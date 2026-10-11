# Multi-vCPU / SMP — Implementation Plan

Concrete, staged build plan for adding deterministic multi-vCPU support. The *design
rationale* lives in [`smp-determinism.md`](smp-determinism.md) (cooperative serialization,
memory-access granularity, SC-then-TSO, exploration). **This** doc maps that design onto the
*current code* and sequences the work with milestones, validation gates, and effort estimates.

Grounded in an architecture sweep of `src/x86/{exec,memory,mmu,lapic,ioapic,pit,cpuid,platform}.rs`,
`src/{lib,seed,state}.rs` (2026-07-16).

---

## 0. The one-line goal

Boot Linux with N>1 CPUs and reproduce **true SMP data-race bugs** (missing barriers, lost
updates on shared memory) **deterministically** — the one bug class a single vCPU structurally
cannot exhibit (`tools/workloads/dst/race.c` shows no lost updates on 1 vCPU today). Every
interleaving and memory-visibility choice is `f(seed, state)`; a found bug is one reproducible seed.

---

## 1. Where the 1-CPU boundary sits today (the seam)

The architecture is remarkably ready — the per-vCPU/shared split is already clean:

| State | Scope today | Location |
|---|---|---|
| All CPU exec state (GPRs, RIP, RFLAGS, CRx, segs, MSRs, FPU/XMM/YMM, MXCSR) | **per-vCPU** (in `X86Cpu`) | `exec.rs:42-175` |
| LAPIC (timer only) | **per-vCPU** (in `X86Cpu.lapic`) | `exec.rs:78`, `lapic.rs` |
| `halted`, `rdrand_rng`, decode cache | **per-vCPU** (in `X86Cpu`) | `exec.rs` |
| Guest RAM (flat `Vec<u8>`, SC) | **shared** | `memory.rs` |
| IoBus (serial/PIT/RTC), IOAPIC, virtio | **shared** | `exec.rs:6315`, `ioapic.rs` |
| Event queue, virtual clock, RNG streams | **shared** (`DeterministicCore`) | `lib.rs:25-39` |
| `pending: VecDeque<u8>`, `delayed`, `sched_jitter` | **machine-level** (must become per-vCPU) | `exec.rs:6317-6334` |
| **Hard gate:** `assert_eq!(machine.vcpus, 1, …)` | — | `state.rs:88` |

**What's entirely absent (must be built):**
- **IPI / APIC ICR** (regs `0x300/0x310`) — no core can signal another. `lapic.rs` models only the timer.
- **INIT / SIPI** — no AP-startup sequence; Linux brings APs online with INIT-SIPI-SIPI.
- **Store buffers / memory ordering** — RAM is flat SC; `MFENCE/LFENCE/SFENCE` are unimplemented
  (`#UD`); `LOCK` prefix is a **no-op** (`decode.rs:1956` "trivial on a single vCPU"); `XADD`/`CMPXCHG`
  have precise-exception semantics but zero cross-vCPU atomicity.
- **Cross-core IOAPIC routing** — redirection entries have no destination-CPU field.
- **N-CPU enumeration** — ACPI MADT (`platform.rs:258`) and CPUID topology (`cpuid.rs:28,69-75`)
  hardcode 1.

---

## 2. Global logical clock — the ordering backbone

Today `instruction_count` (in `DeterministicCore::VirtualClock`, `clock.rs`) is *the* clock and it
increments once per instruction of the single vCPU. For N vCPUs it becomes the **global logical
step counter** (Lamport clock): incremented once per *scheduled vCPU step*, imposing the total order
on all cross-core events and keying the event queue.

- Per-vCPU instruction counts still exist (for per-core `RDTSC`), derived as coherent functions of
  the global clock (real cores have skewed TSCs; we make them deterministic and relatable).
- Idle/halt: a halted vCPU is simply not selected; the scheduler fast-forwards the global clock to
  the next event targeting a runnable/wakeable vCPU (generalizes the existing `step_halted` idle
  fast-forward, `exec.rs:6759`).

---

## 3. Staged build order

Each stage is independently valuable, independently gated, and re-proves determinism before the next.

### Stage 0 — Boot N vCPUs, SC, lock-step (no exploration yet) — *make it work*

Goal: Linux boots with `nr_cpus=2`, both CPUs online (`dmesg: smpboot: Brought up 2 CPUs`),
byte-identical run-to-run. This is the hardest *plumbing* stage (AP startup) but uses the simplest
scheduler (round-robin, no seeded interleaving).

1. **Enumeration.** Lift `state.rs:88`; thread `MachineSpec.vcpus` everywhere. CPUID topology
   (`cpuid.rs` leaf 1 EBX[23:16], leaf 0xB subleaves 0/1) and ACPI MADT (`platform.rs:246-263`,
   loop N Local-APIC entries, fix table length) advertise N. Populate `ConfigIdentity.topology_hash`.
2. **N contexts.** `X86Machine.cpu: X86Cpu` → `cpus: Vec<X86Cpu>` with per-vCPU APIC ID; move
   `pending`/`delayed`/halt-wake into each `X86Cpu`. Shared `memory`/`io`/`core` unchanged. Per-vCPU
   `rdrand`/entropy streams via `seed.derive(&format!("vcpu-{id}-rdrand"))`.
3. **IPI path.** Implement LAPIC **ICR** write (`0x300`/`0x310`): decode destination + delivery mode
   (FIXED / INIT / SIPI). Route FIXED → target vCPU's `pending`. This is the new `lapic.rs` surface.
4. **AP startup state machine.** Per-vCPU `ApState { Running, WaitForSipi, SipiLatched(vec) }`. APs
   start in `WaitForSipi` (halted, not selected). **INIT** resets the AP; **SIPI** latches the
   startup vector and makes the AP runnable at real-mode `CS:IP = (vec<<8):0` — the trampoline the
   kernel places low. (Riskiest sub-item: real-mode AP trampoline → the same real→protected→long-mode
   bring-up the BSP already does, but entered via SIPI. Reuse the existing mode-transition code.)
5. **IOAPIC destination.** Add destination-APIC-ID to redirection entries; `vector_for_gsi` returns
   `(vector, dest_cpu)` so device interrupts route to a specific vCPU (default: BSP, as today).
6. **Run loop.** Replace `run(budget)`/`step()` with a **round-robin scheduler loop**: pick next
   runnable vCPU, run it one instruction (or a fixed quantum), advance the global clock, poll events,
   deliver due interrupts to per-vCPU pending. Skip halted vCPUs; fast-forward when all idle.
7. **Determinism + snapshot.** Extend `full_state_digest` / `X86Snapshot` over the `Vec<X86Cpu>` +
   per-vCPU streams + scheduler cursor + global clock. Re-prove boot-twice byte-identical.

**Gate:** boots to userspace with 2 (then 4) CPUs online; `state_digest` byte-identical ×N; existing
single-vCPU gate still green when `vcpus=1` (the N=1 path must be behavior-identical → no golden shift).

*Effort: large (the AP-startup + ICR + run-loop refactor is the bulk of the whole feature).*

### Stage 1 — Deterministic interleaving (SC) + seeded scheduling — *make it a DST substrate*

Goal: reproduce lost-update data races deterministically.

1. **Scheduling granularity → per-memory-access.** A vCPU free-runs between memory-relevant ops and
   yields a scheduling point at loads/stores/atomics/fences/MMIO (the sweet spot from
   `smp-determinism.md §2.1`). Non-memory instructions don't switch.
2. **Seeded scheduler.** New `derive("smp-scheduler")` stream picks the next vCPU at each point.
   Random-schedule-per-seed first (each seed = one interleaving).
3. **Atomics as scheduling points.** `LOCK` RMW / `XADD` / `CMPXCHG` = one indivisible point
   (read+write, no interleave within). Still SC memory (immediate visibility).

**Gate:** `tools/workloads/dst/race.c` (shared counter, no lost updates on 1 vCPU) now **exhibits
lost updates under some seeds**, each reproducible; the conservation invariant fails and minimizes to
a seed. Determinism preserved (same seed → same interleaving → same digest).

*Effort: medium. The memory-access-granularity yield is the core mechanism; the scheduler is small.*

### Stage 2 — Software store buffers (x86-TSO) — *find missing-barrier bugs*

Goal: reproduce store-buffer reordering bugs (the SC model in Stage 1 structurally *cannot*).

1. **Per-vCPU store buffer** intercepting `mem_write` (`exec.rs:1185`). Loads (`mem_read`,
   `exec.rs:1144`) check the **own** buffer first (store-forwarding) then shared RAM.
2. **Seeded drain** (`derive("smp-store-drain")`) chooses when each buffered store publishes to
   shared memory — with the option to delay to the **maximum legal window** to blow a race open
   (adversarial: better than real silicon, which drains in microseconds).
3. **Fences.** Implement `MFENCE/LFENCE/SFENCE` (today `#UD`) as buffer drains/orders per the x86-TSO
   operational model; `LOCK` ops drain+order.

**Gate:** a store-load reordering test (Dekker / SB litmus) that is impossible under SC now manifests
under some seed and reproduces; all Stage-0/1 gates still green.

*Effort: medium-large. Correctness of the TSO operational model + store-forwarding is the risk.*

### Stage 3 — Exploration (DPOR / PCT) — *make the space tractable*

Random-per-seed finds shallow bugs; the interleaving space is exponential. Add **PCT** (probabilistic
priority scheduling with a depth-`d` bound) and **DPOR** (skip interleavings that only reorder
independent accesses). Coverage-guided steering toward new conflicting-access pairs. *Effort: medium;
can be incremental after Stage 1/2 deliver value.*

*Status (2026-07-30): PCT done (Stage 3a, `LS_SMP_PCT`, 2026-07-20). DPOR done at litmus scale
(Stage 3b, `x86::dpor_explore` in `src/x86/dpor.rs`): forced-schedule replay + per-step access
tracing + vector-clock race detection + backtracking driver, validated complete vs a brute-force
schedule enumeration (SB litmus: 4 executions vs 70 schedules, exactly the SC outcome set).
Coverage-guided steering done (Stage 3c, same day): boot-scale race-pair coverage sensor
(`LS_SMP_COV`, observer-only, validated vs exhaustive ground truth) + novelty-steered config
explorer (`x86::cov_explore` — finds a rare pair a blind same-budget sweep cannot).
SC only for DPOR; DPOR-over-TSO and sleep sets remain open.*

### Stage 4 — RunIdentity + model versioning (woven through, not last)

`vcpus`, `memory_model_hash` (SC | x86-TSO), `scheduler_policy_hash` (random | PCT | DPOR) join
`RunIdentity` (`state.rs`) — each is a *different world* and a reviewed model-version bump, not a
silent edit. Snapshots already extend in Stage 0. Done alongside each stage as those knobs appear.

---

## 4. Validation strategy

- **Correctness anchor:** the existing single-vCPU gate must stay byte-identical on the `vcpus=1`
  path at every stage (the N=1 code path is behavior-preserving).
- **Boot proof:** Linux `smpboot: Brought up N CPUs` + a userspace program that pins threads to
  distinct CPUs and observes real parallelism-shaped behavior.
- **The money shot (Stage 1):** `race.c` lost-update reproduced + minimized to a seed — the thing a
  single vCPU cannot do. Add an SMP determinism gate: boot-twice byte-identical over N contexts.
- **Litmus tests (Stage 2):** SB/Dekker/message-passing litmus programs vs. the x86-TSO expected
  outcome set.
- **Cross-check:** where feasible, compare against QEMU `-smp N` for functional (not timing) parity.

---

## 5. Risks & open questions

- **AP real-mode trampoline (Stage 0)** is the single riskiest item: SIPI enters the AP in real mode
  at `(vec<<8):0`; the kernel's trampoline then re-does real→protected→long-mode. We must drive the
  AP through the same mode transitions the BSP uses, entered from SIPI rather than the boot entry.
- **Scheduling granularity vs. throughput:** per-memory-access switching is the fidelity sweet spot
  but adds a yield check to the hot load/store path. Measure MIPS impact (the new perf line makes
  this visible per gate); consider a fast path when only one vCPU is runnable.
- **Decode cache under SMP:** keyed on `(rip, cr3)`, per-vCPU today. A store to a code page must flush
  *all* vCPUs' caches (cross-core SMC). Wire into the existing `code_page_bits` mechanism.
- **TSO faithfulness (Stage 2):** "faithful to which model" is a spec commitment — implement the
  published x86-TSO operational model, pin it in `memory_model_hash`, and litmus-test it.
- **Digest/state growth:** the whole-machine digest now folds N contexts + N store buffers; ensure the
  `vcpus=1` digest is bit-identical to today's (tag per-vCPU state by index; empty store buffers fold
  to the legacy value) so no golden re-baseline is forced by merely *adding* the machinery.

---

## 6. Recommended first move

Stage 0 is the gating deliverable and the bulk of the risk. Suggested first PR slice: **enumeration +
`Vec<X86Cpu>` + ICR/INIT/SIPI + round-robin loop**, validated by "2 CPUs online, boots to userspace,
byte-identical ×2, `vcpus=1` path unchanged." Everything after (seeded interleaving, TSO, DPOR) is
additive on top of a booting SMP machine.
