# SMP Determinism Design

Status: design anchor for **lifting GOLDEN_TARGET rule 15** (the single-vCPU
restriction). Nothing here is built or scheduled to be built before the
single-vCPU machine is solid; this document exists so that when we add multiple
virtual CPUs we do it deterministically from the start instead of bolting it on.

Companion: [cpu-determinism-contract.md](cpu-determinism-contract.md),
[platform-determinism-contract.md](platform-determinism-contract.md),
[GOLDEN_TARGET.md](../GOLDEN_TARGET.md) sections 7, 10, 16, 17, 22 (rule 15).

## 0. The problem, stated precisely

A single-vCPU deterministic machine reproduces concurrency bugs that manifest via
**preemption** (one core, threads interleaved at timer ticks): lost wakeups,
atomicity violations, check-then-act, signal races. It does **not** reproduce a
bug that requires **two threads running at once on two cores with hardware
store/load reordering** — e.g. a missing memory barrier. That class needs true
multi-core, which is the genuinely hard problem this document addresses.

The hard problem splits into **two layers that must not be conflated**:

- **Layer 1 — Interleaving (control).** With N cores there are N instruction
  streams; which core advances next is nondeterministic on real hardware. This is
  the *scheduling* problem.
- **Layer 2 — Memory ordering (visibility).** Even for a *fixed* interleaving,
  real CPUs do not expose one consistent memory. x86 is **TSO**: a per-core store
  buffer makes stores visible to other cores asynchronously, with store-load
  reordering. ARM is weaker. This is the *memory-model* problem.

**Layer 1 is the easy half** (pick the next core from the seed). **Layer 2 is the
hard half**: a missing-barrier bug *cannot manifest at all* under a model that
applies every store to one shared memory immediately, because that model is
**sequentially consistent (SC)** and SC forbids the reordering the bug needs.
Solving only Layer 1 yields deterministic SMP that still cannot find reordering
bugs.

## 1. The liberating idea

**We never need actual simultaneity.** Two vCPUs never literally run at once. We
*simulate the observable effects* of concurrent execution as a
deterministically-chosen interleaving over a modeled (possibly relaxed) memory
subsystem. No real parallelism, full reproducibility of parallel bugs.

```text
true multi-core bug  =  a deterministically chosen interleaving
                        over a deterministically modeled memory subsystem
```

Determinism holds because every choice — which vCPU steps next, when a buffered
store drains, when an IPI lands — is a pure function of `(seed, global state)`.

## 2. Architecture: cooperative serialization

Instantiate N vCPU contexts sharing guest physical memory, but **never run them in
parallel**. Multiplex them onto one deterministic engine driven by a deterministic
scheduler:

```text
loop:
  vcpu = scheduler.pick_next(seed_stream, global_logical_state)   # deterministic
  vcpu.run_until_next_scheduling_point()                          # free run between points
  commit_memory_effects(vcpu)                                     # per the memory model
  deliver_due_cross_core_events()                                 # IPIs, timers, shootdowns
  global_step += 1
```

N parallel streams collapse into **one deterministic sequence of vCPU steps** —
the single timeline that makes the run reproducible.

### 2.1 Scheduling granularity

Only memory-relevant operations can affect another core, so switch only there.

| Granularity | Fidelity | Cost | Notes |
|---|---|---|---|
| Per-instruction | max | highest | switch after every instruction |
| **Per-memory-access** | **high** | **moderate** | **sweet spot**: switch only at loads/stores/atomics/fences/MMIO; run freely between |
| Per-quantum (k instrs) | medium | low | round-robin like QEMU TCG |
| Per-sync-point (locks only) | low | lowest | assumes data-race-free; **cannot find data races** — too coarse for bug-finding |

Default target: **per-memory-access**. Non-memory instructions execute without a
scheduling point.

## 3. Layer 2: the memory model (staged)

### 3.1 Stage A — Sequential consistency (first)

Serialize at memory-access granularity; apply each store to shared memory
immediately; loads read shared memory. Fully deterministic, simple, and it still
finds the **large majority** of real concurrency bugs (lost wakeups, atomicity
violations, ordering violations, check-then-act). Blind spot: pure store-buffer
reordering bugs.

### 3.2 Stage B — Software store buffers (TSO / weak)

Each vCPU gets a modeled **store buffer**:

- Stores enter the issuing vCPU's buffer (not shared memory yet).
- Loads check the **own** buffer first (store-forwarding), then shared memory.
- The **scheduler deterministically chooses when each buffered store drains** to
  shared memory; drain points are drawn from the seed.
- Fences/atomics drain and order the buffer per the architecture's formal model
  (x86-TSO operational model; ARMv8 weak model when targeting aarch64 guests).

This reproduces missing-barrier bugs — deterministically, because every drain
decision is `f(seed, state)`.

**Adversarial advantage:** a controllable store buffer is *better than real
hardware* at finding these bugs. Real silicon drains in microseconds; we can delay
a drain to its **maximum legal window** to blow the race wide open — the same
"explore the nastiest schedule" idea as fault injection, applied to memory.

### 3.3 Atomics, fences, cross-core control

- `LOCK` RMW: one scheduling point, read+write indivisible (no interleave within),
  orders the buffer per its implicit fence.
- `MFENCE` / `LFENCE` / `SFENCE`, acquire/release: drain/order the buffer per model.
- IPIs, TLB shootdowns, cross-core interrupt delivery: placed on the deterministic
  event queue, delivered at the target vCPU's next scheduling boundary.
- Atomicity unit (cache-line vs word) and unaligned/split-access behavior are
  pinned (extends CPU contract §3/§12).

## 4. Time becomes a global logical clock

A single `instruction_count` no longer suffices to order cross-core events.

- Introduce a **global logical step counter** (Lamport-style), incremented once
  per scheduled vCPU step. It imposes the total order on all cross-core events and
  keys the event queue.
- Per-vCPU instruction counts still exist (for per-core `RDTSC`), but per-core TSCs
  are coherent functions of the global logical clock, not independent — real cores
  have skewed TSCs; we make them deterministic and relatable.
- Idle/halt: a halted vCPU is simply not selected; the scheduler advances the
  global clock to the next event that targets a runnable or wakeable vCPU
  (extends the single-vCPU idle fast-forward, rule 11).

## 5. Determinism is necessary but not sufficient — exploration

Determinism makes a found bug reproducible; it does nothing to *find* it among
exponentially many interleavings. The SMP layer needs an exploration strategy
(this is the deferred exploration layer, now with concurrency teeth):

- **Random schedule per seed** — each seed is one interleaving; run many.
- **PCT (Probabilistic Concurrency Testing)** — randomized priority scheduling
  with a provable probability bound for catching a depth-`d` bug.
- **DPOR (Dynamic Partial-Order Reduction)** — treat two interleavings as
  equivalent when they only reorder *independent* (non-conflicting) memory ops, and
  explore one representative. This is what makes the space tractable.
- **Active delay injection** — insert a scheduler delay right after a suspicious
  store to widen a specific race window (targeted, like fault-policy timing).
- **Coverage-guided** — track observed conflicting-access pairs / interleavings and
  steer toward new ones.

## 6. Fit with the existing substrate

The current design already assumes a single deterministic timeline, so the changes
are additive:

- **vCPU contexts**: N register files over shared guest physical memory
  (`memory.rs` already models shared RAM).
- **Scheduler**: a new `derive("smp-scheduler")` stream and store-drain stream
  `derive("smp-store-drain")`; replace the single run loop with the pick-next loop.
- **Event queue**: already deterministic; rekey events on the global logical clock;
  add IPI / TLB-shootdown event kinds.
- **Snapshots** (Phase 5): extend the state vector to N vCPU contexts, all store
  buffers, the scheduler cursor, and the global logical clock. Restore-equivalence
  (rule 14) must continue to hold.
- **Conformance harness**: same boot-twice + cross-host + bisect, now over the
  multi-vCPU full-state digest.

## 7. RunIdentity changes

A different memory model or scheduler is a **different world** and must be in
identity:

```text
RunIdentity += {
  vcpus,                 // N
  memory_model_hash,     // SC | x86-TSO | armv8-weak
  scheduler_policy_hash, // random | PCT(params) | DPOR | ...
}
```

Changing any of these is a reviewed model-version bump, not a silent edit.

## 8. Honest tradeoffs

- **SC model**: deterministic, simple, finds most bugs — but **false negatives** on
  reordering bugs.
- **Faithful relaxed model**: finds reordering bugs, but is a real
  memory-subsystem simulator: slower, and "faithful to which model" is a spec
  commitment (implement the formal operational x86-TSO / ARMv8 model).
- **Serialization cost**: controlling interleaving means no intra-run host
  parallelism. DST throughput is reclaimed the same way as single-vCPU — run many
  seeds in parallel *across* runs.
- **Exploration is mandatory**: without DPOR/PCT the interleaving space is
  intractable.

## 9. Build order (after single-vCPU machine is solid)

1. Finish single-vCPU determinism end-to-end (current track).
2. N vCPUs + **SC** memory model at memory-access granularity + random/PCT
   scheduling. Deterministic, catches most concurrency bugs.
3. **Software store buffers (x86-TSO)** for true reordering bugs.
4. **DPOR** to make exploration efficient.
5. **ARMv8 weak model** if/when aarch64 guests are targeted.

## 10. Resolution of rule 15

Rule 15 ("a single v1 run is single-vCPU; parallelism is across runs only") is a
**staging decision, not a permanent wall**. The resolution to "cannot reproduce
true multi-core simultaneity" is: simulate N vCPUs on one deterministic engine,
switch at memory-access granularity, model the store buffers, and let the seed
deterministically choose both the interleaving and the store-drain order. That
reproduces — and can aggressively hunt — exactly the missing-barrier bugs rule 15
currently excludes, while preserving `same RunIdentity -> same VM state sequence`.
