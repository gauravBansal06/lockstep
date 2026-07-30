# Working in this repo

Lockstep is a **from-scratch deterministic x86-64 machine** in Rust — its own CPU,
MMU, and devices, **not** QEMU/KVM. It's the substrate for Deterministic Simulation
Testing. Read [`Progress.md`](Progress.md) for status, the roadmap, and the doc map.
This file is *how to work here without breaking things*.

---

## 🔴 Two rules that override everything

1. **Everything builds, tests, and boots on an x86-64 Linux host.** Lockstep emulates
   x86-64 hardware and the differential compares against a real x86-64 CPU, so the
   build/test/gate must run on x86-64 Linux (edit anywhere, but build and run there).

2. **Determinism is sacrosanct.** A run must be byte-identical run-to-run, controlled
   only by `seed + spec + image + build`, with zero host coupling (no wall-clock, host
   RNG, host threads, `HashMap` iteration, raw-pointer values, uninitialized memory, or
   host libm/FMA). The *property* must never break. Golden *values* may re-baseline
   **only** at a deliberate, understood config change (a CPUID flip, a digest-formula
   change, an identity-string change, or a new opcode that legitimately shifts a boot's
   instruction stream) — and in the same change the property must be **re-proven**
   (byte-identical N×) with app-level correctness (invariants + differential) shown intact.

---

## The loop

```bash
# build + test (release). cargo test covers the unit suite incl. the determinism,
# SMP, and TSO litmus tests.
cargo build --release
cargo test --release --lib          # expect the full suite green

# regression gate — run after every change. It boots ten real language runtimes and
# asserts each is byte-identical run-to-run, its invariants hold, and its golden hash
# is unchanged, plus a differential vs real silicon.
bash scripts/concurrent-gate.sh
```

A run's determinism is gated three ways: **(1)** the unit suite, **(2)** the differential
vs real silicon (new/changed instructions), **(3)** the ten-runtime concurrent gate
(whole-system byte-identity + invariants). A change that isn't a deliberate config change
must leave every gate golden unchanged.

---

## Validating new instructions: differential vs real silicon

The gold standard. `tools/diff/gen_sse.py` emits randomized operand vectors; the same
bytes run on Lockstep (`--difftest` / `--difftest256` for 256-bit) **and** on the real
CPU (`tools/diff/cputest`, built from `cputest.c` + `trampoline.S`). They must match
bit-for-bit.

```bash
python3 tools/diff/gen_sse.py 1 40 > /tmp/g.txt
./target/release/lockstep --difftest /tmp/g.txt > /tmp/e.txt 2>/dev/null
(cd tools/diff && gcc -O0 -no-pie cputest.c trampoline.S -o /tmp/c && /tmp/c /tmp/g.txt > /tmp/r.txt)
diff -q /tmp/e.txt /tmp/r.txt && echo MATCH
```

**Implementation-defined ops are excluded** from the strict differential (DPPS/DPPD
intermediate precision, RCP/RSQRT approximations) — they aren't bit-exact across CPUs,
so no single CPU can be an oracle. Lockstep still runs them deterministically.

---

## ISA build discipline (Phase A / Phase B)

- **Phase A** — implement & validate new opcodes at the *current* CPUID. The feature
  isn't advertised, so no guest emits it and gate digests stay identical (no re-baseline).
  Validate via the differential.
- **Phase B** — only when a family is complete, flip the CPUID bit to advertise it. That
  is a deliberate config change: re-baseline goldens and re-prove the property.

The ISA surface is `x86-64-v3` (SSE/AVX/AVX2/FMA/BMI/F16C) plus x87 and an 80-bit
soft-float path, all differentially validated. See `Progress.md` for the detailed history.

---

## Determinism & SMP knobs

- `--smp N` — boot N vCPUs (real AP bring-up trampoline).
- `SF_SMP_SCHED=Q` — seed-driven interleaving exploration (Stage 1): each core runs a
  seed-drawn quantum, then the scheduler switches to a seed-chosen core. `0` = fixed
  round-robin (byte-identical baseline).
- `SF_SMP_TSO=W` — x86-TSO store buffers (Stage 2): stores linger a seed-drawn window
  before becoming globally visible → store→load reordering is reachable. `0` = write-through
  (sequential consistency, byte-identical baseline).
- `SF_SMP_PCT=depth[,k]` — Probabilistic Concurrency Testing (Stage 3a): a priority schedule
  with `depth-1` seed-drawn priority-change points, giving a provable ≥`1/(n·k^(depth-1))`
  chance of hitting any depth-`d` bug per seed. `depth=0` = off (byte-identical baseline).
- **DPOR (Stage 3b)** — systematic schedule exploration is a *library API*, not an env knob:
  `x86::dpor_explore` (built on `X86Machine::set_sched_script` forced-schedule replay +
  `set_dpor_record` access tracing, `src/x86/dpor.rs`) enumerates **all** schedule-reachable
  outcomes of a small terminating multi-vCPU program, skipping interleavings that only
  reorder independent accesses. SC only (`tso_max == 0`); litmus scale. The hooks are inert
  (one predicted-false compare) on the baseline path.
- `SF_SMP_COV=W` — race-pair coverage sensor (Stage 3c): observes, O(1) per RAM access,
  cross-vCPU conflicting-access pairs (by code location, within a `W`-instruction window)
  into a signature bitmap; the run prints a deterministic `=> race_cov pairs=… bitmap_digest=…`
  report. Observer-only (digests identical on/off); unset = off, byte-identical baseline.
  The feedback signal for coverage-guided exploration (`x86::cov_explore` steers schedule
  configs by novelty, abandoning saturated axes).
- `SF_SCHED_JITTER=n`, `--disk`, `SF_BLK_FAULT=...`, `SF_NET_FAULT=...` — seed-driven
  scheduling/fault exploration axes.

Any exploration knob left at its default must produce a byte-identical baseline run — new
seed-driven streams are folded into the state digest **only** when the knob is active.

---

## Code gotchas

- The big execute `match op { … }` binds `Op` fields **by value** (`Op` is `Copy`). Do
  **not** write `*l256`/`*bytes`/`*ieee` in arms (E0614) — use the binding directly.
- Keep the proven **SSE execute arms untouched** when adding VEX work — add VEX-only free
  helpers (or reuse existing free fns). Zero regression to validated paths; the differential
  validates the new behavior independently.
- The whole-machine `state_digest` folds new state (YMM upper halves, SMP scheduler RNG,
  TSO store buffers) **only when non-default/non-empty**, so a run that never uses a feature
  reproduces the legacy digest exactly — no re-baseline from merely adding the field.
- Identity strings that the guest can observe (CPU brand via CPUID, ACPI OEM IDs) and the
  seed-derivation / build-hash prefixes are part of the deterministic identity — changing
  them re-baselines every golden and is a Phase-B-style deliberate change.

---

## Where things are

- `src/x86/decode.rs` — instruction decoder (incl. `decode_vex_body`, x87).
- `src/x86/exec.rs` — `X86Cpu`/`X86Machine`, the execute match, free helper fns,
  `full_state_digest`, the SMP scheduler + TSO store buffer, and the test module.
- `src/x86/cpuid.rs` — advertised feature bits (the Phase-B flip point) + CPU brand.
- `scripts/concurrent-gate.sh` — the ten-runtime regression gate + goldens.
- `tools/diff/` — the differential harness (`gen_sse.py`, `cputest.c`, `trampoline.S`).
- `tools/workloads/` — the gate workloads (ten languages, plus DST race/order programs).
- `docs/` — determinism contracts, coverage matrix, frozen values (see the doc map in
  `Progress.md`).
