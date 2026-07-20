# Seed exploration — what can be varied, and the principle

DST is only useful if different seeds **explore different executions** of the *same*
program while each stays **deterministically reproducible**. This doc enumerates the
perturbation axes the substrate exposes, what's demonstrated, and where it can grow.

## The governing principle

We run a **real, unmodified guest kernel + real application binaries**. We do **not**
intercept syscalls or rewrite the guest. Therefore nondeterminism can only be injected
**at the boundaries we actually emulate**:

1. the **CPU / interrupt surface** — *when* interrupts (esp. the timer) are delivered,
   which drives the guest kernel's scheduler and thus thread/goroutine interleaving;
2. the **virtio device surface** — *when* and *how* device I/O completes (latency,
   ordering, errors).

Everything above those boundaries (the kernel's scheduling logic, the app's threads,
libc) is executed faithfully. So "what can be varied" == "what crosses an emulated
boundary." Adding a new axis means emulating a new surface (e.g. a NIC).

## Axes that exist today (all seeded → reproducible)

| axis | knob | what it perturbs |
|---|---|---|
| CPU schedule | `SF_SCHED_JITTER=<max instr>` | the timer-IRQ delivery point → preemption points → thread/goroutine interleaving |
| disk I/O latency | `SF_BLK_FAULT=seed,0,0,<jitter>` | when virtio-blk completions land (can reorder in-flight requests) → who proceeds first |
| disk faults | `SF_BLK_FAULT=seed,<eio‰>,<torn‰>,0` | injected EIO errors / torn writes → fault-resilience + timing perturbation |
| network drop / partition | `--net2 --net-fault seed,<drop‰>[,pstart,pend]` | the two-guest switch drops frames / cuts the link for an instruction window → TCP retransmit + partition tolerance |
| **network I/O-completion order** | `--net2 --net-jitter seed[,<max instr>]` | per-frame RX-delivery jitter → **which socket becomes readable first** → **event-loop** (asyncio/libuv) processing order |
| RNG / entropy | `--seed` (+ `--rngdev` virtio-rng) | `/dev/urandom`, RDRAND are seed-derived → app-level randomness; `--rngdev` seeds crng fast+deterministically |

### Demonstrated on the stateful service (`tools/workloads/statesvc`)

Same workload, fixed `--seed 1`, varying only the disk axis:

| config | machine digest | final STATE-HASH | INV-CONSERVED / DISK-MATCH |
|---|---|---|---|
| baseline (no faults) | `7e08d44815e00ac0` | `506a6121ae1bb283` | ✓ / ✓ |
| torn-write 20% | `6a0609f63142a308` | `506a6121ae1bb283` | ✓ / ✓ |
| torn-write 20% (repeat) | `6a0609f63142a308` | `506a6121ae1bb283` | ✓ / ✓ (**deterministic under faults**) |
| EIO 30% | `bc40c07b2c660efe` | `bfa6d75c4c55e504` | ✓ / ✓ |
| I/O latency/reorder | `463c66aba359cdc5` | `28d7f443ea5cc506` | ✓ / ✓ |

Key reading: **EIO and I/O-latency change the *outcome*** (`bfa6…`, `28d7…`), not just the
machine state — delaying or failing a disk completion makes a different goroutine win the
lock, so a different set of transfers succeeds. This is the same race class as timer
jitter, reached through the I/O surface. The conservation invariant holds across **all**
explored executions; a buggy (lock-removed, or crash-recovery-from-torn-disk) variant
would violate it under some seeds — that's the DST signal.

`SF_SCHED_JITTER` and `SF_BLK_FAULT` **compose**: the explored space is
{schedule} × {disk latency} × {disk faults}, all from one seed.

### Demonstrated on the distributed service (`tools/workloads/dist`, `--net2`)

The bank-service split across **two guest kernels** (node A=client, node B=server) over the
emulated virtio-net switch. All 10 language runtimes run; the invariant (money conserved,
total=2400) is machine-checked across every network schedule/fault. Two things this reaches
that the disk/CPU axes cannot:

**Network faults — invariant survives drops AND a partition** (C, `--seed 1`):

| config | applied/failed | INV-CONSERVED | note |
|---|---|---|---|
| clean link | 2133/867 | ✓ | deterministic 2× |
| 100‰ drop | 2129/871 | ✓ | TCP retransmits; deterministic 2× under loss |
| mid-run partition | 2155/845 | ✓ | link cut for an instruction window; fewer applied, still conserved |

**I/O-completion jitter explores EVENT-LOOP runtimes** — the key result. A single-threaded
event loop (asyncio, libuv) has **no thread interleaving**, so `SF_SCHED_JITTER` (timer/CPU)
cannot perturb it. But *the order I/O completions land* — which socket `epoll` reports
readable first — **is** its nondeterminism, and that crosses the virtio-net boundary.
`--net-jitter` reorders RX delivery per frame (seeded), so the loop processes requests in a
different order → a different set of transfers wins → a different final state. Same workload,
`--seed 1`, varying only `--net-jitter`:

| runtime | baseline | seed 1 | seed 1 (repeat) | seed 2 | seed 3 |
|---|---|---|---|---|---|
| **python** asyncio | `e423e849…` | `93b0cf12…` | `93b0cf12…` ✓det | `dc717ad5…` | `13f625e2…` |
| **node** libuv | `95521738…` | `a9d8261e…` | `a9d8261e…` ✓det | `cfe3d3b3…` | `5c4bc82e…` |

Every cell is `INV-CONSERVED true`. Baseline + 3 seeds = **four distinct schedules per
runtime**, each byte-reproducible; the invariant holds across all of them. This is the
event-loop analogue of timer-jitter exploration, reached through the I/O surface — exactly
the "nondeterminism only at emulated boundaries" principle. (`SF_SCHED_JITTER` remains the
lever for *thread*-model runtimes; `--net-jitter` composes with it and with `--net-fault`.)

## Gaps — the next axes to add (need a new emulated surface)

- **Finer interrupt-delivery jitter** — vary delivery order/coalescing of *non-timer*
  IRQs, and the timer phase independently of the scheduler tick.
- **Memory layout** — seed-vary mmap/heap base (currently `nokaslr` for determinism) to
  surface pointer/layout-dependent bugs (as Antithesis does). Must stay seed-deterministic.
- **Syscall-level perturbation** (EINTR, partial read/write) is *not* directly available —
  we don't intercept syscalls; it must manifest through a device (e.g. virtio-blk EIO
  already makes the guest's `write()` return EIO).
