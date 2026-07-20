# Lockstep

**Determinism you can explore. Every bug, on demand. Reproduce anything.**

Lockstep is a from-scratch **deterministic x86-64 machine** — its own CPU, MMU, and
devices, written in Rust, with **no** dependency on QEMU or KVM. A run is a pure
function of `(seed, image, build)`: byte-for-byte identical every time, with zero
coupling to the host clock, host RNG, host threads, or host architecture.

That determinism is the foundation for **Deterministic Simulation Testing (DST)**:
boot your real system inside Lockstep, then explore thousands of timing, scheduling,
and fault schedules by varying a seed — surfacing the race conditions and heisenbugs
that normal testing never hits, and **replaying any failure bit-for-bit**.

## What it does today

- **Boots real Linux and runs real userspace** (ring-3), not a toy. Ten language
  runtimes run **byte-identical to QEMU**: C (glibc), C++, Go, Rust, CPython, Node/V8,
  .NET/C#, OpenJDK/JVM, PHP, Perl, Ruby.
- **Bit-exact x86-64** up to `x86-64-v3` (SSE/AVX/AVX2/FMA/BMI/F16C), plus x87 and an
  80-bit soft-float path, all differentially validated against real silicon.
- **Multi-vCPU SMP** that boots Linux with 2+ processors — deterministically:
  - **Stage 0** — application processors come online through the real
    real-mode → protected → long-mode bring-up trampoline.
  - **Stage 1** — *seed-driven interleaving exploration* (`SF_SMP_SCHED`): different
    seeds realize different, each-reproducible thread interleavings, exposing true
    data races a single core cannot.
  - **Stage 2** — *x86-TSO store buffers* (`SF_SMP_TSO`): store→load reordering
    becomes observable, so the Store-Buffering / Dekker class of bug (impossible under
    sequential consistency) is reachable and reproducible.
- **Controllable I/O + fault injection** at every emulated boundary (disk EIO / torn
  writes / latency, network, timer jitter) — all seed-driven and reproducible.
- **Snapshot / restore** of the whole machine: rewind to any instruction and replay.

## How determinism is kept

Virtual time is the retired-instruction count; every memory operation across all vCPUs
falls into a single deterministic order; interrupts and device completions are delivered
at fixed instruction counts, not wall-clock. Nondeterminism is only ever *injected*
deliberately, at emulated boundaries, and always as a function of the seed — so
"which thread ran on which core, and when" is reproducible, never a function of host
timing. Every change is gated against byte-identical goldens for ten real language
runtimes plus a differential against real silicon.

## Quick start

```bash
cargo build --release

# Boot a Linux bzImage + initrd; the run is fully deterministic.
./target/release/lockstep --seed <64-hex> --boot vmlinuz --initrd initrd.gz \
    --disk disk.img --ram-mb 1024 --smp 2

# Explore concurrency: different seeds → different, each-reproducible schedules.
SF_SMP_SCHED=50 ./target/release/lockstep --seed <seed> --boot ... --smp 2
# Weak memory: x86-TSO store buffers → store→load reordering.
SF_SMP_TSO=50  ./target/release/lockstep --seed <seed> --boot ... --smp 2
```

> **Note:** Lockstep emulates x86-64 hardware and is validated on x86-64 Linux hosts
> (the differential-vs-silicon tests compare against the real CPU).

See [`Progress.md`](Progress.md) for the full roadmap and status, and [`docs/`](docs)
for the determinism contract and design records.
