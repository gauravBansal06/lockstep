# x86-64 Platform and Boot Determinism Contract

Status: design contract (no real platform implemented yet). This document pins the
deterministic behavior of everything *around* the CPU core: boot handoff, memory
map, platform tables, interrupt controllers, timers, devices, and the host-side
rules — so that a real guest OS boots to a byte-identical state sequence under the
same RunIdentity.

Companion to [cpu-determinism-contract.md](cpu-determinism-contract.md). Related
GOLDEN_TARGET sections: 3, 5, 7, 8, 9, 10, 11, 19, 20, 22.

## 0. The reduction this contract serves

With the CPU core exposing no nondeterministic primitive (companion doc), the only
remaining ways nondeterminism can enter guest-visible state are: the **boot
handoff**, the **platform tables**, the **timers/interrupt controllers**, the
**devices**, and the **emulator's own implementation**. Each is pinned below. The
test of completeness is: boot the same kernel+initrd+cmdline twice (and on two host
architectures) and get identical full-state digests at every checkpoint.

## 1. Boot model: direct kernel boot

No firmware/BIOS/UEFI (Firecracker-style). The VMM constructs long mode and jumps
straight to the kernel's 64-bit entry. This removes the single largest legacy
nondeterminism surface.

Pinned handoff (the guest's first observed state):

- **Images**: kernel (64-bit entry of the bzImage / extracted vmlinux), `initrd`,
  and `cmdline` loaded at **fixed guest-physical addresses**, deterministically.
  Their hashes are in RunIdentity (`guest_kernel_hash`, `guest_init_image_hash`,
  `guest_kernel_cmdline_hash`).
- **boot_params (zero page)**: byte-deterministic. Every field the kernel reads
  (e820 table, cmdline pointer, initrd addr/size, setup_header fields, screen_info
  zeroed, etc.) is fixed; all reserved/unused bytes are **zeroed**.
- **Page tables**: VMM-built identity map covering low RAM; entries deterministic.
- **Entry register state** (must be specified exactly, example shape):
  - `RIP` = 64-bit entry point
  - `RSI` = guest-physical address of boot_params
  - `CR0` = PE | PG (| MP | NE)
  - `CR3` = boot page-table base
  - `CR4` = PAE (| PGE)
  - `EFER` = LME | LMA | NXE (| SCE)
  - `RFLAGS` = 0x2 (reserved bit only; IF=0)
  - GDT loaded with flat 64-bit code/data; `CS`/`SS`/`DS`/`ES`/`FS`/`GS` set
  - `RSP` = fixed boot stack; all other GPRs = 0
  - IDTR/LDTR/TR/DR* = fixed reset values
  - x87/SSE state = fixed reset (control word, MXCSR defaults)

The entire reset/boot register file is part of the determinism contract — "fixed"
is not enough; the exact values are committed and covered by `machine_spec_hash`.

## 2. Guest physical memory map (e820)

A fixed, byte-deterministic e820 map, part of `device_topology_hash`:

- usable RAM region(s) of the configured size,
- reserved legacy hole below 1 MiB,
- reserved region for ACPI tables,
- reserved MMIO gap below 4 GiB for device windows (LAPIC, IOAPIC, virtio, HPET),
- (optional) RAM above 4 GiB.

Region ordering, addresses, and types are fixed. Guest RAM is zero-initialized
(GOLDEN_TARGET rule 5); no host memory contents ever leak in.

## 3. Platform tables (ACPI, etc.) — byte determinism rules

All VMM-generated tables MUST be byte-identical across runs and hosts. The rules
that make table generation deterministic:

- **No timestamps, no host data.** OEM IDs, OEM table IDs, and revisions are fixed
  string/number constants.
- **All reserved and padding bytes zeroed** before checksum.
- **Checksums** computed over the fully-populated, deterministic byte image.
- **Stable ordering** of every list (MADT entries, etc.) — never hash-map order.

Minimum table set for long-mode Linux:

- **RSDP** → **XSDT** → **FADT** (→ **DSDT**), **MADT** (local APIC + IOAPIC +
  interrupt source overrides), optionally **MCFG** (PCI ECAM) and **HPET**.
- **DSDT**: minimal AML describing the platform; fixed bytes.
- **MP tables / SMBIOS**: optional; if present, fixed bytes.

The chosen table set and their bytes are covered by `device_model_hash` /
`device_topology_hash`.

## 4. Interrupt controllers

- **Local APIC** + **IOAPIC** (PIC/8259 present but masked; modern Linux uses
  APIC). State (IRR/ISR/TPR/PPR, LVT, redirection entries) is deterministic.
- Interrupt **routing** (device IRQ → IOAPIC pin → vector) is fixed by the MADT
  and deterministic.
- **EOI**, masking, and level/edge semantics pinned.
- The CPU-side *selection and timing* of delivery is the companion doc §11; this
  doc fixes *which device raises which line* and *when*, always via the
  deterministic event queue, never host scheduling.

## 5. Timers and the cross-calibration invariant

This is the subtle one. A kernel **calibrates clock sources against each other at
boot** (e.g., counts TSC ticks across a PIT/HPET interval). If the sources are not
all exact functions of one virtual-time base, calibration yields a host-dependent
result and the guest diverges — even with a deterministic TSC.

Contract: every clock source is an exact, pinned function of the single virtual
time base `T` (instruction-count × ticks-per-instruction):

- **TSC** = `T` (+ MSR offset), invariant, at a **fixed nominal frequency** `f_tsc`.
- **PIT (8254)** counts at fixed nominal 1.193182 MHz **mapped onto `T`** by a
  fixed ratio.
- **HPET** main counter at a fixed femtosecond period, mapped onto `T`.
- **Local APIC timer** (periodic / one-shot / TSC-deadline) driven by `T`.
- **RTC/CMOS** advances from `initial_virtual_clock_epoch` by `T`.

**Invariant:** any ratio the kernel measures between these is a fixed constant
independent of host, because all derive from `T`. The nominal frequencies and the
`T`-mappings are committed constants (part of `machine_spec_hash`).

Idle/halt: when the vCPU halts (`HLT`) waiting for an interrupt, virtual time
**fast-forwards to the next scheduled event's instruction count** (GOLDEN_TARGET
rule 11, already implemented in the core), then the timer/interrupt fires. All
clock sources read consistently across the jump because they share `T`.

## 6. Devices

General device rules (GOLDEN_TARGET §8–§10):

- **Data vs timing separated.** Device *data* is deterministic (images, seeded
  models). Device *completion timing* is a deterministic event-queue decision,
  never host I/O completion.
- **DMA visibility**: a device write to guest RAM becomes CPU-visible only at the
  deterministic completion point chosen by the event model.
- **Enumeration order** (PCI bus scan, or virtio-mmio fixed addresses) is fixed and
  part of `device_topology_hash`.

### 6.1 virtio (blk / net / rng)

> **Implementation status (2026-06-12, Review 3): completion timing is v0
> SYNCHRONOUS, not event-queued.** `virtio_blk` and `virtio_rng` currently
> process the queue and write DMA-visible results + raise the completion IRQ
> *synchronously at QueueNotify*. This is deterministic (tied to guest
> instruction progress), but it is **not** the scheduled event-queue completion
> the bullets below describe, and it cannot yet perturb device timing for
> seed-driven exploration. Moving completion into the deterministic event queue
> is planned; until then treat the "scheduled at an instruction count" wording as
> the target, not the current behavior.

- Transport: virtio-mmio at fixed addresses, or virtio-pci with fixed BARs and
  enumeration order. Choice is fixed in `device_topology_hash`.
- **Feature negotiation** offers a fixed feature set; deterministic.
- **Virtqueue processing**: the device consumes the available ring in the order the
  guest published; it writes the used ring deterministically. When multiple
  requests are in flight, **completion order is a deterministic policy** (default:
  FIFO by submission; later, a seed-scheduled order may be used to *explore* fault
  interleavings — but that lives in the deferred exploration layer, not here).
- **Completion interrupt** is raised at a scheduled instruction count via the event
  queue, subject to the (deterministic) interrupt-suppression flags.
- **virtio-blk**: backing image is the deterministic disk model; faults
  (EIO/timeout/ENOSPC) come from the seeded fault model (already prototyped in the
  toy `device.rs`).
- **virtio-net**: Pure Mode = seeded simulator (loss/dup/reorder/partition, already
  prototyped); Capture Mode = transcript (deferred).
- **virtio-rng**: bytes from the seeded entropy stream.

### 6.2 Serial 16550

- Output is deterministic: bytes accepted immediately; TX-empty / THR-empty
  interrupts scheduled via the event queue. LSR/IIR/IER/FCR semantics pinned.
- Input in Pure Mode is forbidden unless seed-generated or transcript-captured
  (Capture Mode, deferred). No host stdin timing crosses the boundary.

### 6.3 RTC / CMOS

- Initial wall-clock = `initial_virtual_clock_epoch` (RunIdentity); advances with
  `T`. No host wall clock, ever (GOLDEN_TARGET rule 1).

## 7. OS-determinism reduction notes (why HW determinism is sufficient)

With the above honored, the OS's apparent randomness becomes deterministic. The
couplings worth keeping in mind because they are the easiest to break:

- **Entropy from interrupt timing.** Linux mixes the cycle counter sampled at IRQ
  time into its entropy pool. Requirement met because: every interrupt's
  instruction count is event-queue-scheduled, and the TSC read in the IRQ path is
  the deterministic virtual time.
- **Early RNG for KASLR / `AT_RANDOM` / siphash keys / SLAB freelist / stack
  canaries.** All pull from the kernel's early entropy, which pulls from
  `RDRAND`/`RDSEED`/virtio-rng. Requirement met because those are seeded
  (companion §7, and virtio-rng above). KASLR therefore lands at a deterministic
  address; `nokaslr` is **not** required (but if set, it is in the cmdline hash).
- **Scheduler decisions** are driven by deterministic timer ticks and deterministic
  runqueue state → deterministic on a single vCPU.
- **PIDs, inode allocation, readdir order, dcache hashing** are deterministic given
  deterministic FS operations on a deterministic disk image and seeded hash keys.

Conclusion: there is no separate "make the OS deterministic" task. There is only
"leave no nondeterministic primitive observable," which is exactly the union of
this contract and the CPU contract.

## 8. Host-side determinism (emulator implementation)

Restating GOLDEN_TARGET §20 as platform-build requirements:

- No randomized-iteration containers for anything affecting guest state, event
  order, tables, snapshots, or digests — ordered/insertion-stable only.
- No host pointer/ASLR/thread-id/fd values mixed into guest-visible state or
  digests.
- No host wall clock / host RNG / host thread-scheduling outcome in engine
  decisions.
- Async host I/O completion timing is hidden behind the event queue; a host disk
  or network error is **not** a guest fault unless the deterministic fault model
  says so.
- Canonical serialization for hashing/snapshots; cryptographic digest (BLAKE3/
  SHA-256) for the real machine (the toy FNV digest is local-only).

## 9. Determinism conformance harness (the gate)

The Phase 5 divergence detector and checkpoints are the foundation. For the real
machine this MUST become an always-on gate:

1. Boot the same `(kernel, initrd, cmdline, disk image, seed, machine spec)` twice.
2. Checkpoint the **full-state digest** (CPU + FP/SSE + MSRs + RAM + devices +
   event queue + RNG cursors) at a fixed virtual-time cadence.
3. Assert the two checkpoint chains are identical; run one instance on an x86-64
   host and one on an aarch64 host to enforce **host-independence**.
4. On mismatch, **bisect** to the first diverging instruction and report it.

Every unpinned primitive in either contract surfaces here as a bisectable
divergence rather than a silent difference. This gate is built and exercised
against the toy first, then turned on the moment the real core boots anything.

## 10. Open decisions (must be fixed before/with implementation)

- virtio transport: mmio vs pci (mmio is simpler and more deterministic to
  enumerate; pci is closer to real hardware).
- Exact e820 layout and the reserved MMIO window addresses.
- Which ACPI tables are minimally required for the chosen kernel config, and their
  committed bytes.
- Nominal `f_tsc`, PIT/HPET/APIC-timer frequencies and their `T`-mappings.
- Multi-request virtio completion-order policy default (FIFO) and where the
  seed-scheduled variant will plug in later.
- Exact boot-entry register file values.
