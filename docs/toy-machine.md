# Toy Machine

> **Status: historical / minimal reference.** The toy machine was the first
> executable guest target — it proved the deterministic contract (seed, clock,
> event queue, state digest, snapshot/restore) before real CPU complexity. It is
> **superseded** by the real x86-64 machine (which now boots Linux and runs real
> runtimes), but kept as a clean, isolated determinism example — `src/toy.rs`,
> the `--toy` flag, and its tests still build and pass. Safe to remove later if it
> stops earning its keep.

The toy machine is deliberately not x86 or ARM. Its purpose is to prove the
deterministic hypervisor contract before real CPU complexity enters the system.

## Components

- 4 general-purpose 64-bit registers
- program counter
- zero flag
- halted flag
- byte-addressed RAM
- deterministic core for virtual clock and events
- small MMIO surface
- deterministic toy block, network, and entropy devices

## Instruction Format

All instructions are bytecode in guest RAM.

```text
0x00                         NOP
0x01                         HALT
0x10 reg imm64               LOAD_IMM
0x11 dst src                 ADD
0x12 dst src                 XOR
0x20 reg addr16              LOAD64
0x21 reg addr16              STORE64
0x30 addr16                  JMP
0x31 reg addr16              JZ
0x40 port reg                MMIO_WRITE
0x41 reg port                MMIO_READ
```

## MMIO Ports

```text
0x00 read  -> virtual instruction count
0x01 read  -> virtual ticks
0x02 read  -> deterministic guest entropy
0x10 write -> schedule deterministic disk completion for request id in register
0x11 write -> schedule deterministic network packet for packet id in register
0x12 write -> schedule deterministic guest entropy event for stream id in register
```

## Toy Devices

The toy machine routes MMIO through explicit device models:

- block device: owns deterministic block contents and schedules completions
- network device: schedules packet delivery or deterministic drops
- entropy device: owns the guest entropy stream
- device scheduler: derives disk/network/entropy timing from seed

This keeps the data/timing split visible:

```text
device state changes are deterministic
guest-visible completion timing is a deterministic event
```

## Determinism Proof

For this machine:

```text
same seed + same program + same RAM size -> same registers + same RAM + same digest
```

Different seeds may change MMIO entropy and deterministic event timing.
