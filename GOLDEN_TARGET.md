# Lockstep Golden Target

This document is the architectural anchor for Lockstep. When design
or implementation choices conflict, prefer the choice that preserves this target.

> This is the **vision / source of truth** — intentionally broader than the
> current code. For *what is actually built today* (and the in-progress / queued
> roadmap) see [`Progress.md`](Progress.md); for how to work in the repo see
> [`CLAUDE.md`](CLAUDE.md).

## 1. North Star

Build a deterministic VM environment where a whole Linux system can be executed
and re-executed exactly.

The long-term target:

```text
same seed_256
+ same virtual machine spec
+ same guest image/environment
+ same hypervisor build
= same VM state sequence
```

Same VM state sequence means not just the same logs or same final result. It
means the same:

- CPU registers
- guest RAM
- virtual disk state
- virtual network state
- virtual clock
- interrupt/device state
- pending event queue
- guest-visible entropy
- final state digest

The system should eventually support both x86-64 and ARM64 guests, with the
first serious target being a single-vCPU Linux guest.

## 2. Why A Hypervisor

Language/runtime-level deterministic testing is powerful but limited. A Go
runtime, JVM, C++ program, database, or closed-source binary can expose
nondeterminism below the language layer.

Syscall interception is also incomplete because guests can observe time or
kernel behavior through paths that do not always pass through ordinary syscalls:

- vDSO time reads
- JIT-generated code
- runtime schedulers
- signals
- io_uring
- kernel scheduling
- device timing
- hardware timestamp/random instructions

Therefore Lockstep anchors below the guest OS, at the virtual hardware boundary.

The guest should see:

```text
virtual CPU
virtual RAM
virtual timers
virtual disk
virtual network
virtual interrupt controller
virtual entropy source
```

The guest must not see:

```text
host wall clock
host RNG
host scheduler timing
host disk latency
host network timing
host CPU feature variation
uninitialized host memory
```

## 3. Determinism Contract

A run identity is:

```text
RunIdentity {
  mode: Pure | Capture | ReplayCapture,
  seed_256,
  arch: x86_64 | aarch64,
  virtual_cpu_model,
  machine_spec_hash,
  hypervisor_build_hash,
  guest_kernel_hash,
  guest_init_image_hash,
  guest_kernel_cmdline_hash,
  disk_base_image_hash,
  initial_virtual_clock_epoch,
  device_topology_hash,
  device_model_hash,
  capture_transcript_hash?, // capture/replay modes only
}
```

The guarantee:

```text
same RunIdentity -> same ordered VM state sequence
```

The seed alone is not the entire identity. The seed drives exploration and
guest-visible uncertainty. The machine spec, images, and implementation versions
freeze the world being explored.

For a software-emulated Pure Mode backend, the target portability guarantee is:

```text
same RunIdentity + same hypervisor build
-> same VM state sequence on any supported host CPU/OS
```

Host-architecture independence is part of the target. A run should not produce
different guest-visible results merely because the emulator process runs on an
x86-64 host, ARM64 host, or a different machine in the fleet.

## 4. Pure Mode

Pure Mode is the real deterministic simulation target.

In Pure Mode:

```text
guest-visible input = f(seed, machine_spec, initial_images, deterministic_models)
```

Allowed:

- deterministic virtual clock
- seed-driven guest entropy
- deterministic virtual network
- deterministic disk timing
- deterministic disk/network fault injection
- deterministic interrupt delivery
- deterministic simulated services

Forbidden unless explicitly modeled:

- real outbound network
- host wall clock
- host RNG
- host disk completion timing
- live host file reads
- interactive user input
- hardware feature passthrough that can vary across machines

Pure Mode is closed-world by default. If the workload needs dependencies, prefer
running those dependencies inside the deterministic guest world or modeling them
as deterministic services.

## 5. Capture Mode

Capture Mode is for workloads that must touch the outside world.

First run:

```text
external world -> capture transcript
```

Replay:

```text
capture transcript -> guest
```

Capture Mode can reproduce a captured run, but it is weaker than Pure Mode
because the external world is not derived from the seed. Replay identity must
include the transcript hash.

Examples of captured inputs:

- DNS responses
- TCP stream bytes
- packet arrival timing
- HTTP/API responses
- serial/stdin input
- host file reads if allowed

The design rule:

```text
No uncontrolled external input may cross into guest-visible state.
It must be generated from seed, frozen in the initial image, or captured.
```

## 6. Seed Model

The seed is a 256-bit root value.

Do not use it as one global random stream. Use a seed tree so subsystems do not
accidentally couple to each other:

```text
root_seed
  derive("guest-rng")
  derive("virtual-clock")
  derive("event-scheduler")
  derive("disk/faults/0")
  derive("network/link/0")
  derive("network/link/1")
  derive("process-faults")
  derive("service-model/dns")
```

The seed drives guest-visible uncertainty:

- `/dev/random`, `/dev/urandom`, `getrandom`
- `RDRAND`, `RDSEED`, or ARM equivalent entropy exposure
- virtio-rng
- network latency, loss, duplication, reorder, partitions
- disk delay, errors, corruption, ENOSPC, timeout
- power loss/crash points
- deterministic service mock behavior
- scheduler/event exploration choices

The seed does not change deterministic CPU instruction semantics. CPU execution
should remain a pure function of CPU state, memory, and instruction definition.

## 7. Virtual Time

Guest-visible time must not come from the host wall clock.

Baseline model:

```text
virtual_time = instruction_count * ticks_per_instruction
```

Instruction-count time is the active-execution clock. Idle states need a second
rule. When the guest executes an idle/halt instruction such as x86 `HLT`, ARM
`WFI`, or an equivalent wait state, and no instruction can retire until an
interrupt, the hypervisor must fast-forward virtual progress to the next pending
deterministic event:

```text
if guest_is_idle:
  instruction_count = next_event.instruction_count
  deliver next_event
  resume guest execution
```

Without this rule, a guest can deadlock in virtual time: instruction count stops,
but the timer interrupt needed to wake the guest is scheduled in the future.

All timer-like guest-visible events should be scheduled by deterministic VM
progress:

```text
event_at = current_instruction_count + deterministic_delay
```

This includes:

- timer interrupts
- disk completions
- network packet arrivals
- fault injection
- virtual device callbacks

Events are delivered at deterministic architectural boundaries. The default
boundary is between complete guest instructions. Real ISA backends must define
the exact interruptible point for restartable or multi-step instructions such as
x86 string operations, faults mid-instruction, and architecturally restartable
loads/stores.

Virtual time must feed:

- timestamp instructions such as x86 `RDTSC` and `RDTSCP`
- Linux clock sources observed through vDSO/syscalls
- RTC/CMOS or equivalent platform clock
- timer devices
- guest scheduler tick source

## 8. Disk Model

Disk data and disk timing are separate.

Disk data:

```text
base_image + deterministic guest writes -> deterministic disk state
```

Disk timing:

```text
completion interrupt = deterministic event queue decision
```

Host I/O completion timing must never directly decide when the guest sees a
completion.

Writes do not need to be recorded in Pure Mode if replay starts from the same
base disk image and re-executes the same guest operations. The virtual disk state
will reconstruct deterministically.

Record logs or snapshots are still useful for:

- faster rewind
- checkpoint restore
- divergence debugging
- Capture Mode external inputs

Disk faults must be deterministic:

- `EIO`
- timeout
- delayed completion
- ENOSPC
- torn write
- corruption after crash

## 9. Network Model

Real outbound network is incompatible with Pure Mode.

Even if the guest sends the same packet at the same virtual instruction count,
the outside world can vary:

- DNS
- remote service state
- latency
- packet loss
- load balancer choice
- TLS/session behavior
- server randomness
- rate limits

Pure Mode network:

```text
virtual_network = simulator(seed, topology, protocol_model)
```

Capture Mode network:

```text
virtual_network = captured transcript
```

The virtual network should support deterministic:

- delivery
- delay
- loss
- duplication
- reordering
- partition
- reconnect/reset
- DNS

## 10. CPU And Memory Model

The hypervisor must control:

- CPU feature identity
- instruction execution backend
- privileged state
- exceptions
- interrupts
- timestamp instructions
- hardware RNG instructions
- performance counters
- page table effects
- guest physical memory
- MMIO
- DMA
- initial RAM contents

Initial RAM must be deterministic. Guest-visible memory must never contain
uninitialized host memory.

For a VM, address translation has two layers:

```text
guest virtual address -> guest physical address -> host backing memory
```

The guest kernel controls the first layer. The hypervisor controls the second.

Floating-point and SIMD semantics are part of CPU determinism. A software
emulator must define exact bit-level behavior for:

- rounding modes
- x87/SSE/AVX or ARM FP/SIMD state
- NaN payload propagation
- denormals
- flush-to-zero and denormals-are-zero behavior
- exception flags
- transcendental or helper-operation results

The preferred Pure Mode baseline is soft-float or another canonical software
implementation. If host FP is used anywhere, the implementation must pin and
verify rounding and denormal behavior and still satisfy host-architecture
independence.

Initial CPU state is guest-visible. Boot register values, Linux boot protocol
structures, memory maps, initrd placement, device probe order, and platform
tables such as ACPI or device tree must be deterministic and covered by
RunIdentity through machine/device topology hashes.

DMA and MMIO ordering are deterministic event surfaces. A device write to guest
memory becomes visible to the CPU only at a deterministic point chosen by the
event model, not when a host thread or host I/O callback happens to complete.

The exact, testable semantics for the first real target (x86-64) live in the
determinism contracts, which turn the principles in this section into pinned
rules (undefined flags, soft-float, CPUID, MSR map, precise exception/interrupt
delivery, MMU A/D bits):

- CPU: [docs/cpu-determinism-contract.md](docs/cpu-determinism-contract.md)
- Platform/boot/devices: [docs/platform-determinism-contract.md](docs/platform-determinism-contract.md)

## 11. What Must Be Virtualized Or Emulated

Minimum eventual Linux-capable surface:

- CPU
- RAM
- boot path
- interrupt controller
- timer
- serial console
- block device
- entropy device
- optional network device

x86-64 examples:

- fixed CPUID
- `RDTSC`/`RDTSCP`
- `RDRAND`/`RDSEED`
- APIC/IOAPIC/PIC model
- PIT/HPET/local APIC timer or simplified deterministic replacement
- serial 16550
- virtio-blk
- virtio-net
- virtio-rng
- MMIO/PIO

ARM64 examples:

- fixed ID registers
- exception levels
- GIC
- generic timer
- device tree
- UART
- virtio-mmio

## 12. Emulation Vs Virtualization

Emulation:

```text
Reproduce the behavior of a device/platform in software.
```

Virtualization:

```text
Expose an isolated virtual version of a real resource.
```

The rule is not "hardware is virtualized and software is emulated." Real systems
often mix both.

Examples:

- QEMU TCG emulates the CPU in software.
- QEMU emulates many devices.
- KVM virtualizes CPU execution using hardware virtualization extensions.
- Firecracker virtualizes CPU/memory through KVM and implements virtual devices.
- A hypervisor often virtualizes CPU/memory and emulates devices.
- Software interfaces can also be emulated, such as syscall or API emulation.

Best mental model:

```text
Emulation: I reproduce behavior, possibly on totally different hardware.
Virtualization: I multiplex/isolate a real resource behind a similar interface.
```

## 13. QEMU Inspiration

QEMU is a full machine emulator and VMM. It can run with:

- TCG: software CPU emulation
- KVM: hardware-assisted virtualization

QEMU TCG plus instruction counting is the most relevant existing model for Pure
Mode determinism.

Borrow from QEMU:

- instruction-count based virtual time
- deterministic record/replay insight
- fixed virtual CPU model
- virtual device state machines
- timer scheduling tied to virtual progress
- disk replay lesson: data and completion order are both important
- machine snapshots
- clear separation between CPU execution, devices, and machine model

Do not blindly borrow:

- huge legacy device surface
- default open-world networking
- host-timed async I/O as guest-visible behavior
- assumption that `icount` alone creates a product-level deterministic guarantee

Important distinction:

```text
QEMU icount = useful primitive
QEMU record/replay = log-based reproducibility
Lockstep Pure Mode = fresh deterministic execution from seed + spec
```

Plain QEMU TCG with `icount` is not automatically Pure Mode. It becomes close
only under strict closed-world conditions where all guest-visible inputs are
deterministic.

## 14. KVM Inspiration

KVM is a Linux kernel virtualization facility. It lets guest code run directly on
real CPU hardware using virtualization extensions.

KVM gives speed, but not easy Pure Mode determinism.

Borrow from KVM:

- vCPU run loop shape
- VM exits as control points
- guest RAM slots
- interrupt injection concepts
- CPU state save/restore
- second-level address translation concepts: EPT/NPT/stage-2

Use KVM later as a performance backend only after the determinism contract is
clear.

KVM risks:

- real CPU timing leaks
- instruction-count scheduling is harder
- hidden hardware differences
- performance counters and timestamp behavior need strict control
- host scheduling affects when exits are handled unless masked by deterministic
  event rules

## 15. Firecracker Inspiration

Firecracker is a small Rust VMM built around KVM and a minimal device model.

Firecracker is not a deterministic emulator. It is still valuable as a design
reference for a minimal production VMM.

Borrow from Firecracker:

- Rust implementation style
- one VM per process
- small machine model
- direct Linux kernel boot
- minimal device set
- virtio-first devices
- snapshot/restore discipline
- jailer/seccomp/namespaces as later host-hardening patterns
- clean control plane separation

Do not borrow as-is for Pure Mode:

- KVM-first CPU execution
- host-timed I/O completion behavior
- real network passthrough
- multi-vCPU execution

## 16. Antithesis-Style Target

The public Antithesis idea we care about:

```text
ordinary software runs inside a deterministic simulated environment;
bugs can be reproduced exactly.
```

The useful conceptual target:

- deterministic hypervisor boundary
- single logical CPU first
- virtual time
- all nondeterminism controlled by the platform
- fault injection from seed
- replayable failing executions
- many seeds explored in parallel
- state snapshots/checkpoints
- property-based bug finding

Parallelism is across runs, not inside a single run. A single v1 run is
single-vCPU and logically single-threaded at the virtual CPU boundary. Guest
threads still exist and are preempted/interleaved by deterministic virtual
interrupts, but true simultaneous multi-core execution is out of scope for v1.
That means v1 will not expose every cross-core data race, weak-memory ordering
bug, or lock-free algorithm bug.

We should not depend on unverified implementation details. The architectural
lesson is enough: push determinism below the OS/runtime boundary and make the
guest universe explicit.

## 17. Performance Strategy

Do not weaken determinism for speed in the core contract.

Safe optimizations:

- single vCPU
- one deterministic event loop
- minimal device set
- direct kernel boot
- no GUI/audio/USB
- virtio-style devices
- copy-on-write disk
- dirty-page tracking
- periodic snapshots
- incremental state hashing
- derived PRNG streams
- deterministic async host I/O where host completion timing is hidden
- DBT/JIT for software CPU emulation

Possible backend strategy:

```text
most deterministic: software emulation / DBT
fast experimental: KVM with strict traps and deterministic event injection
capture mode: KVM plus transcript boundary, where acceptable
```

The golden target remains backend-independent:

```text
guest-visible state must be a deterministic function of RunIdentity.
```

If DBT/JIT is used for CPU emulation, code-cache invalidation for self-modifying
guest code is a determinism surface. Translation cache state must be invisible to
the guest except through architecturally defined behavior.

## 18. Exploration And Property Layer

Lockstep is a DST engine, not just a deterministic VM. The deterministic
hypervisor is the substrate. The engine also needs a search-and-detect layer.

The guest or workload harness must be able to communicate with the engine through
a deterministic channel such as a reserved MMIO port, hypercall, virtio device,
or serial protocol. This channel is for:

- safety assertions: something must always hold
- liveness/progress assertions: something must eventually happen
- test verdicts
- named checkpoints
- coverage/progress signals
- structured debug events

Bug classes must be distinguished:

- guest property violation
- guest panic/crash
- guest deadlock or no progress for a configured virtual budget
- guest timeout/quiescence
- emulator limitation or unimplemented instruction/device
- host/platform failure outside the simulation

Exploration policy is separate from deterministic execution. A seed can drive:

- fault schedule choices
- packet delay/loss/reorder choices
- disk fault choices
- workload random choices
- branch points for future search

The policy layer must eventually support:

- random seed exploration
- coverage-guided exploration
- schedule/fault budget control
- failure minimization or shrinking
- replay and counterfactual branching from snapshots

Run termination is part of the contract. A run may end by halt, instruction
budget, virtual-time budget, property violation, explicit success verdict,
deadlock, or emulator limitation. The termination policy must be included in the
run configuration identity.

## 19. Snapshots, Replay, And Digests

A snapshot must capture the complete deterministic state vector:

- CPU registers and privileged state
- guest RAM
- virtual disk state or CoW layer
- virtual network state
- virtual clock
- pending event queue
- interrupt/device state
- every derived RNG stream position
- capture transcript cursor, if any

Restore equivalence is mandatory:

```text
run to snapshot -> restore -> continue
= same continuation as never taking the snapshot
```

State digests must eventually use a cryptographic hash such as SHA-256 or BLAKE3
over canonical state serialization. Toy FNV-style digests are acceptable only for
early local divergence checks.

Golden digest discipline:

```text
digest changed intentionally -> reviewed hypervisor_build_hash/model-version bump
digest changed unintentionally -> regression
```

## 20. Engine-Internal Determinism Hygiene

The emulator implementation itself must obey deterministic rules.

- Do not use randomized iteration order for anything that affects guest state,
  event order, snapshots, or digests. Prefer ordered containers such as
  `BTreeMap`, or explicitly stable insertion-order containers.
- Do not mix host pointer values, process addresses, ASLR artifacts, thread IDs,
  file descriptor numbers, or allocation addresses into guest-visible state.
- Do not let host wall clock, host RNG, or host thread scheduling outcome affect
  deterministic engine decisions.
- Use explicit wrapping or checked arithmetic for architectural behavior. Do not
  rely on build-mode-dependent overflow semantics.
- Serialize state canonically for hashing and snapshots.
- Keep host errors separate from modeled guest-visible faults. A host disk error
  is not automatically a guest disk error unless the deterministic fault model
  says so.

## 21. First Build Direction

Start below real x86/ARM complexity.

First prove the core mathematical contract:

```text
same seed + same toy machine spec -> same registers + same RAM + same digest
different seed -> different deterministic event schedule
```

Then grow the toy machine:

- RAM
- toy CPU registers
- toy instructions
- MMIO
- timer interrupt
- disk completion event
- network packet event
- entropy device

Only after that should we bind the design to a real backend such as:

- custom software emulator
- QEMU-based prototype
- KVM/Firecracker-style VMM experiment

## 22. Non-Negotiable Rules

1. Host wall clock must never directly enter guest-visible state.
2. Host RNG must never directly enter guest-visible state.
3. Real outbound network is not Pure Mode.
4. Host disk latency must not decide guest disk completion timing.
5. Initial RAM must be deterministic.
6. CPU feature identity must be fixed.
7. All async guest-visible events must be ordered by deterministic VM progress.
8. Seed streams must be independently derived by subsystem label.
9. Same RunIdentity must reproduce the same VM state sequence.
10. Capture Mode must include transcript identity.
11. Idle/halt execution must fast-forward to the next deterministic event.
12. FP/SIMD behavior must be canonical and host-independent.
13. Engine-internal data structures must not leak host nondeterminism.
14. Snapshots must include event queues and every RNG stream cursor.
15. A single v1 run is single-vCPU; parallelism is across runs only.
