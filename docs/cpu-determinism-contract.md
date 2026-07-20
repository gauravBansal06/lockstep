# x86-64 CPU Determinism Contract

Status: design contract (no real CPU core implemented yet). This document turns
the GOLDEN_TARGET principle "CPU execution is a pure function of CPU state,
memory, and instruction definition" into an exact, testable specification for a
single-vCPU x86-64 software interpreter.

It exists so that two correct-looking implementations cannot diverge: every
architecturally *undefined*, *model-specific*, or *host-influenced* behavior is
pinned to one host-independent value here.

Related: [platform-determinism-contract.md](platform-determinism-contract.md),
[GOLDEN_TARGET.md](../GOLDEN_TARGET.md) sections 6, 7, 10, 19, 22.

## 0. The reduction this contract serves

A guest OS amplifies any single nondeterministic primitive into full divergence
within millions of instructions (KASLR, entropy pool, scheduler, hash seeds all
ride on hardware reads). Therefore the CPU must expose **zero** nondeterministic
primitive. Every item below is "what an OS can observe that is not already a pure
function of RunIdentity," made into a fixed rule.

## 1. Scope and non-goals

In scope (v1):

- IA-32e **long mode (64-bit) only**. The VMM hands the guest off already in long
  mode (see platform contract); the guest never leaves it. Real/protected/compat
  mode execution is out of scope.
- **Single vCPU.** No inter-processor interrupts, no memory-ordering interleaving.
- **Soft-float** for all x87/SSE/AVX. No reliance on host FPU.
- Deterministic, host-independent results on any supported host CPU/OS.

Explicitly excluded and must be unreachable or fault deterministically:

- SMM (System Management Mode) and SMIs — never entered.
- Microcode / model-specific undocumented behavior.
- VMX/SVM (nested virtualization) — `#UD` or CPUID-absent.
- MPX, CET, PKU, AMX, AVX-512 (v1) — CPUID-absent; encodings `#UD`.
- Hardware performance monitoring as a real counter — see §8.
- Real-mode BIOS services, legacy `DAA`/`AAA` family (invalid in 64-bit anyway).

## 2. Execution model

```text
(state', mem', events) = execute(state, mem, instr_def, virtual_time)
```

- `execute` reads only: architectural CPU state, guest memory/MMIO, the
  instruction definition, and virtual time / seeded streams via the defined
  instructions (RDTSC, RDRAND, …). Nothing else — no host clock, no host RNG, no
  host pointer, no uninitialized value.
- One instruction is the unit of atomic progress. Interrupts and asynchronous
  events are delivered only at the boundaries defined in §10–§11.
- `virtual_time` advances by the machine's ticks-per-instruction (GOLDEN_TARGET
  §7). Instruction *cost* is uniform and virtual; it never depends on host timing.

## 3. Integer and flag semantics (RFLAGS)

Defined-flag results follow the Intel SDM exactly. The determinism work is the
**undefined** flags: the SDM leaves several flags "undefined" after certain
instructions, which means real CPUs differ. We pin them.

Rule: **every architecturally-undefined flag bit is written to a fixed,
documented value, not left to chance.** Default policy unless a per-instruction
row overrides it: undefined flag → **cleared to 0**. A guest that reads an
undefined flag is buggy; this contract only guarantees we are *deterministic*
and host-independent about it.

Instructions whose undefined flags MUST be pinned (non-exhaustive; the
implementation MUST carry the full table and a test per row):

| Instruction(s) | Defined | Undefined → pinned value |
|---|---|---|
| `MUL`,`IMUL` (1-op) | CF, OF | SF, ZF, AF, PF → 0 |
| `DIV`,`IDIV` | (none) | CF, OF, SF, ZF, AF, PF → 0 |
| `AND`,`OR`,`XOR`,`TEST` | SF, ZF, PF; OF=CF=0 | AF → 0 |
| `SHL`/`SHR`/`SAR` count>1 | CF, SF, ZF, PF | OF → 0; AF → 0 |
| `SHL`/`SHR`/`SAR` count==0 | (flags unchanged) | (none) |
| `SHLD`,`SHRD` | CF, SF, ZF, PF | OF (count≠1) → 0; AF → 0 |
| `BSF`,`BSR` | ZF (src==0) | SF, AF, PF, CF, OF → 0; **dest if src==0 → unchanged** |
| `BT`,`BTS`,`BTR`,`BTC` | CF | OF, SF, AF, PF → 0 |
| `INC`,`DEC` | OF, SF, ZF, AF, PF (CF unchanged) | (none) |
| `RCL`/`RCR`/`ROL`/`ROR` | CF; OF (count==1) | OF (count>1) → 0 |

Other pinned integer behaviors:

- `DIV`/`IDIV` by zero or quotient overflow → `#DE` with no partial commit.
  (Includes the qword `i128::MIN / -1` corner: it raises `#DE` and must never
  overflow/panic the host — Review 4 C1.)
- `RCR`/`RCL` OF (count==1) = XOR of the two most-significant bits of the result
  (Review 4 H1).
- `IMUL`/`MUL` partial-register and sign-extension rules per SDM.
- Writing a 32-bit GPR zero-extends to 64 bits (standard) — must be exact.
- 8/16-bit partial-register writes preserve upper bits deterministically.
- `BSWAP`, `MOVSX`/`MOVZX`/`MOVSXD`, `CDQE`/`CQO` sign/zero extension exact.
- Scalar `MOVSS`/`MOVSD`: memory source zero-extends the destination XMM;
  register→register **preserves** the upper bits (Review 4 C2).

**Intentional deterministic divergences** (deterministic, but NOT guaranteed to
equal silicon — a guest relying on these is buggy; we only promise reproducibility):

- `SHLD`/`SHRD` with `count ≥ operand width` is architecturally undefined; we
  produce a fixed value (`src << (count % bits)`), not whatever a given CPU does.
- The MMU sets the **Accessed** bit on every entry used by a translation *even
  when the walk ultimately raises a permission `#PF`* (it does not roll back A).
  Deterministic and defensible; pinned here so it is not mistaken for a bug.

## 4. Floating point and SIMD (soft-float mandate)

> **Implementation status (2026-07-15, Phase 3d): cross-x86-host MET;
> cross-architecture-host MET for the modeled surface bar one niche item.**
> The executor computes the round-nearest SSE/x87 base results (+,−,×,÷,√,
> int↔float, compares) with host `f64`/`f32`. On any IEEE-754 host these five
> basic operations in round-nearest are *uniquely determined and bit-identical*
> (the standard mandates correct rounding) — so this is cross-*architecture*
> safe (incl. aarch64), not a divergence. Everything that is **not** an
> IEEE-basic-op in round-nearest is computed in our own deterministic software on
> top, so it too is host-independent:
>
> - **MXCSR/FCW rounding-control (RC) — DONE (3a).** All four directed modes for
>   SSE/AVX and x87 ADD/SUB/MUL/DIV/SQRT: host round-nearest result + the EXACT
>   residual sign (`two_sum` for ±, fused `mul_add`/integer for ×÷√), then step to
>   the adjacent representable value. All operands are IEEE-deterministic, so this
>   is cross-host identical. Validated bit-exact vs Alder Lake (>340k vectors).
> - **FTZ/DAZ — DONE (3b).** Flush-to-zero of subnormal results and denormals-are-
>   zero of inputs, deterministic when the guest sets them.
> - **MXCSR/x87 exception status flags (IE/DE/ZE/OE/UE/PE) — DONE (3b, 3d-1).**
>   Set deterministically incl. the masked-underflow (UE) and div-by-subnormal
>   corners (exact integer residual, `div_resid_sign`).
> - **x87 80-bit extended precision + transcendentals — DONE (3c, 3d-2/3).** A
>   full software `F80` (`src/x86/f80.rs`): add/sub/mul/div/sqrt with PC/RC,
>   80-bit subnormals, the canonical 80-bit constants, FSCALE, FPREM/FPREM1, and
>   int↔80-bit conversions — validated bit-exact vs silicon (~4k x87 vectors).
>   Transcendentals (`FSIN`/`FCOS`/`FSINCOS`/`FPTAN`/`FPATAN`/`F2XM1`/`FYL2X`/
>   `FYL2XP1`) use committed soft-float polynomials + Cody–Waite reduction, never
>   host libm — so the result is part of `hypervisor_build_hash`.
>
> The rest of the machine has **no** other host-dependent primitive: audited clean
> of host clock, host RNG, threading, `HashMap`/`HashSet` iteration, pointer-value
> exposure, uninitialized memory, FMA contraction, host libm, and external crates
> (zero dependencies). NaN results are canonicalized from operand bits
> (`fp_result_f64`/`_f32`): a source NaN propagates quieted, a fresh invalid op
> yields the x86 QNaN indefinite — host-architecture-independent.
>
> **Remaining for full cross-*architecture* identity:**
> 1. **Large-argument trig reduction** (`FSIN`/`FCOS`… for |x| ≳ 2^20): the
>    3-part Cody–Waite reduction loses accuracy for huge arguments; Payne–Hanek is
>    the follow-up. This is *deterministic* today (still host-independent) — it is
>    an accuracy gap, not a host-dependence gap. (Silicon's own large-arg x87 trig
>    is implementation-defined, so there is no cross-CPU oracle here anyway.)
> 2. **End-to-end validation on an actual aarch64 build host** — the contract is
>    met by construction (every path above is IEEE-basic-op or committed
>    soft-float), but a physical non-x86 run has not yet been performed.

Soft-float is **mandatory**, not preferred. Recommended reference semantics:
Berkeley SoftFloat / IEEE-754, adapted to x86 quirks. Pin:

- **Rounding**: per x87 control word (RC field) and `MXCSR.RC`. Default
  round-nearest-even. All four modes implemented in software.
- **Precision**: x87 80-bit extended internally; control-word precision field
  (PC) honored. SSE/AVX are 32/64-bit per operation.
- **Denormals**: full IEEE gradual underflow by default. `MXCSR.FTZ` and
  `MXCSR.DAZ` honored *deterministically* when the guest sets them.
- **NaN**: SNaN → QNaN conversion with a fixed canonical payload; QNaN
  propagation follows x86 rule (result takes the NaN of the **source** operand
  when both are NaN; single-NaN operand propagates that NaN, made quiet). The
  canonical default QNaN value is pinned.
- **Exception flags**: `MXCSR`/x87 status word IE/DE/ZE/OE/UE/PE bits set
  deterministically; masked-exception default results pinned.
- **FMA** (if FMA exposed via CPUID): single rounding step, software-exact.
- **Legacy transcendental x87 ops** (`FSIN`,`FCOS`,`FSINCOS`,`FPTAN`,`FPATAN`,
  `F2XM1`,`FYL2X`,`FYL2XP1`): real CPUs differ here and host libm is **not**
  bit-reproducible across hosts. Contract: implement each with a **fixed software
  algorithm** committed to the repo (documented polynomial/argument reduction),
  so the result is part of `hypervisor_build_hash`. Never call host libm.
- `MXCSR` reserved bits, `LDMXCSR` of reserved → `#GP`, deterministic.

A guest may legitimately depend on FP bit patterns (glibc, JITs). FP determinism
is therefore first-class, not a corner case.

## 5. CPUID (fixed CPU identity)

CPUID is a **constant table** keyed by `(leaf, subleaf)`. It is part of
`virtual_cpu_model` in RunIdentity; changing it is a reviewed model-version bump.

Requirements:

- Fixed vendor string, family/model/stepping, and a fixed brand string.
- Feature flags expose **only** what we emulate deterministically. v1 baseline:
  FPU, TSC, MSR, PAE, CX8, APIC, SEP, PGE, CMOV, CLFSH, MMX, FXSR, SSE, SSE2,
  SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, RDRAND, RDSEED, NX, SYSCALL, RDTSCP, LM,
  invariant-TSC. (Target roughly x86-64-v2; AVX optional but if present must be
  soft-float exact.)
- Absent in v1: AVX-512, MPX, CET, PKU, TSX/RTM, AMX, SGX, hardware PMU
  (`PDCM`/architectural-perfmon off), VMX/SVM.
- Hypervisor present bit (CPUID.1:ECX[31]) may be set; the hypervisor CPUID leaf
  range (0x4000_0000+) is fixed if exposed.
- Leaves the guest reads for topology (0xB/0x1F) report a single logical CPU.

Any leaf/subleaf not in the table returns a fixed value (the highest-leaf
fallback rule per SDM), deterministically.

## 6. Model-Specific Registers (MSRs)

The MSR surface is large and is currently the **biggest unpinned area**. Contract:

- There is a documented **MSR map**. Each entry specifies: address, readable,
  writable, reset value, and read/write semantics.
- **Any MSR not in the map → `#GP(0)`**, deterministically. (Kernels probe with
  `rdmsr_safe`/`wrmsr_safe` and tolerate `#GP`; this is the correct default.)
- All values host-independent.

Minimum map for Linux long-mode boot (implementation MUST carry the full list):

| MSR | Addr | Semantics |
|---|---|---|
| IA32_TSC | 0x10 | read = virtual time (§8); write sets a deterministic offset |
| IA32_APIC_BASE | 0x1B | fixed base, BSP + x2APIC-enable bits per model |
| IA32_MISC_ENABLE | 0x1A0 | fixed value (fast-strings on, monitor off, …) |
| IA32_EFER | 0xC000_0080 | SCE/LME/LMA/NXE bits, RW |
| IA32_STAR/LSTAR/CSTAR/SFMASK | 0xC000_0081..84 | SYSCALL config, RW storage |
| IA32_FS_BASE | 0xC000_0100 | RW; also via `WRFSBASE` |
| IA32_GS_BASE | 0xC000_0101 | RW; also via `WRGSBASE` |
| IA32_KERNEL_GS_BASE | 0xC000_0102 | RW; swapped by `SWAPGS` |
| IA32_TSC_AUX | 0xC000_0103 | RW; returned by RDTSCP/RDPID |
| IA32_ARCH_CAPABILITIES | 0x10A | fixed (report mitigations as chosen, stable) |
| IA32_SPEC_CTRL / PRED_CMD | 0x48 / 0x49 | RW storage / write-only no-op, deterministic |
| IA32_MTRRcap / def_type / phys* | 0xFE / 0x2FF / … | minimal: report all-WB deterministically |
| IA32_PAT | 0x277 | RW, reset value fixed |
| IA32_TSC_DEADLINE | 0x6E0 | if APIC TSC-deadline mode exposed (§platform) |

PMU MSRs (0xC1+, 0x186+, 0x38D+): either absent (CPUID perfmon off → `#GP`) or a
deterministic counter; v1 = absent.

## 7. Random-number instructions

`RDRAND`, `RDSEED` draw from a dedicated seeded stream (`derive("cpu-rng")`,
independent of `guest-rng` so device entropy and CPU entropy do not couple).

- Default: **always succeed** (CF=1), value = next stream draw. Simple and fully
  deterministic.
- Optional (seed-driven) failure modeling: a deterministic schedule may force
  CF=0 (retry) to exercise guest retry loops; if enabled it is part of the seed
  contract, never host-driven.
- `RDSEED` vs `RDRAND` draw from sub-labeled streams so they don't alias.

## 8. Timestamp and performance instructions

- `RDTSC`/`RDTSCP` return virtual time (instruction-count-derived, GOLDEN_TARGET
  §7), plus `IA32_TSC` offset. `RDTSCP` and `RDPID` also return `IA32_TSC_AUX`.
- TSC is **invariant** (constant rate vs virtual time); the rate is a fixed
  nominal frequency pinned in the platform contract so guest calibration is
  reproducible.
- `RDPMC` → `#GP` in v1 (no PMU). If a deterministic PMU is later added, it counts
  retired virtual instructions, not host cycles.
- `CPUID` is a serializing instruction but in a single-vCPU model serialization is
  a no-op for state.

## 9. Control registers, segmentation, system state

- CR0 (PE,MP,EM,TS,NE,WP,PG), CR2 (last `#PF` linear address), CR3 (page-table
  base + PCID), CR4 (PAE,PGE,OSFXSR,OSXMMEXCPT,FSGSBASE,…), CR8 (TPR), EFER:
  exact bit semantics, deterministic reset values (platform contract).
- Long-mode segmentation: CS/SS/DS/ES base=0, limit ignored; FS/GS base from
  MSRs. Segment-register loads validate selectors deterministically; null-selector
  rules pinned. `SWAPGS`, `WRFSBASE`/`WRGSBASE` defined.
- GDTR/IDTR/LDTR/TR loaded by VMM at boot; descriptor reads deterministic.
- Debug registers DR0–DR3/DR6/DR7: storage + breakpoint match semantics
  deterministic; `#DB` delivery pinned (§10). Single-step (`RFLAGS.TF`) trap
  timing pinned.
- `CLI`/`STI`/`PUSHF`/`POPF` effects on IF/IOPL deterministic.

## 10. Exceptions (precise-exception model)

Exceptions are classified and committed precisely:

- **Faults** (e.g., `#PF`, `#GP`, `#UD`, `#DE`, `#NM`): reported at the faulting
  instruction with **no architectural side effects committed** beyond those the
  SDM defines (e.g., CR2 set on `#PF`, error code pushed). The instruction is
  restartable.
- **Traps** (e.g., `#DB` single-step, `#BP`/INT3, overflow `INTO`): reported
  *after* the instruction.
- **Aborts** (double fault `#DF`, machine check): deterministic; triple fault →
  deterministic machine reset/halt (no host involvement).
- Error codes, vector numbers, and IDT vectoring (descriptor read, stack switch
  via IST/TSS, CPL checks, pushed frame layout) follow the SDM exactly and
  deterministically.
- Nested exceptions follow the SDM contributory/benign matrix → `#DF` rules.
- **Restartable string ops** (`REP MOVS/STOS/CMPS/SCAS`): RCX/RSI/RDI/RAX reflect
  exact progress; a fault or interrupt is taken at an **iteration boundary**, and
  resuming re-enters with the partial state. The boundary is defined in §11.

## 11. Interrupt delivery and masking

Asynchronous interrupts are delivered only at deterministic points:

- **Boundary**: at instruction retire, and between `REP` iterations. Never
  mid-sub-operation.
- **Masking**: maskable external interrupts gated by `RFLAGS.IF`. `CLI` clears,
  `STI` sets — with the **STI shadow**: interrupts become eligible only *after the
  instruction following* `STI` (one-instruction delay). `MOV SS`/`POP SS` create a
  one-instruction shadow blocking interrupts and `#DB`. These shadows are pinned.
- **NMI**: not masked by IF; once taken, further NMIs blocked until `IRET`
  (NMI-blocking latch) — pinned.
- **Priority** when multiple events are pending at a boundary, highest first:
  1. pending faults/traps from the just-retired instruction,
  2. NMI,
  3. maskable external interrupts, selected by local-APIC priority
     (PPR/TPR; highest in-service-eligible vector wins),
  4. lower-priority pending.
  This total order is fixed so two builds agree on *which* vector is taken when
  several are ready at the same virtual instant.
- **Timing**: the instruction count at which each interrupt becomes pending is
  decided by the deterministic event queue (GOLDEN_TARGET §7) and the platform
  interrupt/timer model (platform contract), never by host scheduling.

This is the contract Linux's entropy pool rides on: `add_interrupt_randomness`
samples the cycle counter at IRQ time, so both the interrupt's instruction count
*and* the TSC it reads must be deterministic — both are, by §8 and the event
queue.

## 12. Memory and MMU

(Memory map, page-table setup, and device MMIO are in the platform contract; the
CPU-side rules:)

- Page-table walks produce `#PF` with exact error-code bits (P/W/U/RSVD/I/D) and
  set CR2 to the faulting linear address.
- **Accessed/Dirty bits**: the A bit is set on a successful translation that uses
  the entry; the D bit is set on the first write through it. The **exact point**
  these writes happen relative to the instruction is pinned: A/D updates are
  committed as part of the translation that the (retiring) instruction performs,
  atomically and deterministically. No speculative A/D writes.
- **TLB model — DECISION (TLB-less by design).** The implementation always walks
  the page tables fresh on every access; it models **no** TLB. A real TLB is a
  microarchitectural cache whose exact contents and refill timing are
  implementation-defined, so omitting it *removes* a nondeterminism source rather
  than adding one — an always-coherent translation is the right model for a DST
  engine. Consequently `INVLPG`, `INVPCID`, `MOV CR3`, and `CR4.PGE`/PCID toggles
  are **no-ops** (there is nothing cached to invalidate) and remain
  architecturally correct: a TLB is permitted to be empty at any instruction
  boundary, so "always empty" is a legal TLB.
  - **No *translation* cache is added** — every access still walks fresh, so the
    A/D-bit write frequency is unchanged and `INVPCID`/`PGE`/PCID tagging stay
    no-ops. (A translation TLB would change A/D frequency and is still not modeled.)
  - **A *decode* cache IS now added for performance** (`exec.rs`, keyed by
    `(rip, cr3)`, generation-flushed). It caches decoded *instructions*, not
    translations, so it does not affect A/D semantics. It is invalidated — an O(1)
    generation bump — on `MOV CR0/CR3/CR4` (address-space switch / PGD reuse /
    paging change), on `INVLPG` (now a real op, not a NOP — single-page remap), and
    on any store to a guest-physical page that holds cached code (self-modifying /
    patched code, tracked by a per-physical-page bitmap). It is excluded from the
    state digest (a pure cache). Validated behaviour-identical: the boot + battle-
    test + all P4 jitter-sweep digests are unchanged with the cache enabled.
- Reserved-bit and NX violations → `#PF` deterministically.
- Misaligned accesses, page-crossing accesses, and lock semantics on a single
  vCPU are deterministic by construction.

## 13. Conformance and testability

> **Run-to-run determinism — PROVEN (2026-06-15).** Two full Alpine-6.12 kernel
> boots (decompression → paging → interrupts → devices → userspace) with identical
> inputs (seed, kernel, cmdline, initrd, instruction budget) produce a
> **byte-identical whole-machine digest** (RAM + every CPU/sys register + all
> device state) after 2,000,000,000 instructions:
> `state_digest=afb6e3a5e78a4e64`, both runs. `run_boot` prints `=> state_digest=…`;
> `scripts/determinism-check.sh <bzImage> [initrd] [max]` is the regression gate
> (boots twice, asserts equality, bisect with a smaller `--max` on divergence).
>
> **Cross-host, same x86 architecture — MET.** Two *different x86 machines* running
> the emulator on identical inputs produce byte-identical state: all integer/logic
> is Rust-spec deterministic; all modeled FP lowers to SSE2 whose IEEE results are
> uniquely determined and identical across x86 CPUs; NaN results are canonicalized
> from bits (§4); and the codebase is audited clean of every other host-dependent
> primitive (no host clock/RNG/threading/`HashMap`/pointers/uninit/FMA/libm, zero
> external crates). The host-*dependent* approximations RCPSS/RSQRTSS (0F 52/53)
> are deliberately not modeled (guest use faults, never silently diverges).
>
> **Cross-host, different architecture (e.g. emulator on aarch64) — MET by
> construction (2026-07-15), physical aarch64 run pending.** The §4 formerly-
> deferred items are all now implemented in host-independent software and
> validated bit-exact vs silicon: MXCSR/FCW rounding modes (3a), FTZ/DAZ (3b),
> MXCSR/x87 exception flags incl. UE/div-by-subnormal (3b/3d-1), and the full x87
> 80-bit path — extended-precision add/sub/mul/div/sqrt, subnormals, constants,
> FSCALE, FPREM/FPREM1, and committed-algorithm transcendentals (3c/3d). Every FP
> path is now either an IEEE-basic-op in round-nearest (bit-identical on any
> conforming host) or our own soft-float. The one open accuracy item — Payne–Hanek
> large-argument trig reduction — is deterministic, so it does not break
> host-independence. Claim ceiling: not yet executed on a physical non-x86 host.

- Every implemented instruction has a determinism test: same input state → same
  output state + flags, and the result is identical when the interpreter binary
  runs on an x86-64 host and an aarch64 host (host-independence gate).
- The MSR map, CPUID table, and undefined-flag table each have an
  exhaustive-coverage test.
- The full-state digest (registers + flags + FP/SSE state + MSRs + control regs)
  feeds the conformance harness (see platform contract §10 and GOLDEN_TARGET §19),
  so any unpinned primitive surfaces as a bisectable divergence rather than a
  silent difference.

## 14. Open decisions (must be fixed before/with implementation)

- Exact CPUID baseline (x86-64-v2 vs v2+AVX) and the frozen feature bitmaps.
- Final MSR map values (reset values, MISC_ENABLE, ARCH_CAPABILITIES contents).
- Whether RDRAND/RDSEED ever model failure, or always succeed in v1.
- Soft-float library choice and the committed algorithms for legacy x87
  transcendentals.
- Whether AVX is in the v1 feature set (cost: more soft-float surface).
