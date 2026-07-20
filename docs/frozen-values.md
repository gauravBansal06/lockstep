# Frozen Values — Freeze v1 (x86-64)

These are the concrete, committed constants that the determinism contracts leave
as "open decisions." Once frozen, they are part of identity: changing any value
here is a **reviewed model-version bump** of `hypervisor_build_hash` /
`virtual_cpu_model` / `machine_spec_hash`, never a silent edit (golden-digest
discipline, GOLDEN_TARGET section 19).

Companion: [cpu-determinism-contract.md](cpu-determinism-contract.md),
[platform-determinism-contract.md](platform-determinism-contract.md).

Scope: single-vCPU, 64-bit long mode, no SMM. The CPU model is **truthful to the
implemented ISA**: CPUID advertises only instruction families the executor fully
implements. As of 2026-06, that is **full x86-64-v3** (the v2 set — SSE3/SSSE3/
SSE4.1/SSE4.2, CMPXCHG16B, POPCNT, LAHF — plus AVX/AVX2/FMA/BMI1/BMI2/F16C/MOVBE/
LZCNT/TZCNT, all VEX-decoded and **differential-validated bit-exact vs silicon**) +
the **MMX CPUID bit** (baseline gate so glibc loads; MM-register ops still `#UD`) +
**x87 FPU**. **CPUID flipped to v3 (2026-06-18)** — a reviewed model bump
(re-baselined + re-proven). XSAVE is advertised and leaf 0xD enumerates x87+SSE+AVX.
**CAVEAT — userspace AVX is not yet exercised:** at leaf-0xD subleaf-1 = `0x3`
(XSAVEC, current) the guest kernel disables system xstate (a buffer-size consistency
check) and falls back to FXSAVE, so runtimes ifunc to SSE2; truly enabling it
(subleaf-1 = `0x1`) hangs the boot in a chain of phase-sensitive early-init timing
loops (clock calibration → `udelay` poll loop → …; see Progress.md "B.4-pre").
AES/PCLMULQDQ off (not implemented). **RDRAND/RDSEED are implemented** (deterministic
seed-derived, `0F C7 /6,/7`) but **not advertised yet** (Phase A). Guest entropy
comes from virtio-rng (seed-derived). FP is currently host `f64`/`f32`,
not yet a canonical soft-float (cpu contract §4 — host-independence across *x86*
hosts is met; cross-arch is *partial*). The current frozen golden digests live in
[`../Progress.md`](../Progress.md) and `scripts/determinism-gate.sh`.

---

## 1. Machine spec defaults

| Field | Freeze v1 |
|---|---|
| `arch` | x86_64 |
| `vcpus` | 1 |
| `memory_bytes` | configurable; default 128 MiB (must be in `machine_spec_hash`) |
| `ticks_per_instruction` | 1 |

## 2. Virtual time and clock frequencies

One virtual-time base drives every clock source, so any calibration the guest
performs is a fixed ratio (platform contract section 5).

```text
base unit      : 1 tick = 1 retired instruction        (ticks_per_instruction = 1)
virtual time   : virtual_ns = instruction_count        (define 1 tick = 1 ns)
nominal f_cpu  : 1.000 GHz
```

Derived sources (all exact functions of `instruction_count`):

| Source | Nominal frequency | Mapping |
|---|---|---|
| TSC | 1.000 GHz (invariant) | `tsc = instruction_count` (+ IA32_TSC offset) |
| PIT (8254) | 1 193 182 Hz | `pit = floor(instruction_count * 1193182 / 1e9)` |
| HPET | 100 MHz (period 10 ns, `10_000_000` fs) | `hpet = floor(instruction_count / 10)` |
| Local APIC timer | 1.000 GHz bus, divider per guest config | counts from `instruction_count` |
| RTC/CMOS | 1 Hz wall | `epoch + floor(instruction_count / 1e9)` seconds |

`initial_virtual_clock_epoch` (RunIdentity) default: **1700000000**
(2023-11-14T22:13:20Z). Fixed unless overridden; always in identity.

## 3. Seed derivation labels (frozen)

The seed tree derives independent streams by exact label string. Adding or
renaming a label is a model bump.

```text
guest-rng            virtual-clock        event-scheduler
cpu-rng              cpu-rng/rdrand       cpu-rng/rdseed
disk/faults/0        network/faults/0     network/link/0
service-model/dns    process-faults
toy-mmio  toy-devices  toy-entropy        (toy machine only)
```

## 4. CPUID (Freeze v1)

Constant table keyed by `(leaf, subleaf)`; part of `virtual_cpu_model`.

- **Vendor** (leaf 0 EBX:EDX:ECX): `"GenuineIntel"`.
- **Max basic leaf** (leaf 0 EAX): `0x0D`. **Max ext leaf** (0x80000000 EAX): `0x80000008`.
- **Signature** (leaf 1 EAX): family 6, model 0x2C, stepping 2 → `0x000206C2`
  (Westmere-class: a low-errata x86-64-v2 identity).
- **Leaf 1 EBX** = `0x0001_0800`: CLFLUSH line size 8 (×8 = 64B), max logical IDs = 1.
- **Leaf 1 EDX** = `0x078B_FBF1`. Bits set: FPU(0), TSC(4), MSR(5), PAE(6),
  MCE(7), CX8(8), APIC(9), SEP(11), MTRR(12), PGE(13), MCA(14), CMOV(15),
  PAT(16), PSE36(17), CLFSH(19), **MMX(23)**, FXSR(24), SSE(25), SSE2(26).
  **MMX(23) is now SET** as the x86-64 baseline gate (glibc's loader rejects a CPU
  below baseline if MMX is missing, even though modern code uses SSE2 not MM regs);
  MM-register instructions themselves still `#UD`.
- **Leaf 1 ECX** = `0xB4D8_3201` (**x86-64-v3, flipped 2026-06-18**). Bits set:
  **SSE3(0), SSSE3(9), FMA(12), CX16(13), SSE4_1(19), SSE4_2(20), MOVBE(22),
  POPCNT(23), XSAVE(26), AVX(28), F16C(29), HYPERVISOR(31)**. OSXSAVE(27) is added
  **dynamically** from CR4.OSXSAVE in the CPUID exec arm (so the static table clears
  it). Still cleared: PCLMULQDQ(1), AESNI(25), RDRAND(30) (RDRAND/RDSEED implemented
  but Phase A — unadvertised).
- **Leaf 0x80000001 ECX** = `0x21`: LAHF_LM(0) + LZCNT/ABM(5).
- **Leaf 7 / subleaf 0 EBX** = `0x128`: **BMI1(3), AVX2(5), BMI2(8)**. FSGSBASE(0),
  RDSEED(18, implemented-but-Phase-A), SMEP/SMAP still clear.
- **Leaf 0xD (XSAVE enumeration)**: subleaf 0 = `[0x7, 0x340, 0x340, 0]` (XCR0 mask
  x87+SSE+AVX = 0x7, max area 832); subleaf 1 = `[0x3, 0, 0, 0]` (XSAVEOPT+XSAVEC —
  see the userspace-AVX caveat in §0); subleaf 2 = `[0x100, 0x240, 0, 0]` (AVX state
  size 256 @ offset 576).
- **Leaf 0x80000001 EDX**: SYSCALL(11), NX(20), RDTSCP(27), LM(29).
  **ECX**: LAHF_LM(0).
- **Leaf 0x80000007 EDX**: invariant TSC (bit 8) set.
- **Leaf 0x80000008 EAX**: phys address bits = 40, linear address bits = 48.
- **Brand string** (0x80000002–4): `"Lockstep Deterministic CPU v1            "`
  (fixed 48 bytes).
- Any leaf/subleaf not specified → SDM "highest-leaf" fallback, deterministic.
- Topology leaves (0x0B/0x1F) report a single logical processor.

## 5. MSR reset values (Freeze v1)

Unknown MSR → `#GP(0)` (CPU contract section 6).

| MSR | Addr | Reset value | Notes |
|---|---|---|---|
| IA32_TSC | 0x10 | 0 | read = virtual time + offset |
| IA32_APIC_BASE | 0x1B | 0xFEE0_0900 | BSP(8), global-enable(11), base 0xFEE00000 |
| IA32_MISC_ENABLE | 0x1A0 | 0x1 | fast-strings enable only; everything else off |
| IA32_PAT | 0x277 | 0x0007_0406_0007_0406 | architectural reset value |
| IA32_MTRR_DEF_TYPE | 0x2FF | 0xC06 | MTRR enable + default type WB(6) |
| IA32_MTRRCAP | 0xFE | 0x0508 | fixed-range + WC supported, 8 var ranges |
| IA32_EFER | 0xC000_0080 | 0x0000_0500 | LME(8)|LMA(10); SCE/NXE set later by guest |
| IA32_STAR | 0xC000_0081 | 0 | RW storage |
| IA32_LSTAR | 0xC000_0082 | 0 | RW storage |
| IA32_CSTAR | 0xC000_0083 | 0 | RW storage |
| IA32_SFMASK | 0xC000_0084 | 0 | RW storage |
| IA32_FS_BASE | 0xC000_0100 | 0 | RW; also WRFSBASE |
| IA32_GS_BASE | 0xC000_0101 | 0 | RW; also WRGSBASE |
| IA32_KERNEL_GS_BASE | 0xC000_0102 | 0 | RW; SWAPGS target |
| IA32_TSC_AUX | 0xC000_0103 | 0 | RW; RDTSCP/RDPID |
| IA32_ARCH_CAPABILITIES | 0x10A | 0 | report no special caps (stable) |
| IA32_SPEC_CTRL | 0x48 | 0 | RW storage, no effect |
| IA32_PRED_CMD | 0x49 | (wo) | write-only no-op |

PMU MSRs (0xC1, 0x186, 0x38D, 0x38F, …): **absent** → `#GP` (perfmon off in CPUID).

## 6. Boot register file (long-mode handoff, Freeze v1)

VMM-built long mode, then jump to the kernel 64-bit entry. Convention follows the
known-good KVM/Firecracker direct-boot state.

Two columns: **Freeze v1** is the intended frozen contract; **Code today** is
what `X86Machine::boot`/`boot_bzimage` actually construct right now. Where they
differ, *Code today* is authoritative until the gap is closed (Review 3 anti-drift
rule). Rows marked *(not built)* are part of the intended contract but the current
boot path does not yet construct them.

| Item | Freeze v1 (intended) | Code today |
|---|---|---|
| `RIP` | kernel 64-bit entry | kernel entry (`kernel_load`, or `+0x200` for bzImage) |
| `RSI` | `0x0000_7000` (zero page) | `0x0000_7000` |
| `RSP` | top of RAM | `ram_bytes` (top of RAM) |
| `RBP`, other GPRs | 0 | 0 |
| `RFLAGS` | `0x2` | `0x2` |
| `CR0` | `0x8005_0033` (PE,MP,ET,NE,WP,AM,PG) | `0x8005_0033` |
| `CR2` | 0 | 0 |
| `CR3` | `0x0000_1000` (PML4 base) | `0x0000_1000` |
| `CR4` | `0x0000_0020` (PAE) | `0x0000_0020` |
| `EFER` | `0x0000_0500` (LME,LMA) | `0x0000_0500` |
| Boot page tables | identity-map low RAM | **PML4 @0x1000, PDPT @0x2000** (1 GiB identity pages) |
| GDT | base `0x500`: null/code/data | *(not built — flat long mode, no GDT in guest memory)* |
| `CS`/`DS/ES/FS/GS/SS` | descriptors as listed | selectors held; flat base 0, no hidden descriptor model *(not built)* |
| GDTR / IDTR | base 0x500 / 0x520 | 0 until the guest loads them (LGDT/LIDT) *(not built)* |
| x87 control word / MXCSR | 0x037F / 0x1F80 | *(not modeled — no x87; MXCSR not exposed)* |

The actual entry `RIP`, page-table contents, and boot register file are committed
in code and covered by `machine_spec_hash`. The *(not built)* rows are tracked as
gaps in `Progress.md` (segmentation/TSS/IST and x87 are out of the current scope).

## 7. Guest physical memory map (e820, Freeze v1)

| Range | Type | Purpose |
|---|---|---|
| `0x0000_0000`–`0x0009_FC00` | usable | low RAM |
| `0x0009_FC00`–`0x000A_0000` | reserved | EBDA |
| `0x000A_0000`–`0x0010_0000` | reserved | legacy / VGA hole |
| `0x0010_0000`–RAM top (≤3 GiB) | usable | main RAM |
| `0xFEC0_0000` | reserved | IOAPIC |
| `0xFED0_0000` | reserved | HPET |
| `0xFEE0_0000` | reserved | Local APIC |
| `0xD000_0000`+ | reserved | virtio-mmio windows (fixed stride) |
| `0x1_0000_0000`+ | usable | RAM above 4 GiB (if memory > 3 GiB) |

Fixed load addresses: `boot_params` @`0x7000`, kernel cmdline @`0x2_0000`,
kernel load @`0x10_0000` (1 MiB, where main usable RAM begins), initrd high (just
below RAM top, page-aligned). All reserved/padding bytes in tables are zeroed
before checksum.

## 8. Change log

- **Freeze v1 (2026-06-11):** initial committed values. x86-64-v2 baseline, AVX
  off, 1 GHz TSC = instruction count, 128 MiB default RAM.
- **Freeze v1.1 (2026-06-12, Review 3 truthfulness pass):** CPUID reduced to the
  implemented ISA — leaf 1 ECX `0x8080_0000` (POPCNT+HYPERVISOR), EDX `0x070B_FBF1`
  (MMX cleared), leaf 7 zero (FSGSBASE/RDSEED dropped). Boot file synced to code:
  CR3/PML4 `0x1000` (was `0x9000`), kernel load `0x10_0000` (was doc `0x100_0000`),
  RSP = top of RAM. GDT/segment-descriptor/IDTR/x87 rows marked *(not built)*. FP
  noted as host `f64`/`f32` (host-independence partial), not yet soft-float.
