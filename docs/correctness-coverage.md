# Correctness & Determinism Coverage Map

The proactive completeness list: everything an x86-64 machine must model to run
real software *deterministically and correctly*, with current status and priority.
This replaces "wait for the next `#UD`" with a deliberate work-list. Validate each
item against real silicon (`tools/diff/`) and lock proven workloads into the
determinism gate (`scripts/determinism-gate.sh`).

Status: ✅ done · 🟡 partial · ❌ missing · ⛔ deliberately out of scope (now)

---

## A. CPU instruction set

| Family | Status | Notes / gaps |
|---|---|---|
| Base x86-64 integer/branch/string | ✅ | reactive gaps closed (POP r/m, moffs MOV, XADD restart, PUSH r/m64…) |
| x87 control (FNINIT/FLDCW/FNSTSW…) | ✅ | |
| SSE / SSE2 (scalar+packed, int, convert) | ✅ | |
| SSE3 / SSSE3 | ✅ | bit-exact vs silicon |
| SSE4.1 / SSE4.2 (incl. CRC32, PCMPxSTRx) | ✅ | bit-exact vs silicon (x86-64-v2) |
| CMPXCHG16B | ✅ | |
| **x87 FPU arithmetic** | 🟡 | stack+load/store/arith/compare/const done; **FPREM/FPREM1/FSCALE done** (exact fdlibm-style, proven by C `fmod`); **FCMOVcc done** (reuses integer `condition()`); **f64-modelled (not 80-bit)**; still missing transcendentals (FSIN/FCOS/FPTAN/FYL2X/F2XM1 — need committed soft-float polynomials), FBLD/FBSTP, FLDENV/FNSTENV/FRSTOR/FNSAVE |
| **AVX / AVX2 (VEX)** | 🟡 | **iter 1-3 done** (all **bit-exact vs silicon**, 31.5k differential vectors): YMM state + VEX C4/C5 decoder (maps 0F/0F38/0F3A); VZEROUPPER/ALL, moves (VMOVUPS/APS/DQU/DQA 128/256, VMOVD/Q); bitwise; FP arith (VADD/SUB/MUL/DIV/MIN/MAX PS/PD/SS/SD); packed-int add/sub; **compares (VCMPPS/PD/SS/SD 32-pred + VPCMPEQ/GT B/W/D/Q)**; **shuffles (VSHUFPS/PD, VPSHUFD/HW/LW, VUNPCK*, VPUNPCK*, VPSHUFB, VPALIGNR)**; **blends (VBLENDPS/PD, VPBLENDW, VBLENDVPS/PD, VPBLENDVB — 128-bit)**; **converts (VCVT PS/PD/DQ/SS/SD — 128-bit)**; min/max; shift-by-xmm (VPSLL/SRL/SRA); pack; VMOVMSK/VPMOVMSKB. **VEX NDD imm-shifts (group 71/72/73 incl VPSLLDQ/VPSRLDQ), VPMUL family**; **256-bit blends (VBLENDPS/PD/VPBLENDW + variable VBLENDV*) and converts (widening VCVTPS2PD/DQ2PD, narrowing VCVTPD2PS/PD2DQ, lane-preserving VCVTDQ2PS/PS2DQ)** — all bit-exact vs silicon via BOTH a 128-bit and a **256-bit YMM differential harness** (`cputest 256` / `--difftest256`; the harness caught a real VSHUFPD-256 imm bug). **broadcasts + lane-cross** (VPBROADCAST B/W/D/Q, VBROADCAST SS/SD/F128/I128, VINSERTF128/VINSERTI128, VEXTRACTF128/VEXTRACTI128, VPERM2F128/I128, VPERMQ/VPERMPD, VPERMD/VPERMPS) — all bit-exact vs silicon (256-bit harness). **FMA** (VFMADD/SUB/NMADD/NMSUB + ADDSUB/SUBADD, 132/213/231, PS/PD/SS/SD, 128+256) DONE — single-rounding via mul_add, bit-exact vs silicon. **AVX/AVX2 ISA complete; CPUID flipped to v3 (2026-06-18, Phase B).** Missing: gather (VGATHER), AVX-512 (⛔). NOTE: DPPS/DPPD excluded from differential (SDM-implementation-defined intermediate precision; not bit-exact across CPUs — like RCP/RSQRT). **CAVEAT:** userspace AVX not yet exercised — the guest kernel disables xstate (FXSAVE fallback) at leaf-0xD subleaf-1=0x3; enabling it (subleaf-1=0x1) hangs the boot in phase-sensitive early-init timing loops (Progress.md "B.4-pre") |
| **FMA + BMI1/BMI2 + F16C + MOVBE + LZCNT/TZCNT** | ✅ | rest of **x86-64-v3** — DONE, all bit-exact vs silicon (54.9k GPR + 75k F16C differential vectors). CPUID v3 level flipped. |
| MMX arithmetic (MM0-7) | ❌ | only the CPUID *bit* advertised (baseline gate); MM-reg ops would `#UD` |
| AES-NI / PCLMULQDQ / SHA | ❌ | crypto accel; CPUID-hidden so code uses SW fallbacks |
| RDRAND / RDSEED | ⛔ | entropy must come from virtio-rng (deterministic) |
| AVX-512 | ⛔ | not needed for target workloads |
| MONITOR/MWAIT, CLWB/CLFLUSHOPT, UMWAIT | 🟡 | treated as no-ops (no cache modelled) |

## B. Determinism-critical numeric modelling

| Item | Status | Notes |
|---|---|---|
| SSE FP = host f64/f32 (IEEE) | 🟡 | deterministic + correct on x86 host; **cross-arch needs soft-float** |
| **MXCSR rounding mode (RC)** | ❌ | always round-nearest — *wrong* if guest sets RZ/RU/RD (silent) |
| **FTZ / DAZ (denormal flush)** | ❌ | not modelled — wrong if guest enables |
| MXCSR / x87 exception status flags | ❌ | not updated (`fetestexcept` wrong) |
| **x87 80-bit precision + rounding** | ❌ | modelled as f64 → diverges from silicon for `long double` |
| x87 transcendentals = committed soft-float | ❌ | must NEVER use host libm (non-deterministic across libms) |
| NaN canonicalization | ✅ | `fp_result_f64/_f32` (dest-first quieted, x86 indefinite) |
| Soft-float library (cross-host/arch identity) | ❌ | §4 of cpu-determinism-contract; the master fix for B |

## C. Memory / MMU

| Item | Status |
|---|---|
| 4-level paging (4K/2M/1G), A/D writeback, #PF, CR2, W/U/NX | ✅ |
| TLB-less (always-coherent = deterministic) | ✅ |
| Decode cache + SMC/remap invalidation (gen + INVLPG + CR3) | ✅ |
| Within-page fast path + fetch translation cache | ✅ |
| SMAP / SMEP enforcement | 🟡 | not enforced (no observed dependency yet) |
| PCID / global-page semantics | 🟡 | TLB-less makes most of this moot |
| 5-level paging (LA57) | ⛔ | |

## D. Devices / platform

| Item | Status |
|---|---|
| LAPIC, IOAPIC, PIT(8254), RTC/CMOS, 16550 serial | ✅ |
| virtio-blk (event-queued completion + fault injection) | ✅ |
| virtio-rng (seed-derived) | ✅ |
| ACPI (RSDP/RSDT/FADT/MADT/DSDT) | 🟡 | minimal; FADT lacks PM blocks |
| **virtio-net** | ❌ | only in-guest loopback works; no external NIC |
| HPET, PCI(e) config space | ❌ | virtio-mmio used instead of virtio-pci |
| SMP / multi-vCPU | ⛔ | last (single-core interleaving first) |

## E. Privilege / system

| Item | Status |
|---|---|
| CPL transitions, IDT/GDT/TSS, RSP0, SYSCALL/SYSRET | ✅ |
| Precise-exception restart (faulting insn re-executes) | ✅ |
| CET (shadow stack / IBT) | 🟡 | ENDBR = NOP; not enforced (glibc marks IBT/SHSTK but doesn't require) |
| Debug registers / hardware breakpoints | 🟡 | stored, not enforced |
| Real/protected/v86 mode | ⛔ | long-mode-only by design |

## F. Determinism infrastructure

| Item | Status |
|---|---|
| Virtual time = instruction count (TSC/PIT/RTC derive) | ✅ |
| Snapshot / restore / divergence | ✅ |
| Seed-driven scheduling jitter (P4 first slice) | 🟡 | timer-IRQ jitter only; extend to IO/IRQ-delivery + futex wakeups |
| Fault injection (virtio-blk EIO/torn/jitter) | ✅ |
| Differential vs silicon — SSE/SSE2/SSE3/SSE4 | ✅ | `tools/diff/gen_sse.py`, ~14k vectors, 0 mismatch |
| Differential vs silicon — **base int, x87, flags** | ❌ | extend the generator (caught only via workloads today) |
| Determinism gate (golden digests, run after every change) | 🟡 | expanding to all proven workloads |

## G. Workload coverage matrix (the breadth proof)

| Workload | libc | Status | Golden |
|---|---|---|---|
| busybox / init | musl | ✅ | |
| Go runtime (goroutines, net/http, race) | musl(static) | ✅ | battle-test |
| CPython 3.12 | musl | ✅ | PY-HASH 3b2639986b6e8ada |
| C++ stress (SSE/STL/exceptions) | glibc static | ✅ | CPP-HASH 30021be53fa64a67 |
| **Node.js / V8 (JIT)** | glibc(v2) | ✅ | NODE-HASH 7b05e6d9b8415e3e |
| **C (static, libm/qsort/x87)** | glibc static | ✅ | C-HASH 353504a2bc8a29ab |
| Rust (static) | musl | 🟡 | runs natively; guest packaging blocked (no musl std / static-PIE loader) |
| PHP | glibc | ❌ | likely v2-ok |
| Java / JVM (C1/C2 JIT) | glibc | ❌ | needs **v3 (AVX2)** in userspace — blocked on B.4-pre |
| .NET / CoreCLR / NativeAOT (C#) | glibc | ❌ | **REQUIRES AVX2/FMA**; NativeAOT ELF built + QEMU-Haswell oracle CS-HASH 9217b151a2cb0087, but aborts on our machine until userspace AVX is enabled (B.4-pre blocker) |
| Redis / Postgres (services) | glibc/musl | ❌ | |

---

## Prioritized build order (proactive)

1. **Expand the determinism gate** to all proven workloads (CPython, Node, C++, battle-test) — lock the contract. *(in progress)*
2. **Rust + C static** workloads — quick wins, musl, no new ISA. Add to gate.
3. **x87 finish**: 80-bit soft-float precision + rounding + the missing transcendentals (committed algorithms) + FCMOVcc. Validate via a new x87 differential.
4. **MXCSR rounding/FTZ/DAZ + exception flags** — the silent-wrong-number class; the differential tester is the only way to catch these.
5. **x86-64-v3** (AVX/AVX2/FMA/BMI/F16C/MOVBE/LZCNT) — ✅ **DONE** (ISA implemented + differential-validated; CPUID flipped). **B.4-pre — enable userspace AVX:** the kernel won't boot with XSAVE/AVX fully enabled (subleaf-1=0x1) — it hangs in a chain of phase-sensitive early-init timing loops (clock calibration → `udelay` poll loop → …). The XRSTOR fpu-init fix cleared the first link. This is the current blocker for Java/.NET/C#. *(in progress)*
6. **Soft-float library** — the master fix for cross-host/cross-arch numeric identity (cpu-determinism-contract §4). Lets B items be both correct *and* host-independent.
7. **virtio-net** (then external services), **P4 seed-exploration breadth**, **SMP** — last.
