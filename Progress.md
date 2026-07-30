# Lockstep — Progress

Living status of the build. **This is the source of truth for "what's done, what's
in progress, what's queued."** Architecture/vision lives in
[`GOLDEN_TARGET.md`](GOLDEN_TARGET.md); the op/feature matrix in
[`docs/correctness-coverage.md`](docs/correctness-coverage.md); how to work in
this repo in [`AGENTS.md`](AGENTS.md).

Legend: `[x]` done · `[~]` partial · `[ ]` not started

> **Note (2026-07-20, "Lockstep" rename):** the project was renamed from its working
> title to **Lockstep**. This changed the guest-visible identity strings (CPU brand,
> ACPI OEM IDs) and internal seed/build-hash prefixes, which deliberately shifted the
> boot instruction stream — so the **golden digest/hash values quoted in older entries
> below predate the rename**. The live regression goldens are in
> [`scripts/concurrent-gate.sh`](scripts/concurrent-gate.sh), re-baselined at the rename
> with the determinism property re-proven (byte-identical run-to-run for all 10 runtimes,
> invariants hold, silicon differential still MATCHes, and the schedule-independent
> python + node hashes were **unchanged** — proof the computation itself did not change).

---

## What this is

A **from-scratch, deterministic x86-64 machine emulator** written in Rust — the
"Pure Mode" substrate for a Deterministic Simulation Testing (DST) engine. It is
its **own software interpreter** (CPU + 4-level-paging MMU + devices), **not**
QEMU/KVM (those are inspiration only; no hardware virtualization, no kernel
module — it runs as an ordinary user-space process). Single-vCPU, long-mode-only.

**The contract that must never break:** a run is byte-identical run-to-run,
controlled only by the 256-bit seed + machine spec + guest image + build — with
**zero** coupling to host wall-clock/RNG/timing/threading. Virtual time = retired
instruction count (1 instr = 1 ns). Golden *values* may re-baseline at a
deliberate, well-understood config change (CPUID flip, digest formula, a new
opcode that changes a boot's instruction stream), but the *property* is sacrosanct
and is re-proven at every such change.

---

## Current status (2026-06-23)

- **`cargo test` = 198 passing** (run on pcmini; see [CLAUDE.md](CLAUDE.md) — all
  build/test/boot happens on the `pcmini-x86` host, never locally).
- **Boots a real Alpine Linux kernel** → initramfs → `/init` → userspace at CPL3 →
  syscalls → poweroff, fully deterministically.
- **ISA: full x86-64-v3** (AVX/AVX2/FMA/BMI1/BMI2/F16C/MOVBE/LZCNT/XSAVE on top of
  the v2 set) + **MMX CPUID bit** (glibc baseline) + **x87 FPU** (FPREM/FSCALE/
  FCMOVcc). **CPUID v3 FLIPPED (2026-06-18)** — AVX/AVX2/FMA/BMI/F16C are advertised,
  decoded, and **differentially validated bit-exact vs real silicon** (the substantive
  work).
- **✅ USERSPACE AVX IS LIVE (B.4, 2026-06-23).** leaf-0xD subleaf-1 flipped `0x3`→`0x1`
  (plain XSAVEOPT) so the kernel keeps system xstate ENABLED — userspace now runs real
  XSAVE-backed YMM instead of the FXSAVE fallback. The `0x1` boot's old calibration hang
  was cured by **IOAPIC masked-at-reset** (a premature timer IRQ in the calibration loop).
  Reaching userspace needed a **VEX op batch** (saturating arith, avg, sadbw, pmovsx/zx,
  vptest, vpinsr/extr, vpblendd, vpermilps/pd, movddup, movntdq, VSQRTSS/SD/PS/PD — the V8
  runtime #UD was **VSQRTSD**, pinpointed via `SF_EXCTRACE=1`) and a **256-bit VPTEST fix**
  (Go's bytealg `memequal` compares whole ymm registers; a 128-bit-only VPTEST mis-set ZF →
  corrupted HTTP parsing → battle printed `400 Bad Request`; full-width VPTEST fixed it).
  **CPython + Node/V8 run real userspace AVX byte-identical to QEMU** (PY/NODE-HASH
  unchanged). All four gate workloads re-baselined at `0x1`, property re-proven byte-
  identical 2×, app-hashes preserved. This unblocks AVX-*requiring* binaries (C#/.NET, JVM).
- **Validated runtimes** (byte-identical to QEMU/host oracle + in the determinism gate):
  Go, C, C++, CPython 3.12, Node.js/V8, **.NET NativeAOT C# (AVX-requiring)**, **OpenJDK 17
  HotSpot JVM (JIT+GC threads)**, **PHP 8.3 (Zend)**, **Perl 5.38**, **Ruby 3.2**, **Rust 1.75**.
  See goldens below.
- **Differential conformance vs real silicon** (Alder Lake, `tools/diff/`):
  **128-bit** (`cputest`) and **256-bit YMM** (`cputest 256`) harnesses, ~40k+
  vectors/run, **0 mismatch** for every implemented op (DPPS/DPPD excluded —
  SDM-implementation-defined precision, like RCP/RSQRT).
- **Perf**: ~19 MIPS (2.6× via mem/fetch caching + `(rip,cr3)`+gen decode cache),
  determinism preserved.

### Determinism gate goldens (x86-64-v3 CPUID, USERSPACE AVX ON, re-baselined 2026-06-23)

Run after every change: `scripts/determinism-gate.sh [--full]`. **Re-baselined 2026-06-23 at
the userspace-AVX flip** (leaf-0xD subleaf-1 `0x3`→`0x1`: the kernel keeps system xstate enabled
→ every guest ifunc now selects true AVX/AVX2 paths). Every `state_digest` moved, but every
**app-level hash is unchanged** (correctness preserved) and each is byte-identical 2× — the
determinism *property* re-proven. battle's race outcome moved `-2 → -6` (still ∈ QEMU's set;
AVX-shifted instruction timing lands the interleaving differently).

| Workload | libc | App-level hash (correctness) | Whole-machine `state_digest` |
|---|---|---|---|
| battle-test (Go race) | musl | `SIMFORGE_RESULT=-6` (∈ QEMU's observed set {-6,-2,4}) | `b2f823c1870fe1f4` |
| C (libm/qsort/x87) | glibc | `C-HASH 353504a2bc8a29ab` | `6b0a7cecf0f04a0b` |
| CPython 3.12 | musl | `PY-HASH 3b2639986b6e8ada` | `9184307f3bbbc535` |
| Node.js / V8 | glibc | `NODE-HASH 7b05e6d9b8415e3e` | `47eb3ec7a18d0502` |
| **.NET NativeAOT C#** (AVX-requiring) | glibc | `CS-HASH 9217b151a2cb0087` | `52b081d72b959c98` |
| **OpenJDK 17 HotSpot JVM** (JIT+GC threads) | glibc | `JVM-HASH 111a5b05b6bfd07b` | `7474a05fe8e824ae` |
| **PHP 8.3 Zend interpreter** | glibc | `PHP-HASH 54e9eca6654b4761` | `8e3492eca47b662e` |
| **Perl 5.38** | glibc | `PERL-HASH 0008b4380008b438` | `ec51752f242ea212` |
| **Ruby 3.2** | glibc | `RUBY-HASH cae625523bbd1932` | `22adb5208ad582e8` |
| **Rust 1.75** (static) | static | `RUST-HASH bccae32f18672ce7` | `d1df7b6b2f03605b` |
| **concurrency-order** (pthreads) | static | `ORDER-HASH 6d15e3daa9690309` (seed1) | `87acda385ca3c8e8` |
| C++ stress | glibc | `CPP-HASH 30021be53fa64a67` | (not yet gated) |

---

## ROADMAP — in progress / queued

**Maintain this section.** When starting an item, move it to *In progress*; when
done, mark `[x]` and pull the next item up. (Mirrors the prioritized build order
in [`docs/correctness-coverage.md`](docs/correctness-coverage.md).)

### ✅ Recently done
- [x] **256-bit differential harness** — `cputest 256` (+ `trampoline.S` `run_test256`,
  emulator `--difftest256`, `gen_sse.py … 256`): loads `ymm0/ymm1` from a buffer,
  runs the op, stores `ymm0`. Caught + fixed a real bug (`VSHUFPD` 256 used the
  wrong imm bits in the high lane). Now in the gate as step [2b].
- [x] **256-bit blends** — `VBLENDPS/PD`, `VPBLENDW`, `VBLENDVPS/PD`, `VPBLENDVB`
  (imm spans lanes for PS/PD; per-lane for PBLENDW). Bit-exact vs silicon.
- [x] **256-bit converts** — widening (`VCVTPS2PD`/`DQ2PD`), narrowing (`VCVTPD2PS`/
  `PD2DQ`/`TPD2DQ`), lane-preserving (`VCVTDQ2PS`/`PS2DQ`/`TPS2DQ`). Bit-exact vs silicon.
- [x] **AVX broadcasts + lane-cross** — `VPBROADCAST{B,W,D,Q}`, `VBROADCAST{SS,SD,F128,I128}`,
  `VINSERTF128`/`VINSERTI128`, `VEXTRACTF128`/`VEXTRACTI128`, `VPERM2F128`/`VPERM2I128`,
  `VPERMQ`/`VPERMPD`, `VPERMD`/`VPERMPS`. Bit-exact vs silicon (the first genuinely
  lane-crossing ops; F128/I128 broadcasts are memory-source → unit-tested, not in
  the reg-only differential).
- [x] **FMA** — `VFMADD/VFMSUB/VFNMADD/VFNMSUB` + `VFMADDSUB/VFMSUBADD`, all three
  forms (132/213/231), PS/PD/SS/SD, 128 & 256. Single-rounding via `f64::mul_add`
  (matches the hardware VFMA exactly); NaN canonicalized in x86 source-priority
  order with the source sign preserved (a subtlety the differential caught — the
  product negation must not flip a propagated NaN's sign). Bit-exact vs silicon.
  **➡ With FMA done, the AVX/AVX2 instruction set is essentially complete.**

### ✅ Done — Flipped CPUID to x86-64-v3 (the big Phase-B transition)
This is what lets real Java/.NET/C#/PHP emit the AVX we built. Sub-steps:
- [x] **B.1 — XSAVE infrastructure** (Phase A, done): XCR0 state, `XGETBV`/`XSETBV`
  (0F 01 D0/D1), `XSAVE`/`XRSTOR` (0F AE /4,/5) saving x87+SSE+**YMM** state in the
  standard (non-compacted) format, CPUID leaf 0xD (XCR0 mask 0x7, AVX state @576).
  Advertises only *plain* XSAVE → kernel will use the simple format. Unit-tested:
  `XSETBV`→`XSAVE`→`VZEROALL`→`XRSTOR` restores the full 256-bit YMM. CPUID XSAVE/AVX
  bits still OFF (B.3), so zero perturbation in Phase A.
- [~] **B.2 — v3 GPR instructions**: **DONE** — LZCNT/TZCNT, MOVBE, BMI1 (ANDN/
  BLSI/BLSR/BLSMSK/BEXTR), BMI2 (BZHI/MULX/PDEP/PEXT/RORX/SARX/SHRX/SHLX), all
  bit-exact vs silicon incl. flags (54.9k GPR differential vectors; found+fixed
  MULX high/low swap). **Remaining: F16C** (VCVTPH2PS/VCVTPS2PH) — needs soft-float
  f16↔f32 conversion; bounded follow-up (not strictly blocking B.3 since binaries'
  AVX2 ifuncs check AVX2 directly, but v3 *requires* F16C for a clean advertisement).
- [x] **B.2 F16C** — VCVTPH2PS/VCVTPS2PH with soft f16↔f32 (all rounding modes,
  subnormals, NaN-quieting); bit-exact vs silicon (75k differential vectors).
- [x] **B.3 — CPUID flipped to v3 (2026-06-18)**: leaf1 +AVX/FMA/XSAVE/F16C/MOVBE
  (OSXSAVE dynamic from CR4), leaf7 +AVX2/BMI1/BMI2, ext +LZCNT; CPUID leaf 0xD +
  XSAVEOPT/XSAVEC (the kernel needs the **compacted** format — plain XSAVE alone is
  rejected). All goldens re-baselined + property re-proven byte-identical 2×, app
  hashes unchanged. *(One cosmetic kernel boot WARN — compacted size-consistency;
  non-fatal/deterministic; advertising XSAVES or a non-zero subleaf-1 EBX hangs early
  boot, so left as-is — TODO clean up.)*

### 🔄 In progress / next
- [x] **B.4-pre — USERSPACE AVX TRULY ENABLED (2026-06-23).** leaf-0xD subleaf-1 flipped
  `0x3`→`0x1` (plain XSAVEOPT): the kernel keeps system xstate ENABLED, so userspace runs real
  XSAVE-backed YMM instead of the FXSAVE fallback. The systemic blocker was the **IOAPIC reset
  state** — redirection entries reset UNMASKED (premature timer IRQ into early-init calibration/
  udelay loops); fixed by masking-at-reset (see below). With that, all four gate workloads boot
  to userspace byte-identically. Then two userspace gaps: (1) a **VEX op batch** for the AVX paths
  glibc/V8/Go now take — `vpinsr/extr`, `vextractps`, `vpblendd`, `vpmovsx/zx`, saturating arith
  (`vpadds/subs/addus/subus`), `vpavg`, `vpsadbw`, `vptest`, `vmovntdq`, `vmovddup`, `vpermilps/pd`,
  and `VSQRTSS/SD/PS/PD` (the V8 runtime #UD was **VSQRTSD**, pinpointed via `SF_EXCTRACE=1`); and
  (2) a **256-bit VPTEST correctness fix** — Go's bytealg `memequal` compares whole ymm registers,
  and a 128-bit-only VPTEST mis-set ZF → corrupted HTTP parsing (battle printed `400 Bad Request`);
  full-width VPTEST fixed it (battle → clean `-6`). **CPython + Node/V8 now run real userspace AVX
  byte-identical to QEMU** (PY/NODE-HASH unchanged); C unchanged; all re-baselined at `0x1`,
  property re-proven byte-identical 2×. C# oracle ready: NativeAOT ELF runs under QEMU-Haswell =
  **CS-HASH 9217b151a2cb0087** (drop `f16c` from `IlcInstructionSet`).
- [x] **IOAPIC masked-at-reset** (2026-06-22): redirection entries now reset with the mask bit (16)
  set, per the Intel IOAPIC spec — prevents premature timer-IRQ delivery into early-init loops. A
  real fidelity fix benefiting all boots; re-baselined the gate goldens (app-hashes preserved).
- [x] **AVX-enabled boot now REACHES USERSPACE (2026-06-22) — the IOAPIC fix advanced it all the way
  through kernel init.** Console confirms `x86/fpu: Supporting XSAVE feature 0x004: 'AVX registers'` +
  `Enabled xstate features 0x7, context size 832, 'standard' format` and `Run /init`. The APIC clock
  calibration completed (`APIC frequency too slow, disabling apic timer` — no hang). **Diagnostic
  breakthrough:** booted the kernel under QEMU with a kallsyms-dump initramfs (`nokaslr` → addresses
  match vmlinux) to resolve every mystery address — that's how the calibration loop was confirmed as
  `setup_boot_APIC_clock`, and the userspace crash was pinpointed. The init (busybox) then crashed on
  missing VEX *scalar* ops (our differential only tested packed/256-bit data ops). **Added 4 missing
  VEX op families** (all glibc/busybox AVX paths): `VMOVQ xmm,xmm/m64` (F3 0F 7E) + `VMOVQ xmm/m64,xmm`
  (66 0F D6); `VCVTSI2SD/SS` (F2/F3 0F 2A, int→scalar FP, upper from vvvv); `VLDMXCSR/VSTMXCSR`
  (0F AE /2,/3); `VUCOMISS/SD`+`VCOMISS/SD` (0F 2E/2F). All inert at `0x3` (glibc uses SSE2 there →
  goldens unchanged). **REMAINING (final blocker): a CORRECTNESS bug** — the C workload now runs glibc's
  FMA math (`__ieee754_pow_fma`, `__tan_fma`) with AVX and hits `*** stack smashing detected ***` (a
  wrong AVX result corrupts the stack), NOT a missing instruction. Next: validate the 4 new ops vs
  silicon (differential) + find which AVX op gives the wrong result (likely a scalar/FMA edge case or
  the new ops). **This is the last hurdle to the first AVX-enabled userspace workload.**
- [x] **FIRST AVX-ENABLED USERSPACE WORKLOAD WORKS (2026-06-23): C workload boots & passes with
  xstate/AVX enabled (`0x1`), C-HASH `353504a2bc8a29ab` — byte-identical to the SSE2 path.** The
  stack-smash was a real bug: VEX `VMOVSS/VMOVSD` (F3/F2 0F 10/11) were decoded as a *packed* 128-bit
  `VMov`, so `vmovsd %xmm,-0x28(%rbp)` wrote **16 bytes instead of 8**, clobbering the stack canary in
  glibc's `__cos_fma`. Fixed with a scalar VEX-move path (4/8 bytes; mem zero-extend/store, reg-reg
  merge from vvvv). Found via the kallsyms map + a stack-walk to `__stack_chk_fail`'s caller. Also
  added VEX scalar ops the workload needed: `vmovq`(F3 7E/66 D6), `vcvtsi2sd/ss`, `vcvttsd2si/ss`,
  `vstmxcsr/vldmxcsr`, `vcomisd/vucomisd`. All inert at `0x3` (glibc uses SSE2 there → goldens unchanged).
- [x] **PROACTIVE OPCODE ENUMERATION — built a decode-coverage probe (2026-06-23).** New CLI
  `--decode-probe <file>`: feeds hex instructions through the real decoder and reports which `#UD`.
  Disassembled all workload binaries (app/busybox/python3.12/libpython/node/libc/libm/libstdc++),
  extracted every VEX (c4/c5) instruction, deduped, probed → the COMPLETE gap list at once (vs reactive
  one-#UD-per-boot). **Most "missing" are DEAD CODE** — ifunc-gated on features we don't advertise:
  FMA4 (`vfmaddsd`/`vfnmaddsd`/`vfmsubsd`), AES-NI (`vaes*`), PCLMULQDQ (`vpclmul*`), AVX-512 mask/IFMA
  (`kmov*`/`kortest*`/`vpmadd52`). **REAL remaining AVX/AVX2 VEX gaps** (to boot Python + Node at `0x1`):
  `vpinsrd/q`, `vpextrd/q/w`, `vextractps`, `vpblendd`, `vpmovzx{bw,bd,wd,dq}`, `vpsubusb/w`, `vpaddsb`,
  `vpavgb`, `vpsadbw`, `vptest`, `vmovntdq`, `vmovddup`, `vpermilpd` — all have SSE equivalents to reuse
  (just need the VEX decode arm + vvvv/l256/YMM-zeroing). **Validation workflow:** re-run `--decode-probe`
  after each (instant coverage, no boot) then boot Python/Node once for correctness. Probe + extracted
  binaries left on pcmini (/tmp/vexm*.txt, /tmp/cextract, /tmp/pyx, /tmp/ndx, /tmp/kallsyms.txt).
- [x] **XRSTOR init-restore now fully matches FNINIT** (2026-06-19): the `xstate_bv=0`
  ("restore to INIT") path was resetting `fpu_cw/fpu_sw/st` but NOT `fpu_tw`/`fpu_top` — now
  it does (SDM-correct; inert at `0x3` where XRSTOR isn't used, so goldens unchanged). A real
  bug, but not the calibration root cause.
- [x] **RDRAND/RDSEED implemented** (Phase A, 2026-06-19): `0F C7 /6`,`/7`; deterministic
  seed-derived stream (`X86Cpu.rdrand_rng`, folded into digest once consumed); CF=1, flags
  cleared per SDM. NOT advertised in CPUID yet. Unit-tested (`rdrand_is_deterministic_and_sets_cf`).
- [x] **B.4 — .NET/CoreCLR C# RUNS (2026-06-23) — first AVX-REQUIRING runtime, in the gate.**
  The NativeAOT C# binary (`/tmp/csir.gz`, glibc) boots at `0x1` and prints `CS-HASH
  9217b151a2cb0087` (= the QEMU-Haswell oracle), `CS-EXIT 0`; byte-identical 2× (digest
  `52b081d72b959c98`). It passed the runtime's CPUID "required instruction sets" check (our full
  v3 satisfies NativeAOT) and ran to completion. The three gaps it hit were all **legacy SSE2**
  ops (not AVX) our other runtimes never emitted: **PEXTRW** (`0F C5`), **PINSRW** (`0F C4`),
  **PSHUFLW/PSHUFHW** (`F2/F3 0F 70` → new `Op::PshufW`, preserves YMM upper). All inert for the
  existing workloads (they pass without them). Added as gate step [7].
- [x] **B.4 — OpenJDK 17 HotSpot JVM RUNS (2026-06-23) — in the gate, ZERO new opcodes needed.**
  Packaged a `jlink`'d minimal `java.base` runtime + glibc + a deterministic pure-compute Java
  workload (`tools/workloads/jvm/`: IEEE strictfp float, `Arrays.sort`, string ops → FNV `JVM-HASH`;
  any conformant JVM gives the same hash, so the host JVM is the oracle). Boots at `0x1` →
  `JVM-HASH 111a5b05b6bfd07b` (= host OpenJDK 17.0.19), `JVM-EXIT 0`; byte-identical 2× (digest
  `7474a05fe8e824ae`). HotSpot's **JIT (C1/C2) + SerialGC run in background threads** and schedule
  deterministically under our instruction-count clock. **No missing opcodes at all** — HotSpot
  CPUID-gates (like V8), and the legacy SSE2 ops added for C# (PSHUFLW/HW etc.) already covered it.
  Gate step [8]. Reproducible build in `tools/workloads/jvm/build.sh`.
- [x] **B.4 — PHP 8.3 Zend interpreter RUNS (2026-06-24) — in the gate, ZERO emulator changes.**
  Boots at `0x1` → `PHP-HASH 54e9eca6654b4761` (= host PHP 8.3.6), `PHP-EXIT 0`; byte-identical 2×
  (digest `8e3492eca47b662e`). **No #UD and no emulator code change** — the only blocker was a
  *workload/environment* bug: php's `date` extension scans `/usr/share/zoneinfo` at startup, and in
  a minimal initrd (no zoneinfo) its absent-DB fallback **overflows a fortified buffer** (`*** buffer
  overflow detected ***`). Crucially this **reproduces identically under QEMU-Haswell**, proving it's
  php/env, *not* our CPU (our emulator faithfully matches real hardware — the whole diagnosis hinged
  on that). Fix: bundle `/usr/share/zoneinfo` + set `TZ=UTC`/`date.timezone=UTC`. Gate step [9].
  Workload + build in `tools/workloads/php/`.
- [x] **B.4 — Perl 5.38, Ruby 3.2, Rust 1.75 RUN (2026-06-24) — all in the gate, zero emulator
  changes.** Perl `PERL-HASH 0008b4380008b438`, Ruby `RUBY-HASH cae625523bbd1932`, Rust (static)
  `RUST-HASH bccae32f18672ce7` — each = its host oracle, byte-identical 2×. Ruby isn't installed on
  pcmini → fetched via `apt-get download` + `dpkg-deb -x` (no sudo) and bundled with its lib tree
  (gotcha: recreate the `libruby-3.2.so.3.2` SONAME symlink, the loader needs it not the `.3.2.3`
  file). Workloads in `tools/workloads/{perl,ruby,rust}/`. **Ten language runtimes now gated.**
- [x] **CLOSED-WORLD DST PROPERTY DEMONSTRATED (phase 0, 2026-06-24).** See
  `tools/workloads/dst/`. A multi-threaded program whose output is the **thread-interleaving order**
  (`order.c`, pthreads+mutex) is: byte-identical run-to-run for a fixed seed (gate step [13],
  `ORDER-HASH 6d15e3daa9690309`); and **explores different interleavings across seeds** with
  `SF_SCHED_JITTER` (seed1→`1dac…`, seed2→`d7ec…`, seed3→`90516…`), each reproducible. On the host
  the same program gives a different order every run. **Honest scope finding:** a pure SMP data race
  (`race.c`, unsynchronised shared counter) does **not** manifest on our single vCPU — it always
  reads the full count (no lost updates), because lost updates need two cores in the read-modify-write
  window *simultaneously*, and a single deterministic vCPU has no simultaneity. So we faithfully
  reproduce + explore **scheduling-mediated** concurrency (event/lock/handler order — the dominant
  real-microservice bug class), but **SMP simultaneous-access races need multi-vCPU modelling**
  (next DST milestone). Memory ✓ / loopback network ✓ (Go battle-test) / disk ✓ (`--disk`), all
  in-guest = closed world.
- [x] **REAL STATEFUL MICROSERVICE WORKLOAD (2026-06-30).** `tools/workloads/statesvc/` — a
  concurrent Go "bank": HTTP server on loopback, client goroutines firing `POST /transfer`,
  in-RAM accounts, periodic state snapshots to **`/dev/vda` (virtio-blk)** read back and verified,
  and a **conservation invariant** (total balance fixed, no negative, disk==memory). Transfers are
  lock-protected *with a sufficiency check* → non-commutative → order-sensitive. Result: byte-
  identical per seed (`7e08…`/STATE-HASH `506a…`); explores per seed (applied/failed 3372–3482,
  STATE-HASH `822e…`/`3a3a…`/`3be6…`); **invariant holds across every explored schedule**. Exercises
  memory + loopback network + disk + concurrency at once. (Gotcha: alpine `virt` kernel has
  `virtio_blk` as a **module** → `insmod virtio_blk.ko` in init, and bring `lo` up for loopback.)
- [x] **SEED EXPLORATION IS MULTI-AXIS (2026-06-30).** See `docs/exploration.md`. Beyond
  `SF_SCHED_JITTER` (CPU schedule) the substrate already exposes a **disk axis**:
  `SF_BLK_FAULT=seed,eio‰,torn‰,jitter` — seeded EIO injection, torn writes, and completion-
  latency/reordering. Demonstrated on statesvc: EIO and I/O-latency **change the outcome**
  (STATE-HASH `bfa6…`, `28d7…`) by shifting which goroutine wins the lock, deterministic under
  faults (torn-repeat identical), invariant preserved. **Principle:** we run the real kernel+apps,
  so nondeterminism can only be injected at emulated boundaries (CPU interrupts + virtio devices);
  the axes compose ({schedule}×{io-latency}×{io-fault}). **Biggest gap = network:** loopback is
  in-guest-kernel (no device), so network partitions/drops/latency need a **virtio-net** with a
  seeded fault policy — the highest-value next exploration axis for distributed microservices.
- [x] **CONCURRENCY DETERMINISM PROVEN LANGUAGE-AGNOSTIC (2026-07-01).** The stateful-service
  pattern (`tools/workloads/concurrent/`, common `SPEC.md`: loopback-TCP bank, concurrent clients,
  shared state, disk snapshots, conservation invariant) implemented in **10 languages**. **ALL 10 fully
  pass** (mem+net+disk+invariant) after the C# disk fix + the Node POPF/IF fix (both below); it's now the
  standing gate (`scripts/concurrent-gate.sh`). *(Originally 8/10 — C# disk + Node were the two hold-outs.)*
  Deterministic per seed + exploring per seed,
  across **both the thread+futex model** (Go, C, C++, Rust, Java, Ruby, Perl) **and the event-loop
  model** (Python asyncio): Go `1cf084…`, C `affd07…`, C++ `c18eff…`, Rust `fff7be…`, Java `482c0c…`,
  Ruby `ccda16…`, Perl `2a2267…`, Python `e423e8…`. **Two isolated caveats:** (a) **C#** — determinism
  + conservation + exploration pass, only `INV-DISK-MATCH` fails (.NET NativeAOT raw block-device I/O
  doesn't round-trip `/dev/vda`; fix = P/Invoke pwrite). (b) **Node HANGS on the emulator but runs
  fine on the host** — its multi-threaded (libuv threadpool + V8) loopback-networking event loop hits
  a **lost-wakeup** on our single-vCPU deterministic scheduler; Python's single-threaded asyncio
  dodges it. Real substrate finding, not a determinism failure. **KEY INSIGHT: event-loop runtimes
  (asyncio) are deterministic but need I/O-completion jitter to *explore* — CPU `SF_SCHED_JITTER`
  doesn't perturb epoll-driven interleaving.** (Note: pcmini `/tmp` was wiped mid-session by a
  shared-host cleanup — gate initrd artifacts pyir/nodeir/csir/jvmir/battle/etc. need rebuilding from
  `tools/workloads/*/build.sh`; the concurrent roots survived under `~/concurrent`.)
- [x] **Node FIXED (2026-07-04) — it was a `POPF`/IF CPU-correctness bug, NOT multi-vCPU.** *(Supersedes
  the 2026-07-02 "single-vCPU concurrency limit / fix=multi-vCPU" conclusion, which was WRONG — QEMU
  `-smp 1` runs the same node fine, refuting the theory.)* `Op::Popf` copied `IF`/`IOPL` from the popped
  value unconditionally; real x86 preserves `IF` in userspace (writable only when `CPL<=IOPL`) and `IOPL`
  outside ring 0. V8's Smi handlers do `push %rax; popf` with bit 9 clear → our emulator cleared userspace
  `IF` → the next demand-paging `#PF` tripped the guest kernel's `BUG_ON(userspace fault, IRQs off)`
  (verified `ud2` at kernel `0xffffffff8108b337`; `testb $0x2,0x91(%r14); je BUG` = pt_regs->flags IF)
  → oops → RC=139. **Fix** (`src/x86/exec.rs` `Op::Popf`): mask `IF` when `CPL>IOPL`, `IOPL` when `CPL≠0`.
  Node now runs **byte-identical to QEMU** (`CSVC-HASH 28d1c4de2c0b133b`). Found via a clean POSITIONAL
  instruction diff (ours vs QEMU) — NOT occurrence-counting bisection (which gave 3 wrong answers via
  loop-aliasing); when the userspace fault proved *correct*, a kernel-path positional diff found the
  `testb IF; je BUG` branch → the wrong IF → `POPF`. cargo test 198/198.
### QUEUED — reconciled & prioritized (2026-07-07)

- [x] **GATE RE-BASELINED + GREEN + STORAGE HARDENED (2026-07-07).** The old `/tmp` compute-gate initrds
  were unrecoverable (build recipes for battle/cir/pyir/nodeir/csir/rustir/orderir were never committed —
  only jvm/php/ruby/perl have `build.sh`). Built a **new persistent gate over the 10 surviving concurrent
  workloads** (`~/concurrent/*/root`) — a stronger gate (mem+net+disk+conservation + ours==QEMU cross-
  check): `scripts/concurrent-gate.sh` = cargo test + SSE/AVX-256 differential-vs-silicon + each of 10
  langs run seed1 2× asserting byte-identical + `INV-CONSERVED` + `INV-DISK-MATCH` + golden hash. **Result:
  198 tests, SSE-MATCH, YMM-MATCH, 10/10 langs PASS** (re-baselined goldens for 6.12.94 + POPF; Python
  `e423e8…` unchanged = correctness proof; Node `28d1c4…` == QEMU oracle). C# needed `LD_LIBRARY_PATH=
  /glibc` for its NativeAOT `/svc`. **Storage fix (root of the recurring `/tmp` wipes):** all durable
  artifacts moved OFF `/tmp` into **`~/sf/`** (home): `kernel/vmlinuz-virt-6.12.94`+`virtio_blk.ko` (the
  matched pair was ONLY in `/tmp`!), `deps/busybox`, `bin/sfhv`, `gate/csvc_<lang>.gz` ×10,
  `concurrent-gate.sh`, `README-RECOVERY.md`. A `/tmp` wipe now only loses scratch. Rule: kernel/initrds/
  binary → `~/sf`, never `/tmp`.
- [x] **0. C# disk FIXED + VERIFIED (2026-07-02).** Replaced .NET `FileStream`/`RandomAccess` (don't
  round-trip a raw block device) with libc **P/Invoke `open`/`pwrite`/`pread`/`fsync`** in `Svc.cs` →
  `INV-DISK-MATCH true` (diskTotal=2400), deterministic A==B (`4213de62899ffa26`). **C# is now 10/10
  fully green** (mem+net+disk+invariant). The earlier "disk fail" was purely a host artifact: the
  `/tmp`-wipe recovery restored a **6.12.81** kernel while `virtio_blk.ko` was **6.12.93** → vermagic
  mismatch → insmod silently failed → no `/dev/vda` (also explains node's `/dev/vda` ENOENT — a red
  herring, unrelated to node's real networking bug). Fixed by fetching a **matched alpine linux-virt
  6.12.94** apk (kernel + module from the same package). ⚠️ **Gate goldens were baselined on 6.12.93 →
  the kernel is now 6.12.94, so the gate needs RE-BASELINING (deliberate kernel bump) + all wiped gate
  initrds rebuilt from `tools/workloads/*/build.sh`.** Reusable recovery: `~/sfhv` = stashed emulator
  binary (survives `target/` wipes); matched kernel+module via the alpine apk.
- [x] **1. virtio-net + seeded network-fault policy — DONE (2026-07-07), Phase 1a + 1b.** Two real guest
  kernels talk over an emulated deterministic switch, with seeded network fault injection. **Phase 1b
  (two-guest switch) VALIDATED:** `--net2` harness (`run_net2` in main.rs) boots two `X86Machine`s (each
  `--netdev`, reflector OFF), steps them round-robin by a fixed instruction quantum, and runs a
  deterministic SWITCH between quanta (drain each NIC's TX, seeded drop/partition, deliver survivors to
  the peer's RX). Guest A (10.0.0.1) **pinged guest B (10.0.0.2): 3/3 replies, 0% loss** (ARP + ICMP both
  crossed). DETERMINISM: byte-identical 2× (digests A=6a40736c… B=7302a968…). FAULT: `--net-fault
  seed,drop‰[,part_start,part_end]` → 100% drop = total loss (dropped=17); seeded 50% drop byte-identical
  2× (A=a025d05a…) AND digest differs from no-fault → faults reproducibly change the outcome = the
  exploration axis. Per-guest serial printed separately (A|/B|) to avoid interleaving. Test initrd needs
  proc/sys/dev dirs (mount fails silently otherwise). Uses the matched 6.12.95 kernel+virtio_net.ko in
  ~/sf/nettest/ (gate stays on 6.12.94). *Phase 1a (below) built the device.* **Full DST network axis is
  now live** — next: richer fault policy (latency/reorder/duplicate, asymmetric partitions), real
  workloads (client A ↔ server B, e.g. the concurrent bank-service split across two nodes), and the
  event-loop I/O-completion jitter that lets asyncio/libuv explore.
  **DISTRIBUTED WORKLOAD VALIDATED (2026-07-07):** the concurrent bank-service SPLIT across two nodes —
  `tools/workloads/dist/distsvc.c` (one static binary; `distsvc server` binds the NIC on node B,
  `distsvc client <peer>` drives 6×500 transfers from node A over TCP). Client A ↔ server B across the
  switch: **all 3000 transactions processed, INV-CONSERVED true (total=2400)**. DETERMINISTIC byte-
  identical 2× (hash 5bfc480a…). FAULT-TOLERANT: under 10‰... er 100‰ drop the invariant STILL HOLDS
  (TCP retransmits; applied 2129/871, hash ea058d25…, byte-identical 2×); under a MID-RUN PARTITION
  (total link cut for an instruction window) the invariant STILL HOLDS (applied 2155/845, hash e949e883…).
  So money is conserved across a clean link, packet loss, AND a partition — each deterministic, each a
  distinct hash (exploration). This is the DST end-goal: a real distributed stateful service whose safety
  invariant is machine-checked across every reproducible network schedule/fault. distsvc gotcha: server
  polls done_conns with a ~40s virtual-time cap so a partition can't hang the run. Test initrd ~/sf/nettest/
  bank.gz (roles via `sf.role=server|client`, `sf.peer=`).

  **ALL 10 LANGUAGE RUNTIMES DISTRIBUTED — 10/10 GREEN (2026-07-08):** ported the bank-service to
  server/client roles in every runtime (`tools/workloads/dist/{c,cpp,go,rust,java,python,node,ruby,perl,csharp}`,
  gate `scripts/dist-gate.sh`). Each runs as two guest kernels (node A=client, node B=server) over the
  virtio-net switch; each asserts **all 3000 requests processed + INV-CONSERVED true + deterministic (run1==run2)**.
  Results (applied/failed, seed=1, 1024MB/guest): c 2133/867 `1e5c6693…`, cpp 2143/857 `5f0050d5…`,
  go 2126/874 `cdd70396…`, rust 2156/844 `7bf9d7e5…`, python 2139/861 `e423e849…` (== its concurrent
  golden — single-thread asyncio is schedule-independent, so loopback vs two-node give the identical final
  state, a clean cross-check), perl 2179/821 `bbb45f14…`, ruby 2116/884 `8b23e3b6…`, node 2133/867
  `95521738…` (**node passes distributed** — the concurrent-mode libuv loopback lost-wakeup does NOT bite
  on the real two-node device path), java 2096/904 `b933db1d…`, csharp 2154/846 `3ddb8397…`. The initrds
  reuse each runtime's userspace from `~/concurrent/<lang>/root` + the 6.12.95 net modules + a two-node
  `/init` that dispatches on `sf.role`. **C# needed an entropy source:** .NET blocks the idle server on
  `getrandom()` until crng seeds, and on two nodes the busy client seeds ~80s earlier and finishes before
  the server can process (→ applied 0). Fix = the deterministic **virtio-rng** device: `insmod rng-core +
  virtio-rng`, boot with `--rngdev` (new flag emits the rng `virtio_mmio.device` token), moved rng to its
  **own GSI 7 / vector 0x32** (was sharing net's GSI 6/0x31). crng then seeds at 0.85s symmetric →
  csharp processes all 3000, conserved, deterministic. Bytes are seed-derived so determinism is preserved;
  cargo test=198. (Diagnosis gotcha: my rng test initrd had dropped `lib64/ld-linux`, so the NativeAOT
  binary silently crashed — the "networking broke" was a red herring; instrument `/init` with echo markers.)

  **EVENT-LOOP I/O-COMPLETION JITTER — asyncio & libuv now EXPLORE (2026-07-08).** New knob
  `--net2 --net-jitter <seed>[,<max_instr>]` (main.rs `parse_net_jitter`) sets a seeded RX-delivery jitter
  policy (`NetFaultPolicy::new(seed,0,max)`, drop=0) on both guests, so each reorders its own incoming
  frames by up to `max` instructions. This is the ONLY nondeterminism a single-threaded event loop exposes:
  `SF_SCHED_JITTER` (timer/CPU) can't perturb a loop with no thread interleaving, but *which socket epoll
  reports readable first* crosses the virtio-net boundary. Reordering RX → the loop processes requests in a
  different order → a different set of transfers wins → different final state, invariant still holds.
  Demonstrated (dist bank, `--seed 1`, varying only `--net-jitter`): **python asyncio** baseline `e423e849`,
  seed1 `93b0cf12` (==repeat, deterministic), seed2 `dc717ad5`, seed3 `13f625e2`; **node libuv** baseline
  `95521738`, seed1 `a9d8261e` (==repeat), seed2 `cfe3d3b3`, seed3 `5c4bc82e` — **4 distinct schedules per
  runtime, each byte-reproducible, all INV-CONSERVED true**. The event-loop analogue of timer-jitter, reached
  through the I/O surface. Composes with `--net-fault` (drop/partition) and `SF_SCHED_JITTER`. Gotcha:
  node runs are heavy — an EMPTY result is a wall-clock `timeout` under host load (per CLAUDE.md), not a
  divergence; re-run longer. Mechanism already existed in `virtio_net.rs` (`deliver_frame` applies
  `fault.next()` extra latency); this just wires a jitter policy onto the net2 guests. See docs/exploration.md.

  **DETERMINISM GATE RE-GREEN + HARDENED (2026-07-10).** The rng/net-jitter binary change surfaced a
  PRE-EXISTING harness flaw: `concurrent-gate.sh` rebuilt each initrd per run with `find . | cpio -H newc`,
  which embeds inode numbers + mtimes and tracks `find` order → every rebuild = different initrd bytes →
  different tmpfs-extraction timing → a different thread PREEMPTION point → shifted CSVC-HASH for the 8
  THREAD-model workloads. NOT an emulator determinism regression: proven by det=ok (byte-identical
  run-to-run) everywhere, identical-source rebuild → byte-identical binary → identical hash (builds
  reproducible), and BOTH single-thread event loops (python `e423e849`, node `28d1c4de`, schedule-INDEPENDENT)
  matching EXACTLY through every build. `find|sort` alone doesn't fix it (cpio metadata still varies). FIX =
  **pin pre-built initrds**: `scripts/pin-initrds.sh` freezes 10 artifacts in `~/sf/gate/pinned/<lang>.gz`;
  the gate REUSES them (no per-run rebuild). Re-baselined the 8 multi-threaded goldens (c 38ab1994, cpp
  1a0ce8ae, go 200a146d, rust dd7309b0, java 33559d5f, ruby 32ee7ffd, perl f654076d, csharp 7ed4facd;
  python/node unchanged). **Gate GREEN + STABLE: confirmation re-run reproduced all 10 exactly →
  `GATE_CONCURRENT_FAIL=0`** (198 tests, SSE-MATCH, YMM-MATCH). Re-pin only when a workload changes.
  Phase 1a (device) details: Built the device
  `src/x86/virtio_net.rs` (legacy virtio-mmio, **RX+TX queues**, fixed MAC via VIRTIO_NET_F_MAC, 10-byte
  net-hdr, deferred RX delivery in virtual time) + `NetFaultPolicy` (SplitMix64: drop_permille + jitter,
  mirrors BlkFaultPolicy). Wired into the machine as `Option<VirtioNet>` — **`None` by default so the
  gate is byte-identical** (digest/MMIO/cmdline only change with `--netdev`). New `--netdev` flag (attaches
  NIC at 0xd0002000:6, appends the cmdline entry) + `SF_NET_FAULT=seed,drop‰,jitter` + `SF_NETIOTRACE`.
  Phase-1a backend = a deterministic **loopback reflector** in the machine step (TX→fault policy→RX).
  **VALIDATED:** matched alpine 6.12.95 kernel+virtio_net.ko in `~/sf/nettest/` (alpine bumped 6.12.94→95;
  used a separate 6.12.95 set for the NET test so the gate stays on 6.12.94) → guest discovers+binds the
  NIC, `eth0` UP/LOWER_UP, **TXdrain=9 / RXdeliver=9** (every transmitted frame — IPv6 RS/MLD + ARP —
  reflected back with correct net-hdr). cargo test 198/198. **NEXT = Phase 1b:** two `Machine` instances +
  a deterministic switch stepping them in lockstep by instruction-count, routing frames between their NICs,
  with drop/latency/reorder/**partition** — the real distributed-DST harness. *(merges former "virtio-net" ×2)*
- [ ] **2. Property/assertion layer + seed-exploration breadth (P4)** — an in-guest assertion channel
  (flag invariant violations to the harness) + a seed sweep/explorer that searches the
  schedule×fault space and shrinks failing repros; jitter more boundaries (LAPIC/futex/IO). Turns the
  substrate from a *regression gate* into an Antithesis-style *bug-finder*.
- [x] **3. Numerical fidelity (the "silent wrong number" + cross-host class)** — COMPLETE (3a/3b/3c/3d all
  done + silicon-validated + gate-confirmed). SSE/AVX directed rounding + FTZ/DAZ + exception flags; full x87
  80-bit soft-float (arith/subnormals/constants/FSCALE/FPREM) + transcendentals incl. Payne-Hanek; cross-host
  numeric-identity contract met by construction (docs updated). Remaining niches: physical aarch64 run,
  FPREM gap≥64 (impl-defined).
  - [x] **3a — MXCSR rounding-control (RC) for SSE/AVX ADD/SUB/MUL/DIV/SQRT (2026-07-13).** Honour the
    directed modes (RC bits 13:14: −inf/+inf/toward-0) that the host always computes as round-nearest.
    Technique (no soft-float): take the host nearest result `n` + the EXACT residual sign (Knuth `two_sum`
    for add/sub, fused `mul_add` for mul/div/sqrt — deterministic + identical on any IEEE host incl aarch64)
    and step to the adjacent representable value. `fbin_rounded_{f64,f32}` / `sqrt_rounded_{f64,f32}` +
    `apply_rc_{f64,f32}` / `next_{up,down}_{f64,f32}` in exec.rs; threaded `rc` through `fbin_lanes`/
    `sqrt_lanes128` + call sites. Overflow-under-directed clamps to max-finite; add/sub exact-zero is −0 under
    round-toward−inf (IEEE §6.3). **Phase-A / inert at default (RC=0 → no-op), so gate goldens byte-unchanged.**
    VALIDATED bit-exact vs silicon: `gen_sse.py` emits RC=1/2/3 variants (wrap op with `ldmxcsr`(mode)…`ldmxcsr`
    (default); emulator `--difftest` sets RSP scratch for the stack `ldmxcsr [rsp]`) — **110,100 vectors MATCH**
    (default + all 3 modes, ps/pd, scalar+packed). Unit test `mxcsr_directed_rounding_div_sqrt`. cargo test=199.
    **MUL-underflow corner CLOSED (2026-07-13):** f64 via a scaled-Dekker residual (`frexp2`/`ldexp`/
    `mul_resid_sign` — normalise both operands, exact product vs n in a common domain); f32 via the always-
    exact f64 residual (f32×f32 ≤48 bits). Validated bit-exact incl. UE/PE flags (`gen_sse.py emit_mul_uf`).
    Full differential now **173,880 vectors MATCH**. Only **divide-by-subnormal** directed rounding remains
    deferred → 3d (rare; a subnormal divisor makes the `x−n·y` residual round).
  - [x] **3b — FTZ/DAZ + MXCSR exception-flag status bits (2026-07-13).**
    - **FTZ/DAZ:** DAZ (bit 6) treats subnormal INPUTS as sign-preserved 0; FTZ (bit 15) flushes subnormal
      arithmetic RESULTS to sign-preserved 0. `daz_{f64,f32}` / `ftz_{f64,f32}_bits`; `fbin_lanes`/
      `sqrt_lanes128` take `mxcsr` (derive rc/ftz/daz). Default OFF → inert.
    - **Exception flags:** the arithmetic/sqrt helpers now return `(bits, exceptions)`; IE/DE/ZE/OE/PE are
      OR'd (sticky) into `self.mxcsr` at each FP op site. IEEE micro-rules the differential pinned down:
      invalid = SNaN input or fresh NaN (inf−inf/0·inf/0÷0/inf÷inf/sqrt<0); ZE = finite÷0; OE = finite-operand
      overflow (excl. ÷0), implies PE; #D fires for a denormal source UNLESS a NaN operand, a ÷0, or a
      negative sqrt determines the result (inf partner does NOT suppress). **UE (underflow) + its coupled PE
      are DEFERRED to 3d** (same subnormal-residual limitation as 3a).
    - **Validated bit-exact vs Alder Lake: 168,120 vectors MATCH** (RC + FTZ/DAZ + exc IE/DE/ZE/OE/PE;
      flags read back via `stmxcsr`→ECX, UE masked). Exception flags fire at DEFAULT mxcsr but are
      **app-invisible** — python concurrent CSVC-HASH `e423e849` + INV-CONSERVED unchanged (the gate keys on
      app hashes, not the mxcsr-mixing machine digest). cargo test=199.
  - [~] **3c** — x87 fidelity.
    - [x] **FCW rounding-control (2026-07-13):** x87 arith (FADD/SUB/MUL/DIV incl. reverse forms) + FSQRT +
      FRNDINT now honour the FCW RC field (bits 10:11) — previously always round-to-nearest. Reuses the SSE
      directed-rounding core (`fpu_apply(op,a,b,rc)` → `fbin_rounded_f64`; FRNDINT → floor/ceil/trunc/nearest).
      `rc==0` (reset default) takes the raw host path → **byte-unchanged, gate-safe**. Unit `x87_rounding_control`,
      cargo test=200.
    - [x] **FCW precision-control + true 80-bit extended precision (2026-07-13)** — DONE. Built isolated
      to de-risk, then wired the whole stack.
      - [x] **F80 soft-float library (`src/x86/f80.rs`):** `F80{sign,exp:i32,mant:u64,class}` (value =
        ±mant·2^(exp−63), integer bit at 63), deterministic INTEGER arithmetic (never host f64 rounding →
        cross-host identical). Complete: exact f64↔80-bit + m80-memory + i64 conversions (round-trip incl.
        subnormals/specials), `round128` (correct-rounding to PC=24/53/64 bits with RC directed modes),
        `mul` (64×64→128), `add`/`sub` (exponent-aligned, exact cancellation via borrow), `div` (long
        division with explicit round-bit from `2·rem` vs divisor + sticky), `sqrt` (integer Newton floor +
        rem-based rounding). **Extended precision DEMONSTRATED:** `1.0 + 2^-60` is exact in 80-bit
        (mant `…0008`) but rounds to 1.0 in f64; PC=53 mode reproduces f64. 6 f80 unit tests.
      - [x] **Stack rewired** `st:[f64;8]` → `[F80;8]`: all x87 loads/stores/arith/FSQRT/FXCH/FCHS/FABS/
        FCMOV/FXTRACT now compute in true 80-bit at the FCW's PC (24/53/64) and RC; digest mixes `st` via
        `to_m80`. f64-bridge helpers kept for the f64-typed paths. All existing FPU tests still pass.
      - [x] **x87 80-bit SILICON DIFFERENTIAL (`gen_x87.py` + `cputest x87` + `--difftest-x87`):** same
        FLD/op/FSTP m80 blob on our soft-float AND real Alder Lake x87 → **2000 vectors add/sub/mul/div/sqrt
        MATCH bit-for-bit** at default FCW (PC=64). x87 basic arith is deterministic across CPUs, so silicon
        is a valid oracle (unlike the impl-defined transcendentals). This caught + fixed a div round-bit
        off-by-1 (round bit was lost when quotient < 2^64; now derived from `2·rem`). cargo test=207.
      - [x] **Gate re-proven on the F80-complete binary (2026-07-14):** installed the fresh build as
        `~/sf/bin/sfhv` and ran the full concurrent gate → **10/10 PASS, `GATE_CONCURRENT_FAIL=0`**, every
        runtime `golden=ok` + conserved/disk/det ok, and **all hashes byte-identical to the pre-rewire
        values** (node `28d1c4de`==QEMU oracle, python `e423e849` anchor unchanged). The 80-bit stack rewire
        is **behavior-neutral** — internals changed, observable output did not.
      - [ ] **deferred to 3d:** 80-bit subnormals (currently flush), div-by-subnormal directed rounding,
        80-bit constants, FPREM/FSCALE full-precision, large-arg trig reduction (Payne-Hanek).
    - [x] **transcendentals (2026-07-13):** F2XM1/FYL2X/FYL2XP1/FPTAN/FPATAN/FXTRACT/FSINCOS/FSIN/FCOS (D9
      F0-FF) — were `#UD`, now decode + execute. **DETERMINISTIC soft-float** (committed constants + f64
      Taylor/atanh polynomials, NEVER host libm → identical on any IEEE host incl. aarch64): `soft_log2`
      (frexp + atanh series), `soft_exp2m1` (expm1, no −1 cancellation), `soft_sincos` (Cody-Waite reduce to
      ±π/4 + Taylor), `soft_atan2` (π/8-reduced Taylor + quadrant). New `Op::Ftrig{FpuTrig}`. Accurate to
      ~1e-13 (impl-defined on silicon, so validated by **determinism + accuracy**, not bit-exactness). Unit
      `x87_transcendentals_deterministic`, cargo test=201, decode-probe 9/9 ok. **ADDITIVE → gate unaffected**
      (no workload emits them). Large-argument trig reduction (|x|≳2^20, Payne-Hanek) is a follow-up.
  - [~] **3d** — cross-host/cross-arch numeric identity + the deferred subnormal corners.
    - [x] **3d-1 — SSE div-by-subnormal + underflow directed rounding (2026-07-14).** Differential-first:
      added `emit_uf_dir` to `gen_sse.py` (add/sub/div, all RC, denormal-heavy operands → results in the
      subnormal grid, compares result AND flags) → probe showed **DIV-only** breakage (2489 mismatches),
      NONE for add/sub. Root cause: the fused `(-n).mul_add(y,x)` div residual **underflows to 0** for a
      subnormal divisor, losing both the sign (→ wrong directed result, 163 cases) and the inexactness
      (→ missing PE flag, 328 cases). Fix: **f32** div computes the residual exactly in **f64** (48-bit
      product, f64 range dwarfs f32 subnormals — no underflow); **f64** div uses a new exact integer
      `div_resid_sign` (compare |x| vs |n·y| by **MSB-normalization** — top-bit exponent then normalized
      u128 mantissa, so it's exact even when a 1-bit subnormal x's exponent is far from n·y's; the first
      fixed-threshold-guard attempt wrongly bailed on `shx=103` and left 131 off-by-1s). **FULL DIFFERENTIAL
      MATCH: 339,840 vectors** (seeds 1–8) incl. all div ufdir + flags; x87 diff still 400 MATCH; cargo
      test=208 (new `div_subnormal_directed_rounding`). Result-neutral at default rc=0 (only the PE/UE flags
      newly fire, app-invisible like 3b) → **GATE 10/10 PASS, `GATE_CONCURRENT_FAIL=0`, all golden=ok**.
    - [x] **3d-2 — F80 80-bit subnormals (2026-07-15).** `round128` + `to_m80` flushed sub-normal-range
      results to zero; now they round onto the 80-bit subnormal grid (ULP 2^-16445). `round128`: for
      `exp < -16382`, `round_shift_u64(mant, rsh, sign, rc)` rounds `mant·2^(exp+16318)` to the grid with
      RC; normalization then covers both true subnormals AND round-up-to-smallest-normal uniformly.
      `to_m80`: subnormals denormalize (integer bit clears, exp field 0) instead of flushing. `gen_x87.py`
      grew `rand_m80_sub` (genuine subnormals + tiny normals) → **x87 DIFFERENTIAL 3000 vectors MATCH incl.
      subnormal** vs Alder Lake; cargo test=209 (new `f80_subnormals`).
    - [x] **3d-3 — F80 constants + FSCALE + FPREM/FPREM1 (2026-07-15).** All were f64-based. Now:
      **80-bit constants** (FLDPI/FLDL2E/… use the canonical extended encodings, not f64-truncated);
      **FSCALE** = `F80::scale_by` (exact exponent shift, over/underflow via subnormal-aware `round128`);
      **FPREM/FPREM1** = `F80::fprem` (exact 80-bit remainder + quotient C-bits for exponent gap < 64,
      C2=0; deterministic partial reduction + C2=1 for gap ≥ 64, which is implementation-defined per SDM so
      excluded from the strict differential). The FSCALE differential also caught a latent **`to_i64` bug**
      (`>> 64` at `e==-1`) that was silently corrupting FIST/FISTP for |value|∈[0.5,1) — fixed (now handles
      all `|value|<1`). **x87 DIFFERENTIAL 3921 vectors MATCH** (arith+subnormal+consts+FSCALE+FPREM/1);
      cargo test=211 (new `f80_scale_and_fprem`, `f80_constants`). Combined 3d-2+3d-3 gate re-proving.
    - [ ] **3d-3 follow-up:** FPREM gap ≥ 64 partial-quotient exactness vs a specific silicon (impl-defined).
    - [x] **3d-4 — Payne-Hanek large-argument trig reduction (2026-07-15).** `soft_sincos` now uses
      Cody-Waite for |x| < 2^20 and **Payne-Hanek** beyond: a 1344-bit `IPIO2` table (2/π in 24-bit
      chunks) + `twopi_bits` window + a windowed big-integer multiply (`payne_hanek`) computes
      x·2/π mod 4 and the reduced r ∈ [−π/4, π/4] EXACTLY in the integer domain — accurate at any
      magnitude, deterministic, no host libm. Validated (transcendentals are impl-defined → no silicon
      oracle) by **accuracy vs libm reference across |x| = 10^7…2^900** (13 sin/cos values within 1e-11)
      + sin²+cos²=1 invariant + determinism. cargo test=209 (added `payne_hanek_large_arg_trig`; also
      removed the now-dead f64 `fprem_core`/`fscale_f64` helpers + their 3 tests). Additive to the x87
      transcendental path (no workload emits FSIN/FCOS) → gate-safe.
  **3d COMPLETE** — all five sub-items (3d-1/2/3/4/5) done and validated. The numerical-fidelity
  milestone (item 3) is fully closed. **Final gate 10/10 PASS, `GATE_CONCURRENT_FAIL=0`, all golden=ok
  + det=ok** on the complete binary (subnormals+constants+FSCALE+FPREM+Payne-Hanek), byte-identical to
  pre-3d goldens → behavior-neutral.
- **Perf instrumentation baked in (2026-07-15):** `run_boot` prints `=> elapsed_ms=/mips=` (wall-clock
  diagnostic, determinism-safe — never in state/digest); `concurrent-gate.sh` appends per-workload
  `instr=/time=/mips=/load1m=` + a `PERF-TOTAL` line, so every future gate auto-reports throughput with
  host-load context. First numbers (under load ~32): ~7 MIPS most runtimes, **Rust 91 MIPS** (compute-bound,
  cache-hot), total 64.2B instr / 94 min / 11.4 MIPS aggregate. Instruction counts byte-identical across
  runs (deterministic virtual time); wall-times drift with load.
    - [x] **3d-5 — cross-host numeric-identity contract (2026-07-15).** Rewrote `docs/cpu-determinism-
      contract.md` §4 + §13: the formerly-deferred items (RC, FTZ/DAZ, exception flags, x87 80-bit +
      transcendentals) are all now DONE and validated bit-exact vs silicon, so cross-*architecture* identity
      is **MET by construction** (every FP path is an IEEE-basic-op in round-nearest — bit-identical on any
      conforming host incl. aarch64 — or our own soft-float). Claim ceiling: not yet run on a physical
      non-x86 host; and the Payne-Hanek accuracy item (3d-4) remains (deterministic, not host-dependent).
- [x] **4. Multi-vCPU / SMP deterministic interleaving** — model >1 vCPU with deterministic
  instruction-granularity interleaving (cooperative serialization, x86-TSO, DPOR/PCT). Unlocks the SMP
  simultaneous-access data-race class (`tools/workloads/dst/race.c` — no lost updates on 1 vCPU).
  **All four stages delivered** (Stage 0 boot → Stage 1 seeded interleaving → Stage 2 x86-TSO →
  Stage 3a PCT + 3b DPOR, below); named follow-ups (DPOR-over-TSO, sleep sets, coverage steering)
  are extensions, not gaps.
  **SCOPED (2026-07-16):** design in [`docs/smp-determinism.md`](docs/smp-determinism.md); concrete staged
  build plan in [`docs/smp-implementation-plan.md`](docs/smp-implementation-plan.md) — Stage 0 (boot N
  vCPUs: enumeration + `Vec<X86Cpu>` + ICR/INIT-SIPI AP startup + round-robin loop, SC lock-step) → Stage 1
  (seeded per-memory-access interleaving → race.c reproduces) → Stage 2 (x86-TSO store buffers → missing-
  barrier bugs) → Stage 3 (DPOR/PCT). Seam is clean (all per-vCPU state already in `X86Cpu`; one gate
  `assert_eq!(vcpus,1)` at `state.rs:88`); absent = IPI/ICR/INIT-SIPI + store buffers + fences (`#UD` today)
  + N-CPU MADT/CPUID.
  **STAGE 0 IN PROGRESS (2026-07-16):** Slice 1 — `X86Machine.cpu` → `cpus: Vec<X86Cpu>` + `current` index +
  accessors (byte-identical at N=1: cargo test=209, gate C-anchor `38ab1994`+`det=ok`+identical instr count).
  Slice 2 — round-robin `pick_next_vcpu()` + restructured `run`/`run_until` (returns `Some(0)` at N=1 → order
  unchanged) + **PROVEN: 2 vCPUs execute round-robin over shared SC memory deterministically**
  (`smp_two_vcpus_shared_counter`: shared counter→10, both HLT, re-run bit-identical; cargo test=210). Slice 3 —
  per-vCPU `pending`/`delayed` (moved into `X86Cpu`; device IRQs route to `cpus[current]`, correct at N=1;
  cargo test=210, byte-identical). Slice 4 — **device-sharing** (the big invasive one): moved `ioapic`/`virtio`/
  `virtio_rng`/`virtio_net` from `X86Cpu` → machine-level shared `IoBus`; threaded `io` through the whole
  memory/execute cascade (~20 methods, 91+ call sites, 4 build layers); added `io()`/`io_mut()`; digest folds
  machine-level devices. cargo test=210 byte-identical N=1; **GATE 10/10 PASS, GATE_CONCURRENT_FAIL=0, all
  golden=ok+det=ok, byte-identical instr counts** → confirmed behavior-neutral. Slice 5 — **IOAPIC-dest routing**:
  `Ioapic::route_for_gsi` returns `(vector, dest APIC ID)` (physical mode bits 56:63; logical→BSP); added
  `X86Cpu.apic_id` + `cpu_for_apic()`; device IRQs (PIT/serial/virtio) route to `cpus[dest].pending` not
  `current`. Byte-identical at N=1 (dest→0→cpus[0]=current); cargo test=211. Slice 6 — **all-cpu digest**:
  current vCPU folded inline (byte-identical); every OTHER vCPU appended tagged-by-index via `fold_cpu_state()`
  + scheduler cursor, only when `vcpus>1` (N=1 loop empty → unchanged); cargo test=211. **SMP substructure
  complete** (N contexts, scheduler, per-cpu interrupts, shared devices, dest routing, all-cpu digest — all
  byte-identical at N=1). Slice 7 — **LAPIC ICR + INIT/SIPI AP-startup MECHANISM**: LAPIC ICR (`0x300`/`0x310`)
  + `take_ipi()` decode (vector/mode/dest/shorthand) + REG_ID reports apic id; `X86Cpu.wait_sipi` (AP parked);
  `deliver_ipi()` (Fixed→pending, INIT→park AP, SIPI→unpark at `rip=vector<<12`); scheduler skips `wait_sipi`
  vCPUs; delivery guarded to `vcpus>1` (N=1 byte-identical). **PROVEN end-to-end: `smp_ap_startup_via_sipi` —
  BSP writes ICR via MMIO → SIPI → AP wakes at vector<<12 → runs its startup code** (+ `icr_write_decodes_ipi`);
  cargo test=213. NOTE: emulator is FLAT-model (rip=linear, no cs.base) so real Linux's 16-bit real-mode AP
  trampoline needs real-mode-execution support (a follow-up); the SIPI mechanism itself is done. Remaining Stage 0:
  N-context build + MADT/CPUID enumeration → N; real-mode for the Linux AP trampoline. Slice 8 — **N-context
  build**: `X86Machine::set_vcpus(n)` attaches N−1 parked APs (each `apic_id`/`lapic.id = i`, `wait_sipi`,
  distinct `seed.derive("rdrand-cpu-i")` entropy); guest RAM + IOAPIC/virtio already shared. `set_vcpus(1)` =
  no-op (byte-identical). The AP-startup test now runs through the real `set_vcpus(2)` path; cargo test=213.
  **Remaining Stage 0: (a) MADT/CPUID enumeration → N (conditional on vcpus>1), (b) REAL-MODE EXECUTION SUPPORT
  — the emulator is flat-model, so Linux's 16-bit real-mode AP trampoline needs 16-bit decode + segmentation
  (`cs.base = cs<<4`) + the ljmp-to-protected-mode transition; this is the true blocker for an actual Linux
  SMP boot and a substantial new subsystem.**
  **REAL-MODE Slice 1 — segmentation + mode foundation (2026-07-16):** `CpuMode{Real,Protected,Long}` +
  `X86Cpu::cpu_mode()` (from CR0.PE/EFER.LMA) + `seg_base(sel)` (real=`sel<<4`, else flat 0); code fetch adds
  `cs_base` (`cs<<4 + IP` in real mode, 0 in long→byte-identical); `effective_address` adds the DS base for the
  default segment; SIPI now sets `CS=vector<<8, IP=0` (fetch at `vector<<12`). The bare/test constructor marks
  the machine long-mode (`efer|=LMA`) so its flat programs keep `cs_base=0` (boot overwrites w/ `EFER_BOOT`
  which has LMA → boot byte-identical). cargo test=213; AP now runs in **real mode** via `cs_base`. Remaining
  real-mode: 16-bit decode + mode transitions + 32-bit protected mode.
  **REAL-MODE Slice 2 — 16-BIT DECODE (2026-07-16, the big piece, DONE):** `DecodeMode{Bits16,32,64}` threaded
  into `decode()` (from `cpu_mode`); `op_size` mode-aware (16-bit default Word, 0x66→Dword); `0x67`→addr-size
  toggle; segment-override prefixes (`2e/36/3e/26`) now set `Seg::{Cs,Ss,Ds,Es}` (base added in
  `effective_address`, 0 in long → byte-identical); **16-bit ModRM decode** (`addr16()` → `[bx+si]`/`[bp+di]`/
  `[si]`/`[disp16]`/… no SIB, disp8/disp16). **PROVEN: `real_mode_16bit_execution` (a CPU in real mode runs
  `mov si,0x200; mov ax,0x1234; mov [si],ax; hlt` correctly) + `smp_ap_startup_via_sipi` now runs a REAL 16-bit
  AP trampoline via SIPI.** cargo test=214, byte-identical at 64-bit. Remaining for Linux AP boot: mode transitions, real-mode stack, 32-bit protected mode, MADT/CPUID enum → N.
  **MODE-TRANSITIONS (in progress):** `mov cr0`/`lgdt`/`lidt` already handled; added **far jmp `0xEA`**
  (`Op::JmpFar{selector,offset}` decode+exec: reload CS, set IP) + `lgdt`/`lidt` mode-aware base read (4 bytes
  in real/protected, 8 in long). cargo test=214, byte-identical (64-bit doesn't use 0xEA). **KEY FINDING — the
  crux of transitions is CS-DESCRIPTOR CACHING:** correct real→protected switching needs cached `cs_base` +
  `code_bits` (16/32/64 from CS.D/L) STORED fields updated only on CS load (far jmp reads GDT[sel].D/L+base),
  NOT recomputed from CR0.PE — because of the "unreal-mode window" (after `mov cr0.PE`, before the `ljmp`, CS
  still holds its cached real-mode base + 16-bit size). Next: `load_cs(sel, mem)` helper (GDT descriptor read) at
  the ~5 CS-load sites; fetch uses cached `cs_base`, decode uses `code_bits`; keep long-mode boot `code_bits=64`
  byte-identical. Then real-mode stack `ss_base`, and MADT/CPUID enum → N. *(Node
  no longer motivates this — its "needs multi-vCPU" diagnosis was WRONG; it was the POPF/IF bug, now
  fixed. All 10 concurrent runtimes incl. libuv/V8 + JIT-threaded JVM run byte-identical on ONE vCPU.
  Multi-vCPU is now purely for true SMP data-race exploration.)* *(merges former "multi-vCPU" ×2)*
  **A-LANE DONE A1–A5 (2026-07-17, cargo test=216, concurrent-gate byte-identical 4/4 so far):**
  **A1 CS-DESCRIPTOR CACHE** — added `X86Cpu.cs_base:u64` + `code_bits:u8` (reset/AP default `0/16`;
  `new_with_config` promotes the BSP to `code_bits=64`). `load_cs(sel,mem,io)` latches base+width from the
  *current* mode at the instant of the load (real: `sel<<4`,16; protected: GDT[sel] base + D→32/16; long:
  base 0, L→64 else D→32). FETCH now reads cached `self.cs_base` and DECODE picks `dmode` from
  `self.code_bits` (NOT recomputed from CR0.PE) — this is what makes the unreal-mode window correct. Only far
  jmp + SIPI route through the cache; iret/syscall/sysret stay long-mode (`code_bits=64`/`cs_base=0`) so the
  boot stream is byte-identical (cache fields are derived, NOT digest-folded). **A2 REAL-MODE STACK** — push/pop
  gained `stack_op_bytes()` (16/32/64 from `code_bits`) + `ss_base()` (`SS<<4` real, 0 flat); the `w==8` path is
  an early return byte-identical to the flat 64-bit model, 16/32-bit wrap SP within width through SS.base.
  **A3 32-BIT PROTECTED EXEC** — mostly free via `code_bits=32`→`Bits32`; fixed `moffs` width (0xA0-A3) to follow
  address size via new `addr_bytes()` (was hardcoded 8 → OOB in 32-bit). **PROVEN:
  `real_to_protected_mode_transition`** (lgdt→set PE→`ljmp 0x08:target`→runs 32-bit code writing 0xCAFEBABE) +
  **`real_mode_stack_push_pop`** (SS=0x2000 base, 16-bit push/pop). **A4 MADT ENUM** — `build_acpi_smp(mem,n)`
  emits one Local-APIC entry per vCPU (IDs 0..n); `build_acpi` = `build_acpi_smp(_,1)`, byte-identical single-CPU
  MADT (id=0 entry unchanged). CPUID left at 1 logical processor (frozen model identity; Linux enumerates via
  MADT not CPUID topology → cosmetic only). **A5 `--smp N` FLAG** — main.rs parses `--smp`, and when N>1 calls
  `set_vcpus(N)` + `build_acpi_smp(mem,N)` after boot; N==1 skips the block entirely → uniprocessor boot
  untouched.
  **★ A6 DONE — 2 REAL CPUs BOOT LINUX (2026-07-17):** `sfhv --smp 2` boots the Alpine 6.12.94 kernel to
  **`smp: Brought up 1 node, 2 CPUs` / `Total of 2 processors activated`** — the AP runs the full real-mode
  INIT-SIPI-SIPI trampoline → 32-bit protected → 64-bit long mode → `secondary_startup_64` → online, then the
  C workload runs (CSVC-HASH). **DETERMINISTIC:** two `--smp 2` runs byte-identical (state_digest
  `aa535b88afa01562`, mem_digest `4510b6b53412b366`, CSVC-HASH `3252bc8180761b56`). Getting the AP through the
  trampoline took **five real CPU bugs**, each found by a new `SF_APTRACE=N` knob (dumps each AP's first N
  instrs: cs/rip/base/cr0/efer/bits/bytes) and hand-decoding the divergence: **(1)** near `CALL`/`JMP`/`Jcc`
  rel decoded as rel32 in 16-bit mode → `rip` became a sign-extended 32-bit garbage address (fix: mode-aware
  `near_rel_bytes`); **(2)** `POPFD`/`PUSHFD` (`66 9d`/`66 9c`) treated as 16-bit → EFLAGS.ID (bit 21) never
  toggled → `verify_cpu` declared "no CPUID/long mode" and the AP hlt-spun (fix: operand-size-aware push/pop
  with `push_sized`/`pop_sized` + width-bounded POPF mask); **(3)** general `push/pop r32` (`66 50`/`66 58`)
  and **(4)** `push imm`/`push reg` (0x50-5F/0x68/0x6A) hardcoded to stack width / Qword → stack imbalance,
  `ret` to garbage (fix: `stack_operand_size`, operand-size data width in exec); **(5)** EFER.LMA not
  auto-set when `mov cr0` enables PG with LME=1 → the `ljmp` into the 64-bit CS mis-read the L=1/D=0 descriptor
  as 16-bit (fix: `LMA = PG && LME` in `set_cr`). All five are byte-identical in long mode by construction
  (64-bit paths keep the exact prior widths/behaviour). **★ A7 DONE — Stage-0 gate GREEN (2026-07-17):** SMP
  determinism proven (2× `--smp 2` byte-identical: state_digest `aa535b88afa01562`); the 10-language concurrent
  gate (vcpus=1) re-run after all five hot-path fixes → **`GATE_CONCURRENT_FAIL=0`, all 10 golden=ok
  byte-identical** (python `e423e84947e598a6` + node `28d1c4de` unchanged), cargo test=216, SSE+YMM
  differentials MATCH. **⇒ Multi-vCPU SMP is real (guest brings up the AP, "Total of 2 processors activated")
  AND deterministic AND zero-regression to the single-CPU substrate.** Debug knob `SF_APTRACE=N` (dump each
  AP's first N instrs) kept, off by default (only active with `--smp`>1).
  **★ SMP STAGE 1 DONE — SEED-DRIVEN INTERLEAVING EXPLORATION (2026-07-17, cargo test=217):** `SF_SMP_SCHED=Q`
  makes the vCPU scheduler run each core for a seed-drawn quantum (`1..=Q`) then switch to a seed-chosen runnable
  core, so different `--seed`s realise different — each byte-reproducible — legal interleavings. `Q=0`/unset =
  the fixed round-robin (byte-identical baseline). Machine-level seed-derived `smp_sched_rng` stream (own
  `derive("smp-sched")`), digest-folded ONLY when `smp_sched!=0` (baseline digest untouched). **Proven on a REAL
  Linux `--smp 2` boot:** baseline `aa535b88afa01562` (unchanged); `SF_SMP_SCHED=50` seed1 → `36a740599b74bd91`
  (reproduced 2×); seed2 → `a55617751b9e13c4` (≠ → explores); all reach "Total of 2 processors activated".
  Unit test `smp_stage1_seed_driven_interleaving` (non-atomic load/inc/store RMW on 2 vCPUs): seeds spread the
  lost-update count, each reproducible, baseline deterministic — the true data race a single vCPU can't show.
  **KEY FIX — YIELD ON `PAUSE`:** naive uniform preemption *timed out* a full boot (900s) because kernel
  spin-waits burn a whole quantum each; added `Op::Pause` (`F3 90`, distinguished from plain NOP via `Rep::F3`)
  which sets a transient `pause_yield` hint the scheduler acts on to switch to another core (pause-loop-exit,
  like KVM PLE) → boots complete fast, more realistic, still deterministic. Baseline byte-identical (PAUSE
  executes as NOP; `pause_yield` not digest-folded; concurrent gate re-run confirms vcpus=1 goldens).
  **★ SMP STAGE 2 DONE — x86-TSO STORE BUFFERS (2026-07-17, cargo test=218):** `SF_SMP_TSO=W` gives each vCPU a
  per-core FIFO store buffer: a normal WB-RAM store lands in the buffer (with a seed-drawn drain deadline in
  `1..=W` instructions) instead of memory; the core's own loads store-forward from it, but other cores only see
  the value once the machine drains it (deterministic FIFO / vCPU-index order, after each instruction). This
  makes **store→load reordering** observable — the Store-Buffering / Dekker outcome where *both* cores read the
  stale value, which Sequential Consistency forbids. `W=0`/unset = write-through (SC, byte-identical baseline;
  buffer + `sb_rng` never touched, digest-folded only when `W!=0`). Fences (`MFENCE/SFENCE/LFENCE` → new
  `Op::Fence`, was `Op::Nop`), LOCKed atomics (XCHG/XADD/CMPXCHG/CMPXCHG16B — bracketed with drains), HLT, and
  page-straddling accesses all drain the buffer. Per-vCPU seed-derived `sb_rng` (`derive("sb-cpu-N")`).
  **PROVEN: unit test `smp_stage2_tso_store_buffering_litmus`** — the SB litmus (CPU0 `x=1;r0=y` ∥ CPU1
  `y=1;r1=x`): under SC (W=0) `r0==r1==0` is impossible across 96 seeds; under TSO (W=50) it's reachable AND
  reproducible per seed. Baseline `--smp 2` boot still `aa535b88afa01562`; concurrent gate (vcpus=1) re-run to
  confirm all 10 goldens byte-identical (the mem_read/mem_write hot path gained only a predicted-false
  `if tso_max!=0` guard). Model is now genuinely x86-TSO (was SC).
  **★ SMP STAGE 3a DONE — PCT SCHEDULER (2026-07-20, cargo test=219):** `SF_SMP_PCT=depth[,k]` runs a
  Probabilistic-Concurrency-Testing schedule: each vCPU gets a distinct seed-permuted priority ≥ `depth`,
  the scheduler always runs the highest-priority *enabled* (making-progress) vCPU, and at each of the
  `depth-1` seed-drawn priority-change points (instruction counts in `[1,k]`, `k`=--max by default) it demotes
  the running vCPU below all others. This gives a **provable ≥ 1/(n·k^(depth-1)) chance of hitting any
  depth-`d` bug per seed** — every schedule reproducible. `depth=0` = OFF (byte-identical baseline; `pct_*`
  never touched, digest-folded only when on). A halted vCPU with nothing pending is treated as *blocked*
  (skipped) so it can't starve the core that would wake it; PAUSE still yields; an idle-fast-forward fallback
  keeps HLT-idle boots progressing. **PROVEN: unit test `smp_stage3_pct_priority_and_change_points`** on the
  non-atomic RMW race: **depth-1 (no change points) → counter always exactly 2N across 32 seeds** (pure
  priority scheduling serializes the two vCPUs → race-FREE), while **depth-2 (one change point) loses an update
  for some seeds** (the change point preempts mid-RMW) — each reproducible; baseline (off) digest-neutral.
  Fields: `pct_depth/pct_prio/pct_change_points/pct_next_cp/pct_low/pct_rng` + `set_pct`, `schedule_next_pct`,
  `pct_enabled`, `pct_highest` in exec.rs; `SF_SMP_PCT` in main.rs.
  **★ SMP STAGE 3b DONE — DPOR (2026-07-30, cargo test=228):** Dynamic Partial-Order Reduction
  (`src/x86/dpor.rs`) — the *systematic* complement to PCT's probabilistic sampling: enumerate **all**
  schedule-reachable outcomes of a (small, terminating) multi-vCPU program while never revisiting
  interleavings that only reorder independent accesses. Three pieces: **(1) forced-schedule replay** —
  `X86Machine::set_sched_script` makes the scheduler follow an explicit per-step vCPU list, then fall back
  to round-robin. Determinism ⇒ replaying a recorded prefix reproduces the pre-state of any step
  bit-for-bit, so backtracking needs NO snapshots or replay log (proven: `dpor_schedule_record_and_replay`
  — a seeded Stage-1 run's schedule, replayed on a fresh machine with exploration OFF, reproduces memory
  digest + every vCPU's registers exactly). **(2) access tracing** — `set_dpor_record(true)` records one
  step record per scheduled step: vCPU, pre-step *enabled set*, and the step's RAM accesses
  (post-translation; device MMIO + instruction fetch excluded), via a per-CPU buffer the machine drains.
  **(3) the driver** — `dpor_explore(builder, max_steps, max_execs)`: run an execution, find its races
  (Flanagan–Godefroid vector-clock happens-before = per-vCPU program order + conflict order; for each step,
  the *last* conflicting step of each other vCPU races unless already HB-ordered through intermediates),
  post the later step's vCPU into the backtrack set at the earlier step's pre-state (all enabled vCPUs if
  it wasn't enabled there), then re-run from the deepest untried alternative until none remain. Vec/BTreeSet
  only → the *exploration itself* is deterministic (asserted: two searches take identical paths).
  **PROVEN systematic + complete + reduced — vs a brute-force enumerator of EVERY schedule (same harness):**
  the SB litmus under SC yields exactly the 3 SC-reachable outcomes {(1,1),(0,1),(1,0)} — never the
  TSO-only (0,0) — in **4 executions vs 70 schedules**; the straight-line non-atomic RMW race yields
  exactly {2, 1(lost update)} in **12 vs 70**; every witness schedule replays to its recorded digest.
  **Off by default and observer-only** (hooks = one predicted-false compare in mem_read/mem_write + the
  scheduler; trace/script state never digest-folded): baseline `--smp 2` Linux boot **byte-identical
  old-binary-vs-new** (state_digest `7821ed9d069d4a8f`, = the post-rename baseline, deterministic 2×,
  "Total of 2 processors activated"); ten-runtime concurrent gate green, all goldens byte-identical;
  SSE + YMM silicon differentials MATCH. Scope: sequential consistency (`tso_max == 0`), litmus scale —
  whole-boot exploration is what Stage-1/PCT sampling is for. *(Follow-ups now tracked after 3c below.)*
  **★ SMP STAGE 3c DONE — COVERAGE-GUIDED EXPLORATION (2026-07-30, cargo test=231):** the feedback
  layer between DPOR (exhaustive, litmus-only) and Stage-1/PCT (blind sampling): an AFL-style
  **race-pair coverage sensor** usable at full boot scale, plus a **novelty-steered config explorer**.
  **(1) The sensor** (`SF_SMP_COV=window` / `set_race_cov`): O(1) per RAM access — a direct-mapped
  shadow table (2^16 slots, qword buckets, last write + last read each) detects a cross-vCPU
  conflicting access within `window` instructions and folds the pair's signature
  hash(rip_earlier, rip_later, kind ∈ {W→R, R→W, W→W}) into a 2^20-bit bitmap; the run prints a
  deterministic `=> race_cov pairs=… bitmap_digest=…` report, and bitmaps UNION across runs.
  Observer-only: digests identical with the sensor on or off; off = one predicted-false compare.
  **(2) Validated against exhaustive ground truth:** on the straight-line RMW litmus, the union over
  ALL 70 schedules (the brute-force enumerator, sensor on) is EXACTLY the program's 3 observable pair
  kinds; DPOR-witness replays observe no phantom pairs; a blind 16-seed Stage-1 sweep reaches the full
  set; per-seed reports byte-reproducible. **(3) The guided explorer** (`x86::cov_explore`): runs
  schedule-config families in order (round-robin → Stage-1 seeds → PCT), abandoning an axis after N
  consecutive runs that light no new bit. **PROVEN on a rare-pair program** (two stores to one address
  ~8000 instructions apart under normal schedules, window 256 — reachable only by running the warm-up
  vCPU to completion first): a blind 25-run Stage-1 sweep finds NOTHING (delaying the fast vCPU needs
  ~120 consecutive quantum draws, p≈2^-120); the guided search under the SAME budget abandons the
  barren axes and finds the pair via PCT priority scheduling in 21 runs (first novelty at run 17),
  byte-reproducibly (two searches identical). *(Probed reality encoded in the test: the first PCT
  priority draw is biased toward vCPU1 for small seeds — hits at seeds 8, 11, 13, 15, 16, 19, 21, 23
  of 1–24 — so the barren limit is 8.)* **(4) BOOT-SCALE:** on the real `--smp 2` Linux boot +
  C workload, the sensor observes **601 distinct cross-CPU pairs** (kernel-internal shared-memory
  contention), byte-identical across runs (`pairs=601 bitmap_digest=d4b041b25dea81f7` 2×), and the
  boot's `state_digest=7821ed9d069d4a8f` is IDENTICAL with the sensor on or off — observer-neutrality
  at full scale, still equal to the pre-DPOR baseline. A seeded schedule EXPLORES the pair space:
  `SF_SMP_SCHED=50` seed2 surfaces **1256 pairs** (`bitmap_digest=ce1915e07fe8c768`) — more than
  double the round-robin union. *(Budget note: a seeded `--smp 2` boot executes ~32B instructions
  vs 1.55B round-robin — the seed-driven scheduler keeps the second core hot instead of
  idle-fast-forwarding it — ≈50 min wall with the sensor on; time-box such runs accordingly.)*
  Ten-runtime concurrent gate re-run on the 3c binary: **GATE_CONCURRENT_FAIL=0, all goldens
  byte-identical** (python `e423e849` / node `28d1c4de` anchors unchanged). `SF_SMP_COV` in
  main.rs; sensor + explorer in exec.rs/dpor.rs. **Remaining SMP (follow-ups): DPOR-over-TSO
  (model buffered-store visibility as extra transitions), sleep sets (drop DPOR's remaining
  redundant revisits).**
- [ ] **5. New runtimes** — more if needed; the substrate is language-agnostic (CPU+kernel level).

---

## Milestone journey (condensed)

Detailed solved-bug notes live in the agent memory log; this is the narrative.

- **Deterministic core + toy machine** — seed derivation, virtual-instruction
  clock, deterministic event queue, state digest, snapshot/restore proven on a
  tiny RAM/CPU/MMIO machine. *(Historical minimal reference; superseded by the
  real x86 machine but kept — see [`docs/toy-machine.md`](docs/toy-machine.md).)*
- **P1 — real Linux boot to userspace (2026-06-15)** — boots Alpine kernel, paged
  long mode, IDT/GDT/TSS, SYSCALL/SYSRET, APIC/PIT/RTC/16550 serial, virtio-blk/
  rng; initramfs → `/init` → CPL3 → syscalls → poweroff. busybox init runs
  end-to-end. (Found+fixed along the way: `xchg rAX,r8-r15` REX.B decode, push/call
  RSP-leak on a faulting store, PUSH r/m64 width, TSC/PIT calibration, TX interrupt.)
- **P2 — full-run determinism proven** — whole-machine `state_digest`; two
  identical boots byte-identical; different seed → different digest; codebase
  audited clean of host coupling.
- **P3 — DST substrate** — controllable I/O (event-queued virtio completion),
  fault injection (EIO/torn/jitter), Go runtime (found+fixed XADD restart bug),
  TCP/net-http loopback, deterministic race reproduction, snapshot/restore.
- **P4 — seed exploration (first slice)** — `SF_SCHED_JITTER` seed-driven timer
  jitter explores interleavings; battle-test spread {-6,0,3}, each reproducible
  (QEMU can't).
- **Perf — 2.6×** — within-page mem fast path, fetch-translation cache,
  `(rip,cr3)`+generation decode cache (gen flush on CR3/INVLPG/SMC); determinism
  preserved. 7.4 → 19.1 MIPS.
- **Correctness gauntlet** — C++ stress byte-identical to QEMU (fixed CMPSD);
  CPython revealed a broad SSE2 gap → fixed 10+ families → CPython byte-identical.
  Differential tester scaled (`tools/diff/gen_sse.py`) → found+fixed SQRTPD.
- **x86-64-v2 ISA** — full SSE3/SSSE3/SSE4.1/SSE4.2 + CMPXCHG16B + PCMPISTRI/M +
  CRC32, bit-exact vs silicon; CPUID flipped (+MMX baseline so glibc loads);
  re-proven. **Node.js/V8 runs byte-identical** (needed x87 → implemented).
- **x87 FPU** — f64-modelled stack; later FPREM/FPREM1/FSCALE (exact fdlibm,
  surfaced by a C `fmod`) + FCMOVcc. *(Re-baseline: native FPREM shifted boot
  instruction streams → battle/CPython/Node digests moved; property re-proven,
  app-hashes unchanged.)*
- **C runtime validated** (glibc-static, libm/qsort/x87) — 5th proven runtime.
- **AVX (VEX) iterations 1–4** — YMM state + VEX C4/C5 decoder (maps 0F/0F38/0F3A);
  VZEROUPPER, moves, bitwise, FP arith, packed-int arith, **compares, shuffles,
  blends, converts, min/max, shifts, packs, VPMUL family, VMOVMSK**, NDD imm-shifts.
  All **bit-exact vs silicon** (128-bit). CPUID AVX bit later flipped at B.3 (Phase B).

---

## Component status

| Area | Status | Notes |
|---|---|---|
| Deterministic core (seed/clock/event-queue/digest) | [x] | |
| Snapshot / restore / divergence | [x] | |
| CPU interpreter (base x86-64 int/branch/string) | [x] | |
| 4-level paging MMU (4K/2M/1G, A/D, #PF, NX) | [x] | TLB-less = always coherent |
| Decode cache + SMC/CR3/INVLPG invalidation | [x] | |
| SSE / SSE2 / SSE3 / SSSE3 / SSE4.1 / SSE4.2 | [x] | bit-exact vs silicon |
| x87 FPU | [~] | f64-modelled; FPREM/FSCALE/FCMOV done; transcendentals/80-bit pending |
| AVX / AVX2 (VEX) | [~] | iter 1–4 done (128-bit); 256-bit + lane-cross + FMA pending |
| MXCSR rounding/FTZ/DAZ + exception flags | [ ] | silent-wrong-number class |
| Soft-float (cross-arch identity) | [ ] | cpu contract §4 |
| Devices: LAPIC/IOAPIC/PIT/RTC/16550/virtio-blk/rng | [x] | |
| virtio-net | [ ] | loopback only |
| Privilege: CPL/IDT/GDT/TSS/SYSCALL, precise-exception restart | [x] | |
| Determinism gate + differential-vs-silicon | [x] | run every change |
| Seed-exploration / property layer (P4) | [~] | first slice; breadth pending |
| SMP / multi-vCPU | [~] | Stages 0–3 done: N-vCPU Linux boot, seeded interleaving, x86-TSO, PCT, DPOR, coverage-guided exploration; DPOR-over-TSO/sleep-sets pending |

---

## Verification / how to run

All builds/tests/boots run on **`pcmini-x86`** (see [CLAUDE.md](CLAUDE.md)).

```bash
# unit + integration tests
cargo test --release --lib                       # 190 passing

# determinism + correctness regression gate (run after every change)
scripts/determinism-gate.sh            # Tier 1: units + SSE/VEX differential + battle + C
scripts/determinism-gate.sh --full     # Tier 2: + CPython + Node

# differential conformance vs real silicon
python3 tools/diff/gen_sse.py 1 40 > /tmp/g.txt
./target/release/lockstep --difftest /tmp/g.txt   # compare vs tools/diff/cputest

# boot a guest (deterministic; prints => state_digest=…)
./target/release/lockstep --seed <64-hex> \
  --boot vmlinuz --initrd initrd.gz --cmdline "console=ttyS0" --ram-mb 1024

# historical minimal determinism demo
cargo run -- --toy --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
```

---

## Doc map

| Doc | Purpose |
|---|---|
| [`GOLDEN_TARGET.md`](GOLDEN_TARGET.md) | Architecture & vision — the north star / source of truth |
| `Progress.md` (this file) | Living status: done / in progress / queued |
| [`CLAUDE.md`](CLAUDE.md) | How to work in this repo (constraints, workflow, gotchas) |
| [`docs/correctness-coverage.md`](docs/correctness-coverage.md) | Op/feature coverage matrix + prioritized build order |
| [`docs/cpu-determinism-contract.md`](docs/cpu-determinism-contract.md) | CPU determinism rules |
| [`docs/platform-determinism-contract.md`](docs/platform-determinism-contract.md) | Device/platform determinism rules |
| [`docs/frozen-values.md`](docs/frozen-values.md) | Identity constants (CPUID, defaults, goldens) |
| [`docs/smp-determinism.md`](docs/smp-determinism.md) | Future SMP design |
| [`docs/toy-machine.md`](docs/toy-machine.md) | Historical minimal determinism reference |
