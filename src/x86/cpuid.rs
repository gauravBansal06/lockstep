//! Fixed CPUID identity (Freeze v1, x86-64-v2 baseline, AVX off).
//!
//! CPUID is a constant table keyed by `(leaf, subleaf)`; it is part of
//! `virtual_cpu_model` in RunIdentity, so changing any value here is a reviewed
//! model bump. Values are committed in `docs/frozen-values.md` section 4.

/// Returns `[EAX, EBX, ECX, EDX]` for `CPUID` with the given leaf (EAX) and
/// subleaf (ECX). Unhandled leaves return zeros (deterministic).
pub fn cpuid(leaf: u32, subleaf: u32) -> [u32; 4] {
    match leaf {
        // Leaf 0: max basic leaf + vendor "GenuineIntel" (EBX:EDX:ECX).
        0x0 => [0x0000_000D, 0x756E_6547, 0x6C65_746E, 0x4965_6E69],
        // Leaf 1: signature, EBX(brand/clflush/maxlogical/apicid), ECX, EDX.
        // Truthfulness rule (Review 3): only advertise instructions the
        // decoder/executor actually implements deterministically.
        //   ECX = 0xB4D8_3201 — the x86-64-v3 set: SSE3(0) | SSSE3(9) | FMA(12) |
        //         CMPXCHG16B(13) | SSE4.1(19) | SSE4.2(20) | MOVBE(22) | POPCNT(23) |
        //         XSAVE(26) | AVX(28) | F16C(29) | HYPERVISOR(31). OSXSAVE(27) is
        //         added DYNAMICALLY in the CPUID execute arm (mirrors CR4.OSXSAVE).
        //         All implemented + validated bit-exact vs silicon (tools/diff).
        //   Still cleared: PCLMULQDQ, AESNI, RDRAND(30). RDRAND/RDSEED are IMPLEMENTED
        //   (deterministic seed-derived, decode 0F C7 /6,/7) but NOT advertised yet
        //   (Phase A): advertising them alone does not unblock the xstate-enabled boot
        //   (the CRNG-seeding deadlock is independent — see the leaf-0xD subleaf-1 note).
        //   EDX advertises MMX (bit 23) even though MMX (MM-register) ops are NOT
        //   modeled: the x86-64 baseline requires the MMX CPUID bit (glibc rejects a
        //   CPU below baseline). Modern code checks it but uses SSE2; an MMX insn #UDs.
        0x1 => [0x0002_06C2, 0x0001_0800, 0xB4D8_3201, 0x078B_FBF1],
        // Leaf 7 subleaf 0: BMI1(EBX bit3) | AVX2(bit5) | BMI2(bit8) = 0x128.
        // RDSEED(bit18) implemented but not advertised (Phase A, see leaf1). FSGSBASE off.
        0x7 if subleaf == 0 => [0x0000_0000, 0x0000_0128, 0x0000_0000, 0x0000_0000],
        0x7 => [0; 4],
        // Leaf 0xB: extended topology — a single logical processor.
        0xb => topology(subleaf),
        // Leaf 0xD: XSAVE state enumeration (x87 + SSE + AVX = XCR0 mask 0x7).
        // Inert until the XSAVE CPUID bit (leaf 1 ECX[26]) is advertised; the OS
        // only queries this leaf when XSAVE is supported. Standard format only
        // (subleaf 1 advertises no XSAVEOPT/XSAVEC/XSAVES).
        //   subleaf 0: EAX=XCR0 mask, EBX/ECX=max area size (832 = 576+256 YMM).
        //   subleaf 2: AVX state — EAX=size(256), EBX=offset(576).
        0xd if subleaf == 0 => [0x0000_0007, 0x0000_0340, 0x0000_0340, 0x0000_0000],
        // subleaf 1 EAX: XSAVEOPT(0) only = 0x1. This keeps the kernel's system xstate
        // ENABLED (real-Haswell value), so userspace runs TRUE AVX (XSAVE-backed YMM)
        // rather than the FXSAVE fallback. This is the B.4 userspace-AVX config and the
        // gate goldens are taken here — it unblocks AVX-*requiring* binaries (C#/.NET).
        //   HISTORY: EAX=0x3 (XSAVEOPT+XSAVEC) also boots, but the kernel then DISABLES
        //   system xstate (a size-consistency check on the compacted format) and falls
        //   back to FXSAVE, so userspace AVX never ran. Flipping to 0x1 once hung the boot
        //   early in CPU/bus-clock calibration; the systemic fix was IOAPIC masked-at-reset
        //   (premature timer IRQ into the calibration loop). With that fixed, 0x1 boots all
        //   four gate workloads to userspace byte-identically. CPython/Node verified running
        //   real userspace AVX (PY-HASH/NODE-HASH byte-identical to QEMU). See Progress.md.
        0xd if subleaf == 1 => [0x0000_0001, 0x0000_0000, 0x0000_0000, 0x0000_0000],
        0xd if subleaf == 2 => [0x0000_0100, 0x0000_0240, 0x0000_0000, 0x0000_0000],
        0xd => [0; 4],
        // Extended leaves.
        0x8000_0000 => [0x8000_0008, 0, 0, 0],
        // ext ECX: LAHF_LM(0) | LZCNT/ABM(5) = 0x21.
        0x8000_0001 => [0, 0, 0x0000_0021, 0x2810_0800],
        0x8000_0002 => brand(0),
        0x8000_0003 => brand(1),
        0x8000_0004 => brand(2),
        0x8000_0007 => [0, 0, 0, 0x0000_0100], // invariant TSC (EDX bit 8)
        0x8000_0008 => [0x0000_3028, 0, 0, 0],  // phys 40 bits, linear 48 bits
        _ => [0; 4],
    }
}

fn topology(subleaf: u32) -> [u32; 4] {
    match subleaf {
        0 => [0, 1, 1 << 8, 0],     // SMT level: 1 logical processor
        1 => [0, 1, 1 | (2 << 8), 0], // core level
        _ => [0, 0, subleaf & 0xff, 0],
    }
}

/// One 16-byte block of the 48-byte processor brand string.
fn brand(block: usize) -> [u32; 4] {
    let mut bytes = [0_u8; 48];
    let name = b"Lockstep Deterministic CPU v1";
    bytes[..name.len()].copy_from_slice(name);
    let mut out = [0_u32; 4];
    for (j, slot) in out.iter_mut().enumerate() {
        let off = block * 16 + j * 4;
        *slot = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_string_is_genuine_intel() {
        let [max, ebx, ecx, edx] = cpuid(0, 0);
        assert_eq!(max, 0x0D);
        let mut vendor = Vec::new();
        vendor.extend_from_slice(&ebx.to_le_bytes());
        vendor.extend_from_slice(&edx.to_le_bytes());
        vendor.extend_from_slice(&ecx.to_le_bytes());
        assert_eq!(&vendor, b"GenuineIntel");
    }

    #[test]
    fn leaf1_features_match_freeze() {
        let [_, _, ecx, edx] = cpuid(1, 0);
        // Implemented features that must stay advertised.
        assert_ne!(edx & (1 << 0), 0, "FPU");
        assert_ne!(edx & (1 << 25), 0, "SSE");
        assert_ne!(edx & (1 << 26), 0, "SSE2");
        assert_ne!(ecx & (1 << 23), 0, "POPCNT");
        // x86-64-v3 set — implemented + differential-validated vs silicon.
        assert_ne!(ecx & (1 << 0), 0, "SSE3");
        assert_ne!(ecx & (1 << 9), 0, "SSSE3");
        assert_ne!(ecx & (1 << 12), 0, "FMA");
        assert_ne!(ecx & (1 << 13), 0, "CMPXCHG16B");
        assert_ne!(ecx & (1 << 19), 0, "SSE4.1");
        assert_ne!(ecx & (1 << 20), 0, "SSE4.2");
        assert_ne!(ecx & (1 << 22), 0, "MOVBE");
        assert_ne!(ecx & (1 << 26), 0, "XSAVE");
        assert_ne!(ecx & (1 << 28), 0, "AVX");
        assert_ne!(ecx & (1 << 29), 0, "F16C");
        // Truthfulness: unimplemented instructions must NOT be advertised.
        assert_eq!(ecx & (1 << 1), 0, "PCLMULQDQ not implemented");
        assert_eq!(ecx & (1 << 25), 0, "AESNI not implemented");
        assert_eq!(ecx & (1 << 30), 0, "RDRAND implemented but not advertised (Phase A)");
        // OSXSAVE (bit 27) is added dynamically from CR4 in the CPUID exec arm,
        // so the static table must leave it clear.
        assert_eq!(ecx & (1 << 27), 0, "OSXSAVE is dynamic, not in the static table");
        // MMX bit advertised (required by the x86-64 baseline ISA level / glibc),
        // even though MMX MM-register ops are not modeled.
        assert_ne!(edx & (1 << 23), 0, "MMX (baseline ISA bit)");
    }

    #[test]
    fn leaf7_advertises_bmi_avx2() {
        let [_, ebx, ecx, edx] = cpuid(7, 0);
        assert_ne!(ebx & (1 << 3), 0, "BMI1");
        assert_ne!(ebx & (1 << 5), 0, "AVX2");
        assert_ne!(ebx & (1 << 8), 0, "BMI2");
        assert_eq!(ebx & (1 << 18), 0, "RDSEED implemented but not advertised (Phase A)");
        assert_eq!(ebx & (1 << 0), 0, "FSGSBASE not implemented");
        assert_eq!(ebx, 0x128);
        assert_eq!(ecx, 0);
        assert_eq!(edx, 0);
    }

    #[test]
    fn extended_leaf_has_lm_and_syscall() {
        let [_, _, _, edx] = cpuid(0x8000_0001, 0);
        assert_ne!(edx & (1 << 11), 0, "SYSCALL");
        assert_ne!(edx & (1 << 20), 0, "NX");
        assert_ne!(edx & (1 << 29), 0, "LM (long mode)");
    }

    #[test]
    fn brand_string_starts_with_expected() {
        let mut s = Vec::new();
        for leaf in 0x8000_0002..=0x8000_0004 {
            for reg in cpuid(leaf, 0) {
                s.extend_from_slice(&reg.to_le_bytes());
            }
        }
        assert!(s.starts_with(b"Lockstep Deterministic CPU v1"));
    }
}
