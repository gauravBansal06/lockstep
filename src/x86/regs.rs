use super::Size;
use crate::state::StateDigest;

// General-purpose register indices (encoding order).
pub const RAX: usize = 0;
pub const RCX: usize = 1;
pub const RDX: usize = 2;
pub const RBX: usize = 3;
pub const RSP: usize = 4;
pub const RBP: usize = 5;
pub const RSI: usize = 6;
pub const RDI: usize = 7;
pub const R8: usize = 8;
pub const R9: usize = 9;
pub const R10: usize = 10;
pub const R11: usize = 11;
pub const R12: usize = 12;
pub const R13: usize = 13;
pub const R14: usize = 14;
pub const R15: usize = 15;

/// Architectural register file for long mode: 16 GPRs, RIP, RFLAGS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Regs {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
}

impl Default for Regs {
    fn default() -> Self {
        Self::new()
    }
}

impl Regs {
    pub fn new() -> Self {
        Self {
            gpr: [0; 16],
            rip: 0,
            // Bit 1 is reserved and always reads as 1.
            rflags: 0x2,
        }
    }

    /// Read a GPR at the given width. `high8` selects the AH/CH/DH/BH legacy
    /// high-byte encoding (only valid for byte width without a REX prefix).
    pub fn read(&self, id: usize, size: Size, high8: bool) -> u64 {
        let full = self.gpr[id];
        match size {
            Size::Byte if high8 => (full >> 8) & 0xff,
            Size::Byte => full & 0xff,
            Size::Word => full & 0xffff,
            Size::Dword => full & 0xffff_ffff,
            Size::Qword => full,
        }
    }

    /// Write a GPR at the given width. A 32-bit write zero-extends to 64 bits;
    /// 8/16-bit writes preserve the upper bits (x86 partial-register rules,
    /// pinned in the CPU determinism contract).
    pub fn write(&mut self, id: usize, size: Size, high8: bool, value: u64) {
        let slot = &mut self.gpr[id];
        match size {
            Size::Byte if high8 => *slot = (*slot & !0xff00) | ((value & 0xff) << 8),
            Size::Byte => *slot = (*slot & !0xff) | (value & 0xff),
            Size::Word => *slot = (*slot & !0xffff) | (value & 0xffff),
            // 32-bit writes clear the upper 32 bits.
            Size::Dword => *slot = value & 0xffff_ffff,
            Size::Qword => *slot = value,
        }
    }
}

// Control-register flag bits the MMU/translation path cares about.
pub const CR0_WP: u64 = 1 << 16;
pub const CR0_PG: u64 = 1 << 31;
pub const EFER_SCE: u64 = 1 << 0;
pub const EFER_NXE: u64 = 1 << 11;

// Long-mode boot register values (Freeze v1 boot register file).
pub const CR0_BOOT: u64 = 0x8005_0033; // PE|MP|ET|NE|WP|AM|PG
pub const CR4_BOOT: u64 = 0x0000_0020; // PAE
pub const EFER_BOOT: u64 = 0x0000_0500; // LME|LMA

/// System/control registers relevant to long-mode paging.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SysRegs {
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub efer: u64,
    /// Interrupt descriptor table base (linear) and limit.
    pub idtr_base: u64,
    pub idtr_limit: u16,
    /// Global descriptor table base (linear) and limit.
    pub gdtr_base: u64,
    pub gdtr_limit: u16,
    /// CR8 (task priority register) and debug registers DR0–DR7.
    pub cr8: u64,
    pub dr: [u64; 8],
    /// Kernel stack pointer for a privilege-increasing interrupt (TSS.RSP0). Used
    /// as a fallback by tests that set it directly; the real boot reads RSP0 live
    /// from the TSS in memory at `tr_base + 4` (the kernel updates it per task
    /// switch without an instruction we could intercept).
    pub tss_rsp0: u64,
    /// Base linear address of the current TSS, captured by `LTR` from its GDT
    /// descriptor. 0 = no TSS loaded (use the `tss_rsp0` fallback).
    pub tr_base: u64,
}

impl SysRegs {
    pub fn paging_enabled(&self) -> bool {
        self.cr0 & CR0_PG != 0
    }
    pub fn write_protect(&self) -> bool {
        self.cr0 & CR0_WP != 0
    }
    pub fn nxe(&self) -> bool {
        self.efer & EFER_NXE != 0
    }
}

/// Model-specific registers backed by the engine (the frozen MSR map). EFER is
/// not here — it aliases `SysRegs::efer`. TSC is computed from virtual time.
/// See `docs/frozen-values.md` for reset values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msrs {
    pub apic_base: u64,
    pub misc_enable: u64,
    pub pat: u64,
    pub mtrr_def_type: u64,
    pub spec_ctrl: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    pub fs_base: u64,
    pub gs_base: u64,
    pub kernel_gs_base: u64,
    pub tsc_aux: u64,
    /// Offset applied to virtual time for the guest-visible TSC (set by WRMSR).
    pub tsc_offset: u64,
}

impl Default for Msrs {
    fn default() -> Self {
        Self {
            apic_base: 0xFEE0_0900,
            misc_enable: 0x1,
            pat: 0x0007_0406_0007_0406,
            mtrr_def_type: 0xC06,
            spec_ctrl: 0,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            fs_base: 0,
            gs_base: 0,
            kernel_gs_base: 0,
            tsc_aux: 0,
            tsc_offset: 0,
        }
    }
}

impl Msrs {
    /// Fold all MSR state into one value for the full-state digest.
    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        for v in [
            self.apic_base,
            self.misc_enable,
            self.pat,
            self.mtrr_def_type,
            self.spec_ctrl,
            self.star,
            self.lstar,
            self.cstar,
            self.sfmask,
            self.fs_base,
            self.gs_base,
            self.kernel_gs_base,
            self.tsc_aux,
            self.tsc_offset,
        ] {
            d.mix_u64(v);
        }
        d.finish()
    }
}
