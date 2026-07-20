//! x86-64 long-mode interpreter (GOLDEN_TARGET section 18 / Phase 8).
//!
//! A deterministic software interpreter: every result is a pure function of CPU
//! state and memory, with no host clock, host RNG, host pointer, or
//! host-architecture dependency (only wrapping integer arithmetic). It implements
//! a broad integer + control-flow subset of long mode with a real ModRM/SIB/REX
//! decoder and the flag semantics pinned in `docs/cpu-determinism-contract.md`
//! (undefined flags cleared to 0), running behind `DeterministicCore` for time.
//!
//! Implemented (partial, see `Progress.md` for the exact surface):
//! 4-level paging/MMU with A/D bits, #PF and canonical-address #GP; IDT
//! interrupt/exception delivery; MSRs (RDMSR/WRMSR with a frozen map); CPUID;
//! SYSCALL/SYSRET/SWAPGS; LAPIC + IOAPIC + a programmable timer; 16550 serial;
//! virtio-blk and virtio-rng; ACPI/e820 boot tables; and an SSE subset (scalar
//! float + SSE integer/conversions).
//!
//! Known partial / out of scope (deterministic `Unsupported` where unimplemented):
//! FP is host `f64`/`f32`, not yet canonical soft-float (host-independence is
//! partial); no x87/MMX; no packed-SSE arithmetic; segmentation beyond flat long
//! mode (no GDT/TSS/IST); STI/MOV-SS interrupt shadows; SMP/multi-vCPU. The CPUID
//! surface advertises **only** what is implemented (Review 3 truthfulness rule).

mod cpuid;
mod decode;
mod exec;
mod flags;
mod ioapic;
mod lapic;
mod mmu;
mod pit;
mod platform;
mod regs;
mod f80;
mod rtc;
mod serial;
mod virtio_blk;
mod virtio_net;
mod virtio_rng;

pub use cpuid::cpuid;
pub use decode::{
    decode, Cond, DecodeMode, DecodedInsn, FBinKind, LoopKind, Op, Operand, PLogicKind, Rep, Seg,
    ShiftKind, StrOp,
};
pub use exec::{X86Cpu, X86Exit, X86Machine, X86Snapshot};
pub use ioapic::{Ioapic, IOAPIC_BASE};
pub use lapic::{Lapic, LAPIC_BASE};
pub use mmu::{translate, Access, PageFaultInfo};
pub use platform::{
    build_acpi, build_acpi_smp, build_boot_params, default_e820, e820_entry, e820_entry_count,
    load_bzimage,
    setup_identity_paging, BootError, BzImageInfo, E820Entry, E820_RESERVED, E820_USABLE,
    RSDP_ADDR,
};
pub use regs::{
    Regs, SysRegs, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, R10, R11, R12, R13, R14, R15, R8, R9,
};
pub use pit::{Pit, PIT_BASE, PIT_FREQ};
pub use rtc::{Rtc, RTC_DATA, RTC_INDEX};
pub use serial::{IoBus, Uart16550, COM1_BASE};
pub use virtio_blk::{VirtioBlk, VIRTIO_BLK_BASE, VIRTIO_BLK_GSI, VIRTIO_BLK_VECTOR};
pub use virtio_net::{NetFaultPolicy, VirtioNet, VIRTIO_NET_BASE, VIRTIO_NET_GSI, VIRTIO_NET_VECTOR};
pub use virtio_rng::{VirtioRng, VIRTIO_RNG_BASE, VIRTIO_RNG_GSI, VIRTIO_RNG_VECTOR};

use crate::memory::MemoryError;

/// Operand width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Byte,
    Word,
    Dword,
    Qword,
}

impl Size {
    pub fn bytes(self) -> usize {
        match self {
            Size::Byte => 1,
            Size::Word => 2,
            Size::Dword => 4,
            Size::Qword => 8,
        }
    }

    pub fn bits(self) -> u32 {
        self.bytes() as u32 * 8
    }

    /// Mask of the significant bits for this width.
    pub fn mask(self) -> u64 {
        match self {
            Size::Qword => u64::MAX,
            other => (1_u64 << other.bits()) - 1,
        }
    }

    /// The sign bit position value for this width.
    pub fn sign_bit(self) -> u64 {
        1_u64 << (self.bits() - 1)
    }
}

/// Errors the interpreter can raise. All are deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X86Error {
    /// Opcode (or prefixed/escape opcode) not implemented by the skeleton.
    Unsupported(String),
    /// Decoder reached an invalid/illegal encoding.
    InvalidEncoding(String),
    /// Guest memory access fault (out of the flat backing store).
    Memory(MemoryError),
    /// `#DE`: divide by zero or quotient overflow. (Will become an IDT-vectored
    /// exception once interrupt delivery exists; deterministic either way.)
    DivideError,
    /// `#PF`: paging translation fault. Carries the SDM error code and the
    /// faulting linear address (mirrored to CR2). Deterministic.
    PageFault { error_code: u64, addr: u64 },
    /// `#GP(0)`: general protection (e.g. unknown MSR, privileged op at CPL>0).
    GeneralProtection,
}

impl From<MemoryError> for X86Error {
    fn from(value: MemoryError) -> Self {
        X86Error::Memory(value)
    }
}
