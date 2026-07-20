//! Execution of decoded instructions, and a thin machine wrapper that runs the
//! CPU behind the deterministic core for virtual time.

use super::decode::{
    decode, Cond, DecodeMode, DecodedInsn, FBinKind, FmaForm, FmaKind, FpuBin, FpuTrig, LoopKind,
    Op, Operand,
    PLogicKind,
    PMulKind, Rep, Seg, ShiftKind,
    StrOp, VBitKind, VCvtKind, VPkind,
};
use super::f80::F80;
use super::flags::{self, AF, CF, DF, IF, OF, PF, SF, ZF};
use super::cpuid::cpuid;
use crate::seed::StreamRng;
use super::ioapic::Ioapic;
use super::lapic::{Ipi, Lapic, DELIVERY_INIT, DELIVERY_STARTUP};
use super::mmu::{self, Access};
use super::virtio_blk::{VirtioBlk, VIRTIO_BLK_GSI, VIRTIO_BLK_VECTOR};
use super::virtio_net::{VirtioNet, VIRTIO_NET_GSI, VIRTIO_NET_VECTOR};
use super::virtio_rng::{VirtioRng, VIRTIO_RNG_GSI, VIRTIO_RNG_VECTOR};
use super::platform;
use super::regs::{
    Msrs, Regs, SysRegs, CR0_BOOT, CR4_BOOT, EFER_BOOT, EFER_SCE, RAX, RBP, RBX, RCX, RDI, RDX,
    RSI, RSP, R11,
};
use super::serial::IoBus;
use super::{Size, X86Error};
use crate::event::EventKind;
use crate::memory::GuestMemory;
use crate::state::{self, ConfigIdentity, MachineSpec, RunIdentity, RunMode, StateDigest};
use crate::{DeterministicCore, Seed256};
use std::collections::VecDeque;

/// Bits POPFQ is allowed to change (CF/PF/AF/ZF/SF/TF/IF/DF/OF/IOPL/NT/RF/AC/ID).
const RFLAGS_MASK: u64 = 0x0025_7fd5;
/// Global System Interrupt for the 8254 PIT channel-0 tick (legacy IRQ0).
const PIT_GSI: u32 = 0;
/// Global System Interrupt for the COM1 16550 UART (legacy IRQ4).
const COM1_GSI: u32 = 4;

const CR0_PE: u64 = 1 << 0; // protection enable
const CR0_PG: u64 = 1 << 31; // paging enable

/// One pending store held in a vCPU's x86-TSO store buffer (Stage 2). Keyed by PHYSICAL
/// address (post-translation) so aliasing virtual addresses coalesce correctly. `deadline`
/// is the global instruction count at/after which the machine may drain it to shared memory
/// (FIFO). Store→load reordering is exactly the window between a store's push and its drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StoreBufEntry {
    pa: u64,
    bytes: u8,
    value: u64,
    deadline: u64,
}
const EFER_LME: u64 = 1 << 8; // long-mode enable
const EFER_LMA: u64 = 1 << 10; // long-mode active

/// The current CPU operating mode, from CR0.PE and EFER.LMA. Determines segmentation (real-mode
/// bases = selector<<4) and, later, the default operand/address size for decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuMode {
    Real,      // 16-bit, segment base = selector<<4
    Protected, // 32-bit protected (we model flat descriptors: base 0)
    Long,      // 64-bit
}

/// The deterministic x86-64 execution core.
#[derive(Debug, Clone, PartialEq)]
pub struct X86Cpu {
    pub regs: Regs,
    pub sys: SysRegs,
    /// Code/stack segment selectors. In flat long mode the bases are 0; the
    /// selector values still matter for the interrupt stack frame.
    pub cs: u16,
    pub ss: u16,
    /// Data-segment selectors ES/DS/FS/GS. In long mode their bases are 0 (FS/GS
    /// bases live in MSRs), but the kernel loads/reads the selectors at boot.
    pub ds: u16,
    pub es: u16,
    pub fs_sel: u16,
    pub gs_sel: u16,
    /// Current privilege level (0 = supervisor, 3 = user). Long mode has no
    /// segment model here, so this is tracked directly; defaults to supervisor.
    pub cpl: u8,
    /// Model-specific registers (frozen map). EFER aliases `sys.efer`.
    pub msrs: Msrs,
    /// XMM (SSE) registers — 16 × 128-bit (the low 128 bits of each YMM).
    pub xmm: [u128; 16],
    /// Upper 128 bits of the YMM (AVX) registers — `ymm[i] = (ymm_hi[i]<<128) | xmm[i]`.
    /// VEX.128 ops zero this half; VEX.256 ops use it. Legacy SSE leaves it intact.
    pub ymm_hi: [u128; 16],
    /// XCR0 (extended-control register 0): which XSAVE state components are enabled.
    /// Reset = 1 (x87). The guest OS sets bit1 (SSE) + bit2 (AVX) via XSETBV.
    pub xcr0: u64,
    /// Deterministic seed-derived entropy stream for RDRAND/RDSEED (its own stream so
    /// "hardware" entropy never couples to guest-syscall or scheduler entropy). Seeded
    /// in `X86Machine::new_with_config`; folded into the state digest.
    pub rdrand_rng: StreamRng,
    /// The seeded origin cursor of `rdrand_rng`; the digest mixes the RDRAND stream
    /// only once it has advanced off this value (i.e. RDRAND/RDSEED was executed), so
    /// machines that never use it reproduce the legacy digest.
    pub rdrand_origin: u64,
    /// Per-CPU Local APIC (MMIO + timer).
    pub lapic: Lapic,
    /// This vCPU's local-APIC / physical ID (BSP = 0). Interrupt destinations route to it.
    pub apic_id: u32,
    // NOTE: IOAPIC + virtio devices moved to the machine-level shared `IoBus` (SMP: one instance
    // for all vCPUs). Only the per-vCPU local APIC (`lapic`) stays here.
    /// Current virtual time snapshot for TSC reads (synced by the machine).
    pub tsc: u64,
    pub halted: bool,
    /// SMP AP bring-up: a secondary CPU sits in this state (not scheduled) after reset/INIT
    /// until it receives a Startup-IPI (SIPI). The BSP is always `false`. At `vcpus == 1` there
    /// are no APs, so this is always `false` and inert.
    pub wait_sipi: bool,
    /// Cached CS-descriptor base and code width — the CS "hidden descriptor" the CPU
    /// latches at every CS load (far jmp / SIPI / reset). Segmentation base and default
    /// operand/address size are read from this cache, NOT recomputed from CR0.PE on every
    /// fetch. That is what makes the *unreal-mode window* correct: after `mov cr0` sets PE
    /// but before the `ljmp` reloads CS, CS must still act with its previously-latched base
    /// and width. `code_bits` ∈ {16,32,64}. Reset/AP real mode = base `cs<<4`, 16-bit.
    /// Long/flat = base 0, 64-bit. These are *derived* cache state (a pure function of the
    /// last CS load), so they are intentionally NOT folded into the state digest.
    pub cs_base: u64,
    pub code_bits: u8,
    /// Transient spin-wait yield hint: set when this vCPU executes a `PAUSE`, consumed by the
    /// SMP scheduler to switch to another core (pause-loop-exit). Not part of guest-visible
    /// state and NOT folded into the digest — it is cleared by the scheduler each time it is
    /// observed, and only ever influences scheduling when seed-driven exploration is enabled.
    pub pause_yield: bool,
    /// x86-TSO store buffer (Stage 2): this vCPU's FIFO of not-yet-globally-visible stores.
    /// Empty unless `tso_max != 0`. When enabled, a normal WB-RAM store lands here (with a
    /// seed-drawn drain deadline) instead of memory; the vCPU's own loads store-forward from
    /// it, but other vCPUs only see the value once the machine drains it. Fences / LOCKed
    /// ops / HLT drain it. Digest-folded only when non-empty, so the baseline is unchanged.
    pub store_buf: VecDeque<StoreBufEntry>,
    /// Max seed-drawn extra lingering (instructions) a buffered store waits before it may
    /// drain. 0 = TSO OFF → stores write straight through (Sequential Consistency, the
    /// byte-identical baseline; the buffer and `sb_rng` are never touched).
    pub tso_max: u64,
    /// Dedicated per-vCPU seed-derived entropy for store-buffer drain deadlines — its own
    /// stream so TSO timing never couples to guest / scheduler / device entropy. Advanced
    /// (and digest-folded) only when `tso_max != 0`.
    pub sb_rng: StreamRng,
    /// This vCPU's pending external interrupt vectors (held while IF is clear) and its
    /// seed-jittered deferred interrupts `(release_at, vector)`. Per-vCPU so each core takes
    /// its own interrupts; at `vcpus == 1` these are the sole machine interrupt queues.
    pub pending: VecDeque<u8>,
    pub delayed: Vec<(u64, u8)>,
    /// x87 FPU control/status/tag words. `fpu_sw` bits 11-13 hold TOP (mirrored
    /// from `fpu_top`); C0-C3 (bits 8,9,10,14) hold the last compare/exam result.
    pub fpu_cw: u16,
    pub fpu_sw: u16,
    pub fpu_tw: u16,
    /// x87 register stack: physical registers R0-R7. ST(i) = `st[(fpu_top+i) & 7]`.
    /// Modelled as host f64 (deterministic); 80-bit extended precision is NOT yet
    /// exact vs silicon (needs soft-float — cpu-determinism-contract §4). `fpu_top`
    /// is the stack-top pointer (0-7); a push pre-decrements it, a pop increments.
    pub st: [F80; 8],
    pub fpu_top: u8,
    /// SSE control/status register (MXCSR); reset value 0x1F80.
    pub mxcsr: u32,
    /// Debug write-watchpoint (physical address); 0 disables. Logs every store
    /// covering this byte with the faulting RIP — used to hunt boot corruption.
    pub watch: u64,
    /// Debug: log every vectored exception except normal page faults.
    pub trace_exc: bool,
    /// Debug: log every guest SYSCALL (number + first args) — userspace probe.
    pub trace_sys: bool,
    /// Debug: like `trace_sys` but only for network syscalls (read/write/epoll/
    /// accept/connect/send/recv) — light enough to trace a heavy runtime's I/O phase
    /// (SYSRET is paired via `last_sys_net`). Off in the gate → zero effect.
    pub trace_net: bool,
    pub last_sys_net: bool,
    pub trace_mem: bool,
    pub last_sys_mem: bool,
    /// For SF_NETTRACE: the epoll_wait/epoll_pwait number + events-buffer pointer
    /// captured at syscall entry, so SYSRET can dump the returned events array
    /// (the one thing a bare rax=count can't show). Debug-only.
    pub last_net_nr: u64,
    pub last_net_buf: u64,
    /// SF_ITRACE: once userspace rip first hits `itrace_rip`, log every subsequent
    /// CPL3 instruction's rip (up to `itrace_n`) — for an instruction-stream diff
    /// against QEMU to find the first divergent branch. Debug-only, off by default.
    pub trace_itrace: bool,
    pub itrace_rip: u64,
    pub itrace_n: u64,
    /// SF_APTRACE: log this (AP) vCPU's (cs, rip, cs_base, cr0, code bytes) for the first
    /// `ap_trace_n` instructions after it starts — for debugging the real-mode AP trampoline
    /// (16-bit decode / mode transitions). Debug-only, off by default; set on APs in main.
    pub trace_ap: bool,
    pub ap_trace_n: u64,
    pub itrace_armed: bool,
    pub itrace_regs: bool,
    pub itrace_idx: u64,
    pub itrace_allcpl: bool,
    pub itrace_fault_cr2: u64,
    /// SF_REGDUMP_RIP: once itrace is armed (past the landmark), dump GPRs + the
    /// GetString arg string each time userspace rip hits this address. Debug-only.
    pub regdump_rip: u64,
    pub regdump_occ: u64,
    pub regdump_count: u64,
    /// SF_WATCHADDR/SF_WATCHVAL: after itrace is armed, log the rip of any store of
    /// `watch_val` to `watch_addr` — to find which instruction wrote a divergent
    /// value to a known stack slot. Debug-only. `cur_rip` = current instruction addr.
    pub watch_addr: u64,
    pub watch_val: u64,
    pub cur_rip: u64,
    /// Debug: when set, dump the delay-loop target + caller stack once it is
    /// clearly stuck (counts hits of the `delay_tsc` compare at 0x81af203a).
    pub trace_delay: bool,
    delay_count: u64,
    /// Debug: log the instruction that sets RSI into the `0x000f…` range.
    pub trace_rsi: bool,
    /// Debug: log System-V args + return address when RIP reaches this value.
    pub fn_watch: u64,
    /// Debug: log stores whose *virtual* destination is in [wwatch_lo, wwatch_hi).
    pub wwatch_lo: u64,
    pub wwatch_hi: u64,
    /// Debug: log stores whose value's low 32 bits match this (to catch a 64-bit
    /// pointer being truncated to 4 bytes).
    pub vwatch: u64,
    /// Decode cache: a direct-mapped table of already-decoded instructions keyed
    /// by (rip, cr3), so hot code is not re-parsed every iteration. Determinism is
    /// preserved: an entry is reused only for the exact (rip, cr3) it was decoded
    /// at (so baked RIP-relative operands stay correct and decodes never cross
    /// address spaces), and the whole cache is flushed when the guest writes to a
    /// page that holds cached code (self-modifying / patched code). A pure cache —
    /// excluded from the state digest.
    decode_cache: Box<[DecodeSlot]>,
    /// Current decode-cache generation; a slot is valid only if its `gen` equals
    /// this. Bumping it is an O(1) flush of the whole cache.
    decode_gen: u64,
    /// One bit per guest-physical page that has ever held a cached instruction.
    /// A store to such a page flushes the decode cache (SMC invalidation). Bits are
    /// only ever set (never cleared): a stale bit at worst causes a spurious flush.
    code_page_bits: Box<[u64]>,
}

/// Decode-cache sizing. 2^15 direct-mapped slots; the bitmap covers 256 MiB of
/// 4 KiB physical pages (the max guest RAM).
const DECODE_CACHE_BITS: usize = 15;
const DECODE_CACHE_SIZE: usize = 1 << DECODE_CACHE_BITS;
const DECODE_CACHE_MASK: u64 = DECODE_CACHE_SIZE as u64 - 1;
const PHYS_BITMAP_PAGES: u64 = 1 << 16; // 65536 pages = 256 MiB
const PHYS_BITMAP_WORDS: usize = (PHYS_BITMAP_PAGES / 64) as usize;

/// One direct-mapped decode-cache slot. A slot is a hit only when its `gen`
/// matches the cache's current generation AND `rip`/`cr3` match. The generation
/// is bumped (an O(1) whole-cache flush) on every event that can change the bytes
/// at a virtual address: CR3 loads (address-space switch / TLB flush / PGD reuse),
/// INVLPG (single-page remap), and stores to a cached code page (self-modifying
/// code). Without this, a recycled cr3 or a remapped page returns a stale decode.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DecodeSlot {
    gen: u64,
    rip: u64,
    cr3: u64,
    insn: Option<DecodedInsn>,
}

impl Default for X86Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl X86Cpu {
    pub fn new() -> Self {
        Self {
            regs: Regs::new(),
            sys: SysRegs::default(),
            cs: 0x08,
            ss: 0x10,
            ds: 0,
            es: 0,
            fs_sel: 0,
            gs_sel: 0,
            cpl: 0,
            msrs: Msrs::default(),
            xmm: [0; 16],
            ymm_hi: [0; 16],
            xcr0: 1, // x87 state always enabled at reset
            rdrand_rng: StreamRng::new(0), // reseeded from the run seed in new_with_config
            rdrand_origin: 0,
            lapic: Lapic::new(),
            apic_id: 0,
            tsc: 0,
            halted: false,
            wait_sipi: false,
            // Reset/AP default = real mode: base 0 (a SIPI re-points it to vector<<12), 16-bit
            // code. `new_with_config` promotes the BSP to long-mode flat (base 0, 64-bit).
            cs_base: 0,
            code_bits: 16,
            pause_yield: false,
            store_buf: VecDeque::new(),
            tso_max: 0,
            sb_rng: StreamRng::new(0), // reseeded per-vCPU in new_with_config / set_vcpus
            pending: VecDeque::new(),
            delayed: Vec::new(),
            fpu_cw: 0x037f, // x87 reset control word (all exceptions masked)
            fpu_sw: 0,
            fpu_tw: 0xffff, // all stack slots empty
            st: [F80::ZERO; 8],
            fpu_top: 0,
            mxcsr: 0x1f80, // SSE reset value (all exceptions masked)
            watch: 0,
            trace_exc: false,
            trace_sys: false,
            trace_net: false,
            last_sys_net: false,
            trace_mem: false,
            last_sys_mem: false,
            last_net_nr: 0,
            last_net_buf: 0,
            trace_itrace: false,
            itrace_rip: 0,
            itrace_n: 0,
            trace_ap: false,
            ap_trace_n: 0,
            itrace_armed: false,
            itrace_regs: false,
            itrace_idx: 0,
            itrace_allcpl: false,
            itrace_fault_cr2: 0,
            regdump_rip: 0,
            regdump_occ: 0,
            regdump_count: 0,
            watch_addr: 0,
            watch_val: 0,
            cur_rip: 0,
            trace_delay: false,
            delay_count: 0,
            trace_rsi: false,
            fn_watch: 0,
            wwatch_lo: 0,
            wwatch_hi: 0,
            vwatch: 0,
            decode_cache: vec![DecodeSlot::default(); DECODE_CACHE_SIZE].into_boxed_slice(),
            decode_gen: 1, // default slots have gen 0 → never falsely match
            code_page_bits: vec![0u64; PHYS_BITMAP_WORDS].into_boxed_slice(),
        }
    }

    /// Mark a guest-physical page as holding cached code (decode-cache miss path).
    #[inline]
    fn mark_code_page(&mut self, page: u64) {
        if page < PHYS_BITMAP_PAGES {
            self.code_page_bits[(page / 64) as usize] |= 1 << (page % 64);
        }
    }

    /// Does this guest-physical page hold cached code? (checked on every store)
    #[inline]
    fn is_code_page(&self, page: u64) -> bool {
        page < PHYS_BITMAP_PAGES
            && self.code_page_bits[(page / 64) as usize] & (1 << (page % 64)) != 0
    }

    /// Invalidate the whole decode cache in O(1) by bumping the generation. Called
    /// on CR3 loads, INVLPG, and stores to a cached code page.
    #[inline]
    pub fn flush_decode_cache(&mut self) {
        self.decode_gen = self.decode_gen.wrapping_add(1);
    }

    /// Read a model-specific register (RDMSR). Unknown MSR → `#GP`.
    fn rdmsr(&self, msr: u32) -> Result<u64, X86Error> {
        if self.cpl != 0 {
            return Err(X86Error::GeneralProtection);
        }
        let m = &self.msrs;
        Ok(match msr {
            0x10 => self.tsc.wrapping_add(m.tsc_offset), // IA32_TSC
            0x17 => 0,                                   // IA32_PLATFORM_ID
            0x8b => 0,                                   // IA32_BIOS_SIGN_ID (microcode rev)
            0x1b => m.apic_base,                         // IA32_APIC_BASE
            0x48 => m.spec_ctrl,                         // IA32_SPEC_CTRL
            0xfe => 0x0508,                              // IA32_MTRRCAP
            0x10a => 0,                                  // IA32_ARCH_CAPABILITIES
            0x1a0 => m.misc_enable,                      // IA32_MISC_ENABLE
            0x277 => m.pat,                              // IA32_PAT
            0x2ff => m.mtrr_def_type,                    // IA32_MTRR_DEF_TYPE
            0xda0 => 0,                                  // IA32_XSS (no supervisor xstates)
            0xc000_0080 => self.sys.efer,                // IA32_EFER (alias)
            0xc000_0081 => m.star,
            0xc000_0082 => m.lstar,
            0xc000_0083 => m.cstar,
            0xc000_0084 => m.sfmask,
            0xc000_0100 => m.fs_base,
            0xc000_0101 => m.gs_base,
            0xc000_0102 => m.kernel_gs_base,
            0xc000_0103 => m.tsc_aux,
            _ => return Err(X86Error::GeneralProtection),
        })
    }

    /// Write a model-specific register (WRMSR). Unknown MSR → `#GP`.
    fn wrmsr(&mut self, msr: u32, value: u64) -> Result<(), X86Error> {
        if self.cpl != 0 {
            return Err(X86Error::GeneralProtection);
        }
        match msr {
            0x10 => self.msrs.tsc_offset = value.wrapping_sub(self.tsc), // IA32_TSC
            0x8b => {} // IA32_BIOS_SIGN_ID: microcode-load trigger, no-op
            0x1b => self.msrs.apic_base = value,
            0x48 => self.msrs.spec_ctrl = value,
            0x49 => {} // IA32_PRED_CMD: write-only no-op
            0x1a0 => self.msrs.misc_enable = value,
            0x277 => self.msrs.pat = value,
            0x2ff => self.msrs.mtrr_def_type = value,
            0xda0 => {} // IA32_XSS: no supervisor xstates supported (kernel writes 0)
            0xc000_0080 => self.sys.efer = value, // IA32_EFER (alias)
            0xc000_0081 => self.msrs.star = value,
            0xc000_0082 => self.msrs.lstar = value,
            0xc000_0083 => self.msrs.cstar = value,
            0xc000_0084 => self.msrs.sfmask = value,
            0xc000_0100 => self.msrs.fs_base = value,
            0xc000_0101 => self.msrs.gs_base = value,
            0xc000_0102 => self.msrs.kernel_gs_base = value,
            0xc000_0103 => self.msrs.tsc_aux = value,
            _ => return Err(X86Error::GeneralProtection),
        }
        Ok(())
    }

    fn get_cr(&self, reg: u8) -> Result<u64, X86Error> {
        Ok(match reg {
            0 => self.sys.cr0,
            2 => self.sys.cr2,
            3 => self.sys.cr3,
            4 => self.sys.cr4,
            8 => self.sys.cr8,
            _ => return Err(X86Error::GeneralProtection),
        })
    }

    fn set_cr(&mut self, reg: u8, value: u64) -> Result<(), X86Error> {
        match reg {
            0 => {
                self.sys.cr0 = value; // paging/PE take effect immediately
                // Long-mode activation: LMA = CR0.PG && EFER.LME. A guest that has set
                // EFER.LME then enables CR0.PG is *entering* long mode, and the CPU sets
                // EFER.LMA automatically — this is the step between the AP trampoline's
                // 32-bit `mov cr0` (PG on) and its `ljmp` into the 64-bit code segment. If
                // we don't set LMA here, the far jump reloads CS while still in "protected"
                // mode and mis-reads the 64-bit (L=1,D=0) descriptor as a 16-bit segment.
                if value & CR0_PG != 0 && self.sys.efer & EFER_LME != 0 {
                    self.sys.efer |= EFER_LMA;
                } else if value & CR0_PG == 0 {
                    self.sys.efer &= !EFER_LMA;
                }
            }
            2 => self.sys.cr2 = value,
            3 => self.sys.cr3 = value, // new page-table root
            4 => self.sys.cr4 = value,
            8 => self.sys.cr8 = value,
            _ => return Err(X86Error::GeneralProtection),
        }
        // CR0/CR3/CR4 change the active address translation (paging enable/WP, the
        // page-table root incl. PGD reuse, PGE/PAE). Any cached decode could now be
        // stale, so flush the decode cache (O(1) generation bump).
        if matches!(reg, 0 | 3 | 4) {
            self.flush_decode_cache();
        }
        Ok(())
    }

    /// True if a maskable external interrupt can be taken right now.
    pub fn can_take_interrupt(&self) -> bool {
        self.regs.rflags & IF != 0 && self.sys.idtr_limit > 0
    }

    // ---- x87 register stack (modelled as f64; see the `st`/`fpu_top` fields) ----

    /// Sync the TOP field (status word bits 11-13) from `fpu_top`.
    fn fpu_sync_sw_top(&mut self) {
        self.fpu_sw = (self.fpu_sw & !0x3800) | ((self.fpu_top as u16 & 7) << 11);
    }
    /// Set the 2-bit tag of a physical register (00=valid, 11=empty).
    fn fpu_set_phys_tag(&mut self, phys: u8, empty: bool) {
        let shift = (phys & 7) * 2;
        self.fpu_tw = (self.fpu_tw & !(3 << shift)) | (if empty { 3 } else { 0 } << shift);
    }
    /// Push an 80-bit value onto the x87 stack (pre-decrement TOP).
    fn fpu_push80(&mut self, val: F80) {
        self.fpu_top = self.fpu_top.wrapping_sub(1) & 7;
        self.st[self.fpu_top as usize] = val;
        self.fpu_set_phys_tag(self.fpu_top, false);
        self.fpu_sync_sw_top();
    }
    /// Pop ST(0) (increments TOP), returning the 80-bit value.
    fn fpu_pop80(&mut self) -> F80 {
        let v = self.st[self.fpu_top as usize];
        self.fpu_set_phys_tag(self.fpu_top, true);
        self.fpu_top = (self.fpu_top + 1) & 7;
        self.fpu_sync_sw_top();
        v
    }
    /// ST(i) as 80-bit.
    fn fpu_st80(&self, i: u8) -> F80 {
        self.st[((self.fpu_top + i) & 7) as usize]
    }
    /// Write ST(i) (80-bit).
    fn fpu_set_st80(&mut self, i: u8, val: F80) {
        self.st[((self.fpu_top + i) & 7) as usize] = val;
    }
    // f64 bridge for impl-defined / algebraic ops (transcendentals, constants, prem, scale).
    fn fpu_push(&mut self, val: f64) {
        self.fpu_push80(F80::from_f64(val));
    }
    fn fpu_pop(&mut self) -> f64 {
        self.fpu_pop80().to_f64()
    }
    fn fpu_st(&self, i: u8) -> f64 {
        self.fpu_st80(i).to_f64()
    }
    fn fpu_set_st(&mut self, i: u8, val: f64) {
        self.fpu_set_st80(i, F80::from_f64(val));
    }
    /// FCW rounding-control (bits 10:11) and precision-control (bits 8:9).
    fn fpu_rc(&self) -> u8 {
        ((self.fpu_cw >> 10) & 3) as u8
    }
    fn fpu_pc(&self) -> u8 {
        ((self.fpu_cw >> 8) & 3) as u8
    }
    /// Set the x87 compare flags C3/C2/C0 (status-word bits 14/10/8) from an
    /// ordered comparison of `a` vs `b` (C1 cleared). Unordered (NaN) → all set.
    fn fpu_set_compare(&mut self, a: f64, b: f64) {
        // C3 C2 C0: equal=100, a<b=001, a>b=000, unordered=111.
        let (c3, c2, c0) = if a.is_nan() || b.is_nan() {
            (true, true, true)
        } else if a < b {
            (false, false, true)
        } else if a > b {
            (false, false, false)
        } else {
            (true, false, false)
        };
        self.fpu_sw &= !0x4700; // clear C3(14) C2(10) C1(9) C0(8)
        if c3 {
            self.fpu_sw |= 1 << 14;
        }
        if c2 {
            self.fpu_sw |= 1 << 10;
        }
        if c0 {
            self.fpu_sw |= 1 << 8;
        }
    }

    /// Fetch, decode, and execute one instruction. A deliverable CPU exception
    /// (`#DE`/`#PF`/`#UD`) is vectored through the IDT when one is installed
    /// (IDTR limit > 0); otherwise it propagates as an error (test/bare mode).
    pub fn step(&mut self, mem: &mut GuestMemory, io: &mut IoBus) -> Result<(), X86Error> {
        self.cur_rip = self.regs.rip;
        if self.trace_ap && self.ap_trace_n > 0 {
            // Real mode has paging off → the fetch physical addr is cs_base+rip. Read a few
            // raw code bytes (debug read, no fault) to hand-decode the trampoline stream.
            let fa = self.cs_base.wrapping_add(self.regs.rip);
            let b = self.read_qword_dbg(mem, fa);
            eprintln!(
                "AP id={} cs={:#x} rip={:#x} base={:#x} cr0={:#x} efer={:#x} bits={} sp={:#x} ax={:#x} @{:#x}: {:016x}",
                self.apic_id, self.cs, self.regs.rip, self.cs_base, self.sys.cr0, self.sys.efer,
                self.code_bits, self.regs.gpr[RSP], self.regs.gpr[RAX], fa, b.swap_bytes()
            );
            self.ap_trace_n -= 1;
        }
        if self.trace_itrace && (self.cpl == 3 || self.itrace_allcpl) {
            if !self.itrace_armed && self.itrace_rip != 0 && self.regs.rip == self.itrace_rip {
                self.itrace_armed = true;
            }
            if self.itrace_armed {
                if self.itrace_n > 0 {
                    if self.itrace_regs {
                        let g = self.regs.gpr;
                        // Non-stack registers only (rbp/rsp/r8 carry a per-run stack
                        // base); idx lets a register-state diff align by instruction.
                        eprintln!(
                            "R idx={} rip={:#x} rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} r9={:#x} r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x} fl={:#x}",
                            self.itrace_idx, self.regs.rip,
                            g[RAX], g[RBX], g[RCX], g[RDX], g[RSI], g[RDI],
                            g[9], g[10], g[11], g[12], g[13], g[14], g[15],
                            self.regs.rflags & 0x8d5
                        );
                    } else {
                        eprintln!("I {:#x}", self.regs.rip);
                    }
                    self.itrace_idx += 1;
                    self.itrace_n -= 1;
                } else {
                    self.trace_itrace = false;
                }
            }
        }
        if self.regdump_rip != 0 && self.cpl == 3 && self.itrace_armed
            && self.regs.rip == self.regdump_rip
        {
            self.regdump_count += 1;
            if self.regdump_occ == 0 || self.regdump_count == self.regdump_occ {
                let g = self.regs.gpr;
                let saved_cr2 = self.sys.cr2;
                let rsp = g[RSP];
                let m0 = self.read_qword_dbg(mem, rsp);
                let m8 = self.read_qword_dbg(mem, rsp.wrapping_add(8));
                self.sys.cr2 = saved_cr2;
                eprintln!(
                    "REGDUMP occ={} rip={:#x} rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} rbp={:#x} rsp={:#x} r8={:#x} r9={:#x} r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x} rflags={:#x} [rsp]={:#x} [rsp+8]={:#x}",
                    self.regdump_count, self.regs.rip,
                    g[RAX], g[RBX], g[RCX], g[RDX], g[RSI], g[RDI], g[RBP], g[RSP],
                    g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15],
                    self.regs.rflags, m0, m8
                );
            }
        }
        if self.trace_delay && self.regs.rip == 0xffffffff81af203a {
            self.delay_count += 1;
            // Once clearly stuck in the long delay, dump the target cycle count
            // and any kernel-text return addresses on the stack (the caller chain
            // up to whoever requested the udelay/mdelay).
            if self.delay_count == 3_000_000 {
                let rsp = self.regs.gpr[RSP];
                eprintln!("DELAY r8(cycles)={:#x} rdi={:#x} rsp={rsp:#x}", self.regs.gpr[8], self.regs.gpr[RDI]);
                for i in 0..64u64 {
                    if let Ok(v) = self.mem_read(mem, io, rsp.wrapping_add(i * 8), 8, Access::Read) {
                        if (0xffffffff81000000..0xffffffff82e00000).contains(&v) {
                            eprintln!("  stack[{i:2}] = {v:#x}");
                        }
                    }
                }
                self.trace_delay = false;
            }
        }
        match self.try_step(mem, io) {
            Ok(()) => Ok(()),
            Err(err) => match exception_vector(&err) {
                Some((vector, error_code)) if self.sys.idtr_limit > 0 => {
                    // TEMP: log userspace (CPL3) faults — the init-crash hunt. Vector
                    // 6=#UD (missing insn), 13=#GP, 14=#PF (segfault), 19=#XM (SSE).
                    // Log non-#PF exceptions, and any fault whose RIP looks like a
                    // 32-bit-truncated kernel address (the boot bug under hunt).
                    let trunc = (0x8000_0000..0x8400_0000).contains(&self.regs.rip);
                    if self.trace_exc && (vector != 14 || trunc) {
                        eprintln!(
                            "EXC vec={vector} rip={:#x} cr2={:#x} err={:#x} ({err:?})",
                            self.regs.rip,
                            self.sys.cr2,
                            error_code.unwrap_or(0)
                        );
                    }
                    // TEMP: dump the guest PTE walk for a userspace #PF in the V8
                    // young-gen region under hunt (demand-paging bug). Shows the raw
                    // PTE flags at fault time vs the reported error code.
                    // Arm the all-CPL itrace when the specific faulting page is hit,
                    // so we capture the kernel #PF handler from delivery to the BUG.
                    if self.itrace_fault_cr2 != 0 && vector == 14
                        && self.sys.cr2 == self.itrace_fault_cr2
                    {
                        self.itrace_armed = true;
                    }
                    if self.trace_exc && vector == 14 && self.cpl == 3
                        && (0xf436_e4c0_000u64..=0xf436_e500_000).contains(&self.sys.cr2)
                    {
                        let va = self.sys.cr2;
                        let ec = error_code.unwrap_or(0);
                        eprint!(
                            "PFWALK cr2={va:#x} err={ec:#x} (P={} W={} U={}) cr3={:#x}",
                            ec & 1, (ec >> 1) & 1, (ec >> 2) & 1, self.sys.cr3
                        );
                        let mut tbl = self.sys.cr3 & !0xfff;
                        for (lvl, sh) in [(4u32, 39u32), (3, 30), (2, 21), (1, 12)] {
                            let idx = (va >> sh) & 0x1ff;
                            let e = mem.read_uint_le(tbl + idx * 8, 8).unwrap_or(0);
                            eprint!(" L{lvl}[{idx}]={e:#x}(P{} W{})", e & 1, (e >> 1) & 1);
                            if e & 1 == 0 || (lvl > 1 && e & 0x80 != 0) {
                                break;
                            }
                            tbl = e & 0x000f_ffff_ffff_f000;
                        }
                        eprintln!();
                    }
                    self.deliver_interrupt(io, mem, vector, error_code)
                }
                _ => Err(err),
            },
        }
    }

    /// Inject an external (maskable) interrupt vector, e.g. a timer. The caller
    /// is responsible for IF/IDT gating via [`can_take_interrupt`].
    pub fn inject_interrupt(&mut self, io: &mut IoBus, mem: &mut GuestMemory, vector: u8) -> Result<(), X86Error> {
        self.halted = false;
        self.deliver_interrupt(io, mem, vector, None)
    }

    /// Deliver a vector through the long-mode IDT: read the gate, push the
    /// interrupt stack frame (SS, RSP, RFLAGS, CS, RIP, [error code]), load the
    /// handler RIP/CS, and clear IF for an interrupt gate.
    fn deliver_interrupt(
        &mut self,
        io: &mut IoBus,
        mem: &mut GuestMemory,
        vector: u8,
        error_code: Option<u64>,
    ) -> Result<(), X86Error> {
        // The interrupted privilege level, captured before any state change. All
        // the implicit accesses below (IDT gate read, kernel-stack pushes) are
        // performed by the CPU at supervisor privilege (CPL 0) regardless of the
        // interrupted CPL — the IDT and kernel stack sit on supervisor-only pages
        // (e.g. the kernel maps its IDT read-only in the cpu_entry_area alias at
        // 0xfffffe0000000000). Walking them with the user CPL would spuriously
        // #PF. So encode the outgoing CPL now, then drop to CPL 0 for delivery.
        let old_cpl = self.cpl;
        let old_cs = ((self.cs & !3) | old_cpl as u16) as u64;
        self.cpl = 0;

        let gate_addr = self.sys.idtr_base + (vector as u64) * 16;
        let lo = self.mem_read(mem, io, gate_addr, 8, Access::Read)?;
        let hi = self.mem_read(mem, io, gate_addr + 8, 8, Access::Read)?;

        let type_attr = (lo >> 40) & 0xff;
        if type_attr & 0x80 == 0 {
            // Not-present gate → would be #GP/#NP; surface deterministically.
            return Err(X86Error::InvalidEncoding(format!(
                "IDT gate {vector} not present"
            )));
        }
        let selector = ((lo >> 16) & 0xffff) as u16;
        let gate_type = type_attr & 0xf; // 0xE interrupt gate, 0xF trap gate
        let offset =
            (lo & 0xffff) | (((lo >> 48) & 0xffff) << 16) | ((hi & 0xffff_ffff) << 32);

        let old_rip = self.regs.rip;
        let old_rsp = self.regs.gpr[RSP];
        if self.trace_exc {
            eprintln!(
                "DELIVER vec={vector} handler={offset:#x} sel={selector:#x} gate={gate_type} old_rip={old_rip:#x} old_rsp={old_rsp:#x}"
            );
            // Userspace stack-fault backtrace: a #PF from CPL3 whose CR2 sits at/just
            // below RSP is a stack push/probe fault (overflow). Walk the RBP chain so
            // we can tell runaway recursion from an undersized stack. Debug-only; the
            // dbg reads never fault the CPU, and we restore CR2 so nothing observable
            // changes (determinism-safe).
            if vector == 14 && old_cpl == 3 {
                let cr2 = self.sys.cr2;
                if cr2 <= old_rsp.wrapping_add(256) && old_rsp.wrapping_sub(cr2) < 0x8000 {
                    let saved_cr2 = self.sys.cr2;
                    eprint!("STKFAULT cr2={cr2:#x} rsp={old_rsp:#x} rip={old_rip:#x} bt=");
                    let mut rbp = self.regs.gpr[RBP];
                    for _ in 0..24 {
                        if rbp == 0 || rbp & 7 != 0 {
                            break;
                        }
                        let ret = self.read_qword_dbg(mem, rbp.wrapping_add(8));
                        eprint!("{ret:#x} ");
                        let next = self.read_qword_dbg(mem, rbp);
                        if next <= rbp {
                            break;
                        }
                        rbp = next;
                    }
                    eprintln!();
                    self.sys.cr2 = saved_cr2;
                }
            }
        }
        let old_ss = self.ss as u64;
        let old_rflags = self.regs.rflags;

        // Target privilege = the handler code-segment selector's RPL (0 for the
        // kernel gates we model). A privilege increase (e.g. CPL 3 → 0) switches
        // to the kernel stack (TSS.RSP0) before pushing the frame; long mode then
        // pushes SS:RSP:RFLAGS:CS:RIP regardless of whether the level changed.
        let new_cpl = (selector & 3) as u8;
        if new_cpl < old_cpl {
            // Switch to the kernel stack. Read RSP0 live from the TSS (the kernel
            // updates it per task switch); fall back to the directly-set field for
            // tests that don't build a TSS + LTR.
            let rsp0 = if self.sys.tr_base != 0 {
                self.mem_read(mem, io, self.sys.tr_base + 4, 8, Access::Read)?
            } else {
                self.sys.tss_rsp0
            };
            if rsp0 != 0 {
                self.regs.gpr[RSP] = rsp0;
            }
        }

        self.push(io, old_ss, mem)?;
        self.push(io, old_rsp, mem)?;
        self.push(io, old_rflags, mem)?;
        self.push(io, old_cs, mem)?;
        self.push(io, old_rip, mem)?;
        if let Some(code) = error_code {
            self.push(io, code, mem)?;
        }

        self.cs = selector;
        self.cpl = new_cpl;
        self.regs.rip = offset;
        // Clear TF and NT always; clear IF for an interrupt gate (0xE), keep it
        // for a trap gate (0xF).
        self.regs.rflags &= !(flags::TF | flags::NT);
        if gate_type == 0xe {
            self.regs.rflags &= !IF;
        }
        Ok(())
    }

    /// IRETQ: pop RIP, CS, RFLAGS, RSP, SS from the interrupt stack frame.
    fn iret(&mut self, io: &mut IoBus, mem: &mut GuestMemory) -> Result<(), X86Error> {
        let rip = self.pop(io, mem)?;
        let cs = self.pop(io, mem)?;
        let rflags = self.pop(io, mem)?;
        let rsp = self.pop(io, mem)?;
        let ss = self.pop(io, mem)?;
        self.regs.rip = rip;
        self.cs = cs as u16;
        // Restore the privilege level from the saved CS RPL (a return to user mode
        // raises CPL again, so later privileged instructions fault correctly).
        self.cpl = (cs & 3) as u8;
        // Bit 1 always reads 1; keep the rest as restored.
        self.regs.rflags = rflags | 0x2;
        self.regs.gpr[RSP] = rsp;
        self.ss = ss as u16;
        Ok(())
    }

    /// SYSCALL: fast entry to CPL 0. Saves RIP→RCX, RFLAGS→R11, loads RIP from
    /// LSTAR, masks RFLAGS by SFMASK, and loads CS/SS from STAR[47:32].
    fn syscall(&mut self) -> Result<(), X86Error> {
        if self.sys.efer & EFER_SCE == 0 {
            return Err(X86Error::InvalidEncoding("SYSCALL with SCE clear".into())); // #UD
        }
        if self.trace_net {
            // network syscalls: read/write/poll/connect/accept/send/recv/epoll*
            let nr = self.regs.gpr[RAX];
            self.last_sys_net = matches!(nr, 0 | 1 | 7 | 19 | 20 | 42 | 43 | 44 | 45 | 46 | 47 | 202 | 232 | 233 | 281 | 288);
            if self.last_sys_net {
                eprintln!(
                    "SYS nr={} fd={:#x} buf={:#x} len={:#x}",
                    nr, self.regs.gpr[RDI], self.regs.gpr[RSI], self.regs.gpr[RDX]
                );
                // epoll_wait(232)/epoll_pwait(281): stash events buffer for SYSRET dump.
                self.last_net_nr = nr;
                self.last_net_buf = if matches!(nr, 232 | 281) { self.regs.gpr[RSI] } else { 0 };
            }
        }
        if self.trace_mem {
            // memory syscalls: mmap(9)/mprotect(10)/munmap(11)/brk(12)/mremap(25)/madvise(28)
            let nr = self.regs.gpr[RAX];
            self.last_sys_mem = matches!(nr, 9 | 10 | 11 | 12 | 25 | 28);
            if self.last_sys_mem {
                eprintln!(
                    "MEM nr={} addr={:#x} len={:#x} prot={:#x} flags={:#x}",
                    nr, self.regs.gpr[RDI], self.regs.gpr[RSI], self.regs.gpr[RDX], self.regs.gpr[10]
                );
            }
        }
        if self.trace_sys {
            eprintln!(
                "SYS nr={} rdi={:#x} rsi={:#x} rdx={:#x} (rip={:#x})",
                self.regs.gpr[RAX],
                self.regs.gpr[RDI],
                self.regs.gpr[RSI],
                self.regs.gpr[RDX],
                self.regs.rip, // already advanced to the return address
            );
        }
        self.regs.gpr[RCX] = self.regs.rip; // return address (next instruction)
        self.regs.gpr[R11] = self.regs.rflags;
        self.regs.rip = self.msrs.lstar;
        // RFLAGS ← (RFLAGS & ~SFMASK), RF cleared, reserved bit 1 set.
        self.regs.rflags = ((self.regs.rflags & !self.msrs.sfmask) | 0x2) & !(1 << 16);
        let sel = ((self.msrs.star >> 32) & 0xffff) as u16;
        self.cs = sel & 0xfffc; // RPL 0
        self.ss = sel.wrapping_add(8);
        self.cpl = 0;
        Ok(())
    }

    /// SYSRETQ: return to CPL 3. RIP←RCX, RFLAGS←R11 (masked), CS/SS from
    /// STAR[63:48] (+16 for the 64-bit user code selector).
    fn sysret(&mut self, mem: &mut GuestMemory) -> Result<(), X86Error> {
        if self.cpl != 0 {
            return Err(X86Error::GeneralProtection);
        }
        if self.trace_mem && self.last_sys_mem {
            eprintln!("MEMRET rax={:#x}", self.regs.gpr[RAX]);
        }
        if self.trace_net && self.last_sys_net {
            let rax = self.regs.gpr[RAX];
            eprintln!("SYSRET rax={:#x}", rax);
            // Dump the epoll events array (events mask + data per ready fd) — CPL is
            // still 0 here so we can read the user buffer; dbg reads never fault and
            // we restore CR2, so nothing observable changes (determinism-safe).
            if matches!(self.last_net_nr, 232 | 281) && self.last_net_buf != 0 && rax < 64 {
                let n = rax as i64;
                if n > 0 {
                    let saved_cr2 = self.sys.cr2;
                    let buf = self.last_net_buf;
                    eprint!("  EPOLL");
                    for i in 0..n as u64 {
                        let ev = self.read_qword_dbg(mem, buf + i * 12) & 0xffff_ffff;
                        let data = self.read_qword_dbg(mem, buf + i * 12 + 4);
                        eprint!(" [ev={ev:#x} data={data:#x}]");
                    }
                    eprintln!();
                    self.sys.cr2 = saved_cr2;
                }
            }
        }
        if self.trace_sys {
            eprintln!("SYSRET rax={:#x} (rip={:#x})", self.regs.gpr[RAX], self.regs.gpr[RCX]);
        }
        self.regs.rip = self.regs.gpr[RCX];
        self.regs.rflags = (self.regs.gpr[R11] & 0x003c_7fd7) | 0x2;
        let sel = ((self.msrs.star >> 48) & 0xffff) as u16;
        self.cs = sel.wrapping_add(16) | 3; // 64-bit user CS, RPL 3
        self.ss = sel.wrapping_add(8) | 3;
        self.cpl = 3;
        Ok(())
    }

    /// Execute one iteration of a string op; with a REP prefix, rewind RIP so the
    /// next step re-executes it (interruptible per the precise-exception model).
    fn exec_string(
        &mut self,
        io: &mut IoBus,
        sop: StrOp,
        size: Size,
        rep: Rep,
        insn_len: u64,
        mem: &mut GuestMemory,
    ) -> Result<(), X86Error> {
        let has_rep = !matches!(rep, Rep::None);
        if has_rep && self.regs.gpr[RCX] == 0 {
            return Ok(());
        }

        let n = size.bytes();
        let delta = if self.regs.rflags & DF != 0 {
            (n as u64).wrapping_neg()
        } else {
            n as u64
        };

        match sop {
            StrOp::Movs => {
                let v = self.mem_read(mem, io, self.regs.gpr[RSI], n, Access::Read)?;
                self.mem_write(mem, io, self.regs.gpr[RDI], n, v)?;
                self.regs.gpr[RSI] = self.regs.gpr[RSI].wrapping_add(delta);
                self.regs.gpr[RDI] = self.regs.gpr[RDI].wrapping_add(delta);
            }
            StrOp::Stos => {
                let v = self.regs.read(RAX, size, false);
                self.mem_write(mem, io, self.regs.gpr[RDI], n, v)?;
                self.regs.gpr[RDI] = self.regs.gpr[RDI].wrapping_add(delta);
            }
            StrOp::Lods => {
                let v = self.mem_read(mem, io, self.regs.gpr[RSI], n, Access::Read)?;
                self.regs.write(RAX, size, false, v);
                self.regs.gpr[RSI] = self.regs.gpr[RSI].wrapping_add(delta);
            }
            StrOp::Scas => {
                let a = self.regs.read(RAX, size, false);
                let m = self.mem_read(mem, io, self.regs.gpr[RDI], n, Access::Read)?;
                flags::sub(&mut self.regs.rflags, a, m, size);
                self.regs.gpr[RDI] = self.regs.gpr[RDI].wrapping_add(delta);
            }
            StrOp::Cmps => {
                let s = self.mem_read(mem, io, self.regs.gpr[RSI], n, Access::Read)?;
                let d = self.mem_read(mem, io, self.regs.gpr[RDI], n, Access::Read)?;
                flags::sub(&mut self.regs.rflags, s, d, size);
                self.regs.gpr[RSI] = self.regs.gpr[RSI].wrapping_add(delta);
                self.regs.gpr[RDI] = self.regs.gpr[RDI].wrapping_add(delta);
            }
        }

        if has_rep {
            self.regs.gpr[RCX] = self.regs.gpr[RCX].wrapping_sub(1);
            let zf = self.regs.rflags & ZF != 0;
            let more = self.regs.gpr[RCX] != 0
                && match (sop, rep) {
                    // REPE/REPZ continues while equal; REPNE/REPNZ while not equal.
                    (StrOp::Scas | StrOp::Cmps, Rep::F3) => zf,
                    (StrOp::Scas | StrOp::Cmps, Rep::F2) => !zf,
                    _ => true, // plain REP for MOVS/STOS/LODS
                };
            if more {
                self.regs.rip = self.regs.rip.wrapping_sub(insn_len);
            }
        }
        Ok(())
    }

    /// Fetch, decode, and execute one instruction (no exception vectoring).
    fn try_step(&mut self, mem: &mut GuestMemory, io: &mut IoBus) -> Result<(), X86Error> {
        if self.halted {
            return Ok(());
        }

        // Decode-cache lookup: reuse the instruction decoded for this exact
        // (rip, cr3). A hit skips fetch+decode entirely (the dominant per-
        // instruction cost on hot loops).
        let rip = self.regs.rip;
        let cr3 = self.sys.cr3;
        let cache_idx = (rip & DECODE_CACHE_MASK) as usize;
        let cached = {
            let slot = &self.decode_cache[cache_idx];
            if slot.gen == self.decode_gen && slot.rip == rip && slot.cr3 == cr3 {
                slot.insn.clone()
            } else {
                None
            }
        };
        let insn = if let Some(insn) = cached {
            insn
        } else {
            // Miss: fetch through the MMU (or identity when paging is off) and decode.
            let paging = self.sys.paging_enabled();
            let wp = self.sys.write_protect();
            let nxe = self.sys.nxe();
            let cpl = self.cpl;
            // Code fetches at the CS *hidden descriptor* base (`cs<<4` in real mode, 0 in
            // long/flat) — read from the cache latched at the last CS load, NOT recomputed
            // from CR0.PE here. This is what makes the unreal-mode window correct.
            let cs_base = self.cs_base;
            // Decoder default operand/address size follows the cached CS code width.
            let dmode = match self.code_bits {
                16 => DecodeMode::Bits16,
                32 => DecodeMode::Bits32,
                _ => DecodeMode::Bits64,
            };
            let result = {
                // Cache the translation of the current code page so the multi-byte
                // fetch does ONE page walk per page, not one per byte (an
                // instruction averages ~4 bytes → ~4 redundant walks). Bytes in a
                // page are physically contiguous; the first translate set the A bit
                // for the whole page, so this is behaviour-identical.
                let mut cached_page = u64::MAX;
                let mut cached_pa_base = 0u64;
                let mut fetch = |off: u64| -> Result<u8, X86Error> {
                    // `off` is the instruction-pointer offset from `decode`; the code linear
                    // address is CS.base + offset (CS.base is 0 in long/protected mode).
                    let va = cs_base.wrapping_add(off);
                    if !mmu::is_canonical(va) {
                        return Err(X86Error::GeneralProtection);
                    }
                    let pa = if paging {
                        let page = va & !0xfff;
                        if page != cached_page {
                            let pa = mmu::translate(mem, cr3, va, Access::Execute, cpl, wp, nxe)
                                .map_err(|f| X86Error::PageFault {
                                    error_code: f.error_code,
                                    addr: f.addr,
                                })?;
                            cached_page = page;
                            cached_pa_base = pa & !0xfff;
                        }
                        cached_pa_base | (va & 0xfff)
                    } else {
                        va
                    };
                    Ok(mem.read_u8(pa)?)
                };
                decode(&mut fetch, rip, dmode)
            };
            let insn = match result {
                Ok(insn) => insn,
                Err(X86Error::PageFault { error_code, addr }) => {
                    self.sys.cr2 = addr;
                    return Err(X86Error::PageFault { error_code, addr });
                }
                Err(other) => {
                    if self.trace_exc {
                        // Dump the undecodable instruction's bytes (SF_EXCTRACE) so a
                        // missing opcode surfaced by a real workload can be identified.
                        let mut bytes = String::new();
                        for i in 0..16u64 {
                            let b = self
                                .translate(mem, rip.wrapping_add(i), Access::Execute)
                                .ok()
                                .and_then(|pa| mem.read_u8(pa).ok());
                            bytes.push_str(&b.map(|b| format!("{b:02x} ")).unwrap_or_default());
                        }
                        eprintln!("DECODE-ERR rip={rip:#x} cpl={} err={other:?} bytes=[{bytes}]", self.cpl);
                    }
                    return Err(other);
                }
            };
            // Record in the cache and mark the instruction's (start) physical page
            // so a later store to it flushes the cache (SMC/patched-code safety).
            // Only cache when we can resolve the page; on the off chance translate
            // disagrees with the just-succeeded fetch, simply skip caching.
            let code_page = if paging {
                mmu::translate(mem, cr3, rip, Access::Execute, cpl, wp, nxe)
                    .ok()
                    .map(|pa| pa >> 12)
            } else {
                Some(rip >> 12)
            };
            if let Some(page) = code_page {
                self.mark_code_page(page);
                self.decode_cache[cache_idx] = DecodeSlot {
                    gen: self.decode_gen,
                    rip,
                    cr3,
                    insn: Some(insn.clone()),
                };
            }
            insn
        };

        // Debug: function-entry watch. When execution reaches `fn_watch`, log the
        // System-V arg registers (rdi/rsi/rdx/rcx) and the return address at [rsp]
        // — used to find which caller passes a bad argument (e.g. the memblock
        // reservation with a virtual/huge size).
        if self.fn_watch != 0 && self.regs.rip == self.fn_watch {
            // memblock.reserved: cnt @ memblock+56, regions ptr @ memblock+80.
            let ret = self.read_qword_dbg(mem, self.regs.gpr[RSP]);
            let cnt = self.read_qword_dbg(mem, 0xffff_ffff_8290_ec58);
            let regions = self.read_qword_dbg(mem, 0xffff_ffff_8290_ec70);
            eprintln!(
                "FNWATCH {:#x} rdi={:#x} rsi={:#x} rdx={:#x} ret={:#x} | reserved.cnt={cnt:#x} regions={regions:#x}",
                self.fn_watch,
                self.regs.gpr[RDI],
                self.regs.gpr[RSI],
                self.regs.gpr[RDX],
                ret
            );
            // On the bad (huge-end) call, dump the caller's loop code (around the
            // return address) so the cnt-bound comparison can be disassembled.
            if self.regs.gpr[RSI] > 0xffff_ffff_0000_0000 {
                let base = ret.wrapping_sub(0x50);
                let mut code = String::new();
                for i in 0..0x80u64 {
                    let b = self
                        .translate(mem, base.wrapping_add(i), Access::Read)
                        .ok()
                        .and_then(|pa| mem.read_u8(pa).ok());
                    code.push_str(&b.map(|b| format!("{b:02x}")).unwrap_or_else(|| "00".into()));
                }
                eprintln!("  CALLERCODE base={base:#x} {code}");
            }
        }

        // RIP advances to the next instruction before relative targets are
        // applied (x86 relative displacements are from end-of-instruction).
        let start_rip = self.regs.rip;
        let old_rsi = self.regs.gpr[RSI];
        self.regs.rip = self.regs.rip.wrapping_add(insn.len);
        let result = self.execute(&insn, mem, io);
        // Debug: catch the instruction that drops RSI's top 12 bits (the boot-spin
        // `rsi=0x000f…` value that should be `0xffff…`). Logs the faulting RIP + the
        // instruction bytes so it can be disassembled directly.
        if self.trace_rsi {
            let v = self.regs.gpr[RSI];
            // The boot-spin bug turns a canonical high address (0xffff…) into a
            // non-canonical one (0x000f…) by clearing the top bits — catch RSI
            // becoming non-canonical.
            // Boot-spin signature: RSI gets an address whose top 12 bits were
            // cleared (`vaddr >> 12` page index used where a physical PFN was
            // expected — a phys/virt mismatch). Dumps the call stack + code window.
            if v != old_rsi && v >> 52 == 0 && v >> 40 != 0 {
                // Dump a window around the instruction (translation is live here),
                // base = start_rip - 0x30, length 0xc0 — enough to disassemble the
                // surrounding loop.
                let base = start_rip.wrapping_sub(0x30);
                let mut code = String::new();
                for i in 0..0xc0u64 {
                    let b = self
                        .translate(mem, base.wrapping_add(i), Access::Read)
                        .ok()
                        .and_then(|pa| mem.read_u8(pa).ok());
                    code.push_str(&b.map(|b| format!("{b:02x}")).unwrap_or_else(|| "00".into()));
                }
                // Walk the stack for kernel-text return addresses (call chain).
                let mut stack = String::new();
                let rsp = self.regs.gpr[RSP];
                for i in 0..48u64 {
                    let a = rsp.wrapping_add(i * 8);
                    if let Some(q) = self
                        .translate(mem, a, Access::Read)
                        .ok()
                        .and_then(|pa| mem.read_uint_le(pa, 8).ok())
                    {
                        if (0xffff_ffff_8100_0000..0xffff_ffff_8300_0000).contains(&q) {
                            stack.push_str(&format!("{q:#x} "));
                        }
                    }
                }
                eprintln!(
                    "RSItrace {old_rsi:#x} -> {v:#x} rip={start_rip:#x}\n  stack: {stack}\n  win={code}"
                );
            }
        }
        match result {
            Ok(()) => Ok(()),
            // Faults are restartable: the architecturally-faulting instruction's
            // RIP must be the one delivered (and re-executed after the handler
            // returns). Without this, a demand-paging #PF on a store would skip
            // the store — corrupting, e.g., the first byte of each new page during
            // kernel decompression.
            Err(e) if is_restartable_fault(&e) => {
                self.regs.rip = start_rip;
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Debug helper: best-effort read of a qword at a virtual address.
    fn read_qword_dbg(&mut self, mem: &mut GuestMemory, va: u64) -> u64 {
        self.translate(mem, va, Access::Read)
            .ok()
            .and_then(|pa| mem.read_uint_le(pa, 8).ok())
            .unwrap_or(0)
    }

    /// Translate a linear address for a data access, setting CR2 on fault.
    fn translate(
        &mut self,
        mem: &mut GuestMemory,
        vaddr: u64,
        access: Access,
    ) -> Result<u64, X86Error> {
        // Non-canonical linear addresses raise #GP before any page walk, in long
        // mode, regardless of whether paging is enabled.
        if !mmu::is_canonical(vaddr) {
            return Err(X86Error::GeneralProtection);
        }
        if !self.sys.paging_enabled() {
            return Ok(vaddr);
        }
        mmu::translate(
            mem,
            self.sys.cr3,
            vaddr,
            access,
            self.cpl,
            self.sys.write_protect(),
            self.sys.nxe(),
        )
        .map_err(|f| {
            self.sys.cr2 = f.addr;
            X86Error::PageFault {
                error_code: f.error_code,
                addr: f.addr,
            }
        })
    }

    /// Read `bytes` from a linear address, translating each byte (handles page
    /// crossing and sets accessed bits).
    // ---- x86-TSO store buffer (Stage 2) ----

    /// Push a normal WB-RAM store into this vCPU's TSO store buffer with a seed-drawn drain
    /// deadline (global instruction count). Only called when `tso_max != 0`.
    fn sb_push(&mut self, pa: u64, bytes: usize, value: u64) {
        let draw = self.sb_rng.next_bounded(self.tso_max);
        // Linger at least 1 instruction so the store is genuinely buffered (deadline > now).
        let deadline = self.tsc.wrapping_add(draw).wrapping_add(1);
        self.store_buf.push_back(StoreBufEntry { pa, bytes: bytes as u8, value, deadline });
    }

    /// Store-forward a load from this vCPU's own buffer (a core always sees its own writes).
    /// Returns `Some(value)` if a single buffered store fully covers `[pa, pa+bytes)` (most
    /// recent wins). If a store PARTIALLY overlaps but none fully covers, drains the buffer to
    /// `mem` and returns `None` so the caller reads the now-correct memory. `None` (read
    /// memory) when nothing in the buffer overlaps.
    fn sb_forward(&mut self, pa: u64, bytes: usize, mem: &mut GuestMemory) -> Option<u64> {
        let lo = pa;
        let hi = pa.wrapping_add(bytes as u64);
        let mut partial = false;
        for e in self.store_buf.iter().rev() {
            let elo = e.pa;
            let ehi = e.pa.wrapping_add(e.bytes as u64);
            if ehi <= lo || elo >= hi {
                continue; // no overlap
            }
            if elo <= lo && hi <= ehi {
                let shift = (lo - elo) * 8;
                let mask = if bytes >= 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 };
                return Some((e.value >> shift) & mask);
            }
            partial = true; // overlaps but does not fully cover → memory merge needed
            break;
        }
        if partial {
            self.sb_drain_all(mem);
        }
        None
    }

    /// Drain this vCPU's ENTIRE store buffer to shared memory in FIFO (program) order — the
    /// effect of a fence, a LOCKed atomic, HLT, or a page-straddling access. No-op (and thus
    /// byte-identical) when the buffer is empty, which it always is with TSO off.
    fn sb_drain_all(&mut self, mem: &mut GuestMemory) {
        while let Some(e) = self.store_buf.pop_front() {
            let _ = mem.write_uint_le(e.pa, e.bytes as usize, e.value);
        }
    }

    fn mem_read(
        &mut self,
        mem: &mut GuestMemory,
        io: &mut IoBus,
        vaddr: u64,
        bytes: usize,
        access: Access,
    ) -> Result<u64, X86Error> {
        let pa0 = self.translate(mem, vaddr, access)?;
        if let Some(off) = Lapic::offset_of(pa0) {
            return Ok(self.lapic.read(off, self.tsc) as u64);
        }
        if let Some(off) = VirtioBlk::offset_of(pa0) {
            return Ok(io.virtio.read(off) as u64);
        }
        if let Some(off) = VirtioRng::offset_of(pa0) {
            return Ok(io.virtio_rng.read(off) as u64);
        }
        if let (Some(net), Some(off)) = (&io.virtio_net, VirtioNet::offset_of(pa0)) {
            return Ok(net.read(off) as u64);
        }
        if let Some(off) = Ioapic::offset_of(pa0) {
            return Ok(io.ioapic.read(off) as u64);
        }
        // Fast path: the whole access lies within one 4 KiB page, so all bytes are
        // physically contiguous from pa0 — one translation instead of one full page
        // walk per byte (an 8-byte read was doing 8 walks). Behaviour-identical: the
        // bytes share a page, so the single translate already set the A bit for them.
        let page_off = (vaddr & 0xfff) as usize;
        if page_off + bytes <= 0x1000 {
            // TSO: a load first checks this core's own store buffer (store forwarding).
            if self.tso_max != 0 {
                if let Some(v) = self.sb_forward(pa0, bytes, mem) {
                    return Ok(v);
                }
            }
            return Ok(mem.read_uint_le(pa0, bytes)?);
        }
        // Slow path: the access straddles a page boundary — translate per byte. TSO: flush this
        // core's buffer first so the per-byte reads see all of its own prior stores.
        if self.tso_max != 0 {
            self.sb_drain_all(mem);
        }
        let mut value = mem.read_u8(pa0)? as u64;
        for i in 1..bytes {
            let pa = self.translate(mem, vaddr.wrapping_add(i as u64), access)?;
            value |= (mem.read_u8(pa)? as u64) << (i * 8);
        }
        Ok(value)
    }

    /// Write the low `bytes` of `value` to a linear address (translated, dirty).
    fn mem_write(
        &mut self,
        mem: &mut GuestMemory,
        io: &mut IoBus,
        vaddr: u64,
        bytes: usize,
        value: u64,
    ) -> Result<(), X86Error> {
        if self.watch_addr != 0 && self.itrace_armed && vaddr == self.watch_addr
            && (self.watch_val == 0 || value == self.watch_val)
        {
            eprintln!(
                "WATCH cur_rip={:#x} writes val={:#x} ({}B) to addr={:#x}",
                self.cur_rip, value, bytes, vaddr
            );
        }
        let pa0 = self.translate(mem, vaddr, Access::Write)?;
        if let Some(off) = Lapic::offset_of(pa0) {
            self.lapic.write(off, value as u32, self.tsc);
            return Ok(());
        }
        if let Some(off) = VirtioBlk::offset_of(pa0) {
            io.virtio.write(off, value as u32, mem, self.tsc);
            return Ok(());
        }
        if let Some(off) = VirtioRng::offset_of(pa0) {
            io.virtio_rng.write(off, value as u32, mem);
            return Ok(());
        }
        if VirtioNet::offset_of(pa0).is_some() {
            if let (Some(net), Some(off)) = (io.virtio_net.as_mut(), VirtioNet::offset_of(pa0)) {
                net.write(off, value as u32, mem, self.tsc);
            }
            return Ok(());
        }
        if let Some(off) = Ioapic::offset_of(pa0) {
            io.ioapic.write(off, value as u32);
            return Ok(());
        }
        // Self-modifying / patched code: if this store lands on a page that holds a
        // cached instruction, the decode cache for it is now stale — flush it. The
        // bitmap test is ~free for the overwhelmingly common data write. (A store
        // straddling into a second code page is handled by the per-byte slow path
        // below, which re-checks each translated page.)
        if self.is_code_page(pa0 >> 12) {
            self.flush_decode_cache();
        }
        if self.wwatch_hi != 0 && (self.wwatch_lo..self.wwatch_hi).contains(&vaddr) {
            eprintln!(
                "WVA va={vaddr:#x} <- {value:#x} ({bytes}B) rip~{:#x}",
                self.regs.rip
            );
        }
        if self.vwatch != 0 && (value & 0xffff_ffff) == (self.vwatch & 0xffff_ffff) {
            eprintln!(
                "VWATCH va={vaddr:#x} <- {value:#x} ({bytes}B) rip~{:#x}",
                self.regs.rip
            );
        }
        if self.watch != 0 && (pa0..pa0 + bytes as u64).contains(&self.watch) {
            let rip = self.regs.rip;
            let mut code = String::new();
            for off in 0..24u64 {
                let va = rip.wrapping_sub(16).wrapping_add(off);
                let b = self
                    .translate(mem, va, Access::Read)
                    .ok()
                    .and_then(|pa| mem.read_u8(pa).ok());
                code.push_str(&match b {
                    Some(b) => format!("{b:02x}"),
                    None => "??".into(),
                });
            }
            eprintln!(
                "WATCH {:#x} by rip~{:#x}: store {bytes}B value={:#018x} pa={:#x} code[rip-16..+8]={}",
                self.watch, rip, value, pa0, code
            );
        }
        // Fast path: whole store within one page → physically contiguous from pa0,
        // a single translation (mirrors mem_read; the per-byte loop did one page
        // walk per byte). The single translate already set the D bit for the page.
        let page_off = (vaddr & 0xfff) as usize;
        if page_off + bytes <= 0x1000 {
            // TSO: a normal WB-RAM store lands in this core's store buffer (with a seed-drawn
            // drain deadline) rather than memory. Device MMIO already returned above, so this
            // is genuine RAM. LOCKed atomics bracket themselves with drains, so their store is
            // flushed within the same instruction (globally visible immediately).
            if self.tso_max != 0 {
                self.sb_push(pa0, bytes, value);
                return Ok(());
            }
            return Ok(mem.write_uint_le(pa0, bytes, value)?);
        }
        // Slow path: the store straddles a page boundary — translate per byte, and
        // flush the decode cache if any byte lands on a cached code page. TSO: flush this
        // core's buffer first so the straddling store is ordered after its own prior stores.
        if self.tso_max != 0 {
            self.sb_drain_all(mem);
        }
        mem.write_uint_le(pa0, 1, value & 0xff)?;
        for i in 1..bytes {
            let pa = self.translate(mem, vaddr.wrapping_add(i as u64), Access::Write)?;
            if self.is_code_page(pa >> 12) {
                self.flush_decode_cache();
            }
            mem.write_uint_le(pa, 1, (value >> (i * 8)) & 0xff)?;
        }
        Ok(())
    }

    fn execute(
        &mut self,
        insn: &DecodedInsn,
        mem: &mut GuestMemory,
        io: &mut IoBus,
    ) -> Result<(), X86Error> {
        match insn.op {
            Op::Mov => {
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)?;
                self.write(io, insn.dst.as_ref().unwrap(), value, mem)?;
            }
            Op::MovToSeg(n) => {
                let sel = self.read(io, insn.src.as_ref().unwrap(), mem)? as u16;
                // In flat long mode loading a data segment selector only records
                // the selector (bases stay 0; FS/GS bases come from their MSRs).
                match n {
                    0 => self.es = sel,
                    1 => return Err(X86Error::GeneralProtection), // MOV to CS is illegal
                    2 => self.ss = sel,
                    3 => self.ds = sel,
                    4 => self.fs_sel = sel,
                    5 => self.gs_sel = sel,
                    _ => return Err(X86Error::InvalidEncoding("bad segment register".into())),
                }
            }
            Op::MovFromSeg(n) => {
                let sel = match n {
                    0 => self.es,
                    1 => self.cs,
                    2 => self.ss,
                    3 => self.ds,
                    4 => self.fs_sel,
                    5 => self.gs_sel,
                    _ => return Err(X86Error::InvalidEncoding("bad segment register".into())),
                };
                self.write(io, insn.dst.as_ref().unwrap(), sel as u64, mem)?;
            }
            Op::Add | Op::Or | Op::And | Op::Sub | Op::Xor | Op::Cmp | Op::Test | Op::Adc
            | Op::Sbb => {
                self.alu(io, insn, mem)?;
            }
            Op::Cmov(cond) => {
                // The memory source is always read (can fault); the write is
                // conditional, so a non-taken 32-bit CMOV does not zero-extend.
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)?;
                if condition(self.regs.rflags, cond) {
                    self.write(io, insn.dst.as_ref().unwrap(), value, mem)?;
                }
            }
            Op::Set(cond) => {
                let value = condition(self.regs.rflags, cond) as u64;
                self.write(io, insn.dst.as_ref().unwrap(), value, mem)?;
            }
            Op::Rdtsc | Op::Rdtscp => {
                let tsc = self.tsc.wrapping_add(self.msrs.tsc_offset);
                self.regs.write(RAX, Size::Dword, false, tsc & 0xffff_ffff);
                self.regs.write(RDX, Size::Dword, false, tsc >> 32);
                if matches!(insn.op, Op::Rdtscp) {
                    self.regs
                        .write(RCX, Size::Dword, false, self.msrs.tsc_aux & 0xffff_ffff);
                }
            }
            Op::Xchg => {
                // XCHG with memory is implicitly LOCKed → a full barrier under TSO: drain this
                // core's buffer first (fence + fresh read), and again after so the swap is
                // globally visible immediately. No-ops with TSO off.
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
                let dst = insn.dst.as_ref().unwrap();
                let src = insn.src.as_ref().unwrap();
                let a = self.read(io, dst, mem)?;
                let b = self.read(io, src, mem)?;
                self.write(io, dst, b, mem)?;
                self.write(io, src, a, mem)?;
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
            }
            Op::Xadd => {
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
                let dst = insn.dst.as_ref().unwrap();
                let src = insn.src.as_ref().unwrap();
                let size = operand_size(dst);
                let d = self.read(io, dst, mem)?;
                let s = self.read(io, src, mem)?;
                let sum = flags::add(&mut self.regs.rflags, d, s, size);
                // Precise-exception safety: commit the (possibly page-faulting)
                // destination memory store BEFORE writing the source register. The
                // source is always a register (XADD r/m, r), so it cannot fault;
                // the r/m destination can. If we wrote the source first and the
                // destination store then faulted (e.g. demand-paging a fresh page),
                // the instruction would restart with the source already clobbered
                // by the old destination value — so on re-execution the delta is
                // wrong and the add is lost. (A 64-bit LOCK XADD silently dropping
                // its delta is exactly how the Go runtime's sysMemStat overflowed.)
                // Same class of bug as the push/RSP-leak fix.
                self.write(io, dst, sum, mem)?;
                self.write(io, src, d, mem)?;
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
            }
            Op::Cmpxchg => {
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let mask = size.mask();
                let acc = self.regs.read(RAX, size, false) & mask;
                let dst_val = self.read(io, dst, mem)? & mask;
                flags::sub(&mut self.regs.rflags, acc, dst_val, size); // CMP acc, dst
                if acc == dst_val {
                    let src_val = self.read(io, insn.src.as_ref().unwrap(), mem)?;
                    self.write(io, dst, src_val, mem)?;
                } else {
                    self.regs.write(RAX, size, false, dst_val);
                }
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
            }
            Op::Bt | Op::Bts | Op::Btr | Op::Btc => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let src = insn.src.as_ref().unwrap();
                // For a memory bit string with the bit index in a register, the
                // index is SIGNED and addresses a bit outside the first operand:
                // byte = effaddr + (offset >> 3), bit = offset & 7. (For a register
                // destination, or an immediate index, the index is masked to the
                // operand width.)
                if matches!(dst, Operand::Mem { .. }) && matches!(src, Operand::Reg { .. }) {
                    let bitoff = sext(self.read(io, src, mem)?, size.bits());
                    let byte_addr = self
                        .effective_address(dst)
                        .wrapping_add((bitoff >> 3) as u64);
                    let bit = (bitoff & 7) as u32;
                    let byte = self.mem_read(mem, io, byte_addr, 1, Access::Read)?;
                    let cf = (byte >> bit) & 1 != 0;
                    set_bit_flags(&mut self.regs.rflags, cf);
                    let new = match insn.op {
                        Op::Bts => byte | (1 << bit),
                        Op::Btr => byte & !(1 << bit),
                        Op::Btc => byte ^ (1 << bit),
                        _ => byte,
                    };
                    if !matches!(insn.op, Op::Bt) {
                        self.mem_write(mem, io, byte_addr, 1, new)?;
                    }
                } else {
                    let bit = self.read(io, src, mem)? % size.bits() as u64;
                    let val = self.read(io, dst, mem)?;
                    let cf = (val >> bit) & 1 != 0;
                    set_bit_flags(&mut self.regs.rflags, cf);
                    let new = match insn.op {
                        Op::Bts => val | (1 << bit),
                        Op::Btr => val & !(1 << bit),
                        Op::Btc => val ^ (1 << bit),
                        _ => val, // BT: test only
                    };
                    if !matches!(insn.op, Op::Bt) {
                        self.write(io, dst, new, mem)?;
                    }
                }
            }
            Op::Bsf | Op::Bsr => {
                let src = insn.src.as_ref().unwrap();
                let size = operand_size(src);
                let value = self.read(io, src, mem)? & size.mask();
                let rflags = &mut self.regs.rflags;
                flags::set_flag(rflags, ZF, value == 0);
                for b in [CF, OF, SF, AF, PF] {
                    flags::set_flag(rflags, b, false);
                }
                if value != 0 {
                    let index = if matches!(insn.op, Op::Bsf) {
                        value.trailing_zeros()
                    } else {
                        63 - value.leading_zeros()
                    };
                    self.write(io, insn.dst.as_ref().unwrap(), index as u64, mem)?;
                }
            }
            Op::Popcnt => {
                let src = insn.src.as_ref().unwrap();
                let size = operand_size(src);
                let value = self.read(io, src, mem)? & size.mask();
                let rflags = &mut self.regs.rflags;
                flags::set_flag(rflags, ZF, value == 0);
                for b in [CF, OF, SF, AF, PF] {
                    flags::set_flag(rflags, b, false);
                }
                self.write(io, insn.dst.as_ref().unwrap(), value.count_ones() as u64, mem)?;
            }
            Op::Lzcnt | Op::Tzcnt => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let width = size.bits() as u32;
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let count = if matches!(insn.op, Op::Lzcnt) {
                    if width == 64 { value.leading_zeros() } else { (value as u32).leading_zeros() }
                } else if value == 0 {
                    width
                } else {
                    value.trailing_zeros()
                };
                self.write(io, dst, count as u64, mem)?;
                let rflags = &mut self.regs.rflags;
                flags::set_flag(rflags, CF, value == 0);
                flags::set_flag(rflags, ZF, count == 0);
                for b in [OF, SF, AF, PF] {
                    flags::set_flag(rflags, b, false);
                }
            }
            Op::Movbe { store } => {
                let dst = insn.dst.as_ref().unwrap();
                let src = insn.src.as_ref().unwrap();
                let size = operand_size(if store { src } else { dst });
                let v = self.read(io, src, mem)? & size.mask();
                let swapped = match size {
                    Size::Word => (v as u16).swap_bytes() as u64,
                    Size::Qword => v.swap_bytes(),
                    _ => (v as u32).swap_bytes() as u64,
                };
                self.write(io, dst, swapped, mem)?;
            }
            Op::Andn { vvvv } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.regs.read(vvvv as usize, size, false) & size.mask();
                let b = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let r = (!a & b) & size.mask();
                self.write(io, dst, r, mem)?;
                set_bmi_flags(&mut self.regs.rflags, r, size, false);
            }
            Op::Blsx { kind } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let s = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let (r, cf) = match kind {
                    1 => (s & s.wrapping_sub(1), s == 0),       // BLSR: clear lowest set
                    2 => (s ^ s.wrapping_sub(1), s == 0),       // BLSMSK: mask to lowest set
                    _ => (s & s.wrapping_neg(), s != 0),        // BLSI(3): isolate lowest set
                };
                let r = r & size.mask();
                self.write(io, dst, r, mem)?;
                set_bmi_flags(&mut self.regs.rflags, r, size, cf);
            }
            Op::Bextr { vvvv } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let width = size.bits() as u32;
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let ctrl = self.regs.read(vvvv as usize, size, false);
                let start = (ctrl & 0xff) as u32;
                let len = ((ctrl >> 8) & 0xff) as u32;
                let r = if start >= width {
                    0
                } else {
                    let sh = value >> start;
                    if len >= width { sh } else { sh & ((1u64 << len) - 1) }
                } & size.mask();
                self.write(io, dst, r, mem)?;
                let rflags = &mut self.regs.rflags;
                flags::set_flag(rflags, ZF, r == 0);
                for b in [CF, OF, SF, AF, PF] {
                    flags::set_flag(rflags, b, false);
                }
            }
            Op::Bzhi { vvvv } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let width = size.bits() as u32;
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let index = (self.regs.read(vvvv as usize, size, false) & 0xff) as u32;
                let r = if index >= width { value } else { value & ((1u64 << index) - 1) } & size.mask();
                self.write(io, dst, r, mem)?;
                set_bmi_flags(&mut self.regs.rflags, r, size, index >= width);
            }
            Op::Mulx { vvvv } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.regs.read(RDX, size, false) & size.mask();
                let b = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let (lo, hibits) = if matches!(size, Size::Qword) {
                    let p = (a as u128) * (b as u128);
                    (p as u64, (p >> 64) as u64)
                } else {
                    let p = a * b;
                    (p & 0xffff_ffff, (p >> 32) & 0xffff_ffff)
                };
                // ModRM.reg (dst) = high half, VEX.vvvv = low half. Write the low
                // (vvvv) first so the high (dst) wins if the two registers overlap.
                self.regs.write(vvvv as usize, size, false, lo);
                self.write(io, dst, hibits, mem)?;
            }
            Op::Pdep { vvvv } | Op::Pext { vvvv } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let value = self.regs.read(vvvv as usize, size, false) & size.mask();
                let mask = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let r = if matches!(insn.op, Op::Pdep { .. }) {
                    pdep(value, mask)
                } else {
                    pext(value, mask)
                } & size.mask();
                self.write(io, dst, r, mem)?;
            }
            Op::Rorx { imm } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let width = size.bits() as u32;
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let n = imm as u32 & (width - 1);
                let r = if width == 64 {
                    value.rotate_right(n)
                } else {
                    (value as u32).rotate_right(n) as u64
                };
                self.write(io, dst, r, mem)?;
            }
            Op::Sarx { kind, vvvv } => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let width = size.bits() as u32;
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)? & size.mask();
                let count = self.regs.read(vvvv as usize, size, false) as u32 & (width - 1);
                let r = match kind {
                    ShiftKind::Sll => value << count,
                    ShiftKind::Srl => value >> count,
                    ShiftKind::Sra => {
                        if width == 64 {
                            ((value as i64) >> count) as u64
                        } else {
                            ((value as i32) >> count) as u64
                        }
                    }
                } & size.mask();
                self.write(io, dst, r, mem)?;
            }
            Op::Str(sop, size, rep) => self.exec_string(io, sop, size, rep, insn.len, mem)?,
            Op::MovCr { write, reg } => {
                if self.cpl != 0 {
                    return Err(X86Error::GeneralProtection);
                }
                let gpr = insn.dst.as_ref().unwrap();
                if write {
                    let v = self.read(io, gpr, mem)?;
                    self.set_cr(reg, v)?;
                } else {
                    let v = self.get_cr(reg)?;
                    self.write(io, gpr, v, mem)?;
                }
            }
            Op::MovDr { write, reg } => {
                if self.cpl != 0 {
                    return Err(X86Error::GeneralProtection);
                }
                let gpr = insn.dst.as_ref().unwrap();
                let idx = (reg & 7) as usize;
                if write {
                    self.sys.dr[idx] = self.read(io, gpr, mem)?;
                } else {
                    let v = self.sys.dr[idx];
                    self.write(io, gpr, v, mem)?;
                }
            }
            Op::CallIndirect => {
                let target = self.read(io, insn.dst.as_ref().unwrap(), mem)?;
                let ret = self.regs.rip;
                self.push(io, ret, mem)?;
                self.regs.rip = target;
            }
            Op::JmpIndirect => {
                self.regs.rip = self.read(io, insn.dst.as_ref().unwrap(), mem)?;
            }
            Op::Pushf => {
                // Flags operand width (PUSHFW=2 / PUSHFD=4 / PUSHFQ=8) from the decoder,
                // honouring the 0x66 prefix; falls back to the stack width if unset.
                let w = insn.rel.map(|r| r as u64).unwrap_or(self.stack_op_bytes());
                let v = self.regs.rflags;
                self.push_sized(io, v, w, mem)?;
            }
            Op::Popf => {
                // POPF(Q) honours CPL/IOPL: IF (bit 9) is only writable when
                // CPL <= IOPL, and IOPL (bits 12-13) only at CPL 0. Otherwise those
                // bits are PRESERVED from the current RFLAGS. Copying them
                // unconditionally lets a userspace `push rax; popf` (as V8's Smi
                // handlers do) clear IF, which then trips the kernel's
                // "userspace fault with IRQs off" BUG on the next #PF.
                //
                // The operand width also bounds which flags are writable: POPFW (2 bytes)
                // touches only the low 16 flags — bits 16-31 (RF/VM/AC/VIF/VIP/ID) are
                // PRESERVED — whereas POPFD (4) can toggle ID (bit 21), which is exactly how
                // `verify_cpu` probes for CPUID support in the real-mode AP trampoline.
                let w = insn.rel.map(|r| r as u64).unwrap_or(self.stack_op_bytes());
                let v = self.pop_sized(io, w, mem)?;
                let old = self.regs.rflags;
                let iopl = ((old >> 12) & 3) as u8;
                let wmask = if w >= 8 { u64::MAX } else { (1u64 << (w * 8)) - 1 };
                let mut mask = RFLAGS_MASK & wmask;
                if self.cpl > iopl {
                    mask &= !0x200; // IF preserved
                }
                if self.cpl != 0 {
                    mask &= !0x3000; // IOPL preserved
                }
                self.regs.rflags = (old & !mask) | (v & mask) | 0x2;
            }
            Op::Lahf => {
                let ah = (self.regs.rflags & 0xd5) | 0x02;
                self.regs.write(RAX, Size::Byte, true, ah);
            }
            Op::Sahf => {
                let ah = self.regs.read(RAX, Size::Byte, true);
                self.regs.rflags = (self.regs.rflags & !0xd5) | (ah & 0xd5);
            }
            Op::XmmLoad(w) => {
                let value = self.read_value128(io, insn.src.as_ref().unwrap(), w, mem)?;
                self.write_value128(io, insn.dst.as_ref().unwrap(), value, w, mem)?;
            }
            Op::XmmStore(w) => {
                let value = self.read_value128(io, insn.src.as_ref().unwrap(), w, mem)?;
                self.write_value128(io, insn.dst.as_ref().unwrap(), value, w, mem)?;
            }
            Op::XmmScalarMov(w) => {
                // MOVSS(4)/MOVSD(8). A register→register move preserves the upper
                // bits of the destination XMM; a memory source zero-extends.
                let src = insn.src.as_ref().unwrap();
                let dst = insn.dst.as_ref().unwrap();
                let lowmask = (1_u128 << (w as u32 * 8)) - 1;
                let value = self.read_value128(io, src, w, mem)? & lowmask;
                match dst {
                    Operand::Xmm { id } => {
                        let id = *id;
                        if matches!(src, Operand::Xmm { .. }) {
                            // reg→reg: merge low bits, preserve [127:w*8].
                            self.xmm[id] = (self.xmm[id] & !lowmask) | value;
                        } else {
                            // mem→reg: zero-extend the destination.
                            self.xmm[id] = value;
                        }
                    }
                    Operand::Mem { .. } => self.write_value128(io, dst, value, w, mem)?,
                    _ => return Err(X86Error::InvalidEncoding("scalar move dest".into())),
                }
            }
            Op::XmmMovHalf { high, store } => {
                let src = insn.src.as_ref().unwrap();
                let dst = insn.dst.as_ref().unwrap();
                let lowmask: u128 = 0xffff_ffff_ffff_ffff;
                if store {
                    // Store one 64-bit half of an XMM to memory (MOVLPS/MOVHPS store).
                    let sid = xmm_id(src)?;
                    let half = if high {
                        (self.xmm[sid] >> 64) & lowmask
                    } else {
                        self.xmm[sid] & lowmask
                    };
                    self.write_value128(io, dst, half, 8, mem)?;
                } else {
                    let id = xmm_id(dst)?;
                    match src {
                        // Register form: cross-half moves MOVHLPS / MOVLHPS.
                        Operand::Xmm { id: sid } => {
                            let sid = *sid;
                            if high {
                                // MOVLHPS: dst[127:64] = src[63:0]
                                self.xmm[id] = (self.xmm[id] & lowmask) | ((self.xmm[sid] & lowmask) << 64);
                            } else {
                                // MOVHLPS: dst[63:0] = src[127:64]
                                self.xmm[id] = (self.xmm[id] & (lowmask << 64)) | ((self.xmm[sid] >> 64) & lowmask);
                            }
                        }
                        // Memory form: load 64 bits into one half, preserve the other.
                        _ => {
                            let m = self.read_value128(io, src, 8, mem)? & lowmask;
                            if high {
                                self.xmm[id] = (self.xmm[id] & lowmask) | (m << 64);
                            } else {
                                self.xmm[id] = (self.xmm[id] & (lowmask << 64)) | m;
                            }
                        }
                    }
                }
            }
            Op::FBin(kind, double, packed) => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let read_bytes = if packed {
                    16
                } else if double {
                    8
                } else {
                    4
                };
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, src, read_bytes, mem)?;
                let (res, exc) = fbin_lanes(a, b, kind, double, packed, self.mxcsr);
                self.xmm[dst_id] = res;
                self.mxcsr |= exc;
            }
            Op::Cmpxchg16b { wide } => {
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
                let dst = insn.dst.as_ref().unwrap();
                let addr = self.effective_address(dst);
                if wide {
                    let lo = self.mem_read(mem, io, addr, 8, Access::Read)?;
                    let hi = self.mem_read(mem, io, addr.wrapping_add(8), 8, Access::Read)?;
                    if lo == self.regs.gpr[RAX] && hi == self.regs.gpr[RDX] {
                        self.mem_write(mem, io, addr, 8, self.regs.gpr[RBX])?;
                        self.mem_write(mem, io, addr.wrapping_add(8), 8, self.regs.gpr[RCX])?;
                        flags::set_flag(&mut self.regs.rflags, ZF, true);
                    } else {
                        self.regs.gpr[RAX] = lo;
                        self.regs.gpr[RDX] = hi;
                        flags::set_flag(&mut self.regs.rflags, ZF, false);
                    }
                } else {
                    let val = self.mem_read(mem, io, addr, 8, Access::Read)?;
                    let acc = (self.regs.gpr[RAX] & 0xffff_ffff)
                        | ((self.regs.gpr[RDX] & 0xffff_ffff) << 32);
                    if val == acc {
                        let store = (self.regs.gpr[RBX] & 0xffff_ffff)
                            | ((self.regs.gpr[RCX] & 0xffff_ffff) << 32);
                        self.mem_write(mem, io, addr, 8, store)?;
                        flags::set_flag(&mut self.regs.rflags, ZF, true);
                    } else {
                        self.regs.write(RAX, Size::Dword, false, val & 0xffff_ffff);
                        self.regs.write(RDX, Size::Dword, false, val >> 32);
                        flags::set_flag(&mut self.regs.rflags, ZF, false);
                    }
                }
                if self.tso_max != 0 {
                    self.sb_drain_all(mem);
                }
            }
            Op::Rdrand { seed } => {
                // Deterministic "hardware" RNG: a seed-derived value (this is a DST
                // machine — reproducible, not host-random). Always reports success
                // (CF=1); OF/SF/ZF/AF/PF cleared per SDM.
                let _ = seed; // RDSEED uses the same deterministic source as RDRAND
                let value = self.rdrand_rng.next_u64();
                self.write(io, insn.dst.as_ref().unwrap(), value, mem)?;
                flags::set_flag(&mut self.regs.rflags, CF, true);
                for b in [OF, SF, ZF, AF, PF] {
                    flags::set_flag(&mut self.regs.rflags, b, false);
                }
            }
            Op::Pcmpstr { explicit, mask_out } => {
                let op1 = self.read_value128(io, insn.dst.as_ref().unwrap(), 16, mem)?;
                let op2 = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let word = imm & 1 != 0;
                let bits: u32 = if word { 16 } else { 8 };
                let num_elem = 128 / bits;
                let mask: u128 = (1u128 << bits) - 1;
                let first_null = |v: u128| -> u32 {
                    (0..num_elem)
                        .find(|&i| (v >> (i * bits)) & mask == 0)
                        .unwrap_or(num_elem)
                };
                let (len1, len2) = if explicit {
                    let clamp = |v: u64| ((v as i64).unsigned_abs() as u32).min(num_elem);
                    (clamp(self.regs.gpr[RAX]), clamp(self.regs.gpr[RDX]))
                } else {
                    (first_null(op1), first_null(op2))
                };
                let intres2 = pcmpstr_intres2(op1, op2, imm, len1, len2, num_elem);

                if mask_out {
                    self.xmm[0] = if imm & 0x40 != 0 {
                        // expanded element mask
                        let mut out = 0u128;
                        for i in 0..num_elem {
                            if intres2 & (1 << i) != 0 {
                                out |= mask << (i * bits);
                            }
                        }
                        out
                    } else {
                        intres2 as u128 // bit mask, zero-extended
                    };
                } else {
                    let idx = if intres2 == 0 {
                        num_elem
                    } else if imm & 0x40 != 0 {
                        31 - intres2.leading_zeros() // most significant
                    } else {
                        intres2.trailing_zeros() // least significant
                    };
                    self.regs.write(RCX, Size::Dword, false, idx as u64);
                }

                flags::set_flag(&mut self.regs.rflags, CF, intres2 != 0);
                flags::set_flag(&mut self.regs.rflags, ZF, len2 < num_elem);
                flags::set_flag(&mut self.regs.rflags, SF, len1 < num_elem);
                flags::set_flag(&mut self.regs.rflags, OF, intres2 & 1 != 0);
                flags::set_flag(&mut self.regs.rflags, AF, false);
                flags::set_flag(&mut self.regs.rflags, PF, false);
            }
            Op::Pinsr { elem } => {
                let dst_id = xmm_dst(insn)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let lanes = (16 / elem) as u32;
                let lane = (imm as u32) & (lanes - 1);
                let bits = elem as u32 * 8;
                let mask: u128 = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
                let v = (self.read(io, insn.src.as_ref().unwrap(), mem)? as u128) & mask;
                self.xmm[dst_id] =
                    (self.xmm[dst_id] & !(mask << (lane * bits))) | (v << (lane * bits));
            }
            Op::Pextr { elem } => {
                let dst = insn.dst.as_ref().unwrap();
                let xmm_val = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let lanes = (16 / elem) as u32;
                let lane = (imm as u32) & (lanes - 1);
                let bits = elem as u32 * 8;
                let mask: u128 = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
                let v = (xmm_val >> (lane * bits)) & mask;
                self.write(io, dst, v as u64, mem)?;
            }
            Op::Crc32 { bytes } => {
                let dst = insn.dst.as_ref().unwrap();
                let src = insn.src.as_ref().unwrap();
                let mut crc = self.read(io, dst, mem)? as u32;
                let data = self.read(io, src, mem)?;
                for i in 0..bytes {
                    crc ^= ((data >> (i * 8)) & 0xff) as u32;
                    for _ in 0..8 {
                        crc = if crc & 1 != 0 {
                            (crc >> 1) ^ 0x82F6_3B78 // reversed CRC-32C polynomial
                        } else {
                            crc >> 1
                        };
                    }
                }
                // The 32-bit CRC zero-extends into the destination (r32 or r64).
                self.write(io, dst, crc as u64, mem)?;
            }
            Op::Blend { elem } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let bits = elem as u32 * 8;
                let mask: u128 = (1u128 << bits) - 1;
                let mut out = 0u128;
                for i in 0..(128 / bits) {
                    let v = if (imm >> i) & 1 != 0 { b } else { a };
                    out |= ((v >> (i * bits)) & mask) << (i * bits);
                }
                self.xmm[dst_id] = out;
            }
            Op::Blendv { elem } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let m = self.xmm[0]; // implicit XMM0 mask
                let bits = elem as u32 * 8;
                let mask: u128 = (1u128 << bits) - 1;
                let sign = 1u128 << (bits - 1);
                let mut out = 0u128;
                for i in 0..(128 / bits) {
                    let pick_src = (m >> (i * bits)) & sign != 0;
                    let v = if pick_src { b } else { a };
                    out |= ((v >> (i * bits)) & mask) << (i * bits);
                }
                self.xmm[dst_id] = out;
            }
            Op::Insertps => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let src_sel = (imm >> 6) & 3;
                let dst_sel = (imm >> 4) & 3;
                let zmask = imm & 0xf;
                let temp = (b >> (src_sel as u32 * 32)) as u32 as u128;
                let mut out = (a & !(0xffff_ffffu128 << (dst_sel as u32 * 32)))
                    | (temp << (dst_sel as u32 * 32));
                for i in 0..4 {
                    if zmask & (1 << i) != 0 {
                        out &= !(0xffff_ffffu128 << (i * 32));
                    }
                }
                self.xmm[dst_id] = out;
            }
            Op::Phminposuw => {
                let dst_id = xmm_dst(insn)?;
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mut min = u16::MAX;
                let mut idx = 0u128;
                for i in 0..8 {
                    let w = (b >> (i * 16)) as u16;
                    if w < min {
                        min = w;
                        idx = i as u128;
                    }
                }
                self.xmm[dst_id] = (min as u128) | (idx << 16);
            }
            Op::Dpp { double } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                self.xmm[dst_id] = if double {
                    let mut sum = 0.0f64;
                    for i in 0..2 {
                        if imm & (0x10 << i) != 0 {
                            sum += f64::from_bits((a >> (i * 64)) as u64)
                                * f64::from_bits((b >> (i * 64)) as u64);
                        }
                    }
                    let s = sum.to_bits() as u128;
                    let mut out = 0u128;
                    for i in 0..2 {
                        if imm & (1 << i) != 0 {
                            out |= s << (i * 64);
                        }
                    }
                    out
                } else {
                    let mut sum = 0.0f32;
                    for i in 0..4 {
                        if imm & (0x10 << i) != 0 {
                            sum += f32::from_bits((a >> (i * 32)) as u32)
                                * f32::from_bits((b >> (i * 32)) as u32);
                        }
                    }
                    let s = sum.to_bits() as u128;
                    let mut out = 0u128;
                    for i in 0..4 {
                        if imm & (1 << i) != 0 {
                            out |= s << (i * 32);
                        }
                    }
                    out
                };
            }
            Op::Mpsadbw => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let src_off = (imm & 3) as u32 * 4;
                let dst_off = ((imm >> 2) & 1) as u32 * 4;
                let byte = |v: u128, n: u32| (v >> (n * 8)) as u8 as i32;
                let mut out = 0u128;
                for i in 0..8u32 {
                    let mut sad = 0i32;
                    for k in 0..4u32 {
                        sad += (byte(a, dst_off + i + k) - byte(b, src_off + k)).abs();
                    }
                    out |= (sad as u16 as u128) << (i * 16);
                }
                self.xmm[dst_id] = out;
            }
            Op::Phadd { elem, sub, sat } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let bits = elem as u32 * 8;
                let per = 128 / bits; // lanes per register
                // Horizontal pairs: dst's pairs fill the low half, src's the high.
                let horiz = |v: u128, out: &mut u128, base: u32| {
                    for j in 0..(per / 2) {
                        let lo = (v >> (2 * j * bits)) as i64;
                        let hi = (v >> ((2 * j + 1) * bits)) as i64;
                        let (x, y) = if elem == 2 {
                            (lo as i16 as i64, hi as i16 as i64)
                        } else {
                            (lo as i32 as i64, hi as i32 as i64)
                        };
                        let r = if sub { x - y } else { x + y };
                        let lane = if sat {
                            (r.clamp(-32768, 32767) as i16 as u16) as u128
                        } else if elem == 2 {
                            (r as u16) as u128
                        } else {
                            (r as u32) as u128
                        };
                        *out |= lane << ((base + j) * bits);
                    }
                };
                let mut out = 0u128;
                horiz(a, &mut out, 0);
                horiz(b, &mut out, per / 2);
                self.xmm[dst_id] = out;
            }
            Op::Psign { elem } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let bits = elem as u32 * 8;
                let mask: u128 = (1u128 << bits) - 1;
                let sign = 1u128 << (bits - 1);
                let mut out = 0u128;
                for i in 0..(128 / bits) {
                    let av = (a >> (i * bits)) & mask;
                    let bv = (b >> (i * bits)) & mask;
                    let lane = if bv == 0 {
                        0
                    } else if bv & sign != 0 {
                        (!av).wrapping_add(1) & mask // negate dst lane
                    } else {
                        av
                    };
                    out |= lane << (i * bits);
                }
                self.xmm[dst_id] = out;
            }
            Op::Pmaddubsw => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id]; // unsigned bytes
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?; // signed bytes
                let mut out = 0u128;
                for j in 0..8u32 {
                    let u0 = (a >> (2 * j * 8)) as u8 as i32;
                    let s0 = (b >> (2 * j * 8)) as u8 as i8 as i32;
                    let u1 = (a >> ((2 * j + 1) * 8)) as u8 as i32;
                    let s1 = (b >> ((2 * j + 1) * 8)) as u8 as i8 as i32;
                    let r = (u0 * s0 + u1 * s1).clamp(-32768, 32767);
                    out |= ((r as i16 as u16) as u128) << (j * 16);
                }
                self.xmm[dst_id] = out;
            }
            Op::Pmulhrsw => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mut out = 0u128;
                for i in 0..8u32 {
                    let x = (a >> (i * 16)) as u16 as i16 as i32;
                    let y = (b >> (i * 16)) as u16 as i16 as i32;
                    let r = (((x * y) >> 14) + 1) >> 1;
                    out |= ((r as i16 as u16) as u128) << (i * 16);
                }
                self.xmm[dst_id] = out;
            }
            Op::Pmovx { src_elem, dst_elem, sign } => {
                let dst_id = xmm_dst(insn)?;
                // src elements come from a register's low lanes or a memory operand
                // of (16/dst_elem * src_elem) bytes; reading 16 covers the reg case.
                let src = insn.src.as_ref().unwrap();
                let nbytes = (16 / dst_elem) * src_elem;
                let b = self.read_value128(io, src, nbytes, mem)?;
                let lanes = 16 / dst_elem;
                let src_bits = src_elem as u32 * 8;
                let src_mask = if src_bits == 128 { u128::MAX } else { (1u128 << src_bits) - 1 };
                let dst_bits = dst_elem as u32 * 8;
                let dst_mask = if dst_bits == 128 { u128::MAX } else { (1u128 << dst_bits) - 1 };
                let mut out = 0u128;
                for i in 0..lanes as u32 {
                    let s = (b >> (i * src_bits)) & src_mask;
                    let ext = if sign && (s >> (src_bits - 1)) & 1 != 0 {
                        // sign-extend: fill the upper bits up to dst width.
                        (s | !src_mask) & dst_mask
                    } else {
                        s & dst_mask
                    };
                    out |= ext << (i * dst_bits);
                }
                self.xmm[dst_id] = out;
            }
            Op::Ptest => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let zf = (a & b) == 0;
                let cf = (b & !a) == 0;
                flags::set_flag(&mut self.regs.rflags, ZF, zf);
                flags::set_flag(&mut self.regs.rflags, CF, cf);
                // PTEST clears OF/AF/PF/SF.
                for f in [OF, AF, PF, SF] {
                    flags::set_flag(&mut self.regs.rflags, f, false);
                }
            }
            Op::VPtest { l256 } => {
                // Full-width test: the upper 128 bits matter when l256 (Go's bytealg
                // memequal/IndexByte compare whole ymm registers).
                let dst_id = xmm_dst(insn)?;
                let a_lo = self.xmm[dst_id];
                let a_hi = if l256 { self.ymm_hi[dst_id] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let zf = (a_lo & b_lo) == 0 && (a_hi & b_hi) == 0;
                let cf = (b_lo & !a_lo) == 0 && (b_hi & !a_hi) == 0;
                flags::set_flag(&mut self.regs.rflags, ZF, zf);
                flags::set_flag(&mut self.regs.rflags, CF, cf);
                for f in [OF, AF, PF, SF] {
                    flags::set_flag(&mut self.regs.rflags, f, false);
                }
            }
            Op::Pabs { elem } => {
                let dst_id = xmm_dst(insn)?;
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let bits = elem as u32 * 8;
                let mask: u128 = (1u128 << bits) - 1;
                let sign = 1u128 << (bits - 1);
                let mut out = 0u128;
                for i in 0..(128 / bits) {
                    let v = (b >> (i * bits)) & mask;
                    // two's-complement abs within the lane width.
                    let abs = if v & sign != 0 { (!v).wrapping_add(1) & mask } else { v };
                    out |= abs << (i * bits);
                }
                self.xmm[dst_id] = out;
            }
            Op::AddSub { double } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = if double {
                    // low lane: a-b ; high lane: a+b.
                    let lo = fp_result_f64(
                        f64::from_bits(a as u64) - f64::from_bits(b as u64),
                        f64::from_bits(a as u64),
                        f64::from_bits(b as u64),
                    ) as u128;
                    let hi = fp_result_f64(
                        f64::from_bits((a >> 64) as u64) + f64::from_bits((b >> 64) as u64),
                        f64::from_bits((a >> 64) as u64),
                        f64::from_bits((b >> 64) as u64),
                    ) as u128;
                    lo | (hi << 64)
                } else {
                    let mut out = 0u128;
                    for i in 0..4 {
                        let x = f32::from_bits((a >> (i * 32)) as u32);
                        let y = f32::from_bits((b >> (i * 32)) as u32);
                        let r = if i % 2 == 0 { x - y } else { x + y };
                        out |= (fp_result_f32(r, x, y) as u128) << (i * 32);
                    }
                    out
                };
            }
            Op::Hadd { double, sub } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = if double {
                    let pair64 = |v: u128| {
                        let x = f64::from_bits(v as u64);
                        let y = f64::from_bits((v >> 64) as u64);
                        let r = if sub { x - y } else { x + y };
                        fp_result_f64(r, x, y) as u128
                    };
                    pair64(a) | (pair64(b) << 64)
                } else {
                    let lane = |v: u128, p: u32| f32::from_bits((v >> (p * 32)) as u32);
                    let pair = |x: f32, y: f32| {
                        let r = if sub { x - y } else { x + y };
                        fp_result_f32(r, x, y) as u128
                    };
                    pair(lane(a, 0), lane(a, 1))
                        | (pair(lane(a, 2), lane(a, 3)) << 32)
                        | (pair(lane(b, 0), lane(b, 1)) << 64)
                        | (pair(lane(b, 2), lane(b, 3)) << 96)
                };
            }
            Op::Movddup => {
                let dst_id = xmm_dst(insn)?;
                let lo = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)? as u64 as u128;
                self.xmm[dst_id] = lo | (lo << 64);
            }
            Op::Movsdup { high } => {
                let dst_id = xmm_dst(insn)?;
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                // MOVSLDUP: lanes 0,0,2,2 ; MOVSHDUP: lanes 1,1,3,3.
                let base = if high { 1 } else { 0 };
                let dw = |i: u32| (b >> (i * 32)) as u32 as u128;
                self.xmm[dst_id] =
                    dw(base) | (dw(base) << 32) | (dw(base + 2) << 64) | (dw(base + 2) << 96);
            }
            Op::Pshufb => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id]; // bytes to shuffle
                let ctrl = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mut out = 0u128;
                for i in 0..16 {
                    let c = (ctrl >> (i * 8)) as u8;
                    if c & 0x80 == 0 {
                        let byte = (a >> ((c & 0x0f) as u32 * 8)) as u8;
                        out |= (byte as u128) << (i * 8);
                    }
                }
                self.xmm[dst_id] = out;
            }
            Op::Pmulld => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mut out = 0u128;
                for i in 0..4 {
                    let r = (a >> (i * 32)) as u32;
                    let s = (b >> (i * 32)) as u32;
                    out |= (r.wrapping_mul(s) as u128) << (i * 32);
                }
                self.xmm[dst_id] = out;
            }
            Op::Palignr => {
                let dst_id = xmm_dst(insn)?;
                let dst = self.xmm[dst_id];
                let src = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let n = (insn.rel.unwrap_or(0) as u8) as u32;
                // Concat dst(high):src(low), shift right by n bytes, keep low 128.
                self.xmm[dst_id] = if n >= 32 {
                    0
                } else if n >= 16 {
                    dst >> ((n - 16) * 8)
                } else if n == 0 {
                    src
                } else {
                    let sh = n * 8;
                    (src >> sh) | (dst << (128 - sh))
                };
            }
            Op::Round { double, scalar } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mode = (insn.rel.unwrap_or(0) as u8) & 0x7; // bit2 = use MXCSR (→ nearest)
                let r64 = |x: f64| round_mode_f64(x, mode).to_bits();
                let r32 = |x: f32| round_mode_f64(x as f64, mode) as f32;
                self.xmm[dst_id] = if double {
                    let r0 = r64(f64::from_bits(b as u64)) as u128;
                    if scalar {
                        (a & (u128::MAX << 64)) | r0
                    } else {
                        (r64(f64::from_bits((b >> 64) as u64)) as u128) << 64 | r0
                    }
                } else if scalar {
                    (a & !0xffff_ffff_u128) | r32(f32::from_bits(b as u32)).to_bits() as u128
                } else {
                    let mut out = 0u128;
                    for i in 0..4 {
                        out |= (r32(f32::from_bits((b >> (i * 32)) as u32)).to_bits() as u128)
                            << (i * 32);
                    }
                    out
                };
            }
            Op::PMul(kind) => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = match kind {
                    PMulKind::Uludq => {
                        let r0 = (a as u32 as u64).wrapping_mul(b as u32 as u64);
                        let r1 = ((a >> 64) as u32 as u64).wrapping_mul((b >> 64) as u32 as u64);
                        (r0 as u128) | ((r1 as u128) << 64)
                    }
                    PMulKind::Maddwd => {
                        let mut out = 0u128;
                        for j in 0..4u32 {
                            let w = |v: u128, k: u32| ((v >> (k * 16)) as u16) as i16 as i32;
                            let s = w(a, 2 * j)
                                .wrapping_mul(w(b, 2 * j))
                                .wrapping_add(w(a, 2 * j + 1).wrapping_mul(w(b, 2 * j + 1)));
                            out |= ((s as u32) as u128) << (j * 32);
                        }
                        out
                    }
                    _ => {
                        let mut out = 0u128;
                        for i in 0..8u32 {
                            let aw = (a >> (i * 16)) as u16;
                            let bw = (b >> (i * 16)) as u16;
                            let r: u16 = match kind {
                                PMulKind::Lw => aw.wrapping_mul(bw),
                                PMulKind::Hw => {
                                    (((aw as i16 as i32) * (bw as i16 as i32)) >> 16) as u16
                                }
                                PMulKind::Huw => (((aw as u32) * (bw as u32)) >> 16) as u16,
                                _ => 0,
                            };
                            out |= (r as u128) << (i * 16);
                        }
                        out
                    }
                };
            }
            Op::CvtF2I { double_src, trunc } => {
                let dst_id = xmm_dst(insn)?;
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = if double_src {
                    // 2 f64 → 2 i32 in the low 64 bits; upper 64 zeroed.
                    let d0 = f64_to_i32(f64::from_bits(b as u64), trunc) as u32 as u128;
                    let d1 = f64_to_i32(f64::from_bits((b >> 64) as u64), trunc) as u32 as u128;
                    d0 | (d1 << 32)
                } else {
                    // 4 f32 → 4 i32.
                    let mut out = 0u128;
                    for i in 0..4 {
                        let v = f64_to_i32(f32::from_bits((b >> (i * 32)) as u32) as f64, trunc);
                        out |= (v as u32 as u128) << (i * 32);
                    }
                    out
                };
            }
            Op::CvtI2F { double_dst } => {
                let dst_id = xmm_dst(insn)?;
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = if double_dst {
                    // 2 i32 (low 64) → 2 f64.
                    let f0 = ((b as u32 as i32) as f64).to_bits() as u128;
                    let f1 = (((b >> 32) as u32 as i32) as f64).to_bits() as u128;
                    f0 | (f1 << 64)
                } else {
                    // 4 i32 → 4 f32.
                    let mut out = 0u128;
                    for i in 0..4 {
                        let f = ((b >> (i * 32)) as u32 as i32) as f32;
                        out |= (f.to_bits() as u128) << (i * 32);
                    }
                    out
                };
            }
            Op::CvtF2F { to_double, scalar } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = if to_double {
                    // single → double. Scalar: low lane, preserve dst high 64.
                    let d0 = (f32::from_bits(b as u32) as f64).to_bits() as u128;
                    if scalar {
                        (a & (u128::MAX << 64)) | d0
                    } else {
                        let d1 = (f32::from_bits((b >> 32) as u32) as f64).to_bits() as u128;
                        d0 | (d1 << 64)
                    }
                } else {
                    // double → single.
                    let s0 = (f64::from_bits(b as u64) as f32).to_bits() as u128;
                    if scalar {
                        (a & !0xffff_ffff_u128) | s0
                    } else {
                        let s1 = (f64::from_bits((b >> 64) as u64) as f32).to_bits() as u128;
                        s0 | (s1 << 32)
                    }
                };
            }
            Op::Pack { dword_src, signed } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mut out = 0u128;
                if dword_src {
                    // 4+4 dwords → 8 words (low from dst). PACKSSDW = signed sat;
                    // PACKUSDW (SSE4.1) = unsigned sat to [0, 65535].
                    let sat = |d: i32| -> u128 {
                        if signed {
                            (d.clamp(-32768, 32767) as i16 as u16) as u128
                        } else {
                            (d.clamp(0, 65535) as u16) as u128
                        }
                    };
                    for i in 0..4 {
                        out |= sat((a >> (i * 32)) as i32) << (i * 16);
                        out |= sat((b >> (i * 32)) as i32) << ((i + 4) * 16);
                    }
                } else {
                    // 8+8 words → 16 bytes, signed or unsigned saturation.
                    let sat = |w: i16| -> u128 {
                        if signed {
                            (w.clamp(-128, 127) as i8 as u8) as u128
                        } else {
                            (w.clamp(0, 255) as u8) as u128
                        }
                    };
                    for i in 0..8 {
                        out |= sat((a >> (i * 16)) as i16) << (i * 8);
                        out |= sat((b >> (i * 16)) as i16) << ((i + 8) * 8);
                    }
                }
                self.xmm[dst_id] = out;
            }
            Op::UnpckFp { high, double } => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, src, 16, mem)?;
                self.xmm[dst_id] = if double {
                    // UNPCKLPD: [dst.lo, src.lo]; UNPCKHPD: [dst.hi, src.hi].
                    let (da, sb) = if high {
                        ((a >> 64) as u64, (b >> 64) as u64)
                    } else {
                        (a as u64, b as u64)
                    };
                    (sb as u128) << 64 | da as u128
                } else {
                    // UNPCKLPS interleaves dwords 0,1; UNPCKHPS dwords 2,3.
                    let base = if high { 2 } else { 0 };
                    let dw = |v: u128, i: u32| (v >> (i * 32)) as u32 as u128;
                    let a0 = dw(a, base);
                    let b0 = dw(b, base);
                    let a1 = dw(a, base + 1);
                    let b1 = dw(b, base + 1);
                    a0 | (b0 << 32) | (a1 << 64) | (b1 << 96)
                };
            }
            Op::PLogic(kind) => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = match kind {
                    PLogicKind::Xor => a ^ b,
                    PLogicKind::Or => a | b,
                    PLogicKind::And => a & b,
                    PLogicKind::Andn => !a & b,
                };
            }
            Op::PackedCmpEq(elem) | Op::PackedCmpGt(elem) => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let gt = matches!(insn.op, Op::PackedCmpGt(_));
                self.xmm[dst_id] = packed_compare(a, b, elem, gt);
            }
            Op::PackedAdd(elem) | Op::PackedSub(elem) => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let sub = matches!(insn.op, Op::PackedSub(_));
                self.xmm[dst_id] = packed_arith(a, b, elem, sub);
            }
            Op::PackedMinMax { max, signed, elem } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = packed_minmax(a, b, elem, max, signed);
            }
            Op::PackedShift { kind, elem } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let src = insn.src.as_ref().unwrap();
                // Count: an immediate, or the low 64 bits of the source xmm/m128.
                let count = match src {
                    Operand::Imm { value, .. } => *value,
                    _ => self.read_value128(io, src, 8, mem)? as u64,
                };
                self.xmm[dst_id] = packed_shift(a, count, elem, kind);
            }
            Op::XmmByteShift { left } => {
                let dst_id = xmm_dst(insn)?;
                let bytes = match insn.src.as_ref().unwrap() {
                    Operand::Imm { value, .. } => *value,
                    _ => return Err(X86Error::InvalidEncoding("PS{R,L}LDQ needs imm8".into())),
                };
                self.xmm[dst_id] = if bytes >= 16 {
                    0
                } else if left {
                    self.xmm[dst_id] << (bytes * 8)
                } else {
                    self.xmm[dst_id] >> (bytes * 8)
                };
            }
            Op::Pshufd => {
                let dst_id = xmm_dst(insn)?;
                let src = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let order = insn.rel.unwrap() as u8;
                let mut r = 0_u128;
                for i in 0..4 {
                    let sel = (order >> (i * 2)) & 3;
                    let lane = (src >> (sel as u32 * 32)) & 0xffff_ffff;
                    r |= lane << (i * 32);
                }
                self.xmm[dst_id] = r;
            }
            Op::PshufW { high } => {
                // Legacy SSE2 PSHUFLW/PSHUFHW: write only xmm (preserve YMM upper).
                let dst_id = xmm_dst(insn)?;
                let src = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let imm = insn.rel.unwrap() as u8;
                self.xmm[dst_id] = pshuf_lanes(src, imm, Some(high));
            }
            Op::Punpck { high, elem } => {
                let dst_id = xmm_dst(insn)?;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                self.xmm[dst_id] = packed_unpack(a, b, elem, high);
            }
            Op::Pmovmskb => {
                let x = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mut mask = 0_u64;
                for i in 0..16 {
                    if (x >> (i * 8 + 7)) & 1 != 0 {
                        mask |= 1 << i;
                    }
                }
                self.write(io, insn.dst.as_ref().unwrap(), mask, mem)?;
            }
            Op::Cvtsi2f(double) => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let isize = operand_size(src);
                let ival = sext(self.read(io, src, mem)?, isize.bits());
                if double {
                    let f = ival as f64;
                    self.xmm[dst_id] =
                        (self.xmm[dst_id] & (u128::MAX << 64)) | (f.to_bits() as u128);
                } else {
                    let f = ival as f32;
                    self.xmm[dst_id] = (self.xmm[dst_id] & !0xffff_ffff_u128) | (f.to_bits() as u128);
                }
            }
            Op::VMovScalarMerge { bytes, vvvv } => {
                // dst[low bytes] = src[low bytes]; dst[upper] = vvvv[upper]; YMM zeroed.
                let dst_id = xmm_id(insn.dst.as_ref().unwrap())?;
                let src_id = xmm_id(insn.src.as_ref().unwrap())?;
                let low_mask = (1u128 << (bytes as u32 * 8)) - 1;
                let result = (self.xmm[vvvv as usize] & !low_mask) | (self.xmm[src_id] & low_mask);
                self.write_ymm_reg(dst_id, result, 0, false);
            }
            Op::VCvtsi2f { double, vvvv } => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let isize = operand_size(src);
                let ival = sext(self.read(io, src, mem)?, isize.bits());
                let src1 = self.xmm[vvvv as usize];
                let result = if double {
                    (src1 & (u128::MAX << 64)) | ((ival as f64).to_bits() as u128)
                } else {
                    (src1 & (u128::MAX << 32)) | ((ival as f32).to_bits() as u128)
                };
                self.write_ymm_reg(dst_id, result, 0, false); // VEX zeroes the YMM upper
            }
            Op::VPinsr { elem, vvvv } => {
                // dst = vvvv with element[imm] replaced by the integer src; YMM zeroed.
                let dst_id = xmm_dst(insn)?;
                let imm = insn.rel.unwrap_or(0) as u8;
                let lanes = (16 / elem) as u32;
                let lane = (imm as u32) & (lanes - 1);
                let bits = elem as u32 * 8;
                let mask: u128 = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
                let v = (self.read(io, insn.src.as_ref().unwrap(), mem)? as u128) & mask;
                let base = self.xmm[vvvv as usize];
                let result = (base & !(mask << (lane * bits))) | (v << (lane * bits));
                self.write_ymm_reg(dst_id, result, 0, false);
            }
            Op::VPmovx { src_elem, dst_elem, sign, l256 } => {
                let dst_id = xmm_dst(insn)?;
                let total_lanes = (if l256 { 32 } else { 16 } / dst_elem) as u32;
                let nbytes = total_lanes * src_elem as u32;
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), nbytes.min(16) as u8, mem)?;
                let src_bits = src_elem as u32 * 8;
                let src_mask = if src_bits == 128 { u128::MAX } else { (1u128 << src_bits) - 1 };
                let dst_bits = dst_elem as u32 * 8;
                let dst_mask = if dst_bits == 128 { u128::MAX } else { (1u128 << dst_bits) - 1 };
                let (mut lo, mut hi) = (0u128, 0u128);
                for i in 0..total_lanes {
                    let s = (b >> (i * src_bits)) & src_mask;
                    let ext = if sign && (s >> (src_bits - 1)) & 1 != 0 {
                        (s | !src_mask) & dst_mask
                    } else {
                        s & dst_mask
                    };
                    let bitpos = i * dst_bits;
                    if bitpos < 128 {
                        lo |= ext << bitpos;
                    } else {
                        hi |= ext << (bitpos - 128);
                    }
                }
                self.write_ymm_reg(dst_id, lo, hi, l256);
            }
            Op::VMovddup { l256 } => {
                let dst_id = xmm_dst(insn)?;
                let (src_lo, src_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let dup = |q: u128| (q & u64::MAX as u128) | ((q & u64::MAX as u128) << 64);
                self.write_ymm_reg(dst_id, dup(src_lo), if l256 { dup(src_hi) } else { 0 }, l256);
            }
            Op::VFSqrt { double, packed, vvvv, l256 } => {
                let dst_id = xmm_dst(insn)?;
                let mxcsr = self.mxcsr;
                if packed {
                    let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                    let (lo, elo) = sqrt_lanes128(b_lo, double, mxcsr);
                    let (hi, ehi) = if l256 { sqrt_lanes128(b_hi, double, mxcsr) } else { (0, 0) };
                    self.mxcsr |= elo | ehi;
                    self.write_ymm_reg(dst_id, lo, hi, l256);
                } else {
                    let bytes = if double { 8 } else { 4 };
                    let b = self.read_value128(io, insn.src.as_ref().unwrap(), bytes, mem)?;
                    let base = self.xmm[vvvv as usize];
                    let (result, exc) = if double {
                        let (r0, e) = sqrt_scalar_f64(b as u64, mxcsr);
                        ((base & (u128::MAX << 64)) | r0 as u128, e)
                    } else {
                        let (r0, e) = sqrt_scalar_f32(b as u32, mxcsr);
                        ((base & !0xffff_ffff_u128) | r0 as u128, e)
                    };
                    self.mxcsr |= exc;
                    self.write_ymm_reg(dst_id, result, 0, false);
                }
            }
            Op::VPermil { double, variable, vvvv, l256 } => {
                let dst_id = xmm_dst(insn)?;
                let (data_lo, data_hi, ctrl_lo, ctrl_hi);
                if variable {
                    // data = vvvv (src1), control = r/m (src2).
                    data_lo = self.xmm[vvvv as usize];
                    data_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                    let (clo, chi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                    ctrl_lo = clo;
                    ctrl_hi = chi;
                } else {
                    // data = r/m, control = imm8 (replicated to each lane below).
                    let (dlo, dhi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                    data_lo = dlo;
                    data_hi = dhi;
                    let imm = insn.rel.unwrap_or(0) as u128;
                    ctrl_lo = imm;
                    ctrl_hi = imm;
                }
                let out_lo = permil_lane(data_lo, double, variable, ctrl_lo, 0);
                let out_hi = if l256 { permil_lane(data_hi, double, variable, ctrl_hi, 1) } else { 0 };
                self.write_ymm_reg(dst_id, out_lo, out_hi, l256);
            }
            Op::Cvttf2si(double) => {
                let dst = insn.dst.as_ref().unwrap();
                let dsize = operand_size(dst);
                let f = if double {
                    f64::from_bits(self.read_value128(io, insn.src.as_ref().unwrap(), 8, mem)? as u64)
                } else {
                    f32::from_bits(self.read_value128(io, insn.src.as_ref().unwrap(), 4, mem)? as u32)
                        as f64
                };
                let r = f64_to_int(f, dsize.bits());
                self.write(io, dst, r, mem)?;
            }
            Op::Movmsk { double } => {
                let x = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let mask = if double {
                    (((x >> 63) & 1) | (((x >> 127) & 1) << 1)) as u64
                } else {
                    let mut m = 0_u64;
                    for i in 0..4 {
                        if (x >> (31 + i * 32)) & 1 != 0 {
                            m |= 1 << i;
                        }
                    }
                    m
                };
                self.write(io, insn.dst.as_ref().unwrap(), mask, mem)?;
            }
            Op::Shuffle { double } => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let imm = insn.rel.unwrap_or(0) as u8;
                let a = self.xmm[dst_id];
                let b = self.read_value128(io, src, 16, mem)?;
                self.xmm[dst_id] = if double {
                    // SHUFPD: low 64 from dst (bit0), high 64 from src (bit1).
                    let lo = if imm & 1 != 0 { (a >> 64) as u64 } else { a as u64 };
                    let hi = if imm & 2 != 0 { (b >> 64) as u64 } else { b as u64 };
                    ((hi as u128) << 64) | lo as u128
                } else {
                    // SHUFPS: dwords 0,1 selected from dst; dwords 2,3 from src.
                    let dword = |v: u128, sel: u8| (v >> ((sel & 3) as u32 * 32)) as u32;
                    let d0 = dword(a, imm) as u128;
                    let d1 = dword(a, imm >> 2) as u128;
                    let d2 = dword(b, imm >> 4) as u128;
                    let d3 = dword(b, imm >> 6) as u128;
                    d0 | (d1 << 32) | (d2 << 64) | (d3 << 96)
                };
            }
            Op::XmmCmp { double, scalar } => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let pred = (insn.rel.unwrap_or(0) as u8) & 0x7;
                let a = self.xmm[dst_id];
                let read_bytes = if scalar {
                    if double {
                        8
                    } else {
                        4
                    }
                } else {
                    16
                };
                let b = self.read_value128(io, src, read_bytes, mem)?;
                // Predicate truth table (SSE imm8 0..7). NaN comparisons are false in
                // Rust, which matches x86: the ordered predicates (EQ/LT/LE/ORD) are
                // false on NaN, the unordered ones (UNORD/NEQ/NLT/NLE) true.
                let test64 = |x: f64, y: f64| -> bool {
                    match pred {
                        0 => x == y,
                        1 => x < y,
                        2 => x <= y,
                        3 => x.is_nan() || y.is_nan(),
                        4 => !(x == y),
                        5 => !(x < y),
                        6 => !(x <= y),
                        _ => !(x.is_nan() || y.is_nan()),
                    }
                };
                let test32 = |x: f32, y: f32| -> bool {
                    match pred {
                        0 => x == y,
                        1 => x < y,
                        2 => x <= y,
                        3 => x.is_nan() || y.is_nan(),
                        4 => !(x == y),
                        5 => !(x < y),
                        6 => !(x <= y),
                        _ => !(x.is_nan() || y.is_nan()),
                    }
                };
                let m64 = |t: bool| if t { u64::MAX as u128 } else { 0 };
                let m32 = |t: bool| if t { u32::MAX as u128 } else { 0 };
                let result = if double {
                    let r0 = m64(test64(f64::from_bits(a as u64), f64::from_bits(b as u64)));
                    if scalar {
                        (a & (u128::MAX << 64)) | r0
                    } else {
                        let r1 = m64(test64(
                            f64::from_bits((a >> 64) as u64),
                            f64::from_bits((b >> 64) as u64),
                        ));
                        (r1 << 64) | r0
                    }
                } else if scalar {
                    let r0 = m32(test32(f32::from_bits(a as u32), f32::from_bits(b as u32)));
                    (a & !0xffff_ffff_u128) | r0
                } else {
                    let mut out = 0u128;
                    for i in 0..4 {
                        let ai = f32::from_bits((a >> (i * 32)) as u32);
                        let bi = f32::from_bits((b >> (i * 32)) as u32);
                        out |= m32(test32(ai, bi)) << (i * 32);
                    }
                    out
                };
                self.xmm[dst_id] = result;
            }
            Op::Ucomis(double) => {
                let a_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let (a, b) = if double {
                    (
                        f64::from_bits(self.xmm[a_id] as u64),
                        f64::from_bits(self.read_value128(io, src, 8, mem)? as u64),
                    )
                } else {
                    (
                        f32::from_bits(self.xmm[a_id] as u32) as f64,
                        f32::from_bits(self.read_value128(io, src, 4, mem)? as u32) as f64,
                    )
                };
                let (zf, pf, cf) = if a.is_nan() || b.is_nan() {
                    (true, true, true)
                } else if a < b {
                    (false, false, true)
                } else if a > b {
                    (false, false, false)
                } else {
                    (true, false, false)
                };
                let rf = &mut self.regs.rflags;
                flags::set_flag(rf, ZF, zf);
                flags::set_flag(rf, PF, pf);
                flags::set_flag(rf, CF, cf);
                for bit in [OF, SF, AF] {
                    flags::set_flag(rf, bit, false);
                }
            }
            Op::FSqrt(double, packed) => {
                let dst_id = xmm_dst(insn)?;
                let src = insn.src.as_ref().unwrap();
                let read_bytes = if packed {
                    16
                } else if double {
                    8
                } else {
                    4
                };
                let b = self.read_value128(io, src, read_bytes, mem)?;
                let dst = self.xmm[dst_id];
                let mxcsr = self.mxcsr;
                let (result, exc) = if packed {
                    sqrt_lanes128(b, double, mxcsr) // both/all lanes
                } else if double {
                    let (bits, e) = sqrt_scalar_f64(b as u64, mxcsr);
                    ((dst & (u128::MAX << 64)) | bits as u128, e)
                } else {
                    let (bits, e) = sqrt_scalar_f32(b as u32, mxcsr);
                    ((dst & !0xffff_ffff_u128) | bits as u128, e)
                };
                self.mxcsr |= exc;
                self.xmm[dst_id] = result;
            }
            Op::Inc => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.read(io, dst, mem)?;
                let r = flags::inc(&mut self.regs.rflags, a, size);
                self.write(io, dst, r, mem)?;
            }
            Op::Dec => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.read(io, dst, mem)?;
                let r = flags::dec(&mut self.regs.rflags, a, size);
                self.write(io, dst, r, mem)?;
            }
            Op::Lea => {
                let dst = insn.dst.as_ref().unwrap();
                let addr = match insn.src.as_ref().unwrap() {
                    Operand::Mem { .. } => self.effective_address(insn.src.as_ref().unwrap()),
                    _ => return Err(X86Error::InvalidEncoding("LEA with non-memory source".into())),
                };
                self.write(io, dst, addr, mem)?;
            }
            Op::Push => {
                let dst = insn.dst.as_ref().unwrap();
                let value = self.read(io, dst, mem)?;
                // In 16/32-bit segments the pushed width follows the OPERAND size (e.g. a
                // `66`-prefixed `push eax` in a 16-bit segment moves 4 bytes) — the stack
                // pointer still wraps within the stack width. Long mode is unchanged: PUSH
                // there is 8 bytes (or 2 with 0x66), which is exactly the operand size.
                let w = if self.code_bits == 64 {
                    self.stack_op_bytes()
                } else {
                    operand_size(dst).bytes() as u64
                };
                self.push_sized(io, value, w, mem)?;
            }
            Op::Pop => {
                let dst = insn.dst.as_ref().unwrap();
                let w = if self.code_bits == 64 {
                    self.stack_op_bytes()
                } else {
                    operand_size(dst).bytes() as u64
                };
                let value = self.pop_sized(io, w, mem)?;
                self.write(io, dst, value, mem)?;
            }
            Op::Jmp => {
                self.regs.rip = self.regs.rip.wrapping_add(insn.rel.unwrap() as u64);
            }
            Op::JmpFar { selector, offset } => {
                // Reload CS through the descriptor cache and set IP absolutely. `load_cs` latches
                // the new base + code width from the *current* mode, so a `mov cr0` that set PE
                // just before this jump makes it land in the 32-bit target segment (real→protected),
                // and a trampoline's final ljmp after enabling PG+LME lands in 64-bit code.
                self.load_cs(selector, mem, io)?;
                self.regs.rip = offset;
            }
            Op::Jcc(cond) => {
                if condition(self.regs.rflags, cond) {
                    self.regs.rip = self.regs.rip.wrapping_add(insn.rel.unwrap() as u64);
                }
            }
            Op::Call => {
                let return_addr = self.regs.rip;
                self.push(io, return_addr, mem)?;
                self.regs.rip = self.regs.rip.wrapping_add(insn.rel.unwrap() as u64);
            }
            Op::Ret => {
                self.regs.rip = self.pop(io, mem)?;
                if let Some(n) = insn.rel {
                    self.regs.gpr[RSP] = self.regs.gpr[RSP].wrapping_add(n as u64);
                }
            }
            Op::Leave => {
                // RSP = RBP; then pop RBP from the new stack top.
                self.regs.gpr[RSP] = self.regs.gpr[RBP];
                self.regs.gpr[RBP] = self.pop(io, mem)?;
            }
            Op::Retf => {
                // LRETQ: pop RIP then CS (64-bit operands). A far return can change
                // privilege; restore CPL from the loaded CS RPL.
                self.regs.rip = self.pop(io, mem)?;
                let cs = self.pop(io, mem)?;
                self.cs = cs as u16;
                self.cpl = (cs & 3) as u8;
                if let Some(n) = insn.rel {
                    self.regs.gpr[RSP] = self.regs.gpr[RSP].wrapping_add(n as u64);
                }
            }
            Op::Movzx => {
                let value = self.read(io, insn.src.as_ref().unwrap(), mem)?;
                self.write(io, insn.dst.as_ref().unwrap(), value, mem)?;
            }
            Op::Movsx => {
                let src = insn.src.as_ref().unwrap();
                let raw = self.read(io, src, mem)?;
                let bits = operand_size(src).bits();
                let shift = 64 - bits;
                let value = (((raw << shift) as i64) >> shift) as u64;
                self.write(io, insn.dst.as_ref().unwrap(), value, mem)?;
            }
            Op::Shl | Op::Shr | Op::Sar | Op::Rol | Op::Ror | Op::Rcl | Op::Rcr => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.read(io, dst, mem)?;
                let count = self.read(io, insn.src.as_ref().unwrap(), mem)? & 0xff;
                let rflags = &mut self.regs.rflags;
                let r = match insn.op {
                    Op::Shl => flags::shl(rflags, a, count, size),
                    Op::Shr => flags::shr(rflags, a, count, size),
                    Op::Sar => flags::sar(rflags, a, count, size),
                    Op::Rol => flags::rol(rflags, a, count, size),
                    Op::Ror => flags::ror(rflags, a, count, size),
                    Op::Rcl => flags::rcl(rflags, a, count, size),
                    Op::Rcr => flags::rcr(rflags, a, count, size),
                    _ => unreachable!(),
                };
                self.write(io, dst, r, mem)?;
            }
            Op::Shld | Op::Shrd => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let d = self.read(io, dst, mem)?;
                let s = self.read(io, insn.src.as_ref().unwrap(), mem)?;
                let count = match insn.rel {
                    Some(n) => n as u64 & 0xff,
                    None => self.regs.read(RCX, Size::Byte, false),
                };
                let rflags = &mut self.regs.rflags;
                let r = if matches!(insn.op, Op::Shld) {
                    flags::shld(rflags, d, s, count, size)
                } else {
                    flags::shrd(rflags, d, s, count, size)
                };
                self.write(io, dst, r, mem)?;
            }
            Op::Bswap => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let v = self.read(io, dst, mem)?;
                let r = match size {
                    Size::Dword => (v as u32).swap_bytes() as u64,
                    Size::Qword => v.swap_bytes(),
                    _ => v,
                };
                self.write(io, dst, r, mem)?;
            }
            Op::Cdq => {
                let size = operand_size(insn.dst.as_ref().unwrap());
                let a = self.regs.read(RAX, size, false);
                let fill = if a & size.sign_bit() != 0 { size.mask() } else { 0 };
                self.regs.write(RDX, size, false, fill);
            }
            Op::Cwde => {
                let size = operand_size(insn.dst.as_ref().unwrap());
                let (src_size, bits) = match size {
                    Size::Word => (Size::Byte, 8),
                    Size::Dword => (Size::Word, 16),
                    _ => (Size::Dword, 32),
                };
                let v = self.regs.read(RAX, src_size, false);
                self.regs.write(RAX, size, false, sext(v, bits) as u64);
            }
            Op::Loop(kind) => {
                let take = if matches!(kind, LoopKind::Jrcxz) {
                    self.regs.gpr[RCX] == 0
                } else {
                    self.regs.gpr[RCX] = self.regs.gpr[RCX].wrapping_sub(1);
                    let rcx_nz = self.regs.gpr[RCX] != 0;
                    let zf = self.regs.rflags & ZF != 0;
                    match kind {
                        LoopKind::Loope => rcx_nz && zf,
                        LoopKind::Loopne => rcx_nz && !zf,
                        _ => rcx_nz,
                    }
                };
                if take {
                    self.regs.rip = self.regs.rip.wrapping_add(insn.rel.unwrap() as u64);
                }
            }
            Op::Not => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.read(io, dst, mem)?;
                self.write(io, dst, !a & size.mask(), mem)?;
            }
            Op::Neg => {
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let a = self.read(io, dst, mem)?;
                let r = flags::sub(&mut self.regs.rflags, 0, a, size);
                self.write(io, dst, r, mem)?;
            }
            Op::Mul => self.mul(io, insn.src.as_ref().unwrap(), mem)?,
            Op::Imul => self.imul(io, insn, mem)?,
            Op::Div => self.divide(io, insn.src.as_ref().unwrap(), mem, false)?,
            Op::Idiv => self.divide(io, insn.src.as_ref().unwrap(), mem, true)?,
            Op::Cli => self.regs.rflags &= !IF,
            Op::Sti => self.regs.rflags |= IF,
            Op::Clc => self.regs.rflags &= !CF,
            Op::Stc => self.regs.rflags |= CF,
            Op::Cmc => self.regs.rflags ^= CF,
            Op::Cld => self.regs.rflags &= !flags::DF,
            Op::Std => self.regs.rflags |= flags::DF,
            Op::Fninit => {
                self.fpu_cw = 0x037f;
                self.fpu_sw = 0;
                self.fpu_tw = 0xffff;
                self.fpu_top = 0;
                self.st = [F80::ZERO; 8];
            }
            Op::Fnclex => self.fpu_sw &= !0x80ff, // clear exception + busy bits
            Op::Fwait => {} // no unmasked pending FP exceptions in this model
            Op::Fldcw => {
                let cw = self.read(io, insn.src.as_ref().unwrap(), mem)? as u16;
                self.fpu_cw = cw;
            }
            Op::Fnstcw => {
                let cw = self.fpu_cw as u64;
                self.write(io, insn.dst.as_ref().unwrap(), cw, mem)?;
            }
            Op::FnstswMem => {
                let sw = self.fpu_sw as u64;
                self.write(io, insn.dst.as_ref().unwrap(), sw, mem)?;
            }
            Op::FnstswAx => {
                self.regs.write(RAX, Size::Word, false, self.fpu_sw as u64);
            }
            Op::FpuLoadMem { bytes, integer } => {
                let addr = self.effective_address(insn.src.as_ref().unwrap());
                let val = if integer {
                    let raw = self.mem_read(mem, io, addr, bytes as usize, Access::Read)?;
                    F80::from_i64(sext(raw, bytes as u32 * 8) as i64)
                } else {
                    match bytes {
                        4 => F80::from_f64(
                            f32::from_bits(self.mem_read(mem, io, addr, 4, Access::Read)? as u32) as f64,
                        ),
                        8 => F80::from_f64(f64::from_bits(self.mem_read(mem, io, addr, 8, Access::Read)?)),
                        _ => {
                            let lo = self.mem_read(mem, io, addr, 8, Access::Read)?;
                            let hi = self.mem_read(mem, io, addr.wrapping_add(8), 2, Access::Read)?;
                            F80::from_m80(lo, hi as u16) // exact 80-bit load — no f64 round-trip
                        }
                    }
                };
                self.fpu_push80(val);
            }
            Op::FpuStoreMem { bytes, integer, pop, trunc } => {
                let addr = self.effective_address(insn.dst.as_ref().unwrap());
                let v = self.fpu_st80(0);
                if integer {
                    let n = v.to_i64(self.fpu_rc(), trunc);
                    let stored = match bytes {
                        2 => if (i16::MIN as i64..=i16::MAX as i64).contains(&n) {
                            (n as u16) as u64
                        } else {
                            0x8000
                        },
                        4 => if (i32::MIN as i64..=i32::MAX as i64).contains(&n) {
                            (n as u32) as u64
                        } else {
                            0x8000_0000
                        },
                        _ => n as u64,
                    };
                    self.mem_write(mem, io, addr, bytes as usize, stored)?;
                } else {
                    match bytes {
                        4 => self.mem_write(mem, io, addr, 4, (v.to_f64() as f32).to_bits() as u64)?,
                        8 => self.mem_write(mem, io, addr, 8, v.to_f64().to_bits())?,
                        _ => {
                            let (m, se) = v.to_m80(); // exact 80-bit store
                            self.mem_write(mem, io, addr, 8, m)?;
                            self.mem_write(mem, io, addr.wrapping_add(8), 2, se as u64)?;
                        }
                    }
                }
                if pop {
                    self.fpu_pop80();
                }
            }
            Op::FpuLdSt { i } => {
                let v = self.fpu_st80(i); // FLD ST(i): exact 80-bit copy
                self.fpu_push80(v);
            }
            Op::FpuStSt { i, pop } => {
                let v = self.fpu_st80(0);
                self.fpu_set_st80(i, v);
                if pop {
                    self.fpu_pop80();
                }
            }
            Op::FpuArithMem { op, bytes, integer } => {
                let addr = self.effective_address(insn.src.as_ref().unwrap());
                let b = if integer {
                    F80::from_i64(
                        sext(self.mem_read(mem, io, addr, bytes as usize, Access::Read)?, bytes as u32 * 8)
                            as i64,
                    )
                } else if bytes == 4 {
                    F80::from_f64(f32::from_bits(self.mem_read(mem, io, addr, 4, Access::Read)? as u32) as f64)
                } else {
                    F80::from_f64(f64::from_bits(self.mem_read(mem, io, addr, 8, Access::Read)?))
                };
                let r = fpu_apply80(op, self.fpu_st80(0), b, self.fpu_pc(), self.fpu_rc());
                self.fpu_set_st80(0, r);
            }
            Op::FpuArithSt { op, i, to_st0, pop } => {
                let (pc, rc) = (self.fpu_pc(), self.fpu_rc());
                let st0 = self.fpu_st80(0);
                let sti = self.fpu_st80(i);
                if to_st0 {
                    self.fpu_set_st80(0, fpu_apply80(op, st0, sti, pc, rc));
                } else {
                    self.fpu_set_st80(i, fpu_apply80(op, sti, st0, pc, rc));
                }
                if pop {
                    self.fpu_pop80();
                }
            }
            Op::Fxch { i } => {
                let a = self.fpu_st80(0);
                let b = self.fpu_st80(i);
                self.fpu_set_st80(0, b);
                self.fpu_set_st80(i, a);
            }
            Op::FpuConst { which } => {
                // Canonical 80-bit x87 constant encodings (mantissa, biased-exp) — full extended
                // precision, not the f64-truncated values. Match real x87 at default RC.
                let (mant, se): (u64, u16) = match which {
                    0 => (0x8000_0000_0000_0000, 0x3fff), // 1.0
                    1 => (0xd49a_784b_cd1b_8afe, 0x4000), // log2(10)
                    2 => (0xb8aa_3b29_5c17_f0bc, 0x3fff), // log2(e)
                    3 => (0xc90f_daa2_2168_c235, 0x4000), // π
                    4 => (0x9a20_9a84_fbcf_f799, 0x3ffd), // log10(2)
                    5 => (0xb172_17f7_d1cf_79ac, 0x3ffe), // ln(2)
                    _ => (0x0000_0000_0000_0000, 0x0000), // 0.0
                };
                self.fpu_push80(F80::from_m80(mant, se));
            }
            Op::Fchs => {
                let v = self.fpu_st80(0).neg(); // exact sign flip
                self.fpu_set_st80(0, v);
            }
            Op::Fabs => {
                let mut v = self.fpu_st80(0); // exact
                v.sign = false;
                self.fpu_set_st80(0, v);
            }
            Op::Fsqrt => {
                let v = self.fpu_st80(0).sqrt(self.fpu_pc(), self.fpu_rc());
                self.fpu_set_st80(0, v);
            }
            Op::Frndint => {
                // FRNDINT rounds ST(0) to an integer per the FCW rounding-control field.
                let x = self.fpu_st(0);
                let v = match (self.fpu_cw >> 10) & 3 {
                    1 => x.floor(),      // toward −inf
                    2 => x.ceil(),       // toward +inf
                    3 => x.trunc(),      // toward zero
                    _ => round_ties_even_f64(x),
                };
                self.fpu_set_st(0, v);
            }
            Op::Fprem { ieee } => {
                // ST(0) = ST(0) − ST(1)·Q at full 80-bit. Exact + complete (C2=0) for an exponent
                // gap < 64; partial (C2=1) for ≥ 64. C0/C3/C1 = quotient bits 2/1/0.
                let (r, q, incomplete) = self.fpu_st80(0).fprem(self.fpu_st80(1), ieee);
                self.fpu_set_st80(0, r);
                self.fpu_sw &= !0x4700; // clear C3(14) C2(10) C1(9) C0(8)
                if q & 1 != 0 {
                    self.fpu_sw |= 1 << 9; // C1 = bit0
                }
                if q & 2 != 0 {
                    self.fpu_sw |= 1 << 14; // C3 = bit1
                }
                if q & 4 != 0 {
                    self.fpu_sw |= 1 << 8; // C0 = bit2
                }
                if incomplete {
                    self.fpu_sw |= 1 << 10; // C2 = partial remainder (gap ≥ 64)
                }
            }
            Op::Fscale => {
                // ST(0) · 2^trunc(ST(1)), exact in 80-bit (exponent shift).
                let v = self.fpu_st80(0).scale_by(self.fpu_st80(1), self.fpu_rc());
                self.fpu_set_st80(0, v);
            }
            Op::Ftrig { kind } => {
                match kind {
                    FpuTrig::Fsin => {
                        let v = soft_sincos(self.fpu_st(0)).0;
                        self.fpu_set_st(0, v);
                    }
                    FpuTrig::Fcos => {
                        let v = soft_sincos(self.fpu_st(0)).1;
                        self.fpu_set_st(0, v);
                    }
                    FpuTrig::Fsincos => {
                        let (s, c) = soft_sincos(self.fpu_st(0));
                        self.fpu_set_st(0, s);
                        self.fpu_push(c);
                    }
                    FpuTrig::Fptan => {
                        let (s, c) = soft_sincos(self.fpu_st(0));
                        self.fpu_set_st(0, s / c);
                        self.fpu_push(1.0);
                    }
                    FpuTrig::Fpatan => {
                        let v = soft_atan2(self.fpu_st(1), self.fpu_st(0));
                        self.fpu_set_st(1, v);
                        self.fpu_pop();
                    }
                    FpuTrig::F2xm1 => {
                        let v = soft_exp2m1(self.fpu_st(0));
                        self.fpu_set_st(0, v);
                    }
                    FpuTrig::Fyl2x => {
                        let v = self.fpu_st(1) * soft_log2(self.fpu_st(0));
                        self.fpu_set_st(1, v);
                        self.fpu_pop();
                    }
                    FpuTrig::Fyl2xp1 => {
                        let v = self.fpu_st(1) * soft_log2(self.fpu_st(0) + 1.0);
                        self.fpu_set_st(1, v);
                        self.fpu_pop();
                    }
                    FpuTrig::Fxtract => {
                        // ST(0) ← unbiased exponent; push the significand (m ∈ [1,2), signed) — exact.
                        let x = self.fpu_st80(0);
                        self.fpu_set_st80(0, F80::from_i64(x.exp as i64));
                        self.fpu_push80(F80 { exp: 0, ..x }); // significand ∈ [1,2)
                    }
                }
                self.fpu_sw &= !0x0400; // C2 = 0: reduction complete (argument in range)
            }
            Op::Fcmov { i, cond } => {
                if condition(self.regs.rflags, cond) {
                    let v = self.fpu_st80(i); // exact 80-bit conditional copy
                    self.fpu_set_st80(0, v);
                }
            }

            // ---- AVX (VEX) — iteration 1 ----
            Op::VZeroUpper { all } => {
                for i in 0..16 {
                    self.ymm_hi[i] = 0;
                    if all {
                        self.xmm[i] = 0;
                    }
                }
            }
            Op::VMov { l256 } => {
                let (lo, hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                match insn.dst.as_ref().unwrap() {
                    Operand::Xmm { id } => self.write_ymm_reg(*id, lo, hi, l256),
                    dst @ Operand::Mem { .. } => {
                        let addr = self.effective_address(dst);
                        self.write_u128_at(io, addr, lo, mem)?;
                        if l256 {
                            self.write_u128_at(io, addr.wrapping_add(16), hi, mem)?;
                        }
                    }
                    _ => return Err(X86Error::InvalidEncoding("vmov bad dst".into())),
                }
            }
            Op::VBitwise { kind, vvvv, l256 } => {
                let (a_lo, a_hi) = (
                    self.xmm[vvvv as usize],
                    if l256 { self.ymm_hi[vvvv as usize] } else { 0 },
                );
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let (lo, hi) = match kind {
                    VBitKind::And => (a_lo & b_lo, a_hi & b_hi),
                    VBitKind::AndN => (!a_lo & b_lo, !a_hi & b_hi),
                    VBitKind::Or => (a_lo | b_lo, a_hi | b_hi),
                    VBitKind::Xor => (a_lo ^ b_lo, a_hi ^ b_hi),
                };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VMovGpr { to_xmm, bytes } => {
                if to_xmm {
                    // GPR/mem → XMM low element; zero the whole YMM above it.
                    let val = self.read_value128(io, insn.src.as_ref().unwrap(), bytes, mem)?;
                    let id = xmm_id(insn.dst.as_ref().unwrap())?;
                    self.write_ymm_reg(id, val, 0, false);
                } else {
                    // XMM low element → GPR/mem.
                    let id = xmm_id(insn.src.as_ref().unwrap())?;
                    let mask = if bytes >= 16 { u128::MAX } else { (1u128 << (bytes as u32 * 8)) - 1 };
                    let val = self.xmm[id] & mask;
                    self.write_value128(io, insn.dst.as_ref().unwrap(), val, bytes, mem)?;
                }
            }
            Op::VFBin { kind, double, packed, vvvv, l256 } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 && packed { self.ymm_hi[vvvv as usize] } else { 0 };
                let src = insn.src.as_ref().unwrap();
                let (b_lo, b_hi) = if packed {
                    self.read_ymm(io, src, l256, mem)?
                } else {
                    // Scalar src2: read only the element from memory (correct fault width).
                    (self.read_value128(io, src, if double { 8 } else { 4 }, mem)?, 0)
                };
                let (lo, elo) = fbin_lanes(a_lo, b_lo, kind, double, packed, self.mxcsr);
                let (hi, ehi) = if packed && l256 {
                    fbin_lanes(a_hi, b_hi, kind, double, true, self.mxcsr)
                } else {
                    (0, 0)
                };
                self.mxcsr |= elo | ehi;
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                // VEX scalar zeroes YMM[255:128]; packed honors L.
                self.write_ymm_reg(id, lo, hi, l256 && packed);
            }
            Op::VPackedArith { elem, sub, vvvv, l256 } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = packed_arith(a_lo, b_lo, elem, sub);
                let hi = if l256 { packed_arith(a_hi, b_hi, elem, sub) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VPacked { kind, vvvv, l256 } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = vpacked_lane(kind, a_lo, b_lo);
                let hi = if l256 {
                    // Shifts take their count from src2's low 128 lane for both halves.
                    let b_for_hi = if matches!(kind, VPkind::Shift { .. }) { b_lo } else { b_hi };
                    vpacked_lane(kind, a_hi, b_for_hi)
                } else {
                    0
                };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VShufps { double, vvvv, l256, imm } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = shuffle_fp(a_lo, b_lo, double, imm);
                // VSHUFPS reuses the same imm8 per 128-bit lane; VSHUFPD has one
                // selector bit per double, so the high lane consumes imm[3:2].
                let hi_imm = if double { imm >> 2 } else { imm };
                let hi = if l256 { shuffle_fp(a_hi, b_hi, double, hi_imm) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VPshuf { high, l256, imm } => {
                let (s_lo, s_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = pshuf_lanes(s_lo, imm, high);
                let hi = if l256 { pshuf_lanes(s_hi, imm, high) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VCmpFp { double, scalar, vvvv, l256, imm } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 && !scalar { self.ymm_hi[vvvv as usize] } else { 0 };
                let src = insn.src.as_ref().unwrap();
                let (b_lo, b_hi) = if scalar {
                    (self.read_value128(io, src, if double { 8 } else { 4 }, mem)?, 0)
                } else {
                    self.read_ymm(io, src, l256, mem)?
                };
                let lo = cmp_fp(a_lo, b_lo, double, scalar, imm);
                let hi = if l256 && !scalar { cmp_fp(a_hi, b_hi, double, false, imm) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256 && !scalar);
            }
            Op::VBlend { elem, vvvv, l256, imm } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = blend_imm(a_lo, b_lo, elem, imm);
                // BLENDPS/PD: the imm spans all lanes, so the high 128 lane consumes
                // the next (16/elem) bits. PBLENDW (elem 2) repeats imm8 per 128 lane.
                let hi_imm = if elem == 2 { imm } else { imm >> (16 / elem as u32) };
                let hi = if l256 { blend_imm(a_hi, b_hi, elem, hi_imm) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VBlendv { elem, vvvv, l256, ctrl } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let c_lo = self.xmm[ctrl as usize];
                let c_hi = if l256 { self.ymm_hi[ctrl as usize] } else { 0 };
                let lo = blendv_lanes(a_lo, b_lo, c_lo, elem);
                let hi = if l256 { blendv_lanes(a_hi, b_hi, c_hi, elem) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VPalignr { vvvv, l256, imm } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                let (b_lo, b_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = palignr_lanes(a_lo, b_lo, imm);
                let hi = if l256 { palignr_lanes(a_hi, b_hi, imm) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VCvt { kind, vvvv, l256: true } => {
                // 256-bit forms: input/output width depends on the convert direction.
                let src = insn.src.as_ref().unwrap();
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                match kind {
                    // Widening (128-bit input → 256-bit output): low input lane → out
                    // low 128, high input lane → out high 128.
                    VCvtKind::F2F { to_double: true, scalar: false } => {
                        let b = self.read_value128(io, src, 16, mem)?; // 4 f32
                        self.write_ymm_reg(id, cvt2_f32_to_f64(b), cvt2_f32_to_f64(b >> 64), true);
                    }
                    VCvtKind::I2F { double_dst: true } => {
                        let b = self.read_value128(io, src, 16, mem)?; // 4 i32
                        self.write_ymm_reg(id, cvt2_i32_to_f64(b), cvt2_i32_to_f64(b >> 64), true);
                    }
                    // Narrowing (256-bit input → 128-bit output, upper zeroed).
                    VCvtKind::F2F { to_double: false, scalar: false } => {
                        let (blo, bhi) = self.read_ymm(io, src, true, mem)?; // 4 f64
                        let out = (cvt2_f64_to_f32(blo as u64, (blo >> 64) as u64) as u128)
                            | ((cvt2_f64_to_f32(bhi as u64, (bhi >> 64) as u64) as u128) << 64);
                        self.write_ymm_reg(id, out, 0, false);
                    }
                    VCvtKind::F2I { double_src: true, trunc } => {
                        let (blo, bhi) = self.read_ymm(io, src, true, mem)?; // 4 f64
                        let out = (cvt2_f64_to_i32(blo as u64, (blo >> 64) as u64, trunc) as u128)
                            | ((cvt2_f64_to_i32(bhi as u64, (bhi >> 64) as u64, trunc) as u128) << 64);
                        self.write_ymm_reg(id, out, 0, false);
                    }
                    // Lane-preserving (256 → 256), apply per 128-bit lane.
                    VCvtKind::I2F { double_dst: false } => {
                        let (blo, bhi) = self.read_ymm(io, src, true, mem)?;
                        self.write_ymm_reg(id, cvt4_i32_to_f32(blo), cvt4_i32_to_f32(bhi), true);
                    }
                    VCvtKind::F2I { double_src: false, trunc } => {
                        let (blo, bhi) = self.read_ymm(io, src, true, mem)?;
                        self.write_ymm_reg(id, cvt4_f32_to_i32(blo, trunc), cvt4_f32_to_i32(bhi, trunc), true);
                    }
                    // Scalar forms have no 256-bit encoding; treat as 128.
                    VCvtKind::F2F { scalar: true, .. } => {
                        let b = self.read_value128(io, src, 16, mem)?;
                        let a = self.xmm[vvvv as usize];
                        let lo = match kind {
                            VCvtKind::F2F { to_double: true, .. } => {
                                (a & (u128::MAX << 64)) | (f32::from_bits(b as u32) as f64).to_bits() as u128
                            }
                            _ => (a & !0xffff_ffff_u128) | (f64::from_bits(b as u64) as f32).to_bits() as u128,
                        };
                        self.write_ymm_reg(id, lo, 0, false);
                    }
                }
            }
            Op::VCvt { kind, vvvv, l256: false } => {
                let b = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let a = self.xmm[vvvv as usize]; // preserved upper bits for scalar F2F
                let lo = match kind {
                    VCvtKind::F2I { double_src, trunc } => {
                        if double_src {
                            let d0 = f64_to_i32(f64::from_bits(b as u64), trunc) as u32 as u128;
                            let d1 = f64_to_i32(f64::from_bits((b >> 64) as u64), trunc) as u32 as u128;
                            d0 | (d1 << 32)
                        } else {
                            let mut out = 0u128;
                            for i in 0..4 {
                                let v = f64_to_i32(f32::from_bits((b >> (i * 32)) as u32) as f64, trunc);
                                out |= (v as u32 as u128) << (i * 32);
                            }
                            out
                        }
                    }
                    VCvtKind::I2F { double_dst } => {
                        if double_dst {
                            let f0 = ((b as u32 as i32) as f64).to_bits() as u128;
                            let f1 = (((b >> 32) as u32 as i32) as f64).to_bits() as u128;
                            f0 | (f1 << 64)
                        } else {
                            let mut out = 0u128;
                            for i in 0..4 {
                                let f = ((b >> (i * 32)) as u32 as i32) as f32;
                                out |= (f.to_bits() as u128) << (i * 32);
                            }
                            out
                        }
                    }
                    VCvtKind::F2F { to_double, scalar } => {
                        if to_double {
                            let d0 = (f32::from_bits(b as u32) as f64).to_bits() as u128;
                            if scalar {
                                (a & (u128::MAX << 64)) | d0
                            } else {
                                let d1 = (f32::from_bits((b >> 32) as u32) as f64).to_bits() as u128;
                                d0 | (d1 << 64)
                            }
                        } else {
                            let s0 = (f64::from_bits(b as u64) as f32).to_bits() as u128;
                            if scalar {
                                (a & !0xffff_ffff_u128) | s0
                            } else {
                                let s1 = (f64::from_bits((b >> 64) as u64) as f32).to_bits() as u128;
                                s0 | (s1 << 32)
                            }
                        }
                    }
                };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, 0, false);
            }
            Op::VBroadcast { elem, l256 } => {
                let src = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let b = broadcast_elem(src, elem);
                // VBROADCASTF128/I128 (elem 16) always fills both 128-bit halves.
                let full = l256 || elem >= 16;
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, b, if full { b } else { 0 }, full);
            }
            Op::VInsert128 { vvvv, lane } => {
                let a_lo = self.xmm[vvvv as usize];
                let a_hi = self.ymm_hi[vvvv as usize];
                let s = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                let (lo, hi) = if lane == 0 { (s, a_hi) } else { (a_lo, s) };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, true);
            }
            Op::VExtract128 { lane } => {
                let sid = xmm_id(insn.src.as_ref().unwrap())?;
                let val = if lane == 0 { self.xmm[sid] } else { self.ymm_hi[sid] };
                match insn.dst.as_ref().unwrap() {
                    Operand::Xmm { id } => self.write_ymm_reg(*id, val, 0, false),
                    dst @ Operand::Mem { .. } => {
                        let addr = self.effective_address(dst);
                        self.write_u128_at(io, addr, val, mem)?;
                    }
                    _ => return Err(X86Error::InvalidEncoding("vextract bad dst".into())),
                }
            }
            Op::VPerm2 { vvvv, imm } => {
                let s1_lo = self.xmm[vvvv as usize];
                let s1_hi = self.ymm_hi[vvvv as usize];
                let (s2_lo, s2_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), true, mem)?;
                let sel = |bits: u8| -> u128 {
                    if bits & 0x8 != 0 {
                        0 // imm bit 3/7: force the lane to zero
                    } else {
                        match bits & 3 {
                            0 => s1_lo,
                            1 => s1_hi,
                            2 => s2_lo,
                            _ => s2_hi,
                        }
                    }
                };
                let lo = sel(imm & 0xf);
                let hi = sel((imm >> 4) & 0xf);
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, true);
            }
            Op::VPermImm { imm } => {
                let (s_lo, s_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), true, mem)?;
                let q = |i: usize| -> u128 {
                    let v = if i < 2 { s_lo } else { s_hi };
                    ((v >> ((i & 1) as u32 * 64)) as u64) as u128
                };
                let g = |sh: u32| q(((imm >> sh) & 3) as usize);
                let lo = g(0) | (g(2) << 64);
                let hi = g(4) | (g(6) << 64);
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, true);
            }
            Op::VPermVar { vvvv } => {
                let idx_lo = self.xmm[vvvv as usize];
                let idx_hi = self.ymm_hi[vvvv as usize];
                let (d_lo, d_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), true, mem)?;
                let dword = |lo: u128, hi: u128, i: usize| -> u128 {
                    let v = if i < 4 { lo } else { hi };
                    ((v >> ((i % 4) as u32 * 32)) as u32) as u128
                };
                let (mut out_lo, mut out_hi) = (0u128, 0u128);
                for i in 0..8 {
                    let idx = (dword(idx_lo, idx_hi, i) & 7) as usize;
                    let d = dword(d_lo, d_hi, idx);
                    if i < 4 {
                        out_lo |= d << (i as u32 * 32);
                    } else {
                        out_hi |= d << ((i - 4) as u32 * 32);
                    }
                }
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, out_lo, out_hi, true);
            }
            Op::VFma { kind, form, double, packed, vvvv, l256 } => {
                let dst_id = xmm_id(insn.dst.as_ref().unwrap())?;
                let d_lo = self.xmm[dst_id];
                let w_lo = self.xmm[vvvv as usize];
                let src = insn.src.as_ref().unwrap();
                let (lo, hi) = if packed {
                    let d_hi = if l256 { self.ymm_hi[dst_id] } else { 0 };
                    let w_hi = if l256 { self.ymm_hi[vvvv as usize] } else { 0 };
                    let (s_lo, s_hi) = self.read_ymm(io, src, l256, mem)?;
                    let (a0, b0, c0) = fma_operands(form, d_lo, w_lo, s_lo);
                    let lo = fma_lanes(a0, b0, c0, kind, double, true, d_lo);
                    let hi = if l256 {
                        let (a1, b1, c1) = fma_operands(form, d_hi, w_hi, s_hi);
                        fma_lanes(a1, b1, c1, kind, double, true, d_hi)
                    } else {
                        0
                    };
                    (lo, hi)
                } else {
                    // Scalar: read only the element; dst's upper bits are preserved.
                    let s = self.read_value128(io, src, if double { 8 } else { 4 }, mem)?;
                    let (a, b, c) = fma_operands(form, d_lo, w_lo, s);
                    (fma_lanes(a, b, c, kind, double, false, d_lo), 0)
                };
                self.write_ymm_reg(dst_id, lo, hi, l256 && packed);
            }
            Op::VCvtph2ps { l256 } => {
                let ph2ps4 = |halves: u128| -> u128 {
                    let mut o = 0u128;
                    for i in 0..4 {
                        o |= (f16_to_f32((halves >> (i * 16)) as u16) as u128) << (i * 32);
                    }
                    o
                };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                if l256 {
                    // 8 halves (128-bit src) → 8 singles (256-bit dst).
                    let s = self.read_value128(io, insn.src.as_ref().unwrap(), 16, mem)?;
                    self.write_ymm_reg(id, ph2ps4(s), ph2ps4(s >> 64), true);
                } else {
                    // 4 halves (low 64 of src) → 4 singles (128-bit dst).
                    let s = self.read_value128(io, insn.src.as_ref().unwrap(), 8, mem)?;
                    self.write_ymm_reg(id, ph2ps4(s), 0, false);
                }
            }
            Op::VCvtps2ph { l256, imm } => {
                // imm8[2]=1 selects MXCSR rounding (we model round-nearest).
                let mode = if imm & 4 != 0 { 0 } else { imm & 3 };
                let ps2ph4 = |s: u128| -> u64 {
                    let mut o = 0u64;
                    for i in 0..4 {
                        o |= (f32_to_f16((s >> (i * 32)) as u32, mode) as u64) << (i * 16);
                    }
                    o
                };
                let (s_lo, s_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let result = if l256 {
                    (ps2ph4(s_lo) as u128) | ((ps2ph4(s_hi) as u128) << 64)
                } else {
                    ps2ph4(s_lo) as u128
                };
                match insn.dst.as_ref().unwrap() {
                    Operand::Xmm { id } => self.write_ymm_reg(*id, result, 0, false),
                    dst @ Operand::Mem { .. } => {
                        let addr = self.effective_address(dst);
                        if l256 {
                            self.write_u128_at(io, addr, result, mem)?;
                        } else {
                            self.mem_write(mem, io, addr, 8, result as u64)?;
                        }
                    }
                    _ => return Err(X86Error::InvalidEncoding("vcvtps2ph bad dst".into())),
                }
            }
            Op::VMovmsk { double, l256 } => {
                let (lo, hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let mut mask = movmsk_of(lo, double);
                if l256 {
                    mask |= movmsk_of(hi, double) << (if double { 2 } else { 4 });
                }
                self.write(io, insn.dst.as_ref().unwrap(), mask, mem)?;
            }
            Op::VPmovmskb { l256 } => {
                let (lo, hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let bmask = |x: u128| -> u64 {
                    let mut m = 0u64;
                    for i in 0..16 {
                        m |= (((x >> (i * 8 + 7)) & 1) as u64) << i;
                    }
                    m
                };
                let mut mask = bmask(lo);
                if l256 {
                    mask |= bmask(hi) << 16;
                }
                self.write(io, insn.dst.as_ref().unwrap(), mask, mem)?;
            }
            Op::VPShiftImm { kind, elem, l256, imm } => {
                let (s_lo, s_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let lo = packed_shift(s_lo, imm as u64, elem, kind);
                let hi = if l256 { packed_shift(s_hi, imm as u64, elem, kind) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::VPByteShiftImm { left, l256, imm } => {
                let (s_lo, s_hi) = self.read_ymm(io, insn.src.as_ref().unwrap(), l256, mem)?;
                let sh = |x: u128| -> u128 {
                    if imm >= 16 { 0 } else if left { x << (imm as u32 * 8) } else { x >> (imm as u32 * 8) }
                };
                let lo = sh(s_lo);
                let hi = if l256 { sh(s_hi) } else { 0 };
                let id = xmm_id(insn.dst.as_ref().unwrap())?;
                self.write_ymm_reg(id, lo, hi, l256);
            }
            Op::Ftst => {
                let a = self.fpu_st(0);
                self.fpu_set_compare(a, 0.0);
            }
            Op::Fxam => {
                let v = self.fpu_st(0);
                // C1 = sign; C3/C2/C0 classify (normal=010, zero=100, NaN=001,
                // inf=011). Empty/denormal approximated as their nearest class.
                self.fpu_sw &= !0x4700;
                if v.is_sign_negative() {
                    self.fpu_sw |= 1 << 9; // C1
                }
                let (c3, c2, c0) = if v.is_nan() {
                    (false, false, true)
                } else if v.is_infinite() {
                    (false, true, true)
                } else if v == 0.0 {
                    (true, false, false)
                } else {
                    (false, true, false)
                };
                if c3 {
                    self.fpu_sw |= 1 << 14;
                }
                if c2 {
                    self.fpu_sw |= 1 << 10;
                }
                if c0 {
                    self.fpu_sw |= 1 << 8;
                }
            }
            Op::FpuCompare { with_mem, mem_bytes, integer, sti, pops, eflags } => {
                let a = self.fpu_st(0);
                let b = if with_mem {
                    let addr = self.effective_address(insn.src.as_ref().unwrap());
                    if integer {
                        sext(self.mem_read(mem, io, addr, mem_bytes as usize, Access::Read)?, mem_bytes as u32 * 8) as i64 as f64
                    } else if mem_bytes == 4 {
                        f32::from_bits(self.mem_read(mem, io, addr, 4, Access::Read)? as u32) as f64
                    } else {
                        f64::from_bits(self.mem_read(mem, io, addr, 8, Access::Read)?)
                    }
                } else {
                    self.fpu_st(sti)
                };
                if eflags {
                    let (zf, pf, cf) = if a.is_nan() || b.is_nan() {
                        (true, true, true)
                    } else if a < b {
                        (false, false, true)
                    } else if a > b {
                        (false, false, false)
                    } else {
                        (true, false, false)
                    };
                    flags::set_flag(&mut self.regs.rflags, ZF, zf);
                    flags::set_flag(&mut self.regs.rflags, PF, pf);
                    flags::set_flag(&mut self.regs.rflags, CF, cf);
                    flags::set_flag(&mut self.regs.rflags, OF, false);
                    flags::set_flag(&mut self.regs.rflags, SF, false);
                    flags::set_flag(&mut self.regs.rflags, AF, false);
                } else {
                    self.fpu_set_compare(a, b);
                }
                for _ in 0..pops {
                    self.fpu_pop();
                }
            }
            Op::Ffree { i } => {
                let phys = (self.fpu_top + i) & 7;
                self.fpu_set_phys_tag(phys, true);
            }
            Op::Fincstp => {
                self.fpu_top = (self.fpu_top + 1) & 7;
                self.fpu_sync_sw_top();
            }
            Op::Fdecstp => {
                self.fpu_top = self.fpu_top.wrapping_sub(1) & 7;
                self.fpu_sync_sw_top();
            }
            Op::Fxsave => {
                // 512-byte FP/SSE save area: FCW/FSW/FTW + MXCSR + XMM0-15. We model
                // no x87 stack registers, so the ST area (offset 32..160) is zeroed.
                let addr = self.effective_address(insn.dst.as_ref().unwrap());
                for i in 0..512u64 {
                    self.mem_write(mem, io, addr + i, 1, 0)?;
                }
                self.mem_write(mem, io, addr, 2, self.fpu_cw as u64)?;
                self.mem_write(mem, io, addr + 2, 2, self.fpu_sw as u64)?;
                self.mem_write(mem, io, addr + 4, 1, 0)?; // abridged FTW (no regs valid)
                self.mem_write(mem, io, addr + 24, 4, self.mxcsr as u64)?;
                self.mem_write(mem, io, addr + 28, 4, 0xffff)?; // MXCSR_MASK
                for i in 0..16u64 {
                    let x = self.xmm[i as usize];
                    self.mem_write(mem, io, addr + 160 + i * 16, 8, x as u64)?;
                    self.mem_write(mem, io, addr + 160 + i * 16 + 8, 8, (x >> 64) as u64)?;
                }
            }
            Op::Fxrstor => {
                let addr = self.effective_address(insn.src.as_ref().unwrap());
                self.fpu_cw = self.mem_read(mem, io, addr, 2, Access::Read)? as u16;
                self.fpu_sw = self.mem_read(mem, io, addr + 2, 2, Access::Read)? as u16;
                self.mxcsr = self.mem_read(mem, io, addr + 24, 4, Access::Read)? as u32;
                for i in 0..16u64 {
                    let lo = self.mem_read(mem, io, addr + 160 + i * 16, 8, Access::Read)? as u128;
                    let hi = self.mem_read(mem, io, addr + 160 + i * 16 + 8, 8, Access::Read)? as u128;
                    self.xmm[i as usize] = lo | (hi << 64);
                }
            }
            Op::Xgetbv => {
                // ECX selects the XCR; only XCR0 (ecx=0) is supported.
                let ecx = self.regs.read(RCX, Size::Dword, false) as u32;
                let v = if ecx == 0 { self.xcr0 } else { 0 };
                self.regs.write(RAX, Size::Dword, false, v & 0xffff_ffff);
                self.regs.write(RDX, Size::Dword, false, v >> 32);
            }
            Op::Xsetbv => {
                let ecx = self.regs.read(RCX, Size::Dword, false) as u32;
                let value = ((self.regs.read(RDX, Size::Dword, false) & 0xffff_ffff) << 32)
                    | (self.regs.read(RAX, Size::Dword, false) & 0xffff_ffff);
                if ecx == 0 {
                    self.xcr0 = value;
                }
            }
            Op::Xsave { compacted } => {
                // Save the components in EDX:EAX & XCR0 (compacted = XSAVEC/XSAVES).
                let addr = self.effective_address(insn.dst.as_ref().unwrap());
                let mask = ((self.regs.read(RDX, Size::Dword, false) & 0xffff_ffff) << 32
                    | self.regs.read(RAX, Size::Dword, false) & 0xffff_ffff)
                    & self.xcr0;
                // Legacy area (x87 comp0 + SSE comp1) — mirror FXSAVE.
                if mask & 0x3 != 0 {
                    for i in 0..512u64 {
                        self.mem_write(mem, io, addr + i, 1, 0)?;
                    }
                    self.mem_write(mem, io, addr, 2, self.fpu_cw as u64)?;
                    self.mem_write(mem, io, addr + 2, 2, self.fpu_sw as u64)?;
                    self.mem_write(mem, io, addr + 24, 4, self.mxcsr as u64)?;
                    self.mem_write(mem, io, addr + 28, 4, 0xffff)?;
                    for i in 0..16u64 {
                        let x = self.xmm[i as usize];
                        self.mem_write(mem, io, addr + 160 + i * 16, 8, x as u64)?;
                        self.mem_write(mem, io, addr + 160 + i * 16 + 8, 8, (x >> 64) as u64)?;
                    }
                }
                // Header: XSTATE_BV at +512. XCOMP_BV at +520 — bit 63 set + the
                // saved components for the compacted format (XSAVEC/XSAVES), else 0.
                self.mem_write(mem, io, addr + 512, 8, mask)?;
                let xcomp_bv = if compacted { (1u64 << 63) | mask } else { 0 };
                self.mem_write(mem, io, addr + 520, 8, xcomp_bv)?;
                for i in 0..6u64 {
                    self.mem_write(mem, io, addr + 528 + i * 8, 8, 0)?;
                }
                // AVX state (comp2): YMM_Hi[0..15] at offset 576.
                if mask & 0x4 != 0 {
                    for i in 0..16u64 {
                        let h = self.ymm_hi[i as usize];
                        self.mem_write(mem, io, addr + 576 + i * 16, 8, h as u64)?;
                        self.mem_write(mem, io, addr + 576 + i * 16 + 8, 8, (h >> 64) as u64)?;
                    }
                }
            }
            Op::Xrstor => {
                let addr = self.effective_address(insn.src.as_ref().unwrap());
                let mask = ((self.regs.read(RDX, Size::Dword, false) & 0xffff_ffff) << 32
                    | self.regs.read(RAX, Size::Dword, false) & 0xffff_ffff)
                    & self.xcr0;
                let xstate_bv = self.mem_read(mem, io, addr + 512, 8, Access::Read)?;
                // Components in mask: restore if XSTATE_BV bit set, else reset to init.
                if mask & 0x1 != 0 {
                    if xstate_bv & 0x1 != 0 {
                        self.fpu_cw = self.mem_read(mem, io, addr, 2, Access::Read)? as u16;
                        self.fpu_sw = self.mem_read(mem, io, addr + 2, 2, Access::Read)? as u16;
                    } else {
                        // x87 component to INIT — must match FNINIT exactly (the path the
                        // FXSAVE/no-xstate boot uses), incl. tag word + TOP, else the two
                        // boots diverge after FPU init.
                        self.fpu_cw = 0x037f;
                        self.fpu_sw = 0;
                        self.fpu_tw = 0xffff;
                        self.fpu_top = 0;
                        self.st = [F80::ZERO; 8];
                    }
                }
                if mask & 0x2 != 0 {
                    if xstate_bv & 0x2 != 0 {
                        self.mxcsr = self.mem_read(mem, io, addr + 24, 4, Access::Read)? as u32;
                        for i in 0..16u64 {
                            let lo = self.mem_read(mem, io, addr + 160 + i * 16, 8, Access::Read)? as u128;
                            let hi = self.mem_read(mem, io, addr + 160 + i * 16 + 8, 8, Access::Read)? as u128;
                            self.xmm[i as usize] = lo | (hi << 64);
                        }
                    } else {
                        self.mxcsr = 0x1f80;
                        self.xmm = [0; 16];
                    }
                }
                if mask & 0x4 != 0 {
                    if xstate_bv & 0x4 != 0 {
                        for i in 0..16u64 {
                            let lo = self.mem_read(mem, io, addr + 576 + i * 16, 8, Access::Read)? as u128;
                            let hi = self.mem_read(mem, io, addr + 576 + i * 16 + 8, 8, Access::Read)? as u128;
                            self.ymm_hi[i as usize] = lo | (hi << 64);
                        }
                    } else {
                        self.ymm_hi = [0; 16];
                    }
                }
            }
            Op::Ldmxcsr => {
                let addr = self.effective_address(insn.src.as_ref().unwrap());
                self.mxcsr = self.mem_read(mem, io, addr, 4, Access::Read)? as u32;
            }
            Op::Stmxcsr => {
                let addr = self.effective_address(insn.dst.as_ref().unwrap());
                self.mem_write(mem, io, addr, 4, self.mxcsr as u64)?;
            }
            Op::Iret => self.iret(io, mem)?,
            Op::Int => {
                let vector = self.read(io, insn.src.as_ref().unwrap(), mem)? as u8;
                self.deliver_interrupt(io, mem, vector, None)?;
            }
            Op::Rdmsr => {
                let msr = self.regs.read(RCX, Size::Dword, false) as u32;
                let value = self.rdmsr(msr)?;
                self.regs.write(RAX, Size::Dword, false, value & 0xffff_ffff);
                self.regs.write(RDX, Size::Dword, false, value >> 32);
            }
            Op::Wrmsr => {
                let msr = self.regs.read(RCX, Size::Dword, false) as u32;
                let lo = self.regs.read(RAX, Size::Dword, false);
                let hi = self.regs.read(RDX, Size::Dword, false);
                self.wrmsr(msr, (hi << 32) | lo)?;
            }
            Op::Cpuid => {
                let leaf = self.regs.read(RAX, Size::Dword, false) as u32;
                let subleaf = self.regs.read(RCX, Size::Dword, false) as u32;
                let [a, b, mut c, d] = cpuid(leaf, subleaf);
                // Leaf 1 ECX.OSXSAVE (bit 27) reflects CR4.OSXSAVE (bit 18): it reads
                // 1 only once the OS has enabled XSAVE. This is what gates AVX use.
                if leaf == 1 {
                    if self.sys.cr4 & (1 << 18) != 0 {
                        c |= 1 << 27;
                    } else {
                        c &= !(1u32 << 27);
                    }
                }
                self.regs.write(RAX, Size::Dword, false, a as u64);
                self.regs.write(RBX, Size::Dword, false, b as u64);
                self.regs.write(RCX, Size::Dword, false, c as u64);
                self.regs.write(RDX, Size::Dword, false, d as u64);
            }
            Op::Lgdt | Op::Lidt => {
                if self.cpl != 0 {
                    return Err(X86Error::GeneralProtection);
                }
                let op = insn.dst.as_ref().unwrap();
                let addr = self.effective_address(op);
                let limit = self.mem_read(mem, io, addr, 2, Access::Read)? as u16;
                // The base is 64-bit in long mode (m16:64) but 32-bit in protected/real mode
                // (m16:32) — reading 8 there would pull in 2 bytes past the 6-byte operand.
                let base_bytes = if matches!(self.cpu_mode(), CpuMode::Long) { 8 } else { 4 };
                let base = self.mem_read(mem, io, addr.wrapping_add(2), base_bytes, Access::Read)?;
                if matches!(insn.op, Op::Lgdt) {
                    self.sys.gdtr_base = base;
                    self.sys.gdtr_limit = limit;
                } else {
                    self.sys.idtr_base = base;
                    self.sys.idtr_limit = limit;
                }
            }
            Op::Verw => {
                // Flat model: the segment is always accessible → ZF = 1.
                flags::set_flag(&mut self.regs.rflags, ZF, true);
            }
            Op::Ltr => {
                // Load the task register: resolve the 64-bit TSS base from its GDT
                // descriptor (16 bytes; base split across bytes 2–4, 7, 8–11) so
                // privilege-increasing interrupts can read TSS.RSP0 live.
                let sel = self.read(io, insn.src.as_ref().unwrap(), mem)? & 0xfff8;
                let desc = self.sys.gdtr_base.wrapping_add(sel);
                let lo = self.mem_read(mem, io, desc, 8, Access::Read)?;
                let hi = self.mem_read(mem, io, desc.wrapping_add(8), 8, Access::Read)?;
                self.sys.tr_base = ((lo >> 16) & 0xff_ffff)
                    | (((lo >> 56) & 0xff) << 24)
                    | ((hi & 0xffff_ffff) << 32);
            }
            Op::Sgdt | Op::Sidt => {
                // Store the descriptor-table register (limit then 8-byte base).
                let op = insn.dst.as_ref().unwrap();
                let addr = self.effective_address(op);
                let (base, limit) = if matches!(insn.op, Op::Sgdt) {
                    (self.sys.gdtr_base, self.sys.gdtr_limit)
                } else {
                    (self.sys.idtr_base, self.sys.idtr_limit)
                };
                self.mem_write(mem, io, addr, 2, limit as u64)?;
                self.mem_write(mem, io, addr.wrapping_add(2), 8, base)?;
            }
            Op::In => {
                let port = self.read(io, insn.src.as_ref().unwrap(), mem)? as u16;
                let dst = insn.dst.as_ref().unwrap();
                let size = operand_size(dst);
                let value = io.read(port, size.bytes());
                self.write(io, dst, value, mem)?;
            }
            Op::Out => {
                let port = self.read(io, insn.dst.as_ref().unwrap(), mem)? as u16;
                let src = insn.src.as_ref().unwrap();
                let size = operand_size(src);
                let value = self.read(io, src, mem)?;
                io.write(port, size.bytes(), value);
            }
            Op::Syscall => self.syscall()?,
            Op::Sysret => self.sysret(mem)?,
            Op::Swapgs => {
                if self.cpl != 0 {
                    return Err(X86Error::GeneralProtection);
                }
                std::mem::swap(&mut self.msrs.gs_base, &mut self.msrs.kernel_gs_base);
            }
            Op::Invlpg => self.flush_decode_cache(),
            Op::Nop => {}
            Op::Pause => {
                // Architecturally a NOP; also flag a spin-wait yield for the SMP scheduler.
                self.pause_yield = true;
            }
            Op::Fence => {
                // MFENCE/SFENCE/LFENCE: drain this core's store buffer so its prior stores are
                // globally visible before any later access. Empty (⇒ no-op) with TSO off.
                self.sb_drain_all(mem);
            }
            Op::Hlt => {
                // A halted core's buffered stores must become globally visible (it will not
                // execute a fence itself). No-op with TSO off.
                self.sb_drain_all(mem);
                self.halted = true;
            }
        }
        Ok(())
    }

    /// One-operand unsigned multiply: {AX | DX:AX | EDX:EAX | RDX:RAX} = acc * src.
    fn mul(&mut self, io: &mut IoBus, src: &Operand, mem: &mut GuestMemory) -> Result<(), X86Error> {
        let size = operand_size(src);
        let mask = size.mask();
        let v = self.read(io, src, mem)? & mask;
        let high_nonzero = if size == Size::Byte {
            let al = self.regs.read(RAX, Size::Byte, false) & 0xff;
            let product = al * v;
            self.regs.write(RAX, Size::Word, false, product);
            product >> 8 != 0
        } else {
            let bits = size.bits();
            let acc = (self.regs.read(RAX, size, false) & mask) as u128;
            let product = acc * v as u128;
            let low = product as u64 & mask;
            let high = (product >> bits) as u64 & mask;
            self.regs.write(RAX, size, false, low);
            self.regs.write(RDX, size, false, high);
            high != 0
        };
        let rflags = &mut self.regs.rflags;
        flags::clear_arith_undefined(rflags);
        flags::set_flag(rflags, CF, high_nonzero);
        flags::set_flag(rflags, OF, high_nonzero);
        Ok(())
    }

    /// IMUL, one-operand (implicit accumulator) and two-operand (`r, r/m`) forms.
    fn imul(&mut self, io: &mut IoBus, insn: &DecodedInsn, mem: &mut GuestMemory) -> Result<(), X86Error> {
        let overflow;
        if let Some(dst) = insn.dst.as_ref() {
            let size = operand_size(dst);
            let bits = size.bits();
            let b = sext(self.read(io, insn.src.as_ref().unwrap(), mem)?, bits) as i128;
            // Three-operand form (IMUL r, r/m, imm) carries the immediate in `rel`
            // and does NOT read the destination; two-operand form reads dst*src.
            let a = match insn.rel {
                Some(imm) => imm as i128,
                None => sext(self.read(io, dst, mem)?, bits) as i128,
            };
            let product = a * b;
            let result = product as u64 & size.mask();
            self.write(io, dst, result, mem)?;
            overflow = sext(result, bits) as i128 != product;
        } else {
            let src = insn.src.as_ref().unwrap();
            let size = operand_size(src);
            let bits = size.bits();
            let v = self.read(io, src, mem)?;
            if size == Size::Byte {
                let al = sext(self.regs.read(RAX, Size::Byte, false), 8);
                let product = al * sext(v, 8);
                self.regs.write(RAX, Size::Word, false, product as u64 & 0xffff);
                overflow = product as i8 as i64 != product;
            } else {
                let mask = size.mask();
                let acc = sext(self.regs.read(RAX, size, false), bits) as i128;
                let product = acc * sext(v, bits) as i128;
                let low = product as u64 & mask;
                let high = (product >> bits) as u64 & mask;
                self.regs.write(RAX, size, false, low);
                self.regs.write(RDX, size, false, high);
                overflow = sext(low, bits) as i128 != product;
            }
        }
        let rflags = &mut self.regs.rflags;
        flags::clear_arith_undefined(rflags);
        flags::set_flag(rflags, CF, overflow);
        flags::set_flag(rflags, OF, overflow);
        Ok(())
    }

    /// DIV / IDIV (one-operand). Raises `#DE` on divide-by-zero or quotient
    /// overflow. All arithmetic flags are undefined → cleared.
    fn divide(&mut self, io: &mut IoBus, src: &Operand, mem: &mut GuestMemory, signed: bool) -> Result<(), X86Error> {
        let size = operand_size(src);
        let mask = size.mask();
        let v = self.read(io, src, mem)? & mask;

        if size == Size::Byte {
            if signed {
                let dividend = sext(self.regs.read(RAX, Size::Word, false), 16);
                let divisor = sext(v, 8);
                if divisor == 0 {
                    return Err(X86Error::DivideError);
                }
                let q = dividend / divisor;
                if !(-128..=127).contains(&q) {
                    return Err(X86Error::DivideError);
                }
                self.regs.write(RAX, Size::Byte, false, q as u64 & 0xff);
                self.regs
                    .write(RAX, Size::Byte, true, (dividend % divisor) as u64 & 0xff);
            } else {
                let dividend = self.regs.read(RAX, Size::Word, false) & 0xffff;
                if v == 0 {
                    return Err(X86Error::DivideError);
                }
                let q = dividend / v;
                if q > 0xff {
                    return Err(X86Error::DivideError);
                }
                self.regs.write(RAX, Size::Byte, false, q);
                self.regs.write(RAX, Size::Byte, true, dividend % v);
            }
        } else {
            let bits = size.bits();
            let hi = self.regs.read(RDX, size, false) & mask;
            let lo = self.regs.read(RAX, size, false) & mask;
            let dividend_u = ((hi as u128) << bits) | lo as u128;
            if signed {
                let dividend = sext128(dividend_u, bits * 2);
                let divisor = sext128(v as u128, bits);
                if divisor == 0 {
                    return Err(X86Error::DivideError);
                }
                // `checked_div` guards the one overflowing case (MIN / -1), which
                // would otherwise panic the host before the range check — the
                // architecturally correct result is #DE. (e.g. qword RDX:RAX =
                // i128::MIN / -1: quotient 2^127 is unrepresentable.)
                let q = match dividend.checked_div(divisor) {
                    Some(q) => q,
                    None => return Err(X86Error::DivideError),
                };
                let qmin = -(1_i128 << (bits - 1));
                let qmax = (1_i128 << (bits - 1)) - 1;
                if q < qmin || q > qmax {
                    return Err(X86Error::DivideError);
                }
                self.regs.write(RAX, size, false, q as u64 & mask);
                self.regs
                    .write(RDX, size, false, (dividend % divisor) as u64 & mask);
            } else {
                let divisor = v as u128;
                if divisor == 0 {
                    return Err(X86Error::DivideError);
                }
                let q = dividend_u / divisor;
                if q > mask as u128 {
                    return Err(X86Error::DivideError);
                }
                self.regs.write(RAX, size, false, q as u64 & mask);
                self.regs
                    .write(RDX, size, false, (dividend_u % divisor) as u64 & mask);
            }
        }
        flags::clear_arith_undefined(&mut self.regs.rflags);
        Ok(())
    }

    fn alu(&mut self, io: &mut IoBus, insn: &DecodedInsn, mem: &mut GuestMemory) -> Result<(), X86Error> {
        let dst = insn.dst.as_ref().unwrap();
        let size = operand_size(dst);
        let a = self.read(io, dst, mem)?;
        let b = self.read(io, insn.src.as_ref().unwrap(), mem)?;
        let rflags = &mut self.regs.rflags;
        let (result, write_back) = match insn.op {
            Op::Add => (flags::add(rflags, a, b, size), true),
            Op::Adc => (flags::adc(rflags, a, b, size), true),
            Op::Sub => (flags::sub(rflags, a, b, size), true),
            Op::Sbb => (flags::sbb(rflags, a, b, size), true),
            Op::Cmp => (flags::sub(rflags, a, b, size), false),
            Op::And => {
                let r = (a & b) & size.mask();
                flags::set_logic_flags(rflags, r, size);
                (r, true)
            }
            Op::Test => {
                let r = (a & b) & size.mask();
                flags::set_logic_flags(rflags, r, size);
                (r, false)
            }
            Op::Or => {
                let r = (a | b) & size.mask();
                flags::set_logic_flags(rflags, r, size);
                (r, true)
            }
            Op::Xor => {
                let r = (a ^ b) & size.mask();
                flags::set_logic_flags(rflags, r, size);
                (r, true)
            }
            _ => unreachable!(),
        };
        if write_back {
            self.write(io, dst, result, mem)?;
        }
        Ok(())
    }

    fn read(&mut self, io: &mut IoBus, op: &Operand, mem: &mut GuestMemory) -> Result<u64, X86Error> {
        Ok(match op {
            Operand::Reg { id, size, high8 } => self.regs.read(*id, *size, *high8),
            Operand::Mem { size, .. } => {
                let addr = self.effective_address(op);
                self.mem_read(mem, io, addr, size.bytes(), Access::Read)?
            }
            Operand::Imm { value, .. } => *value,
            Operand::Xmm { .. } => {
                return Err(X86Error::InvalidEncoding("XMM via scalar read".into()))
            }
        })
    }

    fn write(&mut self, io: &mut IoBus, op: &Operand, value: u64, mem: &mut GuestMemory) -> Result<(), X86Error> {
        match op {
            Operand::Reg { id, size, high8 } => self.regs.write(*id, *size, *high8, value),
            Operand::Mem { size, .. } => {
                let addr = self.effective_address(op);
                self.mem_write(mem, io, addr, size.bytes(), value)?;
            }
            Operand::Imm { .. } | Operand::Xmm { .. } => {
                return Err(X86Error::InvalidEncoding("bad scalar write target".into()))
            }
        }
        Ok(())
    }

    // ---- AVX / YMM (256-bit) helpers ----

    fn read_u128_at(&mut self, io: &mut IoBus, addr: u64, mem: &mut GuestMemory) -> Result<u128, X86Error> {
        let lo = self.mem_read(mem, io, addr, 8, Access::Read)? as u128;
        let hi = self.mem_read(mem, io, addr.wrapping_add(8), 8, Access::Read)? as u128;
        Ok(lo | (hi << 64))
    }

    fn write_u128_at(&mut self, io: &mut IoBus, addr: u64, val: u128, mem: &mut GuestMemory) -> Result<(), X86Error> {
        self.mem_write(mem, io, addr, 8, val as u64)?;
        self.mem_write(mem, io, addr.wrapping_add(8), 8, (val >> 64) as u64)?;
        Ok(())
    }

    /// Read a YMM source operand (XMM register or memory) as `(lo128, hi128)`.
    /// For 128-bit ops the upper half is reported as 0.
    fn read_ymm(
        &mut self,
        io: &mut IoBus,
        op: &Operand,
        l256: bool,
        mem: &mut GuestMemory,
    ) -> Result<(u128, u128), X86Error> {
        match op {
            Operand::Xmm { id } => Ok((self.xmm[*id], if l256 { self.ymm_hi[*id] } else { 0 })),
            Operand::Mem { .. } => {
                let addr = self.effective_address(op);
                let lo = self.read_u128_at(io, addr, mem)?;
                let hi = if l256 { self.read_u128_at(io, addr.wrapping_add(16), mem)? } else { 0 };
                Ok((lo, hi))
            }
            _ => Err(X86Error::InvalidEncoding("ymm read from non-vector operand".into())),
        }
    }

    /// Write a YMM register: low 128 → XMM, upper 128 → `ymm_hi` (zeroed for
    /// 128-bit ops per the VEX rule that clears YMM[255:128]).
    fn write_ymm_reg(&mut self, id: usize, lo: u128, hi: u128, l256: bool) {
        self.xmm[id] = lo;
        self.ymm_hi[id] = if l256 { hi } else { 0 };
    }

    /// Read up to 16 bytes from an SSE source (XMM / GPR / memory), low-aligned
    /// and zero-extended to 128 bits.
    fn read_value128(
        &mut self,
        io: &mut IoBus,
        op: &Operand,
        bytes: u8,
        mem: &mut GuestMemory,
    ) -> Result<u128, X86Error> {
        let mask = if bytes >= 16 {
            u128::MAX
        } else {
            (1_u128 << (bytes as u32 * 8)) - 1
        };
        let value = match op {
            Operand::Xmm { id } => self.xmm[*id],
            Operand::Reg { id, size, high8 } => self.regs.read(*id, *size, *high8) as u128,
            Operand::Mem { .. } => {
                let addr = self.effective_address(op);
                if bytes <= 8 {
                    self.mem_read(mem, io, addr, bytes as usize, Access::Read)? as u128
                } else {
                    let lo = self.mem_read(mem, io, addr, 8, Access::Read)? as u128;
                    let hi = self.mem_read(mem, io, addr.wrapping_add(8), 8, Access::Read)? as u128;
                    lo | (hi << 64)
                }
            }
            Operand::Imm { value, .. } => *value as u128,
        };
        Ok(value & mask)
    }

    /// Write the low `bytes` of an SSE value to an XMM / GPR / memory destination.
    fn write_value128(
        &mut self,
        io: &mut IoBus,
        op: &Operand,
        value: u128,
        bytes: u8,
        mem: &mut GuestMemory,
    ) -> Result<(), X86Error> {
        match op {
            Operand::Xmm { id } => {
                self.xmm[*id] = if bytes >= 16 {
                    value
                } else {
                    value & ((1_u128 << (bytes as u32 * 8)) - 1) // upper bits zeroed
                };
            }
            Operand::Reg { id, size, high8 } => self.regs.write(*id, *size, *high8, value as u64),
            Operand::Mem { .. } => {
                let addr = self.effective_address(op);
                if bytes <= 8 {
                    self.mem_write(mem, io, addr, bytes as usize, value as u64)?;
                } else {
                    self.mem_write(mem, io, addr, 8, value as u64)?;
                    self.mem_write(mem, io, addr.wrapping_add(8), 8, (value >> 64) as u64)?;
                }
            }
            Operand::Imm { .. } => return Err(X86Error::InvalidEncoding("write to imm".into())),
        }
        Ok(())
    }

    /// Current operating mode from CR0.PE / EFER.LMA.
    pub fn cpu_mode(&self) -> CpuMode {
        if self.sys.efer & EFER_LMA != 0 {
            CpuMode::Long
        } else if self.sys.cr0 & CR0_PE != 0 {
            CpuMode::Protected
        } else {
            CpuMode::Real
        }
    }

    /// Linear base for a segment selector. Real mode = `selector<<4`; protected/long mode assume
    /// flat descriptors (base 0 — matching the flat GDT that Linux's boot/trampoline code loads).
    /// So this is 0 in long mode ⇒ the flat-model fetch/EA are byte-identical.
    fn seg_base(&self, selector: u16) -> u64 {
        if matches!(self.cpu_mode(), CpuMode::Real) {
            (selector as u64) << 4
        } else {
            0
        }
    }

    /// Load CS and latch its hidden descriptor (`cs_base` + `code_bits`) — the CPU's
    /// behaviour on any CS reload (far jmp / far ret through a mode switch). The base and
    /// default width come from the *current* mode at the moment of the load, and are then
    /// cached until the next load. This is the whole point of caching: a subsequent `mov
    /// cr0` that flips PE does NOT retroactively change how the just-loaded CS behaves.
    ///
    /// - **Real mode** (PE=0): base `sel<<4`, 16-bit.
    /// - **Protected** (PE=1, LMA=0): read the GDT descriptor → linear base; D/B bit → 32/16.
    /// - **Long** (LMA=1): base forced 0; L=1 → 64-bit, else D → 32/16. This is the path a
    ///   real trampoline takes on its final `ljmp $0x08,$target64` after enabling PG+LME.
    fn load_cs(&mut self, sel: u16, mem: &mut GuestMemory, io: &mut IoBus) -> Result<(), X86Error> {
        self.cs = sel;
        self.cpl = (sel & 3) as u8;
        match self.cpu_mode() {
            CpuMode::Real => {
                self.cs_base = (sel as u64) << 4;
                self.code_bits = 16;
            }
            CpuMode::Protected => {
                let desc = self.sys.gdtr_base.wrapping_add((sel & 0xfff8) as u64);
                let d = self.mem_read(mem, io, desc, 8, Access::Read)?;
                // Descriptor base: bits 16-39 (low 24) + bits 56-63 (high 8).
                self.cs_base = ((d >> 16) & 0xff_ffff) | (((d >> 56) & 0xff) << 24);
                // D/B (bit 54): 1 = 32-bit code segment, 0 = 16-bit.
                self.code_bits = if (d >> 54) & 1 != 0 { 32 } else { 16 };
            }
            CpuMode::Long => {
                let desc = self.sys.gdtr_base.wrapping_add((sel & 0xfff8) as u64);
                let d = self.mem_read(mem, io, desc, 8, Access::Read)?;
                self.cs_base = 0; // CS.base is ignored in 64-bit mode
                // L (bit 53): 1 = 64-bit. Else fall back to D (bit 54) for 32-bit compat.
                self.code_bits = if (d >> 53) & 1 != 0 {
                    64
                } else if (d >> 54) & 1 != 0 {
                    32
                } else {
                    16
                };
            }
        }
        Ok(())
    }

    fn effective_address(&self, op: &Operand) -> u64 {
        match op {
            Operand::Mem {
                base,
                index,
                scale,
                disp,
                seg,
                ..
            } => {
                let mut addr = *disp as u64;
                if let Some(b) = base {
                    addr = addr.wrapping_add(self.regs.gpr[*b]);
                }
                if let Some(i) = index {
                    addr = addr.wrapping_add(self.regs.gpr[*i].wrapping_mul(*scale as u64));
                }
                match seg {
                    Seg::Fs => addr = addr.wrapping_add(self.msrs.fs_base),
                    Seg::Gs => addr = addr.wrapping_add(self.msrs.gs_base),
                    Seg::Cs => addr = addr.wrapping_add(self.seg_base(self.cs)),
                    Seg::Ss => addr = addr.wrapping_add(self.seg_base(self.ss)),
                    Seg::Es => addr = addr.wrapping_add(self.seg_base(self.es)),
                    // Default data segment (DS). Base is 0 in long/protected-flat mode (byte-
                    // identical), and `ds<<4` in real mode.
                    Seg::Ds | Seg::None => addr = addr.wrapping_add(self.seg_base(self.ds)),
                }
                addr
            }
            _ => 0,
        }
    }

    /// Stack slot width for the current CS code width: 8 bytes in long mode, 4 in a 32-bit
    /// protected segment, 2 in real/16-bit mode. (We model stack size = code size, which holds
    /// for flat protected mode and real mode — the cases we run.)
    fn stack_op_bytes(&self) -> u64 {
        match self.code_bits {
            16 => 2,
            32 => 4,
            _ => 8,
        }
    }

    /// Linear base of the stack segment: `SS<<4` in real mode, 0 in protected/long (flat).
    fn ss_base(&self) -> u64 {
        if matches!(self.cpu_mode(), CpuMode::Real) {
            (self.ss as u64) << 4
        } else {
            0
        }
    }

    fn push(&mut self, io: &mut IoBus, value: u64, mem: &mut GuestMemory) -> Result<(), X86Error> {
        self.push_sized(io, value, self.stack_op_bytes(), mem)
    }

    /// Push `data_w` bytes (the OPERAND size — usually the stack width, but PUSHF/PUSHFD sets
    /// it from the operand-size prefix independently of the stack width). SP is decremented by
    /// `data_w` and wraps within the *stack* pointer width. Store-before-commit-RSP for the
    /// precise-exception restart model (see the fault note that used to live here).
    fn push_sized(
        &mut self,
        io: &mut IoBus,
        value: u64,
        data_w: u64,
        mem: &mut GuestMemory,
    ) -> Result<(), X86Error> {
        let dmask = if data_w >= 8 { u64::MAX } else { (1u64 << (data_w * 8)) - 1 };
        let stack_w = self.stack_op_bytes();
        if stack_w == 8 {
            // Long-mode fast path — byte-identical to the flat 64-bit model (data_w==8).
            let rsp = self.regs.gpr[RSP].wrapping_sub(data_w);
            self.mem_write(mem, io, rsp, data_w as usize, value & dmask)?;
            self.regs.gpr[RSP] = rsp;
            return Ok(());
        }
        // 16/32-bit stack: SP wraps within its width; the access goes through SS.base.
        let ptr_mask = (1u64 << (stack_w * 8)) - 1;
        let new_sp = self.regs.gpr[RSP].wrapping_sub(data_w) & ptr_mask;
        let addr = self.ss_base().wrapping_add(new_sp);
        self.mem_write(mem, io, addr, data_w as usize, value & dmask)?;
        self.regs.gpr[RSP] = (self.regs.gpr[RSP] & !ptr_mask) | new_sp;
        Ok(())
    }

    fn pop(&mut self, io: &mut IoBus, mem: &mut GuestMemory) -> Result<u64, X86Error> {
        self.pop_sized(io, self.stack_op_bytes(), mem)
    }

    /// Pop `data_w` bytes (operand size). SP advances by `data_w`, wrapping within the stack
    /// pointer width. The returned value is zero-extended to 64 bits.
    fn pop_sized(
        &mut self,
        io: &mut IoBus,
        data_w: u64,
        mem: &mut GuestMemory,
    ) -> Result<u64, X86Error> {
        let stack_w = self.stack_op_bytes();
        if stack_w == 8 {
            let rsp = self.regs.gpr[RSP];
            let value = self.mem_read(mem, io, rsp, data_w as usize, Access::Read)?;
            self.regs.gpr[RSP] = rsp.wrapping_add(data_w);
            return Ok(value);
        }
        let ptr_mask = (1u64 << (stack_w * 8)) - 1;
        let sp = self.regs.gpr[RSP] & ptr_mask;
        let addr = self.ss_base().wrapping_add(sp);
        let value = self.mem_read(mem, io, addr, data_w as usize, Access::Read)?;
        let new_sp = sp.wrapping_add(data_w) & ptr_mask;
        self.regs.gpr[RSP] = (self.regs.gpr[RSP] & !ptr_mask) | new_sp;
        Ok(value)
    }
}

/// Sign-extend the low `bits` of `value` to a full i64.
fn sext(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// Sign-extend the low `bits` of `value` to a full i128.
fn sext128(value: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((value << shift) as i128) >> shift
}

/// Map a CPU error to its `(vector, error_code)` for IDT delivery, or `None` if
/// it is not a deliverable guest exception (e.g. a host backing-store error).
fn exception_vector(err: &X86Error) -> Option<(u8, Option<u64>)> {
    match err {
        X86Error::DivideError => Some((0, None)),                      // #DE
        X86Error::Unsupported(_) | X86Error::InvalidEncoding(_) => Some((6, None)), // #UD
        X86Error::PageFault { error_code, .. } => Some((14, Some(*error_code))), // #PF
        X86Error::GeneralProtection => Some((13, Some(0))),                      // #GP
        X86Error::Memory(_) => None,
    }
}

/// BT/BTS/BTR/BTC: CF = selected bit; OF/SF/AF/PF undefined→0; ZF unchanged.
fn set_bit_flags(rflags: &mut u64, cf: bool) {
    flags::set_flag(rflags, CF, cf);
    for b in [OF, SF, AF, PF] {
        flags::set_flag(rflags, b, false);
    }
}

/// Truncating float→signed-int conversion matching x86 (NaN / out-of-range
/// yield the "integer indefinite" value: sign bit set, rest zero).
/// Canonical bit pattern for a scalar SSE double result, applying x86's NaN
/// rules from operand *bits* (not host NaN behavior) so the result is identical
/// on any host architecture, not just x86. A source NaN propagates (dest `a`
/// first, then `b`), made quiet; a freshly-produced NaN (invalid op, e.g.
/// `inf−inf`) becomes the x86 QNaN floating-point indefinite. `a`/`b` are the
/// operands (`b` twice for unary sqrt).
// ---- MXCSR rounding-control (RC, bits 13:14) ----
// 0 = round-to-nearest-even (default), 1 = toward −inf, 2 = toward +inf, 3 = toward 0.
// The host computes every basic op in round-to-nearest; to honour a *directed* mode we
// take that nearest result `n` and the EXACT residual sign (exact − n) and step to the
// adjacent representable value when needed. The residual is computed with only host f64
// + the IEEE fused `mul_add` / Knuth `two_sum`, so it is bit-deterministic and identical
// on any IEEE-754 host (incl. aarch64). rc==0 is a no-op, so default behaviour — and thus
// every gate golden — is byte-unchanged.

// ---- MXCSR DAZ (bit 6, inputs) / FTZ (bit 15, outputs) ----
// DAZ: a denormal INPUT is treated as sign-preserved zero before the op. FTZ: a denormal
// RESULT of an arithmetic op is flushed to sign-preserved zero. Both default OFF (0x1f80),
// so no perturbation to the gate. NaN/inf/zero/normal pass through unchanged.
fn daz_f64(x: f64, daz: bool) -> f64 {
    if daz && x.is_subnormal() { f64::from_bits(x.to_bits() & (1 << 63)) } else { x }
}
fn daz_f32(x: f32, daz: bool) -> f32 {
    if daz && x.is_subnormal() { f32::from_bits(x.to_bits() & (1 << 31)) } else { x }
}

fn next_up_f64(x: f64) -> f64 {
    let b = x.to_bits();
    let nb = if x == 0.0 { 1 } else if b >> 63 == 0 { b + 1 } else { b - 1 };
    f64::from_bits(nb)
}
fn next_down_f64(x: f64) -> f64 {
    let b = x.to_bits();
    let nb = if x == 0.0 { 0x8000_0000_0000_0001 } else if b >> 63 == 0 { b - 1 } else { b + 1 };
    f64::from_bits(nb)
}
fn next_up_f32(x: f32) -> f32 {
    let b = x.to_bits();
    let nb = if x == 0.0 { 1 } else if b >> 31 == 0 { b + 1 } else { b - 1 };
    f32::from_bits(nb)
}
fn next_down_f32(x: f32) -> f32 {
    let b = x.to_bits();
    let nb = if x == 0.0 { 0x8000_0001 } else if b >> 31 == 0 { b - 1 } else { b + 1 };
    f32::from_bits(nb)
}

/// Apply MXCSR RC to a finite/inf host-nearest result `n`. `resid` carries the sign of
/// (exact − n) and is 0.0 iff the op was exact. `operands_finite` distinguishes a genuine
/// overflow (finite operands → inf) — which directed modes clamp to the max finite — from
/// a true infinity (an inf operand), which stays inf.
fn apply_rc_f64(n: f64, resid: f64, rc: u8, operands_finite: bool) -> f64 {
    if rc == 0 || n.is_nan() {
        return n;
    }
    if n.is_infinite() {
        if operands_finite {
            return match rc {
                1 => if n > 0.0 { f64::MAX } else { f64::NEG_INFINITY },
                2 => if n > 0.0 { f64::INFINITY } else { f64::MIN },
                3 => if n > 0.0 { f64::MAX } else { f64::MIN },
                _ => n,
            };
        }
        return n;
    }
    if resid == 0.0 {
        return n;
    }
    let above = resid > 0.0; // exact > n
    match rc {
        1 => if above { n } else { next_down_f64(n) },
        2 => if above { next_up_f64(n) } else { n },
        3 => {
            if n == 0.0 { n } else if n > 0.0 {
                if above { n } else { next_down_f64(n) }
            } else if above { next_up_f64(n) } else { n }
        }
        _ => n,
    }
}
fn apply_rc_f32(n: f32, resid: f32, rc: u8, operands_finite: bool) -> f32 {
    if rc == 0 || n.is_nan() {
        return n;
    }
    if n.is_infinite() {
        if operands_finite {
            return match rc {
                1 => if n > 0.0 { f32::MAX } else { f32::NEG_INFINITY },
                2 => if n > 0.0 { f32::INFINITY } else { f32::MIN },
                3 => if n > 0.0 { f32::MAX } else { f32::MIN },
                _ => n,
            };
        }
        return n;
    }
    if resid == 0.0 {
        return n;
    }
    let above = resid > 0.0;
    match rc {
        1 => if above { n } else { next_down_f32(n) },
        2 => if above { next_up_f32(n) } else { n },
        3 => {
            if n == 0.0 { n } else if n > 0.0 {
                if above { n } else { next_down_f32(n) }
            } else if above { next_up_f32(n) } else { n }
        }
        _ => n,
    }
}

// ---- MXCSR exception status bits (sticky, masked behaviour = default 0x1f80) ----
const FE_IE: u32 = 0x01; // invalid operation
const FE_DE: u32 = 0x02; // denormal operand
const FE_ZE: u32 = 0x04; // divide by zero
const FE_OE: u32 = 0x08; // overflow
const FE_UE: u32 = 0x10; // underflow
const FE_PE: u32 = 0x20; // precision (inexact)

fn is_snan_f64(x: f64) -> bool {
    let b = x.to_bits();
    b & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000
        && b & 0x000f_ffff_ffff_ffff != 0 // NaN
        && b & 0x0008_0000_0000_0000 == 0 // quiet bit clear → signaling
}
fn is_snan_f32(x: f32) -> bool {
    let b = x.to_bits();
    b & 0x7f80_0000 == 0x7f80_0000 && b & 0x007f_ffff != 0 && b & 0x0040_0000 == 0
}
fn is_subnormal_bits_f64(b: u64) -> bool {
    b & 0x7ff0_0000_0000_0000 == 0 && b & 0x000f_ffff_ffff_ffff != 0
}
fn is_subnormal_bits_f32(b: u32) -> bool {
    b & 0x7f80_0000 == 0 && b & 0x007f_ffff != 0
}

/// x = m·2^e with m ∈ [1,2) for a finite nonzero x (sign dropped — caller passes |x|).
fn frexp2(x: f64) -> (f64, i32) {
    let b = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exp = (b >> 52) as i32;
    if exp == 0 {
        let lz = b.leading_zeros(); // b ≤ 52-bit subnormal mantissa ⇒ lz ≥ 12
        let shift = lz - 11;
        let mant = (b << shift) & 0x000f_ffff_ffff_ffff;
        (f64::from_bits(mant | (1023u64 << 52)), -1022 - shift as i32)
    } else {
        (f64::from_bits((b & 0x000f_ffff_ffff_ffff) | (1023u64 << 52)), exp - 1023)
    }
}
/// m·2^e, exact (multiply by exact powers of two, chunked to avoid intermediate over/underflow).
fn ldexp(m: f64, e: i32) -> f64 {
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (mut r, mut ee) = (m, e);
    while ee > 1000 {
        r *= 2f64.powi(1000);
        ee -= 1000;
    }
    while ee < -1000 {
        r *= 2f64.powi(-1000);
        ee += 1000;
    }
    r * 2f64.powi(ee)
}
/// sign(x·y − n) computed exactly when the product UNDERFLOWS (n subnormal/0, so the host
/// `mul_add` residual is lost): scaled-Dekker product vs n in a common normal-range domain.
fn mul_resid_sign(x: f64, y: f64, n: f64) -> f64 {
    let (xm, xe) = frexp2(x);
    let (ym, ye) = frexp2(y);
    let hi = xm * ym; // ∈ [1,4)
    let lo = xm.mul_add(ym, -hi); // exact: xm·ym = hi + lo
    let ns = ldexp(n.abs(), -(xe + ye)); // |n|·2^-(xe+ye) ∈ O(1)
    let d = (hi - ns) + lo; // ∝ |x·y| − |n|
    let mag = if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 };
    let prod_neg = (x < 0.0) != (y < 0.0);
    if prod_neg { -mag } else { mag }
}

/// |x| = m·2^e with m an INTEGER mantissa (2^52..2^53 for normals, <2^52 for subnormals), for a
/// finite nonzero x. Sign dropped.
fn f64_mant_exp(x: f64) -> (u64, i32) {
    let b = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let e = (b >> 52) as i32;
    if e == 0 {
        (b, -1074) // subnormal: |x| = frac·2^-1074
    } else {
        ((b & 0x000f_ffff_ffff_ffff) | (1 << 52), e - 1075) // normal: (2^52+frac)·2^(e-1075)
    }
}
/// sign(x − n·y) computed EXACTLY in the integer domain (underflow-proof) — for the DIV residual
/// when the fused `mul_add` residual vanishes in the subnormal region. Returns 1.0 / −1.0 / 0.0.
/// n = round(x/y) ⇒ n·y ≈ x, so the two magnitudes' exponents differ by ≤ ~2 and the alignment
/// shifts stay tiny (the ≥ guards are pure u128-overflow paranoia, never hit for real inputs).
fn div_resid_sign(x: f64, y: f64, n: f64) -> f64 {
    let (mx, ex) = f64_mant_exp(x);
    let (my, ey) = f64_mant_exp(y);
    let (mn, en) = f64_mant_exp(n);
    let sx = x < 0.0;
    let sny = (n < 0.0) != (y < 0.0); // sign of n·y
    if sx != sny {
        // opposite signs ⇒ x − n·y has x's sign, magnitude |x|+|n·y| (nonzero)
        return if sx { -1.0 } else { 1.0 };
    }
    // same sign S: sign(x − n·y) = S · sign(|x| − |n·y|). Compare |x| = ax·2^ex vs |n·y| = p·2^ep
    // EXACTLY via MSB-normalization (compare top-bit exponent, then normalized mantissa) — no
    // huge shifts, so it works for any magnitudes incl. a 1-bit subnormal x whose ex is far from
    // n·y's exponent even when the two values are nearly equal.
    let p: u128 = (mn as u128) * (my as u128); // |n·y| mantissa (≤106 bits), exp en+ey
    let ep = en + ey;
    let ax = mx as u128; // |x| mantissa (≤53 bits), exp ex
    let s1 = ax.leading_zeros();
    let t1 = ex + 127 - s1 as i32; // exponent of |x|'s MSB
    let n1 = ax << s1; // |x| mantissa, MSB at bit 127
    let s2 = p.leading_zeros();
    let t2 = ep + 127 - s2 as i32; // exponent of |n·y|'s MSB
    let n2 = p << s2;
    let s = if sx { -1.0 } else { 1.0 };
    let greater = if t1 != t2 { t1 > t2 } else { n1 > n2 };
    let equal = t1 == t2 && n1 == n2;
    if equal { 0.0 } else if greater { s } else { -s }
}

// ---- x87 transcendentals: DETERMINISTIC soft-float (committed constants + f64 poly, NEVER
// host libm). Real silicon's x87 transcendentals are implementation-defined (~1 ulp, CPU-
// varying), so we define our own reproducible results — identical on any IEEE host. ----
const LN2_C: f64 = 0.693_147_180_559_945_309_4;
const LOG2E_C: f64 = 1.442_695_040_888_963_407_4; // 1/ln2
const TWO_OVER_PI_C: f64 = 0.636_619_772_367_581_343_1;
// π/2 split for Cody–Waite argument reduction (hi has trailing zero bits).
const PIO2_HI: f64 = 1.570_796_326_734_125_614_6;
const PIO2_MID: f64 = 6.077_100_506_506_192e-11;
const PIO2_LO: f64 = 2.022_266_248_795_950_7e-21;

/// log2(x) for finite x > 0. Exponent extraction + atanh series for ln(mantissa).
fn soft_log2(x: f64) -> f64 {
    let (mut m, mut e) = frexp2(x); // m ∈ [1,2)
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    } // m ∈ [1/√2, √2) → small t
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // ln(m) = 2t·Σ t^{2k}/(2k+1)
    let ln_m = 2.0
        * t
        * (1.0
            + t2 * (1.0 / 3.0
                + t2 * (1.0 / 5.0
                    + t2 * (1.0 / 7.0
                        + t2 * (1.0 / 9.0 + t2 * (1.0 / 11.0 + t2 * (1.0 / 13.0)))))));
    e as f64 + ln_m * LOG2E_C
}

/// 2^x − 1 for |x| ≤ 1 (F2XM1): expm1(x·ln2) Taylor — no −1 cancellation near 0.
fn soft_exp2m1(x: f64) -> f64 {
    let z = x * LN2_C; // |z| ≤ ln2
    let mut term = z;
    let mut sum = z;
    for k in 2..=17u32 {
        term *= z / k as f64;
        sum += term;
    }
    sum
}

/// 2/π as 24-bit big-endian fractional chunks (bit k, 1-indexed, is bit 23−((k−1)%24) of
/// chunk (k−1)/24). Enough bits (1344) to reduce any finite f64 exactly (fdlibm `ipio2`).
static IPIO2: [u32; 56] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041,
    0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C,
    0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
    0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F,
    0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D,
    0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
    0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E, 0xE61B08,
];

/// Bits p_start .. p_{start+count−1} of 2/π (start ≥ 1, count ≤ 128) as an integer, MSB first.
fn twopi_bits(start: i32, count: u32) -> u128 {
    let mut acc: u128 = 0;
    for i in 0..count as i32 {
        let k = start + i; // 1-indexed bit
        let bit = if k < 1 {
            0
        } else {
            let idx = (k - 1) as usize;
            let (chunk, off) = (idx / 24, idx % 24);
            if chunk >= IPIO2.len() { 0 } else { (IPIO2[chunk] >> (23 - off)) & 1 }
        };
        acc = (acc << 1) | bit as u128;
    }
    acc
}

/// Payne–Hanek reduction for large |x|: returns (k, r) with x = k·(π/2) + r, r ∈ [−π/4, π/4],
/// computed EXACTLY in the integer domain (no cancellation), so accurate for any magnitude.
fn payne_hanek(x: f64) -> (i64, f64) {
    let bits = x.to_bits();
    let m: u64 = (bits & 0x000f_ffff_ffff_ffff) | (1 << 52); // 53-bit mantissa (|x| ≥ 2^20 ⇒ normal)
    let e = ((bits >> 52) & 0x7ff) as i32 - 1075; // |x| = m · 2^e
    // Window of 128 bits of 2/π starting at k = e−1; |x|·2/π (windowed) = (m·PW)·2^(2−128).
    let pw = twopi_bits(e - 1, 128);
    let (pl, ph) = (pw as u64, (pw >> 64) as u64);
    let plo = (m as u128) * (pl as u128);
    let phi = (m as u128) * (ph as u128);
    let w0 = plo as u64;
    let mid = (plo >> 64) + (phi & 0xffff_ffff_ffff_ffff);
    let w1 = mid as u64;
    // product = w0 + w1·2^64 + w2·2^128; binary point at bit 126 (weight 2^0).
    let q_mod4 = ((w1 >> 62) & 3) as i64;
    let round_up = (w1 >> 61) & 1 == 1; // fraction bit at weight 2^-1
    let frac = (w0 as u128) | (((w1 & ((1u64 << 62) - 1)) as u128) << 64); // fraction, < 2^126
    let (mut k, mf, sgn) = if round_up {
        (q_mod4 + 1, (1u128 << 126) - frac, -1.0) // f' ∈ [−0.5, 0)
    } else {
        (q_mod4, frac, 1.0) // f' ∈ [0, 0.5)
    };
    // r = f'·(π/2); f' = ±mf·2^-126. mf's top 53 bits suffice for a correctly-scaled r.
    let mut r = sgn * (mf as f64) * 2f64.powi(-126) * std::f64::consts::FRAC_PI_2;
    if x < 0.0 {
        k = -k;
        r = -r;
    }
    (k, r)
}

/// (sin x, cos x), deterministic. Cody–Waite reduction for |x| < 2^20; Payne–Hanek beyond.
fn soft_sincos(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let (k, r) = if x.abs() < 1_048_576.0 {
        let k = round_ties_even_f64(x * TWO_OVER_PI_C);
        (k as i64, ((x - k * PIO2_HI) - k * PIO2_MID) - k * PIO2_LO)
    } else {
        payne_hanek(x)
    };
    let r2 = r * r;
    let s = r
        * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0
                    + r2 * (-1.0 / 5040.0
                        + r2 * (1.0 / 362880.0
                            + r2 * (-1.0 / 39916800.0
                                + r2 * (1.0 / 6227020800.0 + r2 * (-1.0 / 1307674368000.0))))))));
    let c = 1.0
        + r2 * (-1.0 / 2.0
            + r2 * (1.0 / 24.0
                + r2 * (-1.0 / 720.0
                    + r2 * (1.0 / 40320.0
                        + r2 * (-1.0 / 3628800.0
                            + r2 * (1.0 / 479001600.0
                                + r2 * (-1.0 / 87178291200.0
                                    + r2 * (1.0 / 20922789888000.0))))))));
    match k.rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// atan(w) for w ≥ 0, reduced to |arg| ≤ tan(π/8) then Taylor.
fn soft_atan_pos(w: f64) -> f64 {
    if w > 0.414_213_562_373_095_1 {
        std::f64::consts::FRAC_PI_4 + soft_atan_small((w - 1.0) / (w + 1.0))
    } else {
        soft_atan_small(w)
    }
}
fn soft_atan_small(t: f64) -> f64 {
    let t2 = t * t;
    t * (1.0
        + t2 * (-1.0 / 3.0
            + t2 * (1.0 / 5.0
                + t2 * (-1.0 / 7.0
                    + t2 * (1.0 / 9.0 + t2 * (-1.0 / 11.0 + t2 * (1.0 / 13.0)))))))
}
/// atan2(y, x) with full quadrant, deterministic.
fn soft_atan2(y: f64, x: f64) -> f64 {
    if x == 0.0 && y == 0.0 {
        return if x.is_sign_negative() {
            if y.is_sign_negative() { -std::f64::consts::PI } else { std::f64::consts::PI }
        } else {
            y // ±0
        };
    }
    let (ax, ay) = (x.abs(), y.abs());
    let a = if ay <= ax {
        soft_atan_pos(ay / ax)
    } else {
        std::f64::consts::FRAC_PI_2 - soft_atan_pos(ax / ay)
    };
    let a = if x < 0.0 { std::f64::consts::PI - a } else { a };
    if y < 0.0 { -a } else { a }
}

/// Directed-rounded ADD/SUB/MUL/DIV for one f64 lane. Returns (result bits, exceptions).
/// Residual (exact − n): `two_sum` for add/sub, fused `mul_add` for mul/div — all exact
/// (mul falls back to a scaled-Dekker sign on underflow, where the fused residual vanishes).
fn fbin_rounded_f64(kind: FBinKind, x: f64, y: f64, rc: u8) -> (u64, u32) {
    let mut exc = 0u32;
    if is_snan_f64(x) || is_snan_f64(y) {
        exc |= FE_IE;
    }
    let of = x.is_finite() && y.is_finite();
    let (n, resid) = match kind {
        FBinKind::Add => {
            let s = x + y;
            let bb = s - x;
            (s, (x - (s - bb)) + (y - bb))
        }
        FBinKind::Sub => {
            let ny = -y;
            let s = x + ny;
            let bb = s - x;
            (s, (x - (s - bb)) + (ny - bb))
        }
        FBinKind::Mul => {
            let n = x * y;
            let resid = if !n.is_finite() {
                0.0
            } else if n == 0.0 {
                if x != 0.0 && y != 0.0 { mul_resid_sign(x, y, n) } else { 0.0 } // full underflow
            } else {
                let r = x.mul_add(y, -n); // exact when it doesn't itself underflow
                if x != 0.0 && y != 0.0 && (r == 0.0 || is_subnormal_bits_f64(r.to_bits())) {
                    mul_resid_sign(x, y, n) // fused residual underflowed → exact sign via scaled Dekker
                } else {
                    r
                }
            };
            (n, resid)
        }
        FBinKind::Div => {
            let n = x / y;
            // sign(x − n·y): fused mul_add is exact in the normal range, but it UNDERFLOWS to 0
            // (losing sign + inexactness) when the residual is subnormal — e.g. a subnormal
            // divisor. Recover exactly via the integer-domain sign there.
            let rs = if !n.is_finite() || x == 0.0 || y == 0.0 {
                0.0
            } else {
                div_resid_sign(x, y, n) // exact, underflow-proof (bisect: unconditional)
            };
            // convert sign(x − n·y) → sign(quotient − n) via × sign(y)
            let resid = if rs == 0.0 { 0.0 } else if (rs > 0.0) == (y > 0.0) { 1.0 } else { -1.0 };
            (n, resid)
        }
        _ => unreachable!(),
    };
    // Invalid op producing a fresh NaN (inf−inf, 0·inf, 0/0, inf/inf).
    if n.is_nan() && !x.is_nan() && !y.is_nan() {
        exc |= FE_IE;
    }
    // Divide-by-zero: finite nonzero / zero.
    if matches!(kind, FBinKind::Div) && y == 0.0 && x.is_finite() && x != 0.0 {
        exc |= FE_ZE;
    }
    // IEEE §6.3: a sum that is exactly zero is −0 under round-toward−inf, +0 otherwise.
    if matches!(kind, FBinKind::Add | FBinKind::Sub) && resid == 0.0 && n == 0.0 && rc == 1 {
        return (fp_result_f64(-0.0, x, y), exc);
    }
    let out = apply_rc_f64(n, resid, rc, of);
    let divzero = matches!(kind, FBinKind::Div) && y == 0.0 && x.is_finite() && x != 0.0;
    if resid != 0.0 && n.is_finite() {
        exc |= FE_PE; // inexact (finite rounded result); inf/nan results aren't "rounded"
    }
    if of && n.is_infinite() && !divzero {
        exc |= FE_OE | FE_PE; // overflow (finite operands → inf, not divide-by-zero) is inexact
    } else if resid != 0.0 && (is_subnormal_bits_f64(out.to_bits()) || out == 0.0) {
        exc |= FE_UE | FE_PE; // masked underflow: tiny (subnormal or flushed to 0) AND inexact
    }
    (fp_result_f64(out, x, y), exc)
}
/// Native-f32 directed rounding — `two_sum`/`mul_add` in f32 are EXACT for normal-range
/// results (incl. widely-separated exponents, where evaluating in f64 would itself round).
/// f32/f64 SUBNORMAL-underflow directed rounding is a documented follow-up (soft-float).
fn fbin_rounded_f32(kind: FBinKind, x: f32, y: f32, rc: u8) -> (u32, u32) {
    let mut exc = 0u32;
    if is_snan_f32(x) || is_snan_f32(y) {
        exc |= FE_IE;
    }
    let of = x.is_finite() && y.is_finite();
    let (n, resid) = match kind {
        FBinKind::Add => {
            let s = x + y;
            let bb = s - x;
            (s, (x - (s - bb)) + (y - bb))
        }
        FBinKind::Sub => {
            let ny = -y;
            let s = x + ny;
            let bb = s - x;
            (s, (x - (s - bb)) + (ny - bb))
        }
        FBinKind::Mul => {
            let n = x * y;
            // f32×f32 is ALWAYS exact in f64 (≤48-bit product, exp in range), so the f64
            // residual is exact for the whole range — no native-f32 underflow to worry about.
            let resid: f32 = if !n.is_finite() {
                0.0
            } else {
                let d = (x as f64) * (y as f64) - (n as f64);
                if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 }
            };
            (n, resid)
        }
        FBinKind::Div => {
            let n = x / y;
            // f32÷f32: x − n·y is EXACT in f64 (n·y ≤48-bit product; f64's range dwarfs f32
            // subnormals, so no underflow) — gives the true residual sign even in the subnormal grid.
            let resid: f32 = if !n.is_finite() || x == 0.0 || y == 0.0 {
                0.0
            } else {
                let d = (x as f64) - (n as f64) * (y as f64);
                if d == 0.0 { 0.0 } else if (d > 0.0) == (y > 0.0) { 1.0 } else { -1.0 }
            };
            (n, resid)
        }
        _ => unreachable!(),
    };
    if n.is_nan() && !x.is_nan() && !y.is_nan() {
        exc |= FE_IE;
    }
    if matches!(kind, FBinKind::Div) && y == 0.0 && x.is_finite() && x != 0.0 {
        exc |= FE_ZE;
    }
    let rs: f32 = if resid > 0.0 { 1.0 } else if resid < 0.0 { -1.0 } else { 0.0 };
    if matches!(kind, FBinKind::Add | FBinKind::Sub) && rs == 0.0 && n == 0.0 && rc == 1 {
        return (fp_result_f32(-0.0, x, y), exc);
    }
    let out = apply_rc_f32(n, rs, rc, of);
    let divzero = matches!(kind, FBinKind::Div) && y == 0.0 && x.is_finite() && x != 0.0;
    if rs != 0.0 && n.is_finite() {
        exc |= FE_PE;
    }
    if of && n.is_infinite() && !divzero {
        exc |= FE_OE | FE_PE;
    } else if rs != 0.0 && (is_subnormal_bits_f32(out.to_bits()) || out == 0.0) {
        exc |= FE_UE | FE_PE;
    }
    (fp_result_f32(out, x, y), exc)
}

/// Directed-rounded SQRT for one f64/f32 lane → (bits, exceptions). SQRT never over/underflows;
/// it flags IE for a negative operand (or SNaN) and PE when inexact.
fn sqrt_rounded_f64(x: f64, rc: u8) -> (u64, u32) {
    let mut exc = 0u32;
    if is_snan_f64(x) || x < 0.0 {
        exc |= FE_IE; // SNaN, or sqrt of a negative (−0 is ==0.0 → excluded)
    }
    let n = x.sqrt();
    let resid = if !n.is_finite() || n == 0.0 {
        0.0
    } else {
        let r = n.mul_add(n, -x); // exact n²−x; exact > n iff x > n² iff r < 0
        if r == 0.0 { 0.0 } else if r < 0.0 { 1.0 } else { -1.0 }
    };
    if resid != 0.0 {
        exc |= FE_PE;
    }
    (fp_result_f64(apply_rc_f64(n, resid, rc, x.is_finite()), x, x), exc)
}
fn sqrt_rounded_f32(x: f32, rc: u8) -> (u32, u32) {
    let mut exc = 0u32;
    if is_snan_f32(x) || x < 0.0 {
        exc |= FE_IE;
    }
    let n = x.sqrt();
    let rs: f32 = if !n.is_finite() || n == 0.0 {
        0.0
    } else {
        let r = n.mul_add(n, -x);
        if r == 0.0 { 0.0 } else if r < 0.0 { 1.0 } else { -1.0 }
    };
    if rs != 0.0 {
        exc |= FE_PE;
    }
    (fp_result_f32(apply_rc_f32(n, rs, rc, x.is_finite()), x, x), exc)
}

fn fp_result_f64(r: f64, a: f64, b: f64) -> u64 {
    const QUIET: u64 = 1 << 51;
    const INDEFINITE: u64 = 0xFFF8_0000_0000_0000;
    if a.is_nan() {
        a.to_bits() | QUIET
    } else if b.is_nan() {
        b.to_bits() | QUIET
    } else if r.is_nan() {
        INDEFINITE
    } else {
        r.to_bits()
    }
}

/// Single-precision counterpart of [`fp_result_f64`].
fn fp_result_f32(r: f32, a: f32, b: f32) -> u32 {
    const QUIET: u32 = 1 << 22;
    const INDEFINITE: u32 = 0xFFC0_0000;
    if a.is_nan() {
        a.to_bits() | QUIET
    } else if b.is_nan() {
        b.to_bits() | QUIET
    } else if r.is_nan() {
        INDEFINITE
    } else {
        r.to_bits()
    }
}

// ---- FMA (fused multiply-add, single rounding) ----
// `f64::mul_add`/`f32::mul_add` are the IEEE fused op (one rounding); on an FMA
// host they lower to the hardware instruction. NaN is canonicalized in the x86
// FMA source-priority order a, b, c so the result is host-independent.

/// One fused f64 lane: `±(a*b)±c`, with the sign pattern from `kind` (lane index
/// `i` selects the alternating sign for ADDSUB/SUBADD). Returns result bits.
fn fma_one_f64(a: f64, b: f64, c: f64, kind: FmaKind, i: u32) -> u64 {
    let (na, nc) = match kind {
        FmaKind::Madd => (a, c),
        FmaKind::Msub => (a, -c),
        FmaKind::Nmadd => (-a, c),
        FmaKind::Nmsub => (-a, -c),
        FmaKind::Maddsub => (a, if i % 2 == 0 { -c } else { c }),
        FmaKind::Msubadd => (a, if i % 2 == 0 { c } else { -c }),
    };
    let r = na.mul_add(b, nc);
    // NaN propagation uses the ORIGINAL source operands (x86 negates the product/
    // addend value, not the propagated source NaN — its sign is preserved).
    const QUIET: u64 = 1 << 51;
    const INDEFINITE: u64 = 0xFFF8_0000_0000_0000;
    if a.is_nan() {
        a.to_bits() | QUIET
    } else if b.is_nan() {
        b.to_bits() | QUIET
    } else if c.is_nan() {
        c.to_bits() | QUIET
    } else if r.is_nan() {
        INDEFINITE
    } else {
        r.to_bits()
    }
}

/// One fused f32 lane.
fn fma_one_f32(a: f32, b: f32, c: f32, kind: FmaKind, i: u32) -> u32 {
    let (na, nc) = match kind {
        FmaKind::Madd => (a, c),
        FmaKind::Msub => (a, -c),
        FmaKind::Nmadd => (-a, c),
        FmaKind::Nmsub => (-a, -c),
        FmaKind::Maddsub => (a, if i % 2 == 0 { -c } else { c }),
        FmaKind::Msubadd => (a, if i % 2 == 0 { c } else { -c }),
    };
    let r = na.mul_add(b, nc);
    const QUIET: u32 = 1 << 22;
    const INDEFINITE: u32 = 0xFFC0_0000;
    if a.is_nan() {
        a.to_bits() | QUIET
    } else if b.is_nan() {
        b.to_bits() | QUIET
    } else if c.is_nan() {
        c.to_bits() | QUIET
    } else if r.is_nan() {
        INDEFINITE
    } else {
        r.to_bits()
    }
}

/// Arrange the dst/vvvv/src 128-bit lanes into (a, b, c) per the FMA form.
fn fma_operands(form: FmaForm, d: u128, w: u128, s: u128) -> (u128, u128, u128) {
    match form {
        FmaForm::F132 => (d, s, w),
        FmaForm::F213 => (w, d, s),
        FmaForm::F231 => (w, s, d),
    }
}

/// Compute one 128-bit FMA lane block. `d` is the dst lane (its upper bits are
/// preserved in the scalar form).
fn fma_lanes(a: u128, b: u128, c: u128, kind: FmaKind, double: bool, packed: bool, d: u128) -> u128 {
    if double {
        let l = |i: u32| -> u128 {
            fma_one_f64(
                f64::from_bits((a >> (i * 64)) as u64),
                f64::from_bits((b >> (i * 64)) as u64),
                f64::from_bits((c >> (i * 64)) as u64),
                kind,
                i,
            ) as u128
        };
        if packed {
            l(0) | (l(1) << 64)
        } else {
            (d & (u128::MAX << 64)) | l(0)
        }
    } else if packed {
        let mut out = 0u128;
        for i in 0..4 {
            out |= (fma_one_f32(
                f32::from_bits((a >> (i * 32)) as u32),
                f32::from_bits((b >> (i * 32)) as u32),
                f32::from_bits((c >> (i * 32)) as u32),
                kind,
                i,
            ) as u128)
                << (i * 32);
        }
        out
    } else {
        let r0 = fma_one_f32(
            f32::from_bits(a as u32),
            f32::from_bits(b as u32),
            f32::from_bits(c as u32),
            kind,
            0,
        ) as u128;
        (d & !0xffff_ffff_u128) | r0
    }
}

/// SSE4.2 string-compare core (PCMPxSTRx). Returns IntRes2 (the per-element
/// comparison bitmask after aggregation + polarity). `len1`/`len2` are the valid
/// element counts of operand1/operand2; `num_elem` is 8 (word) or 16 (byte).
fn pcmpstr_intres2(op1: u128, op2: u128, imm: u8, len1: u32, len2: u32, num_elem: u32) -> u32 {
    let word = imm & 1 != 0;
    let signed = imm & 2 != 0;
    let bits = if word { 16 } else { 8 };
    let mask: u128 = (1u128 << bits) - 1;
    let get = |v: u128, i: u32| (v >> (i * bits)) & mask;
    let sval = |v: u128| -> i64 {
        if signed && v & (1u128 << (bits - 1)) != 0 {
            (v | !mask) as i64
        } else {
            v as i64
        }
    };
    let agg = (imm >> 2) & 3;
    let v1 = |j: u32| j < len1;
    let v2 = |i: u32| i < len2;

    let mut r1: u32 = 0;
    for i in 0..num_elem {
        let bit = match agg {
            0 => v2(i) && (0..num_elem).any(|j| v1(j) && get(op1, j) == get(op2, i)), // equal-any
            1 => {
                // ranges: op1 holds (lo,hi) pairs.
                v2(i) && {
                    let x = sval(get(op2, i));
                    (0..num_elem).step_by(2).any(|j| {
                        v1(j) && v1(j + 1) && sval(get(op1, j)) <= x && x <= sval(get(op1, j + 1))
                    })
                }
            }
            2 => match (v1(i), v2(i)) {
                // equal-each
                (true, true) => get(op1, i) == get(op2, i),
                (false, false) => true,
                _ => false,
            },
            _ => {
                // equal-ordered (substring search): scan the needle op1 against the
                // haystack op2 from position i. A needle that ends (!v1) is a match;
                // a haystack that ends (!v2) or a mismatch is a fail; running past the
                // operand ends the scan keeping the current (matched) state.
                let mut res = true;
                let mut j = 0;
                while j < num_elem && i + j < num_elem {
                    if !v1(j) {
                        break;
                    }
                    if !v2(i + j) || get(op1, j) != get(op2, i + j) {
                        res = false;
                        break;
                    }
                    j += 1;
                }
                res
            }
        };
        if bit {
            r1 |= 1 << i;
        }
    }

    let full = (1u32 << num_elem) - 1;
    if imm & 0x10 == 0 {
        r1 // positive
    } else if imm & 0x20 == 0 {
        (!r1) & full // negative
    } else {
        // masked negative: flip only bits with a valid operand2 element.
        let mut r = r1;
        for i in 0..num_elem {
            if v2(i) {
                r ^= 1 << i;
            }
        }
        r & full
    }
}

/// Round half to even (rustc 1.75 has no `f64::round_ties_even`). Non-finite and
/// already-integral values pass through unchanged.
fn round_ties_even_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    let d = x - f;
    let r = if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i64).rem_euclid(2) == 0 {
        f
    } else {
        f + 1.0
    };
    // Preserve the sign of zero (x86: round(-0.4) → -0.0, not +0.0).
    if r == 0.0 && x.is_sign_negative() {
        -0.0
    } else {
        r
    }
}

/// Round a float to an integral value by the SSE4 ROUND imm8 mode: bit2 set → use
/// MXCSR (we model RC = round-nearest); else bits[1:0] = nearest/down/up/trunc.
fn round_mode_f64(x: f64, mode: u8) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if mode & 0x4 != 0 {
        return round_ties_even_f64(x);
    }
    match mode & 0x3 {
        0 => round_ties_even_f64(x),
        1 => x.floor(),
        2 => x.ceil(),
        _ => x.trunc(),
    }
}

/// Convert a float to i32 for the packed CVT ops: truncate or round-to-nearest-
/// even (the MXCSR default), with NaN / out-of-range → the integer indefinite
/// (0x8000_0000), matching x86.
fn f64_to_i32(x: f64, trunc: bool) -> i32 {
    if x.is_nan() {
        return i32::MIN;
    }
    let r = if trunc {
        x.trunc()
    } else {
        round_ties_even_f64(x)
    };
    if r >= -2_147_483_648.0 && r < 2_147_483_648.0 {
        r as i32
    } else {
        i32::MIN
    }
}

/// Apply an x87 binary arithmetic op in true 80-bit extended precision (R variants reverse
/// the operands), honouring the FCW precision-control (`pc`) and rounding-control (`rc`).
fn fpu_apply80(op: FpuBin, a: F80, b: F80, pc: u8, rc: u8) -> F80 {
    match op {
        FpuBin::Add => a.add(b, pc, rc),
        FpuBin::Sub => a.sub(b, pc, rc),
        FpuBin::SubR => b.sub(a, pc, rc),
        FpuBin::Mul => a.mul(b, pc, rc),
        FpuBin::Div => a.div(b, pc, rc),
        FpuBin::DivR => b.div(a, pc, rc),
    }
}

fn f64_to_int(f: f64, bits: u32) -> u64 {
    let indefinite = 1_u64 << (bits - 1);
    if f.is_nan() {
        return indefinite;
    }
    let t = f.trunc();
    if bits == 64 {
        if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&t) {
            t as i64 as u64
        } else {
            indefinite
        }
    } else if (-2_147_483_648.0..2_147_483_648.0).contains(&t) {
        (t as i32) as u32 as u64
    } else {
        indefinite
    }
}

/// Lane-wise wrapping packed integer add/sub over a 128-bit XMM value.
/// `elem` is the lane width in bytes (1/2/4/8); carries/borrows do not cross
/// lane boundaries (each lane is independent two's-complement modular arithmetic).
/// One 128-bit lane of FP binary arithmetic (ADD/SUB/MUL/DIV/MIN/MAX). `a` is
/// src1 (it supplies the preserved upper bits in scalar form), `b` is src2.
/// Shared by the legacy SSE `FBin` and the VEX `VFBin` execute paths so they are
/// guaranteed bit-identical. MIN/MAX return an operand verbatim (x86 picks src2
/// on NaN/equality); arithmetic canonicalizes NaN via `fp_result_*`.
fn fbin_lanes(a: u128, b: u128, kind: FBinKind, double: bool, packed: bool, mxcsr: u32) -> (u128, u32) {
    let rc = ((mxcsr >> 13) & 3) as u8;
    let daz = mxcsr & (1 << 6) != 0;
    let ftz = mxcsr & (1 << 15) != 0;
    let lane64 = |x: f64, y: f64| -> (u64, u32) {
        let mut e = 0u32;
        // DE fires for a denormal source unless the result is determined without normalizing
        // it: a NaN operand (suppresses #D) or a divide-by-zero. An inf partner does NOT
        // suppress (−inf − denormal still flags DE).
        if !daz
            && !x.is_nan()
            && !y.is_nan()
            && (x.is_subnormal() || y.is_subnormal())
            && !(matches!(kind, FBinKind::Div) && y == 0.0)
        {
            e |= FE_DE;
        }
        let (xd, yd) = (daz_f64(x, daz), daz_f64(y, daz));
        let bits = match kind {
            // MIN/MAX return src2 on NaN/equality; a NaN operand flags IE (they compare).
            FBinKind::Min => {
                if x.is_nan() || y.is_nan() {
                    e |= FE_IE;
                }
                (if xd < yd { xd } else { yd }).to_bits()
            }
            FBinKind::Max => {
                if x.is_nan() || y.is_nan() {
                    e |= FE_IE;
                }
                (if xd > yd { xd } else { yd }).to_bits()
            }
            _ => {
                let (mut bits, be) = fbin_rounded_f64(kind, xd, yd, rc);
                e |= be;
                if ftz && is_subnormal_bits_f64(bits) {
                    e |= FE_UE | FE_PE;
                    bits &= 1 << 63; // flush to sign-preserved zero
                }
                bits
            }
        };
        (bits, e)
    };
    let lane32 = |x: f32, y: f32| -> (u32, u32) {
        let mut e = 0u32;
        // DE fires for a denormal source unless the result is determined without normalizing
        // it: a NaN operand (suppresses #D) or a divide-by-zero. An inf partner does NOT
        // suppress (−inf − denormal still flags DE).
        if !daz
            && !x.is_nan()
            && !y.is_nan()
            && (x.is_subnormal() || y.is_subnormal())
            && !(matches!(kind, FBinKind::Div) && y == 0.0)
        {
            e |= FE_DE;
        }
        let (xd, yd) = (daz_f32(x, daz), daz_f32(y, daz));
        let bits = match kind {
            FBinKind::Min => {
                if x.is_nan() || y.is_nan() {
                    e |= FE_IE;
                }
                (if xd < yd { xd } else { yd }).to_bits()
            }
            FBinKind::Max => {
                if x.is_nan() || y.is_nan() {
                    e |= FE_IE;
                }
                (if xd > yd { xd } else { yd }).to_bits()
            }
            _ => {
                let (mut bits, be) = fbin_rounded_f32(kind, xd, yd, rc);
                e |= be;
                if ftz && is_subnormal_bits_f32(bits) {
                    e |= FE_UE | FE_PE;
                    bits &= 1 << 31;
                }
                bits
            }
        };
        (bits, e)
    };
    let mut exc = 0u32;
    let out = if double {
        let (r0, e0) = lane64(f64::from_bits(a as u64), f64::from_bits(b as u64));
        exc |= e0;
        if packed {
            let (r1, e1) = lane64(f64::from_bits((a >> 64) as u64), f64::from_bits((b >> 64) as u64));
            exc |= e1;
            ((r1 as u128) << 64) | r0 as u128
        } else {
            (a & (u128::MAX << 64)) | r0 as u128
        }
    } else if packed {
        let mut out = 0u128;
        for i in 0..4 {
            let xi = f32::from_bits((a >> (i * 32)) as u32);
            let yi = f32::from_bits((b >> (i * 32)) as u32);
            let (r, e) = lane32(xi, yi);
            exc |= e;
            out |= (r as u128) << (i * 32);
        }
        out
    } else {
        let (r0, e0) = lane32(f32::from_bits(a as u32), f32::from_bits(b as u32));
        exc |= e0;
        (a & !0xffff_ffff_u128) | r0 as u128
    };
    (out, exc)
}

fn packed_arith(a: u128, b: u128, elem: u8, sub: bool) -> u128 {
    let bits = elem as u32 * 8;
    let lane_mask: u128 = (1_u128 << bits) - 1;
    let lanes = 128 / bits;
    let mut out = 0_u128;
    for i in 0..lanes {
        let shift = i * bits;
        let av = (a >> shift) & lane_mask;
        let bv = (b >> shift) & lane_mask;
        let rv = if sub {
            av.wrapping_sub(bv) & lane_mask
        } else {
            av.wrapping_add(bv) & lane_mask
        };
        out |= rv << shift;
    }
    out
}

/// Lane-wise packed min/max. `elem` = lane bytes (1 = bytes, 2 = words); `signed`
/// selects two's-complement ordering (PMINSW/PMAXSW) vs unsigned (PMINUB/PMAXUB).
fn packed_minmax(a: u128, b: u128, elem: u8, max: bool, signed: bool) -> u128 {
    let bits = elem as u32 * 8;
    let lane_mask: u128 = (1_u128 << bits) - 1;
    let sign = 1_u128 << (bits - 1);
    let lanes = 128 / bits;
    let mut out = 0_u128;
    for i in 0..lanes {
        let shift = i * bits;
        let av = (a >> shift) & lane_mask;
        let bv = (b >> shift) & lane_mask;
        // For signed lanes, flip the sign bit so an unsigned compare orders them.
        let (ak, bk) = if signed {
            (av ^ sign, bv ^ sign)
        } else {
            (av, bv)
        };
        let pick_a = if max { ak >= bk } else { ak <= bk };
        out |= (if pick_a { av } else { bv }) << shift;
    }
    out
}

/// Lane-wise packed compare: equality, or signed greater-than. Each lane is set
/// to all-ones on a true result, all-zero otherwise. `elem` = lane bytes (1/2/4).
fn packed_compare(a: u128, b: u128, elem: u8, gt: bool) -> u128 {
    let bits = elem as u32 * 8;
    let lane_mask: u128 = (1_u128 << bits) - 1;
    let sign = 1_u128 << (bits - 1);
    let lanes = 128 / bits;
    let mut out = 0_u128;
    for i in 0..lanes {
        let shift = i * bits;
        let av = (a >> shift) & lane_mask;
        let bv = (b >> shift) & lane_mask;
        let set = if gt {
            // Signed compare: flip the sign bit and compare as unsigned.
            (av ^ sign) > (bv ^ sign)
        } else {
            av == bv
        };
        if set {
            out |= lane_mask << shift;
        }
    }
    out
}

/// Lane-wise packed shift. `count` ≥ lane width yields 0 (logical) or a sign-fill
/// (arithmetic). `elem` = lane bytes (2/4/8); PSRA only valid for 2/4.
fn packed_shift(a: u128, count: u64, elem: u8, kind: ShiftKind) -> u128 {
    let bits = elem as u32 * 8;
    let lane_mask: u128 = (1_u128 << bits) - 1;
    let sign = 1_u128 << (bits - 1);
    let lanes = 128 / bits;
    let mut out = 0_u128;
    for i in 0..lanes {
        let shift = i * bits;
        let v = (a >> shift) & lane_mask;
        let rv = if count >= bits as u64 {
            match kind {
                ShiftKind::Sra if v & sign != 0 => lane_mask, // sign-fill all ones
                _ => 0,
            }
        } else {
            let c = count as u32;
            match kind {
                ShiftKind::Sll => (v << c) & lane_mask,
                ShiftKind::Srl => v >> c,
                ShiftKind::Sra => {
                    let shifted = v >> c;
                    if v & sign != 0 {
                        // Fill the vacated high bits with ones.
                        shifted | (lane_mask & !(lane_mask >> c))
                    } else {
                        shifted
                    }
                }
            }
        };
        out |= (rv & lane_mask) << shift;
    }
    out
}

/// PUNPCK: interleave the low (or high) halves of two 128-bit values at `elem`
/// granularity, taking lanes alternately from `a` (dst) then `b` (src).
fn packed_unpack(a: u128, b: u128, elem: u8, high: bool) -> u128 {
    let bits = elem as u32 * 8;
    let lane_mask: u128 = if bits == 128 { u128::MAX } else { (1_u128 << bits) - 1 };
    let lanes = 128 / bits;
    let out_pairs = lanes / 2;
    let base = if high { lanes / 2 } else { 0 };
    let mut out = 0_u128;
    for i in 0..out_pairs {
        let src_lane = base + i;
        let al = (a >> (src_lane * bits)) & lane_mask;
        let bl = (b >> (src_lane * bits)) & lane_mask;
        out |= al << ((2 * i) * bits);
        out |= bl << ((2 * i + 1) * bits);
    }
    out
}

// ---- VEX vector helpers (128-bit lane block; callers apply per-128 for 256). ----

/// SHUFPS/SHUFPD select (imm8). `double`=SHUFPD.
fn shuffle_fp(a: u128, b: u128, double: bool, imm: u8) -> u128 {
    if double {
        let lo = if imm & 1 != 0 { (a >> 64) as u64 } else { a as u64 };
        let hi = if imm & 2 != 0 { (b >> 64) as u64 } else { b as u64 };
        ((hi as u128) << 64) | lo as u128
    } else {
        let dw = |v: u128, sel: u8| (v >> ((sel & 3) as u32 * 32)) as u32 as u128;
        dw(a, imm) | (dw(a, imm >> 2) << 32) | (dw(b, imm >> 4) << 64) | (dw(b, imm >> 6) << 96)
    }
}

/// VPSHUFD (`high=None`) full-dword shuffle of one source; or VPSHUFLW
/// (`high=Some(false)`) / VPSHUFHW (`high=Some(true)`) word shuffle of one half.
fn pshuf_lanes(src: u128, imm: u8, high: Option<bool>) -> u128 {
    match high {
        None => {
            let mut r = 0u128;
            for i in 0..4 {
                let sel = (imm >> (i * 2)) & 3;
                r |= ((src >> (sel as u32 * 32)) & 0xffff_ffff) << (i * 32);
            }
            r
        }
        Some(hi) => {
            let base = if hi { 4u32 } else { 0 };
            let mut r = src;
            // clear the 4 words being shuffled
            for i in 0..4 {
                r &= !(0xffffu128 << ((base + i) * 16));
            }
            for i in 0..4 {
                let sel = ((imm >> (i * 2)) & 3) as u32;
                let w = (src >> ((base + sel) * 16)) & 0xffff;
                r |= w << ((base + i) * 16);
            }
            r
        }
    }
}

/// The 32 AVX FP compare predicates collapse to 16 boolean relations (the
/// signaling/non-signaling 16-31 give the same result as 0-15).
fn cmp_rel(pred: u8, lt: bool, le: bool, eq: bool, gt: bool, ge: bool, un: bool) -> bool {
    match pred & 0xf {
        0 => eq,             // EQ_OQ
        1 => lt,             // LT_OS
        2 => le,             // LE_OS
        3 => un,             // UNORD_Q
        4 => !eq,            // NEQ_UQ (true on NaN)
        5 => !lt,            // NLT_US
        6 => !le,            // NLE_US
        7 => !un,            // ORD_Q
        8 => eq || un,       // EQ_UQ
        9 => !ge,            // NGE_US
        10 => !gt,           // NGT_US
        11 => false,         // FALSE_OQ
        12 => !eq && !un,    // NEQ_OQ (ordered)
        13 => ge,            // GE_OS
        14 => gt,            // GT_OS
        _ => true,           // 15 TRUE_UQ
    }
}

/// VCMPPS/PD/SS/SD: per-lane FP compare with a 5-bit predicate; true→all-ones.
fn cmp_fp(a: u128, b: u128, double: bool, scalar: bool, pred: u8) -> u128 {
    let t64 = |x: f64, y: f64| -> bool {
        cmp_rel(pred, x < y, x <= y, x == y, x > y, x >= y, x.is_nan() || y.is_nan())
    };
    let t32 = |x: f32, y: f32| -> bool {
        cmp_rel(pred, x < y, x <= y, x == y, x > y, x >= y, x.is_nan() || y.is_nan())
    };
    let m64 = |t: bool| if t { u64::MAX as u128 } else { 0 };
    let m32 = |t: bool| if t { u32::MAX as u128 } else { 0 };
    if double {
        let r0 = m64(t64(f64::from_bits(a as u64), f64::from_bits(b as u64)));
        if scalar {
            (a & (u128::MAX << 64)) | r0
        } else {
            (m64(t64(f64::from_bits((a >> 64) as u64), f64::from_bits((b >> 64) as u64))) << 64) | r0
        }
    } else if scalar {
        let r0 = m32(t32(f32::from_bits(a as u32), f32::from_bits(b as u32)));
        (a & !0xffff_ffff_u128) | r0
    } else {
        let mut out = 0u128;
        for i in 0..4 {
            out |= m32(t32(f32::from_bits((a >> (i * 32)) as u32), f32::from_bits((b >> (i * 32)) as u32))) << (i * 32);
        }
        out
    }
}

/// BLENDPS/PD/PBLENDW immediate blend; bit i picks src2 for lane i.
fn blend_imm(a: u128, b: u128, elem: u8, imm: u8) -> u128 {
    let bits = elem as u32 * 8;
    let mask: u128 = (1u128 << bits) - 1;
    let mut out = 0u128;
    for i in 0..(128 / bits) {
        let v = if (imm >> i) & 1 != 0 { b } else { a };
        out |= ((v >> (i * bits)) & mask) << (i * bits);
    }
    out
}

/// BLENDV variable blend; lane i picks src2 if the sign bit of `ctrl` lane i is set.
fn blendv_lanes(a: u128, b: u128, ctrl: u128, elem: u8) -> u128 {
    let bits = elem as u32 * 8;
    let mask: u128 = (1u128 << bits) - 1;
    let sign = 1u128 << (bits - 1);
    let mut out = 0u128;
    for i in 0..(128 / bits) {
        let v = if (ctrl >> (i * bits)) & sign != 0 { b } else { a };
        out |= ((v >> (i * bits)) & mask) << (i * bits);
    }
    out
}

/// PACKSSWB/PACKSSDW/PACKUSWB/PACKUSDW saturating pack (low halves from `a`).
fn pack_lanes(a: u128, b: u128, dword_src: bool, signed: bool) -> u128 {
    let mut out = 0u128;
    if dword_src {
        let sat = |d: i32| -> u128 {
            if signed { (d.clamp(-32768, 32767) as i16 as u16) as u128 } else { (d.clamp(0, 65535) as u16) as u128 }
        };
        for i in 0..4 {
            out |= sat((a >> (i * 32)) as i32) << (i * 16);
            out |= sat((b >> (i * 32)) as i32) << ((i + 4) * 16);
        }
    } else {
        let sat = |w: i16| -> u128 {
            if signed { (w.clamp(-128, 127) as i8 as u8) as u128 } else { (w.clamp(0, 255) as u8) as u128 }
        };
        for i in 0..8 {
            out |= sat((a >> (i * 16)) as i16) << (i * 8);
            out |= sat((b >> (i * 16)) as i16) << ((i + 8) * 8);
        }
    }
    out
}

/// PALIGNR: byte i of result = byte (i+imm) of the concat [b(low16) : a(high16)].
fn palignr_lanes(a: u128, b: u128, imm: u8) -> u128 {
    let mut out = 0u128;
    for i in 0..16u32 {
        let idx = i + imm as u32;
        let byte: u128 = if idx >= 32 {
            0
        } else if idx >= 16 {
            (a >> ((idx - 16) * 8)) & 0xff
        } else {
            (b >> (idx * 8)) & 0xff
        };
        out |= byte << (i * 8);
    }
    out
}

/// PSHUFB: result byte i = 0 if ctrl[i] sign set, else data byte (ctrl[i] & 0x0f).
fn pshufb_lanes(data: u128, ctrl: u128) -> u128 {
    let mut out = 0u128;
    for i in 0..16u32 {
        let c = (ctrl >> (i * 8)) as u8;
        if c & 0x80 == 0 {
            out |= ((data >> ((c & 0x0f) as u32 * 8)) & 0xff) << (i * 8);
        }
    }
    out
}

/// PMULLW/HW/HUW/UDQ/MADDWD (word/dword multiplies).
fn pmul_lane(kind: PMulKind, a: u128, b: u128) -> u128 {
    match kind {
        PMulKind::Uludq => {
            let r0 = (a as u32 as u64).wrapping_mul(b as u32 as u64);
            let r1 = ((a >> 64) as u32 as u64).wrapping_mul((b >> 64) as u32 as u64);
            (r0 as u128) | ((r1 as u128) << 64)
        }
        PMulKind::Maddwd => {
            let mut out = 0u128;
            for j in 0..4u32 {
                let w = |v: u128, k: u32| ((v >> (k * 16)) as u16) as i16 as i32;
                let s = w(a, 2 * j)
                    .wrapping_mul(w(b, 2 * j))
                    .wrapping_add(w(a, 2 * j + 1).wrapping_mul(w(b, 2 * j + 1)));
                out |= ((s as u32) as u128) << (j * 32);
            }
            out
        }
        _ => {
            let mut out = 0u128;
            for i in 0..8u32 {
                let aw = (a >> (i * 16)) as u16;
                let bw = (b >> (i * 16)) as u16;
                let r: u16 = match kind {
                    PMulKind::Lw => aw.wrapping_mul(bw),
                    PMulKind::Hw => (((aw as i16 as i32) * (bw as i16 as i32)) >> 16) as u16,
                    PMulKind::Huw => (((aw as u32) * (bw as u32)) >> 16) as u16,
                    _ => 0,
                };
                out |= (r as u128) << (i * 16);
            }
            out
        }
    }
}

/// PMULLD (32×32→low32 per dword).
fn pmulld_lane(a: u128, b: u128) -> u128 {
    let mut out = 0u128;
    for i in 0..4 {
        let r = ((a >> (i * 32)) as u32).wrapping_mul((b >> (i * 32)) as u32);
        out |= (r as u128) << (i * 32);
    }
    out
}

/// PMULDQ (signed 32×32→64 of the low dword of each qword lane).
fn pmuldq_lane(a: u128, b: u128) -> u128 {
    let r0 = (a as u32 as i32 as i64).wrapping_mul(b as u32 as i32 as i64) as u64;
    let r1 = ((a >> 64) as u32 as i32 as i64).wrapping_mul((b >> 64) as u32 as i32 as i64) as u64;
    (r0 as u128) | ((r1 as u128) << 64)
}

/// PMADDUBSW (a=unsigned bytes, b=signed bytes; pairwise → saturated word).
fn pmaddubsw_lane(a: u128, b: u128) -> u128 {
    let mut out = 0u128;
    for j in 0..8u32 {
        let u0 = (a >> (2 * j * 8)) as u8 as i32;
        let s0 = (b >> (2 * j * 8)) as u8 as i8 as i32;
        let u1 = (a >> ((2 * j + 1) * 8)) as u8 as i32;
        let s1 = (b >> ((2 * j + 1) * 8)) as u8 as i8 as i32;
        let r = (u0 * s0 + u1 * s1).clamp(-32768, 32767);
        out |= ((r as i16 as u16) as u128) << (j * 16);
    }
    out
}

/// PMULHRSW (signed word multiply, round, take bits [16:1]).
fn pmulhrsw_lane(a: u128, b: u128) -> u128 {
    let mut out = 0u128;
    for i in 0..8u32 {
        let x = (a >> (i * 16)) as u16 as i16 as i32;
        let y = (b >> (i * 16)) as u16 as i16 as i32;
        let r = (((x * y) >> 14) + 1) >> 1;
        out |= ((r as i16 as u16) as u128) << (i * 16);
    }
    out
}

/// One 128-bit lane block of a `VPkind` two-source op. For `Shift`, `b`'s low 64
/// bits are the shift count.
fn vpacked_lane(kind: VPkind, a: u128, b: u128) -> u128 {
    match kind {
        VPkind::CmpEq(e) => packed_compare(a, b, e, false),
        VPkind::CmpGt(e) => packed_compare(a, b, e, true),
        VPkind::MinMax { max, signed, elem } => packed_minmax(a, b, elem, max, signed),
        VPkind::Unpack { high, elem } => packed_unpack(a, b, elem, high),
        VPkind::Shift { kind, elem } => packed_shift(a, b as u64, elem, kind),
        VPkind::Pack { dword_src, signed } => pack_lanes(a, b, dword_src, signed),
        VPkind::Pshufb => pshufb_lanes(a, b),
        VPkind::Mul(k) => pmul_lane(k, a, b),
        VPkind::Mulld => pmulld_lane(a, b),
        VPkind::Muldq => pmuldq_lane(a, b),
        VPkind::Maddubsw => pmaddubsw_lane(a, b),
        VPkind::Mulhrsw => pmulhrsw_lane(a, b),
        VPkind::SatArith { sub, signed, elem } => packed_sat_arith(a, b, elem, sub, signed),
        VPkind::Avg { elem } => packed_avg(a, b, elem),
        VPkind::Sadbw => psadbw_lane(a, b),
    }
}

/// Saturating packed add/sub. `signed` → clamp to i8/i16 range; else u8/u16.
fn packed_sat_arith(a: u128, b: u128, elem: u8, sub: bool, signed: bool) -> u128 {
    let bits = elem as u32 * 8;
    let lanes = 16 / elem as usize;
    let mut out: u128 = 0;
    for i in 0..lanes {
        let sh = i as u32 * bits;
        let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
        let av = (a >> sh) & mask;
        let bv = (b >> sh) & mask;
        let r: u128 = if signed {
            let (lo, hi) = if elem == 1 { (i8::MIN as i64, i8::MAX as i64) } else { (i16::MIN as i64, i16::MAX as i64) };
            let sa = sext_lane(av as u64, bits);
            let sb = sext_lane(bv as u64, bits);
            let v = if sub { sa - sb } else { sa + sb };
            (v.clamp(lo, hi) as u128) & mask
        } else {
            let ua = av as i64;
            let ub = bv as i64;
            let hi = if elem == 1 { u8::MAX as i64 } else { u16::MAX as i64 };
            let v = if sub { ua - ub } else { ua + ub };
            (v.clamp(0, hi) as u128) & mask
        };
        out |= r << sh;
    }
    out
}

/// Square-root every FP lane of a 128-bit value (f64×2 or f32×4).
/// One f64 lane of SQRT with DAZ/FTZ + exception flags → (result bits, exceptions).
fn sqrt_scalar_f64(xb: u64, mxcsr: u32) -> (u64, u32) {
    let rc = ((mxcsr >> 13) & 3) as u8;
    let daz = mxcsr & (1 << 6) != 0;
    let ftz = mxcsr & (1 << 15) != 0;
    let x = f64::from_bits(xb);
    let mut e = 0u32;
    if !daz && x.is_subnormal() && x > 0.0 {
        e |= FE_DE; // a NEGATIVE denormal flags IE (invalid sqrt), not DE
    }
    let (mut bits, se) = sqrt_rounded_f64(daz_f64(x, daz), rc);
    e |= se;
    if ftz && is_subnormal_bits_f64(bits) {
        e |= FE_UE | FE_PE;
        bits &= 1 << 63;
    }
    (bits, e)
}
fn sqrt_scalar_f32(xb: u32, mxcsr: u32) -> (u32, u32) {
    let rc = ((mxcsr >> 13) & 3) as u8;
    let daz = mxcsr & (1 << 6) != 0;
    let ftz = mxcsr & (1 << 15) != 0;
    let x = f32::from_bits(xb);
    let mut e = 0u32;
    if !daz && x.is_subnormal() && x > 0.0 {
        e |= FE_DE; // a NEGATIVE denormal flags IE (invalid sqrt), not DE
    }
    let (mut bits, se) = sqrt_rounded_f32(daz_f32(x, daz), rc);
    e |= se;
    if ftz && is_subnormal_bits_f32(bits) {
        e |= FE_UE | FE_PE;
        bits &= 1 << 31;
    }
    (bits, e)
}
fn sqrt_lanes128(b: u128, double: bool, mxcsr: u32) -> (u128, u32) {
    let mut exc = 0u32;
    let out = if double {
        let (r0, e0) = sqrt_scalar_f64(b as u64, mxcsr);
        let (r1, e1) = sqrt_scalar_f64((b >> 64) as u64, mxcsr);
        exc = e0 | e1;
        (r0 as u128) | ((r1 as u128) << 64)
    } else {
        let mut out = 0u128;
        for i in 0..4 {
            let (r, e) = sqrt_scalar_f32((b >> (i * 32)) as u32, mxcsr);
            exc |= e;
            out |= (r as u128) << (i * 32);
        }
        out
    };
    (out, exc)
}

/// Permute one 128-bit lane for VPERMILPS/PD. `lane` (0/1) selects which imm
/// bits the PD imm form consumes (low lane uses bits 0/1, high lane 2/3).
fn permil_lane(data: u128, double: bool, variable: bool, ctrl: u128, lane: u32) -> u128 {
    let mut out = 0u128;
    if double {
        for j in 0..2u32 {
            let sel = if variable {
                ((ctrl >> (j * 64 + 1)) & 1) as u32
            } else {
                ((ctrl >> (2 * lane + j)) & 1) as u32
            };
            let q = (data >> (sel * 64)) & u64::MAX as u128;
            out |= q << (j * 64);
        }
    } else {
        for i in 0..4u32 {
            let sel = if variable {
                ((ctrl >> (i * 32)) & 3) as u32
            } else {
                ((ctrl >> (2 * i)) & 3) as u32
            };
            let dw = (data >> (sel * 32)) & 0xffff_ffff;
            out |= dw << (i * 32);
        }
    }
    out
}

/// Sign-extend the low `bits` of `v` to i64.
fn sext_lane(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

/// Unsigned rounded average `(a+b+1)>>1` per element (byte or word).
fn packed_avg(a: u128, b: u128, elem: u8) -> u128 {
    let bits = elem as u32 * 8;
    let lanes = 16 / elem as usize;
    let mask = (1u128 << bits) - 1;
    let mut out: u128 = 0;
    for i in 0..lanes {
        let sh = i as u32 * bits;
        let av = ((a >> sh) & mask) as u64;
        let bv = ((b >> sh) & mask) as u64;
        let r = ((av + bv + 1) >> 1) as u128 & mask;
        out |= r << sh;
    }
    out
}

/// PSADBW: |a-b| summed over each 8-byte lane, result in low u16 of that lane.
fn psadbw_lane(a: u128, b: u128) -> u128 {
    let mut out: u128 = 0;
    for lane in 0..2 {
        let mut sum: u32 = 0;
        for byte in 0..8 {
            let sh = (lane * 8 + byte) * 8;
            let av = ((a >> sh) & 0xff) as i32;
            let bv = ((b >> sh) & 0xff) as i32;
            sum += (av - bv).unsigned_abs();
        }
        out |= (sum as u128) << (lane * 64);
    }
    out
}

// ---- element converts (used by the 256-bit VCVT width-changing forms) ----
/// 2 packed f32 (low 64 of `x`) → 2 f64 (128 bits).
fn cvt2_f32_to_f64(x: u128) -> u128 {
    ((f32::from_bits(x as u32) as f64).to_bits() as u128)
        | (((f32::from_bits((x >> 32) as u32) as f64).to_bits() as u128) << 64)
}
/// 2 packed i32 (low 64 of `x`) → 2 f64 (128 bits).
fn cvt2_i32_to_f64(x: u128) -> u128 {
    ((x as u32 as i32 as f64).to_bits() as u128)
        | ((((x >> 32) as u32 as i32 as f64).to_bits() as u128) << 64)
}
/// 2 f64 (lo, hi) → 2 packed f32 (in a u64).
fn cvt2_f64_to_f32(lo: u64, hi: u64) -> u64 {
    ((f64::from_bits(lo) as f32).to_bits() as u64)
        | (((f64::from_bits(hi) as f32).to_bits() as u64) << 32)
}
/// 2 f64 (lo, hi) → 2 packed i32 (in a u64).
fn cvt2_f64_to_i32(lo: u64, hi: u64, trunc: bool) -> u64 {
    (f64_to_i32(f64::from_bits(lo), trunc) as u32 as u64)
        | ((f64_to_i32(f64::from_bits(hi), trunc) as u32 as u64) << 32)
}
/// 4 packed i32 → 4 packed f32 (one 128-bit lane).
fn cvt4_i32_to_f32(x: u128) -> u128 {
    let mut out = 0u128;
    for i in 0..4 {
        let f = ((x >> (i * 32)) as u32 as i32) as f32;
        out |= (f.to_bits() as u128) << (i * 32);
    }
    out
}
/// 4 packed f32 → 4 packed i32 (one 128-bit lane).
fn cvt4_f32_to_i32(x: u128, trunc: bool) -> u128 {
    let mut out = 0u128;
    for i in 0..4 {
        let v = f64_to_i32(f32::from_bits((x >> (i * 32)) as u32) as f64, trunc);
        out |= (v as u32 as u128) << (i * 32);
    }
    out
}

/// Replicate the low `elem`-byte element of `v` across a 128-bit lane (or, when
/// `elem >= 16`, return `v` unchanged — the whole 128-bit broadcast unit).
/// Flags for the BMI ops that set them (ANDN/BLSx/BZHI): ZF/SF from the result,
/// CF per the op, OF cleared (AF/PF undefined → cleared deterministically).
fn set_bmi_flags(rflags: &mut u64, r: u64, size: Size, cf: bool) {
    flags::set_flag(rflags, ZF, r == 0);
    flags::set_flag(rflags, SF, (r >> (size.bits() - 1)) & 1 != 0);
    flags::set_flag(rflags, CF, cf);
    for b in [OF, AF, PF] {
        flags::set_flag(rflags, b, false);
    }
}

/// PDEP: deposit the contiguous low bits of `value` into the set positions of `mask`.
fn pdep(value: u64, mask: u64) -> u64 {
    let mut r = 0u64;
    let mut v = value;
    let mut m = mask;
    while m != 0 {
        let low = m & m.wrapping_neg();
        if v & 1 != 0 {
            r |= low;
        }
        v >>= 1;
        m &= m - 1;
    }
    r
}

/// PEXT: gather the bits of `value` at the set positions of `mask` into the low bits.
fn pext(value: u64, mask: u64) -> u64 {
    let mut r = 0u64;
    let mut m = mask;
    let mut i = 0u32;
    while m != 0 {
        let low = m & m.wrapping_neg();
        if value & low != 0 {
            r |= 1u64 << i;
        }
        i += 1;
        m &= m - 1;
    }
    r
}

fn broadcast_elem(v: u128, elem: u8) -> u128 {
    if elem >= 16 {
        return v;
    }
    let bits = elem as u32 * 8;
    let x = v & ((1u128 << bits) - 1);
    let mut out = 0u128;
    for i in 0..(128 / bits) {
        out |= x << (i * bits);
    }
    out
}

// ---- F16C: IEEE half-precision (binary16) ↔ single (binary32). Software, since
//      Rust 1.75 has no f16. NaN/inf/subnormal handled; deterministic. ----

/// half (binary16) → single (binary32) bit pattern.
fn f16_to_f32(h: u16) -> u32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            sign // ±0
        } else {
            // subnormal: leading set bit at position hb → normalize.
            let hb = 31 - mant.leading_zeros();
            let e = hb + 103; // biased f32 exp = (hb - 24) + 127
            let frac = (mant & ((1 << hb) - 1)) << (23 - hb);
            sign | (e << 23) | frac
        }
    } else if exp == 0x1f {
        if mant != 0 {
            sign | 0x7fc0_0000 | (mant << 13) // NaN → quieted f32 NaN (set bit 22)
        } else {
            sign | 0x7f80_0000 // inf
        }
    } else {
        sign | ((exp + 112) << 23) | (mant << 13) // normal: bias 127-15 = 112
    }
}

/// Round `sig` dropping its low `drop` bits, per the F16C imm8 rounding mode
/// (0=nearest-even, 1=toward −∞, 2=toward +∞, 3=truncate). `neg` = result sign.
fn round_drop(sig: u32, drop: u32, mode: u8, neg: bool) -> u32 {
    if drop == 0 {
        return sig;
    }
    let kept = sig >> drop;
    let rem = sig & ((1u32 << drop) - 1);
    let half = 1u32 << (drop - 1);
    let up = match mode & 3 {
        0 => rem > half || (rem == half && (kept & 1) == 1),
        1 => neg && rem != 0,
        2 => !neg && rem != 0,
        _ => false,
    };
    kept + up as u32
}

/// single (binary32) → half (binary16), with the F16C rounding `mode`.
fn f32_to_f16(f: u32, mode: u8) -> u16 {
    let sign = ((f >> 16) & 0x8000) as u16;
    let neg = (f >> 31) & 1 != 0;
    let exp = ((f >> 23) & 0xff) as i32;
    let mant = f & 0x7f_ffff;
    if exp == 0xff {
        if mant != 0 {
            return sign | 0x7e00 | ((mant >> 13) as u16 & 0x3ff); // qNaN
        }
        return sign | 0x7c00; // inf
    }
    if exp == 0 && mant == 0 {
        return sign; // ±0
    }
    // value = m * 2^p, with m's leading set bit at `msb`.
    let (m, p) = if exp == 0 { (mant, -149) } else { (mant | 0x80_0000, exp - 150) };
    let msb = 31 - m.leading_zeros();
    let b = p + msb as i32; // binary exponent of the value
    if b > 15 {
        let inf = sign | 0x7c00;
        let maxf = sign | 0x7bff;
        return match mode & 3 {
            1 => if neg { inf } else { maxf },
            2 => if neg { maxf } else { inf },
            3 => maxf,
            _ => inf,
        };
    }
    if b >= -14 {
        // normal f16: drop so the leading bit lands at bit 10.
        let drop = msb as i32 - 10;
        let r = if drop > 0 { round_drop(m, drop as u32, mode, neg) } else { m << (-drop) as u32 };
        let mut e16 = b;
        let mut r16 = r;
        if r16 >= 0x800 {
            r16 >>= 1;
            e16 += 1;
            if e16 > 15 {
                return sign | 0x7c00;
            }
        }
        return sign | (((e16 + 15) as u16) << 10) | (r16 as u16 & 0x3ff);
    }
    // subnormal f16: m16 = round(value / 2^-24) = round(m * 2^(p+24)).
    let shift = -(p + 24);
    let r = if shift > 0 { round_drop(m, shift as u32, mode, neg) } else { m << (-shift) as u32 };
    if r == 0 {
        sign
    } else if r >= 0x400 {
        sign | (1u16 << 10) // rounded up to the smallest normal
    } else {
        sign | (r as u16 & 0x3ff)
    }
}

/// Sign-bit mask of a 128-bit FP lane block: 2 bits (PD) or 4 bits (PS).
fn movmsk_of(x: u128, double: bool) -> u64 {
    if double {
        (((x >> 63) & 1) | (((x >> 127) & 1) << 1)) as u64
    } else {
        let mut m = 0u64;
        for i in 0..4 {
            m |= (((x >> (31 + i * 32)) & 1) as u64) << i;
        }
        m
    }
}

/// Whether an error is an x86 *fault* (restarted at the faulting instruction)
/// rather than a trap (reported at the following instruction). All the exceptions
/// we vector are faults: #PF, #GP, #DE, and #UD (unimplemented/illegal encodings).
fn is_restartable_fault(e: &X86Error) -> bool {
    matches!(
        e,
        X86Error::PageFault { .. }
            | X86Error::GeneralProtection
            | X86Error::DivideError
            | X86Error::Unsupported(_)
            | X86Error::InvalidEncoding(_)
    )
}

/// The destination XMM register id of an SSE instruction.
fn xmm_dst(insn: &DecodedInsn) -> Result<usize, X86Error> {
    match insn.dst {
        Some(Operand::Xmm { id }) => Ok(id),
        _ => Err(X86Error::InvalidEncoding("SSE op without XMM dest".into())),
    }
}

/// Extract the register index of an `Operand::Xmm`.
fn xmm_id(op: &Operand) -> Result<usize, X86Error> {
    match op {
        Operand::Xmm { id } => Ok(*id),
        _ => Err(X86Error::InvalidEncoding("expected XMM operand".into())),
    }
}

fn operand_size(op: &Operand) -> Size {
    match op {
        Operand::Reg { size, .. } | Operand::Mem { size, .. } | Operand::Imm { size, .. } => *size,
        Operand::Xmm { .. } => Size::Qword,
    }
}

fn condition(rflags: u64, cond: Cond) -> bool {
    let cf = flags::get_flag(rflags, CF);
    let pf = flags::get_flag(rflags, PF);
    let zf = flags::get_flag(rflags, ZF);
    let sf = flags::get_flag(rflags, SF);
    let of = flags::get_flag(rflags, OF);
    match cond {
        Cond::O => of,
        Cond::No => !of,
        Cond::B => cf,
        Cond::Ae => !cf,
        Cond::E => zf,
        Cond::Ne => !zf,
        Cond::Be => cf || zf,
        Cond::A => !cf && !zf,
        Cond::S => sf,
        Cond::Ns => !sf,
        Cond::P => pf,
        Cond::Np => !pf,
        Cond::L => sf != of,
        Cond::Ge => sf == of,
        Cond::Le => zf || (sf != of),
        Cond::G => !zf && (sf == of),
    }
}

/// Why a run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X86Exit {
    Halted,
    InstructionLimit,
}

/// An x86-64 guest running behind the deterministic core (virtual time only;
/// interrupt injection arrives with the interrupt-controller phase).
#[derive(Debug, Clone)]
pub struct X86Machine {
    core: DeterministicCore,
    /// The vCPU contexts (SMP). `current` indexes the one the scheduler is running.
    /// At `vcpus == 1` this is a single-element Vec and `current` is always 0, so the
    /// whole machine is behavior- and digest-identical to the pre-SMP single-CPU model.
    cpus: Vec<X86Cpu>,
    current: usize,
    memory: GuestMemory,
    io: IoBus,
    /// When true, the seed-derived `DeterministicCore` timer (the toy core's
    /// background `TimerInterrupt`, vector 32) is injected as an x86 external
    /// interrupt. A real x86 guest takes time only from modeled platform timers
    /// (LAPIC, and later PIT/HPET/RTC), so this defaults to `false` and is
    /// enabled only by interrupt-mechanics tests via [`set_core_timer_injection`].
    inject_core_timer: bool,
    /// Profiling (SF_PROFILE): when on, accumulate rdtsc cycles for the three
    /// per-instruction phases [cpu.step, core.step, check_timer]. Diagnostic only.
    prof_on: bool,
    prof: [u64; 3],
    /// P4 seed-exploration: max bounded jitter (instructions) added to the PIT
    /// scheduler-tick delivery instant. 0 = OFF (no stream drawn, baseline run is
    /// byte-identical). Non-zero makes the seed STEER thread interleavings.
    sched_jitter: u64,
    /// SMP Stage-1 seed-driven interleaving exploration: the maximum scheduling QUANTUM
    /// (instructions a vCPU runs before the scheduler may switch). 0 = OFF → the scheduler
    /// falls back to the fixed round-robin `pick_next_vcpu`, so every existing golden
    /// (incl. the byte-identical `--smp 2` boot) is unchanged and no stream is drawn.
    /// Non-zero: at each epoch the seed draws a quantum in `1..=smp_sched` and then a next
    /// runnable vCPU uniformly — so different seeds realise different (each reproducible)
    /// legal interleavings, exposing true SMP data races. This is also the substrate PCT
    /// (Stage 3) will reuse: priorities + seeded change-points layer straight on top.
    smp_sched: u64,
    /// Instructions left in the current vCPU's quantum before the scheduler may switch
    /// (seed-driven mode only). Reset to 0 whenever contention drops below 2 runnable
    /// vCPUs so a fresh quantum is drawn when concurrency resumes.
    smp_quantum_remaining: u64,
    /// Dedicated seed-derived entropy for the SMP scheduler (quantum length + next-vCPU
    /// choice). A separate stream so scheduling never couples to guest / device / jitter
    /// entropy. Only advanced when `smp_sched != 0`; folded into the digest only then, so
    /// the baseline (off) run is byte-for-byte unchanged.
    smp_sched_rng: StreamRng,
    /// SMP Stage-3 PCT (Probabilistic Concurrency Testing): the target bug *depth* `d`.
    /// 0 = OFF (the scheduler falls back to Stage-1/round-robin, byte-identical). When set,
    /// the scheduler runs the highest-priority runnable vCPU (`pct_prio`) and, at each of the
    /// `d-1` seed-drawn `pct_change_points`, demotes the running vCPU below all others —
    /// realising a schedule that provably hits any depth-`d` bug with probability ≥
    /// 1/(n·k^(d-1)) per seed. All seed-derived, so each schedule is reproducible.
    pct_depth: u64,
    /// Per-vCPU PCT priority (index = vCPU). Higher runs first; demotions go strictly below
    /// every initial priority. Empty/ignored when `pct_depth == 0`.
    pct_prio: Vec<i64>,
    /// Sorted instruction-count thresholds (`d-1` of them) at which the running vCPU is
    /// demoted — the PCT priority-change points.
    pct_change_points: Vec<u64>,
    /// Index of the next not-yet-fired change point.
    pct_next_cp: usize,
    /// The priority the next demotion assigns (starts at `d-1`, decreases) — so earlier
    /// change points leave the demoted vCPU above later-demoted ones (paper's ordering).
    pct_low: i64,
    /// Dedicated seed-derived entropy for PCT (priority permutation + change points). Folded
    /// into the digest only when `pct_depth != 0`, so the baseline is byte-for-byte unchanged.
    pct_rng: StreamRng,
}

/// Host timestamp counter for the diagnostic phase profiler (SF_PROFILE). Not
/// part of guest state and never affects determinism — profiling-only.
#[inline(always)]
fn profile_rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: rdtsc is always available on x86-64 and has no memory effects.
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// A full-machine snapshot (see [`X86Machine::snapshot`]). Boxed so the snapshot
/// doesn't bloat the live machine's stack/struct; holds the captured instruction
/// count and state digest for cheap identity checks without a restore.
#[derive(Debug, Clone)]
pub struct X86Snapshot {
    machine: Box<X86Machine>,
    pub instruction_count: u64,
    pub digest: u64,
}

impl X86Snapshot {
    /// Reconstitute a live machine from the snapshot. The returned machine is a
    /// bit-identical deep copy; running it forward reproduces exactly what the
    /// original would have done from this point (determinism ⇒ no replay log).
    pub fn restore(&self) -> X86Machine {
        (*self.machine).clone()
    }
}

impl X86Machine {
    pub fn new(
        seed: Seed256,
        ram_bytes: usize,
        program: &[u8],
        load_addr: u64,
    ) -> Result<Self, X86Error> {
        Self::new_with_config(seed, ram_bytes, program, load_addr, ConfigIdentity::default())
    }

    /// Like [`new`], but attaches a guest-visible [`ConfigIdentity`] (kernel /
    /// cmdline / disk hashes) to the run identity. The boot constructors use this
    /// so two runs with different kernels or command lines have distinct identity.
    fn new_with_config(
        seed: Seed256,
        ram_bytes: usize,
        program: &[u8],
        load_addr: u64,
        config: ConfigIdentity,
    ) -> Result<Self, X86Error> {
        let machine = MachineSpec {
            memory_bytes: ram_bytes as u64,
            ..MachineSpec::x86_64_single_vcpu()
        };
        let identity = RunIdentity::new(RunMode::Pure, seed, machine).with_config(ConfigIdentity {
            hypervisor_build_hash: state::hypervisor_build_hash(),
            ..config
        });
        let mut memory = GuestMemory::new(ram_bytes);
        memory.load(load_addr, program)?;

        let mut cpu = X86Cpu::new();
        cpu.regs.rip = load_addr;
        // The bare/test machine runs flat 64-bit code, so mark it long mode (the boot constructor
        // overwrites cr0/efer with the real long-mode state). Without this the default cr0=0 would
        // read as real mode and apply segment bases to the flat programs.
        cpu.sys.efer |= EFER_LMA;
        // Long-mode flat: latch the CS cache to base 0 / 64-bit code (the default `code_bits`
        // is 16 for a reset AP; the BSP of a bare/test machine executes flat 64-bit).
        cpu.code_bits = 64;
        // Stack at the top of RAM (first push lands fully in bounds).
        cpu.regs.gpr[RSP] = ram_bytes as u64;
        cpu.rdrand_rng = identity.seed().derive("rdrand"); // RDRAND/RDSEED entropy
        cpu.rdrand_origin = cpu.rdrand_rng.state();
        cpu.sb_rng = identity.seed().derive("sb-cpu-0"); // TSO store-buffer drain entropy (BSP)

        // Machine-level shared devices; the virtio-rng entropy stream is seed-derived.
        let mut io = IoBus::new();
        io.virtio_rng.reseed(identity.seed());

        Ok(Self {
            core: DeterministicCore::new(identity),
            cpus: vec![cpu],
            current: 0,
            memory,
            io,
            inject_core_timer: false,
            prof_on: false,
            prof: [0; 3],
            sched_jitter: 0,
            smp_sched: 0,
            smp_quantum_remaining: 0,
            smp_sched_rng: seed.derive("smp-sched"),
            pct_depth: 0,
            pct_prio: Vec::new(),
            pct_change_points: Vec::new(),
            pct_next_cp: 0,
            pct_low: 0,
            pct_rng: seed.derive("pct"),
        })
    }

    /// Bring the machine up to `n` vCPUs by attaching secondary CPUs (APs), each **parked waiting
    /// for its Startup-IPI**. The BSP is index 0; AP `i` gets APIC ID `i`, its own local APIC, and
    /// a distinct seed-derived entropy stream. Guest RAM and the IOAPIC/virtio devices are already
    /// machine-level (shared). `n <= 1` (or ≤ the current count) is a no-op, so `set_vcpus(1)`
    /// leaves the single-CPU machine byte-identical.
    pub fn set_vcpus(&mut self, n: usize) {
        let seed = *self.core.identity().seed();
        while self.cpus.len() < n {
            let id = self.cpus.len() as u32;
            let mut ap = X86Cpu::new();
            ap.apic_id = id;
            ap.lapic.id = id;
            ap.wait_sipi = true; // not scheduled until a SIPI arrives
            ap.halted = true;
            ap.rdrand_rng = seed.derive(&format!("rdrand-cpu-{id}"));
            ap.rdrand_origin = ap.rdrand_rng.state();
            ap.sb_rng = seed.derive(&format!("sb-cpu-{id}"));
            ap.tso_max = self.cpus[0].tso_max; // inherit the BSP's TSO setting
            self.cpus.push(ap);
        }
    }

    /// Enable the per-phase rdtsc profiler (diagnostic only; perturbs timing).
    pub fn set_profile(&mut self, on: bool) {
        self.prof_on = on;
    }

    /// Accumulated profiler cycles: [cpu_step, core_step, check_timer].
    pub fn profile_cycles(&self) -> [u64; 3] {
        self.prof
    }

    /// P4 seed-exploration control: enable bounded seed-driven jitter (in
    /// instructions) on the PIT scheduler-tick delivery instant. With jitter > 0,
    /// different seeds shift preemption points → different thread interleavings →
    /// a spread of per-seed-deterministic outcomes. 0 disables it (baseline).
    pub fn set_sched_jitter(&mut self, max: u64) {
        self.sched_jitter = max;
    }

    /// SMP Stage-1 control: enable seed-driven interleaving exploration by setting the max
    /// scheduling quantum (instructions a vCPU runs before the scheduler may switch). 0
    /// disables it → fixed round-robin, byte-identical baseline. A small quantum (e.g. 1-50)
    /// yields fine-grained interleavings that expose data races; a large one yields coarser,
    /// lock-holding-friendly schedules. Different seeds realise different reproducible schedules.
    pub fn set_smp_sched(&mut self, max_quantum: u64) {
        self.smp_sched = max_quantum;
    }

    /// SMP Stage-3 PCT control: enable Probabilistic Concurrency Testing for target bug depth
    /// `depth` (≥1), with change points drawn uniformly over `[1, k_bound]` instruction counts
    /// (`k_bound` ≈ the run's length in instructions). `depth == 0` disables it (baseline).
    ///
    /// Each vCPU gets a distinct seed-permuted priority ≥ `depth`; the scheduler always runs the
    /// highest-priority runnable vCPU, and at each of the `depth-1` seed-drawn change points it
    /// demotes the running vCPU below all others (to `depth-1`, then `depth-2`, …). A depth-`d`
    /// bug is hit with probability ≥ 1/(n·k^(d-1)) per seed — and every schedule is reproducible.
    /// `depth == 1` (no change points) is pure priority scheduling: each vCPU runs to completion
    /// before the next, i.e. a fully serialized (race-free) execution.
    pub fn set_pct(&mut self, depth: u64, k_bound: u64) {
        self.pct_depth = depth;
        if depth == 0 {
            return;
        }
        let n = self.cpus.len();
        // Distinct initial priorities depth..depth+n-1, then a seed-driven Fisher-Yates shuffle.
        self.pct_prio = (0..n).map(|i| depth as i64 + i as i64).collect();
        for i in (1..n).rev() {
            let j = self.pct_rng.next_bounded((i + 1) as u64) as usize;
            self.pct_prio.swap(i, j);
        }
        self.pct_low = depth as i64 - 1; // first demotion sits just below every initial priority
        self.pct_next_cp = 0;
        // Draw `depth-1` change points uniformly in [1, k_bound], sorted ascending.
        let bound = k_bound.max(1);
        let mut cps: Vec<u64> = (0..depth.saturating_sub(1))
            .map(|_| 1 + self.pct_rng.next_bounded(bound))
            .collect();
        cps.sort_unstable();
        self.pct_change_points = cps;
    }

    /// SMP Stage-2 control: enable the x86-TSO store-buffer model with a max seed-drawn
    /// store-lingering window (instructions). 0 = OFF → stores write through immediately
    /// (Sequential Consistency, the byte-identical baseline). Non-zero: each core's stores are
    /// buffered and become globally visible after a seed-drawn delay, so store→load reordering
    /// (e.g. both cores reading stale in the Store-Buffering / Dekker litmus) becomes reachable
    /// — reproducibly per seed. Applies to every vCPU (they share the setting).
    pub fn set_tso(&mut self, max_linger: u64) {
        for cpu in self.cpus.iter_mut() {
            cpu.tso_max = max_linger;
        }
    }

    /// The identity this run will be compared under (mode/seed/machine + config).
    pub fn identity_digest(&self) -> u64 {
        self.core.identity().digest()
    }

    /// Enable/disable injection of the seed-derived core timer as an x86
    /// interrupt. Off by default (a real platform uses modeled timers); enabled
    /// by tests that exercise the IDT delivery path with a synthetic timer.
    pub fn set_core_timer_injection(&mut self, on: bool) {
        self.inject_core_timer = on;
    }

    /// Construct a machine in the long-mode boot state: kernel loaded at
    /// `kernel_load`, identity paging on, a boot_params zero page (e820 + cmdline)
    /// at 0x7000 with RSI pointing at it, and the Freeze v1 boot register file.
    /// Ready to execute from the kernel entry (GOLDEN_TARGET section 11, platform
    /// contract section 1).
    pub fn boot(
        seed: Seed256,
        ram_bytes: usize,
        kernel: &[u8],
        kernel_load: u64,
        cmdline: &str,
    ) -> Result<Self, X86Error> {
        let config = ConfigIdentity {
            kernel_hash: state::config_hash(kernel),
            cmdline_hash: state::config_hash(cmdline.as_bytes()),
            ..ConfigIdentity::default()
        };
        let mut machine = Self::new_with_config(seed, ram_bytes, &[], 0, config)?;
        machine.memory.load(kernel_load, kernel)?;

        let cr3 = platform::setup_identity_paging(&mut machine.memory)?;
        platform::build_acpi(&mut machine.memory)?;
        const ZERO_PAGE: u64 = 0x7000;
        const CMDLINE_PTR: u64 = 0x2_0000;
        let entries = platform::default_e820(ram_bytes as u64);
        platform::build_boot_params(
            &mut machine.memory,
            ZERO_PAGE,
            &entries,
            CMDLINE_PTR,
            cmdline,
        )?;

        let cpu = &mut machine.cpus[machine.current];
        cpu.regs.rip = kernel_load;
        cpu.regs.gpr[RSI] = ZERO_PAGE;
        cpu.regs.gpr[RSP] = ram_bytes as u64;
        cpu.sys.cr3 = cr3;
        cpu.sys.cr0 = CR0_BOOT;
        cpu.sys.cr4 = CR4_BOOT;
        cpu.sys.efer = EFER_BOOT;
        Ok(machine)
    }

    /// Boot a Linux x86 bzImage: parse the setup_header, load the 64-bit kernel,
    /// build boot_params, and enter at the 64-bit entry in the long-mode state.
    pub fn boot_bzimage(
        seed: Seed256,
        ram_bytes: usize,
        image: &[u8],
        cmdline: &str,
        initrd: Option<&[u8]>,
        disk: Option<Vec<u8>>,
    ) -> Result<Self, X86Error> {
        let config = ConfigIdentity {
            kernel_hash: state::config_hash(image),
            cmdline_hash: state::config_hash(cmdline.as_bytes()),
            initrd_hash: initrd.map(state::config_hash).unwrap_or(0),
            disk_image_hash: disk.as_deref().map(state::config_hash).unwrap_or(0),
            ..ConfigIdentity::default()
        };
        let mut machine = Self::new_with_config(seed, ram_bytes, &[], 0, config)?;
        let info =
            platform::load_bzimage(&mut machine.memory, image, cmdline, ram_bytes as u64, initrd)
                .map_err(|e| X86Error::Unsupported(format!("bzImage load: {e:?}")))?;
        let cr3 = platform::setup_identity_paging(&mut machine.memory)?;
        platform::build_acpi(&mut machine.memory)?;
        // Attach the backing disk to the virtio-mmio block device. The guest sees
        // it once the cmdline carries `virtio_mmio.device=0x200@0xd0000000:5`.
        if let Some(disk) = disk {
            machine.io.virtio = VirtioBlk::new(disk);
            // Optional deterministic fault injection for DST validation:
            //   SF_BLK_FAULT="seed,eio_permille,torn_permille,jitter"
            if let Ok(spec) = std::env::var("SF_BLK_FAULT") {
                let n: Vec<u64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                if n.len() == 4 {
                    machine.io.virtio.set_fault_policy(super::virtio_blk::BlkFaultPolicy::new(
                        n[0], n[1] as u32, n[2] as u32, n[3],
                    ));
                    eprintln!(
                        "SF_BLK_FAULT: seed={} eio={}‰ torn={}‰ jitter={}",
                        n[0], n[1], n[2], n[3]
                    );
                }
            }
        }

        let cpu = &mut machine.cpus[machine.current];
        cpu.regs.rip = info.entry;
        cpu.regs.gpr[RSI] = info.zero_page;
        cpu.regs.gpr[RSP] = ram_bytes as u64;
        cpu.sys.cr3 = cr3;
        cpu.sys.cr0 = CR0_BOOT;
        cpu.sys.cr4 = CR4_BOOT;
        cpu.sys.efer = EFER_BOOT;
        Ok(machine)
    }

    /// A human-readable dump of machine state — the "boot probe" used to see how
    /// far a real kernel got and what stopped it.
    pub fn diagnostics(&mut self) -> String {
        let rip = self.cpus[self.current].regs.rip;
        let mut code = String::new();
        for i in 0..16u64 {
            match self.peek_code(rip.wrapping_add(i)) {
                Some(b) => code.push_str(&format!("{b:02x} ")),
                None => code.push_str(".. "),
            }
        }
        let c = &self.cpus[self.current];
        let r = &c.regs;
        format!(
            "instructions={}\n\
             rip={:#018x} cs={:#06x} cpl={} rflags={:#06x}\n\
             code@rip={}\n\
             rax={:#018x} rbx={:#018x} rcx={:#018x} rdx={:#018x}\n\
             rsi={:#018x} rdi={:#018x} rsp={:#018x} rbp={:#018x}\n\
             cr0={:#x} cr2={:#018x} cr3={:#x} cr4={:#x} efer={:#x}\n\
             fs_base={:#x} gs_base={:#x} kernel_gs_base={:#x}\n\
             idtr={:#x}/{:#x} gdtr={:#x}/{:#x} halted={}",
            self.instruction_count(),
            r.rip, c.cs, c.cpl, r.rflags,
            code.trim_end(),
            r.gpr[0], r.gpr[3], r.gpr[1], r.gpr[2],
            r.gpr[6], r.gpr[7], r.gpr[4], r.gpr[5],
            c.sys.cr0, c.sys.cr2, c.sys.cr3, c.sys.cr4, c.sys.efer,
            c.msrs.fs_base, c.msrs.gs_base, c.msrs.kernel_gs_base,
            c.sys.idtr_base, c.sys.idtr_limit, c.sys.gdtr_base, c.sys.gdtr_limit,
            c.halted,
        )
    }

    /// Best-effort read of one code byte at a linear address (for the boot probe's
    /// disassembly hint). Translates through the guest page tables when paging is
    /// on; returns None on a translation/bounds fault.
    pub fn peek_code(&mut self, va: u64) -> Option<u8> {
        let pa = if self.cpus[self.current].sys.paging_enabled() {
            mmu::translate(
                &mut self.memory,
                self.cpus[self.current].sys.cr3,
                va,
                // Read access (not Execute) so the peek also works on NX data
                // pages — stacks, the memblock arrays, etc.
                Access::Read,
                self.cpus[self.current].cpl,
                self.cpus[self.current].sys.write_protect(),
                self.cpus[self.current].sys.nxe(),
            )
            .ok()?
        } else {
            va
        };
        self.memory.read_u8(pa).ok()
    }

    pub fn step(&mut self) -> Result<(), X86Error> {
        if self.cpus[self.current].halted {
            return self.step_halted();
        }

        // Deliver one pending external interrupt at this boundary if allowed.
        if self.cpus[self.current].can_take_interrupt() {
            if let Some(vector) = self.cpus[self.current].pending.pop_front() {
                self.cpus[self.current].inject_interrupt(&mut self.io, &mut self.memory, vector)?;
            }
        }

        // Expose virtual time to TSC/RDMSR reads and time-derived PIO devices
        // (PIT, RTC) for this instruction.
        let now = self.core.clock().instruction_count();
        self.cpus[self.current].tsc = now;
        self.io.now = now;

        if self.prof_on {
            let t0 = profile_rdtsc();
            self.cpus[self.current].step(&mut self.memory, &mut self.io)?;
            let t1 = profile_rdtsc();
            for event in self.core.step_instruction() {
                if let EventKind::TimerInterrupt { vector } = event.kind {
                    if self.inject_core_timer {
                        self.cpus[self.current].pending.push_back(vector);
                    }
                }
            }
            let t2 = profile_rdtsc();
            self.drain_ready_stores();
            self.check_lapic_timer();
            let t3 = profile_rdtsc();
            self.prof[0] += t1.wrapping_sub(t0);
            self.prof[1] += t2.wrapping_sub(t1);
            self.prof[2] += t3.wrapping_sub(t2);
            return Ok(());
        }

        self.cpus[self.current].step(&mut self.memory, &mut self.io)?;

        // Advance deterministic virtual time; queue any timer interrupts that
        // came due (they are masked by IF / IDT at the next boundary).
        for event in self.core.step_instruction() {
            if let EventKind::TimerInterrupt { vector } = event.kind {
                if self.inject_core_timer {
                    self.cpus[self.current].pending.push_back(vector);
                }
            }
        }
        self.drain_ready_stores();
        self.check_lapic_timer();
        Ok(())
    }

    /// x86-TSO (Stage 2): drain each vCPU's store buffer to shared memory for entries whose
    /// deadline has arrived, in FIFO (program) order per core and vCPU-index order across cores
    /// — a deterministic total order. This is where one core's buffered stores finally become
    /// visible to the others; the delay is what makes store→load reordering observable. No-op
    /// (and byte-identical) with TSO off, since all buffers are then empty.
    fn drain_ready_stores(&mut self) {
        if self.cpus[self.current].tso_max == 0 {
            return;
        }
        let now = self.core.clock().instruction_count();
        for i in 0..self.cpus.len() {
            while let Some(e) = self.cpus[i].store_buf.front().copied() {
                if e.deadline > now {
                    break;
                }
                self.cpus[i].store_buf.pop_front();
                let _ = self.memory.write_uint_le(e.pa, e.bytes as usize, e.value);
            }
        }
    }

    /// Poll device interrupt sources: the LAPIC timer and the virtio devices.
    /// Device IRQs route through the IOAPIC redirection table when programmed,
    /// else fall back to the device's default vector.
    /// Deliver an inter-processor interrupt issued by `self.current` (an ICR write). Resolves
    /// the destination set (physical ID or a shorthand), then applies the delivery mode:
    /// INIT parks the target AP waiting for a SIPI, SIPI starts it at real-mode `CS:IP =
    /// (vector<<8):0` (physical `vector<<12`), and Fixed queues the vector on the target.
    fn deliver_ipi(&mut self, ipi: Ipi) {
        let src = self.current;
        let n = self.cpus.len();
        let targets: Vec<usize> = match ipi.dest_shorthand {
            1 => vec![src],                                            // self
            2 => (0..n).collect(),                                     // all including self
            3 => (0..n).filter(|&i| i != src).collect(),              // all excluding self
            _ => vec![self.cpu_for_apic(ipi.dest)],                    // physical destination
        };
        for t in targets {
            match ipi.delivery_mode {
                DELIVERY_INIT => {
                    if t != src {
                        // Park the AP; a subsequent SIPI releases it. (Full INIT also resets AP
                        // register state — the AP has not run yet, so parking suffices here.)
                        self.cpus[t].wait_sipi = true;
                        self.cpus[t].halted = true;
                    }
                }
                DELIVERY_STARTUP => {
                    if t != src && self.cpus[t].wait_sipi {
                        let cpu = &mut self.cpus[t];
                        cpu.wait_sipi = false;
                        cpu.halted = false;
                        // Real-mode startup: CS selector = vector<<8 (so CS.base = vector<<12) and
                        // IP = 0, so the AP begins fetching at linear/physical `vector<<12`. Latch
                        // the CS descriptor cache directly (SIPI is a hardware CS load, not a far
                        // jmp): base vector<<12, 16-bit real-mode code.
                        cpu.cs = (ipi.vector as u16) << 8;
                        cpu.cs_base = (ipi.vector as u64) << 12;
                        cpu.code_bits = 16;
                        cpu.regs.rip = 0;
                    }
                }
                _ => {
                    // Fixed (and unmodelled modes): queue the vector on the target vCPU.
                    self.cpus[t].pending.push_back(ipi.vector);
                }
            }
        }
    }

    /// Index of the vCPU with the given (physical) APIC ID; falls back to the BSP (index 0)
    /// when no vCPU matches. At `vcpus == 1` every destination resolves to the sole vCPU.
    fn cpu_for_apic(&self, apic_id: u8) -> usize {
        self.cpus
            .iter()
            .position(|c| c.apic_id == apic_id as u32)
            .unwrap_or(0)
    }

    fn check_lapic_timer(&mut self) {
        // SMP: deliver any inter-processor interrupt this vCPU issued (ICR write) in its last
        // instruction. Guarded to multi-vCPU so the single-CPU path is byte-identical — a lone
        // CPU has no other core to signal and (as before) never self-delivers via the ICR.
        if self.cpus.len() > 1 {
            if let Some(ipi) = self.cpus[self.current].lapic.take_ipi() {
                self.deliver_ipi(ipi);
            }
        }
        let now = self.core.clock().instruction_count();
        // P4: release any seed-jittered timer interrupts whose deferred delivery
        // instant has arrived. Same vector for all entries, so order is immaterial.
        if !self.cpus[self.current].delayed.is_empty() {
            let mut i = 0;
            while i < self.cpus[self.current].delayed.len() {
                if self.cpus[self.current].delayed[i].0 <= now {
                    let (_, v) = self.cpus[self.current].delayed.remove(i);
                    self.cpus[self.current].pending.push_back(v);
                } else {
                    i += 1;
                }
            }
        }
        if let Some(vector) = self.cpus[self.current].lapic.timer_fired(now) {
            self.cpus[self.current].pending.push_back(vector);
        }
        // 8254 PIT channel-0 system tick → IRQ0 → IOAPIC GSI 0 (the kernel programs
        // its vector during check_timer). Only delivered when the GSI is programmed
        // and unmasked; this is what advances jiffies.
        if self.io.pit.timer_fired(now) {
            if let Some((v, dest)) = self.io.ioapic.route_for_gsi(PIT_GSI) {
                let idx = self.cpu_for_apic(dest);
                // P4 seed exploration: defer the scheduler tick by a bounded,
                // seed-derived jitter so the preemption lands on a seed-dependent
                // instruction → a different thread interleaving per seed. The draw
                // returns 0 (and consumes nothing) when jitter is disabled, leaving
                // the proven baseline byte-for-byte unchanged.
                let j = self.core.draw_sched_jitter(self.sched_jitter);
                if j == 0 {
                    self.cpus[idx].pending.push_back(v);
                } else {
                    self.cpus[idx].delayed.push((now.saturating_add(j), v));
                }
            }
        }
        // COM1 16550 transmit-empty interrupt → IRQ4 → IOAPIC GSI 4. The 8250 tty
        // tx path enables IER.THRE and waits for this IRQ to drain its buffer to
        // the THR (which our serial captures). Level-triggered while armed; pacing
        // is by IF (the gate clears it, so at most one delivery per ISR cycle), and
        // the not-already-queued guard keeps the queue from growing during an ISR.
        if self.io.serial.tx_irq_pending() {
            if let Some((v, dest)) = self.io.ioapic.route_for_gsi(COM1_GSI) {
                let idx = self.cpu_for_apic(dest);
                if !self.cpus[idx].pending.contains(&v) {
                    self.cpus[idx].pending.push_back(v);
                }
            }
        }
        // Apply any virtio-blk completions whose deterministic deadline arrived
        // (controllable I/O: the completion is event-queued in virtual time, not
        // synchronous with the queue notify), then deliver its interrupt.
        self.io.virtio.poll_completions(now, &mut self.memory);
        if self.io.virtio.take_interrupt().is_some() {
            let (v, dest) = self
                .io
                .ioapic
                .route_for_gsi(VIRTIO_BLK_GSI)
                .unwrap_or((VIRTIO_BLK_VECTOR, 0));
            let idx = self.cpu_for_apic(dest);
            self.cpus[idx].pending.push_back(v);
        }
        if self.io.virtio_rng.take_interrupt().is_some() {
            let (v, dest) = self
                .io
                .ioapic
                .route_for_gsi(VIRTIO_RNG_GSI)
                .unwrap_or((VIRTIO_RNG_VECTOR, 0));
            let idx = self.cpu_for_apic(dest);
            self.cpus[idx].pending.push_back(v);
        }
        // virtio-net: poll RX/TX completions, then run the Phase-1a backend — a
        // deterministic LOOPBACK REFLECTOR (TX frames → fault policy → RX). Phase 1b
        // replaces this with a two-guest switch. Zero-cost when there is no NIC.
        if self.io.virtio_net.is_some() {
            let net = self.io.virtio_net.as_mut().unwrap();
            net.poll_completions(now, &mut self.memory);
            // Single-guest reflector: loop TX→RX. In two-guest mode (reflector off)
            // an external switch owns TX drain + RX inject, so skip this.
            if net.reflector {
                for frame in net.take_tx_frames() {
                    net.deliver_frame(frame, now);
                }
            }
            if net.take_interrupt().is_some() {
                let (v, dest) = self
                    .io
                    .ioapic
                    .route_for_gsi(VIRTIO_NET_GSI)
                    .unwrap_or((VIRTIO_NET_VECTOR, 0));
                let idx = self.cpu_for_apic(dest);
                self.cpus[idx].pending.push_back(v);
            }
        }
    }

    /// HLT waits for an interrupt. Instead of burning instruction-count one tick
    /// at a time, fast-forward virtual time to the next scheduled event (idle
    /// fast-forward, GOLDEN_TARGET rule 11), queue any timer it produces, and
    /// deliver it — which wakes the CPU.
    fn step_halted(&mut self) -> Result<(), X86Error> {
        // The serial THRE line is level-triggered: if the tty tx path armed it
        // while the CPU went idle, wake now rather than fast-forwarding past it.
        if self.cpus[self.current].pending.is_empty() && self.io.serial.tx_irq_pending() {
            if let Some((v, dest)) = self.io.ioapic.route_for_gsi(COM1_GSI) {
                // Only wake this halted vCPU if it is the interrupt's destination.
                if self.cpu_for_apic(dest) == self.current {
                    self.cpus[self.current].pending.push_back(v);
                }
            }
        }
        if self.cpus[self.current].pending.is_empty() {
            // Jump to the earliest of the core event queue, the LAPIC deadline, or
            // the PIT system-tick deadline.
            let core_next = self.core.next_event_instruction();
            let lapic_next = self.cpus[self.current].lapic.next_deadline();
            let pit_next = self.io.pit.next_deadline();
            let virtio_next = self.io.virtio.next_completion_deadline();
            // A seed-jittered deferred interrupt is a future wake too.
            let delayed_next = self.cpus[self.current].delayed.iter().map(|(at, _)| *at).min();
            let target = [core_next, lapic_next, pit_next, virtio_next, delayed_next]
                .into_iter()
                .flatten()
                .min();
            match target {
                None => return Ok(()), // nothing scheduled: stays halted
                Some(t) => {
                    for event in self.core.advance_clock_to(t) {
                        if let EventKind::TimerInterrupt { vector } = event.kind {
                            if self.inject_core_timer {
                                self.cpus[self.current].pending.push_back(vector);
                            }
                        }
                    }
                    self.check_lapic_timer();
                }
            }
        }
        if self.cpus[self.current].can_take_interrupt() {
            if let Some(vector) = self.cpus[self.current].pending.pop_front() {
                self.cpus[self.current].inject_interrupt(&mut self.io, &mut self.memory, vector)?;
            }
        }
        Ok(())
    }

    /// vCPU `idx` is halted with no way ever to wake: interrupts masked / no IDT and nothing
    /// pending, or no future event/timer to fast-forward to.
    fn is_vcpu_terminal(&self, idx: usize) -> bool {
        // A modeled timer with a future deadline (LAPIC or PIT) or a scheduled
        // virtio-blk completion can still wake us. Evaluate `idx`'s OWN wake sources
        // (pending/delayed/interrupt-state) — NOT the currently-running vCPU's — so with
        // >1 vCPU a halted AP is judged terminal by its own queues, not the BSP's. At
        // vcpus==1 `idx == self.current`, so this is byte-identical to the prior code.
        let has_timer = self.cpus[idx].lapic.next_deadline().is_some()
            || self.io.pit.next_deadline().is_some()
            || self.io.virtio.next_completion_deadline().is_some()
            || !self.cpus[idx].delayed.is_empty()
            || self.core.has_events();
        self.cpus[idx].halted
            && self.cpus[idx].pending.is_empty()
            && (!self.cpus[idx].can_take_interrupt() || !has_timer)
    }

    /// Round-robin scheduler: pick the next vCPU to run, scanning from `current+1`. Returns a
    /// vCPU that is running OR halted-but-wakeable (so `step_halted` can idle-fast-forward it);
    /// returns `None` only when EVERY vCPU is terminally halted. At `vcpus == 1` this returns
    /// `Some(0)` until the sole vCPU terminally halts, so single-CPU execution order is unchanged.
    fn pick_next_vcpu(&self) -> Option<usize> {
        let n = self.cpus.len();
        for off in 1..=n {
            let idx = (self.current + off) % n;
            // An AP parked waiting for its SIPI is not runnable; skip it (it wakes when a
            // Startup-IPI clears `wait_sipi`).
            if !self.cpus[idx].wait_sipi && !self.is_vcpu_terminal(idx) {
                return Some(idx);
            }
        }
        None
    }

    /// A vCPU is *runnable* if it is neither parked for a SIPI nor terminally halted
    /// (a halted-but-wakeable vCPU counts, so `step_halted` can idle-fast-forward it).
    fn is_runnable(&self, idx: usize) -> bool {
        !self.cpus[idx].wait_sipi && !self.is_vcpu_terminal(idx)
    }

    /// Choose the vCPU to run the next instruction.
    ///
    /// SMP Stage-1: when seed-driven exploration is OFF (`smp_sched == 0`) — the default and
    /// every golden — this is exactly the fixed round-robin `pick_next_vcpu`, so nothing about
    /// the baseline (incl. the `--smp 2` boot) changes and the scheduler RNG is never drawn.
    ///
    /// When ON and ≥2 vCPUs are runnable, the seed drives the schedule: a vCPU runs for a
    /// bounded *quantum* (`1..=smp_sched` instructions), then the scheduler draws a fresh
    /// quantum and a next runnable vCPU uniformly. Both draws come from the dedicated
    /// `smp_sched_rng` stream, so the whole interleaving is a reproducible function of the
    /// seed — different seeds realise different legal interleavings (and thus expose different
    /// data races), each byte-identical run-to-run.
    fn schedule_next(&mut self) -> Option<usize> {
        if self.pct_depth != 0 {
            return self.schedule_next_pct();
        }
        if self.smp_sched == 0 {
            return self.pick_next_vcpu();
        }
        let n = self.cpus.len();
        let runnable: Vec<usize> = (0..n).filter(|&i| self.is_runnable(i)).collect();
        // Did the vCPU we just ran hit a PAUSE (spin-wait hint)? If so, force a switch so a core
        // spinning on a lock yields to the holder instead of burning its whole quantum. Cleared
        // here whether or not we act on it (single-runnable case has no one to yield to).
        let yielded = self.cpus[self.current].pause_yield;
        self.cpus[self.current].pause_yield = false;
        match runnable.len() {
            0 => None,
            1 => {
                // No scheduling choice; reset the quantum so a fresh one is drawn when
                // contention (≥2 runnable) resumes.
                self.smp_quantum_remaining = 0;
                Some(runnable[0])
            }
            _ => {
                // On a PAUSE yield, switch to a DIFFERENT runnable core (pause-loop-exit): pick
                // uniformly among the others so the spinner gives way to the lock holder.
                if yielded {
                    let others: Vec<usize> =
                        runnable.iter().copied().filter(|&i| i != self.current).collect();
                    let pick = self.smp_sched_rng.next_bounded(others.len() as u64) as usize;
                    self.smp_quantum_remaining = self.smp_sched_rng.next_bounded(self.smp_sched);
                    return Some(others[pick]);
                }
                // Keep running the current vCPU while its quantum lasts and it stays runnable.
                if self.smp_quantum_remaining > 0 && runnable.contains(&self.current) {
                    self.smp_quantum_remaining -= 1;
                    return Some(self.current);
                }
                // New epoch: draw a fresh quantum length and a next runnable vCPU.
                self.smp_quantum_remaining = self.smp_sched_rng.next_bounded(self.smp_sched);
                let pick = self.smp_sched_rng.next_bounded(runnable.len() as u64) as usize;
                Some(runnable[pick])
            }
        }
    }

    /// A vCPU is PCT-*enabled* (able to make progress right now) if it is runnable and either not
    /// halted or holding a pending interrupt that would wake it. A halted vCPU with nothing pending
    /// is *blocked* (e.g. spinning on a HLT waiting for an IPI) — priority scheduling must skip it
    /// so it does not starve the cores that would unblock it.
    fn pct_enabled(&self, i: usize) -> bool {
        self.is_runnable(i) && (!self.cpus[i].halted || !self.cpus[i].pending.is_empty())
    }

    /// Highest-priority vCPU (ties → lowest index) among those matching the filter. `enabled_only`
    /// restricts to PCT-enabled vCPUs; `exclude` skips one index (used for pause-yield).
    fn pct_highest(&self, enabled_only: bool, exclude: Option<usize>) -> Option<usize> {
        let mut best = None;
        let mut best_prio = i64::MIN;
        for i in 0..self.cpus.len() {
            if Some(i) == exclude || !self.is_runnable(i) {
                continue;
            }
            if enabled_only && !self.pct_enabled(i) {
                continue;
            }
            if self.pct_prio[i] > best_prio {
                best_prio = self.pct_prio[i];
                best = Some(i);
            }
        }
        best
    }

    /// SMP Stage-3 PCT scheduler: run the highest-priority *enabled* vCPU, demoting the running
    /// vCPU below all others as each seed-drawn change point's instruction count is reached.
    /// Between change points a vCPU runs to completion (or until it blocks/pauses) — the coarse,
    /// priority-driven schedule PCT's probabilistic bug-depth guarantee relies on.
    fn schedule_next_pct(&mut self) -> Option<usize> {
        // Fire every change point we've reached, demoting the vCPU that was running at that point.
        let now = self.core.clock().instruction_count();
        while self.pct_next_cp < self.pct_change_points.len()
            && now >= self.pct_change_points[self.pct_next_cp]
        {
            self.pct_prio[self.current] = self.pct_low;
            self.pct_low -= 1;
            self.pct_next_cp += 1;
        }
        // A PAUSE spin-wait hint makes the current vCPU step aside (avoids priority-inversion:
        // a high-priority spinner gives way to a lower-priority lock holder).
        let yielded = self.cpus[self.current].pause_yield;
        self.cpus[self.current].pause_yield = false;
        if yielded {
            if let Some(x) = self.pct_highest(true, Some(self.current)) {
                return Some(x);
            }
        }
        // Highest-priority enabled vCPU; else the highest-priority runnable (halted-but-wakeable)
        // one so `step_halted` can idle-fast-forward to the next event. None ⇒ all terminal.
        self.pct_highest(true, None).or_else(|| self.pct_highest(false, None))
    }

    pub fn run(&mut self, max_instructions: u64) -> Result<X86Exit, X86Error> {
        for _ in 0..max_instructions {
            match self.schedule_next() {
                Some(idx) => {
                    self.current = idx;
                    self.step()?;
                }
                None => return Ok(X86Exit::Halted), // all vCPUs terminally halted
            }
        }
        Ok(X86Exit::InstructionLimit)
    }

    /// Run until the absolute virtual instruction count reaches `target` (or the
    /// guest terminally halts). Unlike `run` (a relative step budget), this stops
    /// at an absolute instruction count — the cadence used for snapshots and
    /// checkpoint chains (GOLDEN_TARGET §19).
    pub fn run_until_instruction(&mut self, target: u64) -> Result<X86Exit, X86Error> {
        while self.instruction_count() < target {
            match self.schedule_next() {
                Some(idx) => {
                    self.current = idx;
                    self.step()?;
                }
                None => return Ok(X86Exit::Halted),
            }
        }
        Ok(X86Exit::InstructionLimit)
    }

    /// Capture a full-machine snapshot: a deep copy of ALL deterministic state
    /// (CPU registers/sys/xmm, every device, guest RAM, and the core clock + RNG
    /// streams). Restoring it resumes a bit-identical execution — the substrate
    /// for "rewind to any point and replay" (GOLDEN_TARGET §19). Because the whole
    /// machine derives `Clone` and holds no host handles, the snapshot is just a
    /// deep clone; there is nothing host-coupled to lose.
    pub fn snapshot(&self) -> X86Snapshot {
        X86Snapshot {
            machine: Box::new(self.clone()),
            instruction_count: self.instruction_count(),
            digest: self.full_state_digest(),
        }
    }

    pub fn cpu(&self) -> &X86Cpu {
        &self.cpus[self.current]
    }

    pub fn memory(&self) -> &GuestMemory {
        &self.memory
    }

    /// The COM1 serial console (collected output bytes).
    pub fn serial(&self) -> &super::serial::Uart16550 {
        &self.io.serial
    }

    pub fn memory_mut(&mut self) -> &mut GuestMemory {
        &mut self.memory
    }

    pub fn cpu_mut(&mut self) -> &mut X86Cpu {
        &mut self.cpus[self.current]
    }

    /// Debug: enable AP-trampoline tracing on every application processor (index > 0),
    /// logging each AP's first `n` instructions. The BSP is left untraced. Used by
    /// `SF_APTRACE` to diagnose the real-mode AP bring-up path.
    pub fn set_ap_trace(&mut self, n: u64) {
        for cpu in self.cpus.iter_mut().skip(1) {
            cpu.trace_ap = true;
            cpu.ap_trace_n = n;
        }
    }

    /// The machine-level shared I/O bus (serial, PIT, RTC, IOAPIC, virtio devices).
    pub fn io(&self) -> &IoBus {
        &self.io
    }
    pub fn io_mut(&mut self) -> &mut IoBus {
        &mut self.io
    }

    pub fn instruction_count(&self) -> u64 {
        self.core.clock().instruction_count()
    }

    /// Full deterministic state digest (core + memory + register file).
    pub fn full_state_digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.core.full_state_digest());
        digest.mix_u64(self.memory.digest());
        for reg in self.cpus[self.current].regs.gpr {
            digest.mix_u64(reg);
        }
        digest.mix_u64(self.cpus[self.current].regs.rip);
        digest.mix_u64(self.cpus[self.current].regs.rflags);
        digest.mix_u64(self.cpus[self.current].sys.cr0);
        digest.mix_u64(self.cpus[self.current].sys.cr2);
        digest.mix_u64(self.cpus[self.current].sys.cr3);
        digest.mix_u64(self.cpus[self.current].sys.cr4);
        digest.mix_u64(self.cpus[self.current].sys.efer);
        digest.mix_u64(self.cpus[self.current].sys.idtr_base);
        digest.mix_u64(self.cpus[self.current].sys.idtr_limit as u64);
        digest.mix_u64(self.cpus[self.current].sys.gdtr_base);
        digest.mix_u64(self.cpus[self.current].sys.gdtr_limit as u64);
        digest.mix_u64(self.cpus[self.current].sys.cr8);
        digest.mix_u64(self.cpus[self.current].sys.tss_rsp0);
        digest.mix_u64(self.cpus[self.current].sys.tr_base);
        for d in self.cpus[self.current].sys.dr {
            digest.mix_u64(d);
        }
        for x in self.cpus[self.current].xmm {
            digest.mix_u64(x as u64);
            digest.mix_u64((x >> 64) as u64);
        }
        // YMM upper halves (AVX). Mixed only when non-zero, tagged by index, so an
        // all-zero upper file (no AVX executed) reproduces the legacy digest exactly
        // while live upper halves are still captured for divergence detection.
        for (i, hi) in self.cpus[self.current].ymm_hi.iter().enumerate() {
            if *hi != 0 {
                digest.mix_u64(0x594d_4d00_0000_0000 ^ i as u64); // 'YMM\0' tag + index
                digest.mix_u64(*hi as u64);
                digest.mix_u64((*hi >> 64) as u64);
            }
        }
        // XCR0: mixed only when not at its reset value (1), so a machine that never
        // enables AVX/XSAVE reproduces the legacy digest exactly.
        if self.cpus[self.current].xcr0 != 1 {
            digest.mix_u64(0x5843_5230_0000_0000 ^ self.cpus[self.current].xcr0); // 'XCR0' tag
        }
        digest.mix_u64(self.cpus[self.current].lapic.digest());
        digest.mix_u64(self.io.ioapic.digest());
        digest.mix_u64(self.io.virtio.digest());
        digest.mix_u64(self.io.virtio_rng.digest());
        // virtio-net mixed ONLY when present (default None → no digest change, so
        // every existing workload/gate golden is byte-identical).
        if let Some(net) = &self.io.virtio_net {
            digest.mix_u64(net.digest());
        }
        // RDRAND/RDSEED stream cursor — mixed only once consumed (cursor advanced
        // off its seeded origin), so a machine that never executes RDRAND reproduces
        // the legacy digest exactly.
        if self.cpus[self.current].rdrand_rng.state() != self.cpus[self.current].rdrand_origin {
            digest.mix_u64(0x5244_5241_0000_0000 ^ self.cpus[self.current].rdrand_rng.state()); // 'RDRA' tag
        }
        digest.mix_u64(self.cpus[self.current].cs as u64);
        digest.mix_u64(self.cpus[self.current].ss as u64);
        digest.mix_u64(self.cpus[self.current].ds as u64);
        digest.mix_u64(self.cpus[self.current].es as u64);
        digest.mix_u64(self.cpus[self.current].fs_sel as u64);
        digest.mix_u64(self.cpus[self.current].gs_sel as u64);
        digest.mix_u64(self.cpus[self.current].cpl as u64);
        digest.mix_u64(self.cpus[self.current].msrs.digest());
        digest.mix_u64(self.io.digest());
        digest.mix_u64(self.cpus[self.current].halted as u64);
        digest.mix_u64(self.cpus[self.current].fpu_cw as u64);
        digest.mix_u64(self.cpus[self.current].fpu_sw as u64);
        digest.mix_u64(self.cpus[self.current].fpu_tw as u64);
        digest.mix_u64(self.cpus[self.current].fpu_top as u64);
        for s in self.cpus[self.current].st {
            // Mix the full 80-bit register (mantissa + sign/exp). An empty stack is F80::ZERO
            // → (0, 0), preserving the previous all-zero digest contribution (gate-safe).
            let (m, se) = s.to_m80();
            digest.mix_u64(m);
            digest.mix_u64(se as u64);
        }
        digest.mix_u64(self.cpus[self.current].mxcsr as u64);
        // Pending external interrupts are future-affecting state: two machines
        // identical except for a queued vector diverge at the next IF/IDT
        // boundary, so the digest must cover the whole queue (and its length).
        digest.mix_u64(self.cpus[self.current].pending.len() as u64);
        for vector in &self.cpus[self.current].pending {
            digest.mix_u64(*vector as u64);
        }
        // P4 jitter: deferred timer interrupts (release instant + vector) are also
        // future-affecting state, plus the configured jitter bound.
        digest.mix_u64(self.sched_jitter);
        digest.mix_u64(self.cpus[self.current].delayed.len() as u64);
        for (at, vector) in &self.cpus[self.current].delayed {
            digest.mix_u64(*at);
            digest.mix_u64(*vector as u64);
        }
        // SMP: the current vCPU is folded above (byte-identical to the single-CPU digest). Fold
        // every OTHER vCPU's full state too, tagged by index, plus the scheduler cursor. At
        // `vcpus == 1` this loop is empty and the cursor is not mixed, so the digest is unchanged.
        if self.cpus.len() > 1 {
            digest.mix_u64(0x5643_5055_0000_0000 ^ self.current as u64); // 'VCPU' tag + cursor
            for (i, cpu) in self.cpus.iter().enumerate() {
                if i != self.current {
                    digest.mix_u64(0x5643_5055_0001_0000 ^ i as u64); // per-vCPU index tag
                    fold_cpu_state(cpu, &mut digest);
                }
            }
        }
        // SMP Stage-1 seed-driven scheduler state — mixed ONLY when exploration is enabled
        // (`smp_sched != 0`), so a baseline run (fixed round-robin) reproduces the legacy
        // digest exactly. When on, the quantum bound, the remaining-quantum counter, and the
        // scheduler RNG cursor are all future-affecting and captured for snapshot/divergence.
        if self.smp_sched != 0 {
            digest.mix_u64(0x534d_5053_0000_0000 ^ self.smp_sched); // 'SMPS' tag + quantum bound
            digest.mix_u64(self.smp_quantum_remaining);
            digest.mix_u64(self.smp_sched_rng.state());
        }
        // Stage-3 PCT scheduler state — folded ONLY when PCT is enabled, so the baseline digest
        // is unchanged. Priorities, change-point cursor, next demotion, and the PCT RNG cursor
        // are all future-affecting.
        if self.pct_depth != 0 {
            digest.mix_u64(0x5043_5400_0000_0000 ^ self.pct_depth); // 'PCT\0' tag + depth
            for (i, p) in self.pct_prio.iter().enumerate() {
                digest.mix_u64((i as u64) << 56 ^ (*p as u64));
            }
            digest.mix_u64(self.pct_next_cp as u64);
            digest.mix_u64(self.pct_low as u64);
            digest.mix_u64(self.pct_rng.state());
        }
        // x86-TSO store buffers (Stage 2): fold each vCPU's pending stores + its drain RNG
        // cursor + the TSO bound — ONLY when TSO is enabled, so the SC baseline (empty buffers,
        // untouched RNG) reproduces the legacy digest exactly.
        if self.cpus[self.current].tso_max != 0 {
            digest.mix_u64(0x5453_4f00_0000_0000 ^ self.cpus[self.current].tso_max); // 'TSO\0'
            for (i, cpu) in self.cpus.iter().enumerate() {
                digest.mix_u64(0x5453_4f00_0001_0000 ^ i as u64);
                digest.mix_u64(cpu.sb_rng.state());
                digest.mix_u64(cpu.store_buf.len() as u64);
                for e in &cpu.store_buf {
                    digest.mix_u64(e.pa);
                    digest.mix_u64(e.value);
                    digest.mix_u64((e.bytes as u64) | (e.deadline << 8));
                }
            }
        }
        digest.finish()
    }
}

/// Fold one vCPU's full per-CPU deterministic state into `digest` (used for the non-current
/// vCPUs in the SMP whole-machine digest; the current vCPU is folded inline for byte-identity
/// with the legacy single-CPU digest). Machine-level state (memory, core, `io` devices) is NOT
/// included here — it is folded once at the machine level.
fn fold_cpu_state(cpu: &X86Cpu, digest: &mut StateDigest) {
    for reg in cpu.regs.gpr {
        digest.mix_u64(reg);
    }
    digest.mix_u64(cpu.regs.rip);
    digest.mix_u64(cpu.regs.rflags);
    digest.mix_u64(cpu.apic_id as u64);
    let s = &cpu.sys;
    for v in [
        s.cr0, s.cr2, s.cr3, s.cr4, s.efer, s.idtr_base, s.idtr_limit as u64, s.gdtr_base,
        s.gdtr_limit as u64, s.cr8, s.tss_rsp0, s.tr_base,
    ] {
        digest.mix_u64(v);
    }
    for d in s.dr {
        digest.mix_u64(d);
    }
    for x in cpu.xmm {
        digest.mix_u64(x as u64);
        digest.mix_u64((x >> 64) as u64);
    }
    for (i, hi) in cpu.ymm_hi.iter().enumerate() {
        if *hi != 0 {
            digest.mix_u64(0x594d_4d00_0000_0000 ^ i as u64);
            digest.mix_u64(*hi as u64);
            digest.mix_u64((*hi >> 64) as u64);
        }
    }
    if cpu.xcr0 != 1 {
        digest.mix_u64(0x5843_5230_0000_0000 ^ cpu.xcr0);
    }
    digest.mix_u64(cpu.lapic.digest());
    if cpu.rdrand_rng.state() != cpu.rdrand_origin {
        digest.mix_u64(0x5244_5241_0000_0000 ^ cpu.rdrand_rng.state());
    }
    for v in [cpu.cs, cpu.ss, cpu.ds, cpu.es, cpu.fs_sel, cpu.gs_sel] {
        digest.mix_u64(v as u64);
    }
    digest.mix_u64(cpu.cpl as u64);
    digest.mix_u64(cpu.msrs.digest());
    digest.mix_u64(cpu.halted as u64);
    digest.mix_u64(cpu.wait_sipi as u64);
    digest.mix_u64(cpu.fpu_cw as u64);
    digest.mix_u64(cpu.fpu_sw as u64);
    digest.mix_u64(cpu.fpu_tw as u64);
    digest.mix_u64(cpu.fpu_top as u64);
    for st in cpu.st {
        let (m, se) = st.to_m80();
        digest.mix_u64(m);
        digest.mix_u64(se as u64);
    }
    digest.mix_u64(cpu.mxcsr as u64);
    digest.mix_u64(cpu.pending.len() as u64);
    for vector in &cpu.pending {
        digest.mix_u64(*vector as u64);
    }
    digest.mix_u64(cpu.delayed.len() as u64);
    for (at, vector) in &cpu.delayed {
        digest.mix_u64(*at);
        digest.mix_u64(*vector as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86::regs::{CR0_PG, RAX, RBX, RCX, RDX, R11, R15};
    use crate::x86::Size;

    // STAR: SYSRET base 0x20 (bits 63:48), SYSCALL base 0x08 (bits 47:32).
    const STAR: u64 = (0x20_u64 << 48) | (0x08_u64 << 32);

    /// Install a long-mode interrupt gate at `idt_base[vector]` → `handler`.
    fn set_idt_gate(mem: &mut GuestMemory, idt_base: u64, vector: u64, handler: u64) {
        let selector = 0x08_u64;
        let type_attr = 0x80_u64 | 0xe; // present | interrupt gate
        let lo = (handler & 0xffff)
            | (selector << 16)
            | (type_attr << 40)
            | (((handler >> 16) & 0xffff) << 48);
        let hi = (handler >> 32) & 0xffff_ffff;
        mem.write_uint_le(idt_base + vector * 16, 8, lo).unwrap();
        mem.write_uint_le(idt_base + vector * 16 + 8, 8, hi).unwrap();
    }

    fn seed() -> Seed256 {
        Seed256::from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            .unwrap()
    }

    fn run(program: &[u8], max: u64) -> X86Machine {
        let mut m = X86Machine::new(seed(), 4096, program, 0).unwrap();
        m.run(max).unwrap();
        m
    }

    #[test]
    fn smp_two_vcpus_shared_counter() {
        // SMP Stage 0: two vCPUs each increment a shared 64-bit counter at [0x800] five times,
        // then HLT. Under the round-robin scheduler over sequentially-consistent shared memory,
        // each `inc [mem]` is one atomic step — so the result is exactly 2×5 = 10, deterministically.
        //   mov rcx,5 ; L: inc qword[0x800] ; dec rcx ; jnz L ; hlt
        let prog = [
            0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
            0x48, 0xff, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, // inc qword [0x800]
            0x48, 0xff, 0xc9, // dec rcx
            0x75, 0xf3, // jnz -13 -> inc
            0xf4, // hlt
        ];
        let build = || {
            let mut m = X86Machine::new(seed(), 4096, &prog, 0).unwrap();
            let ap = m.cpus[0].clone(); // second vCPU runs the same program from rip=0
            m.cpus.push(ap);
            m.run(100_000).unwrap();
            m
        };
        let m1 = build();
        assert_eq!(m1.cpus.len(), 2);
        assert_eq!(
            m1.memory.read_u64_le(0x800).unwrap(),
            10,
            "both vCPUs' increments must accumulate over shared SC memory"
        );
        assert!(m1.cpus[0].halted && m1.cpus[1].halted, "both vCPUs reach HLT");
        // Determinism: a re-run is bit-identical in memory + both vCPU contexts.
        let m2 = build();
        assert_eq!(m1.memory.digest(), m2.memory.digest(), "SMP memory deterministic");
        assert_eq!(m1.full_state_digest(), m2.full_state_digest(), "SMP whole-machine deterministic");
        assert_eq!(m1.cpus[0].regs.rip, m2.cpus[0].regs.rip);
        assert_eq!(m1.cpus[1].regs.rip, m2.cpus[1].regs.rip);
    }

    #[test]
    fn smp_stage1_seed_driven_interleaving() {
        // SMP Stage 1: a NON-ATOMIC read-modify-write race. Each of two vCPUs does
        // `counter = counter + 1` as THREE separate instructions (load / inc / store), N times.
        // Because the RMW is not atomic, an interleaving where both cores read the same value
        // loses an update — a *true* SMP data race that a single vCPU cannot produce (there is
        // no second core to interleave between the load and the store). The seed-driven
        // scheduler realises different interleavings for different seeds, each reproducible.
        const N: u64 = 20;
        let prog = [
            0x48, 0xc7, 0xc1, N as u8, 0x00, 0x00, 0x00, // mov rcx, N
            0x48, 0x8b, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, // L: mov rax, [0x800]
            0x48, 0xff, 0xc0, //                              inc rax
            0x48, 0x89, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, // mov [0x800], rax
            0x48, 0xff, 0xc9, //                              dec rcx
            0x75, 0xe8, //                                    jnz L (-24)
            0xf4, //                                          hlt
        ];
        // Build a 2-vCPU machine (both run the same loop from rip=0) with the given seed and
        // scheduler quantum; run to completion and return the final shared counter.
        let run_counter = |seed_hex: &str, quantum: u64| -> u64 {
            let s = Seed256::from_hex(seed_hex).unwrap();
            let mut m = X86Machine::new(s, 4096, &prog, 0).unwrap();
            let ap = m.cpus[0].clone();
            m.cpus.push(ap);
            m.set_smp_sched(quantum);
            m.run(1_000_000).unwrap();
            m.memory.read_u64_le(0x800).unwrap()
        };
        let seed_n = |i: u64| format!("{i:064x}");

        // (1) Reproducible: same seed + same quantum → identical outcome.
        let a = run_counter(&seed_n(1), 3);
        let b = run_counter(&seed_n(1), 3);
        assert_eq!(a, b, "a seed-driven SMP schedule is byte-reproducible");

        // (2) Sound bounds: never fewer than N (a thread's own N increments always land at
        // least once each in the worst case) nor more than 2N (no update appears from nothing).
        assert!((N..=2 * N).contains(&a), "counter {a} within [{N}, {}]", 2 * N);

        // (3) Exploration + the race is real: sweeping seeds yields more than one distinct
        // outcome, and at least one seed loses updates (< 2N) — exactly the data race a single
        // vCPU cannot expose.
        let results: Vec<u64> = (1..=24).map(|i| run_counter(&seed_n(i), 3)).collect();
        let distinct: std::collections::BTreeSet<u64> = results.iter().copied().collect();
        assert!(distinct.len() > 1, "seeds explore different interleavings: {distinct:?}");
        assert!(
            results.iter().any(|&c| c < 2 * N),
            "at least one seed exposes a lost-update race (all = {}): {results:?}",
            2 * N
        );

        // (4) Off by default: quantum 0 (the baseline) is deterministic and never draws the
        // scheduler stream — its whole-machine digest is unaffected by the exploration knob.
        let build0 = || {
            let mut m = X86Machine::new(seed(), 4096, &prog, 0).unwrap();
            let ap = m.cpus[0].clone();
            m.cpus.push(ap);
            m.run(1_000_000).unwrap();
            m
        };
        let base_a = build0();
        let base_b = build0();
        assert_eq!(
            base_a.full_state_digest(),
            base_b.full_state_digest(),
            "baseline (round-robin, smp_sched=0) is deterministic and digest-neutral"
        );
    }

    #[test]
    fn smp_stage2_tso_store_buffering_litmus() {
        // Store-Buffering (SB) / Dekker litmus — the test that separates x86-TSO from SC.
        //   CPU0:  x = 1 ; r0 = y      CPU1:  y = 1 ; r1 = x
        // Sequential Consistency FORBIDS r0==0 && r1==0 (some store must be seen). x86-TSO
        // ALLOWS it: both stores sit in their cores' store buffers when the loads execute, so
        // each core reads the other's *stale* value. x=[0x800] y=[0x808] r0→[0x810] r1→[0x818].
        let mut prog = vec![0u8; 0x200];
        let cpu0 = [
            0xc7, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // mov [0x800],1
            0x8b, 0x04, 0x25, 0x08, 0x08, 0x00, 0x00, //                         mov eax,[0x808]
            0x89, 0x04, 0x25, 0x10, 0x08, 0x00, 0x00, //                         mov [0x810],eax
            0xf4, //                                                             hlt
        ];
        let cpu1 = [
            0xc7, 0x04, 0x25, 0x08, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // mov [0x808],1
            0x8b, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, //                         mov eax,[0x800]
            0x89, 0x04, 0x25, 0x18, 0x08, 0x00, 0x00, //                         mov [0x818],eax
            0xf4, //                                                             hlt
        ];
        prog[..cpu0.len()].copy_from_slice(&cpu0);
        prog[0x100..0x100 + cpu1.len()].copy_from_slice(&cpu1);

        // Run both vCPUs (CPU1 entered at 0x100) under the given TSO window + scheduler quantum;
        // return (r0, r1).
        let run = |seed_hex: &str, tso: u64, quantum: u64| -> (u64, u64) {
            let s = Seed256::from_hex(seed_hex).unwrap();
            let mut m = X86Machine::new(s, 4096, &prog, 0).unwrap();
            let mut ap = m.cpus[0].clone();
            ap.regs.rip = 0x100;
            ap.sb_rng = s.derive("sb-cpu-1"); // its own store-buffer entropy
            m.cpus.push(ap);
            m.set_smp_sched(quantum);
            m.set_tso(tso);
            m.run(200_000).unwrap();
            (
                m.memory.read_u64_le(0x810).unwrap() & 0xffff_ffff,
                m.memory.read_u64_le(0x818).unwrap() & 0xffff_ffff,
            )
        };
        let seed_n = |i: u64| format!("{i:064x}");

        // Under SC (TSO off): r0==0 && r1==0 is IMPOSSIBLE for every seed/schedule.
        for i in 1..=96 {
            let (r0, r1) = run(&seed_n(i), 0, 3);
            assert!(
                !(r0 == 0 && r1 == 0),
                "SC must forbid the both-stale outcome (seed {i}: r0={r0} r1={r1})"
            );
        }
        // Under x86-TSO: the both-stale outcome IS reachable for some seed, and reproducible.
        let mut found = None;
        for i in 1..=96 {
            if run(&seed_n(i), 50, 3) == (0, 0) {
                found = Some(i);
                break;
            }
        }
        let i = found.expect("x86-TSO must make r0==r1==0 reachable for some seed");
        assert_eq!(run(&seed_n(i), 50, 3), (0, 0), "the TSO outcome is reproducible per seed");
    }

    #[test]
    fn smp_stage3_pct_priority_and_change_points() {
        // SMP Stage 3 (PCT): the same non-atomic load/inc/store RMW race as Stage 1, on two
        // vCPUs. PCT schedules by priority with `depth-1` seed-drawn change points:
        //   - depth 1 (NO change points) = pure priority scheduling: the high-priority vCPU runs
        //     to completion, THEN the other — a fully serialized, race-FREE execution, so the
        //     counter is always exactly 2N (no lost updates) regardless of seed.
        //   - depth 2 (one change point) can preempt a vCPU mid-RMW, so for some seeds an update
        //     is lost (counter < 2N) — and each schedule is reproducible.
        const N: u64 = 20;
        let prog = [
            0x48, 0xc7, 0xc1, N as u8, 0x00, 0x00, 0x00, // mov rcx, N
            0x48, 0x8b, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, // L: mov rax, [0x800]
            0x48, 0xff, 0xc0, //                              inc rax
            0x48, 0x89, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, // mov [0x800], rax
            0x48, 0xff, 0xc9, //                              dec rcx
            0x75, 0xe8, //                                    jnz L (-24)
            0xf4, //                                          hlt
        ];
        let run_counter = |seed_hex: &str, depth: u64, k: u64| -> u64 {
            let s = Seed256::from_hex(seed_hex).unwrap();
            let mut m = X86Machine::new(s, 4096, &prog, 0).unwrap();
            let ap = m.cpus[0].clone();
            m.cpus.push(ap);
            m.set_pct(depth, k);
            m.run(1_000_000).unwrap();
            m.memory.read_u64_le(0x800).unwrap()
        };
        let seed_n = |i: u64| format!("{i:064x}");

        // (1) depth 1 = serialized priority schedule → race-free → exactly 2N for EVERY seed.
        for i in 1..=32 {
            assert_eq!(
                run_counter(&seed_n(i), 1, 0),
                2 * N,
                "PCT depth-1 runs each vCPU to completion (no interleave) → no lost update (seed {i})"
            );
        }

        // (2) Reproducible: same seed + same depth → identical outcome.
        assert_eq!(
            run_counter(&seed_n(7), 2, 400),
            run_counter(&seed_n(7), 2, 400),
            "a PCT schedule is byte-reproducible per seed"
        );

        // (3) depth 2 = one seeded change point → CAN preempt mid-RMW → some seed loses an update.
        let mut saw_race = false;
        for i in 1..=64 {
            let c = run_counter(&seed_n(i), 2, 400);
            assert!((N..=2 * N).contains(&c), "counter {c} in [{N},{}] (seed {i})", 2 * N);
            if c < 2 * N {
                saw_race = true;
            }
        }
        assert!(saw_race, "PCT depth-2 change points must expose the lost-update race for some seed");

        // (4) Off by default: pct_depth 0 leaves the round-robin baseline untouched.
        let base = || {
            let mut m = X86Machine::new(seed(), 4096, &prog, 0).unwrap();
            let ap = m.cpus[0].clone();
            m.cpus.push(ap);
            m.run(1_000_000).unwrap();
            m.full_state_digest()
        };
        assert_eq!(base(), base(), "baseline (PCT off) is deterministic and digest-neutral");
    }

    #[test]
    fn real_mode_16bit_execution() {
        // A CPU in real mode decodes 16-bit instructions (default operand/address size 16, 16-bit
        // ModRM forms) and applies segment bases. Program (16-bit):
        //   mov si, 0x0200 ; mov ax, 0x1234 ; mov [si], ax ; hlt
        let prog = [
            0xbe, 0x00, 0x02, // mov si, 0x0200
            0xb8, 0x34, 0x12, // mov ax, 0x1234
            0x89, 0x04, //       mov [si], ax   (16-bit ModRM rm=4 → [si])
            0xf4, //             hlt
        ];
        let mut m = X86Machine::new(seed(), 0x1000, &prog, 0).unwrap();
        // Drop the BSP into real mode (the bare constructor defaults to long mode).
        m.cpus[0].sys.efer &= !(1 << 10); // clear EFER.LMA
        m.cpus[0].sys.cr0 = 0; // clear CR0.PE
        m.cpus[0].cs = 0; // CS.base = 0
        m.cpus[0].cs_base = 0; // latch the real-mode CS descriptor cache (base 0, 16-bit)
        m.cpus[0].code_bits = 16;
        m.cpus[0].regs.rip = 0;
        assert_eq!(m.cpus[0].cpu_mode(), CpuMode::Real);
        m.run(100).unwrap();
        assert_eq!(m.cpus[0].regs.gpr[RSI] & 0xffff, 0x0200);
        assert_eq!(m.cpus[0].regs.gpr[RAX] & 0xffff, 0x1234);
        assert_eq!(
            m.memory().read_uint_le(0x200, 2).unwrap(),
            0x1234,
            "16-bit `mov [si], ax` stored a word via real-mode addressing"
        );
    }

    #[test]
    fn real_to_protected_mode_transition() {
        // The classic bootstrap sequence: run in 16-bit real mode, load a GDT, set CR0.PE,
        // then `ljmp` into a 32-bit flat code segment — and keep executing 32-bit code there.
        // This exercises the CS-descriptor cache across the *unreal-mode window*: the `ljmp`
        // itself is fetched as 16-bit (CS still cached real-mode) even though PE is already
        // set, and only the CS load latches the 32-bit descriptor.
        //
        // 16-bit real-mode code at 0:
        //   0000: 0F 01 16 00 09   lgdt [0x0900]          (16-bit ModRM [disp16])
        //   0005: 0F 20 C0         mov eax, cr0
        //   0008: 83 C8 01         or  ax, 1              (set PE)
        //   000B: 0F 22 C0         mov cr0, eax
        //   000E: EA 13 00 08 00   ljmp 0x08:0x0013       (ptr16:16 — real-mode encoding)
        // 32-bit protected code at 0x13:
        //   0013: B8 BE BA FE CA   mov eax, 0xCAFEBABE    (32-bit imm, default op-size 32)
        //   0018: A3 00 06 00 00   mov [0x600], eax       (32-bit moffs store)
        //   001D: F4               hlt
        let prog = [
            0x0f, 0x01, 0x16, 0x00, 0x09, // lgdt [0x0900]
            0x0f, 0x20, 0xc0, //             mov eax, cr0
            0x83, 0xc8, 0x01, //             or  ax, 1
            0x0f, 0x22, 0xc0, //             mov cr0, eax
            0xea, 0x13, 0x00, 0x08, 0x00, // ljmp 0x08:0x0013
            0xb8, 0xbe, 0xba, 0xfe, 0xca, // mov eax, 0xCAFEBABE
            0xa3, 0x00, 0x06, 0x00, 0x00, // mov [0x600], eax
            0xf4, //                         hlt
        ];
        let mut m = X86Machine::new(seed(), 0x4000, &prog, 0).unwrap();
        // GDT pseudo-descriptor at 0x0900: limit=0x000F (2 entries = 16 bytes), base=0x1000.
        m.memory_mut()
            .load(0x0900, &[0x0f, 0x00, 0x00, 0x10, 0x00, 0x00])
            .unwrap();
        // GDT at 0x1000: [0]=null, [1]=32-bit flat code (base 0, limit 0xFFFFF, G=1 D=1, P S code).
        m.memory_mut()
            .load(
                0x1000,
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, // null descriptor
                    0xff, 0xff, 0x00, 0x00, 0x00, 0x9a, 0xcf, 0x00, // 0x08: 32-bit code
                ],
            )
            .unwrap();
        // Drop the BSP into real mode (the bare constructor defaults to long mode).
        m.cpus[0].sys.efer &= !(1 << 10); // clear EFER.LMA
        m.cpus[0].sys.cr0 = 0; // clear CR0.PE
        m.cpus[0].cs = 0;
        m.cpus[0].cs_base = 0;
        m.cpus[0].code_bits = 16;
        m.cpus[0].regs.rip = 0;
        assert_eq!(m.cpus[0].cpu_mode(), CpuMode::Real);
        m.run(1000).unwrap();
        assert_eq!(m.cpus[0].cpu_mode(), CpuMode::Protected, "PE set ⇒ protected mode");
        assert_eq!(m.cpus[0].cs, 0x08, "CS reloaded by the far jump");
        assert_eq!(m.cpus[0].code_bits, 32, "latched a 32-bit code segment");
        assert_eq!(m.cpus[0].regs.gpr[RAX] & 0xffff_ffff, 0xCAFEBABE);
        assert_eq!(
            m.memory().read_uint_le(0x600, 4).unwrap(),
            0xCAFEBABE,
            "32-bit protected-mode code executed after the transition"
        );
    }

    #[test]
    fn real_mode_stack_push_pop() {
        // Real-mode stack: 16-bit push/pop wrap SP within 16 bits and address through SS.base.
        // With SS=0x2000 (base 0x20000) and SP=0x100, `push ax` writes 2 bytes at 0x200FE.
        //   mov ax, 0xBEEF ; push ax ; pop bx ; hlt
        let prog = [
            0xb8, 0xef, 0xbe, // mov ax, 0xBEEF
            0x50, //             push ax
            0x5b, //             pop bx
            0xf4, //             hlt
        ];
        let mut m = X86Machine::new(seed(), 0x30000, &prog, 0).unwrap();
        m.cpus[0].sys.efer &= !(1 << 10);
        m.cpus[0].sys.cr0 = 0;
        m.cpus[0].cs = 0;
        m.cpus[0].cs_base = 0;
        m.cpus[0].code_bits = 16;
        m.cpus[0].ss = 0x2000; // SS.base = 0x20000
        m.cpus[0].regs.gpr[RSP] = 0x100;
        m.cpus[0].regs.rip = 0;
        m.run(1000).unwrap();
        assert_eq!(m.cpus[0].regs.gpr[RBX] & 0xffff, 0xBEEF, "pop recovered the pushed word");
        assert_eq!(m.cpus[0].regs.gpr[RSP] & 0xffff, 0x100, "SP back to its start");
        assert_eq!(
            m.memory().read_uint_le(0x20000 + 0xfe, 2).unwrap(),
            0xBEEF,
            "the word landed at SS.base + (SP-2)"
        );
    }

    #[test]
    fn smp_ap_startup_via_sipi() {
        // SMP Stage 0 AP bring-up: the BSP writes the LAPIC ICR to send a Startup-IPI (SIPI) to
        // AP apic_id=1 targeting page 0x10 (physical entry 0x10000). The AP — parked waiting for
        // its SIPI — wakes there and writes a marker, exercising the full path:
        //   ICR MMIO write → take_ipi → deliver_ipi → SIPI → AP unpark → AP runs.
        let bsp = [
            0xbb, 0x10, 0x03, 0xe0, 0xfe, // mov ebx, 0xFEE00310   (ICR high)
            0xb8, 0x00, 0x00, 0x00, 0x01, // mov eax, 0x01000000   (destination APIC ID 1)
            0x89, 0x03, //                   mov [rbx], eax
            0xbb, 0x00, 0x03, 0xe0, 0xfe, // mov ebx, 0xFEE00300   (ICR low)
            0xb8, 0x10, 0x06, 0x00, 0x00, // mov eax, 0x610        (SIPI: mode 6, vector 0x10)
            0x89, 0x03, //                   mov [rbx], eax        → issues the SIPI
            0xf4, //                         hlt
        ];
        let mut m = X86Machine::new(seed(), 0x20000, &bsp, 0).unwrap();
        // Bring up a second vCPU (AP apic_id=1), parked waiting for its SIPI.
        m.set_vcpus(2);
        assert_eq!(m.cpus.len(), 2);
        assert!(m.cpus[1].wait_sipi && m.cpus[1].apic_id == 1);
        // AP trampoline at physical 0x10000 — REAL 16-bit code (the AP runs in real mode):
        //   mov word [0x100], 0xABCD ; hlt   (C7 /0 with 16-bit ModRM mod=00 rm=6 = [disp16])
        let tramp = [0xc7, 0x06, 0x00, 0x01, 0xcd, 0xab, 0xf4];
        m.memory_mut().load(0x10000, &tramp).unwrap();
        m.run(100_000).unwrap();
        assert!(!m.cpus[1].wait_sipi, "SIPI must unpark the AP");
        assert!(m.cpus[1].halted, "AP ran its trampoline to the HLT");
        assert_eq!(m.cpus[1].cpu_mode(), CpuMode::Real, "AP starts in real mode");
        assert_eq!(
            m.memory().read_uint_le(0x100, 2).unwrap(),
            0xABCD,
            "AP executed 16-bit real-mode startup code at CS.base = vector<<12"
        );
    }

    #[test]
    fn payne_hanek_large_arg_trig() {
        // 3d-4: large-argument sin/cos vs an accurate reference (libm, which does correct
        // Payne–Hanek reduction). (x_bits, sin_bits, cos_bits).
        let refs: [(u64, u64, u64); 13] = [
            (0x416312d000000000, 0x3fdaea414a8a3352, 0xbfed085be7a8f4a1),
            (0x4202a05f20000000, 0xbfdf334c7896a4e3, 0x3febf098901c931a),
            (0x430c6bf526340000, 0x3feb76f88136ceba, 0xbfe06c154609d33e),
            (0x4415af1d78b58c40, 0xbfe4a5e605fd6450, 0x3fe872720fc60d3d),
            (0x46293e5939a08cea, 0x3f831c608f107767, 0xbfefffa4b11f1b45),
            (0x54b249ad2594c37d, 0xbfd85c5e5b929359, 0x3fed9757496841f5),
            (0x6974e718d7d7625a, 0xbfe49b644938c64c, 0x3fe87b4df51f679e),
            (0x7e37e43c8800759c, 0xbfea2c16b010e385, 0xbfe2699022adc4c1),
            (0x43b0000000000000, 0xbfea94adab06665c, 0xbfe1d146047d6947),
            (0x4c70000000000000, 0xbfdea642f4f78780, 0xbfec17843662ee23),
            (0x7830000000000000, 0x3fefeb394f48dbe5, 0x3fb23884a86c3540),
            (0xc415af1d78b58c40, 0x3fe4a5e605fd6450, 0x3fe872720fc60d3d),
            (0xc3c8493fba64ef00, 0xbfedd5433aee8144, 0x3fd72694242f4cd4),
        ];
        for (xb, sb, cb) in refs {
            let x = f64::from_bits(xb);
            let (s, c) = soft_sincos(x);
            assert!((s - f64::from_bits(sb)).abs() < 1e-11, "sin({x:e}): {s} vs {}", f64::from_bits(sb));
            assert!((c - f64::from_bits(cb)).abs() < 1e-11, "cos({x:e}): {c} vs {}", f64::from_bits(cb));
            // sin²+cos² = 1 invariant.
            assert!((s * s + c * c - 1.0).abs() < 1e-10, "pythag {x:e}");
            // determinism: identical on re-evaluation.
            assert_eq!(soft_sincos(x), (s, c));
        }
        // continuity across the Cody–Waite / Payne–Hanek threshold (2^20): both paths agree.
        let (sa, ca) = soft_sincos(1_048_575.9); // Cody–Waite side
        let (sb, cb) = soft_sincos(1_048_576.1); // Payne–Hanek side
        assert!((sa * sa + ca * ca - 1.0).abs() < 1e-10);
        assert!((sb * sb + cb * cb - 1.0).abs() < 1e-10);
    }

    #[test]
    fn x87_transcendentals_deterministic() {
        use std::f64::consts::PI;
        let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol.max(b.abs() * tol);
        // log2 exact on powers of two, accurate elsewhere.
        assert_eq!(soft_log2(8.0), 3.0);
        assert_eq!(soft_log2(1024.0), 10.0);
        assert!(close(soft_log2(10.0), 3.321928094887362, 1e-13));
        // 2^x − 1 (F2XM1).
        assert!(close(soft_exp2m1(1.0), 1.0, 1e-13));
        assert!(close(soft_exp2m1(0.0), 0.0, 1e-15));
        assert!(close(soft_exp2m1(0.5), 2f64.sqrt() - 1.0, 1e-13));
        // sin/cos at known angles.
        let (s0, c0) = soft_sincos(0.0);
        assert_eq!((s0, c0), (0.0, 1.0));
        let (s6, c6) = soft_sincos(PI / 6.0);
        assert!(close(s6, 0.5, 1e-13) && close(c6, 3f64.sqrt() / 2.0, 1e-13));
        let (s4, _) = soft_sincos(PI / 4.0);
        assert!(close(s4, 0.5f64.sqrt(), 1e-13));
        // atan2 quadrants.
        assert!(close(soft_atan2(1.0, 1.0), PI / 4.0, 1e-13));
        assert!(close(soft_atan2(1.0, -1.0), 3.0 * PI / 4.0, 1e-13));
        assert!(close(soft_atan2(-1.0, 0.0), -PI / 2.0, 1e-13));
        // Determinism: identical on recomputation (pure f64, no host libm).
        assert_eq!(soft_sincos(1.2345), soft_sincos(1.2345));
        assert_eq!(soft_log2(3.7), soft_log2(3.7));
    }

    #[test]
    fn x87_rounding_control() {
        // x87 arith now runs in true 80-bit; FCW PC/RC honoured. Compare at PC=53 vs f64.
        let f = |x: f64| F80::from_f64(x);
        let n = (1.0f64 / 3.0).to_bits();
        assert_eq!(fpu_apply80(FpuBin::Div, f(1.0), f(3.0), 2, 0).to_f64(), 1.0 / 3.0);
        assert_eq!(fpu_apply80(FpuBin::Div, f(1.0), f(3.0), 2, 2).to_f64().to_bits(), n + 1); // +inf
        assert_eq!(fpu_apply80(FpuBin::Div, f(1.0), f(3.0), 2, 1).to_f64().to_bits(), n); // −inf floor
        assert_eq!(fpu_apply80(FpuBin::SubR, f(2.0), f(5.0), 2, 0).to_f64(), 3.0); // 5 − 2
        assert_eq!(fpu_apply80(FpuBin::DivR, f(3.0), f(1.0), 2, 0).to_f64(), 1.0 / 3.0); // 1 / 3
        // 80-bit MUL keeps a cross term a chained f64 would lose (PC=64).
        let a = f(1.0).add(f(2f64.powi(-60)), 3, 0);
        assert_ne!(a.mul(a, 3, 0).mant, 0x8000_0000_0000_0000);
    }

    #[test]
    fn mxcsr_directed_rounding_div_sqrt() {
        // RC: 0=nearest, 1=−inf, 2=+inf, 3=toward-0. Default (0) must be a no-op.
        // f64 1/3: nearest rounds DOWN (n < 1/3) → down=nearest, up=nextUp, toward0=nearest.
        let n64 = (1.0f64 / 3.0).to_bits();
        assert_eq!(fbin_rounded_f64(FBinKind::Div, 1.0, 3.0, 0).0, n64);
        assert_eq!(fbin_rounded_f64(FBinKind::Div, 1.0, 3.0, 1).0, n64); // −inf: floor = n
        assert_eq!(fbin_rounded_f64(FBinKind::Div, 1.0, 3.0, 2).0, n64 + 1); // +inf: nextUp
        assert_eq!(fbin_rounded_f64(FBinKind::Div, 1.0, 3.0, 3).0, n64); // toward 0 (pos) = n
        // Inexact ops flag PE.
        assert_eq!(fbin_rounded_f64(FBinKind::Div, 1.0, 3.0, 0).1, FE_PE);
        // f32 1/3: nearest rounds UP (n > 1/3) → down=nextDown, up=nearest, toward0=nextDown.
        let n32 = (1.0f32 / 3.0).to_bits();
        assert_eq!(fbin_rounded_f32(FBinKind::Div, 1.0, 3.0, 0).0, n32);
        assert_eq!(fbin_rounded_f32(FBinKind::Div, 1.0, 3.0, 1).0, n32 - 1); // −inf: nextDown
        assert_eq!(fbin_rounded_f32(FBinKind::Div, 1.0, 3.0, 2).0, n32); // +inf: ceil = n
        assert_eq!(fbin_rounded_f32(FBinKind::Div, 1.0, 3.0, 3).0, n32 - 1); // toward 0 = nextDown
        // Exact op: identical in every mode, no exceptions.
        for rc in 0..4 {
            assert_eq!(fbin_rounded_f64(FBinKind::Add, 1.0, 2.0, rc), (3.0f64.to_bits(), 0));
            assert_eq!(fbin_rounded_f64(FBinKind::Mul, 2.0, 4.0, rc), (8.0f64.to_bits(), 0));
        }
        // Exception flags: IE on invalid, ZE on divide-by-zero.
        assert_eq!(fbin_rounded_f64(FBinKind::Sub, f64::INFINITY, f64::INFINITY, 0).1, FE_IE);
        assert_eq!(fbin_rounded_f64(FBinKind::Div, 1.0, 0.0, 0).1, FE_ZE);
        assert_eq!(sqrt_rounded_f64(-4.0, 0).1, FE_IE);
        // sqrt(2) f64: nearest, and directed neighbours differ by ≤1 ulp, bracketing.
        let s = 2.0f64.sqrt().to_bits();
        assert_eq!(sqrt_rounded_f64(2.0, 0).0, s);
        let d = sqrt_rounded_f64(2.0, 1).0;
        let u = sqrt_rounded_f64(2.0, 2).0;
        assert!(d == s || d == s - 1);
        assert!(u == s || u == s + 1);
        assert!(f64::from_bits(d) <= f64::from_bits(s) && f64::from_bits(s) <= f64::from_bits(u));
        assert_eq!(sqrt_rounded_f64(4.0, 3).0, 2.0f64.to_bits()); // exact
    }

    #[test]
    fn div_subnormal_directed_rounding() {
        // 3d: div-by-subnormal directed rounding + the PE flag it was missing. div_resid_sign
        // returns sign(x − n·y), exact even when the fused residual underflows.
        assert_eq!(div_resid_sign(1.0, 3.0, 1.0 / 3.0), 1.0); // f64 1/3 rounds DOWN → n·3 < 1
        assert_eq!(div_resid_sign(1.0, 2.0, 0.5), 0.0); // exact → zero residual
        // Smallest-subnormal dividend ÷ a non-power-of-two subnormal → the exact case that broke
        // (ex far from n·y's exponent). Quotient is a large-ish NORMAL, inexact → PE MUST fire.
        let tiny = f64::from_bits(1); // 2^-1074
        let y = f64::from_bits(0x0004_0000_0000_0007); // subnormal, not a power of two
        let (bits0, exc0) = fbin_rounded_f64(FBinKind::Div, tiny, y, 0);
        assert_eq!(bits0, (tiny / y).to_bits()); // rc=0 == host round-nearest
        assert_ne!(exc0 & FE_PE, 0); // inexact → PE (the previously-missing flag)
        // directed neighbours bracket the nearest result by ≤1 ulp, monotone.
        let dn = fbin_rounded_f64(FBinKind::Div, tiny, y, 1).0; // toward −inf
        let up = fbin_rounded_f64(FBinKind::Div, tiny, y, 2).0; // toward +inf
        assert!(dn == bits0 || dn == bits0 - 1);
        assert!(up == bits0 || up == bits0 + 1);
        assert!(f64::from_bits(dn) <= f64::from_bits(bits0) && f64::from_bits(bits0) <= f64::from_bits(up));
    }

    #[test]
    fn mov_imm_and_add_registers() {
        // mov rax, 7 ; mov rbx, 35 ; add rax, rbx ; hlt
        let program = [
            0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
            0x48, 0xc7, 0xc3, 0x23, 0x00, 0x00, 0x00, // mov rbx, 35
            0x48, 0x01, 0xd8, // add rax, rbx
            0xf4, // hlt
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 42);
        assert_eq!(m.cpu().regs.gpr[RBX], 35);
        assert!(m.cpu().halted);
    }

    #[test]
    fn sum_loop_with_conditional_branch() {
        // Sum 1..=5 into rax using a dec/jnz loop.
        //   xor rax, rax        ; 48 31 c0
        //   mov rcx, 5          ; 48 c7 c1 05 00 00 00
        // loop:
        //   add rax, rcx        ; 48 01 c8
        //   dec rcx             ; 48 ff c9
        //   jnz loop            ; 75 f8   (rel8 = -8)
        //   hlt                 ; f4
        let program = [
            0x48, 0x31, 0xc0, // xor rax, rax
            0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
            0x48, 0x01, 0xc8, // add rax, rcx   (loop start @ offset 10)
            0x48, 0xff, 0xc9, // dec rcx
            0x75, 0xf8, // jnz -8 -> back to add
            0xf4, // hlt
        ];
        let m = run(&program, 1000);
        assert_eq!(m.cpu().regs.gpr[RAX], 15); // 5+4+3+2+1
        assert_eq!(m.cpu().regs.gpr[RCX], 0);
    }

    #[test]
    fn push_pop_round_trips_through_memory() {
        // mov rax, 0x1234 ; push rax ; pop rbx ; hlt
        let program = [
            0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, // mov rax, 0x1234
            0x50, // push rax
            0x5b, // pop rbx
            0xf4, // hlt
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x1234);
        // Stack pointer restored.
        assert_eq!(m.cpu().regs.gpr[RSP], 4096);
    }

    #[test]
    fn mem_store_and_load_via_modrm() {
        //   mov rax, 0xdeadbeef
        //   mov [0x800], rax     ; 48 89 04 25 00 08 00 00
        //   mov rbx, [0x800]     ; 48 8b 1c 25 00 08 00 00
        //   hlt
        let program = [
            0x48, 0xc7, 0xc0, 0xef, 0xbe, 0xad, 0xde, // mov rax, 0xdeadbeef (sign-ext imm32)
            0x48, 0x89, 0x04, 0x25, 0x00, 0x08, 0x00, 0x00, // mov [0x800], rax
            0x48, 0x8b, 0x1c, 0x25, 0x00, 0x08, 0x00, 0x00, // mov rbx, [0x800]
            0xf4,
        ];
        let m = run(&program, 100);
        let expected = 0xffff_ffff_dead_beef; // imm32 sign-extended into rax
        assert_eq!(m.cpu().regs.gpr[RAX], expected);
        assert_eq!(m.cpu().regs.gpr[RBX], expected);
        assert_eq!(m.memory().read_uint_le(0x800, 8).unwrap(), expected);
    }

    #[test]
    fn dword_write_zero_extends() {
        // mov eax, 0xffffffff (b8 imm32) then check upper 32 bits cleared
        let program = [
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0x7f, // mov rax, 0x7fffffff
            0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0  -> zero-extends, clears rax
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0);
    }

    #[test]
    fn execution_is_deterministic() {
        let program = [
            0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc3, 0x23, 0x00, 0x00, 0x00,
            0x48, 0x01, 0xd8, 0xf4,
        ];
        let mut a = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        let mut b = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    // --- flag-table coverage (cpu-determinism-contract section 3) ---

    fn flags_after(program: &[u8]) -> u64 {
        run(program, 100).cpu().regs.rflags
    }

    #[test]
    fn add_sets_carry_and_zero() {
        // mov rax, -1 ; add rax, 1 -> result 0, CF=1, ZF=1, OF=0
        let program = [
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax, -1 (sign-ext)
            0x48, 0x83, 0xc0, 0x01, // add rax, 1
            0xf4,
        ];
        let f = flags_after(&program);
        assert!(flags::get_flag(f, CF), "CF set on wrap");
        assert!(flags::get_flag(f, ZF), "ZF set on zero result");
        assert!(!flags::get_flag(f, OF), "OF clear");
    }

    #[test]
    fn add_sets_signed_overflow() {
        // mov eax, 0x7fffffff ; add eax, 1 -> OF=1, SF=1 (signed overflow)
        let program = [
            0xb8, 0xff, 0xff, 0xff, 0x7f, // mov eax, 0x7fffffff
            0x83, 0xc0, 0x01, // add eax, 1
            0xf4,
        ];
        let f = flags_after(&program);
        assert!(flags::get_flag(f, OF), "OF set on signed overflow");
        assert!(flags::get_flag(f, SF), "SF set (result negative)");
        assert!(!flags::get_flag(f, CF), "CF clear (no unsigned carry)");
    }

    #[test]
    fn logic_clears_carry_overflow_and_undefined_af() {
        // mov rax, 0xff ; and rax, 0x0f -> CF=0, OF=0, AF=0 (undefined->0), PF even
        let program = [
            0x48, 0xc7, 0xc0, 0xff, 0x00, 0x00, 0x00, // mov rax, 0xff
            0x48, 0x83, 0xe0, 0x0f, // and rax, 0x0f
            0xf4,
        ];
        let f = flags_after(&program);
        assert!(!flags::get_flag(f, CF));
        assert!(!flags::get_flag(f, OF));
        assert!(!flags::get_flag(f, flags::AF), "AF pinned to 0 per contract");
    }

    #[test]
    fn movzx_and_movsx_extend_correctly() {
        // mov al, 0xff ; movzx rcx, al ; movsx rdx, al ; hlt
        let program = [
            0xb0, 0xff, // mov al, 0xff
            0x48, 0x0f, 0xb6, 0xc8, // movzx rcx, al
            0x48, 0x0f, 0xbe, 0xd0, // movsx rdx, al
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RCX], 0xff);
        assert_eq!(m.cpu().regs.gpr[RDX], 0xffff_ffff_ffff_ffff);
    }

    fn try_run(program: &[u8]) -> Result<X86Machine, X86Error> {
        let mut m = X86Machine::new(seed(), 4096, program, 0)?;
        m.run(100)?;
        Ok(m)
    }

    #[test]
    fn virtio_blk_read_via_guest_mmio() {
        // Guest programs the virtio-mmio block device (base 0xD0000000): page
        // size, queue num/align/pfn, then QueueNotify. The queue + a read request
        // are laid out in memory; the device DMAs sector 1 into the data buffer.
        let program = [
            0x48, 0xb8, 0x00, 0x00, 0x00, 0xd0, 0x00, 0x00, 0x00, 0x00, // mov rax,0xD0000000
            0xc7, 0x40, 0x28, 0x00, 0x10, 0x00, 0x00, // [rax+0x28]=0x1000 (page size)
            0xc7, 0x40, 0x38, 0x08, 0x00, 0x00, 0x00, // [rax+0x38]=8 (queue num)
            0xc7, 0x40, 0x3c, 0x00, 0x10, 0x00, 0x00, // [rax+0x3c]=0x1000 (align)
            0xc7, 0x40, 0x40, 0x01, 0x00, 0x00, 0x00, // [rax+0x40]=1 (pfn)
            0xc7, 0x40, 0x50, 0x00, 0x00, 0x00, 0x00, // [rax+0x50]=0 (notify)
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 0x20000, &program, 0).unwrap();

        let mut disk = vec![0_u8; 4096];
        disk[512] = 0xab; // sector 1
        disk[513] = 0xcd;
        m.io_mut().virtio = VirtioBlk::new(disk);

        {
            let mem = m.memory_mut();
            let desc_base = 0x1000_u64;
            let avail_base = desc_base + 8 * 16;
            mem.write_uint_le(0x4000, 4, 0).unwrap(); // type IN (read)
            mem.write_uint_le(0x4008, 8, 1).unwrap(); // sector 1
            let wd = |mem: &mut GuestMemory, i: u64, addr: u64, len: u32, flags: u16, next: u16| {
                let a = desc_base + i * 16;
                mem.write_uint_le(a, 8, addr).unwrap();
                mem.write_uint_le(a + 8, 4, len as u64).unwrap();
                mem.write_uint_le(a + 12, 2, flags as u64).unwrap();
                mem.write_uint_le(a + 14, 2, next as u64).unwrap();
            };
            wd(mem, 0, 0x4000, 16, 1, 1); // header (NEXT)
            wd(mem, 1, 0x4100, 16, 1 | 2, 2); // data (NEXT|WRITE)
            wd(mem, 2, 0x4200, 1, 2, 0); // status (WRITE)
            mem.write_uint_le(avail_base + 4, 2, 0).unwrap(); // ring[0] = head 0
            mem.write_uint_le(avail_base + 2, 2, 1).unwrap(); // avail.idx = 1
        }

        m.run(100).unwrap();
        assert_eq!(m.memory().read_uint_le(0x4100, 2).unwrap(), 0xcdab, "disk DMA'd in");
        assert_eq!(m.memory().read_uint_le(0x4200, 1).unwrap(), 0, "status OK");
    }

    #[test]
    fn lapic_timer_programmed_by_guest_drives_handler() {
        // Guest programs the Local APIC (MMIO at 0xFEE00000): divide-by-1,
        // periodic LVT timer vector 0x40, initial count 50; then sti; loop hlt.
        // Handler @0x100 increments r15. Background core timer (vec 32) -> iret.
        let mut program = vec![
            0x48, 0xb8, 0x00, 0x00, 0xe0, 0xfe, 0x00, 0x00, 0x00, 0x00, // mov rax,0xFEE00000
            0xb9, 0x0b, 0x00, 0x00, 0x00, // mov ecx,0x0b (divide by 1)
            0x89, 0x88, 0xe0, 0x03, 0x00, 0x00, // mov [rax+0x3e0],ecx
            0xb9, 0x40, 0x00, 0x02, 0x00, // mov ecx,0x20040 (periodic|vec 0x40)
            0x89, 0x88, 0x20, 0x03, 0x00, 0x00, // mov [rax+0x320],ecx
            0xb9, 0x32, 0x00, 0x00, 0x00, // mov ecx,50
            0x89, 0x88, 0x80, 0x03, 0x00, 0x00, // mov [rax+0x380],ecx  (arms timer)
            0xfb, // sti
            0xf4, // hlt   (loop target)
            0xeb, 0xfd, // jmp -3
        ];
        program.resize(0x100, 0x90);
        program.extend([0x49, 0xff, 0xc7, 0xcf]); // inc r15 ; iret  @0x100
        program.resize(0x120, 0x90);
        program.extend([0xcf]); // iret @0x120 (background core-timer handler)

        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        {
            let mem = m.memory_mut();
            set_idt_gate(mem, 0x4000, 0x40, 0x100);
            set_idt_gate(mem, 0x4000, 32, 0x120);
        }
        m.cpu_mut().sys.idtr_base = 0x4000;
        m.cpu_mut().sys.idtr_limit = 0xfff;

        for _ in 0..300 {
            m.step().unwrap();
        }
        assert!(
            m.cpu().regs.gpr[R15] > 0,
            "LAPIC timer fired and ran its handler"
        );
    }

    #[test]
    fn movq_gpr_xmm_round_trip() {
        // mov rax, 0x1122334455667788 ; movq xmm0,rax ; movq rbx,xmm0 ; hlt
        let program = [
            0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, imm64
            0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
            0x66, 0x48, 0x0f, 0x7e, 0xc3, // movq rbx, xmm0
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x1122_3344_5566_7788);
        assert_eq!(m.cpu().xmm[0] as u64, 0x1122_3344_5566_7788);
        assert_eq!(m.cpu().xmm[0] >> 64, 0, "MOVQ zero-extends upper 64");
    }

    #[test]
    fn cvt_int_float_round_trip_and_pcmpeqb() {
        // CVTSI2SD 42 ; CVTTSD2SI -> 42
        let cvt = run(
            &[0x48, 0xc7, 0xc0, 0x2a, 0, 0, 0, 0xf2, 0x48, 0x0f, 0x2a, 0xc0, 0xf2, 0x48, 0x0f, 0x2c, 0xc0, 0xf4],
            100,
        );
        assert_eq!(cvt.cpu().regs.gpr[RAX], 42);
        // PXOR xmm0,xmm0 -> zero, read back
        let px = run(
            &[
                0x48, 0xb8, 0xef, 0xbe, 0xad, 0xde, 0, 0, 0, 0, // mov rax, 0xdeadbeef
                0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0,rax
                0x66, 0x0f, 0xef, 0xc0, // pxor xmm0,xmm0
                0x66, 0x48, 0x0f, 0x7e, 0xc3, // movq rbx,xmm0
                0xf4,
            ],
            100,
        );
        assert_eq!(px.cpu().regs.gpr[RBX], 0, "PXOR self -> 0");
    }

    #[test]
    fn interrupt_from_cpl3_switches_stack_and_restores_cpl_on_iret() {
        // main: int 0x40 ; mov rbx,0x1234 ; hlt
        // handler@0x200: mov [0x9000], rsp ; iret
        let mut program = vec![
            0xcd, 0x40, // int 0x40
            0x48, 0xc7, 0xc3, 0x34, 0x12, 0x00, 0x00, // mov rbx, 0x1234
            0xf4, // hlt
        ];
        program.resize(0x200, 0x90);
        program.extend([
            0x48, 0x89, 0x24, 0x25, 0x00, 0x90, 0x00, 0x00, // mov [0x9000], rsp
            0xcf, // iretq
        ]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        set_idt_gate(m.memory_mut(), 0x4000, 0x40, 0x200);
        let c = m.cpu_mut();
        c.sys.idtr_base = 0x4000;
        c.sys.idtr_limit = 0xfff;
        c.sys.tss_rsp0 = 0x8000; // kernel stack
        c.cpl = 3; // start in user mode
        c.cs = 0x33; // user CS (RPL 3)
        c.ss = 0x2b;
        c.regs.gpr[RSP] = 0x6000; // user stack

        m.run(100).unwrap();

        // The handler ran on the kernel stack (TSS.RSP0 region), not the user one.
        let kstack = m.memory().read_uint_le(0x9000, 8).unwrap();
        assert!(
            (0x7000..0x8000).contains(&kstack),
            "interrupt switched to the kernel stack (rsp={kstack:#x})"
        );
        // IRET returned to user mode (CPL 3) and the post-INT instruction ran there.
        assert_eq!(m.cpu().cpl, 3, "IRET restored CPL 3 from the saved CS RPL");
        assert_eq!(m.cpu().regs.gpr[RBX], 0x1234);
        assert_eq!(m.cpu().regs.gpr[RSP], 0x6000, "user RSP restored by IRET");
    }

    #[test]
    fn guest_reads_rtc_and_pit_through_pio() {
        // RTC: select the seconds register (0), read it -> the frozen epoch's
        // seconds field, 0x20 in BCD (2023-11-14T22:13:20Z).
        let rtc = run(
            &[
                0xb0, 0x00, // mov al, 0
                0xe6, 0x70, // out 0x70, al   (select seconds)
                0xe4, 0x71, // in  al, 0x71
                0xf4,
            ],
            100,
        );
        assert_eq!(rtc.cpu().regs.gpr[RAX] & 0xff, 0x20, "RTC seconds = 0x20 BCD");

        // PIT: program ch0 (lo/hi, mode 2), latch it, read the low byte. The device
        // responds (not the 0xff floating-bus default on both bytes).
        let pit = run(
            &[
                0xb0, 0x34, 0xe6, 0x43, // mov al,0x34 ; out 0x43,al  (ch0 lo/hi mode2)
                0xb0, 0x00, 0xe6, 0x43, // mov al,0    ; out 0x43,al  (latch ch0)
                0xe4, 0x40, // in al, 0x40  (latched low byte)
                0x88, 0xc3, // mov bl, al
                0xe4, 0x40, // in al, 0x40  (latched high byte)
                0xf4,
            ],
            100,
        );
        // Reconstruct the 16-bit latched count; it has counted down below 0xFFFF.
        let count = (pit.cpu().regs.gpr[RBX] & 0xff) | ((pit.cpu().regs.gpr[RAX] & 0xff) << 8);
        assert!(pit.cpu().halted);
        assert!(count <= 0xffff, "PIT returned a real latched count ({count:#x})");
    }

    #[test]
    fn bt_memory_with_register_index_uses_bit_displacement() {
        // bit 72 lives in byte 9 (72/8); BT must address that byte, not mask to 64.
        let program = [
            0x48, 0xc7, 0xc1, 0x48, 0x00, 0x00, 0x00, // mov rcx, 72
            0x48, 0x0f, 0xa3, 0x0c, 0x25, 0x00, 0x20, 0x00, 0x00, // bt [0x2000], rcx
            0x0f, 0x92, 0xc0, // setc al
            0x48, 0xc7, 0xc2, 0x50, 0x00, 0x00, 0x00, // mov rdx, 80
            0x48, 0x0f, 0xab, 0x14, 0x25, 0x00, 0x20, 0x00, 0x00, // bts [0x2000], rdx
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        m.memory_mut().write_uint_le(0x2009, 1, 0x01).unwrap(); // byte 9, bit0 set
        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RAX] & 0xff, 1, "BT bit 72 (byte 9, bit0) -> CF");
        assert_eq!(
            m.memory().read_uint_le(0x200a, 1).unwrap(),
            0x01,
            "BTS bit 80 set byte 10, bit0"
        );
    }

    #[test]
    fn packed_arith_lanes_wrap_independently() {
        // PADDB: 0xFF + 0x01 wraps to 0x00 within the byte (no carry into next).
        let a = 0x01ff_01ff_01ff_01ff_u128;
        let b = 0x0101_0101_0101_0101_u128;
        assert_eq!(packed_arith(a, b, 1, false), 0x0200_0200_0200_0200);
        // PADDW: carry crosses bytes but stays within the 16-bit lane.
        assert_eq!(packed_arith(0x0000_00ff_u128, 0x0000_0001_u128, 2, false), 0x0000_0100);
        // PADDD lane independence: low lane overflows, high lane unaffected.
        assert_eq!(
            packed_arith(0x0000_0001_ffff_ffff_u128, 0x0000_0001_ffff_ffff_u128, 4, false),
            0x0000_0002_ffff_fffe
        );
        // PADDQ: full 64-bit lane add wraps to 0.
        assert_eq!(packed_arith(0xffff_ffff_ffff_ffff_u128, 1, 8, false), 0);
        // PSUBB: 0x00 - 0x01 wraps to 0xFF within the byte.
        assert_eq!(packed_arith(0, 0x0000_0001_u128, 1, true), 0x00ff);
        // PSUBD: two independent borrows, no cross-lane.
        assert_eq!(
            packed_arith(0x0000_0001_0000_0000_u128, 0x0000_0002_0000_0001_u128, 4, true),
            0xffff_ffff_ffff_ffff
        );
    }

    #[test]
    fn packed_compare_eq_and_gt_lanes() {
        // PCMPEQW: per-word equality. The upper 64 bits of both inputs are 0, so
        // those four word-lanes compare equal → all-ones.
        assert_eq!(
            packed_compare(0x0001_0002_0003_0004, 0x0001_9999_0003_9999, 2, false),
            0xffff_ffff_ffff_ffff_ffff_0000_ffff_0000_u128
        );
        // PCMPGTB signed: 1 > 0 true; 0xFF (=-1) > 0 false.
        assert_eq!(packed_compare(0x01, 0x00, 1, true), 0xff);
        assert_eq!(packed_compare(0xff, 0x00, 1, true), 0x00);
        // PCMPGTD: 5 > 3 in the low dword lane.
        assert_eq!(packed_compare(0x0000_0005, 0x0000_0003, 4, true), 0xffff_ffff);
    }

    #[test]
    fn packed_shift_logical_and_arithmetic_lanes() {
        assert_eq!(
            packed_shift(0x0000_0001_0000_0001, 4, 4, ShiftKind::Sll),
            0x0000_0010_0000_0010
        );
        assert_eq!(
            packed_shift(0xff00_0000_0000_0000, 8, 8, ShiftKind::Srl),
            0x00ff_0000_0000_0000
        );
        // PSRAW of a negative word fills with ones.
        assert_eq!(packed_shift(0x8000, 4, 2, ShiftKind::Sra), 0xf800);
        // count ≥ width: logical → 0, arithmetic negative → all ones.
        assert_eq!(packed_shift(0x1234_5678, 64, 4, ShiftKind::Srl), 0);
        assert_eq!(packed_shift(0x8000_0000, 64, 4, ShiftKind::Sra), 0xffff_ffff);
    }

    #[test]
    fn packed_unpack_low_and_high() {
        // PUNPCKLBW: low byte of a then low byte of b.
        assert_eq!(packed_unpack(0xaa, 0xbb, 1, false) & 0xffff, 0xbbaa);
        // PUNPCKLQDQ: low 64 of a, then low 64 of b.
        assert_eq!(
            packed_unpack(0x1111_1111_1111_1111, 0x2222_2222_2222_2222, 8, false),
            0x2222_2222_2222_2222_1111_1111_1111_1111_u128
        );
        // PUNPCKHQDQ: high 64 of a, then high 64 of b.
        let a = 0xaaaa_aaaa_aaaa_aaaa_0000_0000_0000_0000_u128;
        let b = 0xbbbb_bbbb_bbbb_bbbb_0000_0000_0000_0000_u128;
        assert_eq!(
            packed_unpack(a, b, 8, true),
            0xbbbb_bbbb_bbbb_bbbb_aaaa_aaaa_aaaa_aaaa_u128
        );
    }

    #[test]
    fn pslld_imm_and_pshufd_execute_end_to_end() {
        // movq xmm0,rax ; pslld xmm0, 4 ; movq rax,xmm0
        let shl = run(
            &[
                0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // rax=0x100000001
                0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0,rax
                0x66, 0x0f, 0x72, 0xf0, 0x04, // pslld xmm0, 4
                0x66, 0x48, 0x0f, 0x7e, 0xc0, // movq rax,xmm0
                0xf4,
            ],
            100,
        );
        assert_eq!(shl.cpu().regs.gpr[RAX], 0x0000_0010_0000_0010);

        // pshufd xmm0,xmm0,0x00 -> broadcast dword lane 0 to all lanes.
        let shuf = run(
            &[
                0x48, 0xb8, 0x22, 0x22, 0x22, 0x22, 0x11, 0x11, 0x11, 0x11, // rax=0x1111111122222222
                0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0,rax
                0x66, 0x0f, 0x70, 0xc0, 0x00, // pshufd xmm0,xmm0,0
                0x66, 0x48, 0x0f, 0x7e, 0xc0, // movq rax,xmm0
                0xf4,
            ],
            100,
        );
        assert_eq!(shuf.cpu().regs.gpr[RAX], 0x2222_2222_2222_2222);
    }

    #[test]
    fn paddb_psubq_execute_end_to_end() {
        // movq xmm0,rax ; movq xmm1,rbx ; paddb xmm0,xmm1 ; movq rax,xmm0
        let padd = run(
            &[
                0x48, 0xb8, 0xff, 0x01, 0xff, 0x01, 0xff, 0x01, 0xff, 0x01, // mov rax,0x01ff01ff01ff01ff
                0x48, 0xbb, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, // mov rbx,0x0101010101010101
                0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0,rax
                0x66, 0x48, 0x0f, 0x6e, 0xcb, // movq xmm1,rbx
                0x66, 0x0f, 0xfc, 0xc1, // paddb xmm0,xmm1
                0x66, 0x48, 0x0f, 0x7e, 0xc0, // movq rax,xmm0
                0xf4,
            ],
            100,
        );
        assert_eq!(padd.cpu().regs.gpr[RAX], 0x0200_0200_0200_0200);

        // psubq: 5 - 3 = 2 in the low 64-bit lane.
        let psub = run(
            &[
                0x48, 0xc7, 0xc0, 0x05, 0, 0, 0, // mov rax,5
                0x48, 0xc7, 0xc3, 0x03, 0, 0, 0, // mov rbx,3
                0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0,rax
                0x66, 0x48, 0x0f, 0x6e, 0xcb, // movq xmm1,rbx
                0x66, 0x0f, 0xfb, 0xc1, // psubq xmm0,xmm1
                0x66, 0x48, 0x0f, 0x7e, 0xc0, // movq rax,xmm0
                0xf4,
            ],
            100,
        );
        assert_eq!(psub.cpu().regs.gpr[RAX], 2);
    }

    #[test]
    fn addsd_computes_scalar_double() {
        // 1.5 + 2.25 = 3.75 (IEEE-754 double, bit-exact)
        let program = [
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x3f, // mov rax, bits(1.5)
            0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x40, // mov rax, bits(2.25)
            0x66, 0x48, 0x0f, 0x6e, 0xc8, // movq xmm1, rax
            0xf2, 0x0f, 0x58, 0xc1, // addsd xmm0, xmm1
            0x66, 0x48, 0x0f, 0x7e, 0xc3, // movq rbx, xmm0
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(f64::from_bits(m.cpu().regs.gpr[RBX]), 3.75);
    }

    #[test]
    fn movdqa_copies_full_128_bits() {
        // load xmm0 low from rax, set xmm0 fully via two movq? simpler: movq xmm0,rax;
        // movdqa xmm1,xmm0 ; movq rcx,xmm1
        let program = [
            0x48, 0xb8, 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, // mov rax,0x0123456789abcdef
            0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
            0x66, 0x0f, 0x6f, 0xc8, // movdqa xmm1, xmm0
            0x66, 0x48, 0x0f, 0x7e, 0xc9, // movq rcx, xmm1
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RCX], 0x0123_4567_89ab_cdef);
        assert_eq!(m.cpu().xmm[1], m.cpu().xmm[0]);
    }

    #[test]
    fn mov_to_cr3_then_read_back() {
        // mov rax, 0x5000 ; mov cr3, rax ; mov rbx, cr3 ; hlt
        let program = [
            0x48, 0xc7, 0xc0, 0x00, 0x50, 0x00, 0x00, // mov rax, 0x5000
            0x0f, 0x22, 0xd8, // mov cr3, rax
            0x0f, 0x20, 0xdb, // mov rbx, cr3
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().sys.cr3, 0x5000);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x5000);
    }

    #[test]
    fn mov_cr0_writeback() {
        // mov rax, 0x11 ; mov cr0, rax ; mov rbx, cr0 ; hlt
        // (no PG: enabling paging mid-stream without page tables would fault the
        //  next fetch — a real kernel sets CR3 + tables first.)
        let program = [
            0x48, 0xc7, 0xc0, 0x11, 0x00, 0x00, 0x00, // mov rax, 0x11
            0x0f, 0x22, 0xc0, // mov cr0, rax
            0x0f, 0x20, 0xc3, // mov rbx, cr0
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().sys.cr0, 0x11);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x11);
    }

    #[test]
    fn indirect_call_and_ret() {
        // mov rax, target ; call rax ; hlt ;  target: mov rbx,0x77 ; ret
        let mut program = vec![
            0x48, 0xc7, 0xc0, 0x40, 0x00, 0x00, 0x00, // mov rax, 0x40
            0xff, 0xd0, // call rax
            0xf4, // hlt  (return lands here)
        ];
        program.resize(0x40, 0x90);
        program.extend([0x48, 0xc7, 0xc3, 0x77, 0x00, 0x00, 0x00, 0xc3]); // mov rbx,0x77 ; ret
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x77, "indirect call ran target");
        assert!(m.cpu().halted, "RET returned to the HLT after CALL");
    }

    #[test]
    fn indirect_jmp() {
        // mov rax, 0x20 ; jmp rax ; (skipped) mov rbx,1 ; ... 0x20: mov rbx,0x55 ; hlt
        let mut program = vec![
            0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00, // mov rax,0x20
            0xff, 0xe0, // jmp rax
            0x48, 0xc7, 0xc3, 0x01, 0x00, 0x00, 0x00, // mov rbx,1 (should be skipped)
        ];
        program.resize(0x20, 0x90);
        program.extend([0x48, 0xc7, 0xc3, 0x55, 0x00, 0x00, 0x00, 0xf4]); // mov rbx,0x55 ; hlt
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x55, "jumped past the skipped mov");
    }

    #[test]
    fn pushf_popf_round_trips_flags() {
        // mov rax,-1 ; add rax,1 (ZF=1,CF=1) ; pushfq ; pop rbx ; hlt
        let program = [
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax,-1
            0x48, 0x83, 0xc0, 0x01, // add rax,1
            0x9c, // pushfq
            0x5b, // pop rbx
            0xf4,
        ];
        let m = run(&program, 100);
        assert!(m.cpu().regs.gpr[RBX] & ZF != 0, "ZF captured in pushed flags");
        assert!(m.cpu().regs.gpr[RBX] & CF != 0, "CF captured");
    }

    #[test]
    fn lahf_sahf_move_flags_through_ah() {
        // add path to set CF, then lahf ; check AH has CF bit
        let program = [
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax,-1
            0x48, 0x83, 0xc0, 0x01, // add rax,1 -> CF=1,ZF=1; rax=0
            0x9f, // lahf -> AH = flags low byte
            0xf4,
        ];
        let m = run(&program, 100);
        let ah = (m.cpu().regs.gpr[RAX] >> 8) & 0xff;
        assert!(ah & 0x01 != 0, "CF in AH bit0");
        assert!(ah & 0x40 != 0, "ZF in AH bit6");
        assert!(ah & 0x02 != 0, "reserved bit1 always 1");
    }

    #[test]
    fn loop_counts_down() {
        // mov rcx, 5 ; xor rax,rax ; loop_body: inc rax ; loop loop_body ; hlt
        let program = [
            0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00, // mov rcx,5
            0x48, 0x31, 0xc0, // xor rax,rax  (body @ offset 10)
            0x48, 0xff, 0xc0, // inc rax
            0xe2, 0xfb, // loop -5 -> back to inc rax
            0xf4,
        ];
        let m = run(&program, 1000);
        assert_eq!(m.cpu().regs.gpr[RAX], 5, "loop ran 5 times");
        assert_eq!(m.cpu().regs.gpr[RCX], 0);
    }

    #[test]
    fn bswap_reverses_bytes() {
        // mov rax, 0x1122334455667788 ; bswap rax
        let program = [
            0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x48, 0x0f, 0xc8, 0xf4,
        ];
        assert_eq!(run(&program, 100).cpu().regs.gpr[RAX], 0x8877_6655_4433_2211);
    }

    #[test]
    fn push_rolls_back_rsp_when_the_store_faults() {
        // Precise-exception model: if a PUSH's store faults (here: RSP below the
        // mapped RAM), RSP must NOT be left decremented, or the instruction restart
        // would push again and leak 8 bytes — the bug that corrupted busybox's
        // return address when the stack demand-paged during a CALL.
        let program = [0x50, 0xf4]; // push rax ; hlt
        let mut m = X86Machine::new(seed(), 0x1000, &program, 0).unwrap();
        m.cpu_mut().regs.gpr[RSP] = 4; // store target = 4 - 8 wraps far out of bounds
        assert!(m.run(10).is_err(), "the push store faults");
        assert_eq!(m.cpu().regs.gpr[RSP], 4, "RSP rolled back, not decremented");
    }

    #[test]
    fn push_rm64_is_full_width() {
        // PUSH r/m has a default 64-bit operand size in long mode (no REX.W); a
        // 32-bit decode would truncate a high kernel pointer — the bug that
        // corrupted the kernel's int3 frame copy. Store a high 64-bit value, then
        // `push qword [0x200]`, and verify all 8 bytes (not 4) reach the stack.
        let program = [
            // movabs rax, 0xffffffff82e25efd
            0x48, 0xb8, 0xfd, 0x5e, 0xe2, 0x82, 0xff, 0xff, 0xff, 0xff,
            // mov [0x200], rax
            0x48, 0x89, 0x04, 0x25, 0x00, 0x02, 0x00, 0x00,
            // push qword [0x200]   (0xff /6, no REX.W)
            0xff, 0x34, 0x25, 0x00, 0x02, 0x00, 0x00,
            0xf4, // hlt
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RSP], 4096 - 8, "PUSH decremented RSP by 8");
        let pushed = m.memory().read_uint_le(4096 - 8, 8).unwrap();
        assert_eq!(
            pushed, 0xffff_ffff_82e2_5efd,
            "PUSH r/m64 stored the full 64-bit value, not a truncated dword"
        );
    }

    #[test]
    fn fp_nan_is_canonical_and_host_independent() {
        // Fresh invalid op (inf − inf) → x86 QNaN floating-point indefinite,
        // derived from bits so it is identical regardless of host architecture.
        let inf = f64::INFINITY;
        assert_eq!(fp_result_f64(inf - inf, inf, inf), 0xFFF8_0000_0000_0000);
        // A source NaN propagates, quieted, dest operand first.
        let snan = f64::from_bits(0x7FF0_0000_0000_0001);
        assert_eq!(fp_result_f64(f64::NAN, snan, 2.0), 0x7FF8_0000_0000_0001);
        // A normal result passes through unchanged.
        assert_eq!(fp_result_f64(3.5, 1.0, 2.5), 3.5_f64.to_bits());
        // Single precision: fresh invalid → 32-bit indefinite.
        assert_eq!(fp_result_f32(f32::INFINITY - f32::INFINITY, 1.0, 2.0), 0xFFC0_0000);
    }

    #[test]
    fn pextrd_extractps_pull_the_selected_lane() {
        // pextrd eax, xmm0, 2 (66 0f 3a 16 c0 02): eax = dword lane 2 of xmm0.
        let program = [0x66, 0x0f, 0x3a, 0x16, 0xc0, 0x02, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.cpu_mut().xmm[0] = 0xdddddddd_cccccccc_bbbbbbbb_aaaaaaaau128;
        m.run(10).unwrap();
        assert_eq!(m.cpu().regs.gpr[RAX], 0xcccccccc, "lane 2");
        // extractps eax, xmm0, 3 (66 0f 3a 17 c0 03): same lane select for FP dword.
        let p2 = [0x66, 0x0f, 0x3a, 0x17, 0xc0, 0x03, 0xf4];
        let mut m2 = X86Machine::new(seed(), 4096, &p2, 0).unwrap();
        m2.cpu_mut().xmm[0] = 0xdddddddd_cccccccc_bbbbbbbb_aaaaaaaau128;
        m2.run(10).unwrap();
        assert_eq!(m2.cpu().regs.gpr[RAX], 0xdddddddd, "lane 3");
    }

    #[test]
    fn cmpxchg16b_match_stores_and_sets_zf() {
        // lock cmpxchg16b [rsi] (f0 48 0f c7 0e): RDX:RAX vs [rsi]; on match store
        // RCX:RBX + ZF=1, else load [rsi] into RDX:RAX + ZF=0.
        let program = [0xf0, 0x48, 0x0f, 0xc7, 0x0e, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        {
            let c = m.cpu_mut();
            c.regs.gpr[RSI] = 0x800;
            c.regs.gpr[RAX] = 0x1111;
            c.regs.gpr[RDX] = 0x2222;
            c.regs.gpr[RBX] = 0x3333;
            c.regs.gpr[RCX] = 0x4444;
        }
        m.memory_mut().write_uint_le(0x800, 8, 0x1111).unwrap();
        m.memory_mut().write_uint_le(0x808, 8, 0x2222).unwrap();
        m.run(20).unwrap();
        assert_eq!(m.memory().read_uint_le(0x800, 8).unwrap(), 0x3333);
        assert_eq!(m.memory().read_uint_le(0x808, 8).unwrap(), 0x4444);
        assert!(flags::get_flag(m.cpu().regs.rflags, ZF));

        let mut m2 = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        {
            let c = m2.cpu_mut();
            c.regs.gpr[RSI] = 0x800;
            c.regs.gpr[RAX] = 0x9999;
            c.regs.gpr[RDX] = 0x2222;
        }
        m2.memory_mut().write_uint_le(0x800, 8, 0x1111).unwrap();
        m2.memory_mut().write_uint_le(0x808, 8, 0x2222).unwrap();
        m2.run(20).unwrap();
        assert_eq!(m2.cpu().regs.gpr[RAX], 0x1111, "loaded [rsi] low into RAX");
        assert_eq!(m2.cpu().regs.gpr[RDX], 0x2222, "loaded [rsi] high into RDX");
        assert!(!flags::get_flag(m2.cpu().regs.rflags, ZF));
    }

    #[test]
    fn sqrtpd_takes_both_lanes() {
        // sqrtpd xmm0, xmm1 (66 0f 51 c1): both f64 lanes square-rooted.
        let program = [0x66, 0x0f, 0x51, 0xc1, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.cpu_mut().xmm[1] = (16.0f64).to_bits() as u128 | ((9.0f64).to_bits() as u128) << 64;
        m.run(10).unwrap();
        assert_eq!(f64::from_bits(m.cpu().xmm[0] as u64), 4.0);
        assert_eq!(f64::from_bits((m.cpu().xmm[0] >> 64) as u64), 3.0);
    }

    #[test]
    fn movmskps_extracts_sign_bits() {
        // movmskps eax, xmm0 (0f 50 c0): bit i = sign of dword lane i.
        let program = [0x0f, 0x50, 0xc0, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        // lanes (low→high): d0 sign=1, d1=0, d2=1, d3=1 → mask 0b1101
        m.cpu_mut().xmm[0] = 0x80000000_80000000_00000000_80000000u128;
        m.run(10).unwrap();
        assert_eq!(m.cpu().regs.gpr[RAX], 0b1101);
    }

    #[test]
    fn minsd_maxsd_pick_operand_with_src_on_nan() {
        // minsd xmm0,xmm1 (f2 0f 5d c1): (a<b)?a:b ; maxsd (f2 0f 5f c1): (a>b)?a:b
        let mins = [0xf2, 0x0f, 0x5d, 0xc1, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &mins, 0).unwrap();
        m.cpu_mut().xmm[0] = (3.0f64).to_bits() as u128;
        m.cpu_mut().xmm[1] = (5.0f64).to_bits() as u128;
        m.run(10).unwrap();
        assert_eq!(f64::from_bits(m.cpu().xmm[0] as u64), 3.0, "min(3,5)=3");
        // NaN in dst → result is src (x86 semantics).
        let maxs = [0xf2, 0x0f, 0x5f, 0xc1, 0xf4];
        let mut m2 = X86Machine::new(seed(), 4096, &maxs, 0).unwrap();
        m2.cpu_mut().xmm[0] = f64::NAN.to_bits() as u128;
        m2.cpu_mut().xmm[1] = (5.0f64).to_bits() as u128;
        m2.run(10).unwrap();
        assert_eq!(f64::from_bits(m2.cpu().xmm[0] as u64), 5.0, "max(NaN,5)=src=5");
    }

    #[test]
    fn shufps_selects_low_from_dst_high_from_src() {
        // shufps xmm0, xmm2, 0xdd (0f c6 c2 dd): dwords 0,1 from xmm0; 2,3 from xmm2.
        let program = [0x0f, 0xc6, 0xc2, 0xdd, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.cpu_mut().xmm[0] = 0x33333333_22222222_11111111_00000000u128;
        m.cpu_mut().xmm[2] = 0x77777777_66666666_55555555_44444444u128;
        m.run(10).unwrap();
        // imm 0xdd = 0b11_01_11_01 → d0=A1, d1=A3, d2=B1, d3=B3
        assert_eq!(m.cpu().xmm[0], 0x77777777_55555555_33333333_11111111u128);
    }

    #[test]
    fn shufpd_selects_halves() {
        // shufpd xmm0, xmm2, 0x2 (66 0f c6 c2 02): low from dst[0], high from src[1].
        let program = [0x66, 0x0f, 0xc6, 0xc2, 0x02, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.cpu_mut().xmm[0] = 0xbbbb_bbbb_bbbb_bbbb_aaaa_aaaa_aaaa_aaaau128;
        m.cpu_mut().xmm[2] = 0xdddd_dddd_dddd_dddd_cccc_cccc_cccc_ccccu128;
        m.run(10).unwrap();
        // bit0=0 → low = dst low (a); bit1=1 → high = src high (d)
        assert_eq!(m.cpu().xmm[0], 0xdddd_dddd_dddd_dddd_aaaa_aaaa_aaaa_aaaau128);
    }

    #[test]
    fn cmpsd_lt_writes_lane_mask_and_preserves_high() {
        // cmpltsd xmm0, xmm3 (F2 0F C2 C3 01): low lane = (xmm0 < xmm3) ? -1 : 0,
        // high 64 bits of xmm0 preserved (scalar). libm's __sincos_sse2 uses this.
        let program = [0xf2, 0x0f, 0xc2, 0xc3, 0x01, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.cpu_mut().xmm[0] = (7.0f64).to_bits() as u128 | (0x1111u128 << 64);
        m.cpu_mut().xmm[3] = (9.0f64).to_bits() as u128;
        m.run(10).unwrap();
        assert_eq!(m.cpu().xmm[0] as u64, u64::MAX, "7<9 true → all ones");
        assert_eq!(m.cpu().xmm[0] >> 64, 0x1111, "high lane preserved");

        let mut m2 = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m2.cpu_mut().xmm[0] = (9.0f64).to_bits() as u128;
        m2.cpu_mut().xmm[3] = (7.0f64).to_bits() as u128;
        m2.run(10).unwrap();
        assert_eq!(m2.cpu().xmm[0] as u64, 0, "9<7 false → zero");

        // NaN with LT predicate → false (ordered predicate).
        let mut m3 = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m3.cpu_mut().xmm[0] = f64::NAN.to_bits() as u128;
        m3.cpu_mut().xmm[3] = (1.0f64).to_bits() as u128;
        m3.run(10).unwrap();
        assert_eq!(m3.cpu().xmm[0] as u64, 0, "NaN<1 (LT) is false");
    }

    #[test]
    fn pminub_takes_unsigned_byte_minimum() {
        // Lane-wise unsigned byte minimum (the instruction musl's SSE2 strcmp
        // faulted on). xmm0=[..,0xff,0x10], xmm1=[..,0x7f,0x80] → min=[..,0x7f,0x10].
        assert_eq!(packed_minmax(0xff10, 0x7f80, 1, false, false), 0x7f10);
        // Unsigned max: PMAXUB → [0xff, 0x80].
        assert_eq!(packed_minmax(0xff10, 0x7f80, 1, true, false), 0xff80);
        // Signed word min: 0xffff (-1) < 0x0001 → keeps 0xffff.
        assert_eq!(packed_minmax(0xffff, 0x0001, 2, false, true), 0xffff);
        // End-to-end through the decoder: pminub xmm0,xmm1.
        let program = [
            0x48, 0xb8, 0x10, 0xff, 0, 0, 0, 0, 0, 0, // mov rax,0xff10
            0x48, 0x89, 0x04, 0x25, 0x00, 0x02, 0, 0, // mov [0x200],rax
            0x48, 0xb8, 0x80, 0x7f, 0, 0, 0, 0, 0, 0, // mov rax,0x7f80
            0x48, 0x89, 0x04, 0x25, 0x08, 0x02, 0, 0, // mov [0x208],rax
            0x0f, 0x10, 0x04, 0x25, 0x00, 0x02, 0, 0, // movups xmm0,[0x200]
            0x0f, 0x10, 0x0c, 0x25, 0x08, 0x02, 0, 0, // movups xmm1,[0x208]
            0x66, 0x0f, 0xda, 0xc1, // pminub xmm0,xmm1
            0x0f, 0x11, 0x04, 0x25, 0x00, 0x03, 0, 0, // movups [0x300],xmm0
            0xf4,
        ];
        let m = run(&program, 200);
        assert_eq!(m.memory().read_uint_le(0x300, 2).unwrap(), 0x7f10);
    }

    #[test]
    fn movlps_movhps_move_64bit_halves() {
        // Build xmm0 = [hi=0xAABBCCDDEEFF0011 : lo=0x1122334455667788] via two
        // stores + MOVLPS/MOVHPS loads, then store each half back and check.
        // (0F 16 MOVHPS load is the exact instruction busybox's /init faulted on.)
        let program = [
            // mov rax,0x1122334455667788 ; mov [0x200],rax
            0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
            0x48, 0x89, 0x04, 0x25, 0x00, 0x02, 0x00, 0x00,
            // mov rax,0xAABBCCDDEEFF0011 ; mov [0x208],rax
            0x48, 0xb8, 0x11, 0x00, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa,
            0x48, 0x89, 0x04, 0x25, 0x08, 0x02, 0x00, 0x00,
            // movlps xmm0,[0x200]  (low half)   0F 12 /r SIB-disp32
            0x0f, 0x12, 0x04, 0x25, 0x00, 0x02, 0x00, 0x00,
            // movhps xmm0,[0x208]  (high half)  0F 16 /r SIB-disp32
            0x0f, 0x16, 0x04, 0x25, 0x08, 0x02, 0x00, 0x00,
            // movhps [0x300],xmm0  (store high) 0F 17
            0x0f, 0x17, 0x04, 0x25, 0x00, 0x03, 0x00, 0x00,
            // movlps [0x308],xmm0  (store low)  0F 13
            0x0f, 0x13, 0x04, 0x25, 0x08, 0x03, 0x00, 0x00,
            0xf4, // hlt
        ];
        let m = run(&program, 200);
        assert_eq!(
            m.memory().read_uint_le(0x300, 8).unwrap(),
            0xaabb_ccdd_eeff_0011,
            "MOVHPS stored the high quadword"
        );
        assert_eq!(
            m.memory().read_uint_le(0x308, 8).unwrap(),
            0x1122_3344_5566_7788,
            "MOVLPS stored the low quadword"
        );
    }

    #[test]
    fn rdtsc_reads_virtual_time() {
        // nop x3 ; rdtsc ; hlt
        let m = run(&[0x90, 0x90, 0x90, 0x0f, 0x31, 0xf4], 100);
        let tsc = (m.cpu().regs.gpr[RAX] & 0xffff_ffff) | (m.cpu().regs.gpr[RDX] << 32);
        assert!(tsc > 0 && tsc < 20, "TSC ~ instruction count, got {tsc}");
    }

    #[test]
    fn xchg_swaps_operands() {
        // mov rax,1 ; mov rbx,2 ; xchg rax,rbx
        let program = [
            0x48, 0xc7, 0xc0, 0x01, 0, 0, 0, 0x48, 0xc7, 0xc3, 0x02, 0, 0, 0, 0x48, 0x87, 0xd8,
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 2);
        assert_eq!(m.cpu().regs.gpr[RBX], 1);
    }

    #[test]
    fn xchg_rax_r8_is_not_a_nop() {
        // `xchg %rax,%r8` is 0x90 with REX.B — a real swap, NOT a NOP. Decoding
        // it as NOP (the bug that crashed busybox's ash var lookup) would leave
        // both registers unchanged.
        let program = [
            0x48, 0xc7, 0xc0, 0xaa, 0xaa, 0x00, 0x00, // mov rax, 0xAAAA
            0x49, 0xc7, 0xc0, 0xbb, 0xbb, 0x00, 0x00, // mov r8, 0xBBBB
            0x49, 0x90, // xchg rax, r8
            0xf4, // hlt
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0xBBBB, "rax got r8");
        assert_eq!(m.cpu().regs.gpr[8], 0xAAAA, "r8 got rax");
        // Plain 0x90 (no REX.B) must still behave as NOP (xchg rax,rax).
        let m2 = run(&[0x48, 0xc7, 0xc0, 0x07, 0, 0, 0, 0x90, 0xf4], 100);
        assert_eq!(m2.cpu().regs.gpr[RAX], 7, "plain 0x90 is a no-op");
    }

    #[test]
    fn xadd_sums_and_returns_old() {
        // mov rax,10 ; mov rbx,5 ; xadd rax,rbx -> rax=15, rbx=10
        let program = [
            0x48, 0xc7, 0xc0, 0x0a, 0, 0, 0, 0x48, 0xc7, 0xc3, 0x05, 0, 0, 0, 0x48, 0x0f, 0xc1,
            0xd8, 0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 15);
        assert_eq!(m.cpu().regs.gpr[RBX], 10);
    }

    #[test]
    fn lock_xaddq_to_memory_persists_64bit() {
        // Reproduces Go's sysMemStat.add path: a 64-bit LOCK XADDQ on a memory
        // location. mem[0x800] starts 0, then += 0x40000, then += -256 (0xff..00),
        // and must end at 0x3ff00. Also checks the returned "old value" in the reg.
        //   mov rax, 0x800            48 c7 c0 00 08 00 00
        //   xor rcx, rcx              48 31 c9
        //   mov [rax], rcx            48 89 08
        //   mov rbx, 0x40000          48 c7 c3 00 00 04 00
        //   lock xadd [rax], rbx      f0 48 0f c1 18      ; rbx<-0(old), mem<-0x40000
        //   mov rdx, -256             48 c7 c2 00 ff ff ff (sign-extended imm32)
        //   lock xadd [rax], rdx      f0 48 0f c1 10      ; rdx<-0x40000(old), mem<-0x3ff00
        //   hlt                       f4
        let program = [
            0x48, 0xc7, 0xc0, 0x00, 0x08, 0x00, 0x00, 0x48, 0x31, 0xc9, 0x48, 0x89, 0x08, 0x48,
            0xc7, 0xc3, 0x00, 0x00, 0x04, 0x00, 0xf0, 0x48, 0x0f, 0xc1, 0x18, 0x48, 0xc7, 0xc2,
            0x00, 0xff, 0xff, 0xff, 0xf0, 0x48, 0x0f, 0xc1, 0x10, 0xf4,
        ];
        let m = run(&program, 100);
        let mem_val = m.memory().read_uint_le(0x800, 8).unwrap();
        assert_eq!(mem_val, 0x3ff00, "64-bit atomic adds must accumulate in memory");
        assert_eq!(m.cpu().regs.gpr[RBX], 0, "first xadd returns old value 0");
        assert_eq!(m.cpu().regs.gpr[RDX], 0x40000, "second xadd returns old value 0x40000");
    }

    #[test]
    fn rdrand_is_deterministic_and_sets_cf() {
        // RDRAND eax (0f c7 f0) ; RDSEED ecx (0f c7 f9) ; hlt (f4). Both draw from
        // the same seed-derived stream → reproducible (this is a DST machine), and
        // both set CF=1 (success). Implemented but not advertised in CPUID (Phase A).
        let program = [0x0f, 0xc7, 0xf0, 0x0f, 0xc7, 0xf9, 0xf4];
        let m1 = run(&program, 10);
        let m2 = run(&program, 10);
        let v1 = m1.cpu().regs.gpr[RAX] & 0xffff_ffff;
        assert_ne!(v1, 0, "RDRAND should produce a (seed-derived) value");
        assert_eq!(
            v1,
            m2.cpu().regs.gpr[RAX] & 0xffff_ffff,
            "same seed -> same RDRAND value (deterministic)"
        );
        assert_ne!(
            m1.cpu().regs.gpr[RCX] & 0xffff_ffff,
            v1,
            "RDSEED draws the next value, distinct from the RDRAND one"
        );
        assert_ne!(m1.cpu().regs.rflags & CF, 0, "RDRAND/RDSEED set CF=1 (success)");
    }

    #[test]
    fn decode_cache_invalidates_on_self_modifying_code() {
        // The decode cache must not return a stale instruction after the guest
        // rewrites code. A 2-iteration loop runs a TARGET (`inc rax`), then patches
        // TARGET's 3rd byte from 0xc0 (inc) to 0xc8 (dec); iteration 2 must execute
        // the PATCHED `dec rax`. SMC works → rax = +1 then -1 = 0; a stale cache
        // would re-run `inc` → rax = 2.
        //   @00 xor rax,rax            48 31 c0
        //   @03 mov rcx,2              48 c7 c1 02 00 00 00
        //   @0a TARGET inc rax         48 ff c0      ; byte c0 is at 0x0c
        //   @0d mov byte [0x0c],0xc8   c6 04 25 0c 00 00 00 c8
        //   @15 dec rcx                48 ff c9
        //   @18 jnz 0x0a (rel8 -16)    75 f0
        //   @1a hlt                    f4
        let program = [
            0x48, 0x31, 0xc0, 0x48, 0xc7, 0xc1, 0x02, 0x00, 0x00, 0x00, 0x48, 0xff, 0xc0, 0xc6,
            0x04, 0x25, 0x0c, 0x00, 0x00, 0x00, 0xc8, 0x48, 0xff, 0xc9, 0x75, 0xf0, 0xf4,
        ];
        let m = run(&program, 1000);
        assert_eq!(
            m.cpu().regs.gpr[RAX],
            0,
            "self-modified `dec` ran on iteration 2 (cache invalidated), not a stale `inc`"
        );
        assert_eq!(m.cpu().regs.gpr[RCX], 0);
    }

    #[test]
    fn snapshot_restore_is_transparent_on_real_machine() {
        // GOLDEN_TARGET §19: a full-machine snapshot taken mid-run, then restored,
        // must continue to a bit-identical end state — proving "rewind and replay"
        // works on the real x86 machine (CPU + RAM + devices + core clock/RNG).
        //   xor rax,rax ; mov rcx,1000 ; loop: add rax,rcx; dec rcx; jnz loop ; hlt
        let program = [
            0x48, 0x31, 0xc0, // xor rax, rax
            0x48, 0xc7, 0xc1, 0xe8, 0x03, 0x00, 0x00, // mov rcx, 1000
            0x48, 0x01, 0xc8, // add rax, rcx
            0x48, 0xff, 0xc9, // dec rcx
            0x75, 0xf8, // jnz -8
            0xf4, // hlt
        ];
        let build = || X86Machine::new(seed(), 4096, &program, 0).unwrap();

        // Reference: run straight to completion.
        let mut reference = build();
        reference.run(100_000).unwrap();
        let d_ref = reference.full_state_digest();

        // Snapshot mid-run, then continue — must match the reference end state.
        let mut m = build();
        m.run_until_instruction(1500).unwrap();
        let snap = m.snapshot();
        assert_eq!(snap.instruction_count, m.instruction_count());
        assert_eq!(snap.digest, m.full_state_digest(), "snapshot captures the live digest");
        m.run(100_000).unwrap();
        assert_eq!(m.full_state_digest(), d_ref, "snapshot+continue == uninterrupted run");

        // Restore from the snapshot and continue independently — also bit-identical.
        let mut restored = snap.restore();
        assert_eq!(restored.full_state_digest(), snap.digest, "restore reproduces the snapshot");
        restored.run(100_000).unwrap();
        assert_eq!(restored.full_state_digest(), d_ref, "restored run reaches the same end state");
    }

    #[test]
    fn cmpxchg_success_and_failure() {
        // success: rax==dst -> dst=src, ZF=1
        // mov rax,5 ; mov rbx,5 ; mov rcx,99 ; cmpxchg rbx,rcx
        let ok = run(
            &[
                0x48, 0xc7, 0xc0, 0x05, 0, 0, 0, 0x48, 0xc7, 0xc3, 0x05, 0, 0, 0, 0x48, 0xc7, 0xc1,
                0x63, 0, 0, 0, 0x48, 0x0f, 0xb1, 0xcb, 0xf4,
            ],
            100,
        );
        assert_eq!(ok.cpu().regs.gpr[RBX], 99, "exchange happened");
        assert!(flags::get_flag(ok.cpu().regs.rflags, ZF));
        // failure: rax!=dst -> rax=dst, ZF=0
        let fail = run(
            &[
                0x48, 0xc7, 0xc0, 0x05, 0, 0, 0, 0x48, 0xc7, 0xc3, 0x07, 0, 0, 0, 0x48, 0xc7, 0xc1,
                0x63, 0, 0, 0, 0x48, 0x0f, 0xb1, 0xcb, 0xf4,
            ],
            100,
        );
        assert_eq!(fail.cpu().regs.gpr[RAX], 7, "RAX reloaded from dst");
        assert_eq!(fail.cpu().regs.gpr[RBX], 7, "dst unchanged");
    }

    #[test]
    fn bt_and_bts() {
        // mov rax,8 ; bt rax,3 (CF=1) ; bts rax,0 (rax=9)
        let program = [
            0x48, 0xc7, 0xc0, 0x08, 0, 0, 0, // mov rax,8
            0x48, 0x0f, 0xba, 0xe0, 0x03, // bt rax,3
            0x48, 0x0f, 0xba, 0xe8, 0x00, // bts rax,0
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 9, "bit 0 set");
    }

    #[test]
    fn bsf_bsr_and_popcnt() {
        // mov rax,0x100 ; bsf rbx,rax (8) ; bsr rcx,rax (8)
        let m = run(
            &[
                0x48, 0xc7, 0xc0, 0x00, 0x01, 0, 0, 0x48, 0x0f, 0xbc, 0xd8, 0x48, 0x0f, 0xbd, 0xc8,
                0xf4,
            ],
            100,
        );
        assert_eq!(m.cpu().regs.gpr[RBX], 8, "BSF lowest set bit");
        assert_eq!(m.cpu().regs.gpr[RCX], 8, "BSR highest set bit");
        // popcnt: mov rax,0xff ; popcnt rbx,rax -> 8
        let p = run(
            &[0x48, 0xc7, 0xc0, 0xff, 0, 0, 0, 0xf3, 0x48, 0x0f, 0xb8, 0xd8, 0xf4],
            100,
        );
        assert_eq!(p.cpu().regs.gpr[RBX], 8, "POPCNT");
    }

    #[test]
    fn bsf_of_zero_sets_zf_and_keeps_dest() {
        // mov rbx,0xdead ; xor rax,rax ; bsf rbx,rax -> ZF=1, rbx unchanged
        let m = run(
            &[
                0x48, 0xc7, 0xc3, 0xad, 0xde, 0, 0, 0x48, 0x31, 0xc0, 0x48, 0x0f, 0xbc, 0xd8, 0xf4,
            ],
            100,
        );
        assert_eq!(m.cpu().regs.gpr[RBX], 0xdead, "dest unchanged when src=0");
        assert!(flags::get_flag(m.cpu().regs.rflags, ZF));
    }

    #[test]
    fn rep_stosb_fills_memory() {
        // mov rdi,0x800 ; mov rax,0xab ; mov rcx,4 ; rep stosb
        let program = [
            0x48, 0xc7, 0xc7, 0x00, 0x08, 0, 0, // mov rdi,0x800
            0x48, 0xc7, 0xc0, 0xab, 0, 0, 0, // mov rax,0xab
            0x48, 0xc7, 0xc1, 0x04, 0, 0, 0, // mov rcx,4
            0xf3, 0xaa, // rep stosb
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.memory().read_uint_le(0x800, 4).unwrap(), 0xabab_abab);
        assert_eq!(m.cpu().regs.gpr[RCX], 0);
    }

    #[test]
    fn rep_movsb_copies_memory() {
        let program = [
            0x48, 0xc7, 0xc6, 0x00, 0x08, 0, 0, // mov rsi,0x800
            0x48, 0xc7, 0xc7, 0x00, 0x09, 0, 0, // mov rdi,0x900
            0x48, 0xc7, 0xc1, 0x04, 0, 0, 0, // mov rcx,4
            0xf3, 0xa4, // rep movsb
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.memory_mut().write_uint_le(0x800, 4, 0x4433_2211).unwrap();
        m.run(100).unwrap();
        assert_eq!(m.memory().read_uint_le(0x900, 4).unwrap(), 0x4433_2211);
    }

    #[test]
    fn adc_adds_with_carry() {
        // mov rax,-1 ; add rax,1 (CF=1) ; mov rbx,5 ; adc rbx,0 -> 6
        let program = [
            0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax,-1
            0x48, 0x83, 0xc0, 0x01, // add rax,1  -> CF=1
            0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx,5
            0x48, 0x83, 0xd3, 0x00, // adc rbx,0  -> 5+0+1
            0xf4,
        ];
        assert_eq!(run(&program, 100).cpu().regs.gpr[RBX], 6);
    }

    #[test]
    fn sbb_subtracts_with_borrow() {
        // mov rax,0 ; sub rax,1 (CF=1) ; mov rbx,10 ; sbb rbx,0 -> 9
        let program = [
            0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, // mov rax,0
            0x48, 0x83, 0xe8, 0x01, // sub rax,1  -> CF=1
            0x48, 0xc7, 0xc3, 0x0a, 0x00, 0x00, 0x00, // mov rbx,10
            0x48, 0x83, 0xdb, 0x00, // sbb rbx,0  -> 10-0-1
            0xf4,
        ];
        assert_eq!(run(&program, 100).cpu().regs.gpr[RBX], 9);
    }

    #[test]
    fn cmovcc_moves_only_when_taken() {
        // mov rbx,0x1234 ; xor rax,rax (ZF=1) ; cmove rcx,rbx ; cmovne rdx,rbx ; hlt
        let program = [
            0x48, 0xc7, 0xc3, 0x34, 0x12, 0x00, 0x00, // mov rbx,0x1234
            0x48, 0x31, 0xc0, // xor rax,rax -> ZF=1
            0x48, 0x0f, 0x44, 0xcb, // cmove rcx,rbx  (taken)
            0x48, 0x0f, 0x45, 0xd3, // cmovne rdx,rbx (not taken)
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RCX], 0x1234, "CMOVE taken");
        assert_eq!(m.cpu().regs.gpr[RDX], 0, "CMOVNE not taken");
    }

    #[test]
    fn setcc_sets_byte_from_condition() {
        // xor rax,rax (ZF=1) ; sete al ; setne bl ; hlt
        let program = [
            0x48, 0x31, 0xc0, // xor rax,rax
            0x0f, 0x94, 0xc0, // sete al
            0x0f, 0x95, 0xc3, // setne bl
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX] & 0xff, 1, "SETE -> 1");
        assert_eq!(m.cpu().regs.gpr[RBX] & 0xff, 0, "SETNE -> 0");
    }

    #[test]
    fn shl_shifts_and_sets_carry() {
        // mov eax, 1 ; shl eax, 4 -> 16
        let m = run(&[0xb8, 0x01, 0, 0, 0, 0xc1, 0xe0, 0x04, 0xf4], 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 16);
    }

    #[test]
    fn shr_by_one_sets_carry_and_zero() {
        // mov eax, 1 ; shr eax, 1 -> 0, CF=1, ZF=1
        let m = run(&[0xb8, 0x01, 0, 0, 0, 0xd1, 0xe8, 0xf4], 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0);
        assert!(flags::get_flag(m.cpu().regs.rflags, CF));
        assert!(flags::get_flag(m.cpu().regs.rflags, ZF));
    }

    #[test]
    fn sar_is_arithmetic() {
        // mov eax, -8 ; sar eax, 1 -> -4
        let m = run(&[0xb8, 0xf8, 0xff, 0xff, 0xff, 0xd1, 0xf8, 0xf4], 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0xffff_fffc);
    }

    #[test]
    fn rol_byte_wraps_through_carry() {
        // mov al, 0x80 ; rol al, 1 -> 0x01, CF=1
        let m = run(&[0xb0, 0x80, 0xd0, 0xc0, 0xf4], 100);
        assert_eq!(m.cpu().regs.gpr[RAX] & 0xff, 0x01);
        assert!(flags::get_flag(m.cpu().regs.rflags, CF));
    }

    #[test]
    fn mul_produces_high_and_low_halves() {
        // mov eax, 7 ; mov ecx, 6 ; mul ecx -> edx:eax = 42, CF=OF=0
        let program = [
            0xb8, 0x07, 0, 0, 0, // mov eax, 7
            0xb9, 0x06, 0, 0, 0, // mov ecx, 6
            0xf7, 0xe1, // mul ecx
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 42);
        assert_eq!(m.cpu().regs.gpr[RDX], 0);
        assert!(!flags::get_flag(m.cpu().regs.rflags, CF));
        assert!(!flags::get_flag(m.cpu().regs.rflags, OF));
    }

    #[test]
    fn mul_overflow_sets_carry_and_overflow() {
        // mov eax, 0x10000000 ; mov ecx, 0x10 ; mul ecx -> edx=1, eax=0, CF=OF=1
        let program = [
            0xb8, 0x00, 0x00, 0x00, 0x10, // mov eax, 0x10000000
            0xb9, 0x10, 0, 0, 0, // mov ecx, 0x10
            0xf7, 0xe1, // mul ecx
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0);
        assert_eq!(m.cpu().regs.gpr[RDX], 1);
        assert!(flags::get_flag(m.cpu().regs.rflags, CF));
        assert!(flags::get_flag(m.cpu().regs.rflags, OF));
    }

    #[test]
    fn imul_two_operand_signed() {
        // mov eax, -3 ; mov ecx, 5 ; imul eax, ecx -> -15
        let program = [
            0xb8, 0xfd, 0xff, 0xff, 0xff, // mov eax, -3
            0xb9, 0x05, 0, 0, 0, // mov ecx, 5
            0x0f, 0xaf, 0xc1, // imul eax, ecx
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0xffff_fff1); // -15
    }

    #[test]
    fn div_unsigned() {
        // mov edx,0 ; mov eax,42 ; mov ecx,5 ; div ecx -> eax=8, edx=2
        let program = [
            0xba, 0, 0, 0, 0, // mov edx, 0
            0xb8, 0x2a, 0, 0, 0, // mov eax, 42
            0xb9, 0x05, 0, 0, 0, // mov ecx, 5
            0xf7, 0xf1, // div ecx
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 8);
        assert_eq!(m.cpu().regs.gpr[RDX], 2);
    }

    #[test]
    fn idiv_signed_truncates_toward_zero() {
        // -42 / 5 = -8 remainder -2
        // mov eax,-42 ; mov edx,-1 ; mov ecx,5 ; idiv ecx
        let program = [
            0xb8, 0xd6, 0xff, 0xff, 0xff, // mov eax, -42
            0xba, 0xff, 0xff, 0xff, 0xff, // mov edx, -1
            0xb9, 0x05, 0, 0, 0, // mov ecx, 5
            0xf7, 0xf9, // idiv ecx
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0xffff_fff8); // -8
        assert_eq!(m.cpu().regs.gpr[RDX], 0xffff_fffe); // -2
    }

    #[test]
    fn divide_by_zero_raises_de() {
        // mov edx,0 ; mov eax,1 ; mov ecx,0 ; div ecx -> #DE
        let program = [
            0xba, 0, 0, 0, 0, 0xb8, 0x01, 0, 0, 0, 0xb9, 0, 0, 0, 0, 0xf7, 0xf1, 0xf4,
        ];
        assert!(matches!(try_run(&program), Err(X86Error::DivideError)));
    }

    #[test]
    fn neg_and_not() {
        // mov eax,5 ; neg eax -> -5, CF=1
        let neg = run(&[0xb8, 0x05, 0, 0, 0, 0xf7, 0xd8, 0xf4], 100);
        assert_eq!(neg.cpu().regs.gpr[RAX], 0xffff_fffb);
        assert!(flags::get_flag(neg.cpu().regs.rflags, CF));
        // mov eax,0 ; not eax -> 0xffffffff
        let not = run(&[0xb8, 0, 0, 0, 0, 0xf7, 0xd0, 0xf4], 100);
        assert_eq!(not.cpu().regs.gpr[RAX], 0xffff_ffff);
    }

    #[test]
    fn arithmetic_is_deterministic() {
        // A mix of mul/div/shift; same seed -> identical digest.
        let program = [
            0xb8, 0x07, 0, 0, 0, 0xb9, 0x06, 0, 0, 0, 0xf7, 0xe1, // mul
            0xc1, 0xe0, 0x02, // shl eax, 2
            0xf4,
        ];
        let mut a = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        let mut b = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    /// Install identity tables for the low 2 MiB (PD large page) so that code,
    /// stack, and the page tables themselves are reachable with paging on.
    fn install_identity_low(mem: &mut GuestMemory) {
        let pe = 0x7_u64; // present | rw | user
        mem.write_uint_le(0x1000, 8, 0x2000 | pe).unwrap(); // PML4[0] -> PDPT
        mem.write_uint_le(0x2000, 8, 0x3000 | pe).unwrap(); // PDPT[0] -> PD
        mem.write_uint_le(0x3000, 8, pe | 0x80).unwrap(); // PD[0] -> 2 MiB page @ PA 0
    }

    #[test]
    fn paging_translates_a_data_store() {
        // mov rax, 0x1234 ; mov [0x400000], rax ; hlt
        // VA 0x400000 is mapped (4 KiB) to PA 0x80000.
        let program = [
            0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, // mov rax, 0x1234
            0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x40, 0x00, // mov [0x400000], rax
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 0x100000, &program, 0x6000).unwrap();
        {
            let mem = m.memory_mut();
            install_identity_low(mem);
            let pe = 0x7_u64;
            mem.write_uint_le(0x3000 + 2 * 8, 8, 0x4000 | pe).unwrap(); // PD[2] -> PT
            mem.write_uint_le(0x4000, 8, 0x80000 | pe).unwrap(); // PT[0] -> PA 0x80000
        }
        m.cpu_mut().sys.cr3 = 0x1000;
        m.cpu_mut().sys.cr0 |= CR0_PG;

        m.run(100).unwrap();
        // The store through VA 0x400000 landed at physical 0x80000.
        assert_eq!(m.memory().read_uint_le(0x80000, 8).unwrap(), 0x1234);
        // Accessed/Dirty bits got written on the leaf PTE.
        let pte = m.memory().read_uint_le(0x4000, 8).unwrap();
        assert_ne!(pte & (1 << 5), 0, "A set");
        assert_ne!(pte & (1 << 6), 0, "D set on write");
    }

    #[test]
    fn paging_fault_on_unmapped_store_sets_cr2() {
        // Store to VA 0x800000, which is not mapped -> #PF, CR2 = 0x800000.
        let program = [
            0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, // mov rax, 0x1234
            0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x80, 0x00, // mov [0x800000], rax
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 0x100000, &program, 0x6000).unwrap();
        install_identity_low(m.memory_mut());
        m.cpu_mut().sys.cr3 = 0x1000;
        m.cpu_mut().sys.cr0 |= CR0_PG;

        let result = m.run(100);
        assert!(matches!(result, Err(X86Error::PageFault { .. })));
        assert_eq!(m.cpu().sys.cr2, 0x800000);
    }

    #[test]
    fn paged_execution_is_deterministic() {
        let program = [
            0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, 0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x40,
            0x00, 0xf4,
        ];
        let setup = |m: &mut X86Machine| {
            install_identity_low(m.memory_mut());
            let mem = m.memory_mut();
            mem.write_uint_le(0x3000 + 2 * 8, 8, 0x4000 | 0x7).unwrap();
            mem.write_uint_le(0x4000, 8, 0x80000 | 0x7).unwrap();
            m.cpu_mut().sys.cr3 = 0x1000;
            m.cpu_mut().sys.cr0 |= CR0_PG;
        };
        let mut a = X86Machine::new(seed(), 0x100000, &program, 0x6000).unwrap();
        let mut b = X86Machine::new(seed(), 0x100000, &program, 0x6000).unwrap();
        setup(&mut a);
        setup(&mut b);
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn software_interrupt_calls_handler_and_returns() {
        // main @ 0:     int 0x20 ; hlt
        // handler @0x100: mov rbx, 0xbeef ; iret
        let mut program = vec![0xcd, 0x20, 0xf4];
        program.resize(0x100, 0x90);
        program.extend([0x48, 0xc7, 0xc3, 0xef, 0xbe, 0x00, 0x00, 0xcf]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        set_idt_gate(m.memory_mut(), 0x4000, 0x20, 0x100);
        m.cpu_mut().sys.idtr_base = 0x4000;
        m.cpu_mut().sys.idtr_limit = 0xfff;

        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RBX], 0xbeef);
        assert!(m.cpu().halted, "returned via IRET to the HLT after INT");
    }

    #[test]
    fn timer_interrupt_delivered_when_if_set() {
        // main: sti ; loop: nop ; jmp loop      handler @0x100: inc r15 ; iret
        let mut program = vec![0xfb, 0x90, 0xeb, 0xfd];
        program.resize(0x100, 0x90);
        program.extend([0x49, 0xff, 0xc7, 0xcf]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        m.set_core_timer_injection(true); // use the synthetic timer as the IRQ source
        set_idt_gate(m.memory_mut(), 0x4000, 32, 0x100);
        m.cpu_mut().sys.idtr_base = 0x4000;
        m.cpu_mut().sys.idtr_limit = 0xfff;

        for _ in 0..1000 {
            m.step().unwrap();
        }
        assert!(m.cpu().regs.gpr[R15] > 0, "timer fired and handler ran");
    }

    #[test]
    fn cli_masks_timer_interrupt() {
        // main: cli ; loop: nop ; jmp loop  -> IF stays clear, handler never runs
        let mut program = vec![0xfa, 0x90, 0xeb, 0xfd];
        program.resize(0x100, 0x90);
        program.extend([0x49, 0xff, 0xc7, 0xcf]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        m.set_core_timer_injection(true); // a timer exists, but IF gates it
        set_idt_gate(m.memory_mut(), 0x4000, 32, 0x100);
        m.cpu_mut().sys.idtr_base = 0x4000;
        m.cpu_mut().sys.idtr_limit = 0xfff;

        for _ in 0..1000 {
            m.step().unwrap();
        }
        assert_eq!(m.cpu().regs.gpr[R15], 0, "interrupts masked while IF clear");
    }

    #[test]
    fn page_fault_is_vectored_to_handler() {
        // main @0x6000: mov rax,1 ; mov [0x800000], rax   (faults)
        // handler @0x6100: mov rbx, 0xdead ; hlt
        let mut program = vec![
            0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
            0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x80, 0x00, // mov [0x800000], rax
        ];
        program.resize(0x100, 0x90);
        program.extend([0x48, 0xc7, 0xc3, 0xad, 0xde, 0x00, 0x00, 0xf4]);
        let mut m = X86Machine::new(seed(), 0x100000, &program, 0x6000).unwrap();
        install_identity_low(m.memory_mut());
        set_idt_gate(m.memory_mut(), 0x5000, 14, 0x6100);
        m.cpu_mut().sys.cr3 = 0x1000;
        m.cpu_mut().sys.cr0 |= CR0_PG;
        m.cpu_mut().sys.idtr_base = 0x5000;
        m.cpu_mut().sys.idtr_limit = 0xfff;

        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RBX], 0xdead, "#PF vectored to handler");
        assert_eq!(m.cpu().sys.cr2, 0x800000, "CR2 holds the faulting address");
    }

    #[test]
    fn interrupt_delivery_is_deterministic() {
        let build = || {
            let mut program = vec![0xfb, 0x90, 0xeb, 0xfd];
            program.resize(0x100, 0x90);
            program.extend([0x49, 0xff, 0xc7, 0xcf]);
            let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
            m.set_core_timer_injection(true);
            set_idt_gate(m.memory_mut(), 0x4000, 32, 0x100);
            m.cpu_mut().sys.idtr_base = 0x4000;
            m.cpu_mut().sys.idtr_limit = 0xfff;
            m
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..1000 {
            a.step().unwrap();
            b.step().unwrap();
        }
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn hlt_waits_and_timer_wakes_it() {
        // sti ; loop: hlt ; jmp loop     handler @0x100: inc r15 ; iret
        let mut program = vec![0xfb, 0xf4, 0xeb, 0xfd];
        program.resize(0x100, 0x90);
        program.extend([0x49, 0xff, 0xc7, 0xcf]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        m.set_core_timer_injection(true); // synthetic timer wakes the HLT
        set_idt_gate(m.memory_mut(), 0x4000, 32, 0x100);
        m.cpu_mut().sys.idtr_base = 0x4000;
        m.cpu_mut().sys.idtr_limit = 0xfff;

        for _ in 0..50 {
            m.step().unwrap();
        }
        assert!(m.cpu().regs.gpr[R15] > 0, "timer woke the HLT and ran handler");
        assert!(
            m.instruction_count() > 200,
            "virtual time fast-forwarded through idle (count={})",
            m.instruction_count()
        );
    }

    #[test]
    fn hlt_with_interrupts_disabled_is_terminal() {
        // cli ; hlt  -> nothing can wake it; run reports Halted without spinning.
        let program = [0xfa, 0xf4];
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        let exit = m.run(10_000).unwrap();
        assert_eq!(exit, X86Exit::Halted);
        assert!(m.cpu().halted);
        assert!(m.instruction_count() < 100, "did not fast-forward forever");
    }

    #[test]
    fn idle_fast_forward_is_deterministic() {
        let build = || {
            let mut program = vec![0xfb, 0xf4, 0xeb, 0xfd];
            program.resize(0x100, 0x90);
            program.extend([0x49, 0xff, 0xc7, 0xcf]);
            let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
            m.set_core_timer_injection(true);
            set_idt_gate(m.memory_mut(), 0x4000, 32, 0x100);
            m.cpu_mut().sys.idtr_base = 0x4000;
            m.cpu_mut().sys.idtr_limit = 0xfff;
            m
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..50 {
            a.step().unwrap();
            b.step().unwrap();
        }
        assert_eq!(a.full_state_digest(), b.full_state_digest());
        assert_eq!(a.instruction_count(), b.instruction_count());
    }

    #[test]
    fn wrmsr_rdmsr_round_trip() {
        // ecx=LSTAR ; write 0x5678_1234 ; clobber edx:eax ; read back.
        let program = [
            0xb9, 0x82, 0x00, 0x00, 0xc0, // mov ecx, 0xC0000082 (LSTAR)
            0xb8, 0x34, 0x12, 0x00, 0x00, // mov eax, 0x1234
            0xba, 0x78, 0x56, 0x00, 0x00, // mov edx, 0x5678
            0x0f, 0x30, // wrmsr
            0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
            0xba, 0x00, 0x00, 0x00, 0x00, // mov edx, 0
            0x0f, 0x32, // rdmsr
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX] & 0xffff_ffff, 0x1234);
        assert_eq!(m.cpu().regs.gpr[RDX] & 0xffff_ffff, 0x5678);
        assert_eq!(m.cpu().msrs.lstar, 0x5678_0000_1234);
    }

    #[test]
    fn wrmsr_efer_aliases_control_register() {
        // mov ecx, 0xC0000080 (EFER) ; mov eax, 0x501 ; mov edx, 0 ; wrmsr ; hlt
        let program = [
            0xb9, 0x80, 0x00, 0x00, 0xc0, 0xb8, 0x01, 0x05, 0x00, 0x00, 0xba, 0x00, 0x00, 0x00,
            0x00, 0x0f, 0x30, 0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().sys.efer, 0x501, "EFER MSR aliases SysRegs::efer");
    }

    #[test]
    fn unknown_msr_raises_gp() {
        // mov ecx, 0x12345678 ; rdmsr  -> #GP (no IDT -> propagates as error)
        let program = [0xb9, 0x78, 0x56, 0x34, 0x12, 0x0f, 0x32, 0xf4];
        assert!(matches!(
            try_run(&program),
            Err(X86Error::GeneralProtection)
        ));
    }

    #[test]
    fn rdmsr_tsc_reflects_virtual_time() {
        // A few instructions, then read IA32_TSC (0x10); it should be nonzero and
        // equal to the instruction count observed when RDMSR ran.
        let program = [
            0x90, 0x90, 0x90, // nops
            0xb9, 0x10, 0x00, 0x00, 0x00, // mov ecx, 0x10 (IA32_TSC)
            0x0f, 0x32, // rdmsr
            0xf4,
        ];
        let m = run(&program, 100);
        let tsc = m.cpu().regs.gpr[RAX] | (m.cpu().regs.gpr[RDX] << 32);
        assert!(tsc > 0, "TSC advanced with virtual time");
        assert!(tsc < 20, "TSC ~ instruction count at the RDMSR");
    }

    #[test]
    fn msr_writes_are_deterministic() {
        let program = [
            0xb9, 0x00, 0x01, 0x00, 0xc0, // mov ecx, 0xC0000100 (FS_BASE)
            0xb8, 0xfe, 0xca, 0x00, 0x00, // mov eax, 0xcafe
            0xba, 0xef, 0xbe, 0x00, 0x00, // mov edx, 0xbeef
            0x0f, 0x30, 0xf4, // wrmsr ; hlt
        ];
        let mut a = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        let mut b = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.cpu().msrs.fs_base, 0xbeef_0000_cafe);
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn syscall_enters_kernel_handler() {
        // main @0: syscall ; (pad) ; handler @0x200: mov rbx,0xbeef ; hlt
        let mut program = vec![0x0f, 0x05];
        program.resize(0x200, 0x90);
        program.extend([0x48, 0xc7, 0xc3, 0xef, 0xbe, 0x00, 0x00, 0xf4]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        {
            let c = m.cpu_mut();
            c.sys.efer |= 1; // SCE
            c.msrs.lstar = 0x200;
            c.msrs.star = STAR;
            c.msrs.sfmask = 0x200; // mask IF on entry
        }
        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RBX], 0xbeef, "kernel handler ran");
        assert_eq!(m.cpu().regs.gpr[RCX], 2, "RIP of next instr saved in RCX");
        assert_eq!(m.cpu().cpl, 0);
        assert_eq!(m.cpu().cs, 0x08, "CS from STAR[47:32]");
    }

    #[test]
    fn sysret_returns_to_user() {
        // sysret at 0 ; user nops at the RCX target (0x10).
        let mut program = vec![0x0f, 0x07];
        program.resize(0x40, 0x90);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        {
            let c = m.cpu_mut();
            c.regs.gpr[RCX] = 0x10;
            c.regs.gpr[R11] = 0x202;
            c.msrs.star = STAR;
        }
        for _ in 0..5 {
            m.step().unwrap();
        }
        assert_eq!(m.cpu().cpl, 3, "returned to user");
        assert_eq!(m.cpu().cs, 0x33, "64-bit user CS = STAR[63:48]+16 | RPL3");
        assert!(m.cpu().regs.rip >= 0x10, "resumed at the RCX target");
    }

    #[test]
    fn swapgs_swaps_gs_bases() {
        let program = [0x0f, 0x01, 0xf8, 0xf4]; // swapgs ; hlt
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        {
            let c = m.cpu_mut();
            c.msrs.gs_base = 0xaaaa;
            c.msrs.kernel_gs_base = 0xbbbb;
        }
        m.run(100).unwrap();
        assert_eq!(m.cpu().msrs.gs_base, 0xbbbb);
        assert_eq!(m.cpu().msrs.kernel_gs_base, 0xaaaa);
    }

    #[test]
    fn syscall_round_trip_is_deterministic() {
        let build = || {
            let mut program = vec![0x0f, 0x05];
            program.resize(0x200, 0x90);
            program.extend([0x48, 0xc7, 0xc3, 0xef, 0xbe, 0x00, 0x00, 0xf4]);
            let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
            let c = m.cpu_mut();
            c.sys.efer |= 1;
            c.msrs.lstar = 0x200;
            c.msrs.star = STAR;
            m
        };
        let mut a = build();
        let mut b = build();
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn serial_console_collects_outb() {
        // mov edx,0x3f8 ; mov al,'H'; out dx,al ; mov al,'i'; out dx,al ; hlt
        let program = [
            0xba, 0xf8, 0x03, 0x00, 0x00, 0xb0, 0x48, 0xee, 0xb0, 0x69, 0xee, 0xf4,
        ];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.run(100).unwrap();
        assert_eq!(m.serial().output_string(), "Hi");
    }

    #[test]
    fn in_reads_serial_line_status() {
        // mov edx,0x3fd (LSR) ; in al, dx ; hlt
        let program = [0xba, 0xfd, 0x03, 0x00, 0x00, 0xec, 0xf4];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RAX] & 0xff, 0x60, "THR+transmitter empty");
    }

    fn print_kernel() -> Vec<u8> {
        // mov edx,0x3f8 ; mov al,'O'; out dx,al ; mov al,'K'; out dx,al ; hlt
        vec![
            0xba, 0xf8, 0x03, 0x00, 0x00, 0xb0, 0x4f, 0xee, 0xb0, 0x4b, 0xee, 0xf4,
        ]
    }

    #[test]
    fn boot_handoff_sets_long_mode_and_runs_kernel() {
        let mut m =
            X86Machine::boot(seed(), 0x20_0000, &print_kernel(), 0x10_0000, "console=ttyS0")
                .unwrap();
        // Boot state per the platform contract.
        assert_eq!(m.cpu().regs.gpr[RSI], 0x7000, "RSI -> boot_params");
        assert!(m.cpu().sys.paging_enabled(), "long mode paging enabled");
        assert_eq!(m.cpu().regs.rip, 0x10_0000, "entry at kernel load address");
        assert_eq!(platform::e820_entry_count(m.memory(), 0x7000), 4);
        let main_ram = platform::e820_entry(m.memory(), 0x7000, 3);
        assert_eq!(main_ram.addr, 0x10_0000);
        // The kernel boots (paged) and prints over the serial console.
        m.run(1000).unwrap();
        assert_eq!(m.serial().output_string(), "OK");
    }

    /// A minimal valid bzImage: 1 setup sector, then a payload whose 64-bit entry
    /// (offset 0x200) is `entry_code`.
    fn synthetic_bzimage(entry_code: &[u8]) -> Vec<u8> {
        let mut image = vec![0_u8; 1024]; // setup_sects=1 -> pm_offset = 1024
        image[0x1f1] = 1;
        image[0x1fe] = 0x55;
        image[0x1ff] = 0xaa;
        image[0x202..0x206].copy_from_slice(b"HdrS");
        let mut payload = vec![0x90_u8; 0x200];
        payload.extend_from_slice(entry_code);
        image.extend_from_slice(&payload);
        image
    }

    #[test]
    fn boot_bzimage_parses_loads_and_runs() {
        let image = synthetic_bzimage(&print_kernel());
        let mut m =
            X86Machine::boot_bzimage(seed(), 0x20_0000, &image, "console=ttyS0", None, None).unwrap();
        assert_eq!(m.cpu().regs.rip, 0x10_0200, "64-bit entry = load + 0x200");
        assert_eq!(m.cpu().regs.gpr[RSI], 0x7000, "RSI -> boot_params");
        assert!(m.cpu().sys.paging_enabled());
        // setup_header "HdrS" magic was copied into boot_params.
        assert_eq!(m.memory().read_uint_le(0x7000 + 0x202, 4).unwrap(), 0x5372_6448);
        m.run(1000).unwrap();
        assert_eq!(m.serial().output_string(), "OK");
    }

    #[test]
    fn boot_bzimage_places_initrd_and_sets_ramdisk_header() {
        let image = synthetic_bzimage(&print_kernel());
        let initrd: Vec<u8> = (0..64u8).collect(); // 64-byte stand-in cpio
        let ram = 0x20_0000_u64; // 2 MiB
        let m = X86Machine::boot_bzimage(seed(), ram as usize, &image, "console=ttyS0", Some(&initrd), None)
            .unwrap();
        // ramdisk_image (0x218) / ramdisk_size (0x21c) point at the placed initrd.
        let addr = m.memory().read_uint_le(0x7000 + 0x218, 4).unwrap();
        let size = m.memory().read_uint_le(0x7000 + 0x21c, 4).unwrap();
        assert_eq!(size, initrd.len() as u64, "ramdisk_size = initrd length");
        assert!(addr >= 0x10_0000 && addr + size <= ram, "initrd inside RAM");
        assert_eq!(addr & 0xfff, 0, "initrd page-aligned");
        // The bytes actually landed at the advertised address.
        for (i, b) in initrd.iter().enumerate() {
            assert_eq!(m.memory().read_uint_le(addr + i as u64, 1).unwrap(), *b as u64);
        }
        // A different initrd yields a distinct run identity (determinism input).
        let m2 = X86Machine::boot_bzimage(seed(), ram as usize, &image, "console=ttyS0", None, None)
            .unwrap();
        assert_ne!(m.identity_digest(), m2.identity_digest());
    }

    #[test]
    fn diagnostics_report_the_stop_point() {
        // Print 'X', then hit an unimplemented opcode (0x06) -> stops.
        let kernel = [
            0xba, 0xf8, 0x03, 0x00, 0x00, 0xb0, 0x58, 0xee, // mov edx,0x3f8; mov al,'X'; out
            0x06, // unimplemented opcode
        ];
        let mut m = X86Machine::boot(seed(), 0x20_0000, &kernel, 0x10_0000, "x").unwrap();
        let outcome = m.run(1000);
        assert!(matches!(outcome, Err(X86Error::Unsupported(_))));
        let diag = m.diagnostics();
        assert!(diag.contains("rip="));
        assert!(diag.contains("instructions="));
        assert_eq!(m.serial().output_string(), "X", "printed before stopping");
    }

    #[test]
    fn bad_bzimage_is_rejected() {
        let bad = vec![0_u8; 0x300]; // no boot flag / magic
        assert!(X86Machine::boot_bzimage(seed(), 0x10_0000, &bad, "x", None, None).is_err());
    }

    #[test]
    fn boot_is_deterministic() {
        let mut a =
            X86Machine::boot(seed(), 0x20_0000, &print_kernel(), 0x10_0000, "x").unwrap();
        let mut b =
            X86Machine::boot(seed(), 0x20_0000, &print_kernel(), 0x10_0000, "x").unwrap();
        a.run(1000).unwrap();
        b.run(1000).unwrap();
        assert_eq!(a.serial().output_string(), "OK");
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn fs_relative_addressing_uses_fs_base() {
        // mov rax, fs:[0x10]   with FS base 0x1000  ->  reads physical 0x1010
        let program = [0x64, 0x48, 0x8b, 0x04, 0x25, 0x10, 0x00, 0x00, 0x00, 0xf4];
        let mut m = X86Machine::new(seed(), 0x4000, &program, 0).unwrap();
        m.cpu_mut().msrs.fs_base = 0x1000;
        m.memory_mut().write_uint_le(0x1010, 8, 0xdead_beef).unwrap();
        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RAX], 0xdead_beef);
    }

    #[test]
    fn gs_relative_addressing_uses_gs_base() {
        // mov rax, gs:[0]   with GS base 0x2000
        let program = [0x65, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xf4];
        let mut m = X86Machine::new(seed(), 0x4000, &program, 0).unwrap();
        m.cpu_mut().msrs.gs_base = 0x2000;
        m.memory_mut().write_uint_le(0x2000, 8, 0xcafe).unwrap();
        m.run(100).unwrap();
        assert_eq!(m.cpu().regs.gpr[RAX], 0xcafe);
    }

    #[test]
    fn swapgs_then_gs_access_uses_kernel_base() {
        // swapgs ; mov rax, gs:[0]    GS=0xAAAA, KERNEL_GS=0x2000 (data there)
        let program = [
            0x0f, 0x01, 0xf8, // swapgs
            0x65, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, // mov rax, gs:[0]
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 0x4000, &program, 0).unwrap();
        m.cpu_mut().msrs.gs_base = 0xaaaa;
        m.cpu_mut().msrs.kernel_gs_base = 0x2000;
        m.memory_mut().write_uint_le(0x2000, 8, 0x1234_5678).unwrap();
        m.run(100).unwrap();
        // After SWAPGS, GS base is 0x2000, so the load reads our value.
        assert_eq!(m.cpu().regs.gpr[RAX], 0x1234_5678);
    }

    #[test]
    fn lidt_loads_idtr_and_interrupt_delivers() {
        // lidt [0x300] ; int 0x20 ; hlt    handler @0x100: mov rbx,0xbeef ; iret
        let mut program = vec![
            0x0f, 0x01, 0x1c, 0x25, 0x00, 0x03, 0x00, 0x00, // lidt [0x300]
            0xcd, 0x20, // int 0x20
            0xf4,
        ];
        program.resize(0x100, 0x90);
        program.extend([0x48, 0xc7, 0xc3, 0xef, 0xbe, 0x00, 0x00, 0xcf]);
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        {
            let mem = m.memory_mut();
            mem.write_uint_le(0x300, 2, 0x0fff).unwrap(); // IDT limit
            mem.write_uint_le(0x302, 8, 0x4000).unwrap(); // IDT base
            set_idt_gate(mem, 0x4000, 0x20, 0x100);
        }
        m.run(100).unwrap();
        assert_eq!(m.cpu().sys.idtr_base, 0x4000);
        assert_eq!(m.cpu().sys.idtr_limit, 0x0fff);
        assert_eq!(m.cpu().regs.gpr[RBX], 0xbeef, "INT used the LIDT-loaded IDT");
    }

    #[test]
    fn lgdt_loads_gdtr() {
        // lgdt [0x300] ; hlt
        let program = vec![0x0f, 0x01, 0x14, 0x25, 0x00, 0x03, 0x00, 0x00, 0xf4];
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        m.memory_mut().write_uint_le(0x300, 2, 0x0027).unwrap();
        m.memory_mut().write_uint_le(0x302, 8, 0x5000).unwrap();
        m.run(100).unwrap();
        assert_eq!(m.cpu().sys.gdtr_base, 0x5000);
        assert_eq!(m.cpu().sys.gdtr_limit, 0x27);
    }

    #[test]
    fn descriptor_load_is_deterministic() {
        let build = || {
            let program = vec![0x0f, 0x01, 0x1c, 0x25, 0x00, 0x03, 0x00, 0x00, 0xf4];
            let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
            m.memory_mut().write_uint_le(0x300, 2, 0x0fff).unwrap();
            m.memory_mut().write_uint_le(0x302, 8, 0x4000).unwrap();
            m
        };
        let mut a = build();
        let mut b = build();
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn cpuid_returns_vendor_and_features() {
        // xor eax,eax ; cpuid ; hlt   -> leaf 0 vendor in EBX:EDX:ECX
        let m = run(&[0x48, 0x31, 0xc0, 0x0f, 0xa2, 0xf4], 100);
        // "Genu" / "ntel" / "ineI"
        assert_eq!(m.cpu().regs.gpr[RBX] & 0xffff_ffff, 0x756e_6547);
        assert_eq!(m.cpu().regs.gpr[RCX] & 0xffff_ffff, 0x6c65_746e);
        assert_eq!(m.cpu().regs.gpr[RDX] & 0xffff_ffff, 0x4965_6e69);
        assert_eq!(m.cpu().regs.gpr[RAX] & 0xffff_ffff, 0x0d, "max basic leaf");
    }

    #[test]
    fn cpuid_leaf1_reports_long_mode_features() {
        // mov eax,1 ; cpuid ; hlt
        let m = run(&[0xb8, 0x01, 0x00, 0x00, 0x00, 0x0f, 0xa2, 0xf4], 100);
        let edx = m.cpu().regs.gpr[RDX] & 0xffff_ffff;
        let ecx = m.cpu().regs.gpr[RCX] & 0xffff_ffff;
        assert_ne!(edx & (1 << 26), 0, "SSE2");
        assert_ne!(ecx & (1 << 23), 0, "POPCNT");
        assert_eq!(ecx & (1 << 30), 0, "RDRAND not advertised (unimplemented)");
        assert_ne!(ecx & (1 << 28), 0, "AVX on (x86-64-v3)");
    }

    #[test]
    fn pending_interrupt_changes_full_state_digest() {
        // Two identical machines; queue an interrupt on only one. Their
        // full-state digests must differ (pending IRQs are future-affecting).
        let program = [0x90, 0xf4]; // nop ; hlt
        let mut a = X86Machine::new(seed(), 0x1000, &program, 0).unwrap();
        let b = X86Machine::new(seed(), 0x1000, &program, 0).unwrap();
        a.cpus[a.current].pending.push_back(0x42);
        assert_ne!(
            a.full_state_digest(),
            b.full_state_digest(),
            "a queued interrupt must be visible in the digest"
        );
    }

    #[test]
    fn noncanonical_data_access_raises_gp() {
        // mov rbx, 0x0000_8000_0000_0000 (2^47, first non-canonical) ; mov rax,[rbx]
        // A non-canonical linear address must raise #GP, not translate or #PF.
        let prog = [
            0x48, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, // mov rbx, 2^47
            0x48, 0x8b, 0x03, // mov rax, [rbx]
        ];
        let mut m = X86Machine::new(seed(), 0x10000, &prog, 0).unwrap();
        let err = m.run(10).unwrap_err();
        assert_eq!(err, X86Error::GeneralProtection, "non-canonical -> #GP");
    }

    #[test]
    fn core_timer_is_not_injected_by_default() {
        // sti ; loop: nop ; jmp loop, with an IDT handler on vector 32. Without
        // opting into the synthetic core timer, no x86 interrupt is delivered.
        let mut program = vec![0xfb, 0x90, 0xeb, 0xfd];
        program.resize(0x100, 0x90);
        program.extend([0x49, 0xff, 0xc7, 0xcf]); // inc r15 ; iret
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        set_idt_gate(m.memory_mut(), 0x4000, 32, 0x100);
        m.cpu_mut().sys.idtr_base = 0x4000;
        m.cpu_mut().sys.idtr_limit = 0xfff;
        for _ in 0..1000 {
            m.step().unwrap();
        }
        assert_eq!(
            m.cpu().regs.gpr[R15],
            0,
            "no phantom timer IRQ without modeled platform timer"
        );
    }

    #[test]
    fn idiv_min_by_neg_one_is_divide_error_not_panic() {
        // RDX:RAX = i128::MIN (0x8000…0 : 0) ; IDIV RCX with RCX = -1.
        // The quotient 2^127 is unrepresentable -> #DE. Must NOT panic the host.
        let program = [
            0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rdx, 0x8000…
            0x48, 0x31, 0xc0, // xor rax, rax
            0x48, 0xb9, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // mov rcx, -1
            0x48, 0xf7, 0xf9, // idiv rcx
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        assert_eq!(m.run(100), Err(X86Error::DivideError));
    }

    #[test]
    fn movsd_preserves_upper_on_regreg_and_zero_extends_from_mem() {
        // MOVUPS xmm1,[0x2000] ; mov rax,0x3333…; movq xmm0,rax ;
        // MOVSD xmm1,xmm0 (reg→reg: low replaced, upper preserved) ;
        // MOVUPS [0x3000],xmm1 ; MOVSD xmm1,[0x2010] (mem: zero-extend) ;
        // MOVUPS [0x3010],xmm1 ; hlt.
        let program = [
            0x0f, 0x10, 0x0c, 0x25, 0x00, 0x20, 0x00, 0x00, // movups xmm1,[0x2000]
            0x48, 0xb8, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, // mov rax,0x3333…
            0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0,rax
            0xf2, 0x0f, 0x10, 0xc8, // movsd xmm1,xmm0
            0x0f, 0x11, 0x0c, 0x25, 0x00, 0x30, 0x00, 0x00, // movups [0x3000],xmm1
            0xf2, 0x0f, 0x10, 0x0c, 0x25, 0x10, 0x20, 0x00, 0x00, // movsd xmm1,[0x2010]
            0x0f, 0x11, 0x0c, 0x25, 0x10, 0x30, 0x00, 0x00, // movups [0x3010],xmm1
            0xf4,
        ];
        let mut m = X86Machine::new(seed(), 0x10000, &program, 0).unwrap();
        let mem = m.memory_mut();
        mem.write_uint_le(0x2000, 8, 0x1111_1111_1111_1111).unwrap();
        mem.write_uint_le(0x2008, 8, 0x2222_2222_2222_2222).unwrap();
        mem.write_uint_le(0x2010, 8, 0x4444_4444_4444_4444).unwrap();
        mem.write_uint_le(0x2018, 8, 0x5555_5555_5555_5555).unwrap();
        m.run(100).unwrap();
        let mem = m.memory();
        // reg→reg MOVSD: low replaced by 0x3333…, upper 0x2222… PRESERVED.
        assert_eq!(mem.read_uint_le(0x3000, 8).unwrap(), 0x3333_3333_3333_3333);
        assert_eq!(mem.read_uint_le(0x3008, 8).unwrap(), 0x2222_2222_2222_2222);
        // mem MOVSD: low 0x4444…, upper ZEROED (not 0x5555…, and only 8 bytes read).
        assert_eq!(mem.read_uint_le(0x3010, 8).unwrap(), 0x4444_4444_4444_4444);
        assert_eq!(mem.read_uint_le(0x3018, 8).unwrap(), 0);
    }

    #[test]
    fn movsxd_imul3_test_imm_and_multibyte_nop() {
        let program = [
            0x48, 0xb9, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, // mov rcx,0x80000000
            0x48, 0x63, 0xc1, // movsxd rax, ecx
            0x48, 0xc7, 0xc6, 0x05, 0x00, 0x00, 0x00, // mov rsi, 5
            0x48, 0x6b, 0xde, 0x03, // imul rbx, rsi, 3
            0xa8, 0xff, // test al, 0xff   (AL = 0x00 -> ZF)
            0x0f, 0x1f, 0x00, // nop [rax]  (multi-byte NOP)
            0x49, 0xc7, 0xc7, 0x07, 0x00, 0x00, 0x00, // mov r15, 7
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RAX], 0xffff_ffff_8000_0000, "MOVSXD sign-extends");
        assert_eq!(m.cpu().regs.gpr[RBX], 15, "IMUL r,r/m,imm");
        assert_ne!(m.cpu().regs.rflags & (1 << 6), 0, "TEST AL,imm set ZF (AL=0)");
        assert_eq!(m.cpu().regs.gpr[R15], 7, "execution continued past 0F 1F NOP");
    }

    #[test]
    fn cpuid_is_deterministic() {
        let program = [0xb8, 0x01, 0x00, 0x00, 0x00, 0x0f, 0xa2, 0xf4];
        let mut a = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        let mut b = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        a.run(100).unwrap();
        b.run(100).unwrap();
        assert_eq!(a.full_state_digest(), b.full_state_digest());
    }

    #[test]
    fn lea_computes_effective_address() {
        // mov rax, 0x10 ; lea rbx, [rax + rax*2 + 4] -> 0x10 + 0x20 + 4 = 0x34
        let program = [
            0x48, 0xc7, 0xc0, 0x10, 0x00, 0x00, 0x00, // mov rax, 0x10
            0x48, 0x8d, 0x1c, 0x40, // lea rbx, [rax + rax*2 + 0]  (SIB scale=2,index=rax,base=rax)
            0xf4,
        ];
        // disp byte not present in this encoding; expected 0x10 + 0x20 = 0x30
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x30);
        let _ = Size::Qword;
        let _ = RDX;
    }

    // --- AVX / VEX (iteration 1) ---

    fn decode_one(bytes: &[u8]) -> DecodedInsn {
        let mut f = |a: u64| -> Result<u8, X86Error> { Ok(bytes[a as usize]) };
        decode(&mut f, 0, DecodeMode::Bits64).unwrap()
    }

    #[test]
    fn vex_prefix_field_extraction() {
        // C5 F8 77 = VZEROUPPER (L=0); C5 FC 77 = VZEROALL (L=1).
        assert_eq!(decode_one(&[0xc5, 0xf8, 0x77]).op, Op::VZeroUpper { all: false });
        assert_eq!(decode_one(&[0xc5, 0xfc, 0x77]).op, Op::VZeroUpper { all: true });
        // C5 F1 EF C2 = VPXOR xmm0, xmm1, xmm2 (vvvv=1 inverted, pp=66, op EF).
        let d = decode_one(&[0xc5, 0xf1, 0xef, 0xc2]);
        assert_eq!(d.op, Op::VBitwise { kind: VBitKind::Xor, vvvv: 1, l256: false });
        assert_eq!(d.dst, Some(Operand::Xmm { id: 0 }));
        assert_eq!(d.src, Some(Operand::Xmm { id: 2 }));
        assert_eq!(d.len, 4);
        // VEX.256 VPOR ymm3, ymm5, ymm7 : C5 D5 EB DF (vvvv=5 inv, L=1).
        let d = decode_one(&[0xc5, 0xd5, 0xeb, 0xdf]);
        assert_eq!(d.op, Op::VBitwise { kind: VBitKind::Or, vvvv: 5, l256: true });
        assert_eq!(d.dst, Some(Operand::Xmm { id: 3 }));
        assert_eq!(d.src, Some(Operand::Xmm { id: 7 }));
        // C4 E1 F9 6E C8 = VMOVQ xmm1, rax (3-byte VEX, W=1).
        assert_eq!(decode_one(&[0xc4, 0xe1, 0xf9, 0x6e, 0xc8]).op, Op::VMovGpr { to_xmm: true, bytes: 8 });
    }

    #[test]
    fn vex_movq_roundtrip_and_vpxor_zeroing() {
        // mov rax, 0x1122334455667788
        // vmovq xmm1, rax ; vmovq xmm2, rax ; vpxor xmm0, xmm1, xmm2
        // vmovq rbx, xmm0 ; vmovq rcx, xmm1 ; hlt
        let program = [
            0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, imm64
            0xc4, 0xe1, 0xf9, 0x6e, 0xc8, // vmovq xmm1, rax
            0xc4, 0xe1, 0xf9, 0x6e, 0xd0, // vmovq xmm2, rax
            0xc5, 0xf1, 0xef, 0xc2,       // vpxor xmm0, xmm1, xmm2
            0xc4, 0xe1, 0xf9, 0x7e, 0xc3, // vmovq rbx, xmm0
            0xc4, 0xe1, 0xf9, 0x7e, 0xc9, // vmovq rcx, xmm1
            0xf4,
        ];
        let m = run(&program, 100);
        // VPXOR of equal values → 0; VMOVQ roundtrip preserves the value.
        assert_eq!(m.cpu().regs.gpr[RBX], 0, "vpxor xmm1,xmm1 should be 0");
        assert_eq!(m.cpu().regs.gpr[1], 0x1122334455667788, "vmovq roundtrip"); // RCX
        // VMOVQ gpr→xmm zero-extends the whole register (upper 64 bits cleared).
        assert_eq!(m.cpu().xmm[1], 0x1122334455667788u128);
    }

    #[test]
    fn vex_arith_fp_and_int() {
        // VADDSD: 2.0 + 3.0 = 5.0 (0x4014…). VPADDD: lanes {1,2}+{20,10}={21,12}.
        let program = [
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // mov rax, 2.0
            0xc4, 0xe1, 0xf9, 0x6e, 0xc8, // vmovq xmm1, rax
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // mov rax, 3.0
            0xc4, 0xe1, 0xf9, 0x6e, 0xd0, // vmovq xmm2, rax
            0xc5, 0xf3, 0x58, 0xc2,       // vaddsd xmm0, xmm1, xmm2
            0xc4, 0xe1, 0xf9, 0x7e, 0xc3, // vmovq rbx, xmm0
            0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, // mov rax, {1,2}
            0xc4, 0xe1, 0xf9, 0x6e, 0xd8, // vmovq xmm3, rax
            0x48, 0xb8, 0x14, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, // mov rax, {20,10}
            0xc4, 0xe1, 0xf9, 0x6e, 0xe0, // vmovq xmm4, rax
            0xc5, 0xe1, 0xfe, 0xec,       // vpaddd xmm5, xmm3, xmm4
            0xc4, 0xe1, 0xf9, 0x7e, 0xe9, // vmovq rcx, xmm5
            0xf4,
        ];
        let m = run(&program, 200);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x4014000000000000, "vaddsd 2+3=5");
        assert_eq!(m.cpu().regs.gpr[1], 0x0000000C00000015, "vpaddd lanes"); // RCX
        // VEX scalar/128 zeroes the YMM upper half of the destination.
        assert_eq!(m.cpu().ymm_hi[0], 0);
        assert_eq!(m.cpu().ymm_hi[5], 0);
    }

    #[test]
    fn vex_movmsk() {
        // xmm1 low = 0x80000000 → float dword0 sign set; byte3 sign set.
        let program = [
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, // mov rax, 0x80000000
            0xc4, 0xe1, 0xf9, 0x6e, 0xc8, // vmovq xmm1, rax
            0xc5, 0xf8, 0x50, 0xc1,       // vmovmskps eax, xmm1
            0x48, 0x89, 0xc3,             // mov rbx, rax
            0xc5, 0xf9, 0xd7, 0xc1,       // vpmovmskb eax, xmm1
            0x48, 0x89, 0xc1,             // mov rcx, rax
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 1, "vmovmskps: only float0 sign set");
        assert_eq!(m.cpu().regs.gpr[1], 8, "vpmovmskb: only byte3 sign set"); // RCX
    }

    #[test]
    fn vex_iter3_decodes() {
        // C5 F1 76 C2 = VPCMPEQD xmm0, xmm1, xmm2.
        assert_eq!(
            decode_one(&[0xc5, 0xf1, 0x76, 0xc2]).op,
            Op::VPacked { kind: VPkind::CmpEq(4), vvvv: 1, l256: false }
        );
        // C5 F1 C6 C2 1B = VSHUFPD xmm0, xmm1, xmm2, 0x1b.
        assert_eq!(
            decode_one(&[0xc5, 0xf1, 0xc6, 0xc2, 0x1b]).op,
            Op::VShufps { double: true, vvvv: 1, l256: false, imm: 0x1b }
        );
        // C5 F1 C2 C2 05 = VCMPPD xmm0, xmm1, xmm2, 5 (NLT).
        assert_eq!(
            decode_one(&[0xc5, 0xf1, 0xc2, 0xc2, 0x05]).op,
            Op::VCmpFp { double: true, scalar: false, vvvv: 1, l256: false, imm: 5 }
        );
        // C4 E3 79 0C C1 0A = VBLENDPS xmm0, xmm0, xmm1, 0x0a.
        assert_eq!(
            decode_one(&[0xc4, 0xe3, 0x79, 0x0c, 0xc1, 0x0a]).op,
            Op::VBlend { elem: 4, vvvv: 0, l256: false, imm: 0x0a }
        );
    }

    #[test]
    fn vex_mul_and_imm_shift_decode() {
        // C5 F1 D5 C2 = VPMULLW xmm0, xmm1, xmm2.
        assert_eq!(
            decode_one(&[0xc5, 0xf1, 0xd5, 0xc2]).op,
            Op::VPacked { kind: VPkind::Mul(PMulKind::Lw), vvvv: 1, l256: false }
        );
        // C5 F9 72 F1 05 = VPSLLD xmm0, xmm1, 5 (NDD: dst=vvvv=xmm0, rm=xmm1, /6=SLL).
        assert_eq!(
            decode_one(&[0xc5, 0xf9, 0x72, 0xf1, 0x05]).op,
            Op::VPShiftImm { kind: ShiftKind::Sll, elem: 4, l256: false, imm: 5 }
        );
        // C5 F9 73 F9 0C = VPSLLDQ xmm0, xmm1, 12 (/7 = byte shift left).
        assert_eq!(
            decode_one(&[0xc5, 0xf9, 0x73, 0xf9, 0x0c]).op,
            Op::VPByteShiftImm { left: true, l256: false, imm: 12 }
        );
        // C4 E2 79 28 C1 = VPMULDQ xmm0, xmm0, xmm1.
        assert_eq!(
            decode_one(&[0xc4, 0xe2, 0x79, 0x28, 0xc1]).op,
            Op::VPacked { kind: VPkind::Muldq, vvvv: 0, l256: false }
        );
    }

    #[test]
    fn vex_256_decodes() {
        // C5 FD C6 C1 1B = VSHUFPD ymm0, ymm0, ymm1, 0x1b (L=1, double).
        assert_eq!(
            decode_one(&[0xc5, 0xfd, 0xc6, 0xc1, 0x1b]).op,
            Op::VShufps { double: true, vvvv: 0, l256: true, imm: 0x1b }
        );
        // C4 E3 7D 0C C1 0A = VBLENDPS ymm0, ymm0, ymm1, 0x0a (L=1).
        assert_eq!(
            decode_one(&[0xc4, 0xe3, 0x7d, 0x0c, 0xc1, 0x0a]).op,
            Op::VBlend { elem: 4, vvvv: 0, l256: true, imm: 0x0a }
        );
        // C5 FC 5A C1 = VCVTPS2PD ymm0, ymm1 (widening, L=1).
        assert_eq!(
            decode_one(&[0xc5, 0xfc, 0x5a, 0xc1]).op,
            Op::VCvt { kind: VCvtKind::F2F { to_double: true, scalar: false }, vvvv: 0, l256: true }
        );
    }

    #[test]
    fn vex_lanecross_decodes() {
        // C4 E2 7D 58 C1 = VPBROADCASTD ymm0, xmm1.
        assert_eq!(decode_one(&[0xc4, 0xe2, 0x7d, 0x58, 0xc1]).op, Op::VBroadcast { elem: 4, l256: true });
        // C4 E2 7D 1A 00 = VBROADCASTF128 ymm0, [rax] (elem 16, memory).
        assert_eq!(decode_one(&[0xc4, 0xe2, 0x7d, 0x1a, 0x00]).op, Op::VBroadcast { elem: 16, l256: true });
        // C4 E3 7D 18 C1 01 = VINSERTF128 ymm0, ymm0, xmm1, 1.
        assert_eq!(decode_one(&[0xc4, 0xe3, 0x7d, 0x18, 0xc1, 0x01]).op, Op::VInsert128 { vvvv: 0, lane: 1 });
        // C4 E3 7D 19 C8 01 = VEXTRACTF128 xmm0, ymm1, 1.
        assert_eq!(decode_one(&[0xc4, 0xe3, 0x7d, 0x19, 0xc8, 0x01]).op, Op::VExtract128 { lane: 1 });
        // C4 E3 7D 06 C1 21 = VPERM2F128 ymm0, ymm0, ymm1, 0x21.
        assert_eq!(decode_one(&[0xc4, 0xe3, 0x7d, 0x06, 0xc1, 0x21]).op, Op::VPerm2 { vvvv: 0, imm: 0x21 });
        // C4 E3 FD 00 C1 1B = VPERMQ ymm0, ymm1, 0x1b.
        assert_eq!(decode_one(&[0xc4, 0xe3, 0xfd, 0x00, 0xc1, 0x1b]).op, Op::VPermImm { imm: 0x1b });
        // C4 E2 7D 36 C1 = VPERMD ymm0, ymm0, ymm1.
        assert_eq!(decode_one(&[0xc4, 0xe2, 0x7d, 0x36, 0xc1]).op, Op::VPermVar { vvvv: 0 });
    }

    #[test]
    fn vex_broadcast_fills_ymm() {
        // vpbroadcastd ymm0, xmm1 → all 8 dwords = xmm1 low dword; both halves set.
        let program = [
            0x48, 0xb8, 0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00, // mov rax, 0xdeadbeef
            0xc4, 0xe1, 0xf9, 0x6e, 0xc8, // vmovq xmm1, rax
            0xc4, 0xe2, 0x7d, 0x58, 0xc1, // vpbroadcastd ymm0, xmm1
            0xf4,
        ];
        let m = run(&program, 100);
        let want = 0xdeadbeef_deadbeef_deadbeef_deadbeefu128;
        assert_eq!(m.cpu().xmm[0], want, "broadcast low 128");
        assert_eq!(m.cpu().ymm_hi[0], want, "broadcast high 128");
    }

    #[test]
    fn vex_fma_decode_and_execute() {
        // C4 E2 F1 B8 C2 = VFMADD231PD xmm0, xmm1, xmm2 (W1, 231, packed).
        assert_eq!(
            decode_one(&[0xc4, 0xe2, 0xf1, 0xb8, 0xc2]).op,
            Op::VFma { kind: FmaKind::Madd, form: FmaForm::F231, double: true, packed: true, vvvv: 1, l256: false }
        );
        // C4 E2 71 9c C1 = VFNMADD132PS xmm0, xmm1, xmm1 (W0, 132, nmadd).
        assert_eq!(
            decode_one(&[0xc4, 0xe2, 0x71, 0x9c, 0xc1]).op,
            Op::VFma { kind: FmaKind::Nmadd, form: FmaForm::F132, double: false, packed: true, vvvv: 1, l256: false }
        );
        // Execute VFMADD231SD: xmm1*xmm2 + xmm0 = 2*3 + 1 = 7.
        let program = [
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // mov rax, 1.0
            0xc4, 0xe1, 0xf9, 0x6e, 0xc0, // vmovq xmm0, rax
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // mov rax, 2.0
            0xc4, 0xe1, 0xf9, 0x6e, 0xc8, // vmovq xmm1, rax
            0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // mov rax, 3.0
            0xc4, 0xe1, 0xf9, 0x6e, 0xd0, // vmovq xmm2, rax
            0xc4, 0xe2, 0xf1, 0xb9, 0xc2, // vfmadd231sd xmm0, xmm1, xmm2
            0xc4, 0xe1, 0xf9, 0x7e, 0xc3, // vmovq rbx, xmm0
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[RBX], 0x401C000000000000, "2*3+1 = 7.0");
    }

    #[test]
    fn xsave_xrstor_roundtrips_ymm() {
        // Broadcast a value across YMM0, enable AVX state (XCR0=7), XSAVE, clobber
        // with VZEROALL, XRSTOR — the full 256-bit YMM0 must come back intact.
        let program = [
            0x48, 0xb8, 0x01, 0x00, 0xfe, 0xca, 0x00, 0x00, 0x00, 0x00, // mov rax, 0xCAFE0001
            0xc4, 0xe1, 0xf9, 0x6e, 0xc0, // vmovq xmm0, rax
            0xc4, 0xe2, 0x7d, 0x59, 0xc0, // vpbroadcastq ymm0, xmm0
            0xb9, 0x00, 0x00, 0x00, 0x00, // mov ecx, 0
            0xb8, 0x07, 0x00, 0x00, 0x00, // mov eax, 7
            0xba, 0x00, 0x00, 0x00, 0x00, // mov edx, 0
            0x0f, 0x01, 0xd1,             // xsetbv  → XCR0 = 7
            0xbf, 0x00, 0x04, 0x00, 0x00, // mov edi, 0x400  (XSAVE buffer, in 4 KiB RAM)
            0xb8, 0x07, 0x00, 0x00, 0x00, // mov eax, 7  (mask)
            0xba, 0x00, 0x00, 0x00, 0x00, // mov edx, 0
            0x0f, 0xae, 0x27,             // xsave [rdi]
            0xc5, 0xfc, 0x77,             // vzeroall  (clobber all YMM)
            0xb8, 0x07, 0x00, 0x00, 0x00, // mov eax, 7
            0xba, 0x00, 0x00, 0x00, 0x00, // mov edx, 0
            0x0f, 0xae, 0x2f,             // xrstor [rdi]
            0xf4,
        ];
        let m = run(&program, 200);
        let want = 0x00000000cafe0001_00000000cafe0001u128;
        assert_eq!(m.cpu().xcr0, 7, "xsetbv set XCR0");
        assert_eq!(m.cpu().xmm[0], want, "XRSTOR restored YMM0 low 128");
        assert_eq!(m.cpu().ymm_hi[0], want, "XRSTOR restored YMM0 high 128");
    }

    #[test]
    fn movbe_byte_reverses_through_memory() {
        let program = [
            0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, 0x1122334455667788
            0xbb, 0x00, 0x04, 0x00, 0x00, // mov ebx, 0x400
            0x48, 0x89, 0x03,             // mov [rbx], rax
            0x48, 0x0f, 0x38, 0xf0, 0x0b, // movbe rcx, [rbx]
            0xf4,
        ];
        let m = run(&program, 100);
        assert_eq!(m.cpu().regs.gpr[1], 0x8877665544332211, "movbe byte-reverses");
    }

    #[test]
    fn v3_gpr_decodes() {
        // F3 48 0F BD C3 = LZCNT rax, rbx.
        assert_eq!(decode_one(&[0xf3, 0x48, 0x0f, 0xbd, 0xc3]).op, Op::Lzcnt);
        // C4 E2 E0 F2 C1 = ANDN rax, rbx, rcx (vvvv=rbx=3).
        assert_eq!(decode_one(&[0xc4, 0xe2, 0xe0, 0xf2, 0xc1]).op, Op::Andn { vvvv: 3 });
        // C4 E2 FB F6 D9 = MULX rax, rbx, rcx (vvvv=rax=0).
        assert_eq!(decode_one(&[0xc4, 0xe2, 0xfb, 0xf6, 0xd9]).op, Op::Mulx { vvvv: 0 });
        // C4 E3 FB F0 C3 1F = RORX rax, rbx, 0x1f.
        assert_eq!(decode_one(&[0xc4, 0xe3, 0xfb, 0xf0, 0xc3, 0x1f]).op, Op::Rorx { imm: 0x1f });
    }

    #[test]
    fn fcmov_decodes() {
        // DA C8 = FCMOVE st0,st0 ; DB D1 = FCMOVNBE st0,st1.
        assert_eq!(decode_one(&[0xda, 0xc8]).op, Op::Fcmov { i: 0, cond: Cond::E });
        assert_eq!(decode_one(&[0xdb, 0xd1]).op, Op::Fcmov { i: 1, cond: Cond::A });
        assert_eq!(decode_one(&[0xda, 0xc1]).op, Op::Fcmov { i: 1, cond: Cond::B });
    }

    #[test]
    fn vzeroupper_clears_upper_halves() {
        // Seed ymm_hi, execute VZEROUPPER, confirm uppers cleared but xmm intact.
        let program = [0xc5, 0xf8, 0x77, 0xf4]; // vzeroupper ; hlt
        let mut m = X86Machine::new(seed(), 4096, &program, 0).unwrap();
        m.cpu_mut().xmm[3] = 0xdead;
        m.cpu_mut().ymm_hi[3] = 0xbeef;
        m.run(100).unwrap();
        assert_eq!(m.cpu().ymm_hi[3], 0, "upper half cleared");
        assert_eq!(m.cpu().xmm[3], 0xdead, "lower half preserved");
    }
}
