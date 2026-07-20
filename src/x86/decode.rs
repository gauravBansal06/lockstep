//! x86-64 long-mode instruction decoder for the implemented integer subset.
//!
//! Handles legacy/REX prefixes, ModRM, SIB, displacement, and immediates.
//! Register-based memory operands keep their components and are resolved against
//! the register file at execute time; RIP-relative operands are resolved here
//! (RIP at end-of-instruction is known after the immediate).

use super::regs::{RBP, RBX, RDI, RSI, RSP};
use super::{Size, X86Error};

/// A condition code (the `tttn` field shared by Jcc / SETcc / CMOVcc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    O,
    No,
    B,
    Ae,
    E,
    Ne,
    Be,
    A,
    S,
    Ns,
    P,
    Np,
    L,
    Ge,
    Le,
    G,
}

impl Cond {
    fn from_tttn(tttn: u8) -> Cond {
        match tttn & 0xf {
            0x0 => Cond::O,
            0x1 => Cond::No,
            0x2 => Cond::B,
            0x3 => Cond::Ae,
            0x4 => Cond::E,
            0x5 => Cond::Ne,
            0x6 => Cond::Be,
            0x7 => Cond::A,
            0x8 => Cond::S,
            0x9 => Cond::Ns,
            0xa => Cond::P,
            0xb => Cond::Np,
            0xc => Cond::L,
            0xd => Cond::Ge,
            0xe => Cond::Le,
            _ => Cond::G,
        }
    }
}

/// Operation of a decoded instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Mov,
    /// MOV to a segment register (`0x8e`): field = segment number
    /// (0=ES,1=CS,2=SS,3=DS,4=FS,5=GS); source is an r/m16 selector.
    MovToSeg(u8),
    /// MOV from a segment register (`0x8c`): field = segment number; the selector
    /// is stored to the r/m destination.
    MovFromSeg(u8),
    Add,
    Or,
    And,
    Sub,
    Xor,
    Cmp,
    Test,
    Adc,
    Sbb,
    Cmov(Cond),
    Set(Cond),
    Inc,
    Dec,
    Lea,
    Push,
    Pop,
    Jmp,
    /// Far direct jump (`JMP ptr16:N`, opcode 0xEA): reload CS with `selector` and set IP to
    /// `offset`. Used by the real→protected→long mode-switch trampolines.
    JmpFar { selector: u16, offset: u64 },
    Jcc(Cond),
    Call,
    Ret,
    /// Far return (`RETF`/`LRETQ`, 0xcb/0xca): pop RIP then CS (64-bit operands);
    /// `rel` carries the optional stack-adjust immediate.
    Retf,
    /// LEAVE (0xc9): `RSP = RBP; RBP = pop()` — the frame-pointer epilogue.
    Leave,
    Movzx,
    Movsx,
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
    Mul,
    Imul,
    Div,
    Idiv,
    Neg,
    Not,
    Cli,
    Sti,
    /// Clear/set/complement the carry flag (CLC/STC/CMC).
    Clc,
    Stc,
    Cmc,
    /// Clear/set the direction flag (CLD/STD).
    Cld,
    Std,
    Iret,
    Int,
    Rdmsr,
    Wrmsr,
    Cpuid,
    Syscall,
    Sysret,
    Swapgs,
    /// INVLPG: TLB-less by design, so architecturally a NOP — but it must flush the
    /// decode cache (a single-page remap can change the bytes at a cached rip).
    Invlpg,
    Lgdt,
    Lidt,
    /// SGDT/SIDT (0F 01 /0, /1): store the GDTR/IDTR (limit:base) to memory.
    Sgdt,
    Sidt,
    /// VERR/VERW (0F 00 /4, /5): segment-accessible check → sets ZF (always
    /// accessible in this flat model).
    Verw,
    /// LTR (0F 00 /3): load the task register; resolves the TSS base from the GDT
    /// descriptor so privilege-increasing interrupts can read TSS.RSP0.
    Ltr,
    In,
    Out,
    Rdtsc,
    Rdtscp,
    Xchg,
    Xadd,
    Cmpxchg,
    Bt,
    Bts,
    Btr,
    Btc,
    Bsf,
    Bsr,
    // ---- x86-64-v3 GPR instructions (BMI1/BMI2, LZCNT/TZCNT, MOVBE) ----
    Lzcnt, // count leading zeros (CF=src0, ZF=res0)
    Tzcnt, // count trailing zeros
    /// MOVBE: byte-reversed move (load `store=false`, store `store=true`).
    Movbe { store: bool },
    /// ANDN: `dst = ~vvvv & rm` (BMI1). `vvvv` is the inverted source register.
    Andn { vvvv: u8 },
    /// BLSI(3)/BLSMSK(2)/BLSR(1): single-source bit ops; `dst = vvvv` (NDD), `src = rm`.
    Blsx { kind: u8 },
    /// BEXTR: `dst = (rm >> vvvv[7:0]) & ((1<<vvvv[15:8])-1)` (BMI1).
    Bextr { vvvv: u8 },
    /// BZHI: zero bits of `rm` at and above index `vvvv[7:0]` (BMI2).
    Bzhi { vvvv: u8 },
    /// MULX: `rdx * rm` (unsigned, no flags). ModRM.reg (`dst`) = high half,
    /// `vvvv` = low half.
    Mulx { vvvv: u8 },
    /// PDEP/PEXT: parallel bit deposit/extract. value=`vvvv`, mask=`rm` (BMI2).
    Pdep { vvvv: u8 },
    Pext { vvvv: u8 },
    /// RORX: `dst = rm ror imm8` (no flags, BMI2).
    Rorx { imm: u8 },
    /// SARX/SHRX/SHLX: flagless shift; value=`rm`, count=`vvvv` (BMI2).
    Sarx { kind: ShiftKind, vvvv: u8 },
    Popcnt,
    Str(StrOp, Size, Rep),
    MovCr { write: bool, reg: u8 },
    MovDr { write: bool, reg: u8 },
    CallIndirect,
    JmpIndirect,
    Pushf,
    Popf,
    Lahf,
    Sahf,
    /// SSE data move into an XMM register (width 4/8/16 bytes; <16 zero-extends).
    XmmLoad(u8),
    /// SSE data move out of an XMM register (width 4/8/16 bytes).
    XmmStore(u8),
    /// Scalar SSE move MOVSS(4)/MOVSD(8): a memory source zero-extends the
    /// destination XMM, but a register→register move **preserves** the upper
    /// bits of the destination (x86 scalar-move semantics).
    XmmScalarMov(u8),
    /// MOVLPS/MOVHPS/MOVLPD/MOVHPD (0F 12/13/16/17): move a 64-bit half between
    /// memory and the low (`high=false`) or high (`high=true`) quadword of an XMM,
    /// preserving the other half. The register-form load variants are the
    /// cross-half moves MOVHLPS (0F 12) and MOVLHPS (0F 16).
    XmmMovHalf { high: bool, store: bool },
    /// SSE float arithmetic. `double` = f64 lanes (else f32); `packed` = all lanes
    /// (else scalar low-lane only, preserving the rest of dst). F2=SD, F3=SS,
    /// 66=PD, none=PS.
    FBin(FBinKind, bool, bool),
    /// UNPCKLPS/HPS/LPD/HPD (0F 14/15): interleave lanes from dst and src. `high`
    /// selects the high half as the source lanes; `double` = 64-bit lanes.
    UnpckFp { high: bool, double: bool },
    /// PACKSSWB/PACKSSDW/PACKUSWB (0F 63/6B/67): pack two vectors of wider elements
    /// into narrower ones with saturation. `dword_src` = dword→word (else word→byte);
    /// `signed` = signed saturation (else unsigned, byte only).
    Pack { dword_src: bool, signed: bool },
    /// Packed float→int convert. `double_src` = f64×2→i32×2 (CVT[T]PD2DQ, 0F E6),
    /// else f32×4→i32×4 (CVT[T]PS2DQ, 0F 5B). `trunc` = truncate (else round-nearest).
    CvtF2I { double_src: bool, trunc: bool },
    /// Packed int→float convert. `double_dst` = i32×2→f64×2 (CVTDQ2PD, F3 0F E6),
    /// else i32×4→f32×4 (CVTDQ2PS, 0F 5B).
    CvtI2F { double_dst: bool },
    /// Float precision convert (0F 5A). `to_double` = single→double (else double→
    /// single); `scalar` = low lane only (CVTSS2SD/CVTSD2SS) else packed (CVTPS2PD/
    /// CVTPD2PS).
    CvtF2F { to_double: bool, scalar: bool },
    /// Packed integer multiply family (66 prefix): PMULLW (D5), PMULHW (E5),
    /// PMULHUW (E4), PMULUDQ (F4), PMADDWD (F5).
    PMul(PMulKind),
    /// PSHUFB (SSSE3, 66 0F 38 00): shuffle dst's bytes by src's per-byte indices;
    /// a src byte with bit7 set zeroes that result lane.
    Pshufb,
    /// PMULLD (SSE4.1, 66 0F 38 40): packed 32-bit multiply, low 32 bits per lane.
    Pmulld,
    /// PALIGNR (SSSE3, 66 0F 3A 0F): byte-align — concat dst:src, shift right by an
    /// imm8 byte count (in `rel`), take the low 128 bits.
    Palignr,
    /// ROUNDPS/PD/SS/SD (SSE4.1, 66 0F 3A 08-0B): round float lanes to integral by
    /// the imm8 mode (in `rel`). `double` = f64; `scalar` = low lane only.
    Round { double: bool, scalar: bool },
    /// ADDSUBPD/PS (SSE3, 66/F2 0F D0): alternately subtract (even lanes) and add
    /// (odd lanes). `double` = f64 lanes.
    AddSub { double: bool },
    /// HADDPD/PS, HSUBPD/PS (SSE3, 66/F2 0F 7C/7D): horizontal add/sub of adjacent
    /// lane pairs (dst pairs → low half, src pairs → high half). `sub` = subtract.
    Hadd { double: bool, sub: bool },
    /// MOVDDUP (F2 0F 12): duplicate the low f64 into both lanes.
    Movddup,
    /// MOVSLDUP (F3 0F 12): duplicate even f32 lanes; MOVSHDUP (F3 0F 16): odd lanes.
    Movsdup { high: bool },
    /// PMOVSX/PMOVZX (SSE4.1, 0F 38 20-25/30-35): take the low lanes of src and
    /// sign/zero-extend each `src_elem`-byte element to `dst_elem` bytes.
    Pmovx { src_elem: u8, dst_elem: u8, sign: bool },
    /// PTEST (SSE4.1, 66 0F 38 17): set ZF = (dst&src)==0, CF = (src&~dst)==0; no
    /// register write.
    Ptest,
    /// PABSB/W/D (SSSE3, 66 0F 38 1C/1D/1E): per-lane absolute value.
    Pabs { elem: u8 },
    /// CMPXCHG8B (0F C7 /1) / CMPXCHG16B (REX.W): atomic compare-exchange of a
    /// 64/128-bit memory operand against RDX:RAX, storing RCX:RBX on match.
    /// `wide` = 128-bit (the x86-64-v2 form).
    Cmpxchg16b { wide: bool },
    /// RDRAND (0F C7 /6) / RDSEED (0F C7 /7), register form. Writes a seed-derived
    /// deterministic value to the dest GPR and sets CF=1 (always "succeeds"); OF/SF/
    /// ZF/AF/PF cleared. `seed` distinguishes RDSEED (same deterministic source here).
    Rdrand { seed: bool },
    /// PHADDW/D, PHSUBW/D, PHADDSW/PHSUBSW (SSSE3, 0F 38 01-07): horizontal add/sub
    /// of adjacent lane pairs. `elem` 2/4; `sub`; `sat` = signed-saturate (word only).
    Phadd { elem: u8, sub: bool, sat: bool },
    /// PSIGNB/W/D (SSSE3, 0F 38 08-0A): apply src's per-lane sign to dst (negate if
    /// src<0, zero if src==0, else keep).
    Psign { elem: u8 },
    /// PMADDUBSW (SSSE3, 0F 38 04): unsigned×signed byte products summed in pairs,
    /// signed-saturated to words.
    Pmaddubsw,
    /// PMULHRSW (SSSE3, 0F 38 0B): signed word multiply, rounded high 16 bits.
    Pmulhrsw,
    /// BLENDPS/PD, PBLENDW (SSE4.1, 0F 3A 0C/0D/0E): per-lane select dst/src by an
    /// imm8 mask (in `rel`). `elem` = lane bytes (4/8/2).
    Blend { elem: u8 },
    /// BLENDVPS/PD, PBLENDVB (SSE4.1, 0F 38 14/15/10): per-lane select by the high
    /// bit of the corresponding lane in the implicit mask register XMM0. `elem` 4/8/1.
    Blendv { elem: u8 },
    /// INSERTPS (SSE4.1, 0F 3A 21): insert one src dword into a dst lane, then zero
    /// lanes per the imm8 (in `rel`).
    Insertps,
    /// PHMINPOSUW (SSE4.1, 0F 38 41): minimum unsigned word + its index into lane 0.
    Phminposuw,
    /// DPPS/DPPD (SSE4.1, 0F 3A 40/41): dot product, lane select by imm8 (in `rel`).
    Dpp { double: bool },
    /// MPSADBW (SSE4.1, 0F 3A 42): sums of absolute byte differences, imm8 offsets.
    Mpsadbw,
    /// CRC32 (SSE4.2, F2 0F 38 F0/F1): accumulate a CRC-32C of `bytes` source bytes
    /// into the GPR destination.
    Crc32 { bytes: u8 },
    /// PINSRB/W/D/Q (SSE4.1, 0F 3A 20/22, +66 0F C4 for word): insert a GPR/mem
    /// element into the xmm lane selected by imm8. `elem` = lane bytes.
    Pinsr { elem: u8 },
    /// PEXTRB/W/D/Q + EXTRACTPS (SSE4.1, 0F 3A 14/15/16/17): extract the xmm lane
    /// selected by imm8 into a GPR/mem. `elem` = lane bytes.
    Pextr { elem: u8 },
    /// PCMPESTRI/M + PCMPISTRI/M (SSE4.2, 0F 3A 60-63): packed string compare by an
    /// imm8 control. `explicit` = lengths from EAX/EDX (else implicit null-terminated);
    /// `mask_out` = write XMM0 mask (PCMP*STRM) else ECX index (PCMP*STRI). Sets flags.
    Pcmpstr { explicit: bool, mask_out: bool },
    /// Scalar SSE square root.
    FSqrt(bool, bool),
    Rcl,
    Rcr,
    Shld,
    Shrd,
    Bswap,
    /// CDQ/CQO: sign-extend the accumulator into RDX.
    Cdq,
    /// CWDE/CDQE: sign-extend the accumulator in place.
    Cwde,
    Loop(LoopKind),
    /// 128-bit SSE bitwise op (PXOR/POR/PAND/PANDN).
    PLogic(PLogicKind),
    /// Packed compare-equal, lane-wise → all-ones/all-zero. Field = element width
    /// in bytes (1/2/4 = PCMPEQB/W/D).
    PackedCmpEq(u8),
    /// Packed signed compare-greater-than, lane-wise (PCMPGTB/W/D).
    PackedCmpGt(u8),
    /// Packed integer add, lane-wise wrapping. Field = element width in bytes
    /// (1/2/4/8 = PADDB/PADDW/PADDD/PADDQ).
    PackedAdd(u8),
    /// Packed integer subtract, lane-wise wrapping (PSUBB/W/D/Q).
    PackedSub(u8),
    /// Packed lane-wise min/max: PMINUB/PMAXUB (unsigned bytes) and PMINSW/PMAXSW
    /// (signed words). `elem` = element width in bytes (1 or 2).
    PackedMinMax { max: bool, signed: bool, elem: u8 },
    /// Packed shift (PSLL/PSRL/PSRA). Count comes from the source operand
    /// (xmm/m128 low 64 bits, or an immediate). Field = element width in bytes.
    PackedShift { kind: ShiftKind, elem: u8 },
    /// PSRLDQ/PSLLDQ (0F 73 /3, /7): shift the whole 128-bit register by a byte
    /// count (`left` = PSLLDQ). Zeros shift in.
    XmmByteShift { left: bool },
    /// PSHUFD: shuffle the four 32-bit lanes of the source by an imm8 selector.
    Pshufd,
    /// PSHUFLW (`high=false`) / PSHUFHW (`high=true`): legacy SSE2 word-shuffle of the
    /// low (or high) 64 bits by imm8; the other 64 bits copy through. Preserves YMM upper.
    PshufW { high: bool },
    /// PUNPCK{L,H}{BW,WD,DQ,QDQ}: interleave the low (or high) halves, element
    /// width in bytes (1/2/4/8); `high` selects the high-half variant.
    Punpck { high: bool, elem: u8 },
    /// PMOVMSKB: byte sign-bit mask of an XMM into a GPR.
    Pmovmskb,
    /// CVTSI2SD/SS: signed integer → scalar float (`double` = SD).
    Cvtsi2f(bool),
    /// VCVTSI2SD/SS (VEX F2/F3 0F 2A): signed int (GPR/mem) → scalar FP. Unlike the
    /// SSE form, the dest's upper lane comes from the `vvvv` (src1) reg and the YMM
    /// upper half is zeroed.
    VCvtsi2f { double: bool, vvvv: u8 },
    /// VMOVSS/VMOVSD reg-reg merge form (VEX F3/F2 0F 10/11): dst low `bytes` from the
    /// src, dst upper from `vvvv` (src1), YMM upper zeroed. (The MEMORY forms reuse
    /// `VMovGpr` — zero-extend on load, write exactly `bytes` on store. Treating these
    /// scalar moves as a packed 16-byte `VMov` corrupts adjacent memory on store.)
    VMovScalarMerge { bytes: u8, vvvv: u8 },
    /// VPINSRB/D/Q (VEX 0F3A 20/22 ib): dst = `vvvv` with element[imm] replaced by the
    /// integer src (GPR/mem); YMM upper zeroed. `elem` = 1/4/8 bytes.
    VPinsr { elem: u8, vvvv: u8 },
    /// VPMOVZX/SX* (VEX 0F38 20-25/30-35): widen `src_elem`→`dst_elem`, sign/zero-extend;
    /// YMM upper zeroed (l256 = 256-bit dst from 128-bit src).
    VPmovx { src_elem: u8, dst_elem: u8, sign: bool, l256: bool },
    /// VPTEST (VEX 0F38 17): ZF=(dst&src==0), CF=(~dst&src==0) over the FULL width
    /// (256-bit when `l256` — Go's bytealg compares ymm, so the upper half matters).
    VPtest { l256: bool },
    /// VMOVDDUP (VEX F2 0F 12): duplicate the even doubles; YMM upper zeroed (l256).
    VMovddup { l256: bool },
    /// VPERMILPS/PD: in-lane float permute. `double` = PD (2 qwords/lane) else PS
    /// (4 dwords/lane). `variable` = control from the r/m src2 (data in `vvvv`);
    /// else control is `imm8` (in `rel`) and data is the r/m src. YMM upper zeroed.
    VPermil { double: bool, variable: bool, vvvv: u8, l256: bool },
    /// VSQRTSS/SD/PS/PD (VEX 0F 51). `packed`=PS/PD (all lanes, l256-aware), else
    /// scalar: low lane = sqrt(src2), upper bits from `vvvv` (src1). YMM zeroed.
    VFSqrt { double: bool, packed: bool, vvvv: u8, l256: bool },
    /// CVTTSD2SI/SS: scalar float → signed integer, truncating.
    Cvttf2si(bool),
    /// UCOMISD/SS: ordered scalar compare, sets CF/ZF/PF.
    Ucomis(bool),
    /// CMPPS/CMPPD/CMPSS/CMPSD (0F C2): per-lane FP compare by an imm8 predicate
    /// (carried in `rel`), writing all-ones/all-zeros per lane. `double` selects
    /// f64 vs f32; `scalar` compares only the low lane (preserving dst's high bits).
    XmmCmp { double: bool, scalar: bool },
    /// SHUFPS (0F C6) / SHUFPD (66 0F C6): shuffle packed lanes by an imm8 (in
    /// `rel`). The low half is selected from dst, the high half from src.
    Shuffle { double: bool },
    /// MOVMSKPS (0F 50) / MOVMSKPD (66 0F 50): extract the per-lane sign bits of an
    /// xmm into a GPR (4 bits for single, 2 for double).
    Movmsk { double: bool },
    Nop,
    /// PAUSE (`F3 90`): architecturally a NOP, but a spin-wait hint. The SMP scheduler
    /// treats it as a yield point (like a hypervisor's pause-loop-exit) so a core spinning
    /// on a lock immediately gives way to the lock holder instead of burning its quantum.
    Pause,
    /// MFENCE / SFENCE / LFENCE (`0F AE E8..FF`): a memory barrier. Under the x86-TSO model
    /// (Stage 2) it drains this core's store buffer to shared memory; with TSO off the buffer
    /// is always empty, so it behaves exactly like the NOP it used to decode to.
    Fence,
    Hlt,
    /// x87 FPU control-surface ops (state management only, no stack arithmetic).
    Fninit,           // FNINIT: reset CW/SW/TW
    Fnclex,           // FNCLEX: clear exception flags
    Fldcw,            // FLDCW m16: load control word
    Fnstcw,           // FNSTCW m16: store control word
    FnstswAx,         // FNSTSW AX: status word → AX
    FnstswMem,        // FNSTSW m16: status word → memory
    Fwait,            // FWAIT/WAIT: no pending unmasked exceptions → no-op
    /// FXSAVE/FXRSTOR (0F AE /0, /1): save/restore the 512-byte FP/SSE state area
    /// (we model FCW/FSW/FTW + MXCSR + XMM; x87 stack regs are zeroed).
    Fxsave,
    Fxrstor,
    /// LDMXCSR/STMXCSR (0F AE /2, /3): load/store the SSE control/status register.
    Ldmxcsr,
    Stmxcsr,
    /// XSAVE/XSAVEOPT (`compacted=false`) / XSAVEC/XSAVES (`compacted=true`): save
    /// x87+SSE+AVX(YMM) state. For our component set the compacted layout matches
    /// the standard one (AVX is the only extended component, at offset 576); only
    /// the XCOMP_BV header bit differs. Mask = EDX:EAX & XCR0.
    Xsave { compacted: bool },
    /// XRSTOR/XRSTORS (0F AE /5, 0F C7 /3): restore per the XSTATE_BV header.
    Xrstor,
    /// XGETBV/XSETBV (0F 01 D0/D1): read/write an extended control register (XCR0).
    Xgetbv,
    Xsetbv,
    // ---- x87 FPU stack arithmetic ----
    /// FLD/FILD a memory float/int onto the stack. `bytes` = 2/4/8/10; `integer`.
    FpuLoadMem { bytes: u8, integer: bool },
    /// FST/FSTP/FIST/FISTP/FISTTP: store ST(0) to memory. `pop`; `trunc` = FISTTP.
    FpuStoreMem { bytes: u8, integer: bool, pop: bool, trunc: bool },
    /// FLD ST(i): push a copy of ST(i).
    FpuLdSt { i: u8 },
    /// FST/FSTP ST(i): copy ST(0) to ST(i), optional pop.
    FpuStSt { i: u8, pop: bool },
    /// FADD/FSUB/FSUBR/FMUL/FDIV/FDIVR with a memory operand (ST(0) op= mem).
    FpuArithMem { op: FpuBin, bytes: u8, integer: bool },
    /// Register-form arithmetic. `to_st0` → ST(0) = ST(0) op ST(i) (D8); else
    /// ST(i) = ST(i) op ST(0) (DC/DE); `pop` pops after (DE / *P forms).
    FpuArithSt { op: FpuBin, i: u8, to_st0: bool, pop: bool },
    /// FXCH ST(i): swap ST(0) and ST(i).
    Fxch { i: u8 },
    /// FLDZ/FLD1/FLDPI/FLDL2E/FLDL2T/FLDLG2/FLDLN2: push a constant (`which` 0-6).
    FpuConst { which: u8 },
    Fchs,      // FCHS: negate ST(0)
    Fabs,      // FABS
    Fsqrt,     // FSQRT
    Frndint,   // FRNDINT: round ST(0) to integer per RC
    /// FPREM (round-toward-zero quotient) / FPREM1 (round-to-nearest, IEEE):
    /// ST(0) = ST(0) - ST(1)*Q. Algebraic, fully deterministic. We always
    /// complete in one step (C2=0) using an exact f64 remainder.
    Fprem { ieee: bool },
    /// FSCALE: ST(0) = ST(0) * 2^trunc(ST(1)). Algebraic, deterministic.
    Fscale,
    /// x87 transcendentals (D9 F0-FF). Implementation-defined on real silicon (~1 ulp,
    /// CPU-varying) — so we define our OWN DETERMINISTIC results via committed soft-float
    /// polynomials (never host libm), validated by determinism + accuracy, not bit-exactness.
    /// `Fsincos` pushes cos then leaves sin on top; `Fptan` pushes 1.0 after tan; `Fxtract`
    /// splits ST(0) into exponent (→ST(0)) and significand (pushed).
    Ftrig { kind: FpuTrig },
    /// FCMOVcc: if `cond` holds (per EFLAGS), ST(0) = ST(i). No flag changes.
    Fcmov { i: u8, cond: Cond },

    // ---- AVX (VEX-encoded) — iteration 1: vzeroupper, moves, bitwise ----
    /// VZEROUPPER (`all=false`: zero YMM[255:128] of all regs) / VZEROALL
    /// (`all=true`: zero all of YMM0-15). No operands.
    VZeroUpper { all: bool },
    /// VEX move (VMOVUPS/UPD/APS/APD/DQU/DQA): copy `src`→`dst`, zero-extending to
    /// the YMM width. `l256` selects 256-bit; 128-bit zeroes the upper half (VEX rule).
    VMov { l256: bool },
    /// VEX bitwise: `dst = vvvv KIND src` (lane-agnostic over 128/256 bits).
    VBitwise { kind: VBitKind, vvvv: u8, l256: bool },
    /// VMOVD/VMOVQ between a GPR/memory and the low element of an XMM (`bytes` 4/8).
    /// `to_xmm`: GPR→XMM (zero-extends whole YMM); else XMM→GPR.
    VMovGpr { to_xmm: bool, bytes: u8 },
    /// VEX FP arithmetic: `dst = vvvv OP src` (VADD/SUB/MUL/DIV/MIN/MAX, PS/PD/SS/SD).
    /// Scalar forms keep `vvvv`'s upper bits and zero YMM[255:128].
    VFBin { kind: FBinKind, double: bool, packed: bool, vvvv: u8, l256: bool },
    /// VEX packed integer add/sub: `dst = vvvv ± src` per `elem`-byte lane.
    VPackedArith { elem: u8, sub: bool, vvvv: u8, l256: bool },
    /// VEX two-source packed op backed by a shared lane helper (compare, min/max,
    /// unpack, shift-by-xmm, pack, pshufb). `dst = vvvv OP src`.
    VPacked { kind: VPkind, vvvv: u8, l256: bool },
    /// VSHUFPS/VSHUFPD (imm8): `dst = shuffle(vvvv, src, imm)`.
    VShufps { double: bool, vvvv: u8, l256: bool, imm: u8 },
    /// VPSHUFD (`high=None`) / VPSHUFLW (`Some(false)`) / VPSHUFHW (`Some(true)`),
    /// one source + imm8.
    VPshuf { high: Option<bool>, l256: bool, imm: u8 },
    /// VCMPPS/PD/SS/SD (imm8 predicate, 0-31): `dst = cmp(vvvv, src, pred)`.
    VCmpFp { double: bool, scalar: bool, vvvv: u8, l256: bool, imm: u8 },
    /// VBLENDPS/PD / VPBLENDW (imm8 lane select).
    VBlend { elem: u8, vvvv: u8, l256: bool, imm: u8 },
    /// VBLENDVPS/PD / VPBLENDVB (control register = imm8[7:4]).
    VBlendv { elem: u8, vvvv: u8, l256: bool, ctrl: u8 },
    /// VPALIGNR (imm8): byte-align of the concat [src(low):vvvv(high)].
    VPalignr { vvvv: u8, l256: bool, imm: u8 },
    /// VCVT* convert. `vvvv` supplies preserved upper bits in scalar forms. `l256`
    /// selects the 256-bit form (input or output is a YMM, per the convert's width).
    VCvt { kind: VCvtKind, vvvv: u8, l256: bool },
    /// VMOVMSKPS/PD (sign-bit extract to GPR). `double`=PD; `l256`=YMM (8 vs 4/2 bits).
    VMovmsk { double: bool, l256: bool },
    /// VPMOVMSKB (byte sign bits → GPR). `l256`=32 bits, else 16.
    VPmovmskb { l256: bool },
    /// VEX NDD packed shift-by-imm8 (group 71/72/73): dst=vvvv, src=r/m. Element shift.
    VPShiftImm { kind: ShiftKind, elem: u8, l256: bool, imm: u8 },
    /// VPSLLDQ/VPSRLDQ: whole-128-lane byte shift by imm8 (`left`=PSLLDQ).
    VPByteShiftImm { left: bool, l256: bool, imm: u8 },
    /// VPBROADCAST{B,W,D,Q} / VBROADCAST{SS,SD,F128,I128}: replicate the low
    /// `elem`-byte element (or, `elem`=16, the whole 128-bit src) across the dest.
    VBroadcast { elem: u8, l256: bool },
    /// VINSERTF128/VINSERTI128 (imm[0]=lane): dst = vvvv with 128-bit lane replaced by src.
    VInsert128 { vvvv: u8, lane: u8 },
    /// VEXTRACTF128/VEXTRACTI128 (imm[0]=lane): dst(xmm/m128) = lane of src(ymm).
    VExtract128 { lane: u8 },
    /// VPERM2F128/VPERM2I128 (imm): each 128-bit out lane selected from the four
    /// input 128-bit lanes {vvvv.lo, vvvv.hi, src.lo, src.hi} (imm bit3/7 = zero).
    VPerm2 { vvvv: u8, imm: u8 },
    /// VPERMQ/VPERMPD (imm): permute the 4 qwords of `src` by imm8 (2 bits each).
    VPermImm { imm: u8 },
    /// VPERMD/VPERMPS: per-dword cross-lane permute; indices in vvvv, data in src.
    VPermVar { vvvv: u8 },
    /// FMA: fused `±(a*b)±c` with a single rounding. dst is operand 1; `form`
    /// arranges {dst, vvvv, src} into (a, b, c). `double`=PD/SD, `packed`=not scalar.
    VFma { kind: FmaKind, form: FmaForm, double: bool, packed: bool, vvvv: u8, l256: bool },
    /// VCVTPH2PS (F16C): convert packed half-floats → singles. `l256`: 8 halves→8 singles.
    VCvtph2ps { l256: bool },
    /// VCVTPS2PH (F16C): convert packed singles → halves (imm8 rounding). dst is r/m.
    VCvtps2ph { l256: bool, imm: u8 },
    Ftst,      // FTST: compare ST(0) with 0.0
    Fxam,      // FXAM: classify ST(0) into C0-C3
    /// FCOM/FCOMP/FCOMPP/FUCOM* and FCOMI/FUCOMI* compare. `with_mem`/`mem_bytes`/
    /// `integer` for memory operands; `sti` for the register; `pops` (0/1/2);
    /// `eflags` = set EFLAGS (FCOMI family) else x87 C0-C3.
    FpuCompare { with_mem: bool, mem_bytes: u8, integer: bool, sti: u8, pops: u8, eflags: bool },
    Ffree { i: u8 },   // FFREE ST(i)
    Fincstp,           // FINCSTP / FDECSTP
    Fdecstp,
}

/// x87 binary arithmetic selector (R = reversed operands).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpuBin {
    Add,
    Sub,
    SubR,
    Mul,
    Div,
    DivR,
}

/// x87 transcendental selector (D9 F0-FF).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpuTrig {
    F2xm1,   // 2^ST(0) − 1, |ST(0)| ≤ 1
    Fyl2x,   // ST(1)·log2(ST(0)), then pop → ST(0)
    Fptan,   // tan(ST(0)) → ST(0), then push 1.0
    Fpatan,  // atan(ST(1)/ST(0)) → ST(1), then pop
    Fxtract, // ST(0) → exponent; push significand
    Fyl2xp1, // ST(1)·log2(ST(0)+1), then pop
    Fsincos, // sin(ST(0)) → ST(0), push cos(ST(0))
    Fsin,    // sin(ST(0))
    Fcos,    // cos(ST(0))
}

/// 128-bit SSE bitwise operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PLogicKind {
    Xor,
    Or,
    And,
    Andn,
}

/// Packed shift direction/type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    /// Logical left (PSLL).
    Sll,
    /// Logical right (PSRL).
    Srl,
    /// Arithmetic right (PSRA).
    Sra,
}

/// LOOP family variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    Loop,   // dec RCX, jump if RCX != 0
    Loope,  // ...and ZF
    Loopne, // ...and !ZF
    Jrcxz,  // jump if RCX == 0 (no decrement)
}

/// Scalar SSE binary float operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FBinKind {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// VEX bitwise kinds: `dst = src1 KIND src2` over the full register width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VBitKind {
    And,
    AndN, // (!src1) & src2
    Or,
    Xor,
}

/// VEX two-source packed-op kinds backed by shared lane helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VPkind {
    CmpEq(u8),
    CmpGt(u8),
    MinMax { max: bool, signed: bool, elem: u8 },
    Unpack { high: bool, elem: u8 },
    Shift { kind: ShiftKind, elem: u8 },
    Pack { dword_src: bool, signed: bool },
    Pshufb,
    Mul(PMulKind), // PMULLW/HW/HUW/UDQ/MADDWD
    Mulld,
    Muldq,
    Maddubsw,
    Mulhrsw,
    /// Saturating packed add/sub: PADDS/PSUBS (signed) or PADDUS/PSUBUS (unsigned).
    SatArith { sub: bool, signed: bool, elem: u8 },
    /// PAVGB/PAVGW: unsigned rounded average `(a+b+1)>>1`.
    Avg { elem: u8 },
    /// PSADBW: sum of absolute byte differences, one u16 result per 64-bit lane.
    Sadbw,
}

/// FMA sign pattern: `±(a*b) ± c` (AddSub/SubAdd alternate per lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmaKind {
    Madd,    // +ab+c
    Msub,    // +ab-c
    Nmadd,   // -ab+c
    Nmsub,   // -ab-c
    Maddsub, // even lane ab-c, odd lane ab+c
    Msubadd, // even lane ab+c, odd lane ab-c
}

/// FMA operand form (which of dst/vvvv/src are the two multiplicands and the addend).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmaForm {
    F132, // a=dst, b=src, c=vvvv
    F213, // a=vvvv, b=dst, c=src
    F231, // a=vvvv, b=src, c=dst
}

/// VEX convert kinds (128-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VCvtKind {
    F2I { double_src: bool, trunc: bool },
    I2F { double_dst: bool },
    F2F { to_double: bool, scalar: bool },
}

/// Packed integer multiply variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PMulKind {
    /// PMULLW: low 16 bits of word×word.
    Lw,
    /// PMULHW: high 16 bits of signed word×word.
    Hw,
    /// PMULHUW: high 16 bits of unsigned word×word.
    Huw,
    /// PMULUDQ: unsigned low-dword × low-dword → qword, per 64-bit lane.
    Uludq,
    /// PMADDWD: signed word×word pairs summed into dwords.
    Maddwd,
}

/// String operation kind (implicit RSI/RDI/RCX/AL operands).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrOp {
    Movs,
    Stos,
    Lods,
    Scas,
    Cmps,
}

/// Repeat prefix on a string op (raw 0xF3 / 0xF2; meaning depends on the op).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rep {
    None,
    F3,
    F2,
}

/// Segment that supplies the base for a memory operand. In long mode only FS/GS
/// have a (nonzero) base; everything else is flat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seg {
    None,
    Cs,
    Ss,
    Ds,
    Es,
    Fs,
    Gs,
}

/// A decoded operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg {
        id: usize,
        size: Size,
        high8: bool,
    },
    /// Effective address = seg_base + disp + base + index*scale, resolved at
    /// execute time.
    Mem {
        base: Option<usize>,
        index: Option<usize>,
        scale: u8,
        disp: i64,
        size: Size,
        seg: Seg,
        /// RIP-relative: `disp` is relative to the END of the instruction and is
        /// rebased to an absolute address by `finish` (which knows the full length,
        /// including any trailing immediate). Always `false` after `finish`.
        rip_rel: bool,
    },
    Imm {
        value: u64,
        size: Size,
    },
    /// An XMM (SSE) register.
    Xmm {
        id: usize,
    },
}

/// A fully decoded instruction plus its byte length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedInsn {
    pub len: u64,
    pub op: Op,
    pub dst: Option<Operand>,
    pub src: Option<Operand>,
    pub rel: Option<i64>,
}

/// The CPU operating mode the decoder assumes, which sets the DEFAULT operand and address size
/// (togglable by the 0x66 / 0x67 prefixes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeMode {
    Bits16, // real mode / 16-bit: default operand & address size 16
    Bits32, // 32-bit protected: default operand & address size 32
    #[default]
    Bits64, // long mode: default operand 32 (REX.W→64), default address size 64
}

struct Prefixes {
    opsize: bool,
    /// Address-size override (0x67) seen.
    addrsize: bool,
    rex: u8,
    has_rex: bool,
    seg: Seg,
    rep: Rep,
    mode: DecodeMode,
}

impl Default for Prefixes {
    fn default() -> Self {
        Self {
            opsize: false,
            addrsize: false,
            rex: 0,
            has_rex: false,
            seg: Seg::None,
            rep: Rep::None,
            mode: DecodeMode::Bits64,
        }
    }
}

impl Prefixes {
    fn rex_w(&self) -> bool {
        self.rex & 0x8 != 0
    }
    fn rex_r(&self) -> usize {
        ((self.rex >> 2) & 1) as usize
    }
    fn rex_x(&self) -> usize {
        ((self.rex >> 1) & 1) as usize
    }
    fn rex_b(&self) -> usize {
        (self.rex & 1) as usize
    }
}

/// Instruction bytes are fetched on demand via this closure, so the caller can
/// translate the linear instruction-pointer through paging (and set A bits)
/// without the decoder knowing about the MMU.
type Fetch<'a> = &'a mut dyn FnMut(u64) -> Result<u8, X86Error>;

struct Cursor<'a> {
    fetch: Fetch<'a>,
    ip: u64,
}

impl Cursor<'_> {
    fn peek(&mut self) -> Result<u8, X86Error> {
        (self.fetch)(self.ip)
    }
    fn u8(&mut self) -> Result<u8, X86Error> {
        let v = (self.fetch)(self.ip)?;
        self.ip += 1;
        Ok(v)
    }
    fn uint(&mut self, bytes: usize) -> Result<u64, X86Error> {
        let mut v = 0_u64;
        for i in 0..bytes {
            let b = (self.fetch)(self.ip)?;
            self.ip += 1;
            v |= (b as u64) << (i * 8);
        }
        Ok(v)
    }
    /// Read an immediate of `bytes`, optionally sign-extended to 64 bits.
    fn imm(&mut self, bytes: usize, sign: bool) -> Result<u64, X86Error> {
        let raw = self.uint(bytes)?;
        if sign && bytes < 8 {
            let shift = 64 - bytes * 8;
            Ok((((raw << shift) as i64) >> shift) as u64)
        } else {
            Ok(raw)
        }
    }
}

/// Raw ModRM r/m before RIP-relative finalization.
enum RmRaw {
    Reg(usize),
    Mem {
        base: Option<usize>,
        index: Option<usize>,
        scale: u8,
        disp: i64,
        rip_rel: bool,
    },
}

/// Operand size from prefixes; `byte` forces 8-bit.
fn op_size(p: &Prefixes, byte: bool) -> Size {
    if byte {
        Size::Byte
    } else if p.rex_w() {
        Size::Qword
    } else if p.mode == DecodeMode::Bits16 {
        // 16-bit mode: default operand size is Word; 0x66 selects Dword.
        if p.opsize { Size::Dword } else { Size::Word }
    } else {
        // 32-/64-bit mode: default operand size is Dword; 0x66 selects Word.
        if p.opsize { Size::Word } else { Size::Dword }
    }
}

fn reg_operand(num: usize, size: Size, has_rex: bool) -> Operand {
    if size == Size::Byte && !has_rex && (4..=7).contains(&num) {
        Operand::Reg {
            id: num - 4,
            size,
            high8: true,
        }
    } else {
        Operand::Reg {
            id: num,
            size,
            high8: false,
        }
    }
}

/// Decode an x87 escape instruction (D8-DF). Returns the raw `DecodedInsn`
/// (len 0, operands possibly RIP-relative); the caller wraps it with `finish`.
fn decode_x87(c: &mut Cursor, p: &Prefixes, opcode: u8) -> Result<DecodedInsn, X86Error> {
    let modrm = c.peek()?;
    let reg = (modrm >> 3) & 7;
    let i = modrm & 7;
    let plain = |op: Op| DecodedInsn { len: 0, op, dst: None, src: None, rel: None };
    let unsup = || X86Error::Unsupported(format!("x87 {opcode:02x} {modrm:02x}"));
    // reg-field 0-7 → arithmetic kind. `rev` swaps SUB/SUBR and DIV/DIVR (the
    // DC/DE register and integer forms use the reversed sense).
    let arith = |r: u8, rev: bool| -> Option<FpuBin> {
        Some(match r {
            0 => FpuBin::Add,
            1 => FpuBin::Mul,
            4 => if rev { FpuBin::SubR } else { FpuBin::Sub },
            5 => if rev { FpuBin::Sub } else { FpuBin::SubR },
            6 => if rev { FpuBin::DivR } else { FpuBin::Div },
            7 => if rev { FpuBin::Div } else { FpuBin::DivR },
            _ => return None,
        })
    };
    let cmp = |sti: u8, pops: u8, eflags: bool| Op::FpuCompare {
        with_mem: false,
        mem_bytes: 0,
        integer: false,
        sti,
        pops,
        eflags,
    };

    if modrm >> 6 == 3 {
        // ---- register form ----
        let _ = c.u8(); // consume modrm
        let op = match opcode {
            0xd8 => match reg {
                2 => cmp(i, 0, false), // FCOM
                3 => cmp(i, 1, false), // FCOMP
                r => Op::FpuArithSt { op: arith(r, false).ok_or_else(unsup)?, i, to_st0: true, pop: false },
            },
            0xd9 => match modrm {
                0xc0..=0xc7 => Op::FpuLdSt { i },
                0xc8..=0xcf => Op::Fxch { i },
                0xd0 => Op::Fwait, // FNOP
                0xe0 => Op::Fchs,
                0xe1 => Op::Fabs,
                0xe4 => Op::Ftst,
                0xe5 => Op::Fxam,
                0xe8 => Op::FpuConst { which: 0 }, // FLD1
                0xe9 => Op::FpuConst { which: 1 }, // FLDL2T
                0xea => Op::FpuConst { which: 2 }, // FLDL2E
                0xeb => Op::FpuConst { which: 3 }, // FLDPI
                0xec => Op::FpuConst { which: 4 }, // FLDLG2
                0xed => Op::FpuConst { which: 5 }, // FLDLN2
                0xee => Op::FpuConst { which: 6 }, // FLDZ
                0xf0 => Op::Ftrig { kind: FpuTrig::F2xm1 },
                0xf1 => Op::Ftrig { kind: FpuTrig::Fyl2x },
                0xf2 => Op::Ftrig { kind: FpuTrig::Fptan },
                0xf3 => Op::Ftrig { kind: FpuTrig::Fpatan },
                0xf4 => Op::Ftrig { kind: FpuTrig::Fxtract },
                0xf5 => Op::Fprem { ieee: true },  // FPREM1
                0xf6 => Op::Fdecstp,
                0xf7 => Op::Fincstp,
                0xf8 => Op::Fprem { ieee: false }, // FPREM
                0xf9 => Op::Ftrig { kind: FpuTrig::Fyl2xp1 },
                0xfa => Op::Fsqrt,
                0xfb => Op::Ftrig { kind: FpuTrig::Fsincos },
                0xfc => Op::Frndint,
                0xfd => Op::Fscale,
                0xfe => Op::Ftrig { kind: FpuTrig::Fsin },
                0xff => Op::Ftrig { kind: FpuTrig::Fcos },
                _ => return Err(unsup()),
            },
            0xda => match modrm {
                0xe9 => cmp(1, 2, false), // FUCOMPP
                // FCMOVB/E/BE/U ST(0), ST(i) — condition from (modrm-0xc0)>>3.
                0xc0..=0xdf => {
                    let cond = match (modrm >> 3) & 3 {
                        0 => Cond::B,  // FCMOVB
                        1 => Cond::E,  // FCMOVE
                        2 => Cond::Be, // FCMOVBE
                        _ => Cond::P,  // FCMOVU
                    };
                    Op::Fcmov { i, cond }
                }
                _ => return Err(unsup()),
            },
            0xdb => match modrm {
                0xe2 => Op::Fnclex,
                0xe3 => Op::Fninit,
                0xe0 | 0xe1 | 0xe4 => Op::Fwait, // FNENI/FNDISI/FNSETPM
                // FCMOVNB/NE/NBE/NU ST(0), ST(i).
                0xc0..=0xdf => {
                    let cond = match (modrm >> 3) & 3 {
                        0 => Cond::Ae, // FCMOVNB
                        1 => Cond::Ne, // FCMOVNE
                        2 => Cond::A,  // FCMOVNBE
                        _ => Cond::Np, // FCMOVNU
                    };
                    Op::Fcmov { i, cond }
                }
                0xe8..=0xf7 => cmp(i, 0, true), // FUCOMI / FCOMI
                _ => return Err(unsup()),
            },
            0xdc => match reg {
                2 => cmp(i, 0, false),
                3 => cmp(i, 1, false),
                r => Op::FpuArithSt { op: arith(r, true).ok_or_else(unsup)?, i, to_st0: false, pop: false },
            },
            0xdd => match modrm {
                0xc0..=0xc7 => Op::Ffree { i },
                0xd0..=0xd7 => Op::FpuStSt { i, pop: false }, // FST ST(i)
                0xd8..=0xdf => Op::FpuStSt { i, pop: true },  // FSTP ST(i)
                0xe0..=0xe7 => cmp(i, 0, false),              // FUCOM
                0xe8..=0xef => cmp(i, 1, false),              // FUCOMP
                _ => return Err(unsup()),
            },
            0xde => match modrm {
                0xd9 => cmp(1, 2, false), // FCOMPP
                _ => Op::FpuArithSt { op: arith(reg, true).ok_or_else(unsup)?, i, to_st0: false, pop: true },
            },
            0xdf => match modrm {
                0xe0 => Op::FnstswAx,
                0xc0..=0xc7 => Op::Ffree { i }, // FFREEP (pop ignored)
                0xe8..=0xf7 => cmp(i, 1, true), // FUCOMIP / FCOMIP
                _ => return Err(unsup()),
            },
            _ => return Err(unsup()),
        };
        return Ok(plain(op));
    }

    // ---- memory form ----
    let (_r, rm) = decode_modrm(c, p)?;
    // (op, operand size in bytes)
    let (op, bytes): (Op, u8) = match opcode {
        0xd8 => match reg {
            2 => (Op::FpuCompare { with_mem: true, mem_bytes: 4, integer: false, sti: 0, pops: 0, eflags: false }, 4),
            3 => (Op::FpuCompare { with_mem: true, mem_bytes: 4, integer: false, sti: 0, pops: 1, eflags: false }, 4),
            r => (Op::FpuArithMem { op: arith(r, false).ok_or_else(unsup)?, bytes: 4, integer: false }, 4),
        },
        0xd9 => match reg {
            0 => (Op::FpuLoadMem { bytes: 4, integer: false }, 4),
            2 => (Op::FpuStoreMem { bytes: 4, integer: false, pop: false, trunc: false }, 4),
            3 => (Op::FpuStoreMem { bytes: 4, integer: false, pop: true, trunc: false }, 4),
            5 => (Op::Fldcw, 2),
            7 => (Op::Fnstcw, 2),
            _ => return Err(unsup()),
        },
        0xda => match reg {
            2 => (Op::FpuCompare { with_mem: true, mem_bytes: 4, integer: true, sti: 0, pops: 0, eflags: false }, 4),
            3 => (Op::FpuCompare { with_mem: true, mem_bytes: 4, integer: true, sti: 0, pops: 1, eflags: false }, 4),
            r => (Op::FpuArithMem { op: arith(r, false).ok_or_else(unsup)?, bytes: 4, integer: true }, 4),
        },
        0xdb => match reg {
            0 => (Op::FpuLoadMem { bytes: 4, integer: true }, 4),  // FILD m32
            1 => (Op::FpuStoreMem { bytes: 4, integer: true, pop: true, trunc: true }, 4), // FISTTP m32
            2 => (Op::FpuStoreMem { bytes: 4, integer: true, pop: false, trunc: true }, 4),
            3 => (Op::FpuStoreMem { bytes: 4, integer: true, pop: true, trunc: true }, 4),
            5 => (Op::FpuLoadMem { bytes: 10, integer: false }, 10), // FLD m80
            7 => (Op::FpuStoreMem { bytes: 10, integer: false, pop: true, trunc: false }, 10),
            _ => return Err(unsup()),
        },
        0xdc => match reg {
            2 => (Op::FpuCompare { with_mem: true, mem_bytes: 8, integer: false, sti: 0, pops: 0, eflags: false }, 8),
            3 => (Op::FpuCompare { with_mem: true, mem_bytes: 8, integer: false, sti: 0, pops: 1, eflags: false }, 8),
            r => (Op::FpuArithMem { op: arith(r, false).ok_or_else(unsup)?, bytes: 8, integer: false }, 8),
        },
        0xdd => match reg {
            0 => (Op::FpuLoadMem { bytes: 8, integer: false }, 8),
            1 => (Op::FpuStoreMem { bytes: 8, integer: true, pop: true, trunc: true }, 8), // FISTTP m64
            2 => (Op::FpuStoreMem { bytes: 8, integer: false, pop: false, trunc: false }, 8),
            3 => (Op::FpuStoreMem { bytes: 8, integer: false, pop: true, trunc: false }, 8),
            7 => (Op::FnstswMem, 2),
            _ => return Err(unsup()),
        },
        0xde => match reg {
            2 => (Op::FpuCompare { with_mem: true, mem_bytes: 2, integer: true, sti: 0, pops: 0, eflags: false }, 2),
            3 => (Op::FpuCompare { with_mem: true, mem_bytes: 2, integer: true, sti: 0, pops: 1, eflags: false }, 2),
            r => (Op::FpuArithMem { op: arith(r, false).ok_or_else(unsup)?, bytes: 2, integer: true }, 2),
        },
        0xdf => match reg {
            0 => (Op::FpuLoadMem { bytes: 2, integer: true }, 2), // FILD m16
            1 => (Op::FpuStoreMem { bytes: 2, integer: true, pop: true, trunc: true }, 2),
            2 => (Op::FpuStoreMem { bytes: 2, integer: true, pop: false, trunc: true }, 2),
            3 => (Op::FpuStoreMem { bytes: 2, integer: true, pop: true, trunc: true }, 2),
            5 => (Op::FpuLoadMem { bytes: 8, integer: true }, 8), // FILD m64
            7 => (Op::FpuStoreMem { bytes: 8, integer: true, pop: true, trunc: true }, 8),
            _ => return Err(unsup()),
        },
        _ => return Err(unsup()),
    };
    let size = match bytes {
        2 => Size::Word,
        8 => Size::Qword,
        _ => Size::Dword, // 4; m80 handled specially in exec via the operand address
    };
    let mem = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
    // Loads/compare-with-mem read the operand (src); stores/control write it (dst).
    let is_store = matches!(op, Op::FpuStoreMem { .. } | Op::Fnstcw | Op::FnstswMem);
    let (dst, src) = if is_store {
        (Some(mem), None)
    } else {
        (None, Some(mem))
    };
    Ok(DecodedInsn { len: 0, op, dst, src, rel: None })
}

/// Decode the ModRM (and SIB/displacement) byte. Returns the `reg` field number
/// and the raw r/m operand.
/// Whether the current instruction uses 16-bit addressing (16-bit ModRM forms, no SIB): the
/// default in real mode, or under a 0x67 override in 32-bit mode. 64-bit never uses it.
fn addr16(p: &Prefixes) -> bool {
    match p.mode {
        DecodeMode::Bits16 => !p.addrsize,
        DecodeMode::Bits32 => p.addrsize,
        DecodeMode::Bits64 => false,
    }
}

/// Operand size of a PUSH/POP register (0x50-0x5F). PUSH/POP default to the *stack* operand
/// size: 64-bit in long mode (REX.W irrelevant; the 0x66 prefix is intentionally kept as
/// Qword here to preserve byte-identical long-mode behaviour), 32-bit in a 32-bit segment
/// (0x66 → 16), 16-bit in a 16-bit segment (0x66 → 32). This is what a `66`-prefixed
/// `push eax` in the real-mode AP trampoline needs to move 4 bytes, not 2.
fn stack_operand_size(p: &Prefixes) -> Size {
    match p.mode {
        DecodeMode::Bits64 => Size::Qword,
        DecodeMode::Bits32 => {
            if p.opsize {
                Size::Word
            } else {
                Size::Dword
            }
        }
        DecodeMode::Bits16 => {
            if p.opsize {
                Size::Dword
            } else {
                Size::Word
            }
        }
    }
}

/// Byte width of a near JMP/CALL `rel` displacement (0xE8/0xE9 and the 0F 8x near Jcc):
/// rel16 in a 16-bit code segment, rel32 in 32-bit. In long mode the near relative is
/// ALWAYS rel32 (the 0x66 operand-size prefix is ignored for branch displacement), so we
/// force 4 there to keep every validated 64-bit boot byte-identical.
fn near_rel_bytes(p: &Prefixes) -> usize {
    match p.mode {
        DecodeMode::Bits64 => 4,
        _ => match op_size(p, false) {
            Size::Word => 2,
            _ => 4,
        },
    }
}

/// Effective address size in bytes for the current mode + 0x67 override. This is the width
/// of an absolute `moffs` displacement (and the address-register width generally): 2 in
/// 16-bit addressing, 4 in 32-bit, 8 in 64-bit (long mode, where 0x67 drops it to 4).
fn addr_bytes(p: &Prefixes) -> usize {
    if addr16(p) {
        2
    } else {
        match p.mode {
            DecodeMode::Bits64 => 8, // long-mode default (matches the prior hardcoded moffs width)
            _ => 4,                  // Bits32 (the 16-bit-addressing case took the addr16 branch above)
        }
    }
}

fn decode_modrm(c: &mut Cursor, p: &Prefixes) -> Result<(usize, RmRaw), X86Error> {
    let modrm = c.u8()?;
    let md = modrm >> 6;
    let reg = ((modrm >> 3) & 0x7) as usize | (p.rex_r() << 3);
    let rm3 = (modrm & 0x7) as usize;

    if md == 0b11 {
        return Ok((reg, RmRaw::Reg(rm3 | (p.rex_b() << 3))));
    }

    // 16-bit addressing: the rm field selects a fixed base/index pair; there is no SIB byte and
    // the displacement is disp8 (mod=01) or disp16 (mod=10, or mod=00 with rm=6).
    if addr16(p) {
        let (mut base, index): (Option<usize>, Option<usize>) = match rm3 {
            0 => (Some(RBX), Some(RSI)),
            1 => (Some(RBX), Some(RDI)),
            2 => (Some(RBP), Some(RSI)),
            3 => (Some(RBP), Some(RDI)),
            4 => (None, Some(RSI)),
            5 => (None, Some(RDI)),
            6 => (Some(RBP), None), // mod=00 rm=6 is disp16 (fixed up below)
            _ => (Some(RBX), None),
        };
        let disp = match md {
            0b00 => {
                if rm3 == 6 {
                    base = None;
                    c.imm(2, true)? as i64
                } else {
                    0
                }
            }
            0b01 => c.imm(1, true)? as i64,
            _ => c.imm(2, true)? as i64, // mod=10
        };
        return Ok((reg, RmRaw::Mem { base, index, scale: 1, disp, rip_rel: false }));
    }

    // Memory operand.
    let mut base: Option<usize> = None;
    let mut index: Option<usize> = None;
    let mut scale: u8 = 1;
    let mut rip_rel = false;

    if rm3 == 0b100 {
        // SIB byte.
        let sib = c.u8()?;
        scale = 1 << (sib >> 6);
        let idx = ((sib >> 3) & 0x7) as usize | (p.rex_x() << 3);
        let bse = (sib & 0x7) as usize | (p.rex_b() << 3);
        if idx != RSP {
            index = Some(idx);
        }
        if (sib & 0x7) == 0b101 && md == 0 {
            base = None; // disp32, no base
        } else {
            base = Some(bse);
        }
    } else if rm3 == 0b101 && md == 0 {
        // RIP-relative disp32.
        rip_rel = true;
    } else {
        base = Some(rm3 | (p.rex_b() << 3));
    }

    let disp = match md {
        0b00 => {
            if rip_rel || base.is_none() {
                c.imm(4, true)? as i64
            } else {
                0
            }
        }
        0b01 => c.imm(1, true)? as i64,
        0b10 => c.imm(4, true)? as i64,
        _ => unreachable!(),
    };

    Ok((
        reg,
        RmRaw::Mem {
            base,
            index,
            scale,
            disp,
            rip_rel,
        },
    ))
}

/// Resolve a raw r/m into an [`Operand`]. RIP-relative operands keep their raw
/// (instruction-end-relative) displacement and a `rip_rel` flag; `finish` rebases
/// them once the full instruction length (including any trailing immediate) is
/// known. `_ip_end` is unused now (kept for call-site compatibility).
fn finalize_rm(rm: RmRaw, size: Size, _ip_end: u64, has_rex: bool, seg: Seg) -> Operand {
    match rm {
        RmRaw::Reg(num) => reg_operand(num, size, has_rex),
        RmRaw::Mem {
            base,
            index,
            scale,
            disp,
            rip_rel,
        } => {
            // An explicit FS/GS override DOES apply to a RIP-relative operand in
            // long mode (e.g. `mov %rax, %gs:var(%rip)` for per-cpu data) — its
            // base must be added. Only the default segment is ignored, and that is
            // already `Seg::None` with base 0 here, so keep `seg` as decoded.
            Operand::Mem {
                base,
                index,
                scale,
                disp,
                size,
                seg,
                rip_rel,
            }
        }
    }
}

/// Build a two-operand decoded instruction (len filled in by `finish`).
fn sse(op: Op, dst: Operand, src: Operand) -> DecodedInsn {
    DecodedInsn {
        len: 0,
        op,
        dst: Some(dst),
        src: Some(src),
        rel: None,
    }
}

/// Resolve a raw r/m as an XMM register (mod=11) or a memory operand.
fn finalize_rm_xmm(rm: RmRaw, ip_end: u64, seg: Seg) -> Operand {
    match rm {
        RmRaw::Reg(num) => Operand::Xmm { id: num },
        mem => finalize_rm(mem, Size::Qword, ip_end, false, seg),
    }
}

/// Map an ALU opcode group (`opcode >> 3`) to an [`Op`].
fn alu_group(group: u8) -> Option<Op> {
    match group {
        0 => Some(Op::Add),
        1 => Some(Op::Or),
        2 => Some(Op::Adc),
        3 => Some(Op::Sbb),
        4 => Some(Op::And),
        5 => Some(Op::Sub),
        6 => Some(Op::Xor),
        7 => Some(Op::Cmp),
        _ => None,
    }
}

fn group1_op(reg: usize) -> Option<Op> {
    match reg {
        0 => Some(Op::Add),
        1 => Some(Op::Or),
        2 => Some(Op::Adc),
        3 => Some(Op::Sbb),
        4 => Some(Op::And),
        5 => Some(Op::Sub),
        6 => Some(Op::Xor),
        7 => Some(Op::Cmp),
        _ => None,
    }
}

/// Group 2 (shift/rotate) reg field.
fn group2_op(reg: usize) -> Option<Op> {
    match reg {
        0 => Some(Op::Rol),
        1 => Some(Op::Ror),
        2 => Some(Op::Rcl),
        3 => Some(Op::Rcr),
        4 | 6 => Some(Op::Shl), // /6 is SAL == SHL
        5 => Some(Op::Shr),
        7 => Some(Op::Sar),
        _ => None,
    }
}

/// Parsed VEX prefix fields (C4 3-byte / C5 2-byte form). R/X/B are stored
/// inverted in the encoding; here `rex` is the equivalent non-inverted REX byte.
struct VexFields {
    vvvv: u8,  // third (non-destructive) source register, 0-15
    l256: bool, // L bit: true = 256-bit (YMM), false = 128-bit (XMM)
    pp: u8,    // implied legacy prefix: 0=none 1=66 2=F3 3=F2
    map: u8,   // opcode map: 1=0F, 2=0F38, 3=0F3A
    rex: u8,   // synthesized REX (0x40 | W<<3 | R<<2 | X<<1 | B)
}

/// Read a VEX prefix. `two_byte` = C5 (1 payload byte); else C4 (2 payload bytes).
/// In 64-bit mode C4/C5 are unconditionally VEX (the legacy LES/LDS are invalid).
fn read_vex(c: &mut Cursor, two_byte: bool) -> Result<VexFields, X86Error> {
    if two_byte {
        let b = c.u8()?; // R vvvv L pp pp  (R inverted)
        let r = ((!b >> 7) & 1) as u8;
        Ok(VexFields {
            vvvv: (!(b >> 3)) & 0xf,
            l256: (b >> 2) & 1 != 0,
            pp: b & 3,
            map: 1, // C5 implies the 0F map
            rex: 0x40 | (r << 2),
        })
    } else {
        let b1 = c.u8()?; // R X B mmmmm  (R/X/B inverted)
        let b2 = c.u8()?; // W vvvv L pp
        let r = ((!b1 >> 7) & 1) as u8;
        let x = ((!b1 >> 6) & 1) as u8;
        let bb = ((!b1 >> 5) & 1) as u8;
        let w = (b2 >> 7) & 1;
        Ok(VexFields {
            vvvv: (!(b2 >> 3)) & 0xf,
            l256: (b2 >> 2) & 1 != 0,
            pp: b2 & 3,
            map: b1 & 0x1f,
            rex: 0x40 | (w << 3) | (r << 2) | (x << 1) | bb,
        })
    }
}

/// Build a VEX two-source packed op: dst=reg(modrm), src=r/m, vvvv=src1.
fn vex_packed(c: &mut Cursor, p: &Prefixes, kind: VPkind, vvvv: u8, l256: bool) -> Result<DecodedInsn, X86Error> {
    let (reg, rm) = decode_modrm(c, p)?;
    let src = finalize_rm_xmm(rm, c.ip, p.seg);
    Ok(sse(Op::VPacked { kind, vvvv, l256 }, Operand::Xmm { id: reg }, src))
}

/// Decode the body of a VEX instruction (the opcode byte + operands) into a
/// `DecodedInsn` with `len: 0` (the caller's `finish` sets the true length and
/// rebases any RIP-relative operand). Iteration-1 batch: VZEROUPPER/ALL, the VEX
/// move family, VMOVD/Q (GPR↔XMM), and the VEX bitwise ops. Anything else is
/// `Unsupported` (later iterations add map-2/3, arithmetic, FMA, shuffles…).
fn decode_vex_body(c: &mut Cursor, v: &VexFields) -> Result<DecodedInsn, X86Error> {
    let mut p = Prefixes::default();
    p.has_rex = true;
    p.rex = v.rex;
    match v.pp {
        1 => p.opsize = true,
        2 => p.rep = Rep::F3,
        3 => p.rep = Rep::F2,
        _ => {}
    }
    let opcode = c.u8()?;
    let l256 = v.l256;

    // BMI1/BMI2 — VEX-encoded but GPR-operand. W selects 32/64-bit; the operands are
    // GPRs (not XMM), so decode them with finalize_rm / reg_operand.
    let gpr_size = if p.rex_w() { Size::Qword } else { Size::Dword };
    if v.map == 2 {
        match (opcode, v.pp) {
            (0xf2, 0) => {
                // ANDN dst, vvvv, r/m
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::Andn { vvvv: v.vvvv }, reg_operand(reg, gpr_size, true), src));
            }
            (0xf3, 0) => {
                // group: /1 BLSR /2 BLSMSK /3 BLSI. dst = vvvv (NDD), src = r/m.
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                return Ok(sse(
                    Op::Blsx { kind: (reg & 7) as u8 },
                    reg_operand(v.vvvv as usize, gpr_size, true),
                    src,
                ));
            }
            (0xf5, 0) => {
                // BZHI dst, r/m, vvvv(index)
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::Bzhi { vvvv: v.vvvv }, reg_operand(reg, gpr_size, true), src));
            }
            (0xf5, 3) => {
                // PDEP dst, vvvv(value), r/m(mask)  (F2 prefix)
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::Pdep { vvvv: v.vvvv }, reg_operand(reg, gpr_size, true), src));
            }
            (0xf5, 2) => {
                // PEXT dst, vvvv(value), r/m(mask)  (F3 prefix)
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::Pext { vvvv: v.vvvv }, reg_operand(reg, gpr_size, true), src));
            }
            (0xf6, 3) => {
                // MULX vvvv:dst = rdx * r/m  (F2 prefix)
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::Mulx { vvvv: v.vvvv }, reg_operand(reg, gpr_size, true), src));
            }
            (0xf7, pp) => {
                // pp0 BEXTR; ppF3 SARX; ppF2 SHRX; pp66 SHLX. dst, r/m(value), vvvv(ctrl/count).
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
                let dst = reg_operand(reg, gpr_size, true);
                let op = match pp {
                    0 => Op::Bextr { vvvv: v.vvvv },
                    2 => Op::Sarx { kind: ShiftKind::Sra, vvvv: v.vvvv },
                    3 => Op::Sarx { kind: ShiftKind::Srl, vvvv: v.vvvv },
                    _ => Op::Sarx { kind: ShiftKind::Sll, vvvv: v.vvvv }, // pp66 SHLX
                };
                return Ok(sse(op, dst, src));
            }
            _ => {}
        }
    }
    if v.map == 3 && opcode == 0xf0 && v.pp == 3 {
        // RORX dst, r/m, imm8  (F2 prefix)
        let (reg, rm) = decode_modrm(c, &p)?;
        let src = finalize_rm(rm, gpr_size, c.ip, p.has_rex, p.seg);
        let imm = c.u8()?;
        return Ok(sse(Op::Rorx { imm }, reg_operand(reg, gpr_size, true), src));
    }

    if v.map == 1 {
        match opcode {
            // VZEROUPPER (L=0) / VZEROALL (L=1) — no ModRM, requires NP + vvvv=1111.
            0x77 => {
                return Ok(DecodedInsn {
                    len: 0,
                    op: Op::VZeroUpper { all: l256 },
                    dst: None,
                    src: None,
                    rel: None,
                });
            }
            // VMOVSS (F3) / VMOVSD (F2): SCALAR moves (4/8 bytes), 0F 10 = load dir,
            // 0F 11 = store dir. CRITICAL: these move only 4/8 bytes — treating them as
            // a packed 128-bit VMov overwrites adjacent memory on store (e.g. a stack
            // canary → spurious "stack smashing detected" in glibc's AVX math).
            0x10 | 0x11 if matches!(v.pp, 2 | 3) => {
                let bytes: u8 = if v.pp == 3 { 8 } else { 4 }; // pp 3=F2(SD,8B), 2=F3(SS,4B)
                let load = opcode == 0x10;
                let reg_form = c.peek()? >> 6 == 3;
                let (reg, rm) = decode_modrm(c, &p)?;
                let rm_op = finalize_rm_xmm(rm, c.ip, p.seg);
                if reg_form {
                    // 3-operand merge: low from the non-vvvv source, upper from vvvv.
                    let (dst, src) = if load {
                        (Operand::Xmm { id: reg }, rm_op)
                    } else {
                        (rm_op, Operand::Xmm { id: reg })
                    };
                    return Ok(sse(Op::VMovScalarMerge { bytes, vvvv: v.vvvv }, dst, src));
                }
                // memory form: load = zero-extend into xmm; store = write `bytes`.
                return Ok(if load {
                    sse(Op::VMovGpr { to_xmm: true, bytes }, Operand::Xmm { id: reg }, rm_op)
                } else {
                    sse(Op::VMovGpr { to_xmm: false, bytes }, rm_op, Operand::Xmm { id: reg })
                });
            }
            // VEX move loads: VMOVUPS/UPD (10), VMOVAPS/APD (28), VMOVDQA/DQU (6F).
            0x10 | 0x28 | 0x6f => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMov { l256 }, Operand::Xmm { id: reg }, src));
            }
            // VEX move stores: dst = r/m, src = reg.
            0x11 | 0x29 | 0x7f => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMov { l256 }, dst, Operand::Xmm { id: reg }));
            }
            // VMOVD/VMOVQ: GPR/mem → XMM (66 0F 6E), W selects 4/8 bytes.
            0x6e if p.opsize => {
                let bytes = if p.rex_w() { 8 } else { 4 };
                let (reg, rm) = decode_modrm(c, &p)?;
                let sz = if bytes == 8 { Size::Qword } else { Size::Dword };
                let src = finalize_rm(rm, sz, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::VMovGpr { to_xmm: true, bytes }, Operand::Xmm { id: reg }, src));
            }
            // VMOVD/VMOVQ: XMM → GPR/mem (66 0F 7E).
            0x7e if p.opsize => {
                let bytes = if p.rex_w() { 8 } else { 4 };
                let (reg, rm) = decode_modrm(c, &p)?;
                let sz = if bytes == 8 { Size::Qword } else { Size::Dword };
                let dst = finalize_rm(rm, sz, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::VMovGpr { to_xmm: false, bytes }, dst, Operand::Xmm { id: reg }));
            }
            // VMOVQ xmm, xmm/m64 (VEX.128.F3.0F.WIG 7E /r): load low 64 bits, zero the
            // rest of the dest (incl. YMM upper). Same effect as VMOVD/Q-to-xmm but with
            // an xmm/m64 source — glibc/busybox AVX strcmp/memcmp use this (`vmovq (%rdi),%xmm`).
            0x7e if v.pp == 2 => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMovGpr { to_xmm: true, bytes: 8 }, Operand::Xmm { id: reg }, src));
            }
            // VMOVQ xmm/m64, xmm (VEX.128.66.0F.WIG D6 /r): store the low 64 bits.
            0xd6 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMovGpr { to_xmm: false, bytes: 8 }, dst, Operand::Xmm { id: reg }));
            }
            // VEX FP arithmetic: ADD(58) MUL(59) SUB(5C) MIN(5D) DIV(5E) MAX(5F).
            // pp selects type: 0=PS 1=PD 2=SS 3=SD (double=pp∈{1,3}, packed=pp∈{0,1}).
            0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => {
                let kind = match opcode {
                    0x58 => FBinKind::Add,
                    0x59 => FBinKind::Mul,
                    0x5c => FBinKind::Sub,
                    0x5d => FBinKind::Min,
                    0x5e => FBinKind::Div,
                    _ => FBinKind::Max, // 0x5f
                };
                let double = matches!(v.pp, 1 | 3);
                let packed = matches!(v.pp, 0 | 1);
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VFBin { kind, double, packed, vvvv: v.vvvv, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            // VEX packed integer add/sub (pp=66): VPADD B/W/D(FC/FD/FE) Q(D4);
            // VPSUB B/W/D/Q(F8/F9/FA/FB).
            0xfc | 0xfd | 0xfe | 0xd4 | 0xf8 | 0xf9 | 0xfa | 0xfb if p.opsize => {
                let elem = match opcode {
                    0xfc | 0xf8 => 1,
                    0xfd | 0xf9 => 2,
                    0xfe | 0xfa => 4,
                    _ => 8, // d4 / fb
                };
                let sub = matches!(opcode, 0xf8 | 0xf9 | 0xfa | 0xfb);
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VPackedArith { elem, sub, vvvv: v.vvvv, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            // VEX saturating packed add/sub (pp66): VPADDSB/W(EC/ED) VPADDUSB/W(DC/DD)
            // VPSUBSB/W(E8/E9) VPSUBUSB/W(D8/D9). signed=adds/subs, unsigned=addus/subus.
            0xec | 0xed | 0xdc | 0xdd | 0xe8 | 0xe9 | 0xd8 | 0xd9 if p.opsize => {
                let elem = if matches!(opcode, 0xec | 0xdc | 0xe8 | 0xd8) { 1 } else { 2 };
                let sub = matches!(opcode, 0xe8 | 0xe9 | 0xd8 | 0xd9);
                let signed = matches!(opcode, 0xec | 0xed | 0xe8 | 0xe9);
                return vex_packed(c, &p, VPkind::SatArith { sub, signed, elem }, v.vvvv, l256);
            }
            // VPAVGB/W (E0/E3, pp66): unsigned rounded average.
            0xe0 | 0xe3 if p.opsize => {
                let elem = if opcode == 0xe0 { 1 } else { 2 };
                return vex_packed(c, &p, VPkind::Avg { elem }, v.vvvv, l256);
            }
            // VPSADBW (F6, pp66): sum of absolute byte differences per 64-bit lane.
            0xf6 if p.opsize => {
                return vex_packed(c, &p, VPkind::Sadbw, v.vvvv, l256);
            }
            // VMOVNTDQ (E7, pp66): non-temporal store (no cache modeled → VMOVDQA store).
            0xe7 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMov { l256 }, dst, Operand::Xmm { id: reg }));
            }
            // VMOVDDUP (F2 0F 12): duplicate the low double into both halves.
            0x12 if v.pp == 3 => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMovddup { l256 }, Operand::Xmm { id: reg }, src));
            }
            // VSQRTPS(pp0)/VSQRTPD(pp1)/VSQRTSS(pp2)/VSQRTSD(pp3) (0F 51).
            0x51 => {
                let (double, packed) = match v.pp {
                    3 => (true, false),  // F2: scalar double
                    2 => (false, false), // F3: scalar single
                    1 => (true, true),   // 66: packed double
                    _ => (false, true),  // none: packed single
                };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VFSqrt { double, packed, vvvv: v.vvvv, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            // VEX bitwise (3-operand): VANDPS/PD(54) VANDNPS/PD(55) VORPS/PD(56)
            // VXORPS/PD(57) VPAND(DB) VPANDN(DF) VPOR(EB) VPXOR(EF).
            0x54 | 0x55 | 0x56 | 0x57 | 0xdb | 0xdf | 0xeb | 0xef => {
                let kind = match opcode {
                    0x54 | 0xdb => VBitKind::And,
                    0x55 | 0xdf => VBitKind::AndN,
                    0x56 | 0xeb => VBitKind::Or,
                    _ => VBitKind::Xor, // 0x57 | 0xef
                };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VBitwise { kind, vvvv: v.vvvv, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            // packed integer compare (pp=66): VPCMPEQ B/W/D (74/75/76), VPCMPGT (64/65/66).
            0x74 | 0x75 | 0x76 if p.opsize => {
                let e = match opcode { 0x74 => 1, 0x75 => 2, _ => 4 };
                return vex_packed(c, &p, VPkind::CmpEq(e), v.vvvv, l256);
            }
            0x64 | 0x65 | 0x66 if p.opsize => {
                let e = match opcode { 0x64 => 1, 0x65 => 2, _ => 4 };
                return vex_packed(c, &p, VPkind::CmpGt(e), v.vvvv, l256);
            }
            // packed integer min/max (pp=66, 0F map): PMINUB(DA) PMAXUB(DE) PMINSW(EA) PMAXSW(EE).
            0xda | 0xde | 0xea | 0xee if p.opsize => {
                let kind = match opcode {
                    0xda => VPkind::MinMax { max: false, signed: false, elem: 1 },
                    0xde => VPkind::MinMax { max: true, signed: false, elem: 1 },
                    0xea => VPkind::MinMax { max: false, signed: true, elem: 2 },
                    _ => VPkind::MinMax { max: true, signed: true, elem: 2 },
                };
                return vex_packed(c, &p, kind, v.vvvv, l256);
            }
            // packed integer unpack (pp=66): low BW/WD/DQ/QDQ (60/61/62/6C), high (68/69/6A/6D).
            0x60 | 0x61 | 0x62 | 0x6c | 0x68 | 0x69 | 0x6a | 0x6d if p.opsize => {
                let (high, elem) = match opcode {
                    0x60 => (false, 1), 0x61 => (false, 2), 0x62 => (false, 4), 0x6c => (false, 8),
                    0x68 => (true, 1), 0x69 => (true, 2), 0x6a => (true, 4), _ => (true, 8),
                };
                return vex_packed(c, &p, VPkind::Unpack { high, elem }, v.vvvv, l256);
            }
            // FP unpack: VUNPCKL/HPS (pp0, elem4), VUNPCKL/HPD (pp66, elem8). 14=low 15=high.
            0x14 | 0x15 => {
                let elem = if p.opsize { 8 } else { 4 };
                return vex_packed(c, &p, VPkind::Unpack { high: opcode == 0x15, elem }, v.vvvv, l256);
            }
            // packed integer pack (pp=66): PACKSSWB(63) PACKSSDW(6B) PACKUSWB(67).
            0x63 | 0x6b | 0x67 if p.opsize => {
                let kind = match opcode {
                    0x63 => VPkind::Pack { dword_src: false, signed: true },
                    0x6b => VPkind::Pack { dword_src: true, signed: true },
                    _ => VPkind::Pack { dword_src: false, signed: false },
                };
                return vex_packed(c, &p, kind, v.vvvv, l256);
            }
            // packed shift by xmm/m (pp=66): PSLLW/D/Q(F1/F2/F3) PSRLW/D/Q(D1/D2/D3) PSRAW/D(E1/E2).
            0xf1 | 0xf2 | 0xf3 | 0xd1 | 0xd2 | 0xd3 | 0xe1 | 0xe2 if p.opsize => {
                let (kind, elem) = match opcode {
                    0xf1 => (ShiftKind::Sll, 2), 0xf2 => (ShiftKind::Sll, 4), 0xf3 => (ShiftKind::Sll, 8),
                    0xd1 => (ShiftKind::Srl, 2), 0xd2 => (ShiftKind::Srl, 4), 0xd3 => (ShiftKind::Srl, 8),
                    0xe1 => (ShiftKind::Sra, 2), _ => (ShiftKind::Sra, 4),
                };
                return vex_packed(c, &p, VPkind::Shift { kind, elem }, v.vvvv, l256);
            }
            // VPMOVMSKB (66 0F D7): byte sign bits → GPR.
            0xd7 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VPmovmskb { l256 }, reg_operand(reg, Size::Dword, true), src));
            }
            // VMOVMSKPS/PD (0F 50): sign bits → GPR. reg form only.
            0x50 => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VMovmsk { double: p.opsize, l256 }, reg_operand(reg, Size::Dword, true), src));
            }
            // VSHUFPS/PD (C6, imm8).
            0xc6 => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VShufps { double: p.opsize, vvvv: v.vvvv, l256, imm }, Operand::Xmm { id: reg }, src));
            }
            // VPSHUFD (pp66) / VPSHUFHW (ppF3) / VPSHUFLW (ppF2) (70, imm8).
            0x70 => {
                let high = if p.opsize { None } else if matches!(p.rep, Rep::F3) { Some(true) } else { Some(false) };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VPshuf { high, l256, imm }, Operand::Xmm { id: reg }, src));
            }
            // VCMPPS/PD/SS/SD (C2, imm8 predicate).
            0xc2 => {
                let (double, scalar) = match p.rep {
                    Rep::F2 => (true, true),
                    Rep::F3 => (false, true),
                    _ => (p.opsize, false),
                };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VCmpFp { double, scalar, vvvv: v.vvvv, l256, imm }, Operand::Xmm { id: reg }, src));
            }
            // VCVT* (128-bit): 5A (f↔f), 5B (dq↔ps), E6 (dq↔pd).
            0x5a | 0x5b | 0xe6 => {
                let kind = match (opcode, v.pp) {
                    (0x5a, 0) => VCvtKind::F2F { to_double: true, scalar: false },  // PS2PD
                    (0x5a, 1) => VCvtKind::F2F { to_double: false, scalar: false }, // PD2PS
                    (0x5a, 2) => VCvtKind::F2F { to_double: true, scalar: true },   // SS2SD
                    (0x5a, 3) => VCvtKind::F2F { to_double: false, scalar: true },  // SD2SS
                    (0x5b, 0) => VCvtKind::I2F { double_dst: false },               // DQ2PS
                    (0x5b, 1) => VCvtKind::F2I { double_src: false, trunc: false }, // PS2DQ
                    (0x5b, 2) => VCvtKind::F2I { double_src: false, trunc: true },  // TPS2DQ
                    (0xe6, 2) => VCvtKind::I2F { double_dst: true },                // DQ2PD (F3)
                    (0xe6, 3) => VCvtKind::F2I { double_src: true, trunc: false },  // PD2DQ (F2)
                    (0xe6, 1) => VCvtKind::F2I { double_src: true, trunc: true },   // TPD2DQ (66)
                    _ => return Err(X86Error::Unsupported(format!("VEX cvt {opcode:02x} pp{}", v.pp))),
                };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VCvt { kind, vvvv: v.vvvv, l256 }, Operand::Xmm { id: reg }, src));
            }
            // VCVTSI2SD/SS xmm, xmm, r/m (VEX F2/F3 0F 2A): int(GPR/mem) → scalar FP.
            0x2a if matches!(v.pp, 2 | 3) => {
                let double = v.pp == 3; // F2 = SD, F3 = SS
                let isize = if p.rex_w() { Size::Qword } else { Size::Dword };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, isize, c.ip, p.has_rex, p.seg);
                return Ok(sse(Op::VCvtsi2f { double, vvvv: v.vvvv }, Operand::Xmm { id: reg }, src));
            }
            // VCVTTSD2SI/SS2SI (VEX F2/F3 0F 2C): scalar FP → signed int (GPR), truncating.
            0x2c if matches!(v.pp, 2 | 3) => {
                let double = v.pp == 3; // F2 = SD, F3 = SS
                let isize = if p.rex_w() { Size::Qword } else { Size::Dword };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::Cvttf2si(double), reg_operand(reg, isize, p.has_rex), src));
            }
            // VUCOMISS/SD (VEX 0F 2E) / VCOMISS/SD (VEX 0F 2F): scalar ordered compare
            // → EFLAGS. 66 = double (SD), none = single (SS). Same as the SSE form.
            0x2e | 0x2f => {
                let double = p.opsize;
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::Ucomis(double), Operand::Xmm { id: reg }, src));
            }
            // VLDMXCSR (VEX.LZ.0F.WIG AE /2) / VSTMXCSR (/3), m32. MXCSR is a single
            // register; the VEX form behaves identically to the SSE LDMXCSR/STMXCSR.
            0xae => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let mem = finalize_rm(rm, Size::Dword, c.ip, p.has_rex, p.seg);
                let (op, dst, src) = match reg {
                    2 => (Op::Ldmxcsr, None, Some(mem)),
                    3 => (Op::Stmxcsr, Some(mem), None),
                    _ => return Err(X86Error::Unsupported(format!("VEX 0f ae /{reg}"))),
                };
                return Ok(DecodedInsn { len: 0, op, dst, src, rel: None });
            }
            // VPMUL word/dword (pp66): MULLW(D5) MULHW(E5) MULHUW(E4) MULUDQ(F4) MADDWD(F5).
            0xd5 | 0xe5 | 0xe4 | 0xf4 | 0xf5 if p.opsize => {
                let k = match opcode {
                    0xd5 => PMulKind::Lw,
                    0xe5 => PMulKind::Hw,
                    0xe4 => PMulKind::Huw,
                    0xf4 => PMulKind::Uludq,
                    _ => PMulKind::Maddwd,
                };
                return vex_packed(c, &p, VPkind::Mul(k), v.vvvv, l256);
            }
            // VEX NDD packed shift-by-imm (group 71/72/73): dst=vvvv, rm=source, reg=sub-op.
            0x71 | 0x72 | 0x73 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                let elem = match opcode { 0x71 => 2, 0x72 => 4, _ => 8 };
                let op = match (opcode, reg & 7) {
                    (_, 2) => Op::VPShiftImm { kind: ShiftKind::Srl, elem, l256, imm },
                    (0x71, 4) | (0x72, 4) => Op::VPShiftImm { kind: ShiftKind::Sra, elem, l256, imm },
                    (_, 6) => Op::VPShiftImm { kind: ShiftKind::Sll, elem, l256, imm },
                    (0x73, 3) => Op::VPByteShiftImm { left: false, l256, imm },
                    (0x73, 7) => Op::VPByteShiftImm { left: true, l256, imm },
                    (o, r) => return Err(X86Error::Unsupported(format!("VEX grp 0x{o:02x} /{r}"))),
                };
                return Ok(sse(op, Operand::Xmm { id: v.vvvv as usize }, src));
            }
            _ => {}
        }
    }
    if v.map == 2 {
        // 0F38 map.
        match opcode {
            0x00 if p.opsize => {
                // VPSHUFB
                return vex_packed(c, &p, VPkind::Pshufb, v.vvvv, l256);
            }
            // VPERMILPS(0C)/VPERMILPD(0D) variable: data=vvvv, control=r/m.
            0x0c | 0x0d if p.opsize => {
                let double = opcode == 0x0d;
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VPermil { double, variable: true, vvvv: v.vvvv, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            // VPMOVSX* (20-25) / VPMOVZX* (30-35): widen src_elem→dst_elem.
            0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 | 0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35
                if p.opsize =>
            {
                let sign = opcode < 0x30;
                let (src_elem, dst_elem) = match opcode & 0x0f {
                    0x0 => (1, 2), // BW
                    0x1 => (1, 4), // BD
                    0x2 => (1, 8), // BQ
                    0x3 => (2, 4), // WD
                    0x4 => (2, 8), // WQ
                    _ => (4, 8),   // DQ
                };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VPmovx { src_elem, dst_elem, sign, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            // VPTEST (17, pp66): set ZF=(dst&src==0), CF=(~dst&src==0). l256-aware.
            0x17 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VPtest { l256 }, Operand::Xmm { id: reg }, src));
            }
            0x29 | 0x37 if p.opsize => {
                // VPCMPEQQ(29) / VPCMPGTQ(37), elem 8.
                let kind = if opcode == 0x29 { VPkind::CmpEq(8) } else { VPkind::CmpGt(8) };
                return vex_packed(c, &p, kind, v.vvvv, l256);
            }
            0x2b if p.opsize => {
                // VPACKUSDW (unsigned dword→word saturate).
                return vex_packed(c, &p, VPkind::Pack { dword_src: true, signed: false }, v.vvvv, l256);
            }
            0x38 | 0x39 | 0x3a | 0x3b | 0x3c | 0x3d | 0x3e | 0x3f if p.opsize => {
                // VPMIN/MAX S/U B/W/D.
                let kind = match opcode {
                    0x38 => VPkind::MinMax { max: false, signed: true, elem: 1 },
                    0x39 => VPkind::MinMax { max: false, signed: true, elem: 4 },
                    0x3a => VPkind::MinMax { max: false, signed: false, elem: 2 },
                    0x3b => VPkind::MinMax { max: false, signed: false, elem: 4 },
                    0x3c => VPkind::MinMax { max: true, signed: true, elem: 1 },
                    0x3d => VPkind::MinMax { max: true, signed: true, elem: 4 },
                    0x3e => VPkind::MinMax { max: true, signed: false, elem: 2 },
                    _ => VPkind::MinMax { max: true, signed: false, elem: 4 },
                };
                return vex_packed(c, &p, kind, v.vvvv, l256);
            }
            // VPMULLD(40) VPMULDQ(28) VPMADDUBSW(04) VPMULHRSW(0B).
            0x40 if p.opsize => return vex_packed(c, &p, VPkind::Mulld, v.vvvv, l256),
            0x28 if p.opsize => return vex_packed(c, &p, VPkind::Muldq, v.vvvv, l256),
            0x04 if p.opsize => return vex_packed(c, &p, VPkind::Maddubsw, v.vvvv, l256),
            0x0b if p.opsize => return vex_packed(c, &p, VPkind::Mulhrsw, v.vvvv, l256),
            // broadcasts: VPBROADCASTB(78) W(79) D(58) Q(59); VBROADCASTSS(18) SD(19)
            // F128(1A) I128(5A). dst=reg, src=r/m.
            0x78 | 0x79 | 0x58 | 0x59 | 0x18 | 0x19 | 0x1a | 0x5a if p.opsize => {
                let elem = match opcode {
                    0x78 => 1,
                    0x79 => 2,
                    0x58 | 0x18 => 4,
                    0x59 | 0x19 => 8,
                    _ => 16, // 1a / 5a (broadcast 128)
                };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VBroadcast { elem, l256 }, Operand::Xmm { id: reg }, src));
            }
            // VPERMD(36) / VPERMPS(16): indices in vvvv, data in r/m, dst=reg.
            0x36 | 0x16 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VPermVar { vvvv: v.vvvv }, Operand::Xmm { id: reg }, src));
            }
            // VCVTPH2PS (F16C): half-floats → singles. dst=reg, src=r/m.
            0x13 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(Op::VCvtph2ps { l256 }, Operand::Xmm { id: reg }, src));
            }
            // FMA (0F38, pp66): opcode 0x96-0x9F (132) / 0xA6-0xAF (213) / 0xB6-0xBF
            // (231). Low nibble selects the sign pattern + packed/scalar; W = PD/SD.
            0x96..=0x9f | 0xa6..=0xaf | 0xb6..=0xbf if p.opsize => {
                let form = match opcode >> 4 {
                    0x9 => FmaForm::F132,
                    0xa => FmaForm::F213,
                    _ => FmaForm::F231,
                };
                let (kind, packed) = match opcode & 0xf {
                    0x6 => (FmaKind::Maddsub, true),
                    0x7 => (FmaKind::Msubadd, true),
                    0x8 => (FmaKind::Madd, true),
                    0x9 => (FmaKind::Madd, false),
                    0xa => (FmaKind::Msub, true),
                    0xb => (FmaKind::Msub, false),
                    0xc => (FmaKind::Nmadd, true),
                    0xd => (FmaKind::Nmadd, false),
                    0xe => (FmaKind::Nmsub, true),
                    _ => (FmaKind::Nmsub, false), // 0xf
                };
                let double = p.rex_w();
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                return Ok(sse(
                    Op::VFma { kind, form, double, packed, vvvv: v.vvvv, l256 },
                    Operand::Xmm { id: reg },
                    src,
                ));
            }
            _ => {}
        }
    }
    if v.map == 3 {
        // 0F3A map (all take an imm8 trailing byte).
        match opcode {
            0x0c | 0x0d | 0x0e if p.opsize => {
                // VBLENDPS(0c, elem4) / VBLENDPD(0d, elem8) / VPBLENDW(0e, elem2).
                let elem = match opcode { 0x0c => 4, 0x0d => 8, _ => 2 };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VBlend { elem, vvvv: v.vvvv, l256, imm }, Operand::Xmm { id: reg }, src));
            }
            // VPERMILPS(04)/VPERMILPD(05) imm8: data=r/m, control=imm8 (no vvvv).
            0x04 | 0x05 if p.opsize => {
                let double = opcode == 0x05;
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(DecodedInsn {
                    len: 0,
                    op: Op::VPermil { double, variable: false, vvvv: v.vvvv, l256 },
                    dst: Some(Operand::Xmm { id: reg }),
                    src: Some(src),
                    rel: Some(imm as i64),
                });
            }
            // VPBLENDD (0F3A 02 ib): blend dwords by imm8 (AVX2) — same blend as elem=4.
            0x02 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VBlend { elem: 4, vvvv: v.vvvv, l256, imm }, Operand::Xmm { id: reg }, src));
            }
            // VPEXTRB/W/D/Q (14/15/16) + VEXTRACTPS (17): xmm → GPR/mem by imm8 (no vvvv).
            0x14 | 0x15 | 0x16 | 0x17 if p.opsize => {
                let elem: u8 = match opcode { 0x14 => 1, 0x15 => 2, 0x16 if p.rex_w() => 8, _ => 4 };
                let gsize = match elem { 1 => Size::Byte, 2 => Size::Word, 8 => Size::Qword, _ => Size::Dword };
                let (reg, rm) = decode_modrm(c, &p)?;
                let dst = finalize_rm(rm, gsize, c.ip, p.has_rex, p.seg);
                let imm = c.u8()?;
                return Ok(DecodedInsn { len: 0, op: Op::Pextr { elem }, dst: Some(dst), src: Some(Operand::Xmm { id: reg }), rel: Some(imm as i64) });
            }
            // VPINSRB (20) / VPINSRD/Q (22): dst = vvvv with element[imm] = int src (GPR/mem).
            0x20 | 0x22 if p.opsize => {
                let elem: u8 = if opcode == 0x20 { 1 } else if p.rex_w() { 8 } else { 4 };
                let gsize = match elem { 1 => Size::Byte, 8 => Size::Qword, _ => Size::Dword };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm(rm, gsize, c.ip, p.has_rex, p.seg);
                let imm = c.u8()?;
                return Ok(DecodedInsn { len: 0, op: Op::VPinsr { elem, vvvv: v.vvvv }, dst: Some(Operand::Xmm { id: reg }), src: Some(src), rel: Some(imm as i64) });
            }
            0x4a | 0x4b | 0x4c if p.opsize => {
                // VBLENDVPS(4a, elem4) / VBLENDVPD(4b, elem8) / VPBLENDVB(4c, elem1).
                // imm8[7:4] selects the control register.
                let elem = match opcode { 0x4a => 4, 0x4b => 8, _ => 1 };
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let ctrl = c.u8()? >> 4;
                return Ok(sse(Op::VBlendv { elem, vvvv: v.vvvv, l256, ctrl }, Operand::Xmm { id: reg }, src));
            }
            0x0f if p.opsize => {
                // VPALIGNR (imm8).
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VPalignr { vvvv: v.vvvv, l256, imm }, Operand::Xmm { id: reg }, src));
            }
            // VINSERTF128(18)/VINSERTI128(38): dst=reg, src1=vvvv, src2=r/m(xmm/m128), imm[0]=lane.
            0x18 | 0x38 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VInsert128 { vvvv: v.vvvv, lane: imm & 1 }, Operand::Xmm { id: reg }, src));
            }
            // VEXTRACTF128(19)/VEXTRACTI128(39): dst=r/m(xmm/m128), src=reg(ymm), imm[0]=lane.
            0x19 | 0x39 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VExtract128 { lane: imm & 1 }, dst, Operand::Xmm { id: reg }));
            }
            // VPERM2F128(06)/VPERM2I128(46): dst=reg, src1=vvvv, src2=r/m, imm.
            0x06 | 0x46 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VPerm2 { vvvv: v.vvvv, imm }, Operand::Xmm { id: reg }, src));
            }
            // VPERMQ(00)/VPERMPD(01) (VEX.W1): dst=reg, src=r/m, imm permutes 4 qwords.
            0x00 | 0x01 if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let src = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VPermImm { imm }, Operand::Xmm { id: reg }, src));
            }
            // VCVTPS2PH (F16C): singles → half-floats (imm8 rounding). dst=r/m, src=reg.
            0x1d if p.opsize => {
                let (reg, rm) = decode_modrm(c, &p)?;
                let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                let imm = c.u8()?;
                return Ok(sse(Op::VCvtps2ph { l256, imm }, dst, Operand::Xmm { id: reg }));
            }
            _ => {}
        }
    }
    Err(X86Error::Unsupported(format!(
        "VEX map{} opcode {:02x} pp{} L{}",
        v.map, opcode, v.pp, l256 as u8
    )))
}

/// Decode one instruction starting at `rip`, fetching bytes via `fetch`.
pub fn decode(
    fetch: &mut dyn FnMut(u64) -> Result<u8, X86Error>,
    rip: u64,
    mode: DecodeMode,
) -> Result<DecodedInsn, X86Error> {
    let mut c = Cursor { fetch, ip: rip };
    let mut p = Prefixes { mode, ..Prefixes::default() };

    // Prefixes.
    loop {
        match c.peek()? {
            0x66 => {
                p.opsize = true;
                c.ip += 1;
            }
            0x67 => {
                p.addrsize = true; // address-size override
                c.ip += 1;
            }
            0xf0 => {
                c.ip += 1; // LOCK: trivial on a single vCPU
            }
            0xf2 => {
                p.rep = Rep::F2;
                c.ip += 1;
            }
            0xf3 => {
                p.rep = Rep::F3;
                c.ip += 1;
            }
            0x64 => {
                p.seg = Seg::Fs;
                c.ip += 1;
            }
            0x65 => {
                p.seg = Seg::Gs;
                c.ip += 1;
            }
            0x2e => {
                p.seg = Seg::Cs;
                c.ip += 1;
            }
            0x36 => {
                p.seg = Seg::Ss;
                c.ip += 1;
            }
            0x3e => {
                p.seg = Seg::Ds;
                c.ip += 1;
            }
            0x26 => {
                p.seg = Seg::Es;
                c.ip += 1;
            }
            0x40..=0x4f => {
                p.rex = c.peek()?;
                p.has_rex = true;
                c.ip += 1;
            }
            _ => break,
        }
    }

    let opcode = c.u8()?;
    let finish = |c: &Cursor, insn: DecodedInsn| -> DecodedInsn {
        // RIP-relative displacements are relative to the END of the instruction
        // (which, for forms with a trailing immediate, is past that immediate).
        // `c.ip` is now that true end, so rebase any rip-rel operand here.
        let end_ip = c.ip;
        let rebase = |op: Option<Operand>| -> Option<Operand> {
            match op {
                Some(Operand::Mem {
                    base,
                    index,
                    scale,
                    disp,
                    size,
                    seg,
                    rip_rel: true,
                }) => Some(Operand::Mem {
                    base,
                    index,
                    scale,
                    disp: (end_ip as i64).wrapping_add(disp),
                    size,
                    seg,
                    rip_rel: false,
                }),
                other => other,
            }
        };
        DecodedInsn {
            len: c.ip - rip,
            dst: rebase(insn.dst),
            src: rebase(insn.src),
            ..insn
        }
    };

    // ALU reg/mem forms and AL/eAX-imm short forms.
    if let Some(op) = alu_group(opcode >> 3) {
        let form = opcode & 0x7;
        if form <= 3 {
            let byte = form & 1 == 0;
            let to_reg = form & 2 != 0; // direction: reg is destination
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            let reg_op = reg_operand(reg, size, p.has_rex);
            let (dst, src) = if to_reg {
                (reg_op, rm_op)
            } else {
                (rm_op, reg_op)
            };
            return Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(dst),
                    src: Some(src),
                    rel: None,
                },
            ));
        } else if form == 4 || form == 5 {
            let byte = form == 4;
            let size = op_size(&p, byte);
            let imm_bytes = if byte { 1 } else { size.bytes().min(4) };
            let value = c.imm(imm_bytes, true)?;
            return Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(reg_operand(0, size, p.has_rex)), // AL/eAX
                    src: Some(Operand::Imm { value, size }),
                    rel: None,
                },
            ));
        }
    }

    match opcode {
        // VEX prefix (AVX): C5 = 2-byte form, C4 = 3-byte. Unconditionally VEX in
        // 64-bit mode (legacy LES/LDS are invalid here).
        0xc4 | 0xc5 => {
            let v = read_vex(&mut c, opcode == 0xc5)?;
            let insn = decode_vex_body(&mut c, &v)?;
            Ok(finish(&c, insn))
        }
        // Group 1: ALU r/m, imm
        0x80 | 0x81 | 0x83 => {
            let byte = opcode == 0x80;
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let op = group1_op(reg)
                .ok_or_else(|| X86Error::Unsupported(format!("group1 /{reg}")))?;
            let imm_bytes = if opcode == 0x81 { size.bytes().min(4) } else { 1 };
            let value = c.imm(imm_bytes, true)?;
            let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(dst),
                    src: Some(Operand::Imm { value, size }),
                    rel: None,
                },
            ))
        }

        // Group 2: shift/rotate r/m by imm8 / 1 / CL
        0xc0 | 0xc1 | 0xd0 | 0xd1 | 0xd2 | 0xd3 => {
            let byte = opcode & 1 == 0;
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let op = group2_op(reg)
                .ok_or_else(|| X86Error::Unsupported(format!("group2 /{reg}")))?;
            // Count source: imm8 (0xc0/0xc1), 1 (0xd0/0xd1), or CL (0xd2/0xd3).
            let count = match opcode {
                0xc0 | 0xc1 => Operand::Imm {
                    value: c.imm(1, false)?,
                    size: Size::Byte,
                },
                0xd0 | 0xd1 => Operand::Imm {
                    value: 1,
                    size: Size::Byte,
                },
                _ => reg_operand(super::regs::RCX, Size::Byte, p.has_rex),
            };
            let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(dst),
                    src: Some(count),
                    rel: None,
                },
            ))
        }

        // Group 3: TEST/NOT/NEG/MUL/IMUL/DIV/IDIV r/m
        0xf6 | 0xf7 => {
            let byte = opcode == 0xf6;
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            match reg {
                0 => {
                    let imm_bytes = if byte { 1 } else { size.bytes().min(4) };
                    let value = c.imm(imm_bytes, true)?;
                    let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Test,
                            dst: Some(dst),
                            src: Some(Operand::Imm { value, size }),
                            rel: None,
                        },
                    ))
                }
                2..=7 => {
                    let operand = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    let (op, in_dst) = match reg {
                        2 => (Op::Not, true),
                        3 => (Op::Neg, true),
                        4 => (Op::Mul, false),
                        5 => (Op::Imul, false),
                        6 => (Op::Div, false),
                        _ => (Op::Idiv, false),
                    };
                    let (dst, src) = if in_dst {
                        (Some(operand), None)
                    } else {
                        (None, Some(operand))
                    };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst,
                            src,
                            rel: None,
                        },
                    ))
                }
                _ => Err(X86Error::Unsupported(format!("0x{opcode:02x} /{reg}"))),
            }
        }

        // MOV r/m, r  and  MOV r, r/m
        0x88..=0x8b => {
            let byte = opcode & 1 == 0;
            let to_reg = opcode & 2 != 0;
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            let reg_op = reg_operand(reg, size, p.has_rex);
            let (dst, src) = if to_reg {
                (reg_op, rm_op)
            } else {
                (rm_op, reg_op)
            };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Mov,
                    dst: Some(dst),
                    src: Some(src),
                    rel: None,
                },
            ))
        }

        // MOV Sreg, r/m16 (0x8e) and MOV r/m16, Sreg (0x8c). The ModRM reg field
        // selects the segment register (ES/CS/SS/DS/FS/GS).
        0x8c | 0x8e => {
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let rm_op = finalize_rm(rm, Size::Word, c.ip, p.has_rex, p.seg);
            if opcode == 0x8e {
                Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: Op::MovToSeg(reg as u8),
                        dst: None,
                        src: Some(rm_op),
                        rel: None,
                    },
                ))
            } else {
                Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: Op::MovFromSeg(reg as u8),
                        dst: Some(rm_op),
                        src: None,
                        rel: None,
                    },
                ))
            }
        }

        // MOV r/m, imm
        0xc6 | 0xc7 => {
            let byte = opcode == 0xc6;
            let size = op_size(&p, byte);
            let (_reg, rm) = decode_modrm(&mut c, &p)?;
            let imm_bytes = if byte { 1 } else { size.bytes().min(4) };
            let value = c.imm(imm_bytes, true)?;
            let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Mov,
                    dst: Some(dst),
                    src: Some(Operand::Imm { value, size }),
                    rel: None,
                },
            ))
        }

        // MOV r8, imm8
        0xb0..=0xb7 => {
            let num = (opcode - 0xb0) as usize | (p.rex_b() << 3);
            let value = c.imm(1, false)?;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Mov,
                    dst: Some(reg_operand(num, Size::Byte, p.has_rex)),
                    src: Some(Operand::Imm {
                        value,
                        size: Size::Byte,
                    }),
                    rel: None,
                },
            ))
        }

        // MOV r, imm (imm64 with REX.W)
        0xb8..=0xbf => {
            let size = op_size(&p, false);
            let num = (opcode - 0xb8) as usize | (p.rex_b() << 3);
            let value = c.imm(size.bytes(), false)?;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Mov,
                    dst: Some(reg_operand(num, size, p.has_rex)),
                    src: Some(Operand::Imm { value, size }),
                    rel: None,
                },
            ))
        }

        // TEST r/m, r
        0x84 | 0x85 => {
            let byte = opcode == 0x84;
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Test,
                    dst: Some(rm_op),
                    src: Some(reg_operand(reg, size, p.has_rex)),
                    rel: None,
                },
            ))
        }

        // MOVSXD r, r/m32 (movslq) — canonical 32→64 sign-extend. Source is always
        // 32-bit; destination is the reg at operand size (Qword with REX.W).
        0x63 => {
            let dst_size = op_size(&p, false);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let src = finalize_rm(rm, Size::Dword, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Movsx,
                    dst: Some(reg_operand(reg, dst_size, p.has_rex)),
                    src: Some(src),
                    rel: None,
                },
            ))
        }

        // Three-operand IMUL r, r/m, imm: 0x69 = imm32 (sign-extended), 0x6b = imm8.
        // The immediate is carried in `rel`; the executor multiplies r/m by it.
        0x69 | 0x6b => {
            let size = op_size(&p, false);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let src = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            let imm_bytes = if opcode == 0x6b { 1 } else { size.bytes().min(4) };
            let imm = c.imm(imm_bytes, true)?;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Imul,
                    dst: Some(reg_operand(reg, size, p.has_rex)),
                    src: Some(src),
                    rel: Some(imm as i64),
                },
            ))
        }

        // TEST AL, imm8 (0xa8) / TEST eAX, imm32 (0xa9).
        0xa8 | 0xa9 => {
            let byte = opcode == 0xa8;
            let size = op_size(&p, byte);
            let imm_bytes = if byte { 1 } else { size.bytes().min(4) };
            let value = c.imm(imm_bytes, true)?;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Test,
                    dst: Some(reg_operand(0, size, p.has_rex)),
                    src: Some(Operand::Imm { value, size }),
                    rel: None,
                },
            ))
        }

        // LEA r, m
        0x8d => {
            let size = op_size(&p, false);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let src = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Lea,
                    dst: Some(reg_operand(reg, size, p.has_rex)),
                    src: Some(src),
                    rel: None,
                },
            ))
        }

        // INC/DEC ; PUSH ; near-indirect CALL/JMP (r/m)
        0xfe | 0xff => {
            let byte = opcode == 0xfe;
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let op = match reg {
                0 => Op::Inc,
                1 => Op::Dec,
                2 if !byte => Op::CallIndirect,
                4 if !byte => Op::JmpIndirect,
                6 if !byte => Op::Push,
                _ => return Err(X86Error::Unsupported(format!("0x{opcode:02x} /{reg}"))),
            };
            // Near CALL/JMP indirect (/2,/4) read a 64-bit target, and PUSH r/m
            // (/6) has a default 64-bit operand size in long mode (REX.W implied;
            // only a 0x66 prefix narrows it to 16-bit) — never the 32-bit default.
            let size = if matches!(reg, 2 | 4) {
                Size::Qword
            } else if reg == 6 && !byte {
                if p.opsize { Size::Word } else { Size::Qword }
            } else {
                op_size(&p, byte)
            };
            let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(dst),
                    src: None,
                    rel: None,
                },
            ))
        }

        // PUSH/POP r64
        0x50..=0x57 => {
            let num = (opcode - 0x50) as usize | (p.rex_b() << 3);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Push,
                    dst: Some(reg_operand(num, stack_operand_size(&p), p.has_rex)),
                    src: None,
                    rel: None,
                },
            ))
        }
        0x58..=0x5f => {
            let num = (opcode - 0x58) as usize | (p.rex_b() << 3);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Pop,
                    dst: Some(reg_operand(num, stack_operand_size(&p), p.has_rex)),
                    src: None,
                    rel: None,
                },
            ))
        }
        // MOV AL/eAX, moffs and MOV moffs, AL/eAX (0xA0-0xA3): the accumulator
        // loaded from / stored to a memory operand at an absolute address (a 64-bit
        // moffset immediate in long mode). A0/A2 are byte; A0/A1 load, A2/A3 store.
        0xa0..=0xa3 => {
            let byte = opcode & 1 == 0;
            let size = if byte { Size::Byte } else { op_size(&p, false) };
            let moffs = c.imm(addr_bytes(&p), false)?;
            let mem = Operand::Mem {
                base: None,
                index: None,
                scale: 0,
                disp: moffs as i64,
                size,
                seg: p.seg,
                rip_rel: false,
            };
            let acc = reg_operand(0, size, p.has_rex);
            let (dst, src) = if opcode <= 0xa1 { (acc, mem) } else { (mem, acc) };
            Ok(finish(
                &c,
                DecodedInsn { len: 0, op: Op::Mov, dst: Some(dst), src: Some(src), rel: None },
            ))
        }
        // POP r/m64 (0x8F /0) — default 64-bit operand size in long mode. The dst
        // effective address (if RSP-relative) uses RSP *after* the pop (Op::Pop
        // reads the value first, so write() computes the address post-increment).
        0x8f => {
            let (_reg, rm) = decode_modrm(&mut c, &p)?;
            let size = if p.opsize { Size::Word } else { Size::Qword };
            let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn { len: 0, op: Op::Pop, dst: Some(dst), src: None, rel: None },
            ))
        }

        // PUSH imm — the pushed width is the stack operand size (so a 16-bit `push imm8`
        // moves 2 bytes, not 8). imm8 (0x6a) is always 1 byte, sign-extended to that width;
        // imm (0x68) is imm16 in a 16-bit segment, else imm32.
        0x68 | 0x6a => {
            let osize = stack_operand_size(&p);
            let bytes = if opcode == 0x6a { 1 } else { osize.bytes().min(4) };
            let value = c.imm(bytes, true)?;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Push,
                    dst: Some(Operand::Imm { value, size: osize }),
                    src: None,
                    rel: None,
                },
            ))
        }

        // Jcc rel8
        0x70..=0x7f => {
            let rel = c.imm(1, true)? as i64;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Jcc(Cond::from_tttn(opcode - 0x70)),
                    dst: None,
                    src: None,
                    rel: Some(rel),
                },
            ))
        }
        // JMP rel8 / rel32
        0xeb => {
            let rel = c.imm(1, true)? as i64;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Jmp,
                    dst: None,
                    src: None,
                    rel: Some(rel),
                },
            ))
        }
        0xe9 => {
            let rel = c.imm(near_rel_bytes(&p), true)? as i64;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Jmp,
                    dst: None,
                    src: None,
                    rel: Some(rel),
                },
            ))
        }
        // JMP ptr16:N far direct (invalid in 64-bit): offset (operand size) then selector.
        0xea => {
            let obytes = match op_size(&p, false) {
                Size::Word => 2,
                _ => 4,
            };
            let offset = c.imm(obytes, false)?;
            let selector = c.imm(2, false)? as u16;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::JmpFar { selector, offset },
                    dst: None,
                    src: None,
                    rel: None,
                },
            ))
        }
        // CALL rel (rel16 in 16-bit mode, rel32 in 32/64-bit)
        0xe8 => {
            let rel = c.imm(near_rel_bytes(&p), true)? as i64;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Call,
                    dst: None,
                    src: None,
                    rel: Some(rel),
                },
            ))
        }
        // RET
        0xc3 => Ok(finish(
            &c,
            DecodedInsn {
                len: 0,
                op: Op::Ret,
                dst: None,
                src: None,
                rel: None,
            },
        )),
        // LEAVE
        0xc9 => Ok(finish(
            &c,
            DecodedInsn {
                len: 0,
                op: Op::Leave,
                dst: None,
                src: None,
                rel: None,
            },
        )),
        // RET imm16 (pop extra bytes off the stack)
        0xc2 => {
            let n = c.imm(2, false)? as i64;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Ret,
                    dst: None,
                    src: None,
                    rel: Some(n),
                },
            ))
        }
        // Far return RETF/LRETQ (0xcb) and RETF imm16 (0xca). The CS reload after
        // installing a new GDT uses lretq.
        0xca | 0xcb => {
            let n = if opcode == 0xca { c.imm(2, false)? as i64 } else { 0 };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Retf,
                    dst: None,
                    src: None,
                    rel: Some(n),
                },
            ))
        }
        // PUSHFQ / POPFQ / SAHF / LAHF
        0x9c..=0x9f => {
            let op = match opcode {
                0x9c => Op::Pushf,
                0x9d => Op::Popf,
                0x9e => Op::Sahf,
                _ => Op::Lahf,
            };
            // For PUSHF/POPF, carry the flags operand width (2/4/8) so exec pushes/pops the
            // right number of bytes and (for POPF) knows which flag bits are writable. In
            // long mode the default is 8 (PUSHFQ/POPFQ), 2 with a 0x66 prefix; in 16/32-bit
            // it follows op_size (Word→2, Dword→4). Sahf/Lahf take no width (rel = None).
            let rel = if matches!(op, Op::Pushf | Op::Popf) {
                let fw = if p.mode == DecodeMode::Bits64 {
                    if p.opsize {
                        2
                    } else {
                        8
                    }
                } else {
                    match op_size(&p, false) {
                        Size::Word => 2,
                        _ => 4,
                    }
                };
                Some(fw as i64)
            } else {
                None
            };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: None,
                    src: None,
                    rel,
                },
            ))
        }

        // CLI / STI
        0xfa | 0xfb => Ok(finish(
            &c,
            DecodedInsn {
                len: 0,
                op: if opcode == 0xfa { Op::Cli } else { Op::Sti },
                dst: None,
                src: None,
                rel: None,
            },
        )),
        // FWAIT / WAIT — no unmasked pending FP exceptions in this model → no-op.
        0x9b => Ok(finish(
            &c,
            DecodedInsn { len: 0, op: Op::Fwait, dst: None, src: None, rel: None },
        )),
        // x87 escape opcodes D8-DF.
        0xd8..=0xdf => {
            let insn = decode_x87(&mut c, &p, opcode)?;
            Ok(finish(&c, insn))
        }
        // Flag manipulation: CLC/STC/CMC (carry), CLD/STD (direction).
        0xf5 | 0xf8 | 0xf9 | 0xfc | 0xfd => Ok(finish(
            &c,
            DecodedInsn {
                len: 0,
                op: match opcode {
                    0xf5 => Op::Cmc,
                    0xf8 => Op::Clc,
                    0xf9 => Op::Stc,
                    0xfc => Op::Cld,
                    _ => Op::Std, // 0xfd
                },
                dst: None,
                src: None,
                rel: None,
            },
        )),
        // IRET / IRETQ (long mode: always 64-bit pops in this model)
        0xcf => Ok(finish(
            &c,
            DecodedInsn {
                len: 0,
                op: Op::Iret,
                dst: None,
                src: None,
                rel: None,
            },
        )),
        // INT3 / INT n
        0xcc | 0xcd => {
            let vector = if opcode == 0xcc { 3 } else { c.imm(1, false)? };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Int,
                    dst: None,
                    src: Some(Operand::Imm {
                        value: vector,
                        size: Size::Byte,
                    }),
                    rel: None,
                },
            ))
        }

        // XCHG r/m, r
        0x86 | 0x87 => {
            let byte = opcode == 0x86;
            let size = op_size(&p, byte);
            let (reg, rm) = decode_modrm(&mut c, &p)?;
            let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Xchg,
                    dst: Some(rm_op),
                    src: Some(reg_operand(reg, size, p.has_rex)),
                    rel: None,
                },
            ))
        }
        // XCHG rAX, r. The accumulator slot 0x90 needs care: a *plain* 0x90
        // (also `48 90`, `66 90`, and `F3 90` = PAUSE) is a true NOP and must NOT
        // be executed as `xchg eax,eax` (that would zero-extend and clobber the
        // upper 32 bits of RAX). But **`0x90` with REX.B is `xchg rax,r8`** — a
        // real swap. So it is XCHG only when REX selects a non-accumulator reg.
        0x90..=0x97 => {
            let num = (opcode - 0x90) as usize | (p.rex_b() << 3);
            if num == 0 {
                // Plain 0x90 = NOP; `F3 90` = PAUSE (same architectural effect, but a spin-wait
                // hint the SMP scheduler yields on). Both leave RAX untouched.
                let op = if matches!(p.rep, Rep::F3) { Op::Pause } else { Op::Nop };
                Ok(finish(
                    &c,
                    DecodedInsn { len: 0, op, dst: None, src: None, rel: None },
                ))
            } else {
                let size = op_size(&p, false);
                Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: Op::Xchg,
                        dst: Some(reg_operand(0, size, p.has_rex)),
                        src: Some(reg_operand(num, size, p.has_rex)),
                        rel: None,
                    },
                ))
            }
        }
        // String ops: MOVS/CMPS/STOS/LODS/SCAS
        0xa4 | 0xa5 | 0xa6 | 0xa7 | 0xaa | 0xab | 0xac | 0xad | 0xae | 0xaf => {
            let byte = opcode & 1 == 0;
            let size = op_size(&p, byte);
            let sop = match opcode {
                0xa4 | 0xa5 => StrOp::Movs,
                0xa6 | 0xa7 => StrOp::Cmps,
                0xaa | 0xab => StrOp::Stos,
                0xac | 0xad => StrOp::Lods,
                _ => StrOp::Scas,
            };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Str(sop, size, p.rep),
                    dst: None,
                    src: None,
                    rel: None,
                },
            ))
        }

        // IN AL/eAX, imm8  /  OUT imm8, AL/eAX
        0xe4..=0xe7 => {
            let byte = opcode & 1 == 0;
            let size = if byte { Size::Byte } else { op_size(&p, false) };
            let port = Operand::Imm {
                value: c.imm(1, false)?,
                size: Size::Byte,
            };
            let acc = reg_operand(0, size, p.has_rex); // AL/eAX
            let (op, dst, src) = if opcode < 0xe6 {
                (Op::In, acc, port)
            } else {
                (Op::Out, port, acc)
            };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(dst),
                    src: Some(src),
                    rel: None,
                },
            ))
        }
        // IN AL/eAX, DX  /  OUT DX, AL/eAX
        0xec..=0xef => {
            let byte = opcode & 1 == 0;
            let size = if byte { Size::Byte } else { op_size(&p, false) };
            let port = reg_operand(super::regs::RDX, Size::Word, p.has_rex);
            let acc = reg_operand(0, size, p.has_rex);
            let (op, dst, src) = if opcode < 0xee {
                (Op::In, acc, port)
            } else {
                (Op::Out, port, acc)
            };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(dst),
                    src: Some(src),
                    rel: None,
                },
            ))
        }

        // CWDE/CDQE (0x98) and CDQ/CQO (0x99)
        0x98 | 0x99 => {
            let size = op_size(&p, false);
            let op = if opcode == 0x98 { Op::Cwde } else { Op::Cdq };
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op,
                    dst: Some(reg_operand(0, size, p.has_rex)),
                    src: None,
                    rel: None,
                },
            ))
        }
        // LOOPNE/LOOPE/LOOP/JrCXZ rel8
        0xe0..=0xe3 => {
            let kind = match opcode {
                0xe0 => LoopKind::Loopne,
                0xe1 => LoopKind::Loope,
                0xe2 => LoopKind::Loop,
                _ => LoopKind::Jrcxz,
            };
            let rel = c.imm(1, true)? as i64;
            Ok(finish(
                &c,
                DecodedInsn {
                    len: 0,
                    op: Op::Loop(kind),
                    dst: None,
                    src: None,
                    rel: Some(rel),
                },
            ))
        }

        0xf4 => Ok(finish(
            &c,
            DecodedInsn {
                len: 0,
                op: Op::Hlt,
                dst: None,
                src: None,
                rel: None,
            },
        )),

        // Two-byte opcodes.
        0x0f => {
            let op2 = c.u8()?;
            match op2 {
                // Three-byte SSSE3/SSE4 map 0F 38 (66-prefixed xmm forms). Many
                // reuse the element-generic packed compare/min-max helpers.
                0x38 => {
                    let op3 = c.u8()?;
                    // MOVBE (0F 38 F0/F1, no F2): byte-reversed GPR↔memory move.
                    // F0 = load (dst=reg, src=mem); F1 = store (dst=mem, src=reg).
                    if (op3 == 0xf0 || op3 == 0xf1) && !matches!(p.rep, Rep::F2) {
                        let size = if p.opsize {
                            Size::Word
                        } else if p.rex_w() {
                            Size::Qword
                        } else {
                            Size::Dword
                        };
                        let (reg, rm) = decode_modrm(&mut c, &p)?;
                        let mem = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                        let reg_op = reg_operand(reg, size, p.has_rex);
                        let store = op3 == 0xf1;
                        let (dst, src) = if store { (mem, reg_op) } else { (reg_op, mem) };
                        return Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op: Op::Movbe { store }, dst: Some(dst), src: Some(src), rel: None },
                        ));
                    }
                    // CRC32 (F2 0F 38 F0/F1): GPR dst, r/m src — not an xmm op.
                    if op3 == 0xf0 || op3 == 0xf1 {
                        let bytes: u8 = if op3 == 0xf0 {
                            1
                        } else if p.opsize {
                            2
                        } else if p.rex_w() {
                            8
                        } else {
                            4
                        };
                        let src_size = match bytes {
                            1 => Size::Byte,
                            2 => Size::Word,
                            8 => Size::Qword,
                            _ => Size::Dword,
                        };
                        let (reg, rm) = decode_modrm(&mut c, &p)?;
                        let src = finalize_rm(rm, src_size, c.ip, p.has_rex, p.seg);
                        let dsize = if p.rex_w() { Size::Qword } else { Size::Dword };
                        let dst = reg_operand(reg, dsize, p.has_rex);
                        return Ok(finish(
                            &c,
                            DecodedInsn {
                                len: 0,
                                op: Op::Crc32 { bytes },
                                dst: Some(dst),
                                src: Some(src),
                                rel: None,
                            },
                        ));
                    }
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let dst = Operand::Xmm { id: reg };
                    let op = match op3 {
                        0x00 => Op::Pshufb,
                        0x29 => Op::PackedCmpEq(8), // PCMPEQQ (SSE4.1)
                        0x37 => Op::PackedCmpGt(8), // PCMPGTQ (SSE4.2)
                        0x40 => Op::Pmulld,
                        0x38 => Op::PackedMinMax { max: false, signed: true, elem: 1 }, // PMINSB
                        0x39 => Op::PackedMinMax { max: false, signed: true, elem: 4 }, // PMINSD
                        0x3a => Op::PackedMinMax { max: false, signed: false, elem: 2 }, // PMINUW
                        0x3b => Op::PackedMinMax { max: false, signed: false, elem: 4 }, // PMINUD
                        0x3c => Op::PackedMinMax { max: true, signed: true, elem: 1 }, // PMAXSB
                        0x3d => Op::PackedMinMax { max: true, signed: true, elem: 4 }, // PMAXSD
                        0x3e => Op::PackedMinMax { max: true, signed: false, elem: 2 }, // PMAXUW
                        0x3f => Op::PackedMinMax { max: true, signed: false, elem: 4 }, // PMAXUD
                        0x17 => Op::Ptest,
                        0x01 => Op::Phadd { elem: 2, sub: false, sat: false }, // PHADDW
                        0x02 => Op::Phadd { elem: 4, sub: false, sat: false }, // PHADDD
                        0x03 => Op::Phadd { elem: 2, sub: false, sat: true },  // PHADDSW
                        0x05 => Op::Phadd { elem: 2, sub: true, sat: false },  // PHSUBW
                        0x06 => Op::Phadd { elem: 4, sub: true, sat: false },  // PHSUBD
                        0x07 => Op::Phadd { elem: 2, sub: true, sat: true },   // PHSUBSW
                        0x08 => Op::Psign { elem: 1 },
                        0x09 => Op::Psign { elem: 2 },
                        0x0a => Op::Psign { elem: 4 },
                        0x04 => Op::Pmaddubsw,
                        0x0b => Op::Pmulhrsw,
                        0x10 => Op::Blendv { elem: 1 }, // PBLENDVB
                        0x14 => Op::Blendv { elem: 4 }, // BLENDVPS
                        0x15 => Op::Blendv { elem: 8 }, // BLENDVPD
                        0x41 => Op::Phminposuw,
                        0x2b => Op::Pack { dword_src: true, signed: false }, // PACKUSDW
                        0x1c => Op::Pabs { elem: 1 },
                        0x1d => Op::Pabs { elem: 2 },
                        0x1e => Op::Pabs { elem: 4 },
                        0x20 => Op::Pmovx { src_elem: 1, dst_elem: 2, sign: true }, // PMOVSXBW
                        0x21 => Op::Pmovx { src_elem: 1, dst_elem: 4, sign: true }, // PMOVSXBD
                        0x22 => Op::Pmovx { src_elem: 1, dst_elem: 8, sign: true }, // PMOVSXBQ
                        0x23 => Op::Pmovx { src_elem: 2, dst_elem: 4, sign: true }, // PMOVSXWD
                        0x24 => Op::Pmovx { src_elem: 2, dst_elem: 8, sign: true }, // PMOVSXWQ
                        0x25 => Op::Pmovx { src_elem: 4, dst_elem: 8, sign: true }, // PMOVSXDQ
                        0x30 => Op::Pmovx { src_elem: 1, dst_elem: 2, sign: false }, // PMOVZXBW
                        0x31 => Op::Pmovx { src_elem: 1, dst_elem: 4, sign: false }, // PMOVZXBD
                        0x32 => Op::Pmovx { src_elem: 1, dst_elem: 8, sign: false }, // PMOVZXBQ
                        0x33 => Op::Pmovx { src_elem: 2, dst_elem: 4, sign: false }, // PMOVZXWD
                        0x34 => Op::Pmovx { src_elem: 2, dst_elem: 8, sign: false }, // PMOVZXWQ
                        0x35 => Op::Pmovx { src_elem: 4, dst_elem: 8, sign: false }, // PMOVZXDQ
                        _ => return Err(X86Error::Unsupported(format!("0f 38 {op3:02x}"))),
                    };
                    Ok(finish(&c, sse(op, dst, src)))
                }
                // Three-byte SSE4 map 0F 3A (66-prefixed, imm8).
                0x3a => {
                    let op3 = c.u8()?;
                    // PINSR (xmm dst, GPR/mem src) and PEXTR/EXTRACTPS (GPR/mem dst,
                    // xmm src) carry a GPR operand — decode them before the xmm ops.
                    if matches!(op3, 0x20 | 0x22) {
                        let elem: u8 = if op3 == 0x20 {
                            1
                        } else if p.rex_w() {
                            8
                        } else {
                            4
                        };
                        let gsize = match elem {
                            1 => Size::Byte,
                            8 => Size::Qword,
                            _ => Size::Dword,
                        };
                        let (reg, rm) = decode_modrm(&mut c, &p)?;
                        let src = finalize_rm(rm, gsize, c.ip, p.has_rex, p.seg);
                        let imm = c.imm(1, false)?;
                        return Ok(finish(
                            &c,
                            DecodedInsn {
                                len: 0,
                                op: Op::Pinsr { elem },
                                dst: Some(Operand::Xmm { id: reg }),
                                src: Some(src),
                                rel: Some(imm as i64),
                            },
                        ));
                    }
                    if matches!(op3, 0x14 | 0x15 | 0x16 | 0x17) {
                        let elem: u8 = match op3 {
                            0x14 => 1,
                            0x15 => 2,
                            0x16 if p.rex_w() => 8,
                            _ => 4, // PEXTRD (0x16) / EXTRACTPS (0x17)
                        };
                        let gsize = match elem {
                            1 => Size::Byte,
                            2 => Size::Word,
                            8 => Size::Qword,
                            _ => Size::Dword,
                        };
                        let (reg, rm) = decode_modrm(&mut c, &p)?;
                        let dst = finalize_rm(rm, gsize, c.ip, p.has_rex, p.seg);
                        let imm = c.imm(1, false)?;
                        return Ok(finish(
                            &c,
                            DecodedInsn {
                                len: 0,
                                op: Op::Pextr { elem },
                                dst: Some(dst),
                                src: Some(Operand::Xmm { id: reg }),
                                rel: Some(imm as i64),
                            },
                        ));
                    }
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let imm = c.imm(1, false)?;
                    let op = match op3 {
                        0x0f => Op::Palignr,
                        0x08 => Op::Round { double: false, scalar: false }, // ROUNDPS
                        0x09 => Op::Round { double: true, scalar: false },  // ROUNDPD
                        0x0a => Op::Round { double: false, scalar: true },  // ROUNDSS
                        0x0b => Op::Round { double: true, scalar: true },   // ROUNDSD
                        0x0c => Op::Blend { elem: 4 }, // BLENDPS
                        0x0d => Op::Blend { elem: 8 }, // BLENDPD
                        0x0e => Op::Blend { elem: 2 }, // PBLENDW
                        0x21 => Op::Insertps,
                        0x40 => Op::Dpp { double: false }, // DPPS
                        0x41 => Op::Dpp { double: true },  // DPPD
                        0x42 => Op::Mpsadbw,
                        0x60 => Op::Pcmpstr { explicit: true, mask_out: true }, // PCMPESTRM
                        0x61 => Op::Pcmpstr { explicit: true, mask_out: false }, // PCMPESTRI
                        0x62 => Op::Pcmpstr { explicit: false, mask_out: true }, // PCMPISTRM
                        0x63 => Op::Pcmpstr { explicit: false, mask_out: false }, // PCMPISTRI
                        _ => return Err(X86Error::Unsupported(format!("0f 3a {op3:02x}"))),
                    };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(Operand::Xmm { id: reg }),
                            src: Some(src),
                            rel: Some(imm as i64),
                        },
                    ))
                }
                // 0F 0D (PREFETCHW), 0F 18–1F: prefetch hints + the reserved
                // hint-NOP space — multi-byte NOP padding and the CET markers
                // ENDBR64/ENDBR32 (F3 0F 1E FA/FB). All are NOPs when the feature is
                // off / cache is unmodeled. Consume the ModRM/SIB/disp; do nothing.
                0x0d | 0x18 | 0x19 | 0x1a | 0x1b | 0x1c | 0x1d | 0x1e | 0x1f => {
                    let (_reg, _rm) = decode_modrm(&mut c, &p)?;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Nop,
                            dst: None,
                            src: None,
                            rel: None,
                        },
                    ))
                }
                // Group 0F 00: SLDT(/0) STR(/1) LLDT(/2) LTR(/3) VERR(/4) VERW(/5).
                // We don't model the LDT/TSS as segment descriptors; LLDT/LTR just
                // load the selector (no flat-mode effect → NOP; TSS.RSP0 for ring
                // transitions is set via the M4 path). SLDT/STR store a null/zero
                // selector. VERR/VERW report the segment accessible (ZF=1).
                0x00 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    match reg {
                        // LLDT (/2): no LDT modeled (flat long mode) → NOP.
                        2 => Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op: Op::Nop, dst: None, src: None, rel: None },
                        )),
                        // LTR (/3): load the task register from a 16-bit selector;
                        // the executor resolves the TSS base from the GDT so
                        // privilege-increasing interrupts can read TSS.RSP0.
                        3 => {
                            let src = finalize_rm(rm, Size::Word, c.ip, p.has_rex, p.seg);
                            Ok(finish(
                                &c,
                                DecodedInsn {
                                    len: 0,
                                    op: Op::Ltr,
                                    dst: None,
                                    src: Some(src),
                                    rel: None,
                                },
                            ))
                        }
                        0 | 1 => {
                            let dst = finalize_rm(rm, Size::Word, c.ip, p.has_rex, p.seg);
                            Ok(finish(
                                &c,
                                DecodedInsn {
                                    len: 0,
                                    op: Op::Mov,
                                    dst: Some(dst),
                                    src: Some(Operand::Imm { value: 0, size: Size::Word }),
                                    rel: None,
                                },
                            ))
                        }
                        4 | 5 => Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op: Op::Verw, dst: None, src: None, rel: None },
                        )),
                        _ => Err(X86Error::Unsupported(format!("0f 00 /{reg}"))),
                    }
                }
                // Group 0F 01: LGDT (/2), LIDT (/3), SWAPGS (reg form 0xF8).
                0x01 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    match (reg, &rm) {
                        (7, RmRaw::Reg(0)) => Ok(finish(
                            &c,
                            DecodedInsn {
                                len: 0,
                                op: Op::Swapgs,
                                dst: None,
                                src: None,
                                rel: None,
                            },
                        )),
                        (7, RmRaw::Reg(1)) => Ok(finish(
                            &c,
                            DecodedInsn {
                                len: 0,
                                op: Op::Rdtscp,
                                dst: None,
                                src: None,
                                rel: None,
                            },
                        )),
                        // XGETBV (0F 01 D0) / XSETBV (0F 01 D1).
                        (2, RmRaw::Reg(0)) => Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op: Op::Xgetbv, dst: None, src: None, rel: None },
                        )),
                        (2, RmRaw::Reg(1)) => Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op: Op::Xsetbv, dst: None, src: None, rel: None },
                        )),
                        (0 | 1 | 2 | 3, RmRaw::Mem { .. }) => {
                            let operand = finalize_rm(rm, Size::Qword, c.ip, p.has_rex, p.seg);
                            let op = match reg {
                                0 => Op::Sgdt,
                                1 => Op::Sidt,
                                2 => Op::Lgdt,
                                _ => Op::Lidt,
                            };
                            Ok(finish(
                                &c,
                                DecodedInsn { len: 0, op, dst: Some(operand), src: None, rel: None },
                            ))
                        }
                        // INVLPG m (0F 01 /7, memory form): the machine is TLB-less
                        // by design, so there is no translation cache to invalidate.
                        // But the DECODE cache keys on (rip, cr3): a single-page remap
                        // (PTE change + INVLPG, no write to the old page) can change
                        // the bytes at a cached rip, so INVLPG must flush it — see the
                        // TLB note in docs/cpu-determinism-contract.md §12.
                        (7, RmRaw::Mem { .. }) => Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op: Op::Invlpg, dst: None, src: None, rel: None },
                        )),
                        _ => Err(X86Error::Unsupported(format!("0f 01 /{reg}"))),
                    }
                }
                // SYSCALL / SYSRET (treated as SYSRETQ in long mode)
                0x05 | 0x07 => Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: if op2 == 0x05 { Op::Syscall } else { Op::Sysret },
                        dst: None,
                        src: None,
                        rel: None,
                    },
                )),
                // INVD / WBINVD (0F 08 / 0F 09): cache management. No cache is
                // modeled (memory is always coherent), so both are NOPs.
                0x08 | 0x09 => Ok(finish(
                    &c,
                    DecodedInsn { len: 0, op: Op::Nop, dst: None, src: None, rel: None },
                )),
                // Jcc rel32
                0x80..=0x8f => {
                    let rel = c.imm(near_rel_bytes(&p), true)? as i64;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Jcc(Cond::from_tttn(op2 - 0x80)),
                            dst: None,
                            src: None,
                            rel: Some(rel),
                        },
                    ))
                }
                // WRMSR / RDMSR (no explicit operands; implicit ECX, EDX:EAX)
                0x30 | 0x32 => Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: if op2 == 0x30 { Op::Wrmsr } else { Op::Rdmsr },
                        dst: None,
                        src: None,
                        rel: None,
                    },
                )),
                // CPUID (implicit EAX/ECX in, EAX/EBX/ECX/EDX out)
                0xa2 => Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: Op::Cpuid,
                        dst: None,
                        src: None,
                        rel: None,
                    },
                )),
                // SHLD/SHRD r/m, r, imm8|CL
                0xa4 | 0xa5 | 0xac | 0xad => {
                    let size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let count = if op2 & 1 == 0 {
                        Some(c.imm(1, false)? as i64) // imm8 form
                    } else {
                        None // CL form
                    };
                    let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    let op = if op2 < 0xac { Op::Shld } else { Op::Shrd };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(dst),
                            src: Some(reg_operand(reg, size, p.has_rex)),
                            rel: count,
                        },
                    ))
                }
                // BSWAP r32/r64
                0xc8..=0xcf => {
                    let size = op_size(&p, false);
                    let num = (op2 - 0xc8) as usize | (p.rex_b() << 3);
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Bswap,
                            dst: Some(reg_operand(num, size, p.has_rex)),
                            src: None,
                            rel: None,
                        },
                    ))
                }
                // 0F AE: MFENCE/LFENCE/SFENCE (no-ops on single vCPU)
                0xae => {
                    let modrm = c.peek()?;
                    if modrm >> 6 == 3 {
                        // Register form: LFENCE (0xe8)/MFENCE (0xf0)/SFENCE (0xf8). Under the
                        // TSO model these drain the store buffer (Op::Fence); with TSO off the
                        // buffer is empty so it is a NOP, exactly as before.
                        let _ = c.u8();
                        match modrm {
                            0xe8..=0xff => Ok(finish(
                                &c,
                                DecodedInsn { len: 0, op: Op::Fence, dst: None, src: None, rel: None },
                            )),
                            other => Err(X86Error::Unsupported(format!("0f ae {other:#x}"))),
                        }
                    } else {
                        // Memory form: /0 FXSAVE /1 FXRSTOR /2 LDMXCSR /3 STMXCSR
                        // /7 CLFLUSH (NOP).
                        let (reg, rm) = decode_modrm(&mut c, &p)?;
                        let mem = finalize_rm(rm, Size::Qword, c.ip, p.has_rex, p.seg);
                        let op = match reg {
                            0 => Op::Fxsave,
                            1 => Op::Fxrstor,
                            2 => Op::Ldmxcsr,
                            3 => Op::Stmxcsr,
                            4 => Op::Xsave { compacted: false },
                            5 => Op::Xrstor,
                            6 => Op::Xsave { compacted: false }, // XSAVEOPT = standard XSAVE (no init opt)
                            7 => Op::Nop, // CLFLUSH: no cache modeled
                            _ => return Err(X86Error::Unsupported(format!("0f ae /{reg} mem"))),
                        };
                        let (dst, src) = match op {
                            Op::Fxrstor | Op::Ldmxcsr | Op::Xrstor => (None, Some(mem)),
                            _ => (Some(mem), None),
                        };
                        Ok(finish(&c, DecodedInsn { len: 0, op, dst, src, rel: None }))
                    }
                }
                // RDTSC
                0x31 => Ok(finish(
                    &c,
                    DecodedInsn {
                        len: 0,
                        op: Op::Rdtsc,
                        dst: None,
                        src: None,
                        rel: None,
                    },
                )),
                // MOV r64,CRn / MOV CRn,r64 / MOV r64,DRn / MOV DRn,r64
                0x20..=0x23 => {
                    let modrm = c.u8()?; // mod treated as 11; reg=ctrl#, rm=GPR
                    let ctrl = ((modrm >> 3) & 7) | ((p.rex_r() as u8) << 3);
                    let gpr = (modrm & 7) as usize | (p.rex_b() << 3);
                    let op = match op2 {
                        0x20 => Op::MovCr { write: false, reg: ctrl },
                        0x22 => Op::MovCr { write: true, reg: ctrl },
                        0x21 => Op::MovDr { write: false, reg: ctrl },
                        _ => Op::MovDr { write: true, reg: ctrl },
                    };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(reg_operand(gpr, Size::Qword, p.has_rex)),
                            src: None,
                            rel: None,
                        },
                    ))
                }
                // XADD r/m, r  /  CMPXCHG r/m, r
                0xc0 | 0xc1 | 0xb0 | 0xb1 => {
                    let byte = op2 & 1 == 0;
                    let size = op_size(&p, byte);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    let op = if op2 < 0xc0 { Op::Cmpxchg } else { Op::Xadd };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(rm_op),
                            src: Some(reg_operand(reg, size, p.has_rex)),
                            rel: None,
                        },
                    ))
                }
                // BT/BTS/BTR/BTC r/m, r
                0xa3 | 0xab | 0xb3 | 0xbb => {
                    let size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    let op = match op2 {
                        0xa3 => Op::Bt,
                        0xab => Op::Bts,
                        0xb3 => Op::Btr,
                        _ => Op::Btc,
                    };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(rm_op),
                            src: Some(reg_operand(reg, size, p.has_rex)),
                            rel: None,
                        },
                    ))
                }
                // BT/BTS/BTR/BTC r/m, imm8  (group 0F BA)
                0xba => {
                    let size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let imm = c.imm(1, false)?;
                    let rm_op = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    let op = match reg {
                        4 => Op::Bt,
                        5 => Op::Bts,
                        6 => Op::Btr,
                        7 => Op::Btc,
                        _ => return Err(X86Error::Unsupported(format!("0f ba /{reg}"))),
                    };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(rm_op),
                            src: Some(Operand::Imm {
                                value: imm,
                                size: Size::Byte,
                            }),
                            rel: None,
                        },
                    ))
                }
                // BSF / BSR / POPCNT  (r, r/m)
                0xbc | 0xbd | 0xb8 => {
                    let size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    let f3 = matches!(p.rep, Rep::F3);
                    let op = match op2 {
                        0xbc => if f3 { Op::Tzcnt } else { Op::Bsf }, // F3 0F BC = TZCNT
                        0xbd => if f3 { Op::Lzcnt } else { Op::Bsr }, // F3 0F BD = LZCNT
                        _ => Op::Popcnt,
                    };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(reg_operand(reg, size, p.has_rex)),
                            src: Some(src),
                            rel: None,
                        },
                    ))
                }
                // SSE moves. Mandatory prefixes (66/F3) select the variant.
                // MOVD/MOVQ xmm, r/m (66 0F 6E)
                0x6e if p.opsize => {
                    let w = if p.rex_w() { 8 } else { 4 };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, if w == 8 { Size::Qword } else { Size::Dword }, c.ip, p.has_rex, p.seg);
                    Ok(finish(&c, sse(Op::XmmLoad(w), Operand::Xmm { id: reg }, src)))
                }
                // MOVD/MOVQ r/m, xmm (66 0F 7E) ; MOVQ xmm, xmm/m64 (F3 0F 7E)
                0x7e => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    if p.opsize {
                        let w = if p.rex_w() { 8 } else { 4 };
                        let dst = finalize_rm(rm, if w == 8 { Size::Qword } else { Size::Dword }, c.ip, p.has_rex, p.seg);
                        Ok(finish(&c, sse(Op::XmmStore(w), dst, Operand::Xmm { id: reg })))
                    } else if matches!(p.rep, Rep::F3) {
                        let src = finalize_rm_xmm(rm, c.ip, p.seg);
                        Ok(finish(&c, sse(Op::XmmLoad(8), Operand::Xmm { id: reg }, src)))
                    } else {
                        Err(X86Error::Unsupported("0f 7e (no prefix)".into()))
                    }
                }
                // MOVUPS/MOVUPD/MOVSS/MOVSD xmm, xmm/m (load). The mandatory prefix
                // selects width: F3 = MOVSS (scalar 4B), F2 = MOVSD (scalar 8B),
                // 66 = MOVUPD (128b), none = MOVUPS (128b). Scalar reg→reg moves
                // preserve the destination's upper bits (see Op::XmmScalarMov).
                0x10 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let dst = Operand::Xmm { id: reg };
                    let op = match p.rep {
                        Rep::F3 => Op::XmmScalarMov(4),
                        Rep::F2 => Op::XmmScalarMov(8),
                        _ => Op::XmmLoad(16),
                    };
                    Ok(finish(&c, sse(op, dst, src)))
                }
                // ...store form (0x11): destination is the r/m, source is the xmm.
                0x11 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                    let src = Operand::Xmm { id: reg };
                    let op = match p.rep {
                        Rep::F3 => Op::XmmScalarMov(4),
                        Rep::F2 => Op::XmmScalarMov(8),
                        _ => Op::XmmStore(16),
                    };
                    Ok(finish(&c, sse(op, dst, src)))
                }
                // MOVLPS/MOVLPD (0F 12) load to low quad / MOVHLPS (reg form);
                // MOVHPS/MOVHPD (0F 16) load to high quad / MOVLHPS (reg form).
                // The 66 prefix (PD) is identical here. Both move a 64-bit half.
                0x12 | 0x16 => {
                    let high = op2 == 0x16;
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    // SSE3 duplicating moves select on the prefix: F2 0F 12 = MOVDDUP,
                    // F3 0F 12 = MOVSLDUP, F3 0F 16 = MOVSHDUP. Otherwise MOVL/HPS.
                    let op = match (op2, p.rep) {
                        (0x12, Rep::F2) => Op::Movddup,
                        (0x12, Rep::F3) => Op::Movsdup { high: false },
                        (0x16, Rep::F3) => Op::Movsdup { high: true },
                        _ => Op::XmmMovHalf { high, store: false },
                    };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // MOVLPS/MOVLPD (0F 13) store low quad; MOVHPS/MOVHPD (0F 17) store
                // high quad. Destination is always memory.
                0x13 | 0x17 => {
                    let high = op2 == 0x17;
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(
                        &c,
                        sse(Op::XmmMovHalf { high, store: true }, dst, Operand::Xmm { id: reg }),
                    ))
                }
                // MOVDQA/MOVDQU/MOVAPS/MOVAPD xmm, xmm/m128 — always 128-bit (load)
                0x6f | 0x28 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::XmmLoad(16), Operand::Xmm { id: reg }, src)))
                }
                // ...store
                0x7f | 0x29 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::XmmStore(16), dst, Operand::Xmm { id: reg })))
                }
                // SSE 128-bit bitwise (66 prefix): PXOR/POR/PAND/PANDN, and the
                // packed-float forms ANDPS/ANDNPS/ORPS/XORPS (0F 54-57) /
                // ANDPD..XORPD (66 0F 54-57) — all identical bitwise on 128 bits.
                0xef | 0xeb | 0xdb | 0xdf if p.opsize => {
                    let kind = match op2 {
                        0xef => PLogicKind::Xor,
                        0xeb => PLogicKind::Or,
                        0xdb => PLogicKind::And,
                        _ => PLogicKind::Andn,
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::PLogic(kind), Operand::Xmm { id: reg }, src)))
                }
                0x54 | 0x55 | 0x56 | 0x57 => {
                    let kind = match op2 {
                        0x54 => PLogicKind::And,
                        0x55 => PLogicKind::Andn,
                        0x56 => PLogicKind::Or,
                        _ => PLogicKind::Xor, // 0x57
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::PLogic(kind), Operand::Xmm { id: reg }, src)))
                }
                // MOVNTPS/MOVNTPD/MOVNTDQ (0F 2B, 66 0F E7): non-temporal 128-bit
                // store. We have no cache, so it is an ordinary aligned store.
                0x2b => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::XmmStore(16), dst, Operand::Xmm { id: reg })))
                }
                // Packed compare equal (PCMPEQB/W/D) and greater-than (PCMPGTB/W/D),
                // 66 prefix.
                0x74 | 0x75 | 0x76 | 0x64 | 0x65 | 0x66 if p.opsize => {
                    let op = match op2 {
                        0x74 => Op::PackedCmpEq(1),
                        0x75 => Op::PackedCmpEq(2),
                        0x76 => Op::PackedCmpEq(4),
                        0x64 => Op::PackedCmpGt(1),
                        0x65 => Op::PackedCmpGt(2),
                        _ => Op::PackedCmpGt(4), // 0x66
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // Packed shift by xmm/m128 count (66 prefix): PSLLW/D/Q (F1/F2/F3),
                // PSRLW/D/Q (D1/D2/D3), PSRAW/D (E1/E2).
                0xf1 | 0xf2 | 0xf3 | 0xd1 | 0xd2 | 0xd3 | 0xe1 | 0xe2 if p.opsize => {
                    let (kind, elem) = match op2 {
                        0xf1 => (ShiftKind::Sll, 2),
                        0xf2 => (ShiftKind::Sll, 4),
                        0xf3 => (ShiftKind::Sll, 8),
                        0xd1 => (ShiftKind::Srl, 2),
                        0xd2 => (ShiftKind::Srl, 4),
                        0xd3 => (ShiftKind::Srl, 8),
                        0xe1 => (ShiftKind::Sra, 2),
                        _ => (ShiftKind::Sra, 4), // 0xe2
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::PackedShift { kind, elem }, Operand::Xmm { id: reg }, src)))
                }
                // Packed shift by imm8 (66 prefix): group 0F 71/72/73 /reg ib.
                //   71: /6 PSLLW /2 PSRLW /4 PSRAW   72: /6 PSLLD /2 PSRLD /4 PSRAD
                //   73: /6 PSLLQ /2 PSRLQ
                0x71..=0x73 if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                    let imm = c.imm(1, false)?;
                    let imm_op = Operand::Imm { value: imm, size: Size::Byte };
                    // The xmm being shifted is the r/m (a register), in place.
                    if op2 == 0x73 && (reg == 3 || reg == 7) {
                        // PSRLDQ (/3) / PSLLDQ (/7): shift the WHOLE 128-bit register
                        // right/left by `imm` bytes (used in SSE reductions).
                        Ok(finish(&c, sse(Op::XmmByteShift { left: reg == 7 }, dst, imm_op)))
                    } else {
                        let elem = match op2 {
                            0x71 => 2,
                            0x72 => 4,
                            _ => 8, // 0x73
                        };
                        let kind = match reg {
                            6 => ShiftKind::Sll,
                            2 => ShiftKind::Srl,
                            4 => ShiftKind::Sra,
                            _ => return Err(X86Error::Unsupported(format!("0f {op2:02x} /{reg}"))),
                        };
                        Ok(finish(&c, sse(Op::PackedShift { kind, elem }, dst, imm_op)))
                    }
                }
                // PSHUFD (66) / PSHUFLW (F2) / PSHUFHW (F3) xmm, xmm/m128, imm8 (0F 70).
                0x70 if p.opsize || matches!(p.rep, Rep::F2 | Rep::F3) => {
                    let op = if p.opsize {
                        Op::Pshufd
                    } else {
                        Op::PshufW { high: matches!(p.rep, Rep::F3) }
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let imm = c.imm(1, false)?;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(Operand::Xmm { id: reg }),
                            src: Some(src),
                            rel: Some(imm as i64),
                        },
                    ))
                }
                // PUNPCK{L,H}{BW,WD,DQ,QDQ} (66 prefix).
                0x60 | 0x61 | 0x62 | 0x6c | 0x68 | 0x69 | 0x6a | 0x6d if p.opsize => {
                    let (high, elem) = match op2 {
                        0x60 => (false, 1),
                        0x61 => (false, 2),
                        0x62 => (false, 4),
                        0x6c => (false, 8),
                        0x68 => (true, 1),
                        0x69 => (true, 2),
                        0x6a => (true, 4),
                        _ => (true, 8), // 0x6d
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::Punpck { high, elem }, Operand::Xmm { id: reg }, src)))
                }
                // Packed integer add/sub (66 prefix): PADDB/W/D/Q, PSUBB/W/D/Q.
                0xfc | 0xfd | 0xfe | 0xd4 | 0xf8 | 0xf9 | 0xfa | 0xfb if p.opsize => {
                    let op = match op2 {
                        0xfc => Op::PackedAdd(1),
                        0xfd => Op::PackedAdd(2),
                        0xfe => Op::PackedAdd(4),
                        0xd4 => Op::PackedAdd(8),
                        0xf8 => Op::PackedSub(1),
                        0xf9 => Op::PackedSub(2),
                        0xfa => Op::PackedSub(4),
                        _ => Op::PackedSub(8), // 0xfb
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // Packed min/max (66 prefix): PMINUB(DA)/PMAXUB(DE) unsigned bytes,
                // PMINSW(EA)/PMAXSW(EE) signed words — used by SSE2 strcmp/memchr.
                0xda | 0xde | 0xea | 0xee if p.opsize => {
                    let op = match op2 {
                        0xda => Op::PackedMinMax { max: false, signed: false, elem: 1 },
                        0xde => Op::PackedMinMax { max: true, signed: false, elem: 1 },
                        0xea => Op::PackedMinMax { max: false, signed: true, elem: 2 },
                        _ => Op::PackedMinMax { max: true, signed: true, elem: 2 }, // 0xee
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // PMOVMSKB r32, xmm (66 prefix)
                0xd7 if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(
                        &c,
                        sse(Op::Pmovmskb, reg_operand(reg, Size::Dword, p.has_rex), src),
                    ))
                }
                // CVTSI2SD/SS xmm, r/m  (F2/F3)
                0x2a => {
                    let double = matches!(p.rep, Rep::F2);
                    let isize = if p.rex_w() { Size::Qword } else { Size::Dword };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, isize, c.ip, p.has_rex, p.seg);
                    Ok(finish(&c, sse(Op::Cvtsi2f(double), Operand::Xmm { id: reg }, src)))
                }
                // CVTTSD2SI/SS r, xmm/m  (F2/F3)
                0x2c => {
                    let double = matches!(p.rep, Rep::F2);
                    let isize = if p.rex_w() { Size::Qword } else { Size::Dword };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(
                        &c,
                        sse(Op::Cvttf2si(double), reg_operand(reg, isize, p.has_rex), src),
                    ))
                }
                // CMPPS/CMPPD/CMPSS/CMPSD (0F C2 /r ib): FP compare by imm8 predicate.
                //   F2 → CMPSD (scalar double), F3 → CMPSS (scalar single),
                //   66 → CMPPD (packed double), none → CMPPS (packed single).
                0xc2 => {
                    let (double, scalar) = match p.rep {
                        Rep::F2 => (true, true),
                        Rep::F3 => (false, true),
                        Rep::None => (p.opsize, false),
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let pred = c.imm(1, false)?;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::XmmCmp { double, scalar },
                            dst: Some(Operand::Xmm { id: reg }),
                            src: Some(src),
                            rel: Some(pred as i64),
                        },
                    ))
                }
                // MOVMSKPS (0F 50) / MOVMSKPD (66 0F 50): per-lane sign-bit extract.
                0x50 => {
                    let double = p.opsize;
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(
                        &c,
                        sse(
                            Op::Movmsk { double },
                            reg_operand(reg, Size::Dword, p.has_rex),
                            src,
                        ),
                    ))
                }
                // CMPXCHG8B/16B (/1); XRSTORS (/3); XSAVEC (/4); XSAVES (/5).
                0xc7 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    // /6 and /7 with mod=11 are RDRAND / RDSEED (dest = GPR `rm`),
                    // not CMPXCHG/XSAVE (those are the memory forms).
                    if reg == 6 || reg == 7 {
                        let size = if p.rex_w() {
                            Size::Qword
                        } else if p.opsize {
                            Size::Word
                        } else {
                            Size::Dword
                        };
                        let dst = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                        let op = Op::Rdrand { seed: reg == 7 };
                        return Ok(finish(
                            &c,
                            DecodedInsn { len: 0, op, dst: Some(dst), src: None, rel: None },
                        ));
                    }
                    let mem = finalize_rm(rm, Size::Qword, c.ip, p.has_rex, p.seg);
                    let (op, store): (Op, bool) = match reg {
                        1 => (Op::Cmpxchg16b { wide: p.rex_w() }, true),
                        3 => (Op::Xrstor, false),                       // XRSTORS
                        4 | 5 => (Op::Xsave { compacted: true }, true), // XSAVEC / XSAVES
                        _ => return Err(X86Error::Unsupported(format!("0f c7 /{reg}"))),
                    };
                    let (dst, src) = if store { (Some(mem), None) } else { (None, Some(mem)) };
                    Ok(finish(&c, DecodedInsn { len: 0, op, dst, src, rel: None }))
                }
                // ADDSUBPD (66 0F D0) / ADDSUBPS (F2 0F D0).
                0xd0 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::AddSub { double: p.opsize }, Operand::Xmm { id: reg }, src)))
                }
                // HADDPD/PS (66/F2 0F 7C), HSUBPD/PS (66/F2 0F 7D).
                0x7c | 0x7d => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let op = Op::Hadd { double: p.opsize, sub: op2 == 0x7d };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // SHUFPS (0F C6) / SHUFPD (66 0F C6): packed lane shuffle by imm8.
                0xc6 => {
                    let double = p.opsize;
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let imm = c.imm(1, false)?;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Shuffle { double },
                            dst: Some(Operand::Xmm { id: reg }),
                            src: Some(src),
                            rel: Some(imm as i64),
                        },
                    ))
                }
                // PINSRW xmm, r/m16, imm8 (66 0F C4) — insert a word into xmm.
                0xc4 if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, Size::Word, c.ip, p.has_rex, p.seg);
                    let imm = c.imm(1, false)?;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Pinsr { elem: 2 },
                            dst: Some(Operand::Xmm { id: reg }),
                            src: Some(src),
                            rel: Some(imm as i64),
                        },
                    ))
                }
                // PEXTRW r32, xmm, imm8 (66 0F C5) — extract a word to a GPR. NOTE the
                // operands are SWAPPED vs the 0F3A 15 form: reg = GPR dst, rm = xmm src.
                0xc5 if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let imm = c.imm(1, false)?;
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Pextr { elem: 2 },
                            dst: Some(reg_operand(reg, Size::Dword, p.has_rex)),
                            src: Some(src),
                            rel: Some(imm as i64),
                        },
                    ))
                }
                // UCOMISD/COMISD (66) and UCOMISS/COMISS (none)
                0x2e | 0x2f => {
                    let double = p.opsize;
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::Ucomis(double), Operand::Xmm { id: reg }, src)))
                }
                // SSE float arithmetic. F2=scalar double, F3=scalar single,
                // 66=packed double, none=packed single.
                0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => {
                    let (double, packed) = match p.rep {
                        Rep::F2 => (true, false),
                        Rep::F3 => (false, false),
                        Rep::None => (p.opsize, true),
                    };
                    let kind = match op2 {
                        0x58 => FBinKind::Add,
                        0x59 => FBinKind::Mul,
                        0x5c => FBinKind::Sub,
                        0x5d => FBinKind::Min,
                        0x5e => FBinKind::Div,
                        _ => FBinKind::Max,
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(
                        &c,
                        sse(Op::FBin(kind, double, packed), Operand::Xmm { id: reg }, src),
                    ))
                }
                // Packed integer multiply (66): D5/E5/E4/F4/F5.
                0xd5 | 0xe5 | 0xe4 | 0xf4 | 0xf5 if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let kind = match op2 {
                        0xd5 => PMulKind::Lw,
                        0xe5 => PMulKind::Hw,
                        0xe4 => PMulKind::Huw,
                        0xf4 => PMulKind::Uludq,
                        _ => PMulKind::Maddwd,
                    };
                    Ok(finish(&c, sse(Op::PMul(kind), Operand::Xmm { id: reg }, src)))
                }
                // PACKSSWB (0F 63), PACKUSWB (0F 67), PACKSSDW (0F 6B) — 66 prefix.
                0x63 | 0x67 | 0x6b if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let op = Op::Pack {
                        dword_src: op2 == 0x6b,
                        signed: op2 != 0x67,
                    };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // Packed double↔dword convert (0F E6): 66=CVTTPD2DQ, F2=CVTPD2DQ,
                // F3=CVTDQ2PD.
                0xe6 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let op = match p.rep {
                        Rep::F3 => Op::CvtI2F { double_dst: true }, // CVTDQ2PD
                        Rep::F2 => Op::CvtF2I { double_src: true, trunc: false }, // CVTPD2DQ
                        Rep::None if p.opsize => Op::CvtF2I { double_src: true, trunc: true }, // CVTTPD2DQ
                        _ => return Err(X86Error::Unsupported("0f e6".into())),
                    };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // Float precision convert (0F 5A): none=CVTPS2PD, 66=CVTPD2PS,
                // F3=CVTSS2SD, F2=CVTSD2SS.
                0x5a => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let op = match p.rep {
                        Rep::F3 => Op::CvtF2F { to_double: true, scalar: true }, // CVTSS2SD
                        Rep::F2 => Op::CvtF2F { to_double: false, scalar: true }, // CVTSD2SS
                        Rep::None => Op::CvtF2F { to_double: !p.opsize, scalar: false }, // PS2PD / PD2PS
                    };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // Packed single↔dword convert (0F 5B): none=CVTDQ2PS, 66=CVTPS2DQ,
                // F3=CVTTPS2DQ.
                0x5b => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let op = match p.rep {
                        Rep::None if !p.opsize => Op::CvtI2F { double_dst: false }, // CVTDQ2PS
                        Rep::None => Op::CvtF2I { double_src: false, trunc: false }, // CVTPS2DQ (66)
                        Rep::F3 => Op::CvtF2I { double_src: false, trunc: true }, // CVTTPS2DQ
                        _ => return Err(X86Error::Unsupported("0f 5b".into())),
                    };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // UNPCKLPS/HPS (none) and UNPCKLPD/HPD (66): 0F 14 = low, 0F 15 = high.
                0x14 | 0x15 => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    let op = Op::UnpckFp {
                        high: op2 == 0x15,
                        double: p.opsize,
                    };
                    Ok(finish(&c, sse(op, Operand::Xmm { id: reg }, src)))
                }
                // SQRTSD/SS (F2/F3) and SQRTPD/PS (66/none, packed).
                0x51 => {
                    let (double, packed) = match p.rep {
                        Rep::F2 => (true, false),
                        Rep::F3 => (false, false),
                        Rep::None => (p.opsize, true),
                    };
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(
                        &c,
                        sse(Op::FSqrt(double, packed), Operand::Xmm { id: reg }, src),
                    ))
                }
                // MOVQ xmm/m64, xmm (66 0F D6)
                0xd6 if p.opsize => {
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm_xmm(rm, c.ip, p.seg);
                    Ok(finish(&c, sse(Op::XmmStore(8), dst, Operand::Xmm { id: reg })))
                }
                // CMOVcc r, r/m
                0x40..=0x4f => {
                    let size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Cmov(Cond::from_tttn(op2 - 0x40)),
                            dst: Some(reg_operand(reg, size, p.has_rex)),
                            src: Some(src),
                            rel: None,
                        },
                    ))
                }
                // SETcc r/m8
                0x90..=0x9f => {
                    let (_reg, rm) = decode_modrm(&mut c, &p)?;
                    let dst = finalize_rm(rm, Size::Byte, c.ip, p.has_rex, p.seg);
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Set(Cond::from_tttn(op2 - 0x90)),
                            dst: Some(dst),
                            src: None,
                            rel: None,
                        },
                    ))
                }
                // IMUL r, r/m (two-operand)
                0xaf => {
                    let size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, size, c.ip, p.has_rex, p.seg);
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op: Op::Imul,
                            dst: Some(reg_operand(reg, size, p.has_rex)),
                            src: Some(src),
                            rel: None,
                        },
                    ))
                }
                // MOVZX / MOVSX r, r/m8 or r/m16
                0xb6 | 0xb7 | 0xbe | 0xbf => {
                    let src_size = if op2 & 1 == 0 { Size::Byte } else { Size::Word };
                    let dst_size = op_size(&p, false);
                    let (reg, rm) = decode_modrm(&mut c, &p)?;
                    let src = finalize_rm(rm, src_size, c.ip, p.has_rex, p.seg);
                    let op = if op2 < 0xbe { Op::Movzx } else { Op::Movsx };
                    Ok(finish(
                        &c,
                        DecodedInsn {
                            len: 0,
                            op,
                            dst: Some(reg_operand(reg, dst_size, p.has_rex)),
                            src: Some(src),
                            rel: None,
                        },
                    ))
                }
                other => Err(X86Error::Unsupported(format!("0x0f 0x{other:02x}"))),
            }
        }

        other => Err(X86Error::Unsupported(format!("0x{other:02x}"))),
    }
}
