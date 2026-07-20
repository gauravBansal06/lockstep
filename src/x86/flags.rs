//! RFLAGS bit layout and ALU flag computation.
//!
//! Defined flags follow the Intel SDM. Architecturally *undefined* flags are
//! pinned to 0 per `docs/cpu-determinism-contract.md` section 3 (the per-op rules
//! that clear AF/OF are applied by the executor via [`set_logic_flags`]).

use super::Size;

pub const CF: u64 = 1 << 0;
pub const PF: u64 = 1 << 2;
pub const AF: u64 = 1 << 4;
pub const ZF: u64 = 1 << 6;
pub const SF: u64 = 1 << 7;
/// Trap flag (single-step).
pub const TF: u64 = 1 << 8;
/// Interrupt-enable flag.
pub const IF: u64 = 1 << 9;
/// Direction flag — set/cleared by STD/CLD; string ops advance down when set.
pub const DF: u64 = 1 << 10;
pub const OF: u64 = 1 << 11;
/// Nested-task flag.
pub const NT: u64 = 1 << 14;

/// Even-parity of the low byte (PF is set when the count of 1 bits is even).
/// (`% 2 == 0` rather than `is_multiple_of` to keep the crate buildable on the
/// older rustc on the x86 conformance host — see cross-host determinism note.)
pub fn parity(value: u64) -> bool {
    (value as u8).count_ones() % 2 == 0
}

pub fn set_flag(rflags: &mut u64, bit: u64, on: bool) {
    if on {
        *rflags |= bit;
    } else {
        *rflags &= !bit;
    }
}

pub fn get_flag(rflags: u64, bit: u64) -> bool {
    rflags & bit != 0
}

/// Set SF/ZF/PF from a result. Shared by every arithmetic/logic op.
fn set_szp(rflags: &mut u64, result: u64, size: Size) {
    set_flag(rflags, SF, result & size.sign_bit() != 0);
    set_flag(rflags, ZF, result & size.mask() == 0);
    set_flag(rflags, PF, parity(result));
}

/// `a + b`: returns the masked result and sets CF/OF/AF/SF/ZF/PF.
pub fn add(rflags: &mut u64, a: u64, b: u64, size: Size) -> u64 {
    let mask = size.mask();
    let wide = (a & mask) as u128 + (b & mask) as u128;
    let result = (wide as u64) & mask;
    set_flag(rflags, CF, (wide >> size.bits()) & 1 != 0);
    set_flag(rflags, AF, ((a & 0xf) + (b & 0xf)) > 0xf);
    let sign = size.sign_bit();
    let overflow = ((a ^ result) & (b ^ result) & sign) != 0;
    set_flag(rflags, OF, overflow);
    set_szp(rflags, result, size);
    result
}

/// `a - b`: returns the masked result and sets CF/OF/AF/SF/ZF/PF.
pub fn sub(rflags: &mut u64, a: u64, b: u64, size: Size) -> u64 {
    let mask = size.mask();
    let a = a & mask;
    let b = b & mask;
    let result = a.wrapping_sub(b) & mask;
    set_flag(rflags, CF, a < b);
    set_flag(rflags, AF, (a & 0xf) < (b & 0xf));
    let sign = size.sign_bit();
    let overflow = ((a ^ b) & (a ^ result) & sign) != 0;
    set_flag(rflags, OF, overflow);
    set_szp(rflags, result, size);
    result
}

/// `a + b + CF` (ADC). Returns the masked result and sets CF/OF/AF/SF/ZF/PF.
pub fn adc(rflags: &mut u64, a: u64, b: u64, size: Size) -> u64 {
    let cf_in = get_flag(*rflags, CF) as u128;
    let mask = size.mask();
    let wide = (a & mask) as u128 + (b & mask) as u128 + cf_in;
    let result = (wide as u64) & mask;
    set_flag(rflags, CF, (wide >> size.bits()) & 1 != 0);
    set_flag(rflags, AF, ((a & 0xf) as u128 + (b & 0xf) as u128 + cf_in) > 0xf);
    let sign = size.sign_bit();
    set_flag(rflags, OF, ((a ^ result) & (b ^ result) & sign) != 0);
    set_szp(rflags, result, size);
    result
}

/// `a - b - CF` (SBB). Returns the masked result and sets CF/OF/AF/SF/ZF/PF.
pub fn sbb(rflags: &mut u64, a: u64, b: u64, size: Size) -> u64 {
    let cf_in = get_flag(*rflags, CF) as u128;
    let mask = size.mask();
    let a = a & mask;
    let b = b & mask;
    let result = (a as u128).wrapping_sub(b as u128).wrapping_sub(cf_in) as u64 & mask;
    set_flag(rflags, CF, (b as u128 + cf_in) > a as u128);
    set_flag(rflags, AF, ((a & 0xf) as u128) < ((b & 0xf) as u128 + cf_in));
    let sign = size.sign_bit();
    set_flag(rflags, OF, ((a ^ b) & (a ^ result) & sign) != 0);
    set_szp(rflags, result, size);
    result
}

/// Logic ops (AND/OR/XOR/TEST): CF=OF=0, AF undefined→0, SF/ZF/PF defined.
pub fn set_logic_flags(rflags: &mut u64, result: u64, size: Size) {
    set_flag(rflags, CF, false);
    set_flag(rflags, OF, false);
    set_flag(rflags, AF, false);
    set_szp(rflags, result, size);
}

/// INC: like `add 1` but CF is preserved. Sets OF/AF/SF/ZF/PF.
pub fn inc(rflags: &mut u64, a: u64, size: Size) -> u64 {
    let cf = get_flag(*rflags, CF);
    let result = add(rflags, a, 1, size);
    set_flag(rflags, CF, cf);
    result
}

/// DEC: like `sub 1` but CF is preserved. Sets OF/AF/SF/ZF/PF.
pub fn dec(rflags: &mut u64, a: u64, size: Size) -> u64 {
    let cf = get_flag(*rflags, CF);
    let result = sub(rflags, a, 1, size);
    set_flag(rflags, CF, cf);
    result
}

/// Clear the six arithmetic flags. Used where the SDM marks them all undefined
/// (DIV/IDIV) — pinned to 0 per the determinism contract.
pub fn clear_arith_undefined(rflags: &mut u64) {
    for bit in [CF, OF, SF, ZF, AF, PF] {
        set_flag(rflags, bit, false);
    }
}

/// Shift count mask: 6 bits for 64-bit operands, 5 bits otherwise.
fn count_mask(size: Size) -> u64 {
    if matches!(size, Size::Qword) {
        0x3f
    } else {
        0x1f
    }
}

fn sext(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// SHL/SAL. CF = last bit shifted out; OF defined only for count 1 (else 0,
/// undefined); AF undefined → 0.
pub fn shl(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    if count == 0 {
        return a & mask;
    }
    let bits = size.bits() as u64;
    let shifted = ((a & mask) as u128) << count;
    let result = (shifted as u64) & mask;
    let cf = (shifted >> bits) & 1 != 0;
    set_flag(rflags, CF, cf);
    let of = count == 1 && ((result & size.sign_bit() != 0) ^ cf);
    set_flag(rflags, OF, of);
    set_flag(rflags, AF, false);
    set_szp(rflags, result, size);
    result
}

/// SHR (logical). CF = last bit shifted out; OF (count 1) = original MSB.
pub fn shr(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    if count == 0 {
        return a & mask;
    }
    let masked = a & mask;
    let result = masked >> count;
    set_flag(rflags, CF, (masked >> (count - 1)) & 1 != 0);
    set_flag(rflags, OF, count == 1 && (masked & size.sign_bit() != 0));
    set_flag(rflags, AF, false);
    set_szp(rflags, result, size);
    result
}

/// SAR (arithmetic). CF = last bit shifted out; OF (count 1) = 0 always.
pub fn sar(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    if count == 0 {
        return a & mask;
    }
    let signed = sext(a & mask, size.bits());
    let result = (signed >> count.min(63)) as u64 & mask;
    set_flag(rflags, CF, (signed >> (count - 1).min(63)) & 1 != 0);
    set_flag(rflags, OF, false);
    set_flag(rflags, AF, false);
    set_szp(rflags, result, size);
    result
}

/// RCL: rotate left through CF (a `bits+1`-bit rotation). Only CF/OF affected.
pub fn rcl(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let bits = size.bits() as u128;
    let total = bits + 1;
    let count = (count_raw & count_mask(size)) as u128 % total;
    let mask = size.mask();
    if count == 0 {
        return a & mask;
    }
    let cf = get_flag(*rflags, CF) as u128;
    let wide = ((a & mask) as u128) | (cf << bits);
    let rotated = ((wide << count) | (wide >> (total - count))) & ((1 << total) - 1);
    let result = (rotated as u64) & mask;
    let new_cf = (rotated >> bits) & 1 != 0;
    set_flag(rflags, CF, new_cf);
    set_flag(rflags, OF, count == 1 && ((result & size.sign_bit() != 0) ^ new_cf));
    result
}

/// RCR: rotate right through CF. Only CF/OF affected.
pub fn rcr(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let bits = size.bits() as u128;
    let total = bits + 1;
    let count = (count_raw & count_mask(size)) as u128 % total;
    let mask = size.mask();
    if count == 0 {
        return a & mask;
    }
    let cf = get_flag(*rflags, CF) as u128;
    let wide = ((a & mask) as u128) | (cf << bits);
    let rotated = ((wide >> count) | (wide << (total - count))) & ((1 << total) - 1);
    let result = (rotated as u64) & mask;
    let new_cf = (rotated >> bits) & 1 != 0;
    // OF (count 1) = XOR of the two most-significant bits of the result (SDM).
    // The previous `MSB(result) != cf_in` was identically false: for RCR-by-1 the
    // rotated-in top bit *is* the old CF, so it never set OF.
    let of = ((result >> (size.bits() - 1)) ^ (result >> (size.bits() - 2))) & 1 != 0;
    set_flag(rflags, CF, new_cf);
    set_flag(rflags, OF, count == 1 && of);
    result
}

/// SHLD: shift `dst` left by `count`, feeding in the high bits of `src`.
pub fn shld(rflags: &mut u64, dst: u64, src: u64, count_raw: u64, size: Size) -> u64 {
    let bits = size.bits() as u64;
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    if count == 0 {
        return dst & mask;
    }
    let result = if count >= bits {
        // Beyond width is undefined; pick a deterministic value.
        (src << (count % bits)) & mask
    } else {
        (((dst & mask) << count) | ((src & mask) >> (bits - count))) & mask
    };
    let cf = (dst >> (bits - count.min(bits))) & 1 != 0;
    set_flag(rflags, CF, cf);
    set_flag(rflags, AF, false);
    set_flag(rflags, OF, count == 1 && ((dst ^ result) & size.sign_bit() != 0));
    set_szp(rflags, result, size);
    result
}

/// SHRD: shift `dst` right by `count`, feeding in the low bits of `src`.
pub fn shrd(rflags: &mut u64, dst: u64, src: u64, count_raw: u64, size: Size) -> u64 {
    let bits = size.bits() as u64;
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    if count == 0 {
        return dst & mask;
    }
    let result = if count >= bits {
        (src >> (count % bits)) & mask
    } else {
        (((dst & mask) >> count) | ((src & mask) << (bits - count))) & mask
    };
    let cf = (dst >> (count - 1)) & 1 != 0;
    set_flag(rflags, CF, cf);
    set_flag(rflags, AF, false);
    set_flag(rflags, OF, count == 1 && ((dst ^ result) & size.sign_bit() != 0));
    set_szp(rflags, result, size);
    result
}

/// ROL. Only CF and OF are affected (OF defined for count 1, else 0).
pub fn rol(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    let m = a & mask;
    if count == 0 {
        return m;
    }
    let bits = size.bits() as u64;
    let rot = count % bits;
    let result = if rot == 0 {
        m
    } else {
        ((m << rot) | (m >> (bits - rot))) & mask
    };
    let cf = result & 1 != 0;
    set_flag(rflags, CF, cf);
    set_flag(rflags, OF, count == 1 && ((result & size.sign_bit() != 0) ^ cf));
    result
}

/// ROR. Only CF and OF are affected (OF defined for count 1, else 0).
pub fn ror(rflags: &mut u64, a: u64, count_raw: u64, size: Size) -> u64 {
    let count = count_raw & count_mask(size);
    let mask = size.mask();
    let m = a & mask;
    if count == 0 {
        return m;
    }
    let bits = size.bits() as u64;
    let rot = count % bits;
    let result = if rot == 0 {
        m
    } else {
        ((m >> rot) | (m << (bits - rot))) & mask
    };
    let msb = (result >> (bits - 1)) & 1;
    set_flag(rflags, CF, msb != 0);
    let second = (result >> (bits - 2)) & 1;
    set_flag(rflags, OF, count == 1 && ((msb ^ second) != 0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rcr_overflow_flag_is_xor_of_top_two_result_bits() {
        // 8-bit RCR 0x80, 1 with CF=0 -> result 0x40; OF = bit7 ^ bit6 = 0 ^ 1 = 1.
        // The old `MSB(result) != cf_in` form gave 0 here (regression guard).
        let mut rflags = 0x2;
        set_flag(&mut rflags, CF, false);
        let r = rcr(&mut rflags, 0x80, 1, Size::Byte);
        assert_eq!(r, 0x40);
        assert!(get_flag(rflags, OF), "RCR OF must be set for 0x80>>1");
        assert!(!get_flag(rflags, CF), "old bit0 was 0 -> CF clear");

        // RCR 0x01, 1 with CF=0 -> result 0x00, CF=1 (old bit0); OF = 0 ^ 0 = 0.
        let mut rflags = 0x2;
        set_flag(&mut rflags, CF, false);
        let r = rcr(&mut rflags, 0x01, 1, Size::Byte);
        assert_eq!(r, 0x00);
        assert!(get_flag(rflags, CF));
        assert!(!get_flag(rflags, OF));
    }
}
