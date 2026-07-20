#![allow(dead_code)]
// The stack rewire onto this module lands in a follow-up; until then the arithmetic is a
// standalone, unit-tested library (hence some items are not yet referenced from exec.rs).
//! Soft-float **80-bit extended precision** (x87 `long double`) — deterministic, host-
//! independent (integer arithmetic only, NEVER host f64 rounding), so identical on any host.
//!
//! The x87 register stack holds 80-bit values (1 sign + 15 exponent + 64-bit significand with
//! an EXPLICIT integer bit). We model each as a decomposed `F80 { sign, exp, mant, class }`:
//!   value = (-1)^sign · mant · 2^(exp − 63),  with mant bit 63 = 1 for a normal.
//! `exp` is the unbiased exponent. This module is standalone + unit-tested; the exec stack is
//! rewired onto it once validated. Rounding honours the FCW rounding-control (`rc`, bits 10:11)
//! and precision-control (`pc`, bits 8:9: 0=24-bit, 2=53-bit, 3=64-bit/default).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F80Class {
    Zero,
    Normal,
    Inf,
    Nan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F80 {
    pub sign: bool,
    pub exp: i32,   // unbiased; value = mant · 2^(exp-63)
    pub mant: u64,  // bit 63 = integer bit (set for normals)
    pub class: F80Class,
}

const EXP_MAX: i32 = 16384; // unbiased inf/nan exponent (biased 0x7fff)

impl F80 {
    pub const ZERO: F80 = F80 { sign: false, exp: 0, mant: 0, class: F80Class::Zero };

    pub fn zero(sign: bool) -> F80 {
        F80 { sign, exp: 0, mant: 0, class: F80Class::Zero }
    }
    pub fn inf(sign: bool) -> F80 {
        F80 { sign, exp: EXP_MAX, mant: 1 << 63, class: F80Class::Inf }
    }
    /// x87 QNaN floating-point indefinite.
    pub fn nan() -> F80 {
        F80 { sign: true, exp: EXP_MAX, mant: 0xC000_0000_0000_0000, class: F80Class::Nan }
    }
    pub fn is_nan(&self) -> bool {
        self.class == F80Class::Nan
    }

    // ---- conversions to/from f64 (host IEEE double) ----

    /// Exact widening f64 → F80 (every f64 is representable in 80-bit).
    pub fn from_f64(x: f64) -> F80 {
        let bits = x.to_bits();
        let sign = bits >> 63 != 0;
        let bexp = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & 0x000f_ffff_ffff_ffff;
        if bexp == 0x7ff {
            return if frac == 0 { F80::inf(sign) } else { F80::nan() };
        }
        if bexp == 0 {
            if frac == 0 {
                return F80::zero(sign);
            }
            // subnormal f64 = frac·2^-1074. Normalize: shift the top set bit to position 63.
            // value = mant·2^(exp-63), mant = frac<<s ⇒ exp = -1011 - s (s = leading_zeros(frac)).
            let s = frac.leading_zeros();
            return F80 { sign, exp: -1011 - s as i32, mant: frac << s, class: F80Class::Normal };
        }
        // normal f64: 53-bit significand (implicit 1) → 64-bit (explicit 1). `exp` is the
        // true unbiased exponent (value = (1.frac)·2^exp = mant·2^(exp-63)).
        let mant = ((1u64 << 52) | frac) << 11; // integer bit now at 63
        F80 { sign, exp: bexp - 1023, mant, class: F80Class::Normal }
    }

    /// F80 → f64, rounding the 64-bit significand to 53 bits (round-to-nearest-even) with
    /// correct over/underflow. (Directed f64 rounding is a follow-up; the stack round-to-f64
    /// on store uses this.)
    pub fn to_f64(&self) -> f64 {
        match self.class {
            F80Class::Zero => if self.sign { -0.0 } else { 0.0 },
            F80Class::Inf => if self.sign { f64::NEG_INFINITY } else { f64::INFINITY },
            F80Class::Nan => f64::from_bits(0xFFF8_0000_0000_0000),
            F80Class::Normal => {
                let e = self.exp; // value = (1.frac)·2^e (true unbiased exponent)
                if e > 1023 {
                    return if self.sign { f64::NEG_INFINITY } else { f64::INFINITY };
                }
                if e < -1074 {
                    return if self.sign { -0.0 } else { 0.0 };
                }
                // round mant (64-bit, integer bit at 63) to 53 bits.
                if e >= -1022 {
                    // normal f64: keep top 53 bits (implicit 1 + 52 frac), round bit 10, sticky below.
                    let frac53 = self.mant >> 11; // 53 bits incl integer bit
                    let round_bit = (self.mant >> 10) & 1;
                    let sticky = (self.mant & 0x3ff) != 0;
                    let mut m = frac53 & 0x000f_ffff_ffff_ffff; // drop integer bit → 52 frac
                    let mut be = (e + 1023) as u64;
                    // round-to-nearest-even
                    if round_bit == 1 && (sticky || (m & 1) == 1) {
                        m += 1;
                        if m >> 52 != 0 {
                            m = 0;
                            be += 1;
                        }
                    }
                    if be >= 0x7ff {
                        return if self.sign { f64::NEG_INFINITY } else { f64::INFINITY };
                    }
                    f64::from_bits(((self.sign as u64) << 63) | (be << 52) | m)
                } else {
                    // subnormal f64: shift right so the exponent becomes -1022.
                    let shift = (-1022 - e) as u32; // extra right-shift beyond the 11
                    let total = 11 + shift;
                    if total >= 64 {
                        return if self.sign { -0.0 } else { 0.0 };
                    }
                    let m_full = self.mant;
                    let frac = m_full >> total;
                    let round_bit = (m_full >> (total - 1)) & 1;
                    let sticky = (m_full & ((1u64 << (total - 1)) - 1)) != 0;
                    let mut m = frac;
                    if round_bit == 1 && (sticky || (m & 1) == 1) {
                        m += 1;
                    }
                    f64::from_bits(((self.sign as u64) << 63) | m)
                }
            }
        }
    }
}

impl F80 {
    // ---- 80-bit memory format (FLD/FSTP m80): (mantissa u64, sign_exp u16) ----
    pub fn to_m80(&self) -> (u64, u16) {
        match self.class {
            F80Class::Zero => (0, (self.sign as u16) << 15),
            F80Class::Inf => (1 << 63, ((self.sign as u16) << 15) | 0x7fff),
            F80Class::Nan => (self.mant, 0xffff),
            F80Class::Normal => {
                let biased = self.exp + 16383;
                if biased >= 0x7fff {
                    return (1 << 63, ((self.sign as u16) << 15) | 0x7fff); // overflow → inf
                }
                if biased <= 0 {
                    // 80-bit subnormal: denormalize (integer bit clears), exp field 0. The internal
                    // value was already rounded to the subnormal grid by round128, so this right
                    // shift is exact (it just reverses the normalization: mant = m << s, sh = s).
                    let sh = (1 - biased) as u32; // ≥ 1
                    if sh >= 64 {
                        return (0, (self.sign as u16) << 15); // below smallest subnormal → 0
                    }
                    return (self.mant >> sh, (self.sign as u16) << 15);
                }
                (self.mant, ((self.sign as u16) << 15) | biased as u16)
            }
        }
    }
    pub fn from_m80(mant: u64, sign_exp: u16) -> F80 {
        let sign = sign_exp >> 15 != 0;
        let biased = (sign_exp & 0x7fff) as i32;
        if biased == 0x7fff {
            // exp all ones: inf (mant = integer bit only) or NaN.
            return if mant == (1 << 63) { F80::inf(sign) } else { F80::nan() };
        }
        if biased == 0 {
            if mant == 0 {
                return F80::zero(sign);
            }
            // 80-bit subnormal (integer bit 0): normalize.
            let s = mant.leading_zeros();
            return F80 { sign, exp: -16382 - s as i32, mant: mant << s, class: F80Class::Normal };
        }
        if mant >> 63 == 0 {
            // pseudo-denormal / unnormal (integer bit clear with nonzero exp) — treat value as-is.
            if mant == 0 {
                return F80::zero(sign);
            }
            let s = mant.leading_zeros();
            return F80 { sign, exp: (biased - 16383) - s as i32, mant: mant << s, class: F80Class::Normal };
        }
        F80 { sign, exp: biased - 16383, mant, class: F80Class::Normal }
    }

    /// FSCALE: self · 2^trunc(n). Exact in 80-bit (an exponent shift), except at the over/underflow
    /// boundary where round128 handles subnormal rounding per `rc`.
    pub fn scale_by(self, n: F80, rc: u8) -> F80 {
        use F80Class::*;
        match self.class {
            Nan => return F80::nan(),
            Inf | Zero => return self, // inf·2^k=inf, 0·2^k=0 (n finite)
            Normal => {}
        }
        match n.class {
            Nan => return F80::nan(),
            Inf => return if n.sign { F80::zero(self.sign) } else { F80::inf(self.sign) }, // 2^±inf
            Zero => return self,
            Normal => {}
        }
        // trunc(n) toward zero. |n| ≥ 2^15 ⇒ the shift is beyond the exponent range → over/underflow.
        let new_exp: i64 = if n.exp >= 15 {
            if n.sign { -100_000 } else { 100_000 }
        } else {
            self.exp as i64 + n.to_i64(3, true)
        };
        if new_exp > 16383 {
            return F80::inf(self.sign);
        }
        if new_exp < -16382 - 64 {
            return F80::zero(self.sign);
        }
        round128(self.sign, new_exp as i32, (self.mant as u128) << 64, 64, rc)
    }

    /// Build a normalized F80 from a magnitude `rem·2^(scale_exp−63)` (rem a u64 mantissa at some
    /// scale). rem == 0 → signed zero. Exact (no rounding); internal wide-exponent rep holds
    /// subnormals losslessly until `to_m80`.
    fn from_mag(sign: bool, rem: u64, scale_exp: i32) -> F80 {
        if rem == 0 {
            return F80::zero(sign);
        }
        let s = rem.leading_zeros();
        F80 { sign, exp: scale_exp - s as i32, mant: rem << s, class: F80Class::Normal }
    }

    /// FPREM (`ieee=false`, quotient truncated) / FPREM1 (`ieee=true`, round-to-nearest-even).
    /// Returns `(remainder, low-3 quotient bits, incomplete)`. `incomplete` (→ C2) is true only
    /// for an exponent gap ≥ 64, where x87 reduces partially; there the partial quotient is
    /// implementation-defined (SDM), so we pick a deterministic 32-bit chunk. For gap < 64 the
    /// remainder is the EXACT 80-bit result (C2 = 0).
    pub fn fprem(self, div: F80, ieee: bool) -> (F80, u8, bool) {
        use F80Class::*;
        match (self.class, div.class) {
            (Nan, _) | (_, Nan) => return (F80::nan(), 0, false),
            (Inf, _) | (_, Zero) => return (F80::nan(), 0, false), // ∞ mod y, x mod 0 → invalid
            (Zero, _) | (_, Inf) => return (self, 0, false),       // 0 mod y = 0, x mod ∞ = x
            (Normal, Normal) => {}
        }
        let (m0, m1) = (self.mant, div.mant);
        let d = self.exp - div.exp; // exponent gap
        // |x| < |y| (gap < 0, or equal exp with smaller mantissa) → quotient 0, remainder x …
        if d < 0 || (d == 0 && m0 < m1) {
            let mut rem = self; // magnitude = |x|
            let mut q = 0u64;
            // FPREM1 rounds when 2|x| > |y|, only possible for gap d ≥ −1 (else |x| < |y|/2).
            if ieee && d >= -1 {
                let two_x = (m0 as u128) << 1; // 2·m0 at scale e0
                let y_at_e0 = (m1 as u128) << (-d) as u32; // |y| mantissa scaled to e0 (shift ≤ 1)
                if two_x > y_at_e0 {
                    // remainder becomes x − y (opposite sign to x, magnitude |y|−|x|)
                    rem = div.sub_mag(self);
                    rem.sign = !self.sign;
                    q = 1;
                }
            }
            return (rem, (q & 7) as u8, false);
        }
        if d >= 64 {
            // Partial reduction (implementation-defined): peel a 32-bit chunk of the quotient.
            let sh = (d - 32) as u32;
            // q32 = trunc(|x| / (|y|·2^sh)) as a ≤33-bit number; num = m0·2^(d−sh)=m0·2^32.
            let num = (m0 as u128) << 32;
            let q32 = (num / m1 as u128) as u64;
            // remainder = |x| − |y|·2^sh·q32, exact: (m0·2^32 − q32·m1)·2^(e1+sh−63)
            let rem_int = (num - (q32 as u128) * (m1 as u128)) as u64; // < m1 < 2^64
            let rem = F80::from_mag(self.sign, rem_int, div.exp + sh as i32);
            return (rem, (q32 & 7) as u8, true);
        }
        // Complete remainder, gap < 64: q = trunc(|x|/|y|) fits in ≤ d+1 bits.
        // num = m0·2^d (< 2^127), q = num / m1 (≤ 2^64), rem_int = num % m1 (< 2^64).
        let num = (m0 as u128) << d as u32;
        let mut q = (num / m1 as u128) as u64;
        let rem_int = (num % m1 as u128) as u64;
        // remainder magnitude = rem_int · 2^(e1−63); build normalized F80.
        let mut rem = F80::from_mag(self.sign, rem_int, div.exp);
        if ieee {
            // round-to-nearest-even: reduce if 2·rem_int > m1, or (== and q odd).
            let two = (rem_int as u128) << 1;
            let m1u = m1 as u128;
            if two > m1u || (two == m1u && (q & 1) == 1) {
                // remainder ← rem − |y| (magnitude m1 − rem_int, opposite sign to x)
                rem = F80::from_mag(!self.sign, m1 - rem_int, div.exp);
                q = q.wrapping_add(1);
            }
        }
        (rem, (q & 7) as u8, false)
    }

    /// |self| − |other| as a magnitude with self's sign, for two normals with |self| ≥ |other|.
    /// Used only for the FPREM1 |x|<|y| reduction (|y|−|x|). Exact.
    fn sub_mag(self, other: F80) -> F80 {
        let a = F80 { sign: false, ..self };
        let b = F80 { sign: true, ..other };
        a.add(b, 3, 0) // exact subtraction of magnitudes (pc=64, rc=nearest; result is exact here)
    }

    // ---- integer conversions (FILD / FIST) ----
    pub fn from_i64(v: i64) -> F80 {
        if v == 0 {
            return F80::zero(false);
        }
        let sign = v < 0;
        let mag = (v as i128).unsigned_abs() as u64;
        let s = mag.leading_zeros();
        F80 { sign, exp: 63 - s as i32, mant: mag << s, class: F80Class::Normal }
    }
    /// Round to a 64-bit integer per `rc` (FIST/FISTP; `trunc` forces round-toward-zero for FISTTP).
    pub fn to_i64(&self, rc: u8, trunc: bool) -> i64 {
        match self.class {
            F80Class::Zero => 0,
            F80Class::Nan | F80Class::Inf => i64::MIN, // x87 integer indefinite
            F80Class::Normal => {
                let e = self.exp; // value = (1.frac)·2^e
                if e < 0 {
                    // |value| < 1 → integer part 0 (a nonzero Normal); may round to 0 or ±1.
                    let up = match if trunc { 3 } else { rc } {
                        1 => self.sign,  // toward −inf: negative → −1
                        2 => !self.sign, // toward +inf: positive → +1
                        3 => false,      // toward zero
                        // nearest: |value| ≥ 0.5 only at e == −1; > 0.5 ⇔ mant > 2^63
                        // (== 0.5 ⇔ mant == 2^63 → ties to even → 0).
                        _ => e == -1 && self.mant > (1 << 63),
                    };
                    let m = if up { 1 } else { 0 };
                    return if self.sign { -m } else { m };
                }
                if e >= 63 {
                    return i64::MIN; // out of range → indefinite
                }
                let int = self.mant >> (63 - e); // integer part
                let shift = 63 - e;
                let frac_mask = (1u64 << shift) - 1;
                let frac = self.mant & frac_mask;
                let half = 1u64 << (shift - 1);
                let up = match if trunc { 3 } else { rc } {
                    1 => self.sign && frac != 0,
                    2 => !self.sign && frac != 0,
                    3 => false,
                    _ => frac > half || (frac == half && (int & 1) == 1),
                };
                let mag = int + up as u64;
                if self.sign { -(mag as i64) } else { mag as i64 }
            }
        }
    }
}

/// Precision-control (FCW bits 8:9) → significand bit count.
fn pc_bits(pc: u8) -> u32 {
    match pc {
        0 => 24, // single
        2 => 53, // double
        _ => 64, // 64 = extended (default); 1 is reserved → treat as 64
    }
}

/// Round an exact significand `mant` (u128, integer bit at 127 for a normalized nonzero
/// value: value = ±mant·2^(exp−127)) to `prec` bits with FCW rounding-control `rc`, producing
/// a normalized F80. Overflow → ±inf. (80-bit subnormal output is a follow-up — results below
/// the normal range are flushed to signed zero here.)
fn round128(sign: bool, mut exp: i32, mant: u128, prec: u32, rc: u8) -> F80 {
    if mant == 0 {
        return F80::zero(sign);
    }
    let drop = 128 - prec;
    let kept = mant >> drop; // prec-bit significand, integer bit at prec−1
    let round_bit = (mant >> (drop - 1)) & 1;
    let sticky = (mant & ((1u128 << (drop - 1)) - 1)) != 0;
    let mut sig = kept;
    let up = match rc {
        1 => sign && (round_bit == 1 || sticky),  // toward −inf
        2 => !sign && (round_bit == 1 || sticky), // toward +inf
        3 => false,                               // toward zero (truncate)
        _ => round_bit == 1 && (sticky || (sig & 1) == 1), // nearest-even
    };
    if up {
        sig += 1;
        if sig >> prec != 0 {
            sig >>= 1;
            exp += 1;
        }
    }
    if exp > 16383 {
        return F80::inf(sign);
    }
    if exp < -16382 {
        // 80-bit subnormal: round the exact value to the subnormal grid (ULP = 2^-16445, the
        // 64-bit-mantissa memory grid — the m80 format has no precision-control, so PC<64
        // subnormals still land on this grid). m = round(mant · 2^(exp+16318)) = round(mant >> rsh).
        let rsh = (-exp - 16318) as u32; // exp ≤ -16383 ⇒ rsh ≥ 65
        let m = round_shift_u64(mant, rsh, sign, rc);
        if m == 0 {
            return F80::zero(sign);
        }
        // Normalize (integer bit → 63). m == 2^63 lands here too → the smallest NORMAL, exp -16382.
        let s = m.leading_zeros();
        return F80 { sign, exp: -16382 - s as i32, mant: m << s, class: F80Class::Normal };
    }
    F80 { sign, exp, mant: (sig as u64) << (64 - prec), class: F80Class::Normal }
}

/// Round `mant·2^-rsh` (mant u128 with the integer bit at 127) to an integer per RC, for 80-bit
/// subnormal formation. Result ≤ 2^63 (2^63 = rounded up to the smallest normal).
fn round_shift_u64(mant: u128, rsh: u32, sign: bool, rc: u8) -> u64 {
    let (kept, round_bit, sticky) = if rsh >= 128 {
        let rb = if rsh == 128 { (mant >> 127) & 1 } else { 0 };
        let st = if rsh == 128 { mant & ((1u128 << 127) - 1) != 0 } else { mant != 0 };
        (0u128, rb, st)
    } else {
        (mant >> rsh, (mant >> (rsh - 1)) & 1, mant & ((1u128 << (rsh - 1)) - 1) != 0)
    };
    let up = match rc {
        1 => sign && (round_bit == 1 || sticky),  // toward −inf
        2 => !sign && (round_bit == 1 || sticky), // toward +inf
        3 => false,                               // toward zero
        _ => round_bit == 1 && (sticky || (kept & 1) == 1), // nearest-even
    };
    (kept + up as u128) as u64
}

impl F80 {
    fn is_special(&self) -> bool {
        !matches!(self.class, F80Class::Normal)
    }
    pub fn neg(self) -> F80 {
        F80 { sign: !self.sign, ..self }
    }

    /// 80-bit multiply with FCW precision/rounding control.
    pub fn mul(self, b: F80, pc: u8, rc: u8) -> F80 {
        use F80Class::*;
        match (self.class, b.class) {
            (Nan, _) | (_, Nan) => return F80::nan(),
            (Inf, Zero) | (Zero, Inf) => return F80::nan(), // 0·∞
            (Inf, _) | (_, Inf) => return F80::inf(self.sign ^ b.sign),
            (Zero, _) | (_, Zero) => return F80::zero(self.sign ^ b.sign),
            (Normal, Normal) => {}
        }
        let sign = self.sign ^ b.sign;
        // 64×64 → 128-bit product; both operands have integer bit at 63, so the product's
        // integer bit is at 126 or 127. value = product·2^(a.exp+b.exp − 126).
        let prod = (self.mant as u128) * (b.mant as u128);
        let (mant, exp) = if prod >> 127 != 0 {
            (prod, self.exp + b.exp + 1)
        } else {
            (prod << 1, self.exp + b.exp)
        };
        round128(sign, exp, mant, pc_bits(pc), rc)
    }

    /// 80-bit add with FCW precision/rounding control (subtraction = `a.add(b.neg())`).
    pub fn add(self, b: F80, pc: u8, rc: u8) -> F80 {
        use F80Class::*;
        match (self.class, b.class) {
            (Nan, _) | (_, Nan) => return F80::nan(),
            (Inf, Inf) => {
                return if self.sign == b.sign { F80::inf(self.sign) } else { F80::nan() };
            }
            (Inf, _) => return self,
            (_, Inf) => return b,
            (Zero, Zero) => {
                // −0 + −0 = −0; else +0 (except round-toward−inf → −0).
                let s = (self.sign && b.sign) || rc == 1;
                return F80::zero(s);
            }
            (Zero, _) => return b,
            (_, Zero) => return self,
            (Normal, Normal) => {}
        }
        // Order so |a| ≥ |b| by exponent; equal exponents keep order.
        let (a, b) = if (self.exp, self.mant) >= (b.exp, b.mant) { (self, b) } else { (b, self) };
        let shift = (a.exp - b.exp) as u32;
        let am = (a.mant as u128) << 64; // integer bit at 127
        // shift b right by `shift`, capturing sticky into bit 0
        let bm_full = (b.mant as u128) << 64;
        let (bm, bsticky) = if shift >= 128 {
            (0u128, bm_full != 0)
        } else if shift == 0 {
            (bm_full, false)
        } else {
            (bm_full >> shift, (bm_full & ((1u128 << shift) - 1)) != 0)
        };
        let mut exp = a.exp;
        let sign = a.sign;
        let mant;
        if a.sign == b.sign {
            // Magnitudes add. Sum can reach bit 128 → shift right 1, bump exponent.
            let (m, carry) = am.overflowing_add(bm);
            if carry {
                let s_out = (m & 1) != 0;
                mant = ((m >> 1) | (1u128 << 127)) | ((bsticky || s_out) as u128);
                exp += 1;
            } else {
                mant = m | (bsticky as u128);
            }
        } else {
            // Magnitudes subtract (|a| ≥ |b|). Bits are kept in the 128-bit field: for
            // exponent-diff ≤ 64 no bits are lost (exact); beyond that a truncated tail
            // becomes a sticky, and the borrow (−1 when sticky) correctly places the exact
            // value in (am−bm−1, am−bm) with the residual (1−frac) as a low sticky.
            let mut m = am - bm - (bsticky as u128);
            if m == 0 {
                return F80::zero(rc == 1); // exact cancellation → +0 (−0 under round-toward−inf)
            }
            let lz = m.leading_zeros();
            m <<= lz;
            exp -= lz as i32;
            mant = m | (bsticky as u128);
        }
        round128(sign, exp, mant, pc_bits(pc), rc)
    }

    pub fn sub(self, b: F80, pc: u8, rc: u8) -> F80 {
        self.add(b.neg(), pc, rc)
    }

    /// 80-bit divide with FCW precision/rounding control.
    pub fn div(self, b: F80, pc: u8, rc: u8) -> F80 {
        use F80Class::*;
        match (self.class, b.class) {
            (Nan, _) | (_, Nan) => return F80::nan(),
            (Inf, Inf) | (Zero, Zero) => return F80::nan(),
            (Inf, _) => return F80::inf(self.sign ^ b.sign),
            (_, Inf) => return F80::zero(self.sign ^ b.sign),
            (_, Zero) => return F80::inf(self.sign ^ b.sign), // finite/0 = ±inf (#Z on hw)
            (Zero, _) => return F80::zero(self.sign ^ b.sign),
            (Normal, Normal) => {}
        }
        let sign = self.sign ^ b.sign;
        let bm = b.mant as u128;
        // q = ⌊(a.mant·2^64)/b.mant⌋ ∈ (2^63, 2^65); the NEXT quotient bit (round bit) is
        // ⌊2·rem/b.mant⌋ and everything below it is the sticky. Pack q + round + sticky with 2
        // spare low bits so round128 rounds correctly regardless of q's bit-length.
        let num = (self.mant as u128) << 64;
        let q = num / bm;
        let rem = num % bm;
        let r2 = rem << 1;
        let round_bit = (r2 >= bm) as u128;
        let sticky = (if round_bit == 1 { r2 != bm } else { rem != 0 }) as u128;
        let mant_raw = (q << 2) | (round_bit << 1) | sticky;
        let lz = mant_raw.leading_zeros();
        let exp = self.exp - b.exp + 61 - lz as i32;
        round128(sign, exp, mant_raw << lz, pc_bits(pc), rc)
    }

    /// 80-bit square root with FCW precision/rounding control (RC direction only affects the
    /// last bit). Exact 64-bit floor sqrt + a round/sticky derived from the remainder.
    pub fn sqrt(self, pc: u8, rc: u8) -> F80 {
        use F80Class::*;
        match self.class {
            Nan => return F80::nan(),
            Zero => return F80::zero(self.sign), // sqrt(±0) = ±0
            Inf => return if self.sign { F80::nan() } else { F80::inf(false) },
            Normal => {}
        }
        if self.sign {
            return F80::nan(); // sqrt of a negative
        }
        // value = mant·2^(exp−63) = mant·2^p. Make the radicand's power even for a clean split.
        let p = self.exp - 63;
        let (radm, hexp, k): (u128, i32, u32) = if p & 1 == 0 {
            (self.mant as u128, p / 2, 64)
        } else {
            ((self.mant as u128) << 1, (p - 1) / 2, 62)
        };
        let x = radm << k; // ∈ [2^126, 2^128)
        let r = isqrt_u128(x); // 64-bit floor sqrt, integer bit at 63
        let rem = x - r * r;
        // round-to-nearest boundary at 64 bits: fractional ≥ 0.5 ⇔ rem > r (never an exact tie).
        let round_up = if rem > r { 1u128 << 63 } else { 0 };
        let mant = (r << 64) | round_up | ((rem != 0) as u128);
        let exp = hexp - (k / 2) as i32 + 63;
        round128(self.sign, exp, mant, pc_bits(pc), rc)
    }
}

/// floor(√n) for a u128 (result < 2^64). Integer Newton iteration.
fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = 1u128 << ((130 - n.leading_zeros()) / 2);
    loop {
        let x1 = (x + n / x) >> 1;
        if x1 >= x {
            return x;
        }
        x = x1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(x: f64) {
        assert_eq!(F80::from_f64(x).to_f64().to_bits(), x.to_bits(), "roundtrip {x:e}");
    }

    #[test]
    fn f80_f64_roundtrip() {
        for &x in &[
            0.0, -0.0, 1.0, -1.0, 2.5, -3.75, 0.1, 1e308, -1e-308, 123456.789,
            f64::MIN_POSITIVE, f64::MAX, std::f64::consts::PI,
        ] {
            roundtrip(x);
        }
        // subnormal f64 round-trips exactly through 80-bit.
        roundtrip(f64::from_bits(1)); // smallest subnormal
        roundtrip(f64::from_bits(0x000f_ffff_ffff_ffff)); // largest subnormal
        // specials
        assert!(F80::from_f64(f64::NAN).to_f64().is_nan());
        assert_eq!(F80::from_f64(f64::INFINITY).to_f64(), f64::INFINITY);
        assert_eq!(F80::from_f64(f64::NEG_INFINITY).to_f64(), f64::NEG_INFINITY);
    }

    #[test]
    fn f80_arith_basic() {
        let f = |x: f64| F80::from_f64(x);
        // exact results round-trip through f64
        assert_eq!(f(3.0).mul(f(7.0), 3, 0).to_f64(), 21.0);
        assert_eq!(f(0.5).mul(f(0.5), 3, 0).to_f64(), 0.25);
        assert_eq!(f(1.5).add(f(2.5), 3, 0).to_f64(), 4.0);
        assert_eq!(f(1.0).add(f(1.0), 3, 0).to_f64(), 2.0);
        assert_eq!(f(-3.0).mul(f(2.0), 3, 0).to_f64(), -6.0);
        // specials
        assert!(f(1.0).mul(f(f64::NAN), 3, 0).is_nan());
        assert!(f(0.0).mul(f(f64::INFINITY), 3, 0).is_nan());
        assert_eq!(f(f64::INFINITY).add(f(1.0), 3, 0).to_f64(), f64::INFINITY);
    }

    #[test]
    fn f80_div_sqrt_sub() {
        let f = |x: f64| F80::from_f64(x);
        // divide
        assert_eq!(f(1.0).div(f(2.0), 3, 0).to_f64(), 0.5);
        assert_eq!(f(21.0).div(f(7.0), 3, 0).to_f64(), 3.0);
        assert_eq!(f(1.0).div(f(4.0), 3, 0).to_f64(), 0.25);
        assert_eq!(f(-6.0).div(f(2.0), 3, 0).to_f64(), -3.0);
        assert_eq!(f(1.0).div(f(0.0), 3, 0).class, F80Class::Inf);
        // divide at PC=53 matches f64
        assert_eq!(f(1.0).div(f(3.0), 2, 0).to_f64(), 1.0f64 / 3.0);
        assert_eq!(f(2.0).div(f(7.0), 2, 0).to_f64(), 2.0f64 / 7.0);
        // sqrt exact
        assert_eq!(f(4.0).sqrt(3, 0).to_f64(), 2.0);
        assert_eq!(f(9.0).sqrt(3, 0).to_f64(), 3.0);
        assert_eq!(f(0.25).sqrt(3, 0).to_f64(), 0.5);
        assert_eq!(f(1.0).sqrt(3, 0).to_f64(), 1.0);
        assert!(f(-1.0).sqrt(3, 0).is_nan());
        // sqrt at PC=53 == correctly-rounded f64 sqrt
        assert_eq!(f(2.0).sqrt(2, 0).to_f64(), 2.0f64.sqrt());
        assert_eq!(f(3.0).sqrt(2, 0).to_f64(), 3.0f64.sqrt());
        assert_eq!(f(1.5).sqrt(2, 0).to_f64(), 1.5f64.sqrt());
        assert_eq!(f(1e10).sqrt(2, 0).to_f64(), 1e10f64.sqrt());
        // subtract incl. deep cancellation (exact in 80-bit)
        assert_eq!(f(3.0).sub(f(1.0), 3, 0).to_f64(), 2.0);
        assert_eq!(f(1.0).sub(f(0.5), 3, 0).to_f64(), 0.5);
        let c = f(1.0).add(f(2f64.powi(-60)), 3, 0).sub(f(1.0), 3, 0); // (1+2^-60)−1
        assert_eq!(c.to_f64(), 2f64.powi(-60));
    }

    #[test]
    fn f80_subnormals() {
        // 3d-2: m80 subnormal round-trip (biased exp field 0, integer bit clear).
        for &m in &[1u64, 0x40, 0x1234_5678_9abc_def0, 0x7fff_ffff_ffff_ffff] {
            assert_eq!(F80::from_m80(m, 0).to_m80(), (m, 0), "subnormal m80 round-trip {m:#x}");
        }
        let half = F80::from_m80(1 << 63, 16382); // biased 16382 ⇒ exp -1 (= 0.5)
        // smallest normal (2^-16382) × 0.5 = 2^-16383 → a subnormal (mant 2^62, exp field 0).
        let min_normal = F80::from_m80(1 << 63, 1); // biased 1 ⇒ exp -16382
        assert_eq!(min_normal.mul(half, 3, 0).to_m80(), (1 << 62, 0), "underflow into subnormal grid");
        // smallest subnormal (2^-16445) × 0.5 = 2^-16446 = exactly ½ ULP → nearest-even → 0,
        // but toward +inf → rounds UP to the smallest subnormal.
        let smallest = F80::from_m80(1, 0);
        assert_eq!(smallest.mul(half, 3, 0).class, F80Class::Zero);
        assert_eq!(smallest.mul(half, 3, 2).to_m80(), (1, 0));
        // subnormal + subnormal stays subnormal (exact).
        assert_eq!(F80::from_m80(3, 0).add(F80::from_m80(5, 0), 3, 0).to_m80(), (8, 0));
        // a subnormal rounding UP across the boundary to the smallest normal.
        let big_sub = F80::from_m80(0x7fff_ffff_ffff_ffff, 0); // largest subnormal
        let up = big_sub.add(smallest, 3, 0); // +1 ULP → 2^63 = smallest normal
        assert_eq!(up.to_m80(), (1 << 63, 1));
    }

    #[test]
    fn f80_scale_and_fprem() {
        let f = |x: f64| F80::from_f64(x);
        // 3d-3 FSCALE: x·2^trunc(n).
        assert_eq!(f(3.0).scale_by(f(4.0), 0).to_f64(), 48.0);
        assert_eq!(f(1.0).scale_by(f(-2.0), 0).to_f64(), 0.25);
        assert_eq!(f(1.0).scale_by(f(2.9), 0).to_f64(), 4.0); // trunc(2.9)=2
        assert_eq!(f(5.0).scale_by(f(0.5), 0).to_f64(), 5.0); // trunc(0.5)=0 → unchanged
        assert_eq!(f(5.0).scale_by(f(-0.5), 0).to_f64(), 5.0);
        assert!(f(f64::INFINITY).scale_by(f(-5.0), 0).to_f64().is_infinite());
        // 3d-3 FPREM (trunc quotient): remainder carries the dividend's sign, |r| < |divisor|.
        let (r, _q, inc) = f(17.0).fprem(f(5.0), false);
        assert_eq!(r.to_f64(), 2.0);
        assert!(!inc);
        assert_eq!(f(-17.0).fprem(f(5.0), false).0.to_f64(), -2.0);
        assert_eq!(f(1.0).fprem(f(3.0), false).0.to_f64(), 1.0); // |x|<|y| → x
        assert_eq!(f(15.0).fprem(f(5.0), false).0.to_f64(), 0.0); // exact multiple → 0
        // 3d-3 FPREM1 (round-nearest): remainder in [−|y|/2, |y|/2].
        assert_eq!(f(17.0).fprem(f(5.0), true).0.to_f64(), 2.0);
        assert_eq!(f(18.0).fprem(f(5.0), true).0.to_f64(), -2.0); // 18 − 4·5
        assert_eq!(f(1.0).fprem(f(3.0), true).0.to_f64(), 1.0); // 1 < 3/2 → stays
        assert_eq!(f(2.0).fprem(f(3.0), true).0.to_f64(), -1.0); // 2 > 3/2 → 2−3
    }

    #[test]
    fn f80_constants() {
        // 80-bit constant encodings → nearest f64.
        assert_eq!(F80::from_m80(0xc90f_daa2_2168_c235, 0x4000).to_f64(), std::f64::consts::PI);
        assert_eq!(F80::from_m80(0xb172_17f7_d1cf_79ac, 0x3ffe).to_f64(), std::f64::consts::LN_2);
        assert_eq!(F80::from_m80(0x8000_0000_0000_0000, 0x3fff).to_f64(), 1.0);
        // the 80-bit π carries bits below f64's 53-bit significand (the whole point).
        assert_ne!(F80::from_m80(0xc90f_daa2_2168_c235, 0x4000).mant & 0x7ff, 0);
    }

    #[test]
    fn f80_extended_precision() {
        // 1.0 + 2^-60 is EXACT in 80-bit (64-bit significand) but rounds to 1.0 in f64.
        let sum = F80::from_f64(1.0).add(F80::from_f64(2f64.powi(-60)), 3, 0); // pc=64, rc=nearest
        assert_eq!(sum.exp, 0);
        assert_eq!(sum.mant, 0x8000_0000_0000_0008); // integer bit + bit 3 (=2^-60 relative)
        assert_eq!(1.0f64 + 2f64.powi(-60), 1.0); // f64 loses it — the whole point of 80-bit
        // at PC=53 (double) the extra bit is rounded away, matching f64.
        let sum53 = F80::from_f64(1.0).add(F80::from_f64(2f64.powi(-60)), 2, 0);
        assert_eq!(sum53.to_f64(), 1.0);
    }

    #[test]
    fn f80_m80_and_int() {
        let f = |x: f64| F80::from_f64(x);
        for &x in &[1.0, -2.5, 1e100, f64::MIN_POSITIVE, 0.1] {
            let (m, se) = f(x).to_m80();
            assert_eq!(F80::from_m80(m, se).to_f64().to_bits(), x.to_bits(), "m80 {x:e}");
        }
        // canonical 80-bit encodings
        assert_eq!(f(1.0).to_m80(), (0x8000_0000_0000_0000, 0x3fff));
        assert_eq!(f(-1.0).to_m80(), (0x8000_0000_0000_0000, 0xbfff));
        assert_eq!(f(2.0).to_m80(), (0x8000_0000_0000_0000, 0x4000));
        // integers
        assert_eq!(F80::from_i64(5).to_f64(), 5.0);
        assert_eq!(F80::from_i64(-42).to_f64(), -42.0);
        assert_eq!(F80::from_i64(1 << 40).to_f64(), (1u64 << 40) as f64);
        assert_eq!(f(3.7).to_i64(0, false), 4); // nearest
        assert_eq!(f(3.7).to_i64(0, true), 3); // truncate
        assert_eq!(f(-3.7).to_i64(3, false), -3); // toward zero
        assert_eq!(f(2.5).to_i64(0, false), 2); // nearest-even
        assert_eq!(f(3.5).to_i64(0, false), 4); // nearest-even
    }

    #[test]
    fn f80_class() {
        assert_eq!(F80::from_f64(0.0).class, F80Class::Zero);
        assert_eq!(F80::from_f64(1.5).class, F80Class::Normal);
        assert_eq!(F80::from_f64(f64::INFINITY).class, F80Class::Inf);
        assert_eq!(F80::from_f64(f64::NAN).class, F80Class::Nan);
    }
}

