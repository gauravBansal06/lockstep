#!/usr/bin/env python3
# Generate differential test vectors for the SSE/SSE2 ops, comparing the emulator
# against real silicon. Each vector loads xmm0 = [rax(lo), rbx(hi)] and
# xmm1 = [rcx(lo), rdx(hi)] from the GPR inputs, runs ONE op, then extracts the
# 128-bit result back into rax(lo)/rbx(hi) so the GPR-only harness can compare it.
# FP inputs are heavy on the corner cases (0/-0/inf/NaN/denormal/huge) where
# rounding and NaN handling diverge.
import struct, random, sys

def d2b(x):  # f64 -> u64 bits
    return struct.unpack('<Q', struct.pack('<d', x))[0]

def f2b(x):  # f32 -> u32 bits
    return struct.unpack('<I', struct.pack('<f', x))[0]

SPECIAL_D = [0.0, -0.0, 1.0, -1.0, 2.5, -3.75, 0.5, 123456.789, -0.000123,
             1e308, -1e-308, 2.2250738585072014e-308, float('inf'), float('-inf'), float('nan')]
SPECIAL_F = [0.0, -0.0, 1.0, -1.0, 2.5, -3.75, 1e38, -1e-38, float('inf'), float('nan')]

def rand_f64_bits(rng):
    r = rng.random()
    if r < 0.45: return d2b(rng.choice(SPECIAL_D))
    if r < 0.65: return rng.getrandbits(64)
    return d2b(rng.uniform(-1e6, 1e6) * (10.0 ** rng.randint(-12, 12)))

def rand_2xf32_bits(rng):
    def one():
        if rng.random() < 0.5: return f2b(rng.choice(SPECIAL_F))
        return f2b(rng.uniform(-1e5, 1e5) * (10.0 ** rng.randint(-8, 8)))
    return one() | (one() << 32)

def rand_int_bits(rng):
    r = rng.random()
    if r < 0.3: return rng.choice([0, 1, 0xffffffffffffffff, 0x8000000080000000,
                                   0x7fff7fff7fff7fff, 0x0001000100010001])
    return rng.getrandbits(64)

# Wrapper: build xmm0/xmm1 from rax/rbx/rcx/rdx, then extract result lo/hi to rax/rbx.
PREFIX = bytes([0x66,0x48,0x0f,0x6e,0xc0, 0x66,0x48,0x0f,0x6e,0xd3, 0x66,0x0f,0x6c,0xc2,
                0x66,0x48,0x0f,0x6e,0xc9, 0x66,0x48,0x0f,0x6e,0xda, 0x66,0x0f,0x6c,0xcb])
SUFFIX = bytes([0x66,0x48,0x0f,0x7e,0xc0, 0x66,0x0f,0x73,0xd8,0x08, 0x66,0x48,0x0f,0x7e,0xc3])

# (name, op-bytes on xmm0,xmm1, input-kind, optional imm)
FP64 = 'f64'; FP32 = '2xf32'; INT = 'int'
OPS = [
    ('addpd', [0x66,0x0f,0x58,0xc1], FP64), ('subpd', [0x66,0x0f,0x5c,0xc1], FP64),
    ('mulpd', [0x66,0x0f,0x59,0xc1], FP64), ('divpd', [0x66,0x0f,0x5e,0xc1], FP64),
    ('minpd', [0x66,0x0f,0x5d,0xc1], FP64), ('maxpd', [0x66,0x0f,0x5f,0xc1], FP64),
    ('addps', [0x0f,0x58,0xc1], FP32), ('subps', [0x0f,0x5c,0xc1], FP32),
    ('mulps', [0x0f,0x59,0xc1], FP32), ('minps', [0x0f,0x5d,0xc1], FP32),
    ('maxps', [0x0f,0x5f,0xc1], FP32),
    ('addsd', [0xf2,0x0f,0x58,0xc1], FP64), ('divsd', [0xf2,0x0f,0x5e,0xc1], FP64),
    ('minsd', [0xf2,0x0f,0x5d,0xc1], FP64), ('maxsd', [0xf2,0x0f,0x5f,0xc1], FP64),
    ('sqrtsd', [0xf2,0x0f,0x51,0xc1], FP64), ('sqrtpd', [0x66,0x0f,0x51,0xc1], FP64),
    ('unpcklps', [0x0f,0x14,0xc1], INT), ('unpckhps', [0x0f,0x15,0xc1], INT),
    ('unpcklpd', [0x66,0x0f,0x14,0xc1], INT), ('unpckhpd', [0x66,0x0f,0x15,0xc1], INT),
    ('cvttpd2dq', [0x66,0x0f,0xe6,0xc1], FP64), ('cvtpd2dq', [0xf2,0x0f,0xe6,0xc1], FP64),
    ('cvtdq2pd', [0xf3,0x0f,0xe6,0xc1], INT), ('cvtps2pd', [0x0f,0x5a,0xc1], FP32),
    ('cvtpd2ps', [0x66,0x0f,0x5a,0xc1], FP64), ('cvtss2sd', [0xf3,0x0f,0x5a,0xc1], FP32),
    ('cvtsd2ss', [0xf2,0x0f,0x5a,0xc1], FP64), ('cvtdq2ps', [0x0f,0x5b,0xc1], INT),
    ('cvtps2dq', [0x66,0x0f,0x5b,0xc1], FP32), ('cvttps2dq', [0xf3,0x0f,0x5b,0xc1], FP32),
    ('pmuludq', [0x66,0x0f,0xf4,0xc1], INT), ('pmullw', [0x66,0x0f,0xd5,0xc1], INT),
    ('pmulhw', [0x66,0x0f,0xe5,0xc1], INT), ('pmulhuw', [0x66,0x0f,0xe4,0xc1], INT),
    ('pmaddwd', [0x66,0x0f,0xf5,0xc1], INT), ('packsswb', [0x66,0x0f,0x63,0xc1], INT),
    ('packssdw', [0x66,0x0f,0x6b,0xc1], INT), ('packuswb', [0x66,0x0f,0x67,0xc1], INT),
    # SSSE3 / SSE4.1 / SSE4.2 (0F 38 map).
    ('pshufb', [0x66,0x0f,0x38,0x00,0xc1], INT), ('pmulld', [0x66,0x0f,0x38,0x40,0xc1], INT),
    ('pcmpeqq', [0x66,0x0f,0x38,0x29,0xc1], INT), ('pcmpgtq', [0x66,0x0f,0x38,0x37,0xc1], INT),
    ('pminsb', [0x66,0x0f,0x38,0x38,0xc1], INT), ('pminsd', [0x66,0x0f,0x38,0x39,0xc1], INT),
    ('pminuw', [0x66,0x0f,0x38,0x3a,0xc1], INT), ('pminud', [0x66,0x0f,0x38,0x3b,0xc1], INT),
    ('pmaxsb', [0x66,0x0f,0x38,0x3c,0xc1], INT), ('pmaxsd', [0x66,0x0f,0x38,0x3d,0xc1], INT),
    ('pmaxuw', [0x66,0x0f,0x38,0x3e,0xc1], INT), ('pmaxud', [0x66,0x0f,0x38,0x3f,0xc1], INT),
    # SSE3
    ('addsubpd', [0x66,0x0f,0xd0,0xc1], FP64), ('addsubps', [0xf2,0x0f,0xd0,0xc1], FP32),
    ('haddpd', [0x66,0x0f,0x7c,0xc1], FP64), ('haddps', [0xf2,0x0f,0x7c,0xc1], FP32),
    ('hsubpd', [0x66,0x0f,0x7d,0xc1], FP64), ('hsubps', [0xf2,0x0f,0x7d,0xc1], FP32),
    ('movddup', [0xf2,0x0f,0x12,0xc1], FP64), ('movsldup', [0xf3,0x0f,0x12,0xc1], FP32),
    ('movshdup', [0xf3,0x0f,0x16,0xc1], FP32),
    # SSE4.1 / SSSE3 widening, abs, unsigned pack.
    ('pmovsxbw', [0x66,0x0f,0x38,0x20,0xc1], INT), ('pmovsxbd', [0x66,0x0f,0x38,0x21,0xc1], INT),
    ('pmovsxbq', [0x66,0x0f,0x38,0x22,0xc1], INT), ('pmovsxwd', [0x66,0x0f,0x38,0x23,0xc1], INT),
    ('pmovsxwq', [0x66,0x0f,0x38,0x24,0xc1], INT), ('pmovsxdq', [0x66,0x0f,0x38,0x25,0xc1], INT),
    ('pmovzxbw', [0x66,0x0f,0x38,0x30,0xc1], INT), ('pmovzxbd', [0x66,0x0f,0x38,0x31,0xc1], INT),
    ('pmovzxbq', [0x66,0x0f,0x38,0x32,0xc1], INT), ('pmovzxwd', [0x66,0x0f,0x38,0x33,0xc1], INT),
    ('pmovzxwq', [0x66,0x0f,0x38,0x34,0xc1], INT), ('pmovzxdq', [0x66,0x0f,0x38,0x35,0xc1], INT),
    ('pabsb', [0x66,0x0f,0x38,0x1c,0xc1], INT), ('pabsw', [0x66,0x0f,0x38,0x1d,0xc1], INT),
    ('pabsd', [0x66,0x0f,0x38,0x1e,0xc1], INT), ('packusdw', [0x66,0x0f,0x38,0x2b,0xc1], INT),
    # SSSE3 horizontal / sign / madd
    ('phaddw', [0x66,0x0f,0x38,0x01,0xc1], INT), ('phaddd', [0x66,0x0f,0x38,0x02,0xc1], INT),
    ('phaddsw', [0x66,0x0f,0x38,0x03,0xc1], INT), ('phsubw', [0x66,0x0f,0x38,0x05,0xc1], INT),
    ('phsubd', [0x66,0x0f,0x38,0x06,0xc1], INT), ('phsubsw', [0x66,0x0f,0x38,0x07,0xc1], INT),
    ('psignb', [0x66,0x0f,0x38,0x08,0xc1], INT), ('psignw', [0x66,0x0f,0x38,0x09,0xc1], INT),
    ('psignd', [0x66,0x0f,0x38,0x0a,0xc1], INT), ('pmaddubsw', [0x66,0x0f,0x38,0x04,0xc1], INT),
    ('pmulhrsw', [0x66,0x0f,0x38,0x0b,0xc1], INT),
    # SSE4.1 variable blends (mask = XMM0 = dst in this wrapper) + phminposuw
    ('pblendvb', [0x66,0x0f,0x38,0x10,0xc1], INT), ('blendvps', [0x66,0x0f,0x38,0x14,0xc1], FP32),
    ('blendvpd', [0x66,0x0f,0x38,0x15,0xc1], FP64), ('phminposuw', [0x66,0x0f,0x38,0x41,0xc1], INT),
]

# AVX (VEX) — the 3-operand form `vop xmm0, xmm0, xmm1` reuses the SSE wrapper
# (dst=src1=xmm0, src2=xmm1). Bytes: C5 (0xF8|pp) <opcode> C1.  pp: ps=0 pd=1 ss=2 sd=3.
def _vex(pp, opcode):
    return [0xc5, 0xf8 | pp, opcode, 0xc1]
VEX_OPS = [
    # FP packed double / single
    ('vaddpd', _vex(1,0x58), FP64), ('vsubpd', _vex(1,0x5c), FP64),
    ('vmulpd', _vex(1,0x59), FP64), ('vdivpd', _vex(1,0x5e), FP64),
    ('vminpd', _vex(1,0x5d), FP64), ('vmaxpd', _vex(1,0x5f), FP64),
    ('vaddps', _vex(0,0x58), FP32), ('vsubps', _vex(0,0x5c), FP32),
    ('vmulps', _vex(0,0x59), FP32), ('vdivps', _vex(0,0x5e), FP32),
    ('vminps', _vex(0,0x5d), FP32), ('vmaxps', _vex(0,0x5f), FP32),
    # FP scalar double / single
    ('vaddsd', _vex(3,0x58), FP64), ('vsubsd', _vex(3,0x5c), FP64),
    ('vmulsd', _vex(3,0x59), FP64), ('vdivsd', _vex(3,0x5e), FP64),
    ('vminsd', _vex(3,0x5d), FP64), ('vmaxsd', _vex(3,0x5f), FP64),
    ('vaddss', _vex(2,0x58), FP32), ('vsubss', _vex(2,0x5c), FP32),
    ('vmulss', _vex(2,0x59), FP32), ('vdivss', _vex(2,0x5e), FP32),
    # bitwise
    ('vpand', _vex(1,0xdb), INT), ('vpandn', _vex(1,0xdf), INT),
    ('vpor', _vex(1,0xeb), INT), ('vpxor', _vex(1,0xef), INT),
    ('vandpd', _vex(1,0x54), FP64), ('vandps', _vex(0,0x54), FP32),
    ('vorpd', _vex(1,0x56), FP64), ('vxorpd', _vex(1,0x57), FP64),
    # packed integer add/sub
    ('vpaddb', _vex(1,0xfc), INT), ('vpaddw', _vex(1,0xfd), INT),
    ('vpaddd', _vex(1,0xfe), INT), ('vpaddq', _vex(1,0xd4), INT),
    ('vpsubb', _vex(1,0xf8), INT), ('vpsubw', _vex(1,0xf9), INT),
    ('vpsubd', _vex(1,0xfa), INT), ('vpsubq', _vex(1,0xfb), INT),
    # packed integer compare
    ('vpcmpeqb', _vex(1,0x74), INT), ('vpcmpeqw', _vex(1,0x75), INT),
    ('vpcmpeqd', _vex(1,0x76), INT), ('vpcmpgtb', _vex(1,0x64), INT),
    ('vpcmpgtw', _vex(1,0x65), INT), ('vpcmpgtd', _vex(1,0x66), INT),
    # packed integer min/max (0F map)
    ('vpminub', _vex(1,0xda), INT), ('vpmaxub', _vex(1,0xde), INT),
    ('vpminsw', _vex(1,0xea), INT), ('vpmaxsw', _vex(1,0xee), INT),
    # unpack (int + fp)
    ('vpunpcklbw', _vex(1,0x60), INT), ('vpunpcklwd', _vex(1,0x61), INT),
    ('vpunpckldq', _vex(1,0x62), INT), ('vpunpcklqdq', _vex(1,0x6c), INT),
    ('vpunpckhbw', _vex(1,0x68), INT), ('vpunpckhwd', _vex(1,0x69), INT),
    ('vpunpckhdq', _vex(1,0x6a), INT), ('vpunpckhqdq', _vex(1,0x6d), INT),
    ('vunpcklps', _vex(0,0x14), INT), ('vunpckhps', _vex(0,0x15), INT),
    ('vunpcklpd', _vex(1,0x14), INT), ('vunpckhpd', _vex(1,0x15), INT),
    # pack
    ('vpacksswb', _vex(1,0x63), INT), ('vpackssdw', _vex(1,0x6b), INT),
    ('vpackuswb', _vex(1,0x67), INT),
    # shift by xmm
    ('vpsllw', _vex(1,0xf1), INT), ('vpslld', _vex(1,0xf2), INT), ('vpsllq', _vex(1,0xf3), INT),
    ('vpsrlw', _vex(1,0xd1), INT), ('vpsrld', _vex(1,0xd2), INT), ('vpsrlq', _vex(1,0xd3), INT),
    ('vpsraw', _vex(1,0xe1), INT), ('vpsrad', _vex(1,0xe2), INT),
    # converts (128-bit)
    ('vcvtps2pd', _vex(0,0x5a), FP32), ('vcvtpd2ps', _vex(1,0x5a), FP64),
    ('vcvtss2sd', _vex(2,0x5a), FP32), ('vcvtsd2ss', _vex(3,0x5a), FP64),
    ('vcvtdq2ps', _vex(0,0x5b), INT), ('vcvtps2dq', _vex(1,0x5b), FP32),
    ('vcvttps2dq', _vex(2,0x5b), FP32), ('vcvtdq2pd', _vex(2,0xe6), INT),
    ('vcvtpd2dq', _vex(3,0xe6), FP64), ('vcvttpd2dq', _vex(1,0xe6), FP64),
    # multiplies (0F map)
    ('vpmullw', _vex(1,0xd5), INT), ('vpmulhw', _vex(1,0xe5), INT),
    ('vpmulhuw', _vex(1,0xe4), INT), ('vpmuludq', _vex(1,0xf4), INT),
    ('vpmaddwd', _vex(1,0xf5), INT),
    # F16C: VCVTPH2PS xmm0, xmm1 (4 halves → 4 singles)
    ('vcvtph2ps', [0xc4,0xe2,0x79,0x13,0xc1], INT),
    # multiplies (0F38 map) — C4 3-byte, byte2 0x79 (vvvv=1111, pp66)
    ('vpmulld', [0xc4,0xe2,0x79,0x40,0xc1], INT),
    ('vpmuldq', [0xc4,0xe2,0x79,0x28,0xc1], INT),
    ('vpmaddubsw', [0xc4,0xe2,0x79,0x04,0xc1], INT),
    ('vpmulhrsw', [0xc4,0xe2,0x79,0x0b,0xc1], INT),
]
# VEX NDD imm-shifts (group 71/72/73): `vpXXX xmm0, xmm1, imm`. dst=vvvv=xmm0,
# rm=xmm1, reg field=sub-op. C5 byte 0xf9 (vvvv=1111, pp66); modrm reg=sub-op rm=001.
def _vshift(grp, sub):
    return [0xc5, 0xf9, grp, 0xc0 | (sub << 3) | 1]
VEX_SHIFT_IMM = [
    ('vpsrlw', _vshift(0x71, 2)), ('vpsraw', _vshift(0x71, 4)), ('vpsllw', _vshift(0x71, 6)),
    ('vpsrld', _vshift(0x72, 2)), ('vpsrad', _vshift(0x72, 4)), ('vpslld', _vshift(0x72, 6)),
    ('vpsrlq', _vshift(0x73, 2)), ('vpsllq', _vshift(0x73, 6)),
    ('vpsrldq', _vshift(0x73, 3)), ('vpslldq', _vshift(0x73, 7)),
]
# VEX ops carrying a trailing imm8 (predicate / shuffle / blend control).
VEX_IMM_OPS = [
    ('vshufps', _vex(0,0xc6), FP32, 256), ('vshufpd', _vex(1,0xc6), FP64, 256),
    ('vpshufd', _vex(1,0x70), INT, 256), ('vpshufhw', _vex(2,0x70), INT, 256),
    ('vpshuflw', _vex(3,0x70), INT, 256),
    ('vcmpps', _vex(0,0xc2), FP32, 32), ('vcmppd', _vex(1,0xc2), FP64, 32),
    ('vcmpss', _vex(2,0xc2), FP32, 32), ('vcmpsd', _vex(3,0xc2), FP64, 32),
    ('vblendps', [0xc4,0xe3,0x79,0x0c,0xc1], FP32, 256),
    ('vblendpd', [0xc4,0xe3,0x79,0x0d,0xc1], FP64, 256),
    ('vpblendw', [0xc4,0xe3,0x79,0x0e,0xc1], INT, 256),
    ('vpalignr', [0xc4,0xe3,0x79,0x0f,0xc1], INT, 256),
    # F16C VCVTPS2PH xmm0, xmm1, imm (4 singles → 4 halves; dst=rm=xmm0, modrm C8)
    ('vcvtps2ph', [0xc4,0xe3,0x79,0x1d,0xc8], FP32, 4),
    # variable blends (128) — ctrl = imm8[7:4] = 0 → xmm0 is the mask.
    ('vblendvps', [0xc4,0xe3,0x79,0x4a,0xc1], FP32, 1),
    ('vblendvpd', [0xc4,0xe3,0x79,0x4b,0xc1], FP64, 1),
    ('vpblendvb', [0xc4,0xe3,0x79,0x4c,0xc1], INT, 1),
]
# flag-setting ops: (name, op-bytes, input-kind, flagmask)
FLAG_OPS = [
    ('ptest', [0x66,0x0f,0x38,0x17,0xc1], INT, 0x41),  # ZF|CF
]
# GPR-operand ops run RAW (no SSE wrapper): rax = crc accumulator, rbx = data,
# result back in rax. crc32 reg(eax/rax), r/m(ebx/rbx/bl).
GPR_OPS = [
    ('crc32b', [0xf2,0x0f,0x38,0xf0,0xc3]),       # crc32 eax, bl
    ('crc32d', [0xf2,0x0f,0x38,0xf1,0xc3]),       # crc32 eax, ebx
    ('crc32q', [0xf2,0x48,0x0f,0x38,0xf1,0xc3]),  # crc32 rax, rbx
]
# x86-64-v3 GPR ops (64-bit). Inputs rax/rbx/rcx/rdx; result read from rax (+rbx
# for MULX). (name, bytes, flagmask, imm_max-or-None). Defined flags only.
BMI_OPS = [
    ('lzcnt',  [0xf3,0x48,0x0f,0xbd,0xc3], 0x41, None),  # lzcnt rax, rbx
    ('tzcnt',  [0xf3,0x48,0x0f,0xbc,0xc3], 0x41, None),  # tzcnt rax, rbx
    ('andn',   [0xc4,0xe2,0xe0,0xf2,0xc1], 0xc0, None),  # andn rax, rbx, rcx
    ('blsi',   [0xc4,0xe2,0xf8,0xf3,0xdb], 0xc1, None),  # blsi rax, rbx
    ('blsmsk', [0xc4,0xe2,0xf8,0xf3,0xd3], 0xc1, None),  # blsmsk rax, rbx
    ('blsr',   [0xc4,0xe2,0xf8,0xf3,0xcb], 0xc1, None),  # blsr rax, rbx
    ('bextr',  [0xc4,0xe2,0xe0,0xf7,0xc3], 0x40, None),  # bextr rax, rbx, rcx
    ('bzhi',   [0xc4,0xe2,0xe0,0xf5,0xc3], 0xc1, None),  # bzhi rax, rbx, rcx
    ('mulx',   [0xc4,0xe2,0xfb,0xf6,0xd9], 0x00, None),  # mulx rax, rbx, rcx (rax=hi,rbx=lo)
    ('pdep',   [0xc4,0xe2,0xe3,0xf5,0xc1], 0x00, None),  # pdep rax, rbx, rcx
    ('pext',   [0xc4,0xe2,0xe2,0xf5,0xc1], 0x00, None),  # pext rax, rbx, rcx
    ('rorx',   [0xc4,0xe3,0xfb,0xf0,0xc3], 0x00, 64),    # rorx rax, rbx, imm
    ('sarx',   [0xc4,0xe2,0xe2,0xf7,0xc3], 0x00, None),  # sarx rax, rbx, rcx
    ('shrx',   [0xc4,0xe2,0xe3,0xf7,0xc3], 0x00, None),  # shrx rax, rbx, rcx
    ('shlx',   [0xc4,0xe2,0xe1,0xf7,0xc3], 0x00, None),  # shlx rax, rbx, rcx
]
# imm-byte ops: append a per-vector imm.
IMM_OPS = [
    ('shufps', [0x0f,0xc6,0xc1], FP64), ('shufpd', [0x66,0x0f,0xc6,0xc1], FP64),
    ('cmppd', [0x66,0x0f,0xc2,0xc1], FP64), ('cmpsd', [0xf2,0x0f,0xc2,0xc1], FP64),
    ('cmpps', [0x0f,0xc2,0xc1], FP32), ('cmpss', [0xf3,0x0f,0xc2,0xc1], FP32),
    ('palignr', [0x66,0x0f,0x3a,0x0f,0xc1], INT),  # imm = byte shift (0..31)
    ('blendps', [0x66,0x0f,0x3a,0x0c,0xc1], FP32), ('blendpd', [0x66,0x0f,0x3a,0x0d,0xc1], FP64),
    ('pblendw', [0x66,0x0f,0x3a,0x0e,0xc1], INT), ('insertps', [0x66,0x0f,0x3a,0x21,0xc1], FP32),
    # NOTE: DPPS/DPPD are intentionally NOT differential-tested. The Intel SDM
    # defines their intermediate-sum precision as IMPLEMENTATION-SPECIFIC, so they
    # are not bit-exact across CPUs (off-by-one rounding + NaN-sign on inf inputs
    # vs Alder Lake). The emulator still executes them deterministically; we just
    # cannot assert silicon bit-equality on an implementation-defined op (same
    # reasoning as RCP/RSQRT approximations).
    ('mpsadbw', [0x66,0x0f,0x3a,0x42,0xc1], INT),
    ('pinsrd', [0x66,0x0f,0x3a,0x22,0xc1], INT),  # pinsrd xmm0, ecx, imm(lane)
]
def imm_max_for(name):
    if 'cmp' in name: return 8
    return {'palignr': 32, 'blendps': 16, 'blendpd': 4, 'mpsadbw': 8, 'pinsrd': 4}.get(name, 256)
# round ops: imm = rounding mode (0..15)
ROUND_OPS = [
    ('roundpd', [0x66,0x0f,0x3a,0x09,0xc1], FP64), ('roundps', [0x66,0x0f,0x3a,0x08,0xc1], FP32),
    ('roundsd', [0x66,0x0f,0x3a,0x0b,0xc1], FP64), ('roundss', [0x66,0x0f,0x3a,0x0a,0xc1], FP32),
]

def gen_input(kind, rng):
    if kind == FP64: return rand_f64_bits(rng)
    if kind == FP32: return rand_2xf32_bits(rng)
    return rand_int_bits(rng)

def emit(name, opbytes, kind, n, rng, imm_max=None, flagmask=0):
    print(f"# --- {name} ---")
    for _ in range(n):
        a, b, c, d = (gen_input(kind, rng) for _ in range(4))
        ob = list(opbytes)
        if imm_max is not None:
            ob.append(rng.randrange(imm_max))
        code = PREFIX + bytes(ob) + SUFFIX
        print(f"{a:x} {b:x} {c:x} {d:x} {flagmask:x} : " + " ".join(f"{x:02x}" for x in code))

# Load MXCSR = v via the stack (balanced): sub rsp,8; mov [rsp],v; ldmxcsr [rsp]; add rsp,8.
def set_mxcsr(v):
    return bytes([0x48, 0x83, 0xec, 0x08, 0xc7, 0x04, 0x24]) + struct.pack('<I', v) \
        + bytes([0x0f, 0xae, 0x14, 0x24, 0x48, 0x83, 0xc4, 0x08])

# MXCSR rounding-control (RC bits 13:14): 1=−inf, 2=+inf, 3=toward-0. Wrap the op with
# ldmxcsr(RC) … op … ldmxcsr(default) so the directed mode is honoured then reset (the
# silicon harness runs all vectors in one process; the emulator makes a fresh machine each).
# "Tame" NORMAL-range operands so add/sub/mul/div/sqrt results stay well inside the normal
# f64/f32 range. Directed rounding at the SUBNORMAL-underflow boundary (where the host
# residual also underflows and f64 has no wider type to recover the sign) is a documented
# follow-up needing a scaled residual / soft-float — inert in practice. This validates the
# directed-rounding LOGIC (all three modes, both precisions) bit-exact vs silicon.
def tame_f64(rng):
    m = rng.uniform(1.0, 2.0) * (2.0 ** rng.randint(-120, 120))
    return d2b(m if rng.random() < 0.5 else -m)
def tame_2xf32(rng):
    def one():
        m = rng.uniform(1.0, 2.0) * (2.0 ** rng.randint(-48, 48))
        return f2b(m if rng.random() < 0.5 else -m)
    return one() | (one() << 32)
def tame_input(kind, rng):
    return tame_2xf32(rng) if kind == FP32 else tame_f64(rng)

def emit_rc(name, opbytes, kind, n, rng):
    for rc in (1, 2, 3):
        print(f"# --- {name} rc={rc} ---")
        mx = 0x1f80 | (rc << 13)
        for _ in range(n):
            a, b, c, d = (tame_input(kind, rng) for _ in range(4))
            code = PREFIX + set_mxcsr(mx) + bytes(opbytes) + set_mxcsr(0x1f80) + SUFFIX
            print(f"{a:x} {b:x} {c:x} {d:x} 0 : " + " ".join(f"{x:02x}" for x in code))

# Targeted MUL underflow: denormal-producing operands so the product underflows, exercising
# the scaled-Dekker residual (directed rounding into the subnormal grid) + UE/PE flags. Both
# the result (rax/rbx) and the flags (ECX, full mask) are compared vs silicon.
def emit_mul_uf(name, opbytes, kind, n, rng):
    for rc in (0, 1, 2, 3):
        print(f"# --- {name} uf rc={rc} ---")
        mx = 0x1f80 | (rc << 13)
        for _ in range(n):
            # tiny normals whose product underflows (exp sums below the normal range)
            if kind == FP32:
                def one(): return f2b(rng.uniform(1.0, 2.0) * (2.0 ** rng.randint(-90, -60)) * (1 if rng.random() < 0.5 else -1))
                a, b, c, d = (one() | (one() << 32) for _ in range(4))
            else:
                def one(): return d2b(rng.uniform(1.0, 2.0) * (2.0 ** rng.randint(-720, -540)) * (1 if rng.random() < 0.5 else -1))
                a, b, c, d = (one() for _ in range(4))
            code = PREFIX + set_mxcsr(mx) + bytes(opbytes) + flags_to_ecx(0x3f) + set_mxcsr(0x1f80) + SUFFIX
            print(f"{a:x} {b:x} {c:x} {d:x} 0 : " + " ".join(f"{x:02x}" for x in code))

# Underflow / subnormal-boundary DIRECTED rounding for ADD/SUB/DIV (mul has emit_mul_uf).
# Operands are denormal-heavy so the RESULT lands in / near the subnormal grid, exercising
# directed rounding INTO subnormals + UE/PE flags. For DIV this also covers div-by-subnormal
# (normal/subnormal → huge), subnormal/normal (→ subnormal quotient) and subnormal/subnormal.
# Compares result (rax/rbx) AND flags (ECX, full 0x3f mask) vs silicon.
def emit_uf_dir(name, opbytes, kind, n, rng):
    for rc in (0, 1, 2, 3):
        print(f"# --- {name} ufdir rc={rc} ---")
        mx = 0x1f80 | (rc << 13)
        for _ in range(n):
            a, b, c, d = (denorm_input(kind, rng) for _ in range(4))
            code = PREFIX + set_mxcsr(mx) + bytes(opbytes) + flags_to_ecx(0x3f) + set_mxcsr(0x1f80) + SUFFIX
            print(f"{a:x} {b:x} {c:x} {d:x} 0 : " + " ".join(f"{x:02x}" for x in code))

# Denormal-heavy operands: half subnormals, half tiny normals near the underflow boundary,
# so DAZ (input flush) and FTZ (subnormal-result flush) actually fire.
def denorm_f64(rng):
    if rng.random() < 0.5:
        return (rng.getrandbits(52)) | (rng.getrandbits(1) << 63)  # subnormal (exp field 0)
    m = rng.uniform(1.0, 2.0) * (2.0 ** rng.randint(-1074, -1000))
    return d2b(m if rng.random() < 0.5 else -m)
def denorm_2xf32(rng):
    def one():
        if rng.random() < 0.5:
            return (rng.getrandbits(23)) | (rng.getrandbits(1) << 31)  # f32 subnormal
        return f2b(rng.uniform(1.0, 2.0) * (2.0 ** rng.randint(-149, -120)) * (1 if rng.random() < 0.5 else -1))
    return one() | (one() << 32)
def denorm_input(kind, rng):
    return denorm_2xf32(rng) if kind == FP32 else denorm_f64(rng)

# Capture MXCSR exception status bits into ECX (harness compares rcx): stmxcsr [rsp];
# mov ecx,[rsp]; and ecx,<mask>. SUFFIX (result→rax/rbx) leaves rcx untouched.
def flags_to_ecx(mask):
    return bytes([0x48,0x83,0xec,0x08, 0x0f,0xae,0x1c,0x24, 0x8b,0x0c,0x24,
                  0x83,0xe1, mask, 0x48,0x83,0xc4,0x08])

# MXCSR exception STATUS flags vs silicon. Clear to default, run op, read sticky flags.
# UE (bit 4, underflow) — and PE when coupled with underflow — need the exact residual in the
# subnormal region, which the host loses (same corner as the 3a directed-rounding follow-up →
# item 3d soft-float). So: PE validated on NORMAL-range operands (full mask 0x3f, no underflow);
# IE/DE/ZE/OE validated on specials/denormals with UE|PE masked out (0x0f).
def emit_exc(name, opbytes, kind, n, rng, mask, tame):
    print(f"# --- {name} exc mask={mask:#x} ---")
    for _ in range(n):
        if tame:
            a, b, c, d = (tame_input(kind, rng) for _ in range(4))
        elif rng.random() < 0.4:
            a, b, c, d = (denorm_input(kind, rng) for _ in range(4))
        else:
            a, b, c, d = (gen_input(kind, rng) for _ in range(4))
        code = PREFIX + set_mxcsr(0x1f80) + bytes(opbytes) + flags_to_ecx(mask) + SUFFIX
        print(f"{a:x} {b:x} {c:x} {d:x} 0 : " + " ".join(f"{x:02x}" for x in code))

# FTZ (bit 15) flushes subnormal RESULTS to sign-preserved 0; DAZ (bit 6) treats subnormal
# INPUTS as 0. Test at round-nearest (isolated from RC) with denormal-heavy operands.
def emit_flush(name, opbytes, kind, n, rng):
    for tag, mx in (("ftz", 0x9f80), ("daz", 0x1fc0), ("ftz+daz", 0x9fc0)):
        print(f"# --- {name} {tag} ---")
        for _ in range(n):
            a, b, c, d = (denorm_input(kind, rng) for _ in range(4))
            code = PREFIX + set_mxcsr(mx) + bytes(opbytes) + set_mxcsr(0x1f80) + SUFFIX
            print(f"{a:x} {b:x} {c:x} {d:x} 0 : " + " ".join(f"{x:02x}" for x in code))

# Arithmetic + sqrt ops whose result depends on the rounding mode (scalar + packed, ps/pd).
RC_OPS = [
    ('addpd', [0x66,0x0f,0x58,0xc1], FP64), ('subpd', [0x66,0x0f,0x5c,0xc1], FP64),
    ('mulpd', [0x66,0x0f,0x59,0xc1], FP64), ('divpd', [0x66,0x0f,0x5e,0xc1], FP64),
    ('addps', [0x0f,0x58,0xc1], FP32), ('subps', [0x0f,0x5c,0xc1], FP32),
    ('mulps', [0x0f,0x59,0xc1], FP32), ('divps', [0x0f,0x5e,0xc1], FP32),
    ('addsd', [0xf2,0x0f,0x58,0xc1], FP64), ('subsd', [0xf2,0x0f,0x5c,0xc1], FP64),
    ('mulsd', [0xf2,0x0f,0x59,0xc1], FP64), ('divsd', [0xf2,0x0f,0x5e,0xc1], FP64),
    ('addss', [0xf3,0x0f,0x58,0xc1], FP32), ('subss', [0xf3,0x0f,0x5c,0xc1], FP32),
    ('mulss', [0xf3,0x0f,0x59,0xc1], FP32), ('divss', [0xf3,0x0f,0x5e,0xc1], FP32),
    ('sqrtpd', [0x66,0x0f,0x51,0xc1], FP64), ('sqrtps', [0x0f,0x51,0xc1], FP32),
    ('sqrtsd', [0xf2,0x0f,0x51,0xc1], FP64), ('sqrtss', [0xf3,0x0f,0x51,0xc1], FP32),
]

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = random.Random(seed)
    print("# Auto-generated SSE/SSE2 differential vectors (gen_sse.py).")
    print("# xmm0=[rax,rbx], xmm1=[rcx,rdx]; result lo->rax, hi->rbx; flagmask 0.")
    for name, ob, kind in OPS:
        emit(name, ob, kind, n, rng)
    for name, ob, kind in VEX_OPS:
        emit(name, ob, kind, n, rng)
    for name, ob, kind, im in VEX_IMM_OPS:
        emit(name, ob, kind, n, rng, imm_max=im)
    for name, ob in VEX_SHIFT_IMM:
        emit(name, ob, INT, n, rng, imm_max=20)
    for name, ob, kind in IMM_OPS:
        emit(name, ob, kind, n, rng, imm_max=imm_max_for(name))
    for name, ob, kind in ROUND_OPS:
        emit(name, ob, kind, n, rng, imm_max=16)
    for name, ob, kind in RC_OPS:  # MXCSR directed rounding (RC=1/2/3) vs silicon
        emit_rc(name, ob, kind, n, rng)
    for name, ob, kind in RC_OPS:  # MXCSR FTZ/DAZ flush-to-zero vs silicon
        emit_flush(name, ob, kind, n, rng)
    for name, ob, kind in RC_OPS:  # PE on normal-range operands (full mask)
        emit_exc(name, ob, kind, n, rng, 0x3f, tame=True)
    for name, ob, kind in RC_OPS:  # IE/DE/ZE/OE on specials (UE/PE-under-underflow → 3d div edge)
        emit_exc(name, ob, kind, n, rng, 0x0f, tame=False)
    for name, ob, kind in RC_OPS:  # mul underflow: directed rounding + UE/PE, full mask
        if 'mul' in name:
            emit_mul_uf(name, ob, kind, n, rng)
    for name, ob, kind in RC_OPS:  # 3d: add/sub/div directed rounding into the subnormal grid
        if name[:3] in ('add', 'sub', 'div'):
            emit_uf_dir(name, ob, kind, n, rng)
    for name, ob, kind, fm in FLAG_OPS:
        emit(name, ob, kind, n, rng, flagmask=fm)
    # SSE4.2 string compare: result in ECX (istri) / XMM0 (istrm) + all flags (0x8d5).
    emit('pcmpistri', [0x66,0x0f,0x3a,0x63,0xc1], INT, n, rng, imm_max=128, flagmask=0x8d5)
    emit('pcmpistrm', [0x66,0x0f,0x3a,0x62,0xc1], INT, n, rng, imm_max=128, flagmask=0x8d5)
    for name, ob in GPR_OPS:
        print(f"# --- {name} ---")
        for _ in range(n):
            a, b = rng.getrandbits(64), rng.getrandbits(64)
            print(f"{a:x} {b:x} 0 0 0 : " + " ".join(f"{x:02x}" for x in ob))
    for name, ob, fm, imm_max in BMI_OPS:
        print(f"# --- {name} ---")
        for _ in range(n):
            a, b, c, d = (rng.getrandbits(64) for _ in range(4))
            code = list(ob)
            if imm_max is not None:
                code.append(rng.randrange(imm_max))
            print(f"{a:x} {b:x} {c:x} {d:x} {fm:x} : " + " ".join(f"{x:02x}" for x in code))

# ---- 256-bit (YMM) differential ----
# Blob: vmovdqu ymm0,[rdi] ; vmovdqu ymm1,[rdi+0x20] ; <op ymm0,ymm0,ymm1> ;
#       vmovdqu [rsi],ymm0.  Input = 8 qwords (ymm0||ymm1); output = ymm0 (4 qwords).
PREFIX256 = bytes([0xc5,0xfe,0x6f,0x07, 0xc5,0xfe,0x6f,0x4f,0x20])
SUFFIX256 = bytes([0xc5,0xfe,0x7f,0x06])
def _v256(pp, opcode):        # C5 2-byte, L=1, vvvv=ymm0 (1111)
    return [0xc5, 0xfc | pp, opcode, 0xc1]
def _v256_38(opcode):         # C4 3-byte 0F38, L=1, vvvv=ymm0, pp=66 (byte2 0x7d)
    return [0xc4, 0xe2, 0x7d, opcode, 0xc1]
def _vshift256(grp, sub):     # NDD imm-shift: dst=vvvv=ymm0, rm=ymm1, L=1, pp66
    return [0xc5, 0xfd, grp, 0xc0 | (sub << 3) | 1]
def _v256_3a(opcode):         # C4 0F3A, W0, L=1, vvvv=ymm0, pp66
    return [0xc4, 0xe3, 0x7d, opcode, 0xc1]
def _v256_3a_w1(opcode):      # C4 0F3A, W1, L=1, vvvv=ymm0, pp66
    return [0xc4, 0xe3, 0xfd, opcode, 0xc1]
YMM_OPS = [   # (name, opbytes, kind, imm_max-or-None)
    ('vaddps', _v256(0,0x58), FP32, None), ('vaddpd', _v256(1,0x58), FP64, None),
    ('vsubps', _v256(0,0x5c), FP32, None), ('vmulpd', _v256(1,0x59), FP64, None),
    ('vdivps', _v256(0,0x5e), FP32, None), ('vminpd', _v256(1,0x5d), FP64, None),
    ('vmaxps', _v256(0,0x5f), FP32, None),
    ('vpaddd', _v256(1,0xfe), INT, None), ('vpaddw', _v256(1,0xfd), INT, None),
    ('vpaddb', _v256(1,0xfc), INT, None), ('vpaddq', _v256(1,0xd4), INT, None),
    ('vpsubd', _v256(1,0xfa), INT, None),
    ('vpxor', _v256(1,0xef), INT, None), ('vpand', _v256(1,0xdb), INT, None),
    ('vpor', _v256(1,0xeb), INT, None), ('vxorpd', _v256(1,0x57), FP64, None),
    ('vpcmpeqd', _v256(1,0x76), INT, None), ('vpcmpgtd', _v256(1,0x66), INT, None),
    ('vpcmpeqb', _v256(1,0x74), INT, None),
    ('vpunpckldq', _v256(1,0x62), INT, None), ('vpunpckhdq', _v256(1,0x6a), INT, None),
    ('vpunpcklqdq', _v256(1,0x6c), INT, None),
    ('vpminsw', _v256(1,0xea), INT, None), ('vpmaxub', _v256(1,0xde), INT, None),
    ('vpmullw', _v256(1,0xd5), INT, None), ('vpmaddwd', _v256(1,0xf5), INT, None),
    ('vpmulld', _v256_38(0x40), INT, None), ('vpminsd', _v256_38(0x39), INT, None),
    ('vpacksswb', _v256(1,0x63), INT, None),
    ('vpslld_byxmm', _v256(1,0xf2), INT, None), ('vpsrld_byxmm', _v256(1,0xd2), INT, None),
    ('vshufps', _v256(0,0xc6), FP32, 256), ('vshufpd', _v256(1,0xc6), FP64, 256),
    ('vpshufd', _v256(1,0x70), INT, 256),
    ('vcmpps', _v256(0,0xc2), FP32, 32), ('vcmppd', _v256(1,0xc2), FP64, 32),
    ('vpslld_imm', _vshift256(0x72, 6), INT, 20), ('vpsrld_imm', _vshift256(0x72, 2), INT, 20),
    ('vpslldq_imm', _vshift256(0x73, 7), INT, 20),
    # blends (imm spans lanes for PS/PD; PBLENDW repeats per-128). byte2 0x7d (L=1).
    ('vblendps', [0xc4,0xe3,0x7d,0x0c,0xc1], FP32, 256),
    ('vblendpd', [0xc4,0xe3,0x7d,0x0d,0xc1], FP64, 16),
    ('vpblendw', [0xc4,0xe3,0x7d,0x0e,0xc1], INT, 256),
    # variable blends — ctrl register = imm8[7:4] = 0 → ymm0 (the first input) is
    # the mask. So result lane = (ymm0 sign) ? ymm1 : ymm0. imm_max=1 pins imm8=0.
    ('vblendvps', [0xc4,0xe3,0x7d,0x4a,0xc1], FP32, 1),
    ('vblendvpd', [0xc4,0xe3,0x7d,0x4b,0xc1], FP64, 1),
    ('vpblendvb', [0xc4,0xe3,0x7d,0x4c,0xc1], INT, 1),
    # converts (2-operand vop ymm0,ymm1). widening (128->256): ps2pd, dq2pd.
    # narrowing (256->128): pd2ps, pd2dq, tpd2dq. lane-preserving: dq2ps, ps2dq, tps2dq.
    ('vcvtps2pd', _v256(0,0x5a), FP32, None), ('vcvtpd2ps', _v256(1,0x5a), FP64, None),
    ('vcvtdq2pd', _v256(2,0xe6), INT, None), ('vcvtpd2dq', _v256(3,0xe6), FP64, None),
    ('vcvttpd2dq', _v256(1,0xe6), FP64, None),
    ('vcvtdq2ps', _v256(0,0x5b), INT, None), ('vcvtps2dq', _v256(1,0x5b), FP32, None),
    ('vcvttps2dq', _v256(2,0x5b), FP32, None),
    # broadcasts (register source = low element of xmm1).
    ('vpbroadcastb', _v256_38(0x78), INT, None), ('vpbroadcastw', _v256_38(0x79), INT, None),
    ('vpbroadcastd', _v256_38(0x58), INT, None), ('vpbroadcastq', _v256_38(0x59), INT, None),
    ('vbroadcastss', _v256_38(0x18), FP32, None), ('vbroadcastsd', _v256_38(0x19), FP64, None),
    # insert/extract 128 (imm[0]=lane). extract: reg=ymm1(src), rm=xmm0(dst) → modrm 0xc8.
    ('vinsertf128', _v256_3a(0x18), INT, 2), ('vinserti128', _v256_3a(0x38), INT, 2),
    ('vextractf128', [0xc4,0xe3,0x7d,0x19,0xc8], INT, 2),
    ('vextracti128', [0xc4,0xe3,0x7d,0x39,0xc8], INT, 2),
    # 128-bit lane permute (imm).
    ('vperm2f128', _v256_3a(0x06), INT, 256), ('vperm2i128', _v256_3a(0x46), INT, 256),
    # qword/double cross-lane permute (W1, imm).
    ('vpermq', _v256_3a_w1(0x00), INT, 256), ('vpermpd', _v256_3a_w1(0x01), FP64, 256),
    # per-dword variable cross-lane permute (indices in ymm0, data in ymm1).
    ('vpermd', _v256_38(0x36), INT, None), ('vpermps', _v256_38(0x16), FP32, None),
    # F16C 256: VCVTPH2PS ymm0, xmm1 (8 halves→8 singles); VCVTPS2PH xmm0, ymm1, imm.
    ('vcvtph2ps', [0xc4,0xe2,0x7d,0x13,0xc1], INT, None),
    ('vcvtps2ph', [0xc4,0xe3,0x7d,0x1d,0xc8], FP32, 4),
]
# FMA (0F38): `vfma… op0, op1, op1` = dst=ymm0/xmm0, vvvv=ymm1/xmm1, rm=ymm1/xmm1,
# so the op fuses both inputs. byte2 vvvv=1110 (op1), W picks PS/SS vs PD/SD.
def _fma(w, L, opcode):
    byte2 = (w << 7) | (0xe << 3) | (L << 2) | 1
    return [0xc4, 0xe2, byte2, opcode, 0xc1]
# form base 0x90/0xa0/0xb0 = 132/213/231; low nibble = sign (8 add, a sub, c nmadd,
# e nmsub, 6 maddsub, 7 msubadd; +1 = scalar 9/b/d/f).
for _fb in (0x90, 0xa0, 0xb0):
    for _s in (0x6, 0x7, 0x8, 0xa, 0xc, 0xe):  # packed
        VEX_OPS.append((f"vfma{_fb:02x}{_s:x}ps", _fma(0, 0, _fb | _s), FP32))
        VEX_OPS.append((f"vfma{_fb:02x}{_s:x}pd", _fma(1, 0, _fb | _s), FP64))
        YMM_OPS.append((f"vfma{_fb:02x}{_s:x}ps", _fma(0, 1, _fb | _s), FP32, None))
        YMM_OPS.append((f"vfma{_fb:02x}{_s:x}pd", _fma(1, 1, _fb | _s), FP64, None))
    for _s in (0x9, 0xb, 0xd, 0xf):  # scalar (128 only)
        VEX_OPS.append((f"vfma{_fb:02x}{_s:x}ss", _fma(0, 0, _fb | _s), FP32))
        VEX_OPS.append((f"vfma{_fb:02x}{_s:x}sd", _fma(1, 0, _fb | _s), FP64))
def emit256(name, opbytes, kind, n, rng, imm_max=None):
    print(f"# --- {name} (ymm) ---")
    for _ in range(n):
        qs = [gen_input(kind, rng) for _ in range(8)]
        ob = list(opbytes)
        if imm_max is not None:
            ob.append(rng.randrange(imm_max))
        code = PREFIX256 + bytes(ob) + SUFFIX256
        print(" ".join(f"{q:x}" for q in qs) + " : " + " ".join(f"{x:02x}" for x in code))
def main256():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = random.Random(seed)
    print("# Auto-generated AVX-256 (YMM) differential vectors (gen_sse.py 256).")
    print("# 8 input qwords (ymm0||ymm1); result = ymm0 (4 qwords).")
    for name, ob, kind, im in YMM_OPS:
        emit256(name, ob, kind, n, rng, imm_max=im)

if len(sys.argv) > 3 and sys.argv[3] == "256":
    main256()
else:
    main()
