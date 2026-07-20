#!/usr/bin/env python3
# x87 80-bit differential vectors: run the same FLD/op/FSTP blob on the emulator's true-80-bit
# soft-float AND on real silicon (cputest x87), comparing the 80-bit result bit-for-bit. x87
# basic arithmetic at the default FCW (PC=64 extended, RC=nearest) is deterministic across CPUs,
# so this is a strict conformance check (unlike the impl-defined transcendentals).
#
# Vector line:  A_mant A_se B_mant B_se : <blob bytes>
#   A/B are the two m80 operands (64-bit mantissa incl. explicit integer bit + 16-bit sign/exp).
#   The blob loads them (fld tword[rdi]; fld tword[rdi+10]), runs the op, stores m80 → [rsi].
import random, sys

# blob pieces
FLD_A = [0xdb, 0x2f]        # fld  tword [rdi]
FLD_B = [0xdb, 0x6f, 0x0a]  # fld  tword [rdi+10]
FSTP  = [0xdb, 0x3e]        # fstp tword [rsi]
BIN_OPS = [
    ('faddp', [0xde, 0xc1]),  # ST(1) = A + B, pop
    ('fmulp', [0xde, 0xc9]),  # A * B
    ('fsubp', [0xde, 0xe9]),  # A - B
    ('fdivp', [0xde, 0xf9]),  # A / B
]

def rand_m80(rng, positive=False):
    sign = 0 if positive else rng.randint(0, 1)
    biased = rng.randint(16383 - 500, 16383 + 500)  # moderate exp → stay normal (no over/underflow)
    mant = (1 << 63) | rng.getrandbits(63)          # explicit integer bit + full 63-bit fraction
    return mant, (sign << 15) | biased

def rand_m80_sub(rng, positive=False):
    # Operands that drive results into the 80-bit SUBNORMAL grid (3d-2). Mix of genuine 80-bit
    # subnormals (biased exp 0, integer bit clear) and tiny normals near the underflow boundary.
    sign = 0 if positive else rng.randint(0, 1)
    if rng.random() < 0.4:
        mant = rng.getrandbits(63) or 1          # integer bit (63) CLEAR → subnormal
        return mant, (sign << 15)                # biased exp 0
    biased = rng.randint(1, 300)                 # tiny normal → products underflow to subnormal/0
    mant = (1 << 63) | rng.getrandbits(63)
    return mant, (sign << 15) | biased

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rng = random.Random(seed)
    print("# x87 80-bit differential (gen_x87.py). A_mant A_se B_mant B_se : blob")
    for name, ob in BIN_OPS:
        print(f"# --- {name} ---")
        for _ in range(n):
            am, ase = rand_m80(rng)
            bm, bse = rand_m80(rng)
            blob = FLD_A + FLD_B + ob + FSTP
            print(f"{am:x} {ase:x} {bm:x} {bse:x} : " + " ".join(f"{x:02x}" for x in blob))
    print("# --- fsqrt ---")
    for _ in range(n):
        am, ase = rand_m80(rng, positive=True)
        blob = FLD_A + [0xd9, 0xfa] + FSTP  # fld; fsqrt; fstp
        print(f"{am:x} {ase:x} 0 0 : " + " ".join(f"{x:02x}" for x in blob))
    # 3d-2: subnormal-producing operands for the binary ops (results land in the 80-bit
    # subnormal grid or underflow to zero) + sqrt of a subnormal.
    for name, ob in BIN_OPS:
        print(f"# --- {name} subnormal ---")
        for _ in range(n):
            am, ase = rand_m80_sub(rng)
            bm, bse = rand_m80_sub(rng)
            blob = FLD_A + FLD_B + ob + FSTP
            print(f"{am:x} {ase:x} {bm:x} {bse:x} : " + " ".join(f"{x:02x}" for x in blob))
    print("# --- fsqrt subnormal ---")
    for _ in range(n):
        am, ase = rand_m80_sub(rng, positive=True)
        blob = FLD_A + [0xd9, 0xfa] + FSTP
        print(f"{am:x} {ase:x} 0 0 : " + " ".join(f"{x:02x}" for x in blob))
    # 3d-3: 80-bit constants (FLDZ/FLD1/FLDL2T/FLDL2E/FLDPI/FLDLG2/FLDLN2) — push then FSTP m80.
    CONSTS = [('fld1', 0xe8), ('fldl2t', 0xe9), ('fldl2e', 0xea),
              ('fldpi', 0xeb), ('fldlg2', 0xec), ('fldln2', 0xed), ('fldz', 0xee)]
    for name, op in CONSTS:
        print(f"# --- {name} ---")
        blob = [0xd9, op] + FSTP
        print("0 0 0 0 : " + " ".join(f"{x:02x}" for x in blob))
    # 3d-3: FSCALE — st0 = B·2^trunc(A). A (scale) small-ish incl. over/underflow; B a normal.
    print("# --- fscale ---")
    for _ in range(n):
        # A = scale: value in ~[-2^14, 2^14] so trunc drives normal / over / underflow.
        asign = rng.randint(0, 1)
        abiased = rng.randint(16383 - 2, 16383 + 14)
        am = (1 << 63) | rng.getrandbits(63)
        ase = (asign << 15) | abiased
        bm, bse = rand_m80(rng)  # value
        # fld A; fld B; fscale (st0=B·2^trunc(A), st1=A, NO pop); fstp result; fstp st0 (discard A)
        # — the trailing discard keeps the shared x87 stack balanced (else it overflows → indefinite).
        blob = FLD_A + FLD_B + [0xd9, 0xfd] + FSTP + [0xdd, 0xd8]
        print(f"{am:x} {ase:x} {bm:x} {bse:x} : " + " ".join(f"{x:02x}" for x in blob))
    # 3d-3: FPREM (d9 f8) / FPREM1 (d9 f5): result = B mod A (B=st0 dividend, A=st1 divisor). Keep
    # the exponent gap < 64 → the COMPLETE remainder (C2=0), which is strictly deterministic vs
    # silicon (gap ≥ 64 uses an implementation-defined partial quotient, excluded here).
    for name, op in [('fprem', 0xf8), ('fprem1', 0xf5)]:
        print(f"# --- {name} ---")
        for _ in range(n):
            abiased = rng.randint(16383 - 30, 16383 + 30)          # divisor A exponent
            bbiased = rng.randint(abiased - 40, abiased + 40)      # dividend B: gap ≤ 40 < 64
            am = (1 << 63) | rng.getrandbits(63)
            bm = (1 << 63) | rng.getrandbits(63)
            ase = (rng.randint(0, 1) << 15) | abiased
            bse = (rng.randint(0, 1) << 15) | bbiased
            blob = FLD_A + FLD_B + [0xd9, op] + FSTP + [0xdd, 0xd8]  # fld A; fld B; fprem; fstp; discard A
            print(f"{am:x} {ase:x} {bm:x} {bse:x} : " + " ".join(f"{x:02x}" for x in blob))

main()
