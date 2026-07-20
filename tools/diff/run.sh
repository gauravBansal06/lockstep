#!/usr/bin/env bash
# Differential conformance: emulator (local) vs real x86 silicon (the Linux build host).
# Runs the same instruction vectors on both and diffs the result registers.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
HOST="${LOCKSTEP_HOST:-lockstep-linux}"
REMOTE="${DIFF_HOST:-diff-host}"
# Vector file: a path arg, or the hand-written vectors.txt by default. Regenerate
# the randomized SSE sweep with:  python3 gen_sse.py <seed> <per-op-count> > sse.txt
VECTORS="${1:-$DIR/vectors.txt}"
VNAME="$(basename "$VECTORS")"

echo "[1/3] running vectors on the emulator (local)…"
EMU="$(cd "$ROOT" && cargo run --quiet -- --difftest "$VECTORS" 2>/dev/null)"

echo "[2/3] running vectors on real silicon ($HOST)…"
ssh "$HOST" "mkdir -p $REMOTE"
scp -q "$DIR/trampoline.S" "$DIR/cputest.c" "$VECTORS" "$HOST:$REMOTE/"
REAL="$(ssh "$HOST" "cd $REMOTE && gcc -O0 -no-pie cputest.c trampoline.S -o cputest && ./cputest $VNAME")"

echo "[3/3] diffing…"
if diff <(printf '%s\n' "$EMU") <(printf '%s\n' "$REAL"); then
    n="$(printf '%s\n' "$EMU" | grep -c .)"
    echo "MATCH ✅  emulator == real silicon on $n vectors"
else
    echo "MISMATCH ❌  (left = emulator, right = real silicon)"
    exit 1
fi
