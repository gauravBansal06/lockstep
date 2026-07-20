#!/usr/bin/env bash
# SUPERSEDED — use scripts/concurrent-gate.sh (the live ten-runtime gate). Kept for
# reference only: initrd inputs are not bundled and any hardcoded goldens predate the
# Lockstep identity re-baseline, so they will not match a current build.
# Phase-2 determinism harness: boot the same kernel twice with identical inputs
# and assert the whole-machine state digest (RAM + CPU/sys regs + devices) is
# byte-identical. Run-to-run determinism is the project's core guarantee, so this
# is the regression gate for it. (Cross-host host-independence is a separate
# check — see docs/cpu-determinism-contract.md §13.)
#
# Usage: scripts/determinism-check.sh <bzImage> [initrd] [max-instructions]
#   Run on the x86-64 Linux host where the release binary builds.
set -euo pipefail

BIN=./target/release/lockstep
KERNEL="${1:?usage: determinism-check.sh <bzImage> [initrd] [max]}"
INITRD="${2:-}"
MAX="${3:-2000000000}"
CMDLINE="nokaslr console=ttyS0"

[ -x "$BIN" ] || { echo "build first: cargo build --release"; exit 2; }

args=(--boot "$KERNEL" --cmdline "$CMDLINE" --max "$MAX")
[ -n "$INITRD" ] && args+=(--initrd "$INITRD")

run_digest() { "$BIN" "${args[@]}" 2>/dev/null | grep -aoE 'state_digest=[0-9a-f]+' | head -1; }

echo "determinism check: kernel=$KERNEL initrd=${INITRD:-<none>} max=$MAX"
D1=$(run_digest); echo "run 1: $D1"
D2=$(run_digest); echo "run 2: $D2"

if [ "$D1" = "$D2" ] && [ -n "$D1" ]; then
    echo "PASS: byte-identical whole-machine state after $MAX instructions"
    exit 0
else
    echo "FAIL: digests differ (non-determinism) — bisect with smaller --max"
    exit 1
fi
