#!/bin/bash
# Persistent, home-based determinism/correctness gate over the 10-language
# concurrent bank-service workloads (mem + loopback-net + virtio-blk disk +
# conservation invariant). Everything lives under ~/sf (NOT /tmp, which the host
# wipes). Run from your workstation: `bash scripts/concurrent-gate.sh` (it sshes the build host),
# or on the build host: `ON_PCMINI=1 bash scripts/concurrent-gate.sh`.
#
# Per language it asserts:  seed1 run A == run B (determinism, byte-identical),
# INV-CONSERVED true, INV-DISK-MATCH true.  It PRINTS the seed1 CSVC-HASH so a
# re-baseline is a copy-paste. Multi-threaded hashes are schedule-dependent (the
# real correctness anchor is the invariants + the ours==QEMU cross-check); the
# single-threaded Python asyncio hash is stable and pinned as a golden.
set -u
HOST="${LOCKSTEP_HOST:-lockstep-linux}"
ON_PCMINI="${ON_PCMINI:-}"
SEED1=0000000000000000000000000000000000000000000000000000000000000001

# ---- goldens: RE-BASELINED 2026-07-07 for kernel 6.12.94-0-virt + the POPF/IF fix ----
# (POPF now honours CPL/IOPL so userspace IF is preserved; this fixed Node's kernel-BUG
# crash. Multi-threaded hashes are schedule-dependent but the schedule is deterministic
# under fixed seed+kernel+emulator, so they're stable regression anchors. Python asyncio
# is single-threaded → its hash is schedule-INDEPENDENT and the strongest correctness pin.
# node == QEMU oracle 28d1c4de2c0b133b.)  All 10 also assert INV-CONSERVED + INV-DISK-MATCH
# + byte-identical run-to-run.

remote() { if [ -n "$ON_PCMINI" ]; then bash -c "$1"; else ssh "$HOST" "$1"; fi; }

read -r -d '' BODY <<'EOS'
set -u
SF=~/sf
K=$SF/kernel/vmlinuz-virt-6.12.94
BB=$SF/deps/busybox
VBLK=$SF/kernel/virtio_blk.ko
BIN=$SF/bin/sfhv
GATE=$SF/gate
SEED1=0000000000000000000000000000000000000000000000000000000000000001
mkdir -p $GATE
fail=0

# lang -> runcmd (entrypoints inside the initrd; glibc runtimes need LD_LIBRARY_PATH=/glibc)
declare -A CMD=(
  [c]="/svc" [cpp]="/svc" [go]="/csvc" [rust]="/svc"
  [csharp]="LD_LIBRARY_PATH=/glibc /svc"
  [node]="LD_LIBRARY_PATH=/glibc /node /svc.js"
  [python]="LD_LIBRARY_PATH=/glibc /python3 /svc.py"
  [java]="LD_LIBRARY_PATH=/glibc /jdk/bin/java -cp / Svc"
  [perl]="LD_LIBRARY_PATH=/glibc PERL5LIB=/perllib /perl /svc.pl"
  [ruby]="LD_LIBRARY_PATH=/glibc RUBYLIB=/rubylib /ruby /svc.rb"
)
# RE-BASELINED 2026-07-20 at the Lockstep identity rename (CPU brand + ACPI OEM IDs +
# seed/build-hash prefixes changed → the boot instruction stream shifted). The 8 multi-threaded
# values moved (their timer-driven preemption schedule shifted); python + node are single-thread
# event loops, so their hashes are schedule-INDEPENDENT and are UNCHANGED across the rename
# (e423e849 / 28d1c4de) — the real correctness anchors, alongside INV-* + det + the differential.
# Property re-proven: det=ok (byte-identical run-to-run) for all 10, invariants hold, silicon
# differential still MATCHes.
declare -A GOLD=(
  [c]="4857f875aea017ad" [cpp]="64e91731e6463786" [go]="8349ba3994d4b9f1"
  [rust]="18ee6577defa196f" [java]="b24e485e0cc5bd06" [ruby]="3ab6c4fa007bf64f"
  [perl]="bf2642c71637eaa8" [python]="e423e84947e598a6" [csharp]="1bbac2de9db67d23"
  [node]="28d1c4de2c0b133b"
)
ORDER="c cpp go rust java ruby perl python csharp node"

runone() { # $1=initrd $2=img [$3=full-output-file] -> marker lines (full run output → $3)
  $BB dd if=/dev/zero of=$2 bs=1024 count=1024 2>/dev/null
  local out="${3:-$GATE/_run_tmp.txt}"  # must be a real file (we grep it) — NOT /dev/null
  timeout 1500 $BIN --seed $SEED1 --boot $K --cmdline "nokaslr console=ttyS0" \
    --initrd $1 --disk $2 --ram-mb 1024 --max 100000000000 2>/dev/null > "$out"
  grep -aE "CSVC-HASH|INV-CONSERVED|INV-DISK-MATCH|NODE-RC|CSVC-EXIT" "$out"
}
# 1-minute host load average (context for a slow run — the emulator's own MIPS is deterministic-safe
# and printed by run_boot as `=> elapsed_ms=/mips=`; load explains WHY a given run was fast/slow).
hostload() { uptime | sed -E 's/.*load average: *([0-9.]+).*/\1/'; }

# Use PINNED, pre-built initrds ($SF/gate/pinned/<lang>.gz). Rebuilding per run is
# non-reproducible — busybox `cpio -H newc` embeds inode numbers + mtimes and `find`
# order tracks filesystem state, so each rebuild yields different initrd bytes → a
# different tmpfs-extraction timing → a different multi-thread preemption point → a
# shifted CSVC-HASH (single-thread python/node are immune). Freezing the artifact keeps
# the multi-threaded goldens stable regression anchors. Re-pin (scripts/pin-initrds.sh)
# only when a workload changes. The determinism *property* (det, run-to-run byte-identical)
# and correctness anchors (python/node schedule-independent hashes + INV-*) never depended
# on this — only the schedule-dependent multi-threaded hash values did.
PIN=$SF/gate/pinned
tot_instr=0; tot_ms=0
for L in $ORDER; do
  [ -f "$PIN/$L.gz" ] || { echo "[$L] SKIP (no pinned $PIN/$L.gz — run pin-initrds.sh)"; continue; }
  load=$(hostload)                                            # host load at run start
  A="$(runone $PIN/$L.gz $GATE/csvc_$L.img $GATE/full_$L.txt)"
  # perf from run A's full emulator output (baked into run_boot; wall-clock, determinism-safe).
  instr=$(grep -aoE "instructions=[0-9]+" $GATE/full_$L.txt | head -1 | cut -d= -f2)
  ems=$(grep -aoE "elapsed_ms=[0-9]+" $GATE/full_$L.txt | head -1 | cut -d= -f2)
  mips=$(grep -aoE "mips=[0-9.]+" $GATE/full_$L.txt | head -1 | cut -d= -f2)
  B="$(runone $PIN/$L.gz $GATE/csvc_$L.img)"                  # 2nd run: determinism check
  HA=$(echo "$A" | grep -aoE "CSVC-HASH [0-9a-f]+" | awk '{print $2}')
  cons=$(echo "$A" | grep -aq "INV-CONSERVED true" && echo ok || echo BAD)
  disk=$(echo "$A" | grep -aq "INV-DISK-MATCH true" && echo ok || echo BAD)
  det=$([ "$A" = "$B" ] && echo ok || echo BAD)
  gold=$([ "$HA" = "${GOLD[$L]:-}" ] && echo ok || echo "SHIFTED(${GOLD[$L]:-none})")
  st="PASS"
  [ "$cons" = ok ] && [ "$disk" = ok ] && [ "$det" = ok ] && [ "$gold" = ok ] || { st="FAIL"; fail=1; }
  tot_instr=$((tot_instr + ${instr:-0})); tot_ms=$((tot_ms + ${ems:-0}))
  printf "[%-6s] %s  hash=%s  conserved=%s disk=%s det=%s golden=%s  | instr=%s time=%ss mips=%s load1m=%s\n" \
    "$L" "$st" "${HA:-none}" "$cons" "$disk" "$det" "$gold" \
    "${instr:-?}" "$(awk "BEGIN{printf \"%.1f\", ${ems:-0}/1000}")" "${mips:-?}" "$load"
done
echo "GATE_CONCURRENT_FAIL=$fail"
# Aggregate throughput (sum over the 10 workloads' first runs).
awk "BEGIN{printf \"PERF-TOTAL instructions=%d wall_s=%.1f mips=%.2f (host load1m=$(hostload))\n\", \
  $tot_instr, $tot_ms/1000, ($tot_ms>0)?$tot_instr/($tot_ms/1000)/1e6:0}"
EOS

echo "=== [1] unit suite ==="
remote "cd ~/lockstep && touch src/x86/*.rs src/lib.rs src/main.rs && cargo build --release 2>&1 | grep -E '^error' && exit 1; cargo test --release --lib 2>&1 | grep 'test result'"

echo "=== [2] SSE + AVX-256 differential vs silicon ==="
remote "cd ~/lockstep && python3 tools/diff/gen_sse.py 1 30 > /tmp/g.txt && python3 tools/diff/gen_sse.py 7 30 >> /tmp/g.txt && ~/sf/bin/sfhv --difftest /tmp/g.txt > /tmp/ge.txt 2>/dev/null && (cd tools/diff && gcc -O0 -no-pie cputest.c trampoline.S -o /tmp/gc 2>/dev/null && /tmp/gc /tmp/g.txt > /tmp/gr.txt 2>/dev/null) && diff -q /tmp/ge.txt /tmp/gr.txt >/dev/null && echo SSE-MATCH || echo SSE-MISMATCH; python3 tools/diff/gen_sse.py 1 25 256 > /tmp/y.txt && python3 tools/diff/gen_sse.py 7 25 256 >> /tmp/y.txt && ~/sf/bin/sfhv --difftest256 /tmp/y.txt > /tmp/ye.txt 2>/dev/null && /tmp/gc /tmp/y.txt 256 > /tmp/yr.txt 2>/dev/null && diff -q /tmp/ye.txt /tmp/yr.txt >/dev/null && echo YMM-MATCH || echo YMM-MISMATCH"

echo "=== [3] 10-language concurrent workloads (determinism + invariants) ==="
echo "Python asyncio golden (single-thread, stable): e423e84947e598a6"
remote "$BODY"
