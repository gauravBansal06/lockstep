#!/usr/bin/env bash
# SUPERSEDED — use scripts/concurrent-gate.sh (the live ten-runtime gate). Kept for
# reference only: initrd inputs are not bundled and any hardcoded goldens predate the
# Lockstep identity re-baseline, so they will not match a current build.
# Determinism & correctness regression gate — the "must never break" guard.
#
# Run on(the build host) after EVERY change. Tier 1 (fast, always): unit suite + SSE
# differential vs silicon + battle-test golden + boot-twice. Tier 2 (--full):
# the heavy real workloads (CPython, Node) — each proven byte-identical to QEMU.
#
# Golden digests are for the CURRENT machine config (x86-64-v2 CPUID). A deliberate
# config change (CPUID / digest-formula) re-baselines them: update here in the same
# commit and re-prove run-to-run determinism. Every NEW proven workload gets added.
set -euo pipefail

HOST="${LOCKSTEP_HOST:-lockstep-linux}"
DIR="${BOOT_DIR:-lockstep}"
K="/tmp/vmlinuz-virt-93"
SEED1="0000000000000000000000000000000000000000000000000000000000000001"

# ---- golden values (x86-64-v3 CPUID, USERSPACE AVX ON: leaf-0xD subleaf-1 = 0x1) ----
# RE-BASELINED 2026-06-23 at the userspace-AVX flip (B.4): leaf-0xD subleaf-1 0x3→0x1 makes
# the kernel KEEP system xstate enabled, so userspace finally runs real AVX (XSAVE-backed YMM)
# instead of the FXSAVE fallback. Every guest's ifunc dispatch now selects true AVX/AVX2 paths,
# so every whole-machine state_digest moved. This unblocks AVX-*requiring* binaries (C#/.NET).
# Reaching userspace required a VEX op batch (saturating arith, avg, sadbw, pmovsx/zx, vptest,
# vpinsr/extr, vpblendd, vpermilps/pd, movddup, movntdq, VSQRTSS/SD/PS/PD — the V8 #UD was
# VSQRTSD) and a 256-bit VPTEST fix (Go's bytealg memequal compares ymm; a 128-bit-only VPTEST
# mis-set ZF → corrupted HTTP parsing → battle printed "400 Bad Request"; full-width fixed it).
# Property RE-PROVEN: boot-twice byte-identical for all 4. App-level correctness PRESERVED
# (C-HASH/PY-HASH/NODE-HASH IDENTICAL to pre-flip). battle race -2 → -6 (∈ QEMU's set {-6,-2,4};
# the AVX-shifted instruction timing lands the interleaving differently, still deterministic).
G_BATTLE_DIGEST="b2f823c1870fe1f4"; G_BATTLE_RESULT="-6"  # /tmp/battle.gz  (Go, musl)
G_C_HASH="353504a2bc8a29ab";       G_C_DIGEST="6b0a7cecf0f04a0b"     # /tmp/cir.gz   (C, glibc static)
G_CPY_HASH="3b2639986b6e8ada";     G_CPY_DIGEST="9184307f3bbbc535"  # /tmp/pyir.gz  (CPython, musl)
G_NODE_HASH="7b05e6d9b8415e3e";    G_NODE_DIGEST="47eb3ec7a18d0502" # /tmp/nodeir.gz (Node/V8, glibc)
G_CS_HASH="9217b151a2cb0087";      G_CS_DIGEST="52b081d72b959c98"   # /tmp/csir.gz  (.NET NativeAOT C#, glibc — AVX-REQUIRING)
G_JVM_HASH="111a5b05b6bfd07b";     G_JVM_DIGEST="7474a05fe8e824ae"  # /tmp/jvmir.gz (OpenJDK 17 HotSpot, glibc — JIT+GC threads)
G_PHP_HASH="54e9eca6654b4761";     G_PHP_DIGEST="8e3492eca47b662e"  # /tmp/phpir.gz (PHP 8.3 Zend interp, glibc)
G_PERL_HASH="0008b4380008b438";    G_PERL_DIGEST="ec51752f242ea212" # /tmp/perlir.gz (Perl 5.38, glibc)
G_RUBY_HASH="cae625523bbd1932";    G_RUBY_DIGEST="22adb5208ad582e8" # /tmp/rubyir.gz (Ruby 3.2, glibc)
G_RUST_HASH="bccae32f18672ce7";    G_RUST_DIGEST="d1df7b6b2f03605b" # /tmp/rustir.gz (Rust 1.75 static)
# Concurrency-determinism check: a multi-threaded (pthreads+mutex) program whose output is the
# thread INTERLEAVING ORDER. seed1/no-jitter => one fixed schedule, byte-identical run-to-run.
# (Seed-driven SF_SCHED_JITTER explores OTHER interleavings — see tools/workloads/dst/README.md.)
G_ORDER_HASH="6d15e3daa9690309";   G_ORDER_DIGEST="87acda385ca3c8e8" # /tmp/orderir.gz (pthreads schedule)

fail() { echo "GATE FAIL: $*" >&2; exit 1; }
run() { # $1=initrd  $2=extra-env  -> prints the marker lines
    ssh "$HOST" "cd $DIR && $2 timeout 2500 ./target/release/lockstep --seed $SEED1 \
        --boot $K --cmdline 'nokaslr console=ttyS0' --initrd $1 --ram-mb 1024 --max 100000000000 \
        2>/dev/null | grep -aE 'SIMFORGE_RESULT|C-HASH|PY-HASH|NODE-HASH|CS-HASH|JVM-HASH|PHP-HASH|PERL-HASH|RUBY-HASH|RUST-HASH|ORDER-HASH|state_digest'"
}

echo "[1] unit suite…"
ssh "$HOST" "cd $DIR && touch src/x86/*.rs src/lib.rs && cargo build --release 2>&1 | grep -E 'error' && exit 1
    cargo test --release --lib 2>&1 | grep -E 'test result'" | tee /tmp/gate1.txt
grep -q '0 failed' /tmp/gate1.txt || fail "unit tests"

echo "[2] SSE/SSE2/SSE3/SSE4 differential vs real silicon (must be 0 mismatch)…"
ssh "$HOST" "cd $DIR && python3 tools/diff/gen_sse.py 1 30 > /tmp/g.txt && python3 tools/diff/gen_sse.py 7 30 >> /tmp/g.txt
    ./target/release/lockstep --difftest /tmp/g.txt > /tmp/ge.txt 2>/dev/null
    cd tools/diff && gcc -O0 -no-pie cputest.c trampoline.S -o /tmp/gc 2>/dev/null && /tmp/gc /tmp/g.txt > /tmp/gr.txt 2>/dev/null
    diff -q /tmp/ge.txt /tmp/gr.txt >/dev/null && echo SSE-MATCH || echo SSE-MISMATCH" | tee /tmp/gate2.txt
grep -q SSE-MATCH /tmp/gate2.txt || fail "SSE/VEX-128 differential vs silicon"

echo "[2b] AVX-256 (YMM) differential vs real silicon (must be 0 mismatch)…"
ssh "$HOST" "cd $DIR && python3 tools/diff/gen_sse.py 1 25 256 > /tmp/y.txt && python3 tools/diff/gen_sse.py 7 25 256 >> /tmp/y.txt
    ./target/release/lockstep --difftest256 /tmp/y.txt > /tmp/ye.txt 2>/dev/null
    /tmp/gc /tmp/y.txt 256 > /tmp/yr.txt 2>/dev/null
    diff -q /tmp/ye.txt /tmp/yr.txt >/dev/null && echo YMM-MATCH || echo YMM-MISMATCH" | tee /tmp/gate2b.txt
grep -q YMM-MATCH /tmp/gate2b.txt || fail "AVX-256 differential vs silicon"

echo "[3] battle-test (Go) — digest=$G_BATTLE_DIGEST result=$G_BATTLE_RESULT, boot-twice…"
A="$(run /tmp/battle.gz '')"; B="$(run /tmp/battle.gz '')"
echo "$A"
echo "$A" | grep -q "state_digest=$G_BATTLE_DIGEST" || fail "battle digest moved"
echo "$A" | grep -q "SIMFORGE_RESULT=$G_BATTLE_RESULT" || fail "battle result moved"
[ "$A" = "$B" ] || fail "battle not deterministic run-to-run"

echo "[4] C (glibc static, x87 FPREM/libm) — C-HASH=$G_C_HASH digest=$G_C_DIGEST, boot-twice…"
CA="$(run /tmp/cir.gz '')"; CB="$(run /tmp/cir.gz '')"
echo "$CA"
echo "$CA" | grep -q "C-HASH $G_C_HASH" || fail "C hash moved (correctness!)"
echo "$CA" | grep -q "state_digest=$G_C_DIGEST" || fail "C digest moved"
[ "$CA" = "$CB" ] || fail "C not deterministic run-to-run"

if [ "${1:-}" = "--full" ]; then
    echo "[5] CPython (musl) — PY-HASH=$G_CPY_HASH digest=${G_CPY_DIGEST} ..."
    C="$(run /tmp/pyir.gz '')"
    echo "$C"
    echo "$C" | grep -q "PY-HASH $G_CPY_HASH" || fail "CPython hash moved (correctness!)"
    echo "$C" | grep -q "state_digest=$G_CPY_DIGEST" || fail "CPython digest moved"

    echo "[6] Node/V8 (glibc) — NODE-HASH=$G_NODE_HASH digest=${G_NODE_DIGEST} ..."
    N="$(run /tmp/nodeir.gz '')"
    echo "$N"
    echo "$N" | grep -q "NODE-HASH $G_NODE_HASH" || fail "Node hash moved (correctness!)"
    echo "$N" | grep -q "state_digest=$G_NODE_DIGEST" || fail "Node digest moved"

    echo "[7] .NET NativeAOT C# (glibc, AVX-REQUIRING) — CS-HASH=$G_CS_HASH digest=${G_CS_DIGEST} ..."
    S="$(run /tmp/csir.gz '')"
    echo "$S"
    echo "$S" | grep -q "CS-HASH $G_CS_HASH" || fail "C# hash moved (correctness!)"
    echo "$S" | grep -q "state_digest=$G_CS_DIGEST" || fail "C# digest moved"

    echo "[8] OpenJDK 17 HotSpot JVM (glibc, JIT+GC threads) — JVM-HASH=$G_JVM_HASH digest=${G_JVM_DIGEST} ..."
    J="$(run /tmp/jvmir.gz '')"
    echo "$J"
    echo "$J" | grep -q "JVM-HASH $G_JVM_HASH" || fail "JVM hash moved (correctness!)"
    echo "$J" | grep -q "state_digest=$G_JVM_DIGEST" || fail "JVM digest moved"

    echo "[9] PHP 8.3 Zend interpreter (glibc) — PHP-HASH=$G_PHP_HASH digest=${G_PHP_DIGEST} ..."
    P="$(run /tmp/phpir.gz '')"
    echo "$P"
    echo "$P" | grep -q "PHP-HASH $G_PHP_HASH" || fail "PHP hash moved (correctness!)"
    echo "$P" | grep -q "state_digest=$G_PHP_DIGEST" || fail "PHP digest moved"

    echo "[10] Perl 5.38 (glibc) — PERL-HASH=$G_PERL_HASH digest=${G_PERL_DIGEST} ..."
    PL="$(run /tmp/perlir.gz '')"
    echo "$PL"
    echo "$PL" | grep -q "PERL-HASH $G_PERL_HASH" || fail "Perl hash moved (correctness!)"
    echo "$PL" | grep -q "state_digest=$G_PERL_DIGEST" || fail "Perl digest moved"

    echo "[11] Ruby 3.2 (glibc) — RUBY-HASH=$G_RUBY_HASH digest=${G_RUBY_DIGEST} ..."
    RB="$(run /tmp/rubyir.gz '')"
    echo "$RB"
    echo "$RB" | grep -q "RUBY-HASH $G_RUBY_HASH" || fail "Ruby hash moved (correctness!)"
    echo "$RB" | grep -q "state_digest=$G_RUBY_DIGEST" || fail "Ruby digest moved"

    echo "[12] Rust 1.75 static (compute) — RUST-HASH=$G_RUST_HASH digest=${G_RUST_DIGEST} ..."
    RS="$(run /tmp/rustir.gz '')"
    echo "$RS"
    echo "$RS" | grep -q "RUST-HASH $G_RUST_HASH" || fail "Rust hash moved (correctness!)"
    echo "$RS" | grep -q "state_digest=$G_RUST_DIGEST" || fail "Rust digest moved"

    echo "[13] concurrency determinism (pthreads schedule order) — ORDER-HASH=$G_ORDER_HASH digest=${G_ORDER_DIGEST} ..."
    OR="$(run /tmp/orderir.gz '')"
    echo "$OR"
    echo "$OR" | grep -q "ORDER-HASH $G_ORDER_HASH" || fail "concurrency schedule moved (determinism!)"
    echo "$OR" | grep -q "state_digest=$G_ORDER_DIGEST" || fail "concurrency digest moved"
fi

echo "GATE PASS ✅  determinism + correctness contract intact"
