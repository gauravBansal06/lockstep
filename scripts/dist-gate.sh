#!/bin/bash
# Distributed bank-service gate — the 10-language concurrent workload SPLIT across TWO
# nodes over the emulated deterministic virtio-net switch (--net2). Runs ENTIRELY on
#(the build host) (detached); writes per-lang results to ~/dist/res/<lang>.txt, a rolling
# ~/dist/SUMMARY.txt, and finally touches ~/dist/DIST_DONE.
#
# Per language it asserts, over two guest kernels (node A=client, node B=server):
#   - all 3000 requests processed  (APPLIED+FAILED == 3000)
#   - INV-CONSERVED true           (money conserved across the network)
#   - deterministic                (run1 == run2: same B digest + CSVC-HASH + counts)
#
# Reuses each runtime's userspace from ~/concurrent/<lang>/root (already solved: glibc,
# jdk, node, python3, ruby, perl, dotnet-NativeAOT) — swaps in the distributed program,
# the 6.12.95 net modules, and a two-node /init that dispatches on sf.role.
set -u
SF=~/sf
SRC=~/dist-src
D=~/dist
K=$SF/nettest/vmlinuz-virt-6.12.95
BB=$SF/deps/busybox
BIN=$SF/bin/sfhv
NETMODS="$SF/nettest/failover.ko $SF/nettest/net_failover.ko $SF/nettest/virtio_net.ko"
SEED=0000000000000000000000000000000000000000000000000000000000000001
mkdir -p $D/res $D/build
: > $D/SUMMARY.txt
rm -f $D/DIST_DONE

log() { echo "$@" | tee -a $D/SUMMARY.txt; }

# lang -> in-initrd run command (env prefixes + program; role args appended by /init)
declare -A CMD=(
  [c]="/distsvc" [cpp]="/distsvc" [go]="/distsvc" [rust]="/distsvc"
  [csharp]="LD_LIBRARY_PATH=/glibc /distsvc"
  [node]="LD_LIBRARY_PATH=/glibc /node /distsvc.js"
  [python]="LD_LIBRARY_PATH=/glibc /python3 /distsvc.py"
  [java]="LD_LIBRARY_PATH=/glibc /jdk/bin/java -cp / Distsvc"
  [perl]="LD_LIBRARY_PATH=/glibc PERL5LIB=/perllib /perl /distsvc.pl"
  [ruby]="LD_LIBRARY_PATH=/glibc RUBYLIB=/rubylib /ruby /distsvc.rb"
)
ORDER="c cpp go rust python perl ruby node java csharp"

# ---------- build each distributed program ----------
build() {
  local L=$1 bdir=$D/build/$L
  rm -rf $bdir; mkdir -p $bdir
  case $L in
    c)    gcc -O2 -static -pthread $SRC/c/distsvc.c -o $bdir/distsvc 2>$bdir/err ;;
    cpp)  g++ -O2 -static -pthread $SRC/cpp/distsvc.cpp -o $bdir/distsvc 2>$bdir/err ;;
    go)   cp $SRC/go/main.go $bdir/; ( cd $bdir && echo 'module distsvc
go 1.22' > go.mod && CGO_ENABLED=0 GOCACHE=$bdir/.gocache GOPATH=$bdir/.gopath \
          go build -o distsvc main.go ) 2>$bdir/err ;;
    rust) rustc -O -C target-feature=+crt-static $SRC/rust/distsvc.rs -o $bdir/distsvc 2>$bdir/err ;;
    java) cp $SRC/java/Distsvc.java $bdir/; /usr/bin/javac --release 17 -d $bdir $bdir/Distsvc.java 2>$bdir/err ;;
    csharp)
          local pj=~/concurrent/csharp/proj
          cp $SRC/csharp/Distsvc.cs $pj/Program.cs
          ( cd $pj && dotnet publish -c Release -r linux-x64 -o out2 \
              /p:PublishAot=true 2>&1 ) >$bdir/err
          # find the produced native binary
          cp $pj/out2/svc $bdir/distsvc 2>/dev/null || cp $pj/out2/proj $bdir/distsvc 2>/dev/null \
            || cp $(find $pj/out2 -maxdepth 1 -type f -executable ! -name '*.so' | head -1) $bdir/distsvc 2>/dev/null ;;
    python|node|ruby|perl) : ;;  # interpreted: nothing to compile
  esac
}

# ---------- assemble the two-node initrd ----------
assemble() {
  local L=$1 R=$D/$L/root bdir=$D/build/$L C=~/concurrent/$L/root
  rm -rf $R; mkdir -p $R/dev $R/proc $R/sys
  cp $BB $R/busybox; chmod +x $R/busybox
  cp $NETMODS $R/
  # .NET blocks the (idle) server on getrandom() until crng seeds; on two nodes the
  # busy client seeds ~80s earlier and races ahead, so the server applies 0. Give C#
  # the deterministic virtio-rng (seed-driven entropy on its own GSI 7) so crng seeds
  # fast and symmetrically. Harmless for the others (they don't pass --rngdev).
  if [ "$L" = csharp ]; then cp $SF/nettest/rng-core.ko $SF/nettest/virtio-rng.ko $R/; fi
  # reuse the runtime from the concurrent root (skip busybox/init/virtio_blk/old program)
  for f in glibc lib64 usr jdk python3 node ruby rubylib perl perllib; do
    [ -e "$C/$f" ] && cp -a "$C/$f" "$R/"
  done
  case $L in
    c|cpp|go|rust|csharp) cp $bdir/distsvc $R/distsvc; chmod +x $R/distsvc ;;
    java)   cp $bdir/Distsvc.class $R/ ;;
    python) cp $SRC/python/distsvc.py $R/ ;;
    node)   cp $SRC/node/distsvc.js $R/ ;;
    ruby)   cp $SRC/ruby/distsvc.rb $R/ ;;
    perl)   cp $SRC/perl/distsvc.pl $R/ ;;
  esac
  # two-node /init: bring up eth0, dispatch on sf.role
  cat > $R/init <<EOF
#!/busybox sh
/busybox mount -t proc proc /proc 2>/dev/null
/busybox mount -t sysfs sys /sys 2>/dev/null
/busybox mount -t devtmpfs dev /dev 2>/dev/null
[ -f /rng-core.ko ] && /busybox insmod /rng-core.ko 2>/dev/null
[ -f /virtio-rng.ko ] && /busybox insmod /virtio-rng.ko 2>/dev/null
/busybox insmod /failover.ko 2>/dev/null
/busybox insmod /net_failover.ko 2>/dev/null
/busybox insmod /virtio_net.ko 2>/dev/null
IP=""; ROLE=""; PEER=""
for x in \$(/busybox cat /proc/cmdline); do
  case \$x in sf.ip=*) IP=\${x#sf.ip=};; sf.role=*) ROLE=\${x#sf.role=};; sf.peer=*) PEER=\${x#sf.peer=};; esac
done
/busybox ip addr add \$IP dev eth0
/busybox ip link set eth0 up
/busybox sleep 1
if [ "\$ROLE" = server ]; then ${CMD[$L]} server
else /busybox sleep 1; ${CMD[$L]} client \$PEER; fi
/busybox poweroff -f
EOF
  chmod +x $R/init
  ( cd $R && find . | $BB cpio -o -H newc 2>/dev/null | gzip -9 ) > $D/$L.gz
}

# ---------- run one two-guest execution, echo the B(server) + switch lines ----------
runone() {
  local L=$1
  local rng=""; [ "$L" = csharp ] && rng="--rngdev"   # entropy source for .NET (see assemble)
  timeout 1800 $BIN --seed $SEED --net2 $rng --boot $K --initrd $D/$L.gz \
    --cmdline-a "sf.ip=10.0.0.1/24 sf.role=client sf.peer=10.0.0.2" \
    --cmdline-b "sf.ip=10.0.0.2/24 sf.role=server" \
    --ram-mb 1024 --quantum 200000 --max 300000000000 2>&1 \
    | grep -aE "^B\| (CSVC-APPLIED|INV-CONSERVED|CSVC-HASH)|net2: (digest|switch)"
}

log "=== distributed bank gate  $(date) ==="
fail=0
for L in $ORDER; do
  log "----- $L -----"
  build $L
  if [ "$L" != python ] && [ "$L" != node ] && [ "$L" != ruby ] && [ "$L" != perl ]; then
    if [ ! -s $D/build/$L/distsvc ] && [ ! -s $D/build/$L/Distsvc.class ]; then
      log "[$L] BUILD-FAIL"; head -5 $D/build/$L/err | sed 's/^/    /' | tee -a $D/SUMMARY.txt; fail=1; continue
    fi
  fi
  assemble $L
  A="$(runone $L)"; echo "$A" > $D/res/$L.run1.txt
  applied=$(echo "$A" | grep -aoE "CSVC-APPLIED [0-9]+ CSVC-FAILED [0-9]+" | head -1)
  na=$(echo "$applied" | awk '{print $2}'); nf=$(echo "$applied" | awk '{print $4}')
  tot=$(( ${na:-0} + ${nf:-0} ))
  cons=$(echo "$A" | grep -aq "INV-CONSERVED true" && echo ok || echo BAD)
  hashA=$(echo "$A" | grep -aoE "CSVC-HASH [0-9a-f]+" | awk '{print $2}')
  digA=$(echo "$A" | grep -aoE "net2: digest .*")
  if [ "$cons" = ok ] && [ "$tot" = 3000 ]; then
    B="$(runone $L)"; echo "$B" > $D/res/$L.run2.txt
    hashB=$(echo "$B" | grep -aoE "CSVC-HASH [0-9a-f]+" | awk '{print $2}')
    digB=$(echo "$B" | grep -aoE "net2: digest .*")
    det=$([ "$A" = "$B" ] && echo ok || echo BAD)
    st="PASS"; [ "$det" = ok ] || { st="FAIL-NONDET"; fail=1; }
    log "[$L] $st  applied=$na failed=$nf (tot=$tot) conserved=$cons det=$det hash=$hashA"
  else
    log "[$L] FAIL  applied=$na failed=$nf (tot=$tot) conserved=$cons hash=${hashA:-none}"
    echo "$A" | sed 's/^/    /' | tee -a $D/SUMMARY.txt
    fail=1
  fi
done
log "=== DIST_GATE_FAIL=$fail  $(date) ==="
touch $D/DIST_DONE
