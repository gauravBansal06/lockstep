#!/bin/bash
# Build each concurrent workload initrd ONCE and freeze it under ~/sf/gate/pinned/.
# The gate reuses these frozen artifacts instead of rebuilding (cpio embeds inode/mtime
# that vary per rebuild → boot-timing → multi-threaded schedule shift). Frozen bytes =
# stable multi-threaded goldens. Rebuild a pinned initrd only when its workload changes.
set -u
SF=~/sf; BB=$SF/deps/busybox; VBLK=$SF/kernel/virtio_blk.ko
OUT=$SF/gate/pinned; mkdir -p $OUT
declare -A CMD=(
  [c]="/svc" [cpp]="/svc" [go]="/csvc" [rust]="/svc"
  [csharp]="LD_LIBRARY_PATH=/glibc /svc"
  [node]="LD_LIBRARY_PATH=/glibc /node /svc.js"
  [python]="LD_LIBRARY_PATH=/glibc /python3 /svc.py"
  [java]="LD_LIBRARY_PATH=/glibc /jdk/bin/java -cp / Svc"
  [perl]="LD_LIBRARY_PATH=/glibc PERL5LIB=/perllib /perl /svc.pl"
  [ruby]="LD_LIBRARY_PATH=/glibc RUBYLIB=/rubylib /ruby /svc.rb"
)
for L in c cpp go rust java ruby perl python csharp node; do
  R=~/concurrent/$L/root
  [ -d "$R" ] || { echo "[$L] SKIP no root"; continue; }
  cp $BB $R/busybox; chmod +x $R/busybox; cp $VBLK $R/virtio_blk.ko
  mkdir -p $R/dev $R/proc $R/sys
  printf '#!/busybox sh\n/busybox mount -t proc proc /proc 2>/dev/null\n/busybox mount -t sysfs sys /sys 2>/dev/null\n/busybox mount -t devtmpfs dev /dev 2>/dev/null\n/busybox insmod /virtio_blk.ko 2>/dev/null\n/busybox ip link set lo up 2>/dev/null\n%s\n/busybox poweroff -f\n' "${CMD[$L]}" > $R/init
  chmod +x $R/init
  # canonical-ish build (sorted names, fixed mtimes, gzip -n) — and then FROZEN
  find $R -exec touch -d @1577836800 {} + 2>/dev/null
  ( cd $R && find . | LC_ALL=C sort | $BB cpio -o -H newc 2>/dev/null | gzip -9n ) > $OUT/$L.gz
  echo "[$L] pinned $(du -h $OUT/$L.gz | cut -f1)  md5=$(md5sum $OUT/$L.gz | awk '{print $1}')"
done
echo "PINNED_OK"
touch $SF/PIN_DONE
