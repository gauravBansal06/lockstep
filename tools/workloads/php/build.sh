#!/bin/bash
# Build the PHP gate workload initrd (/tmp/phpir.gz) on the build host.
set -e
WORK=/tmp/phpwork; ROOT=$WORK/root
rm -rf $WORK; mkdir -p $ROOT/glibc $ROOT/lib64 $ROOT/usr/share $ROOT/dev $ROOT/proc $ROOT/sys
cd ~/phpbuild

echo "[1] reference hash on HOST php (-n: no ini/extensions)"
php -n -d date.timezone=UTC work.php | tee /tmp/php_ref.txt

echo "[2] assemble initrd root"
cp /usr/bin/php $ROOT/php; cp work.php $ROOT/work.php
cp /tmp/csir/busybox $ROOT/busybox
chmod +x $ROOT/busybox $ROOT/php

echo "[3] loader + recursive deps"
LOADER=$(ldd /usr/bin/php | grep -oE '/lib64/ld-linux[^ ]*' | head -1)
[ -z "$LOADER" ] && LOADER=/lib64/ld-linux-x86-64.so.2
cp "$LOADER" $ROOT/lib64/
ldd /usr/bin/php | grep -oE '/[^ ]+\.so[^ ]*' | sort -u | while read so; do
  [ -f "$so" ] && cp -n "$so" $ROOT/glibc/ 2>/dev/null || true
done
echo "libs: $(ls $ROOT/glibc | wc -l)"

echo "[4] timezone DB (php's date ext overflows a buffer if zoneinfo is absent)"
cp -a /usr/share/zoneinfo $ROOT/usr/share/

echo "[5] init"
cat > $ROOT/init <<'INIT'
#!/busybox sh
/busybox mount -t proc proc /proc 2>/dev/null
/busybox mount -t sysfs sys /sys 2>/dev/null
export LD_LIBRARY_PATH=/glibc
export TZ=UTC
echo PHP-BOOT
/php -n -d date.timezone=UTC /work.php
/busybox poweroff -f
INIT
chmod +x $ROOT/init

echo "[6] pack"
( cd $ROOT && find . | /tmp/csir/busybox cpio -o -H newc 2>/dev/null | gzip -9 ) > /tmp/phpir.gz
echo "phpir.gz: $(du -h /tmp/phpir.gz | cut -f1)"
