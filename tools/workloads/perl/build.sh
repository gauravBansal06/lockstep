#!/bin/bash
set -e
WORK=/tmp/perlwork; ROOT=$WORK/root
rm -rf $WORK; mkdir -p $ROOT/glibc $ROOT/lib64 $ROOT/dev $ROOT/proc $ROOT/sys
cd ~/perlbuild
echo "[1] HOST reference"; perl work.pl | tee /tmp/perl_ref.txt
echo "[2] assemble"; cp $(readlink -f $(which perl)) $ROOT/perl; cp work.pl $ROOT/work.pl
cp /tmp/csir/busybox $ROOT/busybox; chmod +x $ROOT/busybox $ROOT/perl
echo "[3] loader+deps"; LOADER=$(ldd $(which perl)|grep -oE '/lib64/ld-linux[^ ]*'|head -1); [ -z "$LOADER" ] && LOADER=/lib64/ld-linux-x86-64.so.2
cp "$LOADER" $ROOT/lib64/
ldd $(which perl)|grep -oE '/[^ ]+\.so[^ ]*'|sort -u|while read so; do [ -f "$so" ]&&cp -n "$so" $ROOT/glibc/ 2>/dev/null||true; done
echo "libs: $(ls $ROOT/glibc|wc -l)"
echo "[4] init"; cat > $ROOT/init <<INIT
#!/busybox sh
/busybox mount -t proc proc /proc 2>/dev/null
export LD_LIBRARY_PATH=/glibc
echo PERL-BOOT
/perl /work.pl
/busybox poweroff -f
INIT
chmod +x $ROOT/init
echo "[5] pack"; ( cd $ROOT && find . | /tmp/csir/busybox cpio -o -H newc 2>/dev/null | gzip -9 ) > /tmp/perlir.gz
echo "perlir.gz: $(du -h /tmp/perlir.gz|cut -f1)"
