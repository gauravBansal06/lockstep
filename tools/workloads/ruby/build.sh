#!/bin/bash
set -e
EXT=/tmp/rubyext           # extracted .deb tree
WORK=/tmp/rubywork; ROOT=$WORK/root
rm -rf $WORK; mkdir -p $ROOT/glibc $ROOT/lib64 $ROOT/rubylib $ROOT/dev $ROOT/proc $ROOT/sys
cd ~/rubybuild
RB=$EXT/usr/bin/ruby3.2
LIBR=$(find $EXT -name 'libruby-3.2.so*' | head -1)
echo "[1] HOST reference"
env LD_LIBRARY_PATH=$(dirname $LIBR) RUBYLIB=$EXT/usr/lib/ruby/3.2.0:$EXT/usr/lib/x86_64-linux-gnu/ruby/3.2.0 \
    $RB --disable-gems work.rb | tee /tmp/ruby_ref.txt
echo "[2] assemble"
cp $RB $ROOT/ruby; cp $LIBR $ROOT/glibc/; cp work.rb $ROOT/work.rb
# recreate libruby SONAME symlinks (loader needs libruby-3.2.so.3.2, not the full .3.2.3)
( cd $ROOT/glibc && B=$(basename $LIBR); ln -sf $B ${B%.*}; ln -sf $B libruby-3.2.so )
cp /tmp/csir/busybox $ROOT/busybox; chmod +x $ROOT/busybox $ROOT/ruby
cp -a $EXT/usr/lib/ruby/3.2.0/. $ROOT/rubylib/
cp -a $EXT/usr/lib/x86_64-linux-gnu/ruby/3.2.0/. $ROOT/rubylib/   # C extensions (.so)
echo "[3] loader + deps (ruby + libruby + every ruby .so ext)"
LOADER=$(ldd $RB | grep -oE '/lib64/ld-linux[^ ]*' | head -1); [ -z "$LOADER" ] && LOADER=/lib64/ld-linux-x86-64.so.2
cp "$LOADER" $ROOT/lib64/
{ ldd $RB; ldd $LIBR; find $ROOT/rubylib -name '*.so' -exec ldd {} \; ; } 2>/dev/null \
  | grep -oE '/[^ ]+\.so[^ ]*' | sort -u | while read so; do [ -f "$so" ] && cp -n "$so" $ROOT/glibc/ 2>/dev/null || true; done
echo "libs: $(ls $ROOT/glibc | wc -l)"
echo "[4] init"
cat > $ROOT/init <<INIT
#!/busybox sh
/busybox mount -t proc proc /proc 2>/dev/null
export LD_LIBRARY_PATH=/glibc
export RUBYLIB=/rubylib
echo RUBY-BOOT
/ruby --disable-gems /work.rb
/busybox poweroff -f
INIT
chmod +x $ROOT/init
echo "[5] pack"
( cd $ROOT && find . | /tmp/csir/busybox cpio -o -H newc 2>/dev/null | gzip -9 ) > /tmp/rubyir.gz
echo "rubyir.gz: $(du -h /tmp/rubyir.gz | cut -f1)"
