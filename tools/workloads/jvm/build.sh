#!/bin/bash
set -e
JH=/usr/lib/jvm/java-17-openjdk-amd64
WORK=/tmp/jvmwork
RT=$WORK/jdk
ROOT=$WORK/root
rm -rf $WORK; mkdir -p $WORK $ROOT
cd ~/jvmbuild

echo "[1] compile"
$JH/bin/javac -d $WORK Main.java

echo "[2] jlink minimal runtime (java.base)"
rm -rf $RT
$JH/bin/jlink --module-path $JH/jmods --add-modules java.base \
  --strip-debug --no-header-files --no-man-pages --compress=2 --output $RT

echo "[3] reference hash on HOST jvm"
$RT/bin/java -XX:+UseSerialGC -cp $WORK Main | tee /tmp/jvm_ref.txt

echo "[4] assemble initrd root"
mkdir -p $ROOT/jdk $ROOT/glibc $ROOT/lib64 $ROOT/dev $ROOT/proc $ROOT/sys
cp -a $RT/. $ROOT/jdk/
cp $WORK/Main.class $ROOT/
cp /tmp/csir/busybox $ROOT/busybox
chmod +x $ROOT/busybox

echo "[5] collect glibc loader + all .so deps"
LOADER=$(ldd $ROOT/jdk/bin/java | grep -oE '/lib64/ld-linux[^ ]*' | head -1)
[ -z "$LOADER" ] && LOADER=/lib64/ld-linux-x86-64.so.2
cp "$LOADER" $ROOT/lib64/
# gather deps from java launcher + every JDK native lib
for f in $ROOT/jdk/bin/java $(find $ROOT/jdk/lib -name '*.so'); do
  ldd "$f" 2>/dev/null | grep -oE '/[^ ]+\.so[^ ]*' 
done | sort -u | while read so; do
  [ -f "$so" ] && cp -n "$so" $ROOT/glibc/ 2>/dev/null || true
done
echo "glibc libs bundled: $(ls $ROOT/glibc | wc -l)"

echo "[6] init script"
cat > $ROOT/init <<'INIT'
#!/busybox sh
/busybox mount -t proc proc /proc 2>/dev/null
/busybox mount -t sysfs sys /sys 2>/dev/null
export LD_LIBRARY_PATH=/glibc:/jdk/lib:/jdk/lib/server
export JAVA_HOME=/jdk
echo JVM-BOOT
/jdk/bin/java -XX:+UseSerialGC -XX:-UsePerfData -Xss512k -Xms32m -Xmx128m -cp / Main
/busybox poweroff -f
INIT
chmod +x $ROOT/init

echo "[7] pack cpio.gz"
( cd $ROOT && find . | /tmp/csir/busybox cpio -o -H newc 2>/dev/null | gzip -9 ) > /tmp/jvmir.gz
echo "jvmir.gz size: $(du -h /tmp/jvmir.gz | cut -f1)"
