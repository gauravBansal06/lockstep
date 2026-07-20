# Concurrent stateful service — Perl (ithreads + threads::shared + IO::Socket). See ../SPEC.md.
use strict; use warnings;
use threads;
use threads::shared;
use IO::Socket::INET;

my ($N, $START, $CLIENTS, $PERCLIENT, $SNAPIVL, $PORT) = (16, 150, 6, 500, 137, 9000);
my @bal : shared = ($START) x $N;
my $applied : shared = 0;
my $failed : shared  = 0;
my $lk : shared = 0;

sub snapshot {                      # caller holds lock($lk); fresh handle = ithreads-safe
    open(my $d, '+<', '/dev/vda') or return;
    binmode($d);
    my $buf = pack('VV', 0x53464244, $applied) . pack('q<*', @bal);
    $buf .= "\0" x (512 - length($buf));
    sysseek($d, 0, 0); syswrite($d, $buf, 512);
    close($d);
}
sub apply {
    my ($frm, $to, $amt) = @_;
    lock($lk);
    if ($bal[$frm] >= $amt) {
        $bal[$frm] -= $amt; $bal[$to] += $amt; $applied++;
        snapshot() if $applied % $SNAPIVL == 0;
        return 1;
    }
    $failed++;
    return 0;
}

$| = 1;
print "CSVC-START perl\n";
my $srv = IO::Socket::INET->new(LocalAddr => '127.0.0.1', LocalPort => $PORT,
    Proto => 'tcp', Listen => 16, ReuseAddr => 1) or do { print "CSVC-LISTEN-ERR\n"; exit 1; };
threads->create(sub {
    while (my $conn = $srv->accept()) {
        threads->create(sub {
            my $c = shift;
            while (my $line = <$c>) {
                my ($f, $t, $a) = split ' ', $line;
                print $c (apply($f, $t, $a) ? "1\n" : "0\n");
            }
            close($c);
        }, $conn)->detach();
    }
})->detach();

my @threads;
for my $idx (0 .. $CLIENTS - 1) {
    push @threads, threads->create(sub {
        my $i = shift;
        my $state = ($i + 1) * 1000 + 1;
        my $conn = IO::Socket::INET->new(PeerAddr => '127.0.0.1', PeerPort => $PORT, Proto => 'tcp') or return;
        for (1 .. $PERCLIENT) {
            $state = ($state * 16807) % 2147483647; my $f = $state % $N;
            $state = ($state * 16807) % 2147483647; my $t = $state % $N;
            $state = ($state * 16807) % 2147483647; my $a = $state % 90 + 10;
            print $conn "$f $t $a\n";
            my $r = <$conn>; last unless defined $r;
        }
        close($conn);
    }, $idx);
}
$_->join for @threads;

snapshot();
my $total = 0; $total += $_ for @bal;
my $mb = pack('q<*', @bal);
my $hash;
{
    use integer;                    # native 64-bit wrap for FNV-1a
    $hash = -3750763034362895579;   # 0xcbf29ce484222325
    for my $byte (unpack('C*', $mb)) {
        $hash = ($hash ^ $byte) * 1099511628211;
    }
}
my $hexh = sprintf("%016x", $hash & 0xFFFFFFFFFFFFFFFF);
my $disktotal = -1;
if (open(my $d, '<', '/dev/vda')) {
    binmode($d); my $rb; sysread($d, $rb, 512); close($d);
    if (defined $rb && length($rb) >= 512 && unpack('V', substr($rb, 0, 4)) == 0x53464244) {
        $disktotal = 0; $disktotal += $_ for unpack('q<*', substr($rb, 8, $N * 8));
    }
}
printf("CSVC-APPLIED %d CSVC-FAILED %d\n", $applied, $failed);
printf("INV-CONSERVED %s (total=%d want=2400)\n", ($total == $N * $START) ? "true" : "false", $total);
printf("INV-DISK-MATCH %s (diskTotal=%d)\n", ($disktotal == $total) ? "true" : "false", $disktotal);
printf("CSVC-HASH %s\n", $hexh);
print "CSVC-EXIT 0\n";
