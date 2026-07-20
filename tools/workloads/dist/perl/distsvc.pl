# Distributed bank-service — Perl (ithreads + threads::shared + IO::Socket), the concurrent
# svc.pl SPLIT into two roles over a real network. See ../c/distsvc.c.
#   distsvc.pl server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
#   distsvc.pl client <peer_ip>  → node A: 6 workers × 500 requests to the server
use strict; use warnings;
use threads;
use threads::shared;
use IO::Socket::INET;

my ($N, $START, $CLIENTS, $PERCLIENT, $PORT) = (16, 150, 6, 500, 9000);
my @bal : shared = ($START) x $N;
my $applied : shared = 0;
my $failed : shared  = 0;
my $done : shared    = 0;
my $lk : shared = 0;

sub apply {
    my ($frm, $to, $amt) = @_;
    lock($lk);
    if ($bal[$frm] >= $amt) {
        $bal[$frm] -= $amt; $bal[$to] += $amt; $applied++;
        return 1;
    }
    $failed++;
    return 0;
}

sub run_server {
    $| = 1;
    print "CSVC-START perl-server\n";
    my $srv = IO::Socket::INET->new(LocalAddr => '0.0.0.0', LocalPort => $PORT,
        Proto => 'tcp', Listen => 16, ReuseAddr => 1) or do { print "CSVC-LISTEN-ERR\n"; exit 1; };
    print "CSVC-LISTEN 9000\n";
    threads->create(sub {
        while (my $conn = $srv->accept()) {
            threads->create(sub {
                my $c = shift;
                while (my $line = <$c>) {
                    my ($f, $t, $a) = split ' ', $line;
                    print $c (apply($f, $t, $a) ? "1\n" : "0\n");
                }
                close($c);
                { lock($lk); $done++; }
            }, $conn)->detach();
        }
    })->detach();
    for (1 .. 4000) {           # ~40s virtual-time cap so a partition can't hang
        my $d; { lock($lk); $d = $done; }
        last if $d >= $CLIENTS;
        select(undef, undef, undef, 0.01);
    }
    my $total = 0; $total += $_ for @bal;
    my $mb = pack('q<*', @bal);
    my $hash;
    {
        use integer;
        $hash = -3750763034362895579;   # 0xcbf29ce484222325
        for my $byte (unpack('C*', $mb)) {
            $hash = ($hash ^ $byte) * 1099511628211;
        }
    }
    my $hexh = sprintf("%016x", $hash & 0xFFFFFFFFFFFFFFFF);
    printf("CSVC-APPLIED %d CSVC-FAILED %d\n", $applied, $failed);
    printf("INV-CONSERVED %s (total=%d want=2400)\n", ($total == $N * $START) ? "true" : "false", $total);
    printf("CSVC-HASH %s\n", $hexh);
    print "CSVC-EXIT 0\n";
}

sub run_client {
    my ($peer) = @_;
    $| = 1;
    print "CSVC-START perl-client\n";
    my @threads;
    for my $idx (0 .. $CLIENTS - 1) {
        push @threads, threads->create(sub {
            my $i = shift;
            my $state = ($i + 1) * 1000 + 1;
            my $conn;
            for (1 .. 80) {     # the server may still be booting
                $conn = IO::Socket::INET->new(PeerAddr => $peer, PeerPort => $PORT, Proto => 'tcp');
                last if $conn;
                select(undef, undef, undef, 0.1);
            }
            return unless $conn;
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
    print "CSVC-CLIENT-DONE\n";
}

if (@ARGV >= 1 && $ARGV[0] eq 'server') {
    run_server();
} elsif (@ARGV >= 2 && $ARGV[0] eq 'client') {
    run_client($ARGV[1]);
} else {
    print "usage: distsvc.pl server | distsvc.pl client <peer_ip>\n";
    exit 1;
}
