# Deterministic pure-compute Perl (core builtins only: no modules/pragmas).
print "PERL-START\n";
srand(987654321);
my $acc = 0.0;
for my $i (1..200000) { $acc += sqrt($i) * 1.0000001 + ($i % 97)/7.0; }
my @arr;
push @arr, int(rand()*1e9) for 1..5000;
@arr = sort { $a <=> $b } @arr;
my $sum = 0; $sum += $_ for @arr;
my $blob = sprintf("%.12e|%d|%d|%d|%d", $acc, $sum, $arr[0], $arr[2500], $arr[4999]);
$blob .= "|" . join(",", @arr[0..199]);
$blob .= "|" . join("", map { sprintf("%x",$_) } @arr[0..999]);
my $ck1 = unpack("%32C*", $blob);
my $ck2 = unpack("%32C*", scalar reverse($blob));
printf "PERL-HASH %08x%08x\n", $ck1, $ck2;
print "PERL-EXIT 0\n";
