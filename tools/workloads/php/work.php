<?php
echo "PHP-START\n";
mt_srand(987654321);
// 1) IEEE float accumulation (sqrt is correctly-rounded; no transcendentals)
$acc = 0.0;
for ($i = 1; $i <= 200000; $i++) {
    $acc += sqrt((float)$i) * 1.0000001 + (float)($i % 97) / 7.0;
}
$parts = [];
$parts[] = sprintf("%.12e", $acc);
// 2) deterministic PRNG fill + sort
$arr = [];
for ($i = 0; $i < 5000; $i++) { $arr[] = mt_rand(); }
sort($arr);
$parts[] = implode(",", array_slice($arr, 0, 100));
$parts[] = (string)array_sum($arr);
// 3) string building + hashing
$s = "";
for ($i = 0; $i < 2000; $i++) { $s .= dechex($arr[$i]); }
$parts[] = (string)strlen($s);
$parts[] = md5($s);
$parts[] = hash('crc32b', $s);
$blob = implode("|", $parts);
echo "PHP-HASH " . substr(hash('sha256', $blob), 0, 16) . "\n";
echo "PHP-EXIT 0\n";
