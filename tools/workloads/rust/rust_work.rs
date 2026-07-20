// Deterministic pure-compute Rust workload (gate correctness vs host oracle).
fn main() {
    println!("RUST-START");
    const FNV: u64 = 1099511628211;
    let mut h: u64 = 14695981039346656037;
    let mut acc: f64 = 0.0;
    for i in 1..=200000u64 {
        acc += (i as f64).sqrt() * 1.0000001 + ((i % 97) as f64) / 7.0;
        let v = (acc * 1000.0) as i64 ^ (i.wrapping_mul(2654435761)) as i64;
        h = (h ^ (v as u64 & 0xff)).wrapping_mul(FNV);
    }
    // deterministic PRNG fill + sort
    let mut a = [0u32; 5000];
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for x in a.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = (s >> 33) as u32;
    }
    a.sort();
    for &x in a.iter() { h = (h ^ x as u64).wrapping_mul(FNV); }
    let mut blob = String::new();
    for i in 0..1000 { blob.push_str(&format!("{:x}", a[i])); }
    for b in blob.bytes() { h = (h ^ b as u64).wrapping_mul(FNV); }
    println!("RUST-HASH {:016x}", h);
    println!("RUST-EXIT 0");
}
