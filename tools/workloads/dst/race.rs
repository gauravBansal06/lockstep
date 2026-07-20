// Racy multi-threaded Rust: threads do a NON-atomic-style read-modify-write on a
// shared counter via a Relaxed atomic (lost updates), so the outcome depends on
// scheduling. Safe Rust, but a genuine data race in effect. DST must pin it.
use std::sync::atomic::{AtomicI64, Ordering::Relaxed};
use std::sync::Arc;
use std::thread;
const THREADS: usize = 8;
const ITERS: i64 = 100_000;
fn main() {
    println!("RACE-START");
    let counter = Arc::new(AtomicI64::new(0));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let c = counter.clone();
        handles.push(thread::spawn(move || {
            let mut s: i64 = 0;
            for _ in 0..ITERS {
                let v = c.load(Relaxed);      // racy load
                c.store(v + 1, Relaxed);      // racy store: lost-update window
                s = s.wrapping_mul(1000003).wrapping_add(c.load(Relaxed));
            }
            s
        }));
    }
    let mut h: u64 = counter.load(Relaxed) as u64;
    for handle in handles {
        let s = handle.join().unwrap();
        h = h.wrapping_mul(1000003).wrapping_add(s as u64);
    }
    println!("RACE-COUNTER {} of {}", counter.load(Relaxed), THREADS as i64 * ITERS);
    println!("RACE-HASH {:016x}", h);
    println!("RACE-EXIT 0");
}
