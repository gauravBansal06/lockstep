// Concurrent stateful service — Rust (std::thread + Mutex + TcpListener). See ../SPEC.md.
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex};
use std::thread;

const N: usize = 16;
const START: i64 = 150;
const CLIENTS: usize = 6;
const PERCLIENT: usize = 500;
const SNAPIVL: i64 = 137;
const PORT: &str = "127.0.0.1:9000";

struct State {
    bal: [i64; N],
    applied: i64,
    failed: i64,
    disk: Option<File>,
}

fn snapshot(s: &State) {
    if let Some(d) = &s.disk {
        let mut buf = [0u8; 512];
        buf[0..4].copy_from_slice(&0x5346_4244u32.to_le_bytes());
        buf[4..8].copy_from_slice(&(s.applied as u32).to_le_bytes());
        for i in 0..N {
            buf[8 + i * 8..16 + i * 8].copy_from_slice(&s.bal[i].to_le_bytes());
        }
        let _ = d.write_at(&buf, 0);
        let _ = d.sync_all();
    }
}
fn apply(st: &Mutex<State>, frm: usize, to: usize, amt: i64) -> bool {
    let mut s = st.lock().unwrap();
    if s.bal[frm] >= amt {
        s.bal[frm] -= amt;
        s.bal[to] += amt;
        s.applied += 1;
        if s.applied % SNAPIVL == 0 {
            snapshot(&s);
        }
        true
    } else {
        s.failed += 1;
        false
    }
}
fn handle(st: Arc<Mutex<State>>, stream: TcpStream) {
    let mut r = BufReader::new(stream.try_clone().unwrap());
    let mut w = stream;
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let p: Vec<i64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if p.len() < 3 {
            break;
        }
        let ok = apply(&st, p[0] as usize, p[1] as usize, p[2]);
        if w.write_all(if ok { b"1\n" } else { b"0\n" }).is_err() {
            break;
        }
    }
}
fn main() {
    println!("CSVC-START rust");
    let disk = OpenOptions::new().read(true).write(true).open("/dev/vda").ok();
    let st = Arc::new(Mutex::new(State { bal: [START; N], applied: 0, failed: 0, disk }));
    let listener = TcpListener::bind(PORT).expect("bind");
    {
        let st = st.clone();
        thread::spawn(move || {
            for s in listener.incoming() {
                if let Ok(s) = s {
                    let st = st.clone();
                    thread::spawn(move || handle(st, s));
                }
            }
        });
    }
    let mut handles = vec![];
    for c in 0..CLIENTS {
        handles.push(thread::spawn(move || {
            let mut state: i64 = (c as i64 + 1) * 1000 + 1;
            let mut next = || {
                state = (state * 16807) % 2147483647;
                state
            };
            if let Ok(mut conn) = TcpStream::connect(PORT) {
                let mut r = BufReader::new(conn.try_clone().unwrap());
                for _ in 0..PERCLIENT {
                    let frm = next() % N as i64;
                    let to = next() % N as i64;
                    let amt = next() % 90 + 10;
                    if write!(conn, "{} {} {}\n", frm, to, amt).is_err() {
                        break;
                    }
                    let mut line = String::new();
                    if r.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    let s = st.lock().unwrap();
    snapshot(&s);
    let total: i64 = s.bal.iter().sum();
    let mut mb = [0u8; N * 8];
    for i in 0..N {
        mb[i * 8..i * 8 + 8].copy_from_slice(&s.bal[i].to_le_bytes());
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in mb.iter() {
        h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    let mut disktotal: i64 = -1;
    if let Some(d) = &s.disk {
        let mut rb = [0u8; 512];
        if d.read_at(&mut rb, 0).is_ok()
            && u32::from_le_bytes(rb[0..4].try_into().unwrap()) == 0x5346_4244
        {
            disktotal = 0;
            for i in 0..N {
                disktotal += i64::from_le_bytes(rb[8 + i * 8..16 + i * 8].try_into().unwrap());
            }
        }
    }
    println!("CSVC-APPLIED {} CSVC-FAILED {}", s.applied, s.failed);
    println!("INV-CONSERVED {} (total={} want=2400)", total == N as i64 * START, total);
    println!("INV-DISK-MATCH {} (diskTotal={})", disktotal == total, disktotal);
    println!("CSVC-HASH {:016x}", h);
    println!("CSVC-EXIT 0");
}
