// Distributed bank-service — Rust (std::thread + Mutex + TcpListener), the concurrent
// svc.rs SPLIT into two roles over a real network. See ../c/distsvc.c.
//   distsvc server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
//   distsvc client <peer_ip>  → node A: 6 workers × 500 requests to the server
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const N: usize = 16;
const START: i64 = 150;
const CLIENTS: usize = 6;
const PERCLIENT: usize = 500;

struct State {
    bal: [i64; N],
    applied: i64,
    failed: i64,
    done_conns: usize,
}

fn apply(st: &Mutex<State>, frm: usize, to: usize, amt: i64) -> bool {
    let mut s = st.lock().unwrap();
    if s.bal[frm] >= amt {
        s.bal[frm] -= amt;
        s.bal[to] += amt;
        s.applied += 1;
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
    st.lock().unwrap().done_conns += 1;
}
fn run_server() {
    println!("CSVC-START rust-server");
    let st = Arc::new(Mutex::new(State { bal: [START; N], applied: 0, failed: 0, done_conns: 0 }));
    let listener = TcpListener::bind("0.0.0.0:9000").expect("bind");
    println!("CSVC-LISTEN 9000");
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
    for _ in 0..4000 {   // ~40s virtual-time cap so a partition can't hang
        if st.lock().unwrap().done_conns >= CLIENTS {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let s = st.lock().unwrap();
    let total: i64 = s.bal.iter().sum();
    let mut mb = [0u8; N * 8];
    for i in 0..N {
        mb[i * 8..i * 8 + 8].copy_from_slice(&s.bal[i].to_le_bytes());
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in mb.iter() {
        h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    println!("CSVC-APPLIED {} CSVC-FAILED {}", s.applied, s.failed);
    println!("INV-CONSERVED {} (total={} want=2400)", total == N as i64 * START, total);
    println!("CSVC-HASH {:016x}", h);
    println!("CSVC-EXIT 0");
}
fn run_client(peer: &str) {
    println!("CSVC-START rust-client");
    let addr = format!("{}:9000", peer);
    let mut handles = vec![];
    for c in 0..CLIENTS {
        let addr = addr.clone();
        handles.push(thread::spawn(move || {
            let mut state: i64 = (c as i64 + 1) * 1000 + 1;
            let mut next = || {
                state = (state * 16807) % 2147483647;
                state
            };
            let mut conn = None;
            for _ in 0..80 {   // the server may still be booting
                if let Ok(s) = TcpStream::connect(&addr) {
                    conn = Some(s);
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            if let Some(mut conn) = conn {
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
    println!("CSVC-CLIENT-DONE");
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "server" {
        run_server();
    } else if args.len() >= 3 && args[1] == "client" {
        run_client(&args[2]);
    } else {
        println!("usage: distsvc server | distsvc client <peer_ip>");
        std::process::exit(1);
    }
}
