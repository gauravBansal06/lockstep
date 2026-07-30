use lockstep::x86::{X86Exit, X86Machine};
use lockstep::{
    DeterministicCore, MachineSpec, RunIdentity, RunMode, Seed256, ToyMachine,
};

const DEFAULT_SEED: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed_hex = parse_seed_arg(&args).unwrap_or_else(|| DEFAULT_SEED.to_string());
    if args.iter().any(|arg| arg == "--toy") {
        run_toy(&seed_hex);
        return;
    }
    if args.iter().any(|a| a == "--net2") {
        run_net2(&seed_hex, &args);
        return;
    }
    if let Some(path) = parse_value_arg(&args, "--boot") {
        let cmdline = parse_value_arg(&args, "--cmdline")
            .unwrap_or_else(|| "console=ttyS0".to_string());
        let budget = parse_value_arg(&args, "--max")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(200_000_000);
        let initrd = parse_value_arg(&args, "--initrd");
        let disk = parse_value_arg(&args, "--disk");
        let ram_mb = parse_value_arg(&args, "--ram-mb")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256);
        let netdev = args.iter().any(|a| a == "--netdev");
        let smp = parse_value_arg(&args, "--smp")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1)
            .max(1);
        run_boot(
            &seed_hex,
            &path,
            &cmdline,
            budget,
            initrd.as_deref(),
            disk.as_deref(),
            ram_mb,
            netdev,
            smp,
        );
        return;
    }
    if let Some(path) = parse_value_arg(&args, "--difftest256") {
        run_difftest256(&seed_hex, &path);
        return;
    }
    if let Some(path) = parse_value_arg(&args, "--difftest-x87") {
        run_difftest_x87(&seed_hex, &path);
        return;
    }
    if let Some(path) = parse_value_arg(&args, "--difftest") {
        run_difftest(&seed_hex, &path);
        return;
    }
    if let Some(path) = parse_value_arg(&args, "--decode-probe") {
        run_decode_probe(&path);
        return;
    }

    let seed = Seed256::from_hex(&seed_hex).expect("seed must be 64 hex characters");
    let identity = RunIdentity::new(RunMode::Pure, seed, MachineSpec::x86_64_single_vcpu());
    let mut core = DeterministicCore::new(identity);

    core.run_instructions(10_000);

    println!("mode=pure");
    println!("arch=x86_64");
    println!("instructions={}", core.clock().instruction_count());
    println!("virtual_ticks={}", core.clock().virtual_ticks());
    println!("state_digest={:016x}", core.digest());
}

fn run_toy(seed_hex: &str) {
    let seed = Seed256::from_hex(seed_hex).expect("seed must be 64 hex characters");
    let mut machine = ToyMachine::new(seed, 512, &sample_program()).expect("toy machine boots");
    let exit = machine.run(200).expect("toy machine runs");

    println!("mode=pure");
    println!("machine=toy");
    println!("exit={exit:?}");
    println!("pc={}", machine.cpu().pc);
    println!("regs={:?}", machine.cpu().regs);
    println!("ram_digest={:016x}", machine.memory().digest());
    println!("state_digest={:016x}", machine.digest());
}

/// Attempt to boot a Linux x86 bzImage and report exactly how far it got — the
/// boot probe that drives empirical opcode/device discovery.
fn run_boot(
    seed_hex: &str,
    path: &str,
    cmdline: &str,
    budget: u64,
    initrd_path: Option<&str>,
    disk_path: Option<&str>,
    ram_mb: usize,
    netdev: bool,
    smp: u32,
) {
    let seed = Seed256::from_hex(seed_hex).expect("seed must be 64 hex characters");
    let image = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let initrd = initrd_path
        .map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("read initrd {p}: {e}")));
    let disk = disk_path.map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("read disk {p}: {e}")));
    let ram: usize = ram_mb * 1024 * 1024;

    // Tell the kernel where the virtio-mmio devices live (size@base:irq): the disk
    // at 0xd0000000:5 and (with --netdev) the NIC at 0xd0002000:6.
    let mut cmdline = cmdline.to_string();
    if disk.is_some() && !cmdline.contains("0xd0000000") {
        cmdline.push_str(" virtio_mmio.device=0x200@0xd0000000:5");
    }
    if netdev && !cmdline.contains("0xd0002000") {
        cmdline.push_str(" virtio_mmio.device=0x200@0xd0002000:6");
    }
    let cmdline = cmdline.as_str();

    println!("booting {} ({} bytes), cmdline={:?}", path, image.len(), cmdline);
    if let Some(ref ird) = initrd {
        println!("initrd {} ({} bytes)", initrd_path.unwrap(), ird.len());
    }
    if let Some(ref d) = disk {
        println!("disk {} ({} bytes, {} sectors)", disk_path.unwrap(), d.len(), d.len() / 512);
    }
    let mut machine = match X86Machine::boot_bzimage(seed, ram, &image, cmdline, initrd.as_deref(), disk) {
        Ok(m) => m,
        Err(e) => {
            println!("load failed: {e:?}");
            return;
        }
    };

    // --smp N (N>1): attach N-1 parked APs and re-emit the MADT so the guest kernel
    // enumerates and INIT/SIPI-brings-up all N processors. At N==1 this whole block is
    // skipped, so a uniprocessor boot's memory + instruction stream are untouched
    // (the SMP mechanism is inert until a second vCPU actually exists).
    if smp > 1 {
        use lockstep::x86::build_acpi_smp;
        machine.set_vcpus(smp as usize);
        if let Err(e) = build_acpi_smp(machine.memory_mut(), smp) {
            println!("MADT (smp={smp}) build failed: {e:?}");
            return;
        }
        // SF_APTRACE=N: dump each AP's first N instructions (cs/rip/base/cr0/bytes) — debug
        // the real-mode trampoline. Set on every AP (index>0); the BSP is untraced.
        if let Ok(n) = std::env::var("SF_APTRACE") {
            let n: u64 = n.trim().parse().unwrap_or(300);
            machine.set_ap_trace(n);
            eprintln!("SF_APTRACE: tracing {} instrs on each of {} AP(s)", n, smp - 1);
        }
        println!("SMP: {smp} vCPUs (1 BSP + {} AP(s)), MADT advertises APIC IDs 0..{smp}", smp - 1);
    }

    // --netdev: attach a virtio-net NIC (fixed MAC 52:54:00:12:34:56). Optional
    // seeded fault policy via SF_NET_FAULT=seed,drop_permille,jitter. The Phase-1a
    // backend is a deterministic loopback reflector (in the machine step loop).
    if netdev {
        use lockstep::x86::{NetFaultPolicy, VirtioNet};
        let mut net = VirtioNet::new([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        if let Ok(spec) = std::env::var("SF_NET_FAULT") {
            let p: Vec<u64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if p.len() == 3 {
                net.set_fault_policy(NetFaultPolicy::new(p[0], p[1] as u32, p[2]));
                println!("virtio-net fault policy: seed={} drop={}‰ jitter={}", p[0], p[1], p[2]);
            }
        }
        machine.io_mut().virtio_net = Some(net);
        println!("virtio-net attached at 0xd0002000 (MAC 52:54:00:12:34:56)");
    }

    if let Ok(d) = std::env::var("SF_DUMP") {
        if let Ok(addr) = u64::from_str_radix(d.trim().trim_start_matches("0x"), 16) {
            let mut hex = String::new();
            for i in 0..192u64 {
                hex.push_str(&format!("{:02x}", machine.peek_code(addr + i).unwrap_or(0)));
            }
            eprintln!("DUMP {addr:#x}: {hex}");
        }
    }
    if std::env::var("SF_EXCTRACE").is_ok() {
        machine.cpu_mut().trace_exc = true;
    }
    if std::env::var("SF_RSITRACE").is_ok() {
        machine.cpu_mut().trace_rsi = true;
    }
    if std::env::var("SF_SYSTRACE").is_ok() {
        machine.cpu_mut().trace_sys = true;
    }
    if std::env::var("SF_NETTRACE").is_ok() {
        machine.cpu_mut().trace_net = true;
    }
    if std::env::var("SF_MEMTRACE").is_ok() {
        machine.cpu_mut().trace_mem = true;
    }
    let itrace_rip = std::env::var("SF_ITRACE_RIP").ok()
        .map(|r| u64::from_str_radix(r.trim_start_matches("0x"), 16).expect("SF_ITRACE_RIP hex"));
    let itrace_fault = std::env::var("SF_ITRACE_FAULT_CR2").ok()
        .map(|r| u64::from_str_radix(r.trim_start_matches("0x"), 16).expect("SF_ITRACE_FAULT_CR2 hex"));
    if itrace_rip.is_some() || itrace_fault.is_some() {
        let n = std::env::var("SF_ITRACE_N").ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(50_000);
        let cpu = machine.cpu_mut();
        cpu.trace_itrace = true;
        cpu.itrace_rip = itrace_rip.unwrap_or(0);
        cpu.itrace_fault_cr2 = itrace_fault.unwrap_or(0);
        cpu.itrace_n = n;
        if std::env::var("SF_ITRACE_REGS").is_ok() {
            cpu.itrace_regs = true;
        }
        if std::env::var("SF_ITRACE_ALLCPL").is_ok() {
            cpu.itrace_allcpl = true;
        }
    }
    if let Ok(rip) = std::env::var("SF_REGDUMP_RIP") {
        let rip = u64::from_str_radix(rip.trim_start_matches("0x"), 16).expect("SF_REGDUMP_RIP hex");
        machine.cpu_mut().regdump_rip = rip;
    }
    if let Ok(occ) = std::env::var("SF_REGDUMP_OCC") {
        machine.cpu_mut().regdump_occ = occ.parse::<u64>().unwrap_or(0);
    }
    if let Ok(a) = std::env::var("SF_WATCHADDR") {
        let a = u64::from_str_radix(a.trim_start_matches("0x"), 16).expect("SF_WATCHADDR hex");
        machine.cpu_mut().watch_addr = a;
        if let Ok(v) = std::env::var("SF_WATCHVAL") {
            machine.cpu_mut().watch_val = u64::from_str_radix(v.trim_start_matches("0x"), 16).expect("SF_WATCHVAL hex");
        }
    }
    if std::env::var("SF_DELAYTRACE").is_ok() {
        machine.cpu_mut().trace_delay = true;
    }
    if let Ok(a) = std::env::var("SF_FNWATCH") {
        if let Ok(addr) = u64::from_str_radix(a.trim().trim_start_matches("0x"), 16) {
            machine.cpu_mut().fn_watch = addr;
        }
    }
    if let Ok(v) = std::env::var("SF_VWATCH") {
        if let Ok(val) = u64::from_str_radix(v.trim().trim_start_matches("0x"), 16) {
            machine.cpu_mut().vwatch = val;
        }
    }
    if let Ok(r) = std::env::var("SF_WVA") {
        // "lo-hi" hex virtual-address range to watch for stores.
        if let Some((lo, hi)) = r.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (
                u64::from_str_radix(lo.trim().trim_start_matches("0x"), 16),
                u64::from_str_radix(hi.trim().trim_start_matches("0x"), 16),
            ) {
                machine.cpu_mut().wwatch_lo = lo;
                machine.cpu_mut().wwatch_hi = hi;
            }
        }
    }
    if let Ok(w) = std::env::var("SF_WATCH") {
        if let Ok(addr) = u64::from_str_radix(w.trim().trim_start_matches("0x"), 16) {
            machine.cpu_mut().watch = addr;
            eprintln!("watchpoint set on {addr:#x}");
        }
    }
    if std::env::var("SF_PROFILE").is_ok() {
        machine.set_profile(true);
    }
    // P4 seed exploration: SF_SCHED_JITTER=<max instr> defers each scheduler tick
    // by a bounded seed-derived jitter, so different --seed values explore
    // different thread interleavings (each still fully deterministic for its seed).
    if let Ok(j) = std::env::var("SF_SCHED_JITTER") {
        if let Ok(max) = j.trim().parse::<u64>() {
            machine.set_sched_jitter(max);
            eprintln!("SF_SCHED_JITTER: max={max} instr (seed-driven schedule perturbation)");
        }
    }
    // SMP Stage-1 seed-driven interleaving exploration: SF_SMP_SCHED=<max quantum instr>
    // makes the vCPU scheduler run each core for a seed-drawn quantum (1..=max) then switch to
    // a seed-chosen runnable core, so different --seed values realise different reproducible
    // interleavings (exposing SMP data races). 0/unset = fixed round-robin (byte-identical).
    if let Ok(q) = std::env::var("SF_SMP_SCHED") {
        if let Ok(max) = q.trim().parse::<u64>() {
            machine.set_smp_sched(max);
            eprintln!("SF_SMP_SCHED: max_quantum={max} instr (seed-driven SMP interleaving)");
        }
    }
    // SMP Stage-2 x86-TSO store buffers: SF_SMP_TSO=<max linger instr> makes each core's stores
    // buffer and become globally visible after a seed-drawn delay, so store→load reordering
    // (Dekker/Store-Buffering) is reachable per seed. 0/unset = write-through (SC, byte-identical).
    if let Ok(t) = std::env::var("SF_SMP_TSO") {
        if let Ok(max) = t.trim().parse::<u64>() {
            machine.set_tso(max);
            eprintln!("SF_SMP_TSO: max_linger={max} instr (x86-TSO store buffers)");
        }
    }
    // SMP Stage-3 PCT: SF_SMP_PCT=<depth>[,<k_bound>] runs a Probabilistic-Concurrency-Testing
    // schedule for the given target bug depth — highest-priority runnable vCPU + (depth-1)
    // seed-drawn priority-change points, giving a ≥1/(n·k^(depth-1)) chance of hitting any
    // depth-d bug per seed. k_bound (change-point instruction range) defaults to --max.
    if let Ok(p) = std::env::var("SF_SMP_PCT") {
        let parts: Vec<&str> = p.trim().split(',').collect();
        if let Ok(depth) = parts[0].parse::<u64>() {
            let k_bound = parts
                .get(1)
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(budget);
            machine.set_pct(depth, k_bound);
            eprintln!("SF_SMP_PCT: depth={depth} k_bound={k_bound} (probabilistic concurrency testing)");
        }
    }
    // SMP Stage-3c race-pair coverage sensor: SF_SMP_COV=<window instr> observes, O(1) per RAM
    // access, cross-vCPU conflicting-access pairs (by code location) within the window and
    // accumulates them in a signature bitmap — the feedback signal for coverage-guided schedule
    // exploration. Observer-only: the run's digests are identical with it on or off; the
    // deterministic `=> race_cov` line below is the per-run report.
    if let Ok(c) = std::env::var("SF_SMP_COV") {
        if let Ok(window) = c.trim().parse::<u64>() {
            if window > 0 {
                machine.set_race_cov(window);
                eprintln!("SF_SMP_COV: window={window} instr (race-pair coverage sensor)");
            }
        }
    }
    // Optional snapshot/restore fidelity demo (GOLDEN_TARGET §19): run to an
    // absolute instruction count, snapshot the WHOLE machine, finish the original,
    // then restore the snapshot and finish again — the two end states must be
    // byte-identical ("rewind to any point and replay" on a real workload).
    let snapshot = std::env::var("SF_SNAPSHOT_AT")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|at| {
            let _ = machine.run_until_instruction(at);
            let snap = machine.snapshot();
            println!(
                "=> SNAPSHOT at instr={} digest={:016x}",
                snap.instruction_count, snap.digest
            );
            snap
        });

    // Wall-clock timing is a DIAGNOSTIC ONLY: measured around the run, printed, and never read
    // back into any guest state, register, memory, or digest — so determinism is unaffected.
    let t0 = std::time::Instant::now();
    let outcome = machine.run(budget);
    let elapsed = t0.elapsed();
    match outcome {
        Ok(X86Exit::Halted) => println!("=> guest HALTED"),
        Ok(X86Exit::InstructionLimit) => println!("=> hit instruction budget ({budget})"),
        Err(e) => println!("=> STOPPED: {e:?}"),
    }
    if let Some(snap) = snapshot {
        let d_end = machine.full_state_digest();
        let end_instr = machine.instruction_count();
        let mut restored = snap.restore();
        let _ = restored.run(budget);
        let ok = restored.full_state_digest() == d_end && restored.instruction_count() == end_instr;
        println!(
            "=> SNAPSHOT-CHECK {} (snapshot@{} → end_digest={:016x}, restored_end={:016x})",
            if ok { "PASS" } else { "FAIL" },
            snap.instruction_count,
            d_end,
            restored.full_state_digest()
        );
    }
    // Whole-machine determinism fingerprint: RAM + all CPU/sys regs + devices.
    // Two runs with identical inputs (seed/kernel/cmdline/initrd/budget) must
    // print byte-identical digests — the Phase-2 boot-twice determinism check.
    println!("=> state_digest={:016x}", machine.full_state_digest());
    // Guest-RAM-only digest: distinguishes real guest divergence (kernel/userspace
    // memory differs) from seed bookkeeping folded into the full digest.
    println!("=> mem_digest={:016x}", machine.memory().digest());
    // Stage-3c coverage report (only when SF_SMP_COV is on): deterministic per run.
    {
        let (pairs, digest) = machine.race_cov_report();
        if !machine.race_cov_bitmap().is_empty() {
            println!("=> race_cov pairs={pairs} bitmap_digest={digest:016x}");
        }
    }
    let instrs = machine.instruction_count();
    println!("=> instructions={instrs}");
    // Throughput diagnostic (wall-clock only; not part of the deterministic state).
    let secs = elapsed.as_secs_f64();
    let mips = if secs > 0.0 { instrs as f64 / secs / 1e6 } else { 0.0 };
    println!("=> elapsed_ms={} mips={mips:.2}", elapsed.as_millis());
    if std::env::var("SF_PROFILE").is_ok() {
        let p = machine.profile_cycles();
        let tot = (p[0] + p[1] + p[2]).max(1);
        println!(
            "=> profile cycles: cpu_step={} ({:.1}%)  core_step={} ({:.1}%)  check_timer={} ({:.1}%)",
            p[0], 100.0 * p[0] as f64 / tot as f64,
            p[1], 100.0 * p[1] as f64 / tot as f64,
            p[2], 100.0 * p[2] as f64 / tot as f64,
        );
    }
    println!("{}", machine.diagnostics());
    let serial = machine.serial().output_string();
    println!("--- serial console ({} bytes) ---", serial.len());
    print!("{serial}");
    if !serial.ends_with('\n') {
        println!();
    }
    // The decompressor stub writes "Decompressing Linux..." to the VGA text buffer
    // at 0xb8000 (char + attribute per cell), independent of any console setting.
    let vga = dump_vga(&machine);
    println!("--- VGA text buffer (0xb8000) ---\n{vga}");
    // Search RAM for the decompressed kernel ELF header ("ELF\x02\x01\x01") to
    // inspect how (and whether) decompression corrupted it.
    let mem = machine.memory();
    let needle = [0x45u8, 0x4c, 0x46, 0x02, 0x01, 0x01];
    let mut found = 0;
    for addr in (0x10_0000u64..(256 * 1024 * 1024)).step_by(1) {
        if mem.read_uint_le(addr, 1).unwrap_or(0) as u8 == needle[0]
            && (1..6).all(|i| mem.read_uint_le(addr + i as u64, 1).unwrap_or(0) as u8 == needle[i])
        {
            let b0 = mem.read_uint_le(addr - 1, 1).unwrap_or(0) as u8;
            let mut hdr = String::new();
            for i in 0..16u64 {
                hdr.push_str(&format!("{:02x} ", mem.read_uint_le(addr - 1 + i, 1).unwrap_or(0) as u8));
            }
            println!("ELF-ish @ {:#x}: byte0={:#04x} (want 0x7f) | {}", addr - 1, b0, hdr.trim_end());
            found += 1;
            if found >= 4 {
                break;
            }
        }
    }
    if found == 0 {
        println!("(no decompressed ELF header found in RAM)");
    }
    // Scan RAM for the kernel printk log buffer — it holds the boot messages even
    // when no console is active. Look for the signature strings, then dump the
    // printable text around them.
    let sig = b"Linux version";
    for addr in (0x10_0000u64..(256 * 1024 * 1024)).step_by(1) {
        if (0..sig.len()).all(|i| mem.read_uint_le(addr + i as u64, 1).unwrap_or(0) as u8 == sig[i]) {
            // Dump the printk ring-buffer text region: printable runs become text,
            // non-printable record headers become newlines, so each message shows.
            let mut txt = String::new();
            let mut last_print = true;
            for i in 0..16384u64 {
                let b = mem.read_uint_le(addr + i, 1).unwrap_or(0) as u8;
                if (0x20..0x7f).contains(&b) {
                    txt.push(b as char);
                    last_print = true;
                } else if last_print {
                    txt.push('\n');
                    last_print = false;
                }
            }
            println!("--- printk log buffer @ {addr:#x} ---");
            for line in txt.lines().filter(|l| l.len() > 3) {
                println!("{line}");
            }
            break;
        }
    }
}

// ---- Phase 1b: two-guest deterministic network harness ----

fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn drop_frame(state: &mut u64, drop_permille: u32) -> bool {
    if drop_permille == 0 {
        return false;
    }
    (splitmix64_step(state) % 1000) < drop_permille as u64
}

/// `--net-fault seed,drop_permille[,partition_start,partition_end]` (partition in
/// total virtual instructions) → (prng_state, drop_permille, partition_window).
fn parse_switch_fault(args: &[String]) -> (u64, u32, Option<(u64, u64)>) {
    if let Some(spec) = parse_value_arg(args, "--net-fault") {
        let p: Vec<u64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let state = p.first().copied().unwrap_or(1) ^ 0x1234_5678_9abc_def0;
        let drop = p.get(1).copied().unwrap_or(0) as u32;
        let part = if p.len() >= 4 { Some((p[2], p[3])) } else { None };
        return (state, drop, part);
    }
    (1, 0, None)
}

/// `--net-jitter <seed>[,<max_instr>]` → per-frame RX-delivery jitter (reorder) for
/// the two-guest switch. Unlike CPU jitter (SF_SCHED_JITTER, which explores THREAD
/// interleavings), this perturbs the ORDER in which I/O completions land — the only
/// nondeterminism an EVENT-LOOP runtime (asyncio / libuv, single-threaded) exposes:
/// epoll returns readable sockets in a different order, so the loop processes them
/// differently. Seed-driven, so each seed replays exactly; different seeds explore.
fn parse_net_jitter(args: &[String]) -> Option<(u64, u64)> {
    let spec = parse_value_arg(args, "--net-jitter")?;
    let p: Vec<u64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let seed = p.first().copied().unwrap_or(1);
    let max = p.get(1).copied().unwrap_or(50_000);
    Some((seed, max))
}

/// Boot TWO guests (each with a virtio-net NIC, reflector OFF), step them
/// round-robin by a fixed instruction quantum, and run a deterministic SWITCH
/// between quanta: drain each NIC's TX, apply a seeded fault policy (drop /
/// partition), deliver survivors to the peer's RX. Fully deterministic — a seed
/// replays the exact interleaving + fault schedule. The real distributed-DST
/// substrate (client in A, server in B, faults on the wire).
fn run_net2(seed_hex: &str, args: &[String]) {
    use lockstep::x86::{NetFaultPolicy, VirtioNet, X86Exit};
    let seed = Seed256::from_hex(seed_hex).expect("seed must be 64 hex characters");
    let kernel = parse_value_arg(args, "--boot").expect("--net2 needs --boot <kernel>");
    let initrd_path = parse_value_arg(args, "--initrd").expect("--net2 needs --initrd <initrd>");
    let base_cmd =
        parse_value_arg(args, "--cmdline").unwrap_or_else(|| "nokaslr console=ttyS0".into());
    let ram_mb = parse_value_arg(args, "--ram-mb").and_then(|s| s.parse().ok()).unwrap_or(512usize);
    let budget = parse_value_arg(args, "--max")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(40_000_000_000);
    let quantum = parse_value_arg(args, "--quantum")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200_000);
    let cmd_a = parse_value_arg(args, "--cmdline-a").unwrap_or_default();
    let cmd_b = parse_value_arg(args, "--cmdline-b").unwrap_or_default();
    let (mut sw_state, drop_permille, part) = parse_switch_fault(args);

    let image = std::fs::read(&kernel).unwrap_or_else(|e| panic!("read {kernel}: {e}"));
    let initrd =
        std::fs::read(&initrd_path).unwrap_or_else(|e| panic!("read initrd {initrd_path}: {e}"));
    let ram = ram_mb * 1024 * 1024;

    // Optional deterministic entropy source (virtio-rng at 0xd0001000, seed-driven).
    // Some runtimes (.NET) block on getrandom() until the kernel CRNG seeds; without a
    // fast source the two guests seed at very different virtual times and the client
    // races ahead of the server. The device is always present in the machine — this
    // just tells the guest kernel to bind it. Deterministic: bytes derive from the seed.
    let rng_tok = if args.iter().any(|a| a == "--rngdev") {
        " virtio_mmio.device=0x200@0xd0001000:7"
    } else {
        ""
    };
    let boot = |extra: &str, mac: [u8; 6]| -> X86Machine {
        let cmd = format!("{base_cmd} {extra}{rng_tok} virtio_mmio.device=0x200@0xd0002000:6");
        eprintln!("net2: booting guest cmdline={cmd:?}");
        let mut m = X86Machine::boot_bzimage(seed, ram, &image, &cmd, Some(&initrd), None)
            .expect("net2 boot failed");
        let mut net = VirtioNet::new(mac);
        net.set_external_backend();
        m.io_mut().virtio_net = Some(net);
        m
    };
    let mut a = boot(&cmd_a, [0x52, 0x54, 0x00, 0x00, 0x00, 0x01]);
    let mut b = boot(&cmd_b, [0x52, 0x54, 0x00, 0x00, 0x00, 0x02]);
    // I/O-completion jitter (event-loop exploration): each guest reorders its own
    // incoming frames from a distinct seed stream, deterministically.
    if let Some((jseed, jmax)) = parse_net_jitter(args) {
        a.io_mut().virtio_net.as_mut().unwrap().set_fault_policy(NetFaultPolicy::new(jseed, 0, jmax));
        b.io_mut().virtio_net.as_mut().unwrap().set_fault_policy(NetFaultPolicy::new(jseed ^ 0xA5A5_5A5A, 0, jmax));
        eprintln!("net2: io-jitter seed={jseed} max={jmax}");
    }
    eprintln!(
        "net2: two guests up, quantum={quantum} drop={drop_permille}permille partition={part:?}"
    );

    let (mut a_done, mut b_done, mut total) = (false, false, 0u64);
    let (mut fwd_ab, mut fwd_ba, mut dropped) = (0u64, 0u64, 0u64);
    while (!a_done || !b_done) && total < budget {
        if !a_done {
            a_done = matches!(a.run(quantum), Ok(X86Exit::Halted));
        }
        if !b_done {
            b_done = matches!(b.run(quantum), Ok(X86Exit::Halted));
        }
        total = total.saturating_add(2 * quantum);
        let partitioned = part.map_or(false, |(s, e)| total >= s && total < e);
        // A -> B
        let bt = b.instruction_count();
        let frames = a.io_mut().virtio_net.as_mut().unwrap().take_tx_frames();
        for f in frames {
            if partitioned || drop_frame(&mut sw_state, drop_permille) {
                dropped += 1;
                continue;
            }
            b.io_mut().virtio_net.as_mut().unwrap().deliver_frame(f, bt);
            fwd_ab += 1;
        }
        // B -> A
        let at = a.instruction_count();
        let frames = b.io_mut().virtio_net.as_mut().unwrap().take_tx_frames();
        for f in frames {
            if partitioned || drop_frame(&mut sw_state, drop_permille) {
                dropped += 1;
                continue;
            }
            a.io_mut().virtio_net.as_mut().unwrap().deliver_frame(f, at);
            fwd_ba += 1;
        }
    }
    eprintln!("net2: done A_halted={a_done} B_halted={b_done} total_instr={total}");
    eprintln!("net2: switch forwarded A->B={fwd_ab} B->A={fwd_ba} dropped={dropped}");
    // Per-guest serial (captured in each Uart) — printed separately so the two
    // consoles don't interleave.
    for line in a.serial().output_string().lines() {
        eprintln!("A| {line}");
    }
    for line in b.serial().output_string().lines() {
        eprintln!("B| {line}");
    }
    eprintln!(
        "net2: digest A={:016x} B={:016x}",
        a.full_state_digest(),
        b.full_state_digest()
    );
}

fn dump_vga(machine: &X86Machine) -> String {
    let mem = machine.memory();
    let mut out = String::new();
    for row in 0..25u64 {
        let mut line = String::new();
        for col in 0..80u64 {
            let cell = 0xb_8000 + (row * 80 + col) * 2;
            let ch = mem.read_uint_le(cell, 1).unwrap_or(0) as u8;
            line.push(if (0x20..0x7f).contains(&ch) { ch as char } else { ' ' });
        }
        let trimmed = line.trim_end();
        if !trimmed.is_empty() {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    if out.is_empty() {
        out.push_str("(empty)\n");
    }
    out
}

/// Differential conformance: run each test vector on the emulator and print the
/// result registers + masked flags, identically to the real-CPU C harness.
/// Vector line: `rax rbx rcx rdx flagmask : <hex instruction bytes>`.
/// (GPR indices: RAX=0, RCX=1, RDX=2, RBX=3.)
/// Decode-coverage probe: read hex-byte instructions (one per line, e.g. "c5 fb 2c d2"),
/// decode each via the real decoder, and print only the ones it can't decode (`UD ...`).
/// Used to enumerate ALL missing opcodes across the workload binaries at once, instead of
/// finding them one #UD at a time. Fetches past the line's bytes return 0 (decode-only).
fn run_decode_probe(path: &str) {
    use lockstep::x86::{decode, DecodeMode, X86Error};
    let data = std::fs::read_to_string(path).expect("probe file");
    let mut ud = 0u64;
    let mut ok = 0u64;
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let bytes: Vec<u8> = line
            .split_whitespace()
            .filter_map(|h| u8::from_str_radix(h, 16).ok())
            .collect();
        if bytes.is_empty() {
            continue;
        }
        let mut fetch = |addr: u64| -> Result<u8, X86Error> {
            Ok(bytes.get(addr as usize).copied().unwrap_or(0))
        };
        match decode(&mut fetch, 0, DecodeMode::Bits64) {
            Ok(_) => ok += 1,
            Err(e) => {
                ud += 1;
                println!("UD {line}  ({e:?})");
            }
        }
    }
    eprintln!("decode-probe: {ok} ok, {ud} unsupported");
}

fn run_difftest(seed_hex: &str, path: &str) {
    let seed = Seed256::from_hex(seed_hex).expect("seed must be 64 hex characters");
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (regs_part, code_part) = line.split_once(':').expect("vector needs ':'");
        let vals: Vec<u64> = regs_part
            .split_whitespace()
            .map(|x| u64::from_str_radix(x, 16).expect("hex reg"))
            .collect();
        let mut bytes: Vec<u8> = code_part
            .split_whitespace()
            .map(|h| u8::from_str_radix(h, 16).expect("hex byte"))
            .collect();
        bytes.push(0xf4); // HLT terminator (the C harness uses RET instead)

        let mut m = X86Machine::new(seed, 0x10000, &bytes, 0).unwrap();
        {
            let c = m.cpu_mut();
            c.regs.gpr[0] = vals[0]; // RAX
            c.regs.gpr[3] = vals[1]; // RBX
            c.regs.gpr[1] = vals[2]; // RCX
            c.regs.gpr[2] = vals[3]; // RDX
            c.regs.gpr[4] = 0xf000; // RSP — scratch stack for blobs that ldmxcsr [rsp]
        }
        let mask = vals.get(4).copied().unwrap_or(0);
        let _ = m.run(1_000_000);
        let c = m.cpu();
        println!(
            "{:016x} {:016x} {:016x} {:016x} {:06x}",
            c.regs.gpr[0],
            c.regs.gpr[3],
            c.regs.gpr[1],
            c.regs.gpr[2],
            c.regs.rflags & mask
        );
    }
}

/// 256-bit (YMM) differential mode. Each vector line is 8 input qwords (ymm0
/// then ymm1, low-to-high) `:` instruction bytes. The bytes are a self-contained
/// blob that loads ymm0/ymm1 from `[rdi]`/`[rdi+0x20]`, runs the op, and stores
/// ymm0 to `[rsi]`. We point rdi at a 64-byte input buffer (phys 0x8000) and rsi
/// at a 32-byte output buffer (phys 0x9000) in the flat, unpaged bare machine,
/// then print the 4 output qwords. Mirrors the silicon `cputest 256` harness.
fn run_difftest256(seed_hex: &str, path: &str) {
    const IN_BUF: u64 = 0x8000;
    const OUT_BUF: u64 = 0x9000;
    let seed = Seed256::from_hex(seed_hex).expect("seed must be 64 hex characters");
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (regs_part, code_part) = line.split_once(':').expect("vector needs ':'");
        let vals: Vec<u64> = regs_part
            .split_whitespace()
            .map(|x| u64::from_str_radix(x, 16).expect("hex reg"))
            .collect();
        let mut bytes: Vec<u8> = code_part
            .split_whitespace()
            .map(|h| u8::from_str_radix(h, 16).expect("hex byte"))
            .collect();
        bytes.push(0xf4); // HLT terminator

        let mut m = X86Machine::new(seed, 0x10000, &bytes, 0).unwrap();
        for (i, v) in vals.iter().take(8).enumerate() {
            m.memory_mut()
                .write_u64_le(IN_BUF + (i as u64) * 8, *v)
                .unwrap();
        }
        {
            let c = m.cpu_mut();
            c.regs.gpr[7] = IN_BUF; // RDI -> input buffer
            c.regs.gpr[6] = OUT_BUF; // RSI -> output buffer
        }
        let _ = m.run(1_000_000);
        let out: Vec<u64> = (0..4)
            .map(|i| m.memory().read_u64_le(OUT_BUF + i * 8).unwrap())
            .collect();
        println!(
            "{:016x} {:016x} {:016x} {:016x}",
            out[0], out[1], out[2], out[3]
        );
    }
}

/// x87 80-bit differential. Each vector: `A_mant A_se B_mant B_se : <blob>` where A/B are
/// the two m80 operands (mantissa u64 + sign/exp u16). The blob loads them (`fld tword[rdi]`,
/// `fld tword[rdi+10]`), runs the x87 op, and stores the m80 result to `[rsi]`. We print the
/// 80-bit result (mantissa + sign/exp). Mirrors `cputest x87` on real silicon (x87 basic
/// arithmetic at PC=64 is deterministic across CPUs, so this is bit-validatable).
fn run_difftest_x87(seed_hex: &str, path: &str) {
    const IN_BUF: u64 = 0x8000;
    const OUT_BUF: u64 = 0x9000;
    let seed = Seed256::from_hex(seed_hex).expect("seed must be 64 hex characters");
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (regs_part, code_part) = line.split_once(':').expect("vector needs ':'");
        let vals: Vec<u64> = regs_part
            .split_whitespace()
            .map(|x| u64::from_str_radix(x, 16).expect("hex reg"))
            .collect();
        let mut bytes: Vec<u8> = code_part
            .split_whitespace()
            .map(|h| u8::from_str_radix(h, 16).expect("hex byte"))
            .collect();
        bytes.push(0xf4); // HLT terminator
        let mut m = X86Machine::new(seed, 0x10000, &bytes, 0).unwrap();
        let g = |i: usize| vals.get(i).copied().unwrap_or(0);
        {
            let mem = m.memory_mut();
            mem.write_u64_le(IN_BUF, g(0)).unwrap();
            mem.write_uint_le(IN_BUF + 8, 2, g(1)).unwrap();
            mem.write_u64_le(IN_BUF + 10, g(2)).unwrap();
            mem.write_uint_le(IN_BUF + 18, 2, g(3)).unwrap();
        }
        {
            let c = m.cpu_mut();
            c.regs.gpr[7] = IN_BUF; // RDI
            c.regs.gpr[6] = OUT_BUF; // RSI
        }
        let _ = m.run(1_000_000);
        let rm = m.memory().read_u64_le(OUT_BUF).unwrap();
        let rse = m.memory().read_uint_le(OUT_BUF + 8, 2).unwrap();
        println!("{rm:016x} {rse:04x}");
    }
}

fn parse_seed_arg(args: &[String]) -> Option<String> {
    parse_value_arg(args, "--seed")
}

fn parse_value_arg(args: &[String], flag: &str) -> Option<String> {
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if arg == flag {
            return it.next().cloned();
        }
    }
    None
}

fn sample_program() -> Vec<u8> {
    let mut program = Vec::new();
    program.extend([0x10, 0x00]);
    program.extend(7_u64.to_le_bytes());
    program.extend([0x10, 0x01]);
    program.extend(35_u64.to_le_bytes());
    program.extend([0x11, 0x00, 0x01]);
    program.extend([0x21, 0x00]);
    program.extend(200_u16.to_le_bytes());
    program.extend([0x41, 0x02, 0x02]);
    program.extend([0x40, 0x10, 0x00]);
    program.extend([0x40, 0x11, 0x02]);
    program.extend([0x40, 0x12, 0x00]);
    program.extend([0x00; 80]);
    program.push(0x01);
    program
}
