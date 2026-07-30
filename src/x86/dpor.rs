//! SMP Stage 3b — Dynamic Partial-Order Reduction (DPOR) over vCPU schedules.
//!
//! Where PCT (Stage 3a) samples schedules probabilistically, DPOR explores them
//! *systematically*: run an execution, record every shared-memory access, find the
//! racing (conflicting, unordered) access pairs, and re-run with the schedule
//! steered to flip each pair's order — skipping every interleaving that only
//! reorders independent steps. On this machine a "thread" is a vCPU, a "step" is
//! one scheduled instruction, and replay needs no state saving at all: the machine
//! is deterministic, so re-running a schedule prefix from reset reproduces the
//! pre-state of any step bit-for-bit.
//!
//! This module holds the trace data model (this slice), and the race analysis +
//! exploration driver (later slices). Scope: sequential consistency (`tso_max == 0`);
//! DPOR over TSO store buffers would need buffered stores modelled as extra
//! visibility events and is a documented follow-up.

use std::collections::BTreeSet;

/// One RAM data access executed by a vCPU during a single instruction: the
/// post-translation physical range and whether it was a write. Device MMIO is
/// excluded (devices are machine-level and their handlers run serialized), as is
/// instruction fetch (code is not data; SMC races are out of scope). A
/// page-straddling access is recorded per byte by the slow path, so the ranges
/// here are always physically contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemAccess {
    pub pa: u64,
    pub bytes: u8,
    pub write: bool,
}

impl MemAccess {
    pub fn load(pa: u64, bytes: u8) -> Self {
        Self { pa, bytes, write: false }
    }

    pub fn store(pa: u64, bytes: u8) -> Self {
        Self { pa, bytes, write: true }
    }

    /// Two accesses *conflict* when their physical ranges overlap and at least one
    /// is a write — the DPOR dependency test (reads commute; disjoint accesses
    /// commute; only write-involved overlap makes execution order observable).
    pub fn conflicts(&self, other: &MemAccess) -> bool {
        (self.write || other.write)
            && self.pa < other.pa + other.bytes as u64
            && other.pa < self.pa + self.bytes as u64
    }
}

/// One executed machine step: which vCPU ran, the set of vCPUs *enabled* at the
/// step's pre-state (bit i = vCPU i; DPOR's backtrack rule asks "was q enabled
/// here?"), and the RAM accesses the step performed. A step that idle-fast-forwards
/// a halted-but-wakeable vCPU records no accesses.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepRecord {
    pub vcpu: u32,
    pub enabled: u32,
    pub accesses: Vec<MemAccess>,
}

impl StepRecord {
    /// Step-level dependency: any of this step's accesses conflicts with any of the
    /// other's. (Steps of the same vCPU are always ordered by program order and are
    /// never asked.)
    pub fn conflicts(&self, other: &StepRecord) -> bool {
        self.accesses.iter().any(|a| other.accesses.iter().any(|b| a.conflicts(b)))
    }
}

/// A recorded execution: the schedule actually taken plus each step's accesses.
/// The DPOR driver replays `schedule()` prefixes on a fresh machine (determinism ⇒
/// bit-identical pre-states) and analyses the records for racing pairs to seed
/// backtrack points.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecTrace {
    pub steps: Vec<StepRecord>,
}

impl ExecTrace {
    /// The schedule this execution took — one vCPU index per machine step,
    /// replayable via `X86Machine::set_sched_script`.
    pub fn schedule(&self) -> Vec<u32> {
        self.steps.iter().map(|s| s.vcpu).collect()
    }
}

/// A *race* in a recorded execution: two conflicting steps by different vCPUs that are
/// not ordered by happens-before through the rest of the trace — i.e. flipping their
/// order is a genuinely different partial order (a different execution DPOR must visit),
/// not just a commuted equivalent. `earlier`/`later` are step indices into the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Race {
    pub earlier: usize,
    pub later: usize,
}

/// Find every race of a recorded execution — the classic DPOR detection (Flanagan &
/// Godefroid) with vector clocks. Happens-before is the transitive closure of per-vCPU
/// program order plus conflict order among earlier steps. For each step `j` by vCPU `q`
/// and each other vCPU `p`, the *last* `p`-step conflicting with `j` races with it unless
/// it already happens-before `j` through intermediate steps (e.g. `p` wrote, some vCPU
/// read that value, and `q` conflicted with the reader before `j`).
///
/// Deterministic by construction: plain index iteration, no hashing. Cost is O(steps ×
/// vCPUs × accesses) with a backward scan per (step, vCPU) — fine at litmus scale, which
/// is DPOR's domain (whole-boot traces are what PCT/Stage-1 sampling is for).
pub fn find_races(trace: &ExecTrace, ncpus: usize) -> Vec<Race> {
    // cv[p] = vCPU p's clock vector: cv[p][r] counts the steps of vCPU r that happen-before
    // p's next step. step_clock[i] = the executing vCPU's clock just after step i;
    // step_ord[i] = step i's 1-based ordinal among its own vCPU's steps.
    let mut cv: Vec<Vec<u64>> = vec![vec![0; ncpus]; ncpus];
    let mut step_clock: Vec<Vec<u64>> = Vec::with_capacity(trace.steps.len());
    let mut step_ord: Vec<u64> = Vec::with_capacity(trace.steps.len());
    // Step indices per vCPU, in order — scanned backwards for "last conflicting step".
    let mut steps_of: Vec<Vec<usize>> = vec![Vec::new(); ncpus];
    let mut races = Vec::new();

    for (j, sj) in trace.steps.iter().enumerate() {
        let q = sj.vcpu as usize;
        assert!(q < ncpus, "trace names vCPU {q} but ncpus is {ncpus}");
        // For each other vCPU: locate its last step conflicting with j. Race-check against
        // q's clock BEFORE j's own dependency edges are merged, then merge that step's
        // clock (it dominates the clocks of that vCPU's earlier conflicting steps, so one
        // join per vCPU closes the dependency edges into j).
        for p in 0..ncpus {
            if p == q {
                continue;
            }
            let last_conflict = steps_of[p].iter().rev().copied().find(|&i| {
                trace.steps[i].conflicts(sj)
            });
            if let Some(i) = last_conflict {
                if step_ord[i] > cv[q][p] {
                    races.push(Race { earlier: i, later: j });
                }
                for r in 0..ncpus {
                    cv[q][r] = cv[q][r].max(step_clock[i][r]);
                }
            }
        }
        // Program order: j is now q's latest step.
        cv[q][q] += 1;
        step_ord.push(cv[q][q]);
        step_clock.push(cv[q].clone());
        steps_of[q].push(j);
    }
    races
}

/// One frame of the DPOR search stack: the scheduling decision taken before step `k` of
/// the current execution, the vCPUs enabled at that (replayable, deterministic) state,
/// and the alternatives still to try from it. Frames for the shared prefix persist across
/// executions — their `backtrack`/`done` sets accumulate; frames past a backtrack point
/// are discarded and rebuilt from the next execution's trace.
#[derive(Debug, Clone)]
struct Frame {
    chosen: u32,
    enabled: u32,
    backtrack: BTreeSet<u32>,
    done: BTreeSet<u32>,
}

/// The result of a [`dpor_explore`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DporReport {
    /// Maximal executions actually run.
    pub executions: u64,
    /// Distinct final memory digests, each with one witness schedule that reaches it
    /// (replayable via `X86Machine::set_sched_script`), in discovery order.
    pub outcomes: Vec<(u64, Vec<u32>)>,
    /// Races posted to backtrack sets across all executions (a diagnostic).
    pub races_seen: u64,
    /// True when exploration ran to natural completion; false when an execution hit
    /// `max_steps` before halting or the `max_execs` budget stopped the search.
    pub complete: bool,
}

/// Systematically explore the schedule space of a (small, terminating) multi-vCPU
/// program with Dynamic Partial-Order Reduction. Each iteration replays a schedule
/// prefix on a FRESH machine from `build` (determinism ⇒ bit-identical revisit — no
/// snapshots needed), continues round-robin to completion while recording the trace,
/// then posts each race's alternative vCPU to the backtrack set of the race's earlier
/// step and re-runs from the deepest untried alternative. Interleavings that only
/// reorder independent (non-conflicting) steps are never revisited — that is the
/// reduction.
///
/// Requirements: the builder must yield identical machines (same seed/program/vCPUs)
/// with every exploration knob OFF — sequential consistency (`tso_max == 0`; DPOR over
/// TSO buffers is a follow-up) and no seed-driven scheduling (the driver owns the
/// schedule). `max_steps` must cover the program's longest execution for the
/// exploration to be exhaustive (`complete` reports this).
pub fn dpor_explore<F: FnMut() -> super::exec::X86Machine>(
    mut build: F,
    max_steps: u64,
    max_execs: u64,
) -> DporReport {
    let mut frames: Vec<Frame> = Vec::new();
    let mut prefix: Vec<u32> = Vec::new();
    let mut outcomes: Vec<(u64, Vec<u32>)> = Vec::new();
    let mut executions = 0u64;
    let mut races_seen = 0u64;
    let mut complete = true;

    loop {
        // Run one maximal execution: forced prefix, then the round-robin default.
        let mut m = build();
        let ncpus = m.vcpus();
        assert!(m.cpu().tso_max == 0, "DPOR requires sequential consistency (tso_max == 0)");
        m.set_dpor_record(true);
        m.set_sched_script(prefix.clone());
        let exit = m.run(max_steps).expect("DPOR workload must not fault");
        if exit == super::exec::X86Exit::InstructionLimit {
            complete = false; // not a maximal execution: the budget cut it short
        }
        executions += 1;
        let trace = m.take_exec_trace();
        assert!(
            trace.steps.len() >= prefix.len(),
            "the forced prefix must replay in full (recorded {} of {} steps)",
            trace.steps.len(),
            prefix.len()
        );
        let digest = m.memory().digest();
        if !outcomes.iter().any(|(d, _)| *d == digest) {
            outcomes.push((digest, trace.schedule()));
        }

        // Extend the frame stack over the defaulted suffix of this execution.
        for st in trace.steps.iter().skip(frames.len()) {
            frames.push(Frame {
                chosen: st.vcpu,
                enabled: st.enabled,
                backtrack: BTreeSet::new(),
                done: BTreeSet::from([st.vcpu]),
            });
        }

        // Post each race to the backtrack set of its earlier step's pre-state: try the
        // later step's vCPU there (or, if it was not enabled there, every enabled vCPU).
        for race in find_races(&trace, ncpus) {
            races_seen += 1;
            let f = &mut frames[race.earlier];
            let q = trace.steps[race.later].vcpu;
            if f.enabled & (1 << q) != 0 {
                f.backtrack.insert(q);
            } else {
                for r in 0..ncpus as u32 {
                    if f.enabled & (1 << r) != 0 {
                        f.backtrack.insert(r);
                    }
                }
            }
        }

        // Backtrack: deepest frame with an untried alternative; truncate deeper frames,
        // switch its choice, and re-run. No such frame ⇒ the space is covered.
        let next = (0..frames.len()).rev().find_map(|k| {
            frames[k]
                .backtrack
                .difference(&frames[k].done)
                .next()
                .copied()
                .map(|c| (k, c))
        });
        match next {
            None => break,
            Some((k, c)) => {
                if executions >= max_execs {
                    complete = false;
                    break;
                }
                frames.truncate(k + 1);
                frames[k].chosen = c;
                frames[k].done.insert(c);
                prefix = frames.iter().map(|f| f.chosen).collect();
            }
        }
    }
    DporReport { executions, outcomes, races_seen, complete }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a step by `vcpu` with the given accesses (enabled mask irrelevant here).
    fn step(vcpu: u32, accesses: &[MemAccess]) -> StepRecord {
        StepRecord { vcpu, enabled: 0b11, accesses: accesses.to_vec() }
    }

    #[test]
    fn access_conflicts() {
        let r = MemAccess::load(0x800, 8);
        let w = MemAccess::store(0x800, 8);
        let w2 = MemAccess::store(0x804, 4); // overlaps [0x800,0x808)
        let far = MemAccess::store(0x900, 8);
        assert!(!r.conflicts(&r), "read/read never conflicts");
        assert!(r.conflicts(&w) && w.conflicts(&r), "read/write overlap conflicts");
        assert!(w.conflicts(&w2), "partial overlap conflicts");
        assert!(!w.conflicts(&far) && !r.conflicts(&far), "disjoint never conflicts");
        // Adjacent-but-touching ranges do not overlap.
        let adj = MemAccess::store(0x808, 8);
        assert!(!w.conflicts(&adj), "adjacent ranges are disjoint");
    }

    #[test]
    fn races_in_sb_shape() {
        // The Store-Buffering shape: T0 {W x; R y}, T1 {W y; R x}. In THIS trace order only
        // the adjacent (Ry, Wy) pair races: the (Wx, Rx) pair is transitively ordered
        // through it (Wx →po Ry →dep Wy →po Rx), so reversing it is not reachable without
        // first reversing (Ry, Wy) — DPOR reports just the adjacent race, and the flipped
        // execution it then generates is where the second pair gets its turn to race. (The
        // exhaustive-coverage proof lives in the machine-level driver tests in exec.rs.)
        let x = 0x800;
        let y = 0x808;
        let t = ExecTrace {
            steps: vec![
                step(0, &[MemAccess::store(x, 8)]), // 0: T0 W x
                step(0, &[MemAccess::load(y, 8)]),  // 1: T0 R y
                step(1, &[MemAccess::store(y, 8)]), // 2: T1 W y
                step(1, &[MemAccess::load(x, 8)]),  // 3: T1 R x
            ],
        };
        assert_eq!(
            find_races(&t, 2),
            vec![Race { earlier: 1, later: 2 }],
            "the adjacent unordered pair races; the transitively ordered pair does not"
        );
    }

    #[test]
    fn happens_before_filters_transitive_order() {
        // T0 writes x; T1 reads x (dependency edge 0→1) then writes x. The (0, 2) pair
        // conflicts but is ordered through step 1 (0 →conflict 1 →program-order 2), so the
        // ONLY race is (0, 1) — the vector-clock filter must drop (0, 2).
        let x = 0x800;
        let t = ExecTrace {
            steps: vec![
                step(0, &[MemAccess::store(x, 8)]), // 0: T0 W x
                step(1, &[MemAccess::load(x, 8)]),  // 1: T1 R x   (races with 0)
                step(1, &[MemAccess::store(x, 8)]), // 2: T1 W x   (ordered after 0 via 1)
            ],
        };
        assert_eq!(find_races(&t, 2), vec![Race { earlier: 0, later: 1 }]);
    }

    #[test]
    fn races_last_conflicting_step_only() {
        // T0 does W x then W y; T1's step accesses BOTH x and y. Only T0's LAST conflicting
        // step (the y write) can race — the earlier x write is ordered before it by program
        // order, hence before T1's step through the direct edge from the later one.
        let x = 0x800;
        let y = 0x808;
        let t = ExecTrace {
            steps: vec![
                step(0, &[MemAccess::store(x, 8)]),                          // 0: T0 W x
                step(0, &[MemAccess::store(y, 8)]),                          // 1: T0 W y
                step(1, &[MemAccess::load(x, 8), MemAccess::load(y, 8)]),    // 2: T1 R x,y
            ],
        };
        assert_eq!(find_races(&t, 2), vec![Race { earlier: 1, later: 2 }]);
    }

    #[test]
    fn no_races_without_writes_or_overlap() {
        let x = 0x800;
        let t = ExecTrace {
            steps: vec![
                step(0, &[MemAccess::load(x, 8)]),           // read-read: no race
                step(1, &[MemAccess::load(x, 8)]),
                step(0, &[MemAccess::store(0x900, 8)]),      // disjoint writes: no race
                step(1, &[MemAccess::store(0x908, 8)]),
                step(1, &[]),                                 // idle fast-forward: no accesses
            ],
        };
        assert!(find_races(&t, 2).is_empty());
    }

    #[test]
    fn three_vcpus_transitive_chain() {
        // T0 W x → T1 reads x, writes y → T2 reads y, then reads x. T2's x-read is ordered
        // after T0's write through the T1 chain, so the races are exactly (0,1) and (2,3):
        // the two direct dependency handoffs, not the transitive (0, 4) pair.
        let x = 0x800;
        let y = 0x808;
        let t = ExecTrace {
            steps: vec![
                step(0, &[MemAccess::store(x, 8)]), // 0: T0 W x
                step(1, &[MemAccess::load(x, 8)]),  // 1: T1 R x   (race with 0)
                step(1, &[MemAccess::store(y, 8)]), // 2: T1 W y
                step(2, &[MemAccess::load(y, 8)]),  // 3: T2 R y   (race with 2)
                step(2, &[MemAccess::load(x, 8)]),  // 4: T2 R x   (HB-ordered after 0)
            ],
        };
        assert_eq!(
            find_races(&t, 3),
            vec![Race { earlier: 0, later: 1 }, Race { earlier: 2, later: 3 }]
        );
    }
}
