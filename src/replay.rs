//! Replay, state-hash checkpoints, and divergence detection.
//!
//! These tools sit on top of the deterministic toy machine and implement the
//! debugging half of GOLDEN_TARGET section 19: record full-state digests at a
//! fixed virtual-time cadence, then compare two runs to find the first
//! instruction count at which they diverge. Because the engine is deterministic,
//! two runs with the same RunIdentity must produce identical checkpoint chains;
//! any divergence is either a different identity or a determinism regression.

use crate::toy::{ToyError, ToyMachine};

/// A full-state digest captured at a specific virtual instruction count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    pub instruction_count: u64,
    pub digest: u64,
}

/// Run `machine` forward, recording a checkpoint at instruction 0 and then every
/// `step` instructions until `max_instructions` is reached or the guest halts.
/// `step` must be non-zero.
pub fn checkpoints(
    machine: &mut ToyMachine,
    step: u64,
    max_instructions: u64,
) -> Result<Vec<Checkpoint>, ToyError> {
    assert!(step > 0, "checkpoint step must be non-zero");

    let mut out = vec![Checkpoint {
        instruction_count: machine.instruction_count(),
        digest: machine.full_state_digest(),
    }];

    let mut target = machine.instruction_count().saturating_add(step);
    while target <= max_instructions {
        let exit = machine.run_until_instruction(target)?;
        out.push(Checkpoint {
            instruction_count: machine.instruction_count(),
            digest: machine.full_state_digest(),
        });
        if machine.cpu().halted {
            let _ = exit;
            break;
        }
        target = target.saturating_add(step);
    }

    Ok(out)
}

/// First instruction count at which two checkpoint chains disagree, or `None`
/// if they match over their common length. A mismatch in either the digest or
/// the instruction-count cadence counts as divergence.
pub fn first_divergence(a: &[Checkpoint], b: &[Checkpoint]) -> Option<u64> {
    for (ca, cb) in a.iter().zip(b.iter()) {
        if ca.instruction_count != cb.instruction_count || ca.digest != cb.digest {
            return Some(ca.instruction_count);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DiskFaultConfig, FaultPolicy, NetFaultConfig};
    use crate::seed::Seed256;

    fn seed(hex_prefix: &str) -> Seed256 {
        let mut hex = hex_prefix.to_string();
        while hex.len() < 64 {
            hex.push('0');
        }
        Seed256::from_hex(&hex).unwrap()
    }

    fn program() -> Vec<u8> {
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
        // Loop a few MMIO reads so there is live state across checkpoints.
        program.extend([0x41, 0x00, 0x00]);
        program.extend([0x41, 0x03, 0x01]);
        program.extend([0x30]);
        program.extend(0_u16.to_le_bytes());
        program
    }

    fn faulty() -> FaultPolicy {
        FaultPolicy {
            disk: DiskFaultConfig {
                eio_percent: 30,
                timeout_percent: 20,
                nospace_percent: 10,
                timeout_delay: 500,
            },
            net: NetFaultConfig {
                loss_percent: 40,
                duplicate_percent: 20,
                reorder_percent: 20,
                ..NetFaultConfig::disabled()
            },
        }
    }

    #[test]
    fn snapshot_restore_is_equivalent_to_no_snapshot() {
        let prog = program();
        let mut machine = ToyMachine::new(seed("01"), 512, &prog).unwrap();

        machine.run_until_instruction(500).unwrap();
        let snapshot = machine.snapshot();

        // Continue the original run.
        machine.run_until_instruction(1_500).unwrap();
        let continued = machine.full_state_digest();

        // Restore from the snapshot and run the same distance.
        let mut restored = snapshot.restore();
        restored.run_until_instruction(1_500).unwrap();

        assert_eq!(snapshot.instruction_count(), 500);
        assert_eq!(restored.full_state_digest(), continued);
    }

    #[test]
    fn same_seed_runs_never_diverge() {
        let prog = program();
        let mut a = ToyMachine::new(seed("01"), 512, &prog).unwrap();
        let mut b = ToyMachine::new(seed("01"), 512, &prog).unwrap();

        let ca = checkpoints(&mut a, 100, 2_000).unwrap();
        let cb = checkpoints(&mut b, 100, 2_000).unwrap();

        assert_eq!(ca, cb);
        assert_eq!(first_divergence(&ca, &cb), None);
        assert!(ca.len() > 1);
    }

    #[test]
    fn different_seed_runs_diverge() {
        let prog = program();
        let mut a = ToyMachine::new(seed("01"), 512, &prog).unwrap();
        let mut b = ToyMachine::new(seed("02"), 512, &prog).unwrap();

        let ca = checkpoints(&mut a, 100, 2_000).unwrap();
        let cb = checkpoints(&mut b, 100, 2_000).unwrap();

        assert!(first_divergence(&ca, &cb).is_some());
    }

    #[test]
    fn fault_injection_diverges_from_clean_run() {
        let prog = program();
        let mut clean = ToyMachine::new(seed("01"), 512, &prog).unwrap();
        let mut faulty_machine =
            ToyMachine::with_faults(seed("01"), 512, &prog, faulty()).unwrap();

        let clean_checks = checkpoints(&mut clean, 100, 2_000).unwrap();
        let faulty_checks = checkpoints(&mut faulty_machine, 100, 2_000).unwrap();

        assert!(first_divergence(&clean_checks, &faulty_checks).is_some());
    }
}
