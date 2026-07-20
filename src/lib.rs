pub mod clock;
pub mod device;
pub mod event;
pub mod memory;
pub mod replay;
pub mod seed;
pub mod state;
pub mod toy;
pub mod x86;

pub use clock::VirtualClock;
pub use device::{
    BlockDevice, DeviceError, DeviceScheduler, DiskFault, DiskFaultConfig, DiskStatus,
    EntropyDevice, FaultModel, FaultPolicy, NetFaultConfig, NetworkDevice, BLOCK_SIZE,
};
pub use event::{DetEvent, EventKind, EventQueue};
pub use memory::{GuestMemory, MemoryError};
pub use replay::{checkpoints, first_divergence, Checkpoint};
pub use seed::{Seed256, StreamRng};
pub use state::{MachineSpec, RunIdentity, RunMode, StateDigest};
pub use toy::{ExitReason, ToyCpu, ToyError, ToyMachine, ToySnapshot};
pub use x86::{X86Cpu, X86Error, X86Exit, X86Machine};

#[derive(Debug, Clone)]
pub struct DeterministicCore {
    identity: RunIdentity,
    clock: VirtualClock,
    events: EventQueue,
    guest_rng: StreamRng,
    /// Timer/event scheduling entropy — kept strictly separate from `guest_rng`
    /// so scheduler timing never couples to guest-visible entropy consumption
    /// (GOLDEN_TARGET §6/§20 stream independence).
    scheduler_rng: StreamRng,
    /// P4 seed-exploration entropy: drives bounded jitter on interrupt-delivery
    /// instants so a seed STEERS thread interleavings. A separate stream so it
    /// never perturbs guest entropy or the toy event scheduler.
    sched_jitter_rng: StreamRng,
    digest: StateDigest,
}

impl DeterministicCore {
    pub fn new(identity: RunIdentity) -> Self {
        let mut scheduler_rng = identity.seed().derive("event-scheduler");
        let mut events = EventQueue::new();
        let timer_delay = 10 + scheduler_rng.next_bounded(90);
        events.push(DetEvent::new(
            timer_delay,
            EventKind::TimerInterrupt { vector: 32 },
        ));

        Self {
            guest_rng: identity.seed().derive("guest-rng"),
            sched_jitter_rng: identity.seed().derive("sched-jitter"),
            scheduler_rng,
            clock: VirtualClock::new(identity.machine().ticks_per_instruction),
            events,
            digest: StateDigest::new(),
            identity,
        }
    }

    /// Draw a bounded scheduling-jitter value (instructions) from the dedicated
    /// seed-derived P4 stream. Returns 0 when `max` is 0 (jitter disabled), so the
    /// stream is not consumed and the baseline run is byte-for-byte unchanged.
    pub fn draw_sched_jitter(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.sched_jitter_rng.next_bounded(max + 1)
    }

    pub fn step_instruction(&mut self) -> Vec<DetEvent> {
        self.clock.step_instruction();
        self.digest.mix_u64(self.clock.instruction_count());

        let mut delivered = Vec::new();
        while let Some(event) = self.events.pop_ready(self.clock.instruction_count()) {
            self.apply_event(&event);
            delivered.push(event);
        }
        delivered
    }

    pub fn run_instructions(&mut self, count: u64) {
        for _ in 0..count {
            self.step_instruction();
        }
    }

    pub fn fast_forward_to_next_event(&mut self) -> Option<Vec<DetEvent>> {
        let next_instruction = self.events.next_instruction()?;
        self.clock.fast_forward_to_instruction(next_instruction);

        let mut delivered = Vec::new();
        while let Some(event) = self.events.pop_ready(self.clock.instruction_count()) {
            self.apply_event(&event);
            delivered.push(event);
        }
        Some(delivered)
    }

    pub fn schedule_event(&mut self, event: DetEvent) {
        self.events.push(event);
    }

    pub fn clock(&self) -> &VirtualClock {
        &self.clock
    }

    /// Whether any future event is scheduled (something that could wake an idle
    /// CPU). The periodic timer keeps this true during normal operation.
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Instruction count of the next scheduled event, if any.
    pub fn next_event_instruction(&self) -> Option<u64> {
        self.events.next_instruction()
    }

    /// Fast-forward the clock to `target` and deliver every event due by then.
    /// Used to advance idle time to an external deadline (e.g. an APIC timer).
    pub fn advance_clock_to(&mut self, target: u64) -> Vec<DetEvent> {
        if target > self.clock.instruction_count() {
            self.clock.fast_forward_to_instruction(target);
        }
        let mut delivered = Vec::new();
        while let Some(event) = self.events.pop_ready(self.clock.instruction_count()) {
            self.apply_event(&event);
            delivered.push(event);
        }
        delivered
    }

    pub fn digest(&self) -> u64 {
        self.digest.finish()
    }

    /// Complete deterministic-core state for snapshots and divergence checks:
    /// the running digest, the clock, the guest RNG cursor, and the pending
    /// event queue (GOLDEN_TARGET section 19, rule 14).
    pub fn full_state_digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.digest.finish());
        digest.mix_u64(self.clock.instruction_count());
        digest.mix_u64(self.guest_rng.state());
        digest.mix_u64(self.scheduler_rng.state());
        digest.mix_u64(self.sched_jitter_rng.state());
        digest.mix_u64(self.events.digest());
        digest.finish()
    }

    pub fn identity(&self) -> &RunIdentity {
        &self.identity
    }

    fn apply_event(&mut self, event: &DetEvent) {
        self.digest.mix_u64(event.at_instruction);
        match &event.kind {
            EventKind::TimerInterrupt { vector } => {
                self.digest.mix_bytes(b"timer");
                self.digest.mix_u64(*vector as u64);
                // Reschedule from the dedicated scheduler stream, NOT guest_rng
                // (Review 4 M3: avoid coupling scheduler timing to guest entropy).
                let delay = 50 + self.scheduler_rng.next_bounded(100);
                self.events.push(DetEvent::new(
                    self.clock.instruction_count() + delay,
                    EventKind::TimerInterrupt { vector: *vector },
                ));
            }
            EventKind::DiskCompletion { request_id, status } => {
                self.digest.mix_bytes(b"disk");
                self.digest.mix_u64(*request_id);
                self.digest.mix_u64(*status as u64);
            }
            EventKind::NetworkPacket { link_id, packet_id } => {
                self.digest.mix_bytes(b"net");
                self.digest.mix_u64(*link_id);
                self.digest.mix_u64(*packet_id);
            }
            EventKind::GuestEntropy { stream_id, bytes } => {
                self.digest.mix_bytes(b"entropy");
                self.digest.mix_u64(*stream_id);
                self.digest.mix_bytes(bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(seed_hex: &str) -> RunIdentity {
        RunIdentity::new(
            RunMode::Pure,
            Seed256::from_hex(seed_hex).unwrap(),
            MachineSpec::x86_64_single_vcpu(),
        )
    }

    #[test]
    fn sched_jitter_is_seed_deterministic_and_off_by_default() {
        // P4: the scheduling-jitter stream must be reproducible per seed, vary by
        // seed, and consume NOTHING (no state change) when disabled so the proven
        // baseline run is byte-for-byte unchanged.
        let id = identity("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        let mut a = DeterministicCore::new(id.clone());
        let mut b = DeterministicCore::new(id);
        let sa: Vec<u64> = (0..32).map(|_| a.draw_sched_jitter(5000)).collect();
        let sb: Vec<u64> = (0..32).map(|_| b.draw_sched_jitter(5000)).collect();
        assert_eq!(sa, sb, "same seed → identical jitter sequence");
        assert!(sa.iter().any(|&j| j != 0), "jitter actually varies");
        assert!(sa.iter().all(|&j| j <= 5000), "jitter is bounded by max");

        let mut c = DeterministicCore::new(identity(
            "ff0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        ));
        let sc: Vec<u64> = (0..32).map(|_| c.draw_sched_jitter(5000)).collect();
        assert_ne!(sa, sc, "different seed → different jitter sequence");

        // max = 0 draws nothing: the digest is unchanged across many calls.
        let mut d = DeterministicCore::new(identity(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        ));
        let before = d.full_state_digest();
        for _ in 0..100 {
            assert_eq!(d.draw_sched_jitter(0), 0);
        }
        assert_eq!(d.full_state_digest(), before, "disabled jitter consumes no entropy");
    }

    #[test]
    fn same_seed_produces_same_digest() {
        let id = identity("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        let mut a = DeterministicCore::new(id.clone());
        let mut b = DeterministicCore::new(id);

        a.run_instructions(1_000);
        b.run_instructions(1_000);

        assert_eq!(a.clock().instruction_count(), b.clock().instruction_count());
        assert_eq!(a.clock().virtual_ticks(), b.clock().virtual_ticks());
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn different_seed_changes_digest() {
        let mut a = DeterministicCore::new(identity(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        ));
        let mut b = DeterministicCore::new(identity(
            "f00102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        ));

        a.run_instructions(1_000);
        b.run_instructions(1_000);

        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn idle_fast_forward_delivers_next_event() {
        let id = identity("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        let mut core = DeterministicCore::new(id);
        core.schedule_event(DetEvent::new(
            5,
            EventKind::DiskCompletion {
                request_id: 9,
                status: 0,
            },
        ));

        let delivered = core.fast_forward_to_next_event().unwrap();

        assert_eq!(core.clock().instruction_count(), 5);
        assert!(matches!(
            delivered.first().unwrap().kind,
            EventKind::DiskCompletion { request_id: 9, .. }
        ));
    }
}
