use crate::event::{DetEvent, EventKind};
use crate::seed::{Seed256, StreamRng};
use crate::state::StateDigest;

#[derive(Debug, Clone)]
pub struct EntropyDevice {
    rng: StreamRng,
}

impl EntropyDevice {
    pub fn new(seed: &Seed256, label: &str) -> Self {
        Self {
            rng: seed.derive(label),
        }
    }

    pub fn read_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    pub fn rng_cursor(&self) -> u64 {
        self.rng.state()
    }

    pub fn request_event(
        &mut self,
        now_instruction: u64,
        stream_id: u64,
        scheduler: &mut DeviceScheduler,
    ) -> DetEvent {
        let mut bytes = vec![0_u8; 8];
        self.rng.fill_bytes(&mut bytes);
        DetEvent::new(
            now_instruction + scheduler.entropy_delay(),
            EventKind::GuestEntropy { stream_id, bytes },
        )
    }
}

#[derive(Debug, Clone)]
pub struct BlockDevice {
    blocks: Vec<[u8; BLOCK_SIZE]>,
    completed_requests: u64,
    eio_completions: u64,
    timeout_completions: u64,
    nospace_completions: u64,
}

pub const BLOCK_SIZE: usize = 512;

impl BlockDevice {
    pub fn new(block_count: usize) -> Self {
        Self {
            blocks: vec![[0; BLOCK_SIZE]; block_count],
            completed_requests: 0,
            eio_completions: 0,
            timeout_completions: 0,
            nospace_completions: 0,
        }
    }

    pub fn read_block(&self, block_index: u64) -> Result<[u8; BLOCK_SIZE], DeviceError> {
        let index = self.block_index(block_index)?;
        Ok(self.blocks[index])
    }

    pub fn write_block(
        &mut self,
        block_index: u64,
        data: &[u8; BLOCK_SIZE],
    ) -> Result<(), DeviceError> {
        let index = self.block_index(block_index)?;
        self.blocks[index] = *data;
        Ok(())
    }

    /// Build the deterministic completion event for a request.
    ///
    /// Disk *data* is already deterministic; this decides completion *timing*
    /// and *fault status* from the seed-derived fault model, never from host I/O.
    /// The timing scheduler is always advanced by exactly one draw per request so
    /// the timing stream stays in lockstep regardless of which fault is chosen; a
    /// timeout fault adds a long completion delay on top of that base delay.
    pub fn submit_completion(
        &self,
        now_instruction: u64,
        request_id: u64,
        scheduler: &mut DeviceScheduler,
        faults: &mut FaultModel,
    ) -> DetEvent {
        let fault = faults.disk_fault();
        let base_delay = scheduler.disk_delay();
        let extra_delay = if matches!(fault, DiskFault::Timeout) {
            faults.policy().disk.timeout_delay
        } else {
            0
        };
        DetEvent::new(
            now_instruction + base_delay + extra_delay,
            EventKind::DiskCompletion {
                request_id,
                status: fault.status() as u8,
            },
        )
    }

    pub fn complete(&mut self, status: u8) {
        match DiskStatus::from_u8(status) {
            DiskStatus::Ok => self.completed_requests = self.completed_requests.wrapping_add(1),
            DiskStatus::Eio => self.eio_completions = self.eio_completions.wrapping_add(1),
            DiskStatus::Timeout => {
                self.timeout_completions = self.timeout_completions.wrapping_add(1)
            }
            DiskStatus::NoSpace => {
                self.nospace_completions = self.nospace_completions.wrapping_add(1)
            }
        }
    }

    pub fn completed_requests(&self) -> u64 {
        self.completed_requests
    }

    pub fn eio_completions(&self) -> u64 {
        self.eio_completions
    }

    pub fn timeout_completions(&self) -> u64 {
        self.timeout_completions
    }

    pub fn nospace_completions(&self) -> u64 {
        self.nospace_completions
    }

    pub fn digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.blocks.len() as u64);
        digest.mix_u64(self.completed_requests);
        digest.mix_u64(self.eio_completions);
        digest.mix_u64(self.timeout_completions);
        digest.mix_u64(self.nospace_completions);
        for block in &self.blocks {
            digest.mix_bytes(block);
        }
        digest.finish()
    }

    fn block_index(&self, block_index: u64) -> Result<usize, DeviceError> {
        let index =
            usize::try_from(block_index).map_err(|_| DeviceError::BlockOutOfRange(block_index))?;
        if index >= self.blocks.len() {
            return Err(DeviceError::BlockOutOfRange(block_index));
        }
        Ok(index)
    }
}

#[derive(Debug, Clone)]
pub struct NetworkDevice {
    link_id: u64,
    delivered_packets: u64,
    dropped_packets: u64,
    duplicated_packets: u64,
    reordered_packets: u64,
    partitioned_packets: u64,
    /// Instruction count until which the link is partitioned (down). While
    /// `now < partition_until` every submitted packet is dropped.
    partition_until: u64,
}

impl NetworkDevice {
    pub fn new(link_id: u64) -> Self {
        Self {
            link_id,
            delivered_packets: 0,
            dropped_packets: 0,
            duplicated_packets: 0,
            reordered_packets: 0,
            partitioned_packets: 0,
            partition_until: 0,
        }
    }

    /// Deterministically decide how a submitted packet is delivered.
    ///
    /// Returns the (possibly empty) set of delivery events. The seed-derived
    /// fault model decides loss, duplication, reorder, and partition; the timing
    /// scheduler decides base latency. Order of fault rolls is fixed so the fault
    /// stream stays deterministic:
    /// partition -> loss -> reorder -> duplicate.
    pub fn submit_packet(
        &mut self,
        now_instruction: u64,
        packet_id: u64,
        scheduler: &mut DeviceScheduler,
        faults: &mut FaultModel,
    ) -> Vec<DetEvent> {
        // Partition: already-down link drops unconditionally; otherwise a fresh
        // partition may begin now and take this packet down with it.
        if now_instruction < self.partition_until {
            self.partitioned_packets = self.partitioned_packets.wrapping_add(1);
            return Vec::new();
        }
        if faults.net_partition_start() {
            self.partition_until = now_instruction + faults.policy().net.partition_window;
            self.partitioned_packets = self.partitioned_packets.wrapping_add(1);
            return Vec::new();
        }

        // Loss.
        if faults.net_loss() {
            self.dropped_packets = self.dropped_packets.wrapping_add(1);
            return Vec::new();
        }

        // Latency, optionally pushed out by a reorder fault so a later packet can
        // overtake this one in the deterministic event queue.
        let mut delay = scheduler.network_delay();
        if faults.net_reorder() {
            delay = delay.wrapping_add(faults.policy().net.reorder_extra_delay);
            self.reordered_packets = self.reordered_packets.wrapping_add(1);
        }

        let mut events = vec![DetEvent::new(
            now_instruction + delay,
            EventKind::NetworkPacket {
                link_id: self.link_id,
                packet_id,
            },
        )];

        // Duplication: a second copy arrives slightly later.
        if faults.net_duplicate() {
            self.duplicated_packets = self.duplicated_packets.wrapping_add(1);
            let dup_delay = delay.wrapping_add(faults.policy().net.duplicate_extra_delay);
            events.push(DetEvent::new(
                now_instruction + dup_delay,
                EventKind::NetworkPacket {
                    link_id: self.link_id,
                    packet_id,
                },
            ));
        }

        events
    }

    pub fn deliver(&mut self) {
        self.delivered_packets = self.delivered_packets.wrapping_add(1);
    }

    pub fn delivered_packets(&self) -> u64 {
        self.delivered_packets
    }

    pub fn dropped_packets(&self) -> u64 {
        self.dropped_packets
    }

    pub fn duplicated_packets(&self) -> u64 {
        self.duplicated_packets
    }

    pub fn reordered_packets(&self) -> u64 {
        self.reordered_packets
    }

    pub fn partitioned_packets(&self) -> u64 {
        self.partitioned_packets
    }

    pub fn digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.link_id);
        digest.mix_u64(self.delivered_packets);
        digest.mix_u64(self.dropped_packets);
        digest.mix_u64(self.duplicated_packets);
        digest.mix_u64(self.reordered_packets);
        digest.mix_u64(self.partitioned_packets);
        digest.mix_u64(self.partition_until);
        digest.finish()
    }
}

#[derive(Debug, Clone)]
pub struct DeviceScheduler {
    rng: StreamRng,
}

impl DeviceScheduler {
    pub fn new(seed: &Seed256, label: &str) -> Self {
        Self {
            rng: seed.derive(label),
        }
    }

    pub fn disk_delay(&mut self) -> u64 {
        3 + self.rng.next_bounded(20)
    }

    pub fn network_delay(&mut self) -> u64 {
        5 + self.rng.next_bounded(30)
    }

    pub fn entropy_delay(&mut self) -> u64 {
        2 + self.rng.next_bounded(10)
    }

    pub fn rng_cursor(&self) -> u64 {
        self.rng.state()
    }
}

/// Deterministic, seed-driven fault injection for disk and network devices.
///
/// Fault decisions use their own derived RNG streams, independent of device
/// timing, so enabling faults never perturbs the timing stream and a disabled
/// policy draws no randomness at all (keeping fault-free digests stable).
#[derive(Debug, Clone)]
pub struct FaultModel {
    disk_rng: StreamRng,
    net_rng: StreamRng,
    policy: FaultPolicy,
}

impl FaultModel {
    pub fn new(seed: &Seed256, policy: FaultPolicy) -> Self {
        policy.validate();
        Self {
            disk_rng: seed.derive("disk/faults/0"),
            net_rng: seed.derive("network/faults/0"),
            policy,
        }
    }

    pub fn policy(&self) -> &FaultPolicy {
        &self.policy
    }

    /// Disk and network fault stream cursors, for snapshots (GOLDEN_TARGET rule 14).
    pub fn rng_cursors(&self) -> (u64, u64) {
        (self.disk_rng.state(), self.net_rng.state())
    }

    /// Pick a disk fault for one request. Draws a single value only when at
    /// least one disk fault is configured, then buckets it cumulatively.
    pub fn disk_fault(&mut self) -> DiskFault {
        let disk = &self.policy.disk;
        let total = disk.eio_percent as u64
            + disk.timeout_percent as u64
            + disk.nospace_percent as u64;
        if total == 0 {
            return DiskFault::None;
        }

        let roll = self.disk_rng.next_bounded(100);
        let eio = disk.eio_percent as u64;
        let timeout = eio + disk.timeout_percent as u64;
        let nospace = timeout + disk.nospace_percent as u64;
        if roll < eio {
            DiskFault::Eio
        } else if roll < timeout {
            DiskFault::Timeout
        } else if roll < nospace {
            DiskFault::NoSpace
        } else {
            DiskFault::None
        }
    }

    pub fn net_partition_start(&mut self) -> bool {
        Self::roll(&mut self.net_rng, self.policy.net.partition_percent)
    }

    pub fn net_loss(&mut self) -> bool {
        Self::roll(&mut self.net_rng, self.policy.net.loss_percent)
    }

    pub fn net_reorder(&mut self) -> bool {
        Self::roll(&mut self.net_rng, self.policy.net.reorder_percent)
    }

    pub fn net_duplicate(&mut self) -> bool {
        Self::roll(&mut self.net_rng, self.policy.net.duplicate_percent)
    }

    fn roll(rng: &mut StreamRng, percent: u8) -> bool {
        percent > 0 && rng.next_bounded(100) < percent as u64
    }
}

/// Probability and shape of injected faults. Percentages are out of 100.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultPolicy {
    pub disk: DiskFaultConfig,
    pub net: NetFaultConfig,
}

impl FaultPolicy {
    pub fn disabled() -> Self {
        Self {
            disk: DiskFaultConfig::disabled(),
            net: NetFaultConfig::disabled(),
        }
    }

    fn validate(&self) {
        self.disk.validate();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskFaultConfig {
    pub eio_percent: u8,
    pub timeout_percent: u8,
    pub nospace_percent: u8,
    /// Extra completion delay (in instructions) added when a timeout fault hits.
    pub timeout_delay: u64,
}

impl DiskFaultConfig {
    pub fn disabled() -> Self {
        Self {
            eio_percent: 0,
            timeout_percent: 0,
            nospace_percent: 0,
            timeout_delay: 10_000,
        }
    }

    fn validate(&self) {
        let total = self.eio_percent as u64
            + self.timeout_percent as u64
            + self.nospace_percent as u64;
        assert!(total <= 100, "disk fault percentages must sum to <= 100");
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetFaultConfig {
    pub loss_percent: u8,
    pub duplicate_percent: u8,
    pub reorder_percent: u8,
    pub partition_percent: u8,
    /// How long (in instructions) a partition keeps the link down once started.
    pub partition_window: u64,
    /// Extra latency added to a reordered packet.
    pub reorder_extra_delay: u64,
    /// Gap between the original packet and its duplicate.
    pub duplicate_extra_delay: u64,
}

impl NetFaultConfig {
    pub fn disabled() -> Self {
        Self {
            loss_percent: 0,
            duplicate_percent: 0,
            reorder_percent: 0,
            partition_percent: 0,
            partition_window: 500,
            reorder_extra_delay: 100,
            duplicate_extra_delay: 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskFault {
    None,
    Eio,
    Timeout,
    NoSpace,
}

impl DiskFault {
    pub fn status(self) -> DiskStatus {
        match self {
            DiskFault::None => DiskStatus::Ok,
            DiskFault::Eio => DiskStatus::Eio,
            DiskFault::Timeout => DiskStatus::Timeout,
            DiskFault::NoSpace => DiskStatus::NoSpace,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskStatus {
    Ok = 0,
    Eio = 1,
    Timeout = 2,
    NoSpace = 3,
}

impl DiskStatus {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => DiskStatus::Eio,
            2 => DiskStatus::Timeout,
            3 => DiskStatus::NoSpace,
            _ => DiskStatus::Ok,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    BlockOutOfRange(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> Seed256 {
        Seed256::from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            .unwrap()
    }

    #[test]
    fn block_device_separates_data_from_completion_timing() {
        let seed = seed();
        let mut scheduler_a = DeviceScheduler::new(&seed, "devices");
        let mut scheduler_b = DeviceScheduler::new(&seed, "devices");
        let mut faults_a = FaultModel::new(&seed, FaultPolicy::disabled());
        let mut faults_b = FaultModel::new(&seed, FaultPolicy::disabled());
        let mut block_a = BlockDevice::new(2);
        let mut block_b = BlockDevice::new(2);
        let mut data = [0_u8; BLOCK_SIZE];
        data[0] = 7;

        block_a.write_block(1, &data).unwrap();
        block_b.write_block(1, &data).unwrap();

        let event_a = block_a.submit_completion(10, 99, &mut scheduler_a, &mut faults_a);
        let event_b = block_b.submit_completion(10, 99, &mut scheduler_b, &mut faults_b);

        assert_eq!(
            block_a.read_block(1).unwrap(),
            block_b.read_block(1).unwrap()
        );
        assert_eq!(event_a, event_b);
        assert_eq!(block_a.digest(), block_b.digest());
    }

    #[test]
    fn disabled_faults_inject_nothing() {
        let seed = seed();
        let mut scheduler = DeviceScheduler::new(&seed, "devices");
        let mut faults = FaultModel::new(&seed, FaultPolicy::disabled());
        let block = BlockDevice::new(4);
        let mut net = NetworkDevice::new(0);

        for request in 0..50 {
            let event = block.submit_completion(0, request, &mut scheduler, &mut faults);
            assert!(matches!(
                event.kind,
                EventKind::DiskCompletion {
                    status: 0, // DiskStatus::Ok
                    ..
                }
            ));
            let packets = net.submit_packet(0, request, &mut scheduler, &mut faults);
            assert_eq!(packets.len(), 1);
        }
        assert_eq!(net.dropped_packets(), 0);
        assert_eq!(net.duplicated_packets(), 0);
        assert_eq!(net.partitioned_packets(), 0);
    }

    #[test]
    fn disk_faults_are_seed_stable_and_injected() {
        let seed = seed();
        let policy = FaultPolicy {
            disk: DiskFaultConfig {
                eio_percent: 20,
                timeout_percent: 15,
                nospace_percent: 10,
                timeout_delay: 1_000,
            },
            net: NetFaultConfig::disabled(),
        };
        let mut scheduler_a = DeviceScheduler::new(&seed, "devices");
        let mut scheduler_b = DeviceScheduler::new(&seed, "devices");
        let mut faults_a = FaultModel::new(&seed, policy.clone());
        let mut faults_b = FaultModel::new(&seed, policy);
        let mut block_a = BlockDevice::new(1);
        let mut block_b = BlockDevice::new(1);

        for request in 0..200 {
            let event_a = block_a.submit_completion(0, request, &mut scheduler_a, &mut faults_a);
            let event_b = block_b.submit_completion(0, request, &mut scheduler_b, &mut faults_b);
            assert_eq!(event_a, event_b);
            if let EventKind::DiskCompletion { status, .. } = event_a.kind {
                block_a.complete(status);
                block_b.complete(status);
            }
        }

        // Same seed -> identical fault histogram.
        assert_eq!(block_a.digest(), block_b.digest());
        // Faults of each kind were actually injected.
        assert!(block_a.eio_completions() > 0);
        assert!(block_a.timeout_completions() > 0);
        assert!(block_a.nospace_completions() > 0);
        assert!(block_a.completed_requests() > 0);
    }

    #[test]
    fn network_loss_is_seed_stable() {
        let seed = seed();
        let policy = FaultPolicy {
            disk: DiskFaultConfig::disabled(),
            net: NetFaultConfig {
                loss_percent: 25,
                ..NetFaultConfig::disabled()
            },
        };
        let mut scheduler_a = DeviceScheduler::new(&seed, "net");
        let mut scheduler_b = DeviceScheduler::new(&seed, "net");
        let mut faults_a = FaultModel::new(&seed, policy.clone());
        let mut faults_b = FaultModel::new(&seed, policy);
        let mut net_a = NetworkDevice::new(0);
        let mut net_b = NetworkDevice::new(0);

        let mut events_a = Vec::new();
        let mut events_b = Vec::new();
        for packet in 0..40 {
            events_a.push(net_a.submit_packet(100, packet, &mut scheduler_a, &mut faults_a));
            events_b.push(net_b.submit_packet(100, packet, &mut scheduler_b, &mut faults_b));
        }

        assert_eq!(events_a, events_b);
        assert_eq!(net_a.dropped_packets(), net_b.dropped_packets());
        assert!(net_a.dropped_packets() > 0);
    }

    #[test]
    fn network_duplicate_and_reorder_inject_extra_events() {
        let seed = seed();
        let policy = FaultPolicy {
            disk: DiskFaultConfig::disabled(),
            net: NetFaultConfig {
                duplicate_percent: 30,
                reorder_percent: 30,
                ..NetFaultConfig::disabled()
            },
        };
        let mut scheduler = DeviceScheduler::new(&seed, "net");
        let mut faults = FaultModel::new(&seed, policy);
        let mut net = NetworkDevice::new(0);

        let mut total_events = 0;
        for packet in 0..60 {
            total_events += net.submit_packet(100, packet, &mut scheduler, &mut faults).len();
        }

        assert!(net.duplicated_packets() > 0);
        assert!(net.reordered_packets() > 0);
        // Duplicates mean more delivery events than packets submitted.
        assert!(total_events > 60);
    }

    #[test]
    fn network_partition_drops_packets_during_window() {
        let seed = seed();
        let policy = FaultPolicy {
            disk: DiskFaultConfig::disabled(),
            net: NetFaultConfig {
                partition_percent: 100, // partition starts on the first submit
                partition_window: 50,
                ..NetFaultConfig::disabled()
            },
        };
        let mut scheduler = DeviceScheduler::new(&seed, "net");
        let mut faults = FaultModel::new(&seed, policy);
        let mut net = NetworkDevice::new(0);

        // First submit opens the partition window; the rest fall inside it and
        // are dropped by the window without consuming further fault rolls.
        for packet in 0..10 {
            let events = net.submit_packet(packet, packet, &mut scheduler, &mut faults);
            assert!(events.is_empty(), "link is partitioned, nothing delivers");
        }
        assert_eq!(net.partitioned_packets(), 10);
        assert_eq!(net.delivered_packets(), 0);
        // Partition drops are accounted separately from loss drops.
        assert_eq!(net.dropped_packets(), 0);
    }

    #[test]
    fn partition_window_recovers_then_delivers() {
        let seed = seed();
        // One-shot partition: open a window, but never re-partition afterwards.
        let policy_open = NetFaultConfig {
            partition_percent: 100,
            partition_window: 50,
            ..NetFaultConfig::disabled()
        };
        let mut scheduler = DeviceScheduler::new(&seed, "net");
        let mut faults = FaultModel::new(
            &seed,
            FaultPolicy {
                disk: DiskFaultConfig::disabled(),
                net: policy_open,
            },
        );
        let mut net = NetworkDevice::new(0);

        // Opens partition until instruction 50.
        assert!(net.submit_packet(0, 0, &mut scheduler, &mut faults).is_empty());
        // Inside the window: dropped by the window check (no new roll).
        assert!(net.submit_packet(10, 1, &mut scheduler, &mut faults).is_empty());
        // Past the window with partition disabled going forward: delivers.
        faults = FaultModel::new(
            &seed,
            FaultPolicy {
                disk: DiskFaultConfig::disabled(),
                net: NetFaultConfig::disabled(),
            },
        );
        let events = net.submit_packet(100, 2, &mut scheduler, &mut faults);
        assert_eq!(events.len(), 1);
    }
}
