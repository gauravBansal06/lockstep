use crate::device::{
    BlockDevice, DeviceScheduler, EntropyDevice, FaultModel, FaultPolicy, NetworkDevice,
};
use crate::event::{DetEvent, EventKind};
use crate::memory::{GuestMemory, MemoryError};
use crate::state::{MachineSpec, RunIdentity, RunMode, StateDigest};
use crate::{DeterministicCore, Seed256};

const REG_COUNT: usize = 4;

#[derive(Debug, Clone)]
pub struct ToyMachine {
    core: DeterministicCore,
    cpu: ToyCpu,
    memory: GuestMemory,
    block: BlockDevice,
    network: NetworkDevice,
    entropy: EntropyDevice,
    device_scheduler: DeviceScheduler,
    faults: FaultModel,
    disk_completions: u64,
    network_packets: u64,
    timer_interrupts: u64,
}

impl ToyMachine {
    pub fn new(seed: Seed256, ram_bytes: usize, program: &[u8]) -> Result<Self, ToyError> {
        Self::with_faults(seed, ram_bytes, program, FaultPolicy::disabled())
    }

    pub fn with_faults(
        seed: Seed256,
        ram_bytes: usize,
        program: &[u8],
        faults: FaultPolicy,
    ) -> Result<Self, ToyError> {
        let machine = MachineSpec {
            memory_bytes: ram_bytes as u64,
            ..MachineSpec::x86_64_single_vcpu()
        };
        let identity = RunIdentity::new(RunMode::Pure, seed, machine);
        let mut memory = GuestMemory::new(ram_bytes);
        memory.load(0, program)?;

        Ok(Self {
            block: BlockDevice::new(8),
            network: NetworkDevice::new(0),
            entropy: EntropyDevice::new(identity.seed(), "toy-entropy"),
            device_scheduler: DeviceScheduler::new(identity.seed(), "toy-devices"),
            faults: FaultModel::new(identity.seed(), faults),
            core: DeterministicCore::new(identity),
            cpu: ToyCpu::new(),
            memory,
            disk_completions: 0,
            network_packets: 0,
            timer_interrupts: 0,
        })
    }

    pub fn run(&mut self, max_instructions: u64) -> Result<ExitReason, ToyError> {
        for _ in 0..max_instructions {
            if self.cpu.halted {
                return Ok(ExitReason::Halted);
            }
            self.step()?;
        }
        Ok(ExitReason::InstructionLimit)
    }

    /// Replay/step forward until the virtual clock reaches `target_instruction`,
    /// or the guest halts. Used for snapshot replay and checkpoint stepping
    /// (GOLDEN_TARGET section 19).
    pub fn run_until_instruction(
        &mut self,
        target_instruction: u64,
    ) -> Result<ExitReason, ToyError> {
        while self.core.clock().instruction_count() < target_instruction {
            if self.cpu.halted {
                return Ok(ExitReason::Halted);
            }
            self.step()?;
        }
        Ok(ExitReason::InstructionLimit)
    }

    pub fn instruction_count(&self) -> u64 {
        self.core.clock().instruction_count()
    }

    /// Capture a complete in-memory snapshot. Because every component is `Clone`
    /// and carries its full state (RAM, devices, event queue, RNG cursors), the
    /// snapshot is a faithful copy and restore is equivalent to never snapshotting
    /// (GOLDEN_TARGET section 19, rule 14).
    pub fn snapshot(&self) -> ToySnapshot {
        ToySnapshot {
            machine: self.clone(),
        }
    }

    pub fn step(&mut self) -> Result<(), ToyError> {
        if self.cpu.halted {
            return Ok(());
        }

        let opcode = self.fetch_u8()?;
        match opcode {
            0x00 => {}
            0x01 => self.cpu.halted = true,
            0x10 => {
                let reg = self.fetch_reg()?;
                let value = self.fetch_u64()?;
                self.cpu.regs[reg] = value;
                self.cpu.zero = value == 0;
            }
            0x11 => {
                let dst = self.fetch_reg()?;
                let src = self.fetch_reg()?;
                let value = self.cpu.regs[dst].wrapping_add(self.cpu.regs[src]);
                self.cpu.regs[dst] = value;
                self.cpu.zero = value == 0;
            }
            0x12 => {
                let dst = self.fetch_reg()?;
                let src = self.fetch_reg()?;
                let value = self.cpu.regs[dst] ^ self.cpu.regs[src];
                self.cpu.regs[dst] = value;
                self.cpu.zero = value == 0;
            }
            0x20 => {
                let reg = self.fetch_reg()?;
                let address = self.fetch_u16()? as u64;
                let value = self.memory.read_u64_le(address)?;
                self.cpu.regs[reg] = value;
                self.cpu.zero = value == 0;
            }
            0x21 => {
                let reg = self.fetch_reg()?;
                let address = self.fetch_u16()? as u64;
                self.memory.write_u64_le(address, self.cpu.regs[reg])?;
            }
            0x30 => {
                self.cpu.pc = self.fetch_u16()? as u64;
            }
            0x31 => {
                let reg = self.fetch_reg()?;
                let address = self.fetch_u16()? as u64;
                if self.cpu.regs[reg] == 0 {
                    self.cpu.pc = address;
                }
            }
            0x40 => {
                let port = self.fetch_u8()?;
                let reg = self.fetch_reg()?;
                self.mmio_write(port, self.cpu.regs[reg])?;
            }
            0x41 => {
                let reg = self.fetch_reg()?;
                let port = self.fetch_u8()?;
                let value = self.mmio_read(port)?;
                self.cpu.regs[reg] = value;
                self.cpu.zero = value == 0;
            }
            _ => return Err(ToyError::InvalidOpcode(opcode)),
        }

        let delivered = self.core.step_instruction();
        for event in delivered {
            self.handle_event(event);
        }

        Ok(())
    }

    pub fn cpu(&self) -> &ToyCpu {
        &self.cpu
    }

    pub fn memory(&self) -> &GuestMemory {
        &self.memory
    }

    pub fn digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.core.digest());
        digest.mix_u64(self.memory.digest());
        digest.mix_u64(self.cpu.pc);
        digest.mix_u64(self.cpu.zero as u64);
        digest.mix_u64(self.cpu.halted as u64);
        for reg in self.cpu.regs {
            digest.mix_u64(reg);
        }
        digest.mix_u64(self.block.digest());
        digest.mix_u64(self.network.digest());
        digest.mix_u64(self.disk_completions);
        digest.mix_u64(self.network_packets);
        digest.mix_u64(self.timer_interrupts);
        digest.finish()
    }

    /// Full deterministic state digest, including everything `digest()` covers
    /// plus the pending event queue and every RNG stream cursor. This is the
    /// digest used for snapshots, checkpoints, and divergence detection so two
    /// runs that look identical on guest-visible state but differ in pending
    /// events or RNG position are still distinguished (GOLDEN_TARGET rule 14).
    pub fn full_state_digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.core.full_state_digest());
        digest.mix_u64(self.memory.digest());
        digest.mix_u64(self.cpu.pc);
        digest.mix_u64(self.cpu.zero as u64);
        digest.mix_u64(self.cpu.halted as u64);
        for reg in self.cpu.regs {
            digest.mix_u64(reg);
        }
        digest.mix_u64(self.block.digest());
        digest.mix_u64(self.network.digest());
        digest.mix_u64(self.entropy.rng_cursor());
        digest.mix_u64(self.device_scheduler.rng_cursor());
        let (disk_cursor, net_cursor) = self.faults.rng_cursors();
        digest.mix_u64(disk_cursor);
        digest.mix_u64(net_cursor);
        digest.mix_u64(self.disk_completions);
        digest.mix_u64(self.network_packets);
        digest.mix_u64(self.timer_interrupts);
        digest.finish()
    }

    fn mmio_read(&mut self, port: u8) -> Result<u64, ToyError> {
        match port {
            0x00 => Ok(self.core.clock().instruction_count()),
            0x01 => Ok(self.core.clock().virtual_ticks()),
            0x02 => Ok(self.entropy.read_u64()),
            _ => Err(ToyError::InvalidMmioPort(port)),
        }
    }

    fn mmio_write(&mut self, port: u8, value: u64) -> Result<(), ToyError> {
        match port {
            0x10 => {
                let now = self.core.clock().instruction_count();
                let event = self.block.submit_completion(
                    now,
                    value,
                    &mut self.device_scheduler,
                    &mut self.faults,
                );
                self.core.schedule_event(event);
                Ok(())
            }
            0x11 => {
                let now = self.core.clock().instruction_count();
                let events = self.network.submit_packet(
                    now,
                    value,
                    &mut self.device_scheduler,
                    &mut self.faults,
                );
                for event in events {
                    self.core.schedule_event(event);
                }
                Ok(())
            }
            0x12 => {
                let event = self.entropy.request_event(
                    self.core.clock().instruction_count(),
                    value,
                    &mut self.device_scheduler,
                );
                self.core.schedule_event(event);
                Ok(())
            }
            _ => Err(ToyError::InvalidMmioPort(port)),
        }
    }

    fn handle_event(&mut self, event: DetEvent) {
        match event.kind {
            EventKind::TimerInterrupt { .. } => {
                self.timer_interrupts = self.timer_interrupts.wrapping_add(1);
                self.cpu.regs[3] = self.cpu.regs[3].wrapping_add(1);
            }
            EventKind::DiskCompletion { request_id, status } => {
                self.block.complete(status);
                self.disk_completions = self.disk_completions.wrapping_add(1);
                self.cpu.regs[1] = request_id ^ status as u64;
            }
            EventKind::NetworkPacket { packet_id, .. } => {
                self.network.deliver();
                self.network_packets = self.network_packets.wrapping_add(1);
                self.cpu.regs[2] = packet_id;
            }
            EventKind::GuestEntropy { bytes, .. } => {
                let mut value = 0_u64;
                for (index, byte) in bytes.iter().take(8).enumerate() {
                    value |= (*byte as u64) << (index * 8);
                }
                self.cpu.regs[0] = value;
            }
        }
    }

    fn fetch_u8(&mut self) -> Result<u8, ToyError> {
        let value = self.memory.read_u8(self.cpu.pc)?;
        self.cpu.pc = self.cpu.pc.checked_add(1).ok_or(ToyError::PcOverflow)?;
        Ok(value)
    }

    fn fetch_u16(&mut self) -> Result<u16, ToyError> {
        let value = self.memory.read_u16_le(self.cpu.pc)?;
        self.cpu.pc = self.cpu.pc.checked_add(2).ok_or(ToyError::PcOverflow)?;
        Ok(value)
    }

    fn fetch_u64(&mut self) -> Result<u64, ToyError> {
        let value = self.memory.read_u64_le(self.cpu.pc)?;
        self.cpu.pc = self.cpu.pc.checked_add(8).ok_or(ToyError::PcOverflow)?;
        Ok(value)
    }

    fn fetch_reg(&mut self) -> Result<usize, ToyError> {
        let reg = self.fetch_u8()? as usize;
        if reg >= REG_COUNT {
            return Err(ToyError::InvalidRegister(reg as u8));
        }
        Ok(reg)
    }
}

/// An immutable, restorable snapshot of a complete toy machine.
#[derive(Debug, Clone)]
pub struct ToySnapshot {
    machine: ToyMachine,
}

impl ToySnapshot {
    /// Reconstruct a runnable machine. The result continues exactly as the
    /// original would have at snapshot time.
    pub fn restore(&self) -> ToyMachine {
        self.machine.clone()
    }

    pub fn instruction_count(&self) -> u64 {
        self.machine.instruction_count()
    }

    pub fn state_digest(&self) -> u64 {
        self.machine.full_state_digest()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToyCpu {
    pub pc: u64,
    pub regs: [u64; REG_COUNT],
    pub zero: bool,
    pub halted: bool,
}

impl ToyCpu {
    fn new() -> Self {
        Self {
            pc: 0,
            regs: [0; REG_COUNT],
            zero: false,
            halted: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    Halted,
    InstructionLimit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToyError {
    Memory(MemoryError),
    InvalidOpcode(u8),
    InvalidRegister(u8),
    InvalidMmioPort(u8),
    PcOverflow,
}

impl From<MemoryError> for ToyError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(hex_prefix: &str) -> Seed256 {
        let mut hex = hex_prefix.to_string();
        while hex.len() < 64 {
            hex.push('0');
        }
        Seed256::from_hex(&hex).unwrap()
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

    #[test]
    fn same_seed_program_reaches_same_state() {
        let program = sample_program();
        let mut a = ToyMachine::new(seed("01"), 512, &program).unwrap();
        let mut b = ToyMachine::new(seed("01"), 512, &program).unwrap();

        assert_eq!(a.run(200).unwrap(), ExitReason::Halted);
        assert_eq!(b.run(200).unwrap(), ExitReason::Halted);

        assert_eq!(a.cpu(), b.cpu());
        assert_eq!(a.memory().digest(), b.memory().digest());
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn different_seed_program_changes_state_through_mmio_and_events() {
        let program = sample_program();
        let mut a = ToyMachine::new(seed("01"), 512, &program).unwrap();
        let mut b = ToyMachine::new(seed("02"), 512, &program).unwrap();

        a.run(200).unwrap();
        b.run(200).unwrap();

        assert_ne!(a.digest(), b.digest());
    }

    fn fault_policy() -> FaultPolicy {
        FaultPolicy {
            disk: crate::device::DiskFaultConfig {
                eio_percent: 40,
                timeout_percent: 30,
                nospace_percent: 20,
                timeout_delay: 1_000,
            },
            net: crate::device::NetFaultConfig {
                loss_percent: 50,
                duplicate_percent: 30,
                reorder_percent: 30,
                ..crate::device::NetFaultConfig::disabled()
            },
        }
    }

    #[test]
    fn faulty_run_is_deterministic_under_same_seed_and_policy() {
        let program = sample_program();
        let mut a = ToyMachine::with_faults(seed("01"), 512, &program, fault_policy()).unwrap();
        let mut b = ToyMachine::with_faults(seed("01"), 512, &program, fault_policy()).unwrap();

        assert_eq!(a.run(200).unwrap(), ExitReason::Halted);
        assert_eq!(b.run(200).unwrap(), ExitReason::Halted);

        assert_eq!(a.cpu(), b.cpu());
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn injected_faults_change_the_state_sequence() {
        let program = sample_program();
        let mut clean = ToyMachine::new(seed("01"), 512, &program).unwrap();
        let mut faulty =
            ToyMachine::with_faults(seed("01"), 512, &program, fault_policy()).unwrap();

        clean.run(200).unwrap();
        faulty.run(200).unwrap();

        // Same seed, same program, but the fault model diverts the run.
        assert_ne!(clean.digest(), faulty.digest());
    }
}
