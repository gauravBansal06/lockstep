#[derive(Debug, Clone)]
pub struct VirtualClock {
    instruction_count: u64,
    ticks_per_instruction: u64,
}

impl VirtualClock {
    pub fn new(ticks_per_instruction: u64) -> Self {
        assert!(ticks_per_instruction > 0);
        Self {
            instruction_count: 0,
            ticks_per_instruction,
        }
    }

    pub fn step_instruction(&mut self) {
        self.instruction_count = self
            .instruction_count
            .checked_add(1)
            .expect("virtual instruction counter overflowed");
    }

    pub fn fast_forward_to_instruction(&mut self, target_instruction: u64) {
        assert!(
            target_instruction >= self.instruction_count,
            "virtual clock cannot move backwards"
        );
        self.instruction_count = target_instruction;
    }

    pub fn instruction_count(&self) -> u64 {
        self.instruction_count
    }

    pub fn virtual_ticks(&self) -> u64 {
        self.instruction_count
            .checked_mul(self.ticks_per_instruction)
            .expect("virtual clock overflowed")
    }
}
