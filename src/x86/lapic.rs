//! Minimal Local APIC with a deterministic timer.
//!
//! MMIO at 0xFEE0_0000. The timer deadline is an absolute virtual instruction
//! count, so firing is a pure function of instruction count (no host timing).
//! This is the per-CPU interrupt source a kernel programs in place of the legacy
//! PIT. APIC priority/IPIs/IOAPIC are not modeled yet (single vCPU).

use crate::state::StateDigest;

pub const LAPIC_BASE: u64 = 0xFEE0_0000;
const LAPIC_LEN: u64 = 0x1000;

// Register offsets.
const REG_ID: u64 = 0x20;
const REG_VERSION: u64 = 0x30;
const REG_TPR: u64 = 0x80;
const REG_EOI: u64 = 0xB0;
const REG_SVR: u64 = 0xF0;
const REG_ICR_LOW: u64 = 0x300;
const REG_ICR_HIGH: u64 = 0x310;
const REG_LVT_TIMER: u64 = 0x320;
const REG_INIT_COUNT: u64 = 0x380;
const REG_CUR_COUNT: u64 = 0x390;
const REG_DIV_CONFIG: u64 = 0x3E0;

const LVT_MASKED: u32 = 1 << 16;
const LVT_PERIODIC: u32 = 1 << 17;

/// IPI delivery modes (ICR bits 8:10).
pub const DELIVERY_FIXED: u8 = 0;
pub const DELIVERY_INIT: u8 = 5;
pub const DELIVERY_STARTUP: u8 = 6;

/// A decoded inter-processor interrupt from an ICR write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ipi {
    pub vector: u8,
    pub delivery_mode: u8,   // 0=Fixed, 5=INIT, 6=Startup(SIPI)
    pub dest_shorthand: u8,  // 0=field, 1=self, 2=all-incl-self, 3=all-excl-self
    pub dest: u8,            // physical destination APIC ID (when shorthand=0)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lapic {
    /// This LAPIC's physical APIC ID (reported in bits 24:31 of the ID register).
    pub id: u32,
    svr: u32,
    tpr: u32,
    lvt_timer: u32,
    initial_count: u32,
    divide_config: u32,
    /// Absolute instruction count at which the timer next fires.
    deadline: u64,
    /// Reload period (in instructions) for periodic mode.
    period: u64,
    armed: bool,
    /// Interrupt Command Register (low/high dwords) and a flag set when the low dword is
    /// written (which issues the IPI on real hardware). Consumed by the machine each step, so
    /// this never contributes to the between-instruction digest.
    icr_low: u32,
    icr_high: u32,
    ipi_pending: bool,
}

impl Lapic {
    pub fn new() -> Self {
        Self {
            svr: 0xff,
            lvt_timer: LVT_MASKED,
            ..Self::default()
        }
    }

    /// Returns the register offset if `phys` is inside the LAPIC MMIO window.
    pub fn offset_of(phys: u64) -> Option<u64> {
        if (LAPIC_BASE..LAPIC_BASE + LAPIC_LEN).contains(&phys) {
            Some(phys - LAPIC_BASE)
        } else {
            None
        }
    }

    fn divisor(&self) -> u64 {
        let sel = (self.divide_config & 0b11) | ((self.divide_config >> 1) & 0b100);
        if sel == 0b111 {
            1
        } else {
            1 << (sel + 1)
        }
    }

    pub fn read(&self, offset: u64, now: u64) -> u32 {
        match offset {
            REG_ID => self.id << 24, // APIC ID lives in bits 24:31
            REG_VERSION => 0x0005_0014, // version 0x14, max LVT 5
            REG_TPR => self.tpr,
            REG_SVR => self.svr,
            REG_LVT_TIMER => self.lvt_timer,
            REG_INIT_COUNT => self.initial_count,
            REG_CUR_COUNT if self.armed => {
                (self.deadline.saturating_sub(now) / self.divisor()) as u32
            }
            REG_CUR_COUNT => 0,
            REG_DIV_CONFIG => self.divide_config,
            _ => 0,
        }
    }

    pub fn write(&mut self, offset: u64, value: u32, now: u64) {
        match offset {
            REG_TPR => self.tpr = value,
            REG_EOI => {} // acknowledge — single-vCPU model has nothing to pop
            REG_SVR => self.svr = value,
            REG_LVT_TIMER => self.lvt_timer = value,
            REG_DIV_CONFIG => self.divide_config = value,
            REG_INIT_COUNT => {
                self.initial_count = value;
                self.arm(now);
            }
            REG_ICR_HIGH => self.icr_high = value,
            REG_ICR_LOW => {
                // Writing the low dword issues the IPI; the machine picks it up post-step.
                self.icr_low = value;
                self.ipi_pending = true;
            }
            _ => {}
        }
    }

    /// If an IPI was issued (ICR low dword written) since the last call, decode and consume it.
    pub fn take_ipi(&mut self) -> Option<Ipi> {
        if !self.ipi_pending {
            return None;
        }
        self.ipi_pending = false;
        Some(Ipi {
            vector: (self.icr_low & 0xff) as u8,
            delivery_mode: ((self.icr_low >> 8) & 0x7) as u8,
            dest_shorthand: ((self.icr_low >> 18) & 0x3) as u8,
            dest: ((self.icr_high >> 24) & 0xff) as u8,
        })
    }

    fn arm(&mut self, now: u64) {
        if self.initial_count != 0 && self.lvt_timer & LVT_MASKED == 0 {
            self.period = self.initial_count as u64 * self.divisor();
            self.deadline = now + self.period;
            self.armed = true;
        } else {
            self.armed = false;
        }
    }

    /// The instruction count of the next timer fire, if armed and unmasked.
    pub fn next_deadline(&self) -> Option<u64> {
        if self.armed && self.lvt_timer & LVT_MASKED == 0 {
            Some(self.deadline)
        } else {
            None
        }
    }

    /// If the timer is due at `now`, return its vector and re-arm/disarm.
    pub fn timer_fired(&mut self, now: u64) -> Option<u8> {
        if self.armed && self.lvt_timer & LVT_MASKED == 0 && now >= self.deadline {
            let vector = (self.lvt_timer & 0xff) as u8;
            if self.lvt_timer & LVT_PERIODIC != 0 {
                self.deadline = now + self.period;
            } else {
                self.armed = false;
            }
            Some(vector)
        } else {
            None
        }
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        for v in [self.svr, self.tpr, self.lvt_timer, self.initial_count, self.divide_config] {
            d.mix_u64(v as u64);
        }
        d.mix_u64(self.deadline);
        d.mix_u64(self.period);
        d.mix_u64(self.armed as u64);
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_timer_fires_once() {
        let mut l = Lapic::new();
        l.write(REG_DIV_CONFIG, 0b1011, 0); // divide by 1
        l.write(REG_LVT_TIMER, 0x40, 0); // vector 0x40, one-shot, unmasked
        l.write(REG_INIT_COUNT, 100, 0); // arm at now=0 -> deadline 100
        assert_eq!(l.next_deadline(), Some(100));
        assert_eq!(l.timer_fired(50), None);
        assert_eq!(l.timer_fired(100), Some(0x40));
        assert_eq!(l.timer_fired(200), None, "one-shot disarms");
    }

    #[test]
    fn icr_write_decodes_ipi() {
        let mut l = Lapic::new();
        assert_eq!(l.take_ipi(), None); // nothing issued yet
        // Startup-IPI to physical APIC ID 3, vector 0x10: high dword dest, then low dword issues.
        l.write(REG_ICR_HIGH, 3 << 24, 0);
        l.write(REG_ICR_LOW, (DELIVERY_STARTUP as u32) << 8 | 0x10, 0);
        assert_eq!(
            l.take_ipi(),
            Some(Ipi { vector: 0x10, delivery_mode: DELIVERY_STARTUP, dest_shorthand: 0, dest: 3 })
        );
        assert_eq!(l.take_ipi(), None, "IPI consumed once");
        // INIT with an all-excluding-self shorthand (bits 18:19 = 3).
        l.write(REG_ICR_LOW, (DELIVERY_INIT as u32) << 8 | 3 << 18, 0);
        let ipi = l.take_ipi().unwrap();
        assert_eq!(ipi.delivery_mode, DELIVERY_INIT);
        assert_eq!(ipi.dest_shorthand, 3);
        // The ID register reports the APIC ID in bits 24:31.
        l.id = 5;
        assert_eq!(l.read(REG_ID, 0), 5 << 24);
    }

    #[test]
    fn periodic_timer_reloads() {
        let mut l = Lapic::new();
        l.write(REG_DIV_CONFIG, 0b1011, 0);
        l.write(REG_LVT_TIMER, 0x20020, 0); // periodic (bit17) + vector 0x20
        l.write(REG_INIT_COUNT, 100, 0);
        assert_eq!(l.timer_fired(100), Some(0x20));
        assert_eq!(l.next_deadline(), Some(200));
        assert_eq!(l.timer_fired(200), Some(0x20));
    }
}
