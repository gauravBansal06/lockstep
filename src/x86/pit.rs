//! Intel 8254 Programmable Interval Timer (PIT), ports 0x40–0x43.
//!
//! The PIT's only deterministic-machine role here is as a **readable counter for
//! clock calibration**: early boot calibrates the TSC and the LAPIC timer against
//! the PIT. The counter is a pure function of virtual time (instruction count),
//! per `docs/frozen-values.md` §2: `pit_ticks = floor(now * 1193182 / 1e9)`, and
//! each channel counts down from its reload value (mode 2/3 style).
//!
//! Generating the periodic IRQ0 is intentionally NOT modeled — the LAPIC timer
//! already provides the deterministic periodic-interrupt path; the PIT here is the
//! calibration reference the kernel reads. Latching captures the live count at a
//! deterministic instant (the guest's instruction count), so two runs latch and
//! read identical values.

use crate::state::StateDigest;

/// PIT input frequency in Hz (the classic 1.193182 MHz).
pub const PIT_FREQ: u64 = 1_193_182;
/// Virtual-time base: 1 instruction = 1 ns (1 GHz), per frozen-values §2.
const NS_PER_SEC: u64 = 1_000_000_000;

pub const PIT_BASE: u16 = 0x40;
const PIT_CMD: u16 = 3; // offset of the mode/command port (0x43)

/// One PIT channel's programmable state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Channel {
    reload: u16,
    /// Latched count (captured by a latch command); `None` = read the live count.
    latched: Option<u16>,
    /// Access mode: 1 = lobyte, 2 = hibyte, 3 = lo then hi.
    access: u8,
    /// Operating mode (command bits 3:1): 2 = rate generator, 3 = square wave —
    /// both periodic; the system timer uses one of these.
    mode: u8,
    /// Toggle for lo/hi byte during a two-byte read.
    read_hi: bool,
    /// Toggle for lo/hi byte during a two-byte write.
    write_hi: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pit {
    ch: [Channel; 3],
    /// Channel-0 periodic IRQ0 (the system tick): next fire instruction count
    /// (0 = disarmed) and the period in instructions. The kernel's `check_timer`
    /// requires these interrupts to advance `jiffies` or it panics.
    ch0_next_irq: u64,
    ch0_period: u64,
    /// Channel-2 one-shot (mode 0) gate state and arm time. Channel 2 is the
    /// TSC/clock-calibration timer: `quick_pit_calibrate`/`pit_calibrate_tsc`
    /// enable the gate via port 0x61, program a reload, then watch the count
    /// (port 0x42) decrement and the OUT bit (port 0x61 bit 5) go high at zero.
    ch2_gate: bool,
    /// Virtual time (instruction count) at which channel 2 began counting down
    /// (0 = not armed). Counting is relative to this so the kernel sees the
    /// count start at `reload` and decrease monotonically, as real hardware.
    ch2_start: u64,
}

impl Pit {
    pub fn new() -> Self {
        Self::default()
    }

    /// The live down-counter for a channel at virtual time `now` (instructions).
    /// Counts down from the reload value (0 means 65536), wrapping — exactly a
    /// deterministic function of `now`.
    fn live_count(&self, ch: usize, now: u64) -> u16 {
        let reload = if self.ch[ch].reload == 0 {
            0x1_0000
        } else {
            self.ch[ch].reload as u64
        };
        // Channel 2 (mode 0, one-shot) counts down from `reload` to 0 starting at
        // the arm instant and then saturates at 0 — the calibration timer. Until
        // the gate is enabled and a reload is loaded it holds at the reload value.
        if ch == 2 {
            if !self.ch2_gate || self.ch2_start == 0 {
                return reload as u16;
            }
            let elapsed_ns = now.saturating_sub(self.ch2_start);
            let elapsed = (elapsed_ns as u128 * PIT_FREQ as u128 / NS_PER_SEC as u128) as u64;
            return (reload.saturating_sub(elapsed)) as u16;
        }
        let elapsed = (now as u128 * PIT_FREQ as u128 / NS_PER_SEC as u128) as u64;
        (reload - 1 - (elapsed % reload)) as u16
    }

    /// Channel-2 OUT line (port 0x61 bit 5): in mode 0 it is low while counting
    /// and goes high once the count reaches terminal count (0).
    pub fn ch2_output(&self, now: u64) -> bool {
        if !self.ch2_gate || self.ch2_start == 0 {
            return false;
        }
        self.live_count(2, now) == 0
    }

    /// Drive the channel-2 gate (port 0x61 bit 0). A rising edge re-arms the
    /// countdown from the current instant if a reload is already loaded.
    pub fn set_gate2(&mut self, enabled: bool, now: u64) {
        if enabled && !self.ch2_gate {
            self.ch2_start = now.max(1);
        }
        self.ch2_gate = enabled;
    }

    pub fn read(&mut self, reg: u16, now: u64) -> u8 {
        let ch = reg as usize;
        if ch > 2 {
            return 0xff; // the command port is write-only
        }
        let count = self.ch[ch].latched.unwrap_or_else(|| self.live_count(ch, now));
        let access = self.ch[ch].access;
        // lobyte-only / hibyte-only / lo-then-hi.
        match access {
            1 => count as u8,
            2 => (count >> 8) as u8,
            _ => {
                let hi = self.ch[ch].read_hi;
                self.ch[ch].read_hi = !hi;
                if hi {
                    // High byte completes a latched read.
                    self.ch[ch].latched = None;
                    (count >> 8) as u8
                } else {
                    count as u8
                }
            }
        }
    }

    pub fn write(&mut self, reg: u16, value: u8, now: u64) {
        if reg == PIT_CMD {
            let ch = (value >> 6) & 3;
            if ch == 3 {
                return; // read-back command: not modeled
            }
            let access = (value >> 4) & 3;
            if access == 0 {
                // Latch command: capture the live count now.
                let c = ch as usize;
                self.ch[c].latched = Some(self.live_count(c, now));
                self.ch[c].read_hi = false;
            } else {
                let c = ch as usize;
                self.ch[c].access = access;
                self.ch[c].mode = (value >> 1) & 7;
                self.ch[c].read_hi = false;
                self.ch[c].write_hi = false;
            }
            return;
        }
        let ch = reg as usize;
        if ch > 2 {
            return;
        }
        let access = self.ch[ch].access;
        let complete = match access {
            1 => {
                self.ch[ch].reload = (self.ch[ch].reload & 0xff00) | value as u16;
                true
            }
            2 => {
                self.ch[ch].reload = (self.ch[ch].reload & 0x00ff) | ((value as u16) << 8);
                true
            }
            _ => {
                if self.ch[ch].write_hi {
                    self.ch[ch].reload = (self.ch[ch].reload & 0x00ff) | ((value as u16) << 8);
                } else {
                    self.ch[ch].reload = (self.ch[ch].reload & 0xff00) | value as u16;
                }
                self.ch[ch].write_hi = !self.ch[ch].write_hi;
                !self.ch[ch].write_hi // true once the high byte completes the reload
            }
        };
        // Channel 0 in a periodic mode (rate generator / square wave) arms the
        // system-tick IRQ0 once the reload is fully programmed.
        if ch == 0 && complete && matches!(self.ch[0].mode, 2 | 3) {
            let reload = if self.ch[0].reload == 0 {
                0x1_0000
            } else {
                self.ch[0].reload as u64
            };
            self.ch0_period =
                ((reload as u128 * NS_PER_SEC as u128) / PIT_FREQ as u128).max(1) as u64;
            self.ch0_next_irq = now + self.ch0_period;
        }
        // Channel 2: loading a fresh reload (while gated) starts the one-shot
        // countdown at this instant — the calibration timer's arming point.
        if ch == 2 && complete {
            self.ch2_start = if self.ch2_gate { now.max(1) } else { 0 };
        }
    }

    /// Poll the channel-0 system tick: returns true once per elapsed period,
    /// advancing the next deadline (catching up if virtual time jumped ahead).
    pub fn timer_fired(&mut self, now: u64) -> bool {
        if self.ch0_next_irq != 0 && self.ch0_period != 0 && now >= self.ch0_next_irq {
            while self.ch0_next_irq <= now {
                self.ch0_next_irq += self.ch0_period;
            }
            true
        } else {
            false
        }
    }

    /// Next channel-0 IRQ0 instruction count (for idle fast-forward).
    pub fn next_deadline(&self) -> Option<u64> {
        if self.ch0_next_irq != 0 {
            Some(self.ch0_next_irq)
        } else {
            None
        }
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        for c in &self.ch {
            d.mix_u64(c.reload as u64);
            d.mix_u64(c.latched.map(|v| v as u64 | (1 << 32)).unwrap_or(0));
            d.mix_bytes(&[c.access, c.mode, c.read_hi as u8, c.write_hi as u8]);
        }
        d.mix_u64(self.ch0_next_irq);
        d.mix_u64(self.ch0_period);
        d.mix_u64(self.ch2_start);
        d.mix_bytes(&[self.ch2_gate as u8]);
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latched_count_is_deterministic_function_of_time() {
        let mut a = Pit::new();
        let mut b = Pit::new();
        // Program channel 0, lo/hi access, reload 0 (= 65536).
        a.write(PIT_CMD, 0b0011_0100, 0);
        b.write(PIT_CMD, 0b0011_0100, 0);
        // Latch at the same virtual time -> identical value.
        let now = 5_000_000; // 5 ms
        a.write(PIT_CMD, 0b0000_0000, now); // latch ch0
        b.write(PIT_CMD, 0b0000_0000, now);
        let lo_a = a.read(0, now);
        let hi_a = a.read(0, now);
        let lo_b = b.read(0, now);
        let hi_b = b.read(0, now);
        assert_eq!((lo_a, hi_a), (lo_b, hi_b));
        // The counter has counted down from 0xFFFF (non-trivial after 5 ms).
        let count = lo_a as u16 | ((hi_a as u16) << 8);
        assert!(count < 0xffff);
    }

    #[test]
    fn counter_counts_down_with_time() {
        let mut pit = Pit::new();
        pit.write(PIT_CMD, 0b0011_0100, 0); // ch0, lo/hi, mode 2
        let early = pit.live_count(0, 1000);
        let later = pit.live_count(0, 2000);
        assert!(later < early, "counter decreases as virtual time advances");
    }
}
