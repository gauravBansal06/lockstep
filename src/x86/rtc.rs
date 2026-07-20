//! MC146818-style RTC / CMOS, ports 0x70 (index) and 0x71 (data).
//!
//! The wall-clock time the guest reads is a pure function of virtual time:
//! `unix_seconds = clock_epoch + floor(now / 1e9)` (frozen-values §2), so two runs
//! with the same identity read identical timestamps. The update-in-progress bit
//! (Status A bit 7) is always 0 — there is no asynchronous update, so the guest
//! never spins waiting for one and never observes a torn read.
//!
//! Non-time CMOS registers are a 128-byte RAM the guest can read/write freely.

use crate::state::StateDigest;

pub const RTC_INDEX: u16 = 0x70;
pub const RTC_DATA: u16 = 0x71;

/// Frozen virtual-clock epoch (2023-11-14T22:13:20Z), per frozen-values §2.
/// Fixed here until the loader threads `virtual_clock_epoch` from RunIdentity.
const CLOCK_EPOCH: u64 = 1_700_000_000;
const NS_PER_SEC: u64 = 1_000_000_000;

// Time register indices.
const R_SEC: u8 = 0x00;
const R_MIN: u8 = 0x02;
const R_HOUR: u8 = 0x04;
const R_DOW: u8 = 0x06;
const R_DOM: u8 = 0x07;
const R_MONTH: u8 = 0x08;
const R_YEAR: u8 = 0x09;
const R_STATUS_A: u8 = 0x0a;
const R_STATUS_B: u8 = 0x0b;
const R_STATUS_C: u8 = 0x0c;
const R_STATUS_D: u8 = 0x0d;
const R_CENTURY: u8 = 0x32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rtc {
    index: u8,
    ram: [u8; 128],
}

impl Default for Rtc {
    fn default() -> Self {
        let mut ram = [0_u8; 128];
        ram[R_STATUS_A as usize] = 0x26; // 32.768 kHz base, 1024 Hz rate (typical)
        ram[R_STATUS_B as usize] = 0x02; // 24-hour, BCD
        ram[R_STATUS_D as usize] = 0x80; // battery (CMOS) valid
        Self { index: 0, ram }
    }
}

/// Civil date (year, month, day) from days since the Unix epoch.
/// Howard Hinnant's `civil_from_days`, valid for the proleptic Gregorian calendar.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

impl Rtc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Selecting an index: high bit is the NMI-disable flag, masked off.
    pub fn write_index(&mut self, value: u8) {
        self.index = value & 0x7f;
    }

    fn binary_mode(&self) -> bool {
        self.ram[R_STATUS_B as usize] & 0x04 != 0
    }

    /// Encode a value per the current data mode (BCD unless Status B bit 2 set).
    fn encode(&self, v: u8) -> u8 {
        if self.binary_mode() {
            v
        } else {
            ((v / 10) << 4) | (v % 10)
        }
    }

    /// The current wall clock derived from virtual time `now` (instruction count).
    fn datetime(&self, now: u64) -> (i64, u32, u32, u32, u32, u32, u32) {
        let secs = CLOCK_EPOCH + now / NS_PER_SEC;
        let days = (secs / 86_400) as i64;
        let tod = secs % 86_400;
        let (y, mo, d) = civil_from_days(days);
        let dow = ((days + 4).rem_euclid(7)) as u32 + 1; // 1 = Sunday
        (
            y,
            mo,
            d,
            (tod / 3600) as u32,
            (tod % 3600 / 60) as u32,
            (tod % 60) as u32,
            dow,
        )
    }

    pub fn read_data(&self, now: u64) -> u8 {
        let (y, mo, d, h, mi, s, dow) = self.datetime(now);
        match self.index {
            R_SEC => self.encode(s as u8),
            R_MIN => self.encode(mi as u8),
            R_HOUR => self.encode(h as u8),
            R_DOW => self.encode(dow as u8),
            R_DOM => self.encode(d as u8),
            R_MONTH => self.encode(mo as u8),
            R_YEAR => self.encode((y % 100) as u8),
            R_CENTURY => self.encode((y / 100) as u8),
            R_STATUS_A => self.ram[R_STATUS_A as usize] & 0x7f, // update-in-progress always 0
            R_STATUS_B | R_STATUS_C | R_STATUS_D => self.ram[self.index as usize],
            i => self.ram[i as usize],
        }
    }

    pub fn write_data(&mut self, value: u8) {
        // Time registers are derived (read-only here); other CMOS RAM is writable.
        match self.index {
            R_SEC | R_MIN | R_HOUR | R_DOW | R_DOM | R_MONTH | R_YEAR | R_CENTURY => {}
            R_STATUS_C | R_STATUS_D => {} // read-only
            i => self.ram[i as usize] = value,
        }
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.index as u64);
        d.mix_bytes(&self.ram);
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_frozen_epoch_in_bcd() {
        let rtc = Rtc::new();
        // At now = 0 the time is exactly the epoch 2023-11-14T22:13:20Z.
        let read = |idx: u8| {
            let mut r = rtc.clone();
            r.write_index(idx);
            r.read_data(0)
        };
        assert_eq!(read(R_SEC), 0x20, "20 seconds in BCD");
        assert_eq!(read(R_MIN), 0x13, "13 minutes");
        assert_eq!(read(R_HOUR), 0x22, "22 hours (24h)");
        assert_eq!(read(R_DOM), 0x14, "14th");
        assert_eq!(read(R_MONTH), 0x11, "November = 11");
        assert_eq!(read(R_YEAR), 0x23, "year 23");
        assert_eq!(read(R_CENTURY), 0x20, "century 20");
    }

    #[test]
    fn update_in_progress_is_never_set() {
        let mut rtc = Rtc::new();
        rtc.write_index(R_STATUS_A);
        for now in [0_u64, 1_000_000, 999_999_999, 5_000_000_000] {
            assert_eq!(rtc.read_data(now) & 0x80, 0, "UIP must stay 0 for determinism");
        }
    }

    #[test]
    fn cmos_ram_is_writable_but_time_is_not() {
        let mut rtc = Rtc::new();
        rtc.write_index(0x40); // a scratch CMOS byte
        rtc.write_data(0xab);
        assert_eq!(rtc.read_data(0), 0xab);
        // Writing a time register is ignored (time is derived).
        rtc.write_index(R_SEC);
        rtc.write_data(0x55);
        assert_eq!(rtc.read_data(0), 0x20, "seconds still derived from the clock");
    }

    #[test]
    fn time_advances_one_second_per_1e9_instructions() {
        let mut rtc = Rtc::new();
        rtc.write_index(R_SEC);
        assert_eq!(rtc.read_data(0), 0x20);
        assert_eq!(rtc.read_data(NS_PER_SEC), 0x21, "one second later");
    }
}
