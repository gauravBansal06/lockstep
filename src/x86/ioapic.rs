//! Minimal I/O APIC: the indirect MMIO register interface + redirection table.
//!
//! A kernel programs it to route device IRQs (GSIs) to interrupt vectors. We use
//! it to map a device's GSI to a vector; when no entry is programmed the device
//! falls back to its default vector (so tests work without IOAPIC setup).

use crate::state::StateDigest;

pub const IOAPIC_BASE: u64 = 0xFEC0_0000;
const IOAPIC_LEN: u64 = 0x20;
const NUM_REDIR: usize = 24;

const REG_SELECT: u64 = 0x00;
const REG_WINDOW: u64 = 0x10;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ioapic {
    select: u32,
    id: u32,
    /// 24 redirection entries (64-bit each: vector, delivery, mask bit16, …).
    redir: [u64; NUM_REDIR],
}

impl Ioapic {
    pub fn new() -> Self {
        let mut io = Self::default();
        // Intel IOAPIC reset state: EVERY redirection entry has the mask bit (16) SET.
        // The OS programs an entry's vector while masked, then explicitly unmasks it
        // when ready to receive the IRQ. Our old all-zero default left entries
        // *unmasked*, so the moment the kernel wrote a vector (early, still intending
        // it masked) we began delivering the timer IRQ — feeding it into early-init
        // udelay/clock-calibration poll loops and perturbing them (a determinism-
        // neutral but boot-divergent fidelity bug).
        for e in io.redir.iter_mut() {
            *e = 1 << 16;
        }
        io
    }

    pub fn offset_of(phys: u64) -> Option<u64> {
        if (IOAPIC_BASE..IOAPIC_BASE + IOAPIC_LEN).contains(&phys) {
            Some(phys - IOAPIC_BASE)
        } else {
            None
        }
    }

    pub fn read(&self, offset: u64) -> u32 {
        match offset {
            REG_SELECT => self.select,
            REG_WINDOW => self.read_reg(self.select),
            _ => 0,
        }
    }

    pub fn write(&mut self, offset: u64, value: u32) {
        match offset {
            REG_SELECT => self.select = value & 0xff,
            REG_WINDOW => self.write_reg(self.select, value),
            _ => {}
        }
    }

    fn read_reg(&self, idx: u32) -> u32 {
        match idx {
            0x00 => self.id,
            0x01 => 0x0017_0011, // version 0x11, max redirection entry = 23
            0x02 => 0,           // arbitration
            0x10..=0x3f => {
                let n = ((idx - 0x10) / 2) as usize;
                if (idx - 0x10) & 1 == 0 {
                    self.redir[n] as u32
                } else {
                    (self.redir[n] >> 32) as u32
                }
            }
            _ => 0,
        }
    }

    fn write_reg(&mut self, idx: u32, value: u32) {
        match idx {
            0x00 => self.id = value,
            0x10..=0x3f => {
                let n = ((idx - 0x10) / 2) as usize;
                if (idx - 0x10) & 1 == 0 {
                    self.redir[n] = (self.redir[n] & !0xffff_ffff) | value as u64;
                } else {
                    self.redir[n] = (self.redir[n] & 0xffff_ffff) | ((value as u64) << 32);
                }
            }
            _ => {}
        }
    }

    /// Vector for a programmed, unmasked GSI redirection entry.
    pub fn vector_for_gsi(&self, gsi: u32) -> Option<u8> {
        self.route_for_gsi(gsi).map(|(v, _)| v)
    }

    /// `(vector, destination APIC ID)` for a programmed, unmasked GSI. The destination is the
    /// physical-mode APIC ID (redirection-entry bits 56:63). Logical destination mode (bit 11)
    /// is not yet modeled — it falls back to APIC ID 0 (the BSP), which is correct for the
    /// single-CPU case the kernel programs today.
    pub fn route_for_gsi(&self, gsi: u32) -> Option<(u8, u8)> {
        let n = gsi as usize;
        if n >= NUM_REDIR {
            return None;
        }
        let entry = self.redir[n];
        let masked = entry & (1 << 16) != 0;
        let vector = (entry & 0xff) as u8;
        if masked || vector == 0 {
            return None;
        }
        let logical = entry & (1 << 11) != 0;
        let dest = if logical { 0 } else { ((entry >> 56) & 0xff) as u8 };
        Some((vector, dest))
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.id as u64);
        // `select` is future-affecting: it decides which register the next
        // REG_WINDOW read/write touches.
        d.mix_u64(self.select as u64);
        for e in self.redir {
            d.mix_u64(e);
        }
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirection_entry_routes_gsi() {
        let mut io = Ioapic::new();
        // Program GSI 5 -> vector 0x42, unmasked: write low dword of redir[5]
        // (register index 0x10 + 2*5 = 0x1a).
        io.write(REG_SELECT, 0x1a);
        io.write(REG_WINDOW, 0x42);
        assert_eq!(io.vector_for_gsi(5), Some(0x42));
        assert_eq!(io.vector_for_gsi(6), None);
        // Mask it (bit 16) -> no delivery.
        io.write(REG_SELECT, 0x1a);
        io.write(REG_WINDOW, 0x42 | (1 << 16));
        assert_eq!(io.vector_for_gsi(5), None);
    }

    #[test]
    fn route_extracts_physical_destination() {
        let mut io = Ioapic::new();
        // GSI 5 → vector 0x42 (low dword, unmasked), physical dest APIC ID 3 (entry bits 56:63
        // = high-dword bits 24:31).
        io.write(REG_SELECT, 0x1a);
        io.write(REG_WINDOW, 0x42);
        io.write(REG_SELECT, 0x1b);
        io.write(REG_WINDOW, 3 << 24);
        assert_eq!(io.route_for_gsi(5), Some((0x42, 3)));
        // Logical destination mode (bit 11) → not modeled → falls back to APIC ID 0 (BSP).
        io.write(REG_SELECT, 0x1a);
        io.write(REG_WINDOW, 0x42 | (1 << 11));
        assert_eq!(io.route_for_gsi(5), Some((0x42, 0)));
    }
}
