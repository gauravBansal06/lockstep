//! 16550 UART serial console (COM1) and the port-I/O bus.
//!
//! Output-only console: bytes written to the transmit register are collected
//! deterministically into a buffer (the "console"). Line status always reports
//! the transmitter empty, so the guest's `outb` loop never stalls. Input is not
//! modeled in Pure Mode (it would be a seed/transcript source — platform
//! contract section 6.2).

use super::ioapic::Ioapic;
use super::pit::{Pit, PIT_BASE};
use super::rtc::{Rtc, RTC_DATA, RTC_INDEX};
use super::virtio_blk::VirtioBlk;
use super::virtio_net::VirtioNet;
use super::virtio_rng::VirtioRng;
use crate::state::StateDigest;

/// A minimal 16550 UART. Registers are the standard offsets from the base port.
#[derive(Debug, Clone, Default)]
pub struct Uart16550 {
    output: Vec<u8>,
    ier: u8,
    lcr: u8,
    mcr: u8,
    scr: u8,
    dll: u8,
    dlm: u8,
}

impl Uart16550 {
    pub fn new() -> Self {
        Self::default()
    }

    fn dlab(&self) -> bool {
        self.lcr & 0x80 != 0
    }

    /// True while the transmit-holding-register-empty (THRE) interrupt is armed.
    /// The transmitter is modeled as always empty (bytes drain instantly), so the
    /// only gate is IER bit 1 (ETBEI). Drives the COM1 IRQ4 line.
    pub fn tx_irq_pending(&self) -> bool {
        self.ier & 0x02 != 0
    }

    /// Read a register (offset 0..7 from the UART base).
    pub fn read(&self, reg: u16) -> u8 {
        match reg {
            0 => if self.dlab() { self.dll } else { 0 }, // RBR: no input
            1 => if self.dlab() { self.dlm } else { self.ier },
            // IIR: report the THRE interrupt (0x02) while it is armed, else "no
            // interrupt pending" (0x01). Reading IIR is how the 8250 ISR learns
            // the source; bits 7:6 (FIFO) stay 0 — we present as a 16450.
            2 => if self.tx_irq_pending() { 0x02 } else { 0x01 },
            3 => self.lcr,
            4 => self.mcr,
            5 => 0x60,  // LSR: THR empty (0x20) + transmitter empty (0x40)
            6 => 0x00,  // MSR
            7 => self.scr,
            _ => 0xff,
        }
    }

    /// Write a register (offset 0..7 from the UART base).
    pub fn write(&mut self, reg: u16, value: u8) {
        match reg {
            0 => {
                if self.dlab() {
                    self.dll = value;
                } else {
                    self.output.push(value); // THR -> console
                }
            }
            1 => {
                if self.dlab() {
                    self.dlm = value;
                } else {
                    self.ier = value;
                }
            }
            2 => {}        // FCR: FIFO control, ignored
            3 => self.lcr = value,
            4 => self.mcr = value,
            5 | 6 => {}    // LSR/MSR read-only
            7 => self.scr = value,
            _ => {}
        }
    }

    pub fn output(&self) -> &[u8] {
        &self.output
    }

    pub fn output_string(&self) -> String {
        String::from_utf8_lossy(&self.output).into_owned()
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.output.len() as u64);
        d.mix_bytes(&self.output);
        d.mix_bytes(&[self.ier, self.lcr, self.mcr, self.scr, self.dll, self.dlm]);
        d.finish()
    }
}

/// COM1 base port.
pub const COM1_BASE: u16 = 0x3f8;

/// Port-I/O bus: routes `IN`/`OUT` to devices. Unknown ports read all-ones and
/// ignore writes (the deterministic "floating bus" convention). `now` is the
/// current instruction count, refreshed by the machine each step so time-derived
/// devices (PIT, RTC) read a deterministic value.
#[derive(Debug, Clone)]
pub struct IoBus {
    pub serial: Uart16550,
    pub pit: Pit,
    pub rtc: Rtc,
    pub now: u64,
    /// Writable bits of port 0x61 (NMI status & control / keyboard-controller
    /// port B): bit 0 = PIT channel-2 gate, bit 1 = speaker data. The other bits
    /// (refresh toggle, channel-2 OUT) are derived from virtual time on read.
    pub port61: u8,
    /// Machine-level SHARED devices — one instance for all vCPUs (the local APIC stays
    /// per-vCPU in `X86Cpu`; the IOAPIC and virtio devices are system-wide).
    pub ioapic: Ioapic,
    pub virtio: VirtioBlk,
    pub virtio_rng: VirtioRng,
    pub virtio_net: Option<VirtioNet>,
}

/// Port 0x61: NMI status and control register (a.k.a. PC speaker / port B).
pub const PORT_B: u16 = 0x61;
/// The hardware refresh-request line on bit 4 of port 0x61 toggles roughly every
/// 15.085 µs; some legacy delay loops poll it as a coarse timebase.
const REFRESH_PERIOD_NS: u64 = 15_085;

impl IoBus {
    pub fn new() -> Self {
        Self {
            serial: Uart16550::new(),
            pit: Pit::new(),
            rtc: Rtc::new(),
            now: 0,
            port61: 0,
            ioapic: Ioapic::new(),
            virtio: VirtioBlk::new(Vec::new()),
            virtio_rng: VirtioRng::new(),
            virtio_net: None,
        }
    }

    pub fn read(&mut self, port: u16, _size: usize) -> u64 {
        match port {
            COM1_BASE..=0x3ff => self.serial.read(port - COM1_BASE) as u64,
            PIT_BASE..=0x43 => self.pit.read(port - PIT_BASE, self.now) as u64,
            PORT_B => {
                // Low writable bits (gate/speaker) reflect what was last written;
                // bit 4 is the refresh toggle and bit 5 the PIT ch-2 OUT line.
                let mut v = self.port61 & 0x0f;
                if (self.now / REFRESH_PERIOD_NS) & 1 == 1 {
                    v |= 0x10;
                }
                if self.pit.ch2_output(self.now) {
                    v |= 0x20;
                }
                v as u64
            }
            RTC_DATA => self.rtc.read_data(self.now) as u64,
            _ => 0xff,
        }
    }

    pub fn write(&mut self, port: u16, _size: usize, value: u64) {
        if std::env::var("LS_IOTRACE").is_ok() && (COM1_BASE..=0x3ff).contains(&port) {
            eprintln!(
                "IO com1 reg{} <- {:#04x} (lcr={:#04x} dlab={})",
                port - COM1_BASE,
                value as u8,
                self.serial.lcr,
                self.serial.lcr & 0x80 != 0
            );
        }
        match port {
            COM1_BASE..=0x3ff => self.serial.write(port - COM1_BASE, value as u8),
            PIT_BASE..=0x43 => self.pit.write(port - PIT_BASE, value as u8, self.now),
            PORT_B => {
                self.port61 = value as u8 & 0x03;
                self.pit.set_gate2(value as u8 & 0x01 != 0, self.now);
            }
            RTC_INDEX => self.rtc.write_index(value as u8),
            RTC_DATA => self.rtc.write_data(value as u8),
            _ => {}
        }
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.serial.digest());
        d.mix_u64(self.pit.digest());
        d.mix_u64(self.rtc.digest());
        d.mix_bytes(&[self.port61]);
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmit_collects_output() {
        let mut uart = Uart16550::new();
        for b in b"OK" {
            uart.write(0, *b);
        }
        assert_eq!(uart.output_string(), "OK");
        // Line status reports ready to transmit.
        assert_eq!(uart.read(5) & 0x20, 0x20);
    }

    #[test]
    fn unknown_port_reads_floating_bus() {
        let mut bus = IoBus::new();
        assert_eq!(bus.read(0x80, 1), 0xff);
    }

    #[test]
    fn thre_interrupt_arms_with_ier_bit1() {
        let mut uart = Uart16550::new();
        // No THRE-enable yet: no interrupt, IIR reports "none pending" (0x01).
        assert!(!uart.tx_irq_pending());
        assert_eq!(uart.read(2), 0x01);
        // Kernel's tty tx path enables the transmit-empty interrupt (IER bit 1).
        uart.write(1, 0x02);
        assert!(uart.tx_irq_pending(), "THRE interrupt armed");
        assert_eq!(uart.read(2), 0x02, "IIR identifies the THRE source");
        // Disabling it (buffer drained → __stop_tx) lowers the line again.
        uart.write(1, 0x00);
        assert!(!uart.tx_irq_pending());
        assert_eq!(uart.read(2), 0x01);
    }
}
