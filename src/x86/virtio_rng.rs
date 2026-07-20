//! virtio-rng (entropy) over legacy virtio-mmio. The device fills the guest's
//! writable buffers with deterministic seed-derived bytes (GOLDEN_TARGET §6:
//! guest entropy is a function of the seed, never the host RNG).

use crate::memory::GuestMemory;
use crate::seed::{Seed256, StreamRng};
use crate::state::StateDigest;

pub const VIRTIO_RNG_BASE: u64 = 0xD000_1000;
const VIRTIO_RNG_LEN: u64 = 0x200;
// Distinct GSI/vector from virtio-net (GSI 6 / 0x31) so the rng and NIC never share an
// IRQ line when both are active (the csharp distributed case binds both).
pub const VIRTIO_RNG_VECTOR: u8 = 0x32;
pub const VIRTIO_RNG_GSI: u32 = 7;

const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtioRng {
    rng: StreamRng,
    guest_page_size: u32,
    queue_num: u32,
    queue_align: u32,
    queue_pfn: u32,
    status: u32,
    interrupt_status: u32,
    last_avail: u16,
    irq_pending: bool,
}

impl Default for VirtioRng {
    fn default() -> Self {
        Self {
            rng: StreamRng::new(0),
            guest_page_size: 4096,
            queue_num: 0,
            queue_align: 4096,
            queue_pfn: 0,
            status: 0,
            interrupt_status: 0,
            last_avail: 0,
            irq_pending: false,
        }
    }
}

impl VirtioRng {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed the entropy stream from the run seed (called by the machine).
    pub fn reseed(&mut self, seed: &Seed256) {
        self.rng = seed.derive("virtio-rng");
    }

    pub fn offset_of(phys: u64) -> Option<u64> {
        if (VIRTIO_RNG_BASE..VIRTIO_RNG_BASE + VIRTIO_RNG_LEN).contains(&phys) {
            Some(phys - VIRTIO_RNG_BASE)
        } else {
            None
        }
    }

    pub fn read(&self, off: u64) -> u32 {
        match off {
            0x000 => 0x7472_6976, // "virt"
            0x004 => 1,           // legacy
            0x008 => 4,           // entropy device
            0x00c => 0x554d_4551,
            0x034 => 8, // QueueNumMax
            0x040 => self.queue_pfn,
            0x060 => self.interrupt_status,
            0x070 => self.status,
            _ => 0,
        }
    }

    pub fn write(&mut self, off: u64, value: u32, mem: &mut GuestMemory) {
        match off {
            0x028 => self.guest_page_size = value,
            0x038 => self.queue_num = value,
            0x03c => self.queue_align = value,
            0x040 => self.queue_pfn = value,
            0x050 => self.process_queue(mem),
            0x064 => self.interrupt_status &= !value,
            0x070 => self.status = value,
            _ => {}
        }
    }

    fn process_queue(&mut self, mem: &mut GuestMemory) {
        if self.queue_num == 0 || self.queue_pfn == 0 {
            return;
        }
        let num = self.queue_num as u64;
        let desc_base = self.queue_pfn as u64 * self.guest_page_size as u64;
        let avail_base = desc_base + num * 16;
        let used_base = align(avail_base + 6 + num * 2, self.queue_align as u64);
        let avail_idx = mem.read_uint_le(avail_base + 2, 2).unwrap_or(0) as u16;

        while self.last_avail != avail_idx {
            let slot = self.last_avail as u64 % num;
            let mut idx = mem.read_uint_le(avail_base + 4 + slot * 2, 2).unwrap_or(0) as u16;
            let mut written = 0_u32;
            // Bound the chain walk to the descriptor-table size so a cyclic chain
            // (every descriptor with NEXT set) is dropped instead of hanging the
            // host (Review 4 H3).
            let mut steps = 0_u64;
            loop {
                let a = desc_base + idx as u64 * 16;
                let addr = mem.read_uint_le(a, 8).unwrap_or(0);
                let len = mem.read_uint_le(a + 8, 4).unwrap_or(0) as u32;
                let flags = mem.read_uint_le(a + 12, 2).unwrap_or(0) as u16;
                let next = mem.read_uint_le(a + 14, 2).unwrap_or(0) as u16;
                if flags & VRING_DESC_F_WRITE != 0 {
                    for i in 0..len as u64 {
                        let _ = mem.write_uint_le(addr + i, 1, self.rng.next_u64() & 0xff);
                    }
                    written += len;
                }
                if flags & VRING_DESC_F_NEXT == 0 {
                    break;
                }
                steps += 1;
                if steps >= num {
                    break;
                }
                idx = next;
            }

            let used_idx = mem.read_uint_le(used_base + 2, 2).unwrap_or(0) as u16;
            let used_slot = used_idx as u64 % num;
            let e = used_base + 4 + used_slot * 8;
            let head = mem.read_uint_le(avail_base + 4 + slot * 2, 2).unwrap_or(0);
            let _ = mem.write_uint_le(e, 4, head);
            let _ = mem.write_uint_le(e + 4, 4, written as u64);
            let _ = mem.write_uint_le(used_base + 2, 2, used_idx.wrapping_add(1) as u64);
            self.last_avail = self.last_avail.wrapping_add(1);
        }
        self.interrupt_status |= 1;
        self.irq_pending = true;
    }

    pub fn take_interrupt(&mut self) -> Option<u8> {
        if self.irq_pending {
            self.irq_pending = false;
            Some(VIRTIO_RNG_VECTOR)
        } else {
            None
        }
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.rng.state());
        for v in [
            self.guest_page_size,
            self.queue_num,
            self.queue_align,
            self.queue_pfn,
            self.status,
            self.interrupt_status,
            self.last_avail as u32,
        ] {
            d.mix_u64(v as u64);
        }
        d.mix_u64(self.irq_pending as u64);
        d.finish()
    }
}

fn align(x: u64, a: u64) -> u64 {
    if a == 0 {
        x
    } else {
        (x + a - 1) & !(a - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_buffer_with_deterministic_entropy() {
        let seed = Seed256::from_hex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        )
        .unwrap();
        let mut a = VirtioRng::new();
        let mut b = VirtioRng::new();
        a.reseed(&seed);
        b.reseed(&seed);

        let run = |dev: &mut VirtioRng| -> Vec<u8> {
            let mut mem = GuestMemory::new(0x10000);
            dev.write(0x028, 0x1000, &mut mem);
            dev.write(0x038, 8, &mut mem);
            dev.write(0x03c, 0x1000, &mut mem);
            dev.write(0x040, 1, &mut mem);
            let desc_base = 0x1000_u64;
            let avail_base = desc_base + 8 * 16;
            mem.write_uint_le(desc_base, 8, 0x4000).unwrap(); // buffer @ 0x4000
            mem.write_uint_le(desc_base + 8, 4, 16).unwrap();
            mem.write_uint_le(desc_base + 12, 2, VRING_DESC_F_WRITE as u64).unwrap();
            mem.write_uint_le(avail_base + 4, 2, 0).unwrap();
            mem.write_uint_le(avail_base + 2, 2, 1).unwrap();
            dev.write(0x050, 0, &mut mem);
            (0..16).map(|i| mem.read_uint_le(0x4000 + i, 1).unwrap() as u8).collect()
        };

        let out_a = run(&mut a);
        let out_b = run(&mut b);
        assert_eq!(out_a, out_b, "same seed -> same entropy");
        assert!(out_a.iter().any(|&x| x != 0), "buffer was filled");
        assert_eq!(a.take_interrupt(), Some(VIRTIO_RNG_VECTOR));
    }
}
