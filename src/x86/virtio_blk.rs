//! Minimal virtio-blk over the legacy virtio-mmio transport (version 1).
//!
//! One virtqueue, READ/WRITE requests, backed by an in-memory deterministic
//! disk. Requests are processed synchronously when the guest writes QueueNotify
//! (deterministic: tied to guest progress, not host I/O timing — GOLDEN_TARGET
//! §8), then a completion interrupt is signalled (polled by the machine).
//! Descriptor/ring DMA uses guest-physical addresses directly.

use crate::memory::GuestMemory;
use crate::state::StateDigest;

pub const VIRTIO_BLK_BASE: u64 = 0xD000_0000;
const VIRTIO_BLK_LEN: u64 = 0x200;
/// Default interrupt vector for the block device (used if the IOAPIC has no
/// redirection entry for its GSI).
pub const VIRTIO_BLK_VECTOR: u8 = 0x30;
/// Global system interrupt line the block device raises (routed via the IOAPIC).
pub const VIRTIO_BLK_GSI: u32 = 5;

const SECTOR: u64 = 512;
const QUEUE_NUM_MAX: u32 = 8;

// virtio-mmio register offsets (legacy).
const MAGIC: u64 = 0x000;
const VERSION: u64 = 0x004;
const DEVICE_ID: u64 = 0x008;
const VENDOR_ID: u64 = 0x00c;
const HOST_FEATURES: u64 = 0x010;
const GUEST_PAGE_SIZE: u64 = 0x028;
const QUEUE_SEL: u64 = 0x030;
const QUEUE_NUM_MAX_REG: u64 = 0x034;
const QUEUE_NUM: u64 = 0x038;
const QUEUE_ALIGN: u64 = 0x03c;
const QUEUE_PFN: u64 = 0x040;
const QUEUE_NOTIFY: u64 = 0x050;
const INTERRUPT_STATUS: u64 = 0x060;
const INTERRUPT_ACK: u64 = 0x064;
const STATUS: u64 = 0x070;
const CONFIG: u64 = 0x100; // capacity (sectors), 64-bit

// virtqueue descriptor flags.
const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

// virtio-blk request types.
const VIRTIO_BLK_T_IN: u32 = 0; // read (device -> guest)
const VIRTIO_BLK_T_OUT: u32 = 1; // write (guest -> device)

// virtio-blk completion status (written into the final status descriptor byte).
const VIRTIO_BLK_S_OK: u8 = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;

/// SplitMix64 — a small, fully deterministic PRNG. Drives fault scheduling so a
/// DST run replays the exact same I/O-fault sequence for a given seed.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What befalls a single block request. Chosen deterministically per request from
/// the fault policy's PRNG stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    /// Normal completion.
    None,
    /// I/O error: the device moves no data and reports S_IOERR to the guest.
    Eio,
    /// Torn transfer: only the first half of each data buffer is moved, yet the
    /// device reports success — silent partial corruption (the nastiest class of
    /// bug a DST harness should be able to surface and reproduce).
    Torn,
}

/// Deterministic, seed-driven fault-injection policy for the block device. A DST
/// run configures one; replaying the same seed reproduces the identical sequence
/// of EIO / torn / latency-jitter decisions. Absent a policy, the device never
/// faults and behaves byte-identically to a plain disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlkFaultPolicy {
    /// SplitMix64 state — advances once per submitted request.
    state: u64,
    /// P(EIO) per request, in parts-per-thousand.
    eio_permille: u32,
    /// P(torn write) per request, in parts-per-thousand.
    torn_permille: u32,
    /// Max extra completion latency (instructions) added per request. Non-zero
    /// jitter lets in-flight requests complete out of submission order (reorder).
    jitter: u64,
}

impl BlkFaultPolicy {
    pub fn new(seed: u64, eio_permille: u32, torn_permille: u32, jitter: u64) -> Self {
        Self {
            state: seed ^ 0xD1B5_4A32_D192_ED03,
            eio_permille,
            torn_permille,
            jitter,
        }
    }

    /// Draw the fault and extra latency for the next request; advances the stream.
    fn next(&mut self) -> (Fault, u64) {
        let r = splitmix64(&mut self.state);
        let roll = (r % 1000) as u32;
        let fault = if roll < self.eio_permille {
            Fault::Eio
        } else if roll < self.eio_permille + self.torn_permille {
            Fault::Torn
        } else {
            Fault::None
        };
        let extra = if self.jitter == 0 {
            0
        } else {
            (r >> 32) % (self.jitter + 1)
        };
        (fault, extra)
    }

    fn digest(&self, d: &mut StateDigest) {
        d.mix_u64(self.state);
        d.mix_u64(self.eio_permille as u64);
        d.mix_u64(self.torn_permille as u64);
        d.mix_u64(self.jitter);
    }
}

/// A processed request awaiting completion. The data move happens immediately on
/// notify, but the used-ring update + interrupt are deferred to `complete_at`
/// (instruction count) — controllable I/O: the completion can be slid in virtual
/// time relative to the CPU, deterministically.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pending {
    head: u16,
    written: u32,
    complete_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtioBlk {
    disk: Vec<u8>,
    guest_page_size: u32,
    queue_num: u32,
    queue_align: u32,
    queue_pfn: u32,
    status: u32,
    interrupt_status: u32,
    last_avail: u16,
    irq_pending: bool,
    /// In-flight requests whose completion is scheduled in virtual time.
    pending: Vec<Pending>,
    /// Completion latency in instructions (virtual time). Deterministic; a future
    /// DST layer varies it per seed to explore I/O/CPU interleavings.
    latency: u64,
    /// Optional deterministic fault-injection policy. `None` = a perfect disk.
    fault: Option<BlkFaultPolicy>,
}

/// Default deterministic disk-completion latency (~a few µs at 1 instr = 1 ns).
pub const VIRTIO_BLK_LATENCY: u64 = 20_000;

struct Desc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

impl VirtioBlk {
    pub fn new(disk: Vec<u8>) -> Self {
        Self {
            disk,
            guest_page_size: 4096,
            queue_num: 0,
            queue_align: 4096,
            queue_pfn: 0,
            status: 0,
            interrupt_status: 0,
            last_avail: 0,
            irq_pending: false,
            pending: Vec::new(),
            latency: VIRTIO_BLK_LATENCY,
            fault: None,
        }
    }

    /// Install a deterministic fault-injection policy (DST control surface).
    pub fn set_fault_policy(&mut self, policy: BlkFaultPolicy) {
        self.fault = Some(policy);
    }

    pub fn offset_of(phys: u64) -> Option<u64> {
        if (VIRTIO_BLK_BASE..VIRTIO_BLK_BASE + VIRTIO_BLK_LEN).contains(&phys) {
            Some(phys - VIRTIO_BLK_BASE)
        } else {
            None
        }
    }

    pub fn read(&self, off: u64) -> u32 {
        let cap = self.disk.len() as u64 / SECTOR;
        let v = match off {
            MAGIC => 0x7472_6976,   // "virt"
            VERSION => 1,           // legacy
            DEVICE_ID => 2,         // block
            VENDOR_ID => 0x554d_4551,
            HOST_FEATURES => 0,     // no negotiated features
            QUEUE_NUM_MAX_REG => QUEUE_NUM_MAX,
            QUEUE_PFN => self.queue_pfn,
            INTERRUPT_STATUS => self.interrupt_status,
            STATUS => self.status,
            // Config space (virtio_blk_config): capacity is the first 8 bytes.
            // The LEGACY virtio-mmio driver reads config BYTE-BY-BYTE, so any
            // offset 0x100..0x108 must return the correct bytes of the capacity
            // (a 1/2/4-byte access takes the low bytes of what we return here).
            o if (CONFIG..CONFIG + 8).contains(&o) => {
                ((cap >> ((o - CONFIG) * 8)) & 0xffff_ffff) as u32
            }
            _ => 0,
        };
        if std::env::var("SF_VIOTRACE").is_ok() {
            eprintln!("VIO rd off={off:#05x} -> {v:#x}");
        }
        v
    }

    pub fn write(&mut self, off: u64, value: u32, mem: &mut GuestMemory, now: u64) {
        if std::env::var("SF_VIOTRACE").is_ok() {
            eprintln!("VIO wr off={off:#05x} <- {value:#x}");
        }
        match off {
            GUEST_PAGE_SIZE => self.guest_page_size = value,
            QUEUE_SEL => {}
            QUEUE_NUM => self.queue_num = value,
            QUEUE_ALIGN => self.queue_align = value,
            QUEUE_PFN => self.queue_pfn = value,
            QUEUE_NOTIFY => self.process_queue(mem, now),
            INTERRUPT_ACK => self.interrupt_status &= !value,
            STATUS => self.status = value,
            _ => {}
        }
    }

    /// Virtqueue layout helper: (descriptor base, ring entries, used-ring base).
    fn ring_layout(&self) -> (u64, u64, u64) {
        let num = self.queue_num as u64;
        let desc_base = self.queue_pfn as u64 * self.guest_page_size as u64;
        let avail_base = desc_base + num * 16;
        let used_base = align(avail_base + 6 + num * 2, self.queue_align as u64);
        (desc_base, num, used_base)
    }

    /// Next scheduled completion deadline in virtual time (for idle fast-forward).
    pub fn next_completion_deadline(&self) -> Option<u64> {
        self.pending.iter().map(|p| p.complete_at).min()
    }

    /// Apply any completions whose deadline has arrived: write the used-ring entry
    /// and raise the device interrupt. Called by the machine each step.
    pub fn poll_completions(&mut self, now: u64, mem: &mut GuestMemory) {
        if self.pending.is_empty() {
            return;
        }
        let (_, num, used_base) = self.ring_layout();
        let mut completed = false;
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].complete_at <= now {
                let p = self.pending.remove(i);
                let used_idx = mem.read_uint_le(used_base + 2, 2).unwrap_or(0) as u16;
                let e = used_base + 4 + (used_idx as u64 % num) * 8;
                let _ = mem.write_uint_le(e, 4, p.head as u64);
                let _ = mem.write_uint_le(e + 4, 4, p.written as u64);
                let _ = mem.write_uint_le(used_base + 2, 2, used_idx.wrapping_add(1) as u64);
                completed = true;
            } else {
                i += 1;
            }
        }
        if completed {
            self.interrupt_status |= 1;
            self.irq_pending = true;
        }
    }

    fn read_desc(&self, mem: &GuestMemory, desc_base: u64, idx: u16) -> Desc {
        let a = desc_base + idx as u64 * 16;
        Desc {
            addr: mem.read_uint_le(a, 8).unwrap_or(0),
            len: mem.read_uint_le(a + 8, 4).unwrap_or(0) as u32,
            flags: mem.read_uint_le(a + 12, 2).unwrap_or(0) as u16,
            next: mem.read_uint_le(a + 14, 2).unwrap_or(0) as u16,
        }
    }

    fn process_queue(&mut self, mem: &mut GuestMemory, now: u64) {
        if self.queue_num == 0 || self.queue_pfn == 0 {
            return;
        }
        let (desc_base, num, _) = self.ring_layout();
        let avail_base = desc_base + num * 16;

        let avail_idx = mem.read_uint_le(avail_base + 2, 2).unwrap_or(0) as u16;
        while self.last_avail != avail_idx {
            let ring_slot = self.last_avail as u64 % num;
            let head = mem.read_uint_le(avail_base + 4 + ring_slot * 2, 2).unwrap_or(0) as u16;
            // Draw this request's fate from the (deterministic) fault policy
            // before touching any data: EIO suppresses the transfer, torn moves
            // only half, and `extra` jitter can slide its completion past a later
            // request's — producing reordered completions, reproducibly.
            let (fault, extra) = self
                .fault
                .as_mut()
                .map(|p| p.next())
                .unwrap_or((Fault::None, 0));
            // The data transfer is applied now (the bytes move), but the guest is
            // not told until the scheduled completion — that is the controllable
            // I/O latency. Stacking `latency` lets several queued requests complete
            // in submission order at deterministic, spaced instants.
            let written = self.process_request(mem, desc_base, head, fault);
            let base = self.next_completion_deadline().unwrap_or(now);
            let complete_at = base.max(now).saturating_add(self.latency).saturating_add(extra);
            self.pending.push(Pending { head, written, complete_at });
            self.last_avail = self.last_avail.wrapping_add(1);
        }
    }

    /// Walk a request's descriptor chain (header, data..., status) and return the
    /// number of bytes written into guest buffers (for the used-ring length).
    fn process_request(
        &mut self,
        mem: &mut GuestMemory,
        desc_base: u64,
        head: u16,
        fault: Fault,
    ) -> u32 {
        let hdr = self.read_desc(mem, desc_base, head);
        let req_type = mem.read_uint_le(hdr.addr, 4).unwrap_or(0) as u32;
        let sector = mem.read_uint_le(hdr.addr + 8, 8).unwrap_or(0);
        let mut written = 0_u32;

        let mut cur = hdr;
        // Data descriptors until the final (status) descriptor. The descriptor
        // table has at most `queue_num` entries, so a well-formed chain visits no
        // more than that; bound the walk to drop a malicious/looping chain (every
        // descriptor with NEXT set) instead of hanging the host (Review 4 H3).
        let max_descs = self.queue_num as u64;
        let mut steps = 0_u64;
        while cur.flags & VRING_DESC_F_NEXT != 0 {
            steps += 1;
            if steps > max_descs {
                break;
            }
            let d = self.read_desc(mem, desc_base, cur.next);
            let is_status = d.flags & VRING_DESC_F_NEXT == 0;
            if is_status {
                // Final byte: completion status. EIO reports an error; everything
                // else (including a silently-torn transfer) reports success.
                let status = match fault {
                    Fault::Eio => VIRTIO_BLK_S_IOERR,
                    _ => VIRTIO_BLK_S_OK,
                };
                let _ = mem.write_uint_le(d.addr, 1, status as u64);
                written += 1;
                break;
            }
            // A data buffer: READ copies disk -> guest (device-writable buffer),
            // WRITE copies guest -> disk. A READ only fills descriptors the guest
            // marked writable (honor the flag rather than assert on it, so a
            // malformed/adversarial chain is handled gracefully, not panicked).
            //
            // Fault handling: EIO moves nothing (the data buffer is left as the
            // guest set it). Torn moves only the first half of the buffer but the
            // device still reports the full length and S_OK — modelling silent
            // partial I/O. The device still claims `d.len` in the used-ring so the
            // guest believes the whole transfer happened.
            let off = sector * SECTOR;
            let moved = match fault {
                Fault::Eio => 0,
                Fault::Torn => d.len as u64 / 2,
                Fault::None => d.len as u64,
            };
            if req_type == VIRTIO_BLK_T_IN && d.flags & VRING_DESC_F_WRITE != 0 {
                for i in 0..moved {
                    let byte = self.disk.get((off + i) as usize).copied().unwrap_or(0);
                    let _ = mem.write_uint_le(d.addr + i, 1, byte as u64);
                }
                written += d.len;
            } else if req_type == VIRTIO_BLK_T_OUT {
                for i in 0..moved {
                    let byte = mem.read_uint_le(d.addr + i, 1).unwrap_or(0) as u8;
                    if let Some(slot) = self.disk.get_mut((off + i) as usize) {
                        *slot = byte;
                    }
                }
            }
            cur = d;
        }
        written
    }

    /// Take a pending completion interrupt, if any.
    pub fn take_interrupt(&mut self) -> Option<u8> {
        if self.irq_pending {
            self.irq_pending = false;
            Some(VIRTIO_BLK_VECTOR)
        } else {
            None
        }
    }

    pub fn disk(&self) -> &[u8] {
        &self.disk
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.disk.len() as u64);
        d.mix_bytes(&self.disk);
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
        d.mix_u64(self.latency);
        match &self.fault {
            Some(f) => {
                d.mix_u64(1);
                f.digest(&mut d);
            }
            None => d.mix_u64(0),
        }
        for p in &self.pending {
            d.mix_u64(p.head as u64);
            d.mix_u64(p.written as u64);
            d.mix_u64(p.complete_at);
        }
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
    fn config_capacity_reads_byte_by_byte() {
        // The legacy virtio-mmio driver reads config space one byte at a time;
        // assembling capacity from individual byte reads must equal the true
        // sector count (the bug that made the guest see a 0-block disk).
        let blk = VirtioBlk::new(vec![0_u8; 1024 * 1024]); // 2048 sectors
        let mut cap = 0_u64;
        for i in 0..8u64 {
            let byte = (blk.read(CONFIG + i) & 0xff) as u64;
            cap |= byte << (i * 8);
        }
        assert_eq!(cap, 2048, "capacity assembled from byte reads = 2048 sectors");
        // A 32-bit read at the aligned offsets still works too.
        assert_eq!(blk.read(CONFIG), 2048);
        assert_eq!(blk.read(CONFIG + 4), 0);
    }

    #[test]
    fn read_request_copies_disk_to_guest() {
        let mut disk = vec![0_u8; 4096];
        disk[512] = 0xab; // sector 1, byte 0
        disk[513] = 0xcd;
        let mut blk = VirtioBlk::new(disk);
        let mut mem = GuestMemory::new(0x20000);

        // Lay out a queue at PFN 1 (page 0x1000), num=8.
        blk.write(GUEST_PAGE_SIZE, 0x1000, &mut mem, 0);
        blk.write(QUEUE_NUM, 8, &mut mem, 0);
        blk.write(QUEUE_ALIGN, 0x1000, &mut mem, 0);
        blk.write(QUEUE_PFN, 1, &mut mem, 0);

        let desc_base = 0x1000_u64;
        let num = 8_u64;
        let avail_base = desc_base + num * 16;
        // header @ 0x4000, data buffer @ 0x4100, status @ 0x4200
        mem.write_uint_le(0x4000, 4, VIRTIO_BLK_T_IN as u64).unwrap(); // read
        mem.write_uint_le(0x4008, 8, 1).unwrap(); // sector 1
        // desc 0 = header (NEXT), desc 1 = data (NEXT|WRITE), desc 2 = status (WRITE)
        let wd = |mem: &mut GuestMemory, i: u64, addr: u64, len: u32, flags: u16, next: u16| {
            let a = desc_base + i * 16;
            mem.write_uint_le(a, 8, addr).unwrap();
            mem.write_uint_le(a + 8, 4, len as u64).unwrap();
            mem.write_uint_le(a + 12, 2, flags as u64).unwrap();
            mem.write_uint_le(a + 14, 2, next as u64).unwrap();
        };
        wd(&mut mem, 0, 0x4000, 16, VRING_DESC_F_NEXT, 1);
        wd(&mut mem, 1, 0x4100, 16, VRING_DESC_F_NEXT | VRING_DESC_F_WRITE, 2);
        wd(&mut mem, 2, 0x4200, 1, VRING_DESC_F_WRITE, 0);
        // avail: idx=1, ring[0]=head 0
        mem.write_uint_le(avail_base + 4, 2, 0).unwrap();
        mem.write_uint_le(avail_base + 2, 2, 1).unwrap();

        blk.write(QUEUE_NOTIFY, 0, &mut mem, 0);

        // The data transfer is applied immediately on notify...
        assert_eq!(mem.read_uint_le(0x4100, 1).unwrap(), 0xab);
        assert_eq!(mem.read_uint_le(0x4101, 1).unwrap(), 0xcd);
        assert_eq!(mem.read_uint_le(0x4200, 1).unwrap(), 0, "status OK");
        // ...but the completion interrupt is deferred until the deadline arrives.
        assert_eq!(blk.take_interrupt(), None, "completion deferred (controllable I/O)");
        blk.poll_completions(VIRTIO_BLK_LATENCY, &mut mem);
        assert_eq!(blk.take_interrupt(), Some(VIRTIO_BLK_VECTOR));
    }

    /// Build a single READ request (header @0x4000, data @0x4100 len `dlen`,
    /// status @0x4200) into the queue and notify. Returns nothing; inspect memory.
    fn submit_read(blk: &mut VirtioBlk, mem: &mut GuestMemory, sector: u64, dlen: u32) {
        blk.write(GUEST_PAGE_SIZE, 0x1000, mem, 0);
        blk.write(QUEUE_NUM, 8, mem, 0);
        blk.write(QUEUE_ALIGN, 0x1000, mem, 0);
        blk.write(QUEUE_PFN, 1, mem, 0);
        let desc_base = 0x1000_u64;
        let num = 8_u64;
        let avail_base = desc_base + num * 16;
        mem.write_uint_le(0x4000, 4, VIRTIO_BLK_T_IN as u64).unwrap();
        mem.write_uint_le(0x4008, 8, sector).unwrap();
        let wd = |mem: &mut GuestMemory, i: u64, addr: u64, len: u32, flags: u16, next: u16| {
            let a = desc_base + i * 16;
            mem.write_uint_le(a, 8, addr).unwrap();
            mem.write_uint_le(a + 8, 4, len as u64).unwrap();
            mem.write_uint_le(a + 12, 2, flags as u64).unwrap();
            mem.write_uint_le(a + 14, 2, next as u64).unwrap();
        };
        wd(mem, 0, 0x4000, 16, VRING_DESC_F_NEXT, 1);
        wd(mem, 1, 0x4100, dlen, VRING_DESC_F_NEXT | VRING_DESC_F_WRITE, 2);
        wd(mem, 2, 0x4200, 1, VRING_DESC_F_WRITE, 0);
        let slot = blk.last_avail as u64 % num;
        mem.write_uint_le(avail_base + 4 + slot * 2, 2, 0).unwrap();
        mem.write_uint_le(avail_base + 2, 2, blk.last_avail.wrapping_add(1) as u64)
            .unwrap();
        blk.write(QUEUE_NOTIFY, 0, mem, 0);
    }

    #[test]
    fn eio_fault_reports_ioerr_and_moves_no_data() {
        let mut disk = vec![0_u8; 4096];
        for b in disk.iter_mut().take(8) {
            *b = 0xee; // sector 0 marker
        }
        let mut blk = VirtioBlk::new(disk);
        // 1000 permille EIO => every request faults, deterministically.
        blk.set_fault_policy(BlkFaultPolicy::new(1, 1000, 0, 0));
        let mut mem = GuestMemory::new(0x20000);
        // Pre-stain the guest buffer so we can prove the device wrote nothing.
        for i in 0..8 {
            mem.write_uint_le(0x4100 + i, 1, 0x55).unwrap();
        }
        submit_read(&mut blk, &mut mem, 0, 8);
        assert_eq!(mem.read_uint_le(0x4100, 1).unwrap(), 0x55, "no data moved on EIO");
        assert_eq!(mem.read_uint_le(0x4200, 1).unwrap(), VIRTIO_BLK_S_IOERR as u64);
    }

    #[test]
    fn torn_fault_moves_half_but_reports_success() {
        let mut disk = vec![0_u8; 4096];
        for b in disk.iter_mut().take(8) {
            *b = 0xab;
        }
        let mut blk = VirtioBlk::new(disk);
        blk.set_fault_policy(BlkFaultPolicy::new(7, 0, 1000, 0)); // always torn
        let mut mem = GuestMemory::new(0x20000);
        for i in 0..8 {
            mem.write_uint_le(0x4100 + i, 1, 0x00).unwrap();
        }
        submit_read(&mut blk, &mut mem, 0, 8);
        // Only the first half (4 of 8) bytes are delivered; the rest stay stale.
        assert_eq!(mem.read_uint_le(0x4100, 1).unwrap(), 0xab);
        assert_eq!(mem.read_uint_le(0x4103, 1).unwrap(), 0xab);
        assert_eq!(mem.read_uint_le(0x4104, 1).unwrap(), 0x00, "torn: upper half not moved");
        // ...yet the device claims success — the silent-corruption case.
        assert_eq!(mem.read_uint_le(0x4200, 1).unwrap(), VIRTIO_BLK_S_OK as u64);
    }

    #[test]
    fn fault_policy_is_deterministic_for_a_seed() {
        // Same seed => identical fault/jitter draw sequence (replayable DST runs).
        let mut a = BlkFaultPolicy::new(42, 300, 300, 1000);
        let mut b = BlkFaultPolicy::new(42, 300, 300, 1000);
        for _ in 0..64 {
            assert_eq!(a.next(), b.next());
        }
        // A different seed diverges (otherwise the seed would be inert).
        let mut c = BlkFaultPolicy::new(43, 300, 300, 1000);
        let mut d = BlkFaultPolicy::new(42, 300, 300, 1000);
        let seq_c: Vec<_> = (0..64).map(|_| c.next()).collect();
        let seq_d: Vec<_> = (0..64).map(|_| d.next()).collect();
        assert_ne!(seq_c, seq_d);
    }

    #[test]
    fn no_policy_means_no_faults() {
        // The default device must behave byte-identically to a perfect disk.
        let mut disk = vec![0_u8; 4096];
        for b in disk.iter_mut().take(8) {
            *b = 0x9c;
        }
        let mut blk = VirtioBlk::new(disk);
        let mut mem = GuestMemory::new(0x20000);
        submit_read(&mut blk, &mut mem, 0, 8);
        assert_eq!(mem.read_uint_le(0x4100, 1).unwrap(), 0x9c);
        assert_eq!(mem.read_uint_le(0x4107, 1).unwrap(), 0x9c, "full transfer");
        assert_eq!(mem.read_uint_le(0x4200, 1).unwrap(), VIRTIO_BLK_S_OK as u64);
    }

    #[test]
    fn cyclic_descriptor_chain_terminates() {
        // A malicious descriptor 0 whose NEXT points back to itself must NOT hang
        // the host — the walk is bounded by queue_num (Review 4 H3).
        let mut blk = VirtioBlk::new(vec![0_u8; 4096]);
        let mut mem = GuestMemory::new(0x20000);
        blk.write(GUEST_PAGE_SIZE, 0x1000, &mut mem, 0);
        blk.write(QUEUE_NUM, 8, &mut mem, 0);
        blk.write(QUEUE_ALIGN, 0x1000, &mut mem, 0);
        blk.write(QUEUE_PFN, 1, &mut mem, 0);

        let desc_base = 0x1000_u64;
        let avail_base = desc_base + 8 * 16;
        mem.write_uint_le(0x4000, 4, VIRTIO_BLK_T_IN as u64).unwrap();
        // desc 0: header with NEXT pointing at itself -> a cycle.
        mem.write_uint_le(desc_base, 8, 0x4000).unwrap();
        mem.write_uint_le(desc_base + 8, 4, 16).unwrap();
        mem.write_uint_le(desc_base + 12, 2, VRING_DESC_F_NEXT as u64).unwrap();
        mem.write_uint_le(desc_base + 14, 2, 0).unwrap(); // next = 0 (self)
        mem.write_uint_le(avail_base + 4, 2, 0).unwrap();
        mem.write_uint_le(avail_base + 2, 2, 1).unwrap();

        // Completes (returns) instead of looping forever.
        blk.write(QUEUE_NOTIFY, 0, &mut mem, 0);
        blk.poll_completions(VIRTIO_BLK_LATENCY, &mut mem);
        assert_eq!(blk.take_interrupt(), Some(VIRTIO_BLK_VECTOR));
    }
}
