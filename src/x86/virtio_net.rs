//! Minimal virtio-net over the legacy virtio-mmio transport (version 1).
//!
//! Two virtqueues — RX (0, device→guest) and TX (1, guest→device) — a fixed MAC
//! (VIRTIO_NET_F_MAC), and the 10-byte legacy `virtio_net_hdr` on every frame.
//! Fully deterministic: TX frames are drained on QueueNotify (tied to guest
//! progress, not host timing); RX delivery + the used-ring/interrupt are deferred
//! to a virtual-time deadline (controllable I/O). A DST harness connects the
//! device via [`take_tx_frames`] / [`deliver_frame`]:
//!   * Phase 1a bring-up = a loopback reflector (TX → fault policy → RX).
//!   * Phase 1b = a deterministic switch routing between two guests' NICs.
//! Absent a fault policy the link is lossless and byte-identical to a plain NIC.

use crate::memory::GuestMemory;
use crate::state::StateDigest;

pub const VIRTIO_NET_BASE: u64 = 0xD000_2000; // blk=0xD0000000, rng=0xD0001000
const VIRTIO_NET_LEN: u64 = 0x200;
/// Default interrupt vector (used if the IOAPIC has no redirection for the GSI).
pub const VIRTIO_NET_VECTOR: u8 = 0x31;
/// Global system interrupt line the NIC raises (blk uses GSI 5).
pub const VIRTIO_NET_GSI: u32 = 6;

const QUEUE_NUM_MAX: u32 = 256;
/// Legacy `virtio_net_hdr` size without MRG_RXBUF (flags,gso_type,hdr_len,
/// gso_size,csum_start,csum_offset = 1+1+2+2+2+2).
const NET_HDR_LEN: u64 = 10;

// virtio-mmio register offsets (legacy) — identical layout to virtio-blk.
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
const CONFIG: u64 = 0x100; // virtio_net_config: mac[6], status(u16)

// virtqueue descriptor flags.
const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

// feature bits.
const VIRTIO_NET_F_MAC: u32 = 1 << 5;

const RX: usize = 0;
const TX: usize = 1;

/// Default RX-delivery latency in instructions (virtual time; ~a few µs).
pub const VIRTIO_NET_LATENCY: u64 = 20_000;

/// SplitMix64 — deterministic PRNG driving the fault schedule (matches virtio-blk).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn align(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

/// Deterministic, seed-driven network fault policy. Replaying the same seed
/// reproduces the identical drop / latency-jitter / partition decisions. A DST
/// run configures one; absent it, the link is perfect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetFaultPolicy {
    state: u64,
    /// P(drop) per frame, in parts-per-thousand (frame is silently lost).
    drop_permille: u32,
    /// Max extra RX-delivery latency (instructions) per frame; non-zero jitter
    /// lets frames arrive out of send order (reorder), reproducibly.
    jitter: u64,
}

/// What befalls a single frame in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NetFault {
    None,
    Drop,
}

impl NetFaultPolicy {
    pub fn new(seed: u64, drop_permille: u32, jitter: u64) -> Self {
        Self { state: seed ^ 0x2545_F491_4F6C_DD1D, drop_permille, jitter }
    }

    /// Draw the fate + extra latency for the next frame; advances the stream.
    fn next(&mut self) -> (NetFault, u64) {
        let r = splitmix64(&mut self.state);
        let fault = if ((r % 1000) as u32) < self.drop_permille {
            NetFault::Drop
        } else {
            NetFault::None
        };
        let extra = if self.jitter == 0 { 0 } else { (r >> 32) % (self.jitter + 1) };
        (fault, extra)
    }

    fn digest(&self, d: &mut StateDigest) {
        d.mix_u64(self.state);
        d.mix_u64(self.drop_permille as u64);
        d.mix_u64(self.jitter);
    }
}

/// A frame awaiting RX delivery: the raw ethernet bytes and the virtual-time
/// instant at which the guest is allowed to see it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RxPending {
    frame: Vec<u8>,
    deliver_at: u64,
}

/// A completed request (RX or TX) awaiting its used-ring/interrupt at `complete_at`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pending {
    queue: usize,
    head: u16,
    written: u32,
    complete_at: u64,
}

/// Per-virtqueue state (RX and TX each have their own num/pfn/last_avail).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Queue {
    num: u32,
    pfn: u32,
    last_avail: u16,
}

struct Desc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtioNet {
    mac: [u8; 6],
    guest_page_size: u32,
    queue_align: u32,
    queue_sel: u32,
    queues: [Queue; 2],
    status: u32,
    interrupt_status: u32,
    irq_pending: bool,
    /// Frames the GUEST transmitted, drained by the backend (switch/reflector).
    tx_out: Vec<Vec<u8>>,
    /// Incoming frames scheduled for RX delivery (from the backend).
    rx_queue: Vec<RxPending>,
    /// Completions (RX + TX) scheduled in virtual time.
    pending: Vec<Pending>,
    latency: u64,
    fault: Option<NetFaultPolicy>,
    /// When true (single-guest `--netdev`) the machine step loops TX back to RX
    /// itself. When false (two-guest mode) an external switch drains TX via
    /// [`take_tx_frames`] and injects RX via [`deliver_frame`].
    pub reflector: bool,
}

impl VirtioNet {
    pub fn new(mac: [u8; 6]) -> Self {
        Self {
            mac,
            guest_page_size: 4096,
            queue_align: 4096,
            queue_sel: 0,
            queues: Default::default(),
            status: 0,
            interrupt_status: 0,
            irq_pending: false,
            tx_out: Vec::new(),
            rx_queue: Vec::new(),
            pending: Vec::new(),
            latency: VIRTIO_NET_LATENCY,
            fault: None,
            reflector: true,
        }
    }

    pub fn set_fault_policy(&mut self, policy: NetFaultPolicy) {
        self.fault = Some(policy);
    }

    /// Switch to external-backend mode: the machine step no longer reflects TX→RX;
    /// a switch drains TX and injects RX. (Two-guest / distributed DST.)
    pub fn set_external_backend(&mut self) {
        self.reflector = false;
    }

    pub fn offset_of(phys: u64) -> Option<u64> {
        (VIRTIO_NET_BASE..VIRTIO_NET_BASE + VIRTIO_NET_LEN)
            .contains(&phys)
            .then(|| phys - VIRTIO_NET_BASE)
    }

    pub fn read(&self, off: u64) -> u32 {
        let q = &self.queues[self.queue_sel.min(1) as usize];
        let v = match off {
            MAGIC => 0x7472_6976, // "virt"
            VERSION => 1,         // legacy
            DEVICE_ID => 1,       // network
            VENDOR_ID => 0x554d_4551,
            HOST_FEATURES => VIRTIO_NET_F_MAC,
            QUEUE_NUM_MAX_REG => QUEUE_NUM_MAX,
            QUEUE_PFN => q.pfn,
            INTERRUPT_STATUS => self.interrupt_status,
            STATUS => self.status,
            // virtio_net_config: mac[0..6] then status(u16). Legacy driver reads
            // config byte-by-byte, so return the byte(s) at the requested offset.
            o if (CONFIG..CONFIG + 6).contains(&o) => self.mac[(o - CONFIG) as usize] as u32,
            _ => 0,
        };
        if std::env::var("SF_NETIOTRACE").is_ok() {
            eprintln!("NET rd off={off:#05x} -> {v:#x}");
        }
        v
    }

    pub fn write(&mut self, off: u64, value: u32, mem: &mut GuestMemory, now: u64) {
        if std::env::var("SF_NETIOTRACE").is_ok() {
            eprintln!("NET wr off={off:#05x} <- {value:#x} (qsel={})", self.queue_sel);
        }
        match off {
            GUEST_PAGE_SIZE => self.guest_page_size = value,
            QUEUE_SEL => self.queue_sel = value,
            QUEUE_NUM => self.queues[self.queue_sel.min(1) as usize].num = value,
            QUEUE_ALIGN => self.queue_align = value,
            QUEUE_PFN => self.queues[self.queue_sel.min(1) as usize].pfn = value,
            QUEUE_NOTIFY => {
                // value = the notified queue index (0=RX, 1=TX).
                if value as usize == TX {
                    self.process_tx(mem, now);
                }
                // RX notify just replenishes buffers; delivery is pulled in poll.
                self.try_deliver_rx(mem, now);
            }
            INTERRUPT_ACK => self.interrupt_status &= !value,
            STATUS => self.status = value,
            _ => {}
        }
    }

    /// Virtqueue layout for queue `qi`: (descriptor base, ring entries, used base).
    fn ring_layout(&self, qi: usize) -> (u64, u64, u64) {
        let q = &self.queues[qi];
        let num = q.num as u64;
        let desc_base = q.pfn as u64 * self.guest_page_size as u64;
        let avail_base = desc_base + num * 16;
        let used_base = align(avail_base + 6 + num * 2, self.queue_align as u64);
        (desc_base, num, used_base)
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

    /// Drain the TX virtqueue: for each request, strip the 10-byte virtio_net_hdr,
    /// collect the ethernet frame into `tx_out` (for the backend), and complete
    /// the descriptor. Deterministic — tied to the QueueNotify, not host timing.
    fn process_tx(&mut self, mem: &mut GuestMemory, now: u64) {
        let q = self.queues[TX].clone();
        if q.num == 0 || q.pfn == 0 {
            return;
        }
        let (desc_base, num, _) = self.ring_layout(TX);
        let avail_base = desc_base + num * 16;
        let avail_idx = mem.read_uint_le(avail_base + 2, 2).unwrap_or(0) as u16;
        while self.queues[TX].last_avail != avail_idx {
            let slot = self.queues[TX].last_avail as u64 % num;
            let head = mem.read_uint_le(avail_base + 4 + slot * 2, 2).unwrap_or(0) as u16;
            // Gather the whole chain into one buffer, then drop the net header.
            let mut buf = Vec::new();
            let mut cur = self.read_desc(mem, desc_base, head);
            let mut steps = 0u64;
            loop {
                for i in 0..cur.len as u64 {
                    buf.push(mem.read_uint_le(cur.addr + i, 1).unwrap_or(0) as u8);
                }
                if cur.flags & VRING_DESC_F_NEXT == 0 || steps > num {
                    break;
                }
                steps += 1;
                cur = self.read_desc(mem, desc_base, cur.next);
            }
            if buf.len() as u64 > NET_HDR_LEN {
                if std::env::var("SF_NETIOTRACE").is_ok() {
                    eprintln!("NET TXdrain frame {} B", buf.len() - NET_HDR_LEN as usize);
                }
                self.tx_out.push(buf[NET_HDR_LEN as usize..].to_vec());
            }
            let complete_at = now.saturating_add(self.latency);
            self.pending.push(Pending { queue: TX, head, written: 0, complete_at });
            self.queues[TX].last_avail = self.queues[TX].last_avail.wrapping_add(1);
        }
    }

    /// Deliver any RX-pending frames whose deadline has passed into guest-posted RX
    /// buffers (prepending a zeroed virtio_net_hdr), scheduling their completions.
    fn try_deliver_rx(&mut self, mem: &mut GuestMemory, now: u64) {
        let q = self.queues[RX].clone();
        if q.num == 0 || q.pfn == 0 || self.rx_queue.is_empty() {
            return;
        }
        let (desc_base, num, _) = self.ring_layout(RX);
        let avail_base = desc_base + num * 16;
        let avail_idx = mem.read_uint_le(avail_base + 2, 2).unwrap_or(0) as u16;
        let mut i = 0;
        while i < self.rx_queue.len() {
            if self.rx_queue[i].deliver_at > now {
                i += 1;
                continue;
            }
            if self.queues[RX].last_avail == avail_idx {
                break; // no free RX buffer; leave the frame queued
            }
            let pend = self.rx_queue.remove(i);
            let slot = self.queues[RX].last_avail as u64 % num;
            let head = mem.read_uint_le(avail_base + 4 + slot * 2, 2).unwrap_or(0) as u16;
            // Write net-hdr (zeroed) + frame into the (device-writable) chain.
            let mut written = 0u32;
            let mut to_write = vec![0u8; NET_HDR_LEN as usize];
            to_write.extend_from_slice(&pend.frame);
            let mut wpos = 0usize;
            let mut cur = self.read_desc(mem, desc_base, head);
            let mut steps = 0u64;
            loop {
                if cur.flags & VRING_DESC_F_WRITE != 0 {
                    for k in 0..cur.len as u64 {
                        if wpos >= to_write.len() {
                            break;
                        }
                        let _ = mem.write_uint_le(cur.addr + k, 1, to_write[wpos] as u64);
                        wpos += 1;
                        written += 1;
                    }
                }
                if cur.flags & VRING_DESC_F_NEXT == 0 || steps > num || wpos >= to_write.len() {
                    break;
                }
                steps += 1;
                cur = self.read_desc(mem, desc_base, cur.next);
            }
            if std::env::var("SF_NETIOTRACE").is_ok() {
                eprintln!("NET RXdeliver head={head} written={written} (frame {} B)", pend.frame.len());
            }
            let complete_at = now.saturating_add(self.latency);
            self.pending.push(Pending { queue: RX, head, written, complete_at });
            self.queues[RX].last_avail = self.queues[RX].last_avail.wrapping_add(1);
        }
    }

    /// Next scheduled event (RX delivery or completion) for idle fast-forward.
    pub fn next_deadline(&self) -> Option<u64> {
        let a = self.pending.iter().map(|p| p.complete_at).min();
        let b = self.rx_queue.iter().map(|r| r.deliver_at).min();
        [a, b].into_iter().flatten().min()
    }

    /// Apply due RX deliveries + completions; raise the interrupt if anything
    /// completed. Called by the machine each step.
    pub fn poll_completions(&mut self, now: u64, mem: &mut GuestMemory) {
        self.try_deliver_rx(mem, now);
        if self.pending.is_empty() {
            return;
        }
        let mut completed = false;
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].complete_at <= now {
                let p = self.pending.remove(i);
                let (_, num, used_base) = self.ring_layout(p.queue);
                if num == 0 {
                    continue;
                }
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

    // ---- backend interface (reflector now, deterministic switch in Phase 1b) ----

    /// Drain the frames the guest has transmitted (for the backend to route).
    pub fn take_tx_frames(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.tx_out)
    }

    /// Inject a frame for the guest to receive, applying the fault policy (drop /
    /// jitter). `now` is the current virtual time; delivery is scheduled after the
    /// link latency + jittered extra. Dropped frames vanish (reproducibly).
    pub fn deliver_frame(&mut self, frame: Vec<u8>, now: u64) {
        let (fault, extra) = self.fault.as_mut().map(|p| p.next()).unwrap_or((NetFault::None, 0));
        if fault == NetFault::Drop {
            return;
        }
        self.rx_queue.push(RxPending { frame, deliver_at: now.saturating_add(extra) });
    }

    pub fn take_interrupt(&mut self) -> Option<u8> {
        if self.irq_pending {
            self.irq_pending = false;
            Some(VIRTIO_NET_VECTOR)
        } else {
            None
        }
    }

    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_bytes(&self.mac);
        for v in [
            self.guest_page_size, self.queue_align, self.queue_sel,
            self.status, self.interrupt_status,
        ] {
            d.mix_u64(v as u64);
        }
        for q in &self.queues {
            d.mix_u64(q.num as u64);
            d.mix_u64(q.pfn as u64);
            d.mix_u64(q.last_avail as u64);
        }
        for f in &self.tx_out {
            d.mix_bytes(f);
        }
        for r in &self.rx_queue {
            d.mix_bytes(&r.frame);
            d.mix_u64(r.deliver_at);
        }
        for p in &self.pending {
            d.mix_u64(p.queue as u64);
            d.mix_u64(p.head as u64);
            d.mix_u64(p.written as u64);
            d.mix_u64(p.complete_at);
        }
        d.mix_u64(self.latency);
        d.mix_u64(self.reflector as u64);
        if let Some(f) = &self.fault {
            f.digest(&mut d);
        }
        d.finish()
    }
}
