use crate::state::StateDigest;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetEvent {
    pub at_instruction: u64,
    pub kind: EventKind,
    sequence: u64,
}

impl DetEvent {
    pub fn new(at_instruction: u64, kind: EventKind) -> Self {
        Self {
            at_instruction,
            kind,
            sequence: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventKind {
    TimerInterrupt { vector: u8 },
    DiskCompletion { request_id: u64, status: u8 },
    NetworkPacket { link_id: u64, packet_id: u64 },
    GuestEntropy { stream_id: u64, bytes: Vec<u8> },
}

#[derive(Debug, Clone, Default)]
pub struct EventQueue {
    heap: BinaryHeap<QueueEntry>,
    next_sequence: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, mut event: DetEvent) {
        event.sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("event sequence overflowed");
        self.heap.push(QueueEntry(event));
    }

    pub fn pop_ready(&mut self, instruction_count: u64) -> Option<DetEvent> {
        let entry = self.heap.peek()?;
        if entry.0.at_instruction > instruction_count {
            return None;
        }
        self.heap.pop().map(|entry| entry.0)
    }

    pub fn next_instruction(&self) -> Option<u64> {
        self.heap.peek().map(|entry| entry.0.at_instruction)
    }

    /// Order-independent digest of the pending queue for snapshots and divergence
    /// checks. `BinaryHeap` iteration order is unspecified, so events are sorted
    /// by their deterministic `(at_instruction, sequence)` key before hashing.
    pub fn digest(&self) -> u64 {
        let mut entries: Vec<&DetEvent> = self.heap.iter().map(|entry| &entry.0).collect();
        entries.sort_by(|a, b| {
            a.at_instruction
                .cmp(&b.at_instruction)
                .then_with(|| a.sequence.cmp(&b.sequence))
        });

        let mut digest = StateDigest::new();
        digest.mix_u64(self.next_sequence);
        digest.mix_u64(entries.len() as u64);
        for event in entries {
            digest.mix_u64(event.at_instruction);
            digest.mix_u64(event.sequence);
            mix_event_kind(&mut digest, &event.kind);
        }
        digest.finish()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

fn mix_event_kind(digest: &mut StateDigest, kind: &EventKind) {
    match kind {
        EventKind::TimerInterrupt { vector } => {
            digest.mix_bytes(b"timer");
            digest.mix_u64(*vector as u64);
        }
        EventKind::DiskCompletion { request_id, status } => {
            digest.mix_bytes(b"disk");
            digest.mix_u64(*request_id);
            digest.mix_u64(*status as u64);
        }
        EventKind::NetworkPacket { link_id, packet_id } => {
            digest.mix_bytes(b"net");
            digest.mix_u64(*link_id);
            digest.mix_u64(*packet_id);
        }
        EventKind::GuestEntropy { stream_id, bytes } => {
            digest.mix_bytes(b"entropy");
            digest.mix_u64(*stream_id);
            digest.mix_bytes(bytes);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueueEntry(DetEvent);

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .at_instruction
            .cmp(&self.0.at_instruction)
            .then_with(|| other.0.sequence.cmp(&self.0.sequence))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_pop_by_instruction_then_insertion_order() {
        let mut queue = EventQueue::new();
        queue.push(DetEvent::new(
            20,
            EventKind::DiskCompletion {
                request_id: 1,
                status: 0,
            },
        ));
        queue.push(DetEvent::new(10, EventKind::TimerInterrupt { vector: 32 }));
        queue.push(DetEvent::new(
            10,
            EventKind::NetworkPacket {
                link_id: 7,
                packet_id: 1,
            },
        ));

        assert!(queue.pop_ready(9).is_none());
        assert!(matches!(
            queue.pop_ready(10).unwrap().kind,
            EventKind::TimerInterrupt { .. }
        ));
        assert!(matches!(
            queue.pop_ready(10).unwrap().kind,
            EventKind::NetworkPacket { .. }
        ));
        assert!(queue.pop_ready(10).is_none());
        assert!(matches!(
            queue.pop_ready(20).unwrap().kind,
            EventKind::DiskCompletion { .. }
        ));
    }

    #[test]
    fn exposes_next_event_instruction() {
        let mut queue = EventQueue::new();
        assert_eq!(queue.next_instruction(), None);
        queue.push(DetEvent::new(30, EventKind::TimerInterrupt { vector: 32 }));
        queue.push(DetEvent::new(10, EventKind::TimerInterrupt { vector: 33 }));
        assert_eq!(queue.next_instruction(), Some(10));
    }
}
