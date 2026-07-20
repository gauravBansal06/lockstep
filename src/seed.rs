use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seed256([u8; 32]);

impl Seed256 {
    pub fn from_hex(input: &str) -> Result<Self, SeedParseError> {
        let input = input.strip_prefix("0x").unwrap_or(input);
        if input.len() != 64 {
            return Err(SeedParseError::WrongLength(input.len()));
        }

        let mut bytes = [0_u8; 32];
        for (index, chunk) in input.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex_nibble(chunk[0])?;
            let low = decode_hex_nibble(chunk[1])?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn derive(&self, label: &str) -> StreamRng {
        let mut state = fnv1a64(b"lockstep-seed-v1");
        state = fnv1a64_continue(state, &self.0);
        state = fnv1a64_continue(state, label.as_bytes());
        StreamRng::new(state)
    }
}

impl fmt::Debug for Seed256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Seed256(")?;
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedParseError {
    WrongLength(usize),
    InvalidHex(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamRng {
    state: u64,
}

impl StreamRng {
    pub fn new(state: u64) -> Self {
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix64(self.state)
    }

    /// Current stream position. Snapshots must capture this so a restored run
    /// continues drawing identical values (GOLDEN_TARGET rule 14).
    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn next_bounded(&mut self, upper_exclusive: u64) -> u64 {
        assert!(upper_exclusive > 0);
        self.next_u64() % upper_exclusive
    }

    pub fn fill_bytes(&mut self, output: &mut [u8]) {
        for chunk in output.chunks_mut(8) {
            let value = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&value[..chunk.len()]);
        }
    }
}

fn decode_hex_nibble(byte: u8) -> Result<u8, SeedParseError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(SeedParseError::InvalidHex(byte)),
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_continue(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv1a64_continue(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_256_bit_hex_seed() {
        let seed =
            Seed256::from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
                .unwrap();
        assert_eq!(seed.bytes()[0], 0);
        assert_eq!(seed.bytes()[31], 31);
    }

    #[test]
    fn derived_streams_are_stable_and_independent() {
        let seed =
            Seed256::from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
                .unwrap();
        let mut a = seed.derive("network");
        let mut b = seed.derive("network");
        let mut c = seed.derive("disk");

        assert_eq!(a.next_u64(), b.next_u64());
        assert_ne!(a.next_u64(), c.next_u64());
    }
}
