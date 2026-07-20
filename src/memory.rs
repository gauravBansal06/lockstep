use crate::state::StateDigest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestMemory {
    bytes: Vec<u8>,
}

impl GuestMemory {
    pub fn new(size: usize) -> Self {
        Self {
            bytes: vec![0; size],
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn load(&mut self, address: u64, data: &[u8]) -> Result<(), MemoryError> {
        let range = self.range(address, data.len())?;
        self.bytes[range].copy_from_slice(data);
        Ok(())
    }

    pub fn read_u8(&self, address: u64) -> Result<u8, MemoryError> {
        let index = self.index(address)?;
        Ok(self.bytes[index])
    }

    pub fn read_u16_le(&self, address: u64) -> Result<u16, MemoryError> {
        let bytes = self.read_array::<2>(address)?;
        Ok(u16::from_le_bytes(bytes))
    }

    pub fn read_u64_le(&self, address: u64) -> Result<u64, MemoryError> {
        let bytes = self.read_array::<8>(address)?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn write_u64_le(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        let range = self.range(address, 8)?;
        self.bytes[range].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Read `bytes` (1..=8) as a little-endian unsigned integer zero-extended to
    /// u64. The x86 interpreter uses this for all operand sizes.
    pub fn read_uint_le(&self, address: u64, bytes: usize) -> Result<u64, MemoryError> {
        debug_assert!((1..=8).contains(&bytes));
        let range = self.range(address, bytes)?;
        let mut value = 0_u64;
        for (index, byte) in self.bytes[range].iter().enumerate() {
            value |= (*byte as u64) << (index * 8);
        }
        Ok(value)
    }

    /// Write the low `bytes` (1..=8) of `value` little-endian.
    pub fn write_uint_le(
        &mut self,
        address: u64,
        bytes: usize,
        value: u64,
    ) -> Result<(), MemoryError> {
        debug_assert!((1..=8).contains(&bytes));
        let range = self.range(address, bytes)?;
        for (index, slot) in self.bytes[range].iter_mut().enumerate() {
            *slot = (value >> (index * 8)) as u8;
        }
        Ok(())
    }

    pub fn digest(&self) -> u64 {
        let mut digest = StateDigest::new();
        digest.mix_u64(self.bytes.len() as u64);
        digest.mix_bytes(&self.bytes);
        digest.finish()
    }

    fn read_array<const N: usize>(&self, address: u64) -> Result<[u8; N], MemoryError> {
        let range = self.range(address, N)?;
        let mut output = [0_u8; N];
        output.copy_from_slice(&self.bytes[range]);
        Ok(output)
    }

    fn index(&self, address: u64) -> Result<usize, MemoryError> {
        let index = usize::try_from(address).map_err(|_| MemoryError::AddressTooLarge(address))?;
        if index >= self.bytes.len() {
            return Err(MemoryError::OutOfBounds {
                address,
                len: 1,
                memory_len: self.bytes.len(),
            });
        }
        Ok(index)
    }

    fn range(&self, address: u64, len: usize) -> Result<std::ops::Range<usize>, MemoryError> {
        let start = usize::try_from(address).map_err(|_| MemoryError::AddressTooLarge(address))?;
        let end = start.checked_add(len).ok_or(MemoryError::OutOfBounds {
            address,
            len,
            memory_len: self.bytes.len(),
        })?;
        if end > self.bytes.len() {
            return Err(MemoryError::OutOfBounds {
                address,
                len,
                memory_len: self.bytes.len(),
            });
        }
        Ok(start..end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    AddressTooLarge(u64),
    OutOfBounds {
        address: u64,
        len: usize,
        memory_len: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_is_zero_initialized_and_bounds_checked() {
        let mut memory = GuestMemory::new(16);
        assert_eq!(memory.read_u64_le(0).unwrap(), 0);
        memory.write_u64_le(8, 0xfeed).unwrap();
        assert_eq!(memory.read_u64_le(8).unwrap(), 0xfeed);
        assert!(memory.write_u64_le(12, 1).is_err());
    }
}
