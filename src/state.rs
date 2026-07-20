use crate::seed::Seed256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Pure,
    Capture,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineSpec {
    pub arch: Arch,
    pub vcpus: u8,
    pub memory_bytes: u64,
    pub ticks_per_instruction: u64,
}

impl MachineSpec {
    pub fn x86_64_single_vcpu() -> Self {
        Self {
            arch: Arch::X86_64,
            vcpus: 1,
            memory_bytes: 128 * 1024 * 1024,
            ticks_per_instruction: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

/// Hashes of the guest-visible configuration that, together with `mode`/`seed`/
/// `machine`, fully identify a run (GOLDEN_TARGET §17). Two runs that differ in
/// any of these can legitimately produce different transcripts, so the identity
/// must distinguish them — comparing full-state digests across runs is only
/// meaningful when the identities match.
///
/// Fields with no computed source yet hold the placeholder `0` (e.g. no initrd,
/// no disk image). As loaders learn to hash their inputs, the placeholders are
/// replaced with real values; `0` explicitly means "absent / not yet computed".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConfigIdentity {
    /// Hash of the frozen virtual CPU model (CPUID/MSR surface). Placeholder 0.
    pub cpu_model_hash: u64,
    /// Hash of the machine spec (memory size, vcpus, timing). Placeholder 0.
    pub machine_spec_hash: u64,
    /// Hash identifying the emulator build itself.
    pub hypervisor_build_hash: u64,
    /// Hash of the guest kernel image bytes (0 when no kernel loaded).
    pub kernel_hash: u64,
    /// Hash of the initrd image bytes (0 when absent).
    pub initrd_hash: u64,
    /// Hash of the kernel command line (0 when empty).
    pub cmdline_hash: u64,
    /// Hash of the backing disk image (0 when absent).
    pub disk_image_hash: u64,
    /// Virtual clock epoch the run started from. Placeholder 0.
    pub virtual_clock_epoch: u64,
    /// Hash of the device topology. Placeholder 0.
    pub topology_hash: u64,
}

/// FNV-1a over bytes; the project's stable, host-independent digest primitive
/// (same construction as [`StateDigest`]). Used to fold identity inputs.
pub fn config_hash(bytes: &[u8]) -> u64 {
    let mut d = StateDigest::new();
    d.mix_bytes(bytes);
    d.finish()
}

/// A compile-time-stable hash standing in for the emulator build identity.
pub fn hypervisor_build_hash() -> u64 {
    config_hash(concat!("lockstep ", env!("CARGO_PKG_VERSION")).as_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunIdentity {
    mode: RunMode,
    seed: Seed256,
    machine: MachineSpec,
    config: ConfigIdentity,
}

impl RunIdentity {
    pub fn new(mode: RunMode, seed: Seed256, machine: MachineSpec) -> Self {
        assert_eq!(machine.vcpus, 1, "v0 deterministic core requires one vCPU");
        let config = ConfigIdentity {
            hypervisor_build_hash: hypervisor_build_hash(),
            ..ConfigIdentity::default()
        };
        Self {
            mode,
            seed,
            machine,
            config,
        }
    }

    /// Attach (or replace) the guest-visible configuration identity.
    pub fn with_config(mut self, config: ConfigIdentity) -> Self {
        self.config = config;
        self
    }

    pub fn mode(&self) -> RunMode {
        self.mode
    }

    pub fn seed(&self) -> &Seed256 {
        &self.seed
    }

    pub fn machine(&self) -> &MachineSpec {
        &self.machine
    }

    pub fn config(&self) -> &ConfigIdentity {
        &self.config
    }

    /// A single hash over the whole identity — the key under which cross-run
    /// full-state digests may be compared.
    pub fn digest(&self) -> u64 {
        let mut d = StateDigest::new();
        d.mix_u64(self.mode as u64);
        d.mix_bytes(self.seed.bytes());
        d.mix_u64(self.machine.memory_bytes);
        d.mix_u64(self.machine.vcpus as u64);
        d.mix_u64(self.machine.ticks_per_instruction);
        d.mix_u64(self.machine.arch as u64);
        let c = &self.config;
        for h in [
            c.cpu_model_hash,
            c.machine_spec_hash,
            c.hypervisor_build_hash,
            c.kernel_hash,
            c.initrd_hash,
            c.cmdline_hash,
            c.disk_image_hash,
            c.virtual_clock_epoch,
            c.topology_hash,
        ] {
            d.mix_u64(h);
        }
        d.finish()
    }
}

#[derive(Debug, Clone)]
pub struct StateDigest {
    hash: u64,
}

impl StateDigest {
    pub fn new() -> Self {
        Self {
            hash: 0xcbf2_9ce4_8422_2325,
        }
    }

    pub fn mix_u64(&mut self, value: u64) {
        self.mix_bytes(&value.to_le_bytes());
    }

    pub fn mix_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.hash ^= *byte as u64;
            self.hash = self.hash.wrapping_mul(0x0000_0100_0000_01b3);
            self.hash ^= self.hash >> 32;
        }
    }

    pub fn finish(&self) -> u64 {
        self.hash
    }
}

impl Default for StateDigest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> Seed256 {
        Seed256::from_hex("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
            .unwrap()
    }

    #[test]
    fn identity_distinguishes_config() {
        let base = RunIdentity::new(RunMode::Pure, seed(), MachineSpec::x86_64_single_vcpu());
        let kernel_a = base.clone().with_config(ConfigIdentity {
            kernel_hash: config_hash(b"vmlinuz-a"),
            ..*base.config()
        });
        let kernel_b = base.clone().with_config(ConfigIdentity {
            kernel_hash: config_hash(b"vmlinuz-b"),
            ..*base.config()
        });
        // Same mode/seed/machine but a different kernel -> different identity.
        assert_ne!(kernel_a.digest(), kernel_b.digest());
        // Identity is stable for the same inputs.
        assert_eq!(kernel_a.digest(), kernel_a.clone().digest());
        // The build hash is populated (non-placeholder).
        assert_ne!(base.config().hypervisor_build_hash, 0);
    }
}
