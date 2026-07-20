//! Platform/boot setup: identity paging, the e820 memory map, and the Linux
//! boot_params ("zero page") handoff. See `docs/platform-determinism-contract.md`
//! sections 1–2 and `docs/frozen-values.md` section 7.

use crate::memory::{GuestMemory, MemoryError};

// boot_params field offsets (Linux x86 struct boot_params).
const E820_ENTRIES_OFFSET: u64 = 0x1e8; // u8 count
const CMD_LINE_PTR_OFFSET: u64 = 0x228; // u32
const E820_TABLE_OFFSET: u64 = 0x2d0; // boot_e820_entry[]
const E820_ENTRY_SIZE: u64 = 20; // packed { u64 addr; u64 size; u32 type; }

pub const E820_USABLE: u32 = 1;
pub const E820_RESERVED: u32 = 2;

/// One e820 memory-map entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct E820Entry {
    pub addr: u64,
    pub size: u64,
    pub kind: u32,
}

/// Build a fixed e820 map for `mem_bytes` of RAM (Freeze v1 layout).
pub fn default_e820(mem_bytes: u64) -> Vec<E820Entry> {
    vec![
        E820Entry { addr: 0x0, size: 0x9_fc00, kind: E820_USABLE }, // low RAM
        E820Entry { addr: 0x9_fc00, size: 0x400, kind: E820_RESERVED }, // EBDA
        E820Entry { addr: 0xf_0000, size: 0x1_0000, kind: E820_RESERVED }, // legacy
        E820Entry {
            addr: 0x10_0000,
            size: mem_bytes.saturating_sub(0x10_0000),
            kind: E820_USABLE,
        }, // main RAM
    ]
}

/// Install identity page tables mapping the low 1 GiB (one PDPT 1 GiB page).
/// Returns the value for CR3. PML4 @ 0x1000, PDPT @ 0x2000.
pub fn setup_identity_paging(mem: &mut GuestMemory) -> Result<u64, MemoryError> {
    const PRESENT_RW: u64 = 0b11;
    const PS: u64 = 1 << 7;
    mem.write_uint_le(0x1000, 8, 0x2000 | PRESENT_RW)?; // PML4[0] -> PDPT
    mem.write_uint_le(0x2000, 8, PRESENT_RW | PS)?; // PDPT[0] -> 1 GiB page @ PA 0
    Ok(0x1000)
}

fn write_e820(mem: &mut GuestMemory, zero_page: u64, entries: &[E820Entry]) -> Result<(), MemoryError> {
    mem.write_uint_le(zero_page + E820_ENTRIES_OFFSET, 1, entries.len() as u64)?;
    for (i, e) in entries.iter().enumerate() {
        let off = zero_page + E820_TABLE_OFFSET + i as u64 * E820_ENTRY_SIZE;
        mem.write_uint_le(off, 8, e.addr)?;
        mem.write_uint_le(off + 8, 8, e.size)?;
        mem.write_uint_le(off + 16, 4, e.kind as u64)?;
    }
    Ok(())
}

fn write_cmdline(
    mem: &mut GuestMemory,
    zero_page: u64,
    cmd_line_ptr: u64,
    cmdline: &str,
) -> Result<(), MemoryError> {
    mem.write_uint_le(zero_page + CMD_LINE_PTR_OFFSET, 4, cmd_line_ptr)?;
    mem.load(cmd_line_ptr, cmdline.as_bytes())?;
    mem.write_uint_le(cmd_line_ptr + cmdline.len() as u64, 1, 0)?; // NUL
    Ok(())
}

/// Write a Linux boot_params zero page: the e820 map, the command line pointer,
/// and the command line string itself.
pub fn build_boot_params(
    mem: &mut GuestMemory,
    zero_page: u64,
    entries: &[E820Entry],
    cmd_line_ptr: u64,
    cmdline: &str,
) -> Result<(), MemoryError> {
    // Zero the 4 KiB page first (deterministic, no stale bytes).
    for i in 0..4096_u64 {
        mem.write_uint_le(zero_page + i, 1, 0)?;
    }
    write_e820(mem, zero_page, entries)?;
    write_cmdline(mem, zero_page, cmd_line_ptr, cmdline)?;
    Ok(())
}

/// Result of locating the bootable parts of a bzImage.
#[derive(Debug, Clone, Copy)]
pub struct BzImageInfo {
    pub entry: u64,
    pub kernel_load: u64,
    pub zero_page: u64,
}

/// Why a bzImage could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootError {
    TooSmall,
    BadBootFlag,
    BadHeaderMagic,
    InitrdTooBig,
    Memory(MemoryError),
}

impl From<MemoryError> for BootError {
    fn from(e: MemoryError) -> Self {
        BootError::Memory(e)
    }
}

// setup_header field offsets within the bzImage / boot_params.
const HDR_SETUP_SECTS: usize = 0x1f1;
const HDR_BOOT_FLAG: usize = 0x1fe;
const HDR_MAGIC: usize = 0x202; // "HdrS"
const HDR_RAMDISK_IMAGE: u64 = 0x218; // initrd load address (physical, low 32 bits)
const HDR_RAMDISK_SIZE: u64 = 0x21c; // initrd size in bytes (low 32 bits)
const HDR_END: usize = 0x268; // end of the copied setup_header region

/// Parse a Linux x86 bzImage, load its 64-bit protected-mode kernel at 1 MiB,
/// and build the boot_params zero page (setup_header copied in, e820 + cmdline
/// applied on top). Returns the 64-bit entry point (`kernel_load + 0x200`).
///
/// This follows the documented 64-bit boot protocol. It is the loader a real
/// kernel needs; the CPU/device surface required to actually *run* one is filled
/// in empirically (see the `--boot` runner's diagnostics).
pub fn load_bzimage(
    mem: &mut GuestMemory,
    image: &[u8],
    cmdline: &str,
    ram_bytes: u64,
    initrd: Option<&[u8]>,
) -> Result<BzImageInfo, BootError> {
    if image.len() < HDR_END {
        return Err(BootError::TooSmall);
    }
    let boot_flag = u16::from_le_bytes([image[HDR_BOOT_FLAG], image[HDR_BOOT_FLAG + 1]]);
    if boot_flag != 0xaa55 {
        return Err(BootError::BadBootFlag);
    }
    if &image[HDR_MAGIC..HDR_MAGIC + 4] != b"HdrS" {
        return Err(BootError::BadHeaderMagic);
    }

    let setup_sects = match image[HDR_SETUP_SECTS] {
        0 => 4,
        n => n as usize,
    };
    let pm_offset = (setup_sects + 1) * 512;

    let kernel_load = 0x10_0000_u64;
    let zero_page = 0x7000_u64;
    let cmdline_ptr = 0x2_0000_u64;

    if pm_offset < image.len() {
        mem.load(kernel_load, &image[pm_offset..])?;
    }

    // boot_params: zero, copy the setup_header, then apply e820 + cmdline on top
    // (so our cmd_line_ptr wins over the image's zero value).
    for i in 0..4096_u64 {
        mem.write_uint_le(zero_page + i, 1, 0)?;
    }
    for (i, byte) in image.iter().enumerate().take(HDR_END).skip(HDR_SETUP_SECTS) {
        mem.write_uint_le(zero_page + i as u64, 1, *byte as u64)?;
    }
    write_e820(mem, zero_page, &default_e820(ram_bytes))?;
    write_cmdline(mem, zero_page, cmdline_ptr, cmdline)?;
    // The boot protocol requires the bootloader to identify itself; 0xFF =
    // "undefined/unknown loader". Some early-boot paths key off a non-zero value.
    mem.write_uint_le(zero_page + 0x210, 1, 0xff)?; // type_of_loader
    // screen_info (boot_params offset 0): advertise an 80x25 VGA text console so
    // the decompressor's __putstr and the kernel's vgacon emit to 0xb8000 (our
    // VGA text buffer) — the deterministic, always-available boot console.
    mem.write_uint_le(zero_page + 0x06, 1, 0x03)?; // orig_video_mode = 3 (80x25 color)
    mem.write_uint_le(zero_page + 0x07, 1, 80)?; // orig_video_cols
    mem.write_uint_le(zero_page + 0x0e, 1, 25)?; // orig_video_lines
    mem.write_uint_le(zero_page + 0x0f, 1, 0x22)?; // orig_video_isVGA = VIDEO_TYPE_VGAC
    mem.write_uint_le(zero_page + 0x10, 2, 16)?; // orig_video_points

    // initrd / initramfs: place it high in RAM (page-aligned, just under the top),
    // copy it in, and point the setup_header's ramdisk_image/ramdisk_size at it.
    // The kernel reserves this region from boot_params and unpacks it as the
    // initial root filesystem. We keep it below 4 GiB so the 32-bit header fields
    // suffice (ext_ramdisk_image/_size stay zero).
    if let Some(initrd) = initrd {
        let size = initrd.len() as u64;
        // Highest page-aligned base that fits, leaving a one-page guard at the top.
        let base = ram_bytes
            .saturating_sub(size)
            .saturating_sub(0x1000)
            & !0xfff;
        // It must sit above the loaded kernel image region (load + payload) to
        // avoid being clobbered by decompression.
        if size == 0 || base < kernel_load + (image.len() as u64) {
            return Err(BootError::InitrdTooBig);
        }
        mem.load(base, initrd)?;
        mem.write_uint_le(zero_page + HDR_RAMDISK_IMAGE, 4, base & 0xffff_ffff)?;
        mem.write_uint_le(zero_page + HDR_RAMDISK_SIZE, 4, size & 0xffff_ffff)?;
    }

    Ok(BzImageInfo {
        entry: kernel_load + 0x200,
        kernel_load,
        zero_page,
    })
}

// ACPI table physical addresses (Freeze v1, in the reserved legacy region).
pub const RSDP_ADDR: u64 = 0xE_0000;
const RSDT_ADDR: u64 = 0xF_0010;
const FADT_ADDR: u64 = 0xF_0100;
const MADT_ADDR: u64 = 0xF_0200;
const DSDT_ADDR: u64 = 0xF_0300;

fn write_bytes(mem: &mut GuestMemory, addr: u64, bytes: &[u8]) -> Result<(), MemoryError> {
    for (i, b) in bytes.iter().enumerate() {
        mem.write_uint_le(addr + i as u64, 1, *b as u64)?;
    }
    Ok(())
}

/// Build an ACPI table: 36-byte header + body, with a zero-sum checksum.
fn acpi_table(sig: &[u8; 4], revision: u8, body: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    t.extend_from_slice(sig);
    t.extend_from_slice(&((36 + body.len()) as u32).to_le_bytes());
    t.push(revision);
    t.push(0); // checksum placeholder (offset 9)
    t.extend_from_slice(b"LOCKST"); // OEM ID (6)
    t.extend_from_slice(b"LOCKSTEP"); // OEM table ID (8)
    t.extend_from_slice(&1_u32.to_le_bytes()); // OEM revision
    t.extend_from_slice(b"LKST"); // creator ID
    t.extend_from_slice(&1_u32.to_le_bytes()); // creator revision
    t.extend_from_slice(body);
    let sum = t.iter().fold(0_u8, |a, &b| a.wrapping_add(b));
    t[9] = 0_u8.wrapping_sub(sum);
    t
}

/// Write the ACPI tables (RSDP + RSDT + FADT + MADT + DSDT). The RSDP lands at
/// [`RSDP_ADDR`] where a kernel scans for "RSD PTR ". The MADT advertises the
/// Local APIC (0xFEE00000) and the I/O APIC (0xFEC00000). Single-CPU shorthand
/// for [`build_acpi_smp`] — byte-identical to the historical one-LAPIC MADT.
pub fn build_acpi(mem: &mut GuestMemory) -> Result<(), MemoryError> {
    build_acpi_smp(mem, 1)
}

/// Like [`build_acpi`], but emits one MADT Local-APIC entry per vCPU (APIC IDs
/// `0..cpus`), so a guest kernel enumerates and brings up all `cpus` processors.
/// At `cpus == 1` the MADT is byte-for-byte the legacy single-LAPIC table, so a
/// uniprocessor boot's ACPI region — and thus its whole instruction stream — is
/// unchanged (no golden re-baseline for the default path).
pub fn build_acpi_smp(mem: &mut GuestMemory, cpus: u32) -> Result<(), MemoryError> {
    let cpus = cpus.max(1);
    let dsdt = acpi_table(b"DSDT", 2, &[]); // empty AML body
    write_bytes(mem, DSDT_ADDR, &dsdt)?;

    let mut fadt_body = vec![0_u8; 80]; // FADT length 116 (ACPI 1.0)
    // FADT.DSDT pointer at table offset 40 == body offset 4.
    fadt_body[4..8].copy_from_slice(&(DSDT_ADDR as u32).to_le_bytes());
    write_bytes(mem, FADT_ADDR, &acpi_table(b"FACP", 1, &fadt_body))?;

    let mut madt = Vec::new();
    madt.extend_from_slice(&0xFEE0_0000_u32.to_le_bytes()); // Local APIC address
    madt.extend_from_slice(&1_u32.to_le_bytes()); // flags: PCAT_COMPAT
    // One Local-APIC (type 0) entry per vCPU: ACPI processor id = APIC id = index, enabled.
    for id in 0..cpus {
        madt.extend_from_slice(&[0, 8, id as u8, id as u8]); // type 0, len 8, acpi id, apic id
        madt.extend_from_slice(&1_u32.to_le_bytes()); // flags: enabled
    }
    madt.extend_from_slice(&[1, 12, 0, 0]); // I/O APIC entry: type 1, len 12, id 0, rsvd
    madt.extend_from_slice(&0xFEC0_0000_u32.to_le_bytes()); // I/O APIC address
    madt.extend_from_slice(&0_u32.to_le_bytes()); // GSI base 0
    write_bytes(mem, MADT_ADDR, &acpi_table(b"APIC", 1, &madt))?;

    let mut rsdt = Vec::new();
    rsdt.extend_from_slice(&(FADT_ADDR as u32).to_le_bytes());
    rsdt.extend_from_slice(&(MADT_ADDR as u32).to_le_bytes());
    write_bytes(mem, RSDT_ADDR, &acpi_table(b"RSDT", 1, &rsdt))?;

    // RSDP (ACPI 1.0, 20 bytes).
    let mut rsdp = Vec::new();
    rsdp.extend_from_slice(b"RSD PTR ");
    rsdp.push(0); // checksum (offset 8)
    rsdp.extend_from_slice(b"LOCKST"); // OEM ID
    rsdp.push(0); // revision (ACPI 1.0)
    rsdp.extend_from_slice(&(RSDT_ADDR as u32).to_le_bytes());
    let sum = rsdp.iter().fold(0_u8, |a, &b| a.wrapping_add(b));
    rsdp[8] = 0_u8.wrapping_sub(sum);
    write_bytes(mem, RSDP_ADDR, &rsdp)?;
    Ok(())
}

/// Read back the e820 entry count from a zero page (for inspection/tests).
pub fn e820_entry_count(mem: &GuestMemory, zero_page: u64) -> u8 {
    mem.read_uint_le(zero_page + E820_ENTRIES_OFFSET, 1).unwrap_or(0) as u8
}

/// Read back one e820 entry from a zero page.
pub fn e820_entry(mem: &GuestMemory, zero_page: u64, index: u64) -> E820Entry {
    let off = zero_page + E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
    E820Entry {
        addr: mem.read_uint_le(off, 8).unwrap_or(0),
        size: mem.read_uint_le(off + 8, 8).unwrap_or(0),
        kind: mem.read_uint_le(off + 16, 4).unwrap_or(0) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acpi_tables_are_valid() {
        let mut mem = GuestMemory::new(0x10_0000);
        build_acpi(&mut mem).unwrap();
        // RSDP signature + zero checksum.
        let mut sig = [0_u8; 8];
        for (i, s) in sig.iter_mut().enumerate() {
            *s = mem.read_uint_le(RSDP_ADDR + i as u64, 1).unwrap() as u8;
        }
        assert_eq!(&sig, b"RSD PTR ");
        let rsdp_sum =
            (0..20).fold(0_u8, |a, i| a.wrapping_add(mem.read_uint_le(RSDP_ADDR + i, 1).unwrap() as u8));
        assert_eq!(rsdp_sum, 0, "RSDP checksum valid");
        // RSDT points at FADT/MADT; MADT has the right signature.
        let rsdt_ptr = mem.read_uint_le(RSDP_ADDR + 16, 4).unwrap();
        assert_eq!(rsdt_ptr, RSDT_ADDR);
        let madt_sig = mem.read_uint_le(MADT_ADDR, 4).unwrap();
        assert_eq!(madt_sig, u32::from_le_bytes(*b"APIC") as u64);
        // MADT table checksum is zero too.
        let len = mem.read_uint_le(MADT_ADDR + 4, 4).unwrap();
        let madt_sum =
            (0..len).fold(0_u8, |a, i| a.wrapping_add(mem.read_uint_le(MADT_ADDR + i, 1).unwrap() as u8));
        assert_eq!(madt_sum, 0, "MADT checksum valid");
    }

    #[test]
    fn boot_params_round_trip() {
        let mut mem = GuestMemory::new(0x40000);
        let entries = default_e820(0x2000_0000); // 512 MiB
        build_boot_params(&mut mem, 0x7000, &entries, 0x2_0000, "console=ttyS0").unwrap();

        assert_eq!(e820_entry_count(&mem, 0x7000), 4);
        let main = e820_entry(&mem, 0x7000, 3);
        assert_eq!(main.addr, 0x10_0000);
        assert_eq!(main.kind, E820_USABLE);
        assert_eq!(main.size, 0x2000_0000 - 0x10_0000);
        // cmdline pointer and string.
        assert_eq!(mem.read_uint_le(0x7000 + 0x228, 4).unwrap(), 0x2_0000);
        assert_eq!(mem.read_uint_le(0x2_0000, 1).unwrap(), b'c' as u64);
    }
}
