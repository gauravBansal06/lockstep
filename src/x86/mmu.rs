//! 4-level long-mode paging (PML4 → PDPT → PD → PT).
//!
//! Translates a 48-bit canonical virtual address to a guest-physical address,
//! supporting 4 KiB / 2 MiB / 1 GiB pages. Accessed/Dirty bits are written back
//! deterministically as part of the translation that uses the entry (CPU
//! determinism contract section 12); permission violations and non-present
//! entries raise `#PF` with the SDM error-code bits and the faulting address for
//! CR2.
//!
//! Not yet modeled (deterministic `Unsupported`/skipped, noted): reserved-bit
//! `#PF`, PCID, global-page TLB nuance, SMEP/SMAP, 5-level paging.

use crate::memory::GuestMemory;

// Page-table entry flags.
const PRESENT: u64 = 1 << 0;
const RW: u64 = 1 << 1;
const US: u64 = 1 << 2;
const ACCESSED: u64 = 1 << 5;
const DIRTY: u64 = 1 << 6;
const PS: u64 = 1 << 7;
const NX: u64 = 1 << 63;

/// 4 KiB frame address mask within a PTE / CR3 (bits 51:12).
const ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

/// The kind of access being translated (drives permission checks and A/D bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    Execute,
}

/// Whether a 64-bit linear address is canonical for 48-bit (4-level) paging:
/// bits 63:47 must all be copies of bit 47 (the sign extension). A non-canonical
/// address must raise `#GP` (or `#SS` for stack references) *before* any page
/// walk — it never page-faults and never yields a translation. This is a
/// guest-visible determinism boundary: a sloppy walker would translate garbage.
pub fn is_canonical(vaddr: u64) -> bool {
    let sign = (vaddr >> 47) & 1;
    let upper = vaddr >> 47;
    if sign == 0 {
        upper == 0
    } else {
        upper == 0x1_ffff
    }
}

/// A page-fault: the SDM error code and the faulting linear address (→ CR2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageFaultInfo {
    pub error_code: u64,
    pub addr: u64,
}

/// Build a `#PF` error code. P=present(protection vs not-present), W=write,
/// U=user, I=instruction-fetch.
fn err_code(present: bool, write: bool, user: bool, instr: bool) -> u64 {
    (present as u64) | ((write as u64) << 1) | ((user as u64) << 2) | ((instr as u64) << 4)
}

/// Translate `vaddr` under the page tables rooted at `cr3`.
pub fn translate(
    mem: &mut GuestMemory,
    cr3: u64,
    vaddr: u64,
    access: Access,
    cpl: u8,
    write_protect: bool,
    nxe: bool,
) -> Result<u64, PageFaultInfo> {
    let user = cpl == 3;
    let write = access == Access::Write;
    let fetch = access == Access::Execute;

    let indices = [
        (vaddr >> 39) & 0x1ff,
        (vaddr >> 30) & 0x1ff,
        (vaddr >> 21) & 0x1ff,
        (vaddr >> 12) & 0x1ff,
    ];

    let mut table = cr3 & ADDR_MASK;
    let mut writable = true;
    let mut user_ok = true;
    let mut nx_seen = false;

    let mut level = 0;
    let (page_base, offset_mask, leaf_entry_addr) = loop {
        let entry_addr = table + indices[level] * 8;
        let entry = mem
            .read_uint_le(entry_addr, 8)
            .map_err(|_| PageFaultInfo {
                error_code: err_code(false, write, user, fetch),
                addr: vaddr,
            })?;

        if entry & PRESENT == 0 {
            return Err(PageFaultInfo {
                error_code: err_code(false, write, user, fetch),
                addr: vaddr,
            });
        }

        writable &= entry & RW != 0;
        user_ok &= entry & US != 0;
        nx_seen |= entry & NX != 0;

        // Accessed bit: set on every entry used by the translation.
        if entry & ACCESSED == 0 {
            let _ = mem.write_uint_le(entry_addr, 8, entry | ACCESSED);
        }

        // A leaf is the PT entry, or a large-page entry (PS=1) at PDPT/PD.
        let leaf = level == 3 || (level >= 1 && entry & PS != 0);
        if leaf {
            let (base, mask) = match level {
                1 => (entry & 0x000f_ffff_c000_0000, 0x3fff_ffff), // 1 GiB
                2 => (entry & 0x000f_ffff_ffe0_0000, 0x001f_ffff), // 2 MiB
                _ => (entry & ADDR_MASK, 0x0000_0fff),             // 4 KiB
            };
            break (base, mask, entry_addr);
        }

        table = entry & ADDR_MASK;
        level += 1;
    };

    // Permission checks.
    if user && !user_ok {
        return Err(PageFaultInfo {
            error_code: err_code(true, write, user, fetch),
            addr: vaddr,
        });
    }
    if write && !writable && (user || write_protect) {
        return Err(PageFaultInfo {
            error_code: err_code(true, true, user, fetch),
            addr: vaddr,
        });
    }
    if fetch && nxe && nx_seen {
        return Err(PageFaultInfo {
            error_code: err_code(true, write, user, true),
            addr: vaddr,
        });
    }

    // Dirty bit: set on the leaf entry for a write.
    if write {
        if let Ok(entry) = mem.read_uint_le(leaf_entry_addr, 8) {
            if entry & DIRTY == 0 {
                let _ = mem.write_uint_le(leaf_entry_addr, 8, entry | DIRTY);
            }
        }
    }

    Ok(page_base | (vaddr & offset_mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lay out a simple identity-ish page-table tree in physical memory:
    //   PML4 @ 0x1000, PDPT @ 0x2000, PD @ 0x3000, PT @ 0x4000.
    // The single PT maps one 4 KiB virtual page to a chosen physical frame.
    fn build_tables(mem: &mut GuestMemory, vaddr: u64, frame: u64, flags: u64) {
        let pml4 = 0x1000_u64;
        let pdpt = 0x2000_u64;
        let pd = 0x3000_u64;
        let pt = 0x4000_u64;
        let base = PRESENT | RW | US;
        mem.write_uint_le(pml4 + ((vaddr >> 39) & 0x1ff) * 8, 8, pdpt | base)
            .unwrap();
        mem.write_uint_le(pdpt + ((vaddr >> 30) & 0x1ff) * 8, 8, pd | base)
            .unwrap();
        mem.write_uint_le(pd + ((vaddr >> 21) & 0x1ff) * 8, 8, pt | base)
            .unwrap();
        mem.write_uint_le(pt + ((vaddr >> 12) & 0x1ff) * 8, 8, frame | flags)
            .unwrap();
    }

    #[test]
    fn canonical_address_boundaries() {
        // Low half: 0 .. 2^47-1 canonical; first non-canonical just above.
        assert!(is_canonical(0));
        assert!(is_canonical(0x0000_7fff_ffff_ffff));
        assert!(!is_canonical(0x0000_8000_0000_0000));
        // High half: 0xffff_8000.. canonical; the gap below it is not.
        assert!(is_canonical(0xffff_8000_0000_0000));
        assert!(is_canonical(u64::MAX));
        assert!(!is_canonical(0xffff_7fff_ffff_ffff));
        assert!(!is_canonical(0x1234_5678_9abc_def0));
    }

    #[test]
    fn translates_4kib_page_with_offset() {
        let mut mem = GuestMemory::new(0x10000);
        let vaddr = 0x1_2345;
        build_tables(&mut mem, vaddr, 0x8000, PRESENT | RW | US);
        let pa = translate(&mut mem, 0x1000, vaddr, Access::Read, 0, false, false).unwrap();
        assert_eq!(pa, 0x8000 | (vaddr & 0xfff));
    }

    #[test]
    fn not_present_faults_with_p0() {
        let mut mem = GuestMemory::new(0x10000);
        // No tables written: PML4 entry is zero (not present).
        let fault = translate(&mut mem, 0x1000, 0x4000, Access::Read, 0, false, false).unwrap_err();
        assert_eq!(fault.addr, 0x4000);
        assert_eq!(fault.error_code & 1, 0, "P bit clear for not-present");
    }

    #[test]
    fn write_to_readonly_user_page_faults() {
        let mut mem = GuestMemory::new(0x10000);
        let vaddr = 0x9000;
        // Read-only (no RW) but user-accessible leaf.
        build_tables(&mut mem, vaddr, 0xc000, PRESENT | US);
        let fault =
            translate(&mut mem, 0x1000, vaddr, Access::Write, 3, false, false).unwrap_err();
        assert_eq!(fault.error_code & 1, 1, "P set (protection)");
        assert_eq!((fault.error_code >> 1) & 1, 1, "W set");
        assert_eq!((fault.error_code >> 2) & 1, 1, "U set");
    }

    #[test]
    fn accessed_and_dirty_bits_are_set() {
        let mut mem = GuestMemory::new(0x10000);
        let vaddr = 0x1000;
        build_tables(&mut mem, vaddr, 0x8000, PRESENT | RW | US);
        translate(&mut mem, 0x1000, vaddr, Access::Write, 0, false, false).unwrap();
        let pt_entry = mem
            .read_uint_le(0x4000 + ((vaddr >> 12) & 0x1ff) * 8, 8)
            .unwrap();
        assert_ne!(pt_entry & ACCESSED, 0, "A set");
        assert_ne!(pt_entry & DIRTY, 0, "D set on write");
        // A read elsewhere should not set D.
        let other = 0x2000;
        build_tables(&mut mem, other, 0xa000, PRESENT | RW | US);
        translate(&mut mem, 0x1000, other, Access::Read, 0, false, false).unwrap();
        let entry = mem
            .read_uint_le(0x4000 + ((other >> 12) & 0x1ff) * 8, 8)
            .unwrap();
        assert_ne!(entry & ACCESSED, 0, "A set on read");
        assert_eq!(entry & DIRTY, 0, "D not set on read");
    }

    #[test]
    fn two_mib_large_page() {
        let mut mem = GuestMemory::new(0x10000);
        let vaddr = 0x20_5678_u64;
        let base = PRESENT | RW | US;
        mem.write_uint_le(0x1000 + ((vaddr >> 39) & 0x1ff) * 8, 8, 0x2000 | base)
            .unwrap();
        mem.write_uint_le(0x2000 + ((vaddr >> 30) & 0x1ff) * 8, 8, 0x3000 | base)
            .unwrap();
        // PD entry with PS=1 → 2 MiB page at physical 0x20_0000.
        mem.write_uint_le(
            0x3000 + ((vaddr >> 21) & 0x1ff) * 8,
            8,
            0x20_0000 | base | PS,
        )
        .unwrap();
        let pa = translate(&mut mem, 0x1000, vaddr, Access::Read, 0, false, false).unwrap();
        assert_eq!(pa, 0x20_0000 | (vaddr & 0x1f_ffff));
    }

    #[test]
    fn nx_blocks_instruction_fetch() {
        let mut mem = GuestMemory::new(0x10000);
        let vaddr = 0x7000;
        build_tables(&mut mem, vaddr, 0xd000, PRESENT | RW | US | NX);
        // Data read is fine.
        assert!(translate(&mut mem, 0x1000, vaddr, Access::Read, 0, false, true).is_ok());
        // Fetch faults with the I bit when NXE is on.
        let fault =
            translate(&mut mem, 0x1000, vaddr, Access::Execute, 0, false, true).unwrap_err();
        assert_eq!((fault.error_code >> 4) & 1, 1, "I bit set");
    }
}
