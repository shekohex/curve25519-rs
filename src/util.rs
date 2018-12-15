extern "C" {
    pub fn fixed_time_eq_asm(lhsp: *mut u8, rhsp: *mut u8, count: usize)
        -> u32;
}

/// Compare two vectors using a fixed number of operations. If the two vectors
/// are not of equal length, the function returns false immediately.
pub fn fixed_time_eq(lhs: &[u8], rhs: &[u8]) -> bool {
    if lhs.len() != rhs.len() {
        false
    } else {
        let count = lhs.len();

        unsafe {
            let lhsp = lhs.get_unchecked(0);
            let rhsp = rhs.get_unchecked(0);
            fixed_time_eq_asm(*lhsp as *mut u8, *rhsp as *mut u8, count) == 0
        }
    }
}
