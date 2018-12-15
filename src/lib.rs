#![warn(clippy::all)]
#![allow(
    clippy::suspicious_arithmetic_impl,
    clippy::many_single_char_names,
    clippy::unknown_clippy_lints
)]
#![no_std]

mod statics;
mod util;
use crate::{
    statics::{BI, FE_D, FE_D2, FE_ONE, FE_SQRTM1, FE_ZERO, GE_PRECOMP_BASE},
    util::fixed_time_eq,
};
use core::{
    cmp::{min, Eq, PartialEq},
    ops::{Add, Mul, Sub},
};
use rand::{rngs::OsRng, Error as RndError, Rng};

/// Here the field is \Z/(2^255-19).
///
/// An element t, entries t\[0\]...t\[9\], represents the integer
/// `t[0]+2^26 t[1]+2^51 t[2]+2^77 t[3]+2^102 t[4]+...+2^230 t[9]`.
/// Bounds on each t\[i\] vary depending on context.
#[derive(Clone, Copy)]
pub struct FieldElement(pub [i32; 10]);

impl PartialEq for FieldElement {
    fn eq(&self, other: &FieldElement) -> bool {
        let &FieldElement(self_elems) = self;
        let &FieldElement(other_elems) = other;
        self_elems.to_vec() == other_elems.to_vec()
    }
}

impl Eq for FieldElement {}

#[inline]
fn load_4u(s: &[u8]) -> u64 {
    u64::from(s[0])
        | (u64::from(s[1]) << 8)
        | (u64::from(s[2]) << 16)
        | (u64::from(s[3]) << 24)
}

#[inline]
fn load_4i(s: &[u8]) -> i64 { load_4u(s) as i64 }

#[inline]
fn load_3u(s: &[u8]) -> u64 {
    u64::from(s[0]) | (u64::from(s[1]) << 8) | (u64::from(s[2]) << 16)
}

#[inline]
fn load_3i(s: &[u8]) -> i64 { load_3u(s) as i64 }

impl Add for FieldElement {
    type Output = FieldElement;

    // `h = f + g`
    // Can overlap `h` with `f` or `g`.
    //
    // Preconditions:
    //    |f| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    //    |g| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    //
    // Postconditions:
    //    |h| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
    fn add(self, rhs: FieldElement) -> FieldElement {
        let FieldElement(f) = self;
        let FieldElement(g) = rhs;
        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;
        let [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9] = g;
        let h0 = f0 + g0;
        let h1 = f1 + g1;
        let h2 = f2 + g2;
        let h3 = f3 + g3;
        let h4 = f4 + g4;
        let h5 = f5 + g5;
        let h6 = f6 + g6;
        let h7 = f7 + g7;
        let h8 = f8 + g8;
        let h9 = f9 + g9;
        FieldElement([h0, h1, h2, h3, h4, h5, h6, h7, h8, h9])
    }
}

impl Sub for FieldElement {
    type Output = FieldElement;

    // `h = f - g`
    // Can overlap `h` with `f` or `g`.
    //
    // Preconditions:
    //    |f| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    //    |g| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    //
    // Postconditions:
    //    |h| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
    fn sub(self, rhs: FieldElement) -> FieldElement {
        let FieldElement(f) = self;
        let FieldElement(g) = rhs;

        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;
        let [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9] = g;
        let h0 = f0 - g0;
        let h1 = f1 - g1;
        let h2 = f2 - g2;
        let h3 = f3 - g3;
        let h4 = f4 - g4;
        let h5 = f5 - g5;
        let h6 = f6 - g6;
        let h7 = f7 - g7;
        let h8 = f8 - g8;
        let h9 = f9 - g9;
        FieldElement([h0, h1, h2, h3, h4, h5, h6, h7, h8, h9])
    }
}

impl Mul for FieldElement {
    type Output = FieldElement;

    // `h = f * g`
    // Can overlap h with `f` or `g`.
    //
    // Preconditions:
    //    |f| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
    //    |g| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
    //
    // Postconditions:
    //    |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    //
    // Notes on implementation strategy:
    //
    // Using schoolbook multiplication.
    // Karatsuba would save a little in some cost models.
    //
    // Most multiplications by 2 and 19 are 32-bit precomputations;
    // cheaper than 64-bit postcomputations.
    //
    // There is one remaining multiplication by 19 in the carry chain;
    // one *19 precomputation can be merged into this,
    // but the resulting data flow is considerably less clean.
    //
    // There are 12 carries below.
    // 10 of them are 2-way parallelizable and vectorizable.
    // Can get away with 11 carries, but then data flow is much deeper.
    //
    // With tighter constraints on inputs can squeeze carries into int32.
    fn mul(self, rhs: FieldElement) -> FieldElement {
        let FieldElement(f) = self;
        let FieldElement(g) = rhs;
        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;
        let [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9] = g;

        let g1_19 = 19 * g1; // 1.4*2^29
        let g2_19 = 19 * g2; // 1.4*2^30; still ok
        let g3_19 = 19 * g3;
        let g4_19 = 19 * g4;
        let g5_19 = 19 * g5;
        let g6_19 = 19 * g6;
        let g7_19 = 19 * g7;
        let g8_19 = 19 * g8;
        let g9_19 = 19 * g9;
        let f1_2 = 2 * f1;
        let f3_2 = 2 * f3;
        let f5_2 = 2 * f5;
        let f7_2 = 2 * f7;
        let f9_2 = 2 * f9;
        let f0g0 = i64::from(f0) * i64::from(g0);
        let f0g1 = i64::from(f0) * i64::from(g1);
        let f0g2 = i64::from(f0) * i64::from(g2);
        let f0g3 = i64::from(f0) * i64::from(g3);
        let f0g4 = i64::from(f0) * i64::from(g4);
        let f0g5 = i64::from(f0) * i64::from(g5);
        let f0g6 = i64::from(f0) * i64::from(g6);
        let f0g7 = i64::from(f0) * i64::from(g7);
        let f0g8 = i64::from(f0) * i64::from(g8);
        let f0g9 = i64::from(f0) * i64::from(g9);
        let f1g0 = i64::from(f1) * i64::from(g0);
        let f1g1_2 = i64::from(f1_2) * i64::from(g1);
        let f1g2 = i64::from(f1) * i64::from(g2);
        let f1g3_2 = i64::from(f1_2) * i64::from(g3);
        let f1g4 = i64::from(f1) * i64::from(g4);
        let f1g5_2 = i64::from(f1_2) * i64::from(g5);
        let f1g6 = i64::from(f1) * i64::from(g6);
        let f1g7_2 = i64::from(f1_2) * i64::from(g7);
        let f1g8 = i64::from(f1) * i64::from(g8);
        let f1g9_38 = i64::from(f1_2) * i64::from(g9_19);
        let f2g0 = i64::from(f2) * i64::from(g0);
        let f2g1 = i64::from(f2) * i64::from(g1);
        let f2g2 = i64::from(f2) * i64::from(g2);
        let f2g3 = i64::from(f2) * i64::from(g3);
        let f2g4 = i64::from(f2) * i64::from(g4);
        let f2g5 = i64::from(f2) * i64::from(g5);
        let f2g6 = i64::from(f2) * i64::from(g6);
        let f2g7 = i64::from(f2) * i64::from(g7);
        let f2g8_19 = i64::from(f2) * i64::from(g8_19);
        let f2g9_19 = i64::from(f2) * i64::from(g9_19);
        let f3g0 = i64::from(f3) * i64::from(g0);
        let f3g1_2 = i64::from(f3_2) * i64::from(g1);
        let f3g2 = i64::from(f3) * i64::from(g2);
        let f3g3_2 = i64::from(f3_2) * i64::from(g3);
        let f3g4 = i64::from(f3) * i64::from(g4);
        let f3g5_2 = i64::from(f3_2) * i64::from(g5);
        let f3g6 = i64::from(f3) * i64::from(g6);
        let f3g7_38 = i64::from(f3_2) * i64::from(g7_19);
        let f3g8_19 = i64::from(f3) * i64::from(g8_19);
        let f3g9_38 = i64::from(f3_2) * i64::from(g9_19);
        let f4g0 = i64::from(f4) * i64::from(g0);
        let f4g1 = i64::from(f4) * i64::from(g1);
        let f4g2 = i64::from(f4) * i64::from(g2);
        let f4g3 = i64::from(f4) * i64::from(g3);
        let f4g4 = i64::from(f4) * i64::from(g4);
        let f4g5 = i64::from(f4) * i64::from(g5);
        let f4g6_19 = i64::from(f4) * i64::from(g6_19);
        let f4g7_19 = i64::from(f4) * i64::from(g7_19);
        let f4g8_19 = i64::from(f4) * i64::from(g8_19);
        let f4g9_19 = i64::from(f4) * i64::from(g9_19);
        let f5g0 = i64::from(f5) * i64::from(g0);
        let f5g1_2 = i64::from(f5_2) * i64::from(g1);
        let f5g2 = i64::from(f5) * i64::from(g2);
        let f5g3_2 = i64::from(f5_2) * i64::from(g3);
        let f5g4 = i64::from(f5) * i64::from(g4);
        let f5g5_38 = i64::from(f5_2) * i64::from(g5_19);
        let f5g6_19 = i64::from(f5) * i64::from(g6_19);
        let f5g7_38 = i64::from(f5_2) * i64::from(g7_19);
        let f5g8_19 = i64::from(f5) * i64::from(g8_19);
        let f5g9_38 = i64::from(f5_2) * i64::from(g9_19);
        let f6g0 = i64::from(f6) * i64::from(g0);
        let f6g1 = i64::from(f6) * i64::from(g1);
        let f6g2 = i64::from(f6) * i64::from(g2);
        let f6g3 = i64::from(f6) * i64::from(g3);
        let f6g4_19 = i64::from(f6) * i64::from(g4_19);
        let f6g5_19 = i64::from(f6) * i64::from(g5_19);
        let f6g6_19 = i64::from(f6) * i64::from(g6_19);
        let f6g7_19 = i64::from(f6) * i64::from(g7_19);
        let f6g8_19 = i64::from(f6) * i64::from(g8_19);
        let f6g9_19 = i64::from(f6) * i64::from(g9_19);
        let f7g0 = i64::from(f7) * i64::from(g0);
        let f7g1_2 = i64::from(f7_2) * i64::from(g1);
        let f7g2 = i64::from(f7) * i64::from(g2);
        let f7g3_38 = i64::from(f7_2) * i64::from(g3_19);
        let f7g4_19 = i64::from(f7) * i64::from(g4_19);
        let f7g5_38 = i64::from(f7_2) * i64::from(g5_19);
        let f7g6_19 = i64::from(f7) * i64::from(g6_19);
        let f7g7_38 = i64::from(f7_2) * i64::from(g7_19);
        let f7g8_19 = i64::from(f7) * i64::from(g8_19);
        let f7g9_38 = i64::from(f7_2) * i64::from(g9_19);
        let f8g0 = i64::from(f8) * i64::from(g0);
        let f8g1 = i64::from(f8) * i64::from(g1);
        let f8g2_19 = i64::from(f8) * i64::from(g2_19);
        let f8g3_19 = i64::from(f8) * i64::from(g3_19);
        let f8g4_19 = i64::from(f8) * i64::from(g4_19);
        let f8g5_19 = i64::from(f8) * i64::from(g5_19);
        let f8g6_19 = i64::from(f8) * i64::from(g6_19);
        let f8g7_19 = i64::from(f8) * i64::from(g7_19);
        let f8g8_19 = i64::from(f8) * i64::from(g8_19);
        let f8g9_19 = i64::from(f8) * i64::from(g9_19);
        let f9g0 = i64::from(f9) * i64::from(g0);
        let f9g1_38 = i64::from(f9_2) * i64::from(g1_19);
        let f9g2_19 = i64::from(f9) * i64::from(g2_19);
        let f9g3_38 = i64::from(f9_2) * i64::from(g3_19);
        let f9g4_19 = i64::from(f9) * i64::from(g4_19);
        let f9g5_38 = i64::from(f9_2) * i64::from(g5_19);
        let f9g6_19 = i64::from(f9) * i64::from(g6_19);
        let f9g7_38 = i64::from(f9_2) * i64::from(g7_19);
        let f9g8_19 = i64::from(f9) * i64::from(g8_19);
        let f9g9_38 = i64::from(f9_2) * i64::from(g9_19);
        let mut h0 = f0g0
            + f1g9_38
            + f2g8_19
            + f3g7_38
            + f4g6_19
            + f5g5_38
            + f6g4_19
            + f7g3_38
            + f8g2_19
            + f9g1_38;
        let mut h1 = f0g1
            + f1g0
            + f2g9_19
            + f3g8_19
            + f4g7_19
            + f5g6_19
            + f6g5_19
            + f7g4_19
            + f8g3_19
            + f9g2_19;
        let mut h2 = f0g2
            + f1g1_2
            + f2g0
            + f3g9_38
            + f4g8_19
            + f5g7_38
            + f6g6_19
            + f7g5_38
            + f8g4_19
            + f9g3_38;
        let mut h3 = f0g3
            + f1g2
            + f2g1
            + f3g0
            + f4g9_19
            + f5g8_19
            + f6g7_19
            + f7g6_19
            + f8g5_19
            + f9g4_19;
        let mut h4 = f0g4
            + f1g3_2
            + f2g2
            + f3g1_2
            + f4g0
            + f5g9_38
            + f6g8_19
            + f7g7_38
            + f8g6_19
            + f9g5_38;
        let mut h5 = f0g5
            + f1g4
            + f2g3
            + f3g2
            + f4g1
            + f5g0
            + f6g9_19
            + f7g8_19
            + f8g7_19
            + f9g6_19;
        let mut h6 = f0g6
            + f1g5_2
            + f2g4
            + f3g3_2
            + f4g2
            + f5g1_2
            + f6g0
            + f7g9_38
            + f8g8_19
            + f9g7_38;
        let mut h7 = f0g7
            + f1g6
            + f2g5
            + f3g4
            + f4g3
            + f5g2
            + f6g1
            + f7g0
            + f8g9_19
            + f9g8_19;
        let mut h8 = f0g8
            + f1g7_2
            + f2g6
            + f3g5_2
            + f4g4
            + f5g3_2
            + f6g2
            + f7g1_2
            + f8g0
            + f9g9_38;
        let mut h9 =
            f0g9 + f1g8 + f2g7 + f3g6 + f4g5 + f5g4 + f6g3 + f7g2 + f8g1 + f9g0;
        let mut carry0;
        let carry1;
        let carry2;
        let carry3;
        let mut carry4;
        let carry5;
        let carry6;
        let carry7;
        let carry8;
        let carry9;

        // |h0| <= (1.1*1.1*2^52*(1+19+19+19+19)+1.1*1.1*2^50*(38+38+38+38+38))
        //   i.e. |h0| <= 1.2*2^59; narrower ranges for h2, h4, h6, h8
        // |h1| <= (1.1*1.1*2^51*(1+1+19+19+19+19+19+19+19+19))
        //   i.e. |h1| <= 1.5*2^58; narrower ranges for h3, h5, h7, h9

        carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        // |h0| <= 2^25
        // |h4| <= 2^25
        // |h1| <= 1.51*2^58
        // |h5| <= 1.51*2^58

        carry1 = (h1 + (1 << 24)) >> 25;
        h2 += carry1;
        h1 -= carry1 << 25;
        carry5 = (h5 + (1 << 24)) >> 25;
        h6 += carry5;
        h5 -= carry5 << 25;
        // |h1| <= 2^24; from now on fits into int32
        // |h5| <= 2^24; from now on fits into int32
        // |h2| <= 1.21*2^59
        // |h6| <= 1.21*2^59

        carry2 = (h2 + (1 << 25)) >> 26;
        h3 += carry2;
        h2 -= carry2 << 26;
        carry6 = (h6 + (1 << 25)) >> 26;
        h7 += carry6;
        h6 -= carry6 << 26;
        // |h2| <= 2^25; from now on fits into int32 unchanged
        // |h6| <= 2^25; from now on fits into int32 unchanged
        // |h3| <= 1.51*2^58
        // |h7| <= 1.51*2^58

        carry3 = (h3 + (1 << 24)) >> 25;
        h4 += carry3;
        h3 -= carry3 << 25;
        carry7 = (h7 + (1 << 24)) >> 25;
        h8 += carry7;
        h7 -= carry7 << 25;
        // |h3| <= 2^24; from now on fits into int32 unchanged
        // |h7| <= 2^24; from now on fits into int32 unchanged
        // |h4| <= 1.52*2^33
        // |h8| <= 1.52*2^33

        carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        carry8 = (h8 + (1 << 25)) >> 26;
        h9 += carry8;
        h8 -= carry8 << 26;
        // |h4| <= 2^25; from now on fits into int32 unchanged
        // |h8| <= 2^25; from now on fits into int32 unchanged
        // |h5| <= 1.01*2^24
        // |h9| <= 1.51*2^58

        carry9 = (h9 + (1 << 24)) >> 25;
        h0 += carry9 * 19;
        h9 -= carry9 << 25;
        // |h9| <= 2^24; from now on fits into int32 unchanged
        // |h0| <= 1.8*2^37

        carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        // |h0| <= 2^25; from now on fits into int32 unchanged
        // |h1| <= 1.01*2^24

        FieldElement([
            h0 as i32, h1 as i32, h2 as i32, h3 as i32, h4 as i32, h5 as i32,
            h6 as i32, h7 as i32, h8 as i32, h9 as i32,
        ])
    }
}

impl FieldElement {
    pub fn from_bytes(s: &[u8]) -> FieldElement {
        let mut h0 = load_4i(&s[0..4]);
        let mut h1 = load_3i(&s[4..7]) << 6;
        let mut h2 = load_3i(&s[7..10]) << 5;
        let mut h3 = load_3i(&s[10..13]) << 3;
        let mut h4 = load_3i(&s[13..16]) << 2;
        let mut h5 = load_4i(&s[16..20]);
        let mut h6 = load_3i(&s[20..23]) << 7;
        let mut h7 = load_3i(&s[23..26]) << 5;
        let mut h8 = load_3i(&s[26..29]) << 4;
        let mut h9 = (load_3i(&s[29..32]) & 8_388_607) << 2;

        let carry9 = (h9 + (1 << 24)) >> 25;
        h0 += carry9 * 19;
        h9 -= carry9 << 25;
        let carry1 = (h1 + (1 << 24)) >> 25;
        h2 += carry1;
        h1 -= carry1 << 25;
        let carry3 = (h3 + (1 << 24)) >> 25;
        h4 += carry3;
        h3 -= carry3 << 25;
        let carry5 = (h5 + (1 << 24)) >> 25;
        h6 += carry5;
        h5 -= carry5 << 25;
        let carry7 = (h7 + (1 << 24)) >> 25;
        h8 += carry7;
        h7 -= carry7 << 25;

        let carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        let carry2 = (h2 + (1 << 25)) >> 26;
        h3 += carry2;
        h2 -= carry2 << 26;
        let carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        let carry6 = (h6 + (1 << 25)) >> 26;
        h7 += carry6;
        h6 -= carry6 << 26;
        let carry8 = (h8 + (1 << 25)) >> 26;
        h9 += carry8;
        h8 -= carry8 << 26;

        FieldElement([
            h0 as i32, h1 as i32, h2 as i32, h3 as i32, h4 as i32, h5 as i32,
            h6 as i32, h7 as i32, h8 as i32, h9 as i32,
        ])
    }

    // Preconditions:
    //   |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    //
    // Write p=2^255-19; q=floor(h/p).
    // Basic claim: q = floor(2^(-255)(h + 19 2^(-25)h9 + 2^(-1))).
    //
    // Proof:
    //   Have |h|<=p so |q|<=1 so |19^2 2^(-255) q|<1/4.
    //   Also have |h-2^230 h9|<2^230 so |19 2^(-255)(h-2^230 h9)|<1/4.
    //
    //   Write y=2^(-1)-19^2 2^(-255)q-19 2^(-255)(h-2^230 h9).
    //   Then 0<y<1.
    //
    //   Write r=h-pq.
    //   Have 0<=r<=p-1=2^255-20.
    //   Thus 0<=r+19(2^-255)r<r+19(2^-255)2^255<=2^255-1.
    //
    //   Write x=r+19(2^-255)r+y.
    //   Then 0<x<2^255 so floor(2^(-255)x) = 0 so floor(q+2^(-255)x) = q.
    //
    //   Have q+2^(-255)x = 2^(-255)(h + 19 2^(-25) h9 + 2^(-1))
    //   so floor(2^(-255)(h + 19 2^(-25) h9 + 2^(-1))) = q.
    pub fn to_bytes(&self) -> [u8; 32] {
        let &FieldElement(es) = self;
        let mut h0 = es[0];
        let mut h1 = es[1];
        let mut h2 = es[2];
        let mut h3 = es[3];
        let mut h4 = es[4];
        let mut h5 = es[5];
        let mut h6 = es[6];
        let mut h7 = es[7];
        let mut h8 = es[8];
        let mut h9 = es[9];
        let mut q;

        q = (19 * h9 + (1 << 24)) >> 25;
        q = (h0 + q) >> 26;
        q = (h1 + q) >> 25;
        q = (h2 + q) >> 26;
        q = (h3 + q) >> 25;
        q = (h4 + q) >> 26;
        q = (h5 + q) >> 25;
        q = (h6 + q) >> 26;
        q = (h7 + q) >> 25;
        q = (h8 + q) >> 26;
        q = (h9 + q) >> 25;

        // Goal: Output h-(2^255-19)q, which is between 0 and 2^255-20.
        h0 += 19 * q;
        // Goal: Output h-2^255 q, which is between 0 and 2^255-20.

        let carry0 = h0 >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        let carry1 = h1 >> 25;
        h2 += carry1;
        h1 -= carry1 << 25;
        let carry2 = h2 >> 26;
        h3 += carry2;
        h2 -= carry2 << 26;
        let carry3 = h3 >> 25;
        h4 += carry3;
        h3 -= carry3 << 25;
        let carry4 = h4 >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        let carry5 = h5 >> 25;
        h6 += carry5;
        h5 -= carry5 << 25;
        let carry6 = h6 >> 26;
        h7 += carry6;
        h6 -= carry6 << 26;
        let carry7 = h7 >> 25;
        h8 += carry7;
        h7 -= carry7 << 25;
        let carry8 = h8 >> 26;
        h9 += carry8;
        h8 -= carry8 << 26;
        let carry9 = h9 >> 25;
        h9 -= carry9 << 25;
        // h10 = carry9

        // Goal: Output h0+...+2^255 h10-2^255 q, which is between 0 and
        // 2^255-20. Have h0+...+2^230 h9 between 0 and 2^255-1;
        // evidently 2^255 h10-2^255 q = 0.
        // Goal: Output h0+...+2^230 h9.
        [
            h0 as u8,
            (h0 >> 8) as u8,
            (h0 >> 16) as u8,
            ((h0 >> 24) | (h1 << 2)) as u8,
            (h1 >> 6) as u8,
            (h1 >> 14) as u8,
            ((h1 >> 22) | (h2 << 3)) as u8,
            (h2 >> 5) as u8,
            (h2 >> 13) as u8,
            ((h2 >> 21) | (h3 << 5)) as u8,
            (h3 >> 3) as u8,
            (h3 >> 11) as u8,
            ((h3 >> 19) | (h4 << 6)) as u8,
            (h4 >> 2) as u8,
            (h4 >> 10) as u8,
            (h4 >> 18) as u8,
            h5 as u8,
            (h5 >> 8) as u8,
            (h5 >> 16) as u8,
            ((h5 >> 24) | (h6 << 1)) as u8,
            (h6 >> 7) as u8,
            (h6 >> 15) as u8,
            ((h6 >> 23) | (h7 << 3)) as u8,
            (h7 >> 5) as u8,
            (h7 >> 13) as u8,
            ((h7 >> 21) | (h8 << 4)) as u8,
            (h8 >> 4) as u8,
            (h8 >> 12) as u8,
            ((h8 >> 20) | (h9 << 6)) as u8,
            (h9 >> 2) as u8,
            (h9 >> 10) as u8,
            (h9 >> 18) as u8,
        ]
    }

    pub fn maybe_swap_with(&mut self, other: &mut FieldElement, do_swap: i32) {
        let &mut FieldElement(f) = self;
        let &mut FieldElement(g) = other;
        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;
        let [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9] = g;
        let mut x0 = f0 ^ g0;
        let mut x1 = f1 ^ g1;
        let mut x2 = f2 ^ g2;
        let mut x3 = f3 ^ g3;
        let mut x4 = f4 ^ g4;
        let mut x5 = f5 ^ g5;
        let mut x6 = f6 ^ g6;
        let mut x7 = f7 ^ g7;
        let mut x8 = f8 ^ g8;
        let mut x9 = f9 ^ g9;
        let b = -do_swap;
        x0 &= b;
        x1 &= b;
        x2 &= b;
        x3 &= b;
        x4 &= b;
        x5 &= b;
        x6 &= b;
        x7 &= b;
        x8 &= b;
        x9 &= b;
        *self = FieldElement([
            f0 ^ x0,
            f1 ^ x1,
            f2 ^ x2,
            f3 ^ x3,
            f4 ^ x4,
            f5 ^ x5,
            f6 ^ x6,
            f7 ^ x7,
            f8 ^ x8,
            f9 ^ x9,
        ]);
        *other = FieldElement([
            g0 ^ x0,
            g1 ^ x1,
            g2 ^ x2,
            g3 ^ x3,
            g4 ^ x4,
            g5 ^ x5,
            g6 ^ x6,
            g7 ^ x7,
            g8 ^ x8,
            g9 ^ x9,
        ]);
    }

    pub fn maybe_set(&mut self, other: &FieldElement, do_swap: i32) {
        let &mut FieldElement(f) = self;
        let &FieldElement(g) = other;
        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;
        let [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9] = g;
        let mut x0 = f0 ^ g0;
        let mut x1 = f1 ^ g1;
        let mut x2 = f2 ^ g2;
        let mut x3 = f3 ^ g3;
        let mut x4 = f4 ^ g4;
        let mut x5 = f5 ^ g5;
        let mut x6 = f6 ^ g6;
        let mut x7 = f7 ^ g7;
        let mut x8 = f8 ^ g8;
        let mut x9 = f9 ^ g9;
        let b = -do_swap;
        x0 &= b;
        x1 &= b;
        x2 &= b;
        x3 &= b;
        x4 &= b;
        x5 &= b;
        x6 &= b;
        x7 &= b;
        x8 &= b;
        x9 &= b;
        *self = FieldElement([
            f0 ^ x0,
            f1 ^ x1,
            f2 ^ x2,
            f3 ^ x3,
            f4 ^ x4,
            f5 ^ x5,
            f6 ^ x6,
            f7 ^ x7,
            f8 ^ x8,
            f9 ^ x9,
        ]);
    }

    // h = f * 121666
    // Can overlap h with f.
    //
    // Preconditions:
    //    |f| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
    //
    // Postconditions:
    //    |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    fn mul_121666(&self) -> FieldElement {
        let &FieldElement(f) = self;

        let mut h0 = i64::from(f[0]) * 121_666;
        let mut h1 = i64::from(f[1]) * 121_666;
        let mut h2 = i64::from(f[2]) * 121_666;
        let mut h3 = i64::from(f[3]) * 121_666;
        let mut h4 = i64::from(f[4]) * 121_666;
        let mut h5 = i64::from(f[5]) * 121_666;
        let mut h6 = i64::from(f[6]) * 121_666;
        let mut h7 = i64::from(f[7]) * 121_666;
        let mut h8 = i64::from(f[8]) * 121_666;
        let mut h9 = i64::from(f[9]) * 121_666;

        let carry9 = (h9 + (1 << 24)) >> 25;
        h0 += carry9 * 19;
        h9 -= carry9 << 25;
        let carry1 = (h1 + (1 << 24)) >> 25;
        h2 += carry1;
        h1 -= carry1 << 25;
        let carry3 = (h3 + (1 << 24)) >> 25;
        h4 += carry3;
        h3 -= carry3 << 25;
        let carry5 = (h5 + (1 << 24)) >> 25;
        h6 += carry5;
        h5 -= carry5 << 25;
        let carry7 = (h7 + (1 << 24)) >> 25;
        h8 += carry7;
        h7 -= carry7 << 25;

        let carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        let carry2 = (h2 + (1 << 25)) >> 26;
        h3 += carry2;
        h2 -= carry2 << 26;
        let carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        let carry6 = (h6 + (1 << 25)) >> 26;
        h7 += carry6;
        h6 -= carry6 << 26;
        let carry8 = (h8 + (1 << 25)) >> 26;
        h9 += carry8;
        h8 -= carry8 << 26;

        FieldElement([
            h0 as i32, h1 as i32, h2 as i32, h3 as i32, h4 as i32, h5 as i32,
            h6 as i32, h7 as i32, h8 as i32, h9 as i32,
        ])
    }

    // h = f * f
    // Can overlap h with f.
    //
    // Preconditions:
    //    |f| bounded by 1.1*2^26,1.1*2^25,1.1*2^26,1.1*2^25,etc.
    //
    // Postconditions:
    //    |h| bounded by 1.1*2^25,1.1*2^24,1.1*2^25,1.1*2^24,etc.
    // See fe_mul.c for discussion of implementation strategy.
    fn square(&self) -> FieldElement {
        let &FieldElement(f) = self;

        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;

        let f0_2 = 2 * f0;
        let f1_2 = 2 * f1;
        let f2_2 = 2 * f2;
        let f3_2 = 2 * f3;
        let f4_2 = 2 * f4;
        let f5_2 = 2 * f5;
        let f6_2 = 2 * f6;
        let f7_2 = 2 * f7;
        let f5_38 = 38 * f5; // 1.31*2^30
        let f6_19 = 19 * f6; // 1.31*2^30
        let f7_38 = 38 * f7; // 1.31*2^30
        let f8_19 = 19 * f8; // 1.31*2^30
        let f9_38 = 38 * f9; // 1.31*2^30
        let f0f0 = i64::from(f0) * i64::from(f0);
        let f0f1_2 = i64::from(f0_2) * i64::from(f1);
        let f0f2_2 = i64::from(f0_2) * i64::from(f2);
        let f0f3_2 = i64::from(f0_2) * i64::from(f3);
        let f0f4_2 = i64::from(f0_2) * i64::from(f4);
        let f0f5_2 = i64::from(f0_2) * i64::from(f5);
        let f0f6_2 = i64::from(f0_2) * i64::from(f6);
        let f0f7_2 = i64::from(f0_2) * i64::from(f7);
        let f0f8_2 = i64::from(f0_2) * i64::from(f8);
        let f0f9_2 = i64::from(f0_2) * i64::from(f9);
        let f1f1_2 = i64::from(f1_2) * i64::from(f1);
        let f1f2_2 = i64::from(f1_2) * i64::from(f2);
        let f1f3_4 = i64::from(f1_2) * i64::from(f3_2);
        let f1f4_2 = i64::from(f1_2) * i64::from(f4);
        let f1f5_4 = i64::from(f1_2) * i64::from(f5_2);
        let f1f6_2 = i64::from(f1_2) * i64::from(f6);
        let f1f7_4 = i64::from(f1_2) * i64::from(f7_2);
        let f1f8_2 = i64::from(f1_2) * i64::from(f8);
        let f1f9_76 = i64::from(f1_2) * i64::from(f9_38);
        let f2f2 = i64::from(f2) * i64::from(f2);
        let f2f3_2 = i64::from(f2_2) * i64::from(f3);
        let f2f4_2 = i64::from(f2_2) * i64::from(f4);
        let f2f5_2 = i64::from(f2_2) * i64::from(f5);
        let f2f6_2 = i64::from(f2_2) * i64::from(f6);
        let f2f7_2 = i64::from(f2_2) * i64::from(f7);
        let f2f8_38 = i64::from(f2_2) * i64::from(f8_19);
        let f2f9_38 = i64::from(f2) * i64::from(f9_38);
        let f3f3_2 = i64::from(f3_2) * i64::from(f3);
        let f3f4_2 = i64::from(f3_2) * i64::from(f4);
        let f3f5_4 = i64::from(f3_2) * i64::from(f5_2);
        let f3f6_2 = i64::from(f3_2) * i64::from(f6);
        let f3f7_76 = i64::from(f3_2) * i64::from(f7_38);
        let f3f8_38 = i64::from(f3_2) * i64::from(f8_19);
        let f3f9_76 = i64::from(f3_2) * i64::from(f9_38);
        let f4f4 = i64::from(f4) * i64::from(f4);
        let f4f5_2 = i64::from(f4_2) * i64::from(f5);
        let f4f6_38 = i64::from(f4_2) * i64::from(f6_19);
        let f4f7_38 = i64::from(f4) * i64::from(f7_38);
        let f4f8_38 = i64::from(f4_2) * i64::from(f8_19);
        let f4f9_38 = i64::from(f4) * i64::from(f9_38);
        let f5f5_38 = i64::from(f5) * i64::from(f5_38);
        let f5f6_38 = i64::from(f5_2) * i64::from(f6_19);
        let f5f7_76 = i64::from(f5_2) * i64::from(f7_38);
        let f5f8_38 = i64::from(f5_2) * i64::from(f8_19);
        let f5f9_76 = i64::from(f5_2) * i64::from(f9_38);
        let f6f6_19 = i64::from(f6) * i64::from(f6_19);
        let f6f7_38 = i64::from(f6) * i64::from(f7_38);
        let f6f8_38 = i64::from(f6_2) * i64::from(f8_19);
        let f6f9_38 = i64::from(f6) * i64::from(f9_38);
        let f7f7_38 = i64::from(f7) * i64::from(f7_38);
        let f7f8_38 = i64::from(f7_2) * i64::from(f8_19);
        let f7f9_76 = i64::from(f7_2) * i64::from(f9_38);
        let f8f8_19 = i64::from(f8) * i64::from(f8_19);
        let f8f9_38 = i64::from(f8) * i64::from(f9_38);
        let f9f9_38 = i64::from(f9) * i64::from(f9_38);
        let mut h0 = f0f0 + f1f9_76 + f2f8_38 + f3f7_76 + f4f6_38 + f5f5_38;
        let mut h1 = f0f1_2 + f2f9_38 + f3f8_38 + f4f7_38 + f5f6_38;
        let mut h2 = f0f2_2 + f1f1_2 + f3f9_76 + f4f8_38 + f5f7_76 + f6f6_19;
        let mut h3 = f0f3_2 + f1f2_2 + f4f9_38 + f5f8_38 + f6f7_38;
        let mut h4 = f0f4_2 + f1f3_4 + f2f2 + f5f9_76 + f6f8_38 + f7f7_38;
        let mut h5 = f0f5_2 + f1f4_2 + f2f3_2 + f6f9_38 + f7f8_38;
        let mut h6 = f0f6_2 + f1f5_4 + f2f4_2 + f3f3_2 + f7f9_76 + f8f8_19;
        let mut h7 = f0f7_2 + f1f6_2 + f2f5_2 + f3f4_2 + f8f9_38;
        let mut h8 = f0f8_2 + f1f7_4 + f2f6_2 + f3f5_4 + f4f4 + f9f9_38;
        let mut h9 = f0f9_2 + f1f8_2 + f2f7_2 + f3f6_2 + f4f5_2;

        let carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        let carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;

        let carry1 = (h1 + (1 << 24)) >> 25;
        h2 += carry1;
        h1 -= carry1 << 25;
        let carry5 = (h5 + (1 << 24)) >> 25;
        h6 += carry5;
        h5 -= carry5 << 25;

        let carry2 = (h2 + (1 << 25)) >> 26;
        h3 += carry2;
        h2 -= carry2 << 26;
        let carry6 = (h6 + (1 << 25)) >> 26;
        h7 += carry6;
        h6 -= carry6 << 26;

        let carry3 = (h3 + (1 << 24)) >> 25;
        h4 += carry3;
        h3 -= carry3 << 25;
        let carry7 = (h7 + (1 << 24)) >> 25;
        h8 += carry7;
        h7 -= carry7 << 25;

        let carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        let carry8 = (h8 + (1 << 25)) >> 26;
        h9 += carry8;
        h8 -= carry8 << 26;

        let carry9 = (h9 + (1 << 24)) >> 25;
        h0 += carry9 * 19;
        h9 -= carry9 << 25;

        let carrya = (h0 + (1 << 25)) >> 26;
        h1 += carrya;
        h0 -= carrya << 26;

        FieldElement([
            h0 as i32, h1 as i32, h2 as i32, h3 as i32, h4 as i32, h5 as i32,
            h6 as i32, h7 as i32, h8 as i32, h9 as i32,
        ])
    }

    fn square_and_double(&self) -> FieldElement {
        let &FieldElement(f) = self;

        let [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = f;

        let f0_2 = 2 * f0;
        let f1_2 = 2 * f1;
        let f2_2 = 2 * f2;
        let f3_2 = 2 * f3;
        let f4_2 = 2 * f4;
        let f5_2 = 2 * f5;
        let f6_2 = 2 * f6;
        let f7_2 = 2 * f7;
        let f5_38 = 38 * f5; // 1.959375*2^30
        let f6_19 = 19 * f6; // 1.959375*2^30
        let f7_38 = 38 * f7; // 1.959375*2^30
        let f8_19 = 19 * f8; // 1.959375*2^30
        let f9_38 = 38 * f9; // 1.959375*2^30
        let f0f0 = i64::from(f0) * i64::from(f0);
        let f0f1_2 = i64::from(f0_2) * i64::from(f1);
        let f0f2_2 = i64::from(f0_2) * i64::from(f2);
        let f0f3_2 = i64::from(f0_2) * i64::from(f3);
        let f0f4_2 = i64::from(f0_2) * i64::from(f4);
        let f0f5_2 = i64::from(f0_2) * i64::from(f5);
        let f0f6_2 = i64::from(f0_2) * i64::from(f6);
        let f0f7_2 = i64::from(f0_2) * i64::from(f7);
        let f0f8_2 = i64::from(f0_2) * i64::from(f8);
        let f0f9_2 = i64::from(f0_2) * i64::from(f9);
        let f1f1_2 = i64::from(f1_2) * i64::from(f1);
        let f1f2_2 = i64::from(f1_2) * i64::from(f2);
        let f1f3_4 = i64::from(f1_2) * i64::from(f3_2);
        let f1f4_2 = i64::from(f1_2) * i64::from(f4);
        let f1f5_4 = i64::from(f1_2) * i64::from(f5_2);
        let f1f6_2 = i64::from(f1_2) * i64::from(f6);
        let f1f7_4 = i64::from(f1_2) * i64::from(f7_2);
        let f1f8_2 = i64::from(f1_2) * i64::from(f8);
        let f1f9_76 = i64::from(f1_2) * i64::from(f9_38);
        let f2f2 = i64::from(f2) * i64::from(f2);
        let f2f3_2 = i64::from(f2_2) * i64::from(f3);
        let f2f4_2 = i64::from(f2_2) * i64::from(f4);
        let f2f5_2 = i64::from(f2_2) * i64::from(f5);
        let f2f6_2 = i64::from(f2_2) * i64::from(f6);
        let f2f7_2 = i64::from(f2_2) * i64::from(f7);
        let f2f8_38 = i64::from(f2_2) * i64::from(f8_19);
        let f2f9_38 = i64::from(f2) * i64::from(f9_38);
        let f3f3_2 = i64::from(f3_2) * i64::from(f3);
        let f3f4_2 = i64::from(f3_2) * i64::from(f4);
        let f3f5_4 = i64::from(f3_2) * i64::from(f5_2);
        let f3f6_2 = i64::from(f3_2) * i64::from(f6);
        let f3f7_76 = i64::from(f3_2) * i64::from(f7_38);
        let f3f8_38 = i64::from(f3_2) * i64::from(f8_19);
        let f3f9_76 = i64::from(f3_2) * i64::from(f9_38);
        let f4f4 = i64::from(f4) * i64::from(f4);
        let f4f5_2 = i64::from(f4_2) * i64::from(f5);
        let f4f6_38 = i64::from(f4_2) * i64::from(f6_19);
        let f4f7_38 = i64::from(f4) * i64::from(f7_38);
        let f4f8_38 = i64::from(f4_2) * i64::from(f8_19);
        let f4f9_38 = i64::from(f4) * i64::from(f9_38);
        let f5f5_38 = i64::from(f5) * i64::from(f5_38);
        let f5f6_38 = i64::from(f5_2) * i64::from(f6_19);
        let f5f7_76 = i64::from(f5_2) * i64::from(f7_38);
        let f5f8_38 = i64::from(f5_2) * i64::from(f8_19);
        let f5f9_76 = i64::from(f5_2) * i64::from(f9_38);
        let f6f6_19 = i64::from(f6) * i64::from(f6_19);
        let f6f7_38 = i64::from(f6) * i64::from(f7_38);
        let f6f8_38 = i64::from(f6_2) * i64::from(f8_19);
        let f6f9_38 = i64::from(f6) * i64::from(f9_38);
        let f7f7_38 = i64::from(f7) * i64::from(f7_38);
        let f7f8_38 = i64::from(f7_2) * i64::from(f8_19);
        let f7f9_76 = i64::from(f7_2) * i64::from(f9_38);
        let f8f8_19 = i64::from(f8) * i64::from(f8_19);
        let f8f9_38 = i64::from(f8) * i64::from(f9_38);
        let f9f9_38 = i64::from(f9) * i64::from(f9_38);
        let mut h0 = f0f0 + f1f9_76 + f2f8_38 + f3f7_76 + f4f6_38 + f5f5_38;
        let mut h1 = f0f1_2 + f2f9_38 + f3f8_38 + f4f7_38 + f5f6_38;
        let mut h2 = f0f2_2 + f1f1_2 + f3f9_76 + f4f8_38 + f5f7_76 + f6f6_19;
        let mut h3 = f0f3_2 + f1f2_2 + f4f9_38 + f5f8_38 + f6f7_38;
        let mut h4 = f0f4_2 + f1f3_4 + f2f2 + f5f9_76 + f6f8_38 + f7f7_38;
        let mut h5 = f0f5_2 + f1f4_2 + f2f3_2 + f6f9_38 + f7f8_38;
        let mut h6 = f0f6_2 + f1f5_4 + f2f4_2 + f3f3_2 + f7f9_76 + f8f8_19;
        let mut h7 = f0f7_2 + f1f6_2 + f2f5_2 + f3f4_2 + f8f9_38;
        let mut h8 = f0f8_2 + f1f7_4 + f2f6_2 + f3f5_4 + f4f4 + f9f9_38;
        let mut h9 = f0f9_2 + f1f8_2 + f2f7_2 + f3f6_2 + f4f5_2;
        let mut carry0: i64;
        let carry1: i64;
        let carry2: i64;
        let carry3: i64;
        let mut carry4: i64;
        let carry5: i64;
        let carry6: i64;
        let carry7: i64;
        let carry8: i64;
        let carry9: i64;

        h0 += h0;
        h1 += h1;
        h2 += h2;
        h3 += h3;
        h4 += h4;
        h5 += h5;
        h6 += h6;
        h7 += h7;
        h8 += h8;
        h9 += h9;

        carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;
        carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;

        carry1 = (h1 + (1 << 24)) >> 25;
        h2 += carry1;
        h1 -= carry1 << 25;
        carry5 = (h5 + (1 << 24)) >> 25;
        h6 += carry5;
        h5 -= carry5 << 25;

        carry2 = (h2 + (1 << 25)) >> 26;
        h3 += carry2;
        h2 -= carry2 << 26;
        carry6 = (h6 + (1 << 25)) >> 26;
        h7 += carry6;
        h6 -= carry6 << 26;

        carry3 = (h3 + (1 << 24)) >> 25;
        h4 += carry3;
        h3 -= carry3 << 25;
        carry7 = (h7 + (1 << 24)) >> 25;
        h8 += carry7;
        h7 -= carry7 << 25;

        carry4 = (h4 + (1 << 25)) >> 26;
        h5 += carry4;
        h4 -= carry4 << 26;
        carry8 = (h8 + (1 << 25)) >> 26;
        h9 += carry8;
        h8 -= carry8 << 26;

        carry9 = (h9 + (1 << 24)) >> 25;
        h0 += carry9 * 19;
        h9 -= carry9 << 25;

        carry0 = (h0 + (1 << 25)) >> 26;
        h1 += carry0;
        h0 -= carry0 << 26;

        FieldElement([
            h0 as i32, h1 as i32, h2 as i32, h3 as i32, h4 as i32, h5 as i32,
            h6 as i32, h7 as i32, h8 as i32, h9 as i32,
        ])
    }

    pub fn invert(&self) -> FieldElement {
        let z1 = *self;

        // qhasm: z2 = z1^2^1
        let z2 = z1.square();
        // qhasm: z8 = z2^2^2
        let z8 = z2.square().square();
        // qhasm: z9 = z1*z8
        let z9 = z1 * z8;

        // qhasm: z11 = z2*z9
        let z11 = z2 * z9;

        // qhasm: z22 = z11^2^1
        let z22 = z11.square();

        // qhasm: z_5_0 = z9*z22
        let z_5_0 = z9 * z22;

        // qhasm: z_10_5 = z_5_0^2^5
        let z_10_5 = (0..5).fold(z_5_0, |z_5_n, _| z_5_n.square());

        // qhasm: z_10_0 = z_10_5*z_5_0
        let z_10_0 = z_10_5 * z_5_0;

        // qhasm: z_20_10 = z_10_0^2^10
        let z_20_10 = (0..10).fold(z_10_0, |x, _| x.square());

        // qhasm: z_20_0 = z_20_10*z_10_0
        let z_20_0 = z_20_10 * z_10_0;

        // qhasm: z_40_20 = z_20_0^2^20
        let z_40_20 = (0..20).fold(z_20_0, |x, _| x.square());

        // qhasm: z_40_0 = z_40_20*z_20_0
        let z_40_0 = z_40_20 * z_20_0;

        // qhasm: z_50_10 = z_40_0^2^10
        let z_50_10 = (0..10).fold(z_40_0, |x, _| x.square());

        // qhasm: z_50_0 = z_50_10*z_10_0
        let z_50_0 = z_50_10 * z_10_0;

        // qhasm: z_100_50 = z_50_0^2^50
        let z_100_50 = (0..50).fold(z_50_0, |x, _| x.square());

        // qhasm: z_100_0 = z_100_50*z_50_0
        let z_100_0 = z_100_50 * z_50_0;

        // qhasm: z_200_100 = z_100_0^2^100
        let z_200_100 = (0..100).fold(z_100_0, |x, _| x.square());

        // qhasm: z_200_0 = z_200_100*z_100_0
        // asm 1: fe_mul(>z_200_0=fe#3,<z_200_100=fe#4,<z_100_0=fe#3);
        // asm 2: fe_mul(>z_200_0=t2,<z_200_100=t3,<z_100_0=t2);
        let z_200_0 = z_200_100 * z_100_0;

        // qhasm: z_250_50 = z_200_0^2^50
        let z_250_50 = (0..50).fold(z_200_0, |x, _| x.square());

        // qhasm: z_250_0 = z_250_50*z_50_0
        let z_250_0 = z_250_50 * z_50_0;

        // qhasm: z_255_5 = z_250_0^2^5
        let z_255_5 = (0..5).fold(z_250_0, |x, _| x.square());

        // qhasm: z_255_21 = z_255_5*z11
        // asm 1: fe_mul(>z_255_21=fe#12,<z_255_5=fe#2,<z11=fe#1);
        // asm 2: fe_mul(>z_255_21=out,<z_255_5=t1,<z11=t0);
        z_255_5 * z11
    }

    fn is_nonzero(&self) -> bool {
        let bs = self.to_bytes();
        let zero = [0; 32];
        !fixed_time_eq(bs.as_ref(), zero.as_ref())
    }

    fn is_negative(&self) -> bool { (self.to_bytes()[0] & 1) != 0 }

    fn neg(&self) -> FieldElement {
        let &FieldElement(f) = self;
        FieldElement([
            -f[0], -f[1], -f[2], -f[3], -f[4], -f[5], -f[6], -f[7], -f[8],
            -f[9],
        ])
    }

    fn pow25523(&self) -> FieldElement {
        let z2 = self.square();
        let z8 = (0..2).fold(z2, |x, _| x.square());
        let z9 = *self * z8;
        let z11 = z2 * z9;
        let z22 = z11.square();
        let z_5_0 = z9 * z22;
        let z_10_5 = (0..5).fold(z_5_0, |x, _| x.square());
        let z_10_0 = z_10_5 * z_5_0;
        let z_20_10 = (0..10).fold(z_10_0, |x, _| x.square());
        let z_20_0 = z_20_10 * z_10_0;
        let z_40_20 = (0..20).fold(z_20_0, |x, _| x.square());
        let z_40_0 = z_40_20 * z_20_0;
        let z_50_10 = (0..10).fold(z_40_0, |x, _| x.square());
        let z_50_0 = z_50_10 * z_10_0;
        let z_100_50 = (0..50).fold(z_50_0, |x, _| x.square());
        let z_100_0 = z_100_50 * z_50_0;
        let z_200_100 = (0..100).fold(z_100_0, |x, _| x.square());
        let z_200_0 = z_200_100 * z_100_0;
        let z_250_50 = (0..50).fold(z_200_0, |x, _| x.square());
        let z_250_0 = z_250_50 * z_50_0;
        let z_252_2 = (0..2).fold(z_250_0, |x, _| x.square());
        z_252_2 * *self
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct GeP2 {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct GeP3 {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
    t: FieldElement,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct GeP1P1 {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
    t: FieldElement,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct GePrecomp {
    y_plus_x: FieldElement,
    y_minus_x: FieldElement,
    xy2d: FieldElement,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct GeCached {
    y_plus_x: FieldElement,
    y_minus_x: FieldElement,
    z: FieldElement,
    t2d: FieldElement,
}

impl GeP1P1 {
    fn to_p2(&self) -> GeP2 {
        GeP2 {
            x: self.x * self.t,
            y: self.y * self.z,
            z: self.z * self.t,
        }
    }

    fn to_p3(&self) -> GeP3 {
        GeP3 {
            x: self.x * self.t,
            y: self.y * self.z,
            z: self.z * self.t,
            t: self.x * self.y,
        }
    }
}

impl GeP2 {
    fn zero() -> GeP2 {
        GeP2 {
            x: FE_ZERO,
            y: FE_ONE,
            z: FE_ONE,
        }
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let recip = self.z.invert();
        let x = self.x * recip;
        let y = self.y * recip;
        let mut bs = y.to_bytes();
        bs[31] ^= (if x.is_negative() { 1 } else { 0 }) << 7;
        bs
    }

    fn dbl(&self) -> GeP1P1 {
        let xx = self.x.square();
        let yy = self.y.square();
        let b = self.z.square_and_double();
        let a = self.x + self.y;
        let aa = a.square();
        let y3 = yy + xx;
        let z3 = yy - xx;
        let x3 = aa - y3;
        let t3 = b - z3;

        GeP1P1 {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }

    fn slide(a: &[u8]) -> [i8; 256] {
        let mut r = [0i8; 256];
        for i in 0..256 {
            r[i] = (1 & (a[i >> 3] >> (i & 7))) as i8;
        }
        for i in 0..256 {
            if r[i] != 0 {
                for b in 1..min(7, 256 - i) {
                    if r[i + b] != 0 {
                        if r[i] + (r[i + b] << b) <= 15 {
                            r[i] += r[i + b] << b;
                            r[i + b] = 0;
                        } else if r[i] - (r[i + b] << b) >= -15 {
                            r[i] -= r[i + b] << b;
                            for k in r.iter_mut().skip(i + b) {
                                if *k == 0 {
                                    *k = 1;
                                    break;
                                }
                                *k = 0;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        r
    }

    // r = a * A + b * B
    // where a = a[0]+256*a[1]+...+256^31 a[31].
    // and b = b[0]+256*b[1]+...+256^31 b[31].
    // B is the Ed25519 base point (x,4/5) with x positive.
    pub fn double_scalarmult_vartime(
        a_scalar: &[u8],
        a_point: GeP3,
        b_scalar: &[u8],
    ) -> GeP2 {
        let aslide = GeP2::slide(a_scalar);
        let bslide = GeP2::slide(b_scalar);

        let mut ai = [GeCached {
            y_plus_x: FE_ZERO,
            y_minus_x: FE_ZERO,
            z: FE_ZERO,
            t2d: FE_ZERO,
        }; 8]; // A,3A,5A,7A,9A,11A,13A,15A
        ai[0] = a_point.to_cached();
        let a2 = a_point.dbl().to_p3();
        ai[1] = (a2 + ai[0]).to_p3().to_cached();
        ai[2] = (a2 + ai[1]).to_p3().to_cached();
        ai[3] = (a2 + ai[2]).to_p3().to_cached();
        ai[4] = (a2 + ai[3]).to_p3().to_cached();
        ai[5] = (a2 + ai[4]).to_p3().to_cached();
        ai[6] = (a2 + ai[5]).to_p3().to_cached();
        ai[7] = (a2 + ai[6]).to_p3().to_cached();

        let mut r = GeP2::zero();

        let mut i: usize = 255;
        loop {
            if aslide[i] != 0 || bslide[i] != 0 {
                break;
            }
            if i == 0 {
                return r;
            }
            i -= 1;
        }

        loop {
            let mut t = r.dbl();
            if aslide[i] > 0 {
                t = t.to_p3() + ai[(aslide[i] / 2) as usize];
            } else if aslide[i] < 0 {
                t = t.to_p3() - ai[(-aslide[i] / 2) as usize];
            }

            if bslide[i] > 0 {
                t = t.to_p3() + BI[(bslide[i] / 2) as usize];
            } else if bslide[i] < 0 {
                t = t.to_p3() - BI[(-bslide[i] / 2) as usize];
            }

            r = t.to_p2();

            if i == 0 {
                return r;
            }
            i -= 1;
        }
    }
}

impl GeP3 {
    pub fn from_bytes_negate_vartime(s: &[u8]) -> Option<GeP3> {
        let y = FieldElement::from_bytes(s);
        let z = FE_ONE;
        let y_squared = y.square();
        let u = y_squared - FE_ONE;
        let v = (y_squared * FE_D) + FE_ONE;
        let v_raise_3 = v.square() * v;
        let v_raise_7 = v_raise_3.square() * v;
        let uv7 = v_raise_7 * u; // Is this commutative? u comes second in the code, but not in the
                                 // notation...

        let mut x = uv7.pow25523() * v_raise_3 * u;

        let vxx = x.square() * v;
        let check = vxx - u;
        if check.is_nonzero() {
            let check2 = vxx + u;
            if check2.is_nonzero() {
                return None;
            }
            x = x * FE_SQRTM1;
        }

        if x.is_negative() == ((s[31] >> 7) != 0) {
            x = x.neg();
        }

        let t = x * y;

        Some(GeP3 { x, y, z, t })
    }

    fn to_p2(&self) -> GeP2 {
        GeP2 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    fn to_cached(&self) -> GeCached {
        GeCached {
            y_plus_x: self.y + self.x,
            y_minus_x: self.y - self.x,
            z: self.z,
            t2d: self.t * FE_D2,
        }
    }

    fn zero() -> GeP3 {
        GeP3 {
            x: FE_ZERO,
            y: FE_ONE,
            z: FE_ONE,
            t: FE_ZERO,
        }
    }

    fn dbl(&self) -> GeP1P1 { self.to_p2().dbl() }

    pub fn to_bytes(&self) -> [u8; 32] {
        let recip = self.z.invert();
        let x = self.x * recip;
        let y = self.y * recip;
        let mut bs = y.to_bytes();
        bs[31] ^= (if x.is_negative() { 1 } else { 0 }) << 7;
        bs
    }
}

impl Add<GeCached> for GeP3 {
    type Output = GeP1P1;

    fn add(self, _rhs: GeCached) -> GeP1P1 {
        let y1_plus_x1 = self.y + self.x;
        let y1_minus_x1 = self.y - self.x;
        let a = y1_plus_x1 * _rhs.y_plus_x;
        let b = y1_minus_x1 * _rhs.y_minus_x;
        let c = _rhs.t2d * self.t;
        let zz = self.z * _rhs.z;
        let d = zz + zz;
        let x3 = a - b;
        let y3 = a + b;
        let z3 = d + c;
        let t3 = d - c;

        GeP1P1 {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }
}

impl Add<GePrecomp> for GeP3 {
    type Output = GeP1P1;

    fn add(self, _rhs: GePrecomp) -> GeP1P1 {
        let y1_plus_x1 = self.y + self.x;
        let y1_minus_x1 = self.y - self.x;
        let a = y1_plus_x1 * _rhs.y_plus_x;
        let b = y1_minus_x1 * _rhs.y_minus_x;
        let c = _rhs.xy2d * self.t;
        let d = self.z + self.z;
        let x3 = a - b;
        let y3 = a + b;
        let z3 = d + c;
        let t3 = d - c;

        GeP1P1 {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }
}

impl Sub<GeCached> for GeP3 {
    type Output = GeP1P1;

    fn sub(self, _rhs: GeCached) -> GeP1P1 {
        let y1_plus_x1 = self.y + self.x;
        let y1_minus_x1 = self.y - self.x;
        let a = y1_plus_x1 * _rhs.y_minus_x;
        let b = y1_minus_x1 * _rhs.y_plus_x;
        let c = _rhs.t2d * self.t;
        let zz = self.z * _rhs.z;
        let d = zz + zz;
        let x3 = a - b;
        let y3 = a + b;
        let z3 = d - c;
        let t3 = d + c;

        GeP1P1 {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }
}

impl Sub<GePrecomp> for GeP3 {
    type Output = GeP1P1;

    fn sub(self, _rhs: GePrecomp) -> GeP1P1 {
        let y1_plus_x1 = self.y + self.x;
        let y1_minus_x1 = self.y - self.x;
        let a = y1_plus_x1 * _rhs.y_minus_x;
        let b = y1_minus_x1 * _rhs.y_plus_x;
        let c = _rhs.xy2d * self.t;
        let d = self.z + self.z;
        let x3 = a - b;
        let y3 = a + b;
        let z3 = d - c;
        let t3 = d + c;

        GeP1P1 {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }
}

#[inline]
fn equal(b: u8, c: u8) -> i32 {
    let x = b ^ c; // 0: yes; 1..255: no
    let mut y = u32::from(x); // 0: yes; 1..255: no
    y = y.wrapping_sub(1); // 4294967295: yes; 0..254: no
    y >>= 31; // 1: yes; 0: no
    y as i32
}

#[inline]
fn negative(b: i8) -> u8 {
    let mut x = i64::from(b) as u64;
    x >>= 63; // 1: yes; 0: no
    x as u8
}

impl GePrecomp {
    fn zero() -> GePrecomp {
        GePrecomp {
            y_plus_x: FE_ONE,
            y_minus_x: FE_ONE,
            xy2d: FE_ZERO,
        }
    }

    pub fn maybe_set(&mut self, other: &GePrecomp, do_swap: i32) {
        self.y_plus_x.maybe_set(&other.y_plus_x, do_swap);
        self.y_minus_x.maybe_set(&other.y_minus_x, do_swap);
        self.xy2d.maybe_set(&other.xy2d, do_swap);
    }

    pub fn select(pos: usize, b: i8) -> GePrecomp {
        let bnegative: u8 = negative(b);
        let babs: u8 = (b - (((-(bnegative as i8)) & b) << 1)) as u8;
        let mut t = GePrecomp::zero();
        t.maybe_set(&GE_PRECOMP_BASE[pos][0], equal(babs, 1));
        t.maybe_set(&GE_PRECOMP_BASE[pos][1], equal(babs, 2));
        t.maybe_set(&GE_PRECOMP_BASE[pos][2], equal(babs, 3));
        t.maybe_set(&GE_PRECOMP_BASE[pos][3], equal(babs, 4));
        t.maybe_set(&GE_PRECOMP_BASE[pos][4], equal(babs, 5));
        t.maybe_set(&GE_PRECOMP_BASE[pos][5], equal(babs, 6));
        t.maybe_set(&GE_PRECOMP_BASE[pos][6], equal(babs, 7));
        t.maybe_set(&GE_PRECOMP_BASE[pos][7], equal(babs, 8));
        let minus_t = GePrecomp {
            y_plus_x: t.y_minus_x,
            y_minus_x: t.y_plus_x,
            xy2d: t.xy2d.neg(),
        };
        t.maybe_set(&minus_t, i32::from(bnegative));
        t
    }
}

// h = a * B
// where a = a[0]+256*a[1]+...+256^31 a[31]
// B is the Ed25519 base point (x,4/5) with x positive.
//
// Preconditions:
//   a[31] <= 127
#[doc(hidden)]
pub fn ge_scalarmult_base(a: &[u8]) -> GeP3 {
    let mut es: [i8; 64] = [0; 64];
    let mut r: GeP1P1;
    let mut s: GeP2;
    let mut t: GePrecomp;

    for i in 0..32 {
        es[2 * i] = (a[i] & 15) as i8;
        es[2 * i + 1] = ((a[i] >> 4) & 15) as i8;
    }
    // each es[i] is between 0 and 15
    // es[63] is between 0 and 7

    let mut carry: i8 = 0;
    for i in es.iter_mut().take(63) {
        *i += carry;
        carry = *i + 8;
        carry >>= 4;
        *i -= carry << 4;
    }
    es[63] += carry;
    // each es[i] is between -8 and 8

    let mut h = GeP3::zero();
    for i in (1..64).step_by(2) {
        t = GePrecomp::select(i / 2, es[i]);
        r = h + t;
        h = r.to_p3();
    }

    r = h.dbl();
    s = r.to_p2();
    r = s.dbl();
    s = r.to_p2();
    r = s.dbl();
    s = r.to_p2();
    r = s.dbl();
    h = r.to_p3();

    for i in (0..64).step_by(2) {
        t = GePrecomp::select(i / 2, es[i]);
        r = h + t;
        h = r.to_p3();
    }

    h
}
// Input:
//     s[0]+256*s[1]+...+256^63*s[63] = s
//
// Output:
//     s[0]+256*s[1]+...+256^31*s[31] = s mod l
//     where l = 2^252 + `27742317777372353535851937790883648493`.
//     Overwrites s in place.
#[doc(hidden)]
pub fn sc_reduce(s: &mut [u8]) {
    let mut s0: i64 = 2_097_151 & load_3i(s);
    let mut s1: i64 = 2_097_151 & (load_4i(&s[2..6]) >> 5);
    let mut s2: i64 = 2_097_151 & (load_3i(&s[5..8]) >> 2);
    let mut s3: i64 = 2_097_151 & (load_4i(&s[7..11]) >> 7);
    let mut s4: i64 = 2_097_151 & (load_4i(&s[10..14]) >> 4);
    let mut s5: i64 = 2_097_151 & (load_3i(&s[13..16]) >> 1);
    let mut s6: i64 = 2_097_151 & (load_4i(&s[15..19]) >> 6);
    let mut s7: i64 = 2_097_151 & (load_3i(&s[18..21]) >> 3);
    let mut s8: i64 = 2_097_151 & load_3i(&s[21..24]);
    let mut s9: i64 = 2_097_151 & (load_4i(&s[23..27]) >> 5);
    let mut s10: i64 = 2_097_151 & (load_3i(&s[26..29]) >> 2);
    let mut s11: i64 = 2_097_151 & (load_4i(&s[28..32]) >> 7);
    let mut s12: i64 = 2_097_151 & (load_4i(&s[31..35]) >> 4);
    let mut s13: i64 = 2_097_151 & (load_3i(&s[34..37]) >> 1);
    let mut s14: i64 = 2_097_151 & (load_4i(&s[36..40]) >> 6);
    let mut s15: i64 = 2_097_151 & (load_3i(&s[39..42]) >> 3);
    let mut s16: i64 = 2_097_151 & load_3i(&s[42..45]);
    let mut s17: i64 = 2_097_151 & (load_4i(&s[44..48]) >> 5);
    let s18: i64 = 2_097_151 & (load_3i(&s[47..50]) >> 2);
    let s19: i64 = 2_097_151 & (load_4i(&s[49..53]) >> 7);
    let s20: i64 = 2_097_151 & (load_4i(&s[52..56]) >> 4);
    let s21: i64 = 2_097_151 & (load_3i(&s[55..58]) >> 1);
    let s22: i64 = 2_097_151 & (load_4i(&s[57..61]) >> 6);
    let s23: i64 = load_4i(&s[60..64]) >> 3;
    let mut carry0: i64;
    let mut carry1: i64;
    let mut carry2: i64;
    let mut carry3: i64;
    let mut carry4: i64;
    let mut carry5: i64;
    let mut carry6: i64;
    let mut carry7: i64;
    let mut carry8: i64;
    let mut carry9: i64;
    let mut carry10: i64;
    let mut carry11: i64;
    let carry12: i64;
    let carry13: i64;
    let carry14: i64;
    let carry15: i64;
    let carry16: i64;

    s11 += s23 * 666_643;
    s12 += s23 * 470_296;
    s13 += s23 * 654_183;
    s14 -= s23 * 997_805;
    s15 += s23 * 136_657;
    s16 -= s23 * 683_901;

    s10 += s22 * 666_643;
    s11 += s22 * 470_296;
    s12 += s22 * 654_183;
    s13 -= s22 * 997_805;
    s14 += s22 * 136_657;
    s15 -= s22 * 683_901;

    s9 += s21 * 666_643;
    s10 += s21 * 470_296;
    s11 += s21 * 654_183;
    s12 -= s21 * 997_805;
    s13 += s21 * 136_657;
    s14 -= s21 * 683_901;

    s8 += s20 * 666_643;
    s9 += s20 * 470_296;
    s10 += s20 * 654_183;
    s11 -= s20 * 997_805;
    s12 += s20 * 136_657;
    s13 -= s20 * 683_901;

    s7 += s19 * 666_643;
    s8 += s19 * 470_296;
    s9 += s19 * 654_183;
    s10 -= s19 * 997_805;
    s11 += s19 * 136_657;
    s12 -= s19 * 683_901;

    s6 += s18 * 666_643;
    s7 += s18 * 470_296;
    s8 += s18 * 654_183;
    s9 -= s18 * 997_805;
    s10 += s18 * 136_657;
    s11 -= s18 * 683_901;

    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry12 = (s12 + (1 << 20)) >> 21;
    s13 += carry12;
    s12 -= carry12 << 21;
    carry14 = (s14 + (1 << 20)) >> 21;
    s15 += carry14;
    s14 -= carry14 << 21;
    carry16 = (s16 + (1 << 20)) >> 21;
    s17 += carry16;
    s16 -= carry16 << 21;

    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    carry13 = (s13 + (1 << 20)) >> 21;
    s14 += carry13;
    s13 -= carry13 << 21;
    carry15 = (s15 + (1 << 20)) >> 21;
    s16 += carry15;
    s15 -= carry15 << 21;

    s5 += s17 * 666_643;
    s6 += s17 * 470_296;
    s7 += s17 * 654_183;
    s8 -= s17 * 997_805;
    s9 += s17 * 136_657;
    s10 -= s17 * 683_901;

    s4 += s16 * 666_643;
    s5 += s16 * 470_296;
    s6 += s16 * 654_183;
    s7 -= s16 * 997_805;
    s8 += s16 * 136_657;
    s9 -= s16 * 683_901;

    s3 += s15 * 666_643;
    s4 += s15 * 470_296;
    s5 += s15 * 654_183;
    s6 -= s15 * 997_805;
    s7 += s15 * 136_657;
    s8 -= s15 * 683_901;

    s2 += s14 * 666_643;
    s3 += s14 * 470_296;
    s4 += s14 * 654_183;
    s5 -= s14 * 997_805;
    s6 += s14 * 136_657;
    s7 -= s14 * 683_901;

    s1 += s13 * 666_643;
    s2 += s13 * 470_296;
    s3 += s13 * 654_183;
    s4 -= s13 * 997_805;
    s5 += s13 * 136_657;
    s6 -= s13 * 683_901;

    s0 += s12 * 666_643;
    s1 += s12 * 470_296;
    s2 += s12 * 654_183;
    s3 -= s12 * 997_805;
    s4 += s12 * 136_657;
    s5 -= s12 * 683_901;
    s12 = 0;

    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;

    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;

    s0 += s12 * 666_643;
    s1 += s12 * 470_296;
    s2 += s12 * 654_183;
    s3 -= s12 * 997_805;
    s4 += s12 * 136_657;
    s5 -= s12 * 683_901;
    s12 = 0;

    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry11 = s11 >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;

    s0 += s12 * 666_643;
    s1 += s12 * 470_296;
    s2 += s12 * 654_183;
    s3 -= s12 * 997_805;
    s4 += s12 * 136_657;
    s5 -= s12 * 683_901;

    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;

    s[0] = s0 as u8;
    s[1] = (s0 >> 8) as u8;
    s[2] = ((s0 >> 16) | (s1 << 5)) as u8;
    s[3] = (s1 >> 3) as u8;
    s[4] = (s1 >> 11) as u8;
    s[5] = ((s1 >> 19) | (s2 << 2)) as u8;
    s[6] = (s2 >> 6) as u8;
    s[7] = ((s2 >> 14) | (s3 << 7)) as u8;
    s[8] = (s3 >> 1) as u8;
    s[9] = (s3 >> 9) as u8;
    s[10] = ((s3 >> 17) | (s4 << 4)) as u8;
    s[11] = (s4 >> 4) as u8;
    s[12] = (s4 >> 12) as u8;
    s[13] = ((s4 >> 20) | (s5 << 1)) as u8;
    s[14] = (s5 >> 7) as u8;
    s[15] = ((s5 >> 15) | (s6 << 6)) as u8;
    s[16] = (s6 >> 2) as u8;
    s[17] = (s6 >> 10) as u8;
    s[18] = ((s6 >> 18) | (s7 << 3)) as u8;
    s[19] = (s7 >> 5) as u8;
    s[20] = (s7 >> 13) as u8;
    s[21] = s8 as u8;
    s[22] = (s8 >> 8) as u8;
    s[23] = ((s8 >> 16) | (s9 << 5)) as u8;
    s[24] = (s9 >> 3) as u8;
    s[25] = (s9 >> 11) as u8;
    s[26] = ((s9 >> 19) | (s10 << 2)) as u8;
    s[27] = (s10 >> 6) as u8;
    s[28] = ((s10 >> 14) | (s11 << 7)) as u8;
    s[29] = (s11 >> 1) as u8;
    s[30] = (s11 >> 9) as u8;
    s[31] = (s11 >> 17) as u8;
}

// Input:
//     a[0]+256*a[1]+...+256^31*a[31] = a
//     b[0]+256*b[1]+...+256^31*b[31] = b
//     c[0]+256*c[1]+...+256^31*c[31] = c
//
// Output:
//     s[0]+256*s[1]+...+256^31*s[31] = (ab+c) mod l
//     where l = 2^252 + 27742317777372353535851937790883648493.
#[doc(hidden)]
pub fn sc_muladd(s: &mut [u8], a: &[u8], b: &[u8], c: &[u8]) {
    let a0 = 2_097_151 & load_3i(&a[0..3]);
    let a1 = 2_097_151 & (load_4i(&a[2..6]) >> 5);
    let a2 = 2_097_151 & (load_3i(&a[5..8]) >> 2);
    let a3 = 2_097_151 & (load_4i(&a[7..11]) >> 7);
    let a4 = 2_097_151 & (load_4i(&a[10..14]) >> 4);
    let a5 = 2_097_151 & (load_3i(&a[13..16]) >> 1);
    let a6 = 2_097_151 & (load_4i(&a[15..19]) >> 6);
    let a7 = 2_097_151 & (load_3i(&a[18..21]) >> 3);
    let a8 = 2_097_151 & load_3i(&a[21..24]);
    let a9 = 2_097_151 & (load_4i(&a[23..27]) >> 5);
    let a10 = 2_097_151 & (load_3i(&a[26..29]) >> 2);
    let a11 = load_4i(&a[28..32]) >> 7;
    let b0 = 2_097_151 & load_3i(&b[0..3]);
    let b1 = 2_097_151 & (load_4i(&b[2..6]) >> 5);
    let b2 = 2_097_151 & (load_3i(&b[5..8]) >> 2);
    let b3 = 2_097_151 & (load_4i(&b[7..11]) >> 7);
    let b4 = 2_097_151 & (load_4i(&b[10..14]) >> 4);
    let b5 = 2_097_151 & (load_3i(&b[13..16]) >> 1);
    let b6 = 2_097_151 & (load_4i(&b[15..19]) >> 6);
    let b7 = 2_097_151 & (load_3i(&b[18..21]) >> 3);
    let b8 = 2_097_151 & load_3i(&b[21..24]);
    let b9 = 2_097_151 & (load_4i(&b[23..27]) >> 5);
    let b10 = 2_097_151 & (load_3i(&b[26..29]) >> 2);
    let b11 = load_4i(&b[28..32]) >> 7;
    let c0 = 2_097_151 & load_3i(&c[0..3]);
    let c1 = 2_097_151 & (load_4i(&c[2..6]) >> 5);
    let c2 = 2_097_151 & (load_3i(&c[5..8]) >> 2);
    let c3 = 2_097_151 & (load_4i(&c[7..11]) >> 7);
    let c4 = 2_097_151 & (load_4i(&c[10..14]) >> 4);
    let c5 = 2_097_151 & (load_3i(&c[13..16]) >> 1);
    let c6 = 2_097_151 & (load_4i(&c[15..19]) >> 6);
    let c7 = 2_097_151 & (load_3i(&c[18..21]) >> 3);
    let c8 = 2_097_151 & load_3i(&c[21..24]);
    let c9 = 2_097_151 & (load_4i(&c[23..27]) >> 5);
    let c10 = 2_097_151 & (load_3i(&c[26..29]) >> 2);
    let c11 = load_4i(&c[28..32]) >> 7;
    let mut s0: i64;
    let mut s1: i64;
    let mut s2: i64;
    let mut s3: i64;
    let mut s4: i64;
    let mut s5: i64;
    let mut s6: i64;
    let mut s7: i64;
    let mut s8: i64;
    let mut s9: i64;
    let mut s10: i64;
    let mut s11: i64;
    let mut s12: i64;
    let mut s13: i64;
    let mut s14: i64;
    let mut s15: i64;
    let mut s16: i64;
    let mut s17: i64;
    let mut s18: i64;
    let mut s19: i64;
    let mut s20: i64;
    let mut s21: i64;
    let mut s22: i64;
    let mut s23: i64;
    let mut carry0: i64;
    let mut carry1: i64;
    let mut carry2: i64;
    let mut carry3: i64;
    let mut carry4: i64;
    let mut carry5: i64;
    let mut carry6: i64;
    let mut carry7: i64;
    let mut carry8: i64;
    let mut carry9: i64;
    let mut carry10: i64;
    let mut carry11: i64;
    let mut carry12: i64;
    let mut carry13: i64;
    let mut carry14: i64;
    let mut carry15: i64;
    let mut carry16: i64;
    let carry17: i64;
    let carry18: i64;
    let carry19: i64;
    let carry20: i64;
    let carry21: i64;
    let carry22: i64;

    s0 = c0 + a0 * b0;
    s1 = c1 + a0 * b1 + a1 * b0;
    s2 = c2 + a0 * b2 + a1 * b1 + a2 * b0;
    s3 = c3 + a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
    s4 = c4 + a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
    s5 = c5 + a0 * b5 + a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1 + a5 * b0;
    s6 = c6
        + a0 * b6
        + a1 * b5
        + a2 * b4
        + a3 * b3
        + a4 * b2
        + a5 * b1
        + a6 * b0;
    s7 = c7
        + a0 * b7
        + a1 * b6
        + a2 * b5
        + a3 * b4
        + a4 * b3
        + a5 * b2
        + a6 * b1
        + a7 * b0;
    s8 = c8
        + a0 * b8
        + a1 * b7
        + a2 * b6
        + a3 * b5
        + a4 * b4
        + a5 * b3
        + a6 * b2
        + a7 * b1
        + a8 * b0;
    s9 = c9
        + a0 * b9
        + a1 * b8
        + a2 * b7
        + a3 * b6
        + a4 * b5
        + a5 * b4
        + a6 * b3
        + a7 * b2
        + a8 * b1
        + a9 * b0;
    s10 = c10
        + a0 * b10
        + a1 * b9
        + a2 * b8
        + a3 * b7
        + a4 * b6
        + a5 * b5
        + a6 * b4
        + a7 * b3
        + a8 * b2
        + a9 * b1
        + a10 * b0;
    s11 = c11
        + a0 * b11
        + a1 * b10
        + a2 * b9
        + a3 * b8
        + a4 * b7
        + a5 * b6
        + a6 * b5
        + a7 * b4
        + a8 * b3
        + a9 * b2
        + a10 * b1
        + a11 * b0;
    s12 = a1 * b11
        + a2 * b10
        + a3 * b9
        + a4 * b8
        + a5 * b7
        + a6 * b6
        + a7 * b5
        + a8 * b4
        + a9 * b3
        + a10 * b2
        + a11 * b1;
    s13 = a2 * b11
        + a3 * b10
        + a4 * b9
        + a5 * b8
        + a6 * b7
        + a7 * b6
        + a8 * b5
        + a9 * b4
        + a10 * b3
        + a11 * b2;
    s14 = a3 * b11
        + a4 * b10
        + a5 * b9
        + a6 * b8
        + a7 * b7
        + a8 * b6
        + a9 * b5
        + a10 * b4
        + a11 * b3;
    s15 = a4 * b11
        + a5 * b10
        + a6 * b9
        + a7 * b8
        + a8 * b7
        + a9 * b6
        + a10 * b5
        + a11 * b4;
    s16 =
        a5 * b11 + a6 * b10 + a7 * b9 + a8 * b8 + a9 * b7 + a10 * b6 + a11 * b5;
    s17 = a6 * b11 + a7 * b10 + a8 * b9 + a9 * b8 + a10 * b7 + a11 * b6;
    s18 = a7 * b11 + a8 * b10 + a9 * b9 + a10 * b8 + a11 * b7;
    s19 = a8 * b11 + a9 * b10 + a10 * b9 + a11 * b8;
    s20 = a9 * b11 + a10 * b10 + a11 * b9;
    s21 = a10 * b11 + a11 * b10;
    s22 = a11 * b11;
    s23 = 0;

    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry12 = (s12 + (1 << 20)) >> 21;
    s13 += carry12;
    s12 -= carry12 << 21;
    carry14 = (s14 + (1 << 20)) >> 21;
    s15 += carry14;
    s14 -= carry14 << 21;
    carry16 = (s16 + (1 << 20)) >> 21;
    s17 += carry16;
    s16 -= carry16 << 21;
    carry18 = (s18 + (1 << 20)) >> 21;
    s19 += carry18;
    s18 -= carry18 << 21;
    carry20 = (s20 + (1 << 20)) >> 21;
    s21 += carry20;
    s20 -= carry20 << 21;
    carry22 = (s22 + (1 << 20)) >> 21;
    s23 += carry22;
    s22 -= carry22 << 21;

    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    carry13 = (s13 + (1 << 20)) >> 21;
    s14 += carry13;
    s13 -= carry13 << 21;
    carry15 = (s15 + (1 << 20)) >> 21;
    s16 += carry15;
    s15 -= carry15 << 21;
    carry17 = (s17 + (1 << 20)) >> 21;
    s18 += carry17;
    s17 -= carry17 << 21;
    carry19 = (s19 + (1 << 20)) >> 21;
    s20 += carry19;
    s19 -= carry19 << 21;
    carry21 = (s21 + (1 << 20)) >> 21;
    s22 += carry21;
    s21 -= carry21 << 21;

    s11 += s23 * 666_643;
    s12 += s23 * 470_296;
    s13 += s23 * 654_183;
    s14 -= s23 * 997_805;
    s15 += s23 * 136_657;
    s16 -= s23 * 683_901;

    s10 += s22 * 666_643;
    s11 += s22 * 470_296;
    s12 += s22 * 654_183;
    s13 -= s22 * 997_805;
    s14 += s22 * 136_657;
    s15 -= s22 * 683_901;

    s9 += s21 * 666_643;
    s10 += s21 * 470_296;
    s11 += s21 * 654_183;
    s12 -= s21 * 997_805;
    s13 += s21 * 136_657;
    s14 -= s21 * 683_901;

    s8 += s20 * 666_643;
    s9 += s20 * 470_296;
    s10 += s20 * 654_183;
    s11 -= s20 * 997_805;
    s12 += s20 * 136_657;
    s13 -= s20 * 683_901;

    s7 += s19 * 666_643;
    s8 += s19 * 470_296;
    s9 += s19 * 654_183;
    s10 -= s19 * 997_805;
    s11 += s19 * 136_657;
    s12 -= s19 * 683_901;

    s6 += s18 * 666_643;
    s7 += s18 * 470_296;
    s8 += s18 * 654_183;
    s9 -= s18 * 997_805;
    s10 += s18 * 136_657;
    s11 -= s18 * 683_901;

    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry12 = (s12 + (1 << 20)) >> 21;
    s13 += carry12;
    s12 -= carry12 << 21;
    carry14 = (s14 + (1 << 20)) >> 21;
    s15 += carry14;
    s14 -= carry14 << 21;
    carry16 = (s16 + (1 << 20)) >> 21;
    s17 += carry16;
    s16 -= carry16 << 21;

    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;
    carry13 = (s13 + (1 << 20)) >> 21;
    s14 += carry13;
    s13 -= carry13 << 21;
    carry15 = (s15 + (1 << 20)) >> 21;
    s16 += carry15;
    s15 -= carry15 << 21;

    s5 += s17 * 666_643;
    s6 += s17 * 470_296;
    s7 += s17 * 654_183;
    s8 -= s17 * 997_805;
    s9 += s17 * 136_657;
    s10 -= s17 * 683_901;

    s4 += s16 * 666_643;
    s5 += s16 * 470_296;
    s6 += s16 * 654_183;
    s7 -= s16 * 997_805;
    s8 += s16 * 136_657;
    s9 -= s16 * 683_901;

    s3 += s15 * 666_643;
    s4 += s15 * 470_296;
    s5 += s15 * 654_183;
    s6 -= s15 * 997_805;
    s7 += s15 * 136_657;
    s8 -= s15 * 683_901;

    s2 += s14 * 666_643;
    s3 += s14 * 470_296;
    s4 += s14 * 654_183;
    s5 -= s14 * 997_805;
    s6 += s14 * 136_657;
    s7 -= s14 * 683_901;

    s1 += s13 * 666_643;
    s2 += s13 * 470_296;
    s3 += s13 * 654_183;
    s4 -= s13 * 997_805;
    s5 += s13 * 136_657;
    s6 -= s13 * 683_901;

    s0 += s12 * 666_643;
    s1 += s12 * 470_296;
    s2 += s12 * 654_183;
    s3 -= s12 * 997_805;
    s4 += s12 * 136_657;
    s5 -= s12 * 683_901;
    s12 = 0;

    carry0 = (s0 + (1 << 20)) >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry2 = (s2 + (1 << 20)) >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry4 = (s4 + (1 << 20)) >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry6 = (s6 + (1 << 20)) >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry8 = (s8 + (1 << 20)) >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry10 = (s10 + (1 << 20)) >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;

    carry1 = (s1 + (1 << 20)) >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry3 = (s3 + (1 << 20)) >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry5 = (s5 + (1 << 20)) >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry7 = (s7 + (1 << 20)) >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry9 = (s9 + (1 << 20)) >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry11 = (s11 + (1 << 20)) >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;

    s0 += s12 * 666_643;
    s1 += s12 * 470_296;
    s2 += s12 * 654_183;
    s3 -= s12 * 997_805;
    s4 += s12 * 136_657;
    s5 -= s12 * 683_901;
    s12 = 0;

    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;
    carry11 = s11 >> 21;
    s12 += carry11;
    s11 -= carry11 << 21;

    s0 += s12 * 666_643;
    s1 += s12 * 470_296;
    s2 += s12 * 654_183;
    s3 -= s12 * 997_805;
    s4 += s12 * 136_657;
    s5 -= s12 * 683_901;

    carry0 = s0 >> 21;
    s1 += carry0;
    s0 -= carry0 << 21;
    carry1 = s1 >> 21;
    s2 += carry1;
    s1 -= carry1 << 21;
    carry2 = s2 >> 21;
    s3 += carry2;
    s2 -= carry2 << 21;
    carry3 = s3 >> 21;
    s4 += carry3;
    s3 -= carry3 << 21;
    carry4 = s4 >> 21;
    s5 += carry4;
    s4 -= carry4 << 21;
    carry5 = s5 >> 21;
    s6 += carry5;
    s5 -= carry5 << 21;
    carry6 = s6 >> 21;
    s7 += carry6;
    s6 -= carry6 << 21;
    carry7 = s7 >> 21;
    s8 += carry7;
    s7 -= carry7 << 21;
    carry8 = s8 >> 21;
    s9 += carry8;
    s8 -= carry8 << 21;
    carry9 = s9 >> 21;
    s10 += carry9;
    s9 -= carry9 << 21;
    carry10 = s10 >> 21;
    s11 += carry10;
    s10 -= carry10 << 21;

    s[0] = s0 as u8;
    s[1] = (s0 >> 8) as u8;
    s[2] = ((s0 >> 16) | (s1 << 5)) as u8;
    s[3] = (s1 >> 3) as u8;
    s[4] = (s1 >> 11) as u8;
    s[5] = ((s1 >> 19) | (s2 << 2)) as u8;
    s[6] = (s2 >> 6) as u8;
    s[7] = ((s2 >> 14) | (s3 << 7)) as u8;
    s[8] = (s3 >> 1) as u8;
    s[9] = (s3 >> 9) as u8;
    s[10] = ((s3 >> 17) | (s4 << 4)) as u8;
    s[11] = (s4 >> 4) as u8;
    s[12] = (s4 >> 12) as u8;
    s[13] = ((s4 >> 20) | (s5 << 1)) as u8;
    s[14] = (s5 >> 7) as u8;
    s[15] = ((s5 >> 15) | (s6 << 6)) as u8;
    s[16] = (s6 >> 2) as u8;
    s[17] = (s6 >> 10) as u8;
    s[18] = ((s6 >> 18) | (s7 << 3)) as u8;
    s[19] = (s7 >> 5) as u8;
    s[20] = (s7 >> 13) as u8;
    s[21] = s8 as u8;
    s[22] = (s8 >> 8) as u8;
    s[23] = ((s8 >> 16) | (s9 << 5)) as u8;
    s[24] = (s9 >> 3) as u8;
    s[25] = (s9 >> 11) as u8;
    s[26] = ((s9 >> 19) | (s10 << 2)) as u8;
    s[27] = (s10 >> 6) as u8;
    s[28] = ((s10 >> 14) | (s11 << 7)) as u8;
    s[29] = (s11 >> 1) as u8;
    s[30] = (s11 >> 9) as u8;
    s[31] = (s11 >> 17) as u8;
}

/// Generate a 32-byte curve25519 key, given a 32-byte curve25519 secret key
/// and a 32-byte curve22519 public key.
///
/// If the public argument is the predefined basepoint value (9 followed by all
/// zeros), then this function will calculate a curve25519 public key.
///
/// # Example
///
/// ```rust
/// # use self::curve25519::curve25519;
///
/// let my_secretkey: [u8; 32] = [0; 32]; // Don't really use all zeros as a secret key.
/// let their_publickey: [u8; 32] = [0; 32]; // or a public key of all zeros.
/// let mut basepoint: [u8; 32] = [0; 32];
/// basepoint[0] = 9;
///
/// // Generate a 32-byte curve25519 shared secret key
/// let shared_secret = curve25519(my_secretkey, their_publickey);
///
/// // Generate a 32-byte curve25519 public key.
/// let my_publickey = curve25519(my_secretkey, basepoint);
/// ```
pub fn curve25519(secret: [u8; 32], public: [u8; 32]) -> [u8; 32] {
    let e = secret.as_ref();
    let mut x2;
    let mut z2;
    let mut x3;
    let mut z3;
    let mut swap: i32;
    let mut b: i32;
    let x1 = FieldElement::from_bytes(public.as_ref());
    x2 = FE_ONE;
    z2 = FE_ZERO;
    x3 = x1;
    z3 = FE_ONE;

    swap = 0;
    // pos starts at 254 and goes down to 0
    for pos in (0usize..255).rev() {
        b = i32::from(e[pos / 8] >> (pos & 7));
        b &= 1;
        swap ^= b;
        x2.maybe_swap_with(&mut x3, swap);
        z2.maybe_swap_with(&mut z3, swap);
        swap = b;

        let d = x3 - z3;
        let b = x2 - z2;
        let a = x2 + z2;
        let c = x3 + z3;
        let da = d * a;
        let cb = c * b;
        let bb = b.square();
        let aa = a.square();
        let t0 = da + cb;
        let t1 = da - cb;
        let x4 = aa * bb;
        let e = aa - bb;
        let t2 = t1.square();
        let t3 = e.mul_121666();
        let x5 = t0.square();
        let t4 = bb + t3;
        let z5 = x1 * t2;
        let z4 = e * t4;

        z2 = z4;
        z3 = z5;
        x2 = x4;
        x3 = x5;
    }
    x2.maybe_swap_with(&mut x3, swap);
    z2.maybe_swap_with(&mut z3, swap);

    (z2.invert() * x2).to_bytes()
}

/// Generate a 32-byte curve25519 secret key.
///
/// If you supply a random 32-byte value, that is used as the base.
/// If you don't (i.e. use None for the `rand` arg), then a random 32-byte
/// number will be generated with the best OS random number generator available.
///
/// # Example
///
/// ```rust
/// # use self::curve25519::curve25519_sk;
/// # use rand::Error as RndError;
/// # fn main() -> Result<(), RndError> {
/// // Let curve25519_sk generate the random 32-byte value.
/// let sk1 = curve25519_sk(None)?;
///
/// let myrand: [u8; 32] = [0; 32]; // Don't use all zeros as a random value!
///
/// // Give curve25519_sk a random 32-byte value.
/// let sk2 = curve25519_sk(Some(myrand))?;
/// # Ok(())
/// # }
/// ```
pub fn curve25519_sk(rand: Option<[u8; 32]>) -> Result<[u8; 32], RndError> {
    let mut buf: [u8; 32] = [0; 32];

    // Fill a 32-byte buffer with random values if necessary.
    // Otherwise, use the given 32-byte value.
    let mut rand: [u8; 32] = match rand {
        Some(r) => r,
        None => {
            let mut rng = OsRng::new()?;
            rng.fill(&mut buf);
            buf
        },
    };

    // curve25519 secret key bit manip.
    rand[0] &= 248;
    rand[31] &= 127;
    rand[31] |= 64;

    Ok(rand)
}

/// Generate a 32-byte curve25519 public key.
///
/// Calls curve25519 with the public key set to the basepoint value of 9
/// followed by all zeros.
///
/// # Example
///
/// ```rust
/// # use self::curve25519::curve25519_pk;
///
/// let mysk: [u8; 32] = [0; 32]; // Don't use all zeros as a secret key!
///
/// let my_pk = curve25519_pk(mysk);
/// ```
#[inline]
pub fn curve25519_pk(secret_key: [u8; 32]) -> [u8; 32] {
    let mut basepoint: [u8; 32] = [0; 32];
    basepoint[0] = 9;
    curve25519(secret_key, basepoint)
}

#[cfg(test)]
mod tests {
    use super::{curve25519_pk, curve25519_sk, FieldElement};

    struct CurveGen {
        which: u32,
    }

    impl CurveGen {
        fn new(seed: u32) -> CurveGen { CurveGen { which: seed } }
    }

    impl Iterator for CurveGen {
        type Item = FieldElement;

        fn next(&mut self) -> Option<FieldElement> {
            let mut e: [u8; 32] = [0; 32];
            // .map(|idx| (idx * (1289 + self.which * 761)) as u8)
            // .collect();
            for idx in e.iter_mut() {
                *idx *= (1289 + self.which * 761) as u8;
            }
            e[0] &= 248;
            e[31] &= 127;
            e[31] |= 64;
            Some(FieldElement::from_bytes(e.as_ref()))
        }
    }

    #[test]
    fn from_to_bytes_preserves() {
        for i in 0..50 {
            let mut e: [u8; 32] = [0; 32];
            // .map(|idx| (idx * (1289 + i * 761)) as u8)
            // .collect();
            for idx in e.iter_mut() {
                *idx *= (1289 + i * 761) as u8;
            }
            e[0] &= 248;
            e[31] &= 127;
            e[31] |= 64;
            let fe = FieldElement::from_bytes(e.as_ref());
            let e_preserved = fe.to_bytes();
            assert!(e == e_preserved);
        }
    }

    #[test]
    fn swap_test() {
        let mut f = FieldElement([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let mut g = FieldElement([11, 21, 31, 41, 51, 61, 71, 81, 91, 101]);
        let f_initial = f;
        let g_initial = g;
        f.maybe_swap_with(&mut g, 0);
        assert!(f == f_initial);
        assert!(g == g_initial);

        f.maybe_swap_with(&mut g, 1);
        assert!(f == g_initial);
        assert!(g == f_initial);
    }

    #[test]
    fn mul_assoc() {
        for (x, (y, z)) in CurveGen::new(1)
            .zip(CurveGen::new(2).zip(CurveGen::new(3)))
            .take(40)
        {
            assert!((x * y) * z == x * (y * z));
        }
    }

    #[test]
    fn invert_inverts() {
        for x in CurveGen::new(1).take(40) {
            assert!(x.invert().invert() == x);
        }
    }

    #[test]
    fn square_by_mul() {
        for x in CurveGen::new(1).take(40) {
            assert!(x * x == x.square());
        }
    }

    #[test]
    fn base_example() {
        let sk: [u8; 32] = [
            0x77, 0x07, 0x6d, 0x0a, 0x73, 0x18, 0xa5, 0x7d, 0x3c, 0x16, 0xc1,
            0x72, 0x51, 0xb2, 0x66, 0x45, 0xdf, 0x4c, 0x2f, 0x87, 0xeb, 0xc0,
            0x99, 0x2a, 0xb1, 0x77, 0xfb, 0xa5, 0x1d, 0xb9, 0x2c, 0x2a,
        ];
        let pk = curve25519_pk(curve25519_sk(Some(sk)).unwrap());
        let correct: [u8; 32] = [
            0x85, 0x20, 0xf0, 0x09, 0x89, 0x30, 0xa7, 0x54, 0x74, 0x8b, 0x7d,
            0xdc, 0xb4, 0x3e, 0xf7, 0x5a, 0x0d, 0xbf, 0x3a, 0x0d, 0x26, 0x38,
            0x1a, 0xf4, 0xeb, 0xa4, 0xa9, 0x8e, 0xaa, 0x9b, 0x4e, 0x6a,
        ];
        assert_eq!(pk.to_vec(), correct.to_vec());
    }
}
