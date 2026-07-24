//! Winograd transform kernels, generated from exact Cook-Toom matrices
//! (`scripts/gen_winograd.py`). Do not edit by hand.
//!
//! Interpolation points come in +-a pairs, so the transforms factor into
//! shared even/odd halves: each point pair costs one even and one odd
//! combination plus an add/sub, roughly halving the arithmetic.
//!
//! Points per stage:
//! - `w0`: F(4,7) over {0, 1, -1, 2, -2, 1/2, -1/2, 3, -3} (plus infinity)
//! - `w1`: F(4,5) over {0, 1, -1, 2, -2, 1/2, -1/2} (plus infinity)
//! - `w2`: F(4,3) over {0, 1, -1, 2, -2} (plus infinity)

#![allow(clippy::excessive_precision)]

use fearless_simd::{Simd, SimdBase, SimdFloat, f32x4};

pub(crate) const W0_POINTS: usize = 10;
pub(crate) const W0_G: [[f32; 7]; 10] = [
    [0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        0.027777777777777776,
        0.027777777777777776,
        0.027777777777777776,
        0.027777777777777776,
        0.027777777777777776,
        0.027777777777777776,
        0.027777777777777776,
    ],
    [
        0.027777777777777776,
        -0.027777777777777776,
        0.027777777777777776,
        -0.027777777777777776,
        0.027777777777777776,
        -0.027777777777777776,
        0.027777777777777776,
    ],
    [
        -0.0022222222222222222,
        -0.0044444444444444444,
        -0.008888888888888889,
        -0.017777777777777778,
        -0.035555555555555556,
        -0.07111111111111111,
        -0.14222222222222222,
    ],
    [
        -0.0022222222222222222,
        0.0044444444444444444,
        -0.008888888888888889,
        0.017777777777777778,
        -0.035555555555555556,
        0.07111111111111111,
        -0.14222222222222222,
    ],
    [
        -0.08126984126984127,
        -0.040634920634920635,
        -0.020317460317460317,
        -0.010158730158730159,
        -0.005079365079365079,
        -0.0025396825396825397,
        -0.0012698412698412698,
    ],
    [
        -0.08126984126984127,
        0.040634920634920635,
        -0.020317460317460317,
        0.010158730158730159,
        -0.005079365079365079,
        0.0025396825396825397,
        -0.0012698412698412698,
    ],
    [
        0.00015873015873015873,
        0.0004761904761904762,
        0.0014285714285714286,
        0.004285714285714286,
        0.012857142857142857,
        0.03857142857142857,
        0.11571428571428571,
    ],
    [
        0.00015873015873015873,
        -0.0004761904761904762,
        0.0014285714285714286,
        -0.004285714285714286,
        0.012857142857142857,
        -0.03857142857142857,
        0.11571428571428571,
    ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

#[inline(always)]
pub(crate) fn input_w0<S: Simd>(simd: S, d: &[f32x4<S>; 10]) -> [f32x4<S>; 10] {
    let mut e0 = d[8];
    e0 = d[2].mul_add(f32x4::splat(simd, -9.0), e0);
    e0 = d[4].mul_add(f32x4::splat(simd, 39.25), e0);
    e0 = d[6].mul_add(f32x4::splat(simd, -13.25), e0);
    let mut o0 = d[7];
    o0 = d[1].mul_add(f32x4::splat(simd, -9.0), o0);
    o0 = d[3].mul_add(f32x4::splat(simd, 39.25), o0);
    o0 = d[5].mul_add(f32x4::splat(simd, -13.25), o0);
    let r1 = e0 + o0;
    let r2 = e0 - o0;
    let mut e1 = d[8];
    e1 = d[2].mul_add(f32x4::splat(simd, -2.25), e1);
    e1 = d[4].mul_add(f32x4::splat(simd, 11.5), e1);
    e1 = d[6].mul_add(f32x4::splat(simd, -10.25), e1);
    let mut o1 = d[1] * f32x4::splat(simd, -4.5);
    o1 = d[3].mul_add(f32x4::splat(simd, 23.0), o1);
    o1 = d[5].mul_add(f32x4::splat(simd, -20.5), o1);
    o1 = d[7].mul_add(f32x4::splat(simd, 2.0), o1);
    let r3 = e1 + o1;
    let r4 = e1 - o1;
    let mut e2 = d[8];
    e2 = d[2].mul_add(f32x4::splat(simd, -36.0), e2);
    e2 = d[4].mul_add(f32x4::splat(simd, 49.0), e2);
    e2 = d[6].mul_add(f32x4::splat(simd, -14.0), e2);
    let mut o2 = d[1] * f32x4::splat(simd, -18.0);
    o2 = d[3].mul_add(f32x4::splat(simd, 24.5), o2);
    o2 = d[5].mul_add(f32x4::splat(simd, -7.0), o2);
    o2 = d[7].mul_add(f32x4::splat(simd, 0.5), o2);
    let r5 = e2 + o2;
    let r6 = e2 - o2;
    let mut e3 = d[8] - d[2];
    e3 = d[4].mul_add(f32x4::splat(simd, 5.25), e3);
    e3 = d[6].mul_add(f32x4::splat(simd, -5.25), e3);
    let mut o3 = d[1] * f32x4::splat(simd, -3.0);
    o3 = d[3].mul_add(f32x4::splat(simd, 15.75), o3);
    o3 = d[5].mul_add(f32x4::splat(simd, -15.75), o3);
    o3 = d[7].mul_add(f32x4::splat(simd, 3.0), o3);
    let r7 = e3 + o3;
    let r8 = e3 - o3;
    let mut r0 = d[8];
    r0 = d[0].mul_add(f32x4::splat(simd, 9.0), r0);
    r0 = d[2].mul_add(f32x4::splat(simd, -48.25), r0);
    r0 = d[4].mul_add(f32x4::splat(simd, 52.5), r0);
    r0 = d[6].mul_add(f32x4::splat(simd, -14.25), r0);
    let mut r9 = d[9];
    r9 = d[1].mul_add(f32x4::splat(simd, 9.0), r9);
    r9 = d[3].mul_add(f32x4::splat(simd, -48.25), r9);
    r9 = d[5].mul_add(f32x4::splat(simd, 52.5), r9);
    r9 = d[7].mul_add(f32x4::splat(simd, -14.25), r9);
    [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9]
}

#[inline(always)]
pub(crate) fn output_w0<S: Simd>(simd: S, m: &[f32x4<S>; 10]) -> [f32x4<S>; 4] {
    let s0 = m[1] + m[2];
    let t0 = m[1] - m[2];
    let s1 = m[3] + m[4];
    let t1 = m[3] - m[4];
    let s2 = m[5] + m[6];
    let t2 = m[5] - m[6];
    let s3 = m[7] + m[8];
    let t3 = m[7] - m[8];
    let y0 = (((m[0] + s0) + s1) + s2) + s3;
    let mut y1 = t0;
    y1 = t1.mul_add(f32x4::splat(simd, 2.0), y1);
    y1 = t2.mul_add(f32x4::splat(simd, 0.5), y1);
    y1 = t3.mul_add(f32x4::splat(simd, 3.0), y1);
    let mut y2 = s0;
    y2 = s1.mul_add(f32x4::splat(simd, 4.0), y2);
    y2 = s2.mul_add(f32x4::splat(simd, 0.25), y2);
    y2 = s3.mul_add(f32x4::splat(simd, 9.0), y2);
    let mut y3 = m[9] + t0;
    y3 = t1.mul_add(f32x4::splat(simd, 8.0), y3);
    y3 = t2.mul_add(f32x4::splat(simd, 0.125), y3);
    y3 = t3.mul_add(f32x4::splat(simd, 27.0), y3);
    [y0, y1, y2, y3]
}

pub(crate) const W1_POINTS: usize = 8;
pub(crate) const W1_G: [[f32; 5]; 8] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [
        -0.2222222222222222,
        -0.2222222222222222,
        -0.2222222222222222,
        -0.2222222222222222,
        -0.2222222222222222,
    ],
    [
        -0.2222222222222222,
        0.2222222222222222,
        -0.2222222222222222,
        0.2222222222222222,
        -0.2222222222222222,
    ],
    [
        0.011111111111111112,
        0.022222222222222223,
        0.044444444444444446,
        0.08888888888888889,
        0.17777777777777778,
    ],
    [
        0.011111111111111112,
        -0.022222222222222223,
        0.044444444444444446,
        -0.08888888888888889,
        0.17777777777777778,
    ],
    [
        0.7111111111111111,
        0.35555555555555557,
        0.17777777777777778,
        0.08888888888888889,
        0.044444444444444446,
    ],
    [
        0.7111111111111111,
        -0.35555555555555557,
        0.17777777777777778,
        -0.08888888888888889,
        0.044444444444444446,
    ],
    [0.0, 0.0, 0.0, 0.0, 1.0],
];

#[inline(always)]
pub(crate) fn input_w1<S: Simd>(simd: S, d: &[f32x4<S>; 8]) -> [f32x4<S>; 8] {
    let mut e0 = d[2] + d[6];
    e0 = d[4].mul_add(f32x4::splat(simd, -4.25), e0);
    let mut o0 = d[1] + d[5];
    o0 = d[3].mul_add(f32x4::splat(simd, -4.25), o0);
    let r1 = e0 + o0;
    let r2 = e0 - o0;
    let mut e1 = d[6];
    e1 = d[2].mul_add(f32x4::splat(simd, 0.25), e1);
    e1 = d[4].mul_add(f32x4::splat(simd, -1.25), e1);
    let mut o1 = d[1] * f32x4::splat(simd, 0.5);
    o1 = d[3].mul_add(f32x4::splat(simd, -2.5), o1);
    o1 = d[5].mul_add(f32x4::splat(simd, 2.0), o1);
    let r3 = e1 + o1;
    let r4 = e1 - o1;
    let mut e2 = d[6];
    e2 = d[2].mul_add(f32x4::splat(simd, 4.0), e2);
    e2 = d[4].mul_add(f32x4::splat(simd, -5.0), e2);
    let mut o2 = d[1] * f32x4::splat(simd, 2.0);
    o2 = d[3].mul_add(f32x4::splat(simd, -2.5), o2);
    o2 = d[5].mul_add(f32x4::splat(simd, 0.5), o2);
    let r5 = e2 + o2;
    let r6 = e2 - o2;
    let mut r0 = d[0] - d[6];
    r0 = d[2].mul_add(f32x4::splat(simd, -5.25), r0);
    r0 = d[4].mul_add(f32x4::splat(simd, 5.25), r0);
    let mut r7 = d[7] - d[1];
    r7 = d[3].mul_add(f32x4::splat(simd, 5.25), r7);
    r7 = d[5].mul_add(f32x4::splat(simd, -5.25), r7);
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

#[inline(always)]
pub(crate) fn output_w1<S: Simd>(simd: S, m: &[f32x4<S>; 8]) -> [f32x4<S>; 4] {
    let s0 = m[1] + m[2];
    let t0 = m[1] - m[2];
    let s1 = m[3] + m[4];
    let t1 = m[3] - m[4];
    let s2 = m[5] + m[6];
    let t2 = m[5] - m[6];
    let y0 = ((m[0] + s0) + s1) + s2;
    let mut y1 = t0;
    y1 = t1.mul_add(f32x4::splat(simd, 2.0), y1);
    y1 = t2.mul_add(f32x4::splat(simd, 0.5), y1);
    let mut y2 = s0;
    y2 = s1.mul_add(f32x4::splat(simd, 4.0), y2);
    y2 = s2.mul_add(f32x4::splat(simd, 0.25), y2);
    let mut y3 = m[7] + t0;
    y3 = t1.mul_add(f32x4::splat(simd, 8.0), y3);
    y3 = t2.mul_add(f32x4::splat(simd, 0.125), y3);
    [y0, y1, y2, y3]
}

pub(crate) const W2_POINTS: usize = 6;
pub(crate) const W2_G: [[f32; 3]; 6] = [
    [0.25, 0.0, 0.0],
    [
        -0.16666666666666666,
        -0.16666666666666666,
        -0.16666666666666666,
    ],
    [
        -0.16666666666666666,
        0.16666666666666666,
        -0.16666666666666666,
    ],
    [
        0.041666666666666664,
        0.08333333333333333,
        0.16666666666666666,
    ],
    [
        0.041666666666666664,
        -0.08333333333333333,
        0.16666666666666666,
    ],
    [0.0, 0.0, 1.0],
];

#[inline(always)]
pub(crate) fn input_w2<S: Simd>(simd: S, d: &[f32x4<S>; 6]) -> [f32x4<S>; 6] {
    let mut e0 = d[4];
    e0 = d[2].mul_add(f32x4::splat(simd, -4.0), e0);
    let mut o0 = d[3];
    o0 = d[1].mul_add(f32x4::splat(simd, -4.0), o0);
    let r1 = e0 + o0;
    let r2 = e0 - o0;
    let e1 = d[4] - d[2];
    let mut o1 = d[1] * f32x4::splat(simd, -2.0);
    o1 = d[3].mul_add(f32x4::splat(simd, 2.0), o1);
    let r3 = e1 + o1;
    let r4 = e1 - o1;
    let mut r0 = d[4];
    r0 = d[0].mul_add(f32x4::splat(simd, 4.0), r0);
    r0 = d[2].mul_add(f32x4::splat(simd, -5.0), r0);
    let mut r5 = d[5];
    r5 = d[1].mul_add(f32x4::splat(simd, 4.0), r5);
    r5 = d[3].mul_add(f32x4::splat(simd, -5.0), r5);
    [r0, r1, r2, r3, r4, r5]
}

#[inline(always)]
pub(crate) fn output_w2<S: Simd>(simd: S, m: &[f32x4<S>; 6]) -> [f32x4<S>; 4] {
    let s0 = m[1] + m[2];
    let t0 = m[1] - m[2];
    let s1 = m[3] + m[4];
    let t1 = m[3] - m[4];
    let y0 = (m[0] + s0) + s1;
    let mut y1 = t0;
    y1 = t1.mul_add(f32x4::splat(simd, 2.0), y1);
    let mut y2 = s0;
    y2 = s1.mul_add(f32x4::splat(simd, 4.0), y2);
    let mut y3 = m[5] + t0;
    y3 = t1.mul_add(f32x4::splat(simd, 8.0), y3);
    [y0, y1, y2, y3]
}
