use fearless_simd::{Simd, SimdBase, SimdFloat, f32x4};

#[inline(always)]
pub(crate) fn gelu(x: f32) -> f32 {
    // Tanh-form GELU (Hendrycks & Gimpel), matching how the student was trained.
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    0.5 * x * (1.0 + tanh_approx(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}

/// [7/6] Padé approximation of tanh, accurate to ~1e-4 over the clamped range.
/// Replaces the libm `tanh` call so the GELU step auto-vectorizes (no function call).
#[inline(always)]
fn tanh_approx(x: f32) -> f32 {
    let x = x.clamp(-5.0, 5.0);
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    num / den
}

#[inline(always)]
pub(crate) fn gelu_simd<S: Simd>(simd: S, x: f32x4<S>) -> f32x4<S> {
    let half = f32x4::splat(simd, 0.5);
    let one = f32x4::splat(simd, 1.0);
    let c1 = f32x4::splat(simd, 0.797_884_6);
    let c2 = f32x4::splat(simd, 0.044_715);
    let x2 = x * x;
    let x3 = x * x2;
    let inner = c1 * (x + c2 * x3);
    let t = tanh_approx_simd(simd, inner);
    half * x * (one + t)
}

/// SIMD [7/6] Padé tanh approximation matching `tanh_approx`.
#[inline(always)]
fn tanh_approx_simd<S: Simd>(simd: S, x: f32x4<S>) -> f32x4<S> {
    let neg5 = f32x4::splat(simd, -5.0);
    let pos5 = f32x4::splat(simd, 5.0);
    let x = x.max(neg5).min(pos5);
    let x2 = x * x;
    let c28 = f32x4::splat(simd, 28.0);
    let c378 = f32x4::splat(simd, 378.0);
    let c3150 = f32x4::splat(simd, 3150.0);
    let c17325 = f32x4::splat(simd, 17325.0);
    let c62370 = f32x4::splat(simd, 62370.0);
    let c135135 = f32x4::splat(simd, 135135.0);
    let num = x * (c135135 + x2 * (c17325 + x2 * (c378 + x2)));
    let den = c135135 + x2 * (c62370 + x2 * (c3150 + c28 * x2));
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::{Level, Simd, SimdBase, dispatch, f32x4};

    #[test]
    fn tanh_approx_tracks_reference_over_clamped_range() {
        for i in 0..=240 {
            let x = -6.0 + i as f32 * 12.0 / 240.0;
            let expected = x.clamp(-5.0, 5.0).tanh();
            let actual = tanh_approx(x);

            assert!(
                (actual - expected).abs() <= 1.25e-4,
                "x={x}, actual={actual}, expected={expected}"
            );
            assert!(actual.is_finite(), "x={x}, actual={actual}");
        }
    }

    #[test]
    fn tanh_approx_is_odd_and_clamps_inputs() {
        for x in [0.0, 0.25, 1.0, 3.0, 5.0, 6.0, 32.0] {
            let positive = tanh_approx(x);
            let negative = tanh_approx(-x);
            assert!(
                (positive + negative).abs() <= f32::EPSILON,
                "x={x}, positive={positive}, negative={negative}"
            );
        }

        assert_eq!(tanh_approx(6.0), tanh_approx(5.0));
        assert_eq!(tanh_approx(-6.0), tanh_approx(-5.0));
    }

    #[test]
    fn gelu_tracks_tanh_form_reference() {
        for i in 0..=120 {
            let x = -6.0 + i as f32 * 12.0 / 120.0;
            let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            let actual = gelu(x);

            assert!(
                (actual - expected).abs() <= 2.5e-4,
                "x={x}, actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn simd_approximations_match_scalar_lanes() {
        let level = Level::new();
        dispatch!(level, simd => {
            assert_tanh_approx_simd_matches_scalar(simd);
            assert_gelu_simd_matches_scalar(simd);
        });
    }

    fn assert_tanh_approx_simd_matches_scalar<S: Simd>(simd: S) {
        let inputs = [
            -10.0, -6.0, -5.0, -3.0, -1.25, -0.5, 0.0, 0.5, 1.25, 3.0, 5.0, 10.0,
        ];

        let (inputs, []) = inputs.as_chunks::<4>() else {
            unreachable!();
        };
        for input in inputs {
            let mut actual = [0.0; 4];
            tanh_approx_simd(simd, f32x4::from_slice(simd, input)).store_slice(&mut actual);

            for (&x, &actual) in input.iter().zip(actual.iter()) {
                let expected = tanh_approx(x);
                assert_eq!(actual, expected, "x={x}");
            }
        }
    }

    fn assert_gelu_simd_matches_scalar<S: Simd>(simd: S) {
        let inputs = [
            -10.0, -6.0, -5.0, -3.0, -1.25, -0.5, 0.0, 0.5, 1.25, 3.0, 5.0, 10.0,
        ];

        let (inputs, []) = inputs.as_chunks::<4>() else {
            unreachable!();
        };
        for input in inputs {
            let mut actual = [0.0; 4];
            gelu_simd(simd, f32x4::from_slice(simd, input)).store_slice(&mut actual);

            for (&x, &actual) in input.iter().zip(actual.iter()) {
                let expected = gelu(x);
                assert_eq!(actual, expected, "x={x}");
            }
        }
    }
}
