use fearless_simd::{Simd, SimdBase, SimdFloat, f32x4};

#[inline(always)]
pub(crate) fn gelu(x: f32) -> f32 {
    // Tanh-form GELU (Hendrycks & Gimpel), matching how the student was trained.
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    0.5 * x * (1.0 + tanh_approx(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}

/// [7/6] Padé approximation of tanh, accurate to ~1e-6 over the clamped range.
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
