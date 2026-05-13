use super::super::*;

#[inline(always)]
pub(super) fn conv0_pool_row_dense(
    model: &Model,
    tokens: &[Token; TOKEN_LENGTH],
    pooled_index: usize,
) -> QuantizedVector<CONV0> {
    let position_start = pooled_index * CONV0_POOL;
    let pad = CONV0_KERNEL / 2;
    let mut pooled_row = [0.0; CONV0];
    for position in position_start..position_start + CONV0_POOL {
        let mut row = model.conv0_bias;
        let source_start = position - pad;
        for kernel_position in 0..CONV0_KERNEL {
            let token = tokens[source_start + kernel_position] as usize;
            let lookup_start = (kernel_position * TOKEN_VOCAB_SIZE + token) * CONV0;
            let lookup = &model.conv0_lookup[lookup_start..lookup_start + CONV0];
            super::add_assign::<CONV0>(&mut row, lookup);
        }

        super::add_gelu_quarter::<CONV0>(&mut pooled_row, &row);
    }

    quantize_array(&pooled_row)
}

#[inline(always)]
pub(super) fn accumulate_conv1_row(pooled: &mut [f32; POOLED], row: [f32; CONV1], count: usize) {
    let count = count as f32;
    for out_channel in 0..CONV1 {
        let value = gelu(row[out_channel]);
        pooled[out_channel] = pooled[out_channel].max(value);
        pooled[CONV1 + out_channel] += value * count;
    }
}

#[inline(always)]
pub(super) fn add_gelu_quarter<const N: usize>(target: &mut [f32; N], source: &[f32; N]) {
    for index in 0..N {
        target[index] += gelu(source[index]) * 0.25;
    }
}

#[inline(always)]
pub(super) fn add_assign<const N: usize>(target: &mut [f32; N], source: &[f32]) {
    debug_assert_eq!(source.len(), N);

    let mut index = 0;
    while index + 4 <= N {
        target[index] += source[index];
        target[index + 1] += source[index + 1];
        target[index + 2] += source[index + 2];
        target[index + 3] += source[index + 3];
        index += 4;
    }
    while index < N {
        target[index] += source[index];
        index += 1;
    }
}

#[inline(always)]
pub(super) fn quantize_values<const N: usize>(input: &[f32; N], inv_scale: f32) -> [i8; N] {
    std::array::from_fn(|index| quantize_i8(input[index], inv_scale))
}

#[inline(always)]
pub(super) fn max_abs_array<const N: usize>(input: &[f32; N]) -> f32 {
    let mut max_abs = 0.0f32;
    for &value in input {
        max_abs = max_abs.max(value.abs());
    }
    max_abs
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon", not(miri)),
    all(target_arch = "wasm32", target_feature = "relaxed-simd", not(miri)),
    all(target_arch = "wasm32", target_feature = "simd128", not(miri))
)))]
#[inline(always)]
pub(super) fn dot_i8(left: &[i8], right: &[i8]) -> i32 {
    let mut sum = 0;
    let mut index = 0;
    while index < left.len() {
        sum += left[index] as i32 * right[index] as i32;
        index += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn gelu_slice(values: &mut [f32]) {
    for value in values {
        *value = gelu(*value);
    }
}
