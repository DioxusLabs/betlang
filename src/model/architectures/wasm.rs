use super::super::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn conv0_pool_row_dense(
    model: &Model,
    tokens: &[Token; TOKEN_LENGTH],
    pooled_index: usize,
) -> QuantizedVector<CONV0> {
    use std::arch::wasm32::*;

    let position_start = pooled_index * CONV0_POOL;
    let pad = CONV0_KERNEL / 2;
    let quarter = f32x4_splat(0.25);
    let mut pooled_row = [0.0; CONV0];
    let bias_ptr = model.conv0_bias.as_ptr();
    let lookup_ptr = model.conv0_lookup.as_ptr();

    unsafe {
        let mut channel = 0;
        while channel + 16 <= CONV0 {
            let mut pooled0 = f32x4_splat(0.0);
            let mut pooled1 = f32x4_splat(0.0);
            let mut pooled2 = f32x4_splat(0.0);
            let mut pooled3 = f32x4_splat(0.0);

            for position in position_start..position_start + CONV0_POOL {
                let mut row0 = v128_load(bias_ptr.add(channel).cast());
                let mut row1 = v128_load(bias_ptr.add(channel + 4).cast());
                let mut row2 = v128_load(bias_ptr.add(channel + 8).cast());
                let mut row3 = v128_load(bias_ptr.add(channel + 12).cast());
                let source_start = position - pad;

                for kernel_position in 0..CONV0_KERNEL {
                    let token = tokens[source_start + kernel_position] as usize;
                    let lookup_start =
                        (kernel_position * TOKEN_VOCAB_SIZE + token) * CONV0 + channel;
                    row0 = f32x4_add(row0, v128_load(lookup_ptr.add(lookup_start).cast()));
                    row1 = f32x4_add(row1, v128_load(lookup_ptr.add(lookup_start + 4).cast()));
                    row2 = f32x4_add(row2, v128_load(lookup_ptr.add(lookup_start + 8).cast()));
                    row3 = f32x4_add(row3, v128_load(lookup_ptr.add(lookup_start + 12).cast()));
                }

                pooled0 = f32x4_add(pooled0, f32x4_mul(gelu_f32x4(row0), quarter));
                pooled1 = f32x4_add(pooled1, f32x4_mul(gelu_f32x4(row1), quarter));
                pooled2 = f32x4_add(pooled2, f32x4_mul(gelu_f32x4(row2), quarter));
                pooled3 = f32x4_add(pooled3, f32x4_mul(gelu_f32x4(row3), quarter));
            }

            v128_store(pooled_row.as_mut_ptr().add(channel).cast(), pooled0);
            v128_store(pooled_row.as_mut_ptr().add(channel + 4).cast(), pooled1);
            v128_store(pooled_row.as_mut_ptr().add(channel + 8).cast(), pooled2);
            v128_store(pooled_row.as_mut_ptr().add(channel + 12).cast(), pooled3);
            channel += 16;
        }
    }

    quantize_array(&pooled_row)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(super) fn accumulate_conv1_row(pooled: &mut [f32; POOLED], row: [f32; CONV1], count: usize) {
    use std::arch::wasm32::*;

    let count = f32x4_splat(count as f32);
    let mut out_channel = 0;
    while out_channel + 16 <= CONV1 {
        unsafe {
            let values0 = gelu_f32x4(v128_load(row.as_ptr().add(out_channel).cast()));
            let values1 = gelu_f32x4(v128_load(row.as_ptr().add(out_channel + 4).cast()));
            let values2 = gelu_f32x4(v128_load(row.as_ptr().add(out_channel + 8).cast()));
            let values3 = gelu_f32x4(v128_load(row.as_ptr().add(out_channel + 12).cast()));

            let max0 = v128_load(pooled.as_ptr().add(out_channel).cast());
            let max1 = v128_load(pooled.as_ptr().add(out_channel + 4).cast());
            let max2 = v128_load(pooled.as_ptr().add(out_channel + 8).cast());
            let max3 = v128_load(pooled.as_ptr().add(out_channel + 12).cast());
            v128_store(
                pooled.as_mut_ptr().add(out_channel).cast(),
                f32x4_max(max0, values0),
            );
            v128_store(
                pooled.as_mut_ptr().add(out_channel + 4).cast(),
                f32x4_max(max1, values1),
            );
            v128_store(
                pooled.as_mut_ptr().add(out_channel + 8).cast(),
                f32x4_max(max2, values2),
            );
            v128_store(
                pooled.as_mut_ptr().add(out_channel + 12).cast(),
                f32x4_max(max3, values3),
            );

            let sum0 = v128_load(pooled.as_ptr().add(CONV1 + out_channel).cast());
            let sum1 = v128_load(pooled.as_ptr().add(CONV1 + out_channel + 4).cast());
            let sum2 = v128_load(pooled.as_ptr().add(CONV1 + out_channel + 8).cast());
            let sum3 = v128_load(pooled.as_ptr().add(CONV1 + out_channel + 12).cast());
            v128_store(
                pooled.as_mut_ptr().add(CONV1 + out_channel).cast(),
                f32x4_add(sum0, f32x4_mul(values0, count)),
            );
            v128_store(
                pooled.as_mut_ptr().add(CONV1 + out_channel + 4).cast(),
                f32x4_add(sum1, f32x4_mul(values1, count)),
            );
            v128_store(
                pooled.as_mut_ptr().add(CONV1 + out_channel + 8).cast(),
                f32x4_add(sum2, f32x4_mul(values2, count)),
            );
            v128_store(
                pooled.as_mut_ptr().add(CONV1 + out_channel + 12).cast(),
                f32x4_add(sum3, f32x4_mul(values3, count)),
            );
        }
        out_channel += 16;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(super) fn add_gelu_quarter<const N: usize>(target: &mut [f32; N], source: &[f32; N]) {
    use std::arch::wasm32::*;

    let quarter = f32x4_splat(0.25);
    let mut index = 0;
    while index + 16 <= N {
        unsafe {
            let target0 = v128_load(target.as_ptr().add(index).cast());
            let target1 = v128_load(target.as_ptr().add(index + 4).cast());
            let target2 = v128_load(target.as_ptr().add(index + 8).cast());
            let target3 = v128_load(target.as_ptr().add(index + 12).cast());
            let values0 = f32x4_mul(
                gelu_f32x4(v128_load(source.as_ptr().add(index).cast())),
                quarter,
            );
            let values1 = f32x4_mul(
                gelu_f32x4(v128_load(source.as_ptr().add(index + 4).cast())),
                quarter,
            );
            let values2 = f32x4_mul(
                gelu_f32x4(v128_load(source.as_ptr().add(index + 8).cast())),
                quarter,
            );
            let values3 = f32x4_mul(
                gelu_f32x4(v128_load(source.as_ptr().add(index + 12).cast())),
                quarter,
            );
            v128_store(
                target.as_mut_ptr().add(index).cast(),
                f32x4_add(target0, values0),
            );
            v128_store(
                target.as_mut_ptr().add(index + 4).cast(),
                f32x4_add(target1, values1),
            );
            v128_store(
                target.as_mut_ptr().add(index + 8).cast(),
                f32x4_add(target2, values2),
            );
            v128_store(
                target.as_mut_ptr().add(index + 12).cast(),
                f32x4_add(target3, values3),
            );
        }
        index += 16;
    }
    while index < N {
        target[index] += gelu(source[index]) * 0.25;
        index += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(super) fn add_assign<const N: usize>(target: &mut [f32; N], source: &[f32]) {
    use std::arch::wasm32::*;

    debug_assert_eq!(source.len(), N);

    let mut index = 0;
    while index + 16 <= N {
        unsafe {
            let target0 = v128_load(target.as_ptr().add(index).cast());
            let target1 = v128_load(target.as_ptr().add(index + 4).cast());
            let target2 = v128_load(target.as_ptr().add(index + 8).cast());
            let target3 = v128_load(target.as_ptr().add(index + 12).cast());
            let source0 = v128_load(source.as_ptr().add(index).cast());
            let source1 = v128_load(source.as_ptr().add(index + 4).cast());
            let source2 = v128_load(source.as_ptr().add(index + 8).cast());
            let source3 = v128_load(source.as_ptr().add(index + 12).cast());
            v128_store(
                target.as_mut_ptr().add(index).cast(),
                f32x4_add(target0, source0),
            );
            v128_store(
                target.as_mut_ptr().add(index + 4).cast(),
                f32x4_add(target1, source1),
            );
            v128_store(
                target.as_mut_ptr().add(index + 8).cast(),
                f32x4_add(target2, source2),
            );
            v128_store(
                target.as_mut_ptr().add(index + 12).cast(),
                f32x4_add(target3, source3),
            );
        }
        index += 16;
    }
    while index < N {
        target[index] += source[index];
        index += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn add_quantized_conv1_row(
    output: &mut [f32; CONV1],
    input: &QuantizedVector<CONV0>,
    kernel: &I8Matrix,
    kernel_position: usize,
) {
    use std::arch::wasm32::*;

    let input_offset = kernel_position * CONV0;
    let input_ptr = input.values.as_ptr();
    let weights_ptr = kernel.data.as_ptr();
    let scale = input.scale * kernel.scale;

    unsafe {
        let mut out_channel = 0;
        while out_channel + 4 <= CONV1 {
            let row0 = out_channel * kernel.input_len + input_offset;
            let row1 = row0 + kernel.input_len;
            let row2 = row1 + kernel.input_len;
            let row3 = row2 + kernel.input_len;
            let mut acc0 = i32x4_splat(0);
            let mut acc1 = i32x4_splat(0);
            let mut acc2 = i32x4_splat(0);
            let mut acc3 = i32x4_splat(0);

            let mut chunk = 0;
            while chunk < CONV0 {
                let input_values = v128_load(input_ptr.add(chunk).cast());
                acc0 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row0 + chunk).cast()),
                    acc0,
                );
                acc1 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row1 + chunk).cast()),
                    acc1,
                );
                acc2 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row2 + chunk).cast()),
                    acc2,
                );
                acc3 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row3 + chunk).cast()),
                    acc3,
                );
                chunk += 16;
            }

            output[out_channel] += horizontal_sum_i32x4(acc0) as f32 * scale;
            output[out_channel + 1] += horizontal_sum_i32x4(acc1) as f32 * scale;
            output[out_channel + 2] += horizontal_sum_i32x4(acc2) as f32 * scale;
            output[out_channel + 3] += horizontal_sum_i32x4(acc3) as f32 * scale;
            out_channel += 4;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn dense_quantized_array<const IN: usize, const OUT: usize>(
    input: &[f32; IN],
    kernel: &I8Matrix,
    bias: &[f32; OUT],
) -> [f32; OUT] {
    use std::arch::wasm32::*;

    let input = quantize_array(input);
    let input_ptr = input.values.as_ptr();
    let weights_ptr = kernel.data.as_ptr();
    let scale = input.scale * kernel.scale;
    let mut output = *bias;

    unsafe {
        let mut out_channel = 0;
        while out_channel + 4 <= OUT {
            let row0 = out_channel * kernel.input_len;
            let row1 = row0 + kernel.input_len;
            let row2 = row1 + kernel.input_len;
            let row3 = row2 + kernel.input_len;
            let mut acc0 = i32x4_splat(0);
            let mut acc1 = i32x4_splat(0);
            let mut acc2 = i32x4_splat(0);
            let mut acc3 = i32x4_splat(0);

            let mut chunk = 0;
            while chunk + 16 <= IN {
                let input_values = v128_load(input_ptr.add(chunk).cast());
                acc0 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row0 + chunk).cast()),
                    acc0,
                );
                acc1 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row1 + chunk).cast()),
                    acc1,
                );
                acc2 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row2 + chunk).cast()),
                    acc2,
                );
                acc3 = dot_i8x16_add(
                    input_values,
                    v128_load(weights_ptr.add(row3 + chunk).cast()),
                    acc3,
                );
                chunk += 16;
            }

            output[out_channel] += horizontal_sum_i32x4(acc0) as f32 * scale;
            output[out_channel + 1] += horizontal_sum_i32x4(acc1) as f32 * scale;
            output[out_channel + 2] += horizontal_sum_i32x4(acc2) as f32 * scale;
            output[out_channel + 3] += horizontal_sum_i32x4(acc3) as f32 * scale;

            while chunk < IN {
                output[out_channel] +=
                    input.values[chunk] as f32 * kernel.data[row0 + chunk] as f32 * scale;
                output[out_channel + 1] +=
                    input.values[chunk] as f32 * kernel.data[row1 + chunk] as f32 * scale;
                output[out_channel + 2] +=
                    input.values[chunk] as f32 * kernel.data[row2 + chunk] as f32 * scale;
                output[out_channel + 3] +=
                    input.values[chunk] as f32 * kernel.data[row3 + chunk] as f32 * scale;
                chunk += 1;
            }

            out_channel += 4;
        }

        while out_channel < OUT {
            let row = out_channel * kernel.input_len;
            let mut acc = i32x4_splat(0);
            let mut chunk = 0;
            while chunk + 16 <= IN {
                acc = dot_i8x16_add(
                    v128_load(input_ptr.add(chunk).cast()),
                    v128_load(weights_ptr.add(row + chunk).cast()),
                    acc,
                );
                chunk += 16;
            }
            let mut sum = horizontal_sum_i32x4(acc);
            while chunk < IN {
                sum += input.values[chunk] as i32 * kernel.data[row + chunk] as i32;
                chunk += 1;
            }
            output[out_channel] += sum as f32 * scale;
            out_channel += 1;
        }
    }

    output
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(super) fn quantize_values<const N: usize>(input: &[f32; N], inv_scale: f32) -> [i8; N] {
    use std::arch::wasm32::*;

    let mut values = [0; N];
    let scale = f32x4_splat(inv_scale);
    let half = f32x4_splat(0.5);
    let negative_half = f32x4_splat(-0.5);
    let zero = f32x4_splat(0.0);
    let min = f32x4_splat(-127.0);
    let max = f32x4_splat(127.0);
    let mut index = 0;
    while index + 16 <= N {
        unsafe {
            let values0 = f32x4_mul(v128_load(input.as_ptr().add(index).cast()), scale);
            let values1 = f32x4_mul(v128_load(input.as_ptr().add(index + 4).cast()), scale);
            let values2 = f32x4_mul(v128_load(input.as_ptr().add(index + 8).cast()), scale);
            let values3 = f32x4_mul(v128_load(input.as_ptr().add(index + 12).cast()), scale);
            let rounded0 = round_i32x4_away(values0, half, negative_half, zero, min, max);
            let rounded1 = round_i32x4_away(values1, half, negative_half, zero, min, max);
            let rounded2 = round_i32x4_away(values2, half, negative_half, zero, min, max);
            let rounded3 = round_i32x4_away(values3, half, negative_half, zero, min, max);
            let packed0 = i16x8_narrow_i32x4(rounded0, rounded1);
            let packed1 = i16x8_narrow_i32x4(rounded2, rounded3);
            v128_store(
                values.as_mut_ptr().add(index).cast(),
                i8x16_narrow_i16x8(packed0, packed1),
            );
        }
        index += 16;
    }
    while index < N {
        values[index] = quantize_i8(input[index], inv_scale);
        index += 1;
    }
    values
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(super) fn max_abs_array<const N: usize>(input: &[f32; N]) -> f32 {
    use std::arch::wasm32::*;

    let mut max_values = f32x4_splat(0.0);
    let mut index = 0;
    while index + 4 <= N {
        unsafe {
            max_values = f32x4_max(
                max_values,
                f32x4_abs(v128_load(input.as_ptr().add(index).cast())),
            );
        }
        index += 4;
    }
    let mut max_abs = f32x4_extract_lane::<0>(max_values)
        .max(f32x4_extract_lane::<1>(max_values))
        .max(f32x4_extract_lane::<2>(max_values))
        .max(f32x4_extract_lane::<3>(max_values));
    while index < N {
        max_abs = max_abs.max(input[index].abs());
        index += 1;
    }
    max_abs
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn round_i32x4_away(
    values: std::arch::wasm32::v128,
    half: std::arch::wasm32::v128,
    negative_half: std::arch::wasm32::v128,
    zero: std::arch::wasm32::v128,
    min: std::arch::wasm32::v128,
    max: std::arch::wasm32::v128,
) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;

    let offset = v128_bitselect(half, negative_half, f32x4_ge(values, zero));
    let rounded = f32x4_min(f32x4_max(f32x4_add(values, offset), min), max);
    i32x4_trunc_sat_f32x4(rounded)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn dot_i8x16_add(
    input_values: std::arch::wasm32::v128,
    weights: std::arch::wasm32::v128,
    acc: std::arch::wasm32::v128,
) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;

    #[cfg(target_feature = "relaxed-simd")]
    {
        i32x4_relaxed_dot_i8x16_i7x16_add(input_values, weights, acc)
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        let low = i16x8_extmul_low_i8x16(input_values, weights);
        let high = i16x8_extmul_high_i8x16(input_values, weights);
        i32x4_add(
            i32x4_add(acc, i32x4_extadd_pairwise_i16x8(low)),
            i32x4_extadd_pairwise_i16x8(high),
        )
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn horizontal_sum_i32x4(values: std::arch::wasm32::v128) -> i32 {
    use std::arch::wasm32::*;

    i32x4_extract_lane::<0>(values)
        + i32x4_extract_lane::<1>(values)
        + i32x4_extract_lane::<2>(values)
        + i32x4_extract_lane::<3>(values)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(super) fn gelu_slice(values: &mut [f32]) {
    use std::arch::wasm32::*;

    let mut index = 0;
    while index + 4 <= values.len() {
        unsafe {
            let value = v128_load(values.as_ptr().add(index).cast());
            v128_store(values.as_mut_ptr().add(index).cast(), gelu_f32x4(value));
        }
        index += 4;
    }
    while index < values.len() {
        values[index] = gelu(values[index]);
        index += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn gelu_f32x4(value: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;

    let cubic = f32x4_mul(f32x4_mul(value, value), value);
    let inner = f32x4_mul(
        f32x4_splat(0.797_884_6),
        f32x4_add(value, f32x4_mul(f32x4_splat(0.044_715), cubic)),
    );
    let clamped = f32x4_min(f32x4_max(inner, f32x4_splat(-3.0)), f32x4_splat(3.0));
    let squared = f32x4_mul(clamped, clamped);
    let tanh = f32x4_div(
        f32x4_mul(clamped, f32x4_add(f32x4_splat(27.0), squared)),
        f32x4_add(f32x4_splat(27.0), f32x4_mul(f32x4_splat(9.0), squared)),
    );
    f32x4_mul(
        f32x4_mul(f32x4_splat(0.5), value),
        f32x4_add(f32x4_splat(1.0), tanh),
    )
}
