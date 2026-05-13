use super::super::*;

#[target_feature(enable = "neon")]
pub(super) unsafe fn conv0_pool_row_dense(
    model: &Model,
    tokens: &[Token; TOKEN_LENGTH],
    pooled_index: usize,
) -> QuantizedVector<CONV0> {
    use std::arch::aarch64::*;

    let position_start = pooled_index * CONV0_POOL;
    let pad = CONV0_KERNEL / 2;
    let quarter = vdupq_n_f32(0.25);
    let mut pooled_row = [0.0; CONV0];
    let bias_ptr = model.conv0_bias.as_ptr();
    let lookup_ptr = model.conv0_lookup.as_ptr();

    unsafe {
        let mut channel = 0;
        while channel + 16 <= CONV0 {
            let mut pooled0 = vdupq_n_f32(0.0);
            let mut pooled1 = vdupq_n_f32(0.0);
            let mut pooled2 = vdupq_n_f32(0.0);
            let mut pooled3 = vdupq_n_f32(0.0);

            for position in position_start..position_start + CONV0_POOL {
                let mut row0 = vld1q_f32(bias_ptr.add(channel));
                let mut row1 = vld1q_f32(bias_ptr.add(channel + 4));
                let mut row2 = vld1q_f32(bias_ptr.add(channel + 8));
                let mut row3 = vld1q_f32(bias_ptr.add(channel + 12));
                let source_start = position - pad;

                for kernel_position in 0..CONV0_KERNEL {
                    let token = tokens[source_start + kernel_position] as usize;
                    let lookup_start =
                        (kernel_position * TOKEN_VOCAB_SIZE + token) * CONV0 + channel;
                    row0 = vaddq_f32(row0, vld1q_f32(lookup_ptr.add(lookup_start)));
                    row1 = vaddq_f32(row1, vld1q_f32(lookup_ptr.add(lookup_start + 4)));
                    row2 = vaddq_f32(row2, vld1q_f32(lookup_ptr.add(lookup_start + 8)));
                    row3 = vaddq_f32(row3, vld1q_f32(lookup_ptr.add(lookup_start + 12)));
                }

                pooled0 = vaddq_f32(pooled0, vmulq_f32(gelu_f32x4(row0), quarter));
                pooled1 = vaddq_f32(pooled1, vmulq_f32(gelu_f32x4(row1), quarter));
                pooled2 = vaddq_f32(pooled2, vmulq_f32(gelu_f32x4(row2), quarter));
                pooled3 = vaddq_f32(pooled3, vmulq_f32(gelu_f32x4(row3), quarter));
            }

            vst1q_f32(pooled_row.as_mut_ptr().add(channel), pooled0);
            vst1q_f32(pooled_row.as_mut_ptr().add(channel + 4), pooled1);
            vst1q_f32(pooled_row.as_mut_ptr().add(channel + 8), pooled2);
            vst1q_f32(pooled_row.as_mut_ptr().add(channel + 12), pooled3);
            channel += 16;
        }
    }

    quantize_array(&pooled_row)
}

#[inline(always)]
pub(super) fn accumulate_conv1_row(pooled: &mut [f32; POOLED], row: [f32; CONV1], count: usize) {
    use std::arch::aarch64::*;

    let count = unsafe { vdupq_n_f32(count as f32) };
    let mut out_channel = 0;
    while out_channel + 16 <= CONV1 {
        unsafe {
            let values0 = gelu_f32x4(vld1q_f32(row.as_ptr().add(out_channel)));
            let values1 = gelu_f32x4(vld1q_f32(row.as_ptr().add(out_channel + 4)));
            let values2 = gelu_f32x4(vld1q_f32(row.as_ptr().add(out_channel + 8)));
            let values3 = gelu_f32x4(vld1q_f32(row.as_ptr().add(out_channel + 12)));

            let max0 = vld1q_f32(pooled.as_ptr().add(out_channel));
            let max1 = vld1q_f32(pooled.as_ptr().add(out_channel + 4));
            let max2 = vld1q_f32(pooled.as_ptr().add(out_channel + 8));
            let max3 = vld1q_f32(pooled.as_ptr().add(out_channel + 12));
            vst1q_f32(
                pooled.as_mut_ptr().add(out_channel),
                vmaxq_f32(max0, values0),
            );
            vst1q_f32(
                pooled.as_mut_ptr().add(out_channel + 4),
                vmaxq_f32(max1, values1),
            );
            vst1q_f32(
                pooled.as_mut_ptr().add(out_channel + 8),
                vmaxq_f32(max2, values2),
            );
            vst1q_f32(
                pooled.as_mut_ptr().add(out_channel + 12),
                vmaxq_f32(max3, values3),
            );

            let sum0 = vld1q_f32(pooled.as_ptr().add(CONV1 + out_channel));
            let sum1 = vld1q_f32(pooled.as_ptr().add(CONV1 + out_channel + 4));
            let sum2 = vld1q_f32(pooled.as_ptr().add(CONV1 + out_channel + 8));
            let sum3 = vld1q_f32(pooled.as_ptr().add(CONV1 + out_channel + 12));
            vst1q_f32(
                pooled.as_mut_ptr().add(CONV1 + out_channel),
                vaddq_f32(sum0, vmulq_f32(values0, count)),
            );
            vst1q_f32(
                pooled.as_mut_ptr().add(CONV1 + out_channel + 4),
                vaddq_f32(sum1, vmulq_f32(values1, count)),
            );
            vst1q_f32(
                pooled.as_mut_ptr().add(CONV1 + out_channel + 8),
                vaddq_f32(sum2, vmulq_f32(values2, count)),
            );
            vst1q_f32(
                pooled.as_mut_ptr().add(CONV1 + out_channel + 12),
                vaddq_f32(sum3, vmulq_f32(values3, count)),
            );
        }
        out_channel += 16;
    }
    while out_channel + 4 <= CONV1 {
        unsafe {
            let values = gelu_f32x4(vld1q_f32(row.as_ptr().add(out_channel)));
            let max_values = vld1q_f32(pooled.as_ptr().add(out_channel));
            vst1q_f32(
                pooled.as_mut_ptr().add(out_channel),
                vmaxq_f32(max_values, values),
            );

            let sum_values = vld1q_f32(pooled.as_ptr().add(CONV1 + out_channel));
            vst1q_f32(
                pooled.as_mut_ptr().add(CONV1 + out_channel),
                vaddq_f32(sum_values, vmulq_f32(values, count)),
            );
        }
        out_channel += 4;
    }
}

#[inline(always)]
pub(super) fn add_gelu_quarter<const N: usize>(target: &mut [f32; N], source: &[f32; N]) {
    use std::arch::aarch64::*;

    let quarter = unsafe { vdupq_n_f32(0.25) };
    let mut index = 0;
    while index + 16 <= N {
        unsafe {
            let target0 = vld1q_f32(target.as_ptr().add(index));
            let target1 = vld1q_f32(target.as_ptr().add(index + 4));
            let target2 = vld1q_f32(target.as_ptr().add(index + 8));
            let target3 = vld1q_f32(target.as_ptr().add(index + 12));
            let values0 = vmulq_f32(gelu_f32x4(vld1q_f32(source.as_ptr().add(index))), quarter);
            let values1 = vmulq_f32(
                gelu_f32x4(vld1q_f32(source.as_ptr().add(index + 4))),
                quarter,
            );
            let values2 = vmulq_f32(
                gelu_f32x4(vld1q_f32(source.as_ptr().add(index + 8))),
                quarter,
            );
            let values3 = vmulq_f32(
                gelu_f32x4(vld1q_f32(source.as_ptr().add(index + 12))),
                quarter,
            );
            vst1q_f32(target.as_mut_ptr().add(index), vaddq_f32(target0, values0));
            vst1q_f32(
                target.as_mut_ptr().add(index + 4),
                vaddq_f32(target1, values1),
            );
            vst1q_f32(
                target.as_mut_ptr().add(index + 8),
                vaddq_f32(target2, values2),
            );
            vst1q_f32(
                target.as_mut_ptr().add(index + 12),
                vaddq_f32(target3, values3),
            );
        }
        index += 16;
    }
    while index + 4 <= N {
        unsafe {
            let target_values = vld1q_f32(target.as_ptr().add(index));
            let values = vmulq_f32(gelu_f32x4(vld1q_f32(source.as_ptr().add(index))), quarter);
            vst1q_f32(
                target.as_mut_ptr().add(index),
                vaddq_f32(target_values, values),
            );
        }
        index += 4;
    }
    while index < N {
        target[index] += gelu(source[index]) * 0.25;
        index += 1;
    }
}

#[inline(always)]
pub(super) fn add_assign<const N: usize>(target: &mut [f32; N], source: &[f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(source.len(), N);

    let mut index = 0;
    while index + 16 <= N {
        unsafe {
            let target0 = vld1q_f32(target.as_ptr().add(index));
            let target1 = vld1q_f32(target.as_ptr().add(index + 4));
            let target2 = vld1q_f32(target.as_ptr().add(index + 8));
            let target3 = vld1q_f32(target.as_ptr().add(index + 12));
            let source0 = vld1q_f32(source.as_ptr().add(index));
            let source1 = vld1q_f32(source.as_ptr().add(index + 4));
            let source2 = vld1q_f32(source.as_ptr().add(index + 8));
            let source3 = vld1q_f32(source.as_ptr().add(index + 12));
            vst1q_f32(target.as_mut_ptr().add(index), vaddq_f32(target0, source0));
            vst1q_f32(
                target.as_mut_ptr().add(index + 4),
                vaddq_f32(target1, source1),
            );
            vst1q_f32(
                target.as_mut_ptr().add(index + 8),
                vaddq_f32(target2, source2),
            );
            vst1q_f32(
                target.as_mut_ptr().add(index + 12),
                vaddq_f32(target3, source3),
            );
        }
        index += 16;
    }
    while index + 4 <= N {
        unsafe {
            let target_values = vld1q_f32(target.as_ptr().add(index));
            let source_values = vld1q_f32(source.as_ptr().add(index));
            vst1q_f32(
                target.as_mut_ptr().add(index),
                vaddq_f32(target_values, source_values),
            );
        }
        index += 4;
    }
    while index < N {
        target[index] += source[index];
        index += 1;
    }
}

pub(super) fn precompute_conv1_i8mm_kernel(conv1_kernel: &I8Matrix) -> Option<Conv1I8mmKernel> {
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return None;
    }

    debug_assert_eq!(conv1_kernel.input_len, CONV1_KERNEL * CONV0);
    debug_assert_eq!(conv1_kernel.output_len, CONV1);

    let mut data = Vec::with_capacity(CONV1_KERNEL * (CONV1 / 2) * CONV1_I8MM_CHUNKS * 16);
    for kernel_position in 0..CONV1_KERNEL {
        let input_offset = kernel_position * CONV0;
        for out_pair in 0..CONV1 / 2 {
            let out0 = out_pair * 2;
            let out1 = out0 + 1;
            for chunk in 0..CONV1_I8MM_CHUNKS {
                let chunk_offset = input_offset + chunk * 8;
                data.extend_from_slice(conv1_kernel.row(out0, chunk_offset, 8));
                data.extend_from_slice(conv1_kernel.row(out1, chunk_offset, 8));
            }
        }
    }

    Some(Conv1I8mmKernel { data })
}

#[target_feature(enable = "neon,i8mm")]
pub(super) unsafe fn add_quantized_conv1_pair_i8mm(
    output0: &mut [f32; CONV1],
    output1: &mut [f32; CONV1],
    input0: &QuantizedVector<CONV0>,
    input1: &QuantizedVector<CONV0>,
    kernel: &Conv1I8mmKernel,
    kernel_position: usize,
    kernel_scale: f32,
) {
    use std::arch::aarch64::*;
    use std::arch::asm;

    let scale0 = input0.scale * kernel_scale;
    let scale1 = input1.scale * kernel_scale;
    let input0_ptr = input0.values.as_ptr();
    let input1_ptr = input1.values.as_ptr();
    let kernel_ptr = kernel.data.as_ptr();
    let kernel_base = kernel_position * (CONV1 / 2) * CONV1_I8MM_CHUNKS * 16;

    macro_rules! smmla {
        ($acc:ident, $input_values:ident, $weights:ident) => {
            unsafe {
                asm!(
                    "smmla {acc:v}.4s, {input_values:v}.16b, {weights:v}.16b",
                    acc = inout(vreg) $acc,
                    input_values = in(vreg) $input_values,
                    weights = in(vreg) $weights,
                    options(nostack, nomem)
                );
            }
        };
    }

    let mut out_pair = 0;
    while out_pair + 8 <= CONV1 / 2 {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mut acc4 = vdupq_n_s32(0);
        let mut acc5 = vdupq_n_s32(0);
        let mut acc6 = vdupq_n_s32(0);
        let mut acc7 = vdupq_n_s32(0);
        let pair_base = kernel_base + out_pair * CONV1_I8MM_CHUNKS * 16;

        macro_rules! process_chunk {
            ($chunk:expr) => {{
                let chunk_offset = $chunk * 8;
                let chunk_base = pair_base + $chunk * 16;
                let input_values = unsafe {
                    vcombine_s8(
                        vld1_s8(input0_ptr.add(chunk_offset)),
                        vld1_s8(input1_ptr.add(chunk_offset)),
                    )
                };
                let weights0 = unsafe { vld1q_s8(kernel_ptr.add(chunk_base)) };
                let weights1 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 16)) };
                let weights2 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 32)) };
                let weights3 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 48)) };
                let weights4 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 64)) };
                let weights5 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 80)) };
                let weights6 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 96)) };
                let weights7 =
                    unsafe { vld1q_s8(kernel_ptr.add(chunk_base + CONV1_I8MM_CHUNKS * 112)) };
                smmla!(acc0, input_values, weights0);
                smmla!(acc1, input_values, weights1);
                smmla!(acc2, input_values, weights2);
                smmla!(acc3, input_values, weights3);
                smmla!(acc4, input_values, weights4);
                smmla!(acc5, input_values, weights5);
                smmla!(acc6, input_values, weights6);
                smmla!(acc7, input_values, weights7);
            }};
        }
        process_chunk!(0);
        process_chunk!(1);
        process_chunk!(2);
        process_chunk!(3);
        process_chunk!(4);
        process_chunk!(5);
        process_chunk!(6);
        process_chunk!(7);
        process_chunk!(8);
        process_chunk!(9);

        let out_channel = out_pair * 2;
        let scale0 = vdupq_n_f32(scale0);
        let scale1 = vdupq_n_f32(scale1);

        macro_rules! add_acc_pair {
            ($left:ident, $right:ident, $offset:expr) => {{
                let sums0 = vcombine_s32(vget_low_s32($left), vget_low_s32($right));
                let sums1 = vcombine_s32(vget_high_s32($left), vget_high_s32($right));
                let target0 = vld1q_f32(output0.as_ptr().add(out_channel + $offset));
                let target1 = vld1q_f32(output1.as_ptr().add(out_channel + $offset));
                vst1q_f32(
                    output0.as_mut_ptr().add(out_channel + $offset),
                    vaddq_f32(target0, vmulq_f32(vcvtq_f32_s32(sums0), scale0)),
                );
                vst1q_f32(
                    output1.as_mut_ptr().add(out_channel + $offset),
                    vaddq_f32(target1, vmulq_f32(vcvtq_f32_s32(sums1), scale1)),
                );
            }};
        }

        unsafe {
            add_acc_pair!(acc0, acc1, 0);
            add_acc_pair!(acc2, acc3, 4);
            add_acc_pair!(acc4, acc5, 8);
            add_acc_pair!(acc6, acc7, 12);
        }
        out_pair += 8;
    }
}

#[inline(always)]
pub(super) fn quantize_values<const N: usize>(input: &[f32; N], inv_scale: f32) -> [i8; N] {
    use std::arch::aarch64::*;

    let mut values = [0; N];
    let scale = unsafe { vdupq_n_f32(inv_scale) };
    let mut index = 0;
    while index + 16 <= N {
        unsafe {
            let values0 = vmulq_f32(vld1q_f32(input.as_ptr().add(index)), scale);
            let values1 = vmulq_f32(vld1q_f32(input.as_ptr().add(index + 4)), scale);
            let values2 = vmulq_f32(vld1q_f32(input.as_ptr().add(index + 8)), scale);
            let values3 = vmulq_f32(vld1q_f32(input.as_ptr().add(index + 12)), scale);
            let rounded0 = vcvtaq_s32_f32(values0);
            let rounded1 = vcvtaq_s32_f32(values1);
            let rounded2 = vcvtaq_s32_f32(values2);
            let rounded3 = vcvtaq_s32_f32(values3);
            let packed0 = vcombine_s16(vqmovn_s32(rounded0), vqmovn_s32(rounded1));
            let packed1 = vcombine_s16(vqmovn_s32(rounded2), vqmovn_s32(rounded3));
            vst1q_s8(
                values.as_mut_ptr().add(index),
                vcombine_s8(vqmovn_s16(packed0), vqmovn_s16(packed1)),
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

#[inline(always)]
pub(super) fn max_abs_array<const N: usize>(input: &[f32; N]) -> f32 {
    use std::arch::aarch64::*;

    let mut max_values = unsafe { vdupq_n_f32(0.0) };
    let mut index = 0;
    while index + 4 <= N {
        unsafe {
            let values = vabsq_f32(vld1q_f32(input.as_ptr().add(index)));
            max_values = vmaxq_f32(max_values, values);
        }
        index += 4;
    }
    let mut max_abs = unsafe { vmaxvq_f32(max_values) };
    while index < N {
        max_abs = max_abs.max(input[index].abs());
        index += 1;
    }
    max_abs
}

#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn dot_i8_dotprod(left: &[i8], right: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    use std::arch::asm;

    let mut acc = vdupq_n_s32(0);
    let mut index = 0;
    while index + 16 <= left.len() {
        let left_values = unsafe { vld1q_s8(left.as_ptr().add(index)) };
        let right_values = unsafe { vld1q_s8(right.as_ptr().add(index)) };
        unsafe {
            asm!(
                "sdot {acc:v}.4s, {left_values:v}.16b, {right_values:v}.16b",
                acc = inout(vreg) acc,
                left_values = in(vreg) left_values,
                right_values = in(vreg) right_values,
                options(nostack, nomem)
            );
        }
        index += 16;
    }

    let mut sum = vaddvq_s32(acc);
    while index < left.len() {
        sum += left[index] as i32 * right[index] as i32;
        index += 1;
    }
    sum
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn dot_i8_neon(left: &[i8], right: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_s32(0);
    let mut index = 0;
    while index + 16 <= left.len() {
        let left_values = unsafe { vld1q_s8(left.as_ptr().add(index)) };
        let right_values = unsafe { vld1q_s8(right.as_ptr().add(index)) };
        let low = vmull_s8(vget_low_s8(left_values), vget_low_s8(right_values));
        let high = vmull_s8(vget_high_s8(left_values), vget_high_s8(right_values));
        acc = vaddq_s32(acc, vpaddlq_s16(low));
        acc = vaddq_s32(acc, vpaddlq_s16(high));
        index += 16;
    }

    let mut sum = vaddvq_s32(acc);
    while index < left.len() {
        sum += left[index] as i32 * right[index] as i32;
        index += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn gelu_slice(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut index = 0;
    while index + 4 <= values.len() {
        unsafe {
            let value = vld1q_f32(values.as_ptr().add(index));
            vst1q_f32(values.as_mut_ptr().add(index), gelu_f32x4(value));
        }
        index += 4;
    }
    while index < values.len() {
        values[index] = gelu(values[index]);
        index += 1;
    }
}

#[inline(always)]
unsafe fn gelu_f32x4(value: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    unsafe {
        let cubic = vmulq_f32(vmulq_f32(value, value), value);
        let inner = vmulq_f32(
            vdupq_n_f32(0.797_884_6),
            vaddq_f32(value, vmulq_f32(vdupq_n_f32(0.044_715), cubic)),
        );
        let clamped = vminq_f32(vmaxq_f32(inner, vdupq_n_f32(-3.0)), vdupq_n_f32(3.0));
        let squared = vmulq_f32(clamped, clamped);
        let tanh = vdivq_f32(
            vmulq_f32(clamped, vaddq_f32(vdupq_n_f32(27.0), squared)),
            vaddq_f32(vdupq_n_f32(27.0), vmulq_f32(vdupq_n_f32(9.0), squared)),
        );
        vmulq_f32(
            vmulq_f32(vdupq_n_f32(0.5), value),
            vaddq_f32(vdupq_n_f32(1.0), tanh),
        )
    }
}
