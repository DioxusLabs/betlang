use super::*;

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod aarch64;
#[cfg(any(
    miri,
    not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
mod scalar;
#[cfg(all(target_arch = "wasm32", not(miri)))]
mod wasm;

#[inline(always)]
pub(super) fn conv0_pool_row_dense(
    model: &Model,
    tokens: &[Token; TOKEN_LENGTH],
    pooled_index: usize,
) -> QuantizedVector<CONV0> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        unsafe { aarch64::conv0_pool_row_dense(model, tokens, pooled_index) }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        unsafe { wasm::conv0_pool_row_dense(model, tokens, pooled_index) }
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::conv0_pool_row_dense(model, tokens, pooled_index)
    }
}

#[inline(always)]
pub(super) fn accumulate_conv1_row(pooled: &mut [f32; POOLED], row: [f32; CONV1], count: usize) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        aarch64::accumulate_conv1_row(pooled, row, count);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        wasm::accumulate_conv1_row(pooled, row, count);
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::accumulate_conv1_row(pooled, row, count);
    }
}

#[inline(always)]
pub(super) fn add_gelu_quarter<const N: usize>(target: &mut [f32; N], source: &[f32; N]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        aarch64::add_gelu_quarter(target, source);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        wasm::add_gelu_quarter(target, source);
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::add_gelu_quarter(target, source);
    }
}

#[inline(always)]
pub(super) fn add_assign<const N: usize>(target: &mut [f32; N], source: &[f32]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        aarch64::add_assign(target, source);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        wasm::add_assign(target, source);
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::add_assign(target, source);
    }
}

#[inline(always)]
pub(super) fn add_quantized_conv1_row(
    output: &mut [f32; CONV1],
    input: &QuantizedVector<CONV0>,
    kernel: &I8Matrix,
    kernel_position: usize,
) {
    debug_assert_eq!(kernel.input_len, CONV1_KERNEL * CONV0);
    debug_assert_eq!(kernel.output_len, CONV1);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        unsafe {
            wasm::add_quantized_conv1_row(output, input, kernel, kernel_position);
        }
    }

    #[cfg(any(miri, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let input_offset = kernel_position * CONV0;
        let scale = input.scale * kernel.scale;
        for (out_channel, output_value) in output.iter_mut().enumerate() {
            let weights = kernel.row(out_channel, input_offset, CONV0);
            *output_value += dot_i8(&input.values, weights) as f32 * scale;
        }
    }
}

#[inline(always)]
pub(super) fn dense_quantized_array<const IN: usize, const OUT: usize>(
    input: &[f32; IN],
    kernel: &I8Matrix,
    bias: &[f32; OUT],
) -> [f32; OUT] {
    debug_assert_eq!(kernel.input_len, IN);
    debug_assert_eq!(kernel.output_len, OUT);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        unsafe { wasm::dense_quantized_array(input, kernel, bias) }
    }

    #[cfg(any(miri, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let input = quantize_array(input);
        let scale = input.scale * kernel.scale;
        let mut output = *bias;
        for (out_channel, output_value) in output.iter_mut().enumerate() {
            let weights = kernel.row(out_channel, 0, IN);
            *output_value += dot_i8(&input.values, weights) as f32 * scale;
        }
        output
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
pub(super) fn precompute_conv1_i8mm_kernel(conv1_kernel: &I8Matrix) -> Option<Conv1I8mmKernel> {
    aarch64::precompute_conv1_i8mm_kernel(conv1_kernel)
}

#[cfg(any(miri, not(target_arch = "aarch64")))]
pub(super) fn precompute_conv1_i8mm_kernel(_conv1_kernel: &I8Matrix) -> Option<Conv1I8mmKernel> {
    None
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline(always)]
pub(super) unsafe fn add_quantized_conv1_pair_i8mm(
    output0: &mut [f32; CONV1],
    output1: &mut [f32; CONV1],
    input0: &QuantizedVector<CONV0>,
    input1: &QuantizedVector<CONV0>,
    kernel: &Conv1I8mmKernel,
    kernel_position: usize,
    kernel_scale: f32,
) {
    unsafe {
        aarch64::add_quantized_conv1_pair_i8mm(
            output0,
            output1,
            input0,
            input1,
            kernel,
            kernel_position,
            kernel_scale,
        );
    }
}

#[cfg(any(miri, not(target_arch = "aarch64")))]
#[inline(always)]
pub(super) unsafe fn add_quantized_conv1_pair_i8mm(
    _output0: &mut [f32; CONV1],
    _output1: &mut [f32; CONV1],
    _input0: &QuantizedVector<CONV0>,
    _input1: &QuantizedVector<CONV0>,
    _kernel: &Conv1I8mmKernel,
    _kernel_position: usize,
    _kernel_scale: f32,
) {
    unreachable!()
}

#[inline(always)]
pub(super) fn quantize_values<const N: usize>(input: &[f32; N], inv_scale: f32) -> [i8; N] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        aarch64::quantize_values(input, inv_scale)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        wasm::quantize_values(input, inv_scale)
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::quantize_values(input, inv_scale)
    }
}

#[inline(always)]
pub(super) fn max_abs_array<const N: usize>(input: &[f32; N]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        aarch64::max_abs_array(input)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        wasm::max_abs_array(input)
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::max_abs_array(input)
    }
}

#[cfg(any(miri, not(all(target_arch = "wasm32", target_feature = "simd128"))))]
#[inline(always)]
pub(super) fn dot_i8(left: &[i8], right: &[i8]) -> i32 {
    debug_assert_eq!(left.len(), right.len());

    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { aarch64::dot_i8_dotprod(left, right) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        unsafe { aarch64::dot_i8_neon(left, right) }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", not(miri)),
        all(target_arch = "wasm32", target_feature = "simd128", not(miri))
    )))]
    {
        scalar::dot_i8(left, right)
    }
}

#[inline(always)]
pub(super) fn gelu_slice(values: &mut [f32]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        aarch64::gelu_slice(values);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
        wasm::gelu_slice(values);
    }

    #[cfg(any(
        miri,
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    {
        scalar::gelu_slice(values);
    }
}
