use super::{
    activation::gelu,
    constants::*,
    embedded::MODEL_BYTES,
    layers::{
        Tensor, conv_gelu_global_pool_tensor, conv_gelu_maxpool_tensor, dense_forward,
        embed_position,
    },
    reader::{read_f32_array, read_int4_dequant, read_ternary_dequant},
    tokenizer::tokenize,
    window::TokenWindow,
};
use std::sync::OnceLock;

#[derive(Debug)]
pub(crate) struct Model {
    /// 1024 × 24 dequantized embedding rows.
    pub(crate) embedding: Box<[f32]>,
    /// `[k][in_c][out_c]` — inner kernel row is contiguous over out_channels.
    conv0_kernel: Box<[f32]>,
    conv0_bias: [f32; CONV0],
    conv1_kernel: Box<[f32]>,
    conv1_bias: [f32; CONV1],
    conv2_kernel: Box<[f32]>,
    conv2_bias: [f32; CONV2],
    /// (POOLED, DENSE) flattened.
    dense0_kernel: Box<[f32]>,
    dense0_bias: [f32; DENSE],
    /// (DENSE, CLASSES) flattened.
    pub(crate) output_kernel: Box<[f32]>,
    output_bias: [f32; CLASSES],
}

impl Model {
    fn load() -> Self {
        debug_assert!(MODEL_BYTES.starts_with(&MODEL_MAGIC), "bad MSQ1 magic");
        let mut cur = MODEL_MAGIC.len();
        let scales = read_f32_array::<SCALE_COUNT>(&mut cur);

        // q_hash_embedding: weights [(1024, 24)] int4
        let embedding = read_int4_dequant(&mut cur, BINS * EMBED, scales[0]);

        // q_conv_0: weights [(7, 24, 64)] ternary, bias [(64,)] f32
        let conv0_kernel = read_ternary_dequant(&mut cur, CONV0_KERNEL * EMBED * CONV0, scales[1]);
        let conv0_bias = read_f32_array::<CONV0>(&mut cur);

        // q_conv_1: weights [(5, 64, 128)] ternary, bias [(128,)] f32
        let conv1_kernel = read_ternary_dequant(&mut cur, CONV1_KERNEL * CONV0 * CONV1, scales[2]);
        let conv1_bias = read_f32_array::<CONV1>(&mut cur);

        // q_conv_2: weights [(3, 128, 128)] ternary, bias [(128,)] f32
        let conv2_kernel = read_ternary_dequant(&mut cur, CONV2_KERNEL * CONV1 * CONV2, scales[3]);
        let conv2_bias = read_f32_array::<CONV2>(&mut cur);

        // q_dense_0: weights [(256, 96)] ternary, bias [(96,)] f32
        let dense0_kernel = read_ternary_dequant(&mut cur, POOLED * DENSE, scales[4]);
        let dense0_bias = read_f32_array::<DENSE>(&mut cur);

        // q_output: weights [(96, 48)] int4, bias [(48,)] f32
        let output_kernel = read_int4_dequant(&mut cur, DENSE * CLASSES, scales[5]);
        let output_bias = read_f32_array::<CLASSES>(&mut cur);

        debug_assert_eq!(cur, MODEL_BYTES.len(), "unexpected model payload length");

        Self {
            embedding,
            conv0_kernel,
            conv0_bias,
            conv1_kernel,
            conv1_bias,
            conv2_kernel,
            conv2_bias,
            dense0_kernel,
            dense0_bias,
            output_kernel,
            output_bias,
        }
    }

    pub(crate) fn get() -> &'static Self {
        static MODEL: OnceLock<Model> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    /// Run the full forward pass on a unit-id sequence using the fixed padded
    /// shape used by the shipped Python evaluator.
    pub(crate) fn logits(&self, units: &[i32]) -> [f32; CLASSES] {
        let mut scratch = vec![0.0f32; INFERENCE_SCRATCH].into_boxed_slice();
        debug_assert!(scratch.len() >= INFERENCE_SCRATCH);
        let t1 = MAX_UNITS / CONV0_POOL;
        let t2 = t1 / CONV1_POOL;
        let embed_len = MAX_UNITS * EMBED;
        let pool0_len = t1 * CONV0;
        let pool1_len = t2 * CONV1;
        let unit_count = units.len().min(MAX_UNITS);
        let (activations, conv_scratch) = scratch.split_at_mut(ACTIVATION_SCRATCH);

        {
            let (embed_storage, pool0_storage) = activations.split_at_mut(embed_len);

            // 1) HashEmbedding + GELU.
            let (embed_rows, embed_remainder) = embed_storage.as_chunks_mut::<EMBED>();
            debug_assert!(embed_remainder.is_empty());
            for (&id, dst) in units.iter().take(unit_count).zip(embed_rows.iter_mut()) {
                debug_assert!(id >= 0);
                embed_position(&self.embedding, id as u32, dst);
                for v in dst.iter_mut() {
                    *v = gelu(*v);
                }
            }
            if unit_count < MAX_UNITS {
                embed_rows[unit_count].fill(0.0);
            }

            // 2) Conv0 + GELU + MaxPool(4).
            let pool0 = &mut pool0_storage[..pool0_len];
            let embed_materialized_rows = if unit_count < MAX_UNITS {
                unit_count + 1
            } else {
                MAX_UNITS
            };
            let embed_tensor = Tensor::with_repeated_tail(
                &embed_storage[..embed_materialized_rows * EMBED],
                MAX_UNITS,
                EMBED,
                unit_count,
            );
            let pool0_tensor = conv_gelu_maxpool_tensor(
                embed_tensor,
                &self.conv0_kernel,
                CONV0_KERNEL,
                CONV0,
                &self.conv0_bias,
                CONV0_POOL,
                pool0,
                &mut *conv_scratch,
            );

            // 3) Conv1 + GELU + MaxPool(2).
            let pool1 = &mut embed_storage[..pool1_len];
            let pool1_tensor = conv_gelu_maxpool_tensor(
                pool0_tensor,
                &self.conv1_kernel,
                CONV1_KERNEL,
                CONV1,
                &self.conv1_bias,
                CONV1_POOL,
                pool1,
                &mut *conv_scratch,
            );

            // 4) Conv2 + GELU + GlobalMax/AvgPool.
            let mut pooled = [0.0f32; POOLED];
            let (max_slice, avg_slice) = pooled.split_at_mut(CONV2);
            conv_gelu_global_pool_tensor(
                pool1_tensor,
                &self.conv2_kernel,
                CONV2_KERNEL,
                CONV2,
                &self.conv2_bias,
                max_slice,
                avg_slice,
                pool0_storage,
                &mut *conv_scratch,
            );

            // 5) Dense + GELU.
            let mut dense0_out = [0.0f32; DENSE];
            dense_forward(
                &pooled,
                &self.dense0_kernel,
                &self.dense0_bias,
                &mut dense0_out,
            );
            for v in &mut dense0_out {
                *v = gelu(*v);
            }

            // 6) Output logits.
            let mut logits = [0.0f32; CLASSES];
            dense_forward(
                &dense0_out,
                &self.output_kernel,
                &self.output_bias,
                &mut logits,
            );
            logits
        }
    }

    pub(crate) fn tokenize_units(&self, window: &TokenWindow) -> Vec<i32> {
        tokenize(window)
    }
}
