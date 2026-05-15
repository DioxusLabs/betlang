use super::{
    activation::gelu,
    constants::*,
    embedded::MODEL_BYTES,
    layers::{conv_gelu_global_pool, conv_gelu_maxpool, dense_forward, embed_position},
    reader::{read_f32_array, read_int4_dequant, read_ternary_dequant},
    tokenizer::tokenize,
};
use std::sync::OnceLock;

#[derive(Debug)]
pub(crate) struct Model {
    /// 1536 × 28 dequantized embedding rows.
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
        assert_eq!(
            MODEL_BYTES.len(),
            MODEL_PAYLOAD_LEN,
            "unexpected model payload length"
        );

        let mut cur = 0;

        // q_hash_embedding: weights [(1536, 28)] int4
        let embedding = read_int4_dequant(&mut cur, BINS * EMBED, MODEL_WEIGHT_SCALES[0]);

        // q_conv_0: weights [(7, 28, 96)] ternary, bias [(96,)] f32
        let conv0_kernel = read_ternary_dequant(
            &mut cur,
            CONV0_KERNEL * EMBED * CONV0,
            MODEL_WEIGHT_SCALES[1],
        );
        let conv0_bias = read_f32_array::<CONV0>(&mut cur);

        // q_conv_1: weights [(5, 96, 192)] ternary, bias [(192,)] f32
        let conv1_kernel = read_ternary_dequant(
            &mut cur,
            CONV1_KERNEL * CONV0 * CONV1,
            MODEL_WEIGHT_SCALES[2],
        );
        let conv1_bias = read_f32_array::<CONV1>(&mut cur);

        // q_conv_2: weights [(3, 192, 192)] ternary, bias [(192,)] f32
        let conv2_kernel = read_ternary_dequant(
            &mut cur,
            CONV2_KERNEL * CONV1 * CONV2,
            MODEL_WEIGHT_SCALES[3],
        );
        let conv2_bias = read_f32_array::<CONV2>(&mut cur);

        // q_dense_0: weights [(384, 160)] ternary, bias [(160,)] f32
        let dense0_kernel = read_ternary_dequant(&mut cur, POOLED * DENSE, MODEL_WEIGHT_SCALES[4]);
        let dense0_bias = read_f32_array::<DENSE>(&mut cur);

        // q_output: weights [(160, 67)] int4, bias [(67,)] f32
        let output_kernel = read_int4_dequant(&mut cur, DENSE * CLASSES, MODEL_WEIGHT_SCALES[5]);
        let output_bias = read_f32_array::<CLASSES>(&mut cur);

        assert_eq!(cur, MODEL_BYTES.len(), "unexpected model payload length");

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

    /// Run the full forward pass on a unit-id sequence (length = `len`).
    pub(crate) fn logits(&self, units: &[i32], len: usize) -> [f32; CLASSES] {
        let t = len.min(MAX_UNITS);
        let t1 = t / CONV0_POOL;
        let t2 = t1 / CONV1_POOL;
        let embed_len = t * EMBED;
        let pool0_len = t1 * CONV0;
        let pool1_len = t2 * CONV1;
        let mut scratch = vec![0.0f32; INFERENCE_SCRATCH].into_boxed_slice();
        let (activations, conv_scratch) = scratch.split_at_mut(ACTIVATION_SCRATCH);

        {
            let (embed, pool0_region) = activations.split_at_mut(embed_len);

            // 1) HashEmbedding + GELU.
            let (embed_rows, embed_remainder) = embed.as_chunks_mut::<EMBED>();
            debug_assert!(embed_remainder.is_empty());
            for (pos, dst) in embed_rows.iter_mut().take(t.min(units.len())).enumerate() {
                let id = units[pos];
                if id < 0 {
                    continue;
                }
                embed_position(&self.embedding, id as u32, dst);
                for v in dst.iter_mut() {
                    *v = gelu(*v);
                }
            }

            // 2) Conv0 + GELU + MaxPool(4).
            let pool0 = &mut pool0_region[..pool0_len];
            conv_gelu_maxpool(
                embed,
                t,
                EMBED,
                &self.conv0_kernel,
                CONV0_KERNEL,
                CONV0,
                &self.conv0_bias,
                CONV0_POOL,
                pool0,
                &mut *conv_scratch,
            );
        }

        // 3) Conv1 + GELU + MaxPool(2).
        {
            let (pool1_region, pool0_region) = activations.split_at_mut(embed_len);
            let pool0 = &pool0_region[..pool0_len];
            let pool1 = &mut pool1_region[..pool1_len];
            conv_gelu_maxpool(
                pool0,
                t1,
                CONV0,
                &self.conv1_kernel,
                CONV1_KERNEL,
                CONV1,
                &self.conv1_bias,
                CONV1_POOL,
                pool1,
                &mut *conv_scratch,
            );
        }

        // 4) Conv2 + GELU + GlobalMax/AvgPool.
        let pool1 = &activations[..pool1_len];
        let mut pooled = [0.0f32; POOLED];
        let (max_slice, avg_slice) = pooled.split_at_mut(CONV2);
        conv_gelu_global_pool(
            pool1,
            t2,
            CONV1,
            &self.conv2_kernel,
            CONV2_KERNEL,
            CONV2,
            &self.conv2_bias,
            max_slice,
            avg_slice,
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

    /// Run inference using the padded 2048-position shape used by the shipped
    /// Python evaluator. Shorter runtime sequences must not shrink the CNN,
    /// because pooling/global-average behavior changes materially.
    pub(crate) fn logits_for_runtime_units(&self, units: &[i32]) -> [f32; CLASSES] {
        self.logits(units, MAX_UNITS)
    }

    pub(crate) fn tokenize_units(&self, bytes: &[u8], padding_mask: &[bool]) -> Vec<i32> {
        tokenize(bytes, padding_mask)
    }
}
