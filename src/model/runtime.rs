use super::{
    activation::gelu,
    constants::*,
    embedded::MODEL_BYTES,
    layers::{dense_forward, embed_position},
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
    /// Retained for the naive test oracle; inference uses the packed copies.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) conv0_kernel: Box<[f32]>,
    pub(crate) conv0_bias: [f32; CONV0],
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) conv1_kernel: Box<[f32]>,
    pub(crate) conv1_bias: [f32; CONV1],
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) conv2_kernel: Box<[f32]>,
    pub(crate) conv2_bias: [f32; CONV2],
    /// (POOLED, DENSE) flattened.
    pub(crate) dense0_kernel: Box<[f32]>,
    pub(crate) dense0_bias: [f32; DENSE],
    /// (DENSE, CLASSES) flattened.
    pub(crate) output_kernel: Box<[f32]>,
    pub(crate) output_bias: [f32; CLASSES],
    /// Winograd-transformed conv kernels, packed as `[group][j][in_c][16]`.
    pub(crate) conv0_wino: Box<[f32]>,
    pub(crate) conv1_wino: Box<[f32]>,
    pub(crate) conv2_wino: Box<[f32]>,
    /// Lazily built forward plan (input-independent tail rows).
    forward_plan: OnceLock<super::forward::ForwardPlan>,
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

        let conv0_wino = super::forward::pack_wino_kernel(
            &conv0_kernel,
            CONV0_KERNEL,
            EMBED,
            CONV0,
            super::wino::W0_G.as_flattened(),
            super::wino::W0_POINTS,
        );
        let conv1_wino = super::forward::pack_wino_kernel(
            &conv1_kernel,
            CONV1_KERNEL,
            CONV0,
            CONV1,
            super::wino::W1_G.as_flattened(),
            super::wino::W1_POINTS,
        );
        let conv2_wino = super::forward::pack_wino_kernel(
            &conv2_kernel,
            CONV2_KERNEL,
            CONV1,
            CONV2,
            super::wino::W2_G.as_flattened(),
            super::wino::W2_POINTS,
        );

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
            conv0_wino,
            conv1_wino,
            conv2_wino,
            forward_plan: OnceLock::new(),
        }
    }

    pub(crate) fn get() -> &'static Self {
        static MODEL: OnceLock<Model> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    pub(crate) fn forward_plan(&self) -> &super::forward::ForwardPlan {
        self.forward_plan
            .get_or_init(|| super::forward::ForwardPlan::new(self))
    }

    /// Embed each unit id (pre-activation), writing rows of `EMBED` into `dst`.
    pub(crate) fn embed_units(&self, units: &[i32], dst: &mut [f32]) {
        let (rows, remainder) = dst.as_chunks_mut::<EMBED>();
        debug_assert!(remainder.is_empty());
        debug_assert_eq!(rows.len(), units.len());
        for (&id, dst) in units.iter().zip(rows.iter_mut()) {
            debug_assert!(id >= 0);
            embed_position(&self.embedding, id as u32, dst);
        }
    }

    /// Dense head: `pooled -> GELU(dense0) -> output logits`.
    pub(crate) fn dense_head(&self, pooled: &[f32; POOLED]) -> [f32; CLASSES] {
        let mut dense0_out = [0.0f32; DENSE];
        dense_forward(
            pooled,
            &self.dense0_kernel,
            &self.dense0_bias,
            &mut dense0_out,
        );
        for v in &mut dense0_out {
            *v = gelu(*v);
        }
        let mut logits = [0.0f32; CLASSES];
        dense_forward(
            &dense0_out,
            &self.output_kernel,
            &self.output_bias,
            &mut logits,
        );
        logits
    }

    pub(crate) fn tokenize_units(&self, window: &TokenWindow) -> Vec<i32> {
        tokenize(window)
    }
}
