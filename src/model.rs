use crate::language::{CLASS_LANGUAGES, Language};
use std::sync::OnceLock;

mod architectures;

static MODEL_BYTES: &[u8] = include_bytes!("../assets/magika/source-student-q4.bin");

const MODEL_MAGIC_LEN: usize = 8;
type Token = u16;
const TOKEN_LENGTH: usize = 2_048;
const TOKEN_VOCAB_SIZE: usize = 257;
const PADDING_TOKEN: Token = 256;
const PADDING_TOKEN_INDEX: usize = PADDING_TOKEN as usize;
const MAGIKA_BEG_SIZE: usize = 1_024;
const MAGIKA_END_SIZE: usize = 1_024;
const MAGIKA_BLOCK_SIZE: usize = 4_096;

const EMBED: usize = 40;
const CONV0_KERNEL: usize = 7;
const CONV0: usize = 80;
const CONV0_POOL: usize = 4;
const CONV0_POOL_WIDTH: usize = TOKEN_LENGTH / CONV0_POOL;
const CONV1_KERNEL: usize = 5;
const CONV1: usize = 160;
#[cfg(all(target_arch = "aarch64", not(miri)))]
const CONV1_I8MM_CHUNKS: usize = CONV0 / 8;
const POOLED: usize = CONV1 * 2;
const DENSE: usize = 176;
const HASH_BINS: usize = 256;
const HASH: usize = 64;
const FEATURES: usize = DENSE + HASH;
const CLASSES: usize = 67;
const RELIABLE_LOGIT_MARGIN: f32 = 3.0;

#[derive(Debug)]
struct Model {
    conv0_lookup: Vec<f32>,
    conv0_bias: [f32; CONV0],
    conv1_kernel: I8Matrix,
    conv1_i8mm_kernel: Option<Conv1I8mmKernel>,
    conv1_padding_lookup: [[f32; CONV1]; CONV1_KERNEL],
    conv1_bias: [f32; CONV1],
    dense0_kernel: I8Matrix,
    dense0_bias: [f32; DENSE],
    hash_kernel: I8Matrix,
    hash_bias: [f32; HASH],
    output_kernel: I8Matrix,
    output_bias: [f32; CLASSES],
}

#[derive(Debug)]
struct I8Weights {
    data: Vec<i8>,
    scale: f32,
}

#[derive(Debug)]
struct I8Matrix {
    data: Vec<i8>,
    input_len: usize,
    output_len: usize,
    scale: f32,
}

#[derive(Debug)]
struct Conv1I8mmKernel {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    data: Vec<i8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct QuantizedVector<const N: usize> {
    values: [i8; N],
    scale: f32,
}

struct Tokenized {
    tokens: [Token; TOKEN_LENGTH],
    begin_len: usize,
    end_start: usize,
    end_len: usize,
}

impl Tokenized {
    #[inline(always)]
    fn is_dense(&self) -> bool {
        self.tokens[MAGIKA_BEG_SIZE] != PADDING_TOKEN
    }
}

struct Conv0Pooled {
    rows: Vec<QuantizedVector<CONV0>>,
    row_slots: [u16; CONV0_POOL_WIDTH],
}

impl Model {
    fn load() -> Self {
        assert!(has_model_magic(MODEL_BYTES));
        let metadata_start = rfind_bytes(MODEL_BYTES, br#"{"bits""#).expect("Magika metadata");
        let metadata = std::str::from_utf8(&MODEL_BYTES[metadata_start..]).expect("utf-8 metadata");
        assert!(metadata.contains(r#""architecture":"conv-xwide-hash-hidden""#));
        let scales = parse_scales(metadata);
        assert_eq!(scales.len(), 6);

        let mut cursor = MODEL_MAGIC_LEN;
        let embedding = read_i8_weights(&mut cursor, TOKEN_VOCAB_SIZE * EMBED, scales[0]);
        let conv0_kernel = read_i8_weights(&mut cursor, CONV0_KERNEL * EMBED * CONV0, scales[1]);
        let conv0_bias = read_f32_array::<CONV0>(&mut cursor);
        let conv0_lookup = precompute_conv0_lookup(&embedding, &conv0_kernel);
        let conv0_padding_pooled = precompute_conv0_padding_pooled(&conv0_lookup, &conv0_bias);
        let conv1_kernel = read_i8_matrix(&mut cursor, CONV1_KERNEL * CONV0, CONV1, scales[2]);
        let conv1_i8mm_kernel = precompute_conv1_i8mm_kernel(&conv1_kernel);
        let conv1_padding_lookup =
            precompute_conv1_padding_lookup(&conv0_padding_pooled, &conv1_kernel);
        let conv1_bias = read_f32_array::<CONV1>(&mut cursor);
        let dense0_kernel = read_i8_matrix(&mut cursor, POOLED, DENSE, scales[3]);
        let dense0_bias = read_f32_array::<DENSE>(&mut cursor);
        let hash_kernel = read_i8_matrix(&mut cursor, HASH_BINS, HASH, scales[4]);
        let hash_bias = read_f32_array::<HASH>(&mut cursor);
        let output_kernel = read_i8_matrix(&mut cursor, FEATURES, CLASSES, scales[5]);
        let output_bias = read_f32_array::<CLASSES>(&mut cursor);

        assert_eq!(cursor + 4, metadata_start);
        let metadata_len = u32::from_le_bytes(MODEL_BYTES[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(metadata_len as usize, MODEL_BYTES.len() - metadata_start);

        Self {
            conv0_lookup,
            conv0_bias,
            conv1_kernel,
            conv1_i8mm_kernel,
            conv1_padding_lookup,
            conv1_bias,
            dense0_kernel,
            dense0_bias,
            hash_kernel,
            hash_bias,
            output_kernel,
            output_bias,
        }
    }

    fn get() -> &'static Self {
        static MODEL: OnceLock<Model> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    fn logits(&self, tokenized: &Tokenized) -> [f32; CLASSES] {
        let pooled0 = self.conv0_average_pool(tokenized);
        let pooled = self.conv1_global_pool(&pooled0);
        let mut conv_features =
            dense_quantized_array::<POOLED, DENSE>(&pooled, &self.dense0_kernel, &self.dense0_bias);
        gelu_slice(&mut conv_features);

        let hash_counts = hashed_bigram_features(tokenized);
        let mut hash_features = dense_quantized_array::<HASH_BINS, HASH>(
            &hash_counts,
            &self.hash_kernel,
            &self.hash_bias,
        );
        gelu_slice(&mut hash_features);

        let mut features = [0.0; FEATURES];
        features[..DENSE].copy_from_slice(&conv_features);
        features[DENSE..].copy_from_slice(&hash_features);

        dense_quantized_array::<FEATURES, CLASSES>(
            &features,
            &self.output_kernel,
            &self.output_bias,
        )
    }

    fn conv0_average_pool(&self, tokenized: &Tokenized) -> Conv0Pooled {
        let tokens = &tokenized.tokens;
        if tokenized.is_dense() {
            return self.conv0_average_pool_dense(tokens);
        }

        let mut rows = Vec::with_capacity(64);
        let mut row_slots = [u16::MAX; CONV0_POOL_WIDTH];

        self.push_conv0_pool_span(tokens, &mut row_slots, &mut rows, 0, tokenized.begin_len);
        self.push_conv0_pool_span(
            tokens,
            &mut row_slots,
            &mut rows,
            tokenized.end_start,
            tokenized.end_start + tokenized.end_len,
        );

        Conv0Pooled { rows, row_slots }
    }

    fn push_conv0_pool_span(
        &self,
        tokens: &[Token; TOKEN_LENGTH],
        row_slots: &mut [u16; CONV0_POOL_WIDTH],
        rows: &mut Vec<QuantizedVector<CONV0>>,
        span_start: usize,
        span_end: usize,
    ) {
        if span_start >= span_end {
            return;
        }

        let pad = CONV0_KERNEL / 2;
        let first_position = span_start.saturating_sub(CONV0_POOL - 1 + pad);
        let last_position_exclusive = (span_end + pad).min(TOKEN_LENGTH);
        let start_pool = first_position / CONV0_POOL;
        let end_pool = last_position_exclusive
            .div_ceil(CONV0_POOL)
            .min(CONV0_POOL_WIDTH);

        for (pooled_index, slot) in row_slots
            .iter_mut()
            .enumerate()
            .take(end_pool)
            .skip(start_pool)
        {
            if *slot != u16::MAX {
                continue;
            }

            let position_start = pooled_index * CONV0_POOL;
            if is_full_padding_conv0_pool(tokens, position_start) {
                continue;
            }

            *slot = rows.len() as u16;
            if is_dense_conv0_pool_span(pooled_index, span_start, span_end) {
                rows.push(self.conv0_pool_row_dense(tokens, pooled_index));
            } else {
                rows.push(self.conv0_pool_row(tokens, pooled_index));
            }
        }
    }

    #[inline(always)]
    fn conv0_average_pool_dense(&self, tokens: &[Token; TOKEN_LENGTH]) -> Conv0Pooled {
        let mut rows = Vec::with_capacity(CONV0_POOL_WIDTH);
        let mut row_slots = [u16::MAX; CONV0_POOL_WIDTH];

        row_slots[0] = 0;
        rows.push(self.conv0_pool_row(tokens, 0));

        for (pooled_index, slot) in row_slots
            .iter_mut()
            .enumerate()
            .take(CONV0_POOL_WIDTH - 1)
            .skip(1)
        {
            *slot = rows.len() as u16;
            rows.push(self.conv0_pool_row_dense(tokens, pooled_index));
        }

        row_slots[CONV0_POOL_WIDTH - 1] = rows.len() as u16;
        rows.push(self.conv0_pool_row(tokens, CONV0_POOL_WIDTH - 1));

        Conv0Pooled { rows, row_slots }
    }

    #[inline(always)]
    fn conv0_pool_row(
        &self,
        tokens: &[Token; TOKEN_LENGTH],
        pooled_index: usize,
    ) -> QuantizedVector<CONV0> {
        let position_start = pooled_index * CONV0_POOL;
        let pad = CONV0_KERNEL / 2;
        let mut pooled_row = [0.0; CONV0];
        for position in position_start..position_start + CONV0_POOL {
            let mut row = self.conv0_bias;
            for kernel_position in 0..CONV0_KERNEL {
                let source_position = if kernel_position < pad {
                    let offset = pad - kernel_position;
                    if position < offset {
                        continue;
                    }
                    position - offset
                } else {
                    let source_position = position + (kernel_position - pad);
                    if source_position >= TOKEN_LENGTH {
                        continue;
                    }
                    source_position
                };

                let token = tokens[source_position].min(PADDING_TOKEN) as usize;
                let lookup_start = (kernel_position * TOKEN_VOCAB_SIZE + token) * CONV0;
                let lookup = &self.conv0_lookup[lookup_start..lookup_start + CONV0];
                add_assign::<CONV0>(&mut row, lookup);
            }

            add_gelu_quarter::<CONV0>(&mut pooled_row, &row);
        }

        quantize_array(&pooled_row)
    }

    #[inline(always)]
    fn conv0_pool_row_dense(
        &self,
        tokens: &[Token; TOKEN_LENGTH],
        pooled_index: usize,
    ) -> QuantizedVector<CONV0> {
        architectures::conv0_pool_row_dense(self, tokens, pooled_index)
    }

    fn conv1_global_pool(&self, input: &Conv0Pooled) -> [f32; POOLED] {
        if let Some(kernel) = &self.conv1_i8mm_kernel
            && input.rows.len() == CONV0_POOL_WIDTH
        {
            return self.conv1_global_pool_dense_i8mm(input, kernel);
        }
        if let Some(kernel) = &self.conv1_i8mm_kernel {
            return self.conv1_global_pool_sparse_i8mm(input, kernel);
        }
        let mut pooled = [0.0; POOLED];
        pooled[..CONV1].fill(f32::NEG_INFINITY);

        let mut position = 0;
        while position < CONV0_POOL_WIDTH {
            if input.is_full_padding_conv1_window(position) {
                let run_start = position;
                while position < CONV0_POOL_WIDTH && input.is_full_padding_conv1_window(position) {
                    position += 1;
                }
                let row = self.conv1_row(input, run_start);
                accumulate_conv1_row(&mut pooled, row, position - run_start);
                continue;
            }

            let row = self.conv1_row(input, position);
            accumulate_conv1_row(&mut pooled, row, 1);
            position += 1;
        }

        for out_channel in 0..CONV1 {
            pooled[CONV1 + out_channel] /= CONV0_POOL_WIDTH as f32;
        }

        pooled
    }

    fn conv1_global_pool_sparse_i8mm(
        &self,
        input: &Conv0Pooled,
        kernel: &Conv1I8mmKernel,
    ) -> [f32; POOLED] {
        let mut pooled = [0.0; POOLED];
        pooled[..CONV1].fill(f32::NEG_INFINITY);

        let mut position = 0;
        while position < CONV0_POOL_WIDTH {
            if input.is_full_padding_conv1_window(position) {
                let run_start = position;
                while position < CONV0_POOL_WIDTH && input.is_full_padding_conv1_window(position) {
                    position += 1;
                }
                let row = self.conv1_row(input, run_start);
                accumulate_conv1_row(&mut pooled, row, position - run_start);
                continue;
            }

            if position + 1 < CONV0_POOL_WIDTH && !input.is_full_padding_conv1_window(position + 1)
            {
                let (row0, row1) = self.conv1_pair_rows_i8mm(input, kernel, position);
                accumulate_conv1_row(&mut pooled, row0, 1);
                accumulate_conv1_row(&mut pooled, row1, 1);
                position += 2;
                continue;
            }

            let row = self.conv1_row(input, position);
            accumulate_conv1_row(&mut pooled, row, 1);
            position += 1;
        }

        for out_channel in 0..CONV1 {
            pooled[CONV1 + out_channel] /= CONV0_POOL_WIDTH as f32;
        }

        pooled
    }

    fn conv1_pair_rows_i8mm(
        &self,
        input: &Conv0Pooled,
        kernel: &Conv1I8mmKernel,
        position: usize,
    ) -> ([f32; CONV1], [f32; CONV1]) {
        debug_assert!(position + 1 < CONV0_POOL_WIDTH);

        let mut row0 = self.conv1_bias;
        let mut row1 = self.conv1_bias;

        for kernel_position in 0..CONV1_KERNEL {
            let source_position0 = conv1_source_position(position, kernel_position);
            let source_position1 = conv1_source_position(position + 1, kernel_position);

            match (source_position0, source_position1) {
                (Some(source0), Some(source1)) => {
                    let row0_is_padding = input.is_full_padding_row(source0);
                    let row1_is_padding = input.is_full_padding_row(source1);
                    match (row0_is_padding, row1_is_padding) {
                        (false, false) => unsafe {
                            add_quantized_conv1_pair_i8mm(
                                &mut row0,
                                &mut row1,
                                input.row(source0),
                                input.row(source1),
                                kernel,
                                kernel_position,
                                self.conv1_kernel.scale,
                            );
                        },
                        (false, true) => {
                            add_quantized_conv1_row(
                                &mut row0,
                                input.row(source0),
                                &self.conv1_kernel,
                                kernel_position,
                            );
                            add_assign::<CONV1>(
                                &mut row1,
                                &self.conv1_padding_lookup[kernel_position],
                            );
                        }
                        (true, false) => {
                            add_assign::<CONV1>(
                                &mut row0,
                                &self.conv1_padding_lookup[kernel_position],
                            );
                            add_quantized_conv1_row(
                                &mut row1,
                                input.row(source1),
                                &self.conv1_kernel,
                                kernel_position,
                            );
                        }
                        (true, true) => {
                            add_assign::<CONV1>(
                                &mut row0,
                                &self.conv1_padding_lookup[kernel_position],
                            );
                            add_assign::<CONV1>(
                                &mut row1,
                                &self.conv1_padding_lookup[kernel_position],
                            );
                        }
                    }
                }
                (Some(source0), None) => {
                    if input.is_full_padding_row(source0) {
                        add_assign::<CONV1>(&mut row0, &self.conv1_padding_lookup[kernel_position]);
                    } else {
                        add_quantized_conv1_row(
                            &mut row0,
                            input.row(source0),
                            &self.conv1_kernel,
                            kernel_position,
                        );
                    }
                }
                (None, Some(source1)) => {
                    if input.is_full_padding_row(source1) {
                        add_assign::<CONV1>(&mut row1, &self.conv1_padding_lookup[kernel_position]);
                    } else {
                        add_quantized_conv1_row(
                            &mut row1,
                            input.row(source1),
                            &self.conv1_kernel,
                            kernel_position,
                        );
                    }
                }
                (None, None) => {}
            }
        }

        (row0, row1)
    }

    fn conv1_global_pool_dense_i8mm(
        &self,
        input: &Conv0Pooled,
        kernel: &Conv1I8mmKernel,
    ) -> [f32; POOLED] {
        let mut pooled = [0.0; POOLED];
        pooled[..CONV1].fill(f32::NEG_INFINITY);

        let mut row0 = self.conv1_bias;
        let mut row1 = self.conv1_bias;
        for kernel_position in CONV1_KERNEL / 2..CONV1_KERNEL {
            let source_position0 = kernel_position - CONV1_KERNEL / 2;
            unsafe {
                add_quantized_conv1_pair_i8mm(
                    &mut row0,
                    &mut row1,
                    &input.rows[source_position0],
                    &input.rows[source_position0 + 1],
                    kernel,
                    kernel_position,
                    self.conv1_kernel.scale,
                );
            }
        }
        add_quantized_conv1_row(&mut row1, &input.rows[0], &self.conv1_kernel, 1);
        accumulate_conv1_row(&mut pooled, row0, 1);
        accumulate_conv1_row(&mut pooled, row1, 1);

        let mut position = CONV1_KERNEL / 2;
        while position + 1 < CONV0_POOL_WIDTH - CONV1_KERNEL / 2 {
            row0 = self.conv1_bias;
            row1 = self.conv1_bias;

            for kernel_position in 0..CONV1_KERNEL {
                let source_position0 = position + kernel_position - CONV1_KERNEL / 2;
                let source_position1 = source_position0 + 1;
                unsafe {
                    add_quantized_conv1_pair_i8mm(
                        &mut row0,
                        &mut row1,
                        &input.rows[source_position0],
                        &input.rows[source_position1],
                        kernel,
                        kernel_position,
                        self.conv1_kernel.scale,
                    );
                }
            }

            accumulate_conv1_row(&mut pooled, row0, 1);
            accumulate_conv1_row(&mut pooled, row1, 1);
            position += 2;
        }

        debug_assert_eq!(position, CONV0_POOL_WIDTH - CONV1_KERNEL / 2);
        row0 = self.conv1_bias;
        row1 = self.conv1_bias;
        for kernel_position in 0..CONV1_KERNEL / 2 + 1 {
            let source_position0 = position + kernel_position - CONV1_KERNEL / 2;
            unsafe {
                add_quantized_conv1_pair_i8mm(
                    &mut row0,
                    &mut row1,
                    &input.rows[source_position0],
                    &input.rows[source_position0 + 1],
                    kernel,
                    kernel_position,
                    self.conv1_kernel.scale,
                );
            }
        }
        add_quantized_conv1_row(
            &mut row0,
            &input.rows[CONV0_POOL_WIDTH - 1],
            &self.conv1_kernel,
            3,
        );
        accumulate_conv1_row(&mut pooled, row0, 1);
        accumulate_conv1_row(&mut pooled, row1, 1);

        for out_channel in 0..CONV1 {
            pooled[CONV1 + out_channel] /= CONV0_POOL_WIDTH as f32;
        }

        pooled
    }

    fn conv1_row(&self, input: &Conv0Pooled, position: usize) -> [f32; CONV1] {
        let mut row = self.conv1_bias;
        let pad = CONV1_KERNEL / 2;

        for kernel_position in 0..CONV1_KERNEL {
            let source_position = if kernel_position < pad {
                let offset = pad - kernel_position;
                if position < offset {
                    continue;
                }
                position - offset
            } else {
                let source_position = position + (kernel_position - pad);
                if source_position >= CONV0_POOL_WIDTH {
                    continue;
                }
                source_position
            };

            if input.is_full_padding_row(source_position) {
                let lookup = &self.conv1_padding_lookup[kernel_position];
                add_assign::<CONV1>(&mut row, lookup);
                continue;
            }

            add_quantized_conv1_row(
                &mut row,
                input.row(source_position),
                &self.conv1_kernel,
                kernel_position,
            );
        }

        row
    }
}

impl Conv0Pooled {
    fn is_full_padding_row(&self, index: usize) -> bool {
        self.row_slots[index] == u16::MAX
    }

    fn is_full_padding_conv1_window(&self, position: usize) -> bool {
        let pad = CONV1_KERNEL / 2;
        position >= pad
            && position + pad < CONV0_POOL_WIDTH
            && self.row_slots[position - pad..=position + pad]
                .iter()
                .all(|slot| *slot == u16::MAX)
    }

    fn row(&self, index: usize) -> &QuantizedVector<CONV0> {
        let slot = self.row_slots[index];
        debug_assert_ne!(slot, u16::MAX);
        &self.rows[slot as usize]
    }
}

impl I8Matrix {
    #[inline(always)]
    fn row(&self, output: usize, input_offset: usize, len: usize) -> &[i8] {
        debug_assert!(output < self.output_len);
        debug_assert!(input_offset + len <= self.input_len);
        let start = output * self.input_len + input_offset;
        &self.data[start..start + len]
    }
}

#[inline(always)]
fn accumulate_conv1_row(pooled: &mut [f32; POOLED], row: [f32; CONV1], count: usize) {
    architectures::accumulate_conv1_row(pooled, row, count);
}

#[inline(always)]
fn add_gelu_quarter<const N: usize>(target: &mut [f32; N], source: &[f32; N]) {
    architectures::add_gelu_quarter(target, source);
}

#[inline(always)]
fn add_assign<const N: usize>(target: &mut [f32; N], source: &[f32]) {
    architectures::add_assign(target, source);
}

#[inline(always)]
fn add_scaled_i8_slice(target: &mut [f32], weights: &[i8], offset: usize, scale: f32) {
    let mut index = 0;
    let weights = &weights[offset..offset + target.len()];
    while index + 4 <= target.len() {
        target[index] += scale * weights[index] as f32;
        target[index + 1] += scale * weights[index + 1] as f32;
        target[index + 2] += scale * weights[index + 2] as f32;
        target[index + 3] += scale * weights[index + 3] as f32;
        index += 4;
    }
    if index < target.len() {
        target[index] += scale * weights[index] as f32;
        index += 1;
    }
    if index < target.len() {
        target[index] += scale * weights[index] as f32;
        index += 1;
    }
    if index < target.len() {
        target[index] += scale * weights[index] as f32;
    }
}

fn has_model_magic(bytes: &[u8]) -> bool {
    bytes.len() >= MODEL_MAGIC_LEN
        && bytes[0] == 0x4d
        && bytes[1] == 0x53
        && bytes[2] == 0x51
        && bytes[3] == 0x31
        && bytes[4] == 0x01
        && bytes[5] == 0x00
        && bytes[6] == 0x00
        && bytes[7] == 0x00
}

fn precompute_conv0_lookup(embedding: &I8Weights, kernel: &I8Weights) -> Vec<f32> {
    debug_assert_eq!(embedding.data.len(), TOKEN_VOCAB_SIZE * EMBED);
    debug_assert_eq!(kernel.data.len(), CONV0_KERNEL * EMBED * CONV0);

    let mut lookup = vec![0.0; CONV0_KERNEL * TOKEN_VOCAB_SIZE * CONV0];
    for kernel_position in 0..CONV0_KERNEL {
        for token in 0..TOKEN_VOCAB_SIZE {
            let lookup_start = (kernel_position * TOKEN_VOCAB_SIZE + token) * CONV0;
            let lookup_row = &mut lookup[lookup_start..lookup_start + CONV0];

            for in_channel in 0..EMBED {
                let input_value = embedding.data[token * EMBED + in_channel] as f32;
                if input_value == 0.0 {
                    continue;
                }
                let kernel_start = (kernel_position * EMBED + in_channel) * CONV0;
                add_scaled_i8_slice(
                    lookup_row,
                    &kernel.data,
                    kernel_start,
                    input_value * embedding.scale * kernel.scale,
                );
            }
        }
    }

    lookup
}

fn precompute_conv0_padding_pooled(
    conv0_lookup: &[f32],
    conv0_bias: &[f32; CONV0],
) -> [f32; CONV0] {
    let mut row = *conv0_bias;
    for kernel_position in 0..CONV0_KERNEL {
        let lookup_start = (kernel_position * TOKEN_VOCAB_SIZE + PADDING_TOKEN_INDEX) * CONV0;
        let lookup = &conv0_lookup[lookup_start..lookup_start + CONV0];
        add_assign::<CONV0>(&mut row, lookup);
    }
    for value in &mut row {
        *value = gelu(*value);
    }
    row
}

fn precompute_conv1_padding_lookup(
    conv0_padding_pooled: &[f32; CONV0],
    conv1_kernel: &I8Matrix,
) -> [[f32; CONV1]; CONV1_KERNEL] {
    debug_assert_eq!(conv1_kernel.input_len, CONV1_KERNEL * CONV0);
    debug_assert_eq!(conv1_kernel.output_len, CONV1);

    let mut lookup = [[0.0; CONV1]; CONV1_KERNEL];
    for (kernel_position, lookup_row) in lookup.iter_mut().enumerate() {
        let input_offset = kernel_position * CONV0;
        for (out_channel, output) in lookup_row.iter_mut().enumerate() {
            let weights = conv1_kernel.row(out_channel, input_offset, CONV0);
            let mut sum = 0.0;
            for (input_value, weight) in conv0_padding_pooled.iter().zip(weights) {
                sum += *input_value * *weight as f32;
            }
            *output = sum * conv1_kernel.scale;
        }
    }
    lookup
}

fn precompute_conv1_i8mm_kernel(conv1_kernel: &I8Matrix) -> Option<Conv1I8mmKernel> {
    architectures::precompute_conv1_i8mm_kernel(conv1_kernel)
}

fn is_full_padding_conv0_pool(tokens: &[Token; TOKEN_LENGTH], position_start: usize) -> bool {
    let pad = CONV0_KERNEL / 2;
    if position_start < pad {
        return false;
    }

    let source_start = position_start - pad;
    let source_end = position_start + CONV0_POOL - 1 + pad;
    source_end < TOKEN_LENGTH
        && tokens[source_start..=source_end]
            .iter()
            .all(|token| *token == PADDING_TOKEN)
}

fn is_dense_conv0_pool_span(pooled_index: usize, span_start: usize, span_end: usize) -> bool {
    let pad = CONV0_KERNEL / 2;
    let position_start = pooled_index * CONV0_POOL;
    position_start >= pad
        && position_start >= span_start + pad
        && position_start + CONV0_POOL - 1 + pad < span_end
}

fn conv1_source_position(position: usize, kernel_position: usize) -> Option<usize> {
    let pad = CONV1_KERNEL / 2;
    if kernel_position < pad {
        position.checked_sub(pad - kernel_position)
    } else {
        let source_position = position + (kernel_position - pad);
        (source_position < CONV0_POOL_WIDTH).then_some(source_position)
    }
}

fn read_i8_weights(cursor: &mut usize, values: usize, scale: f32) -> I8Weights {
    let bytes = values.div_ceil(2);
    let start = *cursor;
    *cursor += bytes;

    let mut data = Vec::with_capacity(values);
    for &packed in &MODEL_BYTES[start..start + bytes] {
        if data.len() < values {
            data.push((packed & 0x0f) as i8 - 8);
        }
        if data.len() < values {
            data.push((packed >> 4) as i8 - 8);
        }
    }

    I8Weights { data, scale }
}

fn read_i8_matrix(cursor: &mut usize, input_len: usize, output_len: usize, scale: f32) -> I8Matrix {
    let values = input_len * output_len;
    let bytes = values.div_ceil(2);
    let payload = &MODEL_BYTES[*cursor..*cursor + bytes];
    *cursor += bytes;

    let mut data = vec![0; values];
    for input_index in 0..input_len {
        for output_index in 0..output_len {
            let packed_index = input_index * output_len + output_index;
            let packed = payload[packed_index / 2];
            let nibble = if packed_index.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            };
            data[output_index * input_len + input_index] = nibble as i8 - 8;
        }
    }

    I8Matrix {
        data,
        input_len,
        output_len,
        scale,
    }
}

fn read_f32_array<const N: usize>(cursor: &mut usize) -> [f32; N] {
    let result = std::array::from_fn(|index| read_f32_at(MODEL_BYTES, *cursor + index * 4));
    *cursor += N * 4;
    result
}

fn read_f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn rfind_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .rposition(|window| window == needle)
}

fn parse_scales(metadata: &str) -> Vec<f32> {
    let mut scales = Vec::new();
    let mut rest = metadata;
    while let Some(index) = rest.find(r#""scale":"#) {
        let value_start = index + r#""scale":"#.len();
        let value_end = rest[value_start..]
            .find([',', '}'])
            .map(|end| value_start + end)
            .expect("scale terminator");
        scales.push(rest[value_start..value_end].parse().expect("scale value"));
        rest = &rest[value_end..];
    }
    scales
}

fn dense_quantized_array<const IN: usize, const OUT: usize>(
    input: &[f32; IN],
    kernel: &I8Matrix,
    bias: &[f32; OUT],
) -> [f32; OUT] {
    architectures::dense_quantized_array(input, kernel, bias)
}

fn add_quantized_conv1_row(
    output: &mut [f32; CONV1],
    input: &QuantizedVector<CONV0>,
    kernel: &I8Matrix,
    kernel_position: usize,
) {
    architectures::add_quantized_conv1_row(output, input, kernel, kernel_position);
}

unsafe fn add_quantized_conv1_pair_i8mm(
    output0: &mut [f32; CONV1],
    output1: &mut [f32; CONV1],
    input0: &QuantizedVector<CONV0>,
    input1: &QuantizedVector<CONV0>,
    kernel: &Conv1I8mmKernel,
    kernel_position: usize,
    kernel_scale: f32,
) {
    unsafe {
        architectures::add_quantized_conv1_pair_i8mm(
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

fn quantize_array<const N: usize>(input: &[f32; N]) -> QuantizedVector<N> {
    let max_abs = max_abs_array(input);

    if max_abs == 0.0 || !max_abs.is_finite() {
        return QuantizedVector {
            values: [0; N],
            scale: 1.0,
        };
    }

    let scale = max_abs / 127.0;
    let inv_scale = 1.0 / scale;
    let values = quantize_values(input, inv_scale);

    QuantizedVector { values, scale }
}

#[inline(always)]
fn quantize_values<const N: usize>(input: &[f32; N], inv_scale: f32) -> [i8; N] {
    architectures::quantize_values(input, inv_scale)
}

#[inline(always)]
fn max_abs_array<const N: usize>(input: &[f32; N]) -> f32 {
    architectures::max_abs_array(input)
}

#[inline(always)]
fn quantize_i8(value: f32, inv_scale: f32) -> i8 {
    let scaled = value * inv_scale;
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    rounded.clamp(-127.0, 127.0) as i8
}

#[cfg(miri)]
fn dot_i8(left: &[i8], right: &[i8]) -> i32 {
    architectures::dot_i8(left, right)
}

fn hashed_bigram_features(tokenized: &Tokenized) -> [f32; HASH_BINS] {
    let tokens = &tokenized.tokens;
    let mut counts = [0.0; HASH_BINS];
    let padding_bucket = (PADDING_TOKEN_INDEX * 263 + PADDING_TOKEN_INDEX * 17) & (HASH_BINS - 1);

    add_bigram_span(&mut counts, tokens, 0, tokenized.begin_len);

    if tokenized.begin_len < tokenized.end_start {
        if tokenized.begin_len > 0 {
            add_bigram_count(
                &mut counts,
                tokens[tokenized.begin_len - 1],
                PADDING_TOKEN,
                1.0,
            );
        }

        let padding_len = tokenized.end_start - tokenized.begin_len;
        if padding_len > 1 {
            counts[padding_bucket] += (padding_len - 1) as f32;
        }

        if tokenized.end_len > 0 {
            add_bigram_count(&mut counts, PADDING_TOKEN, tokens[tokenized.end_start], 1.0);
        }
    } else if tokenized.begin_len > 0 && tokenized.end_len > 0 {
        add_bigram_count(
            &mut counts,
            tokens[tokenized.begin_len - 1],
            tokens[tokenized.end_start],
            1.0,
        );
    }

    add_bigram_span(
        &mut counts,
        tokens,
        tokenized.end_start,
        tokenized.end_start + tokenized.end_len,
    );

    for count in &mut counts {
        *count /= (TOKEN_LENGTH - 1) as f32;
    }
    counts
}

fn add_bigram_span(
    counts: &mut [f32; HASH_BINS],
    tokens: &[Token; TOKEN_LENGTH],
    start: usize,
    end: usize,
) {
    if end.saturating_sub(start) < 2 {
        return;
    }

    let mut index = start;
    while index + 1 < end {
        add_bigram_count(counts, tokens[index], tokens[index + 1], 1.0);
        index += 1;
    }
}

#[inline(always)]
fn add_bigram_count(counts: &mut [f32; HASH_BINS], left: Token, right: Token, count: f32) {
    let bucket = (left as usize * 263 + right as usize * 17) & (HASH_BINS - 1);
    counts[bucket] += count;
}

fn gelu_slice(values: &mut [f32]) {
    architectures::gelu_slice(values);
}

fn gelu(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = 0.797_884_6 * (value + 0.044_715 * cubic);
    0.5 * value * (1.0 + fast_tanh(inner))
}

fn fast_tanh(value: f32) -> f32 {
    if value <= -3.0 {
        return -1.0;
    }
    if value >= 3.0 {
        return 1.0;
    }

    let squared = value * value;
    value * (27.0 + squared) / (27.0 + 9.0 * squared)
}

fn magika_tokens(source: &str) -> Option<Tokenized> {
    let bytes = source.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let block = bytes.len().min(MAGIKA_BLOCK_SIZE);
    let stripped_beg = trim_start_ascii(&bytes[..block]);
    if stripped_beg.len() < 8 {
        return None;
    }
    let stripped_end = trim_end_ascii(&bytes[bytes.len() - block..]);

    let mut tokens = [PADDING_TOKEN; TOKEN_LENGTH];
    let beg = &stripped_beg[..stripped_beg.len().min(MAGIKA_BEG_SIZE)];
    let begin_len = beg.len();
    for (index, &byte) in beg.iter().enumerate() {
        tokens[index] = byte as Token;
    }

    let end_len = stripped_end.len().min(MAGIKA_END_SIZE);
    let end = &stripped_end[stripped_end.len() - end_len..];
    let end_start = MAGIKA_BEG_SIZE + (MAGIKA_END_SIZE - end_len);
    for (index, &byte) in end.iter().enumerate() {
        tokens[end_start + index] = byte as Token;
    }

    Some(Tokenized {
        tokens,
        begin_len,
        end_start,
        end_len,
    })
}

fn trim_start_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    &bytes[start..]
}

fn trim_end_ascii(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|index| index + 1)
        .unwrap_or(0);
    &bytes[..end]
}

/// Detect the source language for a source string.
///
/// Use [`Language::slug`] to map the result to an Arborium/tree-sitter language
/// identifier.
pub fn detect(source: &str) -> Option<Language> {
    let tokenized = magika_tokens(source)?;
    let logits = Model::get().logits(&tokenized);
    let class_index = reliable_class_from_logits(&logits)?;
    Some(CLASS_LANGUAGES[class_index])
}

#[cfg(all(test, not(miri)))]
fn predict_class(source: &str) -> Option<(usize, f32, [f32; CLASSES])> {
    let tokenized = magika_tokens(source)?;
    let logits = Model::get().logits(&tokenized);
    let probabilities = softmax(logits);
    let (class_index, probability) = top_probability(&probabilities)?;
    Some((class_index, probability, probabilities))
}

fn reliable_class_from_logits(logits: &[f32; CLASSES]) -> Option<usize> {
    let (class_index, top, runner_up) = top_two_logits(logits)?;
    if top - runner_up >= RELIABLE_LOGIT_MARGIN {
        return Some(class_index);
    }

    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for &logit in logits {
        let value = (logit - top).exp();
        sum += value;
        sum_squares += value * value;
    }

    let probability = 1.0 / sum;
    let mean = 1.0 / CLASSES as f32;
    let variance = ((sum_squares / (sum * sum)) - mean).max(0.0) / (CLASSES - 1) as f32;
    (probability > mean + 2.0 * variance.sqrt()).then_some(class_index)
}

fn top_two_logits(logits: &[f32; CLASSES]) -> Option<(usize, f32, f32)> {
    let mut top = f32::NEG_INFINITY;
    let mut runner_up = f32::NEG_INFINITY;
    let mut class_index = 0;

    for (index, &logit) in logits.iter().enumerate() {
        if logit >= top {
            runner_up = top;
            top = logit;
            class_index = index;
        } else if logit > runner_up {
            runner_up = logit;
        }
    }

    top.is_finite().then_some((class_index, top, runner_up))
}

#[cfg(test)]
fn softmax(logits: [f32; CLASSES]) -> [f32; CLASSES] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    let mut probabilities = [0.0; CLASSES];
    for (probability, logit) in probabilities.iter_mut().zip(logits) {
        *probability = (logit - max).exp();
        sum += *probability;
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

#[cfg(test)]
fn top_probability(probabilities: &[f32; CLASSES]) -> Option<(usize, f32)> {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
}

#[cfg(all(test, miri))]
fn is_reliable(probabilities: &[f32; CLASSES], probability: f32) -> bool {
    let mean = probabilities.iter().sum::<f32>() / CLASSES as f32;
    let variance = probabilities
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / (CLASSES - 1) as f32;
    probability > mean + 2.0 * variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(miri))]
    #[test]
    fn loads_embedded_model() {
        let model = Model::get();
        assert_eq!(
            model.conv0_lookup.len(),
            CONV0_KERNEL * TOKEN_VOCAB_SIZE * CONV0
        );
        assert_eq!(model.output_bias.len(), CLASSES);
    }

    #[cfg(not(miri))]
    #[test]
    fn predicts_rust_fixture_class() {
        let (class, probability, _) =
            predict_class("use std::fmt;\nfn main() { println!(\"hi\"); }").unwrap();
        assert_eq!(class, 49);
        assert!(probability.is_finite());
    }

    #[cfg(not(miri))]
    #[test]
    fn detects_rust_from_source() {
        assert_eq!(
            detect("use std::fmt;\nfn main() { println!(\"hi\"); }"),
            Some(Language::Rust),
        );
        assert_eq!(Language::Rust.slug(), "rust");
    }

    #[cfg(not(miri))]
    #[test]
    fn sparse_conv0_pool_matches_full_scan() {
        let tokenized = magika_tokens(include_str!("../snippets/demo.rs")).unwrap();
        assert!(!tokenized.is_dense());

        let model = Model::get();
        let optimized = model.conv0_average_pool(&tokenized);
        let full_scan = full_scan_sparse_conv0_pool(model, &tokenized.tokens);

        assert_eq!(optimized.row_slots, full_scan.row_slots);
        assert_eq!(optimized.rows, full_scan.rows);
    }

    #[cfg(not(miri))]
    fn full_scan_sparse_conv0_pool(model: &Model, tokens: &[Token; TOKEN_LENGTH]) -> Conv0Pooled {
        let mut rows = Vec::with_capacity(64);
        let mut row_slots = [u16::MAX; CONV0_POOL_WIDTH];

        for (pooled_index, slot) in row_slots.iter_mut().enumerate() {
            let position_start = pooled_index * CONV0_POOL;
            if is_full_padding_conv0_pool(tokens, position_start) {
                continue;
            }

            *slot = rows.len() as u16;
            rows.push(model.conv0_pool_row(tokens, pooled_index));
        }

        Conv0Pooled { rows, row_slots }
    }

    #[cfg(all(not(miri), target_arch = "aarch64"))]
    #[test]
    fn dense_conv1_i8mm_matches_generic_pool() {
        let mut source = String::new();
        while source.len() < MAGIKA_BLOCK_SIZE + 512 {
            source.push_str(include_str!("../snippets/demo.rs"));
            source.push('\n');
        }

        let tokenized = magika_tokens(&source).unwrap();
        assert!(tokenized.is_dense());

        let model = Model::get();
        let pooled0 = model.conv0_average_pool(&tokenized);
        let Some(kernel) = model.conv1_i8mm_kernel.as_ref() else {
            return;
        };

        let fused = model.conv1_global_pool_dense_i8mm(&pooled0, kernel);
        let generic = generic_conv1_global_pool(model, &pooled0);
        for (left, right) in fused.iter().zip(generic) {
            assert!((left - right).abs() < 0.001, "{left} != {right}");
        }
    }

    #[cfg(all(not(miri), target_arch = "aarch64"))]
    #[test]
    fn sparse_conv1_i8mm_matches_generic_pool() {
        let tokenized = magika_tokens(include_str!("../snippets/demo.rs")).unwrap();
        assert!(!tokenized.is_dense());

        let model = Model::get();
        if model.conv1_i8mm_kernel.is_none() {
            return;
        }

        let pooled0 = model.conv0_average_pool(&tokenized);
        let sparse_i8mm = model.conv1_global_pool(&pooled0);
        let generic = generic_conv1_global_pool(model, &pooled0);
        for (left, right) in sparse_i8mm.iter().zip(generic) {
            assert!((left - right).abs() < 0.001, "{left} != {right}");
        }
    }

    #[cfg(all(not(miri), target_arch = "aarch64"))]
    fn generic_conv1_global_pool(model: &Model, input: &Conv0Pooled) -> [f32; POOLED] {
        let mut pooled = [0.0; POOLED];
        pooled[..CONV1].fill(f32::NEG_INFINITY);

        for position in 0..CONV0_POOL_WIDTH {
            let row = model.conv1_row(input, position);
            accumulate_conv1_row(&mut pooled, row, 1);
        }

        for out_channel in 0..CONV1 {
            pooled[CONV1 + out_channel] /= CONV0_POOL_WIDTH as f32;
        }

        pooled
    }

    #[cfg(not(miri))]
    #[test]
    fn handrolled_fuzz_detect_smoke() {
        run_handrolled_fuzz(24);
    }

    #[cfg(not(miri))]
    #[test]
    #[ignore = "longer deterministic fuzz pass"]
    fn handrolled_fuzz_detect_extended() {
        run_handrolled_fuzz(256);
    }

    #[cfg(not(miri))]
    fn run_handrolled_fuzz(cases: usize) {
        const SEEDS: &[&[u8]] = &[
            b"",
            b"       ",
            b"short",
            b"fn main() {}",
            b"#!/usr/bin/env python3\nprint('hello')\n",
            b"{\"name\":\"dioxus-code\",\"version\":\"0.0.1\"}\n",
            b"\0\0\0\0\0\0\0\0",
            b"\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8",
            b"        fn main() {}\n",
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ];

        for seed in SEEDS {
            fuzz_detect_bytes(seed);
        }

        let mut long = Vec::with_capacity(8 * 1024 + 257);
        while long.len() < 8 * 1024 + 257 {
            long.extend_from_slice(b"fn generated_case() { println!(\"hello\"); }\n");
        }
        fuzz_detect_bytes(&long);

        long.clear();
        long.resize(4 * 1024 + 17, b' ');
        long.extend_from_slice(b"let visible_after_leading_block = true;\n");
        fuzz_detect_bytes(&long);

        let mut rng = FuzzRng::new(0x9e37_79b9_7f4a_7c15);
        for case in 0..cases {
            let len = fuzz_len(&mut rng, case);
            let mut data = Vec::with_capacity(len);
            for index in 0..len {
                data.push(fuzz_byte(&mut rng, case, index));
            }

            if len > 0 {
                data[0] = match case % 6 {
                    0 => b' ',
                    1 => b'\n',
                    2 => b'/',
                    3 => b'{',
                    4 => 0,
                    _ => data[0],
                };
            }
            if len > MAGIKA_BLOCK_SIZE {
                data[MAGIKA_BLOCK_SIZE] = match case % 4 {
                    0 => b'\t',
                    1 => b'}',
                    2 => 0xff,
                    _ => data[MAGIKA_BLOCK_SIZE],
                };
            }

            fuzz_detect_bytes(&data);
        }
    }

    #[cfg(not(miri))]
    fn fuzz_detect_bytes(bytes: &[u8]) {
        let source = String::from_utf8_lossy(bytes);
        let detected = detect(source.as_ref());

        if source.trim().is_empty() {
            assert_eq!(detected, None);
        }
        if let Some(language) = detected {
            assert!(!language.slug().is_empty());
        }
        if let Some(tokenized) = magika_tokens(source.as_ref()) {
            assert!(tokenized.tokens.iter().all(|token| *token <= PADDING_TOKEN));
            assert_eq!(
                hashed_bigram_features(&tokenized),
                scan_hashed_bigram_features(&tokenized.tokens),
            );
        }
    }

    #[cfg(not(miri))]
    fn scan_hashed_bigram_features(tokens: &[Token; TOKEN_LENGTH]) -> [f32; HASH_BINS] {
        let mut counts = [0.0; HASH_BINS];
        let padding_bucket =
            (PADDING_TOKEN_INDEX * 263 + PADDING_TOKEN_INDEX * 17) & (HASH_BINS - 1);
        let mut index = 0;
        while index < TOKEN_LENGTH - 1 {
            if tokens[index] == PADDING_TOKEN && tokens[index + 1] == PADDING_TOKEN {
                let run_start = index;
                while index < TOKEN_LENGTH - 1
                    && tokens[index] == PADDING_TOKEN
                    && tokens[index + 1] == PADDING_TOKEN
                {
                    index += 1;
                }
                counts[padding_bucket] += (index - run_start) as f32;
                continue;
            }

            add_bigram_count(&mut counts, tokens[index], tokens[index + 1], 1.0);
            index += 1;
        }
        for count in &mut counts {
            *count /= (TOKEN_LENGTH - 1) as f32;
        }
        counts
    }

    #[cfg(not(miri))]
    fn fuzz_len(rng: &mut FuzzRng, case: usize) -> usize {
        const BOUNDARIES: &[usize] = &[
            0, 1, 7, 8, 9, 255, 256, 1_023, 1_024, 1_025, 2_047, 2_048, 2_049, 4_095, 4_096, 4_097,
            8_191, 8_192,
        ];

        if case.is_multiple_of(3) {
            return (rng.next_usize() % (8 * 1024 + 1)).min(8 * 1024);
        }

        let boundary = BOUNDARIES[case % BOUNDARIES.len()];
        let jitter = (rng.next_usize() % 17) as isize - 8;
        (boundary as isize + jitter).clamp(0, 8 * 1024) as usize
    }

    #[cfg(not(miri))]
    fn fuzz_byte(rng: &mut FuzzRng, case: usize, index: usize) -> u8 {
        match (case.wrapping_mul(31) ^ index) % 16 {
            0 => 0,
            1 => b' ',
            2 => b'\n',
            3 => b'\t',
            4 => b'a' + (rng.next_u8() % 26),
            5 => b'0' + (rng.next_u8() % 10),
            6 => [b'{', b'}', b'[', b']', b'(', b')'][rng.next_usize() % 6],
            7 => [b'#', b'/', b'*', b'=', b';', b'\''][rng.next_usize() % 6],
            8 => 0x7f,
            9 => 0x80,
            10 => 0xc0,
            11 => 0xe0,
            12 => 0xf0,
            13 => 0xff,
            _ => rng.next_u8(),
        }
    }

    #[cfg(not(miri))]
    struct FuzzRng(u64);

    #[cfg(not(miri))]
    impl FuzzRng {
        const fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            let mut value = self.0;
            value ^= value >> 12;
            value ^= value << 25;
            value ^= value >> 27;
            self.0 = value;
            value.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn next_usize(&mut self) -> usize {
            self.next_u64() as usize
        }

        fn next_u8(&mut self) -> u8 {
            self.next_u64() as u8
        }
    }

    #[cfg(miri)]
    #[test]
    fn miri_tokenization_handles_boundaries() {
        assert!(magika_tokens("").is_none());
        assert!(magika_tokens("        ").is_none());
        assert!(magika_tokens("short").is_none());

        let tokenized = magika_tokens("        fn main() {}\n").unwrap();
        assert_eq!(tokenized.tokens[0], b'f' as Token);
        assert_eq!(tokenized.tokens[1], b'n' as Token);
        assert!(tokenized.tokens.iter().all(|token| *token <= PADDING_TOKEN));

        let padded = magika_tokens("abcdefgh").unwrap();
        assert_eq!(padded.tokens[0], b'a' as Token);
        assert_eq!(padded.tokens[MAGIKA_BEG_SIZE], PADDING_TOKEN);
        assert_eq!(padded.tokens[TOKEN_LENGTH - 8], b'a' as Token);
        assert_eq!(padded.tokens[TOKEN_LENGTH - 1], b'h' as Token);
    }

    #[cfg(miri)]
    #[test]
    fn miri_core_math_helpers_are_bounded() {
        let mut logits = [0.0; CLASSES];
        logits[3] = 1.0;
        let probabilities = softmax(logits);
        let sum = probabilities.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 0.0001);
        assert_eq!(top_probability(&probabilities).unwrap().0, 3);
        assert!(!is_reliable(&probabilities, probabilities[0]));

        let quantized = quantize_array(&[0.0, 1.0, -1.0, 0.25]);
        assert_eq!(quantized.values, [0, 127, -127, 32]);
        assert!(quantized.scale.is_finite());

        let zeroed = quantize_array(&[0.0, f32::INFINITY, f32::NAN, -2.0]);
        assert_eq!(zeroed.values, [0; 4]);
        assert_eq!(zeroed.scale, 1.0);

        assert_eq!(dot_i8(&[1, -2, 3], &[4, 5, -6]), -24);

        let mut values = [-3.0, 0.0, 3.0, 6.0];
        gelu_slice(&mut values);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[cfg(miri)]
    #[test]
    fn miri_matrix_rows_are_bounds_checked() {
        let matrix = I8Matrix {
            data: vec![1, 2, 3, 4, 5, 6],
            input_len: 3,
            output_len: 2,
            scale: 0.5,
        };

        assert_eq!(matrix.row(0, 0, 3), &[1, 2, 3]);
        assert_eq!(matrix.row(1, 1, 2), &[5, 6]);
        assert_eq!(matrix.scale, 0.5);
    }
}
