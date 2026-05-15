#!/usr/bin/env python3
"""Train a tiny content-only Magika student with low-bit QAT."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_magika_source_student import (
    MAGIKA_BEG_SIZE,
    SPLITS,
    Teacher,
    cache_meta_path,
    load_teacher,
    magika_features,
    read_training_windows,
    source_paths,
)


TOKEN_LENGTH = 2048
TOKEN_VOCAB_SIZE = 257
PADDING_TOKEN = 256
CHUNK_COUNT = 64
CHUNK_SIZE = TOKEN_LENGTH // CHUNK_COUNT
QAT_MAGIC = b"MSQ1\x01\0\0\0"


@dataclass(frozen=True)
class TokenSplit:
    tokens: np.memmap
    probabilities: np.memmap
    labels: np.memmap
    count: int
    hidden: np.memmap | None = None
    self_probabilities: np.memmap | None = None
    short_slice_probabilities: dict[int, np.memmap] | None = None
    short_slice_confidences: dict[int, np.memmap] | None = None


QAT_ACTIVE = tf.Variable(True, trainable=False, name="qat_active", dtype=tf.bool)


def _quantize_for_bits(source: tf.Tensor, bits: int) -> tf.Tensor:
    if bits == 2:
        # Use mean(abs(nonzero)) so an exported ternary tensor decoded as
        # {-s,0,+s} is idempotent under QAT, while dense FP warmup weights still
        # behave like the original mean-abs ternary recipe.
        abs_source = tf.abs(source)
        nonzero_abs = tf.boolean_mask(abs_source, abs_source > 0.0)
        scale = tf.cond(
            tf.size(nonzero_abs) > 0,
            lambda: tf.reduce_mean(nonzero_abs),
            lambda: tf.constant(1e-6, dtype=source.dtype),
        )
        scale = tf.maximum(scale, tf.constant(1e-6, dtype=source.dtype))
        threshold = 0.7 * scale
        ternary = tf.where(
            source > threshold,
            scale,
            tf.where(source < -threshold, -scale, tf.zeros_like(source)),
        )
        return source + tf.stop_gradient(ternary - source)
    if bits == 4:
        max_abs = tf.maximum(tf.reduce_max(tf.abs(source)), tf.constant(1e-6, dtype=source.dtype))
        scale = max_abs / tf.constant(7.0, dtype=source.dtype)
        quantized = tf.clip_by_value(tf.round(source / scale), -7.0, 7.0) * scale
        return source + tf.stop_gradient(quantized - source)
    max_abs = tf.maximum(tf.reduce_max(tf.abs(source)), tf.constant(1e-6, dtype=source.dtype))
    return tf.quantization.fake_quant_with_min_max_vars(
        source,
        -max_abs,
        max_abs,
        num_bits=bits,
        narrow_range=True,
    )


def fake_quant_weight(weight: tf.Tensor, bits: int) -> tf.Tensor:
    source = tf.cast(weight, tf.float32)
    quantized = tf.cond(
        QAT_ACTIVE,
        lambda: _quantize_for_bits(source, bits),
        lambda: source,
    )
    return tf.cast(quantized, weight.dtype)


class QEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, dims: int, bits: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dims = dims
        self.bits = bits

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="embedding",
            shape=(self.vocab_size, self.dims),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )

    def call(self, inputs, training=False):
        return tf.gather(fake_quant_weight(self.embedding, self.bits), inputs)


class QConv1D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, bits: int, dilation: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.bits = bits
        self.dilation = dilation

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs, training=False):
        output = tf.nn.conv1d(
            inputs,
            fake_quant_weight(self.kernel, self.bits),
            stride=1,
            padding="SAME",
            dilations=self.dilation,
        )
        return output + self.bias


class QDense(tf.keras.layers.Layer):
    def __init__(self, units: int, bits: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.bits = bits

    def build(self, input_shape):
        in_units = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_units, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs, training=False):
        return inputs @ fake_quant_weight(self.kernel, self.bits) + self.bias


class QDepthwiseConv1D(tf.keras.layers.Layer):
    """Depthwise 1D conv with quantized kernel.

    One kernel per input channel (channel_multiplier=1), so the output channel
    count equals the input channel count. Implemented via tf.nn.depthwise_conv2d
    with a height-1 reshape for portability.
    """

    def __init__(self, kernel_size: int, bits: int, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.bits = bits

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.in_channels = in_channels
        # Shape (kernel_size, 1, in_channels): per-channel 1D kernel.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, 1, in_channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(in_channels,), initializer="zeros", trainable=True
        )

    def call(self, inputs, training=False):
        # (B, T, C) -> (B, 1, T, C) for depthwise_conv2d's NHWC layout.
        x4 = tf.expand_dims(inputs, axis=1)
        kernel = fake_quant_weight(self.kernel, self.bits)
        # (kernel_size, 1, C) -> depthwise filter shape (1, kernel_size, C, 1).
        k4 = tf.expand_dims(tf.transpose(kernel, perm=[1, 0, 2]), axis=-1)
        out4 = tf.nn.depthwise_conv2d(
            x4, k4, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC"
        )
        return tf.squeeze(out4, axis=1) + self.bias


def build_conv_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 24, bits, name="q_embedding")(inputs)
    x = QConv1D(48, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(96, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = QDense(160, bits, name="q_dense_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_conv_wide_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 32, bits, name="q_embedding")(inputs)
    x = QConv1D(64, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(128, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = QDense(256, bits, name="q_dense_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_conv_xwide_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, bits, name="q_embedding")(inputs)
    x = QConv1D(80, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(160, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = QDense(224, bits, name="q_dense_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_conv_xwide_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, bits, name="q_embedding")(inputs)
    x = QConv1D(80, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(160, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    x = QDense(224, bits, name="q_dense_0")(pooled)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def hashed_bigram_features(inputs: tf.Tensor, bins: int) -> tf.Tensor:
    tokens = tf.cast(inputs, tf.int32)
    left = tokens[:, :-1]
    right = tokens[:, 1:]
    bucket_ids = tf.math.floormod(left * 263 + right * 17, bins)
    batch_size = tf.shape(tokens)[0]
    offsets = tf.reshape(tf.range(batch_size, dtype=tf.int32) * bins, (-1, 1))
    flat_ids = tf.reshape(bucket_ids + offsets, (-1,))
    counts = tf.math.bincount(
        flat_ids,
        minlength=batch_size * bins,
        maxlength=batch_size * bins,
        dtype=tf.float32,
    )
    return tf.reshape(counts, (batch_size, bins)) / float(TOKEN_LENGTH - 1)


_WORD_MASK = 0x00FFFFFF
# v4 only: bit 24 set if the source word had any uppercase characters. The
# hash itself is computed on case-folded bytes (so `hello`≡`Hello`), but the
# bit lives outside _WORD_MASK so it acts as a style sub-namespace within the
# word flag (no flag in bits 28-30). Distinguishes e.g. Python (mostly snake)
# from Java (lots of CamelCase) without splitting the embedding table.
_STYLE_BIT = 0x01000000
_PUNCT_FLAG = 0x10000000
_INDENT_FLAG = 0x20000000
_NUM_FLAG = 0x40000000
# v3 only: brackets get their own flag so `(` is always identifiable as `(`
# regardless of adjacent punct, and so the model can specialize on bracket
# patterns (e.g. lisp paren density, ts `<...>` generics, lua `[[...]]`).
_BRACKET_FLAG = 0x50000000
# v4 only: `"..."` string literals collapse to one token. Opening quote char
# in low byte (only `"` is recognized; `'` is too ambiguous — apostrophes in
# markdown/lisp would eat the rest of the input).
_STRING_FLAG = 0x70000000


def numpy_word_units_apply(tokens_np: np.ndarray, output_length: int = TOKEN_LENGTH) -> np.ndarray:
    out = np.full((tokens_np.shape[0], output_length), -1, dtype=np.int32)
    prime = 2654435761
    for row in range(tokens_np.shape[0]):
        word: list[int] = []
        out_pos = 0
        tokens = tokens_np[row]
        at_line_start = True
        indent_units = 0
        for col in range(tokens.shape[0]):
            value = int(tokens[col])
            if value >= PADDING_TOKEN:
                if word and out_pos < output_length:
                    h = 0
                    for b in word:
                        h = (h * prime + b) & 0xFFFFFFFF
                    out[row, out_pos] = h & _WORD_MASK
                    out_pos += 1
                word.clear()
                break
            is_word_char = (
                (48 <= value <= 57)
                or (97 <= value <= 122)
                or (65 <= value <= 90)
                or value == 95
            )
            if is_word_char:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                word.append(value)
                continue
            if word and out_pos < output_length:
                h = 0
                for b in word:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = h & _WORD_MASK
                out_pos += 1
            word.clear()
            if value == 10:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                if out_pos < output_length:
                    out[row, out_pos] = 10 | _PUNCT_FLAG
                    out_pos += 1
                at_line_start = True
                indent_units = 0
                continue
            if value == 13:
                continue
            if at_line_start and value in (32, 9):
                indent_units += 1 if value == 32 else 4
                continue
            if at_line_start and indent_units > 0 and out_pos < output_length:
                out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                out_pos += 1
            at_line_start = False
            indent_units = 0
            if value == 32 or value == 9:
                if out_pos < output_length:
                    last_was_space = out_pos > 0 and out[row, out_pos - 1] == (32 | _PUNCT_FLAG)
                    if not last_was_space:
                        out[row, out_pos] = 32 | _PUNCT_FLAG
                        out_pos += 1
                continue
            if out_pos < output_length:
                out[row, out_pos] = value | _PUNCT_FLAG
                out_pos += 1
        if word and out_pos < output_length:
            h = 0
            for b in word:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = h & _WORD_MASK
    return out


def numpy_word_units_apply_v2(tokens_np: np.ndarray, output_length: int = TOKEN_LENGTH) -> np.ndarray:
    """v2 tokenization: collapses number-runs and same-class punct-runs into single units.

    Changes vs v1:
      - Words are letters/underscore only [a-zA-Z_]; digits split off into numbers.
      - Number runs `\\d+(\\.\\d+)?` collapse to ONE unit tagged with _NUM_FLAG
        (hash captures decimal/integer shape, not the exact value).
      - Punct runs (any contiguous non-word, non-space, non-newline chars) hash
        as ONE unit. So `==`, `=>`, `->`, `::`, `**`, `===>` each take one slot.
      - Spaces collapse (same as v1). Newlines and indent stay separate units.
    """
    out = np.full((tokens_np.shape[0], output_length), -1, dtype=np.int32)
    prime = 2654435761
    for row in range(tokens_np.shape[0]):
        word: list[int] = []
        number: list[int] = []
        punct: list[int] = []
        out_pos = 0
        tokens = tokens_np[row]
        at_line_start = True
        indent_units = 0

        for col in range(tokens.shape[0]):
            value = int(tokens[col])
            if value >= PADDING_TOKEN:
                break
            is_letter = (
                (97 <= value <= 122)
                or (65 <= value <= 90)
                or value == 95
            )
            is_digit = 48 <= value <= 57
            is_newline = value == 10
            is_cr = value == 13
            is_space = value == 32 or value == 9

            if not is_letter and word and out_pos < output_length:
                h = 0
                for b in word:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = h & _WORD_MASK
                out_pos += 1
                word.clear()
            if not (is_digit or value == 46) and number and out_pos < output_length:
                h = 0
                for b in number:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
                out_pos += 1
                number.clear()
            need_flush_punct = is_letter or is_digit or is_space or is_newline or is_cr or value == 46
            if need_flush_punct and punct and out_pos < output_length:
                h = 0
                for b in punct:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
                out_pos += 1
                punct.clear()

            if is_letter:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                word.append(value)
                continue
            if is_digit or value == 46:
                # Digit, or `.` immediately following a digit (decimal point).
                if value == 46 and not number:
                    # Lone `.` is punctuation, not number.
                    if at_line_start and indent_units > 0 and out_pos < output_length:
                        out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                        out_pos += 1
                    at_line_start = False
                    indent_units = 0
                    punct.append(value)
                    continue
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                number.append(value)
                continue
            if is_newline:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                if out_pos < output_length:
                    out[row, out_pos] = 10 | _PUNCT_FLAG
                    out_pos += 1
                at_line_start = True
                indent_units = 0
                continue
            if is_cr:
                continue
            if at_line_start and is_space:
                indent_units += 1 if value == 32 else 4
                continue
            if at_line_start and indent_units > 0 and out_pos < output_length:
                out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                out_pos += 1
            at_line_start = False
            indent_units = 0
            if is_space:
                if out_pos < output_length:
                    last_was_space = out_pos > 0 and out[row, out_pos - 1] == (32 | _PUNCT_FLAG)
                    if not last_was_space:
                        out[row, out_pos] = 32 | _PUNCT_FLAG
                        out_pos += 1
                continue
            # Otherwise: punctuation char, accumulate in punct run.
            punct.append(value)
        # End of row: flush any pending word/number/punct.
        if word and out_pos < output_length:
            h = 0
            for b in word:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = h & _WORD_MASK
            out_pos += 1
        if number and out_pos < output_length:
            h = 0
            for b in number:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
            out_pos += 1
        if punct and out_pos < output_length:
            h = 0
            for b in punct:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
            out_pos += 1
    return out


_V3_BRACKET_BYTES = frozenset((40, 41, 91, 93, 123, 125))  # ( ) [ ] { }
# Excluded from bracket isolation: < > stay in punct so multi-char ops like
# `<<`, `>>`, `<=`, `>=`, `=>`, `->` retain their unique hash (v2 behavior).
# Only the unambiguous brackets get isolated.


def numpy_word_units_apply_v3(tokens_np: np.ndarray, output_length: int = TOKEN_LENGTH) -> np.ndarray:
    """v3 tokenization: v2 + case fold + bracket isolation.

    Changes vs v2:
      - Word chars are case-folded before hashing. `Function`, `function`, and
        `FUNCTION` all hash to the same word ID, halving effective vocabulary
        collision rate at no extra cost.
      - Brackets `(`, `)`, `[`, `]`, `{`, `}`, `<`, `>` always emit as a single
        _BRACKET_FLAG token with the bracket byte as the low byte. They flush
        any pending punct buffer and never merge with adjacent punct, so a `(`
        is always identifiable independent of surrounding context. This helps
        the model specialize on bracket-frequency signals that distinguish
        e.g. lisp (paren density), TypeScript generics (`<...>`), and lua
        `[[...]]` long-string syntax.
    """
    out = np.full((tokens_np.shape[0], output_length), -1, dtype=np.int32)
    prime = 2654435761
    for row in range(tokens_np.shape[0]):
        word: list[int] = []
        number: list[int] = []
        punct: list[int] = []
        out_pos = 0
        tokens = tokens_np[row]
        at_line_start = True
        indent_units = 0

        for col in range(tokens.shape[0]):
            value = int(tokens[col])
            if value >= PADDING_TOKEN:
                break
            # Case fold: ASCII A-Z -> a-z. Done at read time so all downstream
            # state (word buffer, punct/number buffers) sees lowercase.
            if 65 <= value <= 90:
                value += 32
            is_letter = (
                (97 <= value <= 122)
                or value == 95
            )
            is_digit = 48 <= value <= 57
            is_newline = value == 10
            is_cr = value == 13
            is_space = value == 32 or value == 9
            is_bracket = value in _V3_BRACKET_BYTES

            if not is_letter and word and out_pos < output_length:
                h = 0
                for b in word:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = h & _WORD_MASK
                out_pos += 1
                word.clear()
            if not (is_digit or value == 46) and number and out_pos < output_length:
                h = 0
                for b in number:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
                out_pos += 1
                number.clear()
            # v3: brackets also force punct flush (they don't merge into runs).
            need_flush_punct = (
                is_letter or is_digit or is_space or is_newline or is_cr
                or is_bracket or value == 46
            )
            if need_flush_punct and punct and out_pos < output_length:
                h = 0
                for b in punct:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
                out_pos += 1
                punct.clear()

            if is_letter:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                word.append(value)
                continue
            if is_digit or value == 46:
                if value == 46 and not number:
                    if at_line_start and indent_units > 0 and out_pos < output_length:
                        out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                        out_pos += 1
                    at_line_start = False
                    indent_units = 0
                    punct.append(value)
                    continue
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                number.append(value)
                continue
            if is_newline:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                if out_pos < output_length:
                    out[row, out_pos] = 10 | _PUNCT_FLAG
                    out_pos += 1
                at_line_start = True
                indent_units = 0
                continue
            if is_cr:
                continue
            if at_line_start and is_space:
                indent_units += 1 if value == 32 else 4
                continue
            if at_line_start and indent_units > 0 and out_pos < output_length:
                out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                out_pos += 1
            at_line_start = False
            indent_units = 0
            if is_space:
                if out_pos < output_length:
                    last_was_space = out_pos > 0 and out[row, out_pos - 1] == (32 | _PUNCT_FLAG)
                    if not last_was_space:
                        out[row, out_pos] = 32 | _PUNCT_FLAG
                        out_pos += 1
                continue
            # v3: brackets emit immediately as their own token; never merged.
            if is_bracket:
                if out_pos < output_length:
                    out[row, out_pos] = value | _BRACKET_FLAG
                    out_pos += 1
                continue
            # Otherwise: punctuation char, accumulate in punct run.
            punct.append(value)
        # End of row: flush any pending word/number/punct.
        if word and out_pos < output_length:
            h = 0
            for b in word:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = h & _WORD_MASK
            out_pos += 1
        if number and out_pos < output_length:
            h = 0
            for b in number:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
            out_pos += 1
        if punct and out_pos < output_length:
            h = 0
            for b in punct:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
            out_pos += 1
    return out


def numpy_word_units_apply_v4(tokens_np: np.ndarray, output_length: int = TOKEN_LENGTH) -> np.ndarray:
    """v4 = v3 + `"..."` string compression + identifier style bit.

    Changes vs v3:
      - `"..."` string literals collapse to a single _STRING_FLAG token (with
        `"` in the low byte). Backslash-escaped chars inside strings are
        skipped (e.g. `\\"` doesn't terminate). Unterminated strings absorb to
        end of input. Only `"` triggers compression — `'` and backtick are
        left as plain chars to avoid eating apostrophes in markdown / lisp
        quotes / OCaml lifetimes / etc.
      - Words still case-fold for the hash (so `hello`≡`Hello`), but a single
        _STYLE_BIT (bit 24) is set if the original word contained any uppercase
        chars. Distinguishes Java/C# (lots of CamelCase) from Python (snake)
        without splitting the embedding namespace.
    """
    out = np.full((tokens_np.shape[0], output_length), -1, dtype=np.int32)
    prime = 2654435761
    for row in range(tokens_np.shape[0]):
        word: list[int] = []
        word_had_upper = False
        number: list[int] = []
        punct: list[int] = []
        out_pos = 0
        tokens = tokens_np[row]
        at_line_start = True
        indent_units = 0
        in_string = False
        string_escape = False

        for col in range(tokens.shape[0]):
            value = int(tokens[col])
            if value >= PADDING_TOKEN:
                break

            # In-string mode: skip everything until matching close `"`.
            if in_string:
                if string_escape:
                    string_escape = False
                elif value == 92:  # backslash
                    string_escape = True
                elif value == 34:  # closing `"`
                    if out_pos < output_length:
                        out[row, out_pos] = _STRING_FLAG | 34
                        out_pos += 1
                    in_string = False
                continue

            saw_upper_now = 65 <= value <= 90
            if saw_upper_now:
                value += 32  # case fold ASCII A-Z -> a-z
            is_letter = (97 <= value <= 122) or value == 95
            is_digit = 48 <= value <= 57
            is_newline = value == 10
            is_cr = value == 13
            is_space = value == 32 or value == 9
            is_bracket = value in _V3_BRACKET_BYTES
            is_dquote = value == 34

            if not is_letter and word and out_pos < output_length:
                h = 0
                for b in word:
                    h = (h * prime + b) & 0xFFFFFFFF
                style = _STYLE_BIT if word_had_upper else 0
                out[row, out_pos] = (h & _WORD_MASK) | style
                out_pos += 1
                word.clear()
                word_had_upper = False
            if not (is_digit or value == 46) and number and out_pos < output_length:
                h = 0
                for b in number:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
                out_pos += 1
                number.clear()
            need_flush_punct = (
                is_letter or is_digit or is_space or is_newline or is_cr
                or is_bracket or is_dquote or value == 46
            )
            if need_flush_punct and punct and out_pos < output_length:
                h = 0
                for b in punct:
                    h = (h * prime + b) & 0xFFFFFFFF
                out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
                out_pos += 1
                punct.clear()

            if is_letter:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                if saw_upper_now:
                    word_had_upper = True
                word.append(value)
                continue
            if is_digit or value == 46:
                if value == 46 and not number:
                    if at_line_start and indent_units > 0 and out_pos < output_length:
                        out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                        out_pos += 1
                    at_line_start = False
                    indent_units = 0
                    punct.append(value)
                    continue
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                number.append(value)
                continue
            if is_newline:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                if out_pos < output_length:
                    out[row, out_pos] = 10 | _PUNCT_FLAG
                    out_pos += 1
                at_line_start = True
                indent_units = 0
                continue
            if is_cr:
                continue
            if at_line_start and is_space:
                indent_units += 1 if value == 32 else 4
                continue
            if at_line_start and indent_units > 0 and out_pos < output_length:
                out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                out_pos += 1
            at_line_start = False
            indent_units = 0
            if is_space:
                if out_pos < output_length:
                    last_was_space = out_pos > 0 and out[row, out_pos - 1] == (32 | _PUNCT_FLAG)
                    if not last_was_space:
                        out[row, out_pos] = 32 | _PUNCT_FLAG
                        out_pos += 1
                continue
            if is_bracket:
                if out_pos < output_length:
                    out[row, out_pos] = value | _BRACKET_FLAG
                    out_pos += 1
                continue
            if is_dquote:
                # Enter string mode. Don't emit the opening quote separately;
                # the eventual _STRING_FLAG token captures that we saw one.
                in_string = True
                string_escape = False
                continue
            punct.append(value)
        # End of row: flush pending word/number/punct (and any unterminated
        # string — emit the _STRING_FLAG so the model still sees that something
        # was a string, even if truncated by the input window).
        if in_string and out_pos < output_length:
            out[row, out_pos] = _STRING_FLAG | 34
            out_pos += 1
        if word and out_pos < output_length:
            h = 0
            for b in word:
                h = (h * prime + b) & 0xFFFFFFFF
            style = _STYLE_BIT if word_had_upper else 0
            out[row, out_pos] = (h & _WORD_MASK) | style
            out_pos += 1
        if number and out_pos < output_length:
            h = 0
            for b in number:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
            out_pos += 1
        if punct and out_pos < output_length:
            h = 0
            for b in punct:
                h = (h * prime + b) & 0xFFFFFFFF
            out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
            out_pos += 1
    return out


def unit_window_bitset_features(
    unit_ids: tf.Tensor,
    bins: int,
    hash_count: int,
    window_size: int,
    max_windows: int,
    clip_to_one: bool = True,
) -> tf.Tensor:
    seq_used = max_windows * window_size
    truncated = unit_ids[:, :seq_used]
    valid = tf.cast(truncated >= 0, tf.float32)
    safe = tf.where(truncated >= 0, truncated, tf.zeros_like(truncated))
    safe_64 = tf.cast(safe, tf.int64)
    batch_size = tf.shape(unit_ids)[0]
    positions = tf.range(seq_used, dtype=tf.int32)
    window_idx = tf.minimum(positions // window_size, max_windows - 1)
    primes = (2654435761, 2246822519, 3266489917, 668265263)
    mask32 = tf.constant(0xFFFFFFFF, dtype=tf.int64)
    batch_offsets = tf.reshape(tf.range(batch_size, dtype=tf.int32) * max_windows * bins, (-1, 1))
    window_offsets = tf.reshape(window_idx * bins, (1, -1))
    flat_parts = []
    weight_parts = []
    for hi in range(hash_count):
        p1 = tf.constant(primes[hi % len(primes)], dtype=tf.int64)
        p2 = tf.constant(primes[(hi + 1) % len(primes)], dtype=tf.int64)
        h = tf.bitwise.bitwise_and(safe_64 * p1, mask32)
        h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 13))
        h = tf.bitwise.bitwise_and(h * p2, mask32)
        bucket_ids = tf.cast(tf.math.floormod(h, bins), tf.int32)
        flat_parts.append(tf.reshape(batch_offsets + window_offsets + bucket_ids, (-1,)))
        weight_parts.append(tf.reshape(valid, (-1,)))
    flat_ids = tf.concat(flat_parts, axis=0)
    weights = tf.concat(weight_parts, axis=0)
    counts = tf.math.unsorted_segment_sum(weights, flat_ids, batch_size * max_windows * bins)
    counts = tf.reshape(counts, (batch_size, max_windows, bins))
    if clip_to_one:
        counts = tf.clip_by_value(counts, 0.0, 1.0)
    return counts


def chunked_bigram_bitset_features(
    inputs: tf.Tensor,
    bins: int = 1024,
    chunk_size: int = 16,
    hash_count: int = 2,
) -> tf.Tensor:
    tokens = tf.cast(inputs, tf.int64)
    left = tokens[:, :-1]
    right = tokens[:, 1:]
    valid_pair = tf.logical_and(left < PADDING_TOKEN, right < PADDING_TOKEN)
    bigrams = left * 256 + right
    batch_size = tf.shape(tokens)[0]
    pair_count = tf.shape(bigrams)[1]
    chunk_count = (TOKEN_LENGTH + chunk_size - 1) // chunk_size
    positions = tf.range(pair_count, dtype=tf.int32)
    chunk_ids = tf.minimum(positions // chunk_size, chunk_count - 1)
    batch_offsets = tf.reshape(tf.range(batch_size, dtype=tf.int32) * chunk_count * bins, (-1, 1))
    chunk_offsets = tf.reshape(chunk_ids * bins, (1, -1))
    primes = (2654435761, 2246822519, 3266489917, 668265263)
    mask32 = tf.constant(0xFFFFFFFF, dtype=tf.int64)
    flat_parts = []
    weight_parts = []
    for index in range(hash_count):
        p1 = tf.constant(primes[index % len(primes)], dtype=tf.int64)
        p2 = tf.constant(primes[(index + 1) % len(primes)], dtype=tf.int64)
        hashed = tf.bitwise.bitwise_and(bigrams * p1, mask32)
        hashed = tf.bitwise.bitwise_xor(hashed, tf.bitwise.right_shift(hashed, 13))
        hashed = tf.bitwise.bitwise_and(hashed * p2, mask32)
        bucket_ids = tf.cast(tf.math.floormod(hashed, bins), tf.int32)
        flat_parts.append(tf.reshape(batch_offsets + chunk_offsets + bucket_ids, (-1,)))
        weight_parts.append(tf.reshape(tf.cast(valid_pair, tf.float32), (-1,)))
    flat_ids = tf.concat(flat_parts, axis=0)
    weights = tf.concat(weight_parts, axis=0)
    counts = tf.math.unsorted_segment_sum(weights, flat_ids, batch_size * chunk_count * bins)
    bitset = tf.clip_by_value(tf.reshape(counts, (batch_size, chunk_count, bins)), 0.0, 1.0)
    return bitset


def build_conv_xwide_hash_hidden_units_model(
    classes: int,
    bits: int,
    dense_units: int,
    hash_units: int,
    *,
    hidden_dim: int = 512,
    embedding: int = 40,
    conv0: int = 80,
    conv1: int = 160,
    proj_bits: int | None = None,
    conv_bits: int | None = None,
    dense_bits: int | None = None,
    output_bits: int | None = None,
) -> tf.keras.Model:
    pb = proj_bits if proj_bits is not None else bits
    cb = conv_bits if conv_bits is not None else bits
    db = dense_bits if dense_bits is not None else bits
    ob = output_bits if output_bits is not None else bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, embedding, pb, name="q_embedding")(inputs)
    x = QConv1D(conv0, 7, cb, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(conv1, 5, cb, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, bits, name="q_hidden_project")(pooled)
    conv_features = QDense(dense_units, db, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(lambda value: hashed_bigram_features(value, 256), name="hash_bigram")(inputs)
    hash_features = QDense(hash_units, db, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, ob, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_hash_hidden_model(classes: int, bits: int, hidden_dim: int = 512) -> tf.keras.Model:
    return build_conv_xwide_hash_hidden_units_model(classes, bits, dense_units=176, hash_units=64, hidden_dim=hidden_dim)


def build_conv_xwide_big_mp_hidden_model(classes: int, bits: int, hidden_dim: int = 512) -> tf.keras.Model:
    # Scaled-up byte-CNN with mixed precision: 4-bit embed/output, 2-bit conv/dense.
    return build_conv_xwide_hash_hidden_units_model(
        classes, bits,
        dense_units=320, hash_units=128,
        hidden_dim=hidden_dim,
        embedding=40, conv0=160, conv1=320,
        proj_bits=4, conv_bits=2, dense_bits=2, output_bits=4,
    )


def build_conv_xwide_med_mp_hidden_model(classes: int, bits: int, hidden_dim: int = 512) -> tf.keras.Model:
    # Medium MP byte-CNN ~100KB target.
    return build_conv_xwide_hash_hidden_units_model(
        classes, bits,
        dense_units=240, hash_units=96,
        hidden_dim=hidden_dim,
        embedding=40, conv0=120, conv1=240,
        proj_bits=4, conv_bits=2, dense_bits=2, output_bits=4,
    )


def build_conv_xwide_small_mp_hidden_model(classes: int, bits: int, hidden_dim: int = 512) -> tf.keras.Model:
    # Small MP byte-CNN ~70KB target.
    return build_conv_xwide_hash_hidden_units_model(
        classes, bits,
        dense_units=176, hash_units=80,
        hidden_dim=hidden_dim,
        embedding=32, conv0=96, conv1=192,
        proj_bits=4, conv_bits=2, dense_bits=2, output_bits=4,
    )


def build_conv_xwide_tiny_mp_hidden_model(classes: int, bits: int, hidden_dim: int = 512) -> tf.keras.Model:
    # Tiny MP byte-CNN ~45KB target.
    return build_conv_xwide_hash_hidden_units_model(
        classes, bits,
        dense_units=128, hash_units=64,
        hidden_dim=hidden_dim,
        embedding=24, conv0=64, conv1=128,
        proj_bits=4, conv_bits=2, dense_bits=2, output_bits=4,
    )


def build_conv_xwide_dilated_tcn_tiny_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Tiny TCN-style byte-CNN with dilated 1D convs.

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Three conv blocks with
    dilations [1, 2, 4] grow the receptive field exponentially at fixed param
    cost so longer multi-byte patterns are still captured at ~45KB export size.
    """
    del bits  # mixed precision is fully specified per-layer below
    pb = 4
    cb = 2
    db = 2
    ob = 4
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 24, pb, name="q_embedding")(inputs)
    # Block 1: dilation 1, then downsample by 4 to keep activations cheap.
    x = QConv1D(64, 5, cb, dilation=1, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    # Block 2: dilation 2 — receptive field grows without extra params.
    x = QConv1D(96, 5, cb, dilation=2, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    # Block 3: dilation 4 — covers ~28 bytes of pre-pool context per stack.
    x = QConv1D(160, 3, cb, dilation=4, name="q_conv_2")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, pb, name="q_hidden_project")(pooled)
    conv_features = QDense(128, db, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(64, db, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, ob, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_separable_se_tiny_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Tiny depthwise-separable byte-CNN with squeeze-and-excite.

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Each conv stage is a
    depthwise spatial conv followed by a 1x1 pointwise QConv1D — cheaper than
    a dense conv of the same width, so we can afford wider channels at ~45KB.
    A squeeze-and-excite (SE) block after the first stage adds cheap channel
    attention to recover some of the inductive bias depthwise loses.
    """
    del bits  # mixed precision is fully specified per-layer below
    pb = 4
    cb = 2
    db = 2
    ob = 4
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 24, pb, name="q_embedding")(inputs)
    # Stage 1: depthwise (kernel 7) over the 24 embedding channels, then
    # pointwise expansion to 96 channels (1x1 conv).
    x = QDepthwiseConv1D(7, cb, name="q_dw_conv_0")(x)
    x = QConv1D(96, 1, cb, name="q_pw_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    # Squeeze-and-excite over the 96-channel sequence: GAP -> small dense ->
    # dense back to 96 -> sigmoid -> per-channel scale.
    se = tf.keras.layers.GlobalAveragePooling1D(name="se0_squeeze")(x)
    se = QDense(12, db, name="q_se0_reduce")(se)  # 96 / 8 = 12
    se = tf.keras.layers.Activation(tf.nn.relu)(se)
    se = QDense(96, db, name="q_se0_expand")(se)
    se = tf.keras.layers.Activation(tf.nn.sigmoid)(se)
    se = tf.keras.layers.Reshape((1, 96), name="se0_reshape")(se)
    x = tf.keras.layers.Multiply(name="se0_scale")([x, se])

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    # Stage 2: depthwise (kernel 5) on 96 channels, pointwise expand to 192.
    x = QDepthwiseConv1D(5, cb, name="q_dw_conv_1")(x)
    x = QConv1D(192, 1, cb, name="q_pw_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, pb, name="q_hidden_project")(pooled)
    conv_features = QDense(128, db, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(64, db, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, ob, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_tiny_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Tiny inception-style multi-scale parallel-branch byte-CNN, ~45KB target.

    Instead of a single conv at one kernel size, run THREE parallel convs at
    kernels {3, 5, 7} on the embedding so short multi-byte ops (e.g. ``::``),
    medium tokens (e.g. ``void``), and longer patterns (e.g. ``#include``) are
    captured in one shot. A second {3, 5} parallel block then refines the
    pooled features. Each branch is narrow (32/48 channels) so total cost stays
    comparable to a single wider conv. Mixed precision: 4-bit emb/hidden/output,
    2-bit conv/dense.
    """
    del bits  # mixed precision is fully specified per-layer below
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 24, 4, name="q_embedding")(inputs)

    # Multi-scale block 1: parallel kernels {3, 5, 7} on the (B, 2048, 24) embedding.
    branch_a = QConv1D(32, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(32, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(32, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])  # (B, 2048, 96)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 512, 96)

    # Multi-scale block 2: parallel kernels {3, 5} on the pooled features.
    branch_d = QConv1D(48, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(48, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 96)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 192)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(128, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(64, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_tiny_plus_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Tiny-plus inception-style byte-CNN at ~50-55KB target.

    Same shape as multi-scale-tiny but with three accuracy-preserving tweaks:
    1. Embedding 28-dim (vs 24): richer per-byte features feed all three branches.
    2. Block-1 convs at 4-bit (vs 2-bit): the first conv sees raw embedding and
       benefits most from precision; later block-2 convs stay at 2-bit.
    3. Slightly wider block-2 + dense (56/160 vs 48/128): more capacity on the
       discriminative head where the model commits to a class.
    """
    del bits  # mixed precision is fully specified per-layer below
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 28, 4, name="q_embedding")(inputs)

    # Multi-scale block 1: parallel kernels {3, 5, 7} on the (B, 2048, 28) embedding.
    # 4-bit conv here for precision near the input (export pack supports 2 or 4).
    branch_a = QConv1D(36, 3, 4, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(36, 5, 4, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(36, 7, 4, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])  # (B, 2048, 108)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 512, 108)

    # Multi-scale block 2: parallel kernels {3, 5} on the pooled features. 2-bit.
    branch_d = QConv1D(56, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(56, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 112)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 224)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(160, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(80, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_sep_se_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Medium depthwise-separable byte-CNN with squeeze-and-excite (~100KB target).

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Same structure as the
    tiny variant but channels scaled ~3x: emb=40, dw_0(k=7)+pw_0=160, SE
    reduction=20 (160/8), dw_1(k=5)+pw_1=384, dense=256, hash=96. Slight
    deviation from spec on pw_1/dense to land near 100KB exported size.
    """
    del bits  # mixed precision is fully specified per-layer below
    pb = 4
    cb = 2
    db = 2
    ob = 4
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, pb, name="q_embedding")(inputs)
    # Stage 1: depthwise (kernel 7) over the 40 embedding channels, then
    # pointwise expansion to 160 channels (1x1 conv).
    x = QDepthwiseConv1D(7, cb, name="q_dw_conv_0")(x)
    x = QConv1D(160, 1, cb, name="q_pw_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    # Squeeze-and-excite over the 160-channel sequence: GAP -> small dense ->
    # dense back to 160 -> sigmoid -> per-channel scale. Reduction = 160/8 = 20.
    se = tf.keras.layers.GlobalAveragePooling1D(name="se0_squeeze")(x)
    se = QDense(20, db, name="q_se0_reduce")(se)
    se = tf.keras.layers.Activation(tf.nn.relu)(se)
    se = QDense(160, db, name="q_se0_expand")(se)
    se = tf.keras.layers.Activation(tf.nn.sigmoid)(se)
    se = tf.keras.layers.Reshape((1, 160), name="se0_reshape")(se)
    x = tf.keras.layers.Multiply(name="se0_scale")([x, se])

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    # Stage 2: depthwise (kernel 5) on 160 channels, pointwise expand to 384.
    x = QDepthwiseConv1D(5, cb, name="q_dw_conv_1")(x)
    x = QConv1D(384, 1, cb, name="q_pw_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, pb, name="q_hidden_project")(pooled)
    conv_features = QDense(256, db, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, db, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, ob, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_dilated_tcn_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Medium TCN-style byte-CNN with dilated 1D convs (~100KB target).

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Three dilated conv
    blocks scaled up from the tiny variant. Channels are 112/160/192 (a hair
    below the spec 128/192/256, which would land around 127KB) to stay within
    the 95-105KB envelope. Kernels 5/5/3 with dilations 1/2/4.
    """
    del bits  # mixed precision is fully specified per-layer below
    pb = 4
    cb = 2
    db = 2
    ob = 4
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, pb, name="q_embedding")(inputs)
    # Block 1: dilation 1, then downsample by 4 to keep activations cheap.
    x = QConv1D(112, 5, cb, dilation=1, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    # Block 2: dilation 2 — receptive field grows without extra params.
    x = QConv1D(160, 5, cb, dilation=2, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    # Block 3: dilation 4 — covers ~28 bytes of pre-pool context per stack.
    x = QConv1D(192, 3, cb, dilation=4, name="q_conv_2")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, pb, name="q_hidden_project")(pooled)
    conv_features = QDense(192, db, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, db, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, ob, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Medium inception-style multi-scale parallel-branch byte-CNN (~100KB target).

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Block 1 has three
    parallel kernels {3, 5, 7} on a 40-channel embedding, each producing 64
    channels (concat = 192). Block 2 has two parallel kernels {3, 5} each
    producing 96 channels (concat = 192). Dense bumped to 224 (vs spec 192) to
    land near 100KB exported size.
    """
    del bits  # mixed precision is fully specified per-layer below
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    # Multi-scale block 1: parallel kernels {3, 5, 7} on the (B, 2048, 40) embedding.
    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])  # (B, 2048, 192)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 512, 192)

    # Multi-scale block 2: parallel kernels {3, 5} on the pooled features.
    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 384)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(224, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_big_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Big inception-style multi-scale parallel-branch byte-CNN (~150KB target).

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Scaled-up version of
    `build_conv_xwide_multiscale_med_hidden_model`: emb=48, block 1 c=96 each
    (288 concat), block 2 c=144 each (288 concat), dense=320, hash=128.
    """
    del bits  # mixed precision is fully specified per-layer below
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 48, 4, name="q_embedding")(inputs)

    # Multi-scale block 1: parallel kernels {3, 5, 7} on the (B, 2048, 48) embedding.
    branch_a = QConv1D(96, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(96, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(96, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])  # (B, 2048, 288)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 512, 288)

    # Multi-scale block 2: parallel kernels {3, 5} on the pooled features.
    branch_d = QConv1D(144, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(144, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 288)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 576)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(320, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(128, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_deep_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Deep inception-style multi-scale byte-CNN with 3 blocks (~95KB target).

    Mixed precision (4-bit emb/output, 2-bit conv/dense). emb=40. Block 1: 3
    branches kernels {3,5,7} c=48 each (144 concat) -> AvgPool(4) -> (B, 512, 144).
    Block 2: 2 branches kernels {3,5} c=72 each (144 concat) -> AvgPool(4) ->
    (B, 128, 144). Block 3: 2 branches kernels {3,5} c=96 each (192 concat) ->
    (B, 128, 192). Pool concat -> 384 -> dense=224, hash=96.
    """
    del bits  # mixed precision is fully specified per-layer below
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    # Multi-scale block 1: parallel kernels {3, 5, 7} on the (B, 2048, 40) embedding.
    branch_a = QConv1D(48, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(48, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(48, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])  # (B, 2048, 144)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 512, 144)

    # Multi-scale block 2: parallel kernels {3, 5}.
    branch_d = QConv1D(72, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(72, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 144)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 128, 144)

    # Multi-scale block 3: parallel kernels {3, 5}.
    branch_f = QConv1D(96, 3, 2, name="q_conv_2f")(x)
    branch_f = tf.keras.layers.Activation(tf.nn.gelu)(branch_f)
    branch_g = QConv1D(96, 5, 2, name="q_conv_2g")(x)
    branch_g = tf.keras.layers.Activation(tf.nn.gelu)(branch_g)
    x = tf.keras.layers.Concatenate()([branch_f, branch_g])  # (B, 128, 192)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 384)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(224, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_xbig_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Extra-big inception-style multi-scale parallel-branch byte-CNN (~200KB target).

    Mixed precision (4-bit emb/output, 2-bit conv/dense). Even bigger than -big:
    emb=48, block 1 c=128 each (384 concat), block 2 c=192 each (384 concat),
    dense=384, hash=160.
    """
    del bits  # mixed precision is fully specified per-layer below
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 48, 4, name="q_embedding")(inputs)

    # Multi-scale block 1: parallel kernels {3, 5, 7} on the (B, 2048, 48) embedding.
    branch_a = QConv1D(128, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(128, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(128, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])  # (B, 2048, 384)

    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)  # (B, 512, 384)

    # Multi-scale block 2: parallel kernels {3, 5} on the pooled features.
    branch_d = QConv1D(192, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(192, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 384)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 768)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(384, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(160, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


class QLearnedQueries(QEmbedding):
    """Stores N learned query vectors with QAT fake-quant.

    Subclass of QEmbedding so it serializes via the existing export path
    (export reads `layer.embedding`). Ignores its input; returns a
    (1, num_queries, key_dim) tensor of quantized queries.
    """

    def __init__(self, num_queries: int, key_dim: int, bits: int, **kwargs):
        super().__init__(num_queries, key_dim, bits, **kwargs)

    def call(self, inputs, training=False):
        del inputs
        return tf.expand_dims(fake_quant_weight(self.embedding, self.bits), axis=0)

    def compute_output_shape(self, input_shape):
        return (1, self.vocab_size, self.dims)


def _attention_pool(
    x: tf.Tensor,
    *,
    num_queries: int,
    key_dim: int,
    value_dim: int,
    bits: int,
    name: str,
) -> tf.Tensor:
    """Learned-query attention pool over a sequence (B, T, C) -> (B, num_queries*value_dim).

    Replaces global mean/max pool with N learned queries that attend over the
    sequence. Query matrix is stored as a QLearnedQueries layer (exports via
    the QEmbedding path); k/v are QDense projections of the conv features.
    """
    keys = QDense(key_dim, bits, name=f"{name}_k")(x)  # (B, T, K)
    values = QDense(value_dim, bits, name=f"{name}_v")(x)  # (B, T, V)

    queries = QLearnedQueries(num_queries, key_dim, bits, name=f"{name}_q")(x)  # (1, Q, K)

    scale = float(key_dim) ** 0.5
    scores = tf.keras.layers.Lambda(
        lambda inputs: tf.matmul(inputs[0], inputs[1], transpose_b=True) / scale,
        output_shape=(num_queries, x.shape[1]),
        name=f"{name}_scores",
    )([queries, keys])  # (B, Q, T)
    attn = tf.keras.layers.Softmax(axis=-1, name=f"{name}_softmax")(scores)
    pooled = tf.keras.layers.Lambda(
        lambda inputs: tf.matmul(inputs[0], inputs[1]),
        output_shape=(num_queries, value_dim),
        name=f"{name}_apply",
    )([attn, values])  # (B, Q, V)
    return tf.keras.layers.Reshape((num_queries * value_dim,), name=f"{name}_flat")(pooled)


def build_conv_xwide_multiscale_attn_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-med + attention pooling for long-range dependencies (~95KB target).

    Same multi-scale conv backbone as multiscale-med (kernels {3,5,7} block-1,
    {3,5} block-2). Replaces global mean+max pool with a learned-query attention
    pool: 4 learnable queries attend over the (B, 512, 192) conv feature map to
    produce 4 pooled vectors of 64 dim each, concatenated to 256 dim.
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    pooled = _attention_pool(
        x, num_queries=4, key_dim=32, value_dim=64, bits=2, name="q_attn"
    )  # (B, 256)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled_all = tf.keras.layers.Concatenate()([pooled, max_pool, avg_pool])  # (B, 640)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled_all)

    conv_features = QDense(192, 2, name="q_dense_0")(pooled_all)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_attn_big_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-big + attention pooling for long-range dependencies (~165KB target).

    Scaled-up version: emb=48, block-1 c=96 each, block-2 c=144 each. Attention
    pool uses 6 queries × 80 value dim (480-d output).
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 48, 4, name="q_embedding")(inputs)

    branch_a = QConv1D(96, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(96, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(96, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(144, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(144, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 288)

    pooled = _attention_pool(
        x, num_queries=6, key_dim=48, value_dim=80, bits=2, name="q_attn"
    )  # (B, 480)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled_all = tf.keras.layers.Concatenate()([pooled, max_pool, avg_pool])  # (B, 1056)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled_all)

    conv_features = QDense(256, 2, name="q_dense_0")(pooled_all)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(128, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def _pyramid_pool(x: tf.Tensor, name: str) -> tf.Tensor:
    """Multi-scale temporal pyramid pool: global mean+max, plus 2-region and 4-region means.

    Captures features at multiple temporal granularities. For (B, T, C):
      - global mean (B, C)
      - global max (B, C)
      - 2-region mean (B, 2*C)
      - 4-region mean (B, 4*C)
    Concatenated to (B, 8*C).
    """
    glob_mean = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_g_mean")(x)
    glob_max = tf.keras.layers.GlobalMaxPooling1D(name=f"{name}_g_max")(x)

    def split_mean(value, parts):
        # value: (B, T, C). Reshape to (B, parts, T/parts, C) and mean over T/parts.
        T = value.shape[1]
        C = value.shape[-1]
        chunk = T // parts
        truncated = value[:, : chunk * parts, :]
        reshaped = tf.reshape(truncated, (-1, parts, chunk, C))
        return tf.reduce_mean(reshaped, axis=2)

    two_region = tf.keras.layers.Lambda(
        lambda v: split_mean(v, 2),
        output_shape=(2, x.shape[-1]),
        name=f"{name}_2region",
    )(x)
    two_region = tf.keras.layers.Reshape((2 * int(x.shape[-1]),), name=f"{name}_2region_flat")(two_region)
    four_region = tf.keras.layers.Lambda(
        lambda v: split_mean(v, 4),
        output_shape=(4, x.shape[-1]),
        name=f"{name}_4region",
    )(x)
    four_region = tf.keras.layers.Reshape((4 * int(x.shape[-1]),), name=f"{name}_4region_flat")(four_region)
    return tf.keras.layers.Concatenate(name=f"{name}_concat")(
        [glob_mean, glob_max, two_region, four_region]
    )


def build_conv_xwide_multiscale_pyramid_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-med + temporal pyramid pool (~100KB target).

    Same conv backbone. Replaces global mean+max pool with a temporal pyramid:
    global mean (192), global max (192), 2-region mean (2*192=384), 4-region
    mean (4*192=768). Concat = 1536 dim. Then dense_0 reduces to 192.
    Captures features at multiple temporal granularities (file start vs middle
    vs end), giving the model explicit positional information without a
    positional encoding.
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    pooled = _pyramid_pool(x, name="pyr")  # (B, 1536)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(192, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def hashed_trigram_features(inputs: tf.Tensor, bins: int) -> tf.Tensor:
    """Bag-of-trigrams via hashing — same idea as hashed_bigram_features but for 3-byte windows."""
    tokens = tf.cast(inputs, tf.int32)
    a = tokens[:, :-2]
    b = tokens[:, 1:-1]
    c = tokens[:, 2:]
    bucket_ids = tf.math.floormod(a * 65537 + b * 257 + c * 7, bins)
    batch_size = tf.shape(tokens)[0]
    offsets = tf.reshape(tf.range(batch_size, dtype=tf.int32) * bins, (-1, 1))
    flat_ids = tf.reshape(bucket_ids + offsets, (-1,))
    counts = tf.math.bincount(
        flat_ids,
        minlength=batch_size * bins,
        maxlength=batch_size * bins,
        dtype=tf.float32,
    )
    return tf.reshape(counts, (batch_size, bins)) / float(TOKEN_LENGTH - 2)


def build_conv_xwide_multiscale_ngram_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-med + bigram + trigram hash features (~105KB target).

    Adds an explicit trigram bag-of-features path alongside the existing bigram
    one. Trigrams capture richer local patterns than bigrams (e.g., "def ",
    "fn ", "pub ", "<?p" Vs random 2-byte sequences). Hashed into 256 buckets.
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 384)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(192, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    bigram_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    bigram_features = QDense(96, 2, name="q_hash_project")(bigram_features)
    bigram_features = tf.keras.layers.Activation(tf.nn.gelu)(bigram_features)

    trigram_features = tf.keras.layers.Lambda(
        lambda value: hashed_trigram_features(value, 256), name="hash_trigram"
    )(inputs)
    trigram_features = QDense(96, 2, name="q_trigram_project")(trigram_features)
    trigram_features = tf.keras.layers.Activation(tf.nn.gelu)(trigram_features)

    merged = tf.keras.layers.Concatenate()([conv_features, bigram_features, trigram_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_multiscale_kitchen_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-med + attention pool + bigram + trigram features (~115KB target).

    Combines the most-promising additions: attention pool for long-range
    dependencies + bigram + trigram bag-of-features. Kitchen-sink test of
    whether the wins compose.
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    attn_pool = _attention_pool(
        x, num_queries=4, key_dim=32, value_dim=64, bits=2, name="q_attn"
    )  # (B, 256)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([attn_pool, max_pool, avg_pool])  # (B, 640)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(160, 2, name="q_dense_0")(pooled)  # reduced from 192 to fit budget
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    bigram_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    bigram_features = QDense(80, 2, name="q_hash_project")(bigram_features)
    bigram_features = tf.keras.layers.Activation(tf.nn.gelu)(bigram_features)

    trigram_features = tf.keras.layers.Lambda(
        lambda value: hashed_trigram_features(value, 256), name="hash_trigram"
    )(inputs)
    trigram_features = QDense(80, 2, name="q_trigram_project")(trigram_features)
    trigram_features = tf.keras.layers.Activation(tf.nn.gelu)(trigram_features)

    merged = tf.keras.layers.Concatenate()([conv_features, bigram_features, trigram_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def _sinusoidal_position_table(length: int, dim: int) -> tf.Tensor:
    """Standard sinusoidal positional encoding (length, dim). Stored fp32."""
    positions = np.arange(length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * -(np.log(10000.0) / dim))
    angles = positions * div_term[None, :]
    pe = np.zeros((length, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return tf.constant(pe, dtype=tf.float32)


def build_conv_xwide_multiscale_pos_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-med + sinusoidal position encoding added to byte embeddings.

    Same arch as multiscale-med but adds a fixed (non-learned) sinusoidal
    position encoding to the byte embeddings before conv. Gives the model
    explicit positional awareness with zero parameter cost (the encoding is
    not trained, just added).
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    pos_table = _sinusoidal_position_table(TOKEN_LENGTH, 40)
    x = tf.keras.layers.Lambda(
        lambda value: value + tf.cast(pos_table, value.dtype),
        name="add_position",
    )(x)

    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 384)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(224, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def _multi_head_self_attention(
    x: tf.Tensor,
    *,
    dim: int,
    num_heads: int,
    bits: int,
    name: str,
) -> tf.Tensor:
    """Multi-head self-attention (B, T, dim) -> (B, T, dim) with QAT projections.

    Splits dim into num_heads heads each of size dim/num_heads. No layer norm
    (handled by residual scaling at the call site). Q, K, V, O projections are
    quantized via QDense.
    """
    assert dim % num_heads == 0, f"dim {dim} must divide num_heads {num_heads}"
    head_dim = dim // num_heads

    qkv_in = x  # (B, T, dim)
    q = QDense(dim, bits, name=f"{name}_q")(qkv_in)
    k = QDense(dim, bits, name=f"{name}_k")(qkv_in)
    v = QDense(dim, bits, name=f"{name}_v")(qkv_in)

    scale = float(head_dim) ** 0.5

    def attend(inputs):
        q_, k_, v_ = inputs
        bsz = tf.shape(q_)[0]
        seq = tf.shape(q_)[1]
        q4 = tf.reshape(q_, (bsz, seq, num_heads, head_dim))
        k4 = tf.reshape(k_, (bsz, seq, num_heads, head_dim))
        v4 = tf.reshape(v_, (bsz, seq, num_heads, head_dim))
        # (B, H, T, D)
        q4 = tf.transpose(q4, (0, 2, 1, 3))
        k4 = tf.transpose(k4, (0, 2, 1, 3))
        v4 = tf.transpose(v4, (0, 2, 1, 3))
        scores = tf.matmul(q4, k4, transpose_b=True) / scale  # (B, H, T, T)
        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, v4)  # (B, H, T, D)
        out = tf.transpose(out, (0, 2, 1, 3))  # (B, T, H, D)
        return tf.reshape(out, (bsz, seq, dim))

    attended = tf.keras.layers.Lambda(
        attend, output_shape=(x.shape[1], dim), name=f"{name}_attend"
    )([q, k, v])
    return QDense(dim, bits, name=f"{name}_o")(attended)


def _transformer_block(
    x: tf.Tensor,
    *,
    dim: int,
    num_heads: int,
    ffn_dim: int,
    bits: int,
    name: str,
) -> tf.Tensor:
    """Pre-activation transformer block (no LayerNorm).

    y = x + gelu(MHSA(x))
    z = y + gelu(FFN(y))
    """
    attn = _multi_head_self_attention(x, dim=dim, num_heads=num_heads, bits=bits, name=f"{name}_mhsa")
    attn = tf.keras.layers.Activation(tf.nn.gelu)(attn)
    y = tf.keras.layers.Add(name=f"{name}_res1")([x, attn])

    ffn = QDense(ffn_dim, bits, name=f"{name}_ffn0")(y)
    ffn = tf.keras.layers.Activation(tf.nn.gelu)(ffn)
    ffn = QDense(dim, bits, name=f"{name}_ffn1")(ffn)
    ffn = tf.keras.layers.Activation(tf.nn.gelu)(ffn)
    z = tf.keras.layers.Add(name=f"{name}_res2")([y, ffn])
    return z


class QLearnedPositions(QEmbedding):
    """Learned position embedding stored via QEmbedding's `embedding` weight.

    Shape (max_length, dim). On call, returns (1, T, dim) sliced to the input
    sequence length T (so it broadcasts over batch when added to (B, T, dim)).
    """

    def __init__(self, max_length: int, dim: int, bits: int, **kwargs):
        super().__init__(max_length, dim, bits, **kwargs)
        self.max_length = max_length

    def call(self, inputs, training=False):
        seq = tf.shape(inputs)[1]
        positions = fake_quant_weight(self.embedding, self.bits)  # (L, D)
        return tf.expand_dims(positions[:seq], axis=0)  # (1, T, D)

    def compute_output_shape(self, input_shape):
        return (1, input_shape[1], self.dims)


def build_conv_xwide_xformer_tiny_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Conv-stem + tiny transformer (~80KB target).

    Byte embedding -> conv stem (kernel 7, gelu) -> AvgPool(16) downsamples
    sequence to 128. Add learned position embedding. 2 transformer blocks
    (dim=80, 4 heads, FFN expansion 2x). Mean+max pool + bigram features.
    First proper attention-based architecture (vs current attn-pool which only
    adds attention to the pooled output).
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 64, 4, name="q_embedding")(inputs)

    x = QConv1D(80, 7, 2, name="q_conv_stem")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=16)(x)  # (B, 128, 80)

    pos = QLearnedPositions(128, 80, 4, name="q_pos_emb")(x)
    x = tf.keras.layers.Lambda(
        lambda inp: inp[0] + tf.cast(inp[1], inp[0].dtype),
        output_shape=(128, 80),
        name="add_pos",
    )([x, pos])

    x = _transformer_block(x, dim=80, num_heads=4, ffn_dim=128, bits=2, name="q_xfb0")
    x = _transformer_block(x, dim=80, num_heads=4, ffn_dim=128, bits=2, name="q_xfb1")

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 160)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(160, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(80, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_xformer_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Conv-stem + transformer (~110KB target).

    Scaled up from xformer-tiny: emb=80, conv-stem 96 channels, transformer
    dim=96 with 4 heads, FFN dim=192, 2 blocks. AvgPool(16) downsamples to
    seq=128 keeping attention O(T^2) tractable.
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 80, 4, name="q_embedding")(inputs)

    x = QConv1D(96, 7, 2, name="q_conv_stem")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=16)(x)  # (B, 128, 96)

    pos = QLearnedPositions(128, 96, 4, name="q_pos_emb")(x)
    x = tf.keras.layers.Lambda(
        lambda inp: inp[0] + tf.cast(inp[1], inp[0].dtype),
        output_shape=(128, 96),
        name="add_pos",
    )([x, pos])

    x = _transformer_block(x, dim=96, num_heads=4, ffn_dim=192, bits=2, name="q_xfb0")
    x = _transformer_block(x, dim=96, num_heads=4, ffn_dim=192, bits=2, name="q_xfb1")

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 192)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(192, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, 4, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def _multi_prototype_output(
    merged: tf.Tensor,
    *,
    classes: int,
    prototypes_per_class: int,
    bits: int,
    name: str,
) -> tf.Tensor:
    """K prototypes per class, max over prototypes -> per-class logit.

    Replaces the standard linear classifier `QDense(classes, ...)` with a
    K-prototype variant: `QDense(classes*K, ...)`, reshape to (B, classes, K),
    take max over K. Lets each class have multiple "templates" to match against,
    which can capture multi-modal class distributions (e.g., short vs long C
    files have different feature signatures).
    """
    flat = QDense(classes * prototypes_per_class, bits, name=name)(merged)
    reshape = tf.keras.layers.Reshape((classes, prototypes_per_class))(flat)
    return tf.keras.layers.Lambda(
        lambda v: tf.reduce_max(v, axis=2),
        output_shape=(classes,),
        name=f"{name}_max",
    )(reshape)


def build_conv_xwide_multiscale_proto_med_hidden_model(
    classes: int, bits: int, hidden_dim: int = 512
) -> tf.keras.Model:
    """Multi-scale-med + 2 prototypes per class in classifier head (~106KB target).

    Same conv backbone and mean+max pooling as multi-scale-med. The final
    classifier is replaced with a 2-prototype-per-class head: 134 prototypes,
    max pooled per class. Tests whether multi-modal class distributions are a
    factor in the 91% test parity ceiling — confusable pairs (vhdl/verilog,
    scss/css) might benefit from per-class multi-prototype matching.
    """
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, 4, name="q_embedding")(inputs)

    branch_a = QConv1D(64, 3, 2, name="q_conv_0a")(x)
    branch_a = tf.keras.layers.Activation(tf.nn.gelu)(branch_a)
    branch_b = QConv1D(64, 5, 2, name="q_conv_0b")(x)
    branch_b = tf.keras.layers.Activation(tf.nn.gelu)(branch_b)
    branch_c = QConv1D(64, 7, 2, name="q_conv_0c")(x)
    branch_c = tf.keras.layers.Activation(tf.nn.gelu)(branch_c)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b, branch_c])
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)

    branch_d = QConv1D(96, 3, 2, name="q_conv_1d")(x)
    branch_d = tf.keras.layers.Activation(tf.nn.gelu)(branch_d)
    branch_e = QConv1D(96, 5, 2, name="q_conv_1e")(x)
    branch_e = tf.keras.layers.Activation(tf.nn.gelu)(branch_e)
    x = tf.keras.layers.Concatenate()([branch_d, branch_e])  # (B, 512, 192)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])  # (B, 384)

    hidden = QDense(hidden_dim, 4, name="q_hidden_project")(pooled)

    conv_features = QDense(224, 2, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(
        lambda value: hashed_bigram_features(value, 256), name="hash_bigram"
    )(inputs)
    hash_features = QDense(96, 2, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    merged = tf.keras.layers.Concatenate()([conv_features, hash_features])  # (B, 320)
    outputs = _multi_prototype_output(
        merged, classes=classes, prototypes_per_class=2, bits=4, name="q_output"
    )
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_hash80_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    return build_conv_xwide_hash_hidden_units_model(classes, bits, dense_units=176, hash_units=80)


def build_conv_xwide_hash96_dense144_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    return build_conv_xwide_hash_hidden_units_model(classes, bits, dense_units=144, hash_units=96)


def build_conv_xwide_hash96_dense160_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    return build_conv_xwide_hash_hidden_units_model(classes, bits, dense_units=160, hash_units=96)


def build_conv_xwide_hash128_dense112_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    return build_conv_xwide_hash_hidden_units_model(classes, bits, dense_units=112, hash_units=128)


def build_conv_e48_c80_c152_hash_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 48, bits, name="q_embedding")(inputs)
    x = QConv1D(80, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(152, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    conv_features = QDense(176, bits, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(lambda value: hashed_bigram_features(value, 256), name="hash_bigram")(inputs)
    hash_features = QDense(64, bits, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


_WORDWIN_CONFIGS: dict[str, dict[str, int]] = {
    "wordwin-w3-m192-b512-k2-hidden": {
        "embedding": 40, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 512, "hash_count": 2, "window_size": 3, "max_windows": 192,
    },
    "wordwin-w3-m192-b512-k3-hidden": {
        "embedding": 40, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 512, "hash_count": 3, "window_size": 3, "max_windows": 192,
    },
    "wordwin-w3-m192-b1024-k2-hidden": {
        "embedding": 32, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
    },
    "wordwin-w3-m192-b1024-k3-hidden": {
        "embedding": 32, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 1024, "hash_count": 3, "window_size": 3, "max_windows": 192,
    },
    "wordwin-w3-m192-b512-k2-hash48-hidden": {
        "embedding": 36, "conv0": 80, "conv1": 152, "dense": 128,
        "bins": 512, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "hash_branch_units": 48,
    },
    "wordwin-w2-m256-b512-k2-hidden": {
        "embedding": 40, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 512, "hash_count": 2, "window_size": 2, "max_windows": 256,
    },
    "wordwin-w2-m192-b1024-k2-hidden": {
        "embedding": 32, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 1024, "hash_count": 2, "window_size": 2, "max_windows": 192,
    },
    "wordwin-w3-m128-b1024-k3-hidden": {
        "embedding": 36, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 1024, "hash_count": 3, "window_size": 3, "max_windows": 128,
    },
    "wordwin-w3-m576-b1024-k2-hidden": {
        "embedding": 32, "conv0": 80, "conv1": 152, "dense": 144,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 576,
    },
    "wordwin-w3-m576-b2048-k2-e20-hidden": {
        "embedding": 20, "conv0": 80, "conv1": 144, "dense": 144,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
    },
    "wordwin-w3-m576-b2048-k2-e16-hidden": {
        "embedding": 16, "conv0": 88, "conv1": 160, "dense": 160,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
    },
    "wordwin-w3-m576-b2048-k2-big-count-hidden": {
        # Bigger model + count features (no bitset clip). ~125KB.
        "embedding": 32, "conv0": 96, "conv1": 176, "dense": 192,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
        "clip_to_one": False,
    },
    "wordwin-w3-m576-b2048-k2-big-hidden": {
        # Bigger model, bitset (clipped) baseline at ~125KB to isolate count vs bitset.
        "embedding": 32, "conv0": 96, "conv1": 176, "dense": 192,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
    },
    "wordwin-w3-m576-b2048-k2-big2-mp-hidden": {
        # Mixed precision: 4-bit projection + output, 2-bit conv0/conv1/dense.
        # Doubles conv/dense channels at same byte budget (~170KB).
        "embedding": 32, "conv0": 160, "conv1": 320, "dense": 320,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
        "clip_to_one": False,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
    },
    "wordwin-w3-m576-b2048-k2-big2-hidden": {
        # All-4-bit reference for big2 (no mixed precision); much bigger budget ~280KB.
        # Useful as a slope reference if MP version underperforms.
        "embedding": 32, "conv0": 160, "conv1": 320, "dense": 320,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
        "clip_to_one": False,
    },
    "wordwin-w3-m576-b2048-k2-big3-mp-hidden": {
        # Even bigger: 1.4x channels vs big2-mp. Mixed precision keeps size ~240KB.
        "embedding": 40, "conv0": 192, "conv1": 384, "dense": 384,
        "bins": 2048, "hash_count": 2, "window_size": 3, "max_windows": 576,
        "clip_to_one": False,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-hidden": {
        # Tiny mixed-precision wordwin targeting ~45-50KB. 4-bit chunk-project +
        # output, 2-bit conv/dense. Word-level bitset features only (no byte branch).
        "embedding": 24, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
    },
    "wordwin-w3-m192-b512-k2-mp-tiny-hidden": {
        # Smaller-bins variant: 512 bins keep chunk_project cost down for the same
        # channel count, freeing budget for deeper conv if needed. ~40KB.
        "embedding": 28, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 512, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-max-hidden": {
        # Same as tiny-b1024 but MaxPool instead of AvgPool between convs — better
        # preserves "did this bucket appear anywhere in this group" semantics for
        # bitset-derived features.
        "embedding": 24, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-max-count-hidden": {
        # Same as tiny-max but clip_to_one=False so the bitset retains MULTIPLICITY
        # (a word that appears twice in a window contributes 2 to its bin instead
        # of 1). Closer to a sum-over-window of per-word embeddings.
        "embedding": 24, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "clip_to_one": False,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-deeper-hidden": {
        # 3 conv layers (adds conv_2=3 kernel) for richer hierarchical features
        # at modest size cost.
        "embedding": 24, "conv0": 64, "conv1": 96, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "conv2": 96, "pool_op": "max",
    },
    "wordwin-w2-m288-b1024-k2-mp-tiny-max-hidden": {
        # Smaller window (2 units) + more windows (288) for finer temporal granularity.
        "embedding": 24, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 2, "max_windows": 288,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordwin-w3-m192-b1024-k3-mp-tiny-max-hidden": {
        # 3 hash functions instead of 2 — denser bitset, more collision-resistant.
        "embedding": 24, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 1024, "hash_count": 3, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-pool2-hidden": {
        # Less aggressive pool (pool_size=2 instead of 4) keeps more temporal detail.
        # conv1 gets a (96, 120) input instead of (48, 120).
        "embedding": 24, "conv0": 64, "conv1": 96, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool_size": 2,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-randproj-hidden": {
        # Fixed random hypersphere projection replaces q_chunk_project — saves
        # the ~12KB the dense projection would cost. Reinvest into wider
        # downstream: conv0 64→96, conv1 120→160, dense 96→160. Still under 50KB.
        "embedding": 48, "conv0": 96, "conv1": 160, "dense": 160,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "random_proj": True,
    },
    "wordwin-w3-m192-b1024-k2-mp-randproj-bigemb-hidden": {
        # Bigger random projection (96-dim) — more capacity to disambiguate
        # bin combinations geometrically, since projection itself costs 0 bytes.
        "embedding": 96, "conv0": 96, "conv1": 144, "dense": 144,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "random_proj": True,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-randproj-match-hidden": {
        # Same downstream as learned-tiny baseline, but random hypersphere
        # projection swaps in for q_chunk_project — pure ~12KB savings, no
        # reinvestment. Lands ~20-22KB.
        "embedding": 24, "conv0": 64, "conv1": 120, "dense": 96,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "random_proj": True,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-csketch-hidden": {
        # Multi-head count sketch projection (K=4 heads of 12 dims each)
        # replaces dense random Gaussian. Each bin contributes to exactly K=4
        # output dims with random signs, preserving the sparsity of the input
        # bitset (~6 bits/window) instead of scattering it. Zero stored params;
        # downstream channels match the random-proj-tiny baseline for A/B.
        "embedding": 48, "conv0": 96, "conv1": 160, "dense": 160,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "count_sketch_heads": 4,
    },
    "wordwin-w3-m192-b1024-k2-mp-csketch-bigemb-hidden": {
        # Larger count sketch (K=8 heads of 12 dims = 96-dim output). More
        # head diversity → better bin recoverability from joint pattern.
        "embedding": 96, "conv0": 96, "conv1": 144, "dense": 144,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "count_sketch_heads": 8,
    },
    "wordwin-w3-m192-b1024-k2-mp-tiny-randproj-multiscale-hidden": {
        # Random hypersphere projection + multi-scale parallel conv_0 (kernels
        # {3,5,7}). Saved chunk_project budget reinvested into 3 parallel kernel
        # heads instead of one wider conv. Target ~45-50KB.
        "embedding": 32, "conv0": 36, "conv1": 144, "dense": 144,
        "bins": 1024, "hash_count": 2, "window_size": 3, "max_windows": 192,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "random_proj": True,
        "conv0_kernels": (3, 5, 7),
    },
}


def build_wordwin_by_name(classes: int, bits: int, architecture: str, hidden_dim: int = 512) -> tf.keras.Model:
    config = _WORDWIN_CONFIGS.get(architecture)
    if config is None:
        raise ValueError(f"unknown wordwin architecture: {architecture}")
    return build_conv_wordwin_hidden_model(classes, bits, hidden_dim=hidden_dim, **config)


def build_conv_wordwin_hidden_model(
    classes: int,
    bits: int,
    *,
    embedding: int,
    conv0: int,
    conv1: int,
    dense: int,
    bins: int,
    hash_count: int,
    window_size: int,
    max_windows: int,
    hash_branch_units: int = 0,
    conv2: int = 0,
    clip_to_one: bool = True,
    proj_bits: int | None = None,
    conv_bits: int | None = None,
    dense_bits: int | None = None,
    output_bits: int | None = None,
    hidden_dim: int = 512,
    pool_op: str = "avg",  # "avg" or "max" between conv_0 and conv_1
    pool_size: int = 4,
    conv0_kernel: int = 7,
    conv1_kernel: int = 5,
    random_proj: bool = False,
    random_proj_seed: int = 0xABCDEF,
    count_sketch_heads: int = 0,  # if >0, replace projection with K-head count sketch
    conv0_kernels: tuple[int, ...] | None = None,  # if set, build multi-scale parallel conv0
    conv1_kernels: tuple[int, ...] | None = None,  # if set, build multi-scale parallel conv1
) -> tf.keras.Model:
    pb = proj_bits if proj_bits is not None else bits
    cb = conv_bits if conv_bits is not None else bits
    db = dense_bits if dense_bits is not None else bits
    ob = output_bits if output_bits is not None else bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    bitset = tf.keras.layers.Lambda(
        lambda value: unit_window_bitset_features(value, bins, hash_count, window_size, max_windows, clip_to_one=clip_to_one),
        name=f"wordwin_w{window_size}_m{max_windows}_b{bins}_h{hash_count}{'_count' if not clip_to_one else ''}",
    )(inputs)
    if count_sketch_heads > 0:
        # Multi-head count sketch (Charikar 2002 / HashEmbedding): each bin
        # maps to exactly K dims (one per head) with a random ±1 sign. Output
        # preserves sparsity (~K active dims per active bin) vs Gaussian RP
        # which scatters every bin across all `embedding` dims. Zero params —
        # regenerated from `random_proj_seed` at load time.
        if embedding % count_sketch_heads != 0:
            raise ValueError(
                f"embedding ({embedding}) must be divisible by count_sketch_heads ({count_sketch_heads})"
            )
        K = count_sketch_heads
        sub_dim = embedding // K
        rng = np.random.default_rng(random_proj_seed)
        proj_np = np.zeros((bins, embedding), dtype=np.float32)
        inv_sqrt_k = 1.0 / np.sqrt(K)
        rows = np.arange(bins)
        for k in range(K):
            head_dims = rng.integers(0, sub_dim, size=bins)
            signs = rng.choice([-1.0, 1.0], size=bins).astype(np.float32)
            cols = k * sub_dim + head_dims
            np.add.at(proj_np, (rows, cols), signs * inv_sqrt_k)
        proj_const = tf.constant(proj_np, dtype=tf.float32)
        def _apply_proj(b: tf.Tensor) -> tf.Tensor:
            b32 = tf.cast(b, tf.float32)
            return tf.cast(b32 @ proj_const, b.dtype)
        x = tf.keras.layers.Lambda(
            _apply_proj,
            name="count_sketch_proj",
            output_shape=(max_windows, embedding),
        )(bitset)
    elif random_proj:
        # Fixed random hypersphere projection: each bin gets a deterministic
        # unit-norm vector in `embedding`-dim space. The projection has no
        # trainable parameters and is regenerated from `random_proj_seed` at
        # load time, so it costs 0 bytes in the deployed .bin (export only
        # serializes QEmbedding/QConv1D/QDense layer weights, not Lambdas).
        rng = np.random.default_rng(random_proj_seed)
        proj_np = rng.standard_normal((bins, embedding)).astype(np.float32)
        proj_np /= np.maximum(np.linalg.norm(proj_np, axis=-1, keepdims=True), 1e-9)
        proj_const = tf.constant(proj_np, dtype=tf.float32)
        def _apply_proj(b: tf.Tensor) -> tf.Tensor:
            b32 = tf.cast(b, tf.float32)
            return tf.cast(b32 @ proj_const, b.dtype)
        x = tf.keras.layers.Lambda(
            _apply_proj,
            name="random_proj",
            output_shape=(max_windows, embedding),
        )(bitset)
    else:
        x = QDense(embedding, pb, name="q_chunk_project")(bitset)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    # Multi-scale parallel conv_0 branches (each branch produces conv0 channels,
    # concatenated). If conv0_kernels is None, falls back to single conv0_kernel.
    if conv0_kernels:
        branches = []
        for i, k in enumerate(conv0_kernels):
            br = QConv1D(conv0, k, cb, name=f"q_conv_0_{chr(ord('a') + i)}")(x)
            br = tf.keras.layers.Activation(tf.nn.gelu)(br)
            branches.append(br)
        x = tf.keras.layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    else:
        x = QConv1D(conv0, conv0_kernel, cb, name="q_conv_0")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    if pool_size > 1:
        if pool_op == "max":
            x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        else:
            x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(x)
    if conv1_kernels:
        branches = []
        for i, k in enumerate(conv1_kernels):
            br = QConv1D(conv1, k, cb, name=f"q_conv_1_{chr(ord('a') + i)}")(x)
            br = tf.keras.layers.Activation(tf.nn.gelu)(br)
            branches.append(br)
        x = tf.keras.layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    else:
        x = QConv1D(conv1, conv1_kernel, cb, name="q_conv_1")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    if conv2 > 0:
        x = QConv1D(conv2, 3, cb, name="q_conv_2")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, bits, name="q_hidden_project")(pooled)
    conv_features = QDense(dense, db, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    if hash_branch_units > 0:
        hash_features = tf.keras.layers.Lambda(
            lambda value: hashed_bigram_features(value, 256), name="hash_bigram",
        )(inputs)
        hash_features = QDense(hash_branch_units, bits, name="q_hash_project")(hash_features)
        hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
        merged = tf.keras.layers.Concatenate()([conv_features, hash_features])
    else:
        merged = conv_features
    outputs = QDense(classes, ob, name="q_output")(merged)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_chunkbitset1024_e40_c72_c152_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    bitset = tf.keras.layers.Lambda(
        lambda value: chunked_bigram_bitset_features(value, bins=1024, chunk_size=16, hash_count=2),
        name="chunkbitset_16_1024_h2",
    )(inputs)
    x = QDense(40, bits, name="q_chunk_project")(bitset)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = QConv1D(72, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(152, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    x = QDense(176, bits, name="q_dense_0")(pooled)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_chunkbitset2048_e36_c72_c136_d144_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    bitset = tf.keras.layers.Lambda(
        lambda value: chunked_bigram_bitset_features(value, bins=2048, chunk_size=64, hash_count=3),
        name="chunkbitset_64_2048_h3",
    )(inputs)
    x = QDense(36, bits, name="q_chunk_project")(bitset)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = QConv1D(72, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(136, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    x = QDense(144, bits, name="q_dense_0")(pooled)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def hash_unit_indices(unit_ids: tf.Tensor, bins: int, hash_count: int, max_units: int) -> tf.Tensor:
    """Hash each unit_id into K buckets in [0, bins). Returns (batch, max_units, K) int32.

    Padding positions (unit_ids < 0) are mapped to bucket 0 by the safe gather; the
    caller must zero them out via a mask applied to the embedding result.
    """
    truncated = unit_ids[:, :max_units]
    safe = tf.where(truncated >= 0, truncated, tf.zeros_like(truncated))
    safe64 = tf.cast(safe, tf.int64)
    primes = (2654435761, 2246822519, 3266489917, 668265263)
    mask32 = tf.constant(0xFFFFFFFF, dtype=tf.int64)
    parts = []
    for hi in range(hash_count):
        p1 = tf.constant(primes[hi % len(primes)], dtype=tf.int64)
        p2 = tf.constant(primes[(hi + 1) % len(primes)], dtype=tf.int64)
        h = tf.bitwise.bitwise_and(safe64 * p1, mask32)
        h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 13))
        h = tf.bitwise.bitwise_and(h * p2, mask32)
        parts.append(tf.cast(tf.math.floormod(h, bins), tf.int32))
    return tf.stack(parts, axis=-1)


def build_word_seq_hashembed_hidden_model(
    classes: int,
    bits: int,
    *,
    embedding: int,
    conv0: int,
    conv1: int,
    dense: int,
    bins: int,
    hash_count: int,
    max_units: int,
    proj_bits: int | None = None,
    conv_bits: int | None = None,
    dense_bits: int | None = None,
    output_bits: int | None = None,
    hidden_dim: int = 512,
    pool_op: str = "max",
    pool_size: int = 4,
    conv0_kernel: int = 7,
    conv1_kernel: int = 5,
    conv0_kernels: tuple[int, ...] | None = None,
    conv1_kernels: tuple[int, ...] | None = None,
    conv2: int | None = None,
    conv2_kernel: int = 3,
    pool2_size: int = 2,
    conv3: int | None = None,
    conv3_kernel: int = 3,
    pool3_size: int = 1,
    conv3_residual: bool = False,
    residual_scale: float = 0.5,
    random_proj: bool = False,
    random_proj_seed: int = 0xABCDEF,
) -> tf.keras.Model:
    pb = proj_bits if proj_bits is not None else bits
    cb = conv_bits if conv_bits is not None else bits
    db = dense_bits if dense_bits is not None else bits
    ob = output_bits if output_bits is not None else bits
    # shape=(None,) accepts any input length up to TOKEN_LENGTH — required for
    # length-bucketed batching where each batch trims to its bucket max.
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)

    indices = tf.keras.layers.Lambda(
        lambda value: hash_unit_indices(value, bins, hash_count, max_units),
        name=f"hash_indices_b{bins}_h{hash_count}_m{max_units}",
        dtype="int32",
    )(inputs)
    valid_mask = tf.keras.layers.Lambda(
        lambda value: tf.cast(value[:, :max_units] >= 0, tf.float32),
        name=f"unit_mask_m{max_units}",
    )(inputs)

    if random_proj:
        # Frozen random embedding table: deterministic unit-norm vectors per
        # bin. Zero exported bytes (regenerated from random_proj_seed at load).
        # Trades the trained 12 KB QEmbedding for budget reinvestment elsewhere
        # (wider convs, more dense). Per prior wordwin experience this loses
        # ~6 pp vs trained projection unless the freed budget recovers it.
        rng = np.random.default_rng(random_proj_seed)
        table_np = rng.standard_normal((bins, embedding)).astype(np.float32)
        table_np /= np.maximum(np.linalg.norm(table_np, axis=-1, keepdims=True), 1e-9)
        table_const = tf.constant(table_np, dtype=tf.float32)
        embed_per_hash = tf.keras.layers.Lambda(
            lambda idx: tf.gather(table_const, idx),
            name="random_embed_lookup",
            output_shape=(max_units, hash_count, embedding),
        )(indices)
    else:
        table = QEmbedding(bins, embedding, pb, name="q_hash_embedding")
        embed_per_hash = table(indices)  # (batch, T, K, emb), T <= max_units
    x = tf.keras.layers.Lambda(
        lambda value: tf.reduce_sum(value, axis=-2),
        name="hash_embed_sum",
    )(embed_per_hash)
    # Mask out padding positions (where unit_id == -1).
    x = tf.keras.layers.Lambda(
        lambda args: args[0] * tf.cast(tf.expand_dims(args[1], -1), args[0].dtype),
        name="apply_pad_mask",
    )([x, valid_mask])
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    if conv0_kernels:
        branches = []
        for i, k in enumerate(conv0_kernels):
            br = QConv1D(conv0, k, cb, name=f"q_conv_0_{chr(ord('a') + i)}")(x)
            br = tf.keras.layers.Activation(tf.nn.gelu)(br)
            branches.append(br)
        x = tf.keras.layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    else:
        x = QConv1D(conv0, conv0_kernel, cb, name="q_conv_0")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    if pool_size > 1:
        if pool_op == "max":
            x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        else:
            x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(x)
    if conv1_kernels:
        branches = []
        for i, k in enumerate(conv1_kernels):
            br = QConv1D(conv1, k, cb, name=f"q_conv_1_{chr(ord('a') + i)}")(x)
            br = tf.keras.layers.Activation(tf.nn.gelu)(br)
            branches.append(br)
        x = tf.keras.layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    else:
        x = QConv1D(conv1, conv1_kernel, cb, name="q_conv_1")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    if conv2 is not None:
        if pool2_size > 1:
            if pool_op == "max":
                x = tf.keras.layers.MaxPooling1D(pool_size=pool2_size)(x)
            else:
                x = tf.keras.layers.AveragePooling1D(pool_size=pool2_size)(x)
        x = QConv1D(conv2, conv2_kernel, cb, name="q_conv_2")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    if conv3 is not None:
        if pool3_size > 1:
            if pool_op == "max":
                x = tf.keras.layers.MaxPooling1D(pool_size=pool3_size)(x)
            else:
                x = tf.keras.layers.AveragePooling1D(pool_size=pool3_size)(x)
        residual = x
        x = QConv1D(conv3, conv3_kernel, cb, name="q_conv_3")(x)
        x = tf.keras.layers.Activation(tf.nn.gelu)(x)
        if conv3_residual:
            residual_channels = int(residual.shape[-1])
            if residual_channels != conv3:
                raise ValueError(
                    "conv3_residual requires conv3 to match the incoming channel count "
                    f"({conv3} != {residual_channels})"
                )
            x = tf.keras.layers.Lambda(
                lambda pair: (pair[0] + pair[1]) * residual_scale,
                name="conv3_residual_blend",
            )([x, residual])
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(hidden_dim, bits, name="q_hidden_project")(pooled)
    feat = QDense(dense, db, name="q_dense_0")(pooled)
    feat = tf.keras.layers.Activation(tf.nn.gelu)(feat)
    outputs = QDense(classes, ob, name="q_output")(feat)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


_WORDSEQ_JSON_PREFIX = "wordseq-json:"


def architecture_uses_word_units(architecture: str) -> bool:
    return architecture.startswith(("wordseq-", "wordwin-", _WORDSEQ_JSON_PREFIX))


def wordseq_config_for_architecture(architecture: str) -> dict:
    if architecture.startswith(_WORDSEQ_JSON_PREFIX):
        config = json.loads(architecture[len(_WORDSEQ_JSON_PREFIX):])
        if not isinstance(config, dict):
            raise ValueError("wordseq-json architecture must decode to an object")
        return dict(config)
    config = _WORDSEQ_CONFIGS.get(architecture)
    if config is None:
        raise ValueError(f"unknown wordseq architecture: {architecture}")
    return dict(config)


_WORDSEQ_CONFIGS: dict[str, dict[str, int]] = {
    "wordseq-b1024-k2-m576-tiny-mp-hidden": {
        # 576-position word sequence with K=2 hashed embedding lookups into a
        # shared (1024, emb) table. Matches the byte-CNN tiny-plus byte budget
        # (~46-50KB) but at word granularity — each conv kernel-step covers
        # ~20-30 bytes of input instead of 1 byte.
        "embedding": 24, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 1024, "hash_count": 2, "max_units": 576,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k2-m576-tiny-mp-multiscale-hidden": {
        # Multi-scale parallel conv_0 (kernels 3, 5, 7) for richer n-gram coverage.
        "embedding": 24, "conv0": 32, "conv1": 144, "dense": 128,
        "bins": 1024, "hash_count": 2, "max_units": 576,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "conv0_kernels": (3, 5, 7),
    },
    "wordseq-b2048-k2-m576-tiny-mp-hidden": {
        # Bigger hash table (2048 buckets) → fewer collisions. Same downstream.
        "embedding": 16, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 2048, "hash_count": 2, "max_units": 576,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k3-m576-tiny-mp-hidden": {
        # K=3 hashes → cleaner bin recovery at cost of 1 extra lookup per token.
        "embedding": 24, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 1024, "hash_count": 3, "max_units": 576,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k2-m1536-tiny-mp-hidden": {
        # FULL coverage: max_units=1536 captures every unit_id in the file,
        # not just the first 576. p90 unit-length is ~947, so this stops
        # truncating ~60% of files mid-stream. Params identical to the
        # 576-variant; only compute / activations grow.
        "embedding": 24, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 1024, "hash_count": 2, "max_units": 1536,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k2-m1024-tiny-mp-hidden": {
        # Designed for the v2 tokenizer: ~98% of files fit in 1024 units post-
        # punct-run collapse + number-run collapse. Half the compute of m1536
        # without giving up coverage.
        "embedding": 24, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 1024, "hash_count": 2, "max_units": 1024,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k2-m2048-tiny-mp-hidden": {
        # Matches the cache row width (TOKEN_LENGTH=2048): true zero-truncation
        # across every file in the corpus. Params identical to m576/m1536.
        "embedding": 24, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 1024, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k2-m1536-tiny-mp-pool8-hidden": {
        # Same full coverage but pool_size=8 (instead of 4) so conv_1 sees a
        # similar-length sequence (192) to the 576-variant after pool_4.
        "embedding": 24, "conv0": 80, "conv1": 160, "dense": 128,
        "bins": 1024, "hash_count": 2, "max_units": 1536,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool_size": 8,
    },
    "wordseq-b1024-k2-m2048-mini-mp-hidden": {
        # Half the channels of -tiny; ~22-25 KB exported. Same m2048 coverage.
        "embedding": 16, "conv0": 48, "conv1": 96, "dense": 80,
        "bins": 1024, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b512-k2-m2048-micro-mp-hidden": {
        # ~10-12 KB exported: half-sized hash table, smaller everything.
        "embedding": 12, "conv0": 32, "conv1": 64, "dense": 48,
        "bins": 512, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b512-k2-m2048-pico-mp-hidden": {
        # ~6 KB exported — about an eighth of tiny.
        "embedding": 8, "conv0": 24, "conv1": 48, "dense": 32,
        "bins": 512, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b256-k2-m2048-pico-mp-hidden": {
        # ~4 KB exported — minimal. 256-bin hash table is tight on collisions.
        "embedding": 8, "conv0": 24, "conv1": 48, "dense": 32,
        "bins": 256, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k3-m2048-nano-3conv-hidden": {
        # Nano with K=3 and 3 conv layers (the recipe that won at mini-class).
        "embedding": 14, "conv0": 32, "conv1": 56, "conv2": 56, "dense": 48,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b512-k3-m2048-micro-3conv-hidden": {
        # Micro K=3 3-conv. Halves the hash table to 512 bins.
        "embedding": 12, "conv0": 24, "conv1": 48, "conv2": 48, "dense": 32,
        "bins": 512, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b512-k3-m2048-pico-3conv-hidden": {
        # Pico K=3 3-conv: ~6 KB target.
        "embedding": 8, "conv0": 20, "conv1": 36, "conv2": 36, "dense": 24,
        "bins": 512, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b1536-k3-m2048-med-3conv-hidden": {
        # ~100 KB target. Bigger emb table, wider conv stack.
        "embedding": 28, "conv0": 96, "conv1": 192, "conv2": 192, "dense": 160,
        "bins": 1536, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b2048-k3-m2048-big-3conv-hidden": {
        # ~200 KB target — the broadest student we train.
        "embedding": 40, "conv0": 144, "conv1": 288, "conv2": 288, "dense": 200,
        "bins": 2048, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b1024-k2-m2048-nano-mp-hidden": {
        # In between mini and micro: ~16-18 KB. Keeps the 1024-bin hash table
        # to avoid extra collisions on common keyword bigrams.
        "embedding": 14, "conv0": 40, "conv1": 80, "dense": 64,
        "bins": 1024, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b2048-k3-m2048-mini-mp-hidden": {
        # Mini with a wider hash table (2048 bins) and K=3 hashes for fewer
        # collisions. Same byte budget: emb dim shrinks to 8 to compensate.
        "embedding": 8, "conv0": 48, "conv1": 96, "dense": 80,
        "bins": 2048, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k3-m2048-mini-mp-hidden": {
        # Same emb table size as mini, but K=3 hashes for cleaner bin recovery.
        # No size change.
        "embedding": 16, "conv0": 48, "conv1": 96, "dense": 80,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1536-k2-m2048-mini-mp-hidden": {
        # Same byte budget as mini, more bins (1536) at the same emb dim (16)
        # → emb grows from 8 KB to 12 KB. Trades conv1 channels (96→64).
        "embedding": 16, "conv0": 48, "conv1": 64, "dense": 80,
        "bins": 1536, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k3-m2048-mini-3conv-hidden": {
        # 3-layer conv stack. emb dim shrinks 16→14 to fit conv2. Uses the K=3
        # hashes that worked best in the variant sweep.
        "embedding": 14, "conv0": 40, "conv1": 80, "conv2": 80, "dense": 64,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b1024-k3-m2048-mini-4conv-hidden": {
        # Depth-only variant of the strongest mini 3-conv recipe. Keeps the same
        # table and dense widths, then adds one more 3-wide conv at the pooled
        # sequence length to test whether extra nonlinear depth helps.
        "embedding": 14, "conv0": 40, "conv1": 80, "conv2": 80, "conv3": 80, "dense": 64,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2, "pool3_size": 1,
        "conv3_residual": True, "residual_scale": 0.5,
    },
    "wordseq-b1024-k2-m2048-mini-3conv-hidden": {
        # Mini 3-conv with K=2 (matches the strongest mini baseline).
        "embedding": 14, "conv0": 40, "conv1": 80, "conv2": 80, "dense": 64,
        "bins": 1024, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b1024-k2-m2048-tiny-3conv-hidden": {
        # Full ~50 KB budget, 3 conv layers. emb same as tiny; conv0/1 channels
        # trimmed slightly to fit a conv2 at 128 ch (k=3) + MaxPool(2).
        "embedding": 24, "conv0": 64, "conv1": 128, "conv2": 128, "dense": 96,
        "bins": 1024, "hash_count": 2, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b1024-k3-m2048-tiny-3conv-hidden": {
        # Same as -tiny-3conv but K=3 hashes.
        "embedding": 24, "conv0": 64, "conv1": 128, "conv2": 128, "dense": 96,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2,
    },
    "wordseq-b1024-k3-m2048-tiny-3conv-randproj-bigger-hidden": {
        # random_proj=True frees the 12 KB embedding table; we reinvest into
        # wider conv1/conv2 + dense. Same 50 KB total. Tests whether trained
        # embedding is irreplaceable or if larger conv capacity can recover.
        # Estimated added bytes: c1 128->192 = +5 KB, c2 128->192 with c1=192
        # input = +9 KB, dense 96->128 with 384 input = +3 KB, total ~+17 KB
        # but minus 12 KB freed embedding => +5 KB net (within ~5 KB margin).
        "embedding": 24, "conv0": 64, "conv1": 192, "conv2": 192, "dense": 128,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "pool2_size": 2, "random_proj": True,
    },
    "wordseq-b1024-k3-m2048-mini-wideemb-hidden": {
        # Same K=3 mini but emb grows 16→24 (12 KB table). Conv channels shrink
        # to compensate (48→32, 96→72, dense 80→48). Tests whether the
        # embedding bottleneck or the conv stack is the binding constraint.
        "embedding": 24, "conv0": 32, "conv1": 72, "dense": 48,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max",
    },
    "wordseq-b1024-k3-m2048-mini-multiscale-hidden": {
        # K=3 mini with multi-scale conv0 (kernels 3, 5, 7). Each branch is
        # 16 channels so concatenated output is 48 ch (matches baseline mini).
        "embedding": 16, "conv0": 16, "conv1": 96, "dense": 80,
        "bins": 1024, "hash_count": 3, "max_units": 2048,
        "proj_bits": 4, "conv_bits": 2, "dense_bits": 2, "output_bits": 4,
        "pool_op": "max", "conv0_kernels": (3, 5, 7),
    },
}


def build_wordseq_by_name(classes: int, bits: int, architecture: str, hidden_dim: int = 512) -> tf.keras.Model:
    config = wordseq_config_for_architecture(architecture)
    return build_word_seq_hashembed_hidden_model(classes, bits, hidden_dim=hidden_dim, **config)


def build_conv_xwide_hash32_chunkside2048_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, bits, name="q_embedding")(inputs)
    x = QConv1D(80, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(152, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    conv_features = QDense(128, bits, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)

    hash_features = tf.keras.layers.Lambda(lambda value: hashed_bigram_features(value, 256), name="hash_bigram")(inputs)
    hash_features = QDense(32, bits, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)

    chunk_side = tf.keras.layers.Lambda(
        lambda value: chunked_bigram_bitset_features(value, bins=2048, chunk_size=64, hash_count=3),
        name="chunkside_64_2048_h3",
    )(inputs)
    chunk_side = QDense(16, bits, name="q_chunk_side_project")(chunk_side)
    chunk_side = tf.keras.layers.Activation(tf.nn.gelu)(chunk_side)
    chunk_side = QConv1D(32, 3, bits, name="q_chunk_side_conv")(chunk_side)
    chunk_side = tf.keras.layers.Activation(tf.nn.gelu)(chunk_side)
    chunk_side_max = tf.keras.layers.GlobalMaxPooling1D()(chunk_side)
    chunk_side_avg = tf.keras.layers.GlobalAveragePooling1D()(chunk_side)
    chunk_side = tf.keras.layers.Concatenate()([chunk_side_max, chunk_side_avg])

    x = tf.keras.layers.Concatenate()([conv_features, hash_features, chunk_side])
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_xwide_hashmlp_hidden_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 40, bits, name="q_embedding")(inputs)
    x = QConv1D(80, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(160, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    conv_features = QDense(176, bits, name="q_dense_0")(pooled)
    conv_features = tf.keras.layers.Activation(tf.nn.gelu)(conv_features)
    hash_features = tf.keras.layers.Lambda(lambda value: hashed_bigram_features(value, 256), name="hash_bigram")(inputs)
    hash_features = QDense(80, bits, name="q_hash_project")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    hash_features = QDense(40, bits, name="q_hash_project_1")(hash_features)
    hash_features = tf.keras.layers.Activation(tf.nn.gelu)(hash_features)
    x = tf.keras.layers.Concatenate()([conv_features, hash_features])
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def build_conv_ternary_big_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 48, bits, name="q_embedding")(inputs)
    x = QConv1D(128, 7, bits, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(256, 5, bits, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = QDense(192, bits, name="q_dense_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_conv_mixed_model(classes: int, bits: int) -> tf.keras.Model:
    del bits
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 48, 2, name="q_embedding")(inputs)
    x = QConv1D(128, 7, 2, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=4)(x)
    x = QConv1D(256, 5, 2, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = QDense(128, 4, name="q_dense_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, 4, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_chunked_flat_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(TOKEN_LENGTH,), dtype=tf.int32)
    x = QEmbedding(TOKEN_VOCAB_SIZE, 16, bits, name="q_embedding")(inputs)
    x = tf.keras.layers.Reshape((CHUNK_COUNT, CHUNK_SIZE * 16))(x)
    x = QDense(96, bits, name="q_chunk_project")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    x = QDense(176, bits, name="q_dense_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    outputs = QDense(classes, bits, name="q_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_model(classes: int, bits: int, architecture: str, hidden_dim: int = 512) -> tf.keras.Model:
    if architecture == "conv":
        return build_conv_model(classes, bits)
    if architecture == "conv-wide":
        return build_conv_wide_model(classes, bits)
    if architecture == "conv-xwide":
        return build_conv_xwide_model(classes, bits)
    if architecture == "conv-xwide-hidden":
        return build_conv_xwide_hidden_model(classes, bits)
    if architecture == "conv-xwide-hash-hidden":
        return build_conv_xwide_hash_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-big-mp-hidden":
        return build_conv_xwide_big_mp_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-med-mp-hidden":
        return build_conv_xwide_med_mp_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-small-mp-hidden":
        return build_conv_xwide_small_mp_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-tiny-mp-hidden":
        return build_conv_xwide_tiny_mp_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-dilated-tcn-tiny-hidden":
        return build_conv_xwide_dilated_tcn_tiny_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-sep-se-tiny-hidden":
        return build_conv_xwide_separable_se_tiny_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-tiny-hidden":
        return build_conv_xwide_multiscale_tiny_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-tiny-plus-hidden":
        return build_conv_xwide_multiscale_tiny_plus_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-dilated-tcn-med-hidden":
        return build_conv_xwide_dilated_tcn_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-sep-se-med-hidden":
        return build_conv_xwide_sep_se_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-med-hidden":
        return build_conv_xwide_multiscale_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-big-hidden":
        return build_conv_xwide_multiscale_big_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-deep-hidden":
        return build_conv_xwide_multiscale_deep_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-xbig-hidden":
        return build_conv_xwide_multiscale_xbig_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-attn-med-hidden":
        return build_conv_xwide_multiscale_attn_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-attn-big-hidden":
        return build_conv_xwide_multiscale_attn_big_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-pyramid-med-hidden":
        return build_conv_xwide_multiscale_pyramid_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-ngram-med-hidden":
        return build_conv_xwide_multiscale_ngram_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-kitchen-med-hidden":
        return build_conv_xwide_multiscale_kitchen_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-pos-med-hidden":
        return build_conv_xwide_multiscale_pos_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-xformer-tiny-hidden":
        return build_conv_xwide_xformer_tiny_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-xformer-med-hidden":
        return build_conv_xwide_xformer_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-multiscale-proto-med-hidden":
        return build_conv_xwide_multiscale_proto_med_hidden_model(classes, bits, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-hash80-hidden":
        return build_conv_xwide_hash80_hidden_model(classes, bits)
    if architecture == "conv-xwide-hash96-dense144-hidden":
        return build_conv_xwide_hash96_dense144_hidden_model(classes, bits)
    if architecture == "conv-xwide-hash96-dense160-hidden":
        return build_conv_xwide_hash96_dense160_hidden_model(classes, bits)
    if architecture == "conv-xwide-hash128-dense112-hidden":
        return build_conv_xwide_hash128_dense112_hidden_model(classes, bits)
    if architecture == "conv-e48-c80-c152-hash-hidden":
        return build_conv_e48_c80_c152_hash_hidden_model(classes, bits)
    if architecture == "conv-chunkbitset1024-e40-c72-c152-hidden":
        return build_conv_chunkbitset1024_e40_c72_c152_hidden_model(classes, bits)
    if architecture == "conv-chunkbitset2048-e36-c72-c136-d144-hidden":
        return build_conv_chunkbitset2048_e36_c72_c136_d144_hidden_model(classes, bits)
    if architecture == "conv-xwide-hash32-chunkside2048-hidden":
        return build_conv_xwide_hash32_chunkside2048_hidden_model(classes, bits)
    if architecture.startswith("wordwin-"):
        return build_wordwin_by_name(classes, bits, architecture, hidden_dim=hidden_dim)
    if architecture.startswith(("wordseq-", _WORDSEQ_JSON_PREFIX)):
        return build_wordseq_by_name(classes, bits, architecture, hidden_dim=hidden_dim)
    if architecture == "conv-xwide-hashmlp-hidden":
        return build_conv_xwide_hashmlp_hidden_model(classes, bits)
    if architecture == "conv-ternary-big":
        return build_conv_ternary_big_model(classes, bits)
    if architecture == "conv-mixed":
        return build_conv_mixed_model(classes, bits)
    if architecture == "chunked-flat":
        return build_chunked_flat_model(classes, bits)
    raise ValueError(f"unknown architecture: {architecture}")


def group_names(labels: list[str]) -> tuple[list[str], np.ndarray]:
    groups = {
        "asm": "low",
        "c": "c_like",
        "cpp": "c_like",
        "objectivec": "c_like",
        "cs": "jvm_dotnet",
        "java": "jvm_dotnet",
        "kotlin": "jvm_dotnet",
        "scala": "jvm_dotnet",
        "javascript": "web",
        "typescript": "web",
        "css": "web",
        "scss": "web",
        "html": "web",
        "vue": "web",
        "erb": "web",
        "json": "data",
        "jsonl": "data",
        "yaml": "data",
        "toml": "data",
        "ini": "data",
        "xml": "data",
        "textproto": "data",
        "ipynb": "data",
        "markdown": "docs",
        "haskell": "functional",
        "lisp": "functional",
        "clojure": "functional",
        "ocaml": "functional",
        "erlang": "functional",
        "elixir": "functional",
        "shell": "shell",
        "batch": "shell",
        "powershell": "shell",
        "dockerfile": "shell",
        "verilog": "hdl",
        "vhdl": "hdl",
    }
    names = sorted(set(groups.get(label, "other") for label in labels))
    index = {name: i for i, name in enumerate(names)}
    label_to_group = np.asarray([index[groups.get(label, "other")] for label in labels], dtype=np.int64)
    return names, label_to_group


def add_group_head(model: tf.keras.Model, bits: int, group_count: int) -> tf.keras.Model:
    if group_count <= 0:
        return model
    pooled = model.get_layer("q_hidden_project").input
    group_logits = QDense(group_count, bits, name="q_group_output")(pooled)
    outputs = list(model.output) if isinstance(model.output, list) else [model.output]
    outputs.append(group_logits)
    return tf.keras.Model(inputs=model.input, outputs=outputs)


def add_trunk_hidden_output(model: tf.keras.Model, hidden_dim: int = 512) -> tf.keras.Model:
    """Use the pooled trunk feature as the hidden-distillation output.

    Exported MSQ1 checkpoints omit q_hidden_project, so progressive shrink
    stages cannot rely on that auxiliary head surviving across checkpoints.
    This wraps the model to emit a stable 512-wide padded/truncated trunk vector
    instead.
    """
    logits = model.output[0] if isinstance(model.output, list) else model.output
    pooled = model.get_layer("q_hidden_project").input

    def _to_width(value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, tf.float32)
        value = value[:, :hidden_dim]
        pad = tf.maximum(hidden_dim - tf.shape(value)[1], 0)
        return tf.pad(value, [[0, 0], [0, pad]])

    hidden = tf.keras.layers.Lambda(
        _to_width,
        output_shape=(hidden_dim,),
        name="trunk_hidden_512",
    )(pooled)
    return tf.keras.Model(inputs=model.input, outputs=[logits, hidden])


def count_paths(split_dir: Path, limit: int | None) -> int:
    return sum(1 for _ in source_paths(split_dir, limit))


def cache_is_current(cache_dir: Path, split: str, classes: int, hidden_dim: int | None) -> bool:
    meta_path = cache_meta_path(cache_dir, split)
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    count = int(meta["count"])
    expected = {
        "tokens": count * TOKEN_LENGTH * np.dtype(np.uint16).itemsize,
        "probabilities": count * classes * np.dtype(np.float32).itemsize,
        "labels": count * np.dtype(np.int64).itemsize,
    }
    if hidden_dim is not None:
        expected["hidden"] = count * hidden_dim * np.dtype(np.float16).itemsize
    for name, size in expected.items():
        path = cache_dir / f"{split}.{name}.mmap"
        if not path.exists() or path.stat().st_size != size:
            return False
    return True


_MULTI_HIDDEN = "concat[LayerNorm_1, mean_pool(LayerNorm_0)]"
_MULTI3_HIDDEN = "concat[LayerNorm_1, mean_pool(LayerNorm_0), mean_pool(Conv_0)]"
_MULTI_LN1 = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_1/AddV2_1:0"
_MULTI_LN0 = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_0/AddV2_1:0"
_MULTI_CONV0 = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Conv_0/AddV2:0"
_MULTI3_HIDDEN_DIM = 1536  # 512 + 512 + 512 (Conv_0 channels=512 in standard_v3_3)


def build_split_cache(
    split_dir: Path,
    cache_dir: Path,
    split: str,
    teacher: Teacher,
    limit: int | None,
    teacher_batch_size: int,
    hidden_output: str | None,
) -> int:
    capacity = count_paths(split_dir, limit)
    if capacity == 0:
        raise RuntimeError(
            f"Cache rebuild for {split!r} would create empty mmaps because "
            f"{split_dir} contains no files. This likely means --dataset is "
            f"pointing one level too high (e.g., a parent directory of a 'files/' "
            f"subdir). Refusing to overwrite the existing cache. "
            f"Pass --dataset <root>/files if that's where the per-split dirs live."
        )
    tokens_path = cache_dir / f"{split}.tokens.mmap"
    probabilities_path = cache_dir / f"{split}.probabilities.mmap"
    labels_path = cache_dir / f"{split}.labels.mmap"
    hidden_path = cache_dir / f"{split}.hidden.mmap"
    multi_mode = hidden_output == _MULTI_HIDDEN
    multi3_mode = hidden_output == _MULTI3_HIDDEN
    if multi3_mode:
        hidden_dim_value = _MULTI3_HIDDEN_DIM
    elif multi_mode:
        hidden_dim_value = 1024
    else:
        hidden_dim_value = 512
    # Delete any stale hidden cache first so a smaller existing file does not
    # cause np.memmap("w+") to silently keep the old (smaller) size.
    if hidden_output and hidden_path.exists():
        hidden_path.unlink()
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="w+", shape=(capacity, TOKEN_LENGTH))
    probabilities = np.memmap(probabilities_path, dtype=np.float32, mode="w+", shape=(capacity, len(teacher.selected_labels)))
    labels = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(capacity,))
    hidden = np.memmap(hidden_path, dtype=np.float16, mode="w+", shape=(capacity, hidden_dim_value)) if hidden_output else None

    if multi3_mode:
        teacher_outputs = ["target_label", _MULTI_LN1, _MULTI_LN0, _MULTI_CONV0]
    elif multi_mode:
        teacher_outputs = ["target_label", _MULTI_LN1, _MULTI_LN0]
    elif hidden_output:
        teacher_outputs = ["target_label", hidden_output]
    else:
        teacher_outputs = ["target_label"]

    pending_tokens: list[list[int]] = []
    written = 0
    seen = 0
    skipped = 0

    def flush() -> None:
        nonlocal written
        if not pending_tokens:
            return
        outputs = teacher.session.run(teacher_outputs, {"bytes": pending_tokens})
        raw = outputs[0].astype(np.float32)
        selected = raw[:, teacher.selected_indices]
        selected_sum = selected.sum(axis=1, keepdims=True)
        keep = selected_sum[:, 0] > 0.0
        selected = selected[keep] / selected_sum[keep]
        kept_tokens = [window for window, should_keep in zip(pending_tokens, keep) if should_keep]
        if kept_tokens:
            end = written + len(kept_tokens)
            tokens[written:end] = np.asarray(kept_tokens, dtype=np.uint16)
            probabilities[written:end] = selected
            labels[written:end] = selected.argmax(axis=1).astype(np.int64)
            if hidden is not None:
                if multi3_mode:
                    ln1 = outputs[1].astype(np.float32)[keep]   # (B, 512)
                    ln0 = outputs[2].astype(np.float32)[keep]   # (B, 512, 256)
                    conv0 = outputs[3].astype(np.float32)[keep] # (B, 512, 508)
                    if conv0.shape[1] != 512:
                        raise RuntimeError(
                            f"Conv_0 channel dim is {conv0.shape[1]}, expected 512"
                        )
                    ln0_pooled = ln0.mean(axis=2)               # (B, 512)
                    conv0_pooled = conv0.mean(axis=2)           # (B, 512)
                    ln1_norm = ln1 / np.maximum(np.linalg.norm(ln1, axis=1, keepdims=True), 1e-6)
                    ln0_norm = ln0_pooled / np.maximum(np.linalg.norm(ln0_pooled, axis=1, keepdims=True), 1e-6)
                    conv0_norm = conv0_pooled / np.maximum(np.linalg.norm(conv0_pooled, axis=1, keepdims=True), 1e-6)
                    combined = np.concatenate([ln1_norm, ln0_norm, conv0_norm], axis=1)
                    hidden[written:end] = combined.astype(np.float16)
                elif multi_mode:
                    ln1 = outputs[1].astype(np.float32)[keep]  # (B, 512)
                    ln0 = outputs[2].astype(np.float32)[keep]  # (B, 512, 256)
                    ln0_pooled = ln0.mean(axis=2)  # (B, 512)
                    ln1_norm = ln1 / np.maximum(np.linalg.norm(ln1, axis=1, keepdims=True), 1e-6)
                    ln0_norm = ln0_pooled / np.maximum(np.linalg.norm(ln0_pooled, axis=1, keepdims=True), 1e-6)
                    combined = np.concatenate([ln1_norm, ln0_norm], axis=1)
                    hidden[written:end] = combined.astype(np.float16)
                else:
                    hidden_values = outputs[1].astype(np.float32)[keep]
                    hidden_norm = np.maximum(np.linalg.norm(hidden_values, axis=1, keepdims=True), 1e-6)
                    hidden[written:end] = (hidden_values / hidden_norm).astype(np.float16)
            written = end
        pending_tokens.clear()

    for path in source_paths(split_dir, limit):
        seen += 1
        windows = read_training_windows(path)
        if windows is None:
            skipped += 1
            continue
        size, prefix, suffix = windows
        token_window = magika_features(size, prefix, suffix)
        if token_window is None:
            skipped += 1
            continue
        pending_tokens.append(token_window)
        if len(pending_tokens) >= teacher_batch_size:
            flush()
        if seen % 10000 == 0:
            print(f"{split}: seen={seen} cached={written} skipped={skipped}", flush=True)

    flush()
    tokens.flush()
    probabilities.flush()
    labels.flush()
    if hidden is not None:
        hidden.flush()
    with tokens_path.open("r+b") as file:
        file.truncate(written * TOKEN_LENGTH * np.dtype(np.uint16).itemsize)
    with probabilities_path.open("r+b") as file:
        file.truncate(written * len(teacher.selected_labels) * np.dtype(np.float32).itemsize)
    with labels_path.open("r+b") as file:
        file.truncate(written * np.dtype(np.int64).itemsize)
    if hidden is not None:
        with hidden_path.open("r+b") as file:
            file.truncate(written * hidden_dim_value * np.dtype(np.float16).itemsize)

    meta = {
        "count": written,
        "seen": seen,
        "skipped": skipped,
        "classes": len(teacher.selected_labels),
        "token_length": TOKEN_LENGTH,
        "labels": teacher.selected_labels,
        "slugs": teacher.selected_slugs,
        "hidden_dim": hidden_dim_value if hidden_output else None,
        "hidden_output": hidden_output,
    }
    cache_meta_path(cache_dir, split).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"{split}: cached={written} skipped={skipped}", flush=True)
    return written


def ensure_cache(
    dataset: Path,
    cache_dir: Path,
    teacher: Teacher,
    limit: int | None,
    teacher_batch_size: int,
    rebuild_cache: bool,
    hidden_output: str | None,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        meta_path = cache_meta_path(cache_dir, split)
        existing_hidden_dim = None
        if meta_path.exists():
            existing_meta = json.loads(meta_path.read_text())
            existing_hidden_dim = existing_meta.get("hidden_dim")
        hidden_dim = existing_hidden_dim if hidden_output else None
        if not rebuild_cache and cache_is_current(cache_dir, split, len(teacher.selected_labels), hidden_dim):
            meta = json.loads(meta_path.read_text())
            if meta.get("hidden_output") == hidden_output:
                counts[split] = int(meta["count"])
                print(f"{split}: using cached {counts[split]} examples (hidden_dim={hidden_dim})", flush=True)
                continue
        counts[split] = build_split_cache(dataset / split, cache_dir, split, teacher, limit, teacher_batch_size, hidden_output)
    return counts


def convert_splits_to_word_units(
    cache_dir: Path,
    train: TokenSplit,
    valid: TokenSplit,
    test: TokenSplit,
    tokenizer_version: int = 1,
) -> tuple[TokenSplit, TokenSplit, TokenSplit]:
    out: list[TokenSplit] = []
    suffix = "" if tokenizer_version == 1 else f"_v{tokenizer_version}"
    if tokenizer_version == 1:
        apply_fn = numpy_word_units_apply
    elif tokenizer_version == 2:
        apply_fn = numpy_word_units_apply_v2
    elif tokenizer_version == 3:
        apply_fn = numpy_word_units_apply_v3
    elif tokenizer_version == 4:
        apply_fn = numpy_word_units_apply_v4
    else:
        raise ValueError(f"unknown tokenizer_version: {tokenizer_version}")
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        units_path = cache_dir / f"{name}.units{suffix}.mmap"
        expected = split.count * TOKEN_LENGTH * np.dtype(np.int32).itemsize
        if units_path.exists() and units_path.stat().st_size == expected:
            print(f"{name}: using cached unit ids (v{tokenizer_version})")
        else:
            print(f"{name}: building unit-id cache v{tokenizer_version} ({split.count} examples)...", flush=True)
            started = time.perf_counter()
            block = 8192
            tokens_arr = np.asarray(split.tokens, dtype=np.int32)
            units_mm = np.memmap(units_path, dtype=np.int32, mode="w+", shape=(split.count, TOKEN_LENGTH))
            for start in range(0, split.count, block):
                end = min(start + block, split.count)
                units_mm[start:end] = apply_fn(tokens_arr[start:end], TOKEN_LENGTH)
            units_mm.flush()
            del units_mm
            print(f"  done in {time.perf_counter() - started:.1f}s", flush=True)
        units_view = np.memmap(units_path, dtype=np.int32, mode="r", shape=(split.count, TOKEN_LENGTH))
        out.append(TokenSplit(
            tokens=units_view,
            probabilities=split.probabilities,
            labels=split.labels,
            count=split.count,
            hidden=split.hidden,
            self_probabilities=split.self_probabilities,
            short_slice_probabilities=split.short_slice_probabilities,
            short_slice_confidences=split.short_slice_confidences,
        ))
    return out[0], out[1], out[2]


def short_slice_meta_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}.short_slice_targets.json"


def short_slice_prob_path(cache_dir: Path, split: str, length: int) -> Path:
    return cache_dir / f"{split}.short_slice_{length}.probabilities.mmap"


def short_slice_conf_path(cache_dir: Path, split: str, length: int) -> Path:
    return cache_dir / f"{split}.short_slice_{length}.confidence.mmap"


def cached_prefix_slice_features(tokens: np.ndarray, length: int) -> list[int] | None:
    prefix = bytes(int(value) for value in tokens[:MAGIKA_BEG_SIZE] if int(value) < PADDING_TOKEN)
    if not prefix:
        return None
    data = prefix[:length]
    return magika_features(len(data), data, data)


def short_slice_cache_is_current(
    target_dir: Path,
    split: str,
    count: int,
    classes: int,
    lengths: tuple[int, ...],
    labels: tuple[str, ...] | None = None,
) -> bool:
    lengths = tuple(sorted(set(int(length) for length in lengths)))
    meta_path = short_slice_meta_path(target_dir, split)
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    if int(meta.get("count", -1)) != count or int(meta.get("classes", -1)) != classes:
        return False
    if labels is not None and tuple(str(value) for value in meta.get("labels", [])) != labels:
        return False
    if tuple(int(value) for value in meta.get("lengths", [])) != tuple(lengths):
        return False
    for length in lengths:
        prob_path = short_slice_prob_path(target_dir, split, length)
        conf_path = short_slice_conf_path(target_dir, split, length)
        if not prob_path.exists() or prob_path.stat().st_size != count * classes * np.dtype(np.float32).itemsize:
            return False
        if not conf_path.exists() or conf_path.stat().st_size != count * np.dtype(np.float32).itemsize:
            return False
    return True


def build_short_slice_target_cache(
    source_cache_dir: Path,
    target_dir: Path,
    split: str,
    teacher: Teacher,
    lengths: tuple[int, ...],
    teacher_batch_size: int,
    rebuild: bool,
) -> None:
    meta = json.loads(cache_meta_path(source_cache_dir, split).read_text())
    count = int(meta["count"])
    classes = len(teacher.selected_labels)
    lengths = tuple(sorted(set(int(length) for length in lengths)))
    if any(length > MAGIKA_BEG_SIZE for length in lengths):
        raise ValueError(
            f"short-slice target lengths must be <= {MAGIKA_BEG_SIZE}; "
            "the source cache only preserves the first Magika prefix window"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    labels = tuple(teacher.selected_labels)
    if not rebuild and short_slice_cache_is_current(target_dir, split, count, classes, lengths, labels):
        print(f"{split}: using cached short-slice teacher targets lengths={lengths}", flush=True)
        return

    tokens = np.memmap(source_cache_dir / f"{split}.tokens.mmap", dtype=np.uint16, mode="r", shape=(count, TOKEN_LENGTH))
    for length in lengths:
        print(f"{split}: building short-slice teacher targets length={length} ({count} examples)...", flush=True)
        started = time.perf_counter()
        probabilities = np.memmap(
            short_slice_prob_path(target_dir, split, length),
            dtype=np.float32,
            mode="w+",
            shape=(count, classes),
        )
        confidences = np.memmap(
            short_slice_conf_path(target_dir, split, length),
            dtype=np.float32,
            mode="w+",
            shape=(count,),
        )
        probabilities.fill(0.0)
        confidences.fill(0.0)
        pending_rows: list[int] = []
        pending_features: list[list[int]] = []

        def flush() -> None:
            if not pending_rows:
                return
            raw = teacher.session.run(["target_label"], {"bytes": pending_features})[0].astype(np.float32)
            selected = raw[:, teacher.selected_indices]
            selected_sum = selected.sum(axis=1, keepdims=True)
            keep = selected_sum[:, 0] > 0.0
            normalized = np.zeros((len(pending_rows), classes), dtype=np.float32)
            normalized[keep] = selected[keep] / selected_sum[keep]
            rows = np.asarray(pending_rows, dtype=np.int64)
            probabilities[rows] = normalized
            confidences[rows] = normalized.max(axis=1)
            pending_rows.clear()
            pending_features.clear()

        for row in range(count):
            features = cached_prefix_slice_features(tokens[row], length)
            if features is None:
                continue
            pending_rows.append(row)
            pending_features.append(features)
            if len(pending_rows) >= teacher_batch_size:
                flush()
            if row > 0 and row % 50000 == 0:
                print(f"  {split} length={length}: row={row}/{count}", flush=True)
        flush()
        probabilities.flush()
        confidences.flush()
        print(f"  done in {time.perf_counter() - started:.1f}s", flush=True)

    payload = {
        "classes": classes,
        "count": count,
        "labels": teacher.selected_labels,
        "lengths": list(lengths),
        "source_cache_dir": str(source_cache_dir),
        "type": "magika_prefix_slice_targets",
        "version": 1,
    }
    short_slice_meta_path(target_dir, split).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ensure_short_slice_target_cache(
    source_cache_dir: Path,
    target_dir: Path,
    teacher: Teacher,
    lengths: tuple[int, ...],
    teacher_batch_size: int,
    rebuild: bool,
    splits: tuple[str, ...] = ("train",),
) -> None:
    for split in splits:
        build_short_slice_target_cache(
            source_cache_dir,
            target_dir,
            split,
            teacher,
            lengths,
            teacher_batch_size,
            rebuild,
        )


def attach_short_slice_targets(
    split_data: TokenSplit,
    target_dir: Path,
    split: str,
    lengths: tuple[int, ...],
    classes: int,
    labels: tuple[str, ...] | None = None,
) -> TokenSplit:
    if not short_slice_cache_is_current(target_dir, split, split_data.count, classes, lengths, labels):
        raise FileNotFoundError(
            f"short-slice target cache missing or stale for {split!r} in {target_dir}; "
            "run with --build-short-slice-target-cache first"
        )
    probabilities = {
        length: np.memmap(
            short_slice_prob_path(target_dir, split, length),
            dtype=np.float32,
            mode="r",
            shape=(split_data.count, classes),
        )
        for length in lengths
    }
    confidences = {
        length: np.memmap(
            short_slice_conf_path(target_dir, split, length),
            dtype=np.float32,
            mode="r",
            shape=(split_data.count,),
        )
        for length in lengths
    }
    print(f"{split}: using short-slice teacher targets from {target_dir}", flush=True)
    return replace(
        split_data,
        short_slice_probabilities=probabilities,
        short_slice_confidences=confidences,
    )


def open_split(
    cache_dir: Path,
    split: str,
    classes: int,
    self_probabilities_dir: Path | None = None,
    fs_labels_dir: Path | None = None,
    load_hidden: bool = True,
) -> TokenSplit:
    meta = json.loads(cache_meta_path(cache_dir, split).read_text())
    count = int(meta["count"])
    hidden_dim = meta.get("hidden_dim") if load_hidden else None
    hidden = (
        np.memmap(cache_dir / f"{split}.hidden.mmap", dtype=np.float16, mode="r", shape=(count, int(hidden_dim)))
        if hidden_dim
        else None
    )
    self_probabilities = None
    if self_probabilities_dir is not None:
        self_path = self_probabilities_dir / f"{split}.self_probabilities.mmap"
        if not self_path.exists():
            raise FileNotFoundError(
                f"--self-probabilities was set but {self_path} is missing; "
                "did you run scripts/cache_self_distill.py first?"
            )
        self_probabilities = np.memmap(
            self_path, dtype=np.float32, mode="r", shape=(count, classes)
        )
    if fs_labels_dir is not None:
        fs_path = fs_labels_dir / f"{split}.fs_labels.mmap"
        if not fs_path.exists():
            raise FileNotFoundError(
                f"--fs-labels-dir was set but {fs_path} is missing; "
                "run scripts/build_fs_labels.py to generate it."
            )
        labels_mm = np.memmap(fs_path, dtype=np.int64, mode="r", shape=(count,))
        print(f"{split}: using filesystem-extension labels from {fs_path}", flush=True)
    else:
        labels_mm = np.memmap(cache_dir / f"{split}.labels.mmap", dtype=np.int64, mode="r", shape=(count,))
    return TokenSplit(
        tokens=np.memmap(cache_dir / f"{split}.tokens.mmap", dtype=np.uint16, mode="r", shape=(count, TOKEN_LENGTH)),
        probabilities=np.memmap(cache_dir / f"{split}.probabilities.mmap", dtype=np.float32, mode="r", shape=(count, classes)),
        labels=labels_mm,
        count=count,
        hidden=hidden,
        self_probabilities=self_probabilities,
    )


def _np_cutmix(
    tokens: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    hidden: np.ndarray | None,
    self_probs: np.ndarray | None,
    cutmix_prob: float,
    rng: np.random.Generator,
):
    bs, L = tokens.shape
    if bs < 2 or cutmix_prob <= 0.0:
        return tokens, probabilities, labels, hidden, self_probs
    shift = int(rng.integers(1, bs))
    partner = (np.arange(bs) + shift) % bs
    k = rng.integers(8, L - 8, size=bs)
    alpha = (k / L).astype(np.float32)
    apply = rng.random(bs) < cutmix_prob
    pos = np.arange(L)[None, :]
    mask = pos < k[:, None]
    out_tokens = np.where(mask, tokens, tokens[partner]).astype(tokens.dtype)
    p_probs = probabilities[partner]
    mixed_probs = alpha[:, None] * probabilities + (1.0 - alpha[:, None]) * p_probs
    p_labels = labels[partner]
    mixed_labels = np.where(alpha > 0.5, labels, p_labels).astype(labels.dtype)
    apply2 = apply[:, None]
    tokens = np.where(apply2, out_tokens, tokens)
    probabilities = np.where(apply2, mixed_probs, probabilities).astype(np.float32)
    labels = np.where(apply, mixed_labels, labels)
    if hidden is not None and hidden.shape[-1] > 0:
        p_hidden = hidden[partner]
        mixed_hidden = alpha[:, None] * hidden + (1.0 - alpha[:, None]) * p_hidden
        hidden = np.where(apply2, mixed_hidden, hidden).astype(np.float32)
    if self_probs is not None and self_probs.shape[-1] > 0:
        p_self = self_probs[partner]
        mixed_self = alpha[:, None] * self_probs + (1.0 - alpha[:, None]) * p_self
        self_probs = np.where(apply2, mixed_self, self_probs).astype(np.float32)
    return tokens, probabilities, labels, hidden, self_probs


def _np_apply_short_slices(
    tokens: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    self_probs: np.ndarray,
    indices: np.ndarray,
    short_slice_prob: float,
    short_slice_lengths: tuple[int, ...],
    rng: np.random.Generator,
    short_slice_probabilities: dict[int, np.memmap] | None = None,
    short_slice_confidences: dict[int, np.memmap] | None = None,
    short_slice_target_min_confidence: float = 0.0,
    short_slice_target_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if short_slice_prob <= 0.0 or not short_slice_lengths or tokens.shape[1] == 0:
        return tokens, probabilities, labels, self_probs
    apply = rng.random(tokens.shape[0]) < short_slice_prob
    choices = np.asarray(short_slice_lengths, dtype=np.int32)
    lengths = choices[rng.integers(0, len(choices), size=tokens.shape[0])]
    valid_lengths = (tokens >= 0).sum(axis=1)
    apply &= lengths < valid_lengths
    if not np.any(apply):
        return tokens, probabilities, labels, self_probs

    has_targets = short_slice_probabilities is not None and short_slice_confidences is not None
    use_fs_label_targets = short_slice_target_mode == "fs-label"
    out_tokens = np.array(tokens, dtype=np.int32, copy=True)
    out_probabilities = probabilities
    out_labels = labels
    out_self_probs = self_probs
    if has_targets or use_fs_label_targets:
        out_probabilities = np.asarray(probabilities, dtype=np.float32).copy()
        out_labels = np.asarray(labels).copy()
        if self_probs.shape[-1] == out_probabilities.shape[-1]:
            out_self_probs = np.asarray(self_probs, dtype=np.float32).copy()

    for length_value in np.unique(lengths[apply]):
        length = int(length_value)
        rows = np.flatnonzero(apply & (lengths == length))
        if has_targets:
            if length not in short_slice_probabilities or length not in short_slice_confidences:
                raise KeyError(f"short-slice target cache is missing length={length}")
            row_indices = indices[rows]
            confidences = np.asarray(short_slice_confidences[length][row_indices], dtype=np.float32)
            rows = rows[confidences >= short_slice_target_min_confidence]
            if rows.size == 0:
                continue
            row_indices = indices[rows]
            targets = np.asarray(short_slice_probabilities[length][row_indices], dtype=np.float32)
            valid_targets = targets.sum(axis=1) > 0.0
            if not np.any(valid_targets):
                continue
            rows = rows[valid_targets]
            targets = targets[valid_targets]
            out_probabilities[rows] = targets
            out_labels[rows] = targets.argmax(axis=1).astype(out_labels.dtype, copy=False)
            if out_self_probs.shape[-1] == targets.shape[-1]:
                out_self_probs[rows] = targets
        elif use_fs_label_targets:
            target_labels = np.asarray(labels[rows], dtype=np.int64)
            valid_targets = (0 <= target_labels) & (target_labels < out_probabilities.shape[-1])
            if not np.any(valid_targets):
                continue
            rows = rows[valid_targets]
            target_labels = target_labels[valid_targets]
            targets = np.zeros((rows.size, out_probabilities.shape[-1]), dtype=np.float32)
            targets[np.arange(rows.size), target_labels] = 1.0
            out_probabilities[rows] = targets
            out_labels[rows] = target_labels.astype(out_labels.dtype, copy=False)
            if out_self_probs.shape[-1] == targets.shape[-1]:
                out_self_probs[rows] = targets
        out_tokens[rows, length:] = -1

    return out_tokens, out_probabilities, out_labels, out_self_probs


_LENGTH_BUCKETS = (128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048)


def compute_unit_lengths(units: np.ndarray | np.memmap) -> np.ndarray:
    """Return per-sample count of valid (>=0) units. Reads in blocks to avoid OOM."""
    n, L = units.shape
    out = np.empty(n, dtype=np.int32)
    block = 8192
    for s in range(0, n, block):
        e = min(s + block, n)
        out[s:e] = (np.asarray(units[s:e]) >= 0).sum(axis=1)
    return out


def bucket_for(length: int) -> int:
    """Smallest bucket size that fits the given length, capped at the max bucket."""
    for b in _LENGTH_BUCKETS:
        if length <= b:
            return b
    return _LENGTH_BUCKETS[-1]


def batches(
    split: TokenSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    balance_power: float = 0.0,
    hard_mask: np.ndarray | None = None,
    hard_oversample_rate: float = 1.0,
    cutmix_prob: float = 0.0,
    unit_lengths: np.ndarray | None = None,
    short_slice_prob: float = 0.0,
    short_slice_lengths: tuple[int, ...] = (),
    short_slice_target_min_confidence: float = 0.0,
    short_slice_target_mode: str = "none",
):
    rng = np.random.default_rng(seed)
    if unit_lengths is not None and shuffle and balance_power == 0.0 and (hard_mask is None or hard_oversample_rate <= 1.0):
        # Length-bucketed shuffle: group same-length samples, trim each batch to
        # the nearest bucket size. Drops in-batch padding from ~67% (random shuffle
        # over 2048-wide rows) to ~10-20%.
        buckets_for_each = np.asarray([bucket_for(int(L)) for L in unit_lengths], dtype=np.int32)
        unique_buckets = sorted(set(int(b) for b in buckets_for_each))
        bucket_to_indices = {b: np.where(buckets_for_each == b)[0] for b in unique_buckets}
        all_batches: list[tuple[int, np.ndarray]] = []
        for b in unique_buckets:
            idx = bucket_to_indices[b].copy()
            rng.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                all_batches.append((b, idx[start : start + batch_size]))
        # Shuffle batch order so gradient noise isn't all-short-then-all-long.
        order = rng.permutation(len(all_batches))
        for j in order:
            bucket, indices = all_batches[j]
            hidden = (
                split.hidden[indices].astype(np.float32)
                if split.hidden is not None
                else np.zeros((len(indices), 0), dtype=np.float32)
            )
            self_probs = (
                split.self_probabilities[indices]
                if split.self_probabilities is not None
                else np.zeros((len(indices), 0), dtype=np.float32)
            )
            t_full = split.tokens[indices].astype(np.int32)
            t = t_full[:, :bucket]
            p = split.probabilities[indices]
            l = split.labels[indices]
            if shuffle and cutmix_prob > 0.0 and len(indices) >= 2:
                t, p, l, hidden, self_probs = _np_cutmix(
                    t, p, l, hidden, self_probs, cutmix_prob, rng
                )
            if shuffle:
                t, p, l, self_probs = _np_apply_short_slices(
                    t,
                    p,
                    l,
                    self_probs,
                    indices,
                    short_slice_prob,
                    short_slice_lengths,
                    rng,
                    split.short_slice_probabilities,
                    split.short_slice_confidences,
                    short_slice_target_min_confidence,
                    short_slice_target_mode,
                )
            yield t, p, l, hidden, self_probs
        return
    use_hard = shuffle and hard_mask is not None and hard_oversample_rate > 1.0
    if shuffle and (balance_power > 0.0 or use_hard):
        weights = np.ones(split.count, dtype=np.float64)
        if balance_power > 0.0:
            counts = np.bincount(np.asarray(split.labels), minlength=int(np.max(split.labels)) + 1).astype(np.float64)
            weights *= 1.0 / np.maximum(counts[np.asarray(split.labels)], 1.0) ** balance_power
        if use_hard:
            weights *= 1.0 + (hard_oversample_rate - 1.0) * np.asarray(hard_mask, dtype=np.float64)
        weights /= weights.sum()
        steps = max(1, split.count // batch_size)
        for _ in range(steps):
            indices = rng.choice(split.count, size=batch_size, replace=True, p=weights)
            hidden = (
                split.hidden[indices].astype(np.float32)
                if split.hidden is not None
                else np.zeros((len(indices), 0), dtype=np.float32)
            )
            self_probs = (
                split.self_probabilities[indices]
                if split.self_probabilities is not None
                else np.zeros((len(indices), 0), dtype=np.float32)
            )
            t, p, l, h, s = (
                split.tokens[indices].astype(np.int32),
                split.probabilities[indices],
                split.labels[indices],
                hidden,
                self_probs,
            )
            if shuffle and cutmix_prob > 0.0:
                t, p, l, h, s = _np_cutmix(t, p, l, h, s, cutmix_prob, rng)
            if shuffle:
                t, p, l, s = _np_apply_short_slices(
                    t,
                    p,
                    l,
                    s,
                    indices,
                    short_slice_prob,
                    short_slice_lengths,
                    rng,
                    split.short_slice_probabilities,
                    split.short_slice_confidences,
                    short_slice_target_min_confidence,
                    short_slice_target_mode,
                )
            yield t, p, l, h, s
        return
    order = rng.permutation(split.count) if shuffle else np.arange(split.count)
    for start in range(0, split.count, batch_size):
        indices = order[start : start + batch_size]
        hidden = (
            split.hidden[indices].astype(np.float32)
            if split.hidden is not None
            else np.zeros((len(indices), 0), dtype=np.float32)
        )
        self_probs = (
            split.self_probabilities[indices]
            if split.self_probabilities is not None
            else np.zeros((len(indices), 0), dtype=np.float32)
        )
        t, p, l, h, s = (
            split.tokens[indices].astype(np.int32),
            split.probabilities[indices],
            split.labels[indices],
            hidden,
            self_probs,
        )
        if shuffle and cutmix_prob > 0.0 and len(indices) >= 2:
            t, p, l, h, s = _np_cutmix(t, p, l, h, s, cutmix_prob, rng)
        if shuffle:
            t, p, l, s = _np_apply_short_slices(
                t,
                p,
                l,
                s,
                indices,
                short_slice_prob,
                short_slice_lengths,
                rng,
                split.short_slice_probabilities,
                split.short_slice_confidences,
                short_slice_target_min_confidence,
                short_slice_target_mode,
            )
        yield t, p, l, h, s


def materialize_split(
    split: TokenSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.asarray(split.tokens, dtype=np.int32)
    probabilities = np.asarray(split.probabilities, dtype=np.float32)
    labels = np.asarray(split.labels, dtype=np.int64)
    if split.hidden is None:
        hidden = np.zeros((split.count, 0), dtype=np.float32)
    else:
        hidden = np.asarray(split.hidden, dtype=np.float32)
    if split.self_probabilities is None:
        self_probs = np.zeros((split.count, 0), dtype=np.float32)
    else:
        self_probs = np.asarray(split.self_probabilities, dtype=np.float32)
    return tokens, probabilities, labels, hidden, self_probs


def _cutmix_batch(
    tokens: tf.Tensor,
    probabilities: tf.Tensor,
    labels: tf.Tensor,
    hidden: tf.Tensor,
    self_probs: tf.Tensor,
    cutmix_prob: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    bs = tf.shape(tokens)[0]
    L = tf.shape(tokens)[1]
    Lf = tf.cast(L, tf.float32)
    shift = tf.random.uniform([], minval=1, maxval=tf.maximum(bs, 2), dtype=tf.int32)
    partner_idx = tf.math.mod(tf.range(bs) + shift, bs)
    k = tf.random.uniform([bs], minval=8, maxval=L - 8, dtype=tf.int32)
    pos = tf.range(L)[None, :]
    mask = pos < k[:, None]
    p_tokens = tf.gather(tokens, partner_idx)
    mixed_tokens = tf.where(mask, tokens, p_tokens)
    alpha = tf.cast(k, tf.float32) / Lf
    a1 = alpha[:, None]
    p_probs = tf.gather(probabilities, partner_idx)
    mixed_probs = a1 * probabilities + (1.0 - a1) * p_probs
    p_labels = tf.gather(labels, partner_idx)
    mixed_labels = tf.where(alpha > 0.5, labels, p_labels)
    if hidden.shape[-1] is not None and hidden.shape[-1] > 0:
        p_hidden = tf.gather(hidden, partner_idx)
        mixed_hidden = a1 * hidden + (1.0 - a1) * p_hidden
    else:
        mixed_hidden = hidden
    if self_probs.shape[-1] is not None and self_probs.shape[-1] > 0:
        p_self = tf.gather(self_probs, partner_idx)
        mixed_self = a1 * self_probs + (1.0 - a1) * p_self
    else:
        mixed_self = self_probs
    apply = tf.random.uniform([bs]) < cutmix_prob
    a2 = apply[:, None]
    out_tokens = tf.where(a2, mixed_tokens, tokens)
    out_probs = tf.where(a2, mixed_probs, probabilities)
    out_labels = tf.where(apply, mixed_labels, labels)
    if hidden.shape[-1] is not None and hidden.shape[-1] > 0:
        out_hidden = tf.where(a2, mixed_hidden, hidden)
    else:
        out_hidden = hidden
    if self_probs.shape[-1] is not None and self_probs.shape[-1] > 0:
        out_self = tf.where(a2, mixed_self, self_probs)
    else:
        out_self = self_probs
    return out_tokens, out_probs, out_labels, out_hidden, out_self


def _short_slice_batch(
    tokens: tf.Tensor,
    probabilities: tf.Tensor,
    labels: tf.Tensor,
    hidden: tf.Tensor,
    self_probs: tf.Tensor,
    short_slice_prob: float,
    short_slice_lengths: tuple[int, ...],
    short_slice_target_mode: str = "none",
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    if short_slice_prob <= 0.0 or not short_slice_lengths:
        return tokens, probabilities, labels, hidden, self_probs
    bs = tf.shape(tokens)[0]
    L = tf.shape(tokens)[1]
    lengths = tf.constant(short_slice_lengths, dtype=tf.int32)
    choice = tf.random.uniform([bs], minval=0, maxval=len(short_slice_lengths), dtype=tf.int32)
    selected = tf.minimum(tf.gather(lengths, choice), L)
    keep = tf.range(L, dtype=tf.int32)[None, :] < selected[:, None]
    sliced = tf.where(keep, tokens, -tf.ones_like(tokens))
    apply = tf.random.uniform([bs]) < short_slice_prob
    out_tokens = tf.where(apply[:, None], sliced, tokens)
    if short_slice_target_mode != "fs-label":
        return out_tokens, probabilities, labels, hidden, self_probs
    fs_targets = tf.one_hot(labels, tf.shape(probabilities)[-1], dtype=probabilities.dtype)
    out_probabilities = tf.where(apply[:, None], fs_targets, probabilities)
    if self_probs.shape[-1] is not None and self_probs.shape[-1] > 0:
        out_self_probs = tf.where(apply[:, None], tf.cast(fs_targets, self_probs.dtype), self_probs)
    else:
        out_self_probs = self_probs
    return out_tokens, out_probabilities, labels, hidden, out_self_probs


def tensor_batches(
    split: TokenSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    balance_power: float = 0.0,
    prefetch: int = 2,
    hard_mask: np.ndarray | None = None,
    hard_oversample_rate: float = 1.0,
    cutmix_prob: float = 0.0,
    short_slice_prob: float = 0.0,
    short_slice_lengths: tuple[int, ...] = (),
    short_slice_target_mode: str = "none",
) -> tf.data.Dataset:
    tokens, probabilities, labels, hidden, self_probs = materialize_split(split)
    use_hard = shuffle and hard_mask is not None and hard_oversample_rate > 1.0
    if shuffle and (balance_power > 0.0 or use_hard):
        weights = np.ones(split.count, dtype=np.float64)
        if balance_power > 0.0:
            counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
            weights *= 1.0 / np.maximum(counts[labels], 1.0) ** balance_power
        if use_hard:
            weights *= 1.0 + (hard_oversample_rate - 1.0) * np.asarray(hard_mask, dtype=np.float64)
        weights /= weights.sum()
        rng = np.random.default_rng(seed)
        steps = max(1, split.count // batch_size)
        indices = rng.choice(split.count, size=steps * batch_size, replace=True, p=weights)
        tokens = tokens[indices]
        probabilities = probabilities[indices]
        labels = labels[indices]
        hidden = hidden[indices]
        self_probs = self_probs[indices]
        dataset = tf.data.Dataset.from_tensor_slices(
            (tokens, probabilities, labels, hidden, self_probs)
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (tokens, probabilities, labels, hidden, self_probs)
        )
        if shuffle:
            dataset = dataset.shuffle(split.count, seed=seed, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if shuffle and cutmix_prob > 0.0:
        dataset = dataset.map(
            lambda t, p, l, h, s: _cutmix_batch(t, p, l, h, s, cutmix_prob),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if shuffle and short_slice_prob > 0.0:
        dataset = dataset.map(
            lambda t, p, l, h, s: _short_slice_batch(
                t, p, l, h, s, short_slice_prob, short_slice_lengths, short_slice_target_mode
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(prefetch)


def distillation_loss(
    probability_batch: tf.Tensor,
    label_batch: tf.Tensor,
    logits: tf.Tensor,
    hidden_batch: tf.Tensor | None,
    student_hidden: tf.Tensor | None,
    group_logits: tf.Tensor | None,
    label_to_group: tf.Tensor | None,
    temperature: float,
    hidden_loss_weight: float,
    hard_loss_weight: float,
    group_loss_weight: float,
    confidence_power: float,
    self_probabilities_batch: tf.Tensor | None = None,
    self_loss_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> tf.Tensor:
    if temperature > 1.0:
        softened = tf.pow(
            tf.maximum(probability_batch, tf.constant(1e-8, dtype=probability_batch.dtype)),
            1.0 / temperature,
        )
        soft_targets = softened / tf.reduce_sum(softened, axis=1, keepdims=True)
        soft_logits = logits / temperature
        soft_scale = temperature * temperature
    else:
        soft_targets = probability_batch
        soft_logits = logits
        soft_scale = 1.0
    soft_per_example = tf.keras.losses.categorical_crossentropy(
        soft_targets, soft_logits, from_logits=True
    ) * soft_scale
    if label_smoothing > 0.0:
        # Per-class smoothing: target = (1 - eps) * one_hot + eps / classes.
        # Equivalent to mixing the hard target with the uniform distribution.
        classes_static = int(logits.shape[-1])
        one_hot = tf.one_hot(label_batch, classes_static, dtype=tf.float32)
        smoothed = one_hot * (1.0 - label_smoothing) + (label_smoothing / classes_static)
        hard_per_example = tf.keras.losses.categorical_crossentropy(
            smoothed, logits, from_logits=True
        )
    else:
        hard_per_example = tf.keras.losses.sparse_categorical_crossentropy(
            label_batch, logits, from_logits=True
        )
    if confidence_power > 0.0:
        weights = tf.reduce_max(probability_batch, axis=1) ** confidence_power
        weights = weights / tf.maximum(tf.reduce_mean(weights), 1e-6)
    else:
        weights = tf.ones_like(soft_per_example)
    weights = tf.cast(weights, tf.float32)
    soft_per_example = tf.cast(soft_per_example, tf.float32)
    hard_per_example = tf.cast(hard_per_example, tf.float32)
    soft_loss = tf.reduce_mean(soft_per_example * weights)
    hard_loss = tf.reduce_mean(hard_per_example * weights)
    hard_loss_weight = tf.cast(hard_loss_weight, tf.float32)
    loss = (1.0 - hard_loss_weight) * soft_loss + hard_loss_weight * hard_loss
    if hidden_loss_weight > 0.0 and hidden_batch is not None and student_hidden is not None:
        student_hidden = tf.math.l2_normalize(tf.cast(student_hidden, tf.float32), axis=1)
        teacher_hidden = tf.cast(hidden_batch, tf.float32)
        cosine = tf.reduce_sum(student_hidden * teacher_hidden, axis=1)
        hidden_loss = tf.reduce_mean((1.0 - cosine) * weights)
        loss = loss + hidden_loss_weight * hidden_loss
    if group_loss_weight > 0.0 and group_logits is not None and label_to_group is not None:
        group_labels = tf.gather(label_to_group, label_batch)
        group_per_example = tf.keras.losses.sparse_categorical_crossentropy(
            group_labels,
            group_logits,
            from_logits=True,
        )
        group_per_example = tf.cast(group_per_example, tf.float32)
        loss = loss + group_loss_weight * tf.reduce_mean(group_per_example * weights)
    if self_loss_weight > 0.0 and self_probabilities_batch is not None:
        if temperature > 1.0:
            self_softened = tf.pow(
                tf.maximum(
                    self_probabilities_batch,
                    tf.constant(1e-8, dtype=self_probabilities_batch.dtype),
                ),
                1.0 / temperature,
            )
            self_targets = self_softened / tf.reduce_sum(self_softened, axis=1, keepdims=True)
            self_logits = logits / temperature
            self_scale = temperature * temperature
        else:
            self_targets = self_probabilities_batch
            self_logits = logits
            self_scale = 1.0
        self_per_example = tf.keras.losses.categorical_crossentropy(
            self_targets, self_logits, from_logits=True
        ) * self_scale
        self_per_example = tf.cast(self_per_example, tf.float32)
        self_loss = tf.reduce_mean(self_per_example * weights)
        loss = loss + tf.cast(self_loss_weight, tf.float32) * self_loss
    return loss


def model_outputs(
    model: tf.keras.Model,
    token_batch: tf.Tensor,
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    prediction = model(token_batch, training=training)
    if isinstance(prediction, (list, tuple)):
        hidden = None
        group_logits = None
        # Heuristic: hidden output is high-dim (>=256), group_logits is small
        # (~10 groups). This correctly handles both single-target (512), multi-
        # target (1024), and 3-target (1536) hidden distillation.
        for value in prediction[1:]:
            if (
                value.shape.rank is not None
                and value.shape[-1] is not None
                and value.shape[-1] >= 256
            ):
                hidden = value
            else:
                group_logits = value
        return prediction[0], hidden, group_logits
    return prediction, None, None


def hidden_or_none(hidden_batch: tf.Tensor) -> tf.Tensor | None:
    if hidden_batch.shape.rank is not None and hidden_batch.shape[-1] == 0:
        return None
    return hidden_batch


def self_probs_or_none(self_probs_batch: tf.Tensor | None) -> tf.Tensor | None:
    """Treat zero-feature self-probability batches (placeholder when the cache is
    absent) as None so the loss skips the new term cleanly."""
    if self_probs_batch is None:
        return None
    if self_probs_batch.shape.rank is not None and self_probs_batch.shape[-1] == 0:
        return None
    return self_probs_batch


def normalize_feature_for_distill(feature: tf.Tensor) -> tf.Tensor:
    feature = tf.cast(feature, tf.float32)
    if feature.shape.rank == 4:
        # HashEmbedding emits (batch, time, hash_count, channels). Match the
        # model's real trunk input by summing the K hash lookups first.
        feature = tf.reduce_sum(feature, axis=-2)
    return feature


def align_parent_feature_to_student(parent_feature: tf.Tensor, student_feature: tf.Tensor) -> tf.Tensor:
    target_width = tf.shape(student_feature)[-1]
    parent_width = tf.shape(parent_feature)[-1]
    keep = tf.minimum(parent_width, target_width)
    aligned = parent_feature[..., :keep]
    pad = tf.maximum(target_width - keep, 0)
    rank = student_feature.shape.rank
    if rank == 2:
        return tf.pad(aligned, [[0, 0], [0, pad]])
    if rank == 3:
        return tf.pad(aligned, [[0, 0], [0, 0], [0, pad]])
    return tf.reshape(aligned, (tf.shape(aligned)[0], -1))


def parent_feature_distillation_loss(parent_feature: tf.Tensor, student_feature: tf.Tensor) -> tf.Tensor:
    student = normalize_feature_for_distill(student_feature)
    parent = normalize_feature_for_distill(parent_feature)
    parent = align_parent_feature_to_student(parent, student)
    if student.shape.rank not in (2, 3):
        student = tf.reshape(student, (tf.shape(student)[0], -1))
        parent = tf.reshape(parent, (tf.shape(parent)[0], -1))
    student_norm = tf.norm(student, axis=-1)
    parent_norm = tf.norm(parent, axis=-1)
    mask = tf.logical_or(student_norm > 1e-6, parent_norm > 1e-6)
    student = tf.math.l2_normalize(student, axis=-1)
    parent = tf.stop_gradient(tf.math.l2_normalize(parent, axis=-1))
    cosine = tf.reduce_sum(student * parent, axis=-1)
    per_item = tf.where(mask, 1.0 - cosine, tf.zeros_like(cosine))
    return tf.reduce_sum(per_item) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)


def make_train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    label_to_group: tf.Tensor | None,
    temperature: float,
    hidden_loss_weight: float,
    hard_loss_weight: float,
    group_loss_weight: float,
    confidence_power: float,
    jit_compile: bool,
    self_loss_weight: float = 0.0,
    label_smoothing: float = 0.0,
    parent_feature_model: tf.keras.Model | None = None,
    student_feature_model: tf.keras.Model | None = None,
    parent_feature_loss_weight: float = 0.0,
):
    @tf.function(jit_compile=jit_compile)
    def train_step(token_batch, probability_batch, label_batch, hidden_batch, self_probs_batch):
        hidden = hidden_or_none(hidden_batch)
        self_probs = self_probs_or_none(self_probs_batch)
        with tf.GradientTape() as tape:
            logits, student_hidden, group_logits = model_outputs(model, token_batch, training=True)
            loss = distillation_loss(
                probability_batch,
                label_batch,
                logits,
                hidden,
                student_hidden,
                group_logits,
                label_to_group,
                temperature,
                hidden_loss_weight,
                hard_loss_weight,
                group_loss_weight,
                confidence_power,
                self_probabilities_batch=self_probs,
                self_loss_weight=self_loss_weight,
                label_smoothing=label_smoothing,
            )
            if (
                parent_feature_loss_weight > 0.0
                and parent_feature_model is not None
                and student_feature_model is not None
            ):
                parent_feature = parent_feature_model(token_batch, training=False)
                student_feature = student_feature_model(token_batch, training=True)
                loss = loss + tf.cast(parent_feature_loss_weight, tf.float32) * parent_feature_distillation_loss(
                    parent_feature,
                    student_feature,
                )
        gradients = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        ]
        optimizer.apply_gradients(grads_and_vars)
        return loss, tf.shape(label_batch)[0]

    return train_step


def make_eval_step(
    model: tf.keras.Model,
    label_to_group: tf.Tensor | None,
    temperature: float,
    hidden_loss_weight: float,
    hard_loss_weight: float,
    group_loss_weight: float,
    confidence_power: float,
    jit_compile: bool,
    self_loss_weight: float = 0.0,
    parent_feature_model: tf.keras.Model | None = None,
    student_feature_model: tf.keras.Model | None = None,
    parent_feature_loss_weight: float = 0.0,
):
    @tf.function(jit_compile=jit_compile)
    def eval_step(token_batch, probability_batch, label_batch, hidden_batch, self_probs_batch):
        hidden = hidden_or_none(hidden_batch)
        self_probs = self_probs_or_none(self_probs_batch)
        logits, student_hidden, group_logits = model_outputs(model, token_batch, training=False)
        loss = distillation_loss(
            probability_batch,
            label_batch,
            logits,
            hidden,
            student_hidden,
            group_logits,
            label_to_group,
            temperature,
            hidden_loss_weight,
            hard_loss_weight,
            group_loss_weight,
            confidence_power,
            self_probabilities_batch=self_probs,
            self_loss_weight=self_loss_weight,
        )
        if (
            parent_feature_loss_weight > 0.0
            and parent_feature_model is not None
            and student_feature_model is not None
        ):
            parent_feature = parent_feature_model(token_batch, training=False)
            student_feature = student_feature_model(token_batch, training=False)
            loss = loss + tf.cast(parent_feature_loss_weight, tf.float32) * parent_feature_distillation_loss(
                parent_feature,
                student_feature,
            )
        predictions = tf.argmax(logits, axis=1, output_type=label_batch.dtype)
        correct = tf.reduce_sum(tf.cast(tf.equal(predictions, label_batch), tf.int64))
        return loss, correct, tf.shape(label_batch)[0]

    return eval_step


def evaluate(
    model: tf.keras.Model,
    split: TokenSplit,
    batch_size: int,
    label_to_group: tf.Tensor | None,
    temperature: float,
    hidden_loss_weight: float,
    hard_loss_weight: float,
    group_loss_weight: float,
    confidence_power: float,
    self_loss_weight: float = 0.0,
) -> tuple[float, float]:
    correct = 0
    losses = []
    for (
        token_batch,
        probability_batch,
        label_batch,
        hidden_batch,
        self_probs_batch,
    ) in batches(split, batch_size, shuffle=False, seed=0):
        prediction = model(token_batch, training=False)
        if isinstance(prediction, (list, tuple)):
            logits, student_hidden, group_logits = model_outputs(model, token_batch, training=False)
        else:
            logits = prediction
            student_hidden = None
            group_logits = None
        self_probs_tf = (
            tf.convert_to_tensor(self_probs_batch, dtype=tf.float32)
            if self_probs_batch is not None
            else None
        )
        loss = distillation_loss(
            tf.convert_to_tensor(probability_batch, dtype=tf.float32),
            tf.convert_to_tensor(label_batch, dtype=tf.int64),
            logits,
            hidden_batch,
            student_hidden,
            group_logits,
            label_to_group,
            temperature,
            hidden_loss_weight,
            hard_loss_weight,
            group_loss_weight,
            confidence_power,
            self_probabilities_batch=self_probs_tf,
            self_loss_weight=self_loss_weight,
        )
        losses.append(float(loss.numpy()) * len(label_batch))
        correct += int((logits.numpy().argmax(axis=1) == label_batch).sum())
    return correct / split.count, sum(losses) / split.count


def evaluate_dataset(
    model: tf.keras.Model,
    eval_step,
    dataset: tf.data.Dataset,
    count: int,
    classes: int,
    collect_confusion: bool = False,
) -> tuple[float, float, float, np.ndarray | None]:
    correct = 0
    loss_sum = 0.0
    seen = 0
    confusion = np.zeros((classes, classes), dtype=np.int64) if collect_confusion else None
    started = time.perf_counter()
    for token_batch, probability_batch, label_batch, hidden_batch, self_probs_batch in dataset:
        loss, batch_correct, batch_size = eval_step(
            token_batch, probability_batch, label_batch, hidden_batch, self_probs_batch
        )
        batch_n = int(batch_size.numpy())
        seen += batch_n
        correct += int(batch_correct.numpy())
        loss_sum += float(loss.numpy()) * batch_n
        if confusion is not None:
            logits, _, _ = model_outputs(model, token_batch, training=False)
            predictions = logits.numpy().argmax(axis=1)
            np.add.at(confusion, (label_batch.numpy(), predictions), 1)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return correct / count, loss_sum / count, seen / elapsed, confusion


def confusion_summary(confusion: np.ndarray, labels: list[str], limit: int) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for actual_index, actual in enumerate(labels):
        total = int(confusion[actual_index].sum())
        if total == 0:
            continue
        correct = int(confusion[actual_index, actual_index])
        for predicted_index, predicted in enumerate(labels):
            if predicted_index == actual_index:
                continue
            count = int(confusion[actual_index, predicted_index])
            if count == 0:
                continue
            rows.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "actual_total": total,
                    "actual_recall": correct / total,
                    "share_of_actual": count / total,
                }
            )
    rows.sort(key=lambda row: int(row["count"]), reverse=True)
    return rows[:limit]


def write_confusion_matrix(path: Path, confusion: np.ndarray, labels: list[str], limit: int) -> None:
    payload = {
        "labels": labels,
        "matrix": confusion.tolist(),
        "top_confusions": confusion_summary(confusion, labels, limit),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def pack_int4(values: np.ndarray) -> bytes:
    flat = values.reshape(-1).astype(np.int8)
    encoded = np.clip(flat + 8, 0, 15).astype(np.uint8)
    packed = bytearray((len(encoded) + 1) // 2)
    for index, value in enumerate(encoded):
        if index % 2 == 0:
            packed[index // 2] |= int(value)
        else:
            packed[index // 2] |= int(value) << 4
    return bytes(packed)


def pack_ternary(values: np.ndarray) -> bytes:
    flat = values.reshape(-1).astype(np.int8)
    encoded = np.where(flat < 0, 0, np.where(flat > 0, 2, 1)).astype(np.uint8)
    packed = bytearray((len(encoded) + 3) // 4)
    for index, value in enumerate(encoded):
        packed[index // 4] |= int(value) << ((index % 4) * 2)
    return bytes(packed)


def quantize_int4(weight: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = max(float(np.max(np.abs(weight))), 1e-6)
    scale = max_abs / 7.0
    quantized = np.clip(np.rint(weight / scale), -7, 7).astype(np.int8)
    return quantized, scale


def quantize_weight(weight: np.ndarray, bits: int) -> tuple[bytes, float, str]:
    if not np.all(np.isfinite(weight)):
        raise ValueError("cannot export non-finite weight tensor")
    if bits == 2:
        # Match _quantize_for_bits(bits=2) and make export->load->export stable.
        abs_weight = np.abs(weight)
        nonzero_abs = abs_weight[abs_weight > 0.0]
        scale = max(float(np.mean(nonzero_abs)) if nonzero_abs.size else 1e-6, 1e-6)
        threshold = 0.7 * scale
        quantized = np.where(weight > threshold, 1, np.where(weight < -threshold, -1, 0)).astype(np.int8)
        return pack_ternary(quantized), scale, "ternary"
    if bits == 4:
        quantized, scale = quantize_int4(weight)
        return pack_int4(quantized), scale, "int4"
    raise ValueError(f"unsupported weight bits: {bits}")


def export_model(
    output: Path,
    model: tf.keras.Model,
    labels: list[str],
    slugs: list[str],
    bits: int,
    architecture: str,
    tokenizer_version: int | None = None,
) -> int:
    metadata = {
        "bits": bits,
        "token_length": TOKEN_LENGTH,
        "architecture": architecture,
        "labels": labels,
        "slugs": slugs,
        "layers": [],
    }
    if tokenizer_version is not None:
        metadata["tokenizer_version"] = tokenizer_version
    blob = bytearray(QAT_MAGIC)
    for layer in model.layers:
        if isinstance(layer, QEmbedding):
            weights = [("embedding", layer.embedding.numpy())]
            biases = []
        elif isinstance(layer, QConv1D):
            weights = [("kernel", layer.kernel.numpy())]
            biases = [("bias", layer.bias.numpy())]
        elif isinstance(layer, QDepthwiseConv1D):
            weights = [("kernel", layer.kernel.numpy())]
            biases = [("bias", layer.bias.numpy())]
        elif isinstance(layer, QDense):
            if layer.name in {"q_hidden_project", "q_group_output"}:
                continue
            weights = [("kernel", layer.kernel.numpy())]
            biases = [("bias", layer.bias.numpy())]
        else:
            continue
        layer_meta = {"name": layer.name, "weights": [], "biases": []}
        layer_bits = getattr(layer, "bits", bits)
        for name, value in weights:
            payload, scale, encoding = quantize_weight(value, layer_bits)
            layer_meta["weights"].append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "scale": scale,
                    "bits": layer_bits,
                    "encoding": encoding,
                    "bytes": len(payload),
                }
            )
            blob.extend(payload)
        for name, value in biases:
            data = value.astype("<f4").tobytes()
            layer_meta["biases"].append({"name": name, "shape": list(value.shape), "bytes": len(data)})
            blob.extend(data)
        metadata["layers"].append(layer_meta)

    metadata_json = json.dumps(metadata, separators=(",", ":")).encode()
    blob.extend(len(metadata_json).to_bytes(4, "little"))
    blob.extend(metadata_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    return len(blob)


def unpack_int4(payload: bytes, shape: list[int]) -> np.ndarray:
    count = int(np.prod(shape))
    raw = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(len(raw) * 2, dtype=np.int8)
    values[0::2] = raw & 0x0F
    values[1::2] = raw >> 4
    values = values[:count].astype(np.float32) - 8.0
    return values.reshape(shape)


def unpack_ternary(payload: bytes, shape: list[int]) -> np.ndarray:
    count = int(np.prod(shape))
    raw = np.frombuffer(payload, dtype=np.uint8)
    codes = np.empty(len(raw) * 4, dtype=np.uint8)
    codes[0::4] = raw & 0x03
    codes[1::4] = (raw >> 2) & 0x03
    codes[2::4] = (raw >> 4) & 0x03
    codes[3::4] = (raw >> 6) & 0x03
    codes = codes[:count]
    out = np.zeros(count, dtype=np.float32)
    out[codes == 0] = -1.0
    out[codes == 2] = 1.0
    return out.reshape(shape)


def load_exported_layer_weights(path: Path) -> tuple[dict[str, list[np.ndarray]], dict]:
    data = path.read_bytes()
    if not data.startswith(QAT_MAGIC):
        raise ValueError(f"{path} is not an MSQ1 checkpoint")
    meta_start = data.rfind(b'{"bits"')
    if meta_start < 0:
        raise ValueError(f"metadata not found in {path}")
    metadata = json.loads(data[meta_start:])
    layers: dict[str, list[np.ndarray]] = {}
    cursor = len(QAT_MAGIC)
    for layer_meta in metadata["layers"]:
        values: list[np.ndarray] = []
        for weight_meta in layer_meta["weights"]:
            nbytes = int(weight_meta["bytes"])
            payload = data[cursor : cursor + nbytes]
            cursor += nbytes
            shape = list(weight_meta["shape"])
            scale = float(weight_meta["scale"])
            encoding = weight_meta["encoding"]
            if encoding == "int4":
                decoded = unpack_int4(payload, shape) * scale
            elif encoding == "ternary":
                decoded = unpack_ternary(payload, shape) * scale
            else:
                raise ValueError(f"unsupported weight encoding {encoding!r}")
            values.append(decoded.astype(np.float32))
        for bias_meta in layer_meta["biases"]:
            nbytes = int(bias_meta["bytes"])
            payload = data[cursor : cursor + nbytes]
            cursor += nbytes
            values.append(np.frombuffer(payload, dtype="<f4").reshape(bias_meta["shape"]).copy())
        layers[str(layer_meta["name"])] = values
    return layers, metadata


def collapse_or_tile_rows(source: np.ndarray, rows: int) -> np.ndarray:
    if source.shape[0] == rows:
        return source
    if source.shape[0] > rows:
        collapsed = np.zeros((rows, source.shape[1]), dtype=np.float32)
        counts = np.zeros((rows,), dtype=np.float32)
        for row in range(source.shape[0]):
            dst = row % rows
            collapsed[dst] += source[row]
            counts[dst] += 1.0
        return collapsed / np.maximum(counts[:, None], 1.0)
    return source[np.arange(rows) % source.shape[0]]


AxisSelectors = dict[tuple[str, int, int], np.ndarray]


def top_channel_indices(scores: np.ndarray, count: int) -> np.ndarray:
    if count >= scores.shape[0]:
        return np.arange(scores.shape[0], dtype=np.int64)
    safe_scores = np.nan_to_num(scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    selected = np.argpartition(-safe_scores, count - 1)[:count]
    return np.array(sorted(int(index) for index in selected), dtype=np.int64)


def l2_by_axis(value: np.ndarray, axis: int) -> np.ndarray:
    axes = tuple(i for i in range(value.ndim) if i != axis)
    return np.sqrt(np.sum(np.square(value.astype(np.float32)), axis=axes))


def layer_weight_shapes(model: tf.keras.Model) -> dict[str, list[tuple[int, ...]]]:
    shapes: dict[str, list[tuple[int, ...]]] = {}
    for layer in model.layers:
        if isinstance(layer, (QEmbedding, QConv1D, QDepthwiseConv1D, QDense)):
            shapes[layer.name] = [tuple(value.shape) for value in layer.get_weights()]
    return shapes


def add_axis_selector(
    selectors: AxisSelectors,
    layer_name: str,
    value_index: int,
    axis: int,
    indices: np.ndarray,
) -> None:
    selectors[(layer_name, value_index, axis)] = indices.astype(np.int64, copy=False)


def add_dense_channel_selectors(
    selectors: AxisSelectors,
    source_layers: dict[str, list[np.ndarray]],
    target_shapes: dict[str, list[tuple[int, ...]]],
) -> None:
    if "q_dense_0" not in source_layers or "q_output" not in source_layers:
        return
    if "q_dense_0" not in target_shapes or "q_output" not in target_shapes:
        return
    dense_kernel = source_layers["q_dense_0"][0]
    dense_bias = source_layers["q_dense_0"][1]
    output_kernel = source_layers["q_output"][0]
    old_units = dense_kernel.shape[1]
    new_units = target_shapes["q_dense_0"][0][1]
    if new_units >= old_units:
        return
    scores = l2_by_axis(output_kernel, 0)
    scores = scores[:old_units] + l2_by_axis(dense_kernel, 1)
    if dense_bias.shape[0] == old_units:
        scores = scores + np.abs(dense_bias.astype(np.float32))
    keep = top_channel_indices(scores, new_units)
    add_axis_selector(selectors, "q_dense_0", 0, 1, keep)
    add_axis_selector(selectors, "q_dense_0", 1, 0, keep)
    add_axis_selector(selectors, "q_output", 0, 0, keep)


def pooled_rows_for_channels(channel_indices: np.ndarray, old_channels: int) -> np.ndarray:
    return np.concatenate([channel_indices, old_channels + channel_indices]).astype(np.int64)


def add_conv_output_selectors(
    selectors: AxisSelectors,
    source_layers: dict[str, list[np.ndarray]],
    target_shapes: dict[str, list[tuple[int, ...]]],
    *,
    layer_name: str,
    consumer_name: str | None,
    dense_consumer: bool = False,
) -> np.ndarray | None:
    if layer_name not in source_layers or layer_name not in target_shapes:
        return None
    kernel = source_layers[layer_name][0]
    bias = source_layers[layer_name][1]
    old_channels = kernel.shape[2]
    new_channels = target_shapes[layer_name][0][2]
    if new_channels >= old_channels:
        return None
    scores = l2_by_axis(kernel, 2)
    if bias.shape[0] == old_channels:
        scores = scores + np.abs(bias.astype(np.float32))
    if consumer_name and consumer_name in source_layers:
        consumer_kernel = source_layers[consumer_name][0]
        if dense_consumer:
            if consumer_kernel.shape[0] == old_channels * 2:
                max_rows = consumer_kernel[:old_channels, :]
                avg_rows = consumer_kernel[old_channels : old_channels * 2, :]
                scores = scores + np.sqrt(np.sum(np.square(max_rows), axis=1))
                scores = scores + np.sqrt(np.sum(np.square(avg_rows), axis=1))
        elif consumer_kernel.ndim == 3 and consumer_kernel.shape[1] == old_channels:
            scores = scores + l2_by_axis(consumer_kernel, 1)
    keep = top_channel_indices(scores, new_channels)
    add_axis_selector(selectors, layer_name, 0, 2, keep)
    add_axis_selector(selectors, layer_name, 1, 0, keep)
    if consumer_name:
        if dense_consumer:
            add_axis_selector(selectors, consumer_name, 0, 0, pooled_rows_for_channels(keep, old_channels))
        else:
            add_axis_selector(selectors, consumer_name, 0, 1, keep)
    return keep


def add_embedding_selectors(
    selectors: AxisSelectors,
    source_layers: dict[str, list[np.ndarray]],
    target_shapes: dict[str, list[tuple[int, ...]]],
) -> None:
    if "q_hash_embedding" not in source_layers or "q_hash_embedding" not in target_shapes:
        return
    if "q_conv_0" not in source_layers:
        return
    embedding = source_layers["q_hash_embedding"][0]
    old_dims = embedding.shape[1]
    new_dims = target_shapes["q_hash_embedding"][0][1]
    if new_dims >= old_dims:
        return
    scores = np.sqrt(np.sum(np.square(embedding.astype(np.float32)), axis=0))
    conv0_kernel = source_layers["q_conv_0"][0]
    if conv0_kernel.ndim == 3 and conv0_kernel.shape[1] == old_dims:
        scores = scores + l2_by_axis(conv0_kernel, 1)
    keep = top_channel_indices(scores, new_dims)
    add_axis_selector(selectors, "q_hash_embedding", 0, 1, keep)
    add_axis_selector(selectors, "q_conv_0", 0, 1, keep)


def build_axis_selectors(
    source_layers: dict[str, list[np.ndarray]],
    target_shapes: dict[str, list[tuple[int, ...]]],
) -> AxisSelectors:
    selectors: AxisSelectors = {}
    add_dense_channel_selectors(selectors, source_layers, target_shapes)
    if "q_conv_2" in source_layers and "q_conv_2" in target_shapes:
        add_conv_output_selectors(
            selectors,
            source_layers,
            target_shapes,
            layer_name="q_conv_2",
            consumer_name="q_dense_0",
            dense_consumer=True,
        )
        add_conv_output_selectors(
            selectors,
            source_layers,
            target_shapes,
            layer_name="q_conv_1",
            consumer_name="q_conv_2",
        )
    else:
        add_conv_output_selectors(
            selectors,
            source_layers,
            target_shapes,
            layer_name="q_conv_1",
            consumer_name="q_dense_0",
            dense_consumer=True,
        )
    add_conv_output_selectors(
        selectors,
        source_layers,
        target_shapes,
        layer_name="q_conv_0",
        consumer_name="q_conv_1",
    )
    add_embedding_selectors(selectors, source_layers, target_shapes)
    return selectors


def apply_axis_selectors(
    source: np.ndarray,
    selectors: AxisSelectors,
    layer_name: str,
    value_index: int,
) -> np.ndarray:
    selected = source
    for axis in range(source.ndim):
        indices = selectors.get((layer_name, value_index, axis))
        if indices is None:
            continue
        if selected.shape[axis] <= indices.shape[0]:
            continue
        selected = np.take(selected, indices, axis=axis)
    return selected


def adapted_array(
    source: np.ndarray,
    target_template: np.ndarray,
    layer_name: str,
    value_index: int,
    selectors: AxisSelectors | None = None,
) -> np.ndarray:
    if selectors:
        source = apply_axis_selectors(source, selectors, layer_name, value_index)
    target = np.array(target_template, dtype=np.float32, copy=True)
    shape = target.shape
    if source.ndim == 1:
        n = min(source.shape[0], shape[0])
        target[:n] = source[:n]
        return target
    if layer_name == "q_hash_embedding" and source.ndim == 2:
        rows = collapse_or_tile_rows(source, shape[0])
        cols = min(rows.shape[1], shape[1])
        target[:, :cols] = rows[:, :cols]
        return target
    if source.ndim == 3:
        k = min(source.shape[0], shape[0])
        src_k0 = max((source.shape[0] - k) // 2, 0)
        dst_k0 = max((shape[0] - k) // 2, 0)
        in_ch = min(source.shape[1], shape[1])
        out_ch = min(source.shape[2], shape[2])
        target[dst_k0 : dst_k0 + k, :in_ch, :out_ch] = source[
            src_k0 : src_k0 + k, :in_ch, :out_ch
        ]
        return target
    if source.ndim == 2:
        cols = min(source.shape[1], shape[1])
        if layer_name == "q_dense_0" and source.shape[0] % 2 == 0 and shape[0] % 2 == 0:
            src_half = source.shape[0] // 2
            dst_half = shape[0] // 2
            half = min(src_half, dst_half)
            target[:half, :cols] = source[:half, :cols]
            target[dst_half : dst_half + half, :cols] = source[src_half : src_half + half, :cols]
        else:
            rows = min(source.shape[0], shape[0])
            target[:rows, :cols] = source[:rows, :cols]
        return target
    return target


def initialize_model_from_checkpoint(
    model: tf.keras.Model,
    checkpoint: Path,
    channel_selection: str = "first",
) -> tuple[set[str], set[str]]:
    source_layers, metadata = load_exported_layer_weights(checkpoint)
    if channel_selection == "first":
        selectors: AxisSelectors = {}
    elif channel_selection == "importance":
        selectors = build_axis_selectors(source_layers, layer_weight_shapes(model))
    else:
        raise ValueError(f"unknown channel selection mode: {channel_selection}")
    initialized: set[str] = set()
    adapted_layers: set[str] = set()
    skipped: list[str] = []
    selector_summaries = [
        f"{layer}[{value_index}].axis{axis}={len(indices)}"
        for (layer, value_index, axis), indices in sorted(selectors.items())
    ]
    for layer in model.layers:
        if not isinstance(layer, (QEmbedding, QConv1D, QDepthwiseConv1D, QDense)):
            continue
        if layer.name not in source_layers:
            skipped.append(layer.name)
            continue
        target_values = layer.get_weights()
        source_values = source_layers[layer.name]
        if not target_values or not source_values:
            continue
        layer_adapted = any(
            source_values[i].shape != target_values[i].shape
            for i in range(min(len(source_values), len(target_values)))
        )
        adapted = [
            adapted_array(source_values[i], target_values[i], layer.name, i, selectors)
            for i in range(min(len(source_values), len(target_values)))
        ]
        if len(adapted) != len(target_values):
            skipped.append(layer.name)
            continue
        layer.set_weights(adapted)
        initialized.add(layer.name)
        if layer_adapted:
            adapted_layers.add(layer.name)
    print(
        "init_from_checkpoint="
        f"{checkpoint} source_architecture={metadata.get('architecture')} "
        f"channel_selection={channel_selection} "
        f"initialized_layers={','.join(sorted(initialized)) if initialized else 'none'} "
        f"adapted_layers={','.join(sorted(adapted_layers)) if adapted_layers else 'none'} "
        f"axis_selectors={','.join(selector_summaries) if selector_summaries else 'none'} "
        f"skipped_layers={','.join(skipped) if skipped else 'none'}",
        flush=True,
    )
    return initialized, adapted_layers


def layer_feature_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    try:
        layer = model.get_layer(layer_name)
    except ValueError as exc:
        available = ",".join(layer.name for layer in model.layers)
        raise ValueError(
            f"feature layer {layer_name!r} not found; available layers: {available}"
        ) from exc
    return tf.keras.Model(inputs=model.input, outputs=layer.output)


def build_parent_feature_model(
    checkpoint: Path,
    classes: int,
    fallback_bits: int,
    hidden_dim: int,
    layer_name: str,
    channel_selection: str = "first",
) -> tuple[tf.keras.Model, str]:
    _, metadata = load_exported_layer_weights(checkpoint)
    architecture = str(metadata["architecture"])
    bits = int(metadata.get("bits", fallback_bits))
    parent = build_model(classes, bits, architecture, hidden_dim=hidden_dim)
    initialize_model_from_checkpoint(parent, checkpoint, channel_selection=channel_selection)
    for layer in parent.layers:
        layer.trainable = False
    feature_model = layer_feature_model(parent, layer_name)
    print(
        "parent_feature_teacher="
        f"{checkpoint} architecture={architecture} layer={layer_name}",
        flush=True,
    )
    return feature_model, architecture


def apply_freeze_policy(
    model: tf.keras.Model,
    initialized_layers: set[str],
    adapted_layers: set[str],
    freeze_policy: str,
    trainable_layers_csv: str | None,
) -> None:
    explicitly_trainable = {
        name.strip()
        for name in (trainable_layers_csv or "").split(",")
        if name.strip()
    }
    if freeze_policy == "none":
        frozen: set[str] = set()
    elif freeze_policy == "unchanged":
        frozen = initialized_layers - adapted_layers
    elif freeze_policy == "initialized":
        frozen = set(initialized_layers)
    else:
        raise ValueError(f"unknown freeze policy: {freeze_policy}")

    for layer in model.layers:
        if layer.name in explicitly_trainable:
            layer.trainable = True
        elif layer.name in frozen:
            layer.trainable = False

    if explicitly_trainable:
        for layer in model.layers:
            if isinstance(layer, (QEmbedding, QConv1D, QDepthwiseConv1D, QDense)):
                layer.trainable = layer.name in explicitly_trainable

    trainable = [
        layer.name
        for layer in model.layers
        if isinstance(layer, (QEmbedding, QConv1D, QDepthwiseConv1D, QDense)) and layer.trainable
    ]
    non_trainable = [
        layer.name
        for layer in model.layers
        if isinstance(layer, (QEmbedding, QConv1D, QDepthwiseConv1D, QDense)) and not layer.trainable
    ]
    print(
        "freeze_policy="
        f"{freeze_policy} explicit_trainable={','.join(sorted(explicitly_trainable)) if explicitly_trainable else 'none'} "
        f"trainable_layers={','.join(trainable) if trainable else 'none'} "
        f"frozen_layers={','.join(non_trainable) if non_trainable else 'none'}",
        flush=True,
    )


def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return tuple(sorted(set(parsed)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--magika-model", type=Path, required=True)
    parser.add_argument("--magika-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="initialize overlapping/smaller layers from an exported MSQ1 checkpoint; "
             "hash tables are collapsed by bucket modulo when shrinking bin counts",
    )
    parser.add_argument(
        "--freeze-init-policy",
        choices=("none", "unchanged", "initialized"),
        default="none",
        help="after --init-from-checkpoint, freeze no initialized layers, only exact-copy "
             "unchanged layers, or all initialized layers",
    )
    parser.add_argument(
        "--channel-selection",
        choices=("first", "importance"),
        default="first",
        help="when shrinking checkpoint tensors, keep the first overlapping channels "
             "or choose channels by simple downstream weight-norm importance",
    )
    parser.add_argument(
        "--trainable-layers",
        default=None,
        help="comma-separated Q layer names to train exclusively after init, e.g. "
             "q_hash_embedding or q_conv_0,q_conv_1. Overrides --freeze-init-policy.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="cap train batches per epoch; useful for short progressive shrink calibration",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--balance-power", type=float, default=0.0)
    parser.add_argument("--hard-mask-mmap", type=Path, default=None,
                        help="uint8 per-train-example mask (1=hard); see scripts/score_train_difficulty.py")
    parser.add_argument("--hard-oversample-rate", type=float, default=1.0,
                        help="hard examples sampled at this rate vs easy (1.0 disables)")
    parser.add_argument("--qat-start-epoch", type=int, default=0,
                        help="train fp16 until this epoch, then enable QAT fake_quant")
    parser.add_argument(
        "--architecture",
        default="chunked-flat",
    )
    parser.add_argument("--teacher-hidden-output")
    parser.add_argument("--hidden-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--hidden-loss-target",
        choices=("head", "trunk"),
        default="head",
        help="use q_hidden_project output (head) or the pooled trunk feature (trunk) "
             "for hidden/intermediate distillation",
    )
    parser.add_argument("--hard-loss-weight", type=float, default=0.25)
    parser.add_argument("--group-loss-weight", type=float, default=0.0)
    parser.add_argument("--confidence-power", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--tf-data-batches", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--eval-initial",
        action="store_true",
        help="evaluate the initialized model before training and use it as the initial best checkpoint.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--confusion-matrix-output", type=Path)
    parser.add_argument("--confusion-matrix-top", type=int, default=20)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--xla", action="store_true")
    parser.add_argument(
        "--self-probabilities",
        type=Path,
        default=None,
        help="directory containing {split}.self_probabilities.mmap files produced by "
             "scripts/cache_self_distill.py (defaults to --cache-dir if a flag-only "
             "string 'cache-dir' is given). Enables an additional soft-CE term against "
             "a previously trained student's predictions.",
    )
    parser.add_argument(
        "--self-loss-weight",
        type=float,
        default=0.0,
        help="weight for the self-distillation soft-CE term (default 0.0 disables).",
    )
    parser.add_argument(
        "--parent-feature-checkpoint",
        type=Path,
        default=None,
        help="MSQ1 checkpoint used as an online parent activation teacher for the "
             "single layer named by --parent-feature-layer.",
    )
    parser.add_argument(
        "--parent-feature-layer",
        default=None,
        help="Layer whose parent/student activations should be matched during this "
             "stage, e.g. apply_pad_mask, q_conv_0, q_conv_1, q_conv_2, q_dense_0.",
    )
    parser.add_argument(
        "--parent-feature-loss-weight",
        type=float,
        default=0.0,
        help="Weight for online parent activation loss. This is stage-local and "
             "does not use hidden mmap caches.",
    )
    parser.add_argument(
        "--cutmix-prob",
        type=float,
        default=0.0,
        help="probability that a given training example is mixed with another via "
             "CutMix (random byte-cut; teacher probs and hidden also mixed linearly). "
             "Magika v3.1+ credits this for accuracy gains. Default 0.0 disables.",
    )
    parser.add_argument(
        "--short-slice-prob",
        type=float,
        default=0.0,
        help="For wordseq/wordwin training only: probability that a training row is "
             "masked after a sampled short unit length. With --short-slice-target-cache-dir, "
             "the row also uses the Magika teacher target for that short prefix; otherwise "
             "it keeps the legacy full-file target. Default 0.0 disables.",
    )
    parser.add_argument(
        "--short-slice-unit-lengths",
        type=parse_positive_int_csv,
        default=parse_positive_int_csv("64,128,256,512"),
        help="Comma-separated unit lengths sampled by --short-slice-prob. Values are "
             "applied after the unit tokenizer, so this is cheap and works with cached "
             "units. Default: 64,128,256,512.",
    )
    parser.add_argument(
        "--short-slice-target-cache-dir",
        type=Path,
        default=None,
        help="directory containing short-slice teacher target mmap files. When set with "
             "--short-slice-prob, sliced rows train against the teacher distribution for "
             "the sliced prefix instead of the original full-file distribution.",
    )
    parser.add_argument(
        "--short-slice-target-mode",
        choices=("none", "teacher-cache", "fs-label"),
        default="none",
        help="Target used for sliced rows. none keeps the current full-file targets; "
             "teacher-cache uses --short-slice-target-cache-dir; fs-label uses a "
             "one-hot filesystem label from --fs-labels-dir.",
    )
    parser.add_argument(
        "--build-short-slice-target-cache",
        action="store_true",
        help="build short-slice teacher target mmap files before training. Defaults to "
             "--cache-dir when --short-slice-target-cache-dir is not set.",
    )
    parser.add_argument(
        "--rebuild-short-slice-target-cache",
        action="store_true",
        help="force rebuilding short-slice teacher target mmap files.",
    )
    parser.add_argument(
        "--prepare-short-slice-target-cache-only",
        action="store_true",
        help="build the regular cache and the train short-slice target cache, then exit.",
    )
    parser.add_argument(
        "--short-slice-target-min-confidence",
        type=float,
        default=0.0,
        help="minimum top teacher probability required to use a cached short-slice "
             "target. Rows below this threshold are left unsliced. Default 0.0.",
    )
    parser.add_argument(
        "--fs-labels-dir",
        type=Path,
        default=None,
        help="directory containing {split}.fs_labels.mmap from build_fs_labels.py. "
             "When set, the hard CE term targets filesystem-extension labels instead "
             "of the teacher's argmax; the soft term still uses the teacher probs. "
             "Reported '*_teacher_parity' becomes '*_fs_accuracy' (same metric, "
             "different label source).",
    )
    parser.add_argument(
        "--unit-tokenizer",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Word-unit tokenizer version. v1: original (each non-word char = 1 unit). "
             "v2: collapses punct-runs into one hashed unit and merges digit-runs into "
             "a single _NUM_FLAG unit. v3: v2 + case-fold word chars + brackets emit as "
             "their own _BRACKET_FLAG token (never merged into adjacent punct). "
             "v4: v3 + `\"...\"` strings collapse to single _STRING_FLAG token + "
             "_STYLE_BIT set on word hash if original had any uppercase. "
             "Each version caches to {split}.units_v{N}.mmap (v1: just .units.mmap).",
    )
    parser.add_argument(
        "--length-buckets",
        action="store_true",
        help="For wordseq/wordwin: bucket training samples by unit length and trim "
             "each batch to its bucket max. Saves ~2-3× train compute on short files "
             "since most files have <1000 units but TOKEN_LENGTH=2048. tf.function "
             "retraces once per distinct bucket size.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Per-class label smoothing eps for the hard CE term. target = (1 - eps) "
             "* one_hot + eps / classes. 0.0 disables. Typical values: 0.05 - 0.1.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default 1e-4. Raise to 1e-3 / 3e-3 for stronger "
             "regularization when the model memorizes the training set.",
    )
    parser.add_argument(
        "--global-clipnorm",
        type=float,
        default=1.0,
        help="AdamW global gradient clip norm. Set <=0 to disable. Default 1.0.",
    )
    args = parser.parse_args()

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not 0.0 <= args.hard_loss_weight <= 1.0:
        raise ValueError("--hard-loss-weight must be in [0, 1]")
    if not 0.0 <= args.short_slice_prob <= 1.0:
        raise ValueError("--short-slice-prob must be in [0, 1]")
    if not 0.0 <= args.short_slice_target_min_confidence <= 1.0:
        raise ValueError("--short-slice-target-min-confidence must be in [0, 1]")
    if args.short_slice_prob > 0.0 and not architecture_uses_word_units(args.architecture):
        raise ValueError("--short-slice-prob only applies to wordseq/wordwin architectures")
    if args.short_slice_target_mode == "teacher-cache" and args.short_slice_target_cache_dir is None:
        raise SystemExit("--short-slice-target-mode=teacher-cache requires --short-slice-target-cache-dir")
    if args.short_slice_target_mode == "fs-label" and args.fs_labels_dir is None:
        raise SystemExit("--short-slice-target-mode=fs-label requires --fs-labels-dir")
    if args.short_slice_target_mode == "none" and args.short_slice_target_cache_dir is not None:
        raise SystemExit("--short-slice-target-cache-dir requires --short-slice-target-mode=teacher-cache")
    short_slice_target_dir = args.short_slice_target_cache_dir
    if args.build_short_slice_target_cache or args.prepare_short_slice_target_cache_only:
        short_slice_target_dir = short_slice_target_dir or args.cache_dir
        if args.short_slice_target_mode == "none":
            args.short_slice_target_mode = "teacher-cache"
    if (
        short_slice_target_dir is not None
        and args.short_slice_prob > 0.0
        and args.tf_data_batches
    ):
        raise SystemExit(
            "--short-slice-target-cache-dir requires NumPy batches; drop --tf-data-batches"
        )
    if (
        short_slice_target_dir is not None
        and args.short_slice_prob > 0.0
        and args.hidden_loss_weight > 0.0
    ):
        raise SystemExit(
            "--short-slice-target-cache-dir cannot be combined with --hidden-loss-weight; "
            "there are no cached hidden targets for sliced prefixes"
        )
    if (
        args.short_slice_target_mode == "fs-label"
        and args.short_slice_prob > 0.0
        and args.hidden_loss_weight > 0.0
    ):
        raise SystemExit(
            "--short-slice-target-mode=fs-label cannot be combined with --hidden-loss-weight; "
            "there are no hidden targets for sliced prefixes"
        )
    if (
        args.short_slice_target_mode == "fs-label"
        and args.short_slice_prob > 0.0
        and args.cutmix_prob > 0.0
    ):
        raise SystemExit(
            "--short-slice-target-mode=fs-label cannot be combined with --cutmix-prob > 0; "
            "sliced examples need to remain single-file examples"
        )

    tf.keras.utils.set_random_seed(args.seed)
    teacher = load_teacher(args.magika_model, args.magika_config)
    classes = len(teacher.selected_labels)
    counts = ensure_cache(
        args.dataset,
        args.cache_dir,
        teacher,
        args.limit_per_split,
        args.teacher_batch_size,
        args.rebuild_cache,
        args.teacher_hidden_output,
    )
    if args.build_short_slice_target_cache or args.prepare_short_slice_target_cache_only:
        assert short_slice_target_dir is not None
        ensure_short_slice_target_cache(
            args.cache_dir,
            short_slice_target_dir,
            teacher,
            args.short_slice_unit_lengths,
            args.teacher_batch_size,
            args.rebuild_short_slice_target_cache,
            splits=("train",),
        )
    if args.prepare_short_slice_target_cache_only:
        print("short_slice_target_cache_ready=true")
        return
    if args.prepare_cache_only:
        print("cache_ready=true")
        return

    self_probabilities_dir = args.self_probabilities
    if self_probabilities_dir is not None and args.self_loss_weight <= 0.0:
        print(
            "warning: --self-probabilities provided but --self-loss-weight=0.0; "
            "self-distillation term will be skipped",
            flush=True,
        )
    if self_probabilities_dir is None and args.self_loss_weight > 0.0:
        raise SystemExit(
            "--self-loss-weight > 0 requires --self-probabilities <dir>"
        )
    if args.parent_feature_loss_weight > 0.0:
        if args.parent_feature_checkpoint is None or not args.parent_feature_layer:
            raise SystemExit(
                "--parent-feature-loss-weight > 0 requires "
                "--parent-feature-checkpoint and --parent-feature-layer"
            )
    load_hidden = args.hidden_loss_weight > 0.0
    train = open_split(args.cache_dir, "train", classes, self_probabilities_dir, args.fs_labels_dir, load_hidden)
    valid = open_split(args.cache_dir, "valid", classes, self_probabilities_dir, args.fs_labels_dir, load_hidden)
    test = open_split(args.cache_dir, "test", classes, self_probabilities_dir, args.fs_labels_dir, load_hidden)
    if self_probabilities_dir is not None:
        print(
            f"self_probabilities_dir={self_probabilities_dir} "
            f"self_loss_weight={args.self_loss_weight}",
            flush=True,
        )
    train_unit_lengths: np.ndarray | None = None
    short_slice_lengths: tuple[int, ...] = ()
    if architecture_uses_word_units(args.architecture):
        train, valid, test = convert_splits_to_word_units(
            args.cache_dir, train, valid, test, tokenizer_version=args.unit_tokenizer
        )
        if args.short_slice_prob > 0.0:
            short_slice_lengths = tuple(sorted(set(args.short_slice_unit_lengths)))
            print(
                f"short_slice_prob={args.short_slice_prob:.3f} "
                f"short_slice_unit_lengths={','.join(str(v) for v in short_slice_lengths)}",
                flush=True,
            )
            if args.short_slice_target_mode == "teacher-cache":
                train = attach_short_slice_targets(
                    train,
                    short_slice_target_dir,
                    "train",
                    short_slice_lengths,
                    classes,
                    tuple(teacher.selected_labels),
                )
                print(
                    f"short_slice_target_min_confidence="
                    f"{args.short_slice_target_min_confidence:.3f}",
                    flush=True,
                )
            elif args.short_slice_target_mode == "fs-label":
                print("short_slice_target_mode=fs-label", flush=True)
        if args.length_buckets:
            train_unit_lengths = compute_unit_lengths(train.tokens)
            from collections import Counter as _C
            bucket_counts = _C(bucket_for(int(L)) for L in train_unit_lengths)
            print(f"length_buckets={dict(sorted(bucket_counts.items()))}", flush=True)
    hard_mask = None
    if args.hard_mask_mmap is not None and args.hard_oversample_rate > 1.0:
        hard_mask = np.array(np.memmap(args.hard_mask_mmap, dtype=np.uint8, mode="r", shape=(train.count,)))
        print(
            f"hard_oversample: rate={args.hard_oversample_rate:.2f} "
            f"hard={int(hard_mask.sum())}/{train.count} ({hard_mask.mean():.4f})",
            flush=True,
        )
    cache_meta = json.loads((args.cache_dir / "train.json").read_text())
    cache_hidden_dim = int(cache_meta.get("hidden_dim") or 512)
    model = build_model(classes, args.weight_bits, args.architecture, hidden_dim=cache_hidden_dim)
    print(f"hidden_dim={cache_hidden_dim}", flush=True)
    group_labels, label_to_group = group_names(teacher.selected_labels)
    label_to_group_tensor = None
    if args.group_loss_weight > 0.0:
        model = add_group_head(model, args.weight_bits, len(group_labels))
        label_to_group_tensor = tf.constant(label_to_group, dtype=tf.int64)
    if args.hidden_loss_weight > 0.0 and args.hidden_loss_target == "trunk":
        model = add_trunk_hidden_output(model, hidden_dim=cache_hidden_dim)
        print(f"hidden_loss_target=trunk hidden_dim={cache_hidden_dim}", flush=True)
    if args.init_from_checkpoint is not None:
        initialized_layers, adapted_layers = initialize_model_from_checkpoint(
            model,
            args.init_from_checkpoint,
            channel_selection=args.channel_selection,
        )
        apply_freeze_policy(
            model,
            initialized_layers,
            adapted_layers,
            args.freeze_init_policy,
            args.trainable_layers,
        )
    parent_feature_model = None
    student_feature_model = None
    if args.parent_feature_loss_weight > 0.0:
        assert args.parent_feature_checkpoint is not None
        assert args.parent_feature_layer is not None
        parent_feature_model, _ = build_parent_feature_model(
            args.parent_feature_checkpoint,
            classes,
            args.weight_bits,
            cache_hidden_dim,
            args.parent_feature_layer,
            channel_selection="first",
        )
        student_feature_model = layer_feature_model(model, args.parent_feature_layer)
        print(
            f"parent_feature_loss_weight={args.parent_feature_loss_weight} "
            f"parent_feature_layer={args.parent_feature_layer}",
            flush=True,
        )
    learning_rate = args.learning_rate
    if args.cosine_decay:
        if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
            raise ValueError("--min-learning-rate-ratio must be in [0, 1]")
        if args.balance_power > 0.0 or hard_mask is not None:
            steps_per_epoch = max(1, train.count // args.batch_size)
        else:
            steps_per_epoch = max(1, math.ceil(train.count / args.batch_size))
        if args.max_train_batches is not None:
            steps_per_epoch = min(steps_per_epoch, max(1, args.max_train_batches))
        decay_steps = max(1, args.epochs * steps_per_epoch)
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps,
            alpha=args.min_learning_rate_ratio,
        )
        print(
            f"learning_rate_schedule=cosine initial={args.learning_rate:.8g} "
            f"min_ratio={args.min_learning_rate_ratio:.6f} decay_steps={decay_steps}",
            flush=True,
        )
    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
    }
    if args.global_clipnorm > 0.0:
        optimizer_kwargs["global_clipnorm"] = args.global_clipnorm
    optimizer = tf.keras.optimizers.AdamW(**optimizer_kwargs)
    train_step = make_train_step(
        model,
        optimizer,
        label_to_group_tensor,
        args.distill_temperature,
        args.hidden_loss_weight,
        args.hard_loss_weight,
        args.group_loss_weight,
        args.confidence_power,
        args.xla,
        self_loss_weight=args.self_loss_weight,
        label_smoothing=args.label_smoothing,
        parent_feature_model=parent_feature_model,
        student_feature_model=student_feature_model,
        parent_feature_loss_weight=args.parent_feature_loss_weight,
    )
    eval_step = make_eval_step(
        model,
        label_to_group_tensor,
        args.distill_temperature,
        args.hidden_loss_weight,
        args.hard_loss_weight,
        args.group_loss_weight,
        args.confidence_power,
        args.xla,
        self_loss_weight=args.self_loss_weight,
        parent_feature_model=parent_feature_model,
        student_feature_model=student_feature_model,
        parent_feature_loss_weight=args.parent_feature_loss_weight,
    )
    best_valid = -1.0
    best_weights = None
    checks_without_improvement = 0
    if args.qat_start_epoch > 0:
        QAT_ACTIVE.assign(False)
        print(f"qat_active=False (will enable at epoch {args.qat_start_epoch})", flush=True)
    else:
        QAT_ACTIVE.assign(True)
    if args.eval_initial:
        initial_export_phase = args.qat_start_epoch == 0 or args.qat_start_epoch >= args.epochs
        valid_accuracy, valid_loss, valid_examples_per_sec, _ = evaluate_dataset(
            model,
            eval_step,
            tensor_batches(valid, args.batch_size, shuffle=False, seed=0, prefetch=args.prefetch),
            valid.count,
            classes,
        )
        print(
            f"epoch=initial valid_loss={valid_loss:.6f} "
            f"valid_teacher_parity={valid_accuracy:.6f} "
            f"valid_examples_per_sec={valid_examples_per_sec:.1f}",
            flush=True,
        )
        if initial_export_phase:
            best_valid = valid_accuracy
            best_weights = model.get_weights()
            size = export_model(
                args.output,
                model,
                teacher.selected_labels,
                teacher.selected_slugs,
                args.weight_bits,
                args.architecture,
                tokenizer_version=args.unit_tokenizer if architecture_uses_word_units(args.architecture) else None,
            )
            print(f"best_checkpoint_model_size_bytes={size}", flush=True)

    for epoch in range(args.epochs):
        if args.qat_start_epoch > 0 and epoch == args.qat_start_epoch:
            QAT_ACTIVE.assign(True)
            print(f"qat_active=True (enabled at epoch {epoch})", flush=True)
        loss_sum = 0.0
        seen = 0
        started = time.perf_counter()
        if not args.tf_data_batches:
            batch_iterable = batches(
                train,
                args.batch_size,
                shuffle=True,
                seed=args.seed + epoch,
                balance_power=args.balance_power,
                hard_mask=hard_mask,
                hard_oversample_rate=args.hard_oversample_rate,
                cutmix_prob=args.cutmix_prob,
                unit_lengths=train_unit_lengths,
                short_slice_prob=args.short_slice_prob,
                short_slice_lengths=short_slice_lengths,
                short_slice_target_min_confidence=args.short_slice_target_min_confidence,
                short_slice_target_mode=args.short_slice_target_mode,
            )
        else:
            batch_iterable = tensor_batches(
                train,
                args.batch_size,
                shuffle=True,
                seed=args.seed + epoch,
                balance_power=args.balance_power,
                prefetch=args.prefetch,
                hard_mask=hard_mask,
                hard_oversample_rate=args.hard_oversample_rate,
                cutmix_prob=args.cutmix_prob,
                short_slice_prob=args.short_slice_prob,
                short_slice_lengths=short_slice_lengths,
                short_slice_target_mode=args.short_slice_target_mode,
            )
        for (
            token_batch,
            probability_batch,
            label_batch,
            hidden_batch,
            self_probs_batch,
        ) in batch_iterable:
            loss, batch_size = train_step(
                token_batch,
                probability_batch,
                label_batch,
                hidden_batch,
                self_probs_batch,
            )
            batch_n = int(batch_size.numpy())
            loss_value = float(loss.numpy())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at epoch={epoch} "
                    f"examples_seen={seen + batch_n}; aborting"
                )
            seen += batch_n
            loss_sum += loss_value * batch_n
            if args.max_train_batches is not None and seen >= args.max_train_batches * args.batch_size:
                break
        train_seconds = max(time.perf_counter() - started, 1e-9)
        train_loss = loss_sum / max(seen, 1)
        train_examples_per_sec = seen / train_seconds

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            valid_accuracy, valid_loss, valid_examples_per_sec, _ = evaluate_dataset(
                model,
                eval_step,
                tensor_batches(valid, args.batch_size, shuffle=False, seed=0, prefetch=args.prefetch),
                valid.count,
                classes,
            )
            print(
                f"epoch={epoch} loss={train_loss:.6f} train_examples_per_sec={train_examples_per_sec:.1f} "
                f"valid_loss={valid_loss:.6f} valid_teacher_parity={valid_accuracy:.6f} "
                f"valid_examples_per_sec={valid_examples_per_sec:.1f}",
                flush=True,
            )
            qat_phase = args.qat_start_epoch == 0 or epoch >= args.qat_start_epoch
            export_phase = qat_phase or args.qat_start_epoch >= args.epochs
            if export_phase and valid_accuracy > best_valid:
                best_valid = valid_accuracy
                best_weights = model.get_weights()
                checks_without_improvement = 0
                size = export_model(
                    args.output,
                    model,
                    teacher.selected_labels,
                    teacher.selected_slugs,
                    args.weight_bits,
                    args.architecture,
                    tokenizer_version=args.unit_tokenizer if architecture_uses_word_units(args.architecture) else None,
                )
                print(f"best_checkpoint_model_size_bytes={size}", flush=True)
            elif not qat_phase:
                pass
            else:
                checks_without_improvement += 1
                if args.early_stop_patience > 0 and checks_without_improvement >= args.early_stop_patience:
                    print(
                        f"early_stopping=true epoch={epoch} checks_without_improvement={checks_without_improvement}",
                        flush=True,
                    )
                    break

    if best_weights is not None:
        model.set_weights(best_weights)
    valid_accuracy, valid_loss, valid_examples_per_sec, _ = evaluate_dataset(
        model,
        eval_step,
        tensor_batches(valid, args.batch_size, shuffle=False, seed=0, prefetch=args.prefetch),
        valid.count,
        classes,
    )
    test_accuracy, test_loss, test_examples_per_sec, test_confusion = evaluate_dataset(
        model,
        eval_step,
        tensor_batches(test, args.batch_size, shuffle=False, seed=0, prefetch=args.prefetch),
        test.count,
        classes,
        collect_confusion=args.confusion_matrix_output is not None or args.confusion_matrix_top > 0,
    )
    size = export_model(
        args.output,
        model,
        teacher.selected_labels,
        teacher.selected_slugs,
        args.weight_bits,
        args.architecture,
        tokenizer_version=args.unit_tokenizer if architecture_uses_word_units(args.architecture) else None,
    )
    print(f"train: {counts['train']} examples")
    print(f"valid: {counts['valid']} examples")
    print(f"test: {counts['test']} examples")
    print(f"valid_teacher_parity={best_valid:.6f}")
    print(f"valid_loss={valid_loss:.6f}")
    print(f"valid_examples_per_sec={valid_examples_per_sec:.1f}")
    print(f"test_teacher_parity={test_accuracy:.6f}")
    print(f"test_loss={test_loss:.6f}")
    print(f"test_examples_per_sec={test_examples_per_sec:.1f}")
    if test_confusion is not None:
        top_confusions = confusion_summary(test_confusion, teacher.selected_labels, args.confusion_matrix_top)
        for row in top_confusions:
            print(
                "confusion "
                f"actual={row['actual']} predicted={row['predicted']} count={row['count']} "
                f"actual_total={row['actual_total']} actual_recall={row['actual_recall']:.6f} "
                f"share_of_actual={row['share_of_actual']:.6f}"
            )
        if args.confusion_matrix_output:
            write_confusion_matrix(args.confusion_matrix_output, test_confusion, teacher.selected_labels, args.confusion_matrix_top)
            print(f"confusion_matrix_output={args.confusion_matrix_output}")
    print(f"model_size_bytes={size}")


if __name__ == "__main__":
    main()
