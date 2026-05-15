//! Inference for the wordseq student.
//!
//! Loads `assets/magika/source-student-q4.bin` (~100 KB MSQ1 export) and
//! runs a forward pass: byte-window tokenization → word-unit tokenization
//! → HashEmbedding lookup (K=3) → 3 conv stages with max-pool → global
//! max+avg pool → 2 dense layers → 67-class logits.
//!
//! Model architecture: `wordseq-b1536-k3-m2048-med-3conv-hidden`
//! - 1536-bin × 28-dim shared HashEmbedding table (4-bit, ~21 KB)
//! - QConv1D k=7 28→96ch (2-bit ternary)
//! - MaxPool(4)
//! - QConv1D k=5 96→192ch (2-bit)
//! - MaxPool(2)
//! - QConv1D k=3 192→192ch (2-bit)
//! - GlobalMax ⊕ GlobalAvg → 384-dim
//! - QDense 384→160 (2-bit) + GELU
//! - QDense 160→67 (4-bit)

mod activation;
mod constants;
mod embedded;
mod layers;
mod metadata;
mod reader;
mod runtime;
#[cfg(test)]
mod tests;
mod tokenizer;
mod window;

use self::{constants::CLASSES, runtime::Model, window::build_window};
use crate::{Detection, Language, language::CLASS_LANGUAGES};

pub(crate) fn detect(source: &[u8]) -> Detection {
    let Some((bytes, pad)) = build_window(source) else {
        return Detection::from_predictions(Vec::new());
    };
    let model = Model::get();
    let units = model.tokenize_units(&bytes, &pad);
    let logits = model.logits_for_runtime_units(&units);
    detection_from_logits(&logits)
}

fn detection_from_logits(logits: &[f32; CLASSES]) -> Detection {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Detection::from_predictions(Vec::new());
    }

    for &logit in logits {
        debug_assert!(logit.is_finite());
    }

    let denominator: f32 = logits.iter().map(|logit| (logit - max).exp()).sum();
    if !denominator.is_finite() || denominator == 0.0 {
        return Detection::from_predictions(Vec::new());
    }

    let mut predictions: Vec<(f32, Language)> = Vec::new();
    for (&logit, &language) in logits.iter().zip(CLASS_LANGUAGES.iter()) {
        let probability = (logit - max).exp() / denominator;
        if let Some((existing, _)) = predictions
            .iter_mut()
            .find(|(_, existing_language)| *existing_language == language)
        {
            *existing += probability;
        } else {
            predictions.push((probability, language));
        }
    }

    predictions.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.slug().cmp(b.1.slug())));
    Detection::from_predictions(predictions)
}
