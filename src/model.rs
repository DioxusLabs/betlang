//! Inference for the wordseq student.
//!
//! Loads `assets/magika/source-student-q4.bin` (weights-only MSQ1 export) and
//! runs a forward pass: byte-window tokenization -> word-unit tokenization
//! -> HashEmbedding lookup (K=3) -> 3 conv stages with max-pool -> global
//! max+avg pool -> 2 dense layers -> 48-class softmax logits.
//!
//! Model architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`
//! - 1024-bin x 24-dim shared HashEmbedding table (4-bit, ~12 KB)
//! - QConv1D k=7 24->64ch (2-bit ternary)
//! - MaxPool(4)
//! - QConv1D k=5 64->128ch (2-bit)
//! - MaxPool(2)
//! - QConv1D k=3 128->128ch (2-bit)
//! - GlobalMax + GlobalAvg -> 256-dim
//! - QDense 256->96 (2-bit) + GELU
//! - QDense 96->48 (4-bit)

mod activation;
mod constants;
mod embedded;
mod forward;
mod layers;
mod reader;
mod runtime;
#[cfg(test)]
mod tests;
mod tokenizer;
mod window;

use self::{constants::CLASSES, runtime::Model, window::build_window};
use crate::{Detection, Language};

pub(crate) fn detect(source: &[u8]) -> Detection {
    let Some(window) = build_window(source) else {
        return Detection::from_predictions(Vec::new());
    };
    let model = Model::get();
    let units = model.tokenize_units(&window);
    let logits = model.logits(&units);
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

    debug_assert_eq!(CLASSES, Language::MODEL_LABEL_COUNT);
    let mut predictions = Vec::with_capacity(logits.len());
    for (index, &logit) in logits.iter().enumerate() {
        let language = Language::from_model_index(index).expect("model label index");
        let probability = (logit - max).exp() / denominator;
        predictions.push((probability, language));
    }

    predictions.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.slug().cmp(b.1.slug())));
    Detection::from_predictions(predictions)
}
