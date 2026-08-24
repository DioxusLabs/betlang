//! Inference for the Shannon n-gram counting-Bloom binary model.
//!
//! Loads `assets/magika/source-bloom.bin` (weights-only MBL5 export) and runs
//! a forward pass: byte-window tokenization -> tokenizer-v3 word units ->
//! deterministic Zobrist/counting-Bloom buckets -> selected-column binary
//! signature -> binary {-1,+1} linear head evaluated with XOR + popcount,
//! plus a per-(class, plane) int8 scale and per-class bias -> 48-class
//! softmax logits.

mod bloom;
mod constants;
#[cfg(test)]
mod tests;
mod tokenizer;
mod window;

use self::{
    bloom::{BloomModel, build_token_window},
    constants::CLASSES,
    tokenizer::tokenize,
    window::build_window,
};
use crate::{Detection, Language};

pub(crate) fn detect(source: &[u8]) -> Detection {
    let Some(tokens) = build_token_window(source) else {
        return Detection::from_predictions(Vec::new());
    };
    let Some(window) = build_window(source) else {
        return Detection::from_predictions(Vec::new());
    };
    let units = tokenize(&window);
    let logits = BloomModel::get().logits(&tokens, &units);
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
