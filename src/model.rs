//! Inference for the Shannon-bit binary CNN detector.
//!
//! Loads `assets/bnn/source-bnn2.bin` (packed "BBN2" export) and runs a
//! forward pass: raw Magika byte window -> canonical Shannon coding into two
//! bitplanes (code bits + codeword boundaries) -> binary conv stages
//! (XNOR + popcount, fixed-point combine, integer thresholds, OR-pooling)
//! -> segmented popcount head -> 48-class softmax logits, averaged over a
//! small ensemble of binary models.

mod bnn;
mod constants;
#[cfg(test)]
mod tests;
mod window;

use self::{bnn::Bnn, constants::CLASSES, window::build_window};
use crate::{Detection, Language};

pub(crate) fn detect(source: &[u8]) -> Detection {
    let Some(window) = build_window(source) else {
        return Detection::from_predictions(Vec::new());
    };
    let model = Bnn::get();
    let planes = model.encode_planes(window.bytes());
    let logits = model.logits(&planes);
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
