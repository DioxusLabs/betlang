//! Run the experimental BTQ1 model without changing `betlang::detect`.
//!
//! cargo run --release --example tiny -- MODEL.bin SOURCE
//! cargo run --release --example tiny -- MODEL.bin --compare CACHE/test.paths.txt

use std::{env, error::Error, fs, io, path::Path, time::Instant};

use rayon::prelude::*;

const BINS: usize = 512;
const HIDDEN: usize = 16;
const CLASSES: usize = 48;
const LABELS: [&str; CLASSES] = [
    "asm",
    "batch",
    "c",
    "clojure",
    "cmake",
    "cobol",
    "cpp",
    "cs",
    "css",
    "dart",
    "dockerfile",
    "elixir",
    "erlang",
    "gemfile",
    "gemspec",
    "go",
    "gradle",
    "groovy",
    "haskell",
    "html",
    "ini",
    "java",
    "javascript",
    "json",
    "julia",
    "kotlin",
    "lisp",
    "lua",
    "markdown",
    "objectivec",
    "ocaml",
    "perl",
    "php",
    "powershell",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "shell",
    "sql",
    "swift",
    "toml",
    "typescript",
    "vba",
    "verilog",
    "xml",
    "yaml",
];

struct Tiny {
    kernel: Vec<f32>,
    bias: Vec<f32>,
    output: Vec<f32>,
    output_bias: Vec<f32>,
}

impl Tiny {
    fn load(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 4752 || bytes[..8] != *b"BTQ1\x01\0\0\0" {
            return Err("expected a 4,752-byte BTQ1 model");
        }
        let floats = |data: &[u8]| {
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let scales = floats(&bytes[8..16]);
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err("invalid weight scales");
        }
        let unpack = |data: &[u8], scale: f32| {
            data.iter()
                .flat_map(|b| [(b & 15) as i8 - 8, (b >> 4) as i8 - 8])
                .map(|v| v as f32 * scale)
                .collect()
        };
        let model = Self {
            kernel: unpack(&bytes[16..4112], scales[0]),
            bias: floats(&bytes[4112..4176]),
            output: unpack(&bytes[4176..4560], scales[1]),
            output_bias: floats(&bytes[4560..4752]),
        };
        if model
            .bias
            .iter()
            .chain(&model.output_bias)
            .any(|v| !v.is_finite())
        {
            return Err("non-finite bias");
        }
        Ok(model)
    }

    fn logits(&self, source: &[u8]) -> Option<[f32; CLASSES]> {
        let inputs = features(source)?;
        let mut hidden = [0.0; HIDDEN];
        hidden.copy_from_slice(&self.bias);
        for (&input, row) in inputs.iter().zip(self.kernel.chunks_exact(HIDDEN)) {
            if input != 0.0 {
                for (value, weight) in hidden.iter_mut().zip(row) {
                    *value += input * weight;
                }
            }
        }
        let mut logits = [0.0; CLASSES];
        logits.copy_from_slice(&self.output_bias);
        for (input, row) in hidden.iter().zip(self.output.chunks_exact(CLASSES)) {
            for (value, weight) in logits.iter_mut().zip(row) {
                *value += input.max(0.0) * weight;
            }
        }
        Some(logits)
    }

    fn detect(&self, source: &[u8]) -> Option<&'static str> {
        let logits = self.logits(source)?;
        let index = (0..CLASSES)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))?;
        Some(LABELS[index])
    }
}

fn hash(bytes: &[u8]) -> u32 {
    bytes.iter().fold(2166136261, |h, b| {
        (h ^ u32::from(b.to_ascii_lowercase())).wrapping_mul(16777619)
    })
}

fn word_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_' || byte >= 128
}

fn whitespace(byte: &u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 11 | 12)
}

fn window(source: &[u8]) -> Option<Vec<u8>> {
    let begin = source[..source.len().min(4096)].trim_ascii_start();
    if begin.len() < 8 {
        return None;
    }
    if begin.len() < 1024 {
        return Some(begin.to_vec());
    }
    let end = source[source.len().saturating_sub(4096)..].trim_ascii_end();
    let mut window = begin[..1024].to_vec();
    if end.len() >= 1024 {
        window.extend_from_slice(&end[end.len() - 1024..]);
    }
    Some(window)
}

fn features(source: &[u8]) -> Option<[f32; BINS]> {
    let bytes = window(source)?;
    let mut counts = [0.0f32; BINS];
    let mut pos = 0;
    let mut previous = 0u32;
    while pos < bytes.len() {
        let start = pos;
        let byte = bytes[pos];
        pos += 1;
        let value = if word_start(byte) {
            while pos < bytes.len() && (word_start(bytes[pos]) || bytes[pos].is_ascii_digit()) {
                pos += 1;
            }
            hash(&bytes[start..pos])
        } else if byte.is_ascii_digit() {
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            hash(b"0")
        } else if byte == b'\n' || !whitespace(&byte) {
            hash(&[byte])
        } else {
            continue;
        };
        counts[value as usize % BINS] += 1.0;
        if previous != 0 {
            let pair = previous.wrapping_mul(16777619) ^ value;
            counts[pair as usize % BINS] += 1.0;
        }
        previous = value;
    }
    for count in &mut counts {
        *count = count.ln_1p();
    }
    let norm = counts.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut counts {
            *value /= norm;
        }
    }
    Some(counts)
}

fn compare(model: &Tiny, root: &Path) -> Result<(), Box<dyn Error>> {
    let mut paths = Vec::new();
    for line in fs::read_to_string(root)?.lines() {
        let path = Path::new(line).to_path_buf();
        let slug = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str());
        let label = LABELS
            .iter()
            .find(|&&label| Some(label) == slug)
            .ok_or("unknown label")?;
        paths.push((*label, path));
    }
    paths.sort();
    if paths.is_empty() {
        return Err("no labeled files found".into());
    }
    let results: Vec<_> = paths
        .par_iter()
        .map(|(truth, path)| {
            let source = fs::read(path)?;
            let tiny = model.detect(&source).unwrap_or("unknown");
            let base = betlang::detect(&source)
                .language()
                .map(|l| l.slug())
                .unwrap_or("unknown");
            Ok((*truth, tiny, base, path))
        })
        .collect::<Result<_, io::Error>>()?;
    println!("truth\ttiny\tbaseline\tpath");
    for (truth, tiny, base, path) in results {
        println!("{truth}\t{tiny}\t{base}\t{}", path.display());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: tiny MODEL.bin SOURCE | MODEL.bin --compare CACHE/test.paths.txt".into(),
        );
    }
    let model = Tiny::load(&fs::read(&args[1])?)?;
    if args[2] == "--compare" {
        let root = args.get(3).ok_or("--compare requires a paths manifest")?;
        return compare(&model, Path::new(root));
    }
    let source = fs::read(&args[2])?;
    if args.get(3).is_some_and(|arg| arg == "--features") {
        println!("{:?}", features(&source));
        return Ok(());
    }
    let start = Instant::now();
    let logits = model.logits(&source);
    let elapsed = start.elapsed();
    println!("{}", model.detect(&source).unwrap_or("unknown"));
    println!("{logits:?}");
    eprintln!("inference_us={:.2}", elapsed.as_secs_f64() * 1e6);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_match_production_order() {
        for (index, label) in LABELS.iter().enumerate() {
            assert_eq!(label.parse::<betlang::Language>().unwrap() as usize, index);
        }
    }

    #[test]
    fn rejects_invalid_models() {
        for size in [0, 8, 4751, 4752, 4753] {
            assert!(Tiny::load(&vec![0; size]).is_err());
        }
    }

    #[test]
    fn rejects_short_inputs_and_normalizes_features() {
        assert!(features(b"   \n\t").is_none());
        assert!(features(b"short").is_none());
        for source in [b"fn main() { 123; }\n".as_slice(), &[255u8; 2048]] {
            let x = features(source).unwrap();
            assert!((x.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }
}
