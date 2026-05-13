#![cfg_attr(target_arch = "wasm32", no_main)]

use betlang::{Language, detect};
use std::sync::OnceLock;

const RUST_SOURCE: &str = include_str!("../snippets/demo.rs");
const FULL_WINDOW_SIZE: usize = 4_608;

#[unsafe(no_mangle)]
pub extern "C" fn detect_empty_smoke() -> i32 {
    i32::from(detect("  \n\t  ").is_some())
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_rust_smoke() -> i32 {
    language_status(detect(RUST_SOURCE), Language::Rust)
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_full_window_smoke() -> i32 {
    language_status(detect(full_window_source()), Language::Rust)
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_short_len() -> u32 {
    RUST_SOURCE.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_full_window_len() -> u32 {
    full_window_source().len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_short_bench(iterations: u32) -> u32 {
    detect_bench(RUST_SOURCE, iterations)
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_full_window_bench(iterations: u32) -> u32 {
    detect_bench(full_window_source(), iterations)
}

fn detect_bench(source: &'static str, iterations: u32) -> u32 {
    let mut detected = 0;
    for _ in 0..iterations {
        if detect(std::hint::black_box(source)) == Some(Language::Rust) {
            detected += 1;
        }
    }
    std::hint::black_box(detected)
}

fn full_window_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE
        .get_or_init(|| {
            let mut source = String::with_capacity(FULL_WINDOW_SIZE);
            while source.len() < FULL_WINDOW_SIZE {
                source.push_str(RUST_SOURCE);
                source.push('\n');
            }
            source
        })
        .as_str()
}

fn language_status(result: Option<Language>, expected: Language) -> i32 {
    match result {
        Some(language) if language == expected && !language.slug().is_empty() => 0,
        Some(_) => 2,
        None => 1,
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn main() {
    assert_eq!(detect_empty_smoke(), 0);
    assert_eq!(detect_rust_smoke(), 0);
    assert_eq!(detect_full_window_smoke(), 0);
}
