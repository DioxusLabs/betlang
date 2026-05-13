#!/usr/bin/env bash
set -euo pipefail

wasm_rustflags="${BETLANG_WASM_RUSTFLAGS:--C target-feature=+simd128,+relaxed-simd}"
RUSTFLAGS="${RUSTFLAGS:-${wasm_rustflags}}" cargo build --target wasm32-unknown-unknown --release --example wasm_smoke
wasm_module="${BETLANG_WASM_MODULE:-target/wasm32-unknown-unknown/release/examples/wasm_smoke.wasm}"
cargo run --release --features wasm-bench --example wasm_bench -- "${wasm_module}" "$@"
