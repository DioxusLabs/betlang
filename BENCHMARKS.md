# Benchmark Baselines

Baselines are informational. They are useful for spotting large regressions, but
not part of Betlang's semver contract.

## 2026-05-15

Environment:

- Host: `aarch64-apple-darwin`, `arm64`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Wasm runtime: Wasmtime `44.0.1` through `examples/wasm_bench`
- Wasm build: `-C target-feature=+simd128,+relaxed-simd`

Native command:

```bash
cargo bench --bench detect
```

| Case | Bytes | Median time | Throughput |
|---|---:|---:|---:|
| short | 68 | 4.5357 ms/inference | 14.992 KB/s |
| full window | 4623 | 4.5321 ms/inference | 1.0200 MB/s |

Wasm command:

```bash
scripts/bench-wasm.sh --samples 5 --sample-ms 250 --warmup-ms 500
```

| Case | Bytes | Median time | Inferences/sec | Throughput |
|---|---:|---:|---:|---:|
| short | 68 | 8.282942 ms/inference | 120.73 | 0.008 MB/s |
| full window | 4623 | 8.268647 ms/inference | 120.94 | 0.559 MB/s |
