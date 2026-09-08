# Benchmark Baselines

Baselines are informational. They are useful for spotting large regressions, but
not part of Betlang's semver contract.

## 2026-07-23

Environment:

- Host: `aarch64-apple-darwin`, `arm64` (M2 Max)
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`

Native command:

```bash
cargo bench --bench detect
```

| Case | Bytes | Median time | Throughput |
|---|---:|---:|---:|
| short | 68 | 23.841 µs/inference | 2.8522 MB/s |
| full window | 4623 | 386.32 µs/inference | 11.967 MB/s |

`detect` reads a fixed window (at most the first and last 4096 bytes), so
per-file cost is constant in file size; the throughput above is the
worst case by construction. A 1 MB input classifies at multiple GB/s.

## 2026-05-15

Environment:

- Host: `aarch64-apple-darwin`, `arm64`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`

Native command:

```bash
cargo bench --bench detect
```

| Case | Bytes | Median time | Throughput |
|---|---:|---:|---:|
| short | 68 | 4.5357 ms/inference | 14.992 KB/s |
| full window | 4623 | 4.5321 ms/inference | 1.0200 MB/s |
