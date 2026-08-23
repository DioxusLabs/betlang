# Benchmark Baselines

Baselines are informational. They are useful for spotting large regressions, but
not part of Betlang's semver contract.

## 2026-08-22 (Shannon-bit binary CNN ensemble)

Environment:

- Host: `aarch64-apple-darwin`, `arm64` (Apple M4 virtual CPU)
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`

Native command:

```bash
cargo bench --bench detect
```

| Case | Bytes | Median time | Throughput |
|---|---:|---:|---:|
| short | 68 | 41.068 ms/inference | 1.6558 KB/s |
| full window | 4623 | 43.355 ms/inference | 106.63 KB/s |

The five-model binary ensemble trades latency for accuracy against the
previous single wordseq model below.

## 2026-05-15 (previous wordseq model)

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
