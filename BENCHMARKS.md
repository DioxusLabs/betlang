# Benchmark Baselines

Baselines are informational. They are useful for spotting large regressions, but
not part of Betlang's semver contract.

## 2026-08-23 (compact Bloom binary model, 26,808-byte MBL4)

Environment:

- Host: `x86_64-unknown-linux-gnu`, `INTEL(R) XEON(R) PLATINUM 8559C`
- Rust: `rustc 1.94.0 (4a4ef493e 2026-03-02)`

Native command:

```bash
cargo bench --bench detect
```

| Case | Bytes | Median time | Throughput |
|---|---:|---:|---:|
| short | 496 | 45.674 µs/inference | 10.860 MB/s |
| full window | 4970 | 73.806 µs/inference | 67.338 MB/s |

The previous convolutional model measured on the same host, snippet, and
command: short 14.201 ms/inference, full window 54.904 ms/inference — the
compact binary XOR/popcount model is ~310x faster on short inputs and ~740x
faster on a full 4 KiB window (and ~8-13x faster than the retired 319,488-bit
large Bloom variant, which measured 595.51 µs short / 610.22 µs full window).

## 2026-05-15 (previous convolutional model)

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
