# Betlang Production-Readiness TODO

This list is based on a local comparison with `../dioxus`, `../dioxus-code`,
and `../dioxus-icons`, plus the current `betlang` checks.

## P0 - Make Release Gates Real

- [x] Fix `cargo fmt --all -- --check`.
  - Fixed formatting in `src/model.rs` and `examples/detect.rs`.
- [x] Fix `cargo clippy --all-targets --all-features -- -D warnings`.
  - Fixed straightforward lints in `src/model.rs`.
  - Used targeted `#[allow(clippy::too_many_arguments)]` on SIMD kernels where
    splitting parameters would obscure the hot path.
- [x] Replace the hollow `handrolled-fuzz` CI job with real tests, or remove it.
  - Removed the zero-test CI job.
- [x] Expand CI to match the published sibling crates' baseline gates.
  - Run fmt, clippy with warnings denied, test with all targets/features,
    docs with `RUSTDOCFLAGS=-Dwarnings`, wasm smoke, and package verification.
  - Keep the wasm smoke matrix, but make native Rust quality gates first-class
    instead of relying on plain `cargo test`.
- [x] Add `cargo package --locked` or equivalent package verification to CI.
  - Package contents now exclude `.claude`, `.github`, training scripts, and
    generated analysis CSV/Markdown files.

## P0 - Validate The Model Contract

- [x] Add golden prediction tests over a representative fixture suite.
  - Remaining: broaden coverage across every public `Language` variant that is
    expected to classify reliably.
  - Remaining: add CLI-level binary/non-UTF-8 fixtures and ambiguous cases.
- [x] Add regression tests for `Language` mapping.
  - Verify the embedded model metadata labels and slugs match
    `CLASS_LANGUAGES`.
  - Catch duplicate/aliased model classes intentionally mapped to the same
    public language, such as `erb`, `gemfile`, `gemspec`, `jsonl`, `shell`,
    `csproj`, `vcxproj`, and `vba`.
- [x] Decide scalar/SIMD parity test policy.
  - Removed local scalar fallback forcing; runtime trusts `fearless_simd`.
  - Wasm smoke still covers scalar wasm, SIMD wasm, and relaxed SIMD wasm.
- [x] Add parser/asset integrity tests for `source-student-q4.bin`.
  - Assert magic, architecture, tokenizer version, tensor shapes, payload
    length, metadata length, and a committed SHA-256 hash.
- [x] Decide and document the detection result contract.
  - `detect` and `detect_bytes` return a `Detection` that stores sorted
    probability/language pairs internally.
  - `Detection::language()` returns the top language, or `None` for empty,
    whitespace-only, and too-short inputs.

## P1 - Public API Polish

- [x] Keep public detection API minimal.
  - Expose `detect`, `detect_bytes`, `Detection`, `Language::slug`, and
    `Detection::language`, plus `FromStr` for slug parsing.
  - Keep thresholds, model labels, and richer helper methods internal until
    there is stronger evidence that callers need them.
- [x] Add `FromStr` for `Language`.
  - Use standard slug parsing instead of a custom lookup method.
- [x] Decide whether the crate should classify bytes directly.
  - A `detect_bytes(&[u8])` API would avoid forcing callers to pre-validate
    UTF-8 when many source scanners naturally work on file bytes.
- [x] Stabilize error and threshold semantics before publishing a non-0.0.x
  release.
  - Because `Language` is `#[non_exhaustive]`, document how new languages and
    model upgrades affect semver.

## P1 - Cargo And Packaging Hygiene

- [x] Add `rust-version`.
  - The sibling crates publish an explicit MSRV; edition 2024 implies a modern
    compiler but does not communicate the tested minimum.
- [x] Add `resolver = "3"` under `[workspace]`.
- [x] Add `readme = "README.md"` and `documentation = "https://docs.rs/betlang"`
  to `[package]`.
- [x] Add `[package.metadata.docs.rs]`.
  - Build docs with the intended feature set and `--cfg docsrs` if needed.
- [x] Define package `include` or `exclude`.
  - Exclude `.claude`, CI internals, bulky analysis artifacts, and anything
    not required by the published crate.
  - Deliberately decide whether training scripts belong in the crate package
    or only in the repository.
- [x] Audit license attribution for the embedded model.
  - The crate is MIT, but the model is trained from Magika teacher outputs and
    should have clear attribution/licensing notes if required.
- [x] Add badge/header polish to README.
  - Match the sibling crates' crates.io version, downloads, docs.rs, and CI
    badges once published.

## P1 - Documentation

- [x] Expand README into a real user guide.
  - Installation snippet.
  - Quick start.
  - API behavior and empty-detection behavior.
  - Supported language list.
  - Performance and memory notes.
  - Wasm support notes.
  - Example CLI usage.
  - License and model attribution.
- [x] Fix model documentation drift.
  - `README.md`, `scripts/TRAINING.md`, and `src/model.rs` currently disagree
    on model size, architecture, and metrics.
  - The embedded metadata says
    `wordseq-b1536-k3-m2048-med-3conv-hidden`, tokenizer version 3.
- [x] Add crate-level docs that can serve as the docs.rs landing page.
  - Include a tested doctest or mark examples as `no_run` only when necessary.
- [x] Add a model card.
  - Dataset source, teacher version, training recipe, evaluation split,
    per-size accuracy, known weaknesses, unsupported file types, and update
    policy.
- [x] Document reproducibility.
  - Exact training command, expected metrics, required external assets,
    artifact hash, and how to regenerate the confusion matrix.

## P1 - Examples And Developer Experience

- [x] Make `examples/detect.rs` production-grade or move it into a binary crate.
  - Add tests around CLI behavior, exit codes, tree scanning, UTF-8 failures,
    and GitHub-style breakdown output.
- [x] Add a minimal example matching the README quick start.
- [x] Add wasm benchmarking instructions that do not require reverse
  engineering the scripts.
- [x] Add benchmark baselines to docs.
  - Track short snippet and full-window throughput on at least one native CPU
    and one wasm runtime.

## P2 - Repository Maintenance

- [x] Expand `.gitignore`.
  - Add `.claude/scheduled_tasks.lock`, `.DS_Store`, coverage artifacts,
    Python caches, and generated benchmark/report outputs.
- [x] Add typo and link checks.
  - `../dioxus` uses typos and lychee; this repo should at least check README
    and docs before release.
- [x] Add issue/PR templates only if this repo will accept external users.
- [x] Add release process notes.
  - Version bump, package verification, tag, publish, docs check, and model
    artifact hash.
- [x] Decide whether `actual_dataset_confusion_by_size.*` are source artifacts
  or generated artifacts.
  - If generated, move them under a reproducible output path and exclude from
    the crate package unless the README needs them.

## Current Check Snapshot

- `cargo fmt --all -- --check`: passes.
- `cargo clippy --all-targets --all-features -- -D warnings`: passes.
- `cargo test --all-targets --all-features`: passes, including model-contract,
  CLI, and representative golden prediction tests.
- `env RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --all-features`: passes.
- `cargo package --locked --allow-dirty`: passes locally. CI runs
  `cargo package --locked` on a clean checkout.
- Wasm smoke passes for scalar, `+simd128`, and `+simd128,+relaxed-simd`.
- `cargo package --list --allow-dirty`: now excludes `.claude`, `.github`,
  training scripts, and generated analysis artifacts. `Cargo.toml.orig` remains
  because Cargo includes it as standard package metadata.
