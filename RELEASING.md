# Releasing Betlang

1. Update the version in `Cargo.toml`.
2. Confirm the embedded model hash in `README.md` and `MODEL_CARD.md` if the
   model artifact changed.
3. Run the native release gates:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-targets --all-features
   env RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --all-features
   cargo package --locked
   ```

4. Inspect `cargo package --list --allow-dirty` before publishing. The package
   should contain runtime source, examples, docs, the confusion image, and the
   embedded model. Training scripts, generated CSV/Markdown analysis, CI files,
   and local assistant files should not be packaged.
5. Publish:

   ```bash
   cargo publish --locked
   ```

6. Tag the release after crates.io accepts the package:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

7. Check the docs.rs build and update README badges after the first published
   version is visible.
