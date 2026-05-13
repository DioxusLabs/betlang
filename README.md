# Betlang

CPU source-language detection for code, backed by the compact Magika student
model.

```rust
let language = betlang::detect("fn main() { println!(\"hi\"); }");
assert_eq!(language, Some(betlang::Language::Rust));
```
