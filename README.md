# Betlang

CPU source-language detection for code, backed by the compact Magika student
model.

```rust
let language = betlang::detect("fn main() { println!(\"hi\"); }");
assert_eq!(language, Some(betlang::Language::Rust));
```

## Confusion by file size

The shipped wordseq model is evaluated below on the held-out `bigorig` test
split. Each panel is a row-normalized confusion matrix for one file-size
bucket: actual labels are rows, predicted labels are columns, and the diagonal
is correct classification.

![Betlang wordseq confusion by file size](assets/confusion-by-size.png)
