# Betlang

CPU source-language detection for code, backed by the compact Magika student
model.

```rust
let language = betlang::detect("fn main() { println!(\"hi\"); }");
assert_eq!(language, Some(betlang::Language::Rust));
```

The embedded production model is a ~100 KB quantized wordseq student using the
v3 tokenizer. On the held-out `bigorig` test split it reaches
`test_teacher_parity=0.932778` and `test_fs_accuracy=0.927934`. This export is
calibrated to recover underrepresented source labels, trading some full-corpus
accuracy for rare-label recall.

## Confusion by file size

The shipped wordseq model is evaluated below on the held-out `bigorig` test
split. Each panel is a row-normalized confusion matrix for one file-size
bucket: actual labels are rows, predicted labels are columns, and the diagonal
is correct classification.

![Betlang wordseq confusion by file size](assets/confusion-by-size.png)
