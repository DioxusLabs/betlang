# Betlang

CPU source-language detection for code, backed by the compact Magika student
model.

```rust
let language = betlang::detect("fn main() { println!(\"hi\"); }");
assert_eq!(language, Some(betlang::Language::Rust));
```

The embedded production model is a 49,847-byte quantized wordseq student using
the v3 tokenizer. It predicts the 48 filesystem-backed source labels present in
the training corpus; unsupported empty labels such as `jsonl`, `matlab`, and
`prolog` are not part of the model head or runtime class mapping. On the
held-out filesystem-label test split it reaches `test_fs_accuracy=0.965473`
with `macro_recall=0.965813`.

## Confusion Matrix

The shipped wordseq model is evaluated below on the held-out filesystem-label
test split. The matrix is row-normalized: actual labels are rows, predicted
labels are columns, and the diagonal is correct classification. The full count
matrix is in `actual_dataset_confusion_by_size.csv`, and the top-confusion
summary is in `actual_dataset_confusion_by_size.md`.

![Betlang wordseq confusion matrix](assets/confusion-by-size.png)
