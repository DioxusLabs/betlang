# Betlang

CPU source-language detection for code, backed by the compact Magika student
model.

```rust
let language = betlang::detect("fn main() { println!(\"hi\"); }");
assert_eq!(language, Some(betlang::Language::Rust));
```

The embedded production model is a 47,840-byte weights-only quantized wordseq
student using the v3 tokenizer. It predicts the 48 filesystem-backed source
labels present in the training corpus; unsupported empty labels such as `jsonl`,
`matlab`, and `prolog` are not part of the model head or runtime class mapping.
On the manifest-aligned held-out filesystem-label test split, the exported
model reaches `test_fs_accuracy=0.965238` with `macro_recall=0.965411`.

## Overall Confusion Matrix

The shipped wordseq model is evaluated below on the held-out filesystem-label
test split. The matrix is row-normalized: actual labels are rows, predicted
labels are columns, and the diagonal is correct classification.

![Betlang wordseq overall confusion matrix](assets/confusion-overall.png)

## Confusion By File Size

The same held-out filesystem-label test split is bucketed by file size below.
Each panel is row-normalized with the same axes as the overall matrix. The full
count and byte totals are in `actual_dataset_confusion_by_size.csv`, and the
per-bucket summary is in `actual_dataset_confusion_by_size.md`.

![Betlang wordseq confusion matrices by file size](assets/confusion-by-size.png)
