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
manifest-aligned held-out filesystem-label test split, the exported model
reaches `test_fs_accuracy=0.965238` with `macro_recall=0.965411`.

## Confusion By File Size

The shipped wordseq model is evaluated below on the held-out filesystem-label
test split, bucketed by file size. Each panel is row-normalized: actual labels
are rows, predicted labels are columns, and the diagonal is correct
classification. The final panel is the overall matrix. The full count and byte
totals are in `actual_dataset_confusion_by_size.csv`, and the per-bucket
summary is in `actual_dataset_confusion_by_size.md`.

![Betlang wordseq confusion matrices by file size](assets/confusion-by-size.png)
