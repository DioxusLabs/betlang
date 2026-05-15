# Generated Reports

This directory holds generated analysis artifacts that are useful for reviewing
model behavior but are not part of the published crate package.

- `actual_dataset_confusion_by_size.csv`
- `actual_dataset_confusion_by_size.md`

The README uses the rendered image at `assets/confusion-by-size.png`; the CSV
and Markdown report can be regenerated with `scripts/confusion_by_size.py` using
the training cache described in `scripts/TRAINING.md`.
