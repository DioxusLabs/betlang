# Betlang Model Card

## Artifact

- File: `assets/magika/source-student-q4.bin`
- Format: weights-only MSQ1 quantized tensor payload
- Size: 45,448 bytes
- SHA-256: `5c7a689935398887f7e76b23de18e9c84128e6c4d600136429625bf893c1d373`
- Architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`
- Tokenizer: word-unit tokenizer version 3
- Output head: 2 model labels exposed one-to-one as public `Kind` variants
  (`prompt`, `shell_command`)

## Intended Use

Betlang is intended for fast routing of terminal-style input: deciding
whether input is a natural-language prompt for a language model or a shell
command, in the style of Warp's Agent Mode auto-detection. It is suitable as
a best-effort classifier for routing input to an AI assistant vs a shell. It
is not intended for security decisions, command validation, or any use where
a misclassification is costly.

## Training Source

The model is trained from scratch with hard labels on a corpus assembled by
`scripts/build_prompt_corpus.py` from public Hugging Face datasets:

- `prompt`: real user prompts from `OpenAssistant/oasst1` (English first
  turns), ShareGPT first human turns (`RyokoAI/ShareGPT52K`), and
  `HuggingFaceH4/no_robots`, plus instruction-style prompts from
  `tatsu-lab/alpaca`, `databricks/databricks-dolly-15k`, and the
  `fka/awesome-chatgpt-prompts` personas.
- `shell_command`: real bash one-liners from the NL2Bash corpus
  (TellinaTool/nl2bash) and example commands from tldr-pages with
  `{{placeholder}}` markers flattened.

There is no teacher model: unlike the earlier source-language student, the
2-class head trains directly on corpus labels with label smoothing.

## Evaluation

Held-out test split of the training corpus (5% of samples, disjoint from
train/valid; ~148k samples total). Metrics are printed by
`scripts/train_prompt_student.py` at export time; for the shipped artifact:

- `test_accuracy=0.992114`
- `prompt` recall `0.988925`
- `shell_command` recall `0.994816`

## Limitations

- The boundary is fuzzy for one-word inputs and command names used in prose;
  Warp pairs a similar classifier with keyword allowlists, shell-history
  matching, and completion-engine heuristics for this reason.
- The corpus is English- and bash-heavy; other languages and exotic shells
  are underrepresented.
- Inputs shorter than 8 non-whitespace bytes return no prediction.

## Quantization

Same scheme as the original betlang student: 4-bit int embedding/output
layers, 2-bit ternary conv/dense layers, quantization-aware training active
for the second half of the schedule, exported through the metadata-free MSQ1
serializer.
