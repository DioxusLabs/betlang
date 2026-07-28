# Betlang Model Card

## Artifact

- File: `assets/magika/source-student-q4.bin`
- Format: weights-only MSQ1 quantized tensor payload
- Size: 45,448 bytes
- SHA-256: `c1c42a31761c0f38f3e67c7a2b319bd852830fcd3c8d8299261f86376ad9f83f`
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
  turns), ShareGPT first human turns (`RyokoAI/ShareGPT52K`),
  `HuggingFaceH4/no_robots`, and developer questions from Stack Overflow
  titles (`pacovaldez/stackoverflow-questions`), plus instruction-style
  prompts from `tatsu-lab/alpaca`, `databricks/databricks-dolly-15k`, and
  the `fka/awesome-chatgpt-prompts` personas, and imperative sysadmin
  English from NL2SH-ALFA training instructions (hard positives: terse
  verb-first requests full of shell vocabulary).
- `shell_command`: real bash one-liners from the NL2Bash corpus
  (TellinaTool/nl2bash), example commands from English tldr-pages with
  `{{placeholder}}` markers flattened, bash tool calls mined from agent
  RL/SFT trajectories (`SWE-bench/SWE-smith-trajectories`), real user shell
  history (`spignelon/bash_history`), NL2SH-ALFA training commands, and
  synthetic hard negatives that
  embed English phrases inside quoted command arguments
  (`git commit -m "..."`, `echo "..."`, `grep -r "..."`).

Split assignment is leakage-aware: samples are assigned to train/valid/test
by hashing a group key (normalized prompt text; shell command template with
quoted strings, numbers, and paths collapsed), so near-duplicate variants
never straddle splits, and each synthetic hard negative follows the split of
the prompt its phrase came from.

Inputs are canonicalized identically at training and inference time
(`src/model/normalize.rs` mirrors the corpus builder): newlines normalized,
tabs/non-breaking spaces mapped to spaces, BOM/zero-width and control
characters removed, space runs collapsed, and edges trimmed.

There is no teacher model: unlike the earlier source-language student, the
2-class head trains directly on corpus labels with label smoothing.

## Evaluation

Held-out test split of the training corpus (5% of samples, disjoint from
train/valid; ~288k samples total). Metrics are printed by
`scripts/train_prompt_student.py` at export time; for the shipped artifact:

- `test_accuracy=0.990885`
- `prompt` recall `0.993067`
- `shell_command` recall `0.988423`

`scripts/ood_benchmark.py` builds a ~4.2k-sample out-of-distribution
benchmark from held-out sources (HelpSteer2 and hh-rlhf prompts, NL2SH-ALFA
test pairs, InterCode-Corrections commands, unseen bash history), excluding
exact matches and train-template overlaps. On it the shipped artifact
scores 99.3% (prompt recall 0.993, shell recall 1.000) vs 99.1% for Warp's
public `bert_tiny_v3.onnx` classifier (17.6 MB, from `warpdotdev/Warp`
`crates/input_classifier`), at ~390x smaller size. Warp additionally runs
heuristics before its model in production.

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
