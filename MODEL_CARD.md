# Betlang Model Card

## Artifact

- File: `assets/magika/source-student-q4.bin`
- Format: weights-only MSQ1 quantized tensor payload
- Size: 45,448 bytes
- SHA-256: `acb7d673d5717698eae4a385ef8d8e6ba2cd735d9edfe0ca99be6731ebaade17`
- Architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`
- Tokenizer: word-unit tokenizer version 3
- Output head: 2 model labels exposed one-to-one as public `Kind` variants
  (`natural_language`, `prompt`)

## Intended Use

Betlang is intended for fast routing of text input: deciding whether a piece
of text is ordinary prose or a prompt written for a language model, in the
style of Warp's natural-language input detection. It is suitable as a
best-effort content classifier for routing input to an AI assistant. It is not
intended for security decisions, content moderation, or any use where a
misclassification is costly.

## Training Source

The model is trained from scratch with hard labels on a corpus assembled by
`scripts/build_prompt_corpus.py` from public Hugging Face datasets:

- `prompt`: instructions from `tatsu-lab/alpaca` (with and without their task
  input blocks), `databricks/databricks-dolly-15k` (instruction-only plus a
  subset with pasted context), and the `fka/awesome-chatgpt-prompts` persona
  prompts.
- `natural_language`: paragraphs from WikiText-103, AG News headlines and
  summaries, and IMDB and Yelp reviews. Long prose samples also contribute a
  random 1-2 sentence slice so the class covers short texts and the model
  cannot use length as a proxy for the label.

There is no teacher model: unlike the earlier source-language student, the
2-class head trains directly on corpus labels with label smoothing.

## Evaluation

Held-out test split of the training corpus (5% of samples, disjoint from
train/valid; 105k samples total). Metrics are printed by
`scripts/train_prompt_student.py` at export time; for the shipped artifact:

- `test_accuracy=0.983271`
- `natural_language` recall `0.976090`
- `prompt` recall `0.995040`

## Limitations

- Prompts are themselves natural language; the boundary is stylistic. Polite
  imperative prose (recipes, how-to steps) can read as a prompt, and prompts
  phrased as plain narrative can read as prose.
- The corpus is English-heavy; other languages are underrepresented.
- Inputs shorter than 8 non-whitespace bytes return no prediction.

## Quantization

Same scheme as the original betlang student: 4-bit int embedding/output
layers, 2-bit ternary conv/dense layers, quantization-aware training active
for the second half of the schedule, exported through the metadata-free MSQ1
serializer.
