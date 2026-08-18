# Qwen3-Embedding-0.6B — Train & Infer (含光 PPU)

Tiny fine-tune + inference for **Qwen3-Embedding-0.6B**, following the model
README and the unified SPEC (`code/_common/SPEC.md`).

## Model notes

- **Type**: text embedding, 0.6B, 28 layers, hidden size 1024, **embedding dim 1024**.
- **Pooling**: last-token (`1_Pooling` `pooling_mode_lasttoken=true`) followed by L2
  normalization (`2_Normalize`). This matches the README "Transformers Usage"
  `last_token_pool` + `F.normalize` path.
- **Instruction-aware**: queries use the prompt stored in
  `config_sentence_transformers.json`:
  `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`.
  Documents are encoded without an instruction.
- **`padding_side="left"`** (per README recommendation) so the last token used
  for pooling sits at the end of each left-padded row.
- **Context**: 32k supported, but training/inference here cap `max_length=512`
  for speed (the 20 test texts are short).
- **No `trust_remote_code`**: `qwen3` is a native model type in `transformers>=4.51`;
  there is no `custom_st.py` in this checkpoint.

## Files

| File | Description |
|------|-------------|
| `test_cases.json` | 20 multilingual `{query, positive, negative}` triplets (EN/CN/ES; geography, physics, history, programming, daily life, chemistry, biology, math, astronomy, deep learning, economics, sports, art, medicine). |
| `train.py` | Fine-tune with an in-batch-negatives (Multiple Negatives Ranking) InfoNCE loss via a manual torch loop on a `SentenceTransformer`; saves to `./finetuned/`. Prints `TRAIN_OK`. |
| `infer.py` | Loads `./finetuned/` if present else the base checkpoint; encodes queries+docs; prints the cosine similarity matrix; asserts dim==1024; writes `results.json`. Prints `INFER_OK`, exits 0. |
| `requirements.txt` | Extra pip dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
export CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints   # default if unset
python train.py
```

- Loads `$CHECKPOINTS_DIR/Qwen3-Embedding-0.6B`.
- 2 epochs, batch 4, lr 2e-5, `max_length=512`.
- Saves a full SentenceTransformer directory to `./finetuned/` (config,
  weights, `modules.json`, `1_Pooling`, `2_Normalize`, tokenizer, prompts).

## Infer

```bash
export CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints
python infer.py
```

- Prefers `./finetuned/` (detected via `finetuned/config.json`); falls back to
  the base checkpoint if no fine-tune has been run.
- Builds the 20×40 cosine similarity matrix (queries × [positives+negatives]).
- Writes `results.json` (matrix + per-case `sim_positive` / `sim_negative`).

## Device

Uses `cuda` when `torch.cuda.is_available()` (the 含光 PPU masquerades as
CUDA inside the `inference-xpu-pytorch` container), otherwise `cpu`. All
tensors are explicitly moved to the selected device.

## Why a manual loop instead of `SentenceTransformerTrainer`?

The README lists the Sentence Transformers API and a plain Transformers loop
side by side. Training here uses a small manual `forward → loss → backward →
optimizer.step` loop on the `SentenceTransformer` object. This is robust to
`SentenceTransformerTrainer` API differences across versions, still goes through
the exact README pooling/normalize pipeline, and saves a checkpoint that
`infer.py` loads with the same `SentenceTransformer` API.
