# bge-reranker-large — train & infer

Tiny fine-tune + inference harness for **`bge-reranker-large`** (BAAI/FlagEmbedding).
Faithful to the model's HF README and the unified `code/_common/SPEC.md`.

## Model summary (from `config.json` / README)

| field | value |
|---|---|
| architecture | `XLMRobertaForSequenceClassification` (xlm-roberta-large) |
| type | cross-encoder reranker |
| output | single relevance logit per `(query, document)` pair → `.logits.view(-1)` |
| num_labels | 1 |
| language | Chinese & English |
| max_length | 512 |
| instruction | none (reranker needs no query instruction) |
| loss | binary relevance / cross-entropy (sigmoid + BCE) |

The README inference snippet (HF transformers):

```python
inputs = tokenizer(pairs, padding=True, truncation=True,
                   return_tensors='pt', max_length=512)
scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
```

This harness reproduces that exactly.

## Files

| file | purpose |
|---|---|
| `test_cases.json` | 20 `{query, document, label}` cases (label ∈ {0,1}, 10/10 split), Chinese + English, topics: geo / physics / history / code / daily-life. Includes the README panda example. |
| `train.py` | fine-tune the 20 cases; saves `./finetuned/`; prints `TRAIN_OK`. |
| `infer.py` | score the 20 `(q,d)` pairs; asserts 20 scores; writes `results.json`; prints `INFER_OK`, `exit 0`. |
| `requirements.txt` | extra pip deps. |

## Setup

The host path defaults work directly:

```bash
export CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints   # default; set for the PPU container
pip install -r requirements.txt
```

Inside the Aliyun PPU PyTorch container (`code/_common/setup_container.sh`), the
checkpoint root is mounted to `/workspace/checkpoints`, so point `CHECKPOINTS_DIR`
there:

```bash
export CHECKPOINTS_DIR=/workspace/checkpoints
cd /workspace/code/bge-reranker-large
```

`MODEL_PATH` is computed as `$CHECKPOINTS_DIR/bge-reranker-large`. The PPU
masquerades as CUDA, so `torch.cuda.is_available()` is `True` and everything runs
on the accelerator.

## Train

```bash
python train.py
```

- Loads `AutoModelForSequenceClassification` + `AutoTokenizer` from `MODEL_PATH`.
- 2 epochs, batch 8, AdamW lr `2e-5`, `max_length=512`.
- Loss = `BCEWithLogitsLoss` on the single logit vs. the 0/1 label — exactly the
  `BinaryCrossEntropyLoss` the SPEC names (the model is "optimized based on
  cross-entropy loss" per the README). Implemented as a **clean manual torch
  loop** (forward → loss → backward → optimizer.step), the SPEC-sanctioned
  fallback that depends only on torch + transformers, so it runs without error
  on the PPU regardless of the installed `sentence-transformers` version.
- Saves model + tokenizer to `./finetuned/`. Prints `TRAIN_OK`.

> README-aligned alternative (if you prefer the high-level API): install
> `sentence-transformers>=3.0` and use `CrossEncoder` + `CrossEncoderTrainer`
> + `sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` — it
> computes the same `BCEWithLogitsLoss(logit, label)`. The manual loop here is
> chosen for maximum portability.

## Infer

```bash
python infer.py
```

- Loads `./finetuned/` if present, else the base `MODEL_PATH`.
- Tokenizes the 20 `(query, document)` pairs (`max_length=512`) and scores them
  with `.logits.view(-1).float()`.
- Prints every score, **asserts exactly 20 scores**, writes `results.json`,
  prints `INFER_OK`, and exits 0.

`results.json` shape:

```json
[
  {"index": 0, "query": "...", "document": "...", "label": 1, "score": 7.312},
  ...
]
```

## Notes

- No network: both scripts load weights from local `MODEL_PATH` / `./finetuned/`.
- `trust_remote_code` is NOT needed here (plain `XLMRobertaForSequenceClassification`).
- The relevance score is unbounded (cross-entropy trained), so higher = more
  relevant; absolute value is not a probability (README FAQ §2).
