# bge-large-zh-v1.5 — training & inference code

Embedding model `BAAI/bge-large-zh-v1.5` (Chinese, BERT-large, CLS pooling,
embedding dim **1024**, max seq length 512). This directory holds a tiny
fine-tune + inference demo that runs on the 含光 PPU (treated as CUDA).

## Files

| File | Purpose |
|------|---------|
| `test_cases.json` | 20 Chinese `(query, positive, negative)` triples (geo / physics / history / code / daily-life) |
| `train.py` | Tiny contrastive fine-tune (MultipleNegativesRankingLoss), saves to `./finetuned/` |
| `infer.py` | Encode 20 cases, cosine similarity, assert dim==1024, write `results.json` |
| `requirements.txt` | Extra pip deps |

## Environment

The checkpoint root is read from the `CHECKPOINTS_DIR` env var (default
`/home/qiyiyan/songchao/checkpoints`). The model path is
`$CHECKPOINTS_DIR/bge-large-zh-v1.5`.

On the host the default path is used directly; inside the PPU container set:

```bash
export CHECKPOINTS_DIR=/workspace/checkpoints
```

Device selection follows the unified spec:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

(On the PPU `torch.cuda.is_available()` is `True` — the PPU masquerades as CUDA —
so all computation lands on-device.)

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` lists `sentence-transformers>=3.0`, `FlagEmbedding`,
`transformers>=4.51`, `torch`, `datasets`. No network download of weights
happens at runtime — the model is loaded from the local checkpoint dir.

## Train

```bash
python train.py
```

- Loads the model from `$CHECKPOINTS_DIR/bge-large-zh-v1.5`.
- Tiny fine-tune on the 20 triples: **2 epochs**, **batch 8**, **lr 2e-5**,
  **max_length 512**.
- Loss: `MultipleNegativesRankingLoss` (in-batch InfoNCE, scale 20). Queries are
  prefixed with the bge retrieval instruction
  `为这个句子生成表示以用于检索相关文章：`; passages are not, per the README.
- Saves to `./finetuned/` in sentence-transformers format and prints `TRAIN_OK`.

Implementation has three robust paths tried in order, so it runs without error
across sentence-transformers versions:

1. `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss` (ST ≥ 3.0, needs `datasets`).
2. Legacy stable `model.fit` + `MultipleNegativesRankingLoss` (works on older/newer ST).
3. Raw `AutoModel` + CLS pooling + L2 normalize (matches the README's HuggingFace
   Transformers usage); ST metadata (`modules.json`, `1_Pooling/`, …) is copied so
   `infer.py` can still load `./finetuned/` via `SentenceTransformer`.

All three save in sentence-transformers format so `infer.py` loads `./finetuned/`
transparently.

## Infer

```bash
python infer.py
```

- Loads `./finetuned/` if present (else the base checkpoint).
- Encodes 20 queries (with instruction), 20 positives, 20 negatives with
  `normalize_embeddings=True`.
- Prints the query×positive and query×negative cosine similarities and the full
  query×positive similarity matrix.
- Asserts the embedding dimension equals **1024** (per the README C-MTEB table).
- Writes `results.json` and prints `INFER_OK`, then exits 0.

## Model notes (from the official README)

- Pooling: **CLS** (`1_Pooling/config.json`: `pooling_mode_cls_token=true`), plus an
  L2 `Normalize` module — so `normalize_embeddings=True` gives cosine similarity.
- Retrieval instruction for short queries: `为这个句子生成表示以用于检索相关文章：`
  (no instruction on passages).
- Embedding dimension: **1024**; max sequence length: **512**.
- Similarity distribution sits roughly in `[0.6, 1]` (contrastive temperature 0.01);
  for retrieval, rank by relative score rather than an absolute threshold.
- This is a standard sentence-transformers model (no `custom_st.py`), so
  `trust_remote_code` is not needed.
