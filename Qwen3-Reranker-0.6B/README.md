# Qwen3-Reranker-0.6B — train & infer

Tiny fine-tune + inference for **Qwen3-Reranker-0.6B**, a 0.6B causal-LM
reranker (100+ languages). The code follows the model README's "Transformers"
usage exactly: a manual `AutoModelForCausalLM` loop that scores each
(query, document) pair by reading the last-token logits for the tokens
`yes` / `no`.

## Files

| file | purpose |
|------|---------|
| `test_cases.json` | 20 `{query, document, label}` cases — 10 relevant (label 1), 10 not (label 0), multilingual (en + zh), topics: geo / physics / history / code / daily-life |
| `train.py` | tiny fine-tune on the 20 cases → saves to `./finetuned/`, prints `TRAIN_OK` |
| `infer.py` | scores the 20 pairs, asserts 20 scores, writes `results.json`, prints `INFER_OK`, `exit 0` |
| `requirements.txt` | extra pip deps |

## Environment

- Hardware: 含光 PPU (PPU-ZW810E). In the Aliyun `inference-xpu-pytorch`
  container `torch.cuda.is_available()` is `True` (PPU masquerades as CUDA), so
  everything is treated as CUDA. Code picks `cuda` and falls back to `cpu`.
- Models are local — no network downloads. The checkpoint root is read from
  the `CHECKPOINTS_DIR` environment variable, defaulting to the host path:

```bash
export CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints
# → MODEL_PATH = $CHECKPOINTS_DIR/Qwen3-Reranker-0.6B
```

Inside the container, point it at the mounted path, e.g.
`export CHECKPOINTS_DIR=/workspace/checkpoints`.

Install deps (torch comes from the image):

```bash
pip install -r requirements.txt
```

Requires `transformers>=4.51.0` for the `qwen3` architecture.

## Run

From this directory:

```bash
# 1) fine-tune on the 20 cases (2 epochs, batch 4, lr 2e-5, max_length 512)
python train.py
# → writes ./finetuned/ and prints TRAIN_OK

# 2) score the 20 pairs (uses ./finetuned/ if present, else the base checkpoint)
python infer.py
# → writes ./results.json and prints INFER_OK
```

## How it works (model-specific notes)

The implementation mirrors the README's "Using Transformers" section:

1. **Tokenizer** loaded with `padding_side="left"` (so the last position is
   always the final real token). `pad_token` is set to `eos_token` if missing.
2. **Prompt scaffolding** — each pair is wrapped as
   `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`, then the
   full sequence is `prefix_tokens + body + suffix_tokens`, where the prefix is
   the system/user turn and the suffix is the assistant turn plus an empty
   reasoning-block scaffold. The exact prefix/suffix byte strings are taken
   verbatim from the checkpoint `README.md` (see `train.py` / `infer.py`).
3. **Scoring** — `model(**inputs).logits[:, -1, :]` reads the last-token
   logits; the `yes` (true, id 9693) and `no` (false, id 2152) columns are
   stacked as `[no, yes]`. The ids come from the checkpoint's
   `1_LogitScore/config.json` (with `convert_tokens_to_ids("yes"/"no")` and
   the verified constants as fallbacks).
4. **Inference score** — `log_softmax([no, yes])` then `.exp()` on the `yes`
   column → `P(yes) ∈ (0,1)` as the relevance probability.
5. **Training** — same logits extraction, but `[no, yes]` is fed to
   `cross_entropy` against the 0/1 label (label 0 → maximize `no`, label 1 →
   maximize `yes`), then `AdamW` `backward`/`step`. Trains in `bfloat16`,
   `max_length=512`, `use_cache=False`.

### Key parameters

| param | value | note |
|-------|-------|------|
| model class | `AutoModelForCausalLM` | causal LM, not a sequence classifier |
| dtype | `bfloat16` | matches checkpoint |
| training max_length | 512 | SPEC: training cap 512 even though model supports 32k |
| inference max_length | 8192 | README default; actual sequences are far shorter |
| lr / epochs / batch | 2e-5 / 2 / 4 | tiny fine-tune |
| token_true_id (`yes`) | 9693 | from `1_LogitScore/config.json` |
| token_false_id (`no`) | 2152 | from `1_LogitScore/config.json` |
| default instruction | `Given a web search query, retrieve relevant passages that answer the query` | README default |

### Why a manual loop (not SentenceTransformers `CrossEncoder`)?

The SPEC mandates the manual transformers loop for Qwen3-Reranker to avoid the
mismatch between ST's `CrossEncoder` wrapper and a causal-LM reranker. The
manual loop gives direct control over the yes/no logit extraction and the
training loss, and runs without error on the PPU.

## Verification checklist

- `train.py` runs on cuda, produces `./finetuned/`, prints `TRAIN_OK`.
- `infer.py` runs on cuda, produces 20 scores, asserts `len == 20`, writes
  `results.json`, prints `INFER_OK`, exits 0.
