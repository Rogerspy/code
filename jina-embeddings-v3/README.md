# jina-embeddings-v3 — train & infer

Production-quality, README-faithful **training + inference** code for
[`jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3) running on the
FlagOS / 含光 PPU (which masquerades as CUDA, so everything is treated as `cuda`).

## Files

| file | purpose |
|------|---------|
| `test_cases.json` | 20 multilingual `(query, positive, negative)` triples |
| `train.py`        | tiny fine-tune (2 epochs) → saves to `./finetuned/`, prints `TRAIN_OK` |
| `infer.py`        | encode queries + docs, cosine sims, asserts dim=1024, writes `results.json`, prints `INFER_OK` |
| `requirements.txt`| extra pip dependencies |
| `finetuned/`      | produced by `train.py` (model + tokenizer) |

## Setup

```bash
pip install -r requirements.txt
```

The model weights are loaded from a local checkpoint (no network). Point the code at the
checkpoints root with an env var (defaults to the host path):

```bash
export CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints
# inside the FlagOS container this is mounted at /workspace/checkpoints, e.g.:
# export CHECKPOINTS_DIR=/workspace/checkpoints
```

`MODEL_PATH` is computed as `$CHECKPOINTS_DIR/jina-embeddings-v3`.

## Train

```bash
cd /home/qiyiyan/songchao/code/jina-embeddings-v3
python train.py
```

- 1–2 epochs, batch 4, lr 2e-5, `max_length=512` (model supports 8192; 512 for speed).
- Loads `AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)` (fp32 on cuda).
- Loss: in-batch infoNCE over `(query, positive)` — the manual equivalent of
  `MultipleNegativesRankingLoss`. Cosine similarities × temperature 20 (ST default).
- Queries use LoRA task `retrieval.query`, positives use `retrieval.passage`. Only the
  LoRA adapters are trainable (`lora_main_params_trainable=false`).
- Saves to `./finetuned/` and prints `TRAIN_OK`.

## Infer

```bash
cd /home/qiyiyan/songchao/code/jina-embeddings-v3
python infer.py
```

- Loads `./finetuned/` if present, otherwise the base checkpoint.
- Encodes the 20 queries (task `retrieval.query`) and positives+negatives
  (task `retrieval.passage`) with the README's AutoModel + mean-pooling + L2-norm path.
- Asserts embedding dimension == **1024** (jina-v3 `hidden_size`).
- Prints per-case cosine similarities and writes `results.json`.
- Prints `INFER_OK` and exits 0.

## Model-specific notes (from the upstream README)

- **Architecture**: Jina-XLM-RoBERTa with 5 LoRA adapters
  (`retrieval.query`, `retrieval.passage`, `separation`, `classification`, `text-matching`).
- **Pooling**: mean pooling (see `1_Pooling/config.json`) + L2 normalization.
- **Task selection**: pass the LoRA task via an `adapter_mask` built from
  `model._adaptation_map[task]` (an `int32` tensor of length = batch).
- **trust_remote_code**: required (`config.json` `auto_map` points to
  `jinaai/xlm-roberta-flash-implementation`; `custom_st.py` is the local ST wrapper).
- **Matryoshka**: supports `truncate_dim` ∈ {32,64,128,256,512,768,1024}. We use the
  full 1024 here (no `truncate_dim`).
- **numpy<2**: required by jina-v3 (see `requirements.txt`).
- **flash-attn disabled**: `config.use_flash_attn = False` at load time to avoid any
  flash-attn / SDPA backend mismatch on the PPU; the jina impl falls back to its
  standard einops attention path. Correctness is unchanged (only speed).
- **dim**: 1024 (`config.json` `hidden_size`).

### Why not `SentenceTransformer(...)` directly?

The local checkpoint ships `custom_st.py` (the sentence-transformers `Transformer`
module) but **no `sentence_bert_config.json`**. `SentenceTransformer(MODEL_PATH)` calls
`custom_st.Transformer.load(MODEL_PATH)`, which requires one of the `sentence_*_config.json`
files and raises `FileNotFoundError` otherwise. The upstream README's *primary* usage is
the `transformers` `AutoModel` path (`AutoModel.from_pretrained(..., trust_remote_code=True)`
+ manual mean pooling), which is fully self-contained and equivalent, so both `train.py`
and `infer.py` use it. This is the SPEC-sanctioned manual torch-loop fallback for embed
models.
