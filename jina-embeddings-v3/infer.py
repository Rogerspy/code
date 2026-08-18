#!/usr/bin/env python
# coding: utf-8
"""Inference for jina-embeddings-v3 on 20 multilingual cases.

Loads ``./finetuned/`` if it exists, otherwise the base checkpoint. Encodes queries
(LoRA task ``retrieval.query``) and documents (positives + negatives, LoRA task
``retrieval.passage``) using the README's AutoModel + manual mean-pooling + L2 norm path.
Asserts the embedding dimension equals 1024, prints cosine similarities, writes
``results.json``, prints ``INFER_OK`` and exits 0.
"""
import glob
import json
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "jina-embeddings-v3")
HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(HERE, "finetuned")
CASES_PATH = os.path.join(HERE, "test_cases.json")
RESULTS_PATH = os.path.join(HERE, "results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_TASK = "retrieval.query"
PASSAGE_TASK = "retrieval.passage"
MAX_LENGTH = 512
EXPECTED_DIM = 1024          # jina-embeddings-v3 hidden_size (README: Matryoshka up to 1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def task_id_of(model, config, task):
    amap = getattr(model, "_adaptation_map", None)
    if amap and task in amap:
        return int(amap[task])
    adapts = getattr(config, "lora_adaptations", None)
    if adapts and task in adapts:
        return int(adapts.index(task))
    raise ValueError(f"Task '{task}' not found in lora_adaptations {adapts}.")


def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def last_hidden_state(model_output):
    if hasattr(model_output, "last_hidden_state"):
        return model_output.last_hidden_state
    return model_output[0]


def encode(texts, model, tokenizer, task_id):
    """Forward pass WITHOUT gradients -> L2-normalized sentence embeddings."""
    model.eval()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    adapter_mask = torch.full((len(texts),), task_id, dtype=torch.int32, device=device)
    with torch.no_grad():
        out = model(**enc, adapter_mask=adapter_mask)
    emb = mean_pool(last_hidden_state(out), enc["attention_mask"])
    return F.normalize(emb, p=2, dim=1)


def load_model(path):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    try:
        config.use_flash_attn = False
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(
        path, config=config, trust_remote_code=True, dtype=torch.float32
    )
    model.to(device).float()
    return model, tokenizer, config


def seed_custom_code_cache(model_dir):
    """Pre-copy the trust_remote_code .py files into the HF modules cache.

    ``model.save_pretrained`` writes the custom modeling files (block.py,
    modeling_lora.py, ...) into the finetuned dir, but transformers' local
    trust_remote_code loader copies only the entrypoint module to the cache and
    then scans *relative imports at the cache path* -- so sibling files like
    block.py are missing there and reload fails with FileNotFoundError. We mirror
    all .py files into the cache (key == dir basename, e.g. "finetuned") so the
    relative-import walk succeeds.
    """
    hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    name = os.path.basename(os.path.normpath(model_dir))
    cache_mod = os.path.join(hf_home, "modules", "transformers_modules", name)
    try:
        os.makedirs(cache_mod, exist_ok=True)
        for py in glob.glob(os.path.join(model_dir, "*.py")):
            shutil.copy(py, cache_mod)
    except Exception as e:
        print(f"[infer] cache seed skipped: {e!r}")


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------
def main():
    if os.path.exists(os.path.join(FINETUNED_DIR, "config.json")):
        load_path = FINETUNED_DIR
        print(f"[infer] using finetuned model: {load_path}")
    else:
        load_path = MODEL_PATH
        print(f"[infer] using base model: {load_path}")

    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Model dir not found: {load_path} (set CHECKPOINTS_DIR)")

    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"
    print(f"[infer] loaded {len(cases)} cases")

    if load_path == FINETUNED_DIR:
        seed_custom_code_cache(FINETUNED_DIR)
    try:
        model, tokenizer, config = load_model(load_path)
        print(f"[infer] model loaded from {load_path}")
    except Exception as e:
        print(f"[infer] finetuned load failed ({e!r}); falling back to base: {MODEL_PATH}")
        load_path = MODEL_PATH
        model, tokenizer, config = load_model(load_path)
    q_id = task_id_of(model, config, QUERY_TASK)
    p_id = task_id_of(model, config, PASSAGE_TASK)
    print(f"[infer] task ids: {QUERY_TASK}={q_id}  {PASSAGE_TASK}={p_id}")

    queries = [c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    negatives = [c["negative"] for c in cases]

    q_emb = encode(queries, model, tokenizer, q_id)
    p_emb = encode(positives, model, tokenizer, p_id)
    n_emb = encode(negatives, model, tokenizer, p_id)

    dim = int(q_emb.shape[1])
    print(f"[infer] embedding dim = {dim}")
    assert dim == EXPECTED_DIM, f"expected embedding dim {EXPECTED_DIM}, got {dim}"

    # cosine similarities (embeddings are already L2-normalized -> dot product == cosine)
    qp = (q_emb * p_emb).sum(dim=1)
    qn = (q_emb * n_emb).sum(dim=1)
    qp_list = [round(float(x), 6) for x in qp.tolist()]
    qn_list = [round(float(x), 6) for x in qn.tolist()]
    sim_qp = (q_emb @ p_emb.t()).tolist()  # [20, 20] query x positive

    print("[infer] per-case cosine similarities (query-positive vs query-negative):")
    for i, (a, b) in enumerate(zip(qp_list, qn_list)):
        tag = cases[i].get("lang", "?") + "/" + cases[i].get("topic", "?")
        print(f"  case {i:2d} [{tag:12s}] qp={a:.4f}  qn={b:.4f}")

    mean_qp = round(float(qp.mean()), 6)
    mean_qn = round(float(qn.mean()), 6)
    print(f"[infer] mean query-positive sim = {mean_qp:.4f}")
    print(f"[infer] mean query-negative sim = {mean_qn:.4f}")

    results = {
        "model": "jina-embeddings-v3",
        "load_path": load_path,
        "device": device,
        "embedding_dim": dim,
        "num_cases": len(cases),
        "query_positive_sims": qp_list,
        "query_negative_sims": qn_list,
        "mean_qp": mean_qp,
        "mean_qn": mean_qn,
        "similarity_matrix_query_positive": sim_qp,
        "cases": cases,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[infer] wrote {RESULTS_PATH}")
    print("INFER_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
