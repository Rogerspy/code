#!/usr/bin/env python
# coding: utf-8
"""Tiny fine-tune for jina-embeddings-v3 on 20 multilingual (query, positive) pairs.

Implementation note
-------------------
The local checkpoint ships ``custom_st.py`` (the sentence-transformers wrapper) but
NOT ``sentence_bert_config.json``, so ``SentenceTransformer(MODEL_PATH)`` cannot be
instantiated directly from the directory (its custom ``Transformer.load`` requires that
config file). We therefore use the README's *primary* transformers path, which is fully
supported and equivalent::

    AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

plus manual mean-pooling + L2 normalization, and train with an in-batch infoNCE loss
(the manual equivalent of ``MultipleNegativesRankingLoss`` over (query, positive)).
Queries use the LoRA task ``retrieval.query``; positives use ``retrieval.passage``.
Only the LoRA adapters are trainable (config ``lora_main_params_trainable=false``), so
this fine-tunes the LoRA adapter for the retrieval task, exactly as the README describes
for ST fine-tuning with a chosen ``default_task``.

The SPEC explicitly sanctions this fallback: "If ST Trainer API uncertain, fall back to
a clean manual torch loop (forward->loss->backward->optimizer.step) that trains on cuda
without error and saves."
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from ppu_evidence import header, assert_on_cuda, mem
from train_logger import GPUMonitor, measure_flops, log_step, log_eval, train_summary, query_ppu_now

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "jina-embeddings-v3")
HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(HERE, "finetuned")
CASES_PATH = os.path.join(HERE, "test_cases.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_TASK = "retrieval.query"
PASSAGE_TASK = "retrieval.passage"
MAX_LENGTH = 512        # train-time truncation (model supports 8192; 512 for speed)
BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-5
SCALE = 20.0            # MultipleNegativesRankingLoss default temperature for cosine sims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def task_id_of(model, config, task):
    """Resolve a LoRA task name to its integer adapter id (README: model._adaptation_map)."""
    amap = getattr(model, "_adaptation_map", None)
    if amap and task in amap:
        return int(amap[task])
    adapts = getattr(config, "lora_adaptations", None)
    if adapts and task in adapts:
        return int(adapts.index(task))
    raise ValueError(f"Task '{task}' not found in lora_adaptations {adapts}.")


def mean_pool(token_embeddings, attention_mask):
    """README mean pooling over token embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def last_hidden_state(model_output):
    if hasattr(model_output, "last_hidden_state"):
        return model_output.last_hidden_state
    return model_output[0]


def encode_grad(texts, model, tokenizer, task_id):
    """Forward pass WITH gradients -> L2-normalized sentence embeddings."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    adapter_mask = torch.full((len(texts),), task_id, dtype=torch.int32, device=device)
    out = model(**enc, adapter_mask=adapter_mask)
    emb = mean_pool(last_hidden_state(out), enc["attention_mask"])
    return F.normalize(emb, p=2, dim=1)


def encode_nograd(texts, model, tokenizer, task_id):
    """No-grad encode for evaluation (reuses encode_grad under torch.no_grad)."""
    with torch.no_grad():
        return encode_grad(texts, model, tokenizer, task_id)


def evaluate(model, tokenizer, cases, q_id, p_id):
    """Epoch-end retrieval accuracy: acc_i = 1 if cos(q_i, positive) > cos(q_i, negative).

    Embeddings come back L2-normalized from encode_*, so dot product == cosine.
    Queries use the retrieval.query LoRA task; positive & negative use retrieval.passage.
    """
    model.eval()
    q_emb = encode_nograd([c["query"] for c in cases], model, tokenizer, q_id)
    p_emb = encode_nograd([c["positive"] for c in cases], model, tokenizer, p_id)
    n_emb = encode_nograd([c["negative"] for c in cases], model, tokenizer, p_id)
    acc = 0
    for i in range(len(cases)):
        if (q_emb[i] @ p_emb[i]).item() > (q_emb[i] @ n_emb[i]).item():
            acc += 1
    model.train()
    return acc


def load_model(path):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # Avoid any flash-attn dependency / SDPA backend mismatch on the PPU (masquerades
    # as CUDA). Falls back to the standard einops attention path of the jina impl.
    try:
        config.use_flash_attn = False
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(
        path, config=config, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.to(device).float()
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    header("jina-embeddings-v3")
    torch.manual_seed(42)
    print(f"[train] device={device}")
    print(f"[train] model_path={MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model dir not found: {MODEL_PATH} (set CHECKPOINTS_DIR)")

    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"
    print(f"[train] loaded {len(cases)} cases")

    model, tokenizer, config = load_model(MODEL_PATH)
    model.train()
    assert_on_cuda(model, "jina-embeddings-v3")

    q_id = task_id_of(model, config, QUERY_TASK)
    p_id = task_id_of(model, config, PASSAGE_TASK)
    print(f"[train] task ids: {QUERY_TASK}={q_id}  {PASSAGE_TASK}={p_id}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {n_trainable:,} / {n_total:,} (LoRA adapters)")

    optimizer = torch.optim.AdamW(trainable, lr=LR)

    queries = [c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    n = len(cases)

    # --- standard AI training logging + PPU GPU monitoring setup ---
    n_steps_per_epoch = sum(1 for i in range(0, n, BATCH_SIZE) if min(BATCH_SIZE, n - i) >= 2)
    n_steps = n_steps_per_epoch * EPOCHS

    # measure FLOPs for one forward+backward step (before the training loop)
    _q_sample = queries[:BATCH_SIZE]
    _p_sample = positives[:BATCH_SIZE]

    def _meas_step():
        q_emb = encode_grad(_q_sample, model, tokenizer, q_id)
        p_emb = encode_grad(_p_sample, model, tokenizer, p_id)
        scores = q_emb @ p_emb.t() * SCALE
        labels = torch.arange(len(_q_sample), device=device)
        loss = F.cross_entropy(scores, labels)
        loss.backward()

    flops_per_step = measure_flops(_meas_step)
    optimizer.zero_grad()  # clear grads left by the measurement backward
    print(f"[train] flops_per_step={flops_per_step:,}")

    monitor = GPUMonitor()
    monitor.start()
    train_start = time.time()
    total_steps = 0
    global_step = 0
    final_acc = 0

    for epoch in range(EPOCHS):
        perm = torch.randperm(n).tolist()
        running, steps = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue  # in-batch infoNCE needs >= 2
            q = [queries[j] for j in idx]
            p = [positives[j] for j in idx]

            _t0 = time.time()
            q_emb = encode_grad(q, model, tokenizer, q_id)
            p_emb = encode_grad(p, model, tokenizer, p_id)

            # cosine similarities (already L2-normalized) * temperature
            scores = q_emb @ p_emb.t() * SCALE
            labels = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            steps += 1
            global_step += 1
            total_steps += 1
            _dt = max(time.time() - _t0, 1e-6)
            _tput = len(idx) / _dt
            log_step(epoch + 1, EPOCHS, global_step, n_steps, float(loss.item()),
                     lr=optimizer.param_groups[0]["lr"], throughput=_tput,
                     gpu=monitor.snapshot_line())
        acc = evaluate(model, tokenizer, cases, q_id, p_id)
        final_acc = acc
        log_eval(epoch + 1, acc, n, running / max(1, steps))
        print(f"[train] epoch {epoch + 1}/{EPOCHS}  mean_loss={running / max(1, steps):.4f}")

    monitor.stop()
    train_time = time.time() - train_start
    gpu_stats = monitor.stats()
    train_summary("jina-embeddings-v3", total_steps, train_time, flops_per_step,
                  gpu_stats, final_acc, n, EPOCHS)

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    print(f"[train] saved finetuned model to {FINETUNED_DIR}")
    mem("after_train")
    print("TRAIN_OK")


if __name__ == "__main__":
    main()
