#!/usr/bin/env python3
"""Tiny fine-tune of Qwen3-Embedding-0.6B on 20 contrastive triplets.

Implements an in-batch-negatives (Multiple Negatives Ranking) InfoNCE
objective with a manual torch loop on top of a SentenceTransformer model.
This keeps the README pipeline (Transformer -> last-token Pooling ->
L2 Normalize) intact and lets us save a fully SentenceTransformer-loadable
checkpoint via model.save(), while avoiding any SentenceTransformerTrainer
API uncertainty.

References: model README "Transformers Usage" + "Sentence Transformers Usage".
"""
import os
import sys
import json
import random
import time

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from ppu_evidence import header, assert_on_cuda, mem
from train_logger import GPUMonitor, measure_flops, log_step, log_eval, train_summary, query_ppu_now

# Locate the checkpoint from the environment (works on host and in the
# PPU container where checkpoints are mounted under /workspace/checkpoints).
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "Qwen3-Embedding-0.6B")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(CODE_DIR, "finetuned")
CASES_PATH = os.path.join(CODE_DIR, "test_cases.json")

# PPU masquerades as CUDA: torch.cuda.is_available() is True in the container.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Query instruction prompt. Mirrors config_sentence_transformers.json
# prompts["query"] and README get_detailed_instruct:
#   f'Instruct: {task}\nQuery:{query}'  (no space after "Query:")
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)

MAX_LENGTH = 512  # SPEC: train with max_length 512 even though model supports 32k.
EPOCHS = 2
BATCH = 4
LR = 2e-5
SCALE = 20.0  # InfoNCE temperature scaling: cosine * scale before cross-entropy.


def embed_with_grad(model, tokenizer, texts):
    """Tokenize texts and run the full SentenceTransformer pipeline under grad."""
    feats = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    feats = {k: v.to(device) for k, v in feats.items()}
    out = model(feats)
    # SentenceTransformer.forward may return the full features dict (with
    # "sentence_embedding") or just the embedding tensor, depending on the
    # installed version. Handle both.
    return out["sentence_embedding"] if isinstance(out, dict) else out


def main():
    header("Qwen3-Embedding-0.6B")
    random.seed(42)
    torch.manual_seed(42)

    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"

    print(f"device={device}")
    print(f"model_path={MODEL_PATH}")

    # padding_side="left" per README so the last pooled token sits at the end
    # of each (left-padded) row. No trust_remote_code: qwen3 is a native
    # transformers>=4.51 model type.
    model = SentenceTransformer(
        MODEL_PATH,
        device=device,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = MAX_LENGTH
    model.train()
    assert_on_cuda(model, "Qwen3-Embedding-0.6B")

    tokenizer = model[0].tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- standard AI training logging + PPU GPU status (additive only) ---
    def step_fn(batch):
        """One forward+backward (no optimizer.step/zero_grad) for FLOP counting."""
        queries = [QUERY_PROMPT + c["query"] for c in batch]
        positives = [c["positive"] for c in batch]
        q_emb = embed_with_grad(model, tokenizer, queries)
        p_emb = embed_with_grad(model, tokenizer, positives)
        scores = (q_emb @ p_emb.t()) * SCALE
        labels = torch.arange(len(batch), device=device)
        loss = F.cross_entropy(scores, labels)
        loss.backward()
        return loss

    def evaluate(cases_eval):
        """Per-epoch triplet accuracy: acc_i = 1 if cos(q_i,p_i) > cos(q_i,n_i)."""
        model.eval()
        with torch.no_grad():
            queries = [QUERY_PROMPT + c["query"] for c in cases_eval]
            positives = [c["positive"] for c in cases_eval]
            negatives = [c["negative"] for c in cases_eval]
            q_emb = embed_with_grad(model, tokenizer, queries)
            p_emb = embed_with_grad(model, tokenizer, positives)
            n_emb = embed_with_grad(model, tokenizer, negatives)
            # Pipeline ends with 2_Normalize -> unit-norm; re-normalize for safety.
            q_emb = F.normalize(q_emb, dim=-1)
            p_emb = F.normalize(p_emb, dim=-1)
            n_emb = F.normalize(n_emb, dim=-1)
            sim_p = (q_emb * p_emb).sum(dim=-1)
            sim_n = (q_emb * n_emb).sum(dim=-1)
            acc = int((sim_p > sim_n).sum().item())
        model.train()
        return acc

    n_steps_total = EPOCHS * ((len(cases) + BATCH - 1) // BATCH)
    sample_batch = [cases[i] for i in range(min(BATCH, len(cases)))]
    flops_per_step = measure_flops(lambda: step_fn(sample_batch))
    optimizer.zero_grad()  # clear grads left by the FLOP-counting backward
    print(f"flops_per_step={flops_per_step}")

    monitor = GPUMonitor()
    monitor.start()
    train_start = time.time()
    final_acc = 0

    indices = list(range(len(cases)))
    step = 0
    for epoch in range(EPOCHS):
        random.shuffle(indices)
        epoch_losses = []
        for start in range(0, len(indices), BATCH):
            batch_idx = indices[start:start + BATCH]
            batch = [cases[i] for i in batch_idx]

            t_step_start = time.time()

            queries = [QUERY_PROMPT + c["query"] for c in batch]
            positives = [c["positive"] for c in batch]

            q_emb = embed_with_grad(model, tokenizer, queries)   # [B, 1024]
            p_emb = embed_with_grad(model, tokenizer, positives)  # [B, 1024]
            # Pipeline ends with 2_Normalize -> embeddings are unit-norm,
            # so the dot product equals cosine similarity.

            scores = (q_emb @ p_emb.t()) * SCALE  # [B, B]
            labels = torch.arange(len(batch_idx), device=device)
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            t_step = time.time() - t_step_start
            throughput = len(batch_idx) / t_step if t_step > 0 else 0.0
            epoch_losses.append(loss.item())
            log_step(
                epoch + 1, EPOCHS, step, n_steps_total, loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                throughput=throughput,
                gpu=monitor.snapshot_line(),
            )

        acc = evaluate(cases)
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        log_eval(epoch + 1, acc, len(cases), mean_loss)
        final_acc = acc

    train_time = time.time() - train_start
    monitor.stop()
    gpu_stats = monitor.stats()

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    model.save(FINETUNED_DIR)
    print(f"saved finetuned model to {FINETUNED_DIR}")
    mem("after_train")
    train_summary(
        "Qwen3-Embedding-0.6B", step, train_time, flops_per_step,
        gpu_stats, final_acc, len(cases), EPOCHS,
    )
    print("TRAIN_OK")


if __name__ == "__main__":
    main()
