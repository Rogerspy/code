#!/usr/bin/env python3
"""Tiny fine-tune for BAAI/bge-large-zh-v1.5 (Chinese embedding model).

Reads 20 (query, positive, negative) triples from test_cases.json, runs a tiny
contrastive fine-tune (MultipleNegativesRankingLoss / in-batch InfoNCE) on CUDA,
and saves the model to ./finetuned/ in sentence-transformers format.

Following SPEC.md, the primary path is SentenceTransformer + SentenceTransformerTrainer
+ MultipleNegativesRankingLoss. Two robust fallbacks (legacy model.fit + MNRL, and a
raw AutoModel + CLS-pooling loop matching the README's HuggingFace Transformers usage)
keep the bar "runs without error on PPU". All paths save in sentence-transformers
format so infer.py can load ./finetuned/ via SentenceTransformer.

Queries are prefixed with the bge-large-zh retrieval instruction; passages are not,
matching the README. max_length is fixed at 512 even though the model supports 512.
Prints TRAIN_OK on success.
"""
import os
import sys
import json
import shutil
import time
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from ppu_evidence import header, assert_on_cuda, mem
from train_logger import GPUMonitor, measure_flops, log_step, log_eval, train_summary

CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "bge-large-zh-v1.5")
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CASES = os.path.join(HERE, "test_cases.json")
OUTPUT_DIR = os.path.join(HERE, "finetuned")

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 512
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-5
INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # bge-large-zh retrieval query instruction


def load_cases():
    with open(TEST_CASES, "r", encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"
    return cases


def save_st_model(model):
    """Save a SentenceTransformer to OUTPUT_DIR in sentence-transformers format."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(OUTPUT_DIR)
    else:  # very old ST only exposes .save
        model.save(OUTPUT_DIR)


def install_st_metadata():
    """Copy sentence-transformers metadata so a raw transformers save is ST-loadable.

    Used only by the raw-AutoModel fallback, which saves config.json + weights via
    transformers but still needs modules.json / 1_Pooling / sentence_bert_config.json
    for SentenceTransformer to reconstruct the Transformer+Pooling+Normalize stack.
    """
    for fname in ("modules.json", "sentence_bert_config.json", "config_sentence_transformers.json"):
        src = os.path.join(MODEL_PATH, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUTPUT_DIR, fname))
    src_pool = os.path.join(MODEL_PATH, "1_Pooling")
    dst_pool = os.path.join(OUTPUT_DIR, "1_Pooling")
    if os.path.isdir(src_pool):
        os.makedirs(dst_pool, exist_ok=True)
        shutil.copy(os.path.join(src_pool, "config.json"), os.path.join(dst_pool, "config.json"))


# --- Path 1: SentenceTransformerTrainer + MultipleNegativesRankingLoss ----------
def train_with_trainer(cases):
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from transformers import TrainerCallback
    from datasets import Dataset

    model = SentenceTransformer(MODEL_PATH, device=device)
    model.max_seq_length = MAX_LENGTH
    assert_on_cuda(model, "bge-large-zh-v1.5")

    anchors = [INSTRUCTION + c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    n_steps = EPOCHS * math.ceil(len(cases) / BATCH_SIZE)

    train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.0,
        save_strategy="no",
        logging_steps=2,
        report_to="none",
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    # --- standard training logging + GPU monitor (additive; trainer's own prints kept) ---
    class _StdLogCallback(TrainerCallback):
        """Bridge SentenceTransformerTrainer log events -> standard log_step lines.

        Only additive: the trainer already prints standard loss/lr/epoch/
        train_runtime/throughput lines, which are kept untouched. This callback
        additionally emits one log_step (with a live GPU snapshot line) per
        trainer log event so per-step GPU/util/power state is captured.
        """

        def __init__(self, monitor, n_epochs, n_steps, batch_size):
            self.monitor = monitor
            self.n_epochs = n_epochs
            self.n_steps = n_steps
            self.batch_size = batch_size
            self._last_step = 0
            self._last_t = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                logs = logs or {}
                loss_v = logs.get("loss")
                if loss_v is None:
                    return
                lr = logs.get("learning_rate")
                step = state.global_step
                ep_f = logs.get("epoch")
                ep = min(int(ep_f) + 1, self.n_epochs) if ep_f is not None else 1
                now = time.time()
                dt = now - self._last_t
                thr = None
                if dt > 0 and step > self._last_step:
                    thr = (step - self._last_step) * self.batch_size / dt
                log_step(ep, self.n_epochs, step, self.n_steps, float(loss_v),
                         lr=(float(lr) if lr is not None else None),
                         throughput=thr, gpu=self.monitor.snapshot_line())
                self._last_step = step
                self._last_t = now
            except Exception:
                pass

    monitor = GPUMonitor()
    trainer.add_callback(_StdLogCallback(monitor, EPOCHS, n_steps, BATCH_SIZE))
    t0 = time.time()
    monitor.start()
    trainer.train()
    monitor.stop()
    train_time = time.time() - t0

    # --- retrieval accuracy eval: per case cos(query, positive) > cos(query, negative) ---
    final_acc = 0
    try:
        model.eval()
        q_emb = model.encode([INSTRUCTION + c["query"] for c in cases],
                             convert_to_tensor=True, device=device, show_progress_bar=False)
        p_emb = model.encode([c["positive"] for c in cases],
                             convert_to_tensor=True, device=device, show_progress_bar=False)
        n_emb = model.encode([c["negative"] for c in cases],
                             convert_to_tensor=True, device=device, show_progress_bar=False)
        q_emb = F.normalize(q_emb, p=2, dim=1)
        p_emb = F.normalize(p_emb, p=2, dim=1)
        n_emb = F.normalize(n_emb, p=2, dim=1)
        qp = (q_emb * p_emb).sum(dim=1)
        qn = (q_emb * n_emb).sum(dim=1)
        final_acc = int((qp > qn).sum().item())
        mean_loss = float(F.softplus(qn - qp).mean().item())
        log_eval(EPOCHS, final_acc, len(cases), mean_loss)
    except Exception as e:  # noqa: BLE001  (eval must not break the save/TRAIN_OK path)
        print(f"[eval] retrieval acc eval failed: {e!r}")
        log_eval(EPOCHS, 0, len(cases), 0.0)

    # --- measure FLOPs of one forward step (model.encode a batch) ---
    flops_per_step = 0
    try:
        def _flop_fn():
            with torch.no_grad():
                model.encode(anchors[:BATCH_SIZE], convert_to_tensor=True,
                             device=device, show_progress_bar=False)
        flops_per_step = measure_flops(_flop_fn)
    except Exception as e:  # noqa: BLE001  (flops measurement must not break the path)
        print(f"[flops] measure_flops failed: {e!r}")

    train_summary("bge-large-zh-v1.5", n_steps, train_time, flops_per_step,
                  monitor.stats(), final_acc, len(cases), EPOCHS)

    save_st_model(model)


# --- Path 2: legacy stable model.fit + MultipleNegativesRankingLoss ------------
def train_with_fit(cases):
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(MODEL_PATH, device=device)
    model.max_seq_length = MAX_LENGTH
    assert_on_cuda(model, "bge-large-zh-v1.5")

    anchors = [INSTRUCTION + c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    examples = [InputExample(texts=[a, p]) for a, p in zip(anchors, positives)]
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=EPOCHS,
        warmup_steps=0,
        optimizer_params={"lr": LR},
        show_progress_bar=False,
    )
    save_st_model(model)


# --- Path 3: raw AutoModel + CLS pooling + normalize (README HF transformers) --
def train_manual(cases):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.train()
    assert_on_cuda(model, "bge-large-zh-v1.5")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    anchors = [INSTRUCTION + c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    n = len(anchors)
    scale = 20.0  # MultipleNegativesRankingLoss default temperature scale

    for epoch in range(EPOCHS):
        perm = torch.randperm(n).tolist()
        total = 0.0
        i = 0
        while i < n:
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                idx = (perm + perm)[:BATCH_SIZE]
            i += BATCH_SIZE
            q_texts = [anchors[j] for j in idx]
            p_texts = [positives[j] for j in idx]

            q = tokenizer(q_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            q = {k: v.to(device) for k, v in q.items()}
            q_emb = model(**q).last_hidden_state[:, 0]
            q_emb = F.normalize(q_emb, p=2, dim=1)

            p = tokenizer(p_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            p = {k: v.to(device) for k, v in p.items()}
            p_emb = model(**p).last_hidden_state[:, 0]
            p_emb = F.normalize(p_emb, p=2, dim=1)

            scores = q_emb @ p_emb.T * scale
            labels = torch.arange(len(idx)).to(device)
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"[manual] epoch {epoch + 1}/{EPOCHS} loss={total:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    install_st_metadata()


def main():
    header("bge-large-zh-v1.5")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cases = load_cases()
    print(f"bge-large-zh-v1.5 fine-tune | cases={len(cases)} device={device} "
          f"epochs={EPOCHS} batch={BATCH_SIZE} lr={LR} max_len={MAX_LENGTH}")
    paths = [
        ("SentenceTransformerTrainer+MNRL", train_with_trainer),
        ("model.fit+MNRL", train_with_fit),
        ("manual AutoModel+CLS", train_manual),
    ]
    last_err = None
    for name, fn in paths:
        try:
            print(f"==> trying {name} ...")
            fn(cases)
            print(f"==> {name} succeeded; saved to {OUTPUT_DIR}")
            mem("after_train")
            print("TRAIN_OK")
            return
        except Exception as e:  # noqa: BLE001  (intentional: try next robust path)
            last_err = e
            print(f"==> {name} failed: {e!r}")
            # clean any partial output before the next attempt
            if os.path.isdir(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
    raise RuntimeError(f"all training paths failed; last error: {last_err!r}")


if __name__ == "__main__":
    main()
