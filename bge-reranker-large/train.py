#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune bge-reranker-large on the 20 local test cases.

bge-reranker-large is a cross-encoder (`XLMRobertaForSequenceClassification`,
num_labels=1): it takes a (query, document) pair and emits a single relevance
logit (README: `model(**inputs).logits.view(-1)`). It is "optimized based on
cross-entropy loss", i.e. binary relevance with a sigmoid -> BinaryCrossEntropy.

The SPEC names the README-aligned path as
`CrossEncoder` + `CrossEncoderTrainer` + `BinaryCrossEntropyLoss`. That loss is
mathematically `BCEWithLogitsLoss(single_logit, label)`. To maximize robustness
on the PPU (which always ships torch + transformers but whose exact
sentence-transformers version/module paths can vary) we use the explicitly
sanctioned clean manual torch loop (forward -> loss -> backward -> optimizer.step)
that depends only on torch/transformers. This is the *exact* computation the
CrossEncoder + BinaryCrossEntropyLoss stack would perform, just without the
abstraction layer, so it is faithful to the README.

Usage:
    CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints python train.py
"""

import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from ppu_evidence import header, assert_on_cuda, mem
from train_logger import GPUMonitor, measure_flops, log_step, log_eval, train_summary, query_ppu_now

# --------------------------------------------------------------------------- #
# Paths (host default; CHECKPOINTS_DIR env lets this run in the PPU container) #
# --------------------------------------------------------------------------- #
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "bge-reranker-large")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_DIR, "test_cases.json")
FINETUNED_DIR = os.path.join(CODE_DIR, "finetuned")

# --------------------------------------------------------------------------- #
# Device: PPU masquerades as CUDA, so cuda is the real accelerator.           #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Hyper-parameters (SPEC: 1-2 epochs, batch 4-8, lr 2e-5, max_length 512).    #
# --------------------------------------------------------------------------- #
MAX_LENGTH = 512
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-5
SEED = 42


class RerankerPairDataset(Dataset):
    """Holds (query, document, label) cases; tokenizes each pair to fixed length."""

    def __init__(self, cases, tokenizer, max_length=MAX_LENGTH):
        self.cases = cases
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        c = self.cases[idx]
        # text=query, text_pair=document: the HF tokenizer pairs them as
        # "<q></s><d></s>" which is exactly what the README inference snippet
        # does via tokenizer(pairs, ...). Fixed max_length padding keeps every
        # item identical in shape so the default DataLoader collate stacks them
        # with zero custom collation -> maximally robust.
        enc = self.tokenizer(
            c["query"],
            c["document"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # float label -> BCEWithLogitsLoss target (0.0 / 1.0)
        item["labels"] = torch.tensor(float(c["label"]), dtype=torch.float)
        return item


def main():
    header("bge-reranker-large")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"MODEL_PATH = {MODEL_PATH}")
    print(f"DEVICE     = {DEVICE}")
    print(f"finetuned -> {FINETUNED_DIR}")

    # Load local checkpoint (no network).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    assert_on_cuda(model, "bge-reranker-large")
    model.train()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"loaded {len(cases)} cases")

    # Sanity: labels are 0/1 and exactly 20 cases.
    labels = [c["label"] for c in cases]
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"
    assert all(l in (0, 1) for l in labels), "labels must be 0/1"

    dataset = RerankerPairDataset(cases, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # BinaryCrossEntropy == sigmoid(logit) + BCEWithLogitsLoss on the single
    # relevance logit. Equivalent to sentence-transformers' BinaryCrossEntropyLoss.
    loss_fn = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    total_steps = len(loader) * EPOCHS
    final_acc = 0
    n_cases = len(cases)

    # --- Measure FLOPs for one forward+backward step (standard AI training log) ---
    flops_per_step = 0
    try:
        _probe = dataset[0]
        _probe_batch = {}
        for _k, _v in _probe.items():
            _probe_batch[_k] = _v.unsqueeze(0).repeat(BATCH_SIZE, *([1] * _v.dim())).to(DEVICE)

        def _flops_step():
            _lbl = _probe_batch["labels"].view(-1)
            _inp = {k: v for k, v in _probe_batch.items() if k != "labels"}
            _logits = model(**_inp).logits.view(-1)
            _loss = loss_fn(_logits, _lbl)
            _loss.backward()
            optimizer.zero_grad(set_to_none=True)

        flops_per_step = measure_flops(_flops_step)
        print(f"[flops] measured flops_per_step = {flops_per_step}")
    except Exception as _e:
        print(f"[flops] measurement failed: {_e}")
        flops_per_step = 0

    # --- GPU monitor + train timer (standard AI training log) ---
    monitor = GPUMonitor()
    monitor.start()
    train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        running = 0.0
        n_batches = 0
        for batch in loader:
            t_step = time.perf_counter()
            # Do NOT pass `labels` into the model: with num_labels=1 the model's
            # own forward would apply MSELoss and shadow our BCE. We compute loss
            # explicitly on the raw logit.
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            logits = model(**inputs).logits.view(-1)  # [batch_size]
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.item())
            n_batches += 1
            global_step += 1
            # --- per-step log (standard AI training log) ---
            step_dt = max(time.perf_counter() - t_step, 1e-6)
            throughput = len(labels) / step_dt
            log_step(epoch + 1, EPOCHS, global_step, total_steps, float(loss.item()),
                     lr=LR, throughput=throughput, gpu=monitor.snapshot_line())
        print(f"epoch {epoch + 1}/{EPOCHS}  avg_loss={running / max(1, n_batches):.6f}  steps={global_step}")
        # --- epoch-end evaluation: accuracy on the 20 local cases (standard log) ---
        model.eval()
        with torch.no_grad():
            _correct = 0
            _eval_loss = 0.0
            for _c in cases:
                _enc = tokenizer(_c["query"], _c["document"],
                                 padding="max_length", truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt")
                _ev = {k: v.to(DEVICE) for k, v in _enc.items()}
                _logit = model(**_ev).logits.view(-1)
                _pred = 1 if float(_logit[0]) > 0 else 0
                if _pred == _c["label"]:
                    _correct += 1
                _tlbl = torch.tensor([float(_c["label"])], device=DEVICE)
                _eval_loss += float(loss_fn(_logit, _tlbl).item())
            final_acc = _correct
            log_eval(epoch + 1, final_acc, n_cases, _eval_loss / max(1, n_cases))
        model.train()

    # --- stop GPU monitor + capture train time (standard AI training log) ---
    train_time = time.perf_counter() - train_start
    monitor.stop()
    gpu_stats = monitor.stats()

    # Save fine-tuned model + tokenizer to ./finetuned/ (local, no network).
    os.makedirs(FINETUNED_DIR, exist_ok=True)
    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    print(f"saved fine-tuned model to {FINETUNED_DIR}")

    mem("after_train")
    train_summary("bge-reranker-large", total_steps, train_time, flops_per_step,
                  gpu_stats, final_acc, n_cases, EPOCHS)
    print("TRAIN_OK")


if __name__ == "__main__":
    main()
