#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference for bge-reranker-large on the 20 local test cases.

Loads ./finetuned/ if it exists (produced by train.py), otherwise falls back
to the base checkpoint. Scores each (query, document) pair with the single
relevance logit the cross-encoder emits (README: `.logits.view(-1)`), prints
the scores, asserts exactly 20, writes results.json, and prints INFER_OK.

Usage:
    CHECKPOINTS_DIR=/home/qiyiyan/songchao/checkpoints python infer.py
"""

import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "bge-reranker-large")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_DIR, "test_cases.json")
FINETUNED_DIR = os.path.join(CODE_DIR, "finetuned")
RESULTS_PATH = os.path.join(CODE_DIR, "results.json")

# PPU masquerades as CUDA; prefer the fine-tuned model if present.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_PATH = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else MODEL_PATH

MAX_LENGTH = 512


def main():
    print(f"LOAD_PATH = {LOAD_PATH}")
    print(f"DEVICE    = {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOAD_PATH)
    model.to(DEVICE)
    model.eval()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"

    # Build (query, document) pairs exactly as the README inference snippet:
    #   inputs = tokenizer(pairs, padding=True, truncation=True,
    #                      return_tensors='pt', max_length=512)
    pairs = [[c["query"], c["document"]] for c in cases]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        scores = model(**inputs).logits.view(-1).float().cpu().tolist()

    # SPEC: assert we get exactly 20 scores.
    assert len(scores) == len(cases) == 20, (
        f"expected 20 scores, got {len(scores)}"
    )

    print(f"\n{'idx':>3}  {'label':>5}  {'score':>10}   query")
    print("-" * 60)
    for i, (c, s) in enumerate(zip(cases, scores)):
        q_preview = c["query"][:30]
        print(f"{i:>3}  {c['label']:>5}  {s:>10.4f}   {q_preview}")

    results = [
        {
            "index": i,
            "query": c["query"],
            "document": c["document"],
            "label": c["label"],
            "score": float(s),
        }
        for i, (c, s) in enumerate(zip(cases, scores))
    ]
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {RESULTS_PATH}")

    print("INFER_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
