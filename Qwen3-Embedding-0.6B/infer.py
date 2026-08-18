#!/usr/bin/env python3
"""Inference for Qwen3-Embedding-0.6B on the 20 test triplets.

Loads ./finetuned/ if present, otherwise the base checkpoint. Encodes queries
(with the README query instruction) and documents, computes the cosine
similarity matrix, asserts the embedding dimension (1024) and the number of
cases, writes results.json, prints INFER_OK and exits 0.

References: model README "Transformers Usage" + "Sentence Transformers Usage".
"""
import os
import json
import sys

import torch
from sentence_transformers import SentenceTransformer

CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "Qwen3-Embedding-0.6B")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(CODE_DIR, "finetuned")
CASES_PATH = os.path.join(CODE_DIR, "test_cases.json")
RESULTS_PATH = os.path.join(CODE_DIR, "results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_DIM = 1024  # README: Qwen3-Embedding-0.6B embedding dimension.

QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)


def main():
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"

    # Prefer the fine-tuned checkpoint; fall back to the base model.
    load_path = (
        FINETUNED_DIR
        if os.path.exists(os.path.join(FINETUNED_DIR, "config.json"))
        else MODEL_PATH
    )
    print(f"device={device}")
    print(f"load_path={load_path}")

    model = SentenceTransformer(
        load_path,
        device=device,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = 512
    model.eval()

    queries = [c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    negatives = [c["negative"] for c in cases]
    docs = positives + negatives  # 40 documents

    # Queries carry the README instruction; documents are encoded plain.
    # The pipeline ends with 2_Normalize, so embeddings are unit-norm and
    # the dot product equals cosine similarity.
    q_emb = model.encode(
        [QUERY_PROMPT + q for q in queries],
        convert_to_tensor=True,
        batch_size=8,
        show_progress_bar=False,
    ).to(device)
    d_emb = model.encode(
        docs,
        convert_to_tensor=True,
        batch_size=8,
        show_progress_bar=False,
    ).to(device)

    # Assertions required by the SPEC.
    assert q_emb.shape[-1] == EXPECTED_DIM, (
        f"query embedding dim {q_emb.shape[-1]} != expected {EXPECTED_DIM}"
    )
    assert d_emb.shape[-1] == EXPECTED_DIM, (
        f"document embedding dim {d_emb.shape[-1]} != expected {EXPECTED_DIM}"
    )
    assert q_emb.shape[0] == 20, f"expected 20 query embeddings, got {q_emb.shape[0]}"
    assert d_emb.shape[0] == 40, f"expected 40 document embeddings, got {d_emb.shape[0]}"

    # Cosine similarity matrix: queries (20) x documents (40).
    sim = q_emb @ d_emb.t()  # [20, 40]
    assert sim.shape[0] == 20 and sim.shape[1] == 40

    print("similarity matrix (queries x [positives+negatives]):")
    print(sim.detach().cpu())

    per_case = []
    for i, c in enumerate(cases):
        per_case.append({
            "query": c["query"],
            "positive": c["positive"],
            "negative": c["negative"],
            "sim_positive": float(sim[i, i].item()),       # positive at column i
            "sim_negative": float(sim[i, 20 + i].item()),   # negative at column 20+i
        })

    results = {
        "model": "Qwen3-Embedding-0.6B",
        "load_path": load_path,
        "device": device,
        "embedding_dim": int(q_emb.shape[-1]),
        "num_cases": len(cases),
        "similarity_matrix_shape": [int(sim.shape[0]), int(sim.shape[1])],
        "similarity_matrix": sim.detach().cpu().tolist(),
        "per_case": per_case,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"wrote {RESULTS_PATH}")
    print("INFER_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
