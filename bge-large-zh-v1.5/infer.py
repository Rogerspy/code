#!/usr/bin/env python3
"""Inference for BAAI/bge-large-zh-v1.5 (Chinese embedding model).

Loads ./finetuned/ if it exists (else the base checkpoint), encodes the 20 test cases
(queries prefixed with the bge retrieval instruction, passages unmodified), computes
cosine similarity, asserts the embedding dimension matches the README (1024), writes
results.json, prints INFER_OK and exits 0. No exceptions are raised on the happy path.
"""
import os
import sys
import json

import torch

CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "bge-large-zh-v1.5")
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CASES = os.path.join(HERE, "test_cases.json")
FINETUNED = os.path.join(HERE, "finetuned")
RESULTS = os.path.join(HERE, "results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_DIM = 1024  # README C-MTEB: bge-large-zh-v1.5 embedding dimension = 1024
INSTRUCTION = "为这个句子生成表示以用于检索相关文章："  # bge-large-zh retrieval query instruction


def load_cases():
    with open(TEST_CASES, "r", encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 20, f"expected 20 cases, got {len(cases)}"
    return cases


def resolve_model_path():
    if os.path.isdir(FINETUNED) and os.path.exists(os.path.join(FINETUNED, "modules.json")):
        return FINETUNED, "fine-tuned"
    return MODEL_PATH, "base"


def main():
    from sentence_transformers import SentenceTransformer

    load_path, kind = resolve_model_path()
    print(f"Loading {kind} model from: {load_path} (device={device})")
    model = SentenceTransformer(load_path, device=device)
    model.max_seq_length = 512

    cases = load_cases()
    queries = [INSTRUCTION + c["query"] for c in cases]
    positives = [c["positive"] for c in cases]
    negatives = [c["negative"] for c in cases]

    q_emb = model.encode(queries, normalize_embeddings=True, convert_to_tensor=True)
    p_emb = model.encode(positives, normalize_embeddings=True, convert_to_tensor=True)
    n_emb = model.encode(negatives, normalize_embeddings=True, convert_to_tensor=True)

    dim = int(q_emb.shape[1])
    print(f"Embedding shape: queries={tuple(q_emb.shape)} dim={dim}")
    assert dim == EXPECTED_DIM, f"embedding dim {dim} != expected {EXPECTED_DIM}"
    print(f"Embedding dim assertion OK ({dim} == {EXPECTED_DIM})")

    # cosine similarity (embeddings already L2-normalized via normalize_embeddings=True)
    qp_sim = q_emb @ p_emb.T
    qn_sim = q_emb @ n_emb.T

    diag_qp = qp_sim.diag().cpu().tolist()
    diag_qn = qn_sim.diag().cpu().tolist()

    print("\nidx  q-pos sim | q-neg sim | query")
    for i, c in enumerate(cases):
        print(f"{i:>3}  {diag_qp[i]:.4f}   | {diag_qn[i]:.4f}   | {c['query'][:30]}")

    correct = int(sum(1 for i in range(len(cases)) if diag_qp[i] > diag_qn[i]))
    print(f"\nRetrieval (q-pos > q-neg): {correct}/{len(cases)}")

    print("\nQuery-Positive cosine similarity matrix (rows=query, cols=positive):")
    mat = qp_sim.cpu().tolist()
    for row in mat:
        print(" ".join(f"{v:5.2f}" for v in row))

    results = {
        "model": "bge-large-zh-v1.5",
        "kind": kind,
        "load_path": load_path,
        "device": device,
        "embedding_dim": dim,
        "num_cases": len(cases),
        "query_positive_sim": diag_qp,
        "query_negative_sim": diag_qn,
        "retrieval_correct": correct,
        "similarity_matrix_qp": mat,
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {RESULTS}")

    assert len(diag_qp) == 20 and len(diag_qn) == 20, "expected 20 scores"
    print("INFER_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
