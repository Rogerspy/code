#!/usr/bin/env python3
"""Inference for Qwen3-Reranker-0.6B on 20 (query, document, label) cases.

Loads ./finetuned/ if it exists, otherwise the base checkpoint, then scores each
(query, document) pair using the README's exact prompt scaffolding:

  AutoModelForCausalLM -> logits[:, -1, :] -> take yes/no logits ->
  stack [no, yes] -> log_softmax -> exp -> P(yes) as the relevance score.

Runs on cuda (PPU masquerades as CUDA). Asserts 20 scores are produced,
writes results.json, prints INFER_OK and exits 0.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths / device
# ---------------------------------------------------------------------------
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "Qwen3-Reranker-0.6B")

HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(HERE, "finetuned")
CASES_PATH = os.path.join(HERE, "test_cases.json")
RESULTS_PATH = os.path.join(HERE, "results.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# README-exact prompt scaffolding (see train.py for the same construction).
# The suffix ends with an empty reasoning-block scaffold; angle-bracket
# fragments are assembled piecewise so nothing in the source is mistaken for a
# template tag.
# ---------------------------------------------------------------------------
_IME = "<" + "|im_end|" + ">"            # im-end marker
_IMS = "<" + "|im_start|" + ">"          # im-start marker
_THINK_OPEN = "<" + "think" + ">"        # reasoning-block open marker
_THINK_CLOSE = "<" + "/think" + ">"      # reasoning-block close marker

PREFIX = (
    _IMS + "system\n"
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    + _IME + "\n" + _IMS + "user\n"
)
SUFFIX = _IME + "\n" + _IMS + "assistant\n" + _THINK_OPEN + "\n\n" + _THINK_CLOSE + "\n\n"

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MAX_LENGTH = 8192  # inference cap (actual sequences are far shorter)

# Verified token ids from the checkpoint's 1_LogitScore/config.json.
_FALLBACK_TRUE_ID = 9693
_FALLBACK_FALSE_ID = 2152


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = INSTRUCTION
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def get_true_false_ids(tokenizer, search_paths):
    """Return (true_id, false_id) for tokens 'yes' and 'no'.

    Search the provided dirs for a LogitScore/logit config first, then fall
    back to convert_tokens_to_ids, then to the verified constants.
    """
    for base in search_paths:
        for rel in (os.path.join("1_LogitScore", "config.json"), "logit_config.json"):
            fp = os.path.join(base, rel)
            if os.path.exists(fp):
                try:
                    cfg = json.load(open(fp, encoding="utf-8"))
                    if "true_token_id" in cfg and "false_token_id" in cfg:
                        return int(cfg["true_token_id"]), int(cfg["false_token_id"])
                except Exception:
                    pass
    t = tokenizer.convert_tokens_to_ids("yes")
    f = tokenizer.convert_tokens_to_ids("no")
    unk = tokenizer.unk_token_id
    if not isinstance(t, int) or t is None or t == unk:
        t = _FALLBACK_TRUE_ID
    if not isinstance(f, int) or f is None or f == unk:
        f = _FALLBACK_FALSE_ID
    return t, f


def build_inputs(tokenizer, texts, prefix_tokens, suffix_tokens):
    enc = tokenizer(
        texts,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i in range(len(enc["input_ids"])):
        enc["input_ids"][i] = prefix_tokens + enc["input_ids"][i] + suffix_tokens
    padded = tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    return padded


def main():
    print("device:", device)

    # Prefer the fine-tuned checkpoint if it has a model config.
    load_path = FINETUNED_DIR if os.path.exists(os.path.join(FINETUNED_DIR, "config.json")) else MODEL_PATH
    print("Loading model from:", load_path)

    tokenizer = AutoTokenizer.from_pretrained(load_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_true_id, token_false_id = get_true_false_ids(tokenizer, [load_path, MODEL_PATH])
    print(f"token_true_id(yes)={token_true_id} token_false_id(no)={token_false_id}")
    for name, idx in (("yes", token_true_id), ("no", token_false_id)):
        print(f"  id {idx} -> {tokenizer.convert_ids_to_tokens(idx)!r} (expected '{name}')")

    model = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    model.config.use_cache = False

    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    cases = json.load(open(CASES_PATH, encoding="utf-8"))
    pairs = [format_instruction(INSTRUCTION, c["query"], c["document"]) for c in cases]
    print(f"loaded {len(pairs)} pairs to score")

    scores = []
    batch_size = 8
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = build_inputs(tokenizer, batch, prefix_tokens, suffix_tokens)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # [B, vocab] (last token, left-padded)

            true_vector = logits[:, token_true_id]    # [B]
            false_vector = logits[:, token_false_id]  # [B]
            batch_scores = torch.stack([false_vector, true_vector], dim=1).float()  # [B, 2]
            batch_scores = F.log_softmax(batch_scores, dim=-1)
            # P(yes) = P(relevant); index 1 corresponds to the "yes"/true column.
            rel = batch_scores[:, 1].exp().tolist()
            scores.extend(rel)

    print("scores:")
    for i, c in enumerate(cases):
        print(f"  [{i:2d}] label={c['label']} score={scores[i]:.6f}  q={c['query'][:40]!r}")

    assert isinstance(scores, list) and len(scores) == 20, \
        f"expected 20 scores, got {len(scores) if isinstance(scores, list) else type(scores)}"

    results = [
        {
            "query": c["query"],
            "document": c["document"],
            "label": c["label"],
            "score": float(scores[i]),
        }
        for i, c in enumerate(cases)
    ]
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("Wrote results to", RESULTS_PATH)

    print("INFER_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
