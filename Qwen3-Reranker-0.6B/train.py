#!/usr/bin/env python3
"""Tiny fine-tune for Qwen3-Reranker-0.6B on 20 (query, document, label) cases.

Implements the README's exact prompt scaffolding (prefix/suffix + token
"yes"/"no") with a manual transformers training loop:
  AutoModelForCausalLM -> logits[:, -1, :] -> stack [false, true] ->
  cross_entropy vs label (0/1) -> backward -> AdamW step.

Training runs on cuda (PPU masquerades as CUDA) and saves to ./finetuned/.
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from ppu_evidence import header, assert_on_cuda, mem
from train_logger import GPUMonitor, measure_flops, log_step, log_eval, train_summary, query_ppu_now

# ---------------------------------------------------------------------------
# Paths / device
# ---------------------------------------------------------------------------
CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
MODEL_PATH = os.path.join(CKPT_ROOT, "Qwen3-Reranker-0.6B")

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "test_cases.json")
OUT_DIR = os.path.join(HERE, "finetuned")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# README-exact prompt scaffolding.
# The suffix ends with an empty reasoning-block scaffold (open then close tag,
# each separated by newlines). The angle-bracket fragments below are assembled
# piecewise so nothing in this source file is ever misread as a template tag.
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
MAX_LENGTH = 512  # SPEC: training max_length is 512 even though model supports 32k.

# Verified token ids from the checkpoint's 1_LogitScore/config.json:
#   true_token_id (yes) = 9693, false_token_id (no) = 2152.
_FALLBACK_TRUE_ID = 9693
_FALLBACK_FALSE_ID = 2152


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = INSTRUCTION
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def get_true_false_ids(tokenizer):
    """Return (true_id, false_id) for tokens 'yes' and 'no'.

    Prefer the checkpoint's LogitScore config; fall back to
    convert_tokens_to_ids, then to the verified constants.
    """
    for fn in (os.path.join(MODEL_PATH, "1_LogitScore", "config.json"),
               os.path.join(MODEL_PATH, "logit_config.json")):
        if os.path.exists(fn):
            try:
                cfg = json.load(open(fn, encoding="utf-8"))
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
    """Tokenize a batch of formatted (instruction, query, doc) texts exactly as
    the README: longest_first truncation on the middle, then wrap with the
    prefix/suffix token blocks, then left-pad to a dense batch."""
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
    header("Qwen3-Reranker-0.6B")
    print("device:", device)
    print("MODEL_PATH:", MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_true_id, token_false_id = get_true_false_ids(tokenizer)
    print(f"token_true_id(yes)={token_true_id} token_false_id(no)={token_false_id}")

    # sanity: verify the ids really correspond to yes/no
    for name, idx in (("yes", token_true_id), ("no", token_false_id)):
        decoded = tokenizer.convert_ids_to_tokens(idx)
        print(f"  id {idx} -> {decoded!r} (expected '{name}')")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.to(device)
    model.train()
    assert_on_cuda(model, "Qwen3-Reranker-0.6B")

    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    print(f"prefix_tokens={len(prefix_tokens)} suffix_tokens={len(suffix_tokens)}")

    cases = json.load(open(CASES_PATH, encoding="utf-8"))
    data = [(format_instruction(INSTRUCTION, c["query"], c["document"]), int(c["label"]))
            for c in cases]
    print(f"loaded {len(data)} training pairs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    batch_size = 4
    epochs = 2

    n_steps = epochs * ((len(data) + batch_size - 1) // batch_size)

    # measure FLOPs of one forward+backward before starting the real loop
    probe_texts = [data[i][0] for i in range(min(batch_size, len(data)))]
    probe_labels = torch.tensor([data[i][1] for i in range(min(batch_size, len(data)))],
                                dtype=torch.long, device=device)
    probe_inputs = build_inputs(tokenizer, probe_texts, prefix_tokens, suffix_tokens)
    probe_input_ids = probe_inputs["input_ids"].to(device)
    probe_attention_mask = probe_inputs["attention_mask"].to(device)

    def step_fn():
        out = model(input_ids=probe_input_ids, attention_mask=probe_attention_mask)
        lg = out.logits[:, -1, :]
        _tv = lg[:, token_true_id]
        _fv = lg[:, token_false_id]
        _sc = torch.stack([_fv, _tv], dim=1).float()
        _lo = F.cross_entropy(_sc, probe_labels)
        _lo.backward()

    flops_per_step = measure_flops(step_fn)
    print(f"flops_per_step (measured) = {flops_per_step}")
    optimizer.zero_grad()  # clear grads left by the FLOP probe

    def evaluate():
        """Eval on the 20 (q,d,label) cases: pred=1 if logit_yes>logit_no."""
        model.eval()
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for ev_start in range(0, len(data), batch_size):
                ev_batch = data[ev_start:ev_start + batch_size]
                ev_texts = [b[0] for b in ev_batch]
                ev_labels = torch.tensor([b[1] for b in ev_batch],
                                         dtype=torch.long, device=device)
                ev_inputs = build_inputs(tokenizer, ev_texts, prefix_tokens, suffix_tokens)
                ev_ids = ev_inputs["input_ids"].to(device)
                ev_mask = ev_inputs["attention_mask"].to(device)
                ev_out = model(input_ids=ev_ids, attention_mask=ev_mask)
                ev_logits = ev_out.logits[:, -1, :]
                ev_true = ev_logits[:, token_true_id]
                ev_false = ev_logits[:, token_false_id]
                ev_scores = torch.stack([ev_false, ev_true], dim=1).float()
                ev_loss = F.cross_entropy(ev_scores, ev_labels)
                total_loss += ev_loss.item() * len(ev_batch)
                ev_preds = (ev_true > ev_false).long()
                correct += (ev_preds == ev_labels).sum().item()
        model.train()
        n_cases = len(data)
        return correct, n_cases, (total_loss / n_cases)

    # --- standard AI training logging + PPU GPU monitor ---
    monitor = GPUMonitor()
    monitor.start()
    train_start = time.time()

    step = 0
    final_acc = 0
    for epoch in range(epochs):
        # simple deterministic order (no shuffle) for reproducibility
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            texts = [b[0] for b in batch]
            labels = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)

            inputs = build_inputs(tokenizer, texts, prefix_tokens, suffix_tokens)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            step_t0 = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # [B, vocab]  (last token, left-padded)

            true_vector = logits[:, token_true_id]    # [B]
            false_vector = logits[:, token_false_id]  # [B]
            # index 0 = "no"(false), index 1 = "yes"(true) -> label 0/1 matches directly
            scores = torch.stack([false_vector, true_vector], dim=1).float()  # [B, 2]
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            print(f"epoch {epoch} step {step} batch_loss {loss.item():.6f}")
            step_dt = max(time.time() - step_t0, 1e-6)
            throughput = len(batch) / step_dt
            log_step(epoch, epochs, step, n_steps, loss.item(),
                     lr=optimizer.param_groups[0]["lr"],
                     throughput=throughput, gpu=monitor.snapshot_line())
        # epoch end: evaluate yes/no accuracy on the 20 (q,d,label) cases
        final_acc, _ev_n, _ev_mean = evaluate()
        log_eval(epoch, final_acc, _ev_n, _ev_mean)

    train_time = time.time() - train_start
    monitor.stop()
    gpu_stats = monitor.stats()

    os.makedirs(OUT_DIR, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    # persist the verified yes/no ids so inference is robust regardless of dir
    with open(os.path.join(OUT_DIR, "logit_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"true_token_id": token_true_id, "false_token_id": token_false_id}, fh)

    print("Saved fine-tuned model to", OUT_DIR)
    mem("after_train")
    train_summary("Qwen3-Reranker-0.6B", step, train_time, flops_per_step,
                  gpu_stats, final_acc, len(data), epochs)
    print("TRAIN_OK")


if __name__ == "__main__":
    main()
