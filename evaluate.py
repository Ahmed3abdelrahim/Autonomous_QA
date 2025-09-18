# evaluate.py
# Comprehensive evaluation for your tiny-from-scratch char-level GPT.
# Metrics:
# - Perplexity on val.txt
# - Exact Match (EM)
# - F1 vs. gold
# - Keyword hit rate
# - Refusal compliance on gold=="" (should refuse)
# - Hallucination rate (non-empty answer where gold=="")
# - Latency p50/p95
# - Answer length stats
# - Basic format checks (starts after "A:", trims before next "Q:"/"<END>")

import json, time, re, math, statistics as st
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import torch
import pickle

from model import GPT, GPTConfig

DATA_DIR = Path("data/autonet_qa")
MODEL_DIR = Path("out-autonet-char")
EVAL_PATH = Path("data/eval.jsonl")

# -----------------------------
# Utilities
# -----------------------------

def load_char_io():
    meta = pickle.load(open(DATA_DIR/"meta.pkl","rb"))
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] if c in stoi else stoi.get("\n", 0) for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    return encode, decode

def load_model(device="cuda"):
    ckpt = torch.load(MODEL_DIR/"ckpt.pt", map_location=device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    # handle torch.compile prefix
    unp = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(unp):
            sd[k[len(unp):]] = sd.pop(k)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    # pull useful params for eval
    block_size = getattr(gptconf, "block_size", 256)
    vocab_size = getattr(gptconf, "vocab_size", None)
    return model, block_size, vocab_size

def trim_generated(full_text: str, prompt: str) -> str:
    body = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    for stop in ["<END>", "\n\nQ:", "\nQ:"]:
        i = body.find(stop)
        if i != -1:
            body = body[:i].rstrip()
            break
    return body.strip()

def generate(model, encode, decode, q: str, device="cuda",
             max_new_tokens=180, temperature=0.2, top_k=None) -> str:
    prompt = f"Q: {q.strip()}\nA:"
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(idx=x, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_k=top_k)
    txt = decode(y[0].tolist())
    ans = trim_generated(txt, prompt)
    return ans

def normalize(s: str) -> str:
    return re.sub(r"\W+"," ", s.lower()).strip()

def f1_score(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    inter = len(set(p) & set(g))
    prec = inter / len(p)
    rec  = inter / len(g)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def keyword_hits(pred: str, kws: List[str]) -> Tuple[int,int]:
    p = normalize(pred)
    return sum(1 for k in kws if normalize(k) in p), len(kws)

_REFUSAL_PATTERNS = [
    "i can’t", "i can't", "i cannot", "cannot comply", "i won’t", "i won't",
    "i do not have enough information", "i don't have enough information",
    "i don’t know", "i don't know", "i’m unable", "i am unable",
    "i should not", "i will not", "cannot provide", "not appropriate",
    "confidential", "private information", "i must refuse", "i refuse"
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(pat in t for pat in _REFUSAL_PATTERNS) or len(normalize(t)) <= 2

# -----------------------------
# Perplexity on val.txt
# -----------------------------

@torch.no_grad()
def perplexity_on_val(model, encode, device="cuda", block_size=256, stride=None) -> float:
    """
    Computes cross-entropy on val.txt with sliding windows and uses the model's own
    (logits, loss) output when targets are provided, avoiding shape mismatches.
    """
    import math, torch.nn.functional as F
    val_text = (DATA_DIR / "val.txt").read_text(encoding="utf-8")
    ids = torch.tensor(encode(val_text), dtype=torch.long, device=device)
    n = ids.numel()
    if stride is None:
        stride = max(1, block_size // 2)

    if n < block_size + 1:
        # not enough context to compute; return inf
        return float("inf")

    losses = []
    for start in range(0, n - block_size - 1, stride):
        x = ids[start : start + block_size].unsqueeze(0)                # [1, T]
        y = ids[start + 1 : start + 1 + block_size].unsqueeze(0)        # [1, T]

        # Prefer using the model's native loss (avoids shape pitfalls)
        try:
            logits, loss = model(x, y)
        except TypeError:
            # Fallback: older signatures; compute loss manually
            logits, _ = model(x)  # [B, T, V] or [B, V]
            if logits.dim() == 3:
                # [B, T, V] vs [B, T]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="mean"
                )
            else:
                # [B, V] vs [B, T] -> use last target token
                loss = F.cross_entropy(logits, y[:, -1], reduction="mean")

        losses.append(float(loss.item()))

    if not losses:
        return float("inf")
    return float(math.exp(sum(losses) / len(losses)))


# -----------------------------
# Main Evaluation
# -----------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode, decode = load_char_io()
    model, block_size, _ = load_model(device=device)

    # 1) Perplexity
    ppl = perplexity_on_val(model, encode, device=device, block_size=block_size, stride=block_size//2)

    # 2) Load eval set
    rows = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines()]

    # 3) Score generated answers
    f1s, ems, khits, lat = [], [], [], []
    should_refuse_total = 0
    correct_refusals = 0
    hallucinations = 0  # non-empty answer when gold == ""

    lens_words = []
    format_ok = 0

    for r in rows:
        q    = r["question"]
        gold = r.get("gold", "")
        kws  = r.get("keywords", [])

        t0 = time.time()
        ans = generate(model, encode, decode, q, device=device,
                       max_new_tokens=220, temperature=0.2, top_k=None)
        lat.append(time.time() - t0)

        # Length and basic formatting
        lens_words.append(len(normalize(ans).split()))
        # simple format check: not empty, no trailing next question signals
        is_ok = bool(ans) and (" Q: " not in ans[:3])  # crude: starts with content
        format_ok += int(is_ok)

        if gold == "":  # expected refusal
            should_refuse_total += 1
            if is_refusal(ans):
                correct_refusals += 1
            else:
                if len(normalize(ans)) > 0:
                    hallucinations += 1
        else:
            # Standard QA metrics
            ems.append(int(normalize(ans) == normalize(gold)))
            f1s.append(f1_score(ans, gold))
            if kws:
                h, n = keyword_hits(ans, kws)
                khits.append(h / n if n else 1.0)

    # Aggregates
    report: Dict[str, Any] = {
        "n_items": len(rows),
        "perplexity_val_txt": ppl,
        "F1_avg": (sum(f1s) / len(f1s)) if f1s else None,
        "EM": (sum(ems) / len(ems)) if ems else None,
        "KW_Hit_avg": (sum(khits) / len(khits)) if khits else None,
        "latency_p50_s": float(st.median(lat)) if lat else None,
        "latency_p95_s": float(sorted(lat)[int(0.95 * len(lat)) - 1]) if len(lat) >= 2 else None,
        "ans_length_words_avg": (sum(lens_words) / len(lens_words)) if lens_words else None,
        "format_pass_rate": format_ok / len(rows) if rows else None,
        "refusal_expected_total": should_refuse_total,
        "refusal_correct": correct_refusals,
        "refusal_accuracy": (correct_refusals / should_refuse_total) if should_refuse_total else None,
        "hallucination_count": hallucinations,
        "hallucination_rate_on_should_refuse": (hallucinations / should_refuse_total) if should_refuse_total else None,
        "notes": {
            "decoding": {"temperature": 0.2, "top_k": None, "max_new_tokens": 220},
            "block_size": block_size,
        }
    }

    Path("eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
