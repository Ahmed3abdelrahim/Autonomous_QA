# evaluate.py
# Toggleable evaluation for your tiny-from-scratch char-level GPT.

import json, time, re, math, statistics as st
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import pickle
import torch.nn.functional as F

from model import GPT, GPTConfig

# =========================
# Metric toggles (flip here)
# =========================
METRICS = {
    "perplexity": True,     # intrinsic LM quality on val.txt
    "f1_em": True,          # F1 and Exact Match vs gold
    "keyword": True,        # keyword hit rate if keywords are provided
    "refusal": True,        # refusal accuracy & hallucination on gold == ""
    "latency": True,        # p50/p95 latency
    "length": False,        # avg answer length (words)
    "format": False,        # basic format check pass rate
}

# Paths
DATA_DIR = Path("data/autonet_qa")
MODEL_DIR = Path("out-autonet-char")
EVAL_PATH = Path("data/eval.jsonl")

# -----------------------------
# IO & model loading
# -----------------------------
def load_char_io():
    meta = pickle.load(open(DATA_DIR / "meta.pkl", "rb"))
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] if c in stoi else stoi.get("\n", 0) for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    return encode, decode

def load_model(device="cuda"):
    ckpt = torch.load(MODEL_DIR / "ckpt.pt", map_location=device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    # handle torch.compile prefixing
    unp = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(unp):
            sd[k[len(unp):]] = sd.pop(k)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    block_size = getattr(gptconf, "block_size", 256)
    return model, block_size

# -----------------------------
# Generation helpers
# -----------------------------
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
    return trim_generated(txt, prompt)

# -----------------------------
# Text metrics
# -----------------------------
def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()

def f1_score(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    inter = len(set(p) & set(g))
    prec = inter / len(p)
    rec  = inter / len(g)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def keyword_hits(pred: str, kws: List[str]) -> Tuple[int, int]:
    p = normalize(pred)
    return sum(1 for k in kws if normalize(k) in p), len(kws)

_REFUSAL_PATTERNS = [
    "i can’t", "i can't", "i cannot", "cannot comply", "i won’t", "i won't",
    "i do not have enough information", "i don't have enough information",
    "i don’t know", "i don't know", "i’m unable", "i am unable",
    "i should not", "i will not", "cannot provide", "not appropriate",
    "confidential", "private information", "i must refuse", "i refuse",
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _REFUSAL_PATTERNS) or len(normalize(t)) <= 2

# -----------------------------
# Perplexity (robust)
# -----------------------------
@torch.no_grad()
def perplexity_on_val(model, encode, device="cuda", block_size=256, stride=None) -> float:
    """
    Computes cross-entropy on val.txt using the model's own loss with targets,
    avoiding shape mismatches. Returns exp(mean_loss).
    """
    val_text = (DATA_DIR / "val.txt").read_text(encoding="utf-8")
    ids = torch.tensor(encode(val_text), dtype=torch.long, device=device)
    n = ids.numel()
    if stride is None:
        stride = max(1, block_size // 2)
    if n < block_size + 1:
        return float("inf")

    losses = []
    for start in range(0, n - block_size - 1, stride):
        x = ids[start : start + block_size].unsqueeze(0)         # [1, T]
        y = ids[start + 1 : start + 1 + block_size].unsqueeze(0) # [1, T]
        try:
            # Preferred: model returns (logits, loss) when targets provided
            _, loss = model(x, y)
        except TypeError:
            # Fallback: manual CE
            logits, _ = model(x)  # [B, T, V]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   y.view(-1), reduction="mean")
        losses.append(float(loss.item()))

    if not losses:
        return float("inf")
    return float(math.exp(sum(losses) / len(losses)))

# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode, decode = load_char_io()
    model, block_size = load_model(device=device)

    report: Dict[str, Any] = {}

    # 1) Perplexity
    if METRICS["perplexity"]:
        report["perplexity_val_txt"] = perplexity_on_val(
            model, encode, device=device, block_size=block_size
        )
    else:
        report["perplexity_val_txt"] = None

    # 2) Load eval items
    rows = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines()]
    report["n_items"] = len(rows)

    # 3) Iterate & collect
    f1s, ems, khits, lat = [], [], [], []
    should_refuse_total = 0
    correct_refusals = 0
    hallucinations = 0
    lens_words = []
    format_ok = 0

    for r in rows:
        q    = r["question"]
        gold = r.get("gold", "")
        kws  = r.get("keywords", [])

        t0 = time.time()
        ans = generate(model, encode, decode, q, device=device,
                       max_new_tokens=220, temperature=0.2, top_k=None)
        if METRICS["latency"]:
            lat.append(time.time() - t0)

        if METRICS["length"]:
            lens_words.append(len(normalize(ans).split()))
        if METRICS["format"]:
            format_ok += int(bool(ans) and (" Q: " not in ans[:3]))

        if gold == "":
            if METRICS["refusal"]:
                should_refuse_total += 1
                if is_refusal(ans):
                    correct_refusals += 1
                else:
                    if len(normalize(ans)) > 0:
                        hallucinations += 1
        else:
            if METRICS["f1_em"]:
                ems.append(int(normalize(ans) == normalize(gold)))
                f1s.append(f1_score(ans, gold))
            if METRICS["keyword"] and kws:
                h, n = keyword_hits(ans, kws)
                khits.append(h / n if n else 1.0)

    # 4) Aggregate
    # Core metrics
    report["F1_avg"] = (sum(f1s) / len(f1s)) if (METRICS["f1_em"] and f1s) else None
    report["EM"] = (sum(ems) / len(ems)) if (METRICS["f1_em"] and ems) else None

    if METRICS["keyword"]:
        report["KW_Hit_avg"] = (sum(khits) / len(khits)) if khits else None
    else:
        report["KW_Hit_avg"] = None

    if METRICS["latency"]:
        report["latency_p50_s"] = float(st.median(lat)) if lat else None
        report["latency_p95_s"] = float(sorted(lat)[int(0.95 * len(lat)) - 1]) if len(lat) >= 2 else None
    else:
        report["latency_p50_s"] = report["latency_p95_s"] = None

    if METRICS["length"]:
        report["ans_length_words_avg"] = (sum(lens_words) / len(lens_words)) if lens_words else None
    else:
        report["ans_length_words_avg"] = None

    if METRICS["format"]:
        report["format_pass_rate"] = (format_ok / len(rows)) if rows else None
    else:
        report["format_pass_rate"] = None

    if METRICS["refusal"]:
        report["refusal_expected_total"] = should_refuse_total
        report["refusal_correct"] = correct_refusals
        report["refusal_accuracy"] = (correct_refusals / should_refuse_total) if should_refuse_total else None
        report["hallucination_count"] = hallucinations
        report["hallucination_rate_on_should_refuse"] = (hallucinations / should_refuse_total) if should_refuse_total else None
    else:
        report["refusal_expected_total"] = report["refusal_correct"] = None
        report["refusal_accuracy"] = report["hallucination_count"] = report["hallucination_rate_on_should_refuse"] = None

    report["notes"] = {
        "decoding": {"temperature": 0.2, "top_k": None, "max_new_tokens": 220},
        "block_size": block_size,
        "metric_toggles": METRICS,
    }

    Path("eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
