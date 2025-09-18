import json, random, os, io
from pathlib import Path

SRC = Path("train.jsonl")
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with io.open(SRC, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

random.seed(1337)
random.shuffle(rows)

def format_pair(q, a):
    # Keep it simple and consistent; this is your LM’s “prompt pattern”
    return f"Q: {q.strip()}\nA: {a.strip()}\n\n"

text = "".join(format_pair(r["question"], r["answer"]) for r in rows)

# 95/5 split (small corpus)
n = int(0.95 * len(rows))
train_text = "".join(format_pair(r["question"], r["answer"]) for r in rows[:n])
val_text   = "".join(format_pair(r["question"], r["answer"]) for r in rows[n:])

# Write plain text files
(OUT_DIR / "train.txt").write_text(train_text, encoding="utf-8")
(OUT_DIR / "val.txt").write_text(val_text, encoding="utf-8")

print("Wrote:", OUT_DIR / "train.txt", OUT_DIR / "val.txt")
