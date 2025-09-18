
# Build a char vocabulary from train+val, then write train.bin / val.bin + meta.pkl
import os, pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent
train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
val_text   = (DATA_DIR / "val.txt").read_text(encoding="utf-8")

# 1) build charset
text_all = train_text + val_text
chars = sorted(list(set(text_all)))
vocab_size = len(chars)
print(f"Vocab size (chars): {vocab_size}")

# 2) mappings
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

def encode(s: str):
    return [stoi[c] for c in s]

# 3) encode to uint16 arrays (char-level fits easily)
train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids   = np.array(encode(val_text),   dtype=np.uint16)

# 4) write .bin
train_ids.tofile(DATA_DIR / "train.bin")
val_ids.tofile(DATA_DIR / "val.bin")
print("Wrote train.bin and val.bin")

# 5) write meta
meta = dict(
    vocab_size=vocab_size,
    stoi=stoi,
    itos=itos,
    # nanoGPT looks for this flag to know it's char-level
    meta_vocab_type="char"
)
with open(DATA_DIR / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print("Wrote meta.pkl")
