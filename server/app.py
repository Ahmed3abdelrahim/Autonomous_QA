import os, pickle
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import GPT, GPTConfig
from server.generator import TextGenerator

# Directories
MODEL_DIR = os.environ.get("MODEL_DIR", "out-autonet-char")
DATA_DIR  = os.environ.get("DATA_DIR", "data/autonet_qa")
DEVICE    = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.bfloat16 if (DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported()) else (torch.float16 if DEVICE.startswith("cuda") else torch.float32)

# Load vocab
with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] if c in stoi else stoi.get("\n", 0) for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# Load model checkpoint
ckpt = torch.load(os.path.join(MODEL_DIR, "ckpt.pt"), map_location=DEVICE)
gptconf = GPTConfig(**ckpt["model_args"])
model = GPT(gptconf)
state_dict = ckpt["model"]
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=True)

# Initialize generator
gen = TextGenerator(model, encode, decode, device=DEVICE, dtype=DTYPE)
app = FastAPI(title="Autonomous Networks QA")

class Input(BaseModel):
    question: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 100

def trim(text: str):
    for stop in ["<END>", "\n\nQ:", "\nQ:"]:
        idx = text.find(stop)
        if idx != -1:
            return text[:idx].rstrip()
    return text

@app.post("/generate")
def generate(inp: Input):
    prompt = f"Q: {inp.question.strip()}\nA:"
    out = gen.generate(
        prompt,
        max_new_tokens=inp.max_new_tokens,
        temperature=inp.temperature,
        top_k=inp.top_k
    )
    body = out[len(prompt):] if out.startswith(prompt) else out
    return {"answer": trim(body)}
