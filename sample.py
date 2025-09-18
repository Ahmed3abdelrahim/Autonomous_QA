"""
Sample from the trained autonet-char model (char-level).
"""

import os
import pickle
import argparse
from contextlib import nullcontext

import torch
from model import GPTConfig, GPT


# ----------------------------------------------------------------------------- #
# Helper: safe decode trimming
# ----------------------------------------------------------------------------- #
def trim_output(generated: str, prompt: str):
    """Trim generated text after first <END> or next Q:"""
    body = generated[len(prompt):] if generated.startswith(prompt) else generated
    # Stop markers
    for stop in ["<END>", "\n\nQ:", "\nQ:"]:
        idx = body.find(stop)
        if idx != -1:
            body = body[:idx].rstrip()
            break
    return (prompt + body).rstrip()


# ----------------------------------------------------------------------------- #
# Load meta (char-level vocab)
# ----------------------------------------------------------------------------- #
def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if meta.get("meta_vocab_type", "") != "char":
        raise RuntimeError("Expected char-level vocab in meta.pkl")
    stoi = meta["stoi"]
    itos = meta["itos"]
    encode = lambda s: [stoi[ch] if ch in stoi else stoi.get("\n", 0) for ch in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode


# ----------------------------------------------------------------------------- #
# Load model
# ----------------------------------------------------------------------------- #
def load_model(out_dir: str, device: str, dtype: str, compile_flag: bool):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    if compile_flag:
        model = torch.compile(model)
    return model


# ----------------------------------------------------------------------------- #
# Args
# ----------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Sample from autonet-char")
    p.add_argument("--out_dir", default="out-autonet-char", type=str)
    p.add_argument("--data_dir", default="data/autonet_qa", type=str)
    p.add_argument("--start", default="Q: What is an autonomous network in simple terms?\nA:", type=str)
    p.add_argument("--num_samples", default=3, type=int)
    p.add_argument("--max_new_tokens", default=200, type=int)
    p.add_argument("--temperature", default=0.8, type=float)
    p.add_argument("--top_k", default=200, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    p.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # dtype
    if args.dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        else:
            dtype = "float16" if "cuda" in args.device else "float32"
    else:
        dtype = args.dtype
    ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    # torch settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1337)
    if "cuda" in args.device:
        torch.cuda.manual_seed(1337)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # load vocab + model
    encode, decode = load_meta(args.data_dir)
    model = load_model(args.out_dir, args.device, dtype, args.compile)

    # prompt
    start_text = args.start
    if start_text.startswith("FILE:"):
        with open(start_text[5:], "r", encoding="utf-8") as f:
            start_text = f.read()
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    # generate
    with torch.no_grad():
        with ctx:
            for i in range(args.num_samples):
                y = model.generate(
                    idx=x,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if args.top_k > 0 else None
                )
                raw = decode(y[0].tolist())
                cleaned = trim_output(raw, start_text)
                print(cleaned)
                print("-" * 79)


if __name__ == "__main__":
    main()
