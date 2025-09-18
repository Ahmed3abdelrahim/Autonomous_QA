import torch
from contextlib import nullcontext

class TextGenerator:
    def __init__(self, model, encode, decode, device="cuda", dtype=torch.float16):
        self.model = model.eval().to(device)
        self.encode = encode
        self.decode = decode
        self.device = device
        self.ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=200, temperature=0.7, top_k=100):
        x = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)[None, ...]
        with self.ctx:
            y = self.model.generate(
                idx=x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        return self.decode(y[0].tolist())
