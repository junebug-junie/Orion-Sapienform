from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.data import load_gpt2_tokenizer
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import HyperbolicGPTV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate from HyperbolicGPT v2 checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--ignore_eos", action="store_true")
    return p.parse_args()


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, eos_token_id, greedy):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.argmax(logits, dim=-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_token_id is not None and bool((nxt == eos_token_id).all().item()):
            break
    return idx


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = HyperbolicGPTConfig.load_json(ckpt_dir / "config.json")
    model = HyperbolicGPTV2(cfg).to(device)
    try:
        payload = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_dir / "model.pt", map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    tok = load_gpt2_tokenizer()
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k, None if args.ignore_eos else tok.eos_token_id, args.greedy)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
