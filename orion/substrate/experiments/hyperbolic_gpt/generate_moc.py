from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.data import load_gpt2_tokenizer
from orion.substrate.experiments.hyperbolic_gpt.moc_patch import apply_moc
from orion.substrate.experiments.hyperbolic_gpt.model import HyperbolicGPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a MoC HyperbolicGPT checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = HyperbolicGPTConfig.load_json(ckpt_dir / "config.json")
    model = HyperbolicGPT(cfg).to(device)
    apply_moc(model, cfg)
    try:
        payload = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_dir / "model.pt", map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    tok = load_gpt2_tokenizer()
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
