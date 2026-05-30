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
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import HyperbolicGPTV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from HyperbolicGPT v2 checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config = HyperbolicGPTConfig.load_json(ckpt_dir / "config.json")
    model = HyperbolicGPTV2(config).to(device)
    ckpt_path = ckpt_dir / "model.pt"
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    tok = load_gpt2_tokenizer()
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
