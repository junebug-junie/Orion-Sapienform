from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.model import HyperbolicGPT


def _pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def main() -> None:
    device = _pick_device()
    cfg = HyperbolicGPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=64,
    )
    model = HyperbolicGPT(cfg).to(device)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    logits, loss = model(x, y)
    assert torch.isfinite(logits).all(), "non-finite logits"
    assert loss is not None and torch.isfinite(loss), "non-finite loss"
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"
    model.eval()
    gen = model.generate(x[:1, :8], max_new_tokens=5, temperature=1.0)
    assert torch.isfinite(gen.float()).all(), "non-finite generation"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
