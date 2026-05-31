from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import summarize_manifold_batch
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import HyperbolicGPTV2


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HyperbolicGPTConfig(vocab_size=1000, block_size=64, n_layer=2, n_head=4, n_embd=128, semantic_adapter_rank=16, margin_gap_loss_weight=0.001)
    model = HyperbolicGPTV2(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    logits, loss, aux = model(x, y, return_aux=True)
    assert loss is not None and torch.isfinite(loss)
    assert torch.isfinite(logits).all()
    metrics = summarize_manifold_batch(logits=logits, hidden=aux["hidden"], tangent=aux.get("tangent"), loss_ce=aux.get("loss_ce", loss.detach()), gap_epsilon=cfg.margin_gap_epsilon)
    loss.backward()
    print("SMOKE TEST V2 PASSED")
    print(metrics)


if __name__ == "__main__":
    main()
