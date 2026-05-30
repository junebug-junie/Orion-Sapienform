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
        semantic_adapter_rank=16,
        margin_gap_loss_weight=0.01,
        margin_gap_epsilon=0.5,
    )
    model = HyperbolicGPTV2(cfg).to(device)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    logits, loss, aux = model(x, y, return_aux=True)
    assert torch.isfinite(logits).all(), "non-finite logits"
    assert loss is not None and torch.isfinite(loss), "non-finite loss"
    assert "hidden" in aux, "missing hidden diagnostics"
    metrics = summarize_manifold_batch(
        logits=logits,
        hidden=aux["hidden"],
        tangent=aux.get("tangent"),
        loss_ce=aux.get("loss_ce", loss.detach()),
        gap_epsilon=cfg.margin_gap_epsilon,
    )
    assert 0.0 <= metrics.gap_frac <= 1.0, "bad gap fraction"
    assert metrics.fisher_trace_proxy >= 0.0, "bad fisher proxy"
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"
    model.eval()
    gen = model.generate(x[:1, :8], max_new_tokens=5, temperature=1.0)
    assert torch.isfinite(gen.float()).all(), "non-finite generation"
    print("SMOKE TEST V2 PASSED")
    print(metrics)


if __name__ == "__main__":
    main()
