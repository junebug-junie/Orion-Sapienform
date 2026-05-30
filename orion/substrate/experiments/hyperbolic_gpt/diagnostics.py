from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.model import HyperbolicGPT


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def safe_perplexity(loss: float, cap: float = 1e4) -> float:
    if not math.isfinite(loss) or loss > 20.0:
        return float("inf")
    try:
        ppl = math.exp(loss)
    except OverflowError:
        return float("inf")
    return min(ppl, cap)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    """Collect params and hyperbolic geometry scalars (handles DDP wrap)."""
    raw = unwrap_model(model)
    total, trainable = count_parameters(raw)
    geo_lambdas: list[float] = []
    curvatures: list[float] = []
    if isinstance(raw, HyperbolicGPT):
        for block in raw.transformer.h:
            attn = block.attn
            geo_lambdas.append(float(attn.geo_lambda.detach().mean().cpu()))
            curvatures.append(float(attn.curvature.detach().mean().cpu()))
    return {
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "geo_lambda_mean": sum(geo_lambdas) / len(geo_lambdas) if geo_lambdas else None,
        "geo_lambda_per_layer": geo_lambdas,
        "curvature_mean": sum(curvatures) / len(curvatures) if curvatures else None,
        "curvature_per_layer": curvatures,
    }


def cuda_memory_mb(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "cuda_memory_allocated_mb": None,
            "cuda_memory_reserved_mb": None,
            "cuda_max_memory_allocated_mb": None,
        }
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "cuda_memory_allocated_mb": round(
            torch.cuda.memory_allocated(idx) / (1024**2), 2
        ),
        "cuda_memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024**2), 2),
        "cuda_max_memory_allocated_mb": round(
            torch.cuda.max_memory_allocated(idx) / (1024**2), 2
        ),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_run_summary(
    out_dir: Path,
    args: Any,
    config: HyperbolicGPTConfig,
    model: torch.nn.Module,
    world_size: int,
    corpus_token_count: int,
) -> None:
    diag = get_model_diagnostics(model)
    tokens_per_step = args.batch_size * world_size * args.grad_accum * config.block_size
    summary = {
        "started_at_unix": time.time(),
        "args": vars(args),
        "config": config.to_dict(),
        "parameter_count": diag["parameter_count"],
        "trainable_parameter_count": diag["trainable_parameter_count"],
        "geo_lambda_mean": diag["geo_lambda_mean"],
        "curvature_mean": diag["curvature_mean"],
        "world_size": world_size,
        "corpus_token_count": corpus_token_count,
        "tokens_per_step": tokens_per_step,
        "estimated_total_tokens": tokens_per_step * args.max_steps,
        "effective_batch_sequences": args.batch_size * world_size * args.grad_accum,
        "effective_batch_tokens": tokens_per_step,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def build_log_record(
    *,
    step: int,
    split: str,
    loss: float,
    lr: float,
    elapsed_seconds: float,
    seconds_per_step: float | None,
    total_tokens_seen: int,
    args: Any,
    config: HyperbolicGPTConfig,
    device: torch.device,
    world_size: int,
    model: torch.nn.Module,
    grad_norm: float | None = None,
) -> dict[str, Any]:
    diag = get_model_diagnostics(model)
    tokens_per_step = args.batch_size * world_size * args.grad_accum * config.block_size
    rec: dict[str, Any] = {
        "step": step,
        "split": split,
        "loss": loss,
        "perplexity": safe_perplexity(loss),
        "lr": lr,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_step": round(seconds_per_step, 4) if seconds_per_step is not None else None,
        "tokens_per_step": tokens_per_step,
        "total_tokens_seen": total_tokens_seen,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "world_size": world_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "device": str(device),
        "geo_lambda": diag["geo_lambda_mean"],
        "curvature": diag["curvature_mean"],
        "grad_norm": grad_norm,
        **cuda_memory_mb(device),
    }
    return rec
