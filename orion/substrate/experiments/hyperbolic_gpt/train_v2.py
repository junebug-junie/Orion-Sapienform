from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.data import (
    iter_batches,
    load_text_file_tokens,
    load_tinystories_tokens,
    shard_token_ids,
)
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import (
    summarize_manifold_batch,
)
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import HyperbolicGPTV2


def ddp_setup() -> tuple[bool, int, int, int]:
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(k in os.environ for k in keys):
        return False, 0, 1, 0
    return (
        True,
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
    )


def ddp_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_mean_scalar(value: float, device: torch.device, is_ddp: bool) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float64)
    if is_ddp:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def save_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    config: HyperbolicGPTConfig,
    rank: int,
) -> None:
    if rank != 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    meta = {"tokenizer": "gpt2", "block_size": config.block_size, "model_class": "HyperbolicGPTV2"}
    torch.save(
        {"model": raw.state_dict(), "config": config.to_dict()},
        out_dir / "model.pt",
    )
    config.save_json(out_dir / "config.json")
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def split_train_eval(token_ids: list[int], eval_frac: float = 0.05) -> tuple[list[int], list[int]]:
    n_eval = max(1, int(len(token_ids) * eval_frac))
    if len(token_ids) <= n_eval + 256:
        return token_ids, token_ids[-n_eval:]
    return token_ids[:-n_eval], token_ids[-n_eval:]


@torch.no_grad()
def eval_loss_and_geometry(
    model: torch.nn.Module,
    eval_ids: list[int],
    config: HyperbolicGPTConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    gap_fracs: list[float] = []
    margins: list[float] = []
    entropies: list[float] = []
    fisher: list[float] = []
    tangent: list[float] = []
    for x, y in iter_batches(eval_ids, config.block_size, batch_size, device):
        logits, loss, aux = model(x, y, return_aux=True)
        if loss is not None:
            loss_ce = aux.get("loss_ce", loss.detach())
            metrics = summarize_manifold_batch(
                logits=logits,
                hidden=aux["hidden"],
                tangent=aux.get("tangent"),
                loss_ce=loss_ce,
                gap_epsilon=config.margin_gap_epsilon,
            )
            losses.append(metrics.loss_ce)
            gap_fracs.append(metrics.gap_frac)
            margins.append(metrics.margin_mean)
            entropies.append(metrics.entropy_mean)
            fisher.append(metrics.fisher_trace_proxy)
            tangent.append(metrics.tangent_energy)
        if len(losses) >= 20:
            break
    model.train()
    denom = max(len(losses), 1)
    return {
        "loss_ce": sum(losses) / denom,
        "gap_frac": sum(gap_fracs) / denom,
        "margin_mean": sum(margins) / denom,
        "entropy_mean": sum(entropies) / denom,
        "fisher_trace_proxy": sum(fisher) / denom,
        "tangent_energy": sum(tangent) / denom,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train manifold-aware HyperbolicGPT v2")
    p.add_argument("--dataset", choices=("tinystories", "text"), required=True)
    p.add_argument("--text_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="./runs/hyperbolic_gpt_v2")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no_hyperbolic_attention", action="store_true", default=False)
    p.add_argument("--semantic_adapter_rank", type=int, default=64)
    p.add_argument("--margin_gap_loss_weight", type=float, default=0.0)
    p.add_argument("--margin_gap_epsilon", type=float, default=0.5)
    p.add_argument("--entropy_floor_loss_weight", type=float, default=0.0)
    p.add_argument("--min_entropy", type=float, default=0.0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    is_ddp, rank, world_size, local_rank = ddp_setup()
    set_seed(args.seed + rank)

    if is_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA (torchrun + nccl)")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config = HyperbolicGPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_hyperbolic_attention=not args.no_hyperbolic_attention,
        semantic_adapter_rank=args.semantic_adapter_rank,
        margin_gap_loss_weight=args.margin_gap_loss_weight,
        margin_gap_epsilon=args.margin_gap_epsilon,
        entropy_floor_loss_weight=args.entropy_floor_loss_weight,
        min_entropy=args.min_entropy,
    )

    if args.dataset == "tinystories":
        try:
            all_ids = load_tinystories_tokens(
                max_docs=args.max_docs, max_tokens=args.max_tokens
            )
        except Exception as exc:
            if not args.text_path:
                raise RuntimeError(
                    "TinyStories unavailable; pass --text_path for local .txt fallback"
                ) from exc
            ddp_print(rank, f"TinyStories failed ({exc}); using --text_path fallback")
            all_ids = load_text_file_tokens(args.text_path, max_tokens=args.max_tokens)
    else:
        if not args.text_path:
            raise ValueError("--text_path required when --dataset text")
        all_ids = load_text_file_tokens(args.text_path, max_tokens=args.max_tokens)

    train_ids, eval_ids = split_train_eval(all_ids)
    if is_ddp:
        train_ids = shard_token_ids(train_ids, rank, world_size)
        eval_ids = shard_token_ids(eval_ids, rank, world_size)

    model = HyperbolicGPTV2(config).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir)
    opt_step = 0
    micro_step = 0
    accum_loss = 0.0
    loss_window = 0
    optimizer.zero_grad(set_to_none=True)
    batch_iter = iter_batches(train_ids, config.block_size, args.batch_size, device)

    def _optimizer_step() -> None:
        nonlocal opt_step, accum_loss, loss_window
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1
        if opt_step % 10 == 0:
            ddp_print(rank, f"step {opt_step} loss {accum_loss / max(loss_window, 1):.4f}")
            accum_loss = 0.0
            loss_window = 0
        if opt_step % args.eval_interval == 0:
            ev = eval_loss_and_geometry(unwrap_model(model), eval_ids, config, device, args.batch_size)
            ev_mean = {k: reduce_mean_scalar(v, device, is_ddp) for k, v in ev.items()}
            ddp_print(
                rank,
                "eval step "
                f"{opt_step} loss_ce {ev_mean['loss_ce']:.4f} "
                f"gap@{config.margin_gap_epsilon:.2f} {ev_mean['gap_frac']:.3f} "
                f"margin {ev_mean['margin_mean']:.3f} "
                f"entropy {ev_mean['entropy_mean']:.3f} "
                f"fisher_proxy {ev_mean['fisher_trace_proxy']:.3f} "
                f"tangent {ev_mean['tangent_energy']:.6f}",
            )
        if opt_step % args.save_interval == 0:
            save_checkpoint(out_dir, model, config, rank)
            ddp_print(rank, f"saved checkpoint to {out_dir}")

    while opt_step < args.max_steps:
        try:
            x, y = next(batch_iter)
        except StopIteration:
            batch_iter = iter_batches(train_ids, config.block_size, args.batch_size, device)
            x, y = next(batch_iter)

        with autocast(enabled=args.amp and device.type == "cuda"):
            _, loss, _ = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at micro_step {micro_step}")
            loss = loss / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += float(loss.item()) * args.grad_accum
        loss_window += 1
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            _optimizer_step()

    if micro_step % args.grad_accum != 0 and opt_step < args.max_steps:
        _optimizer_step()
    save_checkpoint(out_dir, model, config, rank)
    ddp_print(rank, "training complete")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
