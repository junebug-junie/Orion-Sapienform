from __future__ import annotations

import argparse
import json
import os
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
from orion.substrate.experiments.hyperbolic_gpt.model import HyperbolicGPT


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
    meta = {"tokenizer": "gpt2", "block_size": config.block_size}
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
def eval_loss(
    model: torch.nn.Module,
    eval_ids: list[int],
    config: HyperbolicGPTConfig,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for x, y in iter_batches(eval_ids, config.block_size, batch_size, device):
        logits, loss = model(x, y)
        if loss is not None:
            losses.append(float(loss.item()))
        if len(losses) >= 20:
            break
    model.train()
    return sum(losses) / max(len(losses), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HyperbolicGPT")
    p.add_argument("--dataset", choices=("tinystories", "text"), required=True)
    p.add_argument("--text_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="./runs/hyperbolic_gpt")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_ddp, rank, world_size, local_rank = ddp_setup()

    if is_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA (torchrun + nccl)")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = HyperbolicGPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    if args.dataset == "tinystories":
        try:
            all_ids = load_tinystories_tokens()
        except Exception as exc:
            if not args.text_path:
                raise RuntimeError(
                    "TinyStories unavailable; pass --text_path for local .txt fallback"
                ) from exc
            ddp_print(rank, f"TinyStories failed ({exc}); using --text_path fallback")
            all_ids = load_text_file_tokens(args.text_path)
    else:
        if not args.text_path:
            raise ValueError("--text_path required when --dataset text")
        all_ids = load_text_file_tokens(args.text_path)

    train_ids, eval_ids = split_train_eval(all_ids)
    if is_ddp:
        train_ids = shard_token_ids(train_ids, rank, world_size)

    model = HyperbolicGPT(config).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir)
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    batch_iter = iter_batches(train_ids, config.block_size, args.batch_size, device)
    while step < args.max_steps:
        try:
            x, y = next(batch_iter)
        except StopIteration:
            batch_iter = iter_batches(train_ids, config.block_size, args.batch_size, device)
            x, y = next(batch_iter)

        with autocast(enabled=args.amp and device.type == "cuda"):
            _, loss = model(x, y)
            loss = loss / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += float(loss.item()) * args.grad_accum

        if (step + 1) % args.grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 10 == 0:
            ddp_print(rank, f"step {step + 1} loss {accum_loss / 10:.4f}")
            accum_loss = 0.0

        if (step + 1) % args.eval_interval == 0:
            ev = eval_loss(unwrap_model(model), eval_ids, config, device, args.batch_size)
            ev_mean = reduce_mean_scalar(ev, device, is_ddp)
            ddp_print(rank, f"eval step {step + 1} loss {ev_mean:.4f}")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(out_dir, model, config, rank)
            ddp_print(rank, f"saved checkpoint to {out_dir}")

        step += 1

    save_checkpoint(out_dir, model, config, rank)
    ddp_print(rank, "training complete")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
