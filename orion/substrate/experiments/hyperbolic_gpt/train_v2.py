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
from orion.substrate.experiments.hyperbolic_gpt.data import iter_batches, load_fineweb_edu_tokens, load_text_file_tokens, load_tinystories_tokens, shard_token_ids
from orion.substrate.experiments.hyperbolic_gpt.manifold_metrics import summarize_manifold_batch
from orion.substrate.experiments.hyperbolic_gpt.model_v2 import HyperbolicGPTV2


def ddp_setup():
    if all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    return False, 0, 1, 0


def raw(model):
    return model.module if isinstance(model, DDP) else model


def say(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def split(ids, frac=0.05):
    n = max(1, int(len(ids) * frac))
    return ids[:-n], ids[-n:]


def mean(x: float, device, is_ddp: bool):
    t = torch.tensor([x], dtype=torch.float64, device=device)
    if is_ddp:
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return float(t.item())


def load_ids(args, rank: int):
    if args.dataset == "tinystories":
        try:
            return load_tinystories_tokens(args.max_docs, args.max_tokens)
        except Exception as exc:
            if not args.text_path:
                raise RuntimeError("TinyStories unavailable; pass --text_path") from exc
            say(rank, f"TinyStories failed ({exc}); using text fallback")
            return load_text_file_tokens(args.text_path, args.max_tokens)
    if args.dataset == "fineweb_edu":
        try:
            return load_fineweb_edu_tokens(args.max_docs, args.max_tokens, args.fineweb_edu_name)
        except Exception as exc:
            if not args.text_path:
                raise RuntimeError("FineWeb-Edu unavailable; pass --text_path") from exc
            say(rank, f"FineWeb-Edu failed ({exc}); using text fallback")
            return load_text_file_tokens(args.text_path, args.max_tokens)
    if not args.text_path:
        raise ValueError("--text_path required for --dataset text")
    return load_text_file_tokens(args.text_path, args.max_tokens)


@torch.no_grad()
def eval_metrics(model, ids, cfg, device, batch_size):
    model.eval()
    rows = []
    for x, y in iter_batches(ids, cfg.block_size, batch_size, device):
        logits, loss, aux = model(x, y, return_aux=True)
        loss_ce = aux.get("loss_ce", loss.detach())
        rows.append(summarize_manifold_batch(logits=logits, hidden=aux["hidden"], tangent=aux.get("tangent"), loss_ce=loss_ce, gap_epsilon=cfg.margin_gap_epsilon))
        if len(rows) >= 20:
            break
    model.train()
    denom = max(len(rows), 1)
    return {
        "loss_ce": sum(r.loss_ce for r in rows) / denom,
        "gap_frac": sum(r.gap_frac for r in rows) / denom,
        "margin_mean": sum(r.margin_mean for r in rows) / denom,
        "entropy_mean": sum(r.entropy_mean for r in rows) / denom,
        "fisher_trace_proxy": sum(r.fisher_trace_proxy for r in rows) / denom,
        "tangent_energy": sum(r.tangent_energy for r in rows) / denom,
    }


def save(out_dir: Path, model, cfg, rank: int):
    if rank != 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": raw(model).state_dict(), "config": cfg.to_dict()}, out_dir / "model.pt")
    cfg.save_json(out_dir / "config.json")
    (out_dir / "meta.json").write_text(json.dumps({"model_class": "HyperbolicGPTV2", "tokenizer": "gpt2"}), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Train HyperbolicGPT v2 with manifold diagnostics")
    p.add_argument("--dataset", choices=("tinystories", "fineweb_edu", "text"), required=True)
    p.add_argument("--fineweb_edu_name", default="sample-10BT")
    p.add_argument("--text_path", default=None)
    p.add_argument("--out_dir", default="./runs/hypgpt_v2")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--semantic_adapter_rank", type=int, default=128)
    p.add_argument("--margin_gap_loss_weight", type=float, default=0.001)
    p.add_argument("--margin_gap_epsilon", type=float, default=0.5)
    p.add_argument("--entropy_floor_loss_weight", type=float, default=0.0)
    p.add_argument("--min_entropy", type=float, default=0.0)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=2500)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    is_ddp, rank, world, local = ddp_setup()
    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        print(f"[ddp] rank={rank} local_rank={local} world={world} device={device}", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HyperbolicGPTConfig(block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, semantic_adapter_rank=args.semantic_adapter_rank, margin_gap_loss_weight=args.margin_gap_loss_weight, margin_gap_epsilon=args.margin_gap_epsilon, entropy_floor_loss_weight=args.entropy_floor_loss_weight, min_entropy=args.min_entropy)
    ids = load_ids(args, rank)
    train_ids, eval_ids = split(ids)
    if is_ddp:
        train_ids = shard_token_ids(train_ids, rank, world)
        eval_ids = shard_token_ids(eval_ids, rank, world)
    model = HyperbolicGPTV2(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local], output_device=local)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    batch_iter = iter_batches(train_ids, cfg.block_size, args.batch_size, device)
    opt.zero_grad(set_to_none=True)
    step = micro = n = 0
    loss_sum = 0.0
    out_dir = Path(args.out_dir)
    while step < args.max_steps:
        try:
            x, y = next(batch_iter)
        except StopIteration:
            batch_iter = iter_batches(train_ids, cfg.block_size, args.batch_size, device)
            x, y = next(batch_iter)
        with autocast(enabled=args.amp and device.type == "cuda"):
            _, loss, _ = model(x, y)
            loss = loss / args.grad_accum
        scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
        loss_sum += float(loss.item()) * args.grad_accum
        n += 1
        micro += 1
        if micro % args.grad_accum:
            continue
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            say(rank, f"step {step} loss {loss_sum / max(n, 1):.4f}")
            loss_sum = 0.0; n = 0
        if step % args.eval_interval == 0:
            m = eval_metrics(raw(model), eval_ids, cfg, device, args.batch_size)
            m = {k: mean(v, device, is_ddp) for k, v in m.items()}
            say(rank, f"eval step {step} loss_ce {m['loss_ce']:.4f} gap@{cfg.margin_gap_epsilon:.2f} {m['gap_frac']:.3f} margin {m['margin_mean']:.3f} entropy {m['entropy_mean']:.3f} fisher_proxy {m['fisher_trace_proxy']:.3f} tangent {m['tangent_energy']:.6f}")
        if step % args.save_interval == 0:
            save(out_dir, model, cfg, rank)
            say(rank, f"saved checkpoint to {out_dir}")
    save(out_dir, model, cfg, rank)
    say(rank, "training complete")
    if is_ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
