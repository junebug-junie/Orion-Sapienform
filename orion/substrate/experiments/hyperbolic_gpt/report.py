from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "inf" if v > 0 else "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _latest(rows: list[dict[str, Any]], split: str) -> dict[str, Any] | None:
    filtered = [r for r in rows if r.get("split") == split]
    return filtered[-1] if filtered else None


def _best_loss(rows: list[dict[str, Any]], split: str) -> tuple[float | None, int | None]:
    filtered = [r for r in rows if r.get("split") == split and r.get("loss") is not None]
    if not filtered:
        return None, None
    best = min(filtered, key=lambda r: r["loss"])
    return float(best["loss"]), int(best.get("step", -1))


def print_recommendations(
    summary: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    config: HyperbolicGPTConfig | None,
) -> None:
    _section("6. Recommendations")
    tips: list[str] = []
    latest_train = _latest(rows, "train")
    latest_eval = _latest(rows, "eval")
    best_eval_loss, _ = _best_loss(rows, "eval")

    max_mem = None
    for r in rows:
        m = r.get("cuda_max_memory_allocated_mb")
        if m is not None:
            max_mem = max(max_mem or 0.0, float(m))

    if max_mem is not None and max_mem > 0.9 * 32_768:
        tips.append(
            "VRAM peak > ~90% of 32GB: lower --batch_size or --block_size; raise --grad_accum."
        )

    if latest_train and latest_eval:
        tl = latest_train.get("loss")
        el = latest_eval.get("loss")
        if tl is not None and el is not None and el > tl * 1.15:
            tips.append(
                "Eval loss >> train loss: possible overfit — more data, lower LR, or fewer steps."
            )

    train_losses = [
        r["loss"] for r in rows if r.get("split") == "train" and r.get("loss") is not None
    ]
    if len(train_losses) >= 4:
        early = sum(train_losses[:2]) / 2
        late = sum(train_losses[-2:]) / 2
        if late > early * 0.98 and early < 4.0:
            tips.append(
                "Train loss plateaued: try larger model, LR sweep, or inspect hyperbolic penalty scale."
            )

    geo = latest_train.get("geo_lambda") if latest_train else None
    if geo is not None:
        if not math.isfinite(float(geo)) or float(geo) <= 0:
            tips.append("geo_lambda invalid: check for NaNs in hyperbolic path.")
        elif float(geo) > 1.0:
            tips.append("geo_lambda large (>1): hyperbolic penalty may dominate dot-product attention.")

    curv = latest_train.get("curvature") if latest_train else None
    if curv is not None and (
        not math.isfinite(float(curv)) or float(curv) <= 0 or float(curv) > 100
    ):
        tips.append("curvature extreme or invalid: inspect softplus params and fp32 geo ops.")

    if (
        best_eval_loss is not None
        and latest_eval
        and latest_eval.get("loss") == best_eval_loss
    ):
        tips.append("Eval loss still improving: continue run or scale corpus / model width.")

    if not tips:
        tips.append("No strong warnings; review train_log.jsonl for loss / VRAM trends.")
    for t in tips:
        print(f"- {t}")


def print_report(run_dir: Path) -> None:
    config_data = _load_json(run_dir / "config.json")
    meta = _load_json(run_dir / "meta.json")
    summary = _load_json(run_dir / "run_summary.json")
    rows = _load_jsonl(run_dir / "train_log.jsonl")

    config = HyperbolicGPTConfig.from_dict(config_data) if config_data else None

    _section("1. Run identity")
    print(f"run_dir:              {run_dir.resolve()}")
    if summary:
        print(f"parameter_count:      {summary.get('parameter_count', 'n/a')}")
        print(f"corpus_token_count:   {summary.get('corpus_token_count', 'n/a')}")
    if config:
        print(
            f"architecture:         n_layer={config.n_layer} n_head={config.n_head} "
            f"n_embd={config.n_embd} block_size={config.block_size}"
        )
    if summary and summary.get("args"):
        a = summary["args"]
        print(f"dataset:              {a.get('dataset')} max_docs={a.get('max_docs')} max_tokens={a.get('max_tokens')}")
    ckpt = run_dir / "model.pt"
    print(f"checkpoint:           {'yes' if ckpt.is_file() else 'missing'} ({ckpt.name})")
    if meta:
        print(f"meta:                 {meta}")

    _section("2. Hyperparameters")
    if summary:
        print(f"block_size:           {summary.get('config', {}).get('block_size', config.block_size if config else 'n/a')}")
        args = summary.get("args", {})
        print(f"batch_size:           {args.get('batch_size')}")
        print(f"grad_accum:           {args.get('grad_accum')}")
        print(f"world_size:           {summary.get('world_size')}")
        print(f"effective_batch_tokens: {summary.get('effective_batch_tokens')}")
        print(f"lr:                   {args.get('lr')}")
        print(f"max_steps:            {args.get('max_steps')}")
        print(f"estimated_total_tokens: {summary.get('estimated_total_tokens')}")
        print(f"tokens_per_step:      {summary.get('tokens_per_step')}")
    elif config:
        print(f"block_size:           {config.block_size}")

    _section("3. Training progress")
    if not rows:
        print("train_log.jsonl:      not found (re-run with updated train.py)")
    else:
        lt = _latest(rows, "train")
        le = _latest(rows, "eval")
        best_eval, best_step = _best_loss(rows, "eval")
        print(f"latest train loss:    {_fmt(lt.get('loss') if lt else None)}")
        print(f"latest eval loss:     {_fmt(le.get('loss') if le else None)}")
        print(f"best eval loss:       {_fmt(best_eval)} (step {best_step})")
        print(f"latest train ppl:     {_fmt(lt.get('perplexity') if lt else None)}")
        if le:
            print(f"latest eval ppl:      {_fmt(le.get('perplexity'))}")
        if lt:
            print(f"total_tokens_seen:    {lt.get('total_tokens_seen')}")
            print(f"elapsed_seconds:      {_fmt(lt.get('elapsed_seconds'), 1)}")
            print(f"avg seconds/step:     {_fmt(lt.get('seconds_per_step'))}")
            tps = None
            if lt.get("seconds_per_step") and lt.get("tokens_per_step"):
                tps = lt["tokens_per_step"] / lt["seconds_per_step"]
            print(f"avg tokens/sec:       {_fmt(tps, 1)}")

    _section("4. GPU / memory")
    last = rows[-1] if rows else {}
    print(f"allocated_mb:         {_fmt(last.get('cuda_memory_allocated_mb'), 1)}")
    print(f"reserved_mb:          {_fmt(last.get('cuda_memory_reserved_mb'), 1)}")
    print(f"max_allocated_mb:     {_fmt(last.get('cuda_max_memory_allocated_mb'), 1)}")
    max_mb = last.get("cuda_max_memory_allocated_mb")
    if max_mb is not None and float(max_mb) > 0.9 * 32_768:
        print("WARNING:              peak VRAM > ~90% of 32GB reference")

    _section("5. Geometry diagnostics")
    print(f"latest geo_lambda:      {_fmt(last.get('geo_lambda'))}")
    print(f"latest curvature:       {_fmt(last.get('curvature'))}")
    geo = last.get("geo_lambda")
    curv = last.get("curvature")
    if geo is not None and (not math.isfinite(float(geo)) or float(geo) <= 0):
        print("WARNING:              geo_lambda NaN or <= 0")
    if curv is not None and (not math.isfinite(float(curv)) or float(curv) <= 0):
        print("WARNING:              curvature NaN or <= 0")
    if config:
        head_dim = config.head_dim
        scale = 1.0 / math.sqrt(head_dim)
        if geo is not None and float(geo) > 10 * scale:
            print(
                f"NOTE:                 geo_lambda >> dot-scale (~{scale:.4f}); "
                "hyperbolic term may dominate attention logits"
            )

    print_recommendations(summary, rows, config)


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperbolic GPT run diagnostic report")
    p.add_argument("--run_dir", type=str, required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")
    print_report(run_dir)


if __name__ == "__main__":
    main()
