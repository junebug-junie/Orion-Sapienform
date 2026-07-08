#!/usr/bin/env python3
"""Offline phi encoder fit, evaluate, and promote (Plan 2).

Training is numpy-only SGD on InnerStateFeaturesV1 JSONL corpus rows.
Artifacts: manifest.json, weights.npz, probes.json under --out.

Modes:
  default (--corpus + --out): train candidate encoder
  --eval-only --manifest <dir>: run promote gates on fixture corpus
  --promote --version <ver> --encoders-root <path>: symlink active + flip status

Use --legacy-corpus when reading seed-v1 rows without cognitive trajectory features.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/fit_phi_encoder.py` puts scripts/ on sys.path[0],
# which shadows stdlib `platform` via scripts/platform/ and breaks pydantic.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import (
    CorpusStatsV1,
    PhiEncoderManifestV1,
    TrainingStatsV1,
)

_INNER_STATE_PATH = (
    REPO_ROOT / "services" / "orion-spark-introspector" / "app" / "inner_state.py"
)
_SPEC = importlib.util.spec_from_file_location("spark_inner_state_fit", _INNER_STATE_PATH)
assert _SPEC and _SPEC.loader
_inner_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_inner_state)

FELT_DIMENSIONS: tuple[str, ...] = _inner_state.FELT_DIMENSIONS
COGNITIVE_FEATURE_NAMES: tuple[str, ...] = _inner_state.COGNITIVE_FEATURE_NAMES

DEFAULT_FEATURES_VERSION = "seed-v2"
ARCHITECTURE = "mlp_shallow_v1"
PROMOTE_FIXTURE = REPO_ROOT / "fixtures" / "phi_encoder_promote_gate.jsonl"
PROMOTE_MIN_RECON_RATIO = 2.0

# Spec defaults (docs/superpowers/specs/2026-07-08-phi-encoder-plan2-design.md).
# Tests override with --min-rows / --min-hours for synthetic corpora.
DEFAULT_MIN_ROWS = 500
DEFAULT_MIN_HOURS = 4.0
DEFAULT_VARIANCE_FRACTION = 0.8
DEFAULT_VARIANCE_EPS = 1e-6
DEFAULT_HIDDEN_DIM = 16
DEFAULT_LATENT_DIM = 8
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.01
DEFAULT_PHI_WEIGHT = 0.25
DEFAULT_EARLY_STOP_PATIENCE = 15
DEFAULT_HELD_OUT_FRACTION = 0.15


@dataclass(frozen=True)
class LoadedRow:
    inner: InnerStateFeaturesV1
    x: np.ndarray
    phi_target: float
    felt_raw: dict[str, float]


@dataclass(frozen=True)
class CorpusGateConfig:
    min_rows: int = DEFAULT_MIN_ROWS
    min_hours: float = DEFAULT_MIN_HOURS
    variance_fraction: float = DEFAULT_VARIANCE_FRACTION
    variance_eps: float = DEFAULT_VARIANCE_EPS


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    latent_dim: int = DEFAULT_LATENT_DIM
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    phi_weight: float = DEFAULT_PHI_WEIGHT
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION
    seed: int = 42


def input_features(*, legacy_corpus: bool) -> list[str]:
    felt = list(FELT_DIMENSIONS) + ["overall_intensity"]
    if legacy_corpus:
        return felt
    return felt + list(COGNITIVE_FEATURE_NAMES)


def features_version(*, legacy_corpus: bool) -> str:
    return "seed-v1" if legacy_corpus else DEFAULT_FEATURES_VERSION


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline phi encoder fit/promote")
    parser.add_argument("--corpus", type=Path, help="InnerStateFeaturesV1 JSONL corpus")
    parser.add_argument("--out", type=Path, help="Output directory for trained artifacts")
    parser.add_argument("--legacy-corpus", action="store_true", help="seed-v1 felt-only rows (no cognitive features)")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-hours", type=float, default=DEFAULT_MIN_HOURS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-version", type=str, default=None, help="manifest encoder_version (default: out dir name)")
    parser.add_argument("--eval-only", action="store_true", help="Run promote gates on fixture using trained manifest")
    parser.add_argument("--manifest", type=Path, help="Trained artifact directory for --eval-only")
    parser.add_argument("--promote", action="store_true", help="Promote a trained version to active")
    parser.add_argument("--version", type=str, help="Encoder version directory name under --encoders-root")
    parser.add_argument("--encoders-root", type=Path, help="Root directory containing versioned encoders")
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _load_jsonl(path: Path) -> list[InnerStateFeaturesV1]:
    rows: list[InnerStateFeaturesV1] = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(InnerStateFeaturesV1.model_validate_json(text))
        except Exception as exc:  # noqa: BLE001 — corpus loader should skip bad lines with context
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _feature_maps(inner: InnerStateFeaturesV1) -> tuple[dict[str, float], dict[str, float]]:
    raw: dict[str, float] = {}
    scaled: dict[str, float] = {}
    for feat in inner.features:
        raw[feat.name] = float(feat.raw_value)
        scaled[feat.name] = float(feat.scaled_value)
    return raw, scaled


def _row_vector(inner: InnerStateFeaturesV1, names: list[str]) -> tuple[np.ndarray, dict[str, float], float]:
    raw_map, scaled_map = _feature_maps(inner)
    x = np.asarray([scaled_map.get(name, 0.0) for name in names], dtype=np.float64)
    felt_raw = {name: raw_map.get(name, 0.0) for name in FELT_DIMENSIONS}
    felt_raw["overall_intensity"] = raw_map.get("overall_intensity", 0.0)
    return x, felt_raw, float(inner.headline)


def _filter_training_rows(
    rows: Iterable[InnerStateFeaturesV1],
    *,
    feature_names: list[str],
) -> tuple[list[LoadedRow], int]:
    loaded: list[LoadedRow] = []
    excluded = 0
    for inner in rows:
        if inner.phi_health != "ok" or inner.grammar_truth_degraded:
            excluded += 1
            continue
        x, felt_raw, phi_target = _row_vector(inner, feature_names)
        loaded.append(LoadedRow(inner=inner, x=x, phi_target=phi_target, felt_raw=felt_raw))
    return loaded, excluded


def _hours_span(rows: list[LoadedRow]) -> float:
    if not rows:
        return 0.0
    times = [r.inner.generated_at for r in rows]
    start = min(times)
    end = max(times)
    return (end - start).total_seconds() / 3600.0


def _variance_gate(matrix: np.ndarray, *, fraction: float, eps: float) -> tuple[bool, int, int]:
    if matrix.size == 0:
        return False, 0, matrix.shape[1] if matrix.ndim == 2 else 0
    variances = np.var(matrix, axis=0)
    ok_count = int(np.sum(variances > eps))
    needed = int(np.ceil(fraction * matrix.shape[1]))
    return ok_count >= needed, ok_count, needed


def check_corpus_gates(
    rows: list[LoadedRow],
    matrix: np.ndarray,
    *,
    cfg: CorpusGateConfig,
) -> None:
    if len(rows) < cfg.min_rows:
        raise SystemExit(f"corpus gate failed: min_rows={cfg.min_rows} got={len(rows)}")
    hours = _hours_span(rows)
    if hours < cfg.min_hours:
        raise SystemExit(
            f"corpus gate failed: min_hours={cfg.min_hours} got={hours:.3f}"
        )
    ok, got, need = _variance_gate(
        matrix,
        fraction=cfg.variance_fraction,
        eps=cfg.variance_eps,
    )
    if not ok:
        raise SystemExit(
            "corpus gate failed: feature_variance "
            f"need>={need} features with var>{cfg.variance_eps}, got={got}"
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x_center = x - np.mean(x)
    y_center = y - np.mean(y)
    denom = float(np.linalg.norm(x_center) * np.linalg.norm(y_center))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_center, y_center) / denom)


def _init_weights(d_in: int, hidden_dim: int, latent_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 0.05
    return {
        "W1": (rng.standard_normal((d_in, hidden_dim)) * scale).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": (rng.standard_normal((hidden_dim, latent_dim)) * scale).astype(np.float64),
        "b2": np.zeros(latent_dim, dtype=np.float64),
        "W3": (rng.standard_normal((latent_dim, d_in)) * scale).astype(np.float64),
        "b3": np.zeros(d_in, dtype=np.float64),
        "w_phi": (rng.standard_normal(latent_dim) * scale).astype(np.float64),
        "b_phi": np.array(0.0, dtype=np.float64),
    }


def _forward(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_pre = x @ weights["W1"] + weights["b1"]
    h = np.maximum(0.0, h_pre)
    z = h @ weights["W2"] + weights["b2"]
    xhat = z @ weights["W3"] + weights["b3"]
    phi_logit = z @ weights["w_phi"] + float(weights["b_phi"])
    phi = 1.0 / (1.0 + np.exp(-phi_logit))
    return h_pre, h, z, xhat, phi


def _losses(
    x: np.ndarray,
    y_phi: float,
    weights: dict[str, np.ndarray],
    *,
    phi_weight: float,
) -> tuple[float, dict[str, np.ndarray]]:
    h_pre, h, z, xhat, phi = _forward(x, weights)
    recon = xhat - x
    recon_loss = float(np.mean(recon ** 2))
    phi_loss = float((phi - y_phi) ** 2)
    total = recon_loss + phi_weight * phi_loss

    d_xhat = (2.0 / x.size) * recon
    d_z = d_xhat @ weights["W3"].T
    d_W3 = np.outer(z, d_xhat)
    d_b3 = d_xhat

    d_phi = 2.0 * phi_weight * (phi - y_phi) * phi * (1.0 - phi)
    d_z += d_phi * weights["w_phi"]
    d_w_phi = d_phi * z
    d_b_phi = np.array(d_phi, dtype=np.float64)

    d_h = d_z @ weights["W2"].T
    d_W2 = np.outer(h, d_z)
    d_b2 = d_z

    relu_mask = (h_pre > 0.0).astype(np.float64)
    d_h_pre = d_h * relu_mask
    d_W1 = np.outer(x, d_h_pre)
    d_b1 = d_h_pre

    grads = {
        "W1": d_W1,
        "b1": d_b1,
        "W2": d_W2,
        "b2": d_b2,
        "W3": d_W3,
        "b3": d_b3,
        "w_phi": d_w_phi,
        "b_phi": d_b_phi,
    }
    return total, grads


def _apply_grads(weights: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float) -> None:
    for key in weights:
        weights[key] = weights[key] - lr * grads[key]


def _batch_mean_grad(
    batch: list[LoadedRow],
    weights: dict[str, np.ndarray],
    *,
    phi_weight: float,
) -> tuple[float, dict[str, np.ndarray]]:
    total_loss = 0.0
    accum = {key: np.zeros_like(val) for key, val in weights.items()}
    for row in batch:
        loss, grads = _losses(row.x, row.phi_target, weights, phi_weight=phi_weight)
        total_loss += loss
        for key, grad in grads.items():
            accum[key] += grad
    scale = 1.0 / max(1, len(batch))
    for key in accum:
        accum[key] *= scale
    return total_loss * scale, accum


def train_mlp(
    rows: list[LoadedRow],
    *,
    cfg: TrainConfig,
) -> tuple[dict[str, np.ndarray], TrainingStatsV1, np.ndarray]:
    if not rows:
        raise SystemExit("train: no rows after filtering")
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    held_n = max(1, int(len(rows) * cfg.held_out_fraction))
    held_idx = set(indices[:held_n].tolist())
    train_rows = [rows[i] for i in range(len(rows)) if i not in held_idx]
    held_rows = [rows[i] for i in range(len(rows)) if i in held_idx]

    d_in = train_rows[0].x.shape[0]
    weights = _init_weights(d_in, cfg.hidden_dim, cfg.latent_dim, cfg.seed)

    best_held = float("inf")
    best_weights = {k: v.copy() for k, v in weights.items()}
    stale_epochs = 0
    final_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        final_epoch = epoch
        rng.shuffle(train_rows)
        for start in range(0, len(train_rows), 32):
            batch = train_rows[start : start + 32]
            _, grads = _batch_mean_grad(batch, weights, phi_weight=cfg.phi_weight)
            _apply_grads(weights, grads, cfg.lr)

        held_losses = []
        for row in held_rows:
            loss, _ = _losses(row.x, row.phi_target, weights, phi_weight=cfg.phi_weight)
            held_losses.append(loss)
        held_loss = float(np.mean(held_losses)) if held_losses else float("inf")

        if held_loss + 1e-9 < best_held:
            best_held = held_loss
            best_weights = {k: v.copy() for k, v in weights.items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stop_patience:
                break

    train_matrix = np.stack([r.x for r in train_rows], axis=0)
    train_losses = []
    recon_errors = []
    for row in train_rows:
        loss, _ = _losses(row.x, row.phi_target, best_weights, phi_weight=cfg.phi_weight)
        train_losses.append(loss)
        _, _, _, xhat, _ = _forward(row.x, best_weights)
        recon_errors.append(float(np.mean((row.x - xhat) ** 2)))

    stats = TrainingStatsV1(
        epochs=final_epoch,
        final_loss=float(np.mean(train_losses)) if train_losses else 0.0,
        held_out_loss=best_held,
        recon_error_p50=_percentile(recon_errors, 50),
        recon_error_p95=_percentile(recon_errors, 95),
    )
    return best_weights, stats, train_matrix


def compute_probes(rows: list[LoadedRow], weights: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    latent_dim = int(weights["W2"].shape[1])
    z_cols = [[] for _ in range(latent_dim)]
    felt_names = list(FELT_DIMENSIONS) + ["overall_intensity"]
    felt_cols = {name: [] for name in felt_names}
    for row in rows:
        _, _, z, _, _ = _forward(row.x, weights)
        for i in range(latent_dim):
            z_cols[i].append(float(z[i]))
        for name in felt_names:
            felt_cols[name].append(float(row.felt_raw.get(name, 0.0)))

    probes: dict[str, dict[str, float]] = {}
    for i in range(latent_dim):
        z_arr = np.asarray(z_cols[i], dtype=np.float64)
        probes[f"z{i}"] = {
            name: round(_pearson(z_arr, np.asarray(vals, dtype=np.float64)), 4)
            for name, vals in felt_cols.items()
        }
    return probes


def _recon_errors_for_rows(rows: list[InnerStateFeaturesV1], feature_names: list[str], weights: dict[str, np.ndarray]) -> list[float]:
    errors: list[float] = []
    for inner in rows:
        x, _, _ = _row_vector(inner, feature_names)
        _, _, _, xhat, _ = _forward(x, weights)
        errors.append(float(np.mean((x - xhat) ** 2)))
    return errors


def _gate_label(inner: InnerStateFeaturesV1) -> str:
    meta = inner.metadata or {}
    gate = str(meta.get("gate") or "").strip().lower()
    if gate in {"healthy", "degenerate"}:
        return gate
    if inner.phi_health != "ok":
        return "degenerate"
    return "healthy"


def run_promote_gate(manifest_dir: Path, *, fixture_path: Path = PROMOTE_FIXTURE) -> dict[str, float | bool]:
    manifest = PhiEncoderManifestV1.model_validate_json((manifest_dir / "manifest.json").read_text())
    arrays = np.load(manifest_dir / "weights.npz")
    weights = {key: arrays[key] for key in arrays.files}

    fixture_rows = _load_jsonl(fixture_path)
    healthy: list[InnerStateFeaturesV1] = []
    degenerate: list[InnerStateFeaturesV1] = []
    for row in fixture_rows:
        label = _gate_label(row)
        if label == "degenerate":
            degenerate.append(row)
        else:
            healthy.append(row)

    healthy_errs = _recon_errors_for_rows(healthy, manifest.input_features, weights)
    degenerate_errs = _recon_errors_for_rows(degenerate, manifest.input_features, weights)
    healthy_p95 = _percentile(healthy_errs, 95)
    degenerate_p95 = _percentile(degenerate_errs, 95)
    ratio = degenerate_p95 / max(healthy_p95, 1e-9)
    passed = ratio >= PROMOTE_MIN_RECON_RATIO
    return {
        "passed": passed,
        "healthy_p95": healthy_p95,
        "degenerate_p95": degenerate_p95,
        "recon_ratio": ratio,
        "min_ratio": PROMOTE_MIN_RECON_RATIO,
    }


def write_artifacts(
    out_dir: Path,
    *,
    manifest: PhiEncoderManifestV1,
    weights: dict[str, np.ndarray],
    probes: dict[str, dict[str, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_with_probes = manifest.model_copy(update={"probes": probes})
    manifest_path.write_text(manifest_with_probes.model_dump_json(indent=2), encoding="utf-8")
    np.savez(out_dir / "weights.npz", **weights)
    (out_dir / "probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> int:
    if args.corpus is None or args.out is None:
        raise SystemExit("train mode requires --corpus and --out")
    feature_names = input_features(legacy_corpus=args.legacy_corpus)
    fv = features_version(legacy_corpus=args.legacy_corpus)
    raw_rows = _load_jsonl(args.corpus)
    loaded, excluded = _filter_training_rows(raw_rows, feature_names=feature_names)
    matrix = np.stack([r.x for r in loaded], axis=0) if loaded else np.zeros((0, len(feature_names)))
    gate_cfg = CorpusGateConfig(min_rows=args.min_rows, min_hours=args.min_hours)
    check_corpus_gates(loaded, matrix, cfg=gate_cfg)

    train_cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    weights, training_stats, _train_matrix = train_mlp(loaded, cfg=train_cfg)
    probes = compute_probes(loaded, weights)

    times = [r.inner.generated_at for r in loaded]
    encoder_version = args.encoder_version or args.out.name
    manifest = PhiEncoderManifestV1(
        encoder_id=f"phi-encoder:{encoder_version}",
        encoder_version=encoder_version,
        status="candidate",
        architecture=ARCHITECTURE,
        features_version=fv,
        input_features=feature_names,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        corpus=CorpusStatsV1(
            corpus_path=str(args.corpus),
            row_count=len(loaded),
            excluded_degenerate=excluded,
            time_range_start=min(times) if times else None,
            time_range_end=max(times) if times else None,
        ),
        training=training_stats,
        probes=probes,
        git_sha=_git_sha(),
        trained_at=datetime.now(timezone.utc),
    )
    write_artifacts(args.out, manifest=manifest, weights=weights, probes=probes)

    gate = run_promote_gate(args.out)
    print(
        json.dumps(
            {
                "trained": str(args.out),
                "rows": len(loaded),
                "excluded": excluded,
                "promote_gate": gate,
            },
            indent=2,
        )
    )
    return 0


def cmd_eval_only(args: argparse.Namespace) -> int:
    if args.manifest is None:
        raise SystemExit("--eval-only requires --manifest")
    gate = run_promote_gate(args.manifest)
    print(json.dumps({"promote_gate": gate}, indent=2))
    if not gate["passed"]:
        raise SystemExit(
            f"promote gate failed: recon_ratio={gate['recon_ratio']:.4f} "
            f"< min_ratio={gate['min_ratio']}"
        )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    if not args.version or args.encoders_root is None:
        raise SystemExit("--promote requires --version and --encoders-root")
    version_dir = args.encoders_root / args.version
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")

    gate = run_promote_gate(version_dir)
    if not gate["passed"]:
        raise SystemExit(
            f"promote gate failed: recon_ratio={gate['recon_ratio']:.4f} "
            f"< min_ratio={gate['min_ratio']}"
        )

    manifest = PhiEncoderManifestV1.model_validate_json(manifest_path.read_text())
    now = datetime.now(timezone.utc)
    active_link = args.encoders_root / "active"
    if active_link.is_symlink():
        prev_dir = active_link.resolve()
        prev_manifest_path = prev_dir / "manifest.json"
        if prev_manifest_path.is_file() and prev_dir != version_dir.resolve():
            prev = PhiEncoderManifestV1.model_validate_json(prev_manifest_path.read_text())
            if prev.status == "active":
                retired = prev.model_copy(update={"status": "retired"})
                prev_manifest_path.write_text(retired.model_dump_json(indent=2), encoding="utf-8")
    elif active_link.exists():
        raise SystemExit(f"encoders-root/active exists and is not a symlink: {active_link}")

    promoted = manifest.model_copy(update={"status": "active", "promoted_at": now})
    manifest_path.write_text(promoted.model_dump_json(indent=2), encoding="utf-8")

    if active_link.exists() or active_link.is_symlink():
        active_link.unlink()
    active_link.symlink_to(version_dir.resolve(), target_is_directory=True)
    print(json.dumps({"promoted": args.version, "active": str(active_link)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.promote:
        return cmd_promote(args)
    if args.eval_only:
        return cmd_eval_only(args)
    return cmd_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
