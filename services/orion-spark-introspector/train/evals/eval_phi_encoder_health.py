#!/usr/bin/env python3
"""Phi encoder health eval — recon + residual-after-headline-fit across all runs.

φ is trained to predict ``headline``. High φ↔headline correlation is therefore
*expected* and is **not** a success metric. This eval reports:

- reconstruction error (did the latent learn the feature space?)
- residual after fitting ``φ ≈ a·headline + b`` (is φ an identity copy?)
- supervised φ↔headline diagnostics (labeled as non-success)

Every version directory under ``--encoders-root`` is evaluated on the corpus
slice matching that run's ``features_version``, with full manifest metadata
for promotion tracing.

Run:
    python services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py \\
      --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl \\
      --encoders-root /mnt/telemetry/models/phi/encoders

Exit non-zero when the active encoder fails the collapse gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import PhiEncoderManifestV1

DEFAULT_CORPUS = Path("/mnt/telemetry/phi/corpus/inner_state.jsonl")
DEFAULT_ENCODERS_ROOT = Path("/mnt/telemetry/models/phi/encoders")

# Collapse: φ is effectively copying headline on the eval slice.
DEFAULT_COLLAPSE_RESIDUAL_STD = 1e-3
DEFAULT_COLLAPSE_NEAR_ID_FRAC = 0.50
NEAR_ID_ABS_ERR = 1e-4

# Soft recon warning (does not fail by default; active can opt into --fail-on-recon).
DEFAULT_RECON_P95_WARN = 0.25


def _load_phi_encoder_runtime():
    path = REPO / "services" / "orion-spark-introspector" / "app" / "phi_encoder.py"
    spec = importlib.util.spec_from_file_location("spark_phi_encoder_health", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.PhiEncoderRuntime


@dataclass(frozen=True)
class HeadlineFitStats:
    slope: float
    intercept: float
    residual_std: float
    near_identity_frac: float
    phi_headline_corr: float
    phi_abs_err_mean: float
    phi_abs_err_p50: float
    phi_abs_err_p95: float
    headline_std: float
    phi_std: float
    baseline_mae: float
    skill_vs_mean: float


@dataclass(frozen=True)
class ReconStats:
    mean: float
    p50: float
    p95: float


@dataclass
class RunHealth:
    encoder_version: str
    dir_name: str
    is_active_symlink: bool
    status: str
    features_version: str
    architecture: str
    git_sha: str
    trained_at: str | None
    promoted_at: str | None
    parent_version: str | None
    corpus_row_count_train: int | None
    corpus_excluded_train: int | None
    corpus_time_range_start: str | None
    corpus_time_range_end: str | None
    train_final_loss: float | None
    train_held_out_loss: float | None
    train_recon_p50: float | None
    train_recon_p95: float | None
    input_features: list[str] = field(default_factory=list)
    eval_rows: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    recon: dict[str, float] | None = None
    headline_fit: dict[str, float] | None = None
    collapsed: bool = False
    collapse_reasons: list[str] = field(default_factory=list)
    recon_warn: bool = False


def discover_encoder_dirs(encoders_root: Path) -> list[Path]:
    """Return version directories (not the ``active`` symlink itself)."""
    if not encoders_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(encoders_root.iterdir()):
        if child.name == "active":
            continue
        if child.is_dir() and (child / "manifest.json").is_file() and (child / "weights.npz").is_file():
            out.append(child)
    return out


def resolve_active_target(encoders_root: Path) -> Path | None:
    active = encoders_root / "active"
    if not active.exists():
        return None
    try:
        return active.resolve()
    except OSError:
        return None


def load_corpus_rows(corpus_path: Path) -> list[InnerStateFeaturesV1]:
    rows: list[InnerStateFeaturesV1] = []
    if not corpus_path.is_file():
        return rows
    for line_no, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(InnerStateFeaturesV1.model_validate_json(text))
        except Exception as exc:  # noqa: BLE001 — skip bad lines with context
            raise ValueError(f"invalid JSONL at {corpus_path}:{line_no}: {exc}") from exc
    return rows


def filter_eval_rows(
    rows: list[InnerStateFeaturesV1],
    *,
    features_version: str,
) -> list[InnerStateFeaturesV1]:
    return [
        r
        for r in rows
        if r.features_version == features_version
        and r.phi_health == "ok"
        and not r.grammar_truth_degraded
    ]


def headline_fit_stats(phi: np.ndarray, headline: np.ndarray) -> HeadlineFitStats:
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    headline = np.asarray(headline, dtype=np.float64).reshape(-1)
    if phi.size == 0:
        return HeadlineFitStats(
            slope=0.0,
            intercept=0.0,
            residual_std=0.0,
            near_identity_frac=0.0,
            phi_headline_corr=float("nan"),
            phi_abs_err_mean=0.0,
            phi_abs_err_p50=0.0,
            phi_abs_err_p95=0.0,
            headline_std=0.0,
            phi_std=0.0,
            baseline_mae=0.0,
            skill_vs_mean=0.0,
        )
    abs_err = np.abs(phi - headline)
    if phi.size >= 2 and np.std(headline) > 1e-12:
        design = np.column_stack([headline, np.ones(phi.size)])
        slope, intercept = np.linalg.lstsq(design, phi, rcond=None)[0]
        residual = phi - (slope * headline + intercept)
        residual_std = float(np.std(residual))
    else:
        slope, intercept = 0.0, float(np.mean(phi))
        residual_std = float(np.std(phi - intercept))
    if phi.size >= 2 and np.std(phi) > 1e-12 and np.std(headline) > 1e-12:
        corr = float(np.corrcoef(phi, headline)[0, 1])
    else:
        corr = float("nan")
    baseline_mae = float(np.mean(np.abs(headline - np.mean(headline))))
    mae = float(np.mean(abs_err))
    skill = float(1.0 - mae / baseline_mae) if baseline_mae > 1e-12 else 0.0
    return HeadlineFitStats(
        slope=float(slope),
        intercept=float(intercept),
        residual_std=residual_std,
        near_identity_frac=float(np.mean(abs_err < NEAR_ID_ABS_ERR)),
        phi_headline_corr=corr,
        phi_abs_err_mean=mae,
        phi_abs_err_p50=float(np.percentile(abs_err, 50)),
        phi_abs_err_p95=float(np.percentile(abs_err, 95)),
        headline_std=float(np.std(headline)),
        phi_std=float(np.std(phi)),
        baseline_mae=baseline_mae,
        skill_vs_mean=skill,
    )


def recon_stats(errors: np.ndarray) -> ReconStats:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if errors.size == 0:
        return ReconStats(mean=0.0, p50=0.0, p95=0.0)
    return ReconStats(
        mean=float(np.mean(errors)),
        p50=float(np.percentile(errors, 50)),
        p95=float(np.percentile(errors, 95)),
    )


def collapse_reasons(
    fit: HeadlineFitStats,
    *,
    residual_std_max: float,
    near_id_frac_max: float,
    slope_identity_tol: float = 0.05,
) -> list[str]:
    """Flag φ→headline identity collapse — not merely a low-variance φ head.

    A nearly-constant φ can yield tiny residual_std after a flat linear fit
    without copying headline. Require either a high near-identity fraction or
    (tiny residual AND slope≈1).
    """
    reasons: list[str] = []
    if fit.near_identity_frac > near_id_frac_max:
        reasons.append(
            f"near_identity_frac={fit.near_identity_frac:.3f} > {near_id_frac_max:g} "
            f"(|φ−headline|<{NEAR_ID_ABS_ERR:g})"
        )
    slope_is_identity = abs(fit.slope - 1.0) <= slope_identity_tol
    if fit.residual_std < residual_std_max and slope_is_identity:
        reasons.append(
            f"residual_std={fit.residual_std:.6g} < {residual_std_max:g} "
            f"with slope={fit.slope:.4f}≈1 (φ≈headline identity)"
        )
    return reasons


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def manifest_trace(manifest: PhiEncoderManifestV1) -> dict[str, Any]:
    corpus = manifest.corpus
    training = manifest.training
    return {
        "status": manifest.status,
        "features_version": manifest.features_version,
        "architecture": manifest.architecture,
        "git_sha": manifest.git_sha,
        "trained_at": _iso(manifest.trained_at),
        "promoted_at": _iso(manifest.promoted_at),
        "parent_version": manifest.parent_version,
        "corpus_row_count_train": corpus.row_count,
        "corpus_excluded_train": corpus.excluded_degenerate,
        "corpus_time_range_start": _iso(corpus.time_range_start),
        "corpus_time_range_end": _iso(corpus.time_range_end),
        "train_final_loss": training.final_loss,
        "train_held_out_loss": training.held_out_loss,
        "train_recon_p50": training.recon_error_p50,
        "train_recon_p95": training.recon_error_p95,
        "input_features": list(manifest.input_features),
    }


def evaluate_run(
    *,
    run_dir: Path,
    rows: list[InnerStateFeaturesV1],
    active_target: Path | None,
    phi_runtime_cls: Any,
    residual_std_max: float,
    near_id_frac_max: float,
    recon_p95_warn: float,
) -> RunHealth:
    manifest = PhiEncoderManifestV1.model_validate_json((run_dir / "manifest.json").read_text())
    trace = manifest_trace(manifest)
    is_active = active_target is not None and run_dir.resolve() == active_target.resolve()
    base = RunHealth(
        encoder_version=manifest.encoder_version,
        dir_name=run_dir.name,
        is_active_symlink=is_active,
        status=str(trace["status"]),
        features_version=str(trace["features_version"]),
        architecture=str(trace["architecture"]),
        git_sha=str(trace["git_sha"]),
        trained_at=trace["trained_at"],
        promoted_at=trace["promoted_at"],
        parent_version=trace["parent_version"],
        corpus_row_count_train=trace["corpus_row_count_train"],
        corpus_excluded_train=trace["corpus_excluded_train"],
        corpus_time_range_start=trace["corpus_time_range_start"],
        corpus_time_range_end=trace["corpus_time_range_end"],
        train_final_loss=trace["train_final_loss"],
        train_held_out_loss=trace["train_held_out_loss"],
        train_recon_p50=trace["train_recon_p50"],
        train_recon_p95=trace["train_recon_p95"],
        input_features=list(trace["input_features"]),
    )

    rt = phi_runtime_cls.load(run_dir, expected_features_version=manifest.features_version)
    if rt is None:
        base.skipped = True
        base.skip_reason = "runtime_load_failed"
        return base

    eval_rows = filter_eval_rows(rows, features_version=manifest.features_version)
    if not eval_rows:
        base.skipped = True
        base.skip_reason = f"no_ok_rows_for_features_version={manifest.features_version}"
        return base

    phis: list[float] = []
    headlines: list[float] = []
    recons: list[float] = []
    for row in eval_rows:
        x = rt.feature_vector_from_inner(row)
        out = rt.forward(x)
        phis.append(float(out.phi))
        headlines.append(float(row.headline))
        recons.append(float(out.recon_error))

    fit = headline_fit_stats(np.asarray(phis), np.asarray(headlines))
    recon = recon_stats(np.asarray(recons))
    reasons = collapse_reasons(
        fit,
        residual_std_max=residual_std_max,
        near_id_frac_max=near_id_frac_max,
    )
    base.eval_rows = len(eval_rows)
    base.recon = asdict(recon)
    base.headline_fit = asdict(fit)
    base.collapsed = bool(reasons)
    base.collapse_reasons = reasons
    base.recon_warn = recon.p95 > recon_p95_warn
    return base


def build_report(
    runs: list[RunHealth],
    *,
    corpus_path: Path,
    encoders_root: Path,
    residual_std_max: float,
    near_id_frac_max: float,
    recon_p95_warn: float,
) -> dict[str, Any]:
    active_runs = [r for r in runs if r.is_active_symlink]
    active = active_runs[0] if active_runs else None
    comparable = [r for r in runs if not r.skipped and r.recon and r.headline_fit]
    # Rank by recon p95 ascending among same features_version as active (if any).
    ranking: list[dict[str, Any]] = []
    if active is not None:
        peers = [r for r in comparable if r.features_version == active.features_version]
        peers_sorted = sorted(peers, key=lambda r: float(r.recon["p95"]))  # type: ignore[index]
        ranking = [
            {
                "rank": i + 1,
                "encoder_version": r.encoder_version,
                "recon_p95": r.recon["p95"],  # type: ignore[index]
                "residual_std": r.headline_fit["residual_std"],  # type: ignore[index]
                "collapsed": r.collapsed,
                "is_active": r.is_active_symlink,
            }
            for i, r in enumerate(peers_sorted)
        ]

    active_ok = True
    fail_reasons: list[str] = []
    if active is None:
        active_ok = False
        fail_reasons.append("no_active_encoder_symlink_target_in_report")
    elif active.skipped:
        active_ok = False
        fail_reasons.append(f"active_skipped:{active.skip_reason}")
    elif active.collapsed:
        active_ok = False
        fail_reasons.extend(active.collapse_reasons)

    return {
        "ok": active_ok,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "encoders_root": str(encoders_root),
        "gates": {
            "collapse_residual_std_min": residual_std_max,
            "collapse_near_identity_frac_max": near_id_frac_max,
            "near_identity_abs_err": NEAR_ID_ABS_ERR,
            "recon_p95_warn": recon_p95_warn,
            "note": (
                "φ↔headline correlation is a supervised-target diagnostic, not a success metric. "
                "Primary health: recon + residual-after-headline-fit."
            ),
        },
        "fail_reasons": fail_reasons,
        "active_encoder_version": None if active is None else active.encoder_version,
        "ranking_by_recon_p95_same_features_version": ranking,
        "runs": [asdict(r) for r in runs],
    }


def format_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== phi encoder health (all training runs) ===")
    lines.append(f"corpus: {report['corpus_path']}")
    lines.append(f"encoders_root: {report['encoders_root']}")
    lines.append(f"ok={report['ok']} active={report.get('active_encoder_version')}")
    if report["fail_reasons"]:
        lines.append("fail_reasons: " + "; ".join(report["fail_reasons"]))
    lines.append("")
    lines.append(
        f"{'version':28s} {'status':10s} {'fv':8s} {'n':>6s} "
        f"{'recon_p95':>10s} {'resid_std':>10s} {'near_id':>8s} "
        f"{'corr*':>8s} {'flags':20s}"
    )
    for run in report["runs"]:
        if run["skipped"]:
            flags = f"SKIP:{run['skip_reason']}"
            lines.append(
                f"{run['encoder_version']:28s} {run['status']:10s} {run['features_version']:8s} "
                f"{'—':>6s} {'—':>10s} {'—':>10s} {'—':>8s} {'—':>8s} {flags:20s}"
            )
            continue
        recon = run["recon"] or {}
        fit = run["headline_fit"] or {}
        flags_parts: list[str] = []
        if run["is_active_symlink"]:
            flags_parts.append("ACTIVE")
        if run["collapsed"]:
            flags_parts.append("COLLAPSE")
        if run["recon_warn"]:
            flags_parts.append("RECON_WARN")
        flags = ",".join(flags_parts) or "ok"
        corr = fit.get("phi_headline_corr")
        corr_s = f"{corr:.3f}" if isinstance(corr, float) and np.isfinite(corr) else "nan"
        lines.append(
            f"{run['encoder_version']:28s} {run['status']:10s} {run['features_version']:8s} "
            f"{run['eval_rows']:6d} {recon.get('p95', float('nan')):10.4f} "
            f"{fit.get('residual_std', float('nan')):10.6f} "
            f"{fit.get('near_identity_frac', float('nan')):8.3f} "
            f"{corr_s:>8s} {flags:20s}"
        )
    lines.append("")
    lines.append("* corr = φ↔headline (supervised target; not success)")
    if report.get("ranking_by_recon_p95_same_features_version"):
        lines.append("")
        lines.append("ranking (same features_version as active, by recon p95):")
        for row in report["ranking_by_recon_p95_same_features_version"]:
            mark = " <-- active" if row["is_active"] else ""
            lines.append(
                f"  #{row['rank']} {row['encoder_version']} "
                f"recon_p95={row['recon_p95']:.4f} resid_std={row['residual_std']:.6f}"
                f"{mark}"
            )
    lines.append("")
    lines.append("--- run metadata (trace) ---")
    for run in report["runs"]:
        lines.append(
            f"{run['encoder_version']}: git={run['git_sha'][:12]} "
            f"trained={run['trained_at']} promoted={run['promoted_at']} "
            f"train_rows={run['corpus_row_count_train']} "
            f"span={run['corpus_time_range_start']}..{run['corpus_time_range_end']} "
            f"train_loss={run['train_final_loss']} held_out={run['train_held_out_loss']}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--encoders-root", type=Path, default=DEFAULT_ENCODERS_ROOT)
    p.add_argument(
        "--collapse-residual-std",
        type=float,
        default=DEFAULT_COLLAPSE_RESIDUAL_STD,
        help="Fail active if residual std after headline fit is below this",
    )
    p.add_argument(
        "--collapse-near-id-frac",
        type=float,
        default=DEFAULT_COLLAPSE_NEAR_ID_FRAC,
        help="Fail active if fraction of |φ−headline|<1e-4 exceeds this",
    )
    p.add_argument("--recon-p95-warn", type=float, default=DEFAULT_RECON_P95_WARN)
    p.add_argument(
        "--fail-on-recon",
        action="store_true",
        help="Also fail when active recon p95 exceeds --recon-p95-warn",
    )
    p.add_argument("--json", action="store_true", help="Print JSON report only")
    p.add_argument("--out", type=Path, default=None, help="Write full JSON report to path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    PhiEncoderRuntime = _load_phi_encoder_runtime()
    rows = load_corpus_rows(args.corpus)
    active_target = resolve_active_target(args.encoders_root)
    runs: list[RunHealth] = []
    for run_dir in discover_encoder_dirs(args.encoders_root):
        runs.append(
            evaluate_run(
                run_dir=run_dir,
                rows=rows,
                active_target=active_target,
                phi_runtime_cls=PhiEncoderRuntime,
                residual_std_max=args.collapse_residual_std,
                near_id_frac_max=args.collapse_near_id_frac,
                recon_p95_warn=args.recon_p95_warn,
            )
        )
    report = build_report(
        runs,
        corpus_path=args.corpus,
        encoders_root=args.encoders_root,
        residual_std_max=args.collapse_residual_std,
        near_id_frac_max=args.collapse_near_id_frac,
        recon_p95_warn=args.recon_p95_warn,
    )
    if args.fail_on_recon:
        for run in runs:
            if run.is_active_symlink and run.recon_warn and not run.skipped:
                report["ok"] = False
                report["fail_reasons"].append(
                    f"active_recon_p95={run.recon['p95'] if run.recon else None} "
                    f"> warn={args.recon_p95_warn}"
                )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_text_report(report))
        if args.out is not None:
            print(f"\nwrote JSON report: {args.out}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
