"""Eval: phi encoder fit produces bounded recon metrics + probe correlations.

Runs the real ``scripts/fit_phi_encoder.py`` train path on synthetic
InnerStateFeaturesV1 JSONL (120 rows, 2h span) and prints recon_error
p50/p95 plus a probe summary from the written manifest.

Run: python services/orion-spark-introspector/evals/eval_phi_encoder_fit.py
Exits non-zero on failure so it can gate CI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import PhiEncoderManifestV1

FIT_SCRIPT = REPO / "scripts" / "fit_phi_encoder.py"

FELT_NAMES = (
    "coherence",
    "field_intensity",
    "agency_readiness",
    "execution_pressure",
    "reasoning_pressure",
    "resource_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "social_pressure",
    "introspection_pressure",
)
COGNITIVE_NAMES = (
    "recall_gate_fired",
    "reasoning_present",
    "exec_step_fail_rate",
    "execution_friction",
)
INPUT_FEATURES = list(FELT_NAMES) + ["overall_intensity"] + list(COGNITIVE_NAMES)


def _feature(name: str, raw: float, scaled: float | None = None) -> InnerFeatureV1:
    return InnerFeatureV1(
        name=name,
        raw_value=raw,
        scaled_value=raw if scaled is None else scaled,
        source=f"eval.{name}",
    )


def _inner_row(
    *,
    t: datetime,
    scaled_by_name: dict[str, float] | None = None,
    headline: float = 0.5,
) -> InnerStateFeaturesV1:
    scaled_by_name = scaled_by_name or {}
    features = []
    for name in INPUT_FEATURES:
        raw = float(np.sin(t.timestamp() + hash(name) % 7) * 0.4 + 0.5)
        scaled = scaled_by_name.get(name, raw * 0.5)
        features.append(_feature(name, raw, scaled))
    return InnerStateFeaturesV1(
        features_version="seed-v2",
        generated_at=t,
        features=features,
        headline=headline,
        phi_health="ok",
        grammar_truth_degraded=False,
    )


def _synthetic_corpus(n_rows: int = 120, hours_span: float = 2.0) -> list[InnerStateFeaturesV1]:
    start = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    step = timedelta(hours=hours_span / max(1, n_rows - 1))
    rows: list[InnerStateFeaturesV1] = []
    for i in range(n_rows):
        t = start + step * i
        scaled = {name: float(np.sin(i * 0.17 + j * 0.31)) for j, name in enumerate(INPUT_FEATURES)}
        rows.append(_inner_row(t=t, scaled_by_name=scaled, headline=0.3 + (i % 10) * 0.05))
    return rows


def _write_corpus(path: Path, rows: list[InnerStateFeaturesV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n")


def _probe_summary(probes: dict[str, dict[str, float]]) -> dict[str, float]:
    """Top |pearson| per latent dimension for felt features."""
    felt = set(FELT_NAMES) | {"overall_intensity"}
    summary: dict[str, float] = {}
    for z_name, corrs in probes.items():
        felt_corrs = {k: abs(v) for k, v in corrs.items() if k in felt}
        if felt_corrs:
            best_feat, best_val = max(felt_corrs.items(), key=lambda item: item[1])
            summary[f"{z_name}_top"] = round(best_val, 4)
            summary[f"{z_name}_feature"] = best_feat  # type: ignore[assignment]
    return summary


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phi_encoder_fit_eval_") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "train.jsonl"
        out_dir = tmp_path / "encoder_eval"
        _write_corpus(corpus, _synthetic_corpus())

        proc = subprocess.run(
            [
                sys.executable,
                str(FIT_SCRIPT),
                "--corpus",
                str(corpus),
                "--out",
                str(out_dir),
                "--hidden-dim",
                "8",
                "--latent-dim",
                "4",
                "--min-rows",
                "100",
                "--epochs",
                "30",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        if proc.returncode != 0:
            print("EVAL FAIL: fit_phi_encoder returned non-zero")
            print(proc.stderr or proc.stdout)
            return 1

        manifest_path = out_dir / "manifest.json"
        probes_path = out_dir / "probes.json"
        if not manifest_path.is_file() or not probes_path.is_file():
            print("EVAL FAIL: missing manifest.json or probes.json")
            return 1

        manifest = PhiEncoderManifestV1.model_validate_json(manifest_path.read_text())
        probes = json.loads(probes_path.read_text())
        training = manifest.training
        if training is None:
            print("EVAL FAIL: manifest missing training stats")
            return 1

        p50 = training.recon_error_p50
        p95 = training.recon_error_p95
        if p50 is None or p95 is None or not np.isfinite(p50) or not np.isfinite(p95):
            print("EVAL FAIL: recon_error p50/p95 missing or non-finite")
            return 1
        if p50 < 0.0 or p95 < 0.0 or p95 < p50:
            print(f"EVAL FAIL: invalid recon_error percentiles p50={p50} p95={p95}")
            return 1

        summary = _probe_summary(probes)
        print(
            f"EVAL PASS: rows={manifest.corpus.row_count} "
            f"recon_error_p50={p50:.6f} recon_error_p95={p95:.6f}"
        )
        print(f"probe_summary={json.dumps(summary, sort_keys=True)}")
        print(f"latent_dims={len(probes)} input_features={len(manifest.input_features)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
