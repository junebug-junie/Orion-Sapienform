"""Eval: PhiEncoderRuntime forward keeps phi bounded and smooth across ticks.

Trains a tiny encoder via ``scripts/fit_phi_encoder.py``, loads
``PhiEncoderRuntime``, replays consecutive InnerStateFeaturesV1 vectors,
and asserts phi in [0, 1], no NaN, and |delta_phi| < 0.5 between ticks.

Run: python services/orion-spark-introspector/evals/eval_phi_reward_bounded.py
Exits non-zero on failure so it can gate CI.
"""
from __future__ import annotations

import importlib.util
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

FIT_SCRIPT = REPO / "scripts" / "fit_phi_encoder.py"
_PHI_ENCODER_PATH = REPO / "services" / "orion-spark-introspector" / "app" / "phi_encoder.py"

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

MAX_DELTA_PHI = 0.5
N_REPLAY_TICKS = 16


def _load_phi_encoder_runtime():
    spec = importlib.util.spec_from_file_location("spark_phi_encoder_eval", _PHI_ENCODER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.PhiEncoderRuntime


def _feature(name: str, raw: float, scaled: float) -> InnerFeatureV1:
    return InnerFeatureV1(name=name, raw_value=raw, scaled_value=scaled, source=f"eval.{name}")


def _replay_rows(n_ticks: int = N_REPLAY_TICKS) -> list[InnerStateFeaturesV1]:
    """Smoothly varying consecutive ticks for delta-phi checks."""
    start = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    step = timedelta(seconds=30)
    rows: list[InnerStateFeaturesV1] = []
    for i in range(n_ticks):
        t = start + step * i
        features = []
        for j, name in enumerate(INPUT_FEATURES):
            # Small per-tick drift so consecutive phi deltas stay modest.
            scaled = float(0.35 + 0.25 * np.sin(i * 0.11 + j * 0.23))
            raw = scaled + 0.1
            features.append(_feature(name, raw, scaled))
        rows.append(
            InnerStateFeaturesV1(
                features_version="seed-v2",
                generated_at=t,
                features=features,
                headline=0.4 + 0.02 * i,
                phi_health="ok",
                grammar_truth_degraded=False,
            )
        )
    return rows


def _train_corpus_rows(n_rows: int = 120) -> list[InnerStateFeaturesV1]:
    start = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    step = timedelta(hours=2.0 / max(1, n_rows - 1))
    rows: list[InnerStateFeaturesV1] = []
    for i in range(n_rows):
        t = start + step * i
        features = []
        for j, name in enumerate(INPUT_FEATURES):
            raw = float(np.sin(t.timestamp() + j) * 0.4 + 0.5)
            scaled = float(np.sin(i * 0.17 + j * 0.31))
            features.append(_feature(name, raw, scaled))
        rows.append(
            InnerStateFeaturesV1(
                features_version="seed-v2",
                generated_at=t,
                features=features,
                headline=0.3 + (i % 10) * 0.05,
                phi_health="ok",
                grammar_truth_degraded=False,
            )
        )
    return rows


def _write_corpus(path: Path, rows: list[InnerStateFeaturesV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n")


def main() -> int:
    PhiEncoderRuntime = _load_phi_encoder_runtime()
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="phi_reward_bounded_eval_") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "train.jsonl"
        out_dir = tmp_path / "encoder"
        _write_corpus(corpus, _train_corpus_rows())

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
                "--min-hours",
                "1.0",
                "--epochs",
                "25",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        if proc.returncode != 0:
            print("EVAL FAIL: fit_phi_encoder returned non-zero")
            print(proc.stderr or proc.stdout)
            return 1

        rt = PhiEncoderRuntime.load(out_dir, expected_features_version="seed-v2")
        if rt is None:
            print("EVAL FAIL: PhiEncoderRuntime.load returned None")
            return 1

        phis: list[float] = []
        recons: list[float] = []
        for i, inner in enumerate(_replay_rows()):
            x = rt.feature_vector_from_inner(inner)
            out = rt.forward(x)
            phis.append(out.phi)
            recons.append(out.recon_error)

            if not np.isfinite(out.phi):
                failures.append(f"tick {i}: phi is not finite ({out.phi})")
            elif not (0.0 <= out.phi <= 1.0):
                failures.append(f"tick {i}: phi out of [0,1]: {out.phi}")
            if not np.isfinite(out.recon_error):
                failures.append(f"tick {i}: recon_error is not finite ({out.recon_error})")

        for i in range(1, len(phis)):
            delta = abs(phis[i] - phis[i - 1])
            if delta >= MAX_DELTA_PHI:
                failures.append(
                    f"tick {i}: |delta_phi|={delta:.4f} >= {MAX_DELTA_PHI} "
                    f"(phi[{i-1}]={phis[i-1]:.4f} phi[{i}]={phis[i]:.4f})"
                )

    if failures:
        print("EVAL FAIL:")
        for f in failures:
            print(" -", f)
        return 1

    deltas = [abs(phis[i] - phis[i - 1]) for i in range(1, len(phis))]
    print(
        f"EVAL PASS: ticks={len(phis)} "
        f"phi_min={min(phis):.4f} phi_max={max(phis):.4f} "
        f"delta_phi_max={max(deltas):.4f} "
        f"recon_mean={float(np.mean(recons)):.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
