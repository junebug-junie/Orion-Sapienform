from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import PhiEncoderManifestV1

REPO = Path(__file__).resolve().parents[1]
FIT_SCRIPT = REPO / "scripts" / "fit_phi_encoder.py"
PROMOTE_FIXTURE = REPO / "fixtures" / "phi_encoder_promote_gate.jsonl"

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


def _run_fit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _feature(name: str, raw: float, scaled: float | None = None) -> InnerFeatureV1:
    return InnerFeatureV1(
        name=name,
        raw_value=raw,
        scaled_value=raw if scaled is None else scaled,
        source=f"test.{name}",
    )


def _inner_row(
    *,
    t: datetime,
    scaled_by_name: dict[str, float] | None = None,
    phi_health: str = "ok",
    grammar_truth_degraded: bool = False,
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
        phi_health=phi_health,
        grammar_truth_degraded=grammar_truth_degraded,
    )


def _write_corpus(path: Path, rows: list[InnerStateFeaturesV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n")


def _synthetic_corpus(n_rows: int = 120, hours_span: float = 2.0) -> list[InnerStateFeaturesV1]:
    start = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    step = timedelta(hours=hours_span / max(1, n_rows - 1))
    rows: list[InnerStateFeaturesV1] = []
    for i in range(n_rows):
        t = start + step * i
        scaled = {name: float(np.sin(i * 0.17 + j * 0.31)) for j, name in enumerate(INPUT_FEATURES)}
        rows.append(_inner_row(t=t, scaled_by_name=scaled, headline=0.3 + (i % 10) * 0.05))
    return rows


def test_fit_refuses_small_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text("")  # empty
    proc = _run_fit("--corpus", str(corpus), "--out", str(tmp_path / "out"))
    assert proc.returncode != 0
    assert "min_rows" in proc.stdout.lower() or "min_rows" in proc.stderr.lower()


def test_fit_trains_on_synthetic_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    out_dir = tmp_path / "encoder_v1"
    _write_corpus(corpus, _synthetic_corpus())
    proc = _run_fit(
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
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "weights.npz").is_file()
    assert (out_dir / "probes.json").is_file()
    manifest = PhiEncoderManifestV1.model_validate_json((out_dir / "manifest.json").read_text())
    assert manifest.architecture == "mlp_shallow_v1"
    assert manifest.features_version == "seed-v2"
    assert manifest.input_features == INPUT_FEATURES
    weights = np.load(out_dir / "weights.npz")
    for key in ("W1", "b1", "W2", "b2", "W3", "b3", "w_phi", "b_phi"):
        assert key in weights


def test_eval_only_runs_promote_gate(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    out_dir = tmp_path / "encoder_v1"
    _write_corpus(corpus, _synthetic_corpus())
    train = _run_fit(
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
        "40",
    )
    assert train.returncode == 0, train.stderr or train.stdout
    proc = _run_fit("--eval-only", "--manifest", str(out_dir))
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "promote_gate" in (proc.stdout + proc.stderr).lower()


def test_promote_symlinks_active(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    version_dir = tmp_path / "encoders" / "v20260708"
    encoders_root = tmp_path / "encoders"
    _write_corpus(corpus, _synthetic_corpus())
    train = _run_fit(
        "--corpus",
        str(corpus),
        "--out",
        str(version_dir),
        "--hidden-dim",
        "8",
        "--latent-dim",
        "4",
        "--min-rows",
        "100",
        "--epochs",
        "20",
    )
    assert train.returncode == 0, train.stderr or train.stdout
    eval_proc = _run_fit("--eval-only", "--manifest", str(version_dir))
    assert eval_proc.returncode == 0, eval_proc.stderr or eval_proc.stdout
    promote = _run_fit(
        "--promote",
        "--version",
        "v20260708",
        "--encoders-root",
        str(encoders_root),
    )
    assert promote.returncode == 0, promote.stderr or promote.stdout
    active = encoders_root / "active"
    assert active.is_symlink()
    assert active.resolve() == version_dir.resolve()
    manifest = PhiEncoderManifestV1.model_validate_json((version_dir / "manifest.json").read_text())
    assert manifest.status == "active"


@pytest.mark.parametrize("path", [PROMOTE_FIXTURE])
def test_promote_fixture_exists(path: Path) -> None:
    assert path.is_file()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 4
