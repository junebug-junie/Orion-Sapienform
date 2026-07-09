from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1

REPO = Path(__file__).resolve().parents[1]
DIAG_SCRIPT = REPO / "scripts" / "diag.py"

SEEDV4_NAMES = (
    "agency_readiness",
    "execution_pressure",
    "reasoning_pressure",
    "overall_intensity",
    "recall_gate_fired",
    "reasoning_present",
    "execution_load",
    "reasoning_load",
)


def _run_diag(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DIAG_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _feature(name: str, value: float) -> InnerFeatureV1:
    return InnerFeatureV1(name=name, raw_value=value, scaled_value=value, source=f"test.{name}")


def _seedv4_row(*, t: datetime, scaled_by_name: dict[str, float]) -> InnerStateFeaturesV1:
    features = [_feature(name, scaled_by_name.get(name, 0.0)) for name in SEEDV4_NAMES]
    return InnerStateFeaturesV1(
        features_version="seed-v4",
        generated_at=t,
        features=features,
        headline=0.5,
        phi_health="ok",
        grammar_truth_degraded=False,
    )


def _write_corpus(path: Path, rows: list[InnerStateFeaturesV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n")


def _live_corpus(n_rows: int = 60, hours_span: float = 5.0, live_dims: int = 8) -> list[InnerStateFeaturesV1]:
    start = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
    step = timedelta(hours=hours_span / max(1, n_rows - 1))
    rng = np.random.default_rng(0)
    rows: list[InnerStateFeaturesV1] = []
    for i in range(n_rows):
        t = start + step * i
        scaled = {}
        for j, name in enumerate(SEEDV4_NAMES):
            scaled[name] = float(rng.standard_normal()) if j < live_dims else 0.0
        rows.append(_seedv4_row(t=t, scaled_by_name=scaled))
    return rows


def test_diag_reports_missing_corpus(tmp_path: Path) -> None:
    from scripts.diag import _parse_args, run_diag

    args = _parse_args(["--corpus", str(tmp_path / "nope.jsonl")])
    result = run_diag(args)
    assert result["ok"] is False
    assert "not found" in result["reason"]


def test_diag_passes_on_fully_live_seed_v4_corpus(tmp_path: Path) -> None:
    from scripts.diag import _parse_args, run_diag

    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, _live_corpus(n_rows=60, hours_span=5.0, live_dims=8))
    args = _parse_args([
        "--corpus", str(corpus),
        "--features-version", "seed-v4",
        "--min-rows", "50",
        "--min-hours", "4.0",
    ])
    result = run_diag(args)
    assert result["ok"] is True
    assert result["gates"]["min_rows"]["ok"] is True
    assert result["gates"]["min_hours"]["ok"] is True
    assert result["gates"]["variance"]["ok"] is True
    assert result["gates"]["variance"]["got"] == 8
    live_names = {d["name"] for d in result["dims"] if d["live"]}
    assert live_names == set(SEEDV4_NAMES)


def test_diag_fails_variance_gate_when_dims_mostly_frozen(tmp_path: Path) -> None:
    from scripts.diag import _parse_args, run_diag

    corpus = tmp_path / "corpus.jsonl"
    # live_dims=3 -> only agency_readiness/execution_pressure/reasoning_pressure
    # live; reasoning_present AND reasoning_load (indices 5, 7) are both dead,
    # so the seed-v4 gate policy needs 6 (not the generic ceil(0.8*8)=7).
    _write_corpus(corpus, _live_corpus(n_rows=60, hours_span=5.0, live_dims=3))
    args = _parse_args([
        "--corpus", str(corpus),
        "--features-version", "seed-v4",
        "--min-rows", "50",
        "--min-hours", "4.0",
    ])
    result = run_diag(args)
    assert result["ok"] is False
    assert result["gates"]["variance"]["ok"] is False
    assert result["gates"]["variance"]["got"] == 3
    assert result["gates"]["variance"]["need"] == 6
    frozen = [d for d in result["dims"] if not d["live"]]
    assert len(frozen) == 5


def test_diag_fails_on_too_few_rows(tmp_path: Path) -> None:
    from scripts.diag import _parse_args, run_diag

    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, _live_corpus(n_rows=10, hours_span=5.0, live_dims=8))
    args = _parse_args([
        "--corpus", str(corpus),
        "--features-version", "seed-v4",
        "--min-rows", "500",
        "--min-hours", "4.0",
    ])
    result = run_diag(args)
    assert result["ok"] is False
    assert result["gates"]["min_rows"]["ok"] is False
    assert result["gates"]["min_rows"]["got"] == 10


def test_diag_cli_exit_code_matches_gate_result(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, _live_corpus(n_rows=60, hours_span=5.0, live_dims=8))

    passing = _run_diag(
        "--corpus", str(corpus),
        "--features-version", "seed-v4",
        "--min-rows", "50",
        "--min-hours", "4.0",
    )
    assert passing.returncode == 0, passing.stderr or passing.stdout
    payload = json.loads(passing.stdout)
    assert payload["ok"] is True

    failing = _run_diag(
        "--corpus", str(corpus),
        "--features-version", "seed-v4",
        "--min-rows", "5000",
        "--min-hours", "4.0",
    )
    assert failing.returncode == 1
    payload2 = json.loads(failing.stdout)
    assert payload2["ok"] is False


def test_diag_default_features_version_is_seed_v4(tmp_path: Path) -> None:
    from scripts.diag import _parse_args

    args = _parse_args(["--corpus", str(tmp_path / "x.jsonl")])
    assert args.features_version == "seed-v4"


def test_diag_never_raises_on_malformed_jsonl_line(tmp_path: Path) -> None:
    """_load_jsonl raises ValueError on a malformed line -- run_diag must
    degrade to a reported failure, not propagate the crash."""
    from scripts.diag import _parse_args, run_diag

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"not": "valid inner state json"\n')  # missing closing brace
    args = _parse_args(["--corpus", str(corpus), "--features-version", "seed-v4"])
    result = run_diag(args)
    assert result["ok"] is False
    assert "corpus load failed" in result["reason"]
