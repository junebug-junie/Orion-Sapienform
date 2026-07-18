from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
from orion.schemas.telemetry.mood_arc import MoodArcEncoderManifestV1

REPO = Path(__file__).resolve().parents[1]
FIT_SCRIPT = REPO / "scripts" / "fit_mood_arc_encoder.py"


def _run_fit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _make_row(*, t: datetime, channels: dict[str, float], idx: int) -> FieldChannelCorpusRowV1:
    return FieldChannelCorpusRowV1(generated_at=t, tick_id=f"tick_{idx}", channels=channels)


def _write_corpus(path: Path, rows: list[FieldChannelCorpusRowV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def _normal_rows(
    n_rows: int = 1500, seed: int = 0, start: datetime | None = None
) -> list[FieldChannelCorpusRowV1]:
    """Real periodic/trend pattern + noise, same shape as
    test_mood_arc_encoder_fit_script.py's own synthetic-corpus fixture --
    reused here so a trained encoder actually learns something nontrivial to
    reconstruct, rather than the near-constant-baseline pattern a degenerate
    corpus would produce."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows: list[FieldChannelCorpusRowV1] = []
    for i in range(n_rows):
        t = start + timedelta(seconds=2 * i)
        phase = i * 0.05
        coherence = 0.6 + 0.05 * np.sin(phase) + rng.normal(0, 0.01)
        energy = 0.3 + 0.1 * np.sin(phase * 0.7) + rng.normal(0, 0.02)
        novelty = 0.5 + 0.3 * np.sin(phase * 1.3) + rng.normal(0, 0.03)
        channels = {
            "coherence": float(coherence),
            "energy": float(energy),
            "novelty": float(novelty),
        }
        rows.append(_make_row(t=t, channels=channels, idx=i))
    return rows


def _frozen_ratchet_rows(
    n_rows: int, seed: int, start: datetime
) -> list[FieldChannelCorpusRowV1]:
    """Mimics this session's real pre-fix incident (a one-way-ratchet channel
    pinned at 1.0, e.g. bus_health/availability before PR #1108-#1115):
    coherence/energy/novelty all pinned flat at an out-of-distribution
    constant, none of the normal periodic variation an encoder trained on
    _normal_rows() would have learned to reconstruct."""
    rows: list[FieldChannelCorpusRowV1] = []
    for i in range(n_rows):
        t = start + timedelta(seconds=2 * i)
        channels = {"coherence": 1.0, "energy": 1.0, "novelty": 1.0}
        rows.append(_make_row(t=t, channels=channels, idx=i))
    return rows


def _train_small_encoder(tmp_path: Path) -> Path:
    corpus = tmp_path / "train_corpus.jsonl"
    out_dir = tmp_path / "candidate"
    _write_corpus(corpus, _normal_rows())
    proc = _run_fit(
        "train",
        "--corpus", str(corpus),
        "--out", str(out_dir),
        "--hidden-dim", "32",
        "--latent-dim", "16",
        "--epochs", "150",
        "--min-hours", "0.5",
        "--min-rows", "100",
        "--purge-gap-windows", "6",
        "--bootstrap-n", "20",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return out_dir


def test_load_artifacts_round_trips_write_artifacts(tmp_path: Path) -> None:
    from scripts.fit_mood_arc_encoder import load_artifacts

    encoder_dir = _train_small_encoder(tmp_path)
    manifest, weights = load_artifacts(encoder_dir)
    assert isinstance(manifest, MoodArcEncoderManifestV1)
    assert manifest.channel_names
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        assert key in weights
        assert isinstance(weights[key], np.ndarray)


def test_score_windows_returns_one_score_per_window(tmp_path: Path) -> None:
    from scripts.fit_mood_arc_encoder import load_artifacts, score_windows, _load_jsonl

    encoder_dir = _train_small_encoder(tmp_path)
    manifest, weights = load_artifacts(encoder_dir)

    scoring_corpus = tmp_path / "score_corpus.jsonl"
    _write_corpus(scoring_corpus, _normal_rows(n_rows=200, seed=1))
    rows = _load_jsonl(scoring_corpus)

    scored = score_windows(
        rows,
        fields=tuple(manifest.channel_names),
        window_size=manifest.window_size,
        stride=manifest.stride,
        max_gap_sec=manifest.max_gap_sec,
        weights=weights,
    )
    assert len(scored) > 0
    for loss, start, end in scored:
        assert isinstance(loss, float)
        assert loss >= 0.0
        assert start <= end
    # Timestamp order preserved (build_windows()'s own documented guarantee).
    starts = [s for _, s, _ in scored]
    assert starts == sorted(starts)


def test_detect_anomalies_filters_and_sorts_descending() -> None:
    from scripts.fit_mood_arc_encoder import detect_anomalies

    t0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
    scored = [
        (0.001, t0, t0 + timedelta(seconds=1)),
        (0.050, t0 + timedelta(seconds=2), t0 + timedelta(seconds=3)),
        (0.002, t0 + timedelta(seconds=4), t0 + timedelta(seconds=5)),
        (0.100, t0 + timedelta(seconds=6), t0 + timedelta(seconds=7)),
    ]
    anomalies = detect_anomalies(scored, threshold=0.01)
    assert len(anomalies) == 2
    assert [a["recon_loss"] for a in anomalies] == [0.100, 0.050]


def test_detect_anomalies_empty_when_nothing_exceeds_threshold() -> None:
    from scripts.fit_mood_arc_encoder import detect_anomalies

    t0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
    scored = [(0.001, t0, t0), (0.002, t0, t0)]
    assert detect_anomalies(scored, threshold=1.0) == []


def test_acceptance_check_flags_a_known_bad_frozen_ratchet_window(tmp_path: Path) -> None:
    """This is roadmap Item 3's own acceptance check: an encoder trained on
    normal (real periodic) data must retroactively flag at least one window
    from a known-bad period (here: a frozen-ratchet stretch mimicking the
    real bus_health/availability incident this session traced and fixed,
    PRs #1108-#1115) as anomalous."""
    from scripts.fit_mood_arc_encoder import (
        detect_anomalies,
        load_artifacts,
        score_windows,
        _load_jsonl,
    )

    encoder_dir = _train_small_encoder(tmp_path)
    manifest, weights = load_artifacts(encoder_dir)

    bad_corpus = tmp_path / "known_bad.jsonl"
    _write_corpus(
        bad_corpus,
        _frozen_ratchet_rows(n_rows=200, seed=2, start=datetime(2026, 7, 10, tzinfo=timezone.utc)),
    )
    rows = _load_jsonl(bad_corpus)

    scored = score_windows(
        rows,
        fields=tuple(manifest.channel_names),
        window_size=manifest.window_size,
        stride=manifest.stride,
        max_gap_sec=manifest.max_gap_sec,
        weights=weights,
    )
    assert scored, "expected at least one window from the frozen-ratchet corpus"

    threshold = manifest.training.recon_error_p95 * 3.0
    anomalies = detect_anomalies(scored, threshold=threshold)
    assert len(anomalies) > 0, (
        f"expected the frozen-ratchet window(s) to be flagged anomalous; "
        f"got 0 anomalies (threshold={threshold}, "
        f"scores={[s for s, _, _ in scored][:5]}...)"
    )


def test_cli_detect_anomalies_end_to_end(tmp_path: Path) -> None:
    encoder_dir = _train_small_encoder(tmp_path)

    bad_corpus = tmp_path / "known_bad.jsonl"
    _write_corpus(
        bad_corpus,
        _frozen_ratchet_rows(n_rows=200, seed=3, start=datetime(2026, 7, 10, tzinfo=timezone.utc)),
    )

    proc = _run_fit(
        "detect-anomalies",
        "--corpus", str(bad_corpus),
        "--encoder-dir", str(encoder_dir),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    result = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert result["windows_scored"] > 0
    assert result["anomalies_found"] > 0
    assert len(result["anomalies"]) == result["anomalies_found"]


def test_cli_detect_anomalies_min_max_generated_at_scopes_corpus(tmp_path: Path) -> None:
    encoder_dir = _train_small_encoder(tmp_path)

    mixed_corpus = tmp_path / "mixed.jsonl"
    start = datetime(2026, 7, 10, tzinfo=timezone.utc)
    normal_part = _normal_rows(n_rows=100, seed=4, start=start)
    bad_start = normal_part[-1].generated_at + timedelta(seconds=2)
    bad_part = _frozen_ratchet_rows(n_rows=100, seed=5, start=bad_start)
    _write_corpus(mixed_corpus, normal_part + bad_part)

    # Scope to only the bad half via --min-generated-at.
    proc = _run_fit(
        "detect-anomalies",
        "--corpus", str(mixed_corpus),
        "--encoder-dir", str(encoder_dir),
        "--min-generated-at", bad_start.isoformat().replace("+00:00", "Z"),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    result = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert result["anomalies_found"] > 0

    # Scope to only the normal half via --max-generated-at -- should find
    # far fewer (ideally zero) anomalies than scoring the bad half.
    proc2 = _run_fit(
        "detect-anomalies",
        "--corpus", str(mixed_corpus),
        "--encoder-dir", str(encoder_dir),
        "--max-generated-at", bad_start.isoformat().replace("+00:00", "Z"),
    )
    assert proc2.returncode == 0, proc2.stderr or proc2.stdout
    result2 = json.loads(proc2.stdout[proc2.stdout.index("{") :])
    assert result2["anomalies_found"] < result["anomalies_found"]


def test_cli_detect_anomalies_missing_encoder_dir_is_a_clear_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, _normal_rows(n_rows=50, seed=6))

    proc = _run_fit(
        "detect-anomalies",
        "--corpus", str(corpus),
        "--encoder-dir", str(tmp_path / "does-not-exist"),
    )
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "detect-anomalies: could not load encoder artifacts" in combined


def test_cli_detect_anomalies_no_windows_is_a_clear_error(tmp_path: Path) -> None:
    encoder_dir = _train_small_encoder(tmp_path)
    empty_corpus = tmp_path / "empty.jsonl"
    empty_corpus.write_text("")

    proc = _run_fit(
        "detect-anomalies",
        "--corpus", str(empty_corpus),
        "--encoder-dir", str(encoder_dir),
    )
    assert proc.returncode != 0
    assert "no rows" in (proc.stdout + proc.stderr).lower()
