from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.anomaly_scorer import FieldChannelAnomalyScorer
from orion.mood_arc.fit_encoder import init_weights, write_artifacts
from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1
from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
from orion.schemas.telemetry.mood_arc import CorpusStatsV1, MoodArcEncoderManifestV1, TrainingStatsV1

_FIELDS = ("cpu_pressure", "gpu_pressure")
_WINDOW_SIZE = 3


def _write_tiny_encoder(tmp_path: Path, *, recon_error_p95: float = 0.01) -> Path:
    d_in = _WINDOW_SIZE * len(_FIELDS)
    weights = init_weights(d_in, hidden_dim=4, latent_dim=2, seed=0, data_mean=np.zeros(d_in))
    manifest = MoodArcEncoderManifestV1(
        encoder_id="mood-arc-encoder:test.v1",
        encoder_version="test.v1",
        status="candidate",
        architecture="mlp_shallow_v1",
        window_size=_WINDOW_SIZE,
        stride=1,
        max_gap_sec=10.0,
        hidden_dim=4,
        latent_dim=2,
        channel_names=list(_FIELDS),
        corpus=CorpusStatsV1(corpus_path="unused", row_count=100, excluded_degenerate=0),
        training=TrainingStatsV1(
            epochs=1,
            final_loss=0.001,
            held_out_loss=0.001,
            recon_error_p50=0.001,
            recon_error_p95=recon_error_p95,
        ),
        shuffle_baseline_loss=0.02,
        git_sha="deadbeef",
        trained_at=datetime.now(timezone.utc),
    )
    out_dir = tmp_path / "encoder"
    write_artifacts(out_dir, manifest=manifest, weights=weights, probes={})
    return out_dir


def _row(i: int, *, base: datetime) -> FieldChannelCorpusRowV1:
    return FieldChannelCorpusRowV1(
        generated_at=base + timedelta(seconds=2 * i),
        tick_id=f"tick_{i}",
        channels={"cpu_pressure": 0.1 * i, "gpu_pressure": 0.05 * i},
    )


def test_score_latest_returns_none_before_encoder_dir_configured() -> None:
    scorer = FieldChannelAnomalyScorer(encoder_dir="", threshold_multiplier=3.0)
    scorer.append_row(_row(0, base=datetime.now(timezone.utc)))
    assert scorer.score_latest() is None


def test_score_latest_returns_none_when_encoder_dir_does_not_exist(tmp_path: Path) -> None:
    scorer = FieldChannelAnomalyScorer(
        encoder_dir=str(tmp_path / "nonexistent"), threshold_multiplier=3.0
    )
    scorer.append_row(_row(0, base=datetime.now(timezone.utc)))
    assert scorer.score_latest() is None


def test_score_latest_returns_none_below_window_size(tmp_path: Path) -> None:
    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(encoder_dir=str(encoder_dir), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE - 1):
        scorer.append_row(_row(i, base=base))
    assert scorer.score_latest() is None


def test_score_latest_returns_a_real_score_at_window_size(tmp_path: Path) -> None:
    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(encoder_dir=str(encoder_dir), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))

    score = scorer.score_latest()
    assert isinstance(score, FieldChannelAnomalyScoreV1)
    assert score.encoder_id == "mood-arc-encoder:test.v1"
    assert score.window_size == _WINDOW_SIZE
    assert score.recon_error_p95 == 0.01
    assert score.threshold == 0.01 * 3.0
    # top_channels: real per-channel attribution, not a placeholder -- both
    # trained fields present, ranked, each a "channel=mse" string.
    assert len(score.top_channels) == len(_FIELDS)
    assert {entry.split("=")[0] for entry in score.top_channels} == set(_FIELDS)
    assert score.anomalous == (score.recon_loss > score.threshold)
    assert score.attribution_ok is True
    assert score.deviation_direction in {"elevated", "depressed", "mixed"}


def test_buffer_gap_breaks_the_run_and_still_scores_the_latest_contiguous_window(
    tmp_path: Path,
) -> None:
    """A gap exceeding max_gap_sec must not silently crash scoring -- the
    run before the gap is simply excluded (_build_windows_with_span's
    contract), and score_latest() should still find a complete window in
    whatever the buffer holds after the gap."""
    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(encoder_dir=str(encoder_dir), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))
    # Big gap (> max_gap_sec=10.0), then a fresh contiguous run.
    gapped_base = base + timedelta(seconds=1000)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=gapped_base))

    score = scorer.score_latest()
    assert score is not None
    assert score.window_start >= gapped_base


def test_score_latest_returns_none_during_startup_grace_period(tmp_path: Path) -> None:
    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(
        encoder_dir=str(encoder_dir), threshold_multiplier=3.0, startup_grace_sec=3600.0
    )
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))

    # Full buffer, encoder loaded -- would score if not for the grace period.
    assert len(scorer._buffer) == _WINDOW_SIZE
    assert scorer.score_latest() is None


def test_append_row_still_buffers_during_startup_grace_period(tmp_path: Path) -> None:
    """The grace period gates scoring, not buffering -- a real window must
    be ready the instant the grace period ends, not require waiting an
    additional window_size worth of ticks on top of it."""
    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(
        encoder_dir=str(encoder_dir), threshold_multiplier=3.0, startup_grace_sec=3600.0
    )
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))
    assert len(scorer._buffer) == _WINDOW_SIZE

    scorer._startup_grace_sec = 0.0  # simulate grace period having elapsed
    score = scorer.score_latest()
    assert isinstance(score, FieldChannelAnomalyScoreV1)


def test_default_startup_grace_is_zero_and_does_not_block_scoring(tmp_path: Path) -> None:
    """Backward compatibility: callers that don't pass startup_grace_sec
    (e.g. existing tests, or a hypothetical future caller) get immediate
    scoring, same as before this option existed."""
    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(encoder_dir=str(encoder_dir), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))
    assert scorer.score_latest() is not None


def test_attribution_failure_does_not_break_the_core_score(tmp_path: Path, monkeypatch) -> None:
    """top_channel_attribution() raising must not take down score_latest()
    -- the recon_loss/threshold/anomalous decision is the load-bearing part,
    attribution is best-effort extra context."""
    import app.anomaly_scorer as anomaly_scorer_module

    encoder_dir = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(encoder_dir=str(encoder_dir), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated attribution failure")

    monkeypatch.setattr(anomaly_scorer_module, "top_channel_attribution", _raise)

    score = scorer.score_latest()
    assert isinstance(score, FieldChannelAnomalyScoreV1)
    assert score.top_channels == []
    # attribution_ok distinguishes "computation failed" from a genuinely
    # near-zero/mixed window -- both would otherwise look identical
    # downstream (review finding, 2026-07-21).
    assert score.attribution_ok is False
    assert score.deviation_direction == "mixed"


def test_load_failure_is_sticky_and_does_not_retry_every_call(tmp_path: Path) -> None:
    """A malformed encoder_dir (missing manifest.json) should fail once and
    stay disabled, not re-attempt (and re-log) load on every append_row/
    score_latest call in a hot ~2s tick loop."""
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    scorer = FieldChannelAnomalyScorer(encoder_dir=str(bad_dir), threshold_multiplier=3.0)
    assert scorer.score_latest() is None
    assert scorer._load_failed is True
    scorer.append_row(_row(0, base=datetime.now(timezone.utc)))
    assert len(scorer._buffer) == 0
