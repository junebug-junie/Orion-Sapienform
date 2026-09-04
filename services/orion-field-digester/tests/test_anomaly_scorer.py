from __future__ import annotations

import json
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
    """Returns a models_root (containing active.json + the versioned artifact
    dir under it), not the artifact dir directly -- matches what
    FieldChannelAnomalyScorer expects since 2026-09-03, when it moved from a
    literal `encoder_dir` to resolving "whatever's active" via
    resolve_active_encoder_dir()."""
    d_in = _WINDOW_SIZE * len(_FIELDS)
    weights = init_weights(d_in, hidden_dim=4, latent_dim=2, seed=0, data_mean=np.zeros(d_in))
    manifest = MoodArcEncoderManifestV1(
        encoder_id="mood-arc-encoder:test.v1",
        encoder_version="test.v1",
        status="active",
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
    models_root = tmp_path / "models"
    version_dir = models_root / "test.v1"
    write_artifacts(version_dir, manifest=manifest, weights=weights, probes={})
    (models_root / "active.json").write_text(
        json.dumps(
            {
                "encoder_id": manifest.encoder_id,
                "encoder_version": manifest.encoder_version,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "path": str(version_dir),
            }
        ),
        encoding="utf-8",
    )
    return models_root


def _row(i: int, *, base: datetime) -> FieldChannelCorpusRowV1:
    return FieldChannelCorpusRowV1(
        generated_at=base + timedelta(seconds=2 * i),
        tick_id=f"tick_{i}",
        channels={"cpu_pressure": 0.1 * i, "gpu_pressure": 0.05 * i},
    )


def test_score_latest_returns_none_before_models_root_configured() -> None:
    scorer = FieldChannelAnomalyScorer(models_root="", threshold_multiplier=3.0)
    scorer.append_row(_row(0, base=datetime.now(timezone.utc)))
    assert scorer.score_latest() is None


def test_score_latest_returns_none_when_models_root_does_not_exist(tmp_path: Path) -> None:
    scorer = FieldChannelAnomalyScorer(
        models_root=str(tmp_path / "nonexistent"), threshold_multiplier=3.0
    )
    scorer.append_row(_row(0, base=datetime.now(timezone.utc)))
    assert scorer.score_latest() is None


def test_score_latest_returns_none_below_window_size(tmp_path: Path) -> None:
    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(models_root=str(models_root), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE - 1):
        scorer.append_row(_row(i, base=base))
    assert scorer.score_latest() is None


def test_score_latest_returns_a_real_score_at_window_size(tmp_path: Path) -> None:
    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(models_root=str(models_root), threshold_multiplier=3.0)
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
    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(models_root=str(models_root), threshold_multiplier=3.0)
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
    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, startup_grace_sec=3600.0
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
    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, startup_grace_sec=3600.0
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
    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(models_root=str(models_root), threshold_multiplier=3.0)
    base = datetime.now(timezone.utc)
    for i in range(_WINDOW_SIZE):
        scorer.append_row(_row(i, base=base))
    assert scorer.score_latest() is not None


def test_attribution_failure_does_not_break_the_core_score(tmp_path: Path, monkeypatch) -> None:
    """top_channel_attribution() raising must not take down score_latest()
    -- the recon_loss/threshold/anomalous decision is the load-bearing part,
    attribution is best-effort extra context."""
    import app.anomaly_scorer as anomaly_scorer_module

    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(models_root=str(models_root), threshold_multiplier=3.0)
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
    """A models_root with no active.json (nothing was ever promoted to it)
    should fail once and stay disabled, not re-attempt (and re-log) load on
    every append_row/score_latest call in a hot ~2s tick loop -- unlike a
    transient live-enrichment DB error, this is a config/deploy problem that
    needs a human to fix, so it latches."""
    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    scorer = FieldChannelAnomalyScorer(models_root=str(bad_root), threshold_multiplier=3.0)
    assert scorer.score_latest() is None
    assert scorer._load_failed is True
    scorer.append_row(_row(0, base=datetime.now(timezone.utc)))
    assert len(scorer._buffer) == 0


# --- Live enrichment (2026-09-03) --------------------------------------------


class _FakeLiveCursor:
    def __init__(self, conn: "_FakeLiveConn") -> None:
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self._conn.queries.append(query)

    def fetchall(self):
        rows = self._conn.rows_sequence[self._conn.call_index]
        self._conn.call_index += 1
        return rows


class _FakeLiveConn:
    """Same query-sequenced shape as orion/mood_arc/tests/test_corpus_enrichment.py's
    _SequencedFakeConn -- resolve_live_enrichment() issues two queries per
    call (action_warrant, then attention_self_model) with differently-shaped
    row tuples, so a single fixed `rows` list (like the plain _FakeConn used
    elsewhere) can't serve both."""

    def __init__(self, rows_sequence: list[list[tuple]]) -> None:
        self.rows_sequence = rows_sequence
        self.call_index = 0
        self.queries: list[str] = []
        self.closed = False

    def cursor(self):
        return _FakeLiveCursor(self)

    def close(self):
        self.closed = True


def test_append_row_merges_live_enrichment_fields_into_buffer(tmp_path: Path, monkeypatch) -> None:
    """action_warrant (or any of the 5 attention_self_model fields) resolved
    live must land in the buffered row's channels -- these aren't in
    _FIELDS/the manifest's channel_names for this fixture, but append_row()
    must still merge whatever resolve_live_enrichment() returns; whether a
    channel is actually TRAINED on is select_fields()'s concern, not this
    one's."""
    import app.anomaly_scorer as anomaly_scorer_module

    models_root = _write_tiny_encoder(tmp_path)
    now = datetime.now(timezone.utc)
    fake_conn = _FakeLiveConn([[(now, 0.77)], []])  # action_warrant hit, attention_self_model empty
    monkeypatch.setattr(anomaly_scorer_module, "open_readonly_connection", lambda *a, **k: fake_conn)

    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, postgres_uri="postgresql://fake/db"
    )
    scorer.append_row(_row(0, base=now))

    buffered = list(scorer._buffer)
    assert len(buffered) == 1
    assert buffered[0].channels["action_warrant"] == 0.77
    assert buffered[0].channels["cpu_pressure"] == 0.0  # original field untouched


def test_append_row_does_not_mutate_the_passed_in_row(tmp_path: Path, monkeypatch) -> None:
    """Regression test for the shared-reference risk: worker.py's _tick()
    passes the SAME row object to the JSONL corpus sink AND this scorer --
    append_row() must never mutate row.channels in place, only ever buffer a
    fresh copy, or live-enrichment fields would silently leak into the
    training corpus too."""
    import app.anomaly_scorer as anomaly_scorer_module

    models_root = _write_tiny_encoder(tmp_path)
    now = datetime.now(timezone.utc)
    fake_conn = _FakeLiveConn([[(now, 0.5)], []])
    monkeypatch.setattr(anomaly_scorer_module, "open_readonly_connection", lambda *a, **k: fake_conn)

    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, postgres_uri="postgresql://fake/db"
    )
    original_row = _row(0, base=now)
    original_channels_obj = original_row.channels
    original_channels_copy = dict(original_row.channels)

    scorer.append_row(original_row)

    assert original_row.channels is original_channels_obj
    assert original_row.channels == original_channels_copy
    assert "action_warrant" not in original_row.channels
    # ...but the buffered copy DID get it -- confirms this isn't just "live
    # enrichment silently did nothing".
    assert scorer._buffer[0].channels["action_warrant"] == 0.5


def test_append_row_leaves_channels_untouched_when_postgres_uri_unset(
    tmp_path: Path, monkeypatch
) -> None:
    """Backward-compat guard: postgres_uri defaults to "" (existing callers/
    tests that don't pass it), which must never attempt a DB call at all."""
    import app.anomaly_scorer as anomaly_scorer_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("open_readonly_connection must not be called when postgres_uri is unset")

    monkeypatch.setattr(anomaly_scorer_module, "open_readonly_connection", _fail_if_called)

    models_root = _write_tiny_encoder(tmp_path)
    scorer = FieldChannelAnomalyScorer(models_root=str(models_root), threshold_multiplier=3.0)
    row = _row(0, base=datetime.now(timezone.utc))
    scorer.append_row(row)

    assert len(scorer._buffer) == 1
    assert "action_warrant" not in scorer._buffer[0].channels


def test_live_enrichment_db_error_is_non_sticky(tmp_path: Path, monkeypatch) -> None:
    """A transient DB failure (open_readonly_connection returning None, per
    its own documented contract) must disable live enrichment without
    latching _load_failed or otherwise stopping buffering -- once the
    reconnect cooldown (_DB_RECONNECT_COOLDOWN_SEC) elapses, the next tick's
    append_row() retries a fresh connection on its own. The cooldown itself
    (added 2026-09-03, review finding: retrying a real connect on every ~2s
    tick during a sustained outage would stall this worker's whole tick
    loop) is simulated elapsed here by resetting the private deadline
    directly, matching this file's existing convention for simulating time
    passage (see test_append_row_still_buffers_during_startup_grace_period's
    `scorer._startup_grace_sec = 0.0`) -- a real second tick landing inside
    the cooldown window is covered by the next assert below instead."""
    import app.anomaly_scorer as anomaly_scorer_module

    models_root = _write_tiny_encoder(tmp_path)
    now = datetime.now(timezone.utc)
    call_count = {"n": 0}
    fake_conn = _FakeLiveConn([[(now, 0.9)], []])

    def _flaky_connect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None  # simulates open_readonly_connection's own failure contract
        return fake_conn

    monkeypatch.setattr(anomaly_scorer_module, "open_readonly_connection", _flaky_connect)

    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, postgres_uri="postgresql://fake/db"
    )

    scorer.append_row(_row(0, base=now))
    assert scorer._load_failed is False
    assert len(scorer._buffer) == 1
    assert "action_warrant" not in scorer._buffer[0].channels
    assert scorer._db_connect_retry_after is not None  # cooldown armed

    # A real second tick landing inside the cooldown window must not even
    # attempt a reconnect -- confirms the cooldown itself actually works,
    # not just that a later retry eventually succeeds.
    scorer.append_row(_row(1, base=now))
    assert call_count["n"] == 1
    assert "action_warrant" not in scorer._buffer[1].channels

    scorer._db_connect_retry_after = None  # simulate the cooldown having elapsed
    scorer.append_row(_row(2, base=now))
    assert len(scorer._buffer) == 3
    assert call_count["n"] == 2
    assert scorer._buffer[2].channels["action_warrant"] == 0.9


# --- status() (2026-09-04, operator visibility) ------------------------------


def test_status_before_models_root_configured() -> None:
    """No models_root at all: enabled stays True (the scorer itself is
    wired in, per FIELD_CHANNEL_ANOMALY_ENABLED), but nothing encoder-shaped
    has loaded yet -- absence must read as absence, not a fabricated version
    string."""
    scorer = FieldChannelAnomalyScorer(models_root="", threshold_multiplier=3.0)
    status = scorer.status()
    assert status["enabled"] is True
    assert status["load_failed"] is False
    assert "encoder_version" not in status
    assert status["last_live_enrichment_fields"] is None
    assert status["live_enrichment_configured"] is False


def test_status_reports_load_failure(tmp_path: Path) -> None:
    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    scorer = FieldChannelAnomalyScorer(models_root=str(bad_root), threshold_multiplier=3.0)
    scorer.score_latest()  # triggers _ensure_loaded(), which fails and latches
    status = scorer.status()
    assert status["load_failed"] is True
    assert "encoder_version" not in status


def test_status_reports_loaded_encoder_and_live_enrichment_field_names(
    tmp_path: Path, monkeypatch
) -> None:
    """After a real load + a live-enrichment fetch, status() must name the
    actual encoder_version/channels and exactly which of the live-enrichment
    fields landed on the last tick -- values, not just presence, since
    that's the whole point of an operator status check."""
    import app.anomaly_scorer as anomaly_scorer_module

    models_root = _write_tiny_encoder(tmp_path)
    now = datetime.now(timezone.utc)
    fake_conn = _FakeLiveConn([[(now, 0.42)], []])  # action_warrant hit, attention_self_model empty
    monkeypatch.setattr(anomaly_scorer_module, "open_readonly_connection", lambda *a, **k: fake_conn)

    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, postgres_uri="postgresql://fake/db"
    )
    scorer.append_row(_row(0, base=now))

    status = scorer.status()
    assert status["encoder_id"] == "mood-arc-encoder:test.v1"
    assert status["encoder_version"] == "test.v1"
    assert status["window_size"] == _WINDOW_SIZE
    assert status["channels"] == list(_FIELDS)
    assert status["live_enrichment_configured"] is True
    assert status["last_live_enrichment_fields"] == ["action_warrant"]
    assert status["last_live_enrichment_at"] is not None
    assert status["last_live_enrichment_attempt_at"] == status["last_live_enrichment_at"]
    assert status["last_live_enrichment_error"] is None


def test_status_distinguishes_a_stale_success_from_a_live_outage(tmp_path, monkeypatch) -> None:
    """Review finding (2026-09-04): last_live_enrichment_fields/_at only ever
    updated on success, so during a sustained outage they kept showing the
    last-healthy snapshot forever -- indistinguishable from "still healthy"
    on an operator status page. last_live_enrichment_attempt_at/_error must
    move on a failed tick even while the success fields stay frozen at
    whatever they were before the outage started."""
    import app.anomaly_scorer as anomaly_scorer_module

    models_root = _write_tiny_encoder(tmp_path)
    now = datetime.now(timezone.utc)
    fake_conn = _FakeLiveConn([[(now, 0.42)], []])
    monkeypatch.setattr(anomaly_scorer_module, "open_readonly_connection", lambda *a, **k: fake_conn)

    scorer = FieldChannelAnomalyScorer(
        models_root=str(models_root), threshold_multiplier=3.0, postgres_uri="postgresql://fake/db"
    )
    scorer.append_row(_row(0, base=now))
    healthy_status = scorer.status()
    assert healthy_status["last_live_enrichment_error"] is None

    # Outage starts: every subsequent query raises.
    def _raise(*a, **k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(anomaly_scorer_module, "resolve_live_enrichment", _raise)
    later = now + timedelta(seconds=2)
    scorer.append_row(_row(1, base=later))

    status = scorer.status()
    # Frozen at the pre-outage snapshot -- must NOT silently advance.
    assert status["last_live_enrichment_at"] == healthy_status["last_live_enrichment_at"]
    assert status["last_live_enrichment_fields"] == healthy_status["last_live_enrichment_fields"]
    # ...but the attempt/error fields DO move, proving there's a way to tell
    # "healthy 2 seconds ago" apart from "still healthy right now".
    assert status["last_live_enrichment_attempt_at"] != healthy_status["last_live_enrichment_attempt_at"]
    assert status["last_live_enrichment_error"] == "RuntimeError: connection reset by peer"
