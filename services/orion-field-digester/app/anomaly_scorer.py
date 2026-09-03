from __future__ import annotations

import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from orion.db_readonly import open_readonly_connection
from orion.mood_arc.corpus_enrichment import resolve_live_enrichment
from orion.mood_arc.fit_encoder import (
    build_windows,
    deviation_direction,
    load_artifacts,
    mean_signed_deviation,
    resolve_active_encoder_dir,
    score_windows,
    top_channel_attribution,
)
from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1
from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1

logger = logging.getLogger("orion.field.digester.anomaly_scorer")

# Small safety margin over the encoder's own window_size: covers rows that
# fall into the same run but land just past a stride boundary, without
# holding meaningfully more history than one window actually needs.
_BUFFER_MARGIN_ROWS = 10


class FieldChannelAnomalyScorer:
    """Periodic in-process rescoring of the live rolling window of
    field-channel pressures against a trained encoder (orion/mood_arc/
    fit_encoder.py). Loads its artifacts once, lazily, fail-open: a missing
    or malformed models_root disables scoring for this process's lifetime
    (logged once) rather than crashing the tick loop that calls append_row()
    every ~2s.

    `models_root` (2026-09-03, was `encoder_dir` -- see
    FIELD_CHANNEL_ANOMALY_MODELS_ROOT in settings.py) names a mood_arc
    promotion root (an `active.json` written by `fit_encoder.py promote`),
    not a literal versioned artifact directory -- resolved via
    `resolve_active_encoder_dir()` every time `_ensure_loaded()` actually
    (re)loads, so a restart after a new `promote` picks up the new version
    with no code/config change here.

    `postgres_uri` (optional, default "") enables live enrichment of the 6
    v4 channels that `collect_field_channel_pressures()` cannot produce
    in-process -- `action_warrant` and the 5 `attention_self_model` fields
    (`heartbeat_mean_ratio` + per-domain `prediction_error_*`). See
    `orion.mood_arc.corpus_enrichment.resolve_live_enrichment()`. The
    Postgres connection used for this is opened lazily and reused across
    ticks (unlike every other caller of `orion.db_readonly` in this repo,
    which are one-shot batch scripts -- reopening a fresh connection every
    ~2s for signals that update every 2-30s would be pure waste) but is
    deliberately NOT sticky on failure: any exception drops and nulls the
    cached connection, skipping live enrichment for that one tick only, so a
    transient DB blip (restart, brief partition) self-heals within a tick or
    two rather than permanently disabling live enrichment until this whole
    process restarts -- the opposite of _load_failed's sticky, human-needs-
    to-fix-it semantics for a bad models_root, and deliberately so.
    """

    def __init__(
        self,
        *,
        models_root: str,
        threshold_multiplier: float,
        postgres_uri: str = "",
        startup_grace_sec: float = 0.0,
        live_enrichment_lookback_hours: float = 0.25,
    ) -> None:
        self._models_root = models_root
        self._threshold_multiplier = float(threshold_multiplier)
        self._postgres_uri = postgres_uri
        self._live_enrichment_lookback_hours = float(live_enrichment_lookback_hours)
        self._startup_grace_sec = float(startup_grace_sec)
        self._created_at = datetime.now(timezone.utc)
        self._manifest = None
        self._weights: dict | None = None
        self._load_failed = False
        self._buffer: deque[FieldChannelCorpusRowV1] = deque()
        self._db_conn = None  # lazy, non-sticky -- see _fetch_live_enrichment_fields()

    def _ensure_loaded(self) -> bool:
        if self._load_failed:
            return False
        if self._manifest is not None:
            return True
        if not self._models_root:
            self._load_failed = True
            logger.warning("field_channel_anomaly_scorer_no_models_root")
            return False
        try:
            encoder_dir = resolve_active_encoder_dir(Path(self._models_root))
            self._manifest, self._weights = load_artifacts(encoder_dir)
        except Exception:
            self._load_failed = True
            logger.warning(
                "field_channel_anomaly_scorer_load_failed models_root=%s",
                self._models_root,
                exc_info=True,
            )
            return False
        self._buffer = deque(maxlen=self._manifest.window_size + _BUFFER_MARGIN_ROWS)
        logger.info(
            "field_channel_anomaly_scorer_loaded encoder_id=%s encoder_version=%s window_size=%d",
            self._manifest.encoder_id,
            self._manifest.encoder_version,
            self._manifest.window_size,
        )
        return True

    def _fetch_live_enrichment_fields(self, now: datetime) -> dict[str, float]:
        """Best-effort, per-tick. Returns {} (never raises) on any failure --
        the caller falls back to build_windows()'s existing 0.0-fill for
        whichever of the 6 live-enrichment channels stay absent, same as any
        other genuinely-missing channel."""
        if not self._postgres_uri:
            return {}
        if self._db_conn is None:
            self._db_conn = open_readonly_connection(
                self._postgres_uri, connect_timeout=5, statement_timeout_ms=2000
            )
            if self._db_conn is None:
                return {}
        try:
            return resolve_live_enrichment(
                self._db_conn, now=now, lookback_hours=self._live_enrichment_lookback_hours
            )
        except Exception:
            logger.warning("field_channel_anomaly_scorer_live_enrichment_failed", exc_info=True)
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None
            return {}

    def append_row(self, row: FieldChannelCorpusRowV1) -> None:
        """Called from _tick() every poll, independent of whether the JSONL
        corpus sink is enabled -- this buffer is in-memory only and serves a
        different purpose (live rescoring, not training-data collection).

        `row` is buffered via a fresh channels dict (model_copy, not
        in-place mutation) when live enrichment adds fields -- worker.py's
        _tick() passes this SAME row object to the JSONL corpus sink first,
        so mutating row.channels in place here would silently leak
        live-enrichment fields into the training corpus too, an unrelated
        consumer this class must not reach into."""
        if not self._ensure_loaded():
            return
        live_fields = self._fetch_live_enrichment_fields(row.generated_at)
        if live_fields:
            row = row.model_copy(update={"channels": {**row.channels, **live_fields}})
        self._buffer.append(row)

    def score_latest(self) -> FieldChannelAnomalyScoreV1 | None:
        """Scores the most recent complete window in the buffer, if any.
        Returns None when the encoder failed to load, fewer than
        window_size rows have accumulated yet, or the process is still
        within its startup grace period (2026-07-21: the first organic
        firing traced to a cold-start artifact -- reconcile-seeded defaults
        still in the buffer right after a restart) -- all normal, expected
        states, not errors. append_row() keeps buffering during the grace
        period regardless, so a real window is ready the moment it ends."""
        if not self._ensure_loaded():
            return None
        elapsed = (datetime.now(timezone.utc) - self._created_at).total_seconds()
        if elapsed < self._startup_grace_sec:
            return None
        rows = list(self._buffer)
        if len(rows) < self._manifest.window_size:
            return None

        scored = score_windows(
            rows,
            fields=tuple(self._manifest.channel_names),
            window_size=self._manifest.window_size,
            stride=self._manifest.stride,
            max_gap_sec=self._manifest.max_gap_sec,
            weights=self._weights,
        )
        if not scored:
            return None

        recon_loss, window_start, window_end = scored[-1]
        recon_error_p95 = float(self._manifest.training.recon_error_p95)
        threshold = recon_error_p95 * self._threshold_multiplier

        top_channels: list[str] = []
        signed_deviation = 0.0
        direction = "mixed"
        attribution_ok = False
        try:
            # Rebuilds windows from the same rows/params score_windows just
            # used -- score_windows() discards the raw vectors after
            # computing loss, and build_windows() is a cheap, public,
            # already-tested way to get them back without touching that
            # function's contract (other callers depend on it). The buffer
            # is ~window_size rows, so this is negligible extra work.
            vectors = build_windows(
                rows,
                fields=tuple(self._manifest.channel_names),
                window_size=self._manifest.window_size,
                stride=self._manifest.stride,
                max_gap_sec=self._manifest.max_gap_sec,
            )
            if vectors:
                last_window = vectors[-1]
                top_channels = top_channel_attribution(
                    last_window,
                    self._weights,
                    fields=tuple(self._manifest.channel_names),
                    window_size=self._manifest.window_size,
                    limit=3,
                )
                signed_deviation = mean_signed_deviation(
                    last_window,
                    self._weights,
                    window_size=self._manifest.window_size,
                    n_fields=len(self._manifest.channel_names),
                )
                direction = deviation_direction(signed_deviation)
                attribution_ok = True
        except Exception:
            logger.warning("field_channel_anomaly_attribution_failed", exc_info=True)

        return FieldChannelAnomalyScoreV1(
            correlation_id=str(uuid.uuid4()),
            encoder_id=self._manifest.encoder_id,
            encoder_version=self._manifest.encoder_version,
            recon_loss=float(recon_loss),
            recon_error_p95=recon_error_p95,
            threshold_multiplier=self._threshold_multiplier,
            threshold=threshold,
            anomalous=float(recon_loss) > threshold,
            mean_signed_deviation=signed_deviation,
            deviation_direction=direction,
            window_start=window_start,
            window_end=window_end,
            window_size=self._manifest.window_size,
            top_channels=top_channels,
            attribution_ok=attribution_ok,
        )
