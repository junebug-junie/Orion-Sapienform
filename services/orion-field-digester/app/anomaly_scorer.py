from __future__ import annotations

import logging
import uuid
from collections import deque
from pathlib import Path

from orion.mood_arc.fit_encoder import load_artifacts, score_windows
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
    or malformed encoder_dir disables scoring for this process's lifetime
    (logged once) rather than crashing the tick loop that calls append_row()
    every ~2s.
    """

    def __init__(self, *, encoder_dir: str, threshold_multiplier: float) -> None:
        self._encoder_dir = encoder_dir
        self._threshold_multiplier = float(threshold_multiplier)
        self._manifest = None
        self._weights: dict | None = None
        self._load_failed = False
        self._buffer: deque[FieldChannelCorpusRowV1] = deque()

    def _ensure_loaded(self) -> bool:
        if self._load_failed:
            return False
        if self._manifest is not None:
            return True
        if not self._encoder_dir:
            self._load_failed = True
            logger.warning("field_channel_anomaly_scorer_no_encoder_dir")
            return False
        try:
            self._manifest, self._weights = load_artifacts(Path(self._encoder_dir))
        except Exception:
            self._load_failed = True
            logger.warning(
                "field_channel_anomaly_scorer_load_failed encoder_dir=%s",
                self._encoder_dir,
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

    def append_row(self, row: FieldChannelCorpusRowV1) -> None:
        """Called from _tick() every poll, independent of whether the JSONL
        corpus sink is enabled -- this buffer is in-memory only and serves a
        different purpose (live rescoring, not training-data collection)."""
        if not self._ensure_loaded():
            return
        self._buffer.append(row)

    def score_latest(self) -> FieldChannelAnomalyScoreV1 | None:
        """Scores the most recent complete window in the buffer, if any.
        Returns None when the encoder failed to load or fewer than
        window_size rows have accumulated yet -- both are normal, expected
        states (a freshly-started process, or scoring disabled), not
        errors."""
        if not self._ensure_loaded():
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
        return FieldChannelAnomalyScoreV1(
            correlation_id=str(uuid.uuid4()),
            encoder_id=self._manifest.encoder_id,
            encoder_version=self._manifest.encoder_version,
            recon_loss=float(recon_loss),
            recon_error_p95=recon_error_p95,
            threshold_multiplier=self._threshold_multiplier,
            threshold=threshold,
            anomalous=float(recon_loss) > threshold,
            window_start=window_start,
            window_end=window_end,
            window_size=self._manifest.window_size,
        )
