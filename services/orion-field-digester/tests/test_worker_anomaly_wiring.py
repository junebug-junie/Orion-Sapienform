from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import app.worker as worker_module
from app.anomaly_scorer import FieldChannelAnomalyScorer
from app.worker import FieldDigesterWorker


def _make_worker(monkeypatch, *, anomaly_enabled: str = "false") -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_CHANNEL_ANOMALY_ENABLED", anomaly_enabled)
    monkeypatch.setenv("FIELD_CHANNEL_ANOMALY_ENCODER_DIR", "/nonexistent/encoder")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._anomaly_scorer = None
    if worker._settings.field_channel_anomaly_enabled:
        worker._anomaly_scorer = FieldChannelAnomalyScorer(
            encoder_dir=worker._settings.field_channel_anomaly_encoder_dir,
            threshold_multiplier=worker._settings.field_channel_anomaly_threshold_multiplier,
        )
    worker._stop = MagicMock()
    return worker


def test_anomaly_scorer_is_none_when_disabled(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, anomaly_enabled="false")
    assert worker._anomaly_scorer is None


def test_anomaly_scorer_is_constructed_when_enabled(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, anomaly_enabled="true")
    assert isinstance(worker._anomaly_scorer, FieldChannelAnomalyScorer)


@pytest.mark.asyncio
async def test_anomaly_tick_publishes_when_scorer_returns_a_score(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, anomaly_enabled="true")
    fake_score = MagicMock(anomalous=True, correlation_id="c1", recon_loss=0.02, threshold=0.004,
                            window_start="ws", window_end="we")
    monkeypatch.setattr(worker._anomaly_scorer, "score_latest", MagicMock(return_value=fake_score))
    mock_publish = AsyncMock()
    monkeypatch.setattr(worker_module, "publish_anomaly_score", mock_publish)

    await worker._anomaly_tick()

    mock_publish.assert_awaited_once()
    kwargs = mock_publish.call_args.kwargs
    assert kwargs["score"] is fake_score
    assert kwargs["channel"] == worker._settings.channel_field_channel_anomaly_score


@pytest.mark.asyncio
async def test_anomaly_tick_does_not_publish_when_scorer_returns_none(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, anomaly_enabled="true")
    monkeypatch.setattr(worker._anomaly_scorer, "score_latest", MagicMock(return_value=None))
    mock_publish = AsyncMock()
    monkeypatch.setattr(worker_module, "publish_anomaly_score", mock_publish)

    await worker._anomaly_tick()

    mock_publish.assert_not_awaited()
