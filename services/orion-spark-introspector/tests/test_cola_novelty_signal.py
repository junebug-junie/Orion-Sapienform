from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest

from app import worker
from app.settings import settings
from orion.core.bus.bus_schemas import BaseEnvelope


def setup_function(_fn) -> None:
    worker._COLA_NOVELTY_HISTORY.clear()
    worker._COLA_NOVELTY_HISTORY_ORDER.clear()
    worker._ACTIVE_SIGNALS.clear()


def test_novelty_reference_none_when_no_history() -> None:
    assert worker._cola_novelty_reference("session-a") is None


def test_novelty_remember_bounds_per_session_window(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_novelty_window", 3)
    for i in range(5):
        worker._cola_novelty_remember("session-a", np.full(4, float(i), dtype=np.float32))
    history = worker._COLA_NOVELTY_HISTORY["session-a"]
    assert len(history) == 3
    # Oldest entries (0, 1) should have been evicted; only 2, 3, 4 remain.
    assert [float(v[0]) for v in history] == [2.0, 3.0, 4.0]


def test_novelty_remember_bounds_total_session_count(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_novelty_max_sessions", 2)
    worker._cola_novelty_remember("session-a", np.zeros(4, dtype=np.float32))
    worker._cola_novelty_remember("session-b", np.zeros(4, dtype=np.float32))
    worker._cola_novelty_remember("session-c", np.zeros(4, dtype=np.float32))
    assert len(worker._COLA_NOVELTY_HISTORY) == 2
    assert "session-a" not in worker._COLA_NOVELTY_HISTORY
    assert "session-c" in worker._COLA_NOVELTY_HISTORY


@pytest.mark.asyncio
async def test_fetch_cola_understanding_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", False)
    with patch("app.worker.httpx.AsyncClient") as mock_client_cls:
        result = await worker._fetch_cola_understanding("hello", doc_id="d1")
    assert result is None
    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_cola_understanding_fails_open_on_http_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", True)

    class _RaisingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *_args, **_kwargs):
            raise RuntimeError("host unreachable")

    with patch("app.worker.httpx.AsyncClient", return_value=_RaisingClient()):
        result = await worker._fetch_cola_understanding("hello", doc_id="d1")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_cola_understanding_parses_embedding(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", True)

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [0.1, 0.2, 0.3], "embedding_dim": 3}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *_args, **_kwargs):
            return _Resp()

    with patch("app.worker.httpx.AsyncClient", return_value=_Client()):
        result = await worker._fetch_cola_understanding("hello", doc_id="d1")
    assert result is not None
    assert result.tolist() == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_score_cola_novelty_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", False)
    fetch_mock = AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32))
    publish_mock = AsyncMock()
    with patch("app.worker._fetch_cola_understanding", fetch_mock), patch(
        "app.worker._publish_cola_novelty_signal", publish_mock
    ):
        await worker._score_cola_novelty(
            text="hello", doc_id="d1", session_key="s1", correlation_id=uuid4()
        )
    fetch_mock.assert_not_called()
    publish_mock.assert_not_called()
    assert "s1" not in worker._COLA_NOVELTY_HISTORY


@pytest.mark.asyncio
async def test_score_cola_novelty_noop_when_host_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", True)
    fetch_mock = AsyncMock(return_value=None)  # simulates fail-open from _fetch_cola_understanding
    publish_mock = AsyncMock()
    with patch("app.worker._fetch_cola_understanding", fetch_mock), patch(
        "app.worker._publish_cola_novelty_signal", publish_mock
    ):
        await worker._score_cola_novelty(
            text="hello", doc_id="d1", session_key="s1", correlation_id=uuid4()
        )
    publish_mock.assert_not_called()
    assert "s1" not in worker._COLA_NOVELTY_HISTORY


@pytest.mark.asyncio
async def test_score_cola_novelty_first_turn_is_max_novelty(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", True)
    fetch_mock = AsyncMock(return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    publish_mock = AsyncMock()
    with patch("app.worker._fetch_cola_understanding", fetch_mock), patch(
        "app.worker._publish_cola_novelty_signal", publish_mock
    ):
        await worker._score_cola_novelty(
            text="hello", doc_id="d1", session_key="s1", correlation_id=uuid4()
        )
    publish_mock.assert_awaited_once()
    (distance, ), kwargs = publish_mock.call_args
    assert distance == pytest.approx(1.0)
    assert "s1" in worker._COLA_NOVELTY_HISTORY


@pytest.mark.asyncio
async def test_score_cola_novelty_repeated_turn_is_low_novelty(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cola_understand_enable", True)
    same_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    fetch_mock = AsyncMock(return_value=same_vec)
    publish_mock = AsyncMock()
    with patch("app.worker._fetch_cola_understanding", fetch_mock), patch(
        "app.worker._publish_cola_novelty_signal", publish_mock
    ):
        await worker._score_cola_novelty(
            text="hello", doc_id="d1", session_key="s1", correlation_id=uuid4()
        )
        await worker._score_cola_novelty(
            text="hello again", doc_id="d2", session_key="s1", correlation_id=uuid4()
        )
    assert publish_mock.await_count == 2
    second_distance = publish_mock.call_args_list[1].args[0]
    assert second_distance == pytest.approx(0.0, abs=1e-5)


@pytest.mark.asyncio
async def test_publish_cola_novelty_signal_reaches_phi_stats_via_signal_bus() -> None:
    """End-to-end (in-process) proof that a published novelty signal reaches
    the same phi_stats merge orion-cortex-exec reads for metacognition,
    without needing a live bus/model/service stack."""
    captured: dict = {}

    class _FakeBus:
        enabled = True

        async def publish(self, channel, envelope):
            captured["channel"] = channel
            captured["envelope"] = envelope

    worker._pub_bus = _FakeBus()
    try:
        await worker._publish_cola_novelty_signal(0.8, correlation_id=uuid4())
    finally:
        worker._pub_bus = None

    assert captured["channel"] == settings.channel_spark_signal
    env: BaseEnvelope = captured["envelope"]

    baseline = worker._get_phi_stats()
    adjusted_before = worker._apply_signal_deltas(dict(baseline))
    assert adjusted_before["novelty"] == pytest.approx(baseline["novelty"])

    await worker.handle_signal(env)
    adjusted_after = worker._apply_signal_deltas(dict(baseline))
    expected_delta = float(settings.cola_novelty_gain) * 0.8
    assert adjusted_after["novelty"] == pytest.approx(baseline["novelty"] + expected_delta)
