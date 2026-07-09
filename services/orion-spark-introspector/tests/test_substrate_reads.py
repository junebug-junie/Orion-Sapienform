from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from app.substrate_reads import (
    GrammarTruthSnapshot,
    SubstrateReadCache,
    cognitive_lane_dark,
    fetch_execution_trajectory,
    fetch_grammar_truth,
    fetch_reasoning_activity,
)


@pytest.mark.asyncio
async def test_fetch_grammar_truth_degraded(monkeypatch) -> None:
    client = AsyncMock()
    client.get.return_value.json.return_value = {"ok": False, "degraded": True, "degraded_reasons": ["x"]}
    client.get.return_value.raise_for_status = lambda: None
    out = await fetch_grammar_truth(client, "http://substrate/grammar/truth")
    assert out.degraded is True
    assert out.degraded_reasons == ["x"]


@pytest.mark.asyncio
async def test_fetch_grammar_truth_timeout_fail_closed(monkeypatch) -> None:
    client = AsyncMock()
    client.get.side_effect = TimeoutError("slow")
    out = await fetch_grammar_truth(client, "http://substrate/grammar/truth")
    assert out.degraded is True
    assert "http_error" in out.degraded_reasons[0]


def test_cache_reuses_within_ttl() -> None:
    cache = SubstrateReadCache(ttl_sec=2.0)
    cache.put_grammar({"degraded": False})
    assert cache.get_grammar() == {"degraded": False}


@pytest.mark.asyncio
async def test_fetch_reasoning_activity_happy_path() -> None:
    client = AsyncMock()
    projection = {
        "call_count": 5,
        "reasoning_present_rate": 0.2,
        "completion_tokens_sum": 100,
        "thinking_tokens_sum": None,
    }
    client.get.return_value.json.return_value = {"ok": True, "projection": projection}
    client.get.return_value.raise_for_status = lambda: None
    out = await fetch_reasoning_activity(client, "http://thought/projections/reasoning_activity")
    assert out.ok is True
    assert out.projection == projection


@pytest.mark.asyncio
async def test_fetch_reasoning_activity_http_error_fails_closed() -> None:
    client = AsyncMock()
    client.get.side_effect = TimeoutError("slow")
    out = await fetch_reasoning_activity(client, "http://thought/projections/reasoning_activity")
    assert out.ok is False
    assert out.projection is None


@pytest.mark.asyncio
async def test_fetch_reasoning_activity_non_dict_projection_fails_closed() -> None:
    client = AsyncMock()
    client.get.return_value.json.return_value = {"ok": True, "projection": "not-a-dict"}
    client.get.return_value.raise_for_status = lambda: None
    out = await fetch_reasoning_activity(client, "http://thought/projections/reasoning_activity")
    assert out.ok is True
    assert out.projection is None


def test_cache_reuses_reasoning_activity_within_ttl() -> None:
    cache = SubstrateReadCache(ttl_sec=2.0)
    cache.put_reasoning_activity({"ok": True, "projection": {"call_count": 1}})
    assert cache.get_reasoning_activity() == {"ok": True, "projection": {"call_count": 1}}


def test_cache_reasoning_activity_expires_past_ttl() -> None:
    cache = SubstrateReadCache(ttl_sec=0.0)
    cache.put_reasoning_activity({"ok": True, "projection": {}})
    time.sleep(0.01)
    assert cache.get_reasoning_activity() is None


# --- cognitive_lane_dark ------------------------------------------------------
# Regression coverage for the live bug where an UNRELATED reducer's cursor lag
# (chat_grammar_consumer going quiet because no chat happened) tripped
# substrate's blanket `degraded` flag, which spark-introspector used to OR
# into grammar_degraded -- freezing phi_health, and therefore rejecting every
# corpus row at spec 3's ingestion-time health gate, even though
# execution_trajectory (the only reducer phi's cognitive features actually
# depend on) was fully healthy.


def _snapshot(
    *,
    degraded_reasons: list[str] | None = None,
    enabled: bool = True,
    classification: str | None = "healthy",
) -> GrammarTruthSnapshot:
    reducer_health_by_name = {"execution_trajectory": {"classification": classification}} if classification else {}
    return GrammarTruthSnapshot(
        degraded=bool(degraded_reasons),
        degraded_reasons=degraded_reasons or [],
        enabled_reducers={"execution_trajectory": enabled},
        reducer_health_by_name=reducer_health_by_name,
    )


def test_cognitive_lane_dark_false_when_unrelated_reducer_lags() -> None:
    """The exact live bug: chat_grammar_consumer lagging must NOT freeze phi."""
    snap = _snapshot(degraded_reasons=["cursor_lag:chat_grammar_consumer"])
    assert cognitive_lane_dark(snap) is False


def test_cognitive_lane_dark_true_when_execution_trajectory_cursor_lags() -> None:
    """execution_trajectory's OWN cursor lagging must still freeze phi, even
    though `classification` (heartbeat/stream-backlog only) doesn't reflect
    wall-clock cursor lag at all."""
    snap = _snapshot(degraded_reasons=["cursor_lag:execution_grammar_reducer"])
    assert cognitive_lane_dark(snap) is True


def test_cognitive_lane_dark_true_when_multiple_reducers_lag_including_own() -> None:
    snap = _snapshot(
        degraded_reasons=["cursor_lag:chat_grammar_consumer", "cursor_lag:execution_grammar_reducer"]
    )
    assert cognitive_lane_dark(snap) is True


def test_cognitive_lane_dark_true_on_dark_classification_even_without_lag_reason() -> None:
    snap = _snapshot(degraded_reasons=[], classification="dead_no_heartbeat")
    assert cognitive_lane_dark(snap) is True


def test_cognitive_lane_dark_true_when_execution_trajectory_reducer_disabled() -> None:
    snap = _snapshot(degraded_reasons=[], enabled=False)
    assert cognitive_lane_dark(snap) is True


def test_cognitive_lane_dark_false_when_fully_healthy_no_reasons() -> None:
    snap = _snapshot(degraded_reasons=[])
    assert cognitive_lane_dark(snap) is False
