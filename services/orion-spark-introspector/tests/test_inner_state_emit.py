from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

import app.worker as worker
from app.substrate_reads import (
    ExecutionTrajectorySnapshot,
    GrammarTruthSnapshot,
    ReasoningActivitySnapshot,
)


def _mock_healthy_substrate_reads(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=False,
                degraded_reasons=[],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(return_value=ExecutionTrajectorySnapshot(ok=True, projection=None)),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(return_value=ReasoningActivitySnapshot(ok=True, projection=None)),
    )


@pytest.mark.asyncio
async def test_inner_state_tick_correlation_id_is_a_real_uuid(monkeypatch) -> None:
    """2026-07-22 (SelfStateV1 burn): was test_handle_self_state_uuid_crash_fixed,
    a regression for self_state_id (a non-UUID string) crashing pydantic's
    correlation_id coercion. run_inner_state_tick() generates its own tick_id
    the same shape self_state_id used to be (a non-UUID string) -- confirms
    the same UUID5-derivation pattern still holds: the envelope's
    correlation_id is coerced to a real UUID, while the human-readable
    tick_id is preserved in the payload."""
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            captured["channel"] = channel
            captured["env"] = env

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.run_inner_state_tick()

    snap_env = captured["env"]
    # Envelope correlation_id is coerced to a real UUID...
    assert isinstance(snap_env.correlation_id, UUID)
    # ...while the human-readable tick id is preserved in the payload.
    assert snap_env.payload.correlation_id.startswith("inner.tick:")


def test_inner_features_settings_defaults() -> None:
    from app.settings import Settings
    s = Settings()
    assert s.inner_features_enabled is True
    # seed-v5 (2026-07-22, SelfStateV1 burn): seed-v4 depended on SelfStateV1's
    # FELT_DIMENSIONS, which no longer exist.
    assert s.inner_features_version == "seed-v5"
    assert s.channel_inner_features == "orion:self:inner_features"
    assert s.phi_degenerate_streak == 20


@pytest.mark.asyncio
async def test_inner_state_tick_emits_inner_features(monkeypatch, tmp_path) -> None:
    published = []
    broadcasts = []

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published.append((channel, env))

    async def _capture_broadcast(payload):
        broadcasts.append(payload)

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", _capture_broadcast, raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_corpus_path", str(tmp_path / "c.jsonl"), raising=False)
    # fresh module state
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.run_inner_state_tick()

    channels = [c for c, _ in published]
    assert worker.settings.channel_inner_features in channels

    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"


@pytest.mark.asyncio
async def test_handle_trace_ws_phi_uses_last_headline(monkeypatch, tmp_path) -> None:
    """The main-path trace EKG frame (handle_trace, non-heartbeat) must show
    _INNER_LAST_HEADLINE once a tick has populated it, not a stale default."""
    published = []
    broadcasts = []

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published.append((channel, env))

    async def _capture_broadcast(payload):
        broadcasts.append(payload)

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", _capture_broadcast, raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_corpus_path", str(tmp_path / "c.jsonl"), raising=False)
    # fresh module state
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)
    # Force a known headline directly -- no self-state-derived or encoder
    # phi source is guaranteed in this test's mocked environment, so drive
    # the value the trace path is expected to read rather than depend on
    # run_inner_state_tick() producing one incidentally.
    monkeypatch.setattr(worker, "_INNER_LAST_HEADLINE", 0.71, raising=False)
    _mock_healthy_substrate_reads(monkeypatch)

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    # Non-heartbeat trace; spark_meta appraisal makes display_novelty non-None so
    # the 1428 tissue.update frame actually broadcasts.
    trace_payload = {
        "mode": "reasoning",
        "verb": "respond",
        "correlation_id": "trace-corr-ekg",
        "steps": [],
        "metadata": {
            "spark_meta": {
                "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.3}
            }
        },
    }
    trace_env = BaseEnvelope(
        kind="cognition.trace.v1",
        source=ServiceRef(name="cortex-exec", node="athena"),
        payload=trace_payload,
    )
    await worker.handle_trace(trace_env)

    trace_frames = [
        b for b in broadcasts
        if b.get("type") == "tissue.update" and b.get("correlation_id") == "trace-corr-ekg"
    ]
    assert trace_frames, "expected a trace-path tissue.update broadcast"
    assert trace_frames[-1]["stats"]["phi"] == pytest.approx(0.71)


@pytest.mark.asyncio
async def test_inner_state_tick_grammar_truth_freeze(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=True,
                degraded_reasons=["cursor_lag:execution_grammar_reducer"],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(return_value=ExecutionTrajectorySnapshot(ok=False, projection=None)),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(return_value=ReasoningActivitySnapshot(ok=False, projection=None)),
    )
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            if channel == worker.settings.channel_inner_features:
                captured["payload"] = env.payload

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    await worker.run_inner_state_tick()
    assert captured["payload"].phi_health == "frozen"


@pytest.mark.asyncio
async def test_inner_state_tick_includes_cognitive_features(monkeypatch) -> None:
    now = worker.datetime.now(worker.timezone.utc)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=False,
                degraded_reasons=[],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(
            return_value=ExecutionTrajectorySnapshot(
                ok=True,
                projection={
                    "runs": {
                        "a": {
                            "reasoning_present": True,
                            "recall_observed": True,
                            "step_count": 4,
                            "failed_step_count": 0,
                            "pressure_hints": {},
                            "last_updated_at": now.isoformat(),
                        }
                    }
                },
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(return_value=ReasoningActivitySnapshot(ok=False, projection=None)),
    )
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            if channel == worker.settings.channel_inner_features:
                captured["payload"] = env.payload

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    await worker.run_inner_state_tick()
    names = {f.name for f in captured["payload"].features}
    assert "reasoning_present" in names


@pytest.mark.asyncio
async def test_inner_state_tick_healthy_row_appends_to_corpus(monkeypatch) -> None:
    """A healthy row (phi_health='ok', grammar not degraded, cognitive features
    backed by a real execution-trajectory run rather than all '.none') must
    reach the phi training corpus sink."""
    now = worker.datetime.now(worker.timezone.utc)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=False,
                degraded_reasons=[],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(
            return_value=ExecutionTrajectorySnapshot(
                ok=True,
                projection={
                    "runs": {
                        "a": {
                            "reasoning_present": True,
                            "recall_observed": True,
                            "step_count": 4,
                            "failed_step_count": 0,
                            "pressure_hints": {},
                            "last_updated_at": now.isoformat(),
                        }
                    }
                },
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(return_value=ReasoningActivitySnapshot(ok=False, projection=None)),
    )

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            pass

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    mock_sink = MagicMock()
    monkeypatch.setattr(worker, "_INNER_SINK", mock_sink, raising=False)

    await worker.run_inner_state_tick()

    mock_sink.append.assert_called_once()


@pytest.mark.asyncio
async def test_inner_state_tick_unhealthy_row_skips_corpus(monkeypatch) -> None:
    """A frozen/degraded row must NOT reach the phi training corpus sink —
    garbage is rejected at the write boundary, not filtered later at fit time."""
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=True,
                degraded_reasons=["cursor_lag:execution_grammar_reducer"],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(return_value=ExecutionTrajectorySnapshot(ok=False, projection=None)),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(return_value=ReasoningActivitySnapshot(ok=False, projection=None)),
    )

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            pass

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    mock_sink = MagicMock()
    monkeypatch.setattr(worker, "_INNER_SINK", mock_sink, raising=False)

    await worker.run_inner_state_tick()

    mock_sink.append.assert_not_called()


@pytest.mark.asyncio
async def test_inner_state_tick_seed_v4_uses_reasoning_activity_signals(monkeypatch) -> None:
    """With INNER_FEATURES_VERSION=seed-v4 and both substrate + orion-thought
    reads mocked, the emitted row carries the seed-v4 cognitive slot names
    (execution_load/reasoning_load/reasoning_present) instead of seed-v3's
    exec_step_fail_rate/execution_friction pair."""
    now = worker.datetime.now(worker.timezone.utc)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v4", raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=False,
                degraded_reasons=[],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(
            return_value=ExecutionTrajectorySnapshot(
                ok=True,
                projection={
                    "runs": {
                        "a": {
                            "reasoning_present": True,
                            "recall_observed": True,
                            "step_count": 4,
                            "failed_step_count": 0,
                            "pressure_hints": {},
                            "last_updated_at": now.isoformat(),
                        }
                    }
                },
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_reasoning_activity",
        AsyncMock(
            return_value=ReasoningActivitySnapshot(
                ok=True,
                projection={
                    "call_count": 8,
                    "reasoning_present_rate": 0.5,
                    "completion_tokens_sum": 200,
                    "thinking_tokens_sum": None,
                },
            )
        ),
    )
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            if channel == worker.settings.channel_inner_features:
                captured["payload"] = env.payload

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    await worker.run_inner_state_tick()

    names = {f.name for f in captured["payload"].features}
    assert "execution_load" in names
    assert "reasoning_load" in names
    assert "reasoning_present" in names
    assert "exec_step_fail_rate" not in names
    assert "execution_friction" not in names


@pytest.mark.asyncio
async def test_inner_state_tick_corpus_gate_uses_seedv4_cognitive_names(monkeypatch) -> None:
    """The corpus-health gate must check the cognitive names that actually
    exist on the row: seed-v4 rows carry execution_load/reasoning_load, not
    seed-v3's exec_step_fail_rate/execution_friction. Using the wrong name set
    would silently narrow the "all cognitive features dead" check."""
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v4", raising=False)
    monkeypatch.setattr(worker, "fetch_grammar_truth", AsyncMock(
        return_value=GrammarTruthSnapshot(
            degraded=False, degraded_reasons=[],
            enabled_reducers={"execution_trajectory": True},
            reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
        )
    ))
    monkeypatch.setattr(worker, "fetch_execution_trajectory", AsyncMock(
        return_value=ExecutionTrajectorySnapshot(ok=True, projection=None)
    ))
    monkeypatch.setattr(worker, "fetch_reasoning_activity", AsyncMock(
        return_value=ReasoningActivitySnapshot(ok=False, projection=None)
    ))

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            pass

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)
    monkeypatch.setattr(worker, "_INNER_SINK", MagicMock(), raising=False)

    captured_names = {}
    real_gate = worker.is_corpus_row_healthy

    def _spy(inner, *, cognitive_feature_names):
        captured_names["names"] = cognitive_feature_names
        return real_gate(inner, cognitive_feature_names=cognitive_feature_names)

    monkeypatch.setattr(worker, "is_corpus_row_healthy", _spy)

    await worker.run_inner_state_tick()

    assert captured_names["names"] == worker.SEEDV4_COGNITIVE_FEATURE_NAMES
    assert "exec_step_fail_rate" not in captured_names["names"]


@pytest.mark.asyncio
async def test_inner_state_tick_unrelated_cursor_lag_does_not_freeze_or_reject(monkeypatch) -> None:
    """Regression for a live incident: substrate reporting degraded=True
    solely because chat_grammar_consumer (unrelated to phi) is lagging must
    NOT freeze phi_health or trip the corpus-hygiene gate, as long as
    execution_trajectory (the reducer phi's cognitive features depend on) is
    itself healthy."""
    now = worker.datetime.now(worker.timezone.utc)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(worker, "fetch_grammar_truth", AsyncMock(
        return_value=GrammarTruthSnapshot(
            degraded=True,
            degraded_reasons=["cursor_lag:chat_grammar_consumer"],
            enabled_reducers={"execution_trajectory": True},
            reducer_health_by_name={"execution_trajectory": {"classification": "healthy"}},
        )
    ))
    monkeypatch.setattr(worker, "fetch_execution_trajectory", AsyncMock(
        return_value=ExecutionTrajectorySnapshot(
            ok=True,
            projection={
                "runs": {
                    "a": {
                        "reasoning_present": True,
                        "recall_observed": True,
                        "step_count": 4,
                        "failed_step_count": 0,
                        "pressure_hints": {},
                        "last_updated_at": now.isoformat(),
                    }
                }
            },
        )
    ))
    monkeypatch.setattr(worker, "fetch_reasoning_activity", AsyncMock(
        return_value=ReasoningActivitySnapshot(
            ok=True,
            projection={
                "call_count": 8,
                "reasoning_present_rate": 0.5,
                "completion_tokens_sum": 200,
                "thinking_tokens_sum": None,
            },
        )
    ))
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            if channel == worker.settings.channel_inner_features:
                captured["payload"] = env.payload

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)

    mock_sink = MagicMock()
    monkeypatch.setattr(worker, "_INNER_SINK", mock_sink, raising=False)

    await worker.run_inner_state_tick()

    assert captured["payload"].phi_health == "ok"
    assert captured["payload"].grammar_truth_degraded is False
    mock_sink.append.assert_called_once()
