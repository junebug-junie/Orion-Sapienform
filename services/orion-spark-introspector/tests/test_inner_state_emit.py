from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

import app.worker as worker
from app.substrate_reads import (
    ExecutionTrajectorySnapshot,
    GrammarTruthSnapshot,
    ReasoningActivitySnapshot,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

# NOTE: handle_self_state does `SelfStateV1.model_validate(payload)` inside a
# try/except that swallows ValidationError and returns early. A malformed
# payload would therefore NOT exercise the UUID crash at all — the test would
# be testing nothing. So build a REAL, valid SelfStateV1 and dump it. The
# self_state_id is intentionally a non-UUID string (that is the bug under test);
# it lives in the payload, not in a UUID field, so it is valid here.
_NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
_SCORES = {
    "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
    "execution_pressure": 0.0, "reasoning_pressure": 0.05,
    "resource_pressure": 1.0, "reliability_pressure": 1.0,
    "continuity_pressure": 0.0, "introspection_pressure": 0.0,
    "social_pressure": 0.0, "uncertainty": 0.0, "policy_pressure": 0.0,
}


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:tick_abc:policy.v1",  # non-UUID on purpose
        generated_at=_NOW,
        source_field_tick_id="tick_abc",
        source_field_generated_at=_NOW,
        source_attention_frame_id="frame_abc",
        source_attention_generated_at=_NOW,
        overall_intensity=0.4,
        overall_confidence=0.6,
        overall_condition="steady",
        trajectory_condition="stable",
        dimensions={
            k: SelfStateDimensionV1(dimension_id=k, score=v, confidence=0.6)
            for k, v in _SCORES.items()
        },
        dominant_field_channels={
            "contract_pressure": 1.0, "catalog_drift_pressure": 1.0, "bus_health": 1.0,
        },
        dimension_trajectory={},
    )


def _self_state_payload() -> dict:
    return _self_state().model_dump(mode="json")


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
async def test_handle_self_state_uuid_crash_fixed(monkeypatch) -> None:
    captured = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            captured["channel"] = channel
            captured["env"] = env

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _mock_healthy_substrate_reads(monkeypatch)

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )

    # Must NOT raise pydantic ValidationError on correlation_id.
    await worker.handle_self_state(env)

    snap_env = captured["env"]
    # Envelope correlation_id is coerced to a real UUID...
    assert isinstance(snap_env.correlation_id, UUID)
    # ...while the human-readable id is preserved in the payload.
    assert snap_env.payload.correlation_id == "self.state:tick_abc:policy.v1"


def test_inner_features_settings_defaults() -> None:
    from app.settings import Settings
    s = Settings()
    assert s.inner_features_enabled is True
    # Flipped seed-v3 -> seed-v4 once specs 1-3 of the phi corpus-honesty
    # initiative were merged and deployed (chore/enable-reasoning-telemetry-seedv4).
    assert s.inner_features_version == "seed-v4"
    assert s.channel_inner_features == "orion:self:inner_features"
    assert s.phi_degenerate_streak == 20
    assert s.orion_phi_encoder_enabled is False


@pytest.mark.asyncio
async def test_handle_self_state_emits_inner_features_and_honest_phi(monkeypatch, tmp_path) -> None:
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

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    channels = [c for c, _ in published]
    assert worker.settings.channel_inner_features in channels

    # the tissue.update carries the HONEST headline (>0.5), not the 0.01 floor
    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"
    assert tissue[-1]["stats"]["phi"] > 0.5


@pytest.mark.asyncio
async def test_handle_trace_ws_phi_uses_honest_headline(monkeypatch, tmp_path) -> None:
    """The main-path trace EKG frame (handle_trace, non-heartbeat) must show the
    honest headline, not the geometric-coherence floor carried by telem.phi."""
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
    monkeypatch.setattr(worker, "_INNER_LAST_HEADLINE", None, raising=False)
    _mock_healthy_substrate_reads(monkeypatch)

    # Drive a real self-state tick first: this populates _INNER_LAST_HEADLINE
    # (~0.70 honest headline) and _LATEST_SELF_STATE (source for _get_phi_stats).
    ss_env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(ss_env)
    assert worker._INNER_LAST_HEADLINE is not None and worker._INNER_LAST_HEADLINE > 0.5

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
    assert trace_frames[-1]["stats"]["phi"] > 0.5


@pytest.mark.asyncio
async def test_handle_self_state_grammar_truth_freeze(monkeypatch) -> None:
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
    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)
    assert captured["payload"].phi_health == "frozen"


@pytest.mark.asyncio
async def test_handle_self_state_includes_cognitive_features(monkeypatch) -> None:
    now = _NOW
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
    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)
    names = {f.name for f in captured["payload"].features}
    assert "reasoning_present" in names


@pytest.mark.asyncio
async def test_handle_self_state_healthy_row_appends_to_corpus(monkeypatch) -> None:
    """A healthy row (phi_health='ok', grammar not degraded, cognitive features
    backed by a real execution-trajectory run rather than all '.none') must
    reach the phi training corpus sink."""
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
                            "last_updated_at": _NOW.isoformat(),
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

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    mock_sink.append.assert_called_once()


@pytest.mark.asyncio
async def test_handle_self_state_unhealthy_row_skips_corpus(monkeypatch) -> None:
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

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    mock_sink.append.assert_not_called()


@pytest.mark.asyncio
async def test_handle_self_state_seed_v4_uses_reasoning_activity_signals(monkeypatch) -> None:
    """With INNER_FEATURES_VERSION=seed-v4 and both substrate + orion-thought
    reads mocked, the emitted row carries the seed-v4 cognitive slot names
    (execution_load/reasoning_load/reasoning_present) instead of seed-v3's
    exec_step_fail_rate/execution_friction pair."""
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
                            "last_updated_at": _NOW.isoformat(),
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

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    names = {f.name for f in captured["payload"].features}
    assert "execution_load" in names
    assert "reasoning_load" in names
    assert "reasoning_present" in names
    assert "exec_step_fail_rate" not in names
    assert "execution_friction" not in names


@pytest.mark.asyncio
async def test_handle_self_state_corpus_gate_uses_seedv4_cognitive_names(monkeypatch) -> None:
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

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    assert captured_names["names"] == worker.SEEDV4_COGNITIVE_FEATURE_NAMES
    assert "exec_step_fail_rate" not in captured_names["names"]


@pytest.mark.asyncio
async def test_handle_self_state_unrelated_cursor_lag_does_not_freeze_or_reject(monkeypatch) -> None:
    """Regression for the live incident: substrate reporting degraded=True
    solely because chat_grammar_consumer (unrelated to phi) is lagging must
    NOT freeze phi_health or trip the corpus-hygiene gate, as long as
    execution_trajectory (the reducer phi's cognitive features depend on) is
    itself healthy."""
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
                        "last_updated_at": _NOW.isoformat(),
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

    env = BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )
    await worker.handle_self_state(env)

    assert captured["payload"].phi_health == "ok"
    assert captured["payload"].grammar_truth_degraded is False
    mock_sink.append.assert_called_once()
