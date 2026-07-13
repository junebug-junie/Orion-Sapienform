from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.orchestrator as orchestrator
from app.grammar_emit import build_route_arbitration_grammar_events
from app.grammar_publish import publish_orch_route_grammar_trace
from app.settings import Settings
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest


_SAMPLE_ROUTE_METADATA = {
    "execution_lane": "spark",
    "execution_lane_reason": "lane_routing_disabled",
    "mind_requested": False,
    "mind_skip_reason": "mind_enabled_not_true",
    "output_mode": "concise",
}


def _spark_req() -> CortexClientRequest:
    return CortexClientRequest(
        mode="brain",
        verb_name="introspect_spark",
        packs=[],
        options={"timeout_sec": 5.0},
        context=CortexClientContext(
            messages=[],
            raw_user_text="x",
            user_message="x",
            session_id="s",
            user_id="u",
            trace_id="trace-spark",
            metadata={},
        ),
    )


def _settings_routing_on(**overrides: object) -> Settings:
    base = dict(
        exec_lane_routing_enabled=True,
        channel_exec_request="orion:cortex:exec:request",
        channel_exec_request_chat="orion:cortex:exec:request:chat",
        channel_exec_request_spark="orion:cortex:exec:request:spark",
        channel_exec_request_background="orion:cortex:exec:request:background",
        channel_exec_result_prefix="orion:exec:result",
        orion_state_enabled=False,
    )
    base.update(overrides)
    return Settings.model_construct(**base)


# --- 1. build_route_arbitration_grammar_events -----------------------------


def test_build_route_arbitration_grammar_events_shape() -> None:
    events = build_route_arbitration_grammar_events(
        correlation_id="corr-123",
        node_id="athena",
        route_metadata=_SAMPLE_ROUTE_METADATA,
        session_id="sess-1",
        turn_id="turn-1",
    )

    assert len(events) == 2

    trace_started, atom_emitted = events
    assert trace_started.event_kind == "trace_started"
    assert atom_emitted.event_kind == "atom_emitted"

    for event in events:
        assert event.trace_id == "orch.route:athena:corr-123"
        assert event.trace_id.startswith("orch.route:")
        assert event.provenance.source_service == "orion-cortex-orch"
        assert event.layer == "route"
        assert event.dimensions == ["route", "arbitration"]
        assert event.session_id == "sess-1"
        assert event.turn_id == "turn-1"

    atom = atom_emitted.atom
    assert atom is not None
    assert atom.atom_type == "reasoning_step"
    assert atom.semantic_role == "route_arbitration_decided"
    assert atom.layer == "route"
    assert atom.dimensions == ["route", "arbitration"]
    assert atom.confidence == 1.0
    assert atom.text_value is None

    # Round-trip through the same kv-parse regex the route_loop consumer uses
    # (orion/substrate/route_loop/grammar_extract.py's _KV_RE).
    kv_re = re.compile(r"(\w+)=([^,;\s]+)")
    kv = dict(kv_re.findall(atom.summary))
    assert kv["lane"] == "spark"
    assert kv["lane_reason"] == "lane_routing_disabled"
    assert kv["mind_requested"] == "false"
    assert kv["mind_skip_reason"] == "mind_enabled_not_true"
    assert kv["output_mode"] == "concise"


def test_build_route_arbitration_grammar_events_unsafe_values_sanitized() -> None:
    """Values containing commas/semicolons/whitespace must not break the kv-parse."""
    dirty_metadata = {
        "execution_lane": "spark, background",
        "execution_lane_reason": "reason; with semicolon and spaces",
        "mind_requested": True,
        "mind_skip_reason": None,
        "output_mode": None,
    }
    events = build_route_arbitration_grammar_events(
        correlation_id="corr-dirty",
        node_id="athena",
        route_metadata=dirty_metadata,
    )
    atom = events[1].atom
    assert atom is not None
    assert "," not in atom.summary.split("lane=", 1)[1].split(",lane_reason=")[0]

    kv_re = re.compile(r"(\w+)=([^,;\s]+)")
    kv = dict(kv_re.findall(atom.summary))
    assert kv["lane"] == "spark_background"
    assert kv["mind_requested"] == "true"
    assert kv["mind_skip_reason"] == "none"
    assert kv["output_mode"] == "unknown"


def test_build_route_arbitration_grammar_events_session_turn_optional() -> None:
    events = build_route_arbitration_grammar_events(
        correlation_id="corr-noctx",
        node_id="athena",
        route_metadata=_SAMPLE_ROUTE_METADATA,
    )
    for event in events:
        assert event.session_id is None
        assert event.turn_id is None


# --- 2. publish_orch_route_grammar_trace ------------------------------------


def test_publish_orch_route_grammar_trace_noop_when_disabled() -> None:
    bus = MagicMock()
    bus.publish = AsyncMock()
    events = build_route_arbitration_grammar_events(
        correlation_id="corr-1",
        node_id="athena",
        route_metadata=_SAMPLE_ROUTE_METADATA,
    )

    asyncio.run(
        publish_orch_route_grammar_trace(
            bus,
            events,
            correlation_id="corr-1",
            channel="orion:grammar:event",
            enabled=False,
        )
    )

    bus.publish.assert_not_called()


def test_publish_orch_route_grammar_trace_publishes_when_enabled() -> None:
    bus = MagicMock()
    bus.publish = AsyncMock()
    events = build_route_arbitration_grammar_events(
        correlation_id="corr-2",
        node_id="athena",
        route_metadata=_SAMPLE_ROUTE_METADATA,
    )

    asyncio.run(
        publish_orch_route_grammar_trace(
            bus,
            events,
            correlation_id="corr-2",
            channel="orion:grammar:event",
            enabled=True,
        )
    )

    assert bus.publish.call_count == len(events)
    called_channel = bus.publish.call_args_list[0].args[0]
    assert called_channel == "orion:grammar:event"


def test_publish_orch_route_grammar_trace_noop_on_empty_events() -> None:
    bus = MagicMock()
    bus.publish = AsyncMock()

    asyncio.run(
        publish_orch_route_grammar_trace(
            bus,
            [],
            correlation_id="corr-empty",
            channel="orion:grammar:event",
            enabled=True,
        )
    )

    bus.publish.assert_not_called()


def test_publish_orch_route_grammar_trace_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A publish failure must be swallowed, not raised."""
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    events = build_route_arbitration_grammar_events(
        correlation_id="corr-fail",
        node_id="athena",
        route_metadata=_SAMPLE_ROUTE_METADATA,
    )

    # Must not raise.
    asyncio.run(
        publish_orch_route_grammar_trace(
            bus,
            events,
            correlation_id="corr-fail",
            channel="orion:grammar:event",
            enabled=True,
        )
    )


# --- 3. call_verb_runtime smoke: publish failure never affects the response -


def test_call_verb_runtime_grammar_publish_failure_does_not_affect_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_cortex_exec(_bus, *, exec_request_channel: str, plan_request=None, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "status": "success",
            "verb_name": "introspect_spark",
            "final_text": "ok",
            "steps": [],
            "memory_used": False,
            "recall_debug": {},
            "metadata": {},
        }

    monkeypatch.setattr(orchestrator, "call_cortex_exec", fake_call_cortex_exec)
    monkeypatch.setattr(
        orchestrator,
        "get_settings",
        lambda: _settings_routing_on(publish_cortex_orch_grammar=True),
    )

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(orchestrator, "_maybe_fetch_state", _none)

    # Publish raises -- must not propagate into call_verb_runtime's return.
    # orchestrator.py does `from .grammar_publish import publish_orch_route_grammar_trace`
    # as a local import inside call_verb_runtime, so patch the attribute on the
    # source module (app.grammar_publish) rather than on `orchestrator` itself --
    # a local `from x import y` re-resolves `y` off the module at call time.
    import app.grammar_publish as grammar_publish_module

    async def _boom(*_a, **_k):
        raise RuntimeError("publish exploded")

    monkeypatch.setattr(grammar_publish_module, "publish_orch_route_grammar_trace", _boom)

    bus = MagicMock()

    async def _run() -> None:
        res = await orchestrator.call_verb_runtime(
            bus,
            source=ServiceRef(name="cortex-orch", version="0", node="n"),
            client_request=_spark_req(),
            correlation_id="corr-grammar-fail",
            timeout_sec=5.0,
        )
        assert res.ok is True
        assert isinstance(res.output, dict)
        route_meta = res.output.get("_route_metadata")
        assert route_meta is not None
        assert route_meta["execution_lane"] == "spark"
        assert res.output.get("result", {}).get("status") == "success"

    asyncio.run(_run())
