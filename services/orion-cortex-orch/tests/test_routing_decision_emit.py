"""The router must actually publish what it decided, with the gate's effect intact.

The contract tests for this record only pin the schema and the channel entry.
That is not enough: a producer that records `gate_demoted=False` on a demoted
turn, or reads the post-gate depth as the pre-gate one, satisfies every contract
test and silently produces a table whose numbers mean the opposite of what they
claim.
"""
from __future__ import annotations

import asyncio

from app.decision_router import DecisionRouter
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.contracts import AutoDepthDecisionV1, CortexClientRequest
from orion.substrate import mutation_control_surface


def _req(text: str = "refactor this authentication module") -> CortexClientRequest:
    return CortexClientRequest.model_validate(
        {
            "mode": "auto",
            "route_intent": "auto",
            "packs": [],
            "options": {"route_intent": "auto"},
            "recall": {"enabled": True, "required": False, "mode": "hybrid", "profile": None},
            "context": {
                "messages": [{"role": "user", "content": text}],
                "raw_user_text": text,
                "user_message": text,
                "metadata": {},
            },
        }
    )


class _RecordingBus:
    codec = None

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel, msg) -> None:
        self.published.append((channel, msg))


def _route_and_drain(router: DecisionRouter, req: CortexClientRequest) -> None:
    """Run a turn and wait for the detached emit.

    The publish is deliberately fire-and-forget so a hung Redis cannot stall a
    chat turn, which means the test has to drain it explicitly. In the service
    the loop outlives the turn; here it does not.
    """

    async def _go() -> None:
        await router.route(
            req,
            correlation_id="c-emit",
            source=ServiceRef(name="orch", version="0", node="n"),
        )
        if router._emit_tasks:
            await asyncio.gather(*list(router._emit_tasks), return_exceptions=True)

    asyncio.run(_go())


def _decision(**kw) -> AutoDepthDecisionV1:
    base = dict(
        execution_depth=2,
        primary_verb="implement_change",
        confidence=0.85,
        reason="heuristic:engineering",
        source="heuristic",
    )
    base.update(kw)
    return AutoDepthDecisionV1(**base)


def test_a_demoted_turn_is_published_as_demoted(monkeypatch) -> None:
    """The outcome a routing mutation claims to move, recorded truthfully."""
    bus = _RecordingBus()
    router = DecisionRouter(bus)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.95, actor="test_emit")
    monkeypatch.setattr(DecisionRouter, "heuristic_router", lambda self, req, shortlist: _decision())

    _route_and_drain(router, _req())

    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:routing:decision"
    payload = envelope.payload
    assert payload["gate_demoted"] is True
    assert payload["execution_depth_before_gate"] == 2  # what it wanted to do
    assert payload["execution_depth"] == 0              # what the gate forced
    assert payload["routing_threshold"] == 0.95
    assert payload["decision_confidence"] == 0.85


def test_an_ungated_turn_is_published_as_not_demoted(monkeypatch) -> None:
    bus = _RecordingBus()
    router = DecisionRouter(bus)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.10, actor="test_emit")
    monkeypatch.setattr(DecisionRouter, "heuristic_router", lambda self, req, shortlist: _decision())

    _route_and_drain(router, _req())

    payload = bus.published[0][1].payload
    assert payload["gate_demoted"] is False
    assert payload["execution_depth_before_gate"] == payload["execution_depth"] == 2


def test_the_verb_the_gate_discarded_is_still_recorded(monkeypatch) -> None:
    """The gate nulls primary_verb, so reading it post-gate loses exactly the
    thing a before/after comparison wants: what Orion would have done.

    ``_clamp_decision`` is neutralised here because it independently nulls a verb
    that is not in the shortlist -- leaving it in place would let this pass with
    the gate's own null and prove nothing about where the value is captured.
    """
    bus = _RecordingBus()
    router = DecisionRouter(bus)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.95, actor="test_emit")
    monkeypatch.setattr(DecisionRouter, "heuristic_router", lambda self, req, shortlist: _decision())
    monkeypatch.setattr(DecisionRouter, "_clamp_decision", lambda self, decision, shortlist: decision)

    _route_and_drain(router, _req())

    payload = bus.published[0][1].payload
    assert payload["gate_demoted"] is True
    assert payload["execution_depth"] == 0            # the gate did null the verb
    assert payload["primary_verb"] == "implement_change"  # ... and we kept it anyway


def test_free_text_reason_never_reaches_the_bus(monkeypatch) -> None:
    """The LLM router's `reason` is parsed from a completion whose prompt renders
    the user's message. It must not be persisted verbatim."""
    bus = _RecordingBus()
    router = DecisionRouter(bus)
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.10, actor="test_emit")
    monkeypatch.setattr(
        DecisionRouter,
        "heuristic_router",
        lambda self, req, shortlist: _decision(
            reason="user is asking about their medication dosage"
        ),
    )

    _route_and_drain(router, _req())

    payload = bus.published[0][1].payload
    assert payload["reason"] == "unstructured"
    assert "medication" not in str(payload)


def test_a_failing_bus_does_not_break_the_turn(monkeypatch) -> None:
    """Observability degrades; routing does not."""

    class _BrokenBus:
        codec = None

        async def publish(self, channel, msg):
            raise RuntimeError("redis is down")

    router = DecisionRouter(_BrokenBus())
    mutation_control_surface.set_chat_reflective_lane_threshold(value=0.10, actor="test_emit")
    monkeypatch.setattr(DecisionRouter, "heuristic_router", lambda self, req, shortlist: _decision())

    async def _go():
        routed = await router.route(
            _req(), correlation_id="c-broken", source=ServiceRef(name="orch", version="0", node="n")
        )
        await asyncio.gather(*list(router._emit_tasks), return_exceptions=True)
        return routed

    routed = asyncio.run(_go())
    assert routed.decision.execution_depth == 2
