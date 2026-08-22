"""Gates, prompt construction, and delivery for endogenous outreach.

Hand-computed expectations throughout: each gate case sets exactly one field
away from a known-passing baseline, so a test that passes for the wrong reason
(e.g. blocked by a stale default) fails loudly instead.
"""

from __future__ import annotations

import asyncio

import pytest

from scripts.endogenous_outreach import (
    EndogenousOutreach,
    OutreachContext,
    OutreachGateInputs,
    build_outreach_prompt,
    in_quiet_hours,
    is_pass_response,
    looks_like_error_text,
    outreach_block_reason,
)
from scripts.tension_outreach_trigger import TensionTriggerReason


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def _passing_gate(**overrides) -> OutreachGateInputs:
    """Baseline that MUST return None; every gate test perturbs one field."""
    base = dict(
        enabled=True,
        turn_in_flight=False,
        local_hour=14,
        quiet_start_hour=23,
        quiet_end_hour=8,
        seconds_since_last_outreach=10_000.0,
        min_cooldown_sec=2700.0,
        sent_today=0,
        daily_cap=4,
    )
    base.update(overrides)
    return OutreachGateInputs(**base)


def test_baseline_gate_passes() -> None:
    # Guards every other test in this block: if the baseline itself blocked,
    # the perturbation tests would pass vacuously.
    assert outreach_block_reason(_passing_gate()) is None


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"enabled": False}, "disabled"),
        ({"turn_in_flight": True}, "turn_in_flight"),
        ({"local_hour": 2}, "quiet_hours"),
        ({"sent_today": 4}, "daily_cap"),
        ({"seconds_since_last_outreach": 60.0}, "cooldown"),
    ],
)
def test_each_gate_blocks_independently(overrides, expected) -> None:
    assert outreach_block_reason(_passing_gate(**overrides)) == expected


def test_never_reached_outreach_is_not_cooldown_blocked() -> None:
    # None means "never spoken", which must not read as "0 seconds ago".
    assert outreach_block_reason(_passing_gate(seconds_since_last_outreach=None)) is None


def test_daily_cap_negative_one_disables_cap() -> None:
    assert outreach_block_reason(_passing_gate(daily_cap=-1, sent_today=999)) is None


def test_disabled_outranks_every_other_reason() -> None:
    blocked = _passing_gate(enabled=False, turn_in_flight=True, local_hour=2, sent_today=99)
    assert outreach_block_reason(blocked) == "disabled"


@pytest.mark.parametrize(
    "hour,start,end,expected",
    [
        # Wrapping window 23->8: hand-checked at both edges and the interior.
        (23, 23, 8, True),
        (0, 23, 8, True),
        (7, 23, 8, True),
        (8, 23, 8, False),  # end is exclusive
        (22, 23, 8, False),
        # Non-wrapping window 9->17.
        (9, 9, 17, True),
        (16, 9, 17, True),
        (17, 9, 17, False),
        (8, 9, 17, False),
        # Disabled forms.
        (3, -1, 8, False),
        (3, 0, 0, False),
    ],
)
def test_quiet_hours_boundaries(hour, start, end, expected) -> None:
    assert in_quiet_hours(hour, start, end) is expected


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_empty_context_yields_no_prompt() -> None:
    """AGENTS.md §0A: no contentless generation, so no prompt at all."""
    ctx = OutreachContext(curiosity_summaries=[], recent_turns=[], presence=None)
    assert build_outreach_prompt(ctx) == ""


def test_presence_alone_is_not_grounding() -> None:
    # Presence is colour, not substance -- it must not by itself unlock a tick.
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence={"health": "idle", "last_turn_age_sec": 600.0},
    )
    assert build_outreach_prompt(ctx) == ""


def test_prompt_carries_real_signals_and_turns() -> None:
    ctx = OutreachContext(
        curiosity_summaries=["sustained prediction error on node:substrate.execution"],
        recent_turns=[("Juniper", "sup yo"), ("Orion", "that is genuinely exciting")],
        presence={"health": "idle", "last_turn_age_sec": 2460.0},
    )
    prompt = build_outreach_prompt(ctx)
    assert "sustained prediction error on node:substrate.execution" in prompt
    assert "Juniper: sup yo" in prompt
    assert "Orion: that is genuinely exciting" in prompt
    # 2460s / 60 == 41 minutes, rounded by format spec.
    assert "41 minutes ago" in prompt
    assert "PASS" in prompt


def test_curiosity_alone_is_enough_grounding() -> None:
    ctx = OutreachContext(
        curiosity_summaries=["repair pressure rising on node:substrate.route"],
        recent_turns=[],
        presence=None,
    )
    assert "repair pressure rising" in build_outreach_prompt(ctx)


def test_tension_reason_alone_is_enough_grounding() -> None:
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence=None,
        tension_reason=TensionTriggerReason(
            target_id="node:athena", run_length=9, peak_deviation_pressure=0.62
        ),
    )
    prompt = build_outreach_prompt(ctx)
    assert prompt != ""
    assert "node:athena" in prompt
    assert "9 consecutive readings" in prompt


def test_tension_reason_prompt_never_claims_distress() -> None:
    """The gate is a change-detector, not a level-detector -- the prompt must
    not overclaim a distress/concern reading it cannot honestly support."""
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence=None,
        tension_reason=TensionTriggerReason(
            target_id="node:athena", run_length=9, peak_deviation_pressure=0.62
        ),
    )
    prompt = build_outreach_prompt(ctx).lower()
    for banned in ("worried", "concerned", "distress", "alarmed"):
        assert banned not in prompt


def test_sustained_load_pressure_adds_a_second_real_fact_when_nonzero() -> None:
    """2026-08-19: the level-aware half is combined honestly into the same
    prompt -- a real second fact, not folded into or replacing the change
    fact above."""
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence=None,
        tension_reason=TensionTriggerReason(
            target_id="node:athena",
            run_length=9,
            peak_deviation_pressure=0.62,
            sustained_load_pressure=0.71,
        ),
    )
    prompt = build_outreach_prompt(ctx)
    assert "sustained_load_pressure=0.71" in prompt
    # The original change fact must still be present, untouched.
    assert "9 consecutive readings" in prompt


def test_sustained_load_pressure_omitted_from_prompt_when_zero() -> None:
    """0.0 means 'nothing currently loaded_steady' -- a real reading, not
    something worth stating as a fact (matches `orion.field.significance`'s
    own 0.0 convention)."""
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence=None,
        tension_reason=TensionTriggerReason(
            target_id="node:athena",
            run_length=9,
            peak_deviation_pressure=0.62,
            sustained_load_pressure=0.0,
        ),
    )
    prompt = build_outreach_prompt(ctx)
    assert "sustained_load_pressure" not in prompt


def test_tension_reason_with_sustained_load_still_never_claims_distress() -> None:
    """Even with a real, nonzero level-aware reading present, this module
    must not script a feeling for Orion -- that judgment is left to
    generation, not hardcoded into the prompt."""
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence=None,
        tension_reason=TensionTriggerReason(
            target_id="node:athena",
            run_length=9,
            peak_deviation_pressure=0.62,
            sustained_load_pressure=0.71,
        ),
    )
    prompt = build_outreach_prompt(ctx).lower()
    for banned in ("worried", "concerned", "distress", "alarmed"):
        assert banned not in prompt


@pytest.mark.parametrize("raw", ["PASS", " pass ", "PASS.", '"PASS"'])
def test_pass_response_detected(raw) -> None:
    assert is_pass_response(raw) is True


@pytest.mark.parametrize("raw", ["", "I'll pass along the note", "passing thought about you"])
def test_non_pass_response_not_swallowed(raw) -> None:
    assert is_pass_response(raw) is False


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


def _always_fires() -> TensionTriggerReason:
    """Default test evaluator: a real-shaped reason, always returned -- most
    tests here are exercising gates/delivery, not the trigger itself (see
    test_tension_outreach_trigger.py for that)."""
    return TensionTriggerReason(target_id="node:test", run_length=99, peak_deviation_pressure=1.0)


def _outreach(**overrides) -> EndogenousOutreach:
    base = dict(
        enabled=True,
        tick_interval_sec=300.0,
        min_cooldown_sec=0.0,
        daily_cap=4,
        quiet_start_hour=-1,
        quiet_end_hour=-1,
        timeout_sec=5.0,
        notify_channel="orion:notify:in_app",
        fallback_session_id="orion_outreach",
        trigger_evaluator=_always_fires,
    )
    base.update(overrides)
    return EndogenousOutreach(**base)


def _stub_context(monkeypatch, *, summaries=("sustained prediction error on node:x",), turns=()) -> None:
    async def fake_gather(self, session_id):
        return OutreachContext(
            curiosity_summaries=list(summaries),
            recent_turns=list(turns),
            presence=None,
        )

    monkeypatch.setattr(EndogenousOutreach, "_gather_context", fake_gather)


def _stub_generation(monkeypatch, text: str) -> None:
    async def fake_generate(self, prompt, session_id, correlation_id):
        return text, {"stub": True}

    monkeypatch.setattr(EndogenousOutreach, "_generate", fake_generate)


def _final_frame(text: str, *, correlation_id: str = "corr", **extra) -> dict:
    """One real `execute_unified_turn` success frame shape (see
    orion.hub.turn_orchestrator._success_frames) -- the ONLY frame `_generate`
    reads (`type == "final"`)."""
    frame = {
        "type": "final",
        "correlation_id": correlation_id,
        "mode": "orion",
        "llm_response": text,
        "finalize_ran": True,
        "finalize_changed": True,
        "harness_step_count": 1,
        "harness_grounding_status": None,
    }
    frame.update(extra)
    return frame


def _stub_unified_turn(monkeypatch, fake_execute) -> None:
    """Patch the exact call site `_generate` lazily imports at call time
    (`from orion.hub.turn_orchestrator import execute_unified_turn`) --
    patching the module attribute, not a re-import, is what makes this
    visible to that lazy import (same reasoning this suite's own docstring
    already applies to `sys.modules['scripts.*']`)."""
    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", fake_execute)


def test_turn_in_flight_blocks_even_when_forced(monkeypatch) -> None:
    """force= skips only the random roll; safety gates still hold."""
    outreach = _outreach()
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": "live-turn", "kind": "orion"})
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "should never be sent")

    result = asyncio.run(outreach.maybe_outreach(force=True))

    assert result == {**result, "outreach": False, "reason": "turn_in_flight"}
    assert queue.empty()


def test_empty_generation_is_dropped_not_shipped(monkeypatch) -> None:
    outreach = _outreach()
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "   ")

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["reason"] == "empty_generation"
    assert queue.empty()
    assert outreach.status()["sent_today"] == 0


def test_orion_pass_does_not_consume_the_daily_budget(monkeypatch) -> None:
    outreach = _outreach()
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "PASS")

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "orion_passed"
    assert outreach.status()["sent_today"] == 0
    assert outreach.status()["seconds_since_last_outreach"] is None


def test_no_grounding_context_skips_generation(monkeypatch) -> None:
    outreach = _outreach()
    _stub_context(monkeypatch, summaries=(), turns=())

    called = {"n": 0}

    async def fake_generate(self, prompt, session_id, correlation_id):
        called["n"] += 1
        return "unreachable", {}

    monkeypatch.setattr(EndogenousOutreach, "_generate", fake_generate)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "no_grounding_context"
    assert called["n"] == 0


def test_successful_outreach_pushes_to_every_live_socket(monkeypatch) -> None:
    outreach = _outreach()
    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", q1, {"correlation_id": None, "kind": None})
    outreach.register_connection("c2", q2, {"correlation_id": None, "kind": None})
    outreach.note_session("c1", "sess-abc")
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "The execution node has been noisy all afternoon.")

    published: list = []

    async def fake_history(self, **kwargs):
        published.append(("history", kwargs))

    async def fake_notify(self, **kwargs):
        published.append(("notify", kwargs))

    monkeypatch.setattr(EndogenousOutreach, "_publish_history", fake_history)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", fake_notify)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is True
    assert result["session_id"] == "sess-abc"
    for queue in (q1, q2):
        payload = queue.get_nowait()
        assert payload["kind"] == "orion_outreach"
        assert payload["llm_response"] == "The execution node has been noisy all afternoon."
        assert payload["session_id"] == "sess-abc"
        # Must not carry keys that would stomp the UI's live turn panels.
        assert "state" not in payload
        assert "recall_debug" not in payload
        assert "memory_digest" not in payload
    assert [kind for kind, _ in published] == ["history", "notify"]
    assert outreach.status()["sent_today"] == 1


def test_second_outreach_blocked_by_cooldown(monkeypatch) -> None:
    outreach = _outreach(min_cooldown_sec=2700.0)
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "something worth saying")
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    first = asyncio.run(outreach.maybe_outreach())
    second = asyncio.run(outreach.maybe_outreach())

    assert first["outreach"] is True
    assert second["outreach"] is False
    assert second["reason"] == "cooldown"


def test_session_hello_lets_an_idle_tab_be_reachable_in_thread() -> None:
    """Confirmed live on the first real firing (2026-08-14).

    `connections: 1` but `session_id: "orion_outreach"` -- the browser was open
    and had never sent a message, so note_session had never fired and the
    outreach landed in the fallback session instead of the thread on screen.
    The connect-time session_hello frame closes that.
    """
    outreach = _outreach()
    outreach.register_connection("c1", asyncio.Queue(), {"correlation_id": None, "kind": None})

    # Open tab, nothing typed yet: fallback.
    assert outreach._active_session_id() == "orion_outreach"

    # session_hello arrives on connect.
    outreach.note_session("c1", "sess-from-hello")
    assert outreach._active_session_id() == "sess-from-hello"


def test_session_hello_with_no_session_id_is_ignored() -> None:
    """A tab with no stored session must not blank an already-known one."""
    outreach = _outreach()
    outreach.register_connection("c1", asyncio.Queue(), {"correlation_id": None, "kind": None})
    outreach.note_session("c1", "sess-real")

    for empty in (None, "", "   "):
        outreach.note_session("c1", empty)
        assert outreach._active_session_id() == "sess-real"


def test_falls_back_to_configured_session_when_no_socket_reported_one(monkeypatch) -> None:
    outreach = _outreach()
    outreach.register_connection("c1", asyncio.Queue(), {"correlation_id": None, "kind": None})
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "unprompted thought")
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    result = asyncio.run(outreach.maybe_outreach())

    assert result["session_id"] == "orion_outreach"


def test_unregistered_connection_stops_receiving_and_stops_blocking() -> None:
    outreach = _outreach()
    active_turn = {"correlation_id": "live", "kind": "orion"}
    outreach.register_connection("c1", asyncio.Queue(), active_turn)
    assert outreach.status()["block_reason"] == "turn_in_flight"

    outreach.unregister_connection("c1")

    assert outreach.status()["connections"] == 0
    assert outreach.status()["block_reason"] is None


def test_active_turn_is_held_by_reference() -> None:
    """The ws handler mutates its own dict mid-turn; no re-registration call."""
    outreach = _outreach()
    active_turn = {"correlation_id": None, "kind": None}
    outreach.register_connection("c1", asyncio.Queue(), active_turn)
    assert outreach.status()["block_reason"] is None

    active_turn["correlation_id"] = "turn-started"

    assert outreach.status()["block_reason"] == "turn_in_flight"


def test_generation_timeout_returns_empty_not_raise(monkeypatch) -> None:
    outreach = _outreach(timeout_sec=0.01)
    outreach._bus = object()

    async def fake_execute(**kwargs):
        await asyncio.sleep(5)
        return []  # unreachable

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["reason"] == "empty_generation"
    assert result["generation"]["error"] == "timeout"


def test_generation_calls_the_real_unified_turn_pipeline(monkeypatch) -> None:
    """2026-08-19: generation goes through orion.hub.turn_orchestrator.
    execute_unified_turn -- the SAME function websocket_handler.py calls for
    a real client_mode=="orion" turn, not a cheaper substitute. This asserts
    the call shape, not a route (there is no route left to configure --
    that's the harness governor's decision now, identically for outreach and
    real chat)."""
    outreach = _outreach()
    outreach._bus = object()
    harness_bus = object()
    outreach._harness_rpc_bus = harness_bus
    seen: dict = {}

    async def fake_execute(**kwargs):
        seen.update(kwargs)
        return [_final_frame("a real unprompted thought", correlation_id=kwargs["correlation_id"])]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is True
    assert seen["harness_rpc_bus"] is harness_bus
    assert seen["user_message"]  # the built prompt, non-empty
    # no_write: this module's own _deliver() is the sole persistence path --
    # see endogenous_outreach.py's module docstring for why the governor's
    # own persistence step must be suppressed rather than reused.
    assert seen["payload"]["no_write"] is True


def test_generation_threads_fcc_model_label_into_publish_history(monkeypatch) -> None:
    """Regression for the live-confirmed gap (2026-08-19): the final frame's
    fcc_model_label (now exposed by turn_orchestrator._success_frames, see
    that module's own change) must reach _publish_history's model= kwarg so
    chat_history_log.response_identity is the real served identity instead
    of always falling back to speaker='Orion'."""
    outreach = _outreach()
    outreach._bus = object()

    async def fake_execute(**kwargs):
        return [
            _final_frame(
                "a real unprompted thought",
                correlation_id=kwargs["correlation_id"],
                fcc_model_label="MODEL_SONNET",
            )
        ]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)
    captured: dict = {}

    async def fake_publish_history(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(EndogenousOutreach, "_publish_history", fake_publish_history)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is True
    assert captured["model"] == "MODEL_SONNET"


def test_generation_omits_model_when_frame_has_no_fcc_model_label(monkeypatch) -> None:
    outreach = _outreach()
    outreach._bus = object()

    async def fake_execute(**kwargs):
        return [_final_frame("a real unprompted thought", correlation_id=kwargs["correlation_id"])]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)
    captured: dict = {}

    async def fake_publish_history(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(EndogenousOutreach, "_publish_history", fake_publish_history)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is True
    assert captured["model"] is None


class _FakeBus:
    def __init__(self) -> None:
        self.published: list = []
        self.enabled = True

    async def publish(self, channel, envelope) -> None:
        self.published.append((channel, envelope))


def test_notification_payload_validates_against_the_real_schema() -> None:
    """Exercises HubNotificationEvent for real, not a monkeypatched stand-in.

    ``message_id`` is a UUID field but the outreach carries it around as a str,
    so a coercion mistake here would only ever surface at runtime.
    """
    from uuid import UUID, uuid4

    outreach = _outreach()
    bus = _FakeBus()
    outreach._bus = bus
    message_id = str(uuid4())
    # BaseEnvelope.correlation_id is UUID-typed; maybe_outreach always mints
    # str(uuid4()), so the fixture must match production rather than a label.
    correlation_id = str(uuid4())

    asyncio.run(
        outreach._publish_notification(
            text="the execution node has been noisy",
            session_id="sess-abc",
            correlation_id=correlation_id,
            message_id=message_id,
        )
    )

    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:notify:in_app"
    assert envelope.kind == "notify.in_app.v1"
    payload = envelope.payload
    assert payload["body_text"] == "the execution node has been noisy"
    assert payload["event_kind"] == "hub.endogenous_outreach.v1"
    assert payload["session_id"] == "sess-abc"
    assert payload["tags"] == ["endogenous_outreach"]
    assert UUID(payload["message_id"]) == UUID(message_id)


def test_notification_failure_does_not_propagate() -> None:
    class _BrokenBus:
        async def publish(self, channel, envelope):
            raise RuntimeError("bus down")

    outreach = _outreach()
    outreach._bus = _BrokenBus()

    # Must not raise: a delivery failure cannot be allowed to kill the loop.
    asyncio.run(
        outreach._publish_notification(
            text="x", session_id="s", correlation_id="c", message_id=str(__import__("uuid").uuid4())
        )
    )


def test_busy_connection_blocks_even_without_active_turn() -> None:
    """Review finding 1: active_turn is only set by 2 of the UI's 4 modes.

    Quick / Story / Agent fall through the ws handler's general cortex path and
    never populate active_turn, so a gate reading only that key sees an idle
    socket for the whole duration of a real turn.
    """
    outreach = _outreach()
    outreach.register_connection("c1", asyncio.Queue(), {"correlation_id": None, "kind": None})
    assert outreach.status()["block_reason"] is None

    outreach.note_busy("c1")
    assert outreach.status()["block_reason"] == "turn_in_flight"

    outreach.note_idle("c1")
    assert outreach.status()["block_reason"] is None


def test_turn_starting_during_generation_drops_the_outreach(monkeypatch) -> None:
    """Review finding 2 (TOCTOU): the gate must be re-checked after the LLM call.

    Generation is a bus RPC bounded by timeout_sec (default 60s); a gate checked
    only at the top of the tick lets a turn that starts mid-generation get
    talked over.
    """
    outreach = _outreach()
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})
    _stub_context(monkeypatch)

    async def generate_then_user_starts_typing(self, prompt, session_id, correlation_id):
        outreach.note_busy("c1")  # Juniper hits Enter while the LLM is working
        return "something I was thinking about", {"stub": True}

    monkeypatch.setattr(EndogenousOutreach, "_generate", generate_then_user_starts_typing)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["reason"] == "turn_in_flight_after_generation"
    assert queue.empty()
    assert outreach.status()["sent_today"] == 0


def test_force_cannot_override_the_disabled_flag(monkeypatch) -> None:
    """Review finding 3: the debug trigger endpoint is unauthenticated.

    A force= carve-out for "disabled" would mean one POST makes a feature
    documented as off-by-default emit real unsolicited chat.
    """
    outreach = _outreach(enabled=False)
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "should never be sent")

    result = asyncio.run(outreach.maybe_outreach(force=True))

    assert result["outreach"] is False
    assert result["reason"] == "disabled"
    assert queue.empty()


def test_generation_no_longer_configures_execution_policy_or_recall(monkeypatch) -> None:
    """2026-08-19: these are no longer this module's decision to make.

    Old regression (Review findings 4 and 5 on the direct-cortex-client
    path): options["no_write"]/["use_recall"] were inert keys nobody read.
    That whole class of bug is now structurally impossible, not just fixed --
    execute_unified_turn hard-codes every unified turn's
    ContextExecPermissionV1 to read-only (every write/mutate/network/shell
    flag stays at its safe False default) and recall is controlled by
    whatever the harness governor's own turn logic decides, identically for
    outreach and real chat. Nothing here to assert about options/recall any
    more -- see execute_unified_turn's own contract
    (orion/schemas/context_exec.py::ContextExecPermissionV1) for where that
    guarantee actually lives. The one thing THIS module still configures --
    payload["no_write"] suppressing the governor's own chat-history
    persistence -- is asserted in
    test_generation_calls_the_real_unified_turn_pipeline.
    """
    outreach = _outreach()
    outreach._bus = object()

    async def fake_execute(**kwargs):
        return [_final_frame("an unprompted thought", correlation_id=kwargs["correlation_id"])]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    assert asyncio.run(outreach.maybe_outreach())["outreach"] is True


def test_concurrent_ticks_cannot_both_send(monkeypatch) -> None:
    """Review finding 7: the loop and the debug endpoint share cooldown state.

    Counters are only bumped after _deliver, so two overlapping passes would
    each see a clean cooldown.
    """
    outreach = _outreach(min_cooldown_sec=2700.0)
    _stub_context(monkeypatch)
    released = asyncio.Event()

    async def slow_generate(self, prompt, session_id, correlation_id):
        await released.wait()
        return "an unprompted thought", {}

    monkeypatch.setattr(EndogenousOutreach, "_generate", slow_generate)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    async def scenario():
        first = asyncio.create_task(outreach.maybe_outreach())
        await asyncio.sleep(0)  # let `first` reach slow_generate
        second = await outreach.maybe_outreach(force=True)
        released.set()
        return await first, second

    first_result, second_result = asyncio.run(scenario())

    assert first_result["outreach"] is True
    assert second_result["outreach"] is False
    assert second_result["reason"] == "already_sending"
    assert outreach.status()["sent_today"] == 1


def test_quiet_hours_use_configured_zone_not_the_container_clock() -> None:
    """Review finding 6: Hub's container sets no TZ, so naive local == UTC.

    18:00 UTC is 13:00 in Chicago -- inside a UTC-read 17->23 window, outside
    the same window read in Chicago.
    """
    import time as _time
    from datetime import datetime as _dt, timezone as _tz

    utc_1800 = _dt(2026, 8, 14, 18, 0, tzinfo=_tz.utc).timestamp()

    as_utc = _outreach(quiet_start_hour=17, quiet_end_hour=23, timezone_name="UTC")
    as_chicago = _outreach(quiet_start_hour=17, quiet_end_hour=23, timezone_name="America/Chicago")

    assert outreach_block_reason(as_utc._gate_inputs(now=utc_1800)) == "quiet_hours"
    assert outreach_block_reason(as_chicago._gate_inputs(now=utc_1800)) is None
    assert as_chicago.status()["timezone"] == "America/Chicago"
    assert _time  # keep the import meaningful if the assertion set changes


def test_unknown_timezone_falls_back_to_utc_loudly(caplog) -> None:
    with caplog.at_level("ERROR"):
        outreach = _outreach(timezone_name="Mars/Olympus_Mons")
    assert outreach.status()["timezone"] == "UTC"
    assert any("endogenous_outreach_bad_timezone" in r.message for r in caplog.records)


def test_daily_cap_resets_on_the_configured_zones_date_boundary() -> None:
    """The cap rolls at midnight in the configured zone, not at 00:00 UTC."""
    from datetime import datetime as _dt, timezone as _tz

    outreach = _outreach(timezone_name="America/Chicago", daily_cap=1)
    # 04:00 UTC on the 14th is still 23:00 on the 13th in Chicago.
    late_on_13th = _dt(2026, 8, 14, 4, 0, tzinfo=_tz.utc).timestamp()
    # 06:00 UTC is 01:00 on the 14th in Chicago -- a new local day.
    early_on_14th = _dt(2026, 8, 14, 6, 0, tzinfo=_tz.utc).timestamp()

    outreach._gate_inputs(now=late_on_13th)
    outreach._sent_today = 1
    assert outreach_block_reason(outreach._gate_inputs(now=late_on_13th)) == "daily_cap"

    assert outreach_block_reason(outreach._gate_inputs(now=early_on_14th)) is None
    assert outreach._sent_today == 0


@pytest.mark.parametrize(
    "raw",
    [
        # The exact string that reached Juniper's chat thread on 2026-08-14.
        "[Error: llamacpp failed: Client error '400 Bad Request' for url "
        "'http://100.121.214.30:8013/v1/chat/completions']",
        "Error: connection refused",
        "Traceback (most recent call last):\n  File ...",
        "Internal Server Error",
        "llamacpp failed: read timeout after 60s",
    ],
)
def test_error_shaped_text_is_recognised(raw) -> None:
    assert looks_like_error_text(raw) is True


@pytest.mark.parametrize(
    "raw",
    [
        "",
        # Orion's real first outreach -- must NOT be swallowed by the backstop.
        "The codebase is throwing errors I can't map yet. It's like trying to fix "
        "a machine without knowing which parts are broken.",
        "I keep hitting errors in the same place and I think that means something.",
        "There is something about the way a timeout feels from the inside.",
    ],
)
def test_real_prose_about_errors_is_not_swallowed(raw) -> None:
    assert looks_like_error_text(raw) is False


def test_turn_error_frame_is_dropped_not_shipped(monkeypatch) -> None:
    """Regression, generalized for the real pipeline (2026-08-19): a
    turn_error/turn_deferred/turn_degraded frame set (no `type=="final"`
    frame at all) must degrade to empty, never ship a partial_draft or an
    error string as if Orion had said it. Root-caused live, 2026-08-19: this
    exact class of failure (a real 400 from the LLM backend arriving as a
    frame with no clean final text) is what silently broke outreach even
    after the poll-cadence fix (PR #1727)."""
    outreach = _outreach()
    outreach._bus = object()
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})

    async def fake_execute(**kwargs):
        return [
            {
                "type": "turn_error",
                "correlation_id": kwargs["correlation_id"],
                "phase": "harness",
                "error": "llamacpp failed: Client error '400 Bad Request'",
            }
        ]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["reason"] == "empty_generation"
    assert result["generation"]["error"] == "no_final_frame"
    assert result["generation"]["frame_type"] == "turn_error"
    assert queue.empty()
    assert outreach.status()["sent_today"] == 0


def test_context_overflow_final_frame_is_dropped(monkeypatch) -> None:
    """The real pipeline's own context_overflow detection (root-caused
    2026-08-19: the OLD direct-call path had none, only Hub's own
    looks_like_error_text() backstop) -- a `final` frame explicitly flagged
    context_overflow must still be dropped even though it has real text."""
    outreach = _outreach()
    outreach._bus = object()

    async def fake_execute(**kwargs):
        return [
            _final_frame(
                "[context window exceeded]",
                correlation_id=kwargs["correlation_id"],
                context_overflow=True,
            )
        ]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["generation"]["error"] == "context_overflow"


def test_error_shaped_text_dropped_even_in_a_final_frame(monkeypatch) -> None:
    """Backstop for an upstream that reports failure only in the prose, even
    when it arrives inside a real `type=="final"` frame."""
    outreach = _outreach()
    outreach._bus = object()
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})

    async def fake_execute(**kwargs):
        return [
            _final_frame(
                "[Error: llamacpp failed: Client error '400 Bad Request']",
                correlation_id=kwargs["correlation_id"],
            )
        ]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["generation"]["error"] == "error_shaped_text"
    assert queue.empty()


def test_healthy_final_frame_still_sends(monkeypatch) -> None:
    """Guards the tests above: the happy path must not be over-gated."""
    outreach = _outreach()
    outreach._bus = object()
    queue: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", queue, {"correlation_id": None, "kind": None})

    async def fake_execute(**kwargs):
        return [
            _final_frame(
                "I keep circling the same unresolved thing.",
                correlation_id=kwargs["correlation_id"],
            )
        ]

    _stub_unified_turn(monkeypatch, fake_execute)
    _stub_context(monkeypatch)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is True
    assert queue.get_nowait()["llm_response"] == "I keep circling the same unresolved thing."


def test_no_tension_trigger_does_not_fire(monkeypatch) -> None:
    outreach = _outreach(trigger_evaluator=lambda: None)
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "never")

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "no_tension_trigger"


def test_trigger_evaluator_exception_degrades_to_not_firing(monkeypatch) -> None:
    """A broken trigger must not crash the tick -- the honest failure mode is
    silence, not a false positive."""

    def _broken():
        raise RuntimeError("db is down")

    outreach = _outreach(trigger_evaluator=_broken)
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "never")

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "no_tension_trigger"


def test_trigger_evaluator_does_not_block_the_event_loop(monkeypatch) -> None:
    """Regression guard: `_should_roll()` must run its (blocking, synchronous)
    evaluator off the event loop via `asyncio.to_thread`. Proven by racing a
    concurrent coroutine against a slow evaluator and confirming the
    concurrent coroutine's own mark lands BEFORE the evaluator's -- only
    possible if the evaluator actually yielded the loop instead of blocking
    it in-line, which would otherwise freeze every connected websocket and
    in-flight chat turn on Hub's single uvicorn worker for the call's
    duration."""
    import time

    marks: list[str] = []

    def _slow_evaluator():
        time.sleep(0.2)
        marks.append("evaluator_done")
        return None

    outreach = _outreach(trigger_evaluator=_slow_evaluator)
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "never")

    async def _concurrent_marker():
        await asyncio.sleep(0.05)
        marks.append("concurrent_task_ran")

    async def run():
        await asyncio.gather(outreach.maybe_outreach(), _concurrent_marker())

    asyncio.run(run())

    assert marks == ["concurrent_task_ran", "evaluator_done"]


def test_status_does_not_report_a_stale_tension_reason_after_a_blocked_tick(monkeypatch) -> None:
    """A gate (quiet_hours/daily_cap/cooldown/turn_in_flight) blocking BEFORE
    the trigger evaluator ever runs must not leave `status()` reporting an
    earlier tick's reason as if it were still live right now."""
    outreach = _outreach()
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "hi")

    asyncio.run(outreach.maybe_outreach())  # organic: sets a real reason
    assert outreach.status()["last_tension_reason"] is not None

    outreach.register_connection("c1", asyncio.Queue(), {"correlation_id": "live-turn", "kind": "orion"})
    asyncio.run(outreach.maybe_outreach())  # blocked by turn_in_flight before _should_roll runs

    assert outreach.status()["block_reason"] == "turn_in_flight"
    assert outreach.status()["last_tension_reason"] is None


def test_status_reports_sustained_load_pressure_alongside_deviation(monkeypatch) -> None:
    """2026-08-19: the operator-visible debug surface must carry the new
    level-aware number, not just the pre-existing deviation one -- an
    un-updated status() would silently hide half of what the trigger now
    knows (AGENTS.md §0A, "UI/debug surface")."""
    outreach = _outreach(
        trigger_evaluator=lambda: TensionTriggerReason(
            target_id="node:athena",
            run_length=9,
            peak_deviation_pressure=0.62,
            sustained_load_pressure=0.71,
        )
    )
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "hi")

    asyncio.run(outreach.maybe_outreach())

    reason = outreach.status()["last_tension_reason"]
    assert reason is not None
    assert reason["sustained_load_pressure"] == 0.71


def test_forced_outreach_does_not_carry_a_stale_tension_reason(monkeypatch) -> None:
    """A prior organic tick's reason must not leak into a later force=True
    (debug-endpoint) call that never re-evaluated the trigger."""
    outreach = _outreach()
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "hi")

    asyncio.run(outreach.maybe_outreach())  # organic: sets _last_tension_reason
    assert outreach._last_tension_reason is not None

    captured: list[OutreachContext] = []

    async def fake_gather(self, session_id):
        ctx = OutreachContext(curiosity_summaries=["x"], recent_turns=[], presence=None,
                               tension_reason=self._last_tension_reason)
        captured.append(ctx)
        return ctx

    monkeypatch.setattr(EndogenousOutreach, "_gather_context", fake_gather)
    asyncio.run(outreach.maybe_outreach(force=True))

    assert captured[-1].tension_reason is None


def test_disabled_instance_starts_no_task() -> None:
    outreach = _outreach(enabled=False)

    async def run() -> None:
        await outreach.start(bus=None, harness_rpc_bus=None)

    asyncio.run(run())

    assert outreach.status()["running"] is False
    assert outreach.status()["block_reason"] == "disabled"


def test_start_falls_back_to_bus_when_no_harness_rpc_bus_given() -> None:
    """Mirrors websocket_handler.py's own `harness_rpc_bus=rpc_bus or bus`
    convention -- a forked RPC client is preferred (see module docstring for
    why a plain long-lived bus risks a stolen reply), but a direct/test
    caller with only `bus` must not end up with no harness bus at all."""
    outreach = _outreach(enabled=False)
    sentinel_bus = object()

    async def run() -> None:
        await outreach.start(sentinel_bus)

    asyncio.run(run())

    assert outreach._bus is sentinel_bus
    assert outreach._harness_rpc_bus is sentinel_bus


def test_start_prefers_the_forked_harness_rpc_bus_when_given() -> None:
    outreach = _outreach(enabled=False)
    sentinel_bus = object()
    sentinel_rpc_bus = object()

    async def run() -> None:
        await outreach.start(sentinel_bus, harness_rpc_bus=sentinel_rpc_bus)

    asyncio.run(run())

    assert outreach._bus is sentinel_bus
    assert outreach._harness_rpc_bus is sentinel_rpc_bus


def _env_example_value(key: str) -> str:
    """Read one KEY=value line straight out of the checked-in `.env_example`,
    without instantiating `Settings` (see `test_tick_sec_default_is_not_the_
    root_caused_300s_value`'s own comment for why that specifically doesn't
    work here). Shared by every test in this file that needs to assert
    something about the checked-in operator contract, not the runtime
    default -- those are two different files and this repo has been burned
    before by them drifting apart silently."""
    import re
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / ".env_example"
    match = re.search(rf"^{re.escape(key)}=(.+)$", example.read_text(), re.M)
    assert match, f"{key} missing from .env_example"
    return match.group(1).strip()


def test_shipped_timezone_is_a_real_iana_zone() -> None:
    """A typo in .env_example's TZ degrades silently to UTC.

    The constructor's fallback is deliberate (a bad zone must not crash Hub
    startup), which means a misspelling would quietly shift the quiet window by
    hours with only a log line to show for it. This gate turns that into a
    failing test instead.
    """
    from zoneinfo import ZoneInfo

    zone = _env_example_value("HUB_ENDOGENOUS_OUTREACH_TZ")
    ZoneInfo(zone)  # raises if the zone is not real

    # Round-trip through the real constructor: proves it did not fall back.
    outreach = _outreach(timezone_name=zone)
    assert outreach.status()["timezone"] == zone


def test_tick_sec_default_is_not_the_root_caused_300s_value() -> None:
    """Regression guard, 2026-08-19: 300s was root-caused live as the reason
    outreach had never fired -- a real qualifying run's catchable window is
    typically 0-8s, so a 300s poll essentially never observes one (0 of 9
    real episodes caught, replayed against real history). This does not
    pin the exact new value (that's a real, data-derived tuning knob an
    operator may retune from live firing-rate data, same as
    MIN_RUN_LENGTH) -- it only guards against silently drifting back to the
    specific value already proven broken."""
    import re
    from pathlib import Path

    assert float(_env_example_value("HUB_ENDOGENOUS_OUTREACH_TICK_SEC")) != 300.0

    # Source-text check, not a live `Settings()` import. Verified directly,
    # 2026-08-19: `app/settings.py`'s module-level `settings = get_settings()`
    # runs `Settings()` on the very first `from app.settings import
    # <anything>` (any name, including the `Settings` class itself -- Python
    # executes the whole module on first import regardless of which name is
    # pulled from it), and this suite has no fixture supplying the class's
    # other required env keys (CHANNEL_VOICE_*/CHANNEL_COLLAPSE_* etc.), so
    # even a bare `Settings.model_fields[...]` lookup fails before it's ever
    # reached -- reproduced live: `pydantic_core.ValidationError: 5
    # validation errors for Settings`. There is no simpler import-based
    # equivalent available in this environment; the regex is the workaround,
    # not an oversight.
    settings_src = (Path(__file__).resolve().parents[1] / "app" / "settings.py").read_text()
    settings_match = re.search(
        r'HUB_ENDOGENOUS_OUTREACH_TICK_SEC:\s*float\s*=\s*Field\(\s*default=([\d.]+)',
        settings_src,
    )
    assert settings_match, "HUB_ENDOGENOUS_OUTREACH_TICK_SEC Field default not found in settings.py"
    assert float(settings_match.group(1)) != 300.0


# --------------------------------------------------------------------------
# Durable decision log (2026-08-22) -- _record() is the one choke point
# every branch funnels through; these assert the persist hook fires with the
# right forced/tension_reason for each shape, not just the happy path.
# --------------------------------------------------------------------------


def _patch_record_decision(monkeypatch):
    import scripts.endogenous_outreach_decisions as decisions_mod

    calls: list[dict] = []

    def fake_record(result, *, tension_reason=None, forced=False):
        calls.append({"result": result, "tension_reason": tension_reason, "forced": forced})

    monkeypatch.setattr(decisions_mod, "record_decision", fake_record)
    return calls


def test_record_persists_a_blocked_decision_with_the_real_forced_flag(monkeypatch) -> None:
    """The already_sending early-return in maybe_outreach never reaches
    _outreach_once -- this is exactly the path the `_last_forced` comment on
    `maybe_outreach` calls out as needing to be set before that return."""
    outreach = _outreach()
    calls = _patch_record_decision(monkeypatch)

    async def run_both():
        async with outreach._send_lock:
            return await outreach.maybe_outreach(force=True)

    result = asyncio.run(run_both())

    assert result["reason"] == "already_sending"
    assert len(calls) == 1
    assert calls[0]["forced"] is True
    assert calls[0]["result"]["reason"] == "already_sending"


def test_record_persists_no_tension_trigger_with_no_stale_reason(monkeypatch) -> None:
    outreach = _outreach(trigger_evaluator=lambda: None)
    calls = _patch_record_decision(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "no_tension_trigger"
    assert len(calls) == 1
    assert calls[0]["forced"] is False
    assert calls[0]["tension_reason"] is None


def test_record_persists_a_successful_send_with_the_real_tension_reason(monkeypatch) -> None:
    outreach = _outreach()
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "The execution node has been noisy all afternoon.")

    async def fake_history(self, **kwargs):
        return None

    async def fake_notify(self, **kwargs):
        return None

    monkeypatch.setattr(EndogenousOutreach, "_publish_history", fake_history)
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", fake_notify)
    calls = _patch_record_decision(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "sent"
    assert len(calls) == 1
    assert calls[0]["forced"] is False
    assert calls[0]["result"]["reason"] == "sent"
    assert calls[0]["tension_reason"].target_id == "node:test"


def test_record_persists_a_forced_trigger_with_no_stale_tension_reason(monkeypatch) -> None:
    """Mirrors test_forced_outreach_does_not_carry_a_stale_tension_reason --
    the decision log must not misattribute a forced debug trigger to
    whatever the LAST organic tick's reason was."""
    outreach = _outreach()
    outreach._last_tension_reason = _always_fires()  # stale, from a prior organic tick
    _stub_context(monkeypatch, summaries=(), turns=())  # no organic grounding
    calls = _patch_record_decision(monkeypatch)

    # No curiosity/turns/tension_reason -> no_grounding_context, but the
    # important assertion is what tension_reason gets threaded through.
    result = asyncio.run(outreach.maybe_outreach(force=True))

    assert result["reason"] == "no_grounding_context"
    assert len(calls) == 1
    assert calls[0]["forced"] is True
    assert calls[0]["tension_reason"] is None


def test_decision_log_hook_failure_does_not_break_the_tick(monkeypatch) -> None:
    """A broken persist hook must not make maybe_outreach itself raise or
    change its returned result -- same best-effort contract this module's
    other side rails (_push_to_sockets, _publish_notification) already have."""
    import scripts.endogenous_outreach_decisions as decisions_mod

    def broken_record(*args, **kwargs):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(decisions_mod, "record_decision", broken_record)
    outreach = _outreach(trigger_evaluator=lambda: None)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "no_tension_trigger"
