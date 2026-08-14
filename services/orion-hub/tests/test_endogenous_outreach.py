"""Gates, prompt construction, and delivery for endogenous outreach.

Hand-computed expectations throughout: each gate case sets exactly one field
away from a known-passing baseline, so a test that passes for the wrong reason
(e.g. blocked by a stale default) fails loudly instead.
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from scripts.endogenous_outreach import (
    EndogenousOutreach,
    OutreachContext,
    OutreachGateInputs,
    build_outreach_prompt,
    in_quiet_hours,
    is_pass_response,
    outreach_block_reason,
)


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


@pytest.mark.parametrize("raw", ["PASS", " pass ", "PASS.", '"PASS"'])
def test_pass_response_detected(raw) -> None:
    assert is_pass_response(raw) is True


@pytest.mark.parametrize("raw", ["", "I'll pass along the note", "passing thought about you"])
def test_non_pass_response_not_swallowed(raw) -> None:
    assert is_pass_response(raw) is False


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


class _AlwaysRoll(random.Random):
    def random(self) -> float:  # noqa: D102
        return 0.0


def _outreach(**overrides) -> EndogenousOutreach:
    base = dict(
        enabled=True,
        tick_interval_sec=300.0,
        probability=1.0,
        min_cooldown_sec=0.0,
        daily_cap=4,
        quiet_start_hour=-1,
        quiet_end_hour=-1,
        llm_route="quick",
        timeout_sec=5.0,
        notify_channel="orion:notify:in_app",
        fallback_session_id="orion_outreach",
        rng=_AlwaysRoll(),
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

    class _SlowCortex:
        async def chat(self, req, correlation_id=None, **kwargs):
            await asyncio.sleep(5)

    outreach._cortex_client = _SlowCortex()
    _stub_context(monkeypatch)

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is False
    assert result["reason"] == "empty_generation"
    assert result["generation"]["error"] == "timeout"


def test_generation_sends_configured_lane(monkeypatch) -> None:
    # Execution policy / recall assertions live in
    # test_generation_pins_execution_policies_and_disables_recall.
    outreach = _outreach(llm_route="metacog")
    seen: dict = {}

    class _Cortex:
        async def chat(self, req, correlation_id=None, **kwargs):
            seen["req"] = req
            return SimpleNamespace(final_text="a real unprompted thought")

    outreach._cortex_client = _Cortex()
    _stub_context(monkeypatch)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    result = asyncio.run(outreach.maybe_outreach())

    assert result["outreach"] is True
    assert seen["req"].options["llm_route"] == "metacog"


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


def test_generation_pins_execution_policies_and_disables_recall(monkeypatch) -> None:
    """Review findings 4 and 5: the previous keys were inert.

    options["no_write"] is only translated by cortex_request_builder, which
    reads it as a top-level payload key and which this module does not use;
    options["use_recall"] has no reader at all -- recall is controlled by the
    typed `recall` field, which defaults to enabled=True when left None.
    """
    outreach = _outreach()
    seen: dict = {}

    class _Cortex:
        async def chat(self, req, correlation_id=None, **kwargs):
            seen["req"] = req
            return SimpleNamespace(final_text="an unprompted thought")

    outreach._cortex_client = _Cortex()
    _stub_context(monkeypatch)
    monkeypatch.setattr(EndogenousOutreach, "_publish_history", lambda self, **kw: asyncio.sleep(0))
    monkeypatch.setattr(EndogenousOutreach, "_publish_notification", lambda self, **kw: asyncio.sleep(0))

    assert asyncio.run(outreach.maybe_outreach())["outreach"] is True

    req = seen["req"]
    assert req.options["tool_execution_policy"] == "none"
    assert req.options["action_execution_policy"] == "none"
    assert req.options["no_write_active"] is True
    assert req.recall == {"enabled": False}


def test_recall_directive_accepts_the_payload_we_send() -> None:
    """The gateway filters our dict into RecallDirective; enabled must survive."""
    from orion.schemas.cortex.contracts import RecallDirective

    directive = RecallDirective(**{"enabled": False})
    assert directive.enabled is False


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


def test_probability_zero_never_rolls(monkeypatch) -> None:
    outreach = _outreach(probability=0.0, rng=random.Random(1234))
    _stub_context(monkeypatch)
    _stub_generation(monkeypatch, "never")

    result = asyncio.run(outreach.maybe_outreach())

    assert result["reason"] == "not_rolled"


def test_disabled_instance_starts_no_task() -> None:
    outreach = _outreach(enabled=False)

    async def run() -> None:
        await outreach.start(bus=None, cortex_client=None)

    asyncio.run(run())

    assert outreach.status()["running"] is False
    assert outreach.status()["block_reason"] == "disabled"
