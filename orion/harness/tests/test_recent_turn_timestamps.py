"""RECENT CONVERSATION carries real times and elapsed-time markers.

Live 2026-10-16: the block had no times, so Orion read "cut off" (00:22 UTC)
-> "night night" (07:47 UTC) -> "what's the latest?" (16:56 UTC, ~9 h later)
as one moment and apologised for a cut-off that came two turns earlier.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from orion.substrate.appraisal.turn_window import build_turn_window

UTC = timezone.utc


def _prompt(turns, *, now, user_message="what's the latest?"):
    return compile_harness_prefix(
        make_thought(imperative="Continue.", tone="direct"),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message=user_message,
        recent_turns=turns,
        now=now,
    )


def _recent_block(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("RECENT CONVERSATION"))
    end = next(i for i, l in enumerate(lines) if l.startswith("TASK THIS TURN"))
    return lines[start + 1 : end]


def test_live_incident_renders_order_times_and_gaps() -> None:
    turns = [
        TurnWindowMessageV1(role="user", content="queue it up", at=datetime(2026, 10, 16, 0, 22, tzinfo=UTC)),
        TurnWindowMessageV1(role="assistant", content="[Cut short]", at=datetime(2026, 10, 16, 0, 22, tzinfo=UTC)),
        TurnWindowMessageV1(role="user", content="night night", at=datetime(2026, 10, 16, 7, 47, tzinfo=UTC)),
        TurnWindowMessageV1(role="assistant", content="Night night", at=datetime(2026, 10, 16, 7, 47, tzinfo=UTC)),
    ]
    prompt = _prompt(turns, now=datetime(2026, 10, 16, 16, 56, tzinfo=UTC))
    assert _recent_block(prompt) == [
        "- User (Thu Oct 15, 6:22 pm): queue it up",
        "- Orion (you, prior turn, Thu Oct 15, 6:22 pm): [Cut short]",
        "(— about 7 hours pass —)",
        "- User (Fri Oct 16, 1:47 am): night night",
        "- Orion (you, prior turn, Fri Oct 16, 1:47 am): Night night",
        "(— about 9 hours pass —)",
    ]
    lines = prompt.splitlines()
    task = lines.index(next(l for l in lines if l.startswith("TASK THIS TURN")))
    assert lines[task + 1] == "Current time: Fri Oct 16, 10:56 am (Juniper's local time)"
    assert lines[task + 2] == "User message: what's the latest?"


def test_local_time_follows_dst() -> None:
    # 18:00 UTC is noon MDT (UTC-6) in July and 11 am MST (UTC-7) in December.
    summer = TurnWindowMessageV1(role="user", content="hi", at=datetime(2026, 7, 1, 18, 0, tzinfo=UTC))
    winter = TurnWindowMessageV1(role="user", content="hi", at=datetime(2026, 12, 1, 18, 0, tzinfo=UTC))
    assert "- User (Wed Jul 1, 12:00 pm): hi" in _prompt([summer], now=summer.at)
    assert "- User (Tue Dec 1, 11:00 am): hi" in _prompt([winter], now=winter.at)
    # Naive datetimes are UTC (chat_history_log.created_at is naive UTC).
    naive = TurnWindowMessageV1(role="user", content="hi", at=datetime(2026, 12, 1, 18, 0))
    assert "- User (Tue Dec 1, 11:00 am): hi" in _prompt([naive], now=winter.at)


def test_gap_marker_threshold() -> None:
    t0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    under = [
        TurnWindowMessageV1(role="user", content="a", at=t0),
        TurnWindowMessageV1(role="assistant", content="b", at=t0 + timedelta(minutes=60)),
    ]
    # Exactly one hour is still "one sitting": the marker is for MORE than an hour.
    block = _recent_block(_prompt(under, now=t0 + timedelta(minutes=120)))
    assert not any("pass" in l for l in block)

    over = [
        TurnWindowMessageV1(role="user", content="a", at=t0),
        TurnWindowMessageV1(role="assistant", content="b", at=t0 + timedelta(minutes=61)),
    ]
    block = _recent_block(_prompt(over, now=t0 + timedelta(minutes=62)))
    assert block[1] == "(— about 1 hour pass —)"
    assert block[-1].startswith("- Orion")  # last->now is 1 min: no tail marker

    days = _recent_block(_prompt(over[:1], now=t0 + timedelta(days=3)))
    assert days[-1] == "(— about 3 days pass —)"
    # Half-up rounding; hours until 48 h.
    half = _recent_block(_prompt(over[:1], now=t0 + timedelta(hours=2, minutes=30)))
    assert half[-1] == "(— about 3 hours pass —)"
    day_and_half = _recent_block(_prompt(over[:1], now=t0 + timedelta(hours=36)))
    assert day_and_half[-1] == "(— about 36 hours pass —)"


def test_unloadable_timezone_renders_utc_labeled_utc(monkeypatch) -> None:
    """Never label a UTC time as Juniper's local time."""
    monkeypatch.setenv("ORION_SITUATION_TIMEZONE", "Not/AZone")
    t = datetime(2026, 10, 16, 16, 56, tzinfo=UTC)
    prompt = _prompt([TurnWindowMessageV1(role="user", content="hi", at=t)], now=t)
    assert "- User (Fri Oct 16, 4:56 pm): hi" in prompt
    assert "Current time: Fri Oct 16, 4:56 pm (UTC)" in prompt
    assert "Juniper's local time" not in prompt


def test_timezone_follows_the_situation_setting(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SITUATION_TIMEZONE", "America/New_York")
    t = datetime(2026, 10, 16, 16, 56, tzinfo=UTC)
    assert "Current time: Fri Oct 16, 12:56 pm (Juniper's local time)" in _prompt([], now=t)


def test_missing_timestamp_renders_without_time_or_gap() -> None:
    t0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    turns = [
        TurnWindowMessageV1(role="user", content="old", at=t0),
        TurnWindowMessageV1(role="assistant", content="untimed"),
        TurnWindowMessageV1(role="user", content="later", at=t0 + timedelta(hours=5)),
    ]
    block = _recent_block(_prompt(turns, now=t0 + timedelta(hours=10)))
    # No gap is guessed across the untimed message; the timed tail still gets one.
    assert block == [
        "- User (Fri Oct 16, 6:00 am): old",
        "- Orion (you, prior turn): untimed",
        "- User (Fri Oct 16, 11:00 am): later",
        "(— about 5 hours pass —)",
    ]


def test_build_turn_window_keeps_ts_only_when_asked() -> None:
    msgs = [
        {"role": "user", "content": "a", "ts": "2026-10-16T00:22:11+00:00"},
        {"role": "assistant", "content": "b", "ts": "not a time"},
        {"role": "user", "content": "c"},
    ]
    kept = build_turn_window(msgs, keep_timestamps=True)
    assert [m.content for m in kept] == ["a", "b", "c"]
    assert kept[0].at == datetime(2026, 10, 16, 0, 22, 11, tzinfo=UTC)
    assert kept[1].at is None and kept[2].at is None
    assert all(m.at is None for m in build_turn_window(msgs))


def test_untimed_payloads_stay_byte_identical_for_forbid_consumers() -> None:
    """A consumer not yet rebuilt validates with extra='forbid'; a payload
    with no timestamp must not grow an 'at' key it would reject."""
    plain = TurnWindowMessageV1(role="user", content="hi")
    assert plain.model_dump(mode="json") == {"role": "user", "content": "hi"}
    req = PreTurnAppraisalRequestV1(correlation_id="c", session_id="s", turn_window=[plain])
    assert '"at"' not in req.model_dump_json()
    assert req.model_dump(mode="json")["turn_window"] == [{"role": "user", "content": "hi"}]
    timed = TurnWindowMessageV1(role="user", content="hi", at=datetime(2026, 10, 16, tzinfo=UTC))
    dumped = timed.model_dump(mode="json")
    assert dumped["at"].startswith("2026-10-16T00:00:00")
    assert TurnWindowMessageV1.model_validate(dumped).at == timed.at
    # The governor validates HarnessRunRequestV1; its nested recent_turns obey
    # the same rule (only a timed message carries "at").
    from orion.schemas.harness_finalize import HarnessRunRequestV1

    fields = HarnessRunRequestV1.model_fields["recent_turns"]
    assert fields.annotation == list[TurnWindowMessageV1]
