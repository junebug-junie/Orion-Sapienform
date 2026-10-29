"""Strain report door (strain loop part 3): one grounded message per long strained episode.

Rules under test (Juniper, 2026-10-23): the door passes ONLY the arousal_strained refusal; quiet
hours hold it and then release it; an episode under 60 min does not qualify; once per episode;
once per local day; a restart does not re-send; flag off = the old behaviour; no grounding = no
send; recent_chat still blocks.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.regulation import (
    REGULATION_STATE_REDIS_KEY,
    ArousalReadingV1,
    RegulationStateV1,
)
from scripts.arousal_reader import ArousalReader
from scripts.strain_report import (
    STRAIN_HOLDS,
    StrainReportHistory,
    episode_key,
    render_record,
    strain_report_block,
    strain_report_facts,
)
from test_curiosity_investigation import _FakeBus

UTC = timezone.utc
ARC_ID = "a4c4acfbb676ef83"


def _now():
    return datetime.now(UTC)


def _state(*, level="strained", since_ago_min=120.0, reasons=("gpu_queue_sustained",), temp=31.9, queue=7):
    at = _now() - timedelta(seconds=20)
    reading = ArousalReadingV1(
        arousal_level=level,
        since=_now() - timedelta(minutes=since_ago_min),
        observed_at=at,
        reasons=list(reasons),
        cabinet_temp_c=temp,
        gpu_queue_depth=queue,
    )
    return RegulationStateV1(generated_at=at, arousal=reading)


def _arc_for(state, *, label="cabinet_hot+gpu_queue_sustained"):
    return {
        "kind": "strain", "status": "open", "arc_id": ARC_ID,
        "episode_began_at": state.arousal.since.isoformat(), "subject_label": label,
    }


class _Harness:
    """An EndogenousOutreach wired to fake Redis, a fake frame, a fake decision log and a fake
    compose. ``rows`` is the decision log: a restart rebuilds history from it like the SQL."""

    def __init__(self, monkeypatch, *, state=None, enabled=True, rows=(), quiet=(-1, -1),
                 spoke_ago=4000.0, compose="I've been strained for a while; the GPU queue is backed up."):
        from test_endogenous_outreach import _outreach

        import scripts.endogenous_outreach_decisions as decisions

        # () = an empty log; None = the log cannot be read.
        self.rows = None if rows is None else list(rows) if isinstance(rows, tuple) else rows
        self.composed: list = []
        self.delivered: list = []
        self.history_reads = 0
        self.state = state if state is not None else _state()
        self.arc = _arc_for(self.state)
        reader = ArousalReader(enabled=True, max_age_sec=360.0, name="test")
        self.o = _outreach(
            arousal_reader=reader, strain_report_enabled=enabled,
            quiet_start_hour=quiet[0], quiet_end_hour=quiet[1],
        )
        bus = _FakeBus()
        bus.redis.values[REGULATION_STATE_REDIS_KEY] = self.state.model_dump_json()
        self.o._bus = bus
        cls = type(self.o)
        g = cls._outreach_once.__globals__
        harness = self

        def fake_history(**_kw):
            harness.history_reads += 1
            return None if harness.rows is None else [
                {"episode": r.get("strain_episode_since"), "reason": r.get("reason"), "decided_at": r.get("at")}
                for r in harness.rows
                if r.get("source") == "strain_report" and r.get("strain_composed")
            ]

        def fake_record(result, **_kw):
            if harness.rows is not None:
                harness.rows.append(dict(result))

        async def fake_arc(_url, timeout_sec=1.5):
            return (harness.arc, "ok") if harness.arc else (None, "no_open_arc")

        async def fake_generate(self_, prompt, session_id, correlation_id, **_kw):
            harness.composed.append(prompt)
            text = harness.compose(prompt) if callable(harness.compose) else harness.compose
            return text, {"correlation_id": correlation_id}

        async def fake_deliver(self_, **kw):
            harness.delivered.append(kw)

        self.compose = compose
        monkeypatch.setattr(decisions, "fetch_strain_report_history", fake_history)
        monkeypatch.setattr(decisions, "record_decision", fake_record)
        self.intent_ok = True

        def fake_record_now(result):
            if harness.intent_ok and harness.rows is not None:
                harness.rows.append(dict(result))
            return harness.intent_ok

        monkeypatch.setattr(decisions, "record_decision_now", fake_record_now)
        monkeypatch.setitem(g, "fetch_open_strain_arc", fake_arc)
        monkeypatch.setitem(g, "_seconds_since_juniper_spoke", lambda: spoke_ago)
        monkeypatch.setitem(g, "_juniper_spoke_since", lambda epoch: False)
        monkeypatch.setattr(cls, "_generate", fake_generate)
        monkeypatch.setattr(cls, "_deliver", fake_deliver)

        async def no_recover(self_):
            self_._sent_today_recovered = True

        monkeypatch.setattr(cls, "_recover_sent_today", no_recover)

    def set_state(self, state):
        self.state = state
        self.o._bus.redis.values[REGULATION_STATE_REDIS_KEY] = state.model_dump_json()

    def tick(self):
        return asyncio.run(self.o.maybe_outreach())


# --- pure ------------------------------------------------------------------------------


def _block(**kw):
    base = dict(level="strained", since=_now() - timedelta(hours=2), now=_now(), min_episode_sec=3600.0,
                history=StrainReportHistory(), tz=UTC)
    base.update(kw)
    return strain_report_block(**base)


def test_block_qualifies_after_an_hour_only():
    assert _block() is None
    assert _block(since=_now() - timedelta(minutes=59)) == "episode_too_short"
    assert _block(level="idle") == "not_strained"
    assert _block(since=None) == "no_episode_start"


def test_block_once_per_episode_and_per_day_and_retry():
    since = _now() - timedelta(hours=2)
    h = StrainReportHistory()
    h.note(episode=episode_key(since), reason="sent", at=_now(), tz=UTC)
    assert _block(since=since, history=h) == "already_reported_episode"
    # Another episode the same local day.
    assert _block(since=_now() - timedelta(hours=1, minutes=5), history=h) == "already_reported_today"
    h2 = StrainReportHistory()
    h2.note(episode=episode_key(since), reason="empty_generation", at=_now() - timedelta(minutes=5), tz=UTC)
    assert _block(since=since, history=h2) == "retry_wait"
    h2.note(episode=episode_key(since), reason="orion_passed", at=_now(), tz=UTC)
    assert _block(since=since, history=h2) == "orion_declined_episode"


def test_facts_refuse_without_grounding_and_cite_only_the_matching_arc():
    empty = _state(reasons=("strain_holding",), temp=None, queue=None)
    assert strain_report_facts(empty, None) is None
    assert strain_report_facts(None, None) is None
    assert strain_report_facts(_state(level="idle"), None) is None
    s = _state()
    facts = strain_report_facts(s, _arc_for(s))
    assert facts.arc_id == ARC_ID and facts.causes_seen == ("cabinet_hot", "gpu_queue_sustained")
    other = dict(_arc_for(s), episode_began_at=(s.arousal.since - timedelta(hours=3)).isoformat())
    assert strain_report_facts(s, other).arc_id is None
    record = render_record(facts, _now(), UTC)
    assert f"[arc:{ARC_ID}]" in record and "31.9 C" in record and "GPU jobs waiting: 7" in record


def test_every_hold_is_a_registered_strain_reader():
    from orion.inner_state_registry import get

    readers = {c.split(" ")[0] for c in get("regulation.state.v1").cognition_consumers}
    for key, _phrase in STRAIN_HOLDS:
        assert key in readers, key


# --- the tick ----------------------------------------------------------------------------


def test_long_episode_sends_one_grounded_report(monkeypatch):
    h = _Harness(monkeypatch)
    r = h.tick()
    assert (r["outreach"], r["reason"], r["source"], r["door"]) == (True, "sent", "strain_report", "strain_report")
    assert r["strain_episode_since"] == episode_key(h.state.arousal.since)
    assert r["strain_arc_id"] == ARC_ID
    text = h.delivered[0]["text"]
    assert f"[arc:{ARC_ID}]" in text and "Causes now: gpu_queue_sustained" in text
    assert h.delivered[0]["source_tag"] == "strain_report"
    assert "curiosity investigations wait" in h.composed[0]
    # Once per episode.
    r2 = h.tick()
    assert (r2["outreach"], r2["reason"], r2["strain_report"]) == (False, "arousal_strained", "already_reported_episode")
    assert len(h.delivered) == 1


def test_once_per_local_day_across_episodes(monkeypatch):
    h = _Harness(monkeypatch)
    assert h.tick()["reason"] == "sent"
    h.set_state(_state(since_ago_min=70))
    r = h.tick()
    assert r["strain_report"] == "already_reported_today" and len(h.delivered) == 1


def test_restart_does_not_resend(monkeypatch):
    h = _Harness(monkeypatch)
    assert h.tick()["reason"] == "sent"
    again = _Harness(monkeypatch, state=h.state, rows=h.rows)
    r = again.tick()
    assert r["strain_report"] == "already_reported_episode"
    assert again.composed == [] and again.delivered == []


def test_unknown_history_keeps_the_door_shut(monkeypatch):
    h = _Harness(monkeypatch, rows=None)
    r = h.tick()
    assert (r["reason"], r["strain_report"]) == ("arousal_strained", "history_unknown")
    assert h.composed == []


def test_short_episode_does_not_qualify(monkeypatch):
    h = _Harness(monkeypatch, state=_state(since_ago_min=30))
    r = h.tick()
    assert (r["reason"], r["strain_report"]) == ("arousal_strained", "episode_too_short")
    assert h.composed == []


def test_quiet_hours_hold_then_release(monkeypatch):
    hour = _now().hour
    h = _Harness(monkeypatch, quiet=(hour, (hour + 1) % 24))
    assert h.tick()["reason"] == "quiet_hours" and h.composed == []
    h.o.quiet_start_hour = h.o.quiet_end_hour = -1  # quiet hours over, same episode still open
    assert h.tick()["reason"] == "sent"


@pytest.mark.parametrize("gate", ["recent_chat", "turn_in_flight", "disabled", "cooldown"])
def test_only_the_strained_refusal_is_passed(monkeypatch, gate):
    h = _Harness(monkeypatch, spoke_ago=60.0 if gate == "recent_chat" else 4000.0)
    if gate == "turn_in_flight":
        h.o.register_connection("c1", asyncio.Queue(), {"correlation_id": "busy"})
    if gate == "disabled":
        h.o.enabled = False
    if gate == "cooldown":
        import time

        h.o.min_cooldown_sec = 2700.0
        h.o._last_outreach_at = time.time() - 60.0
    r = h.tick()
    assert r["reason"] == gate and h.composed == [] and h.delivered == []


def test_flag_off_is_the_old_refusal(monkeypatch):
    h = _Harness(monkeypatch, enabled=False)
    r = h.tick()
    assert r["reason"] == "arousal_strained" and "strain_report" not in r
    assert h.composed == [] and h.history_reads == 0


def test_no_grounding_no_send(monkeypatch):
    h = _Harness(monkeypatch, state=_state(reasons=("strain_holding",), temp=None, queue=None))
    h.arc = None
    r = h.tick()
    assert (r["reason"], r["strain_report"]) == ("arousal_strained", "no_grounding")
    assert h.composed == [] and h.delivered == []


def test_empty_compose_sends_nothing_and_waits_to_retry(monkeypatch):
    h = _Harness(monkeypatch, compose="")
    assert h.tick()["reason"] == "empty_generation" and h.delivered == []
    assert h.tick()["strain_report"] == "retry_wait"


def test_pass_is_respected_for_the_episode(monkeypatch):
    h = _Harness(monkeypatch, compose="PASS")
    assert h.tick()["reason"] == "orion_passed"
    assert h.tick()["strain_report"] == "orion_declined_episode" and h.delivered == []


def test_episode_change_mid_compose_drops_the_report(monkeypatch):
    h = _Harness(monkeypatch)

    def compose(_prompt):
        h.set_state(_state(since_ago_min=1))
        return "I've been strained a while."

    h.compose = compose
    r = h.tick()
    assert r["reason"] == "strain_episode_changed_after_generation" and h.delivered == []


def test_lost_sent_row_still_blocks_resend_after_restart(monkeypatch):
    """Review finding: the `sent` row is fire-and-forget. The synchronous intent row must be
    enough on its own to stop a re-send after a restart."""
    h = _Harness(monkeypatch)
    assert h.tick()["reason"] == "sent"
    only_intent = [r for r in h.rows if r.get("reason") == "strain_report_sending"]
    assert len(only_intent) == 1
    again = _Harness(monkeypatch, state=h.state, rows=only_intent)
    assert again.tick()["strain_report"] == "already_reported_episode" and again.delivered == []


def test_intent_write_failure_means_no_send(monkeypatch):
    h = _Harness(monkeypatch)
    h.intent_ok = False
    r = h.tick()
    assert r["reason"] == "intent_write_failed" and h.delivered == []


def test_failed_composes_are_capped_per_episode():
    from scripts.strain_report import STRAIN_REPORT_MAX_ATTEMPTS

    since = _now() - timedelta(hours=10)
    h = StrainReportHistory()
    for i in range(STRAIN_REPORT_MAX_ATTEMPTS):
        h.note(episode=episode_key(since), reason="empty_generation", at=_now() - timedelta(hours=5 - i), tz=UTC)
    assert _block(since=since, history=h) == "attempts_exhausted"


def test_strain_report_spends_a_daily_cap_slot(monkeypatch):
    h = _Harness(monkeypatch)
    before = h.o._sent_today
    assert h.tick()["reason"] == "sent"
    assert h.o._sent_today == before + 1


# ---- the prompt states how the GPU queue relates to the strain (live 10-23 invented "heat queued jobs")


def _prompt_facts(causes_now, *, holding=False, queue_holding=False, deferrable=None):
    from zoneinfo import ZoneInfo

    from scripts.strain_report import StrainReportFacts

    since = datetime(2026, 10, 23, 19, 47, tzinfo=timezone.utc)
    facts = StrainReportFacts(
        episode_since=since, observed_at=since + timedelta(hours=3), causes_now=tuple(causes_now),
        causes_seen=("gpu_queue_sustained", "cabinet_hot"), holding=holding, cabinet_temp_c=33.6,
        cabinet_reflex="cabinet_hot", gpu_queue_depth=9, gpu_queue_self_deferrable=deferrable,
        arc_id="26011418aad1aaa2", holds=("curiosity investigations wait",), queue_holding=queue_holding,
    )
    return facts, since + timedelta(hours=3), ZoneInfo("America/Denver")


def test_prompt_says_a_waiting_queue_is_not_a_cause_when_only_heat_is_active():
    from scripts.strain_report import build_strain_report_prompt

    facts, now, tz = _prompt_facts(["cabinet_hot"])
    prompt = build_strain_report_prompt(facts, "Strain record", now, tz)
    assert "GPU jobs waiting now: 9. This is NOT a cause of the strain right now." in prompt
    assert "never say one caused another" in prompt


def test_prompt_marks_the_queue_as_a_cause_when_it_is_one():
    from scripts.strain_report import build_strain_report_prompt

    facts, now, tz = _prompt_facts(["gpu_queue_sustained"])
    prompt = build_strain_report_prompt(facts, "Strain record", now, tz)
    assert "GPU jobs waiting now: 9 (this is one of the causes above)." in prompt
    assert "NOT a cause" not in prompt


def test_prompt_says_the_queue_keeps_a_holding_strain_from_clearing():
    """strain_holding: no active cause, but the load is still at/above the floor -- the queue IS what
    holds the strain, so 'not a cause' would be a new invented fact (review 10-24)."""
    from scripts.strain_report import build_strain_report_prompt

    facts, now, tz = _prompt_facts([], holding=True, queue_holding=True)
    prompt = build_strain_report_prompt(facts, "Strain record", now, tz)
    assert "it is what keeps the strain from clearing" in prompt and "NOT a cause" not in prompt
    # strain_clearing (load below the floor): the queue is not what holds it.
    facts, now, tz = _prompt_facts([], holding=True, queue_holding=False)
    assert "NOT a cause of the strain right now" in build_strain_report_prompt(facts, "Strain record", now, tz)


def test_prompt_names_deferrable_self_work_in_the_count():
    from scripts.strain_report import build_strain_report_prompt

    facts, now, tz = _prompt_facts(["gpu_queue_sustained"], deferrable=4)
    prompt = build_strain_report_prompt(facts, "Strain record", now, tz)
    assert "9 (this is one of the causes above, 4 of them your own deferrable work, which does not count as load)" in prompt
