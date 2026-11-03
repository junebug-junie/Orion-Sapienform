"""Self findings (self-monitor E3) reach Juniper through the real outreach gates, each episode once.

Runs against a real ``EndogenousOutreach`` (only generation, delivery side effects and the
decision-log write are stubbed), like test_reading_finding_outreach.py.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1

T0 = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
WALKWAY = "act.actions.journal.make.walkway_grade"
PRIME = "act.prime.predict"
DREAM = "act.dream.cycle.make"

# The live walkway-grade numbers (patch 1 report, 2026-11-03): 22 of 28 tries never ran.
BROKEN_NUMBERS = dict(claim_kind="success", n_bets=28, n_graded=6, n_failed_to_execute=22, reliability=1.0,
                      reliability_lo=0.6, reliability_hi=1.0, execution_failure=0.79,
                      execution_failure_lo=0.55, streak_kind="failed_to_execute", streak_len=4,
                      reasons=["failed_to_execute 22/28 (lower bound 0.55)"])
RELIABLE_NUMBERS = dict(claim_kind="success", n_bets=20, n_graded=20, n_failed_to_execute=0, reliability=1.0,
                        reliability_lo=0.84, reliability_hi=1.0, reasons=["works 100% (lower bound 0.84)"])
UNRELIABLE_NUMBERS = dict(claim_kind="success", n_bets=40, n_graded=40, n_failed_to_execute=0, reliability=0.02,
                          reliability_lo=0.0, reliability_hi=0.14, reasons=["works 2% (upper bound 0.14 < 0.5)"])


def rec(faculty, status, since, **numbers):
    base = dict(faculty=faculty, sources=["grammar_acts"], gradeability="graded", window_start=since - timedelta(days=7),
                window_end=since, n_bets=0, n_graded=0, n_failed_to_execute=0, n_expired=0, weight_graded=1.0,
                candidate_status=status, status=status, status_since=since, candidate_held=3)
    base.update(numbers)
    return FacultyRecordV1(**base)


def state_json(computed_at, *records):
    return SelfMonitorStateV1(computed_at=computed_at, window_days=7.0, faculties=list(records)).model_dump_json()


class _Redis:
    def __init__(self):
        self.kv: dict[str, str] = {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v):
        self.kv[k] = v


class _Conn:
    def __init__(self, rows):
        self.rows = rows

    async def fetch(self, sql, *args):
        from scripts import self_finding_outreach as sfo

        assert sql == sfo.DECISIONS_SQL
        ids, reasons = set(args[0]), args[1]
        return [r for r in self.rows if r["reason"] in reasons and r.get("source") == "self_finding_outreach"
                and ids & set(json.loads(r["finding_ids"]))]


class _Pool:
    def __init__(self, rows):
        self.rows = rows

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _Conn(pool.rows)

            async def __aexit__(self, *a):
                return False

        return _Ctx()


def _setup(monkeypatch, *, text="My walkway grade journal has mostly stopped running.", clock=None, **outreach_kw):
    from scripts import endogenous_outreach as eo
    from scripts import endogenous_outreach_decisions as decisions_mod
    from scripts import self_finding_outreach as sfo

    rows: list[dict] = []

    def record(result, **kw):
        rows.append({**result, "finding_ids": json.dumps(result.get("finding_ids") or [])})

    monkeypatch.setattr(decisions_mod, "record_decision", record)
    monkeypatch.setattr(eo, "_seconds_since_juniper_spoke", outreach_kw.pop("spoke", lambda: None))
    composed: list[str] = []
    delivered: list[dict] = []
    texts = {"value": text}

    async def fake_generate(self, prompt, session_id, correlation_id, retrieval_query=None):
        composed.append(prompt)
        return texts["value"], {"fcc_model_label": "test"}

    async def fake_deliver(self, **kw):
        delivered.append(kw)

    monkeypatch.setattr(eo.EndogenousOutreach, "_generate", fake_generate)
    monkeypatch.setattr(eo.EndogenousOutreach, "_deliver", fake_deliver)
    base = dict(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=4,
                quiet_start_hour=-1, quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0,
                notify_channel="orion:notify:in_app", fallback_session_id="orion_outreach",
                trigger_evaluator=lambda: None, timezone_name="UTC")
    base.update(outreach_kw)
    outreach = eo.EndogenousOutreach(**base)
    redis = _Redis()
    now = {"t": T0}
    loop = sfo.SelfFindingOutreach(enabled=True, redis_provider=lambda: redis, pool_provider=lambda: _Pool(rows),
                                   outreach_provider=lambda: outreach, clock=clock or (lambda: now["t"]))
    return loop, outreach, redis, rows, composed, delivered, now, texts


def _publish(redis, computed_at, *records):
    from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY

    redis.kv[SELF_MONITOR_REDIS_KEY] = state_json(computed_at, *records)


def _sent(rows):
    return [r for r in rows if r["reason"] == "sent"]


# -- the pure rule ---------------------------------------------------------


def test_finding_kind_is_exactly_the_e3_bar():
    from scripts.self_finding_outreach import finding_kind

    assert finding_kind("reliable", "broken") == "became_broken"
    assert finding_kind("unseen", "broken") == "became_broken"
    assert finding_kind("broken", "reliable") == "recovered"
    assert finding_kind("unreliable", "reliable") == "recovered"
    assert finding_kind("reliable", "unreliable") == "became_unreliable"
    for frm, to in (("unseen", "reliable"), ("unseen", "unreliable"), ("broken", "unreliable"),
                    ("broken", "broken"), ("reliable", "miscalibrated"), ("reliable", "unknown"),
                    ("unknown", "reliable"), ("miscalibrated", "reliable")):
        assert finding_kind(frm, to) is None, (frm, to)


# -- the loop --------------------------------------------------------------


def test_first_evaluation_tells_a_broken_faculty_once_and_nothing_else(monkeypatch):
    loop, _, redis, rows, composed, delivered, now, _ = _setup(monkeypatch)
    _publish(redis, T0, rec(WALKWAY, "broken", T0 - timedelta(hours=1), **BROKEN_NUMBERS),
             rec(DREAM, "reliable", T0 - timedelta(hours=1), **RELIABLE_NUMBERS),
             rec(PRIME, "unreliable", T0 - timedelta(hours=1), **UNRELIABLE_NUMBERS))
    summary = asyncio.run(loop.tick())
    assert summary["new"] == [f"self_finding:{WALKWAY}:broken@{(T0 - timedelta(hours=1)).isoformat()}"]
    assert [r["reason"] for r in summary["results"]] == ["sent"] and len(delivered) == 1
    prompt = composed[0]
    assert "In the last 7 days, 22 of its 28 tries never ran." in prompt
    assert "Juniper has not been told about it before." in prompt
    assert PRIME not in prompt and DREAM not in prompt
    sent = _sent(rows)[0]
    assert sent["source"] == "self_finding_outreach" and sent["finding_kinds"] == ["became_broken"]
    assert f"self_monitor:{WALKWAY}@{T0.isoformat()}" in sent["evidence_refs"]
    assert delivered[0]["provenance"]["prompt_text"] == prompt
    # Same state again, and a newer state with the same held status: nothing more.
    assert asyncio.run(loop.tick())["results"] == []
    now["t"] = T0 + timedelta(hours=1)
    _publish(redis, now["t"], rec(WALKWAY, "broken", T0 - timedelta(hours=1), **BROKEN_NUMBERS))
    assert asyncio.run(loop.tick())["results"] == [] and len(delivered) == 1


def test_recovery_is_one_finding(monkeypatch):
    loop, _, redis, rows, composed, delivered, now, _ = _setup(monkeypatch, text="Walkway grading runs again.")
    since_broken = T0 - timedelta(hours=1)
    _publish(redis, T0, rec(WALKWAY, "broken", since_broken, **BROKEN_NUMBERS))
    asyncio.run(loop.tick())
    now["t"] = T0 + timedelta(days=2)
    _publish(redis, now["t"], rec(WALKWAY, "reliable", now["t"], **RELIABLE_NUMBERS))
    summary = asyncio.run(loop.tick())
    assert summary["new"] == [f"self_finding:{WALKWAY}:reliable@{now['t'].isoformat()}"]
    assert [s["finding_kinds"] for s in _sent(rows)] == [["became_broken"], ["recovered"]]
    assert "It was broken (most of its tries never ran)" in composed[-1]
    assert "Of 20 tries that were checked, about 100% worked" in composed[-1]


def test_reliable_to_unreliable_is_one_finding_and_miscalibrated_between_does_not_hide_it(monkeypatch):
    loop, _, redis, rows, _, delivered, now, _ = _setup(monkeypatch, text="My predictions stopped landing.")
    _publish(redis, T0, rec(PRIME, "reliable", T0, **RELIABLE_NUMBERS))
    assert asyncio.run(loop.tick())["new"] == []  # a first-seen reliable faculty is not news
    now["t"] = T0 + timedelta(hours=5)
    _publish(redis, now["t"], rec(PRIME, "miscalibrated", now["t"], **RELIABLE_NUMBERS))
    assert asyncio.run(loop.tick())["new"] == []
    now["t"] = T0 + timedelta(hours=10)
    _publish(redis, now["t"], rec(PRIME, "unreliable", now["t"], **UNRELIABLE_NUMBERS))
    assert len(asyncio.run(loop.tick())["new"]) == 1
    assert [s["finding_kinds"] for s in _sent(rows)] == [["became_unreliable"]] and len(delivered) == 1


def test_miscalibrated_and_unknown_flips_tell_nothing(monkeypatch):
    loop, _, redis, rows, composed, delivered, now, _ = _setup(monkeypatch)
    _publish(redis, T0, rec(DREAM, "reliable", T0, **RELIABLE_NUMBERS), rec(PRIME, "unknown", T0))
    asyncio.run(loop.tick())
    for i, (a, b) in enumerate((("miscalibrated", "reliable"), ("unknown", "miscalibrated"),
                                ("reliable", "unknown"), ("unknown", "reliable")), start=1):
        now["t"] = T0 + timedelta(hours=4 * i)
        _publish(redis, now["t"], rec(DREAM, a, now["t"], **RELIABLE_NUMBERS), rec(PRIME, b, now["t"], **RELIABLE_NUMBERS))
        assert asyncio.run(loop.tick())["new"] == []
    assert composed == [] and delivered == [] and rows == []


def test_broken_unknown_broken_is_no_news_and_broken_unknown_reliable_is_a_recovery(monkeypatch):
    loop, _, redis, rows, _, _, now, _ = _setup(monkeypatch)
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    asyncio.run(loop.tick())
    now["t"] = T0 + timedelta(hours=4)
    _publish(redis, now["t"], rec(WALKWAY, "unknown", now["t"]))
    asyncio.run(loop.tick())
    now["t"] = T0 + timedelta(hours=8)
    _publish(redis, now["t"], rec(WALKWAY, "broken", now["t"], **BROKEN_NUMBERS))
    assert asyncio.run(loop.tick())["new"] == []
    now["t"] = T0 + timedelta(hours=12)
    _publish(redis, now["t"], rec(WALKWAY, "reliable", now["t"], **RELIABLE_NUMBERS))
    assert len(asyncio.run(loop.tick())["new"]) == 1
    assert [s["finding_kinds"] for s in _sent(rows)] == [["became_broken"], ["recovered"]]


def test_same_episode_is_told_once_even_after_the_state_key_is_lost(monkeypatch):
    from scripts.self_finding_outreach import SELF_FINDING_STATE_KEY

    loop, _, redis, rows, _, delivered, now, _ = _setup(monkeypatch)
    since = T0 - timedelta(hours=1)
    _publish(redis, T0, rec(WALKWAY, "broken", since, **BROKEN_NUMBERS))
    asyncio.run(loop.tick())
    del redis.kv[SELF_FINDING_STATE_KEY]  # Redis lost the cursor
    loop._told.clear()  # and the Hub restarted: only the decision log remembers
    now["t"] = T0 + timedelta(hours=1)
    _publish(redis, now["t"], rec(WALKWAY, "broken", since, **BROKEN_NUMBERS))
    summary = asyncio.run(loop.tick())
    assert len(summary["new"]) == 1 and summary["results"] == [] and summary["pending"] == 0
    assert len(delivered) == 1 and len(_sent(rows)) == 1


def test_stale_absent_and_unparseable_state_tell_nothing(monkeypatch):
    from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY
    from scripts.self_finding_outreach import SELF_FINDING_STATE_KEY

    loop, _, redis, rows, composed, _, now, _ = _setup(monkeypatch)
    assert asyncio.run(loop.tick())["view"] == "absent"
    redis.kv[SELF_MONITOR_REDIS_KEY] = "{not json"
    assert asyncio.run(loop.tick())["view"] == "unparseable"
    _publish(redis, T0 - timedelta(hours=4), rec(WALKWAY, "broken", T0 - timedelta(hours=5), **BROKEN_NUMBERS))
    summary = asyncio.run(loop.tick())
    assert summary["view"] == "stale" and summary["new"] == []
    assert composed == [] and rows == [] and SELF_FINDING_STATE_KEY not in redis.kv


def test_quiet_hours_hold_the_finding_and_it_is_told_later(monkeypatch):
    hour = T0.hour
    loop, outreach, redis, rows, composed, delivered, now, _ = _setup(
        monkeypatch, quiet_start_hour=hour, quiet_end_hour=(hour + 1) % 24)
    import scripts.endogenous_outreach as eo

    monkeypatch.setattr(eo.EndogenousOutreach, "_gate_inputs", _gate_at(eo, T0))
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    for _ in range(3):
        assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["quiet_hours"]
    assert composed == [] and [r["reason"] for r in rows] == ["quiet_hours"]
    outreach.quiet_start_hour = outreach.quiet_end_hour = -1
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"] and len(delivered) == 1


def _gate_at(eo, when):
    """Pin the outreach gate's local hour to ``when`` (the real gate reads the wall clock)."""
    real = eo.EndogenousOutreach._gate_inputs

    def gate(self):
        inp = real(self)
        return inp.__class__(**{**inp.__dict__, "local_hour": when.hour})

    return gate


def test_arousal_strained_still_holds(monkeypatch):
    from scripts import endogenous_outreach as eo

    loop, _, redis, rows, composed, _, _, _ = _setup(monkeypatch)
    real = eo.EndogenousOutreach._gate_inputs
    monkeypatch.setattr(eo.EndogenousOutreach, "_gate_inputs",
                        lambda self: real(self).__class__(**{**real(self).__dict__, "arousal_level": "strained"}))
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["arousal_strained"]
    assert composed == []


def test_naming_another_faculty_is_refused_and_counts_toward_giving_up(monkeypatch):
    loop, _, redis, rows, composed, delivered, now, texts = _setup(
        monkeypatch, text="My act.prime.predict bets and act.actions.journal.make.walkway_grade both broke.")
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    for _ in range(3):
        result = asyncio.run(loop.tick())["results"][0]
        assert result["reason"] == "named_ungrounded_signal" and result["offending_terms"] == [PRIME]
    assert asyncio.run(loop.tick())["pending"] == 0 and delivered == [] and len(composed) == 3


def test_grounding_guard_passes_on_the_composed_fixture(monkeypatch):
    """The message a real record grounds passes both guards: the faculty name check and
    offer_message's closed-vocabulary signal check."""
    from scripts.outreach_vocabulary import find_ungrounded_faculty_mentions, find_ungrounded_signal_mentions

    text = (f"My walkway grade journal ({WALKWAY}) stopped running: 22 of its last 28 tries never ran. "
            "The six that ran all worked, so I think the trigger fails, not the writing.")
    assert find_ungrounded_faculty_mentions(text, [WALKWAY]) == []
    assert find_ungrounded_signal_mentions(text, []) == []
    loop, _, redis, rows, _, delivered, _, _ = _setup(monkeypatch, text=text)
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]
    assert delivered[0]["text"] == text


def test_pass_is_final(monkeypatch):
    loop, _, redis, rows, _, delivered, _, _ = _setup(monkeypatch, text="PASS")
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["orion_passed"]
    assert asyncio.run(loop.tick())["pending"] == 0 and delivered == []


def test_no_daily_cap_and_sends_do_not_consume_it(monkeypatch):
    import time

    loop, outreach, redis, _, _, delivered, _, _ = _setup(monkeypatch, daily_cap=0, min_cooldown_sec=86400.0)
    outreach._last_outreach_at = time.time()
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]
    assert outreach._sent_today == 0


def test_unreadable_decision_log_tells_nothing_but_keeps_the_finding(monkeypatch):
    loop, _, redis, rows, composed, _, _, _ = _setup(monkeypatch)
    loop._pool_provider = lambda: None
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    summary = asyncio.run(loop.tick())
    assert summary["blocked"] == "decision_log_unreadable" and summary["pending"] == 1 and composed == []


def test_count_sent_on_excludes_self_findings():
    import inspect

    from scripts import endogenous_outreach_decisions as d

    assert "'self_finding_outreach'" in inspect.getsource(d.count_sent_on)


def test_faculty_shape_does_not_flag_prose():
    from scripts.outreach_vocabulary import find_ungrounded_faculty_mentions

    text = "I had to act fast. The forecast says rain, and turn signals matter. Vision is hard."
    assert find_ungrounded_faculty_mentions(text, []) == []
    assert find_ungrounded_faculty_mentions("energy.bill and forecast.office.temp_c", []) == [
        "energy.bill", "forecast.office.temp_c"]


def test_shortened_ids_file_names_and_capitals_are_not_flagged():
    from scripts.outreach_vocabulary import find_ungrounded_faculty_mentions

    grounded = ["turn.self.steps"]
    assert find_ungrounded_faculty_mentions("turn.self is off", grounded) == []
    assert find_ungrounded_faculty_mentions("See vision.py and body.json.", grounded) == []
    assert find_ungrounded_faculty_mentions("I act.Then I paused", grounded) == []
    assert find_ungrounded_faculty_mentions("forecast.office was hot", grounded) == ["forecast.office"]
    assert find_ungrounded_faculty_mentions("vision.percept_expectation drifted", grounded) == [
        "vision.percept_expectation"]


def test_a_newer_finding_replaces_an_untold_older_one(monkeypatch):
    loop, outreach, redis, rows, composed, delivered, now, _ = _setup(monkeypatch)
    real = __import__("scripts.endogenous_outreach", fromlist=["x"]).EndogenousOutreach._gate_inputs
    strained = {"on": True}

    def gate(self):
        inp = real(self)
        return inp.__class__(**{**inp.__dict__, "arousal_level": "strained" if strained["on"] else inp.arousal_level})

    monkeypatch.setattr(type(outreach), "_gate_inputs", gate)
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    asyncio.run(loop.tick())  # held by strain
    now["t"] = T0 + timedelta(days=3)
    _publish(redis, now["t"], rec(WALKWAY, "reliable", now["t"], **RELIABLE_NUMBERS))
    strained["on"] = False
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["sent"]
    assert [r["reason"] for r in rows] == ["arousal_strained", "superseded", "sent"]
    assert json.loads(rows[1]["finding_ids"]) == [f"self_finding:{WALKWAY}:broken@{T0.isoformat()}"]
    assert _sent(rows)[0]["finding_kinds"] == ["recovered"]
    assert "It was broken" in composed[0] and "became broken." not in composed[0]


def test_a_waiting_finding_gets_fresh_numbers(monkeypatch):
    loop, outreach, redis, rows, composed, _, now, _ = _setup(monkeypatch)
    loop._pool_provider = lambda: None  # held: decision log unreadable
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    asyncio.run(loop.tick())
    now["t"] = T0 + timedelta(hours=1)
    _publish(redis, now["t"], rec(WALKWAY, "broken", T0, **{**BROKEN_NUMBERS, "n_failed_to_execute": 25}))
    loop._pool_provider = lambda: _Pool(rows)
    asyncio.run(loop.tick())
    assert "25 of its 31 tries never ran" in composed[0]


def test_decision_log_switched_off_tells_nothing(monkeypatch):
    from scripts import endogenous_outreach_decisions as d

    loop, _, redis, rows, composed, _, _, _ = _setup(monkeypatch)
    monkeypatch.setattr(d, "decision_log_enabled", lambda: False)
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    summary = asyncio.run(loop.tick())
    assert summary["blocked"] == "decision_log_unreadable" and composed == []


def test_an_older_snapshot_is_ignored(monkeypatch):
    loop, _, redis, rows, _, delivered, now, _ = _setup(monkeypatch)
    _publish(redis, T0, rec(WALKWAY, "broken", T0, **BROKEN_NUMBERS))
    asyncio.run(loop.tick())
    now["t"] = T0 + timedelta(minutes=10)
    _publish(redis, T0 - timedelta(hours=1), rec(WALKWAY, "reliable", T0 - timedelta(hours=1), **RELIABLE_NUMBERS))
    assert asyncio.run(loop.tick())["new"] == [] and len(delivered) == 1


def test_a_change_without_status_since_is_evaluated_again_later():
    from scripts.self_finding_outreach import OutreachState, evaluate

    no_since = rec(WALKWAY, "broken", T0).model_copy(update={"status_since": None})
    cursor, new = evaluate(OutreachState(), SelfMonitorStateV1(computed_at=T0, window_days=7.0, faculties=[no_since]), now=T0)
    assert new == [] and WALKWAY not in cursor.last_counted
    t1 = T0 + timedelta(hours=1)
    _, new = evaluate(cursor, SelfMonitorStateV1(computed_at=t1, window_days=7.0, faculties=[rec(WALKWAY, "broken", t1)]), now=t1)
    assert [f.kind for f in new] == ["became_broken"]
