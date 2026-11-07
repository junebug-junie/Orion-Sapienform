"""Office visits get attention and one curious message (office camera design step C).

The reach-out runs against a real ``EndogenousOutreach`` (only generation, delivery side effects and the
decision-log write are stubbed), so a gate change in that module shows up here. The database is a fake
that keeps the claim column, so "a restart never sends twice" is exercised through the same SQL calls.
Household names are placeholders.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from orion.physiology.event_wants import MIN_SOURCE_HISTORY, event_want_id

T0 = datetime(2026, 11, 5, 1, 24, 7, tzinfo=timezone.utc)   # 18:24 in Denver
TZ = ZoneInfo("America/Denver")
SID = "cam0-1f48ba3068b16a16cf7ba9b7e68ecabd"

# The real 11-05 visit's first entries (live vision_office_session row).
LIVE_TIMELINE = [
    ("2026-11-05T01:23:48+00:00", "standing in a room, looking at a box on the floor", "e1"),
    ("2026-11-05T01:24:00+00:00", "organizing items in a storage bin", "e2"),
    ("2026-11-05T01:24:15+00:00", "sitting at a desk, working on a computer with a scenic wallpaper", "e3"),
    ("2026-11-05T01:24:30+00:00", "sitting at a desk, opening a box", "e4"),
    ("2026-11-05T01:26:06+00:00", "sitting at a desk, unpacking a plastic bag and handling a piece of equipment", "e5"),
    ("2026-11-05T01:26:54+00:00", "reaching into a cabinet", "e6"),
]


def _tl(entries=LIVE_TIMELINE):
    return [{"at": at, "text": t, "repeats": 1, "event_id": e} for at, t, e in entries]


def _session(sid=SID, *, start=T0, end=T0 + timedelta(minutes=7), timeline=None, basis="prior", people=None,
             unknown=0, ask_id=None, status="closed", reached=None):
    people = people if people is not None else [{"name": "Juniper", "basis": basis, "individual_ids": ["i1"]}]
    return {"session_id": sid, "stream_id": "cam0", "status": status, "started_at": start,
            "last_seen_at": end or start, "ended_at": end, "who_text": "Juniper (probably)",
            "who": {"people": people, "unknown": unknown}, "unknown_people": unknown,
            "timeline": _tl() if timeline is None else timeline, "label": "Juniper (probably) in the office: x",
            "ask_id": ask_id, "reached_out_at": reached}


# --------------------------------------------------------------------------- pure


def test_a_visit_has_a_story_only_with_enough_entries_over_enough_time():
    from scripts.office_reach_out import has_story

    s = _session(status="open", end=None)
    assert not has_story(s, T0 + timedelta(minutes=2, seconds=59), min_entries=4, min_minutes=3)
    assert has_story(s, T0 + timedelta(minutes=3), min_entries=4, min_minutes=3)
    assert not has_story(_session(timeline=_tl()[:3]), T0 + timedelta(hours=1), min_entries=4, min_minutes=3)
    # A closed visit is measured to its end, not to now.
    short = _session(end=T0 + timedelta(minutes=2))
    assert not has_story(short, T0 + timedelta(hours=1), min_entries=4, min_minutes=3)


def test_a_prior_only_who_says_it_is_a_guess_and_a_match_does_not():
    from scripts.office_reach_out import who_line

    assert "probably you. That is a guess" in who_line(_session(basis="prior"))
    assert "did not match your face" in who_line(_session(basis="prior"))
    matched = who_line(_session(basis="matched"))
    assert "guess" not in matched and "matched you" in matched
    nobody = who_line(_session(people=[{"name": "Housemate", "basis": "likely", "individual_ids": []}]))
    assert nobody.startswith("Who: not you") and "Housemate (maybe)" in nobody


def test_an_unknown_visitor_is_mentioned_with_its_card_never_asked_again():
    from scripts.office_reach_out import build_prompt, who_line

    s = _session(unknown=1, ask_id="ask-1")
    assert "someone I don't recognize (there is a \"Who is this?\" card for them in the Hub)" in who_line(s)
    assert "do not ask who they are here" in build_prompt(s, TZ, now=T0)


def test_the_camera_notes_quote_what_was_seen_with_local_times():
    from scripts.office_reach_out import appendix, build_prompt

    notes = appendix(_session(), TZ)
    assert notes.splitlines()[1] == "18:23 standing in a room, looking at a box on the floor"
    assert "18:26 sitting at a desk, unpacking a plastic bag and handling a piece of equipment" in notes
    assert notes.splitlines()[-1].startswith("Who: probably you. That is a guess")
    prompt = build_prompt(_session(), TZ, now=T0)
    assert "from 18:24 to 18:31, America/Denver time" in prompt
    assert "- 18:26 reaching into a cabinet" in prompt
    assert "did NOT recognize her" in prompt and "ONE question" in prompt
    # A matched visit does not carry the "did not recognize" instruction.
    assert "did NOT recognize" not in build_prompt(_session(basis="matched"), TZ, now=T0)


def test_a_long_visit_is_spread_over_the_whole_visit_in_the_notes():
    from scripts.office_reach_out import APPENDIX_ENTRIES_CAP, appendix

    tl = [{"at": (T0 + timedelta(minutes=k)).isoformat(), "text": f"activity number {k}", "repeats": 1,
           "event_id": f"e{k}"} for k in range(20)]
    lines = appendix(_session(timeline=tl, end=T0 + timedelta(minutes=25)), TZ).splitlines()
    shown = lines[1:1 + APPENDIX_ENTRIES_CAP]
    assert shown[0].endswith("activity number 0") and shown[-1].endswith("activity number 19")
    assert lines[1 + APPENDIX_ENTRIES_CAP] == f"(+{20 - APPENDIX_ENTRIES_CAP} more)"


def test_novelty_rests_at_zero_for_a_routine_visit_and_is_one_for_a_new_one():
    from scripts.office_reach_out import distinct_activities, novelty

    routine = distinct_activities(_tl())
    assert novelty(routine, routine) == 0.0
    assert novelty(routine, []) == 1.0
    assert novelty([], routine) is None
    # Re-worded routine (an extra word, "the" for "a") is still the same activity.
    reworded = distinct_activities([{"text": "sitting at a desk, working on a computer"}])
    assert novelty(reworded, distinct_activities([{"text": "sitting at the desk, working on a computer screen"}])) == 0.0


def _history(n, *, timeline=None):
    """n earlier visits with a story, one a day, all doing the same routine desk work."""
    routine = [("x", "sitting at a desk, working on a computer", "r1"), ("x", "reaching into a cabinet", "r2"),
               ("x", "organizing items in a storage bin", "r3"), ("x", "standing in a room, looking around", "r4")]
    out = []
    for k in range(n):
        start = T0 - timedelta(days=n - k)
        tl = [{"at": (start + timedelta(minutes=i)).isoformat(), "text": t, "repeats": 1, "event_id": f"h{k}{e}"}
              for i, (_, t, e) in enumerate(timeline or routine)]
        out.append(_session(f"cam0-h{k}", start=start, end=start + timedelta(minutes=6), timeline=tl))
    return out


def test_no_salience_under_the_markets_minimum_history():
    from scripts.office_reach_out import plan_wants

    plan = plan_wants(_history(MIN_SOURCE_HISTORY - 1) + [_session()], now=T0 + timedelta(minutes=10),
                      lookback=timedelta(hours=12), min_entries=4, min_minutes=3)
    assert [(p["session_id"], p["salience"], p["history_n"]) for p in plan] == [(SID, None, MIN_SOURCE_HISTORY - 1)]


def test_a_new_kind_of_visit_ranks_above_routine_and_routine_rests_at_zero():
    from scripts.office_reach_out import plan_wants

    now = T0 + timedelta(minutes=10)
    hist = _history(MIN_SOURCE_HISTORY + 5)
    new = plan_wants(hist + [_session()], now=now, lookback=timedelta(hours=12), min_entries=4, min_minutes=3)
    assert new[0]["salience"] > 0.9    # unpacking equipment is unlike every routine visit
    routine = _history(1)[0] | {"session_id": "cam0-today", "started_at": T0, "ended_at": T0 + timedelta(minutes=6)}
    rt = plan_wants(hist + [routine], now=now, lookback=timedelta(hours=12), min_entries=4, min_minutes=3)
    assert rt[0]["novelty"] == 0.0 and rt[0]["salience"] == 0.0


# --------------------------------------------------------------------------- the loop


class _Conn:
    def __init__(self, state):
        self.s = state

    async def fetchval(self, sql, *args):
        from scripts import office_reach_out as oro

        if "to_regclass" in sql:
            return self.s.get("table", True)
        if sql == oro.COLUMN_SQL:
            return 2 if self.s.get("columns", True) else 0
        raise AssertionError(sql[:60])

    async def fetch(self, sql, *args):
        from scripts import office_reach_out as oro

        if sql in (oro.SESSIONS_SQL, oro.SESSIONS_SQL_NO_REACH):
            self.s["sessions_sql"].append(sql)
            return [dict(r) for r in self.s["sessions"]]
        if sql == oro.POSTED_SQL:
            return [{"want_id": w} for w in self.s["ledger"] if w in args[0]]
        if sql == oro.FAILURES_SQL:
            return self.s.get("failures", [])
        if sql == oro.CLAIM_SQL:
            sid, at, decision_id = args
            for r in self.s["sessions"]:
                if r["session_id"] == sid and r["reached_out_at"] is None:
                    r["reached_out_at"], r["decision"] = at, decision_id
                    return [{"session_id": sid}]
            return []
        raise AssertionError(sql[:60])

    async def execute(self, sql, *args):
        from scripts import office_reach_out as oro

        assert sql == oro.RELEASE_SQL
        sid, decision_id = args
        self.s["released"].append(sid)
        for r in self.s["sessions"]:
            if r["session_id"] == sid and r.get("decision") == decision_id:
                r["reached_out_at"], r["decision"] = None, None


class _Pool:
    def __init__(self, state):
        self.state = state

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _Conn(pool.state)

            async def __aexit__(self, *a):
                return False

        return _Ctx()


NOW = T0 + timedelta(minutes=20)


def _setup(monkeypatch, *, sessions=None, text="Looked like you were unpacking some equipment at 18:26. What are you building?",
           event_wants=False, bid_outcome="funded", **outreach_kw):
    from scripts import endogenous_outreach as eo
    from scripts import endogenous_outreach_decisions as decisions_mod
    from scripts import office_reach_out as oro

    rows: list[dict] = []
    monkeypatch.setattr(decisions_mod, "record_decision", lambda result, **kw: rows.append(result))
    monkeypatch.setattr(eo, "_seconds_since_juniper_spoke", outreach_kw.pop("spoke", lambda: None))
    composed: list[str] = []
    delivered: list[dict] = []
    lanes: list[str] = []
    texts = list(text) if isinstance(text, list) else None

    async def fake_generate(self, prompt, session_id, correlation_id, retrieval_query=None, lane="agent_first"):
        composed.append(prompt)
        lanes.append(lane)
        return (texts.pop(0) if texts else text), {"fcc_model_label": "test"}

    async def fake_deliver(self, **kw):
        delivered.append(kw)

    monkeypatch.setattr(eo.EndogenousOutreach, "_generate", fake_generate)
    monkeypatch.setattr(eo.EndogenousOutreach, "_deliver", fake_deliver)
    base = dict(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=0,
                quiet_start_hour=-1, quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0,
                notify_channel="orion:notify:in_app", fallback_session_id="orion_outreach",
                trigger_evaluator=lambda: None, timezone_name="UTC")
    base.update(outreach_kw)
    outreach = eo.EndogenousOutreach(**base)
    state = {"sessions": sessions if sessions is not None else [_session()], "ledger": set(), "released": [],
             "sessions_sql": [], "lanes": lanes}
    bids: list = []

    async def bid(want):
        bids.append(want)
        return SimpleNamespace(outcome=bid_outcome, reason="ok", value_rank=0.5)

    loop = oro.OfficeReachOut(reach_out_enabled=True, event_wants_enabled=event_wants,
                              pool_provider=lambda: _Pool(state), outreach_provider=lambda: outreach, bid=bid,
                              timezone_name="America/Denver", clock=lambda: NOW)
    return loop, outreach, state, rows, composed, delivered, bids


def test_one_visit_one_message_with_the_camera_notes_and_provenance(monkeypatch):
    loop, _, state, rows, composed, delivered, _ = _setup(monkeypatch)
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["sent"]
    assert len(composed) == 1 and len(delivered) == 1
    body = delivered[0]["text"]
    assert body.startswith("Looked like you were unpacking some equipment at 18:26. What are you building?\n\n"
                           "What the office camera saw")
    assert "18:26 sitting at a desk, unpacking a plastic bag" in body and "That is a guess" in body
    sent = [r for r in rows if r["reason"] == "sent"][0]
    assert sent["source"] == "office_reach_out" and sent["office_session_id"] == SID
    assert sent["timeline_times"][0] == "2026-11-05T01:23:48+00:00" and sent["who_basis"] == "prior"
    assert f"vision_office_session:{SID}" in sent["evidence_refs"] and "vision_events:e5" in sent["evidence_refs"]
    assert delivered[0]["provenance"]["prompt_text"] == composed[0]
    assert state["sessions"][0]["decision"] == sent["decision_id"]
    # Told once: the claim is in the row, so the next tick (and a restarted Hub) sends nothing.
    assert asyncio.run(loop.tick())["pending"] == 0
    from scripts import office_reach_out as oro

    fresh = oro.OfficeReachOut(reach_out_enabled=True, event_wants_enabled=False, pool_provider=lambda: _Pool(state),
                               outreach_provider=lambda: None, clock=lambda: NOW)
    assert asyncio.run(fresh.tick())["pending"] == 0 and len(delivered) == 1


def test_every_visit_may_reach_out_no_daily_cap(monkeypatch):
    second = _session("cam0-second", start=T0 + timedelta(minutes=9), end=T0 + timedelta(minutes=16))
    loop, _, _, _, _, delivered, _ = _setup(monkeypatch, sessions=[_session(), second], daily_cap=0)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent", "sent"]
    assert len(delivered) == 2


def test_a_visit_without_a_story_is_never_messaged(monkeypatch):
    loop, _, _, rows, composed, _, _ = _setup(monkeypatch, sessions=[_session(timeline=_tl()[:2])])
    assert asyncio.run(loop.tick())["pending"] == 0 and composed == [] and rows == []


def test_an_active_chat_turn_holds_it_and_records_the_block_once(monkeypatch):
    loop, outreach, state, rows, composed, delivered, _ = _setup(monkeypatch)
    outreach.register_connection("c1", asyncio.Queue(), {})
    outreach.note_busy("c1")
    for _ in range(3):
        assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["turn_in_flight"]
    assert composed == [] and delivered == [] and state["sessions"][0]["reached_out_at"] is None
    assert [r["reason"] for r in rows] == ["turn_in_flight"]
    outreach.note_idle("c1")
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]


def test_quiet_hours_delay_the_message(monkeypatch):
    hour = datetime.now(timezone.utc).hour
    # A two-hour window around the wall clock, so the test cannot cross its edge.
    loop, outreach, _, _, composed, delivered, _ = _setup(monkeypatch, quiet_start_hour=(hour - 1) % 24,
                                                         quiet_end_hour=(hour + 1) % 24)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["quiet_hours"]
    assert composed == [] and delivered == []
    outreach.quiet_start_hour = outreach.quiet_end_hour = -1
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]


def test_juniper_typing_during_composition_releases_the_claim(monkeypatch):
    loop, outreach, state, rows, _, delivered, _ = _setup(monkeypatch)
    outreach.register_connection("c1", asyncio.Queue(), {})
    from scripts import endogenous_outreach as eo

    real_generate = eo.EndogenousOutreach._generate

    async def busy_generate(self, prompt, session_id, correlation_id, retrieval_query=None, lane="agent_first"):
        out = await real_generate(self, prompt, session_id, correlation_id, retrieval_query, lane)
        self.note_busy("c1")   # she starts a turn while Orion writes
        return out

    monkeypatch.setattr(eo.EndogenousOutreach, "_generate", busy_generate)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["turn_in_flight"]
    assert delivered == [] and state["released"] == [SID] and state["sessions"][0]["reached_out_at"] is None
    outreach.note_idle("c1")
    monkeypatch.setattr(eo.EndogenousOutreach, "_generate", real_generate)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]


def test_a_pass_or_a_message_with_no_question_is_retried_then_dropped(monkeypatch):
    loop, _, state, rows, composed, delivered, _ = _setup(
        monkeypatch, text=["PASS", "You were unpacking equipment at 18:26.", "PASS", "late?"])
    reasons = [asyncio.run(loop.tick())["results"][0]["reason"] for _ in range(3)]
    assert reasons == ["orion_passed", "no_question", "orion_passed"]
    assert delivered == [] and state["sessions"][0]["reached_out_at"] is None
    assert asyncio.run(loop.tick())["pending"] == 0      # given up after max_attempts (3)
    assert len(composed) == 3 and [r["reason"] for r in rows] == reasons


def test_without_the_reach_out_columns_nothing_is_sent_but_wants_still_post(monkeypatch):
    loop, _, state, _, composed, delivered, bids = _setup(monkeypatch, sessions=_history(25) + [_session()],
                                                          event_wants=True)
    state["columns"] = False
    summary = asyncio.run(loop.tick())
    assert summary["blocked"] == "no_reach_out_columns" and composed == [] and delivered == []
    from scripts import office_reach_out as oro

    assert state["sessions_sql"] == [oro.SESSIONS_SQL_NO_REACH]
    assert [w.event_ref for w in bids] == [f"office_session:{SID}"]


def test_the_visit_posts_one_perception_office_want(monkeypatch):
    from scripts import office_reach_out as oro

    loop, _, state, _, _, _, bids = _setup(monkeypatch, sessions=_history(25) + [_session()], event_wants=True)
    loop.reach_out_enabled = False
    s = asyncio.run(loop.tick())
    assert s["wants"]["posted"] == 1 and len(bids) == 1
    w = bids[0]
    assert w.family == oro.FAMILY == "perception.office" and w.verb == "look" and w.tier == 2
    assert w.want_id == event_want_id("office_session", SID) and w.event_ref == f"office_session:{SID}"
    assert w.salience > 0.9 and w.opened_at == T0
    # Posted once: the next tick does not bid again.
    asyncio.run(loop.tick())
    assert len(bids) == 1


def test_no_want_under_twenty_past_visits_but_the_message_still_goes(monkeypatch):
    loop, _, _, _, _, delivered, bids = _setup(monkeypatch, event_wants=True)
    s = asyncio.run(loop.tick())
    assert s["wants"] == {"posted": 0, "no_history": 1, "already": 0, "unanswered": 0} and bids == []
    assert [r["reason"] for r in s["results"]] == ["sent"] and len(delivered) == 1


def test_the_family_is_registered_for_the_market():
    from pathlib import Path

    from orion.physiology.families import load_families

    fams, err = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
    assert err is None
    cfg = fams.get("perception.office")
    assert cfg is not None and cfg.tier == 2 and cfg.deferrable_window_s == 7200


@pytest.mark.parametrize("key", ["HUB_OFFICE_REACH_OUT_ENABLED", "HUB_OFFICE_EVENT_WANTS_ENABLED"])
def test_flags_ship_on(key):
    from app.settings import Settings

    assert Settings.model_fields[key].default is True


# --------------------------------------------------------------------------- review fixes


def test_a_visit_without_juniper_is_not_messaged_the_card_covers_a_stranger(monkeypatch):
    stranger = _session(people=[], unknown=1, ask_id="ask-1")
    housemate = _session("cam0-hm", people=[{"name": "Housemate", "basis": "named", "individual_ids": ["h"]}])
    loop, _, _, rows, composed, delivered, _ = _setup(monkeypatch, sessions=[stranger, housemate])
    assert asyncio.run(loop.tick())["pending"] == 0 and composed == [] and delivered == [] and rows == []


def test_an_unknown_visitor_is_never_asked_about_even_before_its_card_opens():
    from scripts.office_reach_out import build_prompt

    assert "do not ask who they are here" in build_prompt(_session(unknown=1, ask_id=None), TZ, now=T0)


def test_an_unsure_match_also_may_not_claim_recognition():
    from scripts.office_reach_out import build_prompt

    assert "did NOT recognize her" in build_prompt(_session(basis="likely"), TZ, now=T0)


def test_a_message_claiming_recognition_on_a_guess_is_refused(monkeypatch):
    loop, _, state, rows, _, delivered, _ = _setup(
        monkeypatch, text="I recognized you at the desk at 18:26. What are you building?")
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["claimed_recognition"]
    assert delivered == [] and state["sessions"][0]["reached_out_at"] is None
    # A real match may say so.
    loop, _, _, _, _, delivered, _ = _setup(monkeypatch, sessions=[_session(basis="matched")],
                                            text="I recognized you at the desk at 18:26. What are you building?")
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]


def test_three_questions_is_not_one_question(monkeypatch):
    loop, _, _, _, _, delivered, _ = _setup(monkeypatch, text="Unpacking? Building? Fixing?")
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["no_question"] and delivered == []


def test_a_visit_held_overnight_says_which_day_it_was():
    from scripts.office_reach_out import appendix, build_prompt

    morning = T0 + timedelta(hours=14)   # 08:24 the next day in Denver
    prompt = build_prompt(_session(), TZ, now=morning)
    assert "(yesterday from 18:24 to 18:31" in prompt and "the visit is over, it ended 13 hours ago" in prompt
    assert appendix(_session(), TZ, morning).startswith("What the office camera saw yesterday (")
    assert appendix(_session(), TZ, T0 + timedelta(minutes=20)).startswith("What the office camera saw (")


def test_novelty_compares_what_was_done_not_where():
    from scripts.office_reach_out import distinct_activities, novelty

    eating = distinct_activities([{"text": "sitting at a desk, eating"}])
    typing = distinct_activities([{"text": "sitting at a desk, typing"}])
    assert novelty(eating, typing) == 1.0


def test_a_long_visit_is_measured_on_the_same_slice_as_a_short_one():
    from scripts.office_reach_out import story_prefix

    tl = [{"at": (T0 + timedelta(minutes=k)).isoformat(), "text": f"activity number {k}", "repeats": 1}
          for k in range(30)]
    prefix = story_prefix(_session(timeline=tl, end=T0 + timedelta(minutes=40)), min_entries=4, min_minutes=3)
    assert [e["text"] for e in prefix] == [f"activity number {k}" for k in range(4)]


def test_attempts_logged_before_a_restart_still_count(monkeypatch):
    loop, _, state, _, composed, _, _ = _setup(monkeypatch)
    state["failures"] = [{"sid": SID, "n": 3}]
    assert asyncio.run(loop.tick())["pending"] == 0 and composed == []


def test_a_broken_want_producer_never_holds_the_message(monkeypatch):
    from scripts import office_reach_out as oro

    loop, _, _, _, _, delivered, _ = _setup(monkeypatch, event_wants=True)
    monkeypatch.setattr(oro, "plan_wants", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("want_ledger gone")))
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"] and len(delivered) == 1


def test_office_sends_do_not_count_toward_the_background_daily_cap():
    import inspect

    from scripts import endogenous_outreach_decisions as d

    assert "'office_reach_out'" in inspect.getsource(d.count_sent_on)



def test_office_message_is_composed_on_the_chat_lane(monkeypatch) -> None:
    """2026-11-07 (Juniper): office messages skip the agent lane. Its 210 s timeout made the
    first live office message reach her 5 minutes into the visit."""
    loop, _, state, _, composed, delivered, _ = _setup(monkeypatch)
    asyncio.run(loop.tick())
    assert composed and delivered
    assert state["lanes"] == ["chat"]


def test_chat_lane_compose_is_one_chat_attempt_with_the_full_budget(monkeypatch) -> None:
    from scripts import endogenous_outreach as eo

    calls: list[dict] = []

    async def attempt(self, **kw):
        calls.append(kw)
        return "Hi -- what are you building?", {}

    monkeypatch.setattr(eo.EndogenousOutreach, "_attempt_unified_turn", attempt)
    outreach = eo.EndogenousOutreach(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=0,
                                     quiet_start_hour=-1, quiet_end_hour=-1, timeout_sec=300.0,
                                     agent_lane_timeout_sec=210.0, notify_channel="orion:notify:in_app",
                                     fallback_session_id="orion_outreach", trigger_evaluator=lambda: None,
                                     timezone_name="UTC")
    outreach._bus = object()
    text, debug = asyncio.run(outreach.compose("p", "corr-1", lane="chat"))
    assert text and debug["lane"] == "chat"
    assert len(calls) == 1 and calls[0]["fcc_model_label"] is None and calls[0]["timeout_sec"] == 300.0


def test_economy_sated_hold_delays_the_visit_message_then_it_is_sent(monkeypatch):
    """Body economy Patch D: the office door obeys its binding bid; a hold leaves the visit pending."""
    from orion.schemas.physiology_want import WantDecisionV1

    import scripts.outreach_bind as ob
    from scripts.settings import settings

    monkeypatch.setattr(settings, "ORION_PHYSIOLOGY_BID_HUB_OUTREACH", "on", raising=False)
    answers = [("deferred", "sated", 0.0), ("funded", "net_positive", 0.3)]
    wants = []

    async def abid(want, timeout_sec):
        wants.append(want)
        outcome, reason, net = answers.pop(0)
        now = datetime.now(timezone.utc)
        return WantDecisionV1(decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id,
                              family=want.family, producer=want.producer, verb=want.verb, tier=want.tier,
                              decided_at=now, outcome=outcome, reason=reason, net=net, retry_after_sec=300.0,
                              want=want, why=f"hub.outreach: {reason}")

    async def arelease(want_id):
        return 1

    monkeypatch.setattr(ob, "_abid", abid)
    monkeypatch.setattr(ob, "_arelease", arelease)
    loop, outreach, state, rows, composed, delivered, _ = _setup(monkeypatch)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["economy_held"]
    assert composed == [] and delivered == [] and state["sessions"][0]["reached_out_at"] is None
    assert [r["reason"] for r in rows] == ["economy_held"] and rows[0]["economy"]["reason"] == "sated"
    assert wants[0].cost.juniper_msgs == 0
    for w in outreach._binder.wants.values():
        w.retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"] and len(delivered) == 1
    assert len(wants) == 2 and len({w.want_id for w in wants}) == 1
