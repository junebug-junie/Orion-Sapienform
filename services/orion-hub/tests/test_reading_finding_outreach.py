"""Reading findings reach Juniper through the real outreach gates: one message per read, no cap.

Runs against a real ``EndogenousOutreach`` (only generation, delivery side effects and the
decision-log write are stubbed), so a gate change in that module shows up here.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

T_READ = datetime(2026, 10, 11, 14, 13, 21, tzinfo=timezone.utc)
NOW = T_READ + timedelta(hours=6)
SEED = "digest_item:d7daaa9b-08e5-4db9-89ff-33819093430c:item:cluster:22b63427ceef0ac9:a6250952bd65a37d"
URL = "https://www.nasa.gov/news-release/nasa-to-conduct-flyover-engage-nfl-fans-at-metlife-stadium/"

# The real accepted claim (production, 2026-10-11), as Stage 2 stored it.
LIVE_CLAIM = {
    "quote": "NASA Administrator Jared Isaacman will pilot one of three F-5 fighter jets",
    "receipt": {"reason": "accepted", "outcome": "accepted_provisional", "span_end": 2234, "span_start": 2160,
                "decision_id": "reading-aae77003-1145-5385-926f-cb78f3dc4b37",
                "proposal_id": "reading-86d3b2a0-3ae4-52d5-8561-1e1ddd03b0a9",
                "assertion_id": "assertion-f29721db-52ae-5017-8b26-107fceafbd0d",
                "content_sha256": "f758f22c88470dd4cf5f672a6639ff15ecc10ea5f93db990013a038eb5609961",
                "representation": "tool_digest"},
    "object_id": "sub-concept-wp-read-c79f0617a2ef6b3c", "predicate": "associated_with",
    "subject_id": "sub-concept-wp-read-0568d894ec192184",
    "statement_text": "Per advisory M26-080, NASA Administrator Jared Isaacman pilots one of the three F-5 "
                      "fighter jets of the Freedom 250 flyover team for the MetLife stop.",
}

READ_ROW = {
    "seed_id": SEED, "request_id": None, "url": URL, "title": "NASA to Conduct Flyover, Engage NFL Fans at MetLife Stadium",
    "status": "done", "stage2_status": "done", "created_at": T_READ - timedelta(hours=1),
    "handoff_at": T_READ - timedelta(minutes=30), "stage2_completed_at": T_READ, "landing_at": None,
    "handoff_json": json.dumps({"read_evidence": [{"url": URL, "tool_name": "WebFetch", "content_chars": 5000}]}),
    "stage2_result_json": json.dumps({"summary": "NASA will fly F-5 jets over MetLife Stadium. The administrator pilots one."}),
    "trace_id": None, "stage2_trace_id": None, "why_now": "Space news.",
}


def _link_row(projected=True):
    return {"seed_id": SEED, "stage2_completed_at": T_READ, "claim": json.dumps(LIVE_CLAIM),
            "latest_state": "provisional", "projected_at": T_READ if projected else None}


MOVE_ROW = {"run_id": "e5eb9cfcf7ea", "completed_at": T_READ + timedelta(hours=2),
            "per_prior": json.dumps([{"kind": "tested", "before": 0.55, "after": 0.7,
                                      "prior_id": "world:nasa_outreach_is_spectacle"}])}
PRIOR_TEXTS = {"world:nasa_outreach_is_spectacle": [{
    "prior_id": "world:nasa_outreach_is_spectacle", "claim": "NASA public outreach leans on spectacle",
    "why": "Test 2: nasa.gov/news-release/nasa-to-conduct-flyover-engage-nfl-fans-at-metlife-stadium -- "
           "the administrator himself flies a jet over an NFL game", "last_run_id": "e5eb9cfcf7ea"}]}


# The read's own row: a world-pulse digest item, so the chain is the read alone.
ORIGIN_ROWS = [{"leaf": SEED, "depth": 0, "seed_id": SEED, "kind": "digest_item", "url": URL,
                "title": READ_ROW["title"], "requested_by": "world_pulse",
                "invocation_context": "world_pulse", "why_now": "", "parent_request_id": None}]


class _Conn:
    def __init__(self, state):
        self.s = state

    async def fetch(self, sql, *args):
        from orion.world_pulse_read import findings as wf
        from orion.world_pulse_read import introspect
        from orion.world_pulse_read import origin

        from scripts import reading_finding_outreach as rfo

        if sql == introspect._WINDOW_SQL:
            return [_Rec(r) for r in self.s["reads"]]
        if sql == wf.READ_LINKS_SQL:
            return self.s["links"]
        if sql == wf.RUN_PRIOR_MOVES_SQL:
            return self.s["moves"]
        if sql == rfo.DECISIONS_SQL:
            return self.s["decisions"]
        if sql == rfo.LAST_SENT_SQL:
            self.s["last_sent_asked"].append(args[0])
            return [_Rec({"last_sent": self.s.get("last_sent")})]
        if sql == origin.ORIGIN_SQL:
            self.s["origin_asked"].append(args[0])
            if self.s.get("origin_fails"):
                raise RuntimeError("statement timeout")
            return [r for r in self.s["origin"] if r["leaf"] in args[0]]
        raise AssertionError(sql[:60])


class _Rec(dict):
    def __getitem__(self, k):
        return dict.get(self, k)


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


def _setup(monkeypatch, *, links=None, moves=None, decisions=None, text="I read about NASA's flyover.",
           min_spacing_sec=0.0, last_sent=None, **outreach_kw):
    outreach_kw_last_sent = last_sent
    from scripts import endogenous_outreach as eo
    from scripts import endogenous_outreach_decisions as decisions_mod
    from scripts import reading_finding_outreach as rfo

    rows: list[dict] = []
    monkeypatch.setattr(decisions_mod, "record_decision", lambda result, **kw: rows.append(result))
    monkeypatch.setattr(eo, "_seconds_since_juniper_spoke", outreach_kw.pop("spoke", lambda: None))
    composed: list[str] = []
    delivered: list[dict] = []

    async def fake_generate(self, prompt, session_id, correlation_id, retrieval_query=None):
        composed.append(prompt)
        return text, {"fcc_model_label": "test"}

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
    state = {"reads": [READ_ROW], "links": [_link_row()] if links is None else links,
             "moves": [] if moves is None else moves, "decisions": decisions or [],
             "origin": ORIGIN_ROWS, "origin_asked": [], "last_sent_asked": [],
             "last_sent": outreach_kw_last_sent}

    async def lookup(ids):
        return {i: PRIOR_TEXTS.get(i, []) for i in ids}

    loop = rfo.ReadingFindingOutreach(enabled=True, pool_provider=lambda: _Pool(state),
                                      outreach_provider=lambda: outreach, prior_lookup=lookup,
                                      min_spacing_sec=min_spacing_sec, clock=lambda: NOW)
    return loop, outreach, state, rows, composed, delivered


def test_a_live_accepted_claim_and_a_moved_belief_become_one_message(monkeypatch):
    loop, _, _, rows, composed, delivered = _setup(monkeypatch, moves=[MOVE_ROW])
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["sent"]
    assert len(composed) == 1 and len(delivered) == 1  # bundled: two changes, one message
    prompt = composed[0]
    assert "Per advisory M26-080" in prompt and LIVE_CLAIM["quote"] in prompt
    assert "Belief moved (0.55 -> 0.70): NASA public outreach leans on spectacle" in prompt
    sent = [r for r in rows if r["reason"] == "sent"][0]
    assert sent["source"] == "reading_finding_outreach" and sent["seed_id"] == SEED
    assert sent["finding_ids"] == ["belief:world:nasa_outreach_is_spectacle@e5eb9cfcf7ea",
                                   "link:assertion-f29721db-52ae-5017-8b26-107fceafbd0d"]
    assert sent["finding_kinds"] == ["belief_moved", "link_made"]
    assert "curiosity:e5eb9cfcf7ea" in sent["evidence_refs"]
    assert delivered[0]["text"].startswith("I read about NASA's flyover.\n\nSource: NASA to Conduct Flyover")
    capsule = delivered[0]["provenance"]
    assert capsule["schema"] == "outreach_provenance.v1" and capsule["prompt_text"] == prompt
    assert capsule["decision_id"] == sent["decision_id"]
    # Told once: the next tick (decision row now visible) sends nothing.
    loop2_summary = asyncio.run(loop.tick())
    assert loop2_summary["pending"] == 0 and len(delivered) == 1


def test_no_message_when_nothing_changed(monkeypatch):
    loop, _, _, rows, composed, delivered = _setup(monkeypatch, links=[])
    summary = asyncio.run(loop.tick())
    assert summary["pending"] == 0 and composed == [] and delivered == [] and rows == []


def test_an_unprojected_link_waits(monkeypatch):
    loop, _, _, _, composed, _ = _setup(monkeypatch, links=[_link_row(projected=False)])
    assert asyncio.run(loop.tick())["pending"] == 0 and composed == []


def test_quiet_hours_hold_the_message_and_record_the_block_once(monkeypatch):
    hour = datetime.now(timezone.utc).hour
    loop, outreach, _, rows, composed, delivered = _setup(
        monkeypatch, quiet_start_hour=hour, quiet_end_hour=(hour + 1) % 24)
    for _ in range(3):
        assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["quiet_hours"]
    assert composed == [] and delivered == []
    assert [r["reason"] for r in rows] == ["quiet_hours"]  # one row, not one per tick
    outreach.quiet_start_hour = outreach.quiet_end_hour = -1  # morning
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]


def test_recent_chat_and_turn_in_flight_still_block(monkeypatch):
    loop, _, _, _, composed, _ = _setup(monkeypatch, recent_chat_sec=600.0, spoke=lambda: 30.0)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["recent_chat"]
    assert composed == []
    loop, outreach, _, _, composed, _ = _setup(monkeypatch)
    import asyncio as _a

    outreach.register_connection("c1", _a.Queue(), None)
    outreach._connections["c1"]["busy"] = True  # Juniper's turn is running
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["turn_in_flight"]


def test_no_daily_cap_and_no_cooldown_for_this_kind(monkeypatch):
    loop, outreach, _, _, _, delivered = _setup(monkeypatch, daily_cap=0, min_cooldown_sec=86400.0)
    import time

    outreach._last_outreach_at = time.time()
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]
    assert len(delivered) == 1 and outreach._sent_today == 0  # does not consume the background cap


def test_a_later_finding_of_a_told_read_is_a_new_bundle_with_only_the_new_change(monkeypatch):
    told = [{"reason": "sent", "finding_ids": json.dumps(["link:assertion-f29721db-52ae-5017-8b26-107fceafbd0d"])}]
    loop, _, _, rows, composed, _ = _setup(monkeypatch, moves=[MOVE_ROW], decisions=told)
    asyncio.run(loop.tick())
    sent = [r for r in rows if r["reason"] == "sent"][0]
    assert sent["finding_ids"] == ["belief:world:nasa_outreach_is_spectacle@e5eb9cfcf7ea"]
    assert "Per advisory M26-080" not in composed[0]


def test_pass_is_final_and_empty_generation_gives_up_after_max_attempts(monkeypatch):
    loop, _, _, rows, composed, delivered = _setup(monkeypatch, text="PASS")
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["orion_passed"]
    assert asyncio.run(loop.tick())["pending"] == 0 and delivered == []

    loop, _, _, rows, composed, _ = _setup(monkeypatch, text="")
    for _ in range(3):
        assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["empty_generation"]
    assert asyncio.run(loop.tick())["pending"] == 0 and len(composed) == 3


def test_count_sent_on_excludes_reading_findings():
    from scripts import endogenous_outreach_decisions as d
    import inspect

    assert "'reading_finding_outreach'" in inspect.getsource(d.count_sent_on)


def test_the_recorded_article_summary_is_delivered_under_orions_words(monkeypatch):
    row = {**READ_ROW, "stage2_result_json": json.dumps({
        "summary": "NASA will fly F-5 jets over MetLife Stadium.",
        "article_summary": "NASA will fly three F-5 jets over MetLife Stadium for Freedom 250. "
                           "Administrator Jared Isaacman pilots one of them. The flyover is part of a national tour."})}
    loop, _, state, _, composed, delivered = _setup(monkeypatch)
    state["reads"] = [row]
    asyncio.run(loop.tick())
    assert "Administrator Jared Isaacman pilots one of them." in composed[0]
    text = delivered[0]["text"]
    assert text.startswith("I read about NASA's flyover.\n\nWhat the piece says -- NASA to Conduct Flyover")
    assert text.endswith("The flyover is part of a national tour.")


def test_rate_gate_skip_keeps_every_gate_that_protects_juniper():
    from scripts.endogenous_outreach import OutreachGateInputs, outreach_block_reason

    base = dict(enabled=True, turn_in_flight=False, local_hour=12, quiet_start_hour=-1, quiet_end_hour=-1,
                seconds_since_last_outreach=10.0, min_cooldown_sec=3600.0, sent_today=9, daily_cap=4)
    assert outreach_block_reason(OutreachGateInputs(**base)) == "daily_cap"
    assert outreach_block_reason(OutreachGateInputs(**base), skip_rate_gates=True) is None
    assert outreach_block_reason(OutreachGateInputs(**{**base, "rest_drive_cooldown_sec": 9e9}),
                                 skip_rate_gates=True) is None
    for over, reason in (({"quiet_start_hour": 11, "quiet_end_hour": 13}, "quiet_hours"),
                         ({"arousal_level": "strained"}, "arousal_strained"),
                         ({"turn_in_flight": True}, "turn_in_flight"),
                         ({"enabled": False}, "disabled"),
                         ({"recent_chat_sec": 600.0, "seconds_since_juniper_spoke": 5.0}, "recent_chat")):
        got = outreach_block_reason(OutreachGateInputs(**{**base, **over}), skip_rate_gates=True)
        assert got == reason, (over, got)
    # Door-A unchanged: it still skips quiet hours.
    assert outreach_block_reason(OutreachGateInputs(**{**base, "quiet_start_hour": 11, "quiet_end_hour": 13}),
                                 skip_schedule_gates=True) is None


def test_a_belief_moved_days_after_an_old_read_is_still_told(monkeypatch):
    old = {**READ_ROW, "stage2_completed_at": NOW - timedelta(days=4), "created_at": NOW - timedelta(days=4),
           "handoff_at": NOW - timedelta(days=4)}
    late_move = {**MOVE_ROW, "completed_at": NOW - timedelta(hours=1)}
    loop, _, state, rows, _, _ = _setup(monkeypatch, links=[], moves=[late_move])
    state["reads"] = [old]
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]
    assert [r for r in rows if r["reason"] == "sent"][0]["finding_kinds"] == ["belief_moved"]


def test_each_attempt_gets_its_own_correlation_id(monkeypatch):
    loop, _, _, rows, _, _ = _setup(monkeypatch, text="")
    asyncio.run(loop.tick())
    asyncio.run(loop.tick())
    ids = [r["correlation_id"] for r in rows]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert len({r["bundle_key"] for r in rows}) == 1


def test_a_restart_does_not_resend_a_told_read(monkeypatch):
    told = [{"reason": "sent", "finding_ids": json.dumps(["link:assertion-f29721db-52ae-5017-8b26-107fceafbd0d"])}]
    loop, _, _, _, composed, delivered = _setup(monkeypatch, decisions=told)  # fresh process, empty memory
    assert asyncio.run(loop.tick())["pending"] == 0 and composed == [] and delivered == []


def test_the_prompt_says_how_the_read_started_from_the_queue_record(monkeypatch):
    # Live 2026-10-26: the old prompt asserted "Nobody asked you to read it" and Orion wrote
    # "not because a process scheduled it" about a read the daily digest had started.
    loop, _, state, _, composed, _ = _setup(monkeypatch)
    asyncio.run(loop.tick())
    prompt = composed[0]
    assert f"1. Your daily world-pulse news run put {URL} in your reading queue. This is the read below." in prompt
    assert "Nobody asked" not in prompt and "it changed you" not in prompt
    assert state["origin_asked"] == [[SEED]]  # looked up only for reads with a message to send


def test_no_origin_lookup_when_nothing_to_send(monkeypatch):
    loop, _, state, _, composed, _ = _setup(monkeypatch, links=[])
    asyncio.run(loop.tick())
    assert composed == [] and state["origin_asked"] == []


def test_a_failed_origin_lookup_still_sends_the_message(monkeypatch):
    loop, _, state, _, composed, delivered = _setup(monkeypatch)
    state["origin_fails"] = True
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["sent"] and len(delivered) == 1
    assert "How this read started was not recorded." in composed[0]


# Live 2026-10-28 15:08:09Z: Orion reasoned, ended on a PASS line, and the reasoning plus the
# article summary reached Juniper as a chat message. 35 of 49 reading "sends" in 30 days had this
# shape (the replies ended on a PASS line).
LIVE_REASONED_PASS = (
    "Let me assess whether this single graph link warrants a proactive message to Juniper.\n\n"
    "The change to the graph is one link -- a conceptual refinement of how I think about evolution "
    "strategies. It doesn't affect any running system, no prior needs testing, no action item emerges.\n\n"
    "This is the kind of quiet internal reframe that happens constantly in the curiosity feed -- useful "
    "for coherence, not worth interrupting Juniper for.\n\nPASS"
)


def test_a_reasoned_pass_is_told_as_passed_and_never_delivered(monkeypatch):
    loop, _, _, rows, _, delivered = _setup(monkeypatch, text=LIVE_REASONED_PASS)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["orion_passed"]
    assert delivered == []
    passed = [r for r in rows if r["reason"] == "orion_passed"][0]
    assert passed["pass_shape"] == "after_reasoning"
    assert asyncio.run(loop.tick())["pending"] == 0  # told: never composed again


def _two_reads(state):
    other_seed = SEED + ":b"
    state["reads"] = [READ_ROW, {**READ_ROW, "seed_id": other_seed, "url": URL + "b"}]
    state["links"] = [_link_row(), {**_link_row(), "seed_id": other_seed, "claim": json.dumps(
        {**LIVE_CLAIM, "receipt": {**LIVE_CLAIM["receipt"], "assertion_id": "assertion-second"}})}]
    state["origin"] = ORIGIN_ROWS + [{**ORIGIN_ROWS[0], "leaf": other_seed, "seed_id": other_seed}]


def test_a_morning_pile_goes_out_one_at_a_time(monkeypatch):
    # Live 2026-10-28: findings held through quiet hours all went out between 14:11 and 15:08 UTC.
    loop, _, state, rows, composed, delivered = _setup(monkeypatch, min_spacing_sec=5400.0)
    _two_reads(state)
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["sent", "reading_spacing"]
    assert len(composed) == 1 and len(delivered) == 1  # the second is not even composed
    held = [r for r in rows if r["reason"] == "reading_spacing"]
    assert len(held) == 1 and held[0]["spacing_wait_sec"] == 5400.0
    # Still held next tick, recorded once, and still pending (not dropped).
    summary = asyncio.run(loop.tick())
    assert summary["pending"] == 1 and [r["reason"] for r in summary["results"]] == ["reading_spacing"]
    assert len([r for r in rows if r["reason"] == "reading_spacing"]) == 1
    # The gap passes: the held read goes out.
    loop._clock = lambda: NOW + timedelta(seconds=5401)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent"]
    assert len(delivered) == 2


def test_a_pass_does_not_start_the_wait(monkeypatch):
    loop, _, state, rows, composed, delivered = _setup(monkeypatch, min_spacing_sec=5400.0, text="PASS")
    _two_reads(state)
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["orion_passed", "orion_passed"]
    assert len(composed) == 2 and delivered == []


def test_the_gap_survives_a_restart(monkeypatch):
    # Fresh process, empty memory; the decision log says a reading message went out 10 min ago.
    loop, _, state, _, composed, delivered = _setup(
        monkeypatch, min_spacing_sec=5400.0, last_sent=NOW - timedelta(minutes=10))
    summary = asyncio.run(loop.tick())
    assert [r["reason"] for r in summary["results"]] == ["reading_spacing"]
    assert composed == [] and delivered == []
    assert state["last_sent_asked"] == [NOW - timedelta(seconds=5400)]


def test_zero_spacing_turns_the_wait_off_and_skips_the_query(monkeypatch):
    loop, _, state, _, _, delivered = _setup(monkeypatch, min_spacing_sec=0.0)
    _two_reads(state)
    assert [r["reason"] for r in asyncio.run(loop.tick())["results"]] == ["sent", "sent"]
    assert len(delivered) == 2 and state["last_sent_asked"] == []
