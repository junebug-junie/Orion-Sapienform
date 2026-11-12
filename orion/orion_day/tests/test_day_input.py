"""The honest letter input (living Prime design #3224, constraints 6 and 7): one test per rule."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from orion.orion_day import day_input as di
from orion.orion_day.budget import chronology_citations, extract_refs, material_refs
from orion.orion_day.day_input import build_day_input, latest_self_sense, render_day_input
from orion.orion_day.gather import JUNIPER_CHAT_SOURCES, JUNIPER_CHAT_SQL, TEMPORAL_CORE_EVENTS_SQL, gather_orion_day
from orion.orion_day.temporal_core import render_temporal_core
from orion.orion_day.tests import fixtures as fx
from orion.orion_day.window import orion_day_window
from orion.schemas.orion_day import (
    JuniperChatTurnV1,
    OrionDayLetterSourcesV1,
    OrionDayMaterialV1,
    ReadingItemV1,
    ReverieChainV1,
    ReverieThoughtV1,
    SelfSenseAnswerV1,
)
from orion.schemas.temporal_self import DAY_BLOCK_MAX_CHARS, ExpectationRefV1
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.tests.patch4_fixtures import day_1009

NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
PROMPT = Path(__file__).resolve().parents[2] / "cognition" / "prompts" / "orion_day_note_v1.j2"


def _material(**kw) -> OrionDayMaterialV1:
    return asyncio.run(gather_orion_day(fx.FakeConn(juniper_chat=fx.JUNIPER_CHAT, **kw), fx.LETTER_DATE, now=NOW))


def _bare(**fields) -> OrionDayMaterialV1:
    w = orion_day_window(fx.LETTER_DATE)
    return OrionDayMaterialV1(letter_date=fx.LETTER_DATE, window_start=w.window_start, window_end=w.window_end,
                              gathered_at=NOW, **fields)


# --- Juniper: protected share ------------------------------------------------------------------


def test_juniper_turns_are_gathered_full_and_unsolicited_messages_are_flagged():
    m = _material()
    assert [t.turn_id for t in m.juniper_chat] == ["turn-0001-aaaaaaaa", "turn-0002-bbbbbbbb", "turn-0003-cccccccc"]
    assert m.juniper_chat[0].unsolicited and not m.juniper_chat[0].prompt
    assert m.juniper_chat[1].response == fx.JUNIPER_CHAT[1]["response"].strip()  # never clipped at gather
    assert all(t.at.tzinfo is not None for t in m.juniper_chat)  # naive UTC column -> aware
    assert m.sources["juniper_chat"].status == "ok"


def test_juniper_chat_sql_reads_the_window_in_utc_and_skips_empty_prompts_unless_unsolicited():
    assert "AT TIME ZONE 'UTC'" in JUNIPER_CHAT_SQL and "unsolicited" in JUNIPER_CHAT_SQL
    conn = fx.FakeConn(juniper_chat=fx.JUNIPER_CHAT)
    asyncio.run(gather_orion_day(conn, fx.LETTER_DATE, now=NOW))
    w = orion_day_window(fx.LETTER_DATE)
    assert (JUNIPER_CHAT_SQL, (w.window_start, w.window_end, JUNIPER_CHAT_SOURCES)) in conn.calls
    # Only her rows: Hub chat sources or her browser (NULL source with an origin), never an agent's
    # 'hub' rpc ping or a Collapse Mirror block (live sources, 120 days to 2026-11-12).
    assert "source = ANY($3::text[])" in JUNIPER_CHAT_SQL and "client_meta ? 'origin'" in JUNIPER_CHAT_SQL
    assert "hub" not in JUNIPER_CHAT_SOURCES and "collapse_mirror_reply" not in JUNIPER_CHAT_SOURCES


def test_juniper_turns_survive_any_budget_in_full_and_outreach_is_trimmed():
    m = _material()
    view = render_day_input(m, budget_tokens=500)  # far too small for anything else
    for row in fx.JUNIPER_CHAT[1:]:
        assert row["prompt"] in view.digest_md and row["response"].strip() in view.digest_md
    outreach = fx.JUNIPER_CHAT[0]["response"]
    assert outreach.strip() not in view.digest_md  # longer than OUTREACH_CHARS: trimmed
    assert "unprompted" in view.digest_md and "[juniper_chat:turn-0001-aa]" in view.digest_md


def test_readings_cannot_crowd_juniper_out_and_stay_in_the_tail_cap():
    m = _material()
    many = [ReadingItemV1(seed_id=f"reading:{i:04d}", occurred_at=fx.T0, title=f"News {i}", learned="word " * 2000)
            for i in range(300)]
    view = render_day_input(m.model_copy(update={"readings": many}))
    assert fx.JUNIPER_CHAT[1]["response"].strip() in view.digest_md
    assert view.source_chars["readings"] <= di.TAIL_MAX_CHARS
    assert view.source_chars["juniper"] > 0
    assert "more trimmed items left out to fit" in view.digest_md


def test_a_day_without_juniper_says_so_and_never_invents_a_turn():
    m = _bare(juniper_chat=[JuniperChatTurnV1(turn_id="x1", at=fx.T0, response="I wrote first.", unsolicited=True)])
    view = render_day_input(m)
    assert "Juniper did not write to me this day." in view.digest_md
    assert "Juniper:" not in view.digest_md
    assert _bare(juniper_chat=m.juniper_chat).is_empty()  # outreach alone is not a day with content
    assert not _bare(juniper_chat=[JuniperChatTurnV1(turn_id="x2", at=fx.T0, prompt="hi")]).is_empty()


# --- reveries: across themes, not by salience ----------------------------------------------------


def _reverie_day() -> OrionDayMaterialV1:
    chains, thoughts = [], []
    # Theme "alarm": 60 chains, every thought at salience 0.99. Themes "quiet" and "juniper": low salience.
    for theme, n, sal in (("alarm", 60, 0.99), ("quiet", 6, 0.1), ("juniper", 3, 0.2)):
        for i in range(n):
            cid = f"{theme}-chain-{i:03d}"
            chains.append(ReverieChainV1(chain_id=cid, created_at=fx.T0, theme_key=theme, thought_count=1))
            thoughts.append(ReverieThoughtV1(
                thought_id=f"{theme[:2]}{i:06d}", chain_id=cid, created_at=fx.T0 + timedelta(minutes=10 * i),
                salience=sal, interpretation=f"{theme} thought number {i} about something distinct {i * 7}"))
    return _bare(reverie_chains=chains, reverie_thoughts=thoughts)


def test_reveries_sample_every_theme_even_when_one_theme_is_all_alarms():
    view = render_day_input(_reverie_day())
    for theme in ("alarm", "quiet", "juniper"):
        assert f"[reverie_theme:{theme}]" in view.digest_md
    shown = re.findall(r"\[reverie:(\w{2})", view.digest_md)
    assert shown.count("al") < len(shown)  # the salient theme does not take every slot
    assert shown.count("ju") == 3 and shown.count("qu") == 6  # small themes are shown whole
    assert view.condensed.reverie_themes_included == view.condensed.reverie_themes_total == 3
    assert "salience" not in view.digest_md.split("## Reveries", 1)[1]


def test_reverie_picks_inside_a_theme_spread_over_the_day():
    m = _reverie_day()
    cond = di.OrionDayCondensationV1()
    _, chosen, _ = di.sample_reveries_by_theme(m, 10_000, cond, di.ZoneInfo(m.timezone))
    times = sorted(t.created_at for t in chosen["alarm"])
    all_alarm = sorted(t.created_at for t in m.reverie_thoughts if t.chain_id.startswith("alarm"))
    assert times[0] - all_alarm[0] < all_alarm[-1] - all_alarm[0]  # not just the first hour
    assert times[-1] - times[0] > (all_alarm[-1] - all_alarm[0]) / 2  # spans most of the day


def test_spread_order_visits_every_index_once():
    for n in range(0, 40):
        order = di._spread_order(n)
        assert sorted(order) == list(range(n))
        if n > 2:
            assert order[0] == n // 2


# --- self-sense: once each, latest -------------------------------------------------------------


def test_repeated_self_sense_question_is_shown_once_with_its_latest_answer():
    answers = [SelfSenseAnswerV1(run_id=f"ss{i}", question_key="cannot_do_now", question="What can't you do right now?",
                                 answer_text=f"answer {i}", created_at=fx.T0 + timedelta(hours=i)) for i in range(7)]
    answers.append(SelfSenseAnswerV1(run_id="ssx", question_key="what_are_you", question="What are you?",
                                     answer_text="a mind", created_at=fx.T0))
    picked = latest_self_sense(answers)
    assert [(a.question_key, a.answer_text, n) for a, n in picked] == [
        ("cannot_do_now", "answer 6", 7), ("what_are_you", "a mind", 1)]
    view = render_day_input(_bare(self_sense=answers))
    assert view.digest_md.count("What can't you do right now?") == 1
    assert "answer 6" in view.digest_md and "answer 5" not in view.digest_md
    assert "Asked 7 times this day" in view.digest_md


# --- curiosity: included; clipped only past the cap ------------------------------------------------


def test_curiosity_runs_are_included_in_full_under_the_cap():
    m = _material()
    view = render_day_input(m)
    assert all(f"[curiosity:{r.run_id}]" in view.digest_md for r in m.curiosity_runs)
    assert fx.CURIOSITY_JOURNALS[0]["body"].strip() in view.digest_md
    assert view.condensed.full_text_clip_chars is None


def test_curiosity_bodies_share_one_clip_cap_past_the_full_text_cap():
    m = _material()
    big = [r.model_copy(update={"journal_body": "x" * 40_000}) for r in m.curiosity_runs]
    view = render_day_input(m.model_copy(update={"curiosity_runs": big}))
    assert view.condensed.full_text_clip_chars is not None
    assert view.source_chars["curiosity"] + view.source_chars.get("self_sense", 0) <= di.FULL_TEXT_MAX_CHARS
    assert "clipped here" in view.digest_md


# --- shares --------------------------------------------------------------------------------------


def test_source_shares_cover_the_whole_digest():
    view = render_day_input(_material())
    assert sum(view.source_chars.values()) == len(view.digest_md)
    assert abs(sum(view.source_shares.values()) - 1.0) < 0.01
    assert set(view.source_chars) <= set(di.SOURCES)
    assert {"juniper", "curiosity", "self_sense", "reveries"} <= set(view.source_chars)


def test_letter_sources_carry_input_shares_and_old_rows_still_validate():
    assert OrionDayLetterSourcesV1.model_validate({"by_source": {}}).input_shares == {}
    s = OrionDayLetterSourcesV1(input_chars={"juniper": 10}, input_shares={"juniper": 1.0})
    assert OrionDayLetterSourcesV1.model_validate(s.model_dump(mode="json")) == s


# --- every ref resolves; determinism; entry point ---------------------------------------------------


def test_every_ref_resolves_and_the_view_is_deterministic():
    day, _ = _core_day()
    rows = [{"event_id": "action_outcome:3", "source_kind": "action_outcome", "verdict": "claim_upheld=false",
             "occurred_at": day.closed_at, "payload_json": '{"dispatch_kind": "act:say", "target_id": "hub.outreach"}'}]
    conn = fx.FakeConn(juniper_chat=fx.JUNIPER_CHAT, temporal_day=day.model_dump(mode="json"), temporal_events=rows)
    m = asyncio.run(gather_orion_day(conn, date(2026, 10, 9), now=NOW))  # the closed day's own date
    view = render_day_input(m)
    assert any(r.startswith("arc:") for r in view.included_refs)
    assert "event:action_outcome:3" in view.included_refs
    assert set(extract_refs(view.digest_md)) <= material_refs(m)
    again = render_day_input(OrionDayMaterialV1.model_validate_json(m.model_dump_json()))
    assert again.digest_md == view.digest_md and again.source_shares == view.source_shares


def test_build_day_input_is_the_one_entry_point():
    text, shares, refs, material = asyncio.run(build_day_input(fx.FakeConn(juniper_chat=fx.JUNIPER_CHAT),
                                                               fx.LETTER_DATE, now=NOW))
    assert text == render_day_input(material).digest_md
    assert shares and refs == extract_refs(text)
    assert material.letter_date == fx.LETTER_DATE


def test_note_prompt_carries_the_honesty_line():
    text = PROMPT.read_text()
    assert "only where the record supports it" in text and "Do not manufacture either mood" in text
    assert "BEGIN writing instructions" in text and "END writing instructions" in text


# --- the temporal core (constraint 6) ---------------------------------------------------------------


def _core_day():
    day = day_1009()
    frame = day.frame.model_copy(update={
        "self_change_event_ids": ["action_outcome:1", "action_outcome:2", "action_outcome:3"],
        "expectations_resolved_today": [ExpectationRefV1(
            event_id="action_outcome:3", source_kind="action_outcome",
            committed_at=day.closed_at - timedelta(hours=3), resolved_at=day.closed_at - timedelta(hours=3),
            verdict="claim_upheld=false")],
        "expectations_resolved_total": 1,
    })
    events = {
        "action_outcome:1": {"source_kind": "action_outcome", "verdict": "claim_upheld=true",
                             "payload": {"dispatch_kind": "act:make", "target_id": "actions.journal"}},
        "action_outcome:2": {"source_kind": "action_outcome", "verdict": "claim_upheld=true",
                             "payload": {"dispatch_kind": "act:make", "target_id": "actions.journal"}},
        "action_outcome:3": {"source_kind": "action_outcome", "verdict": "claim_upheld=false",
                             "payload": {"dispatch_kind": "act:say", "target_id": "hub.outreach"}},
    }
    return day.model_copy(update={"frame": frame}), events


def test_temporal_core_names_expectations_acts_threads_and_body_with_citations():
    day, events = _core_day()
    core = render_temporal_core(day, events)
    text = core.text
    assert "Threads with Juniper:" in text and "Open loop open-loop-5cb6cd032923" in text
    assert "act:say hub.outreach" in text and "[event:action_outcome:3]" in text
    assert "act:make actions.journal: 2 (2 held)" in text
    assert "Sleep cycle at 00:33" in text
    refs = re.findall(r"\[((?:arc|event|day):[^\]]+)\]", text)
    assert refs and set(refs) <= set(core.records)
    assert core.records["event:action_outcome:3"]["verdict"] == "claim_upheld=false"


def test_temporal_core_skips_body_arcs_the_chronology_block_already_shows():
    day, events = _core_day()
    block = render_closed_day_block(day)
    shown = frozenset(r[4:] for r in block.records if r.startswith("arc:"))
    core = render_temporal_core(day, events, already_shown=shown)
    assert "Sleep cycle at" not in core.text  # both sleep arcs are in the block


def test_temporal_core_says_when_a_thread_crossed_midnight():
    day = day_1009()
    arcs = [a.model_copy(update={"carried_from_previous_day": True}) if a.kind == "concern" else a for a in day.arcs]
    core = render_temporal_core(day.model_copy(update={"arcs": arcs}), {})
    assert "carried over from the day before" in core.text


def test_temporal_core_is_capped_and_absent_without_a_day():
    assert render_temporal_core(None) is None
    day, events = _core_day()
    core = render_temporal_core(day, events, max_chars=400)
    assert len(core.text) <= 400 and core.lines_dropped > 0 and "left out to fit" in core.text
    assert len(render_temporal_core(day, events).text) <= DAY_BLOCK_MAX_CHARS


def test_gather_builds_the_temporal_core_from_the_closed_day_and_its_event_rows():
    day, _ = _core_day()
    rows = [{"event_id": "action_outcome:3", "source_kind": "action_outcome", "verdict": "claim_upheld=false",
             "occurred_at": day.closed_at, "payload_json": '{"dispatch_kind": "act:say", "target_id": "hub.outreach"}'}]
    conn = fx.FakeConn(temporal_day=day.model_dump(mode="json"), temporal_events=rows)
    m = asyncio.run(gather_orion_day(conn, date(2026, 10, 9), now=NOW))
    assert m.temporal_core is not None and "act:say hub.outreach" in m.temporal_core.text
    assert any(sql == TEMPORAL_CORE_EVENTS_SQL for sql, _ in conn.calls)
    assert m.sources["temporal_core"].status == "ok"


def test_chronology_citations_cover_both_blocks():
    day, events = _core_day()
    m = _bare().model_copy(update={"temporal_day": render_closed_day_block(day),
                                   "temporal_core": render_temporal_core(day, events)})
    shown, labels = chronology_citations(m)
    core_arcs = {r[4:] for r in m.temporal_core.records if r.startswith("arc:")}
    block_arcs = {r[4:] for r in m.temporal_day.records if r.startswith("arc:")}
    assert set(shown) == core_arcs | block_arcs
    assert labels


# --- review follow-ups ---------------------------------------------------------------------------


def test_tail_cuts_and_self_sense_dedupe_are_counted_for_the_email():
    m = _material()
    many = [ReadingItemV1(seed_id=f"reading:{i:04d}", occurred_at=fx.T0, title=f"News {i}", learned="word " * 2000)
            for i in range(300)]
    view = render_day_input(m.model_copy(update={"readings": many}))
    c = view.condensed
    assert c.tail_items_total >= 300 and 0 < c.tail_items_left_out < c.tail_items_total
    assert f"({c.tail_items_left_out} more trimmed items left out to fit" in view.digest_md
    assert c.self_sense_answers_total == len(m.self_sense) and c.self_sense_answers_shown == 1
    assert view.source_chars["readings"] <= di.TAIL_MAX_CHARS


def test_no_reverie_heading_when_nothing_is_showable():
    hollow = [ReverieThoughtV1(thought_id="h0000001", created_at=fx.T0, interpretation="x", hollow=True)]
    view = render_day_input(_bare(reverie_thoughts=hollow, juniper_chat=[
        JuniperChatTurnV1(turn_id="t1", at=fx.T0, prompt="hi", response="hello")]))
    assert "## Reveries" not in view.digest_md
    assert view.condensed.reverie_thoughts_hollow_skipped == 1


def test_a_ref_pasted_into_chat_is_not_a_citation():
    turn = JuniperChatTurnV1(turn_id="t1", at=fx.T0, prompt="what is [arc:deadbeefdeadbeef]?", response="ok")
    view = render_day_input(_bare(juniper_chat=[turn]))
    assert "arc:deadbeefdeadbeef" not in view.included_refs
    assert "(arc:deadbeefdeadbeef)" in view.digest_md


def test_temporal_core_never_leaves_a_heading_over_nothing():
    day, events = _core_day()
    for cap in range(120, 900, 20):
        core = render_temporal_core(day, events, max_chars=cap)
        if core is None:
            continue
        lines = core.text.splitlines()
        for i, line in enumerate(lines):
            if line.endswith(":") and not line.startswith("- "):
                assert i + 1 < len(lines) and lines[i + 1].startswith("- "), (cap, core.text)
