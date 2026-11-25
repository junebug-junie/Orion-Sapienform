"""Time-aware lived recall (issue #3674): the date grammar, the rewrite's WHEN/PAST lines, window
filtering before the pool cut, and the "true then, not now" label for facts an update retired.

Made-up records only; no DB, bus or model (fakes only).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import app.lived_memory as lm
from app.lived_memory import (AS_OF_LABEL, OUT_OF_WINDOW_LINES, OUT_OF_WINDOW_MAX, LivedMemoryConfig,
                              LivedMemoryRecall, rank, render_line, tokenize, trace_of, window_where)
from app.lived_rewrite import RewriteResult, build_prompt, parse_rewrite
from app.lived_time import parse_time_window, window_from_dates
from orion.introspect.semantic_index import SearchConfig
from orion.memory.lived_memory import LivedRecord

# Wednesday 2026-11-25, 12:00 in Denver (MST, UTC-7).
NOW = datetime(2026, 11, 25, 19, 0, tzinfo=timezone.utc)


def _dates(text: str, now: datetime = NOW):
    w = parse_time_window(text, now)
    if w is None:
        return None
    t = w.trace()
    return t["start"], t["end"], t["kind"]


# --- the date grammar ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("What has happened with Amanda's health over the last month or so?", ("2026-10-10", "2026-11-25", "rolling")),
    ("the past few days", ("2026-11-22", "2026-11-25", "rolling")),
    ("in the last 2 weeks", ("2026-11-11", "2026-11-25", "rolling")),
    ("Last week, which game kept crashing?", ("2026-11-16", "2026-11-25", "calendar")),
    ("anything this month?", ("2026-11-01", "2026-11-25", "calendar")),
    ("About a month or so ago, why was the office hot?", ("2026-10-10", "2026-11-10", "ago")),
    ("what about two weeks ago", ("2026-11-04", "2026-11-18", "ago")),
    ("Yesterday I mentioned a mod", ("2026-11-24", "2026-11-24", "relative_day")),
    ("the day before yesterday", ("2026-11-23", "2026-11-23", "relative_day")),
    ("last weekend", ("2026-11-21", "2026-11-22", "weekend")),
    ("last tuesday", ("2026-11-24", "2026-11-24", "weekday")),
    ("What did I tell you about my job on November 20th?", ("2026-11-20", "2026-11-20", "date")),
    ("Where did I spend the day on Saturday, October 18th?", ("2026-10-18", "2026-10-18", "date")),
    ("the 3rd of October", ("2026-10-03", "2026-10-03", "date")),
    ("10/30", ("2026-10-30", "2026-10-30", "date")),
    ("2026-10-30", ("2026-10-30", "2026-10-30", "date")),
    ("What happened in the week of November 2nd to 8th?", ("2026-11-02", "2026-11-08", "range")),
    ("between Oct 3 and Oct 10", ("2026-10-03", "2026-10-10", "range")),
    ("the week of Nov 2", ("2026-11-02", "2026-11-08", "week_of")),
    ("Which days was I in Chicago in October?", ("2026-10-01", "2026-10-31", "month")),
    ("Where was I in early October?", ("2026-10-01", "2026-10-10", "month")),
    ("late September", ("2026-09-21", "2026-09-30", "month")),
    ("since August", ("2026-08-01", "2026-11-25", "since_month")),
    ("in 2025", ("2025-01-01", "2025-12-31", "year")),
])
def test_parser_reads_the_common_time_phrases(text, expected):
    assert _dates(text) == expected


@pytest.mark.parametrize("text", [
    "May I ask you something?",
    "I march on regardless",
    "the last time we talked about it",
    "I have 3 days left on the project",
    "Did I add the second V100 before or after I beat NES Zelda I?",
    "When did I get promoted to Team Lead?",
    "hi Orion, how are you?",
])
def test_parser_finds_no_window_where_no_time_is_named(text):
    assert parse_time_window(text, NOW) is None


def test_a_month_without_a_year_is_the_most_recent_one():
    assert _dates("in December") == ("2025-12-01", "2025-12-31", "month")   # asked in November
    jan = datetime(2027, 1, 10, 19, 0, tzinfo=timezone.utc)
    assert _dates("December 30th", jan)[:2] == ("2026-12-30", "2026-12-30")
    assert _dates("January 3rd", jan)[:2] == ("2027-01-03", "2027-01-03")


def test_relative_days_resolve_in_denver_not_utc():
    # 03:00Z on the 25th is 20:00 on the 24th in Denver: "yesterday" is the 23rd.
    late = datetime(2026, 11, 25, 3, 0, tzinfo=timezone.utc)
    assert _dates("yesterday", late)[:2] == ("2026-11-23", "2026-11-23")
    w = parse_time_window("yesterday", late)
    assert w.start == datetime(2026, 11, 23, 7, 0, tzinfo=timezone.utc)   # Denver midnight, MST


def test_past_state_needs_a_window_wholly_past_and_a_past_tense_state_verb():
    assert parse_time_window("Where was I in early October?", NOW).past_state
    assert not parse_time_window("What did I tell you in September?", NOW).past_state
    assert not parse_time_window("where was I this week?", NOW).past_state   # not over yet


def test_the_window_pads_a_day_each_side():
    w = parse_time_window("on November 20th", NOW)
    assert w.contains(datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc))
    assert not w.contains(datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc))
    assert not w.contains(None)


# --- the rewrite's WHEN / PAST lines --------------------------------------------------------

def test_prompt_says_what_day_it_is_in_denver():
    p = build_prompt("what happened last week?", [], now=NOW)
    assert "Today is Wednesday, 2026-11-25" in p and "WHEN:" in p and "PAST: yes" in p
    assert "Today is" not in build_prompt("hm?", [], short=True, now=NOW)


def test_parse_rewrite_reads_when_and_past_and_keeps_them_out_of_the_queries():
    res = parse_rewrite("Chicago work trip hotel\nJuniper travel October\nWHEN: 2026-10-01 to 2026-10-10\nPAST: yes",
                        now=NOW)
    assert res.queries == ["Chicago work trip hotel", "Juniper travel October"]
    assert res.window is not None and res.window.trace()["start"] == "2026-10-01"
    assert res.window.trace()["end"] == "2026-10-10" and res.window.source == "model"
    assert res.past_state and res.window.past_state


def test_parse_rewrite_drops_an_implausible_or_malformed_window():
    assert parse_rewrite("game crash fix\nWHEN: 2027-03-01 to 2027-03-02", now=NOW).window is None   # future
    assert parse_rewrite("game crash fix\nWHEN: 2026-10-10 to 2026-10-01", now=NOW).window is None   # reversed
    assert parse_rewrite("game crash fix\nWHEN: last week", now=NOW).window is None                 # not dates
    assert parse_rewrite("game crash fix\nWHEN: last week", now=NOW).queries == ["game crash fix"]
    none = parse_rewrite("NONE\nWHEN: 2026-10-01 to 2026-10-10", now=NOW)
    assert none.queries == [] and none.window is None   # small talk carries no window


# --- window filtering in recall -------------------------------------------------------------

def _rec(rid: str, text: str, at: datetime, **kw: Any) -> Dict[str, Any]:
    return {"record_id": rid, "purpose": "happened", "text": text, "occurred_at": at, "importance": 0.8,
            "is_correction": False, "confirmed": False, "label": None, "created_at": at, **kw}


class Pool:
    """asyncpg-shaped fake: the live serving read, the as-of window read, and nothing else."""

    def __init__(self, live: List[Dict[str, Any]], retired: List[Dict[str, Any]] = ()) -> None:
        self.live = live
        self.retired = list(retired)
        self.during_calls: List[tuple] = []

    async def fetch(self, sql: str, *args: Any) -> List[Dict[str, Any]]:
        if "invalid_at AS invalid_at" in sql:  # EPISODE_VALID_DURING_SQL
            self.during_calls.append(args)
            start, end = args[0], args[1]
            rows = [r for r in self.retired if r["occurred_at"] < end and r["invalid_at"] > start]
            if len(args) > 2:
                rows = [r for r in rows if r["record_id"] in set(args[2])]
            return rows
        if "FROM episode_memory_referent" in sql or "FROM referent_alias" in sql or "FROM chat_history_log" in sql:
            return []
        if "FROM episode_memory" not in sql:
            return []
        rows = list(self.live)
        if args:
            rows = [r for r in rows if r["record_id"] in set(args[0])]
        return rows


def _cfg(**kw: Any) -> LivedMemoryConfig:
    base = dict(search=SearchConfig("http://chroma", "http://embed", "orion_lived_memory", 0.0),
                rerank_url="http://rerank", min_fit=0.5, max_lines=4, timeout_sec=2.0, min_query_tokens=3,
                rewrite_timeout_sec=0.5, link_top_k=0)
    base.update(kw)
    return LivedMemoryConfig(**base)


def overlap_reranker(query: str, docs: List[str]):
    q = set(tokenize(query))

    async def go():
        return [(-4.0 + 3.0 * len(q & set(tokenize(d)))) for d in docs]

    return go()


@pytest.fixture
def no_dense(monkeypatch):
    seen: List[Any] = []

    async def embed(*_a, **_k):
        return [0.0], "m"

    async def nearest(*_a, where=None, **_k):
        seen.append(where)
        return []

    monkeypatch.setattr(lm, "embed", embed)
    monkeypatch.setattr(lm, "nearest", nearest)
    return seen


AUG = datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc)
OCT = datetime(2026, 10, 28, 18, 0, tzinfo=timezone.utc)
# Fifteen August records fit the words better than the October one: without the window they
# fill the exact-word top 10 and the October record never reaches the rerank (issue #3881).
CROWD = [_rec(f"aug{i}", f"Amanda health scare hospital visit number {i}", AUG) for i in range(15)]
TARGET = _rec("oct1", "Amanda health news: surgery on her leg veins", OCT)
QUESTION = "What has happened with Amanda's health over the last month or so?"


def _reader(pool, **kw) -> LivedMemoryRecall:
    return LivedMemoryRecall(pool, _cfg(**kw), reranker=overlap_reranker, clock=lambda: NOW)


def test_without_the_window_the_old_crowd_fills_the_pool(no_dense):
    out = asyncio.run(_reader(Pool(CROWD + [TARGET]), time_window=False).recall(QUESTION))
    assert "episode_memory:oct1" not in out.debug["candidate_ids"]
    assert "window" not in out.debug


def test_the_window_brings_in_window_records_into_the_pool_before_the_cut(no_dense):
    out = asyncio.run(_reader(Pool(CROWD + [TARGET])).recall(QUESTION))
    cands = out.debug["candidate_ids"]
    assert cands[0] == "episode_memory:oct1"                      # in-window first, best-retrieved
    assert sum(1 for c in cands if c.startswith("episode_memory:aug")) <= OUT_OF_WINDOW_MAX
    assert "oct1" in [s.rec.record_id for s in out.chosen]        # served, though older rows fit better
    outside = [s for s in out.chosen if s.rec.record_id.startswith("aug")]
    assert len(outside) <= OUT_OF_WINDOW_LINES                    # the cap on lines outside the window
    w = out.debug["window"]
    assert (w["kind"], w["source"], w["in_window"]) == ("rolling", "parser", 1)
    # The meaning search is asked once more, filtered to the padded window in Chroma.
    assert any(isinstance(x, dict) and "$and" in x for x in no_dense)


def test_a_window_with_nothing_in_it_changes_nothing(no_dense):
    q = "What happened with Amanda's health in March?"
    plain = asyncio.run(_reader(Pool(CROWD), time_window=False).recall(q))
    windowed = asyncio.run(_reader(Pool(CROWD)).recall(q))
    assert [s.rec.doc_id for s in windowed.chosen] == [s.rec.doc_id for s in plain.chosen]
    assert windowed.chosen                                         # never emptied by a time word
    assert windowed.debug["window"]["empty"] is True


def test_the_model_window_is_used_only_when_the_parser_found_none(no_dense):
    async def rw(message, context, short=False):
        return RewriteResult(queries=["Amanda leg surgery"],
                             window=window_from_dates(date(2026, 10, 20), date(2026, 11, 5), NOW))

    q = "Tell me again how Amanda's health has been going lately with everything"
    out = asyncio.run(LivedMemoryRecall(Pool(CROWD + [TARGET]), _cfg(), reranker=overlap_reranker, rewriter=rw,
                                        clock=lambda: NOW).recall(q))
    assert out.debug["window"]["source"] == "model"
    assert out.debug["candidate_ids"][0] == "episode_memory:oct1"
    assert trace_of(out.debug)["rewrite_window"]["start"] == "2026-10-20"


def test_now_moves_the_window_for_a_question_written_on_another_day(no_dense):
    out = asyncio.run(_reader(Pool(CROWD + [TARGET])).recall(QUESTION, now=datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)))
    assert out.debug["window"]["start"] == "2026-07-17"   # 1.5 months before 2026-09-01 (Denver)
    assert out.debug["candidate_ids"][0].startswith("episode_memory:aug")


def test_rank_caps_lines_outside_the_window_only_when_something_inside_fits():
    def rec(rid, at):
        return LivedRecord(kind="episode_memory", record_id=rid, purpose="happened", text=rid, occurred_at=at,
                           importance=0.5)

    inside, out1, out2 = rec("in", OCT), rec("o1", AUG), rec("o2", AUG)
    w = parse_time_window("over the last month", NOW)

    def in_w(r):
        return w.contains(r.occurred_at)

    got = rank([(out1, 3.0), (out2, 2.5), (inside, 0.5)], min_fit=0.5, max_lines=4, now=NOW, in_window=in_w)
    assert [s.rec.record_id for s in got] == ["o1", "in"]            # one outside line, the best one
    got = rank([(out1, 3.0), (out2, 2.5), (inside, -9.0)], min_fit=0.5, max_lines=4, now=NOW, in_window=in_w)
    assert [s.rec.record_id for s in got] == ["o1", "o2"]            # nothing inside fits: no cap


def test_window_where_is_a_chroma_range_filter_on_occurred_ts():
    w = parse_time_window("on November 20th", NOW)
    f = window_where(w)
    lo, hi = f["$and"][0]["occurred_ts"]["$gte"], f["$and"][1]["occurred_ts"]["$lt"]
    assert hi - lo == pytest.approx(3 * 86400)                        # the day, padded a day each side


# --- past state: the as-of reader and the label ---------------------------------------------

EARLY_OCT = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
HOME_NOW = _rec("home-now", "Juniper lives in Ogden and works from her office", datetime(2026, 11, 10, tzinfo=timezone.utc))
TRIP = _rec("trip", "Juniper was staying at a hotel on the lake in Chicago for work", EARLY_OCT,
            invalid_at=datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc))


def test_a_past_state_question_reads_retired_facts_and_labels_them_true_then(no_dense):
    pool = Pool([HOME_NOW] + CROWD, retired=[TRIP])
    out = asyncio.run(_reader(pool).recall("Where was Juniper staying for work in early October?"))
    assert out.debug["window"]["past_state"] is True and out.debug["window"]["as_of_rows"] == 1
    assert "episode_memory:trip" in out.debug["candidate_ids"]
    line = next(x for x in out.lines if "episode_memory:trip" in x)
    assert f"{AS_OF_LABEL}: changed 2026-10-08" in line
    # The as-of row was re-checked through the same window reader, by id, before render.
    assert any(len(a) == 3 and a[2] == ["trip"] for a in pool.during_calls)


def test_a_present_tense_question_never_reads_retired_facts(no_dense):
    pool = Pool([HOME_NOW] + CROWD, retired=[TRIP])
    out = asyncio.run(_reader(pool).recall("What did Juniper tell you about work in early October?"))
    assert out.debug["window"]["past_state"] is False
    assert pool.during_calls == []
    assert all("trip" not in c for c in out.debug["candidate_ids"])


def test_a_retired_fact_that_is_gone_at_recheck_is_not_served(no_dense):
    pool = Pool([HOME_NOW], retired=[TRIP])
    reader = _reader(pool)

    orig = pool.fetch

    async def fetch(sql, *args):
        if "invalid_at AS invalid_at" in sql and len(args) > 2:
            return []   # rejected or expired since the candidate read
        return await orig(sql, *args)

    pool.fetch = fetch
    out = asyncio.run(reader.recall("Where was Juniper staying for work in early October?"))
    assert all("trip" not in x for x in out.lines)
    assert out.debug.get("recheck_dropped") == ["episode_memory:trip"]


def test_a_failed_as_of_read_adds_nothing_and_breaks_nothing(no_dense):
    class Broken(Pool):
        async def fetch(self, sql, *args):
            if "invalid_at AS invalid_at" in sql:
                raise RuntimeError("db down")
            return await super().fetch(sql, *args)

    out = asyncio.run(_reader(Broken([HOME_NOW] + CROWD)).recall("Where was Juniper staying for work in early October?"))
    assert out.debug["window"]["as_of_error"] == "RuntimeError"
    assert out.backend == "hybrid"


def test_render_line_marks_only_retired_rows():
    rec = LivedRecord(kind="episode_memory", record_id="r", purpose="happened", text="x", occurred_at=EARLY_OCT,
                      importance=0.5)
    assert AS_OF_LABEL not in render_line(rec)
    old = LivedRecord(kind="episode_memory", record_id="r", purpose="happened", text="x", occurred_at=EARLY_OCT,
                      importance=0.5, invalid_at=datetime(2026, 10, 8, tzinfo=timezone.utc))
    assert render_line(old) == f"- (episode_memory/happened · episode_memory:r · 2026-10-05 · {AS_OF_LABEL}: changed 2026-10-08) x"


def test_the_trace_keeps_dates_and_shape_never_the_message_words(no_dense):
    out = asyncio.run(_reader(Pool(CROWD + [TARGET])).recall(QUESTION))
    w = trace_of(out.debug)["window"]
    assert set(w) <= {"start", "end", "kind", "source", "past_state", "in_window", "as_of_rows",
                      "out_of_window_kept", "empty", "dense_error", "as_of_error"}
    assert "month" not in " ".join(str(v) for v in w.values())
