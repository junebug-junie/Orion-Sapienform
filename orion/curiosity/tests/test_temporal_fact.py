"""`curiosity_today_fact`: today's curiosity time as one cited fact line, never raising."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.temporal_fact import (
    CITATION_CAP,
    curiosity_today_fact,
    thread_run_ids_from_view,
)
from orion.curiosity.worldview import RecentRun, WorldviewSnapshot

# 2026-10-13 21:00 MDT = 2026-10-14 03:00 UTC; local day is 2026-10-13.
NOW = datetime(2026, 10, 14, 3, 0, tzinfo=timezone.utc)
DAY = "2026-10-13"


def _arc(run_id: str, *, began_h: int, minutes: float, kind: str = "curiosity", arc_id: str | None = None) -> dict:
    began = datetime(2026, 10, 13, began_h, 0, tzinfo=timezone.utc)
    return {
        "arc_id": arc_id or f"arc{run_id}", "kind": kind, "day_id": DAY, "subject_ref": run_id,
        "subject_label": "", "began_at": began.isoformat(),
        "ended_at": (began + timedelta(minutes=minutes)).isoformat(), "status": "closed",
        "cumulative_dwell_sec": minutes * 60.0,
    }


def _frame(arcs=(), *, as_of=NOW - timedelta(seconds=60), day=DAY, total=None) -> dict:
    arcs = list(arcs)
    return {"day_id": day, "as_of": as_of.isoformat(), "arcs_today": arcs,
            "arcs_today_total": len(arcs) if total is None else total}


def _arcs_payload(arcs, day=DAY) -> dict:
    return {"day_id": day, "count": len(arcs), "arcs": list(arcs)}


def test_thread_match_counts_only_listed_runs_and_cites_them_first():
    arcs = [_arc("r1", began_h=8, minutes=12), _arc("r2", began_h=10, minutes=20), _arc("r3", began_h=14, minutes=5)]
    line = curiosity_today_fact(_frame(), now=NOW, thread_run_ids=["r1", "r3", "old"],
                                curiosity_arcs=_arcs_payload(arcs))
    assert line == ("TODAY SO FAR: 3 curiosity runs finished today, 37 min in total. "
                    "2 of the earlier runs listed above were today, 17 min of that. "
                    "[arc:arcr3] [arc:arcr1] [arc:arcr2]")


def test_no_thread_match_is_stated_not_hidden():
    line = curiosity_today_fact(_frame(), now=NOW, thread_run_ids=["x", "y"],
                                curiosity_arcs=_arcs_payload([_arc("r1", began_h=8, minutes=0.5)]))
    assert line == ("TODAY SO FAR: 1 curiosity run finished today, under a minute in total. "
                    "None of the earlier runs listed above was today. [arc:arcr1]")


def test_no_thread_ids_gives_day_total_only():
    line = curiosity_today_fact(_frame(), now=NOW, curiosity_arcs=_arcs_payload([_arc("r1", began_h=8, minutes=9)]))
    assert line == "TODAY SO FAR: 1 curiosity run finished today, 9 min in total. [arc:arcr1]"


def test_zero_runs_with_a_complete_list():
    line = curiosity_today_fact(_frame(), now=NOW, curiosity_arcs=_arcs_payload([]))
    assert line == "TODAY SO FAR: no curiosity run has finished yet today."


def test_stale_frame_renders_nothing():
    arcs = _arcs_payload([_arc("r1", began_h=8, minutes=9)])
    assert curiosity_today_fact(_frame(as_of=NOW - timedelta(seconds=901)), now=NOW, curiosity_arcs=arcs) is None
    assert curiosity_today_fact(_frame(as_of=NOW - timedelta(seconds=899)), now=NOW, curiosity_arcs=arcs) is not None


def test_wrong_day_renders_nothing():
    arcs = _arcs_payload([_arc("r1", began_h=8, minutes=9)])
    assert curiosity_today_fact(_frame(day="2026-10-12"), now=NOW, curiosity_arcs=arcs) is None
    # The arc list for another day than the frame is not used; the frame fallback is.
    line = curiosity_today_fact(_frame(), now=NOW, curiosity_arcs=_arcs_payload([_arc("r1", began_h=8, minutes=9)], day="2026-10-12"))
    assert line == "TODAY SO FAR: no curiosity run has finished yet today."


def test_garbage_never_raises():
    for frame in (None, "x", 3, [], {}, {"day_id": DAY}, {"day_id": DAY, "as_of": "not a date"},
                  {"day_id": 5, "as_of": NOW.isoformat()}):
        assert curiosity_today_fact(frame, now=NOW) is None
    bad_rows = [None, "x", {"kind": "curiosity"}, {"kind": "curiosity", "arc_id": "a", "subject_ref": "r",
                                                     "cumulative_dwell_sec": "nan?"}]
    line = curiosity_today_fact(_frame(), now=NOW, thread_run_ids=[None, ""],  # type: ignore[list-item]
                                curiosity_arcs={"day_id": DAY, "arcs": bad_rows})
    assert line == "TODAY SO FAR: 1 curiosity run finished today, under a minute in total. [arc:a]"
    assert curiosity_today_fact(_frame(), now="now", curiosity_arcs=_arcs_payload([])) is None  # type: ignore[arg-type]


def test_citations_capped_and_duplicates_not_double_counted():
    arcs = [_arc(f"r{i}", began_h=7 + i, minutes=10) for i in range(8)]
    line = curiosity_today_fact(_frame(), now=NOW, curiosity_arcs=_arcs_payload(arcs + arcs[:2]))
    assert line is not None and line.count("[arc:") == CITATION_CAP
    assert "8 curiosity runs finished today, 80 min in total" in line
    assert "[arc:arcr7]" in line and "[arc:arcr0]" not in line  # most recent first


def test_capped_frame_without_arc_list_says_nothing():
    listed = [_arc("r1", began_h=8, minutes=9)]
    assert curiosity_today_fact(_frame(listed, total=200), now=NOW) is None
    # Complete when a reverie arc is listed (reverie only fills after every other kind).
    listed_rev = listed + [_arc("c1", began_h=9, minutes=1, kind="reverie")]
    assert curiosity_today_fact(_frame(listed_rev, total=200), now=NOW) == (
        "TODAY SO FAR: 1 curiosity run finished today, 9 min in total. [arc:arcr1]")
    assert curiosity_today_fact(_frame(listed), now=NOW) is not None


def test_no_steering_words():
    arcs = [_arc(f"r{i}", began_h=7 + i, minutes=30) for i in range(6)]
    line = curiosity_today_fact(_frame(), now=NOW, thread_run_ids=["r1", "r2"], curiosity_arcs=_arcs_payload(arcs))
    assert line is not None
    for word in ("should", "consider", "too much", "maybe", "instead", "try"):
        assert word not in line.lower()


def test_thread_ids_only_when_the_thread_section_renders():
    assert thread_run_ids_from_view(WorldviewSnapshot(recent_runs=[RecentRun(run_id="a")])) == []
    view = WorldviewSnapshot(recent_runs=[RecentRun(run_id="a"), RecentRun(run_id="b")])
    assert thread_run_ids_from_view(view) == ["a", "b"]
    assert thread_run_ids_from_view(None) == []


def test_kickoff_includes_line_after_thread_and_omits_when_none():
    view = WorldviewSnapshot(recent_runs=[RecentRun(run_id="a", claims=["c1"]), RecentRun(run_id="b", claims=["c2"])])
    fact = "TODAY SO FAR: 2 curiosity runs finished today, 30 min in total. [arc:x]"
    with_line = build_kickoff_prompt(StudyMaterial(generated_at=NOW), view=view, today_fact=fact)
    assert fact in with_line
    assert with_line.index("THE LAST 2 RUNS") < with_line.index(fact)
    without = build_kickoff_prompt(StudyMaterial(generated_at=NOW), view=view, today_fact=None)
    assert "TODAY SO FAR" not in without
    assert build_kickoff_prompt(StudyMaterial(generated_at=NOW), view=view) == without
    # No graph configured: the day line still shows (it asks for no graph).
    no_graph = build_kickoff_prompt(StudyMaterial(generated_at=NOW), view=view, graph_enabled=False, today_fact=fact)
    assert fact in no_graph
