"""Walkway replay floor (2026-11-03).

Live 10-29..11-03: 26 dream cycles had >= 1 graded walkway candidate, yet 0
`dream_replay_item` rows ever had source_kind='walkway' -- every replay filled
all 12 slots with heavier sources, and walkway_replay_slots was only a ceiling.
Pinned: when walkway candidates exist, the best one (a miss over a met) takes
one slot; no walkway candidates -> the replay is unchanged; the ceiling holds.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone


def _item(ref, src, w, tags=None):
    from app.replay import ReplayItemV1

    return ReplayItemV1(ref_id=ref, source_kind=src, text=f"text {ref}", weight=w,
                        reason="r", tags=tags or [src, ref])


def _strong():
    """12 non-walkway items at >= 0.5, the live shape since 10-29."""
    return (
        [_item(f"metacog:m{i}", "metacog", 1.0 - 0.05 * i) for i in range(6)]
        + [_item(f"arc:a{i}", "arc", 0.71 - 0.05 * i) for i in range(4)]
        + [_item("resonance:r0", "resonance", 1.0), _item("compaction_request:c0", "compaction_request", 0.5)]
    )


def _select(items, k=12, floors=True):
    from app.replay import arc_replay_slots, select_replay, walkway_replay_floor, walkway_replay_slots

    return select_replay(
        items, k,
        source_caps={"arc": arc_replay_slots(k), "walkway": walkway_replay_slots(k)},
        source_floors={"walkway": walkway_replay_floor(k)} if floors else None,
    )


def test_one_missed_walkway_item_gets_a_slot_among_twelve_strong_items():
    from app.replay import WALKWAY_WEIGHT_MISSED

    walk = _item("walkway:e1", "walkway", WALKWAY_WEIGHT_MISSED)
    picked = _select(_strong() + [walk])
    assert len(picked) == 12
    assert [p.ref_id for p in picked if p.source_kind == "walkway"] == ["walkway:e1"]
    # The lightest non-walkway item is the one that gave way.
    assert "compaction_request:c0" not in {p.ref_id for p in picked}
    assert [p.weight for p in picked] == sorted((p.weight for p in picked), reverse=True)


def test_floor_prefers_a_miss_over_a_met():
    from app.replay import WALKWAY_WEIGHT_MET, WALKWAY_WEIGHT_MISSED

    met = _item("walkway:a_met", "walkway", WALKWAY_WEIGHT_MET)
    miss = _item("walkway:z_miss", "walkway", WALKWAY_WEIGHT_MISSED)
    picked = _select(_strong() + [met, miss])
    assert [p.ref_id for p in picked if p.source_kind == "walkway"] == ["walkway:z_miss"]


def test_floor_ties_keep_the_producer_order():
    """walkway_candidates already orders ties newest first; the floor keeps that."""
    from app.replay import WALKWAY_WEIGHT_MISSED

    newest = _item("walkway:zz_newest", "walkway", WALKWAY_WEIGHT_MISSED)
    older = _item("walkway:aa_older", "walkway", WALKWAY_WEIGHT_MISSED)
    picked = _select(_strong() + [newest, older])
    assert [p.ref_id for p in picked if p.source_kind == "walkway"] == ["walkway:zz_newest"]


def test_no_walkway_candidates_leaves_the_replay_unchanged():
    items = _strong()
    assert _select(items) == _select(items, floors=False)


def test_ceiling_still_holds_and_floor_never_exceeds_it():
    from app.replay import select_replay, walkway_replay_slots

    walks = [_item(f"walkway:e{i}", "walkway", 0.95) for i in range(8)]
    picked = _select(_strong() + walks)
    assert sum(p.source_kind == "walkway" for p in picked) == walkway_replay_slots(12) == 2
    # A floor larger than the cap is clipped to the cap; a floor never exceeds k.
    many = select_replay(walks, 12, source_caps={"walkway": 2}, source_floors={"walkway": 9})
    assert len(many) == 2
    assert len(select_replay(walks, 1, source_floors={"walkway": 5})) == 1
    assert select_replay(walks, 0, source_floors={"walkway": 1}) == []


def test_floor_result_independent_of_non_walkway_order():
    from app.replay import WALKWAY_WEIGHT_MISSED

    items = _strong() + [_item("walkway:e1", "walkway", WALKWAY_WEIGHT_MISSED)]
    assert _select(items) == _select(list(reversed(items)))


# --- cycle: a full replay still carries one walkway item ------------------------


NOW = datetime(2026, 11, 3, 13, 0, tzinfo=timezone.utc)


def _full_rows():
    return {
        "metacog": [{"id": f"m{i}", "summary": f"degraded thing {i}", "severity": "critical",
                     "trigger_kind": f"kind_{i}", "trigger_reason": f"reason {i}", "tags": [f"t{i}"]}
                    for i in range(8)],
        "compaction_request": [{"request_id": f"r{i}", "theme": f"theme {i}", "reason": "recurs"}
                               for i in range(8)],
        "resonance": [{"alert_id": f"a{i}", "theme_key": f"res {i}", "violation_count": 9} for i in range(8)],
    }


def test_cycle_replays_one_walkway_item_when_the_replay_is_full():
    from app.cycle import CycleDeps, run_cycle_once

    def walkway(_since, until):
        scored = until - timedelta(minutes=30)
        return {"graded": [{"expectation_id": "e1", "stream_id": "walkway",
                            "subject_key": "individual:dog", "subject_label": "the black dog",
                            "status": "missed", "peak_minute": 460,
                            "window_start": scored - timedelta(minutes=30), "scored_at": scored}],
                "calibration": []}

    async def complete(_prompt):
        return json.dumps({"link": True, "claim": "These recur together", "why": "w"})

    def deps(with_walkway):
        return CycleDeps(
            load_source_rows=lambda since, limit, until=None: {} if until is not None else _full_rows(),
            load_idle_minutes=lambda: 120.0, load_last_window_start=lambda: None,
            load_last_attempt_end=lambda: None, persist_cycle=lambda c: True, complete=complete,
            load_walkway_rows=walkway if with_walkway else None,
        )

    off = asyncio.run(run_cycle_once(deps(False)))
    on = asyncio.run(run_cycle_once(deps(True)))
    assert off.status == on.status == "completed"
    assert len(off.replay) == len(on.replay) == 12
    assert all(r.weight >= 0.5 for r in off.replay)
    assert [r.source_kind for r in on.replay].count("walkway") == 1


def test_small_replay_reserves_no_walkway_floor():
    from app.replay import WALKWAY_WEIGHT_MISSED, walkway_replay_floor

    assert [walkway_replay_floor(k) for k in (0, 1, 5, 6, 12, 24)] == [0, 0, 0, 1, 1, 1]
    picked = _select(_strong() + [_item("walkway:e1", "walkway", WALKWAY_WEIGHT_MISSED)], k=1)
    assert [p.ref_id for p in picked] == ["metacog:m0"]
