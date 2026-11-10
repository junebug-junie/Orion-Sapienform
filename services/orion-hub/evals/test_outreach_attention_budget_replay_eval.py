"""Offline eval of the one-attention-budget replay: a planted day shaped like a live one."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 2026-11-09 08:00 MST = 15:00 UTC.
MORNING = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)


def _load():
    path = Path(__file__).with_name("run_outreach_attention_budget_replay.py")
    spec = importlib.util.spec_from_file_location("run_outreach_attention_budget_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_morning_pile_is_spread_and_the_day_stays_under_the_cap():
    mod = _load()
    # Eight reading findings all wanting out at 08:00 (the post-quiet-hours pile, live 10-28).
    offers = [mod.Offer(at=MORNING + timedelta(minutes=i), source="reading_finding_outreach") for i in range(8)]
    res = mod.simulate(offers)
    sent = sorted(r["at"] for r in res if r["outcome"] == "sent")
    assert 1 <= len(sent) <= 6
    assert sent[0] == MORNING                                       # the first message of the day is cheap
    assert all(b - a >= timedelta(minutes=30) for a, b in zip(sent, sent[1:]))   # spread, not a burst
    assert {r["outcome"] for r in res} <= {"sent", "expired", "cap"}


def test_office_visit_is_never_held_past_its_window():
    mod = _load()
    offers = [mod.Offer(at=MORNING + timedelta(minutes=i), source="reading_finding_outreach") for i in range(2)]
    offers.append(mod.Offer(at=MORNING + timedelta(minutes=5), source="office_reach_out"))
    res = {r["source"]: r for r in mod.simulate(offers) if r["outcome"] == "sent"}
    office = res["office_reach_out"]
    assert office["held_min"] <= 5.0


def test_a_strong_want_goes_late_and_a_weak_one_waits():
    mod = _load()
    # Four messages through the day; then at 21:00 one after a long silence (strong drive) and one
    # right behind it (weak drive: the family just spoke).
    times = [MORNING, MORNING + timedelta(hours=3), MORNING + timedelta(hours=6), MORNING + timedelta(hours=9)]
    offers = [mod.Offer(at=t, source="tick") for t in times]
    late = MORNING + timedelta(hours=13)                            # 21:00 MST
    offers += [mod.Offer(at=late, source="tick"), mod.Offer(at=late + timedelta(minutes=1), source="tick")]
    res = mod.simulate(offers)
    late_strong = [r for r in res if r["outcome"] == "sent" and r["first_at"] == late]
    weak = [r for r in res if r["first_at"] == late + timedelta(minutes=1)][0]
    assert late_strong and late_strong[0]["held_min"] == 0.0 and late_strong[0]["value"] > late_strong[0]["price"]
    assert weak["outcome"] in ("expired", "cap") or weak["held_min"] > 0


def test_curiosity_door_a_needs_only_cap_room_and_strain_report_is_exempt():
    mod = _load()
    offers = [mod.Offer(at=MORNING + timedelta(minutes=i), source="curiosity_outreach") for i in range(7)]
    offers.append(mod.Offer(at=MORNING + timedelta(minutes=30), source="strain_report"))
    res = mod.simulate(offers)
    assert sum(1 for r in res if r["source"] == "curiosity_outreach" and r["outcome"] == "sent") == 6
    assert sum(1 for r in res if r["source"] == "curiosity_outreach" and r["outcome"] == "cap") == 1
    assert [r["outcome"] for r in res if r["source"] == "strain_report"] == ["sent"]
