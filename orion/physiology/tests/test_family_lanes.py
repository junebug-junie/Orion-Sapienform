"""Every lane a want family or a bidding verb names is a lane the body reads.

2026-11-13: c91a21fd8 moved the metacog work class onto gpu3's ``fast`` seat and removed the
``metacog`` seat. families.yaml and 11 verb YAMLs still named ``metacog`` as a lane, so from 04:01Z
every dream.cycle bid was deferred ``no_acceptable_lane`` ("none of ['metacog'] is a lane the body
reads") and Orion stopped sleeping. The body builds one ``lane:<role>`` constraint per pool ROLE
(orion/physiology/body.py lane_readings); a work class is never read.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from orion.physiology.choice import ReservationBook, build_context, decide
from orion.physiology.families import body_lanes, load_families
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R
from orion.schemas.physiology import PhysiologyStateV1
from orion.schemas.physiology_want import WantV1

REPO = Path(__file__).resolve().parents[3]
POOL = REPO / "config" / "gpu_pool.yaml"
FAMILIES = REPO / "config" / "physiology" / "families.yaml"
VERBS = REPO / "orion" / "cognition" / "verbs"
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def test_the_verb_gate_and_the_family_gate_use_the_same_lane_set():
    from orion.cognition.verb_physiology import load_pool_lanes

    assert load_pool_lanes(POOL) == body_lanes(POOL)


def unreadable_family_lanes(families_doc: dict, lanes: frozenset[str]) -> dict[str, list[str]]:
    return {name: bad for name, body in (families_doc.get("families") or {}).items()
            if (bad := [x for x in ((body or {}).get("lanes") or []) if x not in lanes])}


def test_every_family_lane_is_a_lane_the_body_reads():
    doc = yaml.safe_load(FAMILIES.read_text(encoding="utf-8"))
    assert unreadable_family_lanes(doc, body_lanes(POOL)) == {}


def test_the_gate_catches_the_retired_metacog_seat():
    """Mutation check, kept as a test: put metacog back on dream.cycle and the gate must fail."""
    doc = yaml.safe_load(FAMILIES.read_text(encoding="utf-8"))
    doc["families"]["dream.cycle"]["lanes"] = ["metacog"]
    assert unreadable_family_lanes(doc, body_lanes(POOL)) == {"dream.cycle": ["metacog"]}
    # metacog is a pool work CLASS now (routed to the fast seat), never a lane.
    assert "metacog" not in body_lanes(POOL)


def test_every_bidding_verb_lane_is_a_lane_the_body_reads():
    """A cortex-orch dispatch bid carries the verb's lanes, and Choice uses them in place of the
    family's (choice.py: ``want.lanes or cfg.lanes``)."""
    lanes = body_lanes(POOL)
    bad = {}
    for path in sorted(VERBS.glob("*.yaml")):
        block = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("physiology") or {}
        if miss := [x for x in (block.get("lanes") or []) if x not in lanes]:
            bad[path.name] = miss
    assert bad == {}


# --- regression: dream.cycle is placed against the live lane set ---------------------------------

def _price(v=0.0, horizon=3600.0):
    return {"value": v, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": horizon, "why": ""}


def _live_body() -> PhysiologyStateV1:
    lanes = [R(constraint=f"lane:{role}", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
               detail={"slots": 1, "lease_s": 0.0, "window_sec": 3600.0, "deferrable_queued": 0,
                       "deferrable_waiting_sec": 0.0}, price=_price())
             for role in sorted(body_lanes(POOL))]
    base = [R(constraint="heat", unit="c", level=28.0, budget=31.0, budget_kind="soft_line", price=_price(horizon=21600.0)),
            R(constraint="attention:juniper", unit="messages", level=0.0, budget=4.0, budget_kind="daily_cap",
              price=_price(horizon=86400.0))]
    return PhysiologyStateV1(tick_id=f"physiology:{NOW.isoformat()}", generated_at=NOW, constraints=base + lanes)


def _decide_dream(lanes=None):
    fams, err = load_families(FAMILIES)
    assert err is None
    st = _live_body()
    ctx = build_context(fams, [], [], st, NOW, weights={})
    kw = {"lanes": lanes} if lanes is not None else {}
    w = WantV1(want_id="dream-1", family="dream.cycle", producer="orion-dream", verb="sleep", tier=1,
               opened_at=NOW - timedelta(hours=7), **kw)
    return decide(w, st, ctx, ReservationBook(), NOW)


def test_dream_cycle_family_is_placed_on_the_live_lane_set():
    d = _decide_dream()
    assert d.reason != "no_acceptable_lane", d.why
    assert d.lane == "fast"


def test_dream_cycle_verb_bid_is_placed_on_the_live_lane_set():
    """The cortex-orch dispatch bid for the dream_cycle verb carries the verb's own lanes."""
    block = yaml.safe_load((VERBS / "dream_cycle.yaml").read_text(encoding="utf-8"))["physiology"]
    d = _decide_dream(lanes=list(block["lanes"]))
    assert d.reason != "no_acceptable_lane", d.why
    assert d.lane == "fast"


def test_a_metacog_only_dream_bid_is_the_bug():
    """What production did from 04:01Z on 2026-11-13: proves the regression tests above can fail."""
    d = _decide_dream(lanes=["metacog"])
    assert d.reason == "no_acceptable_lane" and d.lane is None
