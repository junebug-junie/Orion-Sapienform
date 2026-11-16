"""evals/event_wants_replay.py --camera-asks: the replay reproduces the live 4b market (issue #3364).

Fixture ``evals/fixtures/camera_asks_live_parity_2026-11-15.json``: the live inputs as of 2026-11-15 00:26Z, written by
``event_wants_replay.py --camera-asks --live --export-parity-fixture`` (ids hashed, no names, answer_detail reduced
to the two flags the grade reads). 280 Orion camera asks and the family's 136 event wants in ``want_ledger``: 120
from the shadow producer (2026-11-04..13, decided ungraded) and the 16 live 4b bids (2026-11-14, all funded).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO = SERVICE_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(SERVICE_ROOT)]

from orion.grammar.camera_ask_acts import REBID_CAP_SEC, REBID_FIRST_SEC, RULES, rebid_after_sec  # noqa: E402
from orion.physiology.binding import REBID_DEFAULT_SEC  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402

spec = importlib.util.spec_from_file_location("ewr", SERVICE_ROOT / "evals/event_wants_replay.py")
ewr = importlib.util.module_from_spec(spec)
sys.modules["ewr"] = ewr          # dataclasses resolve the module by name
spec.loader.exec_module(ewr)

FAMS, _ = load_families(REPO / "config/physiology/families.yaml")
FX = json.loads((SERVICE_ROOT / "evals/fixtures/camera_asks_live_parity_2026-11-15.json").read_text())
ASKS = [{**a, **{k: ewr._ts(a.get(k)) for k in ("created_at", "expires_at", "answered_at")}} for a in FX["asks"]]
LEDGER = FX["ledger"]
LIVE = [r for r in LEDGER if r["producer"] == ewr.BINDING_PRODUCER]
LAST_BID = max(ewr._ts(r["decided_at"]) for r in LIVE)


def test_the_fixture_is_the_live_shape():
    assert len(LIVE) == 16 and all(r["outcome"] == "funded" for r in LIVE)
    shadow = [r for r in LEDGER if r["producer"] != ewr.BINDING_PRODUCER]
    assert len(shadow) == 120 and all(r["graded"] is False for r in shadow)
    assert all(r["graded"] is True for r in LIVE)
    assert sorted(round(r["value_rank"], 6) for r in LIVE)[:2] == [0.025, 0.025]   # the two vehicle bids


def test_the_replay_reproduces_all_16_live_bids_under_the_rule_the_market_ran():
    p = ewr.live_parity(FAMS, ASKS, LEDGER, same_grading=False)
    assert (p["bids"], p["matched"]) == (16, 16), [r for r in p["rows"] if not r["match"]]
    vehicle = [r for r in p["rows"] if r["rule"] == "vehicle"]
    assert len(vehicle) == 2
    for r in vehicle:
        # 3 near-zero shadow-era events of 120 sit below the vehicle's value: rank 0.025, funded.
        assert r["replay_rank"] == pytest.approx(0.025, abs=1e-6) and r["replay_funded"] is True
        assert r["replay_outcome"] == pytest.approx(0.1354, abs=1e-4) and r["replay_reference_n"] == 120


def test_the_full_window_replay_reproduces_them_too():
    """Review finding: the whole 30-day replay, with its stand-ins for the time before the ledger, must still rank
    the live bids against exactly the market's reference (120 rows at 07:00, the earlier live rows after)."""
    p = ewr.live_parity(FAMS, ASKS, LEDGER, same_grading=False, start=ASKS[0]["created_at"])
    assert (p["bids"], p["matched"]) == (16, 16), [r for r in p["rows"] if not r["match"]]
    n = [r["replay_reference_n"] for r in p["rows"]]
    assert n[:9] == [120] * 9                      # the 07:00 batch: the shadow rows only (snapshot 300 s before)
    assert n[-1] == 135 and n == sorted(n)         # each later bid also sees the live bids decided before its snapshot


def test_the_old_replay_reference_did_not_reproduce_them():
    """Before #3364 the replay ranked against its own first bids only (no ledger): the vehicle bids held."""
    out = ewr.camera_asks_gate(FAMS, ASKS, start=ASKS[0]["created_at"], end=LAST_BID + timedelta(seconds=1),
                               same_grading=False, keep_bids=True)
    on_the_day = [x for x in out["rows"] if x["rule"] == "vehicle"
                  and x["bid_log"][0]["at"] >= LAST_BID - timedelta(hours=17)]
    assert len(on_the_day) == 2 and all(x["held_at_all"] for x in on_the_day)


def _probe(same_grading: bool, at: datetime) -> dict:
    return ewr.camera_asks_gate(FAMS, ASKS, start=at, end=at, ledger=LEDGER, same_grading=same_grading,
                                probe_rules=RULES)["next_bid_by_rule"]


def test_the_next_bid_under_same_grading_holds_vehicle_and_funds_confirm_and_same():
    at = LAST_BID + timedelta(minutes=10)
    old, new = _probe(False, at), _probe(True, at)
    assert old["vehicle"]["funded_at_any_salience"] is True and old["vehicle"]["reference_n"] == 136
    # Held even at salience 1.0 (the rank's ceiling), not only at the floor.
    assert new["vehicle"]["held_at_any_salience"] is True and new["vehicle"]["rank_ceiling"] == 0.0
    assert new["vehicle"]["basis"] == "event:graded" and new["vehicle"]["reference_n"] == 16
    assert new["confirm"]["funded_at_any_salience"] and new["same"]["funded_at_any_salience"]


def test_lifecycle_a_held_vehicle_candidate_rebids_on_the_backoff_and_never_opens():
    """A new vehicle ask after the 16 graded bids: held at its first bid, re-bid 600 s doubling to 3 h with no
    deadline, never funded while its expected answer stays under the family mean; its re-bids add no reference
    row (one per want), so the reference stays the 16 graded bids."""
    t0 = LAST_BID + timedelta(minutes=10)
    probe = {"ask_id": "ask-probe-vehicle", "rule": "vehicle", "status": "open", "sightings": 60, "created_at": t0,
             "expires_at": t0 + timedelta(days=7), "answered_at": None, "answer_detail": {}}
    out = ewr.camera_asks_gate(FAMS, ASKS + [probe], start=t0, end=t0 + timedelta(days=2), ledger=LEDGER,
                               keep_bids=True)
    row = next(x for x in out["rows"] if x["ask_id"] == "ask-probe-vehicle")
    assert row["held_at_all"] and row["never_opened"]
    bids = row["bid_log"]
    assert all(b["rank"] == 0.0 and b["basis"] == "event:graded" and b["n_reference"] == 16 for b in bids)
    gaps = [(b["at"] - a["at"]).total_seconds() for a, b in zip(bids, bids[1:])]
    assert gaps[:3] == [REBID_FIRST_SEC, 2 * REBID_FIRST_SEC, 4 * REBID_FIRST_SEC]
    assert gaps == [rebid_after_sec(n + 1, REBID_DEFAULT_SEC) for n in range(len(gaps))] and max(gaps) == REBID_CAP_SEC
    # 2 days at <= 3 h apart: the backoff caps the ledger volume at ~8 a day.
    assert 2 * 86400 / REBID_CAP_SEC <= len(bids) <= 2 * 86400 / REBID_CAP_SEC + 8


def test_the_snapshot_lag_matters_for_parity():
    """The market reads its acts through a 300 s cache. Juniper answered a batch of cards at 23:38-23:41; the
    23:41:37 bid did not see those closes yet (live: hunger 140670 s, rank 1.0). With no lag the replay sees them,
    the hunger falls to 0.29, and that bid no longer matches."""
    p = ewr.live_parity(FAMS, ASKS, LEDGER, same_grading=False, snapshot_lag=timedelta(0))
    miss = [r for r in p["rows"] if not r["match"]]
    assert len(miss) == 1 and miss[0]["decided_at"].startswith("2026-11-14T23:41")


def test_the_gate_counterfactual_holds_vehicles_and_the_as_ledger_view_does_not():
    """The gate asks "had 4b bid every ask": every ask with no 4b bid gets a stand-in, so the shadow era's ungraded
    rows are replaced by graded bids. The as-ledger view is what the market really had: no graded event before
    2026-11-14, so those bids ranked cold and vehicles were funded."""
    start = ASKS[0]["created_at"] + timedelta(days=1)
    end = LAST_BID + timedelta(seconds=1)
    cf = ewr.camera_asks_gate(FAMS, ASKS, start=start, end=end, ledger=LEDGER, counterfactual=True)
    seen = ewr.camera_asks_gate(FAMS, ASKS, start=start, end=end, ledger=LEDGER)
    assert cf["checks"]["vehicle_held_at_least_70pct"] and cf["checks"]["confirm_and_same_held_at_most_10pct"]
    assert seen["by_rule"]["vehicle"]["held_at_all"] < cf["by_rule"]["vehicle"]["held_at_all"]


# --- issue #3381: the per-rule exploration floor -------------------------------------------------------------

def _steady(rule_floor):
    """The steady state of the live market (today's reference, hunger saturated), with every rule's expected answer
    pinned at the fixture's end: sighting (0.80) sits below the family mean, as it does live."""
    end = LAST_BID + timedelta(seconds=1)
    return ewr.camera_asks_gate(FAMS, ASKS, start=end - timedelta(days=30), end=end, ledger=LEDGER, steady_state=True,
                                pin_outcomes=True, rule_floor=rule_floor)


def test_a_rule_below_the_family_mean_starves_on_the_old_rule_and_gets_its_share_with_the_floor():
    from orion.grammar.camera_ask_acts import RULE_EXPLORATION_FLOOR

    old, new = _steady(None), _steady(RULE_EXPLORATION_FLOOR)
    # Old rule: half the sighting asks are never opened (an answered one is lost); no floor opens anywhere.
    assert old["by_rule"]["sighting"]["never_opened_in_horizon"] == 0.5 and old["lost_answers_by_rule"].get("sighting")
    assert all(v["floor_opens"] == 0 for v in old["by_rule"].values())
    # Floor: every sighting ask opens; no answer lost to starvation, only vehicle ones.
    assert new["by_rule"]["sighting"]["never_opened_in_horizon"] == 0.0 and new["by_rule"]["sighting"]["floor_opens"] > 0
    assert set(new["lost_answers_by_rule"]) == {"vehicle"}
    assert new["answered_loss_share"] < old["answered_loss_share"]


def test_the_floor_keeps_vehicle_mostly_held_and_the_4b_gate_passing():
    from orion.grammar.camera_ask_acts import RULE_EXPLORATION_FLOOR

    new = _steady(RULE_EXPLORATION_FLOOR)
    assert all(new["checks"].values()), new["checks"]
    v = new["by_rule"]["vehicle"]
    assert v["never_opened_in_horizon"] >= ewr.GATE_VEHICLE_HELD_MIN and 0 < v["floor_opens"] <= 0.15 * v["asks"]
    # Confirm and same rank above the mean: never held, never floor-opened.
    assert all(new["by_rule"][r]["held_at_all"] == 0 and new["by_rule"][r]["floor_opens"] == 0 for r in ("confirm", "same"))


def test_parity_by_era_compares_each_bid_under_the_rule_live_when_it_was_made():
    before = {"rows": [{"event_ref": "a", "decided_at": "2026-11-14T07:00:00+00:00", "match": True},
                       {"event_ref": "b", "decided_at": "2026-11-15T07:00:00+00:00", "match": False}]}
    after = {"rows": [{"event_ref": "a", "decided_at": "2026-11-14T07:00:00+00:00", "match": False},
                      {"event_ref": "b", "decided_at": "2026-11-15T07:00:00+00:00", "match": True}]}
    p = ewr.parity_by_era(before, after, ewr.RULE_3364_LIVE_FROM)
    assert (p["bids"], p["matched"]) == (2, 2)
    assert [r["rule_era"] for r in p["rows"]] == ["before_3364", "same_grading"]
