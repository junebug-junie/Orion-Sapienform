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
    assert old["vehicle"]["funded"] is True and old["vehicle"]["reference_n"] == 136
    assert new["vehicle"]["funded"] is False and new["vehicle"]["rank"] == 0.0
    assert new["vehicle"]["basis"] == "event:graded" and new["vehicle"]["reference_n"] == 16
    assert new["confirm"]["funded"] and new["same"]["funded"]


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
