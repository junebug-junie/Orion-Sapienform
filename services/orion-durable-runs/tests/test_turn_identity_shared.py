"""The turn id durable-runs stamps on every lease is the one the act ledger re-derives (patch 3a).

orion/grammar/act_cost_ids.py charges a read act for its turns' GPU leases by re-deriving each turn's
id from the run's pool hold. If durable-runs ever changed its rule without the shared function, reads
would silently go back to costing nothing; this pins the two together, against a live read's ids.
"""
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.graph import turn_correlation_id  # noqa: E402
from orion.durable_runs.turn_identity import durable_turn_correlation_id  # noqa: E402

# A live read's stage-1 run, its hold, and the turn id Hub recorded as the seed's trace_id (2026-11-09).
RUN, CORR, HOLD, LIVE_TRACE = ("reading-95c9b912-01a7-5d79-b071-0a4c8fadcab5", "86e84801-7ff4-4cb8-81df-35ab8fc95f25",
                               "fbaf0d796f1247e1a57705763db91f7f", "fd92e090-eb5e-5c35-bcdb-67dae5175579")


def _state(generation=1, fence=0):
    return {"run_id": RUN, "correlation_id": CORR, "attempt": 0,
            "lease": {"lease_id": HOLD, "generation": generation, "role": "agent-gpu2", "holder": f"durable-runs:{RUN}"},
            **({"turn_fence": fence} if fence else {})}


def test_graph_turn_id_is_the_shared_rule_and_matches_a_live_trace():
    assert turn_correlation_id(_state()) == durable_turn_correlation_id(RUN, CORR, HOLD, 1) == LIVE_TRACE


def test_generation_and_fence_change_the_id_the_same_way_in_both():
    assert turn_correlation_id(_state(generation=2)) == durable_turn_correlation_id(RUN, CORR, HOLD, 2) != LIVE_TRACE
    fenced = turn_correlation_id(_state(fence=3))
    assert fenced == durable_turn_correlation_id(RUN, CORR, HOLD, 1, fence=3)
    assert fenced == str(uuid5(NAMESPACE_URL, f"orion:durable:turn:{RUN}:{CORR}:{HOLD}:1:fence:3"))


def test_no_hold_keeps_the_run_correlation_id():
    assert turn_correlation_id({"run_id": RUN, "correlation_id": CORR}) == CORR
