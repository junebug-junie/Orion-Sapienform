"""Patch A of the shared-bet-primitives design changes no live number on the orion-prediction side: turn
and street bet terms, room/body conformal excess, reachable log loss and calibration bins, the grader's
bins, binary entropy (action bets, curiosity) and Wilson (self-monitor, promise eval) reproduce their
pre-refactor values to 1e-9 on live stored rows (orion/prediction/tests/surprisal_golden.py)."""
from __future__ import annotations

import pytest

from orion.prediction.tests import surprisal_golden as golden

DATA = golden.load()
EXPECTED = DATA["expected"]["service"]


@pytest.fixture(scope="module")
def got() -> dict:
    return golden.compute_service(DATA["inputs"], separate=False)


@pytest.mark.parametrize("key", sorted(set(EXPECTED) - set(golden.SEPARATE_SERVICE_KEYS)))
def test_service_output_reproduces_the_pre_refactor_value(got, key):
    assert golden.diffs(got[key], EXPECTED[key], key) == []


def test_the_live_rows_are_not_degenerate():
    turns = [t for t in EXPECTED["turns"] if t[1] == "scored"]
    assert len(turns) >= 200 and sum(t[2][3] > 0 for t in turns) >= 30
    street = EXPECTED["street_terms"]
    assert len(street) >= 300 and {s[2] for s in street} == {True, False}
    assert len(EXPECTED["reachable_log_loss"]) >= 300 and len(EXPECTED["room_body_excess"]) >= 1000


def test_street_clamp_is_symmetric():
    """street_rhythm clamps with the shared clip at P_MIN; that equals its [P_MIN, P_MAX] only while they
    are symmetric."""
    import app.street_rhythm as sr

    assert sr.P_MAX == 1.0 - sr.P_MIN


def test_callers_import_the_shared_math():
    import app.action_bets as ab
    import app.reachable as rc
    import app.surprise as su
    import app.turn_surprise as ts
    from orion.prediction import grading, surprisal

    assert ab.bernoulli_entropy is grading.bernoulli_entropy is surprisal.bernoulli_entropy
    assert rc.log_loss is ts.log_loss is grading.log_loss is surprisal.log_loss
    assert ts.node_value is surprisal.node_value
    assert su.P_FLOOR is surprisal.P_FLOOR


def test_the_self_monitor_wilson_is_95_percent_and_keeps_its_empty_reading():
    """The one Wilson (orion/self_monitor/record.py). Fractional Kish n_eff is pinned in the golden
    ``record_wilson`` key. Empty: the self-monitor reads (0, 1) (an unknown share); the promise eval keeps its
    own None reading before delegating. Both homes keep their pre-refactor empty behavior."""
    from orion.self_monitor import record

    assert record.Z == 1.96
    assert record.wilson(0.5, 0.0) == (0.0, 1.0) and record.wilson(0.5, -2.0) == (0.0, 1.0)
