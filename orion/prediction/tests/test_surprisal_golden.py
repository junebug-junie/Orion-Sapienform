"""Patch A of the shared-bet-primitives design changes no live number: every substrate surprise node
(room, body, room_joint, body_joint, chat_turn, street_rhythm) reproduces its pre-refactor value to 1e-9 on
events replayed from live stored rows, the moved math lives in one place, and street_rhythm stays shadow.

Fixture and how it was built: ``surprisal_golden.py`` (beside this file). The service-side terms (bet terms,
log loss, bins, entropy, Wilson) are pinned by services/orion-prediction/tests/test_surprisal_golden.py.
"""
from __future__ import annotations

import ast
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.prediction import surprisal as sp
from orion.prediction.tests import surprisal_golden as golden

REPO = Path(__file__).resolve().parents[3]
DATA = golden.load()
EXPECTED = DATA["expected"]["substrate"]


@pytest.fixture(scope="module")
def got() -> dict:
    return golden.compute_substrate(DATA["inputs"])


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_substrate_output_reproduces_the_pre_refactor_value(got, key):
    assert golden.diffs(got[key], EXPECTED[key], key) == []


@pytest.mark.parametrize("key,node_ids", [
    ("room_body", {"node:prediction.room", "node:prediction.body"}),
    ("joint", {"node:prediction.room_joint", "node:prediction.body_joint"}),
    ("chat_turn", {"node:prediction.chat_turn"}),
    ("street_rhythm", {"node:prediction.street_rhythm"}),
])
def test_the_golden_fixture_is_not_degenerate(key, node_ids):
    """The fixture would pass trivially if every window read 0: each node needs many accepted live
    outcomes, some windows above the rest point, and some at it."""
    rows = EXPECTED[key]
    accepted = [r for r in rows if r[1] == "accepted"]
    values = [r[2][2] for r in accepted if r[2] is not None]   # None: a window of only 50/50 forecasts
    assert {r[0] for r in accepted} == node_ids
    assert len(accepted) >= 250
    assert sum(v > 0 for v in values) >= 50 and sum(v == 0 for v in values) >= 50


def test_promise_eval_wilson_reproduces_the_pre_refactor_value():
    """Checked here, not in the orion-prediction golden: the promise eval needs jinja2."""
    want = DATA["expected"]["service"]["promise_wilson"]
    assert golden.diffs(golden.compute_promise_wilson(), want, "promise_wilson") == []


def test_street_rhythm_stays_shadow_with_no_band_rule():
    """Patch A adds no calm_band rule for street_rhythm: its rule ships with the go-live decision (#3950)."""
    from orion.schemas import attention_candidate as ac
    from orion.substrate import calm_band as cb
    from orion.substrate.prediction_street_rhythm import STREET_RHYTHM_NODE_ID

    assert ac.STREET_RHYTHM_SHADOW is True
    assert STREET_RHYTHM_NODE_ID in ac.SHADOW_NODE_IDS
    assert STREET_RHYTHM_NODE_ID not in cb.ATTENTION_BAND_RULE_BY_NODE_ID
    assert cb.band_rule(STREET_RHYTHM_NODE_ID) == cb.RULE_EMPIRICAL
    assert not cb.has_calm_band(STREET_RHYTHM_NODE_ID)


def _py_files():
    for root in ("orion", "services", "scripts"):
        for path in (REPO / root).rglob("*.py"):
            if "node_modules" in path.parts or ".venv" in path.parts:
                continue
            yield path


def _defs_and_assigns():
    defs: dict[str, list[str]] = {}
    assigns: dict[str, list[str]] = {}
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(REPO))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(node.name, []).append(rel)
            elif isinstance(node, ast.Assign) and node in tree.body:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assigns.setdefault(t.id, []).append(rel)
    return defs, assigns


@pytest.fixture(scope="module")
def defs_and_assigns():
    return _defs_and_assigns()


@pytest.mark.parametrize("name", ["bernoulli_entropy", "log_loss", "calm_excess_variance", "independent_origin"])
def test_each_moved_function_is_defined_once_in_the_shared_library(defs_and_assigns, name):
    defs, _ = defs_and_assigns
    assert defs.get(name) == ["orion/prediction/surprisal.py"]


def test_z_full_scale_is_one_constant(defs_and_assigns):
    _, assigns = defs_and_assigns
    assert assigns.get("Z_FULL_SCALE") == ["orion/prediction/surprisal.py"]
    from orion.substrate import calm_band, curiosity_seed_bar, prediction_joint_surprise, prediction_street_rhythm
    from orion.substrate import prediction_surprise, prediction_turn_surprise

    for mod in (calm_band, curiosity_seed_bar, prediction_joint_surprise, prediction_street_rhythm,
                prediction_surprise, prediction_turn_surprise):
        assert mod.Z_FULL_SCALE is sp.Z_FULL_SCALE


def test_the_substrate_folds_read_the_shared_library():
    """No substrate surprise module carries its own fold: each calls surprisal.fold."""
    for name in ("prediction_surprise", "prediction_joint_surprise", "prediction_turn_surprise",
                 "prediction_street_rhythm"):
        src = (REPO / "orion" / "substrate" / f"{name}.py").read_text()
        assert "math.sqrt" not in src, name
        assert "fold(" in src, name


# --- the primitives on their own ---------------------------------------------------------------------


def test_bet_terms_mean_zero_and_stated_variance_under_their_own_q():
    for q in (0.02, 0.1, 0.3, 0.7, 0.95):
        hit, miss = sp.bet_terms(q, True, 0.01), sp.bet_terms(q, False, 0.01)
        assert q * hit.excess + (1 - q) * miss.excess == pytest.approx(0.0, abs=1e-12)
        assert q * hit.excess ** 2 + (1 - q) * miss.excess ** 2 == pytest.approx(hit.calm_var, rel=1e-9)
        assert hit.entropy == sp.bernoulli_entropy(q)


def test_clip_is_a_parameter():
    assert sp.bet_terms(1.0, False, 0.02).q == 0.98 and sp.bet_terms(0.0, True, 0.01).q == 0.01
    assert sp.log_loss(1.0, False, 1e-4) == pytest.approx(-math.log(1e-4))
    assert sp.bernoulli_entropy(1.0) == 0.0 and sp.bernoulli_entropy(1.0, clip=0.01) > 0.0


def test_conformal_terms_rest_at_their_calm_point():
    t = sp.conformal_terms(0.5)
    assert t.calm_mean == pytest.approx(0.95) and t.excess == pytest.approx(math.log(2) - 0.95)
    assert sp.conformal_terms(0.0).p == sp.P_FLOOR and sp.conformal_terms(2.0).surprisal == 0.0
    assert t.calm_var == pytest.approx(0.69793, abs=1e-5)
    with pytest.raises(ValueError):
        sp.calm_excess_variance(1.0)


def test_fold_and_node_value():
    assert sp.fold([]) is None
    assert sp.fold([(0.0, 0.0)], min_var=1e-12) is None
    f = sp.fold([(2.0, 1.0), (-1.0, 3.0)])
    assert f.n == 2 and f.total == 1.0 and f.z == 0.5 and f.mean_excess == 0.5
    assert f.value == sp.node_value(0.5) == pytest.approx(0.5 / 3)
    assert sp.node_value(-1.0) == 0.0 and sp.node_value(9.0) == 1.0


def test_independent_origin_and_one_outcome_per_target():
    t0 = datetime(2026, 11, 11, 8, 45, 3, tzinfo=timezone.utc)
    assert sp.independent_origin(t0, 15) and not sp.independent_origin(t0.replace(minute=50), 15)
    assert sp.independent_origin(t0.replace(tzinfo=None), 5)

    class S:
        def __init__(self, target, h, tag):
            self.target_ts, self.horizon_min, self.tag = target, h, tag

    a, b = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc), datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = sp.shortest_horizon_per_target([S(a, 15, "x"), S(a, 5, "y"), S(b, 5, "z"), S(b, 5, "w")])
    assert [r.tag for r in rows] == ["z", "y"]
