"""Calm-theory tails and the per-node attention band rule (#3917)."""
from __future__ import annotations

import math
import random

import pytest

from orion.substrate import calm_band as cb


def test_ids_match_the_producers() -> None:
    from orion.schemas import attention_candidate as ac
    from orion.substrate import prediction_joint_surprise as pj
    from orion.substrate import prediction_surprise as ps
    from orion.substrate import prediction_turn_surprise as pt

    assert cb.ROOM_NODE_ID == ps.ROOM_NODE_ID and cb.BODY_NODE_ID == ps.BODY_NODE_ID
    assert cb.ROOM_JOINT_NODE_ID == pj.ROOM_JOINT_NODE_ID and cb.BODY_JOINT_NODE_ID == pj.BODY_JOINT_NODE_ID
    assert cb.CHAT_TURN_NODE_ID == pt.CHAT_TURN_NODE_ID
    assert cb.PERCEPTION_NODE_ID == ac.PERCEPTION_NODE_ID
    assert cb.PERCEPTION_WALKWAY_NODE_ID == ac.PERCEPTION_WALKWAY_NODE_ID
    assert ps.Z_FULL_SCALE == pj.Z_FULL_SCALE == pt.Z_FULL_SCALE == cb.Z_FULL_SCALE


def test_the_table() -> None:
    t = cb.ATTENTION_BAND_RULE_BY_NODE_ID
    assert t[cb.ROOM_NODE_ID] == t[cb.BODY_NODE_ID] == cb.RULE_FLOORED
    assert t[cb.ROOM_JOINT_NODE_ID] == t[cb.BODY_JOINT_NODE_ID] == cb.RULE_GAMMA
    assert t[cb.CHAT_TURN_NODE_ID] == cb.RULE_CHAT_BETS
    # Cameras: explicitly empirical (dense; #3874 scale-change guard; walkway p99 floor).
    assert t[cb.PERCEPTION_NODE_ID] == t[cb.PERCEPTION_WALKWAY_NODE_ID] == cb.RULE_EMPIRICAL
    assert set(t.values()) <= cb.CALM_RULES | {cb.RULE_EMPIRICAL}
    for nid in ("node:substrate.cabinet", "node:substrate.street", "node:other"):
        assert cb.band_rule(nid) == cb.RULE_EMPIRICAL and cb.calm_percentiles(nid, 1.0) is None


def test_floored_exact_tail_matches_monte_carlo() -> None:
    var = cb.calm_excess_variance()
    for n in (1, 2, 6):
        rng = random.Random(3917 + n)
        draws = 60_000
        hits = sum(
            sum(-math.log(max(rng.random(), cb.P_FLOOR)) - 0.95 for _ in range(n)) / math.sqrt(n * var) >= 1.3
            for _ in range(draws)
        )
        assert cb.standardized_floored_tail(n, 1.3) == pytest.approx(hits / draws, abs=0.006)


def test_floored_tail_is_zero_past_the_cap() -> None:
    # One outcome cannot exceed -ln 0.05 = 3.0 nats: z <= (3.0 - 0.95) / 0.835 = 2.45.
    assert cb.standardized_floored_tail(1, 2.5) == pytest.approx(0.0, abs=1e-12)
    assert cb.floored_sum_tail(3, 0.0) == pytest.approx(1.0)


def test_gamma_tail() -> None:
    assert cb.gamma_tail(1, 2.0) == pytest.approx(math.exp(-2.0))
    assert cb.gamma_tail(3, 0.0) == 1.0
    # Standardized Gamma(n) 95th percentile <= 2.0 at every n (the #3826 forecast bar).
    for n in range(1, 13):
        assert cb.standardized_gamma_tail(n, 2.0) <= 0.05


@pytest.mark.parametrize("rule", [cb.RULE_GAMMA, cb.RULE_FLOORED])
def test_tail_is_the_max_over_window_sizes(rule) -> None:
    one = cb.standardized_gamma_tail if rule == cb.RULE_GAMMA else cb.standardized_floored_tail
    for z in (0.3, 1.0, 1.5, 2.2, 3.0):
        t = cb.z_tail(rule, z)
        assert all(t >= one(n, z) - 1e-12 for n in range(1, cb.MAX_WINDOW_OUTCOMES + 1))
        assert t >= cb.normal_tail(z) - 1e-12


def test_bet_atoms_are_a_standardized_calm_law() -> None:
    for qs in ([0.1], [0.12, 0.83], [0.04, 0.03, 0.6]):
        atoms = cb.bet_window_atoms(qs)
        assert sum(p for _, p in atoms) == pytest.approx(1.0)
        assert sum(z * p for z, p in atoms) == pytest.approx(0.0, abs=1e-12)
        assert sum(z * z * p for z, p in atoms) == pytest.approx(1.0)
    # One bet's rare side reads z = sqrt((1 - q) / q) with calm probability q (Cantelli, tight).
    q = 0.1
    assert cb.bet_window_tail([q], math.sqrt((1 - q) / q)) == pytest.approx(q)


@pytest.mark.parametrize("rule", sorted(cb.CALM_RULES))
def test_cantelli_bounds_every_rule(rule) -> None:
    for z in (0.5, 1.0, 2.0, 2.9):
        assert cb.z_tail(rule, z) <= cb.z_tail(cb.RULE_CANTELLI, z) + 1e-12


@pytest.mark.parametrize("node_id", [cb.ROOM_NODE_ID, cb.BODY_NODE_ID, cb.ROOM_JOINT_NODE_ID,
                                     cb.BODY_JOINT_NODE_ID, cb.CHAT_TURN_NODE_ID])
def test_percentiles_rest_and_rise(node_id) -> None:
    strict0, mid0 = cb.calm_percentiles(node_id, 0.0)
    assert strict0 == 0.0 and mid0 <= 0.5  # rest: nothing calm reads below 0
    prev = -1.0
    for v in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        strict, mid = cb.calm_percentiles(node_id, v)
        assert 0.0 <= strict <= mid <= 1.0 and strict >= prev
        prev = strict
    # The clip: 1.0 is the top atom, ranked above every value below it.
    assert cb.calm_percentiles(node_id, 1.0)[1] > cb.calm_percentiles(node_id, 0.99)[1]
    assert cb.calm_percentiles(node_id, float("nan")) is None


def _first_value_at(node_id: str, pct: float) -> float | None:
    return next((k / 1000 for k in range(1001) if cb.calm_percentiles(node_id, k / 1000)[0] >= pct), None)


def test_where_each_rule_reads_high() -> None:
    """The value at which a calm window reads lower 90 % of the time ("high", eligible)."""
    assert _first_value_at(cb.ROOM_JOINT_NODE_ID, 0.9) == pytest.approx(0.447, abs=0.002)
    assert _first_value_at(cb.ROOM_NODE_ID, 0.9) == pytest.approx(0.540, abs=0.002)
    assert _first_value_at(cb.CHAT_TURN_NODE_ID, 0.9) == pytest.approx(0.521, abs=0.002)
    # Chat's curiosity bar (0.8, #3826) is about 5 % calm, so it reads high for attention too.
    assert 0.95 <= cb.calm_percentiles(cb.CHAT_TURN_NODE_ID, 0.8)[0] < 0.96
    # Joint nodes clip at z 3 before a 1 % calm tail: they never read "unusual" (honest, not a bug).
    assert _first_value_at(cb.ROOM_JOINT_NODE_ID, 0.99) is None


def test_a_new_calibrated_node_can_join_with_cantelli() -> None:
    """node:prediction.street_rhythm (another branch) has a calm theory: excess surprisal of a
    calibrated stated p. Until its own tail is derived, the distribution-free rule fits it."""
    table = {**cb.ATTENTION_BAND_RULE_BY_NODE_ID, "node:prediction.street_rhythm": cb.RULE_CANTELLI}
    assert cb.has_calm_band("node:prediction.street_rhythm", table)
    assert cb.calm_percentiles("node:prediction.street_rhythm", 1.0, table)[0] == pytest.approx(0.9)
    assert cb.calm_percentiles("node:prediction.street_rhythm", 0.9, table)[0] < 0.9


def test_chat_mix_is_the_live_fixture() -> None:
    mix = cb.chat_turn_q_mix()
    assert len(mix) == 71 and all(len(t) == 2 for t in mix)
    assert all(0.0 < q < 1.0 for t in mix for q in t)


def test_unreadable_chat_mix_falls_back_to_cantelli(monkeypatch, tmp_path) -> None:
    """A missing data file must not raise inside the contest's magnitude pass."""
    cb._chat_mixes.cache_clear()
    cb.chat_turn_q_mix.cache_clear()
    monkeypatch.setattr(cb, "CHAT_TURN_Q_MIX_PATH", tmp_path / "missing.json")
    try:
        assert cb.calm_percentiles(cb.CHAT_TURN_NODE_ID, 1.0)[0] == pytest.approx(0.9)
    finally:
        cb._chat_mixes.cache_clear()
        cb.chat_turn_q_mix.cache_clear()


@pytest.mark.parametrize("delta", [0.0, 1e-7, -1e-7])
def test_floored_one_outcome_atom_is_kept(delta) -> None:
    """A one-outcome window at the p floor stores exactly (ln 20 - 0.95) / sd / 3 (calm prob 5 %).
    Stored values carry 6 decimals, so the atom must be matched within a tolerance."""
    v = (math.log(20.0) - 0.95) / math.sqrt(cb.calm_excess_variance()) / 3.0
    for x in (v + delta, round(v, 6)):
        strict, mid = cb.calm_percentiles(cb.ROOM_NODE_ID, x)
        assert strict == pytest.approx(0.95, abs=1e-4)
        assert mid == pytest.approx(0.966, abs=1e-3)


def test_p_floor_matches_orion_prediction() -> None:
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "services" / "orion-prediction" / "app" / "surprise.py"
    tree = ast.parse(src.read_text())
    # One floor: orion-prediction imports it from the shared library this module also reads (Patch A of
    # the shared-bet-primitives design), and assigns no copy of its own.
    assigned = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "P_FLOOR" for t in n.targets)]
    imported = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "orion.prediction.surprisal"
                and any(a.name == "P_FLOOR" for a in n.names)]
    assert not assigned and len(imported) == 1
    from orion.prediction import surprisal

    assert cb.P_FLOOR is surprisal.P_FLOOR == 0.05


def test_bet_atoms_match_brute_force() -> None:
    """Independent enumeration (turn_surprise.py math) against bet_window_tail."""
    import itertools

    qs = [0.12, 0.83, 0.05]
    lgs = [math.log(q / (1 - q)) for q in qs]
    sd = math.sqrt(sum(q * (1 - q) * lg * lg for q, lg in zip(qs, lgs)))
    for zcut in (0.5, 1.5, 2.4, 3.5):
        want = 0.0
        for ys in itertools.product((0, 1), repeat=len(qs)):
            p = math.prod(q if y else 1 - q for q, y in zip(qs, ys))
            e = sum(-(y - q) * lg for q, y, lg in zip(qs, ys, lgs))
            want += p if e / sd >= zcut - 1e-12 else 0.0
        assert cb.bet_window_tail(qs, zcut) == pytest.approx(want)
