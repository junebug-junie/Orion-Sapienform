"""Per-node curiosity seed bars (#3821): each live node crosses at its own calibrated cut."""
from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.substrate import curiosity_seed_bar as csb
from orion.substrate.endogenous_curiosity import (
    EndogenousCuriosityConfig,
    endogenous_curiosity_candidates,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOW = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
FORECAST_BAR = 2.0 / 3.0
CHAT_BAR = 0.8


def _node(node_id: str, value: float, *, age_s: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        metadata={"prediction_error": value},
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=age_s)),
    )


def _seeded(*nodes) -> dict[str, float]:
    out = endogenous_curiosity_candidates(
        nodes=list(nodes), config=EndogenousCuriosityConfig(enabled=True, budget=8), now=NOW
    )
    return {c.focal_node_refs[0]: c.signal_strength for c in out}


def test_ids_match_the_producers() -> None:
    from orion.substrate import prediction_joint_surprise as pj
    from orion.substrate import prediction_surprise as ps
    from orion.substrate import prediction_turn_surprise as pt
    from orion.schemas import attention_candidate as ac

    assert csb.ROOM_NODE_ID == ps.ROOM_NODE_ID and csb.BODY_NODE_ID == ps.BODY_NODE_ID
    assert csb.ROOM_JOINT_NODE_ID == pj.ROOM_JOINT_NODE_ID
    assert csb.BODY_JOINT_NODE_ID == pj.BODY_JOINT_NODE_ID
    assert csb.CHAT_TURN_NODE_ID == pt.CHAT_TURN_NODE_ID
    assert csb.PERCEPTION_NODE_ID == ac.PERCEPTION_NODE_ID
    assert csb.PERCEPTION_WALKWAY_NODE_ID == ac.PERCEPTION_WALKWAY_NODE_ID
    # every producer stores clip(z / 3, 0, 1): the bars are in those units
    assert ps.Z_FULL_SCALE == pj.Z_FULL_SCALE == pt.Z_FULL_SCALE == csb.Z_FULL_SCALE


def _gamma_tail(n: int, x: float) -> float:
    """P(Gamma(n, 1) >= x) for integer n (Poisson sum)."""
    return sum(math.exp(-x) * x**k / math.factorial(k) for k in range(n))


@pytest.mark.parametrize("n", range(1, 13))
def test_forecast_cut_is_calm_p_005_at_every_window_size(n: int) -> None:
    """Calm: excess per outcome is Exp(1) - 1, so the window z is a standardized Gamma(n).
    z >= 2.0 has tail <= 0.05 for every n; the old 1.645 (value 0.55) does not."""
    tail_new = _gamma_tail(n, n + csb.FORECAST_Z_CUT * math.sqrt(n))
    tail_old = _gamma_tail(n, n + 1.645 * math.sqrt(n))
    assert tail_new <= 0.05
    assert tail_old > 0.06


def _chat_turn_calm_tail(qs, zcut: float) -> float:
    """Exact calm P(turn z >= zcut) for one turn's bets at calibrated q's (turn_surprise.py math:
    excess = -(y - q) logit q, calm_var = q (1 - q) logit(q)^2)."""
    qs = [q for q in qs if q is not None]
    total = 0.0
    for ys in itertools.product((0, 1), repeat=len(qs)):
        p, e, v = 1.0, 0.0, 0.0
        for q, y in zip(qs, ys):
            lg = math.log(q / (1 - q))
            p *= q if y else 1 - q
            e += -(y - q) * lg
            v += q * (1 - q) * lg * lg
        if e / math.sqrt(v) >= zcut - 1e-12:
            total += p
    return total


def test_chat_cut_mean_calm_rate_on_live_bets() -> None:
    """The 71 settled chat turns of 2026-11-17..24 (their calibrated q's only): mean exact calm
    crossing rate per turn is <= 5 % at the chat bar and > 5 % at the old 0.55."""
    turns = json.loads((FIXTURES / "chat_turn_calibrated_q_2026-11-24.json").read_text())["turns"]
    assert len(turns) == 71

    def mean_rate(zcut: float) -> float:
        return sum(_chat_turn_calm_tail(t, zcut) for t in turns) / len(turns)

    assert mean_rate(csb.CHAT_TURN_Z_CUT) <= 0.05
    assert mean_rate(1.65) > 0.08


def test_single_bet_calm_tail_is_q_when_reachable() -> None:
    """Why no chat cut below z 4.36 guarantees p <= 0.05 per turn: a lone bet's rare side reads
    z = sqrt((1 - q) / q) with calm probability q (Cantelli's bound, met with equality)."""
    for q in (0.05, 0.1, 0.15):
        z = math.sqrt((1 - q) / q)
        assert _chat_turn_calm_tail([q], z) == pytest.approx(q)


def test_bar_table() -> None:
    bars = csb.CURIOSITY_SEED_BAR_BY_NODE_ID
    for nid in (csb.ROOM_NODE_ID, csb.BODY_NODE_ID, csb.ROOM_JOINT_NODE_ID, csb.BODY_JOINT_NODE_ID):
        assert bars[nid] == pytest.approx(FORECAST_BAR)
    assert bars[csb.CHAT_TURN_NODE_ID] == pytest.approx(CHAT_BAR)
    # camera nodes keep the pre-#3821 bar until Patch 3's scene-change seed
    assert bars[csb.PERCEPTION_NODE_ID] == bars[csb.PERCEPTION_WALKWAY_NODE_ID] == 0.55
    assert csb.seed_bar("node:other", 0.55) == 0.55


def test_room_joint_seeds_at_its_cut_and_not_below() -> None:
    assert csb.ROOM_JOINT_NODE_ID in _seeded(_node(csb.ROOM_JOINT_NODE_ID, 0.67))
    assert _seeded(_node(csb.ROOM_JOINT_NODE_ID, 0.64)) == {}


def test_chat_turn_seeds_at_its_cut_and_not_below() -> None:
    assert csb.CHAT_TURN_NODE_ID in _seeded(_node(csb.CHAT_TURN_NODE_ID, 0.81))
    # 0.75: a q ~ 0.15 'asks a question' turn; common under calm, no seed
    assert _seeded(_node(csb.CHAT_TURN_NODE_ID, 0.75)) == {}


@pytest.mark.parametrize("node_id", [csb.ROOM_JOINT_NODE_ID, csb.CHAT_TURN_NODE_ID, csb.BODY_JOINT_NODE_ID,
                                     csb.ROOM_NODE_ID, csb.BODY_NODE_ID])
@pytest.mark.parametrize("calm", [0.0, 0.2, 0.55])
def test_calm_values_do_not_seed(node_id: str, calm: float) -> None:
    """Calm windows rest at 0; 0.55 (old bar, z 1.65) is now below every forecast/chat bar."""
    assert _seeded(_node(node_id, calm)) == {}


def test_camera_nodes_keep_old_bar_and_strength() -> None:
    got = _seeded(_node(csb.PERCEPTION_NODE_ID, 0.56), _node(csb.PERCEPTION_WALKWAY_NODE_ID, 0.54))
    assert set(got) == {csb.PERCEPTION_NODE_ID}
    assert got[csb.PERCEPTION_NODE_ID] == pytest.approx(0.56)  # identity at a 0.55 bar


def test_strength_reads_055_at_own_bar_and_invokes() -> None:
    got = _seeded(_node(csb.ROOM_JOINT_NODE_ID, FORECAST_BAR + 1e-9), _node(csb.CHAT_TURN_NODE_ID, 1.0))
    assert got[csb.ROOM_JOINT_NODE_ID] == pytest.approx(0.55)
    assert got[csb.CHAT_TURN_NODE_ID] == pytest.approx(1.0)
    # the frontier evaluator invokes only at strength >= 0.5 (frontier_curiosity.py _decide)
    assert min(got.values()) >= 0.5


def test_staleness_decay_still_applies_before_the_bar() -> None:
    # raw 0.8 at 600 s old decays to 0.533 (< 2/3): no seed
    assert _seeded(_node(csb.BODY_JOINT_NODE_ID, 0.8, age_s=600)) == {}
    assert csb.BODY_JOINT_NODE_ID in _seeded(_node(csb.BODY_JOINT_NODE_ID, 0.8, age_s=60))


@pytest.mark.parametrize(
    "node_id",
    ["node:substrate.execution", "node:substrate.biometrics", "node:substrate.chat",  # retired
     "node:substrate.cabinet", "node:substrate.strain", "node:substrate.street"],     # alarms / fenced
)
def test_retired_and_alarm_nodes_never_seed(node_id: str) -> None:
    assert _seeded(_node(node_id, 1.0)) == {}


def test_override_table_from_config() -> None:
    cfg = EndogenousCuriosityConfig(enabled=True, seed_bar_by_node_id={})
    out = endogenous_curiosity_candidates(nodes=[_node(csb.ROOM_JOINT_NODE_ID, 0.6)], config=cfg, now=NOW)
    assert [c.focal_node_refs[0] for c in out] == [csb.ROOM_JOINT_NODE_ID]  # falls back to 0.55
    assert out[0].signal_strength == pytest.approx(0.6)
