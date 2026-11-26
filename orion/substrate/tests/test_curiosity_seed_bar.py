"""Per-node curiosity seed bars (#3821): each live node crosses at its own calibrated cut."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.substrate import calm_band as cb
from orion.substrate import curiosity_seed_bar as csb
from orion.substrate.endogenous_curiosity import (
    EndogenousCuriosityConfig,
    endogenous_curiosity_candidates,
)

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
    """P(Gamma(n, 1) >= x) (orion/substrate/calm_band.py, the one copy of this math)."""
    return cb.gamma_tail(n, x)


def test_floored_room_body_cut() -> None:
    """Floored p (room / body): the same z 2.0 cut is <= 5 % for windows of 2+ outcomes and
    ~7.3 % for a one-outcome window (exact: U <= exp(-(0.95 + 2 sd)))."""
    from orion.prediction.surprisal import calm_excess_variance

    assert cb.calm_excess_variance(cb.P_FLOOR) == pytest.approx(calm_excess_variance(1.0 - 0.95))
    sd = math.sqrt(calm_excess_variance(1.0 - 0.95))
    assert math.exp(-(0.95 + csb.FORECAST_Z_CUT * sd)) == pytest.approx(0.073, abs=0.002)
    assert cb.standardized_floored_tail(1, csb.FORECAST_Z_CUT) == pytest.approx(0.073, abs=0.002)
    for n in (2, 3, 6):
        assert cb.standardized_floored_tail(n, csb.FORECAST_Z_CUT) <= 0.05


@pytest.mark.parametrize("n", range(1, 13))
def test_forecast_cut_is_calm_p_005_at_every_window_size(n: int) -> None:
    """Joint nodes (unfloored conformal p): excess per outcome is Exp(1) - 1, so the window z is a
    standardized Gamma(n). z >= 2.0 has tail <= 0.05 for every n; the old 1.645 (value 0.55) does not."""
    tail_new = _gamma_tail(n, n + csb.FORECAST_Z_CUT * math.sqrt(n))
    tail_old = _gamma_tail(n, n + 1.645 * math.sqrt(n))
    assert tail_new <= 0.05
    assert tail_old > 0.06


def _chat_turn_calm_tail(qs, zcut: float) -> float:
    """Exact calm P(turn z >= zcut) for one turn's bets (calm_band.bet_window_tail)."""
    return cb.bet_window_tail(qs, zcut)


def test_chat_cut_mean_calm_rate_on_live_bets() -> None:
    """The 71 settled chat turns of 2026-11-17..24 (their calibrated q's only): mean exact calm
    crossing rate per turn is <= 5 % at the chat bar and > 5 % at the old 0.55."""
    turns = cb.chat_turn_q_mix()
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
    # camera scene change (PE retire Patch 3): 2.0 nats, calm crossing e^-3 = 5.0 %
    assert bars[csb.PERCEPTION_NODE_ID] == bars[csb.PERCEPTION_WALKWAY_NODE_ID] == 2.0
    assert math.exp(-(2.0 + 1.0)) == pytest.approx(0.0498, abs=1e-4)
    assert csb.seed_bar("node:other", 0.55) == 0.55


def test_room_joint_seeds_at_its_cut_and_not_below() -> None:
    assert csb.ROOM_JOINT_NODE_ID in _seeded(_node(csb.ROOM_JOINT_NODE_ID, 0.67))
    assert _seeded(_node(csb.ROOM_JOINT_NODE_ID, 0.64)) == {}


def test_chat_turn_seeds_at_its_cut_and_not_below() -> None:
    assert csb.CHAT_TURN_NODE_ID in _seeded(_node(csb.CHAT_TURN_NODE_ID, 0.81))
    # 0.75 is z 2.25: a lone rare-side bet at q ~ 0.165; no seed
    assert _seeded(_node(csb.CHAT_TURN_NODE_ID, 0.75)) == {}


def test_chat_bar_single_bet_boundary() -> None:
    """A lone bet's rare side reads z = sqrt((1 - q) / q); the bar z 2.4 sits at q = 1 / (1 + 2.4^2)
    ~ 0.1479. Rarer than that seeds, more common does not."""
    def value(q: float) -> float:
        return min(1.0, math.sqrt((1 - q) / q) / csb.Z_FULL_SCALE)

    assert csb.CHAT_TURN_NODE_ID in _seeded(_node(csb.CHAT_TURN_NODE_ID, value(0.147)))
    assert _seeded(_node(csb.CHAT_TURN_NODE_ID, value(0.149))) == {}


@pytest.mark.parametrize("node_id", [csb.ROOM_JOINT_NODE_ID, csb.CHAT_TURN_NODE_ID, csb.BODY_JOINT_NODE_ID,
                                     csb.ROOM_NODE_ID, csb.BODY_NODE_ID])
@pytest.mark.parametrize("calm", [0.0, 0.2, 0.55])
def test_calm_values_do_not_seed(node_id: str, calm: float) -> None:
    """Calm windows rest at 0; 0.55 (old bar, z 1.65) is now below every forecast/chat bar."""
    assert _seeded(_node(node_id, calm)) == {}


def _camera(node_id: str, nats: float, *, age_s: float = 0.0, stale_pe: float | None = None) -> SimpleNamespace:
    """A camera node as the Patch 3 writer leaves it: scene change in signal_value (nats)."""
    md = {"signal_value": nats, "signal_kind": "scene_change", "signal_scale": "v2"}
    if stale_pe is not None:
        md["prediction_error"] = stale_pe
    return SimpleNamespace(
        node_id=node_id, metadata=md, temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=age_s))
    )


CAMERAS = (csb.PERCEPTION_NODE_ID, csb.PERCEPTION_WALKWAY_NODE_ID)


def test_camera_ceiling_is_the_stage2_cap() -> None:
    from orion.substrate.scene_change import SCENE_CHANGE_CAP_NATS, STAGE2_MAX_VALUES
    from orion.substrate.signal_history import SCENE_CHANGE_CAP_NATS as light_cap

    assert SCENE_CHANGE_CAP_NATS == pytest.approx(math.log(STAGE2_MAX_VALUES + 1) - 1.0)
    assert csb.SCENE_CHANGE_CEILING_NATS == pytest.approx(SCENE_CHANGE_CAP_NATS) == pytest.approx(light_cap)
    assert csb.SCENE_CHANGE_CEILING_NATS == pytest.approx(4.67, abs=0.005)
    for nid in CAMERAS:
        assert csb.seed_ceiling(nid) == pytest.approx(csb.SCENE_CHANGE_CEILING_NATS)
    for nid in (csb.ROOM_NODE_ID, csb.CHAT_TURN_NODE_ID, "node:other"):
        assert csb.seed_ceiling(nid) == 1.0


def test_seed_strength_with_a_ceiling() -> None:
    cap = csb.SCENE_CHANGE_CEILING_NATS
    assert csb.seed_strength(2.0, 2.0, cap) == pytest.approx(0.55)
    for nats in (2.0, 2.5, 3.0, 4.0, 4.66):
        assert csb.seed_strength(nats, 2.0, cap) < 1.0
    assert csb.seed_strength(cap, 2.0, cap) == pytest.approx(1.0)
    # Without the ceiling the old rule (bar >= 1.0 -> 1.0) would flatten every crossing.
    assert csb.seed_strength(2.0, 2.0) == 1.0
    # The default ceiling keeps every forecast node's mapping unchanged.
    assert csb.seed_strength(0.8, FORECAST_BAR) == csb.seed_strength(0.8, FORECAST_BAR, 1.0)


@pytest.mark.parametrize("node_id", CAMERAS)
def test_both_cameras_seed_from_signal_value_at_their_bar(node_id: str) -> None:
    got = _seeded(_camera(node_id, 2.0))
    assert got == {node_id: pytest.approx(0.55)}
    assert _seeded(_camera(node_id, 3.5))[node_id] < 1.0
    assert _seeded(_camera(node_id, 1.99)) == {}
    # nats are never clamped to 0..1 on the way in
    assert _seeded(_camera(node_id, 4.67))[node_id] == pytest.approx(1.0, abs=1e-3)


def test_camera_seed_text_says_scene_change_not_prediction_error() -> None:
    out = endogenous_curiosity_candidates(
        nodes=[_camera(csb.PERCEPTION_WALKWAY_NODE_ID, 2.6)],
        config=EndogenousCuriosityConfig(enabled=True, budget=8), now=NOW,
    )
    (c,) = out
    assert c.evidence_summary == f"unusual scene change on {csb.PERCEPTION_WALKWAY_NODE_ID}"
    assert "prediction error" not in c.evidence_summary
    assert "source:scene_change" in c.notes


def test_a_camera_stale_prediction_error_key_never_seeds() -> None:
    """Kill means kill: the old clipped-z key left on a camera node is never read."""
    assert _seeded(_camera(csb.PERCEPTION_NODE_ID, 0.0, stale_pe=1.0)) == {}


def test_camera_seed_decays_with_the_same_1800s_horizon() -> None:
    # 3.0 nats at 900 s old decays to 1.5 (< 2.0): no seed. At 60 s: 2.9, seeds.
    assert _seeded(_camera(csb.PERCEPTION_NODE_ID, 3.0, age_s=900)) == {}
    assert csb.PERCEPTION_NODE_ID in _seeded(_camera(csb.PERCEPTION_NODE_ID, 3.0, age_s=60))


def test_a_camera_with_no_frames_for_ten_minutes_cannot_seed() -> None:
    """A camera with no frames writes the stale placeholder (0.0 + missing reason) every tick;
    if the tick itself stopped, the last value is 600 s old and decays below the bar."""
    stale = _camera(csb.PERCEPTION_NODE_ID, 0.0)
    stale.metadata["prediction_error_missing_reason"] = "stale"
    assert _seeded(stale) == {}
    # tick stopped 10 min ago on a 2.9-nat reading: 2.9 * (1 - 600/1800) = 1.93 < 2.0
    assert _seeded(_camera(csb.PERCEPTION_NODE_ID, 2.9, age_s=600)) == {}


def test_forecast_bars_unchanged_by_the_camera_patch() -> None:
    bars = csb.CURIOSITY_SEED_BAR_BY_NODE_ID
    assert {n: round(b, 6) for n, b in bars.items() if n not in CAMERAS} == {
        csb.ROOM_NODE_ID: round(FORECAST_BAR, 6), csb.BODY_NODE_ID: round(FORECAST_BAR, 6),
        csb.ROOM_JOINT_NODE_ID: round(FORECAST_BAR, 6), csb.BODY_JOINT_NODE_ID: round(FORECAST_BAR, 6),
        csb.CHAT_TURN_NODE_ID: round(CHAT_BAR, 6),
        # PE retire Patch 4 (2026-11-25): walkway rhythm surprise, chat-turn shape, z >= 1.8 (calm 4.3 %).
        csb.STREET_RHYTHM_NODE_ID: round(1.8 / 3.0, 6),
    }
    assert set(csb.CURIOSITY_SEED_CEILING_BY_NODE_ID) == set(CAMERAS)


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


def test_ranking_change_repair_seed_beats_mapped_forecast_seed() -> None:
    """A forecast reading of 0.8 maps to strength 0.73, so a 0.74 repair seed now outranks it."""
    repair = SimpleNamespace(dimensions={"level": 0.74}, causal_molecule_ids=["m1"], summary="r", confidence=0.6)
    out = endogenous_curiosity_candidates(
        nodes=[_node(csb.BODY_JOINT_NODE_ID, 0.8)], repair_appraisal=repair,
        config=EndogenousCuriosityConfig(enabled=True, budget=1), now=NOW,
    )
    assert [c.notes[-1] for c in out] == ["source:repair_pressure"]
    assert csb.seed_strength(0.8, FORECAST_BAR) == pytest.approx(0.73)


def test_config_table_is_read_only() -> None:
    cfg = EndogenousCuriosityConfig(enabled=True)
    with pytest.raises(TypeError):
        cfg.seed_bar_by_node_id[csb.ROOM_JOINT_NODE_ID] = 0.1  # type: ignore[index]


def test_override_table_from_config() -> None:
    cfg = EndogenousCuriosityConfig(enabled=True, seed_bar_by_node_id={})
    out = endogenous_curiosity_candidates(nodes=[_node(csb.ROOM_JOINT_NODE_ID, 0.6)], config=cfg, now=NOW)
    assert [c.focal_node_refs[0] for c in out] == [csb.ROOM_JOINT_NODE_ID]  # falls back to 0.55
    assert out[0].signal_strength == pytest.approx(0.6)
