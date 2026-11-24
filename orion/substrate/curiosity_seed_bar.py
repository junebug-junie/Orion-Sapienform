"""Per-node prediction-error seed bars for endogenous curiosity. Pure, stdlib only.

Issue #3821. Curiosity used one bar (``min_prediction_error`` 0.55) for every node. Each live seedable
node now has its own bar, set on that node's own calm scale so that under calm (forecasts calibrated,
nothing unusual) it crosses at a stated rate of about 5 % or less of its writes. The bars come from
the math of each node's producer, not from a percentile of its history (a percentile always fires).

The forecast nodes. ``node:prediction.room`` / ``.body`` (orion/substrate/prediction_surprise.py) and
``node:prediction.room_joint`` / ``.body_joint`` (orion/substrate/prediction_joint_surprise.py) all
store ``clip(z / 3, 0, 1)``, where ``z = sum(excess) / sqrt(sum(calm_var))`` over the outcomes of
the last 30 minutes. Each outcome's excess is ``-ln p`` minus its calm mean. For a calibrated forecaster
p is uniform, so ``-ln p`` is Exp(1), and the window sum is Gamma(n, 1). The Gamma tail is skewed to
the right, so the old normal cut (z >= 1.645, value 0.55) is not p 0.05: it is p 0.065 to 0.071 for
n = 1 to 8. The exact 95th percentile of a standardized Gamma(n) is 1.996 at n = 1 and falls with n
(1.842 at n = 6), so for the two joint nodes z >= 2.0 has a calm tail <= 0.05 at every window size.
Bar: 2.0 / 3. The room / body nodes floor p at 0.05, which caps one outcome's excess; with the floor
the calm tail at z >= 2.0 is <= 0.05 for n >= 2 (3.9 % at n = 2, 3.3 % at n = 6) but 7.3 % for a
one-outcome window. Their windows normally hold 2 to 6 outcomes (prediction_surprise.py), so the same
bar is kept; a one-outcome window crosses a little more often than 5 %.

The chat node. ``node:prediction.chat_turn`` (orion/substrate/prediction_turn_surprise.py) uses the
same z over every turn settled in the last 30 minutes, but each turn is the sum of two yes/no bets, so
z takes only a few values. The rates below are for a one-turn window, the worst case; a window
holding two turns (the median in the 12-day turn replay) crosses less often (3.6 % at this bar,
pairing adjacent live turns). One bet whose less
likely side (calibrated probability q) happens reads ``z = sqrt((1 - q) / q)``. A two-point variable
reaches z >= k with calm probability up to ``1 / (1 + k^2)`` (Cantelli's bound, tight here), so no
z cut below 4.36 guarantees p <= 0.05, and the node clips at z = 3. The bar is therefore set from the
exact calm tail of the real bets: enumerate each turn's outcomes under its own calibrated q, and take
the calm crossing rate. On the 71 settled chat turns of 2026-11-17 .. 2026-11-24 the mean calm rate is
8.5 % at the old 0.55, 5.0 % at 0.75, 4.6 % at 0.80. Bar: 0.80 (z >= 2.4). The 5 % is a mean over
turns, not a per-turn guarantee: a turn whose "asks a question" bet sits at q 0.084 crosses with calm
probability 0.084 at any bar below 0.95. The rate depends on the mix of stated probabilities, so the
fixture test (orion/substrate/tests/test_curiosity_seed_bar.py) pins it and the replay eval re-measures it
(services/orion-substrate-runtime/evals/run_curiosity_seed_bar_replay_eval.py).

Camera nodes keep the old bar. ``node:substrate.perception`` / ``perception_walkway`` store a clipped
EWMA z of embedding change, not a forecast miss. Patch 3 of #3781 replaces their seed with a scene-change
seed, so they stay at 0.55 until then.

Any other node uses ``EndogenousCuriosityConfig.min_prediction_error`` (0.55). Retired and alarm nodes
never seed, whatever their bar (endogenous_curiosity.py checks them first).

Signal strength. The frontier evaluator invokes only when the top signal's strength is >= 0.5, and it
ranks seeds of every source by strength. A crossing therefore maps onto one shared scale:
``0.55`` at the node's own bar, ``1.0`` at its ceiling (``seed_strength``). For a 0.55 bar this is the
identity, so the camera nodes rank exactly as before. Ranking change: a node with a higher bar now
ranks below its raw value (a forecast reading of 0.8 has strength 0.73; a chat reading of 0.9 has
0.775), so repair-pressure and open-loop seeds, which keep their raw strength, win budget slots they
used to lose to it.

Rollback: delete an entry (that node falls back to 0.55), or empty the dict.
"""
from __future__ import annotations

from typing import Mapping

# Ids are literals so this module stays import-light; tests check them against the producers.
ROOM_NODE_ID = "node:prediction.room"
BODY_NODE_ID = "node:prediction.body"
ROOM_JOINT_NODE_ID = "node:prediction.room_joint"
BODY_JOINT_NODE_ID = "node:prediction.body_joint"
CHAT_TURN_NODE_ID = "node:prediction.chat_turn"
PERCEPTION_NODE_ID = "node:substrate.perception"
PERCEPTION_WALKWAY_NODE_ID = "node:substrate.perception_walkway"

# Every node stores clip(z / Z_FULL_SCALE, 0, 1).
Z_FULL_SCALE = 3.0
# Standardized Gamma(n) 95th percentile is <= this for every n >= 1 (1.996 at n = 1).
FORECAST_Z_CUT = 2.0
# Chat turns: exact calm crossing rate of the live bets <= 5 % (4.6 % on 2026-11-17..24).
CHAT_TURN_Z_CUT = 2.4
# Pre-#3821 bar; camera nodes and unlisted nodes keep it.
LEGACY_BAR = 0.55
# Strength a crossing gets exactly at its node's bar.
STRENGTH_AT_BAR = 0.55

CURIOSITY_SEED_BAR_BY_NODE_ID: dict[str, float] = {
    ROOM_NODE_ID: FORECAST_Z_CUT / Z_FULL_SCALE,
    BODY_NODE_ID: FORECAST_Z_CUT / Z_FULL_SCALE,
    ROOM_JOINT_NODE_ID: FORECAST_Z_CUT / Z_FULL_SCALE,
    BODY_JOINT_NODE_ID: FORECAST_Z_CUT / Z_FULL_SCALE,
    CHAT_TURN_NODE_ID: CHAT_TURN_Z_CUT / Z_FULL_SCALE,
    PERCEPTION_NODE_ID: LEGACY_BAR,
    PERCEPTION_WALKWAY_NODE_ID: LEGACY_BAR,
}


def seed_bar(node_id: str, default: float, bars: Mapping[str, float] | None = None) -> float:
    """The node's own seed bar, or ``default`` for a node with no entry."""
    table = CURIOSITY_SEED_BAR_BY_NODE_ID if bars is None else bars
    return float(table.get(str(node_id or ""), default))


def seed_strength(error: float, bar: float) -> float:
    """Map a crossing (``error >= bar``) onto the shared strength scale: STRENGTH_AT_BAR at the bar,
    1.0 at the ceiling. Identity when the bar is STRENGTH_AT_BAR."""
    if bar >= 1.0:
        return 1.0
    frac = (float(error) - bar) / (1.0 - bar)
    return max(0.0, min(1.0, STRENGTH_AT_BAR + (1.0 - STRENGTH_AT_BAR) * frac))
