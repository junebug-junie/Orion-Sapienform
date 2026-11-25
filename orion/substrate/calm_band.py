"""Calm-theory tails for the forecast-surprise nodes, and the per-node band rule. Pure, stdlib only.

Issue #3917. World-first attention bands a node by the percentile of its current value. For most
nodes that percentile is taken against the node's own last 7 days, and below 200 readings the node
reads ``insufficient_history`` and never competes. ``node:prediction.chat_turn`` writes once per
settled chat turn (57-80 a week), so it never competed. The own-week percentile already normalizes
for how often a source writes; the fixed count floor put the rate bias back.

The forecast-surprise nodes do not need their own week. Each stores ``clip(z / 3, 0, 1)``, where z is
a standardized sum of calibrated excess surprisal, and the calm distribution of z is known from the
producer's math (PR #3826, ``orion/substrate/curiosity_seed_bar.py``). For these nodes the band comes
from the calm-theory percentile of the current value: "how often would a calm, well-predicted window
read lower than this". No history count is needed, so no floor.

The rules (``ATTENTION_BAND_RULE_BY_NODE_ID``):

- ``gamma`` (``node:prediction.room_joint`` / ``.body_joint``, prediction_joint_surprise.py): one
  outcome's -ln p is Exp(1) for a calibrated conformal p, so a window of n outcomes sums to
  Gamma(n, 1) and z = (G - n) / sqrt(n).
- ``floored`` (``node:prediction.room`` / ``.body``, prediction_surprise.py): p is floored at 0.05,
  so each -ln p is min(Exp(1), ln 20). The exact tail of a sum of n capped exponentials is below
  (``floored_sum_tail``), z = (S - 0.95 n) / sqrt(n * 0.698).
- ``chat_bets`` (``node:prediction.chat_turn``, prediction_turn_surprise.py): each turn is the sum of
  two yes/no bets at calibrated q's, so z takes a few values per turn. The tail is the exact calm
  tail averaged over the live mix of q's (``data/chat_turn_calibrated_q_2026-11-24.json``, 71 turns),
  for one-turn and two-turn windows, the larger of the two (the #3826 bar used the one-turn case).
- ``cantelli``: any standardized calm variable (mean 0, variance 1) has P(z >= k) <= 1 / (1 + k^2).
  Distribution-free, so it fits a new calibrated-excess node before its own tail is derived (e.g.
  ``node:prediction.street_rhythm``). Conservative: under it a value reaches "high" only at z >= 3.
- ``empirical``: the own-7-day percentile with the 200-reading floor, unchanged. Every node not in
  the table, and, explicitly, the two cameras: their scene change is calibrated by construction
  (calm tail e^-(v+1)), but they write thousands of rows a week, the walkway's p99 floor was set
  on its empirical mid-rank, and #3874 relies on insufficient_history after a scale change.
  The alarms are triggers and are judged on their own path.

The window size n is not stored with the value, so for ``gamma`` / ``floored`` the tail is the
largest over n = 1 .. ``MAX_WINDOW_OUTCOMES`` and the normal limit: "a calm window of any size reads
at least this high no more often than this". That never overstates how unusual a reading is.

Rollback: delete a node's entry (it returns to the empirical rule and its 200-reading floor).
"""
from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

ROOM_NODE_ID = "node:prediction.room"
BODY_NODE_ID = "node:prediction.body"
ROOM_JOINT_NODE_ID = "node:prediction.room_joint"
BODY_JOINT_NODE_ID = "node:prediction.body_joint"
CHAT_TURN_NODE_ID = "node:prediction.chat_turn"
PERCEPTION_NODE_ID = "node:substrate.perception"
PERCEPTION_WALKWAY_NODE_ID = "node:substrate.perception_walkway"

# Every forecast-surprise node stores clip(z / Z_FULL_SCALE, 0, 1).
Z_FULL_SCALE = 3.0
# orion-prediction's surprise floor (services/orion-prediction/app/surprise.py P_FLOOR).
P_FLOOR = 0.05
# A 30-min window holds at most ~6 five-min and 2 fifteen-min outcomes (prediction_surprise.py);
# 12 leaves margin. Larger windows are covered by the normal limit.
MAX_WINDOW_OUTCOMES = 12

RULE_GAMMA = "gamma"
RULE_FLOORED = "floored"
RULE_CHAT_BETS = "chat_bets"
RULE_CANTELLI = "cantelli"
RULE_EMPIRICAL = "empirical"
CALM_RULES = frozenset({RULE_GAMMA, RULE_FLOORED, RULE_CHAT_BETS, RULE_CANTELLI})

# The one table: how attention bands each node. Unlisted nodes are RULE_EMPIRICAL.
ATTENTION_BAND_RULE_BY_NODE_ID: dict[str, str] = {
    ROOM_NODE_ID: RULE_FLOORED,
    BODY_NODE_ID: RULE_FLOORED,
    ROOM_JOINT_NODE_ID: RULE_GAMMA,
    BODY_JOINT_NODE_ID: RULE_GAMMA,
    CHAT_TURN_NODE_ID: RULE_CHAT_BETS,
    PERCEPTION_NODE_ID: RULE_EMPIRICAL,
    PERCEPTION_WALKWAY_NODE_ID: RULE_EMPIRICAL,
}

CHAT_TURN_Q_MIX_PATH = Path(__file__).resolve().parent / "data" / "chat_turn_calibrated_q_2026-11-24.json"

_EPS = 1e-12
_WARNED: set[str] = set()


def _warn_once(key: str, exc: Exception) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logging.getLogger(__name__).warning("calm_band_%s err=%s", key, exc)


# ---------------------------------------------------------------------------
# Exact calm tails (#3826's math, factored out of its tests)
# ---------------------------------------------------------------------------


def gamma_tail(n: int, x: float) -> float:
    """P(Gamma(n, 1) >= x) for integer n >= 0 (Poisson sum). Gamma(0) is the point mass at 0."""
    if n <= 0:
        return 1.0 if x <= 0.0 else 0.0
    if x <= 0.0:
        return 1.0
    term = math.exp(-x)
    total = term
    for k in range(1, n):
        term *= x / k
        total += term
    return min(1.0, total)


def standardized_gamma_tail(n: int, z: float) -> float:
    """Joint nodes: P((G - n) / sqrt(n) >= z), G ~ Gamma(n, 1)."""
    return gamma_tail(n, n + z * math.sqrt(n))


def calm_excess_variance(p_floor: float = P_FLOOR) -> float:
    """Var[-ln max(p, P)] for p uniform (== prediction_surprise.calm_excess_variance(1 - P))."""
    p = float(p_floor)
    return 2.0 - 2.0 * p * (1.0 + math.log(1.0 / p)) - (1.0 - p) ** 2


def floored_sum_tail(n: int, s: float, p_floor: float = P_FLOOR) -> float:
    """P(sum of n iid min(E, c) >= s), E ~ Exp(1), c = -ln P. Exact.

    One term's law is an atom P at c plus the sub-density e^-x on [0, c) = h(x) - P h(x - c) (h the
    Exp(1) density). With k terms capped and m = n - k not, the m uncapped part integrates to
    sum_j C(m, j) (-P)^j P(Gamma(m) >= u - j c)."""
    c = -math.log(p_floor)
    total = 0.0
    for k in range(n + 1):
        m = n - k
        u = s - k * c
        inner = 0.0
        for j in range(m + 1):
            inner += math.comb(m, j) * (-p_floor) ** j * gamma_tail(m, u - j * c)
        total += math.comb(n, k) * p_floor**k * inner
    return max(0.0, min(1.0, total))


def standardized_floored_tail(n: int, z: float, p_floor: float = P_FLOOR) -> float:
    """Room / body: P((S - (1 - P) n) / sqrt(n var) >= z)."""
    sd = math.sqrt(n * calm_excess_variance(p_floor))
    return floored_sum_tail(n, (1.0 - p_floor) * n + z * sd, p_floor)


def normal_tail(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def bet_window_atoms(qs: Iterable[float]) -> list[tuple[float, float]]:
    """Every outcome of a window of calibrated yes/no bets as (z, calm probability)
    (prediction_turn_surprise.py / turn_surprise.py: excess = -(y - q) logit q, calm var
    q (1 - q) logit(q)^2)."""
    qs = [float(q) for q in qs if q is not None and 0.0 < float(q) < 1.0]
    if not qs:
        return []
    lgs = [math.log(q / (1.0 - q)) for q in qs]
    var = sum(q * (1.0 - q) * lg * lg for q, lg in zip(qs, lgs))
    if var <= 0.0:
        return []
    sd = math.sqrt(var)
    atoms: list[tuple[float, float]] = [(0.0, 1.0)]
    for q, lg in zip(qs, lgs):
        nxt = []
        for e, p in atoms:
            nxt.append((e - (1.0 - q) * lg, p * q))      # y = 1
            nxt.append((e + q * lg, p * (1.0 - q)))       # y = 0
        atoms = nxt
    return [(e / sd, p) for e, p in atoms]


def bet_window_tail(qs: Iterable[float], z: float) -> float:
    """Exact calm P(window z >= z) for one window of bets."""
    return sum(p for zz, p in bet_window_atoms(qs) if zz >= z - _EPS)


# ---------------------------------------------------------------------------
# Per-rule tail of the stored value
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def chat_turn_q_mix() -> tuple[tuple[float, ...], ...]:
    """The calibrated q's of the live chat turns' bets (no ids, no outcomes)."""
    turns = json.loads(CHAT_TURN_Q_MIX_PATH.read_text())["turns"]
    return tuple(tuple(float(q) for q in t if q is not None) for t in turns)


@lru_cache(maxsize=1)
def _chat_mixes() -> tuple[tuple[tuple[float, float], ...], ...]:
    """Sorted (z, weight) atoms of the calm z for a one-turn and a two-turn window, each turn (pair)
    of the live mix weighted equally."""
    turns = chat_turn_q_mix()
    out = []
    for windows in ([list(t) for t in turns], [list(a) + list(b) for a, b in zip(turns, turns[1:])]):
        if not windows:
            continue
        w = 1.0 / len(windows)
        atoms = [(z, p * w) for qs in windows for z, p in bet_window_atoms(qs)]
        out.append(tuple(sorted(atoms)))
    return tuple(out)


def _mix_tail(atoms: Sequence[tuple[float, float]], z: float, strict: bool) -> float:
    cut = z + _EPS if strict else z - _EPS
    return sum(p for zz, p in atoms if zz > cut)


def z_tail(rule: str, z: float, *, strict: bool = False) -> float:
    """Calm P(z >= z) (``strict``: P(z > z)) under ``rule``. Conservative over the window size."""
    if rule == RULE_GAMMA:
        tails = [standardized_gamma_tail(n, z) for n in range(1, MAX_WINDOW_OUTCOMES + 1)]
    elif rule == RULE_FLOORED:
        tails = [standardized_floored_tail(n, z) for n in range(1, MAX_WINDOW_OUTCOMES + 1)]
    elif rule == RULE_CHAT_BETS:
        try:
            mixes = _chat_mixes()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # An unreadable q mix must not cost the whole contest its magnitudes: fall back to the
            # distribution-free bound (conservative: the node reads "high" only at z >= 3).
            _warn_once("chat_turn_q_mix_unreadable", exc)
            return z_tail(RULE_CANTELLI, z, strict=strict)
        return max((_mix_tail(m, z, strict) for m in mixes), default=1.0)
    elif rule == RULE_CANTELLI:
        return 1.0 if z <= 0.0 else 1.0 / (1.0 + z * z)
    else:
        raise ValueError(f"no calm theory for rule {rule!r}")
    # Continuous in z (an atom only where every outcome is capped): >= and > agree.
    return max(max(tails), normal_tail(z))


def _value_tails(rule: str, value: float) -> tuple[float, float]:
    """(P(stored >= value), P(stored > value)) for a stored clip(z / 3, 0, 1)."""
    v = float(value)
    if v <= 0.0:
        ge = 1.0
        gt = 1.0 if v < 0.0 else z_tail(rule, 0.0, strict=True)
        return ge, gt
    if v >= 1.0:
        # The clip: every z >= 3 stores exactly 1.0; nothing stores more.
        return (z_tail(rule, Z_FULL_SCALE), 0.0) if v == 1.0 else (0.0, 0.0)
    z = v * Z_FULL_SCALE
    return z_tail(rule, z), z_tail(rule, z, strict=True)


def band_rule(node_id: str, table: dict[str, str] | None = None) -> str:
    return (ATTENTION_BAND_RULE_BY_NODE_ID if table is None else table).get(str(node_id or ""), RULE_EMPIRICAL)


def has_calm_band(node_id: str, table: dict[str, str] | None = None) -> bool:
    return band_rule(node_id, table) in CALM_RULES


def calm_percentiles(node_id: str, value: float, table: dict[str, str] | None = None) -> tuple[float, float] | None:
    """(strict-below percentile, mid-rank percentile) of ``value`` under the node's calm theory, or
    None for an empirical node or a non-finite value. Strict-below is the band's input (the same
    convention as the empirical percentile); mid-rank is the ranking key (half the atom at the value)."""
    rule = band_rule(node_id, table)
    if rule not in CALM_RULES:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    ge, gt = _value_tails(rule, v)
    ge = max(0.0, min(1.0, ge))
    gt = max(0.0, min(ge, gt))
    return 1.0 - ge, 1.0 - 0.5 * (ge + gt)
