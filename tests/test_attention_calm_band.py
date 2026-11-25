"""#3917: a sparse, event-written forecast-surprise node competes in world-first attention.

node:prediction.chat_turn writes about 10 times a day. Under the old rule (band against its own
7 days, 200-reading floor) it read insufficient_history forever. Its band now comes from its calm
theory (orion/substrate/calm_band.py), which needs no history.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.attention.world_first import (
    EVENT_ORIENTING_WINDOW_SEC,
    judge_candidate,
    node_candidate,
    node_rank_percentile,
    rank_candidates,
)
from orion.substrate import calm_band as cb
from orion.substrate.prediction_error_magnitude import (
    DEFAULT_MIN_READINGS,
    compute_prediction_error_magnitude,
)

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
CHAT = cb.CHAT_TURN_NODE_ID
DENSE_EMPIRICAL = "node:substrate.cabinet"  # any node with no calm theory


def _sparse_week(per_day: int = 10, value: float = 0.0) -> list[tuple[datetime, float]]:
    """A source writing ``per_day`` times a day for 7 days, all calm (value 0) unless told."""
    step = timedelta(days=1) / per_day
    out, t = [], NOW - timedelta(days=7) + step
    while t < NOW - timedelta(minutes=10):
        out.append((t, value))
        t += step
    return out


def _chat_candidate(value: float, *, age_s: float = 30.0, history=None):
    hist = list(_sparse_week() if history is None else history)
    observed = NOW - timedelta(seconds=age_s)
    hist.append((observed, value))
    mag = compute_prediction_error_magnitude(value=value, observed_at=observed, history=hist, now=NOW,
                                             node_id=CHAT)
    return node_candidate(node_id=CHAT, label="chat turn", magnitude=mag, observed_at=observed, now=NOW,
                          history_values=[v for _, v in hist])


def test_ten_a_day_is_below_the_old_floor() -> None:
    assert len(_sparse_week()) + 1 < DEFAULT_MIN_READINGS


def test_old_rule_starves_the_sparse_source() -> None:
    hist = _sparse_week() + [(NOW, 1.0)]
    mag = compute_prediction_error_magnitude(value=1.0, observed_at=NOW, history=hist, now=NOW)
    assert mag.band == "insufficient_history"
    # Same with an explicit empirical node id: the floor still guards nodes with no calm theory.
    mag = compute_prediction_error_magnitude(value=1.0, observed_at=NOW, history=hist, now=NOW,
                                             node_id=DENSE_EMPIRICAL)
    assert mag.band == "insufficient_history"


def test_a_surprising_turn_on_a_ten_a_day_source_is_eligible() -> None:
    cand = _chat_candidate(1.0)
    assert cand.unusualness.n_readings_7d < DEFAULT_MIN_READINGS
    assert cand.unusualness.band in ("high", "unusual")
    assert cand.unusualness.trend == "insufficient_history"  # trend still needs history
    v = judge_candidate(cand)
    assert v.eligible, v.reason


@pytest.mark.parametrize("value", [0.0, 0.2, 0.45])
def test_a_calm_turn_is_not_eligible(value: float) -> None:
    v = judge_candidate(_chat_candidate(value))
    assert not v.eligible and v.band in ("quiet", "usual")


def test_the_band_is_the_same_whatever_the_history_holds() -> None:
    """Calm theory, not the node's own week: a week full of 1.0s does not make a 1.0 'usual'."""
    a = _chat_candidate(1.0, history=_sparse_week(value=0.0))
    b = _chat_candidate(1.0, history=_sparse_week(value=1.0))
    assert a.unusualness.percentile_now == b.unusualness.percentile_now
    assert a.rank_percentile == b.rank_percentile


def test_a_surprising_turn_wins_against_a_calm_dense_rival() -> None:
    from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

    rival = node_candidate(
        node_id="node:prediction.body", label="body",
        magnitude=compute_prediction_error_magnitude(
            value=0.0, observed_at=NOW, history=[(NOW - timedelta(minutes=5 * i), 0.0) for i in range(2000)],
            now=NOW, node_id="node:prediction.body"),
        observed_at=NOW, now=NOW,
    )
    assert isinstance(rival.unusualness, PredictionErrorMagnitudeV1)
    r = rank_candidates([_chat_candidate(1.0), rival])
    assert r.winner is not None and r.winner.candidate.source_id == CHAT


def test_a_surprising_turn_fades_after_its_orienting_window() -> None:
    late = judge_candidate(_chat_candidate(1.0, age_s=EVENT_ORIENTING_WINDOW_SEC + 5))
    assert not late.eligible


def test_rank_percentile_for_empirical_nodes_is_unchanged() -> None:
    from orion.attention.world_first import midrank_percentile

    vals = [0.0, 0.1, 0.2, 0.5, 0.5]
    assert node_rank_percentile(DENSE_EMPIRICAL, 0.5, vals) == midrank_percentile(0.5, vals)
    assert node_rank_percentile(DENSE_EMPIRICAL, 0.5, []) is None
    assert node_rank_percentile(CHAT, 1.0, []) == cb.calm_percentiles(CHAT, 1.0)[1]


def test_dense_empirical_node_band_unchanged() -> None:
    hist = [(NOW - timedelta(minutes=i), float(i % 10) / 10) for i in range(500)]
    a = compute_prediction_error_magnitude(value=0.95, observed_at=NOW, history=hist, now=NOW)
    b = compute_prediction_error_magnitude(value=0.95, observed_at=NOW, history=hist, now=NOW,
                                           node_id=DENSE_EMPIRICAL)
    assert a == b


def test_history_cache_bands_chat_turn_from_calm_theory() -> None:
    """orion-attention-runtime's path (NodePeHistoryCache -> magnitudes)."""
    rows = [(CHAT, t, v) for t, v in _sparse_week()] + [(CHAT, NOW - timedelta(seconds=20), 1.0)]
    rows += [(DENSE_EMPIRICAL, t, 1.0) for t, _ in _sparse_week()]
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=NOW)
    mags = cache.magnitudes(now=NOW)
    mag, _, mid = mags[CHAT]
    assert mag.band in ("high", "unusual")
    assert mid == pytest.approx(cb.calm_percentiles(CHAT, 1.0)[1])
    assert mags[DENSE_EMPIRICAL][0].band == "insufficient_history"


def test_self_reading_line_names_its_calm_basis() -> None:
    from orion.attention.self_readings import plain_line
    from orion.schemas.self_reading import SelfReadingV1

    def line(node_id: str) -> str:
        return plain_line(SelfReadingV1(
            source_id=node_id, label="x", value_kind="surprise_excess", value=0.9, percentile_now=0.96,
            band="high", score_0_10=10, trend="flat", state="live", unusual=True, ref="pe:x@2026-11-25T12:00:00Z",
        ))

    assert "of calm, well-predicted stretches" in line(CHAT)
    assert "of my last 7 days" in line(DENSE_EMPIRICAL)
