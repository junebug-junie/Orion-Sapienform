"""World-first attention ranking (spec 2026-10-07 self-calibration, section A).

Each test pins one clause of the rule in orion/attention/world_first.py:
world by default, body only when unusual for itself, absent != calm, explicit
no-winner, polarity from the semantic layer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.attention.world_first import (
    CHAT_MIN_TURNS_7D,
    PERCEPTION_NODE_ID,
    WORLD_CHAT_SOURCE_ID,
    chat_candidate,
    chat_rate_magnitude,
    judge_candidate,
    node_candidate,
    node_prediction_error_semantics,
    rank_candidates,
    world_first_enabled,
)
from orion.schemas.attention_candidate import AttentionCandidateV1
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.schemas.registry import resolve

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def mag(pct: float | None, *, band: str | None = None, age: float = 10.0, value: float = 0.3, n: int = 5000):
    if band is None:
        band = (
            "insufficient_history" if pct is None
            else "quiet" if pct < 0.5 else "usual" if pct < 0.9 else "high" if pct < 0.99 else "unusual"
        )
    return PredictionErrorMagnitudeV1(
        value=value, age_sec=age, percentile_now=pct, n_readings_7d=n, band=band, trend="flat"
    )


def body(node_id: str, pct: float | None, **kw) -> AttentionCandidateV1:
    return node_candidate(
        node_id=node_id, label=node_id, magnitude=mag(pct, **kw), observed_at=NOW, now=NOW
    )


def world(pct: float | None, *, absent: bool = False, source_id: str = WORLD_CHAT_SOURCE_ID, **kw):
    return AttentionCandidateV1(
        candidate_id=f"c:{source_id}",
        source_id=source_id,
        source_kind="external",
        label=source_id,
        unusualness=mag(pct, **kw),
        absent=absent,
        absent_reason="read failed" if absent else None,
    )


def test_schema_is_registered_in_both_registries() -> None:
    from orion.schemas.registry import SCHEMA_REGISTRY

    assert resolve("AttentionCandidateV1") is AttentionCandidateV1
    assert SCHEMA_REGISTRY["AttentionCandidateV1"].model is AttentionCandidateV1


def test_flag_defaults_on_and_false_turns_it_off() -> None:
    assert world_first_enabled({}) is True
    assert world_first_enabled({"ATTENTION_WORLD_FIRST_ENABLED": "false"}) is False


def test_calm_body_and_quiet_world_is_an_explicit_no_winner() -> None:
    ranking = rank_candidates(
        [
            body("node:substrate.biometrics", 0.6),
            body("node:substrate.execution", 0.0),
            world(0.0),
        ]
    )
    assert ranking.no_winner and ranking.winner is None
    trace = ranking.trace()
    assert trace["no_winner"] is True and trace["winner"] is None
    assert len(trace["candidates"]) == 3


@pytest.mark.parametrize("pct", [0.0, 0.49, 0.5, 0.89, 0.8999])
def test_internal_below_high_cannot_enter(pct: float) -> None:
    v = judge_candidate(body("node:substrate.biometrics", pct))
    assert not v.eligible


@pytest.mark.parametrize("pct", [0.9, 0.95, 0.99, 1.0])
def test_internal_at_high_or_unusual_enters(pct: float) -> None:
    v = judge_candidate(body("node:substrate.biometrics", pct))
    assert v.eligible and v.band in {"high", "unusual"}


def test_external_enters_when_fresh_and_busier_than_usual() -> None:
    assert judge_candidate(world(0.5)).eligible
    assert judge_candidate(world(0.93)).eligible


def test_external_at_or_below_its_median_does_not_enter() -> None:
    assert not judge_candidate(world(0.0)).eligible
    assert not judge_candidate(world(0.49)).eligible


def test_world_beats_a_usual_body_but_an_unusual_body_interrupts() -> None:
    usual_body = body("node:substrate.biometrics", 0.85)
    busy_world = world(0.93)
    assert rank_candidates([usual_body, busy_world]).winner.candidate.source_id == WORLD_CHAT_SOURCE_ID
    alarm = body("node:substrate.execution", 0.995)
    assert rank_candidates([alarm, busy_world]).winner.candidate.source_id == "node:substrate.execution"


def test_absent_is_not_calm() -> None:
    """A silent source is never eligible -- and never reads as a calm 0 either:
    its verdict says absent, not quiet."""
    v = judge_candidate(world(0.99, absent=True))
    assert not v.eligible and v.reason.startswith("absent") and v.band == "absent"
    calm = judge_candidate(world(0.0))
    assert calm.band == "quiet" and calm.band != v.band


def test_chat_read_failure_is_absent_not_quiet() -> None:
    cand = chat_candidate(None, now=NOW)
    assert cand.absent and cand.absent_reason
    assert judge_candidate(cand).band == "absent"


def test_perception_absence_comes_from_staleness_not_value() -> None:
    """Perception writes 0.0 while its frames are stale (glossary), so a 0
    reading with stale frames must be ABSENT, not quiet."""
    stale = node_candidate(
        node_id=PERCEPTION_NODE_ID, label="camera", magnitude=mag(0.0, value=0.0),
        observed_at=NOW, now=NOW, absent_reason="camera frames stale",
    )
    assert stale.source_kind == "external" and stale.absent
    fresh_calm = node_candidate(
        node_id=PERCEPTION_NODE_ID, label="camera", magnitude=mag(0.0, value=0.0),
        observed_at=NOW, now=NOW,
    )
    assert not fresh_calm.absent and judge_candidate(fresh_calm).band == "quiet"
    old = node_candidate(
        node_id=PERCEPTION_NODE_ID, label="camera", magnitude=mag(0.95, age=900.0),
        observed_at=NOW, now=NOW,
    )
    assert old.absent and not judge_candidate(old).eligible


def test_perception_is_external_everything_else_internal() -> None:
    assert body(PERCEPTION_NODE_ID, 0.6).source_kind == "external"
    assert body("node:substrate.chat", 0.6).source_kind == "internal"
    assert judge_candidate(body(PERCEPTION_NODE_ID, 0.6)).eligible  # world: usual is enough
    assert not judge_candidate(body("node:substrate.chat", 0.6)).eligible  # body: not


def test_insufficient_history_never_enters() -> None:
    assert not judge_candidate(world(0.99, band="insufficient_history")).eligible
    assert not judge_candidate(body("node:substrate.execution", None)).eligible


def test_stale_internal_reading_cannot_interrupt() -> None:
    v = judge_candidate(body("node:substrate.execution", 0.999, age=1801.0))
    assert not v.eligible and v.reason.startswith("stale")


def test_polarity_and_value_kind_come_from_the_semantic_layer() -> None:
    assert node_prediction_error_semantics("node:substrate.biometrics") == ("level", "higher_is_worse")
    # A trigger has no polarity in the semantic layer; firing is the event.
    assert node_prediction_error_semantics("node:substrate.cabinet") == ("trigger", None)
    assert node_prediction_error_semantics("node:substrate.harness_closure")[0] == "placeholder"


def test_trigger_fires_into_eligibility() -> None:
    assert judge_candidate(body("node:substrate.cabinet", 0.97)).eligible


def test_placeholder_is_never_a_measurement() -> None:
    v = judge_candidate(body("node:substrate.harness_closure", 1.0))
    assert not v.eligible and "placeholder" in v.reason


def test_higher_is_better_inverts_the_bad_direction() -> None:
    good_high = body("node:substrate.biometrics", 0.98).model_copy(update={"polarity": "higher_is_better"})
    assert not judge_candidate(good_high).eligible
    bad_low = body("node:substrate.biometrics", 0.02).model_copy(update={"polarity": "higher_is_better"})
    assert judge_candidate(bad_low).eligible


def test_no_declared_polarity_cannot_interrupt() -> None:
    c = body("node:substrate.biometrics", 0.99).model_copy(update={"polarity": None, "value_kind": "level"})
    v = judge_candidate(c)
    assert not v.eligible and "polarity" in v.reason


def test_rank_is_by_own_percentile_borda_only_breaks_ties() -> None:
    a = body("node:substrate.execution", 0.95)
    b = body("node:substrate.route", 0.95)
    c = body("node:substrate.chat", 0.97)
    r = rank_candidates([a, b, c], tie_break={"node:substrate.route": 0.9, "node:substrate.execution": 0.1})
    assert [v.candidate.source_id for v in r.eligible] == [
        "node:substrate.chat", "node:substrate.route", "node:substrate.execution",
    ]


def _turns(*minutes_ago: float) -> list[datetime]:
    return [NOW - timedelta(minutes=m) for m in minutes_ago]


def test_chat_rate_rests_at_zero_when_nobody_talks() -> None:
    history = _turns(*[60 * 24 * d + 30 for d in range(1, 7)])  # one turn a day
    m = chat_rate_magnitude(history, now=NOW)
    assert m.value == 0.0 and m.percentile_now == 0.0 and m.band == "quiet"


def test_chat_burst_is_unusual_against_its_own_week() -> None:
    history = _turns(*[60 * 24 * d + 30 for d in range(1, 7)], 1, 3, 5, 8)
    m = chat_rate_magnitude(history, now=NOW)
    assert m.value == 4.0 and m.band == "unusual"
    cand = chat_candidate(history, now=NOW)
    assert cand.source_kind == "external" and not cand.absent
    assert judge_candidate(cand).eligible


def test_chat_with_too_little_history_says_so() -> None:
    m = chat_rate_magnitude(_turns(*range(CHAT_MIN_TURNS_7D - 1)), now=NOW)
    assert m.band == "insufficient_history"
    assert not judge_candidate(chat_candidate(_turns(1, 2), now=NOW)).eligible


def test_chat_ignores_turns_after_now_and_older_than_a_week() -> None:
    history = _turns(*[60 * 24 * d + 30 for d in range(1, 7)], 60 * 24 * 9, -5)
    m = chat_rate_magnitude(history, now=NOW)
    assert m.value == 0.0
