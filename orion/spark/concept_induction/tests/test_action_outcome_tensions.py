from __future__ import annotations

from datetime import datetime, timezone

from orion.autonomy.models import ActionOutcomeEmitV1
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.tensions import extract_tensions_from_action_outcome

NOW = datetime(2026, 7, 14, 0, 0, 0, tzinfo=timezone.utc)


def _envelope() -> BaseEnvelope:
    return BaseEnvelope(
        kind="action.outcome.emit.v1",
        source=ServiceRef(name="orion-execution-dispatch-runtime", version="0.1.0"),
        payload={},
    )


def _outcome(**overrides) -> ActionOutcomeEmitV1:
    payload = dict(
        subject="orion",
        action_id="dispatch:1",
        kind="inspect",
        summary="steady state",
        success=True,
        surprise=0.0,
        observed_at=NOW,
    )
    payload.update(overrides)
    return ActionOutcomeEmitV1(**payload)


def test_success_inspect_mints_coherence_relief() -> None:
    events = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(kind="inspect")
    )
    assert len(events) == 1
    event = events[0]
    assert event.kind == "tension.satisfaction.v1"
    assert event.drive_impacts == {"coherence": -0.10}
    assert event.magnitude > 0.0


def test_success_summarize_mints_predictive_relief() -> None:
    events = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(kind="summarize")
    )
    assert events[0].drive_impacts == {"predictive": -0.10}


def test_success_observe_mints_lighter_continuity_relief() -> None:
    events = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(kind="observe")
    )
    assert events[0].drive_impacts == {"continuity": -0.05}
    # Lighter than inspect/summarize's relief weight, matching observe's own
    # lower-stakes design intent from P1.
    assert abs(events[0].drive_impacts["continuity"]) < 0.10


def test_failure_mints_nothing() -> None:
    events = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(success=False)
    )
    assert events == []


def test_unknown_success_mints_nothing() -> None:
    events = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(success=None)
    )
    assert events == []


def test_unmapped_kind_mints_nothing() -> None:
    events = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(kind="noop")
    )
    assert events == []


def test_magnitude_is_positive_direction_lives_in_weight() -> None:
    """magnitude stays schema-non-negative; the negative weight, not a
    negative magnitude, is what carries relief through DriveEngine."""
    event = extract_tensions_from_action_outcome(
        envelope=_envelope(), intake_channel="orion:autonomy:action:outcome", outcome=_outcome(kind="inspect")
    )[0]
    assert event.magnitude >= 0.0
    assert all(w < 0.0 for w in event.drive_impacts.values())
