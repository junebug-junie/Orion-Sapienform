from __future__ import annotations

from orion.autonomy.models import AutonomyStateV1
from orion.autonomy.repository import (
    AutonomyLookupV1,
    _drives_facet_ok,
    select_preferred_autonomy_lookup,
)


def test_select_preferred_autonomy_lookup_falls_back_to_relationship_drives() -> None:
    orion_state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        source="graph",
    )
    # dominant_drive/drive_pressures removed from AutonomyStateV1 2026-07-30
    # (chore/delete-orion-drives Wave 2a); _drives_facet_ok now relies solely
    # on subquery_diagnostics["drives"]["row_count"] (set on the enclosing
    # AutonomyLookupV1 below), not on state content.
    relationship_state = AutonomyStateV1(
        subject="relationship",
        model_layer="relationship-model",
        entity_id="relationship:orion|juniper",
        source="graph",
    )
    by_subject = {
        "orion": AutonomyLookupV1(
            subject="orion",
            state=orion_state,
            availability="degraded",
            unavailable_reason="timeout",
            subquery_diagnostics={"drives": {"status": "timeout", "row_count": 0}},
        ),
        "relationship": AutonomyLookupV1(
            subject="relationship",
            state=relationship_state,
            availability="available",
            subquery_diagnostics={"drives": {"status": "ok", "row_count": 12}},
        ),
    }
    selected = select_preferred_autonomy_lookup(by_subject)
    assert selected.selected_subject == "relationship"
    assert selected.contextual_fallback is True
    assert selected.lookup is not None
    assert selected.lookup.state is not None
    assert selected.lookup.state.subject == "relationship"


def test_drives_facet_ok_with_row_count_only() -> None:
    lookup = AutonomyLookupV1(
        subject="relationship",
        state=None,
        availability="available",
        subquery_diagnostics={"drives": {"status": "ok", "row_count": 5}},
    )
    assert _drives_facet_ok(lookup) is True


def test_select_preferred_skips_orion_when_drives_deferred() -> None:
    orion_state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        source="graph",
    )
    # dominant_drive/drive_pressures removed from AutonomyStateV1 2026-07-30
    # (chore/delete-orion-drives Wave 2a); _drives_facet_ok now relies solely
    # on subquery_diagnostics["drives"]["row_count"] (set on the enclosing
    # AutonomyLookupV1 below), not on state content.
    relationship_state = AutonomyStateV1(
        subject="relationship",
        model_layer="relationship-model",
        entity_id="relationship:orion|juniper",
        source="graph",
    )
    by_subject = {
        "orion": AutonomyLookupV1(
            subject="orion",
            state=orion_state,
            availability="available",
            subquery_diagnostics={"drives": {"status": "deferred", "row_count": 0}},
        ),
        "relationship": AutonomyLookupV1(
            subject="relationship",
            state=relationship_state,
            availability="available",
            subquery_diagnostics={"drives": {"status": "ok", "row_count": 3}},
        ),
    }
    selected = select_preferred_autonomy_lookup(by_subject)
    assert selected.selected_subject == "relationship"
    assert selected.contextual_fallback is True


def test_select_preferred_skips_orion_identity_only_when_drives_timeout() -> None:
    """Partial Orion state (identity/goals) must not win over relationship when Orion drives timed out."""
    orion_state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        goal_headlines=[],
        source="graph",
    )
    # dominant_drive/drive_pressures removed from AutonomyStateV1 2026-07-30
    # (chore/delete-orion-drives Wave 2a); _drives_facet_ok now relies solely
    # on subquery_diagnostics["drives"]["row_count"] (set on the enclosing
    # AutonomyLookupV1 below), not on state content.
    relationship_state = AutonomyStateV1(
        subject="relationship",
        model_layer="relationship-model",
        entity_id="relationship:orion|juniper",
        source="graph",
    )
    by_subject = {
        "orion": AutonomyLookupV1(
            subject="orion",
            state=orion_state,
            availability="degraded",
            subquery_diagnostics={
                "identity": {"status": "ok", "row_count": 1},
                "drives": {"status": "timeout", "row_count": 0},
                "goals": {"status": "empty", "row_count": 0},
            },
        ),
        "relationship": AutonomyLookupV1(
            subject="relationship",
            state=relationship_state,
            availability="available",
            subquery_diagnostics={"drives": {"status": "ok", "row_count": 5}},
        ),
    }
    selected = select_preferred_autonomy_lookup(by_subject)
    assert selected.selected_subject == "relationship"
    assert selected.contextual_fallback is True
