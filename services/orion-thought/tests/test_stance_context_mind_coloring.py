from __future__ import annotations

from app.bus_listener import build_stance_react_context
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="where is our work heading?",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "where is our work heading?"},
    )


def test_context_without_coloring_has_no_key() -> None:
    ctx = build_stance_react_context(_request())
    assert "mind_coloring" not in ctx


def test_context_without_coloring_is_baseline() -> None:
    # Passing mind_coloring=None must be byte-identical to omitting it.
    assert build_stance_react_context(_request()) == build_stance_react_context(_request(), mind_coloring=None)


def test_context_with_coloring_adds_key() -> None:
    coloring = {"attention_frontier": [{"label": "x", "summary": "y", "score": 0.5}], "reflective_themes": ["z"]}
    ctx = build_stance_react_context(_request(), mind_coloring=coloring)
    assert ctx["mind_coloring"] == coloring
