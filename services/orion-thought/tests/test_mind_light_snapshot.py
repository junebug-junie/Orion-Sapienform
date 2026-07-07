from __future__ import annotations

from app.mind_enrichment import build_light_mind_request
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _request(*, broadcast=None, stance_inputs=None) -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="I've been thinking about where our work is heading.",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=broadcast,
            broadcast_stale=False,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs=stance_inputs or {"user_message": "..."},
    )


def _broadcast() -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(
        correlation_id="corr-1",
        open_loops=[
            OpenLoopV1(id="ol-1", description="the unresolved deploy decision", why_it_matters="blocks progress"),
            OpenLoopV1(id="ol-2", description="whether the refactor is worth it"),
        ],
    )
    return AttentionBroadcastProjectionV1(
        frame=frame,
        selected_description="the deploy decision is the live thread",
        selected_open_loop_id="ol-1",
        attended_node_ids=["n-1", "n-2"],
    )


def test_snapshot_carries_user_text_and_policy() -> None:
    req = build_light_mind_request(_request(), wall_time_ms=12000, router_profile="default")
    assert req.correlation_id == "corr-1"
    assert req.session_id == "sess-1"
    assert req.trigger == "user_turn"
    assert req.snapshot_inputs["user_text"].startswith("I've been thinking")
    assert req.snapshot_inputs["messages_tail"] == []
    assert req.policy.n_loops_max == 1
    assert req.policy.wall_time_ms_max == 12000
    assert req.policy.router_profile_id == "default"


def test_no_projection_facet_ever() -> None:
    req = build_light_mind_request(_request(broadcast=_broadcast()), wall_time_ms=12000, router_profile="default")
    facets = req.snapshot_inputs.get("facets", {})
    assert "cognitive_projection" not in facets
    assert "cognitive_projection_degraded" not in facets


def test_recall_bundle_folded_only_when_present() -> None:
    without = build_light_mind_request(_request(), wall_time_ms=12000, router_profile="default")
    assert "recall_bundle" not in without.snapshot_inputs.get("facets", {})

    recall = {"fragments": [{"snippet": "we discussed continuity"}], "citations": []}
    with_recall = build_light_mind_request(
        _request(stance_inputs={"user_message": "...", "recall_bundle": recall}),
        wall_time_ms=12000,
        router_profile="default",
    )
    assert with_recall.snapshot_inputs["facets"]["recall_bundle"] == recall


def test_situation_compact_from_broadcast_open_loops() -> None:
    req = build_light_mind_request(_request(broadcast=_broadcast()), wall_time_ms=12000, router_profile="default")
    situation = req.snapshot_inputs["facets"]["situation_compact"]
    text = str(situation)
    assert "deploy decision" in text
    assert "refactor is worth it" in text


def test_no_situation_facet_without_broadcast() -> None:
    req = build_light_mind_request(_request(broadcast=None), wall_time_ms=12000, router_profile="default")
    assert "situation_compact" not in req.snapshot_inputs.get("facets", {})
