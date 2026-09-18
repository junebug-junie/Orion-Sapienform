from __future__ import annotations

from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1
from app.mind_enrichment import build_light_mind_request


def _stance(user_message: str, *, origin: str | None) -> StanceReactRequestV1:
    inputs: dict = {"user_message": user_message}
    if origin is not None:
        inputs["utterance_origin"] = origin
    # Mirror services/orion-thought/tests/test_mind_light_snapshot.py::_request
    return StanceReactRequestV1(
        correlation_id="corr-origin-1",
        session_id="sess-1",
        user_message=user_message,
        association=HubAssociationBundleV1(
            correlation_id="corr-origin-1",
            broadcast=None,
            broadcast_stale=False,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs=inputs,
    )


def test_build_light_mind_request_sets_utterance_origin_field() -> None:
    req = build_light_mind_request(
        _stance("hello from juniper", origin="juniper"),
        wall_time_ms=1000,
        router_profile="default",
    )
    assert req.utterance_origin == "juniper"


def test_situation_compact_includes_origin_prose_for_orion() -> None:
    req = build_light_mind_request(
        _stance("investigate claim X", origin="orion"),
        wall_time_ms=1000,
        router_profile="default",
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    situation = facets.get("situation_compact") or {}
    note = str(situation.get("utterance_origin_note") or "")
    assert "orion" in note.lower()
    assert req.utterance_origin == "orion"


def test_missing_origin_stays_none_and_omits_origin_note() -> None:
    req = build_light_mind_request(
        _stance("legacy caller", origin=None),
        wall_time_ms=1000,
        router_profile="default",
    )
    assert req.utterance_origin is None
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    situation = facets.get("situation_compact") or {}
    assert "utterance_origin_note" not in situation
