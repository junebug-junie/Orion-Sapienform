"""Orion's own self-definition reaches the identity kernel.

`_project_identity_from_beliefs` prepends the `self_definition` snapshot
(orion/substrate/relational/adapters/self_definition_ctx.py) to
`orion_identity_summary` -- outside the 10-line cap, marked, idempotent --
so every consumer of that key (stance brief, chat_general, the grounding
capsule and the harness WHO YOU ARE block) sees Orion's own words with no
template change. When the belief set is degraded the raw felt-state ctx
payload is the fallback, so the line is not silently lost.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.chat_stance import (
    SELF_DEFINITION_MARKER,
    FALLBACK_ORION_IDENTITY_SUMMARY,
    _project_identity_from_beliefs,
)


def _snap(source: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(node_kind="state_snapshot", snapshot_source=source, metadata=metadata)


def _beliefs(*snaps) -> SimpleNamespace:
    orion = SimpleNamespace(concepts=[], tensions=[], goals=[], drives=[], snapshots=list(snaps), events=[], degraded=False, tier_outcomes=[])
    return SimpleNamespace(anchors={"orion": orion})


_AUTHORED = [f"authored line {i}" for i in range(12)]
_DEFINITION = {
    "content": "I am a mesh of services that senses, remembers, dreams and acts.",
    "version": 3,
    "evidence_refs": ["README.md#Project Overview", "dreams: 17 rows"],
    "entry_id": "e1",
    "created_at": "2026-09-08T04:00:00+00:00",
}


def test_definition_is_prepended_outside_the_authored_cap() -> None:
    beliefs = _beliefs(
        _snap("identity_yaml", {"orion_identity_summary": _AUTHORED, "juniper_relationship_summary": ["j"], "response_policy_summary": ["p"]}),
        _snap("self_definition", _DEFINITION),
    )
    out = _project_identity_from_beliefs(beliefs, {})
    lines = out["orion_identity_summary"]
    assert lines[0].startswith(SELF_DEFINITION_MARKER)
    assert "v3" in lines[0] and "2026-09-08" in lines[0] and "2 evidence refs" in lines[0]
    assert _DEFINITION["content"] in lines[0]
    # The authored card keeps its full 10-line cap; the definition is extra.
    assert lines[1:] == _AUTHORED[:10]


def test_projection_is_idempotent_when_the_augmented_list_is_read_back() -> None:
    """The identity_yaml adapter reads ctx, and ctx is updated with the
    projection -- so a cold pull could store the augmented list. The marker
    line must not double up."""
    already = [f"{SELF_DEFINITION_MARKER}, written during my own self-inquiry (v2): older text"] + _AUTHORED[:10]
    beliefs = _beliefs(
        _snap("identity_yaml", {"orion_identity_summary": already, "juniper_relationship_summary": ["j"], "response_policy_summary": ["p"]}),
        _snap("self_definition", _DEFINITION),
    )
    out = _project_identity_from_beliefs(beliefs, {})
    marked = [l for l in out["orion_identity_summary"] if l.startswith(SELF_DEFINITION_MARKER)]
    assert len(marked) == 1
    assert "v3" in marked[0], "the current definition wins over a stale one read back from ctx"


def test_no_definition_leaves_the_kernel_exactly_as_before() -> None:
    beliefs = _beliefs(
        _snap("identity_yaml", {"orion_identity_summary": _AUTHORED, "juniper_relationship_summary": ["j"], "response_policy_summary": ["p"]}),
    )
    out = _project_identity_from_beliefs(beliefs, {})
    assert out["orion_identity_summary"] == _AUTHORED[:10]
    assert out["juniper_relationship_summary"] == ["j"]


def test_empty_definition_content_contributes_nothing() -> None:
    beliefs = _beliefs(
        _snap("identity_yaml", {"orion_identity_summary": _AUTHORED, "juniper_relationship_summary": ["j"], "response_policy_summary": ["p"]}),
        _snap("self_definition", {**_DEFINITION, "content": "   "}),
    )
    out = _project_identity_from_beliefs(beliefs, {})
    assert not any(l.startswith(SELF_DEFINITION_MARKER) for l in out["orion_identity_summary"])


def test_degraded_beliefs_fall_back_to_the_felt_state_ctx_payload() -> None:
    ctx = {"orion_self_definition": _DEFINITION}
    out = _project_identity_from_beliefs(None, ctx)
    assert out["orion_identity_summary"][0].startswith(SELF_DEFINITION_MARKER)
    assert out["orion_identity_summary"][1:] == list(FALLBACK_ORION_IDENTITY_SUMMARY)


def test_long_definitions_are_clipped_for_the_prompt() -> None:
    beliefs = _beliefs(_snap("self_definition", {**_DEFINITION, "content": "x" * 5000}))
    out = _project_identity_from_beliefs(beliefs, {})
    assert len(out["orion_identity_summary"][0]) < 1100
    assert out["orion_identity_summary"][0].endswith("…")


def test_the_producer_is_registered_and_the_adapter_maps_ctx() -> None:
    from app.chat_stance import _build_unification_registry

    ids = [p.producer_id for p in _build_unification_registry().producers]
    assert "self_definition" in ids
    entry = next(p for p in _build_unification_registry().producers if p.producer_id == "self_definition")
    record = entry.adapter_fn({"orion_self_definition": _DEFINITION})
    assert record is not None and record.nodes[0].snapshot_source == "self_definition"
    assert entry.adapter_fn({}) is None
