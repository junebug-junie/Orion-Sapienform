from __future__ import annotations

from orion.schemas.registry import _REGISTRY


def test_agent_trace_models_are_registered() -> None:
    assert "AgentTraceToolStatV1" in _REGISTRY
    assert "AgentTraceStepV1" in _REGISTRY
    assert "AgentTraceSummaryV1" in _REGISTRY


def test_registry_and_schema_registry_agree() -> None:
    """Both registries must know the same schema_ids.

    `resolve()` reads `_REGISTRY`; `SCHEMA_REGISTRY` is the richer map used
    elsewhere. They are hand-maintained separately, so it is possible to add a
    schema to one and not the other -- and a unit check against the wrong one
    then passes while the live bus publish fails with
    `ValueError: Unknown schema_id`.

    That happened on 2026-08-21 with VisionSceneInventoryV1: registered in
    SCHEMA_REGISTRY only, verified green against SCHEMA_REGISTRY, and caught
    solely because a real deploy logged the failure. This makes the divergence
    a test failure instead of a runtime one.
    """
    from orion.schemas.registry import _REGISTRY, SCHEMA_REGISTRY

    only_schema_registry = sorted(set(SCHEMA_REGISTRY) - set(_REGISTRY))
    assert not only_schema_registry, (
        "in SCHEMA_REGISTRY but not _REGISTRY, so resolve() will raise "
        f"'Unknown schema_id' on a live publish: {only_schema_registry}"
    )
