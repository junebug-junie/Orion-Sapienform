"""Regression guard for feedback_two_schema_registries_verify_against_resolve:
a schema present in only one of _REGISTRY/SCHEMA_REGISTRY is half-registered.
resolve() is what the bus actually calls at runtime.
"""
from __future__ import annotations

from orion.schemas import registry


def test_request_schema_resolves():
    model = registry.resolve("AffectGptAssessRequestPayload")
    assert model.__name__ == "AffectGptAssessRequestPayload"


def test_result_schema_resolves_and_has_kind():
    model = registry.resolve("AffectGptAssessResultPayload")
    assert model.__name__ == "AffectGptAssessResultPayload"
    assert "AffectGptAssessResultPayload" in registry.SCHEMA_REGISTRY
    assert registry.SCHEMA_REGISTRY["AffectGptAssessResultPayload"].kind == "affectgpt.assess.result"


def test_juniper_multimodal_affect_resolves_and_has_kind():
    model = registry.resolve("JuniperMultimodalAffectV1")
    assert model.__name__ == "JuniperMultimodalAffectV1"
    assert "JuniperMultimodalAffectV1" in registry.SCHEMA_REGISTRY
    assert (
        registry.SCHEMA_REGISTRY["JuniperMultimodalAffectV1"].kind
        == "affectgpt.juniper_multimodal_affect.v1"
    )
