"""page:juniper on the grounding capsule (core memory pages Phase 1a)."""
from __future__ import annotations

import pytest

from app import grounding_capsule as gc
from app.grounding_capsule import assemble_stance_grounding, build_grounding_capsule, core_pages_for_ctx
from app.settings import Settings

PAYLOAD = {
    "concept_id": "page:juniper",
    "content": "Juniper lives in Millbrook, Vermont [episode_memory:a1b2]. She uses she/her [person_fact:f00d].",
    "version": 2,
    "evidence_refs": ["episode_memory:a1b2", "person_fact:f00d"],
    "created_at": "2026-11-15T08:10:00+00:00",
}
BASE = {
    "orion_identity_summary": ["I am Oríon."],
    "juniper_relationship_summary": ["co-architect", "Juniper expects continuity and memory."],
    "response_policy_summary": ["Speak plainly."],
    "identity_kernel_source": "configured_yaml",
}


def test_settings_flag_ships_on():
    assert Settings().orion_core_pages_enabled is True
    assert Settings(ORION_CORE_PAGES_ENABLED=False).orion_core_pages_enabled is False


def test_capsule_carries_the_page_with_cites_removed():
    ctx = {**BASE, "orion_core_page_juniper": PAYLOAD}
    capsule = build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=True)
    assert len(capsule.core_pages) == 1
    page = capsule.core_pages[0]
    assert page.concept_id == "page:juniper" and page.version == 2 and page.evidence_count == 2
    assert page.text == "Juniper lives in Millbrook, Vermont. She uses she/her."
    assert page.written_at == "2026-11-15"
    # The identity-yaml bullets stay on the capsule untouched; the renderer decides where they go.
    assert capsule.relationship_summary == BASE["juniper_relationship_summary"]
    assert capsule.context_provenance["orion_core_page_juniper"] == "memory_recall"


def test_flag_off_or_no_page_means_no_core_pages():
    ctx = {**BASE, "orion_core_page_juniper": PAYLOAD}
    assert build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=False).core_pages == []
    assert build_grounding_capsule(dict(BASE), pcr_ran=False, core_pages_enabled=True).core_pages == []
    assert core_pages_for_ctx({"orion_core_page_juniper": {"content": ""}}, enabled=True) == []


def _hydrator(calls: list, payload):
    def fake(ctx, *, lanes=None):
        calls.append(lanes)
        if payload is not None:
            ctx["orion_core_page_juniper"] = payload
    return fake


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_assemble_hydrates_only_the_page_lane_when_enabled(monkeypatch, enabled):
    import app.substrate_felt_state_reader as reader

    calls: list = []
    monkeypatch.setattr(reader, "hydrate_felt_state_ctx", _hydrator(calls, PAYLOAD))
    cfg = Settings(ORION_UNIFIED_GROUNDING_ENABLED=True, CHAT_PCR_ENABLED=False, ORION_CORE_PAGES_ENABLED=enabled)
    ctx = dict(BASE)
    # A provenance map cached earlier in the turn must not hide the page's key.
    gc.context_provenance_for_ctx(ctx)
    capsule = await assemble_stance_grounding(
        bus=None, source=None, ctx=ctx, correlation_id="c-1", recall_cfg={}, stance_step_text="{}",
        exec_settings=cfg)
    assert capsule is not None
    if enabled:
        assert calls == [("orion_core_page_juniper",)]
        assert [p.version for p in capsule.core_pages] == [2]
        assert capsule.context_provenance.get("orion_core_page_juniper") == "memory_recall"
    else:
        assert calls == []
        assert capsule.core_pages == []


@pytest.mark.asyncio
async def test_assemble_survives_a_failing_page_read(monkeypatch):
    import app.substrate_felt_state_reader as reader

    def boom(ctx, *, lanes=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(reader, "hydrate_felt_state_ctx", boom)
    cfg = Settings(ORION_UNIFIED_GROUNDING_ENABLED=True, CHAT_PCR_ENABLED=False, ORION_CORE_PAGES_ENABLED=True)
    capsule = await assemble_stance_grounding(
        bus=None, source=None, ctx=dict(BASE), correlation_id="c-1", recall_cfg={}, stance_step_text="{}",
        exec_settings=cfg)
    assert capsule is not None and capsule.core_pages == []
    assert capsule.relationship_summary == BASE["juniper_relationship_summary"]
