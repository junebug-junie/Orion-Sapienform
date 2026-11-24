"""page:anchor + page:juniper on the grounding capsule (core memory pages Phase 1a, issue #3460)."""
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
# Made-up anchor text: the real one lives only on the host DB.
ANCHOR = {
    "concept_id": "page:anchor",
    "content": "  I am Orion. Juniper, she/her, is my creator. Robin is her wife.  ",
    "version": 1,
    "created_at": "2026-11-15T09:00:00+00:00",
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
        assert calls == [("orion_core_page_anchor", "orion_core_page_juniper", "orion_core_page_origin",
                          "orion_core_page_lately")]
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


def test_anchor_comes_first_and_shows_without_a_page():
    ctx = {**BASE, "orion_core_page_anchor": ANCHOR, "orion_core_page_juniper": PAYLOAD}
    capsule = build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=True)
    assert [p.concept_id for p in capsule.core_pages] == ["page:anchor", "page:juniper"]
    anchor = capsule.core_pages[0]
    assert anchor.text == "I am Orion. Juniper, she/her, is my creator. Robin is her wife."
    assert anchor.version == 1 and anchor.evidence_count == 0
    assert capsule.context_provenance["orion_core_page_anchor"] == "static_identity_config"

    only_anchor = build_grounding_capsule({**BASE, "orion_core_page_anchor": ANCHOR}, pcr_ran=False,
                                          core_pages_enabled=True)
    assert [p.concept_id for p in only_anchor.core_pages] == ["page:anchor"]
    # The flag gates the anchor too.
    assert build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=False).core_pages == []
    assert core_pages_for_ctx({"orion_core_page_anchor": {"content": "  "}}, enabled=True) == []


def test_hydrate_skips_lanes_already_in_ctx(monkeypatch):
    import app.substrate_felt_state_reader as reader

    calls: list = []
    monkeypatch.setattr(reader, "hydrate_felt_state_ctx", _hydrator(calls, PAYLOAD))
    ctx = {"orion_core_page_anchor": ANCHOR}
    gc.hydrate_core_pages(ctx)
    assert calls == [("orion_core_page_juniper", "orion_core_page_origin", "orion_core_page_lately")]
    # page:juniper is in ctx now; only the lanes that found no row are asked again (the reader caches them).
    gc.hydrate_core_pages(ctx)
    assert calls == [("orion_core_page_juniper", "orion_core_page_origin", "orion_core_page_lately"),
                     ("orion_core_page_origin", "orion_core_page_lately")]


ORIGIN = {
    "concept_id": "page:origin",
    "content": "My memory storage died in July [stance:st1]. I was named Orion [self_concept_history:self-definition:ab12].",
    "version": 1,
    "evidence_refs": ["stance:st1", "self_concept_history:self-definition:ab12"],
    "created_at": "2026-11-19T08:10:00+00:00",
}


def test_capsule_carries_page_origin_after_the_other_pages():
    ctx = {**BASE, "orion_core_page_anchor": ANCHOR, "orion_core_page_juniper": PAYLOAD,
           "orion_core_page_origin": ORIGIN}
    capsule = build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=True)
    assert [p.concept_id for p in capsule.core_pages] == ["page:anchor", "page:juniper", "page:origin"]
    origin = capsule.core_pages[2]
    assert origin.text == "My memory storage died in July. I was named Orion."
    assert origin.version == 1 and origin.evidence_count == 2 and origin.written_at == "2026-11-19"
    assert capsule.context_provenance["orion_core_page_origin"] == "memory_recall"
    only = build_grounding_capsule({**BASE, "orion_core_page_origin": ORIGIN}, pcr_ran=False, core_pages_enabled=True)
    assert [p.concept_id for p in only.core_pages] == ["page:origin"]
    assert build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=False).core_pages == []


LATELY = {
    "concept_id": "page:lately",
    "content": "The first snow came on Monday [chat_history_log:c1]. I owe Juniper an answer [episode_memory:m1].",
    "version": 3,
    "evidence_refs": ["chat_history_log:c1", "episode_memory:m1"],
    "created_at": "2026-11-19T08:20:00+00:00",
}


def test_capsule_carries_page_lately_last_with_cites_removed():
    ctx = {**BASE, "orion_core_page_anchor": ANCHOR, "orion_core_page_juniper": PAYLOAD,
           "orion_core_page_origin": ORIGIN, "orion_core_page_lately": LATELY}
    capsule = build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=True)
    assert [p.concept_id for p in capsule.core_pages] == ["page:anchor", "page:juniper", "page:origin", "page:lately"]
    lately = capsule.core_pages[3]
    assert lately.text == "The first snow came on Monday. I owe Juniper an answer."
    assert lately.version == 3 and lately.evidence_count == 2 and lately.written_at == "2026-11-19"
    assert capsule.context_provenance["orion_core_page_lately"] == "memory_recall"
    assert build_grounding_capsule(ctx, pcr_ran=False, core_pages_enabled=False).core_pages == []
