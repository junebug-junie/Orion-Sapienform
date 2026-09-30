"""build_llm_view: deterministic, reveries condensed, everything else full, clipping only
under pressure, and every rendered reference resolves to a material item."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.orion_day import budget
from orion.orion_day.budget import build_llm_view, extract_refs, material_refs
from orion.orion_day.gather import gather_orion_day
from orion.orion_day.tests import fixtures as fx


def _material(**conn_kwargs):
    return asyncio.run(gather_orion_day(fx.FakeConn(**conn_kwargs), fx.LETTER_DATE,
                                        now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)))


def test_same_material_gives_the_same_view_byte_for_byte():
    material = _material()
    a, b = build_llm_view(material), build_llm_view(material)
    assert a.model_dump() == b.model_dump()
    reparsed = type(material).model_validate_json(material.model_dump_json())
    assert build_llm_view(reparsed).digest_md == a.digest_md


def test_reveries_are_condensed_and_every_other_section_is_full():
    material = _material()
    view = build_llm_view(material)
    c = view.condensed
    assert c.reverie_thoughts_total == len(fx.REVERIE_THOUGHTS)
    assert 0 < c.reverie_thoughts_included < c.reverie_thoughts_total
    assert c.reverie_thoughts_hollow_skipped > 0
    assert c.reverie_thoughts_duplicate_skipped >= 1  # th-dup-bbbbbbbb repeats thought 6's opening
    assert c.reverie_thoughts_chain_capped > 0
    assert c.full_text_items_clipped == 0 and c.full_text_clip_chars is None
    # Full texts appear whole.
    assert fx.CURIOSITY_JOURNALS[0]["body"].strip() in view.digest_md
    assert material.readings[0].learned in view.digest_md
    assert fx.SELF_SENSE[0]["answer_text"] in view.digest_md
    assert fx.CHAT_COMPACTOR["body"] in view.digest_md
    assert fx.DREAM_NARRATIVES[0]["narrative"] in view.digest_md
    assert fx.VISUAL_REVERIES[0]["description"] in view.digest_md
    # At most one thought per chain in the sample.
    shown = [r for r in view.included_refs if r.startswith("reverie:")]
    chains = {t.chain_id for t in material.reverie_thoughts if f"reverie:{t.thought_id[:8]}" in shown}
    assert len(shown) == c.reverie_thoughts_included == len(chains)


def test_hollow_thoughts_are_never_sampled():
    material = _material()
    view = build_llm_view(material)
    hollow_texts = {t.interpretation for t in material.reverie_thoughts if t.hollow}
    assert hollow_texts  # the fixture has some
    assert not any(text.strip() in view.digest_md for text in hollow_texts)


def test_digest_stays_within_budget_and_clips_evenly_under_pressure():
    material = _material()
    view = build_llm_view(material, budget_tokens=1200)
    assert view.approx_tokens <= view.budget_tokens
    c = view.condensed
    assert c.full_text_clip_chars is not None and c.full_text_items_clipped >= 1
    assert "[clipped here:" in view.digest_md
    # clipping is in the view only: the material still carries the full body
    assert len(material.curiosity_runs[0].journal_body) > c.full_text_clip_chars


def test_default_budget_holds_for_the_fixture_day():
    view = build_llm_view(_material())
    assert view.approx_tokens <= budget.DEFAULT_BUDGET_TOKENS


def test_every_rendered_reference_resolves_to_material():
    material = _material()
    view = build_llm_view(material)
    assert view.included_refs == extract_refs(view.digest_md)
    assert set(view.included_refs) <= material_refs(material)
    assert "curiosity:ab61e4ccd47b" in view.included_refs
    assert "dream_offered:1" in view.included_refs


def test_hypothesis_ids_never_reach_the_model():
    material = _material()
    view = build_llm_view(material)
    assert material.dream_hypotheses and "dh-1a69e4d982ac" not in view.digest_md
    assert "dream_hypothesis:" not in view.digest_md
    assert fx.DREAM_HYPOTHESES[0]["claim"] in view.digest_md  # the claim itself is shown


def test_body_headings_are_demoted_below_item_headings():
    view = build_llm_view(_material())
    assert "\n# What I read" not in view.digest_md
    assert "#### What I read" in view.digest_md


def test_unreadable_sources_are_named_in_the_view():
    view = build_llm_view(_material(fail={"self_sense"}))
    assert "Sources that could not be read" in view.digest_md
    assert "self_sense" in view.digest_md.split("\n\n", 2)[1]


def test_water_fill_cap_is_the_largest_that_fits():
    assert budget._water_fill_cap([100, 5000, 5000], 3100) == max(1500, budget.MIN_CLIP_CHARS)
    assert budget._water_fill_cap([10, 10], 1000) == budget.MIN_CLIP_CHARS  # floor; nothing that short is clipped
