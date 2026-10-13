"""The letter's readings carry what each read changed, from records; a gap is named, not hidden."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from orion.orion_day import gather
from orion.orion_day.budget import build_llm_view, material_ref_records, reading_change_lines
from orion.orion_day.letter_parts import claim_check
from orion.orion_day.tests import fixtures as fx

NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
MOVE = {"run_id": "ab61e4ccd47b", "completed_at": fx._t(200),
        "per_prior": json.dumps([{"kind": "tested", "before": 0.4, "after": 0.65, "prior_id": "self:serving_stack_gguf"}])}


def _gather(conn, **kw):
    return asyncio.run(gather.gather_orion_day(conn, fx.LETTER_DATE, now=NOW, **kw))


async def _lookup(ids):
    return {"self:serving_stack_gguf": [{
        "prior_id": "self:serving_stack_gguf", "claim": "My serving stack can drop llama.cpp",
        "why": "example.org/gguf says transformers loads GGUF directly", "last_run_id": "ab61e4ccd47b"}]}


def test_a_reading_carries_its_accepted_link():
    material = _gather(fx.FakeConn())
    [r] = material.readings
    [f] = r.findings
    assert f.kind == "link_made" and f.projected is True
    assert f.headline == "Link made: GGUF quantization is now loadable by the transformers library."
    assert material.sources["reading_findings"].status == "ok"
    view = build_llm_view(material).digest_md
    assert "What it changed: Link made: GGUF quantization" in view


def test_a_moved_belief_needs_the_graph_and_says_so_without_it():
    with_graph = _gather(fx.FakeConn(prior_moves=[MOVE]), prior_lookup=_lookup)
    kinds = [f.kind for f in with_graph.readings[0].findings]
    assert kinds == ["link_made", "belief_moved"]
    assert with_graph.readings[0].findings[1].headline == "Belief moved (0.40 -> 0.65): My serving stack can drop llama.cpp"

    no_graph = _gather(fx.FakeConn(read_links=[], prior_moves=[MOVE]))
    assert no_graph.readings[0].findings == []
    assert no_graph.sources["reading_beliefs"].error == "no_prior_lookup"
    assert reading_change_lines(no_graph.readings[0], no_graph) == [
        "no link accepted; beliefs not checked (no_prior_lookup)"]


def test_nothing_changed_is_one_honest_line():
    material = _gather(fx.FakeConn(read_links=[]))
    assert reading_change_lines(material.readings[0], material) == ["no link accepted, no belief moved"]


def test_a_failed_findings_read_never_drops_the_reading():
    material = _gather(fx.FakeConn(fail={"reading_findings"}))
    assert len(material.readings) == 1 and material.readings[0].findings == []
    assert material.sources["reading_findings"].status == "error"
    assert reading_change_lines(material.readings[0], material) == ["not checked (the findings read failed)"]


def test_finding_numbers_are_checkable_against_the_reading_record():
    material = _gather(fx.FakeConn(prior_moves=[MOVE]), prior_lookup=_lookup)
    assert "reading:reading:07d283a9" in material_ref_records(material)
    [token] = [t for t in claim_check("Confidence moved to 0.65 today.", material) if t.token == "0.65"]
    assert token.found and "reading:reading:07d283a9" in token.found_in


def test_letter_material_from_before_findings_still_validates():
    from orion.schemas.orion_day import OrionDayMaterialV1

    material = _gather(fx.FakeConn())
    old = material.model_dump(mode="json")
    for r in old["readings"]:
        r.pop("findings"), r.pop("request_id"), r.pop("article_summary")
    assert OrionDayMaterialV1.model_validate(old).readings[0].findings == []
    assert material.window_end - material.window_start == timedelta(days=1)
