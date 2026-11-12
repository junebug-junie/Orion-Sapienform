"""Temporal Self patch 4: the letter's arc citations are gated (unknown removed and counted,
uncited day sentences labelled UNVERIFIED, never retried) and stored in sources.arc_citations."""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.orion_day_graph import check_carry_forward, check_note, finish_detail, letter_row  # noqa: E402
from orion.orion_day.budget import build_llm_view, extract_refs, material_ref_records  # noqa: E402
from orion.orion_day.window import orion_day_window  # noqa: E402
from orion.schemas.orion_day import (  # noqa: E402
    ORION_DAY_CARRY_FORWARD_VERB, ORION_DAY_NOTE_VERB, OrionDayLetterSourcesV1, OrionDayMaterialV1,
    OrionDayRunBriefV1,
)
from orion.temporal_self.day_block import render_closed_day_block  # noqa: E402
from orion.temporal_self.tests.patch4_fixtures import day_1009  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_orion_day_graph as base  # noqa: E402

HARNESS = "2ce0b486a356bbd2"
PAD = " I sat with that for a while, and I am not sure what to make of it yet." * 8


def brief_1009(*, with_block: bool = True) -> OrionDayRunBriefV1:
    w = orion_day_window(date(2026, 10, 9))
    material = OrionDayMaterialV1(
        letter_date=date(2026, 10, 9), window_start=w.window_start, window_end=w.window_end,
        gathered_at=w.window_end + timedelta(hours=8),
        temporal_day=render_closed_day_block(day_1009()) if with_block else None,
    )
    return OrionDayRunBriefV1(letter_date=material.letter_date, window_start=w.window_start,
                              window_end=w.window_end, material=material, llm_view=build_llm_view(material))


def test_block_is_in_the_digest_and_its_refs_resolve():
    brief = brief_1009()
    digest = brief.llm_view.digest_md
    assert "## My day as my own chronology recorded it" in digest
    assert "as recorded at close" in digest
    assert f"arc:{HARNESS}" in brief.llm_view.included_refs and "day:2026-10-09" in brief.llm_view.included_refs
    records = material_ref_records(brief.material)
    assert all(ref in records for ref in extract_refs(digest))
    assert records[f"arc:{HARNESS}"]["subject_label"] == "Harness closure prediction error"


def test_no_closed_day_means_no_block_and_no_check():
    brief = brief_1009(with_block=False)
    assert "My day as my own chronology" not in brief.llm_view.digest_md
    note = "Harness closure prediction error held me." + PAD
    assert check_note(note, brief) == {}


def test_note_known_kept_unknown_removed_uncited_unverified():
    brief = brief_1009()
    note = (f"Harness closure prediction error held me for over two hours [arc:{HARNESS}]. "
            "Juniper spoke at 18:08. I also dreamed of a door [arc:deadbeefdeadbeef]." + PAD)
    out = check_note(note, brief)
    assert "[arc:deadbeefdeadbeef]" not in out["note_md"] and f"[arc:{HARNESS}]" in out["note_md"]
    c = out["note_arc_citations"]
    assert c["cited_valid"] == [f"arc:{HARNESS}"] and c["rejected_unknown"] == ["arc:deadbeefdeadbeef"]
    assert c["uncited_day_sentences"] == 1 and c["label"] == "UNVERIFIED"
    assert c["uncited_examples"] == ["Juniper spoke at 18:08."]


def test_carry_forward_counts_refs_after_the_gate():
    brief = brief_1009()
    carry = (f"- [arc:{HARNESS}] Why did harness closure dominate?\n"
             "- [arc:feedfacefeedface] Look at the invented thing.")
    out = check_carry_forward(carry, brief)
    assert "[arc:feedfacefeedface]" not in out["carry_forward_md"]
    assert out["carry_forward_refs"] == {"valid": 1, "unknown": 0}
    assert out["carry_forward_arc_citations"]["rejected_unknown"] == ["arc:feedfacefeedface"]


def test_letter_row_stores_the_checks_in_sources():
    brief = brief_1009()
    note = f"Harness closure prediction error held me [arc:{HARNESS}]." + PAD
    state = {"run_id": "r", "note_md": note, "carry_forward_md": "- x",
             **check_note(note, brief)}
    sources = OrionDayLetterSourcesV1.model_validate(letter_row(state, brief, datetime.now(timezone.utc))["sources"])
    assert sources.arc_citations["note"].label == "verified"
    assert "carry_forward" not in sources.arc_citations
    assert finish_detail({**state, "brief": {}})["arc_citations"] == {
        "note": {"label": "verified", "cited": 1, "rejected": 0, "uncited": 0}}


def test_full_run_never_retries_for_citations_and_persists_them():
    async def scenario():
        world, saver = base.World(), InMemorySaver()
        world.brief = brief_1009()
        world.replies[ORION_DAY_NOTE_VERB] = [
            "Harness closure prediction error held me most of the day [arc:deadbeefdeadbeef]." + PAD]
        world.replies[ORION_DAY_CARRY_FORWARD_VERB] = [f"- [arc:{HARNESS}] Look at harness closure."]
        snap = await base.drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["llm_attempts"] == {"write_note": 1, "write_carry_forward": 1}  # no retry
        row = world.rows[date(2026, 10, 9)]
        assert "[arc:deadbeefdeadbeef]" not in row["note_md"]
        note_check = row["sources"]["arc_citations"]["note"]
        assert note_check["rejected_unknown"] == ["arc:deadbeefdeadbeef"] and note_check["label"] == "UNVERIFIED"
        assert row["sources"]["arc_citations"]["carry_forward"]["label"] == "verified"
    asyncio.run(scenario())


# --- the note prompt asks for the ids (live 10-12/10-13: 20 and 17 arcs shown, 0 cited) ----------------


def _render_note(digest_md: str) -> str:
    from jinja2 import Environment
    template = (ROOT / "orion" / "cognition" / "prompts" / "orion_day_note_v1.j2").read_text()
    return Environment(autoescape=False).from_string(template).render(
        metadata={"orion_day_input": {"letter_date": "2026-10-09", "timezone": "America/Denver",
                                      "digest_md": digest_md}})


def test_note_prompt_asks_for_the_chronology_brackets_before_and_after_the_record():
    digest = brief_1009().llm_view.digest_md
    rendered = _render_note(digest)
    head, tail = rendered.split(digest)
    assert "Cite your chronology" in head
    assert "copy the [arc:<id>] next to that fact into the same sentence" in head
    assert "[arc:<id>] for one arc, [event:<id>] for one act or expectation, [day:<date>] for a count over the whole day" in head
    assert "a sentence naming an attention subject or a time also needs that fact's [arc:<id>]" in head
    assert "marked unverified" in head and "you do not need to cite" not in head.split("Cite your chronology")[1]
    assert "put the [arc:<id>] shown next to that fact in the same sentence" in tail


def test_note_prompt_without_a_block_asks_for_no_ids():
    rendered = _render_note(brief_1009(with_block=False).llm_view.digest_md)
    assert "[arc:" not in rendered and "Cite your chronology" not in rendered
    assert "(next point)" not in rendered and "you do not need to cite the bracketed references" in rendered


def test_a_note_that_follows_the_prompt_is_verified():
    """The reply shape the prompt asks for (the 10-12 sentence, cited) passes the gate as verified."""
    brief = brief_1009()
    note = ("I talked with Juniper at 15:00 [arc:2545de72655edb66]. "
            "Harness closure prediction error held me for 2 h 15 min over 11 arcs [day:2026-10-09]; "
            f"the longest stretch was 39 min [arc:{HARNESS}]." + PAD)
    c = check_note(note, brief)["note_arc_citations"]
    assert c["label"] == "verified" and c["uncited_day_sentences"] == 0
    assert c["cited_valid"] == ["arc:2545de72655edb66", f"arc:{HARNESS}"] and c["rejected_unknown"] == []


def test_the_live_uncited_sentence_shape_stays_unverified_not_blocked():
    brief = brief_1009()
    note = "I talked with Juniper from 15:00 to 16:12." + PAD
    out = check_note(note, brief)
    assert out["note_md"] == note  # kept verbatim
    assert out["note_arc_citations"]["label"] == "UNVERIFIED"


def test_a_day_bracket_alone_does_not_verify_an_attention_sentence():
    """Why the prompt says [day:] covers only whole-day counts: the gate counts [arc:] ids, so a
    sentence naming an attention subject with only the day bracket stays UNVERIFIED (kept)."""
    brief = brief_1009()
    note = "Harness closure prediction error held me 2 h 15 min over 11 arcs [day:2026-10-09]." + PAD
    out = check_note(note, brief)
    assert out["note_md"] == note and out["note_arc_citations"]["label"] == "UNVERIFIED"


def test_an_arc_shown_only_by_the_temporal_core_is_kept_and_shares_are_stored():
    """Living Prime #3224: the temporal core is a second cited block; its arcs pass the gate, and
    the honest input's per-source shares land in sources.input_shares."""
    from orion.orion_day.day_input import render_day_input
    from orion.orion_day.temporal_core import render_temporal_core

    day = day_1009()
    material = brief_1009(with_block=False).material.model_copy(update={"temporal_core": render_temporal_core(day)})
    brief = OrionDayRunBriefV1(letter_date=material.letter_date, window_start=material.window_start,
                               window_end=material.window_end, material=material,
                               llm_view=render_day_input(material))
    core_arc = next(r[4:] for r in material.temporal_core.records if r.startswith("arc:"))
    note = f"I talked with Juniper [arc:{core_arc}]. And a door [arc:deadbeefdeadbeef]." + PAD
    out = check_note(note, brief)
    assert f"[arc:{core_arc}]" in out["note_md"] and "[arc:deadbeefdeadbeef]" not in out["note_md"]
    state = {"run_id": "r", "note_md": out["note_md"], "carry_forward_md": "- x", **out}
    sources = OrionDayLetterSourcesV1.model_validate(letter_row(state, brief, datetime.now(timezone.utc))["sources"])
    assert sources.input_shares == brief.llm_view.source_shares and sources.input_shares
    assert sources.input_chars["temporal_core"] == len(material.temporal_core.text)


def test_an_event_ref_the_core_did_not_show_is_removed_and_recorded():
    from orion.orion_day.day_input import render_day_input
    from orion.orion_day.temporal_core import render_temporal_core
    from orion.schemas.temporal_self import ExpectationRefV1

    day = day_1009()
    resolved = [ExpectationRefV1(event_id="action_outcome:7", source_kind="action_outcome",
                                 committed_at=day.closed_at - timedelta(hours=2),
                                 resolved_at=day.closed_at - timedelta(hours=2), verdict="claim_upheld=false")]
    day = day.model_copy(update={"frame": day.frame.model_copy(update={
        "expectations_resolved_today": resolved, "expectations_resolved_total": 1})})
    material = brief_1009(with_block=False).material.model_copy(update={"temporal_core": render_temporal_core(day)})
    brief = OrionDayRunBriefV1(letter_date=material.letter_date, window_start=material.window_start,
                               window_end=material.window_end, material=material, llm_view=render_day_input(material))
    note = "One act missed [event:action_outcome:7]. Another I made up [event:action_outcome:999999]." + PAD
    out = check_note(note, brief)
    assert "[event:action_outcome:7]" in out["note_md"] and "999999" not in out["note_md"]
    assert out["note_rejected_events"] == ["action_outcome:999999"]
    state = {"run_id": "r", "carry_forward_md": "- x", **out}
    sources = OrionDayLetterSourcesV1.model_validate(letter_row(state, brief, datetime.now(timezone.utc))["sources"])
    assert sources.rejected_event_refs == {"note": ["action_outcome:999999"]}
