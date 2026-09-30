"""Orion's Day contracts: schema registry, the durable request, brief/letter invariants,
the two prompts' separation, the journal dispatch policy, and world_pulse_read's text_cap."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment
from pydantic import ValidationError

from orion.journaler.dispatch_registry import resolve_policy
from orion.journaler.schemas import JournalEntryWriteV1
from orion.journaler.worker import _TRIGGER_TO_MODE
from orion.orion_day.brief import OrionDayEmptyError, brief_from_material, build_orion_day_request, default_deadline
from orion.orion_day.gather import gather_orion_day
from orion.orion_day.store import letter_from_row
from orion.orion_day.tests import fixtures as fx
from orion.schemas.durable_run import DurableRunRequestV1, DurableRunStateV1
from orion.schemas.orion_day import (
    ORION_DAY_CARRY_FORWARD_VERB,
    ORION_DAY_NOTE_VERB,
    ORION_DAY_WORKFLOW,
    OrionDayLetterV1,
    OrionDayMaterialV1,
    OrionDayRunBriefV1,
    orion_day_journal_entry_id,
    orion_day_run_id,
)
from orion.schemas.registry import resolve
from orion.world_pulse_read import introspect as wp

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "orion" / "cognition" / "prompts"
VERBS = ROOT / "orion" / "cognition" / "verbs"
NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)

# Instruction vocabulary that belongs to the carry-forward call only.
CARRY_FORWARD_WORDS = ("carry", "forward", "future", "tomorrow", "next time", "follow up", "follow-up",
                       "thread", "to pursue", "open question", "revisit")


def _material():
    return asyncio.run(gather_orion_day(fx.FakeConn(), fx.LETTER_DATE, now=NOW))


def _brief():
    return brief_from_material(_material())


def _render(name: str, od: dict) -> str:
    return Environment(autoescape=False).from_string((PROMPTS / name).read_text()).render(
        metadata={"orion_day_input": od})


@pytest.mark.parametrize("name,model", [
    ("OrionDayRunBriefV1", OrionDayRunBriefV1), ("OrionDayMaterialV1", OrionDayMaterialV1),
    ("OrionDayLetterV1", OrionDayLetterV1),
])
def test_registered_in_schema_registry(name, model):
    assert resolve(name) is model


def test_request_is_admitted_on_the_agent_lane_in_background():
    request = build_orion_day_request(_brief())
    assert request.workflow == ORION_DAY_WORKFLOW == "orion_day.letter"
    assert request.run_id == "orion-day-2026-09-29-1"
    assert request.admission.resource == "llm.route.agent" and request.admission.preferred_lane == "agent"
    assert request.admission.priority == "background"
    assert request.admission.deadline_at == default_deadline(date(2026, 9, 29))
    assert request.admission.deadline_at == datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
    # round-trips through the wire form every consumer parses
    again = DurableRunRequestV1.model_validate(json.loads(request.model_dump_json()))
    assert isinstance(again.brief, OrionDayRunBriefV1)
    assert again.brief.llm_view.digest_md == request.brief.llm_view.digest_md


def test_request_asks_for_enough_context_for_digest_note_and_carry_forward():
    from orion.orion_day.brief import minimum_context_tokens

    brief = _brief()
    request = build_orion_day_request(brief)
    need = request.admission.requirements["minimum_context_tokens"]
    assert need == minimum_context_tokens(brief) == brief.llm_view.approx_tokens + 1500 + 12000 + 4000
    # A heavy day (the live 2026-09-29 digest: ~70k estimated tokens) cannot go to the 65,536-token chat card.
    heavy = brief.model_copy(update={"llm_view": brief.llm_view.model_copy(update={"approx_tokens": 69938})})
    assert minimum_context_tokens(heavy) > 65536


def test_request_requires_admission_and_a_matching_brief():
    brief = _brief()
    with pytest.raises(ValidationError, match="require durable resource admission"):
        DurableRunRequestV1(run_id="orion-day-x", workflow="orion_day.letter", correlation_id="c", brief=brief)
    reading_brief = {"seed_id": "s", "stage": 1, "prompt": "p", "session_id": "s", "timeout_sec": 10}
    with pytest.raises(ValidationError):
        DurableRunRequestV1(run_id="orion-day-x", workflow="orion_day.letter", correlation_id="c",
                            brief=reading_brief, admission={"resource": "llm.route.agent"})
    with pytest.raises(ValidationError):
        DurableRunRequestV1(run_id="orion-day-x", workflow="reading.turn", correlation_id="c",
                            brief=brief, admission={"resource": "llm.route.agent"})


def test_state_events_accept_the_new_workflow():
    DurableRunStateV1(run_id="r", workflow="orion_day.letter", thread_id="r", node="finish",
                      status="completed", correlation_id="c")


def test_brief_never_routes_to_chat():
    data = _brief().model_dump(mode="json")
    data["llm_route"] = "chat"
    with pytest.raises(ValidationError):
        OrionDayRunBriefV1.model_validate(data)


def test_brief_window_must_match_material():
    data = _brief().model_dump(mode="json")
    data["window_end"] = (datetime.fromisoformat(data["window_end"]) + timedelta(hours=1)).isoformat()
    with pytest.raises(ValidationError, match="material window"):
        OrionDayRunBriefV1.model_validate(data)


def test_empty_day_raises_instead_of_briefing_an_empty_shell():
    empty = OrionDayMaterialV1(letter_date=fx.LETTER_DATE, window_start=fx.T0, window_end=fx.T0 + timedelta(days=1),
                               gathered_at=NOW, world_pulse_digest=None)
    with pytest.raises(OrionDayEmptyError):
        brief_from_material(empty)


def test_run_id_and_journal_id_are_stable():
    assert orion_day_run_id("2026-09-29") == "orion-day-2026-09-29-1"
    assert orion_day_run_id(date(2026, 9, 29), 2) == "orion-day-2026-09-29-2"
    assert orion_day_journal_entry_id(date(2026, 9, 29)) == orion_day_journal_entry_id("2026-09-29")
    assert orion_day_journal_entry_id("2026-09-29") != orion_day_journal_entry_id("2026-09-30")
    with pytest.raises(ValueError):
        orion_day_run_id("2026-09-29", 0)


def _letter_row(**overrides):
    material = _material()
    row = {
        "letter_date": fx.LETTER_DATE, "run_id": "orion-day-2026-09-29-1",
        "window_start": material.window_start, "window_end": material.window_end,
        "note_md": "A long note.", "carry_forward_md": "- [curiosity:ab61e4ccd47b] test it",
        "material": material.model_dump_json(), "sources": json.dumps({"by_source": {}}),
        "journal_entry_id": orion_day_journal_entry_id(fx.LETTER_DATE), "created_at": NOW,
        "emailed_at": None, "email_notification_id": None, "carry_forward_expires_at": NOW + timedelta(hours=48),
        "carry_forward_offered_at": None, "carry_forward_offered_run_id": None,
    }
    row.update(overrides)
    return row


def test_letter_row_decodes_and_keeps_note_and_carry_forward_apart():
    letter = letter_from_row(_letter_row())
    assert letter.note_md == "A long note." and letter.carry_forward_md.startswith("- [curiosity:")
    with pytest.raises(ValidationError, match="distinct"):
        letter_from_row(_letter_row(carry_forward_md="A long note."))
    with pytest.raises(ValidationError):
        letter_from_row(_letter_row(note_md=""))


# --- the two prompts ---------------------------------------------------------------------------


def test_note_prompt_carries_no_carry_forward_instruction():
    rendered = _render("orion_day_note_v1.j2", {"letter_date": "2026-09-29", "timezone": "America/Denver",
                                               "digest_md": "DIGEST-PLACEHOLDER"})
    instructions = rendered.split("DIGEST-PLACEHOLDER")[0].lower()
    for word in CARRY_FORWARD_WORDS:
        assert word not in instructions, word
    assert "note_md" not in rendered


def test_note_prompt_renders_the_real_digest_and_grounding_rule():
    brief = _brief()
    rendered = _render("orion_day_note_v1.j2", {"letter_date": "2026-09-29", "timezone": "America/Denver",
                                               "digest_md": brief.llm_view.digest_md})
    assert brief.llm_view.digest_md in rendered
    assert "Do not invent events" in rendered
    assert "first person" in rendered


def test_carry_forward_prompt_gets_the_note_and_digest():
    rendered = _render("orion_day_carry_forward_v1.j2", {
        "letter_date": "2026-09-29", "timezone": "America/Denver", "digest_md": "DIGEST", "note_md": "THE-NOTE"})
    assert "DIGEST" in rendered and "THE-NOTE" in rendered
    assert "future curiosity" in rendered
    assert rendered.index("DIGEST") < rendered.index("THE-NOTE")


@pytest.mark.parametrize("verb,template", [
    (ORION_DAY_NOTE_VERB, "orion_day_note_v1.j2"),
    (ORION_DAY_CARRY_FORWARD_VERB, "orion_day_carry_forward_v1.j2"),
])
def test_verb_yaml_points_at_its_own_template_with_a_long_timeout(verb, template):
    data = yaml.safe_load((VERBS / f"{verb}.yaml").read_text())
    assert data["name"] == verb
    assert [s["prompt_template"] for s in data["steps"]] == [template]
    # above the brief's default per-call RPC wait, so the caller times out first
    assert data["timeout_ms"] > OrionDayRunBriefV1.model_fields["timeout_sec"].default * 1000
    assert all(s["timeout_ms"] == data["timeout_ms"] for s in data["steps"])


# --- journal ---------------------------------------------------------------------------------


def test_journal_trigger_is_registered_without_email():
    assert _TRIGGER_TO_MODE["orion_day_letter"] == "daily"
    policy = resolve_policy("orion_day_letter")
    assert policy.email_enabled is False and policy.in_app_enabled is False
    JournalEntryWriteV1(author="orion", mode="daily", body="b", source_kind="orion_day",
                        trigger_kind="orion_day_letter")


# --- world_pulse_read text_cap ---------------------------------------------------------------


def _reading_row(summary: str):
    return {**fx.READING_ROWS[0], "stage2_result_json": json.dumps({"summary": summary})}


def test_reading_item_default_cap_is_unchanged():
    item = wp._item(_reading_row("x" * 2000), request_id=None)
    assert len(item.text) == wp.DEFAULT_TEXT_CAP == 900 and item.truncated is True


def test_reading_item_text_cap_none_keeps_full_text():
    item = wp._item(_reading_row("x" * 2000), request_id=None, text_cap=None)
    assert len(item.text) == 2000 and item.truncated is False


def test_reading_results_default_call_still_clips():
    class Conn:
        async def fetch(self, sql, *args):
            return [{**_reading_row("y" * 1500), "total": 1}]

    result = asyncio.run(wp.reading_results(Conn()))
    assert len(result.items[0].text) == 900
    full = asyncio.run(wp.reading_results(Conn(), text_cap=None))
    assert len(full.items[0].text) == 1500
