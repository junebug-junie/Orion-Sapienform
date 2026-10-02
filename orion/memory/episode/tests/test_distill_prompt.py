"""The distiller prompt renders full turn text, and the parser fails loudly on non-JSON."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.memory.episode.distill import (
    is_workflow_command_reply,
    malformed_item_count,
    parse_distillation,
    render_prompt,
    turns_from_rows,
    turns_from_state,
    turns_to_state,
)

LONG = "x" * 300 + " the important part is at the end"
ROWS = [
    {"correlation_id": "c-2", "prompt": "Run github compactor.", "response": "Workflow: GitHub Compactor",
     "created_at": datetime(2026, 9, 28, 9, 44)},
    {"correlation_id": "c-1", "prompt": LONG, "response": "ok", "created_at": datetime(2026, 9, 28, 9, 6)},
]


def test_turns_are_time_ordered_labelled_and_commands_flagged():
    turns = turns_from_rows(ROWS)
    assert [(t.label, t.correlation_id, t.is_command) for t in turns] == [("t1", "c-1", False), ("t2", "c-2", True)]
    assert turns[0].created_at.tzinfo is not None
    assert turns_from_state(turns_to_state(turns)) == turns


def test_prompt_carries_full_untruncated_text_and_marks_commands():
    prompt = render_prompt(episode_id="ep-1", turns=turns_from_rows(ROWS), candidate_referents=["person:juniper"])
    assert LONG in prompt
    assert "[t2] 2026-09-28 03:44 COMMAND" in prompt          # local time, America/Denver
    assert "- person:juniper" in prompt
    assert '"memories"' in prompt and '"voice"' in prompt


def test_parse_accepts_fenced_json_and_drops_malformed_items():
    text = '```json\n{"memories": [{"purpose": "happened", "voice": "juniper_said", "statement": "s"}, ' \
           '{"purpose": "bogus"}], "questions": []}\n```'
    d = parse_distillation(text)
    assert len(d.memories) == 1
    assert malformed_item_count(text) == 1


@pytest.mark.parametrize("text", ["", "I could not decide.", "{not json}", "[1, 2]"])
def test_parse_raises_on_no_json_object(text):
    with pytest.raises(ValueError):
        parse_distillation(text)


def test_command_detector_is_the_runtime_header():
    assert is_workflow_command_reply("Workflow: Journal Pass")
    assert is_workflow_command_reply("Workflow 'github_compactor_pass' started")
    assert not is_workflow_command_reply("Your workflow sounds good")
