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


def test_prompt_defines_every_stakes_category_with_examples():
    """Every category the schema accepts is defined in the prompt (no drift), with examples, and
    the stakes_reason is required on every memory, including the shape example."""
    from typing import get_args

    from orion.schemas.memory_episode import HIGH_STAKES_REASONS, StakesReason

    prompt = render_prompt(episode_id="ep-1", turns=turns_from_rows(ROWS))
    for reason in HIGH_STAKES_REASONS:
        line = next((ln for ln in prompt.splitlines() if ln.startswith(f'- "{reason}":')), None)
        assert line is not None, reason
        assert "For example:" in line
    assert '"stakes_reason": "none"' in prompt
    assert '"stakes_reason": null' not in prompt
    assert set(get_args(StakesReason)) == HIGH_STAKES_REASONS | {"none"}
    # The shape example parses and passes the consistency check unchanged.
    from orion.memory.episode.validate import resolve_stakes

    start = prompt.index('{"memories"')
    example = parse_distillation(prompt[start:])
    assert {(m.stakes, m.stakes_reason) for m in example.memories} == {("high", "juniper_feelings"), ("low", "none")}
    assert all(resolve_stakes(m.stakes, m.stakes_reason, m.asks_direction)[2] is None for m in example.memories)


def test_template_states_its_version_and_matches_the_schema_default():
    from orion.memory.episode.distill import UNMARKED_TEMPLATE_VERSION, template_prompt_version
    from orion.schemas.memory_episode import MEMORY_EPISODE_DISTILL_PROMPT_VERSION

    assert template_prompt_version() == MEMORY_EPISODE_DISTILL_PROMPT_VERSION == "memory_episode_distill.v3"
    assert "prompt_version" not in render_prompt(episode_id="e", turns=turns_from_rows(ROWS))  # a comment
    assert UNMARKED_TEMPLATE_VERSION == "memory_episode_distill.v2"


def test_unmarked_template_reads_as_v2(tmp_path):
    from orion.memory.episode.distill import template_prompt_version

    old = tmp_path / "old.j2"
    old.write_text("You are Orion. ...", encoding="utf-8")
    assert template_prompt_version(old) == "memory_episode_distill.v2"


def test_report_never_prints_high_none():
    from zoneinfo import ZoneInfo

    from orion.memory.episode.report import render_episode

    ep = {"episode_id": "ep-123456789", "started_at": datetime(2026, 10, 5, 3, tzinfo=timezone.utc),
          "last_turn_at": datetime(2026, 10, 5, 4, tzinfo=timezone.utc), "close_reason": "x"}
    mem = {"purpose": "happened", "voice": "juniper_said", "channel": "chat", "statement": "s",
           "confirmation_state": "pending_confirmation", "stakes": "high"}
    out = "\n".join(render_episode(ep, [], [{**mem, "stakes_reason": None}, {**mem, "stakes_reason": "unjudged"}],
                                   [], {"run_id": "r"}, ZoneInfo("UTC")))
    assert "high: None" not in out
    assert "high: no category" in out and "high: unjudged" in out
