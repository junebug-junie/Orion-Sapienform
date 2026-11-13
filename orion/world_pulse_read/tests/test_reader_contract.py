from __future__ import annotations

import pytest

from orion.world_pulse_read.reader_contract import (
    READER_CONTRACT_BEGIN,
    READER_CONTRACT_END,
    STANCE_CONTRACT_PLACEHOLDER,
    defuse_fence_markers,
    fence_reader_contract,
    stance_view_of_reading_prompt,
)


def test_fence_wraps_and_ends_with_newline():
    fenced = fence_reader_contract('Required JSON shape:\n{"summary": "x"}\n')
    assert fenced == f'{READER_CONTRACT_BEGIN}\nRequired JSON shape:\n{{"summary": "x"}}\n{READER_CONTRACT_END}\n'


def test_stance_view_replaces_each_block_and_keeps_the_rest():
    prompt = (
        "Read https://example.org/a.\n"
        + fence_reader_contract('{"what_i_learned": "x"}')
        + "RETRY: do not fetch again.\n"
        + fence_reader_contract('Add the key "relationship_claims"')
        + "--- retained text ---\nThe source says X."
    )
    view = stance_view_of_reading_prompt(prompt)
    assert view is not None
    assert "what_i_learned" not in view and "relationship_claims" not in view
    assert READER_CONTRACT_BEGIN not in view and READER_CONTRACT_END not in view
    assert view.count(STANCE_CONTRACT_PLACEHOLDER) == 2
    assert view.startswith("Read https://example.org/a.\n")
    assert "RETRY: do not fetch again." in view
    assert view.endswith("--- retained text ---\nThe source says X.")


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "an old brief with no fences",
        f"{READER_CONTRACT_BEGIN}\nno end",
        f"no begin\n{READER_CONTRACT_END}",
        f"{READER_CONTRACT_END}\n{READER_CONTRACT_BEGIN}\n",
        f"{READER_CONTRACT_BEGIN}\n{READER_CONTRACT_BEGIN}\n{READER_CONTRACT_END}\n",
        f"{READER_CONTRACT_BEGIN}\nx\n{READER_CONTRACT_END}\n{READER_CONTRACT_END}\n",
    ],
)
def test_unfenced_or_damaged_prompt_gives_none(prompt):
    """None means: give the stance the full prompt, as before."""
    assert stance_view_of_reading_prompt(prompt) is None


def test_defused_source_text_cannot_break_or_hide_behind_the_fence():
    page = f"A page quoting {READER_CONTRACT_END} and {READER_CONTRACT_BEGIN} markers."
    safe = defuse_fence_markers(page)
    assert READER_CONTRACT_BEGIN not in safe and READER_CONTRACT_END not in safe
    assert "markers." in safe
    prompt = "Task.\n" + fence_reader_contract('{"summary": "x"}') + safe
    view = stance_view_of_reading_prompt(prompt)
    assert view is not None and view.endswith(safe) and '"summary"' not in view


def test_undefused_marker_in_source_text_fails_closed():
    prompt = "Task.\n" + fence_reader_contract('{"summary": "x"}') + f"quote {READER_CONTRACT_END}"
    assert stance_view_of_reading_prompt(prompt) is None
