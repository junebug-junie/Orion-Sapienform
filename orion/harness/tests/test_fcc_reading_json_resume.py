"""A reading turn whose final answer is not the JSON object it was asked for is resumed once.

Live 2026-10-31..11-03 (Qwen3.8-Flash-Next on Strata): 19 readings failed with
"Could not parse JSON object from LLM text". 10 answered in prose, 5 wrote a JSON
object with a syntax error, 3 lost the head of an answer to the output cap (fixed by
the stitcher), 1 had a continuation that re-opened its code fence (fixed in the
stitcher). The texts below are trimmed from the governor's transcripts.
"""
from __future__ import annotations

import pytest

from orion.harness import fcc_motor as motor
from orion.harness.finalize import canonicalize_structured_output
from orion.harness.structured_output import is_structured_output
from orion.harness.tests.test_fcc_reading_undeclared_tool import BASH_CALL, _result, _run

# Pure prose (ed0adaa9, 2026-11-01): a stance summary instead of the reader JSON.
PROSE_STANCE = (
    "Stance — Stage 1 is well-grounded. Against the full article body (one WebFetch, 4769 chars, "
    "no anti-bot/paywall), all eight candidate priors hold on contact; two need wording tightening.\n\n"
    "Prior test (vs. the fetched body):\n- **\"First ...\"** holds."
)
# Preamble only (d7b4197d, 2026-11-01): the model announced the output and stopped.
PREAMBLE_ONLY = ("Now writing the Stage 2 output. The one nontrivial verification came out of re-reading "
                 "the retained table itself, and the hops are all real this-turn results.\n\n")
# Fenced JSON with a single-quoted string in a list (572a5ff5, 2026-10-31).
SINGLE_QUOTED = (
    '```json\n{\n  "summary": "I read this against the retained digest only.",\n'
    '  "link_hints": ["Michigan 2026 Senate debate (final, Oct 21)", \'Muslim Brotherhood allegation\']\n}\n```'
)
# Fenced JSON with an unquoted key opening (cea9591d, 2026-10-31).
UNQUOTED_KEY = (
    '```json\n{\n  "summary": "Stage 1 fetched the seed page three times.",\n  "priors_tested": [\n'
    '    {\n      "verdict": "supported",\n      why_note": "",\n      "why": "The title appears verbatim."\n'
    '    }\n  ]\n}\n```'
)
# Prose that ends on a malformed text tool call (55223448, 2026-11-02): no tool name the
# undeclared-tool detector can read, so it falls to the JSON check.
MANGLED_CALL = ("The legal-tension source came back rich and it corrects the Stage 1 compression.\n\n"
                "<tool_call>\n<functionfunction=WebFetch>\n</function>\n</tool_call>")
ANSWER = '{"summary": "the paper reports excess oxygen", "open_threads": []}'
FENCED_ANSWER = "Here is the Stage 2 output.\n\n```json\n" + ANSWER + "\n```"

BAD_FINALS = [PROSE_STANCE, PREAMBLE_ONLY, SINGLE_QUOTED, UNQUOTED_KEY, MANGLED_CALL, ""]


@pytest.mark.parametrize("text", BAD_FINALS + [ANSWER, FENCED_ANSWER])
def test_the_motor_check_agrees_with_finalize(text):
    """The motor resumes exactly the drafts finalize would reject (one parser on both sides)."""
    try:
        canonicalize_structured_output(text)
        accepted = True
    except ValueError:
        accepted = False
    assert is_structured_output(text) is accepted
    assert (motor.reading_resume_need(text) is None) is accepted


def test_an_undeclared_tool_call_keeps_its_own_reason():
    assert motor.reading_resume_need(BASH_CALL) == (motor.RESUME_UNDECLARED_TOOL, "Bash")
    assert motor.reading_resume_need(PROSE_STANCE) == (motor.RESUME_NOT_JSON, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", BAD_FINALS)
async def test_a_non_json_final_is_resumed_once_with_the_json_note(monkeypatch, bad):
    frames, spawned = await _run(monkeypatch, [_result(bad, "s1"), _result(ANSWER, "s1")])
    finals = [f for f in frames if f["type"] == "final"]
    assert len(finals) == 1 and finals[0]["llm_response"] == ANSWER   # the bad final is never yielded
    record = finals[0]["metadata"]["fcc_reading_json_resume"]
    assert record["session_id"] == "s1" and record["held_text"] == bad.strip()[:500]
    assert "tool" not in record and "fcc_undeclared_tool_resume" not in finals[0]["metadata"]
    assert len(spawned) == 2
    second = list(spawned[1])
    assert second[second.index("--resume") + 1] == "s1"
    note = second[second.index("-p") + 1]
    assert note == motor.JSON_ONLY_NOTE and "only that JSON object, nothing else" in note
    assert second[second.index("--tools") + 1] == "WebFetch,WebSearch"


@pytest.mark.asyncio
async def test_a_second_non_json_final_is_final_no_loop(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(PROSE_STANCE, "s1"), _result(PREAMBLE_ONLY, "s1")])
    assert len(spawned) == 2
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [PREAMBLE_ONLY.strip()]


@pytest.mark.asyncio
async def test_no_json_resume_stacked_on_an_undeclared_tool_resume(monkeypatch):
    """Once per turn total: prose after the tool resume is final, not a second resume."""
    frames, spawned = await _run(monkeypatch, [_result(BASH_CALL, "s1"), _result(PROSE_STANCE, "s1")])
    assert len(spawned) == 2
    finals = [f for f in frames if f["type"] == "final"]
    assert [f["llm_response"] for f in finals] == [PROSE_STANCE]
    assert "fcc_undeclared_tool_resume" in finals[0]["metadata"]
    assert "fcc_reading_json_resume" not in finals[0]["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("good", [ANSWER, FENCED_ANSWER])
async def test_a_parseable_answer_is_not_resumed(monkeypatch, good):
    frames, spawned = await _run(monkeypatch, [_result(good, "s1")])
    assert len(spawned) == 1
    final = [f for f in frames if f["type"] == "final"][0]
    assert final["llm_response"] == good and "fcc_reading_json_resume" not in final["metadata"]


@pytest.mark.asyncio
async def test_prose_is_never_resumed_outside_a_reading_turn(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(PROSE_STANCE, "s1")], reading_only=False)
    assert len(spawned) == 1
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [PROSE_STANCE]


@pytest.mark.asyncio
async def test_no_json_resume_when_the_deadline_is_nearly_spent(monkeypatch):
    frames, spawned = await _run(monkeypatch, [_result(PROSE_STANCE, "s1")], timeout_sec=5.0)
    assert len(spawned) == 1
    assert [f["llm_response"] for f in frames if f["type"] == "final"] == [PROSE_STANCE]
