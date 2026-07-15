from __future__ import annotations

from orion.harness.tool_provenance_audit import detect_tool_provenance_mismatch
from orion.schemas.harness_finalize import GrammarReceiptV1


def _receipt(tool_name: str | None, *, index: int = 0) -> GrammarReceiptV1:
    return GrammarReceiptV1(step_index=index, tool_name=tool_name, summary=f"[{index}] step", grammar_event_id=f"g{index}")


def test_flags_fetch_plus_immediacy_language():
    receipts = [_receipt("mcp__github__get_file_contents")]
    draft = "Yeah, that's pressure gradients actively computing in the background this turn."
    result = detect_tool_provenance_mismatch(draft, receipts)
    assert result is not None
    assert "mcp__github__get_file_contents" in result
    assert "tool_provenance_mismatch" in result


def test_no_flag_for_fetch_without_immediacy_language():
    receipts = [_receipt("mcp__github__get_file_contents")]
    draft = "Here's what I found when I looked at pressure.py just now: it does bounded propagation."
    assert detect_tool_provenance_mismatch(draft, receipts) is None


def test_no_flag_for_immediacy_language_without_fetch():
    receipts = [_receipt(None), _receipt("Bash", index=1)]
    draft = "Lots of signal computing in the background this turn."
    assert detect_tool_provenance_mismatch(draft, receipts) is None


def test_no_flag_for_non_fetch_tool_use():
    receipts = [_receipt("Bash"), _receipt("Grep", index=1)]
    draft = "Something is happening right now in the substrate."
    assert detect_tool_provenance_mismatch(draft, receipts) is None


def test_no_flag_when_no_receipts():
    assert detect_tool_provenance_mismatch("computing this turn", []) is None


def test_no_flag_when_no_draft_text():
    receipts = [_receipt("mcp__github__get_file_contents")]
    assert detect_tool_provenance_mismatch("", receipts) is None


def test_names_multiple_fetch_tools():
    receipts = [
        _receipt("mcp__github__get_file_contents"),
        _receipt("WebFetch", index=1),
    ]
    draft = "This is computing right now."
    result = detect_tool_provenance_mismatch(draft, receipts)
    assert result is not None
    assert "WebFetch" in result
    assert "mcp__github__get_file_contents" in result


def test_case_insensitive_matching():
    receipts = [_receipt("MCP__GITHUB__GET_FILE_CONTENTS")]
    draft = "RIGHT NOW this is live."
    assert detect_tool_provenance_mismatch(draft, receipts) is not None


def test_no_flag_for_generic_use_of_the_word_computing():
    # Regression: a bare "computing" match (no temporal/proximity qualifier)
    # false-positives on ordinary language unrelated to any liveness claim.
    receipts = [_receipt("mcp__github__get_file_contents")]
    draft = "Cloud computing costs are rising, so I pulled this config to check line counts."
    assert detect_tool_provenance_mismatch(draft, receipts) is None
