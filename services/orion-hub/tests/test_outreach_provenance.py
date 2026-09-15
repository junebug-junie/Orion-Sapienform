from __future__ import annotations

from scripts.outreach_provenance import (
    format_outreach_provenance_block,
    merge_situation_with_outreach_provenance,
)


def _capsule(**overrides):
    base = {
        "schema": "outreach_provenance.v1",
        "decision_id": "dec-1",
        "correlation_id": "corr-1",
        "generated_at": "2026-09-15T19:27:05+00:00",
        "lanes": {"priors_count": 3, "tension": False},
        "prompt_text": "You are Orion. Juniper has not asked you anything — FULL PROMPT",
        "summary_line": "Outreach from open priors (3)",
    }
    base.update(overrides)
    return base


def test_format_outreach_provenance_block_includes_prompt_and_anti_confabulation() -> None:
    block = format_outreach_provenance_block(_capsule())
    assert "FULL PROMPT" in block
    # Header uses "unsolicited endogenous outreach" (contiguous "unsolicited outreach"
    # is not present); assert the real anti-confabulation phrases.
    assert "unsolicited endogenous outreach" in block.lower()
    assert "collapse-mirror" in block.lower()
    assert "do not invent" in block.lower()


def test_format_returns_empty_on_bad_capsule() -> None:
    assert format_outreach_provenance_block({}) == ""
    assert format_outreach_provenance_block({"prompt_text": ""}) == ""


def test_merge_situation_appends_block() -> None:
    merged = merge_situation_with_outreach_provenance("local afternoon", "OUTREACH BLOCK")
    assert merged.startswith("local afternoon")
    assert "OUTREACH BLOCK" in merged


def test_merge_situation_block_only() -> None:
    assert merge_situation_with_outreach_provenance(None, "OUTREACH BLOCK") == "OUTREACH BLOCK"


def test_merge_situation_none_when_both_empty() -> None:
    assert merge_situation_with_outreach_provenance(None, None) is None
    assert merge_situation_with_outreach_provenance("  ", "") is None


def test_situation_with_outreach_provenance_merges_block(monkeypatch) -> None:
    from orion.hub import turn_orchestrator as orch

    monkeypatch.setattr(
        orch,
        "fetch_latest_outreach_provenance",
        lambda sid, **kw: _capsule(),
    )
    merged = orch._situation_with_outreach_provenance("local afternoon", "sess-1")
    assert merged is not None
    assert merged.startswith("local afternoon")
    assert "FULL PROMPT" in merged
    assert "unsolicited endogenous outreach" in merged.lower()


def test_situation_with_outreach_provenance_no_session_passthrough() -> None:
    from orion.hub import turn_orchestrator as orch

    assert orch._situation_with_outreach_provenance("local afternoon", None) == "local afternoon"
    assert orch._situation_with_outreach_provenance(None, None) is None
