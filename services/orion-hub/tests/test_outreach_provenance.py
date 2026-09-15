from __future__ import annotations

import pytest

from scripts.outreach_provenance import (
    format_outreach_provenance_block,
    merge_situation_with_outreach_provenance,
    _valid_outreach_capsule,
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
    assert format_outreach_provenance_block(None) == ""


def test_format_returns_empty_on_wrong_schema() -> None:
    assert (
        format_outreach_provenance_block(
            _capsule(schema="outreach_provenance.v0")
        )
        == ""
    )
    assert format_outreach_provenance_block(_capsule(schema="other.v1")) == ""
    # Missing schema key
    bad = _capsule()
    del bad["schema"]
    assert format_outreach_provenance_block(bad) == ""


def test_format_returns_empty_on_non_string_prompt_text() -> None:
    assert format_outreach_provenance_block(_capsule(prompt_text=None)) == ""
    assert format_outreach_provenance_block(_capsule(prompt_text=12345)) == ""
    assert format_outreach_provenance_block(_capsule(prompt_text=["list"])) == ""
    assert format_outreach_provenance_block(_capsule(prompt_text="   ")) == ""


def test_valid_outreach_capsule_rejects_malformed() -> None:
    assert _valid_outreach_capsule(_capsule()) is not None
    assert _valid_outreach_capsule(_capsule(schema="nope")) is None
    assert _valid_outreach_capsule(_capsule(prompt_text=42)) is None
    assert _valid_outreach_capsule("not-a-dict") is None


def test_merge_situation_appends_block() -> None:
    merged = merge_situation_with_outreach_provenance("local afternoon", "OUTREACH BLOCK")
    assert merged.startswith("local afternoon")
    assert "OUTREACH BLOCK" in merged


def test_merge_situation_block_only() -> None:
    assert merge_situation_with_outreach_provenance(None, "OUTREACH BLOCK") == "OUTREACH BLOCK"


def test_merge_situation_none_when_both_empty() -> None:
    assert merge_situation_with_outreach_provenance(None, None) is None
    assert merge_situation_with_outreach_provenance("  ", "") is None


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_merges_block(monkeypatch) -> None:
    from orion.hub import turn_orchestrator as orch

    monkeypatch.setattr(
        orch,
        "fetch_latest_outreach_provenance",
        lambda sid, **kw: _capsule(),
    )
    merged = await orch._situation_with_outreach_provenance("local afternoon", "sess-1")
    assert merged is not None
    assert merged.startswith("local afternoon")
    assert "FULL PROMPT" in merged
    assert "unsolicited endogenous outreach" in merged.lower()


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_skips_bad_schema(monkeypatch) -> None:
    from orion.hub import turn_orchestrator as orch

    monkeypatch.setattr(
        orch,
        "fetch_latest_outreach_provenance",
        lambda sid, **kw: _capsule(schema="wrong.v1"),
    )
    merged = await orch._situation_with_outreach_provenance("local afternoon", "sess-1")
    assert merged == "local afternoon"


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_uses_to_thread(monkeypatch) -> None:
    """Fetch must leave the event loop via asyncio.to_thread."""
    from orion.hub import turn_orchestrator as orch

    calls: list[object] = []

    def _fake_fetch(sid, **kw):
        calls.append(("fetch", sid))
        return _capsule()

    async def _fake_to_thread(fn, /, *args, **kwargs):
        calls.append(("to_thread", fn, args))
        return fn(*args, **kwargs)

    monkeypatch.setattr(orch, "fetch_latest_outreach_provenance", _fake_fetch)
    monkeypatch.setattr(orch.asyncio, "to_thread", _fake_to_thread)

    merged = await orch._situation_with_outreach_provenance("sit", "sess-9")
    assert any(c[0] == "to_thread" for c in calls)
    assert any(c[0] == "fetch" and c[1] == "sess-9" for c in calls)
    assert merged is not None and "FULL PROMPT" in merged


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_no_session_passthrough() -> None:
    from orion.hub import turn_orchestrator as orch

    assert (
        await orch._situation_with_outreach_provenance("local afternoon", None)
        == "local afternoon"
    )
    assert await orch._situation_with_outreach_provenance(None, None) is None
