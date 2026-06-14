from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from orion.core.contracts.recall import MemoryItemV1

from app.render import _estimate_tokens_strict, render_items


def _long_snippet(prefix: str, *, repeats: int = 40) -> str:
    chunk = (
        "SQL timeline evidence about collapse mirror triage and metacog grounding with provenance "
        "source_ref collapse_mirror table row id "
    )
    return f"{prefix} " + (chunk * repeats)


def test_metacog_render_respects_token_and_char_budget_with_long_items() -> None:
    budget_tokens = 256
    budget_chars = 1280
    items = [
        MemoryItemV1(
            id=f"sql-{i}",
            source="sql_timeline",
            source_ref="collapse_mirror",
            score=0.9 - (i * 0.01),
            snippet=_long_snippet(f"item-{i}"),
        )
        for i in range(24)
    ]
    rendered, dropped = render_items(
        items,
        budget_tokens,
        profile_name="journal.daily.metacog.grounded.v1",
        strict_prompt_budget=True,
        render_char_budget=budget_chars,
        max_snippet_chars=280,
    )
    assert rendered
    assert _estimate_tokens_strict(rendered) <= budget_tokens
    assert len(rendered) <= budget_chars
    assert all(len(line) <= 320 for line in rendered.splitlines() if line.startswith("- "))
    assert dropped


def test_metacog_render_snippets_keep_source_provenance() -> None:
    items = [
        MemoryItemV1(
            id="a",
            source="sql_timeline",
            source_ref="collapse_mirror",
            score=0.8,
            snippet="collapse triage stored event with observer_state zen",
        )
    ]
    rendered, _ = render_items(
        items,
        256,
        profile_name="journal.daily.metacog.grounded.v1",
        strict_prompt_budget=True,
        render_char_budget=1280,
        max_snippet_chars=280,
    )
    assert "[sql_timeline:collapse_mirror]" in rendered
    assert "collapse triage" in rendered
