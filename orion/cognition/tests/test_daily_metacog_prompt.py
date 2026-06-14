from __future__ import annotations

import json
import sys
from pathlib import Path

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = ROOT / "services" / "orion-cortex-exec"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(EXEC_ROOT) not in sys.path:
    sys.path.append(str(EXEC_ROOT))

from app.executor import (  # noqa: E402
    _enforce_daily_metacog_prompt_budget,
    _estimate_prompt_tokens,
    _prompt_render_ctx,
    _render_prompt,
)
from orion.cognition.skills_manifest import build_compact_skill_catalog  # noqa: E402


def _load_daily_metacog_template() -> str:
    path = ROOT / "orion" / "cognition" / "prompts" / "daily_metacog_prompt.j2"
    return path.read_text(encoding="utf-8")


def _bounded_memory_digest() -> str:
    lines = [
        "Recent context:",
        "- [sql_timeline:collapse_mirror] metacog draft mode llm with grounded recall evidence",
        "- [rdf:graphdb] observer_state zen with course correction on prompt sizing",
    ]
    return "\n".join(lines)


def test_daily_metacog_rendered_prompt_stays_bounded() -> None:
    template = _load_daily_metacog_template()
    ctx = {
        "request_date": "2026-06-13",
        "timezone": "America/Denver",
        "node": "athena",
        "window_start_utc": "2026-06-12T06:00:00Z",
        "window_end_utc": "2026-06-13T06:00:00Z",
        "memory_digest": _bounded_memory_digest(),
        "skills_catalog_count": 12,
        "skills_catalog_compact": build_compact_skill_catalog(),
        "recall_memory_bundle_debug": {
            "items": [{"id": "x", "snippet": "should not appear in prompt"}],
            "rendered": "debug only",
        },
        "memory_bundle": {
            "rendered": _bounded_memory_digest(),
            "items": [{"id": "raw", "snippet": "raw bundle must not leak"}],
        },
        "recall_prompt_safe_ctx": True,
    }
    prompt = _render_prompt(template, ctx)
    assert len(prompt) <= 8192
    assert _estimate_prompt_tokens(prompt) <= 2200
    assert "raw bundle must not leak" not in prompt
    assert "should not appear in prompt" not in prompt
    _enforce_daily_metacog_prompt_budget(
        prompt=prompt,
        ctx=ctx,
        correlation_id="test-corr",
        verb_name="daily_metacog_v1",
        step_name="draft_daily_metacog",
    )


def test_daily_metacog_prompt_rejects_oversize_without_truncation() -> None:
    template = _load_daily_metacog_template()
    huge_digest = "X" * 9000
    ctx = {
        "request_date": "2026-06-13",
        "timezone": "America/Denver",
        "node": "athena",
        "window_start_utc": "2026-06-12T06:00:00Z",
        "window_end_utc": "2026-06-13T06:00:00Z",
        "memory_digest": huge_digest,
        "skills_catalog_count": 1,
        "skills_catalog_compact": "[]",
    }
    prompt = _render_prompt(template, ctx)
    try:
        _enforce_daily_metacog_prompt_budget(
            prompt=prompt,
            ctx=ctx,
            correlation_id="test-corr",
            verb_name="daily_metacog_v1",
            step_name="draft_daily_metacog",
        )
    except RuntimeError as exc:
        assert "daily_metacog_prompt_over_limit" in str(exc)
    else:
        raise AssertionError("expected daily_metacog_prompt_over_limit")


def test_prompt_render_ctx_strips_debug_recall_bundle_only_when_opted_in() -> None:
    ctx = {
        "memory_digest": "digest",
        "recall_fragments": [{"snippet": "fragment"}],
        "recall_memory_bundle_debug": {"items": [{"id": "1"}]},
        "memory_bundle": {"rendered": "digest", "items": [{"id": "1", "snippet": "raw"}]},
        "recall_prompt_safe_ctx": True,
    }
    slim = _prompt_render_ctx(ctx)
    assert "recall_fragments" not in slim
    assert "recall_memory_bundle_debug" not in slim
    assert slim["memory_bundle"] == {"rendered": "digest"}


def test_prompt_render_ctx_preserves_journal_lane_bundle_by_default() -> None:
    ctx = {
        "memory_digest": "digest",
        "recall_fragments": [{"snippet": "fragment"}],
        "memory_bundle": {"rendered": "digest", "items": [{"id": "1", "snippet": "raw"}]},
    }
    kept = _prompt_render_ctx(ctx)
    assert kept["recall_fragments"] == [{"snippet": "fragment"}]
    assert kept["memory_bundle"]["items"][0]["snippet"] == "raw"
