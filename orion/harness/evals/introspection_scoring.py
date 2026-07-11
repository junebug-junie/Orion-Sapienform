"""Deterministic scoring for the unified-turn introspection eval.

Experiment artifact for the semantic self-indexing proposal (2026-07-11) —
not a runtime schema, memory type, or bus contract. The scorer is lexical on
purpose: assertions pass when any of their any_of substrings appears in the
answer (case-insensitive). Accuracy of the fixture itself is pinned to the
repo by test_introspection_fixture.py, which requires every evidence symbol
to still exist in its named file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "unified_turn_introspection.json"


def load_fixture(path: Path | None = None) -> Dict[str, Any]:
    return json.loads((path or FIXTURE_PATH).read_text(encoding="utf-8"))


def score_answer(answer: str, assertion_ids: Iterable[str], assertions: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return (passed, failed) assertion id lists for one answer."""
    haystack = str(answer or "").lower()
    passed: list[str] = []
    failed: list[str] = []
    for aid in assertion_ids:
        spec = assertions.get(aid)
        needles = [str(n).lower() for n in (spec or {}).get("any_of", [])]
        if needles and any(n in haystack for n in needles):
            passed.append(aid)
        else:
            failed.append(aid)
    return passed, failed


@dataclass
class ToolMetrics:
    tool_calls: int = 0
    tool_names: List[str] = field(default_factory=list)
    unique_files_read: int = 0
    observed_chars: int = 0
    max_context_fill_pct: float = 0.0
    truncated_results: int = 0


def _iter_tool_use_blocks(raw: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    message = raw.get("message")
    if not isinstance(message, Mapping):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "tool_use":
            yield block


def extract_tool_metrics(step_frames: Iterable[Mapping[str, Any]]) -> ToolMetrics:
    """Fold fcc_motor step frames (raw stream-json events) into navigation metrics."""
    metrics = ToolMetrics()
    files: set[str] = set()
    for frame in step_frames:
        step = frame.get("step") if isinstance(frame, Mapping) else None
        if not isinstance(step, Mapping):
            continue
        fill = step.get("context_fill_pct")
        if isinstance(fill, (int, float)):
            metrics.max_context_fill_pct = max(metrics.max_context_fill_pct, float(fill))
        raw = step.get("raw")
        if not isinstance(raw, Mapping):
            continue
        metrics.observed_chars += len(json.dumps(raw, ensure_ascii=False))
        if "orion-fcc-mcp-proxy" in json.dumps(raw, ensure_ascii=False):
            metrics.truncated_results += 1
        for block in _iter_tool_use_blocks(raw):
            metrics.tool_calls += 1
            name = str(block.get("name") or "unknown")
            metrics.tool_names.append(name)
            tool_input = block.get("input")
            if isinstance(tool_input, Mapping):
                fp = tool_input.get("file_path") or tool_input.get("path")
                if isinstance(fp, str) and fp.strip():
                    files.add(fp.strip())
    metrics.unique_files_read = len(files)
    return metrics
