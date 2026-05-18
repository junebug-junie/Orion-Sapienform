"""Extract memory_graph_suggest model text from CortexChatResult (final_text + step payloads)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from orion.schemas.cortex.contracts import CortexChatResult

from scripts.cortex_chat_display import hub_effective_chat_text


def _first_json_object(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : idx + 1]
    return None


def _step_text_candidates(step: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not step or not getattr(step, "result", None):
        return out
    res = step.result
    if not isinstance(res, dict):
        return out
    for service_name, block in res.items():
        if not isinstance(block, dict):
            continue
        for field in ("content", "final_text", "text", "reasoning_content"):
            val = block.get(field)
            if isinstance(val, str) and val.strip():
                out.append((f"{service_name}.{field}", val.strip()))
    return out


def hub_memory_graph_suggest_text(resp: CortexChatResult) -> Tuple[str, Dict[str, Any]]:
    """
    Prefer final_text; fall back to llm_memory_graph_suggest step LLMGatewayService payloads.
    Returns (text, diagnostics) for suggest-route tracing.
    """
    diag: Dict[str, Any] = {
        "selected_text_source": None,
        "final_text_len": 0,
        "candidate_fields": [],
        "step_names": [],
    }
    top = hub_effective_chat_text(resp)
    if top:
        diag["selected_text_source"] = "final_text"
        diag["final_text_len"] = len(top)
        return top, diag

    cr = resp.cortex_result
    steps = list(getattr(cr, "steps", None) or [])
    diag["step_names"] = [str(getattr(s, "step_name", "") or "") for s in steps]
    best = ""
    best_source = ""

    def _step_rank(step: Any) -> int:
        name = str(getattr(step, "step_name", "") or "")
        return 0 if name == "llm_memory_graph_suggest" else 1

    for step in sorted(steps, key=_step_rank):
        step_name = str(getattr(step, "step_name", "") or "")
        for source, candidate in _step_text_candidates(step):
            diag["candidate_fields"].append(
                {"source": source, "len": len(candidate), "step": step_name or None}
            )
            json_blob = _first_json_object(candidate)
            pick = json_blob if json_blob else candidate
            if len(pick) > len(best):
                best = pick
                best_source = source
    if best:
        diag["selected_text_source"] = best_source
        diag["final_text_len"] = len(best)
    meta = getattr(cr, "metadata", None) or {}
    if isinstance(meta, dict) and meta.get("structured_rejection_preview"):
        diag["structured_rejection_preview"] = str(meta["structured_rejection_preview"])[:500]
    return best, diag
