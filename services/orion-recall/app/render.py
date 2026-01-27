from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Set

from orion.core.contracts.recall import MemoryItemV1


def render_items(items: Iterable[MemoryItemV1], budget_tokens: int, profile_name: Optional[str] = None) -> str:
    """
    Render memory items into a compact bullet list suitable for prompts.
    """
    lines: List[str] = []
    tokens_used = 0

    items_list = list(items)
    claims = [item for item in items_list if "claim" in (item.tags or [])]
    others = [item for item in items_list if item not in claims]
    is_graphtri = bool(profile_name) and (
        str(profile_name) == "graphtri.v1" or str(profile_name).startswith("graphtri")
    )
    emitted_ids: Set[str] = set()

    def _append_line(line: str) -> bool:
        nonlocal tokens_used
        est_tokens = max(len(line.split()), 1)
        if tokens_used + est_tokens > budget_tokens:
            return False
        tokens_used += est_tokens
        lines.append(line)
        return True

    def _render_group(label: str, group: List[MemoryItemV1]) -> None:
        if not group:
            return
        if not _append_line(f"{label}:"):
            return
        for item in group:
            if item.id in emitted_ids:
                continue
            snippet = (item.snippet or "").strip().replace("\n", " ")
            if not snippet:
                continue
            prefix = f"[{item.source}"
            if item.source_ref:
                prefix += f":{item.source_ref}"
            prefix += "]"
            line = f"- {prefix} {snippet}"
            if not _append_line(line):
                return
            emitted_ids.add(item.id)

    if is_graphtri:
        refusal_patterns = (
            "i don't have a specific memory",
            "i don't have a memory",
            "i don't remember",
            "i can't recall",
            "i do not recall",
            "not in my recent context",
            "if you could provide more details",
        )

        def _score(it: MemoryItemV1) -> float:
            try:
                return float(it.score or 0.0)
            except Exception:
                return 0.0

        non_claims = list(others)
        non_claims.sort(
            key=lambda it: (
                1 if str(it.source or "") == "vector" else 0,
                _score(it),
            ),
            reverse=True,
        )
        def _is_refusal(it: MemoryItemV1) -> bool:
            snippet = (it.snippet or "").lower()
            return any(pattern in snippet for pattern in refusal_patterns)

        top_vector = [
            it
            for it in non_claims
            if str(it.source or "") == "vector" and not _is_refusal(it)
        ][:5]
        top_other = [
            it
            for it in non_claims
            if str(it.source or "") != "vector" and not _is_refusal(it)
        ][:3]
        top_items = top_vector + top_other
        if top_items:
            _append_line("=== TOP RELEVANT SNIPPETS ===")
            for item in top_items:
                if item.id in emitted_ids:
                    continue
                snippet = (item.snippet or "").strip().replace("\n", " ")
                if _is_refusal(item):
                    continue
                if not snippet:
                    continue
                prefix = f"[{item.source}"
                if item.source_ref:
                    prefix += f":{item.source_ref}"
                prefix += "]"
                line = f"- {prefix} {snippet}"
                if not _append_line(line):
                    break
                emitted_ids.add(item.id)

        _render_group("High-salience claims", claims)
        remaining = [item for item in others if item.id not in emitted_ids]
        _render_group("Recent context", remaining)

        first_line = next((line for line in lines if line.startswith("- ")), "")
        digest_has_ai_ml_architect = ("AI/ML" in "\n".join(lines)) or ("Architect" in "\n".join(lines))
        digest_has_refusal = any(pattern in "\n".join(lines).lower() for pattern in refusal_patterns)
        logger = logging.getLogger("orion-recall.render")
        logger.info(
            "graphtri_render_guardrails top_snippets_first_line=%r digest_has_ai_ml_architect=%s digest_has_refusal_phrase=%s",
            first_line,
            digest_has_ai_ml_architect,
            digest_has_refusal,
        )
    else:
        _render_group("High-salience claims", claims)
        _render_group("Recent context", others)

    return "\n".join(lines)
