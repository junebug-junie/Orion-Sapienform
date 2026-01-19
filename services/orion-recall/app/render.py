from __future__ import annotations

from typing import Iterable, List

from orion.core.contracts.recall import MemoryItemV1


def render_items(items: Iterable[MemoryItemV1], budget_tokens: int) -> str:
    """
    Render memory items into a compact bullet list suitable for prompts.
    """
    lines: List[str] = []
    tokens_used = 0

    items_list = list(items)
    claims = [item for item in items_list if "claim" in (item.tags or [])]
    others = [item for item in items_list if item not in claims]

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

    _render_group("High-salience claims", claims)
    _render_group("Recent context", others)

    return "\n".join(lines)
