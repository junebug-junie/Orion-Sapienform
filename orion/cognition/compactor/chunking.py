from __future__ import annotations

import json
from typing import Any, Sequence


def json_char_len(obj: Any) -> int:
    """Serialized size of one digest input item (what the prompt pays for)."""
    return len(json.dumps(obj, ensure_ascii=False, default=str))


def chunk_items_by_char_budget(items: Sequence[Any], *, budget_chars: int) -> list[list[Any]]:
    """Split `items` (order preserved) into chunks whose serialized size fits `budget_chars`.

    Every item lands in exactly one chunk -- nothing is dropped. An item larger
    than the budget on its own gets a chunk to itself (callers cap single items
    well under the budget, so this is a pathological-input guard, not a path).
    """
    budget = max(1, int(budget_chars))
    chunks: list[list[Any]] = []
    current: list[Any] = []
    used = 0
    for item in items:
        size = json_char_len(item)
        if current and used + size > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(item)
        used += size
    if current:
        chunks.append(current)
    return chunks
