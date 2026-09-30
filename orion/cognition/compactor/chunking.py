from __future__ import annotations

import json
from typing import Any, Sequence


_HTMLSAFE_CHARS = ("<", ">", "&", "'")


def json_char_len(obj: Any) -> int:
    """Rendered size of `obj` as the digest prompt actually pays for it.

    The templates render input with Jinja's `tojson(indent=2)`, which uses
    `ensure_ascii=True` (every non-ASCII char -> 6-char `\\uXXXX`) and
    HTML-escapes `< > & '` to 6-char `\\u00XX`. Measuring compact
    `ensure_ascii=False` JSON undercounted PR-report prose ~2x (review
    finding, 2026-09-30). Indentation is measured at the item's own level; the
    extra nesting indent per line is small and covered by the budget's slack.
    """
    text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    return len(text) + 5 * sum(text.count(ch) for ch in _HTMLSAFE_CHARS)


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
