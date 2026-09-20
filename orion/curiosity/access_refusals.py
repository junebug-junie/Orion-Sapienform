"""Count access-refusal hop notes for hire-handoff pressure.

Allow-list substrings (case-insensitive match within each note):
``permission denied``, ``permissiondenied``, ``acl``, ``insufficient_privilege``.
"""

from __future__ import annotations

from typing import Sequence

ACCESS_REFUSAL_THRESHOLD = 2
_NEEDLES = (
    "permission denied",
    "permissiondenied",
    "acl",
    "insufficient_privilege",
)


def count_access_refusals(notes: Sequence[str]) -> int:
    n = 0
    for note in notes:
        low = str(note).lower()
        if any(needle in low for needle in _NEEDLES):
            n += 1
    return n
