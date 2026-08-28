"""Provenance de-duplication helpers shared by the block-merging paths.

Lives in its own module rather than in ``windowing.py`` because
``windowing.py`` imports ``semantic_segmentation``, so the reverse import
needed to share these would be circular.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def dedup_extend(target: List[str], values: Iterable[str]) -> List[str]:
    """Append each of ``values`` to ``target`` unless already present.

    Explicitly NOT ``target.extend(v for v in values if v not in target)``:
    that only works because CPython's list.extend consumes its argument one
    item at a time, so the membership test sees earlier appends. Materialize
    the comprehension first -- or swap in a set -- and duplicates come back
    silently. Review finding, 2026-08-28; the pattern was repeated in three
    places, so it lives here once with the invariant written down.
    """
    for value in values:
        if value and value not in target:
            target.append(value)
    return target


def dedup_row_provenance(
    row_ids: Sequence[str], timestamps: Sequence[str]
) -> tuple[List[str], List[str]]:
    """Drop duplicate row_ids, keeping timestamps positionally aligned.

    Splitting text columns means one source row can back several blocks, so
    merging those blocks would otherwise make ``provenance.row_ids`` (and
    ``SegmentRecord.size``/``row_ids_count``) overcount the rows a segment
    actually covers. ``windowing._block_from_units`` already did this; review
    finding 2026-08-28 noted the two merge paths did not.
    """
    seen_ids: List[str] = []
    seen_ts: List[str] = []
    for idx, row_id in enumerate(row_ids):
        if row_id in seen_ids:
            continue
        seen_ids.append(row_id)
        if idx < len(timestamps):
            seen_ts.append(timestamps[idx])
    return seen_ids, seen_ts
