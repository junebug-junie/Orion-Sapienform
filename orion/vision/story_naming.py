"""Who a walkway motion story is about: the one naming rule, shared.

orion-sql-writer (app/vision_motion.py) names a story when it writes it; the
Hub (scripts/motion_story_routes.py) re-derives the name when it shows the
story, with the names individuals carry NOW, so a name Juniper gives later
shows on every story it covers. Both must use the same rule, so it lives here.

The rule (motion stories proposal 2026-10-16, rule 6), over the story's crops:

- every crop that carries an individual id carries a NAMED one, and all of
  them carry the same name (case-insensitive). An unnamed id next to a named
  one: ``other_id``; two names: ``two_named``. Several ids of one name are fine.
- no crop touches the frame edge (``edge_clipped``);
- the name covers >= ``share`` (2/3) of all crops (``low_share``).

Crops with no id at all do not block a name. Pure: no I/O.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

NAME_SHARE = 2.0 / 3.0
NOUNS = {"person": "a person", "dog": "a dog", "bicycle": "a bicycle", "stroller": "a stroller",
         "vehicle": "a vehicle"}


def name_from_crops(crops: Sequence[Tuple[Optional[str], bool]], labels: Dict[str, str],
                    share: float = NAME_SHARE) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(individual_id, label, withheld_reason) for crops given as
    ``(individual_id, edge_clipped)``. The returned id is the one on most crops."""
    ids = [iid for iid, _ in crops if iid]
    named = [i for i in ids if (labels.get(i) or "").strip()]
    if not named:
        return None, None, None
    if len({labels[i].strip().casefold() for i in named}) > 1:
        return None, None, "two_named"
    if len(named) != len(ids):
        return None, None, "other_id"
    if any(clipped for _, clipped in crops):
        return None, None, "edge_clipped"
    if not crops or len(named) / len(crops) < share:
        return None, None, "low_share"
    iid = max(sorted(set(named)), key=named.count)
    return iid, labels[iid].strip(), None


def evidence_crops(evidence: Iterable[dict]) -> list[Tuple[Optional[str], bool]]:
    """A stored story's ``evidence`` list as ``name_from_crops`` input."""
    out = []
    for e in evidence or []:
        if isinstance(e, dict):
            out.append((e.get("individual_id") or None, bool(e.get("clipped"))))
    return out


def subject(kind: str, name: Optional[str]) -> str:
    """The sentence's subject, capitalized: the name, else "A person"."""
    s = (name or "").strip() or NOUNS.get(kind, f"a {kind}")
    return s[0].upper() + s[1:]


def rename_subject(sentence: str, kind: str, old_name: Optional[str], new_name: Optional[str]) -> str:
    """The stored sentence with its subject swapped from ``old_name`` (the name it
    was written with, or none) to ``new_name``. Unchanged if the sentence does not
    start with the old subject (never guess where a name sits)."""
    old = subject(kind, old_name)
    new = subject(kind, new_name)
    if old == new or not sentence.startswith(old):
        return sentence
    rest = sentence[len(old):]
    if rest and rest[0].isalnum():  # "Al" must not match "Alex ..."
        return sentence
    return new + rest
