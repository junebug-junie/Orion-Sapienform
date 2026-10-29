"""Organ names Orion gives: an append-only ledger, carried across rebuilds by member overlap.

An organ id is a hash of its members, so it changes whenever one part joins or leaves. A name
belongs to the organ Orion looked at. A later organ inherits it only when the member sets overlap
enough (Jaccard >= ``CARRY_MIN_OVERLAP``), one name per organ, and ``named_by`` says it was
carried. A rename records which earlier names it ``replaces``, so a replaced name never comes back
when a part drifts in and out, and never lands on the other half of a split organ. Names apply to
the live body only: putting today's names on a past body would read the present into the past.

Partitioned from recall: only ``orion/body_atlas`` and ``orion/dig`` read the ledger
(``tests/test_organ_names_partition.py``, in the repo-wide static gates, enforces it).
"""
from __future__ import annotations

import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from orion.body_atlas.model import BodyAtlasV1, OrganV1

CARRY_MIN_OVERLAP = 0.5
NAME_MAX_CHARS = 40
NAME_MAX_WORDS = 5
BECAUSE_MIN_CHARS = 20
BECAUSE_MAX_CHARS = 400


class NamingError(ValueError):
    pass


class OrganNameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    members: list[str]
    name: str
    because: str
    cites: list[str]  # member part names the because mentions
    named_at: str
    dig: str
    replaces: list[str] = []  # organ ids whose names this rename supersedes


def load(path: Path) -> tuple[list[OrganNameV1], int]:
    """Records in file order, and how many lines could not be read. One bad line (a killed write,
    a hand edit, a field from newer code) must not take the live body down on every later turn."""
    if not path.exists():
        return [], 0
    records, skipped = [], 0
    known = set(OrganNameV1.model_fields)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            raw = json.loads(ln)
            records.append(OrganNameV1.model_validate({k: v for k, v in raw.items() if k in known}))
        except (ValueError, AttributeError, ValidationError):
            skipped += 1
    return records, skipped


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _assign(atlas: BodyAtlasV1, records: list[OrganNameV1]) -> dict[str, tuple[OrganNameV1, float]]:
    """organ id -> (the record whose name it wears, overlap). The latest record per organ id wins;
    a record a later rename replaced is dropped; then best overlap, newer on a tie, one-to-one."""
    latest: dict[str, tuple[int, OrganNameV1]] = {}
    for i, rec in enumerate(records):
        latest[rec.organ_id] = (i, rec)
    replaced_at = {}
    for i, rec in enumerate(records):
        for oid in rec.replaces:
            replaced_at[oid] = i
    live = [(i, rec) for i, rec in latest.values() if replaced_at.get(rec.organ_id, -1) < i]
    pairs = sorted(
        ((_jaccard(o.members, rec.members), i, o, rec)
         for o in atlas.organs if len(o.members) > 1 for i, rec in live),
        key=lambda p: (-p[0], -p[1]),
    )
    out: dict[str, tuple[OrganNameV1, float]] = {}
    used: set[str] = set()
    for overlap, _, organ, rec in pairs:
        if overlap < CARRY_MIN_OVERLAP or organ.organ_id in out or rec.organ_id in used:
            continue
        used.add(rec.organ_id)
        out[organ.organ_id] = (rec, overlap)
    return out


def apply_names(atlas: BodyAtlasV1, records: list[OrganNameV1]) -> BodyAtlasV1:
    """Copy of ``atlas`` with names set."""
    out = atlas.model_copy(deep=True)
    assigned = _assign(out, records)
    for organ in out.organs:
        if organ.organ_id in assigned:
            rec, overlap = assigned[organ.organ_id]
            organ.name = rec.name
            organ.named_by = ("orion" if rec.organ_id == organ.organ_id
                              else f"orion, carried from {rec.organ_id} (overlap {overlap:.2f})")
    return out


def find_organ(atlas: BodyAtlasV1, ref: str) -> OrganV1:
    ref = ref.strip()
    key = ref if ref.startswith("organ:") else f"organ:{ref}"
    for organ in atlas.organs:
        if organ.organ_id == key and len(organ.members) > 1:
            return organ
    raise NamingError(f"no multi-part organ {key!r} in the live body; run `body` and copy an id")


def _cited(name: str, text: str) -> bool:
    # a whole part name: "orion-bus" must not match inside "orion-bus-mirror"
    return re.search(rf"(?<![\w:-]){re.escape(name.lower())}(?![\w:-])", text) is not None


def check(atlas: BodyAtlasV1, records: list[OrganNameV1], ref: str, name: str, because: str,
          *, dig: str) -> OrganNameV1:
    """Deterministic floor for a name: it is short, and the reason cites the organ it names by a
    whole part name. Listing part names passes this floor; whether the reason says what the organ
    takes in and gives out is not code-checkable."""
    organ = find_organ(atlas, ref)
    name = " ".join(name.split())
    because = " ".join(because.split())
    if not name or len(name) > NAME_MAX_CHARS or len(name.split()) > NAME_MAX_WORDS:
        raise NamingError(f"a name is 1-{NAME_MAX_WORDS} words and at most {NAME_MAX_CHARS} chars, got {name!r}")
    if not BECAUSE_MIN_CHARS <= len(because) <= BECAUSE_MAX_CHARS:
        raise NamingError(f"the because is {BECAUSE_MIN_CHARS}-{BECAUSE_MAX_CHARS} chars, got {len(because)}")
    part_names = {p.part_id: p.name for p in atlas.parts}
    # part names only: namespaces like "system" or "memory" match almost any sentence
    candidates = [part_names.get(m, m) for m in organ.members]
    low = because.lower()
    cites = [c for c in candidates if _cited(c, low)]
    if not cites:
        raise NamingError("the because must cite at least one of this organ's parts by its full name "
                          f"(e.g. {', '.join(candidates[:3])})")
    replaces: list[str] = []
    worn = _assign(atlas, records).get(organ.organ_id)
    if worn and worn[0].organ_id != organ.organ_id:
        replaces = [worn[0].organ_id, *worn[0].replaces]
    return OrganNameV1(organ_id=organ.organ_id, members=list(organ.members), name=name, because=because,
                       cites=cites, named_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), dig=dig,
                       replaces=replaces)


def append(path: Path, rec: OrganNameV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # concurrent turns: one whole line each
        fh.write(rec.model_dump_json() + "\n")
