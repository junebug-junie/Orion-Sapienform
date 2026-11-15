"""Organ ids that stay put: an append-only ledger, carried across rebuilds by member overlap (#3402).

``build_atlas`` makes an organ's id from a hash of its members, so the id changed whenever one part
joined or left. Orion's two largest organs (about 70 and 32 parts) got a new id on each of five
daily snapshots in a row, so Orion could not follow an organ across days by id.

Here an organ of the live body takes the id of the earlier organ it overlaps most: Jaccard >=
``names.CARRY_MIN_OVERLAP`` (0.5, the rule the names and the body diff already use), best overlap
first, newer on a tie, one to one. A new id is minted only when nothing matches; it is the member
hash, so an id keeps the shape ``organ:`` + 10 hex that every reader expects. The hash itself stays
on the organ as ``member_hash``.

The ledger holds one line per organ per change of membership: the id and the members it has now.
Overlap is measured against the organ's LATEST members, so an organ that drifts one part at a time
keeps its id however far it moves from where it started. Only multi-part organs are in the ledger:
a part alone is its own hash, which cannot drift.

Ids apply to the live body only. A past day's body keeps its member hashes: matching it against
today's ledger would read the present into the past.

Partitioned like the other self ledgers: only ``orion/body_atlas`` and ``orion/dig`` read it
(``tests/test_organ_names_partition.py``).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from orion.body_atlas.model import BodyAtlasV1
from orion.body_atlas.names import CARRY_MIN_OVERLAP

LEDGER_FILE = "organ_ids.jsonl"


class OrganIdV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    member_hash: str
    members: list[str]  # sorted part ids, as of ``as_of``
    as_of: str  # the atlas that showed these members


def member_hash(members: list[str], salt: int = 0) -> str:
    """``organ:`` + 10 hex of the sorted members. ``members`` are part ids (``part:<name>``); the
    hash is over the bare names, which is the id ``build_atlas`` has always made."""
    raw = "|".join(sorted(m.removeprefix("part:") for m in members)) + (f"#{salt}" if salt else "")
    return "organ:" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def load(path: Path) -> tuple[list[OrganIdV1], int]:
    """Records in file order, and how many lines could not be read (a killed write, a hand edit):
    one bad line must not take the live body down."""
    if not path.exists():
        return [], 0
    records, skipped = [], 0
    known = set(OrganIdV1.model_fields)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], 1
    for ln in lines:
        if not ln.strip():
            continue
        try:
            raw = json.loads(ln)
            records.append(OrganIdV1.model_validate({k: v for k, v in raw.items() if k in known}))
        except (ValueError, AttributeError, ValidationError):
            skipped += 1
    return records, skipped


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def assign(atlas: BodyAtlasV1, records: list[OrganIdV1]) -> tuple[BodyAtlasV1, list[OrganIdV1]]:
    """A copy of ``atlas`` with carried ids (organs and organ flows), and the ledger lines to append:
    one per organ that is new or whose members changed. Pure; the same atlas and ledger give the
    same answer."""
    out = atlas.model_copy(deep=True)
    latest: dict[str, tuple[int, OrganIdV1]] = {}
    for i, rec in enumerate(records):
        latest[rec.organ_id] = (i, rec)
    multi = [o for o in out.organs if len(o.members) > 1]
    pairs = sorted(
        ((_jaccard(o.members, rec.members), i, j, rec) for j, o in enumerate(multi) for i, rec in latest.values()),
        key=lambda p: (-p[0], -p[1], p[2]),
    )
    took: dict[int, OrganIdV1] = {}
    used: set[str] = set()
    for overlap, _, j, rec in pairs:
        if overlap < CARRY_MIN_OVERLAP:
            break
        if j in took or rec.organ_id in used:
            continue
        used.add(rec.organ_id)
        took[j] = rec
    # An id is never handed out twice: not one the ledger ever held, not one a part alone wears now.
    taken = {r.organ_id for r in records} | {o.organ_id for o in out.organs if len(o.members) == 1}
    renamed: dict[str, str] = {}
    new: list[OrganIdV1] = []
    for j, organ in enumerate(multi):
        members = sorted(organ.members)
        hashed = member_hash(members)
        rec = took.get(j)
        if rec is not None:
            organ_id = rec.organ_id
            changed = rec.members != members
        else:
            organ_id, salt = hashed, 0
            while organ_id in taken:  # an earlier organ wore these exact members and has moved on
                salt += 1
                organ_id = member_hash(members, salt)
            changed = True
        taken.add(organ_id)
        if changed:
            new.append(OrganIdV1(organ_id=organ_id, member_hash=hashed, members=members, as_of=atlas.as_of))
        renamed[organ.organ_id] = organ_id
        organ.organ_id = organ_id
    for flow in out.organ_flows:
        flow.source = renamed.get(flow.source, flow.source)
        flow.target = renamed.get(flow.target, flow.target)
    return out, new


def append(path: Path, records: list[OrganIdV1]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(r.model_dump_json() + "\n" for r in records))


def carry(atlas: BodyAtlasV1, path: Path) -> BodyAtlasV1:
    """The live body with its ledger ids, and the ledger brought up to date. Read, assign and
    append run under one lock: two turns that see a new organ at the same moment, with members a
    part apart, must not mint two ids for it (the second reads the first's line and carries it).
    A ledger that cannot be locked or written (a read-only home) still gives carried ids for this
    read; it only cannot learn from it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = (path.parent / f".{path.stem}.lock").open("a")
    except OSError:
        return assign(atlas, load(path)[0])[0]
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out, new = assign(atlas, load(path)[0])
        try:
            append(path, new)
        except OSError:
            pass
    return out
