"""One record per person: which camera clusters point at which person (2026-10-22).

Spec: docs/superpowers/specs/2026-10-22-people-one-record-design.md.

Pure planner. The caller reads the named camera clusters, the "same person?"
verdicts, the people in ``config/person_model/people.yaml``, and the current
``person`` / ``person_link`` rows, and ``plan()`` says what to write. The
writer is ``services/orion-sql-writer/app/person_record_loop.py``.

Rules (the dangerous failure is two real people merged under one name):

- A link comes only from a name the interlocutor typed (``vision_individual.label``) or a
  ``people.yaml`` row. Never from a picture match or an embedding.
- A typed name joins the person whose name folds the same (``normalize.fold``).
  The camera albums already treat one name as one person, and the interlocutor does not
  answer a card per cluster pair (PR #2847's reasoning).
- Guard: a cluster never joins a person when the interlocutor said "different people"
  about it and one of that person's clusters. Two people with one name stay
  unlinked and are reported as a conflict, never merged.
- A rename revokes the old link and writes the new one. A cleared name revokes.
- Only person-like detector kinds become people. A named bicycle or car whose
  name is a known person is linked as that person's belonging; any other named
  thing ("no", "I do not know") is ignored.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from orion.person_model.normalize import fold
from orion.schemas.person_model import PersonRefV1

PERSON_KINDS = frozenset({"person"})
SOURCE_INDIVIDUAL = "vision_individual"
SOURCE_BELONGING = "vision_belonging"
DECIDED_CAMERA = "camera_name"
DECIDED_PEOPLE_YAML = "people_yaml"
_NAMESPACE = uuid.UUID("6f1c2a64-3c1e-5b8e-9f3a-0c2e7d4b9a11")


@dataclass(frozen=True)
class Cluster:
    individual_id: str
    kind: str
    label: str
    label_ask_id: Optional[str] = None


@dataclass(frozen=True)
class Person:
    person_id: str
    display_name: str
    role: str = "other"
    status: str = "active"


@dataclass(frozen=True)
class Link:
    person_id: str
    source_kind: str
    source_ref: str
    decided_by: str
    evidence_ref: Optional[str] = None

    @property
    def link_id(self) -> str:
        raw = f"{self.source_kind}|{self.source_ref}|{self.person_id}|{self.evidence_ref or ''}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Conflict:
    individual_id: str
    name_key: str
    reason: str                 # different_verdict | same_name_people
    person_ids: Tuple[str, ...]


@dataclass
class Plan:
    new_persons: List[Person] = field(default_factory=list)
    add_links: List[Link] = field(default_factory=list)
    revoke: List[Tuple[str, str]] = field(default_factory=list)   # (source_kind, source_ref)
    conflicts: List[Conflict] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.new_persons or self.add_links or self.revoke)


def name_key(name: str) -> str:
    return fold(name or "")


def minted_person_id(key: str) -> str:
    return f"person-{uuid.uuid5(_NAMESPACE, key)}"


def plan(
    clusters: Iterable[Cluster],
    different_pairs: Iterable[Tuple[str, str]],
    people: Iterable[PersonRefV1],
    persons: Iterable[Person],
    active_links: Dict[Tuple[str, str], str],
) -> Plan:
    """What to write so ``person``/``person_link`` match the named clusters.

    ``active_links`` maps (source_kind, source_ref) -> person_id for links
    with ``revoked_at IS NULL``. Deterministic: same input, same plan."""
    out = Plan()
    by_id: Dict[str, Person] = {p.person_id: p for p in persons}
    for ref in people:
        if ref.node_id not in by_id:
            p = Person(ref.node_id, ref.display_name, ref.role)
            by_id[p.person_id] = p
            out.new_persons.append(p)
    # A people.yaml person owns its name: a camera-minted row with the same name
    # (minted before the yaml row existed) stops being a candidate, so the name
    # does not split into two people and its links move to the referent.
    yaml_ids = {ref.node_id for ref in people}
    by_key: Dict[str, List[str]] = {}
    for p in sorted(by_id.values(), key=lambda p: p.person_id):
        by_key.setdefault(name_key(p.display_name), []).append(p.person_id)
    for key, ids in by_key.items():
        preferred = [i for i in ids if i in yaml_ids]
        if preferred:
            by_key[key] = preferred

    different: Dict[str, Set[str]] = {}
    for a, b in different_pairs:
        different.setdefault(a, set()).add(b)
        different.setdefault(b, set()).add(a)

    links = dict(active_links)
    clusters_of: Dict[str, Set[str]] = {}
    for (kind, ref), pid in links.items():
        if kind == SOURCE_INDIVIDUAL:
            clusters_of.setdefault(pid, set()).add(ref)

    def unlink(kind: str, ref: str) -> None:
        pid = links.pop((kind, ref), None)
        if pid is not None:
            clusters_of.get(pid, set()).discard(ref)
            out.revoke.append((kind, ref))

    def link(person_id: str, kind: str, c: Cluster) -> None:
        links[(kind, c.individual_id)] = person_id
        if kind == SOURCE_INDIVIDUAL:
            clusters_of.setdefault(person_id, set()).add(c.individual_id)
        out.add_links.append(Link(person_id, kind, c.individual_id, DECIDED_CAMERA, c.label_ask_id))

    named = sorted((c for c in clusters if name_key(c.label)), key=lambda c: c.individual_id)
    people_clusters = [c for c in named if c.kind in PERSON_KINDS]
    seen_people = {c.individual_id for c in people_clusters}
    for c in people_clusters:
        key = name_key(c.label)
        current = links.get((SOURCE_INDIVIDUAL, c.individual_id))
        clash = different.get(c.individual_id, set())
        candidates = list(by_key.get(key, []))
        if (current is not None and current in candidates
                and not (clash & clusters_of.get(current, set()))):
            continue
        allowed = [pid for pid in candidates if not (clash & clusters_of.get(pid, set()))]
        if current is not None:
            unlink(SOURCE_INDIVIDUAL, c.individual_id)
        if len(allowed) == 1:
            link(allowed[0], SOURCE_INDIVIDUAL, c)
        elif not candidates:
            p = Person(minted_person_id(key), " ".join(c.label.split()))
            by_id[p.person_id] = p
            by_key[key] = [p.person_id]
            out.new_persons.append(p)
            link(p.person_id, SOURCE_INDIVIDUAL, c)
        else:
            reason = "different_verdict" if not allowed else "same_name_people"
            out.conflicts.append(Conflict(c.individual_id, key, reason, tuple(candidates)))

    seen_belongings: Set[str] = set()
    for c in named:
        if c.kind in PERSON_KINDS:
            continue
        candidates = by_key.get(name_key(c.label), [])
        current = links.get((SOURCE_BELONGING, c.individual_id))
        target = candidates[0] if len(candidates) == 1 else None
        if target is not None:
            seen_belongings.add(c.individual_id)
        if current == target:
            continue
        if current is not None:
            unlink(SOURCE_BELONGING, c.individual_id)
        if target is not None:
            link(target, SOURCE_BELONGING, c)

    # A cluster whose name was cleared, or that is gone, no longer points anywhere.
    for (kind, ref) in sorted(k for k in active_links if k in links):
        if kind == SOURCE_INDIVIDUAL and ref not in seen_people:
            unlink(kind, ref)
        elif kind == SOURCE_BELONGING and ref not in seen_belongings:
            unlink(kind, ref)
    return out


def clusters_from_rows(rows: Sequence[Sequence[object]]) -> List[Cluster]:
    """(individual_id, kind, label, label_ask_id) rows -> clusters."""
    return [Cluster(str(r[0]), str(r[1] or ""), str(r[2] or ""), (str(r[3]) if r[3] else None)) for r in rows]
