"""Which substrate node does a writer referent mean? (memory Stage 2 spec 1.2-1.3). Pure.

Postgres ``referent_alias`` is the source of truth: a node exists because it has a
``key`` alias row, and its promotion state is that row's state.

What kind of name each name is comes from the distiller (``alias_kind``: proper_name or
descriptor, for the key's own name and every alias). A name with no judgment is a
descriptor. Code never classifies names by their words.

Identity rule: a referent resolves onto an EXISTING node only by
1. its exact key (a key row for this key, any state but rejected); or
2. a grounded proper name (the key's own name or an alias the distiller called a
   proper_name, found in Juniper's verified words) that exactly equals a live proper name
   of exactly one node of the SAME kind, with nothing contradicting it: if the key's own
   name is itself a grounded proper name, it must belong to that same node.
Anything else that lands on another node's proper name is ambiguous: a NEW node in state
``proposed`` and an identity question. Descriptors ("my boss") never decide identity, in
either direction: they attach as expiring names, and a collision on one is a question.

Aliases on the resolved node:
- ``alias_grounding_v1``: a name found word-bounded in Juniper's verified prompt quotes
  becomes ``provisional`` (kill switch: ``grounding_auto_accept``); anything else stays
  ``proposed``, which never resolves and is never displayed;
- a descriptor lives until ``last use + 90 days``; reuse extends it;
- an alias's class is frozen at insert; a frozen descriptor still never decides identity;
- collision: an alias already live on ANOTHER node is stored ``proposed`` here and asked about.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .aliases import DESCRIPTOR_TTL_DAYS, alias_in_text, normalize_alias, slug_text

REFERENT_PRODUCER = "memory.referents"
REFERENT_NAMESPACE = uuid.UUID("2c0f3c55-8a1e-4f60-b0d2-7a5e1c9d4b13")
LIVE_STATES = frozenset({"provisional", "canonical"})

# Kinds whose identity question goes to Juniper in conversation (people, places and events
# are hers to name). Every other kind is something Orion can investigate itself.
JUNIPER_ANSWERS_KINDS = frozenset({"person", "place", "event"})
# Juniper and Orion are in nearly every memory; a relationship "with Juniper" says nothing.
SELF_REFERENT_KEYS = frozenset({"person:juniper", "person:orion"})


def node_id_for_key(key: str) -> str:
    """Stable at mint, never recomputed from a later label (#2497: ids are not label hashes;
    the ambiguity path mints with a different seed)."""
    return f"referent-{uuid.uuid5(REFERENT_NAMESPACE, key)}"


def kind_of(key: str) -> str:
    return str(key).partition(":")[0]


def alias_class_for(alias_kind: Optional[str]) -> str:
    """The stored class: 'name' only for the distiller's proper_name; everything else descriptor."""
    return "name" if alias_kind == "proper_name" else "descriptor"


@dataclass(frozen=True)
class AliasRow:
    node_id: str
    alias_norm: str
    alias_text: str
    alias_class: str          # key | name | descriptor
    referent_kind: str
    promotion_state: str
    admitted_by: str
    proposed_by: str
    grounded_in: Optional[str] = None
    valid_until: Optional[datetime] = None

    def live(self, now: datetime) -> bool:
        if self.promotion_state not in LIVE_STATES:
            return False
        return self.alias_class != "descriptor" or (self.valid_until is not None and self.valid_until > now)


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    scope: str
    answer_via: str
    referent_keys: tuple[str, ...]
    node_ids: tuple[str, ...]
    reason: str


@dataclass
class Resolution:
    key: str
    node_id: str
    minted: bool
    via: str                  # exact_key | name_match | ambiguous | minted
    new_rows: list[AliasRow] = field(default_factory=list)
    refreshed: list[AliasRow] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


@dataclass(frozen=True)
class ReferentPolicy:
    grounding_auto_accept: bool = True        # MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT
    cooccurrence_auto_accept: bool = True     # MEMORY_COOCCURRENCE_AUTO_ACCEPT


class AliasIndex:
    """The alias table as resolution sees it, updated as one persist adds rows."""

    def __init__(self, rows: Iterable[AliasRow]) -> None:
        self.rows: dict[tuple[str, str], AliasRow] = {(r.node_id, r.alias_norm): r for r in rows}

    def add(self, row: AliasRow) -> None:
        self.rows[(row.node_id, row.alias_norm)] = row

    def key_row(self, key: str) -> Optional[AliasRow]:
        found = [r for r in self.rows.values() if r.alias_class == "key" and r.alias_norm == key
                 and r.promotion_state != "rejected"]
        return min(found, key=lambda r: r.node_id) if found else None

    def node_key_row(self, node_id: str) -> Optional[AliasRow]:
        keys = [r for r in self.rows.values() if r.node_id == node_id and r.alias_class == "key"]
        return min(keys, key=lambda r: r.alias_norm) if keys else None

    def node_state(self, node_id: str) -> Optional[str]:
        row = self.node_key_row(node_id)
        return row.promotion_state if row else None

    def node_kind(self, node_id: str) -> Optional[str]:
        row = self.node_key_row(node_id)
        return row.referent_kind if row else None

    def node_label(self, node_id: str) -> str:
        row = self.node_key_row(node_id)
        return slug_text(row.alias_norm) if row else node_id

    def live_rows(self, alias_norm: str, now: datetime) -> list[AliasRow]:
        return [r for r in self.rows.values() if r.alias_norm == alias_norm and r.alias_class != "key"
                and r.live(now)]

    def live_name_norms(self, node_id: str, now: datetime) -> list[str]:
        return sorted(r.alias_norm for r in self.rows.values()
                      if r.node_id == node_id and r.alias_class != "key" and r.live(now))


@dataclass(frozen=True)
class CandidateAlias:
    text: str
    norm: str
    cls: str                  # name | descriptor (from the distiller's alias_kind)
    grounded_in: Optional[str]
    is_key_name: bool = False


def candidate_aliases(key: str, key_alias_kind: Optional[str], aliases: Iterable[tuple[str, Optional[str]]],
                      prompt_quotes: list[tuple[str, str]]) -> list[CandidateAlias]:
    """The key's own name plus the writer's aliases, each with the distiller's class and a
    grounding check against ``prompt_quotes`` ((source_id, quote) of Juniper's verified words)."""
    out: dict[str, CandidateAlias] = {}
    for text, kind, is_key in [(key.partition(":")[2], key_alias_kind, True),
                               *[(t, k, False) for t, k in aliases]]:
        norm = normalize_alias(text)
        if not norm or norm in out:
            continue
        grounded = next((sid for sid, quote in prompt_quotes if alias_in_text(norm, quote)), None)
        display = slug_text(key) if is_key else (str(text).strip() or norm)
        out[norm] = CandidateAlias(text=display, norm=norm, cls=alias_class_for(kind),
                                   grounded_in=f"chat_prompt:{grounded}" if grounded else None, is_key_name=is_key)
    return list(out.values())


def _question(kind: str, text: str, keys: tuple[str, ...], node_ids: tuple[str, ...], reason: str) -> Question:
    juniper = kind in JUNIPER_ANSWERS_KINDS
    return Question(
        question_id=str(uuid.uuid5(REFERENT_NAMESPACE, "question|" + "|".join(sorted(node_ids)) + "|" + text)),
        text=text, scope="juniper" if juniper else "self",
        answer_via="conversation" if juniper else "investigation",
        referent_keys=keys, node_ids=node_ids, reason=reason,
    )


def resolve_referent(
    key: str,
    candidates: list[CandidateAlias],
    index: AliasIndex,
    *,
    now: datetime,
    last_use: datetime,
    policy: ReferentPolicy,
    proposed_by: str = "episode_writer",
) -> Resolution:
    kind = kind_of(key)

    def finish(res: Resolution) -> Resolution:
        _admit_aliases(res, kind, candidates, index, now=now, last_use=last_use, policy=policy,
                       proposed_by=proposed_by)
        return res

    existing = index.key_row(key)
    if existing is not None:
        return finish(Resolution(key=key, node_id=existing.node_id, minted=False, via="exact_key"))

    # Identity is decided by grounded proper names only, against other nodes' proper names.
    proper = [c for c in candidates if c.cls == "name" and c.grounded_in]
    hits: dict[str, set[str]] = {}
    for cand in proper:
        for row in index.live_rows(cand.norm, now):
            if row.alias_class == "name":
                hits.setdefault(cand.norm, set()).add(row.node_id)
    matched: set[str] = set().union(*hits.values()) if hits else set()
    same_kind = {n for n in matched if index.node_kind(n) == kind}
    key_name = next((c for c in proper if c.is_key_name), None)
    # "person:taylor" + alias "Morgan" (on Morgan's node): the key's own proper name says a
    # different thing than the alias does -- that is a question, not a merge.
    contradicted = key_name is not None and hits.get(key_name.norm, set()) != same_kind
    if len(same_kind) == 1 and matched == same_kind and not contradicted:
        return finish(Resolution(key=key, node_id=next(iter(same_kind)), minted=False, via="name_match"))

    ambiguous = bool(matched)
    node_id = node_id_for_key(key) if not ambiguous else node_id_for_key(f"{key}|ambiguous")
    row = AliasRow(node_id=node_id, alias_norm=key, alias_text=key, alias_class="key", referent_kind=kind,
                   promotion_state="proposed" if ambiguous else "provisional",
                   admitted_by="ambiguous" if ambiguous else "minted", proposed_by=proposed_by)
    index.add(row)
    res = Resolution(key=key, node_id=node_id, minted=True, via="ambiguous" if ambiguous else "minted",
                     new_rows=[row])
    for other_id in sorted(matched):
        other_key = index.node_key_row(other_id)
        res.questions.append(_question(
            kind, f"Is '{slug_text(key)}' the same as '{index.node_label(other_id)}'?",
            (key, other_key.alias_norm if other_key else other_id), (node_id, other_id), "referent_identity"))
    return finish(res)


def _admit_aliases(res: Resolution, kind: str, candidates: list[CandidateAlias], index: AliasIndex, *,
                   now: datetime, last_use: datetime, policy: ReferentPolicy, proposed_by: str) -> None:
    node_state = index.node_state(res.node_id)
    for cand in candidates:
        valid_until = last_use + timedelta(days=DESCRIPTOR_TTL_DAYS) if cand.cls == "descriptor" else None
        mine = index.rows.get((res.node_id, cand.norm))
        if mine is not None:
            # Juniper used it again: a descriptor lives 90 days past its LAST use.
            if (mine.alias_class == "descriptor" and cand.grounded_in and valid_until is not None
                    and (mine.valid_until is None or valid_until > mine.valid_until)):
                refreshed = replace(mine, valid_until=valid_until)
                index.add(refreshed)
                res.refreshed.append(refreshed)
            continue
        elsewhere = [r for r in index.live_rows(cand.norm, now) if r.node_id != res.node_id]
        if elsewhere:
            state, admitted = "proposed", "collision"
            # A node minted on the ambiguity path already asks "is X the same as Y".
            for other in ([] if res.via == "ambiguous" else elsewhere):
                res.questions.append(_question(
                    kind, f"Does '{cand.text}' mean '{index.node_label(other.node_id)}' or "
                          f"'{index.node_label(res.node_id)}'?",
                    tuple(sorted({res.key, (index.node_key_row(other.node_id) or other).alias_norm})),
                    (res.node_id, other.node_id), "alias_collision"))
        elif cand.grounded_in and policy.grounding_auto_accept and node_state in LIVE_STATES:
            state, admitted = "provisional", "alias_grounding_v1"
        elif cand.grounded_in and node_state in LIVE_STATES:
            state, admitted = "proposed", "alias_grounding_v1_disabled"
        elif cand.grounded_in:
            # Grounded, but the node itself is an unanswered identity question.
            state, admitted = "proposed", "node_not_accepted"
        else:
            state, admitted = "proposed", "ungrounded"
        row = AliasRow(node_id=res.node_id, alias_norm=cand.norm, alias_text=cand.text, alias_class=cand.cls,
                       referent_kind=index.node_kind(res.node_id) or kind, promotion_state=state,
                       admitted_by=admitted, proposed_by=proposed_by, grounded_in=cand.grounded_in,
                       valid_until=valid_until)
        index.add(row)
        res.new_rows.append(row)
