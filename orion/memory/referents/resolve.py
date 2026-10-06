"""Which substrate node does a writer referent mean? (memory Stage 2 spec 1.2-1.3). Pure.

Postgres ``referent_alias`` is the source of truth: a node exists because it has a
``key`` alias row, and its promotion state is that row's state. Resolution, per
referent ``{key, aliases}``:

1. exact key: a key row for this key (any state but rejected) -> that node;
2. kind refinement: same slug under an interchangeable kind (project/service) -> that
   node, plus a key row for the new key. Not a merge: the same thing, filed again;
3. name match: the key's own name or one of its grounded aliases equals a live proper
   name (never a descriptor) of exactly one node of a compatible kind -> that node;
4. ambiguous (two or more nodes, or a node of an incompatible kind) -> a NEW node in
   state ``proposed`` plus an identity question. Nothing merges;
5. otherwise a new ``provisional`` node.

Aliases on the resolved node:
- ``alias_grounding_v1``: an alias found word-bounded in one of Juniper's verified
  prompt quotes becomes ``provisional`` (kill switch: ``grounding_auto_accept``);
  anything else stays ``proposed``, which never resolves and is never indexed;
- a descriptor (aliases.alias_class) lives until ``last use + 90 days``;
- collision: an alias already live on ANOTHER node is stored ``proposed`` on this
  one and becomes a question. Never a silent merge.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .aliases import DESCRIPTOR_TTL_DAYS, alias_class, alias_in_text, normalize_alias, slug_text, TokenFrequency

REFERENT_PRODUCER = "memory.referents"
REFERENT_NAMESPACE = uuid.UUID("2c0f3c55-8a1e-4f60-b0d2-7a5e1c9d4b13")
LIVE_STATES = frozenset({"provisional", "canonical"})

# Kinds whose identity question goes to Juniper in conversation (people, places and events
# are hers to name). Every other kind is something Orion can investigate itself.
JUNIPER_ANSWERS_KINDS = frozenset({"person", "place", "event"})
# The same slug under these kinds is one thing filed twice ("project:orion-hub" and
# "service:orion-hub"), so resolution step 2 joins them.
INTERCHANGEABLE_KINDS = frozenset({"project", "service"})
# Juniper and Orion are in nearly every memory; a relationship "with Juniper" says nothing.
SELF_REFERENT_KEYS = frozenset({"person:juniper", "person:orion"})


def node_id_for_key(key: str) -> str:
    """Stable at mint, never recomputed from a later label (#2497: ids are not label hashes;
    the ambiguity path mints with a different seed)."""
    return f"referent-{uuid.uuid5(REFERENT_NAMESPACE, key)}"


def kind_of(key: str) -> str:
    return str(key).partition(":")[0]


def compatible(kind_a: str, kind_b: str) -> bool:
    return kind_a == kind_b or (kind_a in INTERCHANGEABLE_KINDS and kind_b in INTERCHANGEABLE_KINDS)


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
    via: str                  # exact_key | kind_refined | name_match | ambiguous | minted
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

    def live_names(self, alias_norm: str, now: datetime) -> list[AliasRow]:
        return [r for r in self.rows.values() if r.alias_norm == alias_norm and r.alias_class != "key"
                and r.live(now)]

    def live_name_norms(self, node_id: str, now: datetime) -> list[str]:
        return sorted(r.alias_norm for r in self.rows.values()
                      if r.node_id == node_id and r.alias_class != "key" and r.live(now))


@dataclass(frozen=True)
class CandidateAlias:
    text: str
    norm: str
    cls: str
    grounded_in: Optional[str]


def candidate_aliases(key: str, aliases: Iterable[str], prompt_quotes: list[tuple[str, str]],
                      frequency: dict[str, TokenFrequency]) -> list[CandidateAlias]:
    """The key's own name plus the writer's aliases, classified and checked for grounding.

    ``prompt_quotes``: (source_id, quote) of verified chat_prompt evidence: Juniper's words.
    ``frequency``: first-token document frequency in Juniper's prompts (aliases.alias_class).
    """
    from .aliases import first_token

    out: dict[str, CandidateAlias] = {}
    for text in [slug_text(key), *aliases]:
        norm = normalize_alias(text)
        if not norm or norm in out or ":" in norm:
            continue
        grounded = next((sid for sid, quote in prompt_quotes if alias_in_text(norm, quote)), None)
        token = first_token(norm)
        # The writer's own key name for the thing ("circe" for project:circe) is its proper
        # name, however often Juniper says it. Live 2026-10-06: "circe", "chicago" and "space"
        # are frequent in her prompts and were wrongly classed as relative descriptors.
        is_key_name = norm == slug_text(key)
        cls = "name" if is_key_name else alias_class(norm, frequency.get(token) if token else None)
        out[norm] = CandidateAlias(text=str(text).strip() or norm, norm=norm, cls=cls,
                                   grounded_in=f"chat_prompt:{grounded}" if grounded else None)
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
    grounded = [c for c in candidates if c.grounded_in]

    def finish(res: Resolution) -> Resolution:
        _admit_aliases(res, kind, candidates, index, now=now, last_use=last_use, policy=policy,
                       proposed_by=proposed_by)
        return res

    existing = index.key_row(key)
    if existing is not None:
        return finish(Resolution(key=key, node_id=existing.node_id, minted=False, via="exact_key"))

    slug = key.partition(":")[2]
    if kind in INTERCHANGEABLE_KINDS:
        for other_kind in sorted(INTERCHANGEABLE_KINDS - {kind}):
            other = index.key_row(f"{other_kind}:{slug}")
            if other is not None and other.promotion_state in LIVE_STATES:
                row = AliasRow(node_id=other.node_id, alias_norm=key, alias_text=key, alias_class="key",
                               referent_kind=other.referent_kind, promotion_state=other.promotion_state,
                               admitted_by="kind_refined", proposed_by=proposed_by)
                index.add(row)
                return finish(Resolution(key=key, node_id=other.node_id, minted=False, via="kind_refined",
                                         new_rows=[row]))

    names = {slug_text(key)} | {c.norm for c in grounded}
    matched: dict[str, AliasRow] = {}
    for norm in sorted(names):
        for row in index.live_names(norm, now):
            # A relative name ("my boss") never decides identity: who it points at changes.
            # It can only collide with another node's name (-> a question), never resolve.
            if row.alias_class == "name":
                matched.setdefault(row.node_id, row)
    same = [n for n in matched if compatible(kind, index.node_kind(n) or "")]
    other = [n for n in matched if n not in same]
    if len(same) == 1 and not other:
        return finish(Resolution(key=key, node_id=same[0], minted=False, via="name_match"))

    ambiguous = bool(matched)
    node_id = node_id_for_key(key) if not ambiguous else node_id_for_key(f"{key}|ambiguous")
    state = "proposed" if ambiguous else "provisional"
    row = AliasRow(node_id=node_id, alias_norm=key, alias_text=key, alias_class="key", referent_kind=kind,
                   promotion_state=state, admitted_by="ambiguous" if ambiguous else "minted",
                   proposed_by=proposed_by)
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
        elsewhere = [r for r in index.live_names(cand.norm, now) if r.node_id != res.node_id]
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
