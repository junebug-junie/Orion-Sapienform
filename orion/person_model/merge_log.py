"""Merge log format with exact unmerge (spec section 5, "Reversible and replayable").

A merge-log entry lists every write that applying it would make, each with the
value it replaces (``before``). ``unmerge_ops`` inverts an entry from the entry
alone, in reverse order, so an unmerge never has to guess what the world looked
like. Nothing is ever deleted: absorbed nodes are marked, not removed.

P2 only *builds* entries (dry run). ``apply_ops`` here runs against the
``Store`` protocol; the in-memory store in the tests is the only implementation
until the live apply step (after the interlocutor reviews the cleanup).
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Protocol, Tuple

from orion.person_model.normalize import fold
from orion.schemas.person_model import EntityMergeLogEntryV1, PersonLinkageDecisionV1

_NS = uuid.UUID("a3c9e2b4-5f61-4d07-8e2a-9b1c3d4e5f60")
ADMITTED_BY = "person_linkage_v1"


def merge_id(survivor: str, absorbed: str, version: str) -> str:
    return str(uuid.uuid5(_NS, f"{survivor}|{absorbed}|{version}"))


class Store(Protocol):
    def get_prop(self, node_id: str, prop: str) -> Optional[object]: ...
    def set_prop(self, node_id: str, prop: str, value: Optional[object]) -> None: ...
    def has_edge(self, src: str, dst: str, rel: str) -> bool: ...
    def add_edge(self, src: str, dst: str, rel: str) -> None: ...
    def remove_edge(self, src: str, dst: str, rel: str) -> None: ...
    def get_alias(self, node_id: str, alias_norm: str) -> Optional[Dict[str, object]]: ...
    def put_alias(self, node_id: str, alias_norm: str, row: Optional[Dict[str, object]]) -> None: ...


class MergeConflict(ValueError):
    """The absorbed node is already merged into a different survivor."""


def plan_ops(d: PersonLinkageDecisionV1, store: Store, *, now_iso: str,
             created_aliases: Optional[Dict[Tuple[str, str], str]] = None,
             merge_id_: str = "") -> List[Dict[str, object]]:
    """Ops for one decision, with ``before`` read from ``store`` (read-only).

    ``created_aliases`` maps (survivor, alias_norm) to the merge that created
    the alias earlier in the same plan; a later merge that needs the same alias
    records a ``claim_alias`` instead of a second write, so undoing either one
    alone keeps the alias while the other still needs it."""
    survivor, absorbed = d.anchor_id, d.candidate_id
    created_aliases = created_aliases if created_aliases is not None else {}
    if d.relation == "same_as":
        current = store.get_prop(absorbed, "merged_into")
        if current not in (None, survivor):
            raise MergeConflict(f"{absorbed} is already merged into {current}")
        alias_norm = fold(d.candidate_label)
        ops: List[Dict[str, object]] = [
            {"op": "set_prop", "node_id": absorbed, "prop": "merged_into", "value": survivor,
             "before": store.get_prop(absorbed, "merged_into")},
            {"op": "add_edge", "src": absorbed, "dst": survivor, "rel": "same_as",
             "existed": store.has_edge(absorbed, survivor, "same_as")},
        ]
        key = (survivor, alias_norm)
        if key in created_aliases:
            ops.append({"op": "claim_alias", "node_id": survivor, "alias_norm": alias_norm, "created_by": created_aliases[key]})
        elif store.get_alias(survivor, alias_norm) is None:
            # An alias the survivor already had is left exactly as it is.
            ops.append({"op": "put_alias", "node_id": survivor, "alias_norm": alias_norm,
                        "value": {"alias_text": d.candidate_label, "alias_class": "name", "admitted_by": ADMITTED_BY},
                        "before": None})
            created_aliases[key] = merge_id_
        return ops
    if d.relation == "facet_of":
        return [{"op": "add_edge", "src": absorbed, "dst": survivor, "rel": "about",
                 "existed": store.has_edge(absorbed, survivor, "about")}]
    if d.relation == "junk":
        return [{"op": "set_prop", "node_id": absorbed, "prop": "linkage_flag", "value": "junk",
                 "before": store.get_prop(absorbed, "linkage_flag")}]
    if d.relation == "not_home":
        # candidate_id is "<place node>#alias:<alias_norm>"; the alias stops being valid now.
        node_id, alias_norm = _split_alias_ref(d.candidate_id)
        before = store.get_alias(node_id, alias_norm)
        if before is None:
            raise ValueError(f"not_home: alias {alias_norm!r} on {node_id} does not exist")
        after = dict(before)
        after["valid_until"] = now_iso
        return [{"op": "put_alias", "node_id": node_id, "alias_norm": alias_norm, "value": after, "before": before}]
    return []


def _split_alias_ref(ref: str) -> Tuple[str, str]:
    node_id, _, alias = ref.partition("#alias:")
    if not alias:
        raise ValueError(f"not an alias ref: {ref}")
    return node_id, alias


def build_entry(d: PersonLinkageDecisionV1, store: Store, *, decided_by: str, decision: str, now_iso: str,
                created_aliases: Optional[Dict[Tuple[str, str], str]] = None) -> EntityMergeLogEntryV1:
    if d.relation not in ("same_as", "facet_of", "junk", "not_home"):
        raise ValueError(f"no merge-log entry for relation {d.relation}")
    mid = merge_id(d.anchor_id, d.candidate_id, d.linkage_version)
    return EntityMergeLogEntryV1(
        merge_id=mid,
        survivor_node_id=d.anchor_id,
        absorbed_node_id=d.candidate_id,
        relation=d.relation,
        score=d.score,
        features_json=list(d.features),
        linkage_version=d.linkage_version,
        decision=decision,  # type: ignore[arg-type]
        decided_by=decided_by,
        ops=plan_ops(d, store, now_iso=now_iso, created_aliases=created_aliases, merge_id_=mid),
    )


class OverlayStore:
    """Reads fall through to ``base`` unless this plan already wrote the key, so
    entries planned in sequence see each other's writes. Never writes ``base``."""

    def __init__(self, base: Store):
        self.base = base
        self.props: Dict[Tuple[str, str], object] = {}
        self.edges: Dict[Tuple[str, str, str], bool] = {}
        self.aliases: Dict[Tuple[str, str], object] = {}

    def get_prop(self, n, p):
        return self.props[(n, p)] if (n, p) in self.props else self.base.get_prop(n, p)

    def set_prop(self, n, p, v):
        self.props[(n, p)] = v

    def has_edge(self, s, d, r):
        return self.edges[(s, d, r)] if (s, d, r) in self.edges else self.base.has_edge(s, d, r)

    def add_edge(self, s, d, r):
        self.edges[(s, d, r)] = True

    def remove_edge(self, s, d, r):
        self.edges[(s, d, r)] = False

    def get_alias(self, n, a):
        return self.aliases[(n, a)] if (n, a) in self.aliases else self.base.get_alias(n, a)

    def put_alias(self, n, a, row):
        self.aliases[(n, a)] = row


def plan_entries(decisions: List[PersonLinkageDecisionV1], store: Store, *, now_iso: str,
                 decided_by=lambda d: "person_linkage_v1", decision=lambda d: "auto"):
    """Entries for ``decisions`` in order, each planned against the store as the
    earlier entries would leave it. Returns (entries, conflicts)."""
    overlay = OverlayStore(store)
    created: Dict[Tuple[str, str], str] = {}
    entries, conflicts = [], []
    for d in decisions:
        try:
            e = build_entry(d, overlay, decided_by=decided_by(d), decision=decision(d), now_iso=now_iso, created_aliases=created)
        except MergeConflict as exc:
            conflicts.append((d, str(exc)))
            continue
        apply_ops(overlay, e.ops)
        entries.append(e)
    return entries, conflicts


def apply_ops(store: Store, ops: List[Dict[str, object]]) -> None:
    for op in ops:
        kind = op["op"]
        if kind == "set_prop":
            store.set_prop(op["node_id"], op["prop"], op["value"])  # type: ignore[arg-type]
        elif kind == "add_edge":
            store.add_edge(op["src"], op["dst"], op["rel"])  # type: ignore[arg-type]
        elif kind == "remove_edge":
            store.remove_edge(op["src"], op["dst"], op["rel"])  # type: ignore[arg-type]
        elif kind == "put_alias":
            store.put_alias(op["node_id"], op["alias_norm"], op["value"])  # type: ignore[arg-type]
        elif kind == "claim_alias":
            continue  # bookkeeping only: the alias was written by an earlier entry
        else:
            raise ValueError(f"unknown op {kind}")


def _alias_users(entries, key) -> int:
    return sum(
        1 for e in entries for op in e.ops
        if op["op"] in ("claim_alias", "put_alias") and (op["node_id"], op["alias_norm"]) == key
        and (op["op"] == "claim_alias" or op.get("before") is None)
    )


def unmerge_ops(entry: EntityMergeLogEntryV1, still_active: List[EntityMergeLogEntryV1] = ()) -> List[Dict[str, object]]:
    """Exact inverse of ``entry.ops``, in reverse order.

    ``still_active``: the other applied entries that stay applied. An alias this
    entry created, or claimed from an earlier entry, is removed only when no
    still-active entry needs it, so entries can be undone one at a time in any
    order and the last undo restores the original store exactly."""
    others = [e for e in still_active if e.merge_id != entry.merge_id]
    inv: List[Dict[str, object]] = []
    for op in reversed(entry.ops):
        kind = op["op"]
        if kind == "set_prop":
            inv.append({"op": "set_prop", "node_id": op["node_id"], "prop": op["prop"], "value": op.get("before")})
        elif kind == "add_edge":
            if not op.get("existed"):
                inv.append({"op": "remove_edge", "src": op["src"], "dst": op["dst"], "rel": op["rel"]})
        elif kind == "put_alias":
            key = (op["node_id"], op["alias_norm"])
            if op.get("before") is None and _alias_users(others, key):
                continue  # another applied merge still needs this alias
            inv.append({"op": "put_alias", "node_id": op["node_id"], "alias_norm": op["alias_norm"], "value": op.get("before")})
        elif kind == "claim_alias":
            key = (op["node_id"], op["alias_norm"])
            if not _alias_users(others, key):
                inv.append({"op": "put_alias", "node_id": op["node_id"], "alias_norm": op["alias_norm"], "value": None})
        else:
            raise ValueError(f"cannot invert op {kind}")
    return inv
