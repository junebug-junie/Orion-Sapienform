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


def plan_ops(d: PersonLinkageDecisionV1, store: Store, *, now_iso: str) -> List[Dict[str, object]]:
    """Ops for one decision, with ``before`` read from ``store`` (read-only)."""
    survivor, absorbed = d.anchor_id, d.candidate_id
    if d.relation == "same_as":
        alias_norm = fold(d.candidate_label)
        return [
            {"op": "set_prop", "node_id": absorbed, "prop": "merged_into", "value": survivor,
             "before": store.get_prop(absorbed, "merged_into")},
            {"op": "add_edge", "src": absorbed, "dst": survivor, "rel": "same_as",
             "existed": store.has_edge(absorbed, survivor, "same_as")},
            {"op": "put_alias", "node_id": survivor, "alias_norm": alias_norm,
             "value": {"alias_text": d.candidate_label, "alias_class": "name", "admitted_by": ADMITTED_BY},
             "before": store.get_alias(survivor, alias_norm)},
        ]
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
        after = dict(before or {})
        after["valid_until"] = now_iso
        return [{"op": "put_alias", "node_id": node_id, "alias_norm": alias_norm, "value": after, "before": before}]
    return []


def _split_alias_ref(ref: str) -> Tuple[str, str]:
    node_id, _, alias = ref.partition("#alias:")
    if not alias:
        raise ValueError(f"not an alias ref: {ref}")
    return node_id, alias


def build_entry(d: PersonLinkageDecisionV1, store: Store, *, decided_by: str, decision: str, now_iso: str) -> EntityMergeLogEntryV1:
    if d.relation not in ("same_as", "facet_of", "junk", "not_home"):
        raise ValueError(f"no merge-log entry for relation {d.relation}")
    return EntityMergeLogEntryV1(
        merge_id=merge_id(d.anchor_id, d.candidate_id, d.linkage_version),
        survivor_node_id=d.anchor_id,
        absorbed_node_id=d.candidate_id,
        relation=d.relation,
        score=d.score,
        features_json=list(d.features),
        linkage_version=d.linkage_version,
        decision=decision,  # type: ignore[arg-type]
        decided_by=decided_by,
        ops=plan_ops(d, store, now_iso=now_iso),
    )


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
        else:
            raise ValueError(f"unknown op {kind}")


def unmerge_ops(entry: EntityMergeLogEntryV1) -> List[Dict[str, object]]:
    """Exact inverse of ``entry.ops``, in reverse order, from the entry alone."""
    inv: List[Dict[str, object]] = []
    for op in reversed(entry.ops):
        kind = op["op"]
        if kind == "set_prop":
            inv.append({"op": "set_prop", "node_id": op["node_id"], "prop": op["prop"], "value": op.get("before")})
        elif kind == "add_edge":
            if not op.get("existed"):
                inv.append({"op": "remove_edge", "src": op["src"], "dst": op["dst"], "rel": op["rel"]})
        elif kind == "put_alias":
            inv.append({"op": "put_alias", "node_id": op["node_id"], "alias_norm": op["alias_norm"], "value": op.get("before")})
        else:
            raise ValueError(f"cannot invert op {kind}")
    return inv
