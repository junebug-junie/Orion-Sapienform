"""One-time relabel of choice-built tendencies from their own stored choices (#3444). PURE: no I/O.

A tendency's label is set when choices fold into it (``orion.personality.graph._Tick.reinforce``) and
never recomputed. So label-rule fixes (#3379, 2026-11-15: no category leads, no "(+k more)" tails, a
quoted title stands alone) reach a tendency only when it gets another choice, and most tendencies are
1-of-1. This module rebuilds a label the way the reducer would have built it under the CURRENT rules,
from the node's own evidence:

* each evidence row's excerpt is remade with the current ``pair_label`` from the stored choice
  (``personality_choice``): chosen text and declined texts in shown order (``choice_sides``). A Prime
  option text from before 2026-11-15 is first remade with Prime's current ``option_text`` (subject first);
  that rewrite is a fixed point on current texts;
* the founding choice is the base, at ``match_threshold`` similarity (``_Tick.create``);
* each later choice, in fold order (commit time, then occurred_at, then id), takes the base as
  ``reinforce`` does: a closer match than the base so far, or a subject over a category-only base, and
  never a category-only excerpt over a subject. Juniper-"no" absorbs and own-name echoes never take it;
* counts stay what the node has ("n of m": ``evidence_count`` of ``evidence_count + opposed_count``).

Only ``label`` and ``label_similarity`` change, and only on a node whose label moves. Weight, status,
centroid, counts and ``updated_at`` are untouched.

Writers: ``services/orion-durable-runs`` ``POST /personality/relabel`` (under the reducer's tick lock,
one transaction) and ``scripts/personality_relabel_choice_nodes.py`` (read-only dry run, snapshot,
then that endpoint).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.personality.graph import (
    DEFAULT_PARAMS,
    EVIDENCE_PREFIX,
    ORIGIN_CHOICE,
    GraphParams,
    category_led,
    choice_label,
    decline_text,
    pair_label,
)

# Every non-retired choice node. Both SQL texts take no parameters, so psycopg and asyncpg run them as is.
RELABEL_NODES_SQL = f"""
SELECT disposition_id, status, label, label_similarity, evidence_count, opposed_count
FROM personality_disposition
WHERE origin = '{ORIGIN_CHOICE}' AND status <> 'retired'
ORDER BY disposition_id
"""
# Their folded choices, in fold order: one tick commits its rows together (same created_at) after
# sorting them by (occurred_at, evidence_id) in Python string order, hence COLLATE "C".
RELABEL_EVIDENCE_SQL = f"""
SELECT e.disposition_id, e.evidence_id, e.occurred_at, e.created_at, e.similarity, e.meta, e.excerpt,
       c.choice_id, c.lane, c.declined_all, c.chosen, c.alternatives
FROM personality_evidence e
JOIN personality_disposition d ON d.disposition_id = e.disposition_id
LEFT JOIN personality_choice c ON c.choice_id = substr(e.evidence_id, {len(EVIDENCE_PREFIX) + 1})
WHERE d.origin = '{ORIGIN_CHOICE}' AND d.status <> 'retired' AND starts_with(e.evidence_id, '{EVIDENCE_PREFIX}')
ORDER BY e.disposition_id, e.created_at, e.occurred_at, e.evidence_id COLLATE "C"
"""


def _obj(value: Any) -> Any:
    """jsonb as psycopg (decoded) or asyncpg (text) returns it."""
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def choice_sides(lane: str, declined_all: bool, chosen_text: Optional[str],
                 alternatives: Sequence[tuple[str, Optional[int]]]) -> tuple[str, list[str]]:
    """(chosen-side text, declined texts in shown order). ``declined_all`` (or no chosen option) uses the
    lane's decline phrase as the chosen side. ``alternatives``: (text, market_rank), unranked last."""
    chosen = decline_text(lane) if declined_all or chosen_text is None else chosen_text
    ordered = sorted(alternatives, key=lambda a: (a[1] is None, a[1] or 0))
    return chosen, [t for t, _ in ordered]


def excerpt_from_choice_row(row: Mapping[str, Any]) -> Optional[str]:
    """The current ``pair_label`` of a stored ``personality_choice`` row; None when the row is missing."""
    if not row.get("choice_id"):
        return None
    chosen = _obj(row.get("chosen"))
    alts = _obj(row.get("alternatives")) or []
    chosen_text = chosen.get("text") if isinstance(chosen, Mapping) else None
    pairs = [(str(a.get("text") or ""), a.get("market_rank")) for a in alts if isinstance(a, Mapping)]
    lane = str(row.get("lane") or "")
    if lane == "prime":
        # Prime records before 2026-11-15 led with the category ("a promise I made in a curiosity run (1
        # promise, <ref>)"), which leaves a label nothing but the category. Prime's current record text maker
        # rewrites them deterministically; a text already in the current form passes through unchanged.
        chosen_text = _prime_text(chosen_text) if chosen_text is not None else None
        pairs = [(_prime_text(t), r) for t, r in pairs]
    c, declined = choice_sides(lane, bool(row.get("declined_all")), chosen_text, pairs)
    return pair_label(c, declined)


def _prime_text(text: str) -> str:
    from orion.prime.choice import ShownConcern, option_text

    return option_text(ShownConcern("relabel", text)) if text else text


@dataclass(frozen=True)
class Member:
    """One folded choice of a node, in fold order."""

    evidence_id: str
    similarity: Optional[float]
    excerpt: str  # remade with the current label maker (stored excerpt when the choice row is gone)
    takes_label: bool = True  # False: a Juniper-"no" absorb or an own-name echo (never the label)
    remade: bool = True  # False: the choice row was missing, the stored excerpt is used


@dataclass(frozen=True)
class Relabel:
    disposition_id: str
    status: str
    old_label: str
    new_label: str
    old_similarity: float
    new_similarity: float
    evidence: int
    base_evidence_id: str
    anomalies: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """The label moves. ``label_similarity`` is written only with it: it is the bar the base's own
        match set, so with an unchanged label (same base text) the stored bar stays."""
        return self.old_label != self.new_label


def replay_label(members: Sequence[Member], *, evidence_count: int, opposed_count: int,
                 params: GraphParams = DEFAULT_PARAMS) -> tuple[str, float, str]:
    """(label, label_similarity, evidence id of the base) the reducer's rules give these choices.
    ``members`` in fold order, founder first; must be non-empty."""
    founder = members[0]
    base, sim, base_id = founder.excerpt, params.match_threshold, founder.evidence_id
    for m in members[1:]:
        ex = m.excerpt or ""
        if not m.takes_label or not ex.strip():
            continue
        s = float(m.similarity) if m.similarity is not None else float("-inf")
        stale = category_led(base) and not category_led(ex)
        demotes = category_led(ex) and not category_led(base)
        if not demotes and (s > sim or stale):
            base, sim, base_id = ex, max(s, sim), m.evidence_id
    n = max(0, int(evidence_count))
    return choice_label(base, n, n + max(0, int(opposed_count))), sim, base_id


def members_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Member]]:
    """``RELABEL_EVIDENCE_SQL`` rows -> disposition_id -> members, in the rows' (fold) order."""
    out: dict[str, list[Member]] = {}
    for r in rows:
        meta = _obj(r.get("meta")) or {}
        remade = excerpt_from_choice_row(r)
        takes = not (isinstance(meta, Mapping) and (meta.get("juniper_no") or meta.get("cites_own_name")))
        out.setdefault(str(r["disposition_id"]), []).append(Member(
            evidence_id=str(r["evidence_id"]),
            similarity=float(r["similarity"]) if r.get("similarity") is not None else None,
            excerpt=remade if remade is not None else str(r.get("excerpt") or ""),
            takes_label=takes, remade=remade is not None))
    return out


def plan_relabel(nodes: Iterable[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]], *,
                 params: GraphParams = DEFAULT_PARAMS) -> list[Relabel]:
    """One ``Relabel`` per node in ``RELABEL_NODES_SQL`` order (changed or not). A node with no folded
    choice keeps its label (anomaly ``no_evidence``)."""
    by_node = members_from_rows(evidence)
    out: list[Relabel] = []
    for n in nodes:
        did = str(n["disposition_id"])
        old_label = str(n.get("label") or "")
        old_sim = float(n.get("label_similarity") or 0.0)
        members = by_node.get(did) or []
        anomalies: list[str] = []
        if not members:
            out.append(Relabel(did, str(n.get("status") or ""), old_label, old_label, old_sim, old_sim, 0, "",
                               ("no_evidence",)))
            continue
        if members[0].similarity is None or abs(members[0].similarity - 1.0) > 1e-6:
            anomalies.append("founder_similarity_not_1")
        missing = sum(1 for m in members if not m.remade)
        if missing:
            anomalies.append(f"choice_rows_missing:{missing}")
        if len(members) != int(n.get("evidence_count") or 0):
            anomalies.append(f"evidence_rows:{len(members)}_vs_count:{int(n.get('evidence_count') or 0)}")
        label, sim, base_id = replay_label(members, evidence_count=int(n.get("evidence_count") or 0),
                                           opposed_count=int(n.get("opposed_count") or 0), params=params)
        out.append(Relabel(did, str(n.get("status") or ""), old_label, label, old_sim, sim, len(members), base_id,
                           tuple(anomalies)))
    return out


# The write: only while the row still holds the label the plan was made from.
RELABEL_UPDATE_SQL = f"""
UPDATE personality_disposition SET label = %(new_label)s, label_similarity = %(new_similarity)s
WHERE disposition_id = %(disposition_id)s AND origin = '{ORIGIN_CHOICE}' AND status <> 'retired'
  AND label = %(old_label)s
"""


def approved(plan: Sequence[Relabel], expected: Optional[Mapping[str, Sequence[str]]]) -> tuple[list[Relabel], list[str]]:
    """(changes to write, disposition ids refused). With ``expected`` (disposition_id -> [old_label,
    new_label] from a reviewed dry run) only changes that match it exactly are written; every other planned
    change, and every expected id the plan no longer changes the same way, is refused. Without it: every
    change."""
    changes = [r for r in plan if r.changed]
    if expected is None:
        return changes, []
    want = {k: tuple(v) for k, v in expected.items()}
    ok = [r for r in changes if want.get(r.disposition_id) == (r.old_label, r.new_label)]
    ok_ids = {r.disposition_id for r in ok}
    refused = sorted({r.disposition_id for r in changes if r.disposition_id not in ok_ids} | (set(want) - ok_ids))
    return ok, refused


def relabel_view(r: Relabel) -> dict:
    """JSON form (the endpoint's response, the script's snapshot rows)."""
    return {"disposition_id": r.disposition_id, "status": r.status, "old_label": r.old_label,
            "new_label": r.new_label, "old_similarity": r.old_similarity, "new_similarity": r.new_similarity,
            "evidence": r.evidence, "base_evidence_id": r.base_evidence_id, "anomalies": list(r.anomalies),
            "changed": r.changed}
