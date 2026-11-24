"""Precision eval for the supersede classifier (issue #3659). Pure scoring; no database, no model.

A labelled item is one (older row, newer row) pair with the answer a careful reader gives:
``act`` (the newer row updates or corrects the older one, so the older one is no longer current)
or ``no_act`` (both stay current: refine, compatible, unrelated, near-duplicate). Items come
from two places:

- synthetic, made-up fixtures checked into the repo
  (``orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl``);
- real rows, built host-side by ``scripts/memory_supersede_eval.py build`` into
  ``~/.orion/evals/memory_supersede_labelled.jsonl`` (mode 600, never in git): Juniper's own
  confirmation-loop revisions (positives), and hard negatives (the watched trip rows against the
  home correction, the pet rows, pairs the 2026-11-17 dry run vetoed, near-duplicates).

The decision for an item is the whole write-time path: the deterministic eligibility rules
(``can_be_superseded`` / ``can_correct``) when ``apply_rules`` is set, then the classifier. A
false act on any negative fails the eval. The distill step applies links only while the most
recent recorded run for the current check version passed (``supersede.eval_gate``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.memory.episode.supersede import MemoryRow, can_be_superseded, can_correct

SYNTHETIC_PATH = Path(__file__).resolve().parent / "fixtures" / "supersede_eval_synthetic.jsonl"
# Written after the v3.1 checklist was designed and scored, so the checklist never saw them: the
# out-of-sample check (2026-11-18). Part of every full run from then on.
HELDOUT_PATH = Path(__file__).resolve().parent / "fixtures" / "supersede_eval_heldout.jsonl"
LABELS = ("act", "no_act")
# The minimum real-data coverage for a run to count: an eval with no real negatives proves nothing.
MIN_REAL_POSITIVES = 1
MIN_REAL_HARD_NEGATIVES = 5
MIN_TRUE_POSITIVES = 1


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    kind: str                 # "synthetic" | "real"
    source: str               # e.g. "confirmation_revise", "watched_trip", "dry_run_veto", "near_duplicate"
    label: str                # "act" | "no_act"
    expected_relation: str    # update | correction | refine | compatible | unrelated
    hard_negative: bool
    old: MemoryRow
    new: MemoryRow
    apply_rules: bool = True  # False: judge the classifier alone (rows whose live state changed since)
    # Other older rows asked alongside ``old`` for the same new row (the clear-target rule of
    # Juniper's path is per new row). Empty: a single pair.
    others: tuple[MemoryRow, ...] = ()


def _dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _row(d: Mapping[str, Any]) -> MemoryRow:
    return MemoryRow(
        memory_id=str(d["memory_id"]), statement=" ".join(str(d.get("statement") or "").split()),
        purpose=str(d.get("purpose") or "about_juniper"), created_at=_dt(d.get("created_at")),
        confirmation_state=str(d.get("confirmation_state") or "auto"), episode_id=d.get("episode_id"),
        occurred_at=_dt(d.get("occurred_at")), voice=str(d.get("voice") or ""),
        juniper_quoted=bool(d.get("juniper_quoted")))


def row_dict(r: MemoryRow) -> dict[str, Any]:
    return {"memory_id": r.memory_id, "statement": r.statement, "purpose": r.purpose,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
            "confirmation_state": r.confirmation_state, "episode_id": r.episode_id, "voice": r.voice,
            "juniper_quoted": r.juniper_quoted}


def item_from_dict(d: Mapping[str, Any]) -> EvalItem:
    label = str(d["label"])
    if label not in LABELS:
        raise ValueError(f"bad label {label!r} on {d.get('item_id')}")
    return EvalItem(item_id=str(d["item_id"]), kind=str(d.get("kind") or "synthetic"), source=str(d["source"]),
                    label=label, expected_relation=str(d.get("expected_relation") or ""),
                    hard_negative=bool(d.get("hard_negative")), old=_row(d["old"]), new=_row(d["new"]),
                    apply_rules=bool(d.get("apply_rules", True)),
                    others=tuple(_row(o) for o in (d.get("others") or ())))


def item_to_dict(it: EvalItem) -> dict[str, Any]:
    return {"item_id": it.item_id, "kind": it.kind, "source": it.source, "label": it.label,
            "expected_relation": it.expected_relation, "hard_negative": it.hard_negative,
            "apply_rules": it.apply_rules, "old": row_dict(it.old), "new": row_dict(it.new),
            **({"others": [row_dict(o) for o in it.others]} if it.others else {})}


def load_items(path: Path) -> list[EvalItem]:
    if not path.exists():
        return []
    return [item_from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def base_id(item_id: str) -> str:
    """The item id without its repeat suffix ("<id>@<n>", scripts/memory_supersede_eval.py --repeats)."""
    return item_id.rsplit("@", 1)[0]


def rule_refusal(old: MemoryRow, new: MemoryRow) -> Optional[str]:
    """The deterministic write-time rules, before any model call (None = eligible)."""
    if not can_correct(new):
        return "new_cannot_correct"
    if not can_be_superseded(old):
        return "old_protected"
    if new.episode_id and old.episode_id == new.episode_id:
        return "same_episode"
    return None


@dataclass
class Decision:
    item_id: str
    acted: bool
    relation: Optional[str] = None
    stopped_by: Optional[str] = None   # "rule" | "model" | "error" | None (acted)
    detail: str = ""
    votes: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def score(items: Sequence[EvalItem], decisions: Iterable[Decision]) -> dict[str, Any]:
    """Precision / recall on act, and every false act by source. ``passed`` needs zero false acts,
    zero errors, and enough real items to mean something."""
    by_id = {d.item_id: d for d in decisions}
    tp = fp = fn = tn = errors = 0
    false_acts: list[dict[str, str]] = []
    by_source: dict[str, dict[str, int]] = {}
    relation_hits = relation_n = 0
    for it in items:
        d = by_id.get(it.item_id)
        src = by_source.setdefault(f"{it.kind}:{it.source}", {"n": 0, "acted": 0, "correct": 0})
        src["n"] += 1
        if d is None or d.error:
            errors += 1
            continue
        src["acted"] += int(d.acted)
        if it.label == "act":
            if d.acted:
                tp += 1
                src["correct"] += 1
                if it.expected_relation in ("update", "correction"):
                    relation_n += 1
                    relation_hits += int(d.relation == it.expected_relation)
            else:
                fn += 1
        else:
            if d.acted:
                fp += 1
                false_acts.append({"item_id": it.item_id, "source": f"{it.kind}:{it.source}",
                                   "relation": str(d.relation)})
            else:
                tn += 1
                src["correct"] += 1
    real = [it for it in items if it.kind == "real"]
    # Coverage floors count DISTINCT pairs: a repeated run (item ids "<id>@<n>") must not halve them.
    real_pos = len({base_id(it.item_id) for it in real if it.label == "act"})
    real_hard = len({base_id(it.item_id) for it in real if it.label == "no_act" and it.hard_negative})
    hard_false = sum(1 for f in false_acts
                     if any(it.item_id == f["item_id"] and it.hard_negative for it in items))
    fail: list[str] = []
    if fp:
        fail.append(f"false_acts={fp}")
    if tp < MIN_TRUE_POSITIVES:  # abstaining on everything is safe, but proves the classifier works for nothing
        fail.append(f"true_positives={tp}<{MIN_TRUE_POSITIVES}")
    if errors:
        fail.append(f"errors={errors}")
    if real_pos < MIN_REAL_POSITIVES:
        fail.append(f"real_positives={real_pos}<{MIN_REAL_POSITIVES}")
    if real_hard < MIN_REAL_HARD_NEGATIVES:
        fail.append(f"real_hard_negatives={real_hard}<{MIN_REAL_HARD_NEGATIVES}")
    acted = tp + fp
    return {
        "n": len(items), "real": len(real), "synthetic": len(items) - len(real),
        "positives": tp + fn, "negatives": fp + tn, "real_positives": real_pos, "real_hard_negatives": real_hard,
        "acted": acted, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": errors,
        "precision": (tp / acted) if acted else None, "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "hard_negative_false_acts": hard_false, "false_acts": false_acts,
        "relation_accuracy": (relation_hits / relation_n) if relation_n else None,
        "by_source": by_source, "passed": not fail, "fail_reasons": fail,
    }


def decision_dict(d: Decision) -> dict[str, Any]:
    return asdict(d)
