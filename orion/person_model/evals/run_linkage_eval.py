#!/usr/bin/env python3
"""Linkage calibration eval: auto-merge precision, recall, and review-band size.

Inputs: a dry-run ``decisions.jsonl`` (scripts/run_person_linkage.py) and labels,
either ``cleanup_labels.jsonl`` ({"decision_id", "label"}) or the CSV the
interlocutor exports from the HTML review (``your_choice`` approve/reject on each
card; approve on a same_as card = same person).

Labels: same | facet | junk | different | not_home | distinct.

Reports, for the configured thresholds:
- auto-merge precision on labeled same_as auto-merges (gate: must be 1.0),
- recall of labeled true matches that auto-merge,
- band size (cards that need the interlocutor) = expected ask volume,
- the thresholds the labels support: upper* = highest labeled non-match score + margin,
  lower* = lowest labeled match score - margin, and the band size under them.

Exit 1 if auto-merge precision < 1.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MARGIN = 1.0


def load_labels(path: Path, decisions: dict) -> dict:
    if path.suffix == ".csv":
        out = {}
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d = decisions.get(row["decision_id"])
                # Pre-approved defaults she never touched are Orion's own output,
                # not labels: tuning on them would fit the thresholds to themselves.
                if not d or not row.get("your_choice") or row.get("defaulted") == "yes":
                    continue
                rel = d["relation"]
                ok = row["your_choice"] == "approve"
                if rel == "same_as":
                    out[d["decision_id"]] = "same" if ok else "different"
                elif rel == "no_match":
                    out[d["decision_id"]] = "different" if ok else "same"
                elif rel == "facet_of":
                    out[d["decision_id"]] = "facet" if ok else "different"
                elif rel == "junk":
                    out[d["decision_id"]] = "junk" if ok else "different"
                elif rel in ("not_home", "distinct"):
                    out[d["decision_id"]] = rel if ok else "same"
                # One card can answer duplicate questions (same anchor, same folded label).
                for extra in (row.get("also_answers") or "").split():
                    if extra in decisions and d["decision_id"] in out:
                        out[extra] = out[d["decision_id"]]
        return out
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["decision_id"]: r["label"] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path(__file__).with_name("cleanup_labels.jsonl"))
    args = ap.parse_args()
    decisions = {d["decision_id"]: d for d in (json.loads(x) for x in args.decisions.read_text(encoding="utf-8").splitlines() if x.strip())}
    labels = load_labels(args.labels, decisions)
    missing = [k for k in labels if k not in decisions]
    scored = [(decisions[k], v) for k, v in labels.items() if k in decisions]
    groups = {}
    for d in decisions.values():
        groups[d["group"]] = groups.get(d["group"], 0) + 1

    auto = [(d, v) for d, v in scored if d["relation"] == "same_as" and d["group"] == "auto_merge"]
    tp = sum(1 for _, v in auto if v == "same")
    precision = tp / len(auto) if auto else 1.0
    true_same = [(d, v) for d, v in scored if v == "same"]
    recall = (sum(1 for d, _ in true_same if d["group"] == "auto_merge") / len(true_same)) if true_same else 1.0
    rule_ok = sum(1 for d, v in scored if d["rule"] and {"facet_of": "facet", "junk": "junk", "not_home": "not_home", "distinct": "distinct", "no_match": "different"}.get(d["relation"]) == v)
    rule_n = sum(1 for d, _ in scored if d["rule"] and d["rule"] not in ("possible_match_band", "cooccurring_names_guard"))

    pos = [d["score"] for d, v in scored if v == "same" and d["score"] is not None]
    neg = [d["score"] for d, v in scored if v != "same" and d["score"] is not None]
    upper_star = (max(neg) + MARGIN) if neg else None
    lower_star = (min(pos) - MARGIN) if pos else None
    band_star = None
    if upper_star is not None and lower_star is not None:
        band_star = sum(1 for d in decisions.values() if d["score"] is not None and lower_star < d["score"] < upper_star)

    report = {
        "decisions": len(decisions),
        "labeled": len(scored),
        "labels_not_in_run": len(missing),
        "groups": groups,
        "expected_ask_volume": groups.get("needs_your_call", 0),
        "auto_merge_precision": round(precision, 3),
        "auto_merge_n": len(auto),
        "true_match_recall_auto": round(recall, 3),
        "true_matches_labeled": len(true_same),
        "rule_outcomes_correct": f"{rule_ok}/{rule_n}",
        "labeled_match_scores": sorted(pos),
        "labeled_nonmatch_scores_top5": sorted(neg)[-5:],
        "thresholds_configured": {"upper": next(iter(decisions.values()))["upper_threshold"], "lower": next(iter(decisions.values()))["lower_threshold"]},
        "thresholds_labels_support": {"upper": upper_star, "lower": lower_star, "band_size": band_star, "margin": MARGIN},
    }
    print(json.dumps(report, indent=2))
    return 0 if precision == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
