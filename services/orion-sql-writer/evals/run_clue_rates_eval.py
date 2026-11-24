"""Read-only: how often each object clue sits next to each named person vs the street (2026-11-24).

Design: docs/superpowers/specs/2026-11-24-walkway-clue-evidence-design.md (patch 2, revised).

Juniper's "How could you tell?" reasons name clues such as a stroller or a dog.
The detector already boxes strollers and dogs, so no new picture reader is
needed for them. This eval asks, from live data, whether a clue separates a
named person from the street:

- Answer key: the person album (``vision_person_exemplar`` polarity +1, not
  revoked, and not also -1 for the same name). That is ~99.8% crops of clusters
  she named that passed the album's purity check (``cluster_label``), plus the
  few she picked one by one. A crop under two names is dropped.
- Street: person crops whose cluster has no name. It includes regulars not named
  yet, which pulls street rates toward theirs (conservative: weights shrink to 0).
- Unit: a PASSAGE = one person (cluster) on one stream within one minute.
  Frames of one walk past are not independent, so the passage counts are the
  ones to trust; per-crop counts are printed beside them.
- A clue is "next to" a person crop when a box of that kind is in the SAME
  frame (same stream, same ``observed_at``) and its centre is within
  ``--near`` person-heights sideways of the person's centre.
- Weight of evidence per (person, clue): ``ln(P(clue | person) / P(clue | street))``
  with add-0.5 smoothing. Negative = seeing the clue is evidence it is NOT them.

Nothing is written. Names are printed to the terminal only.

Usage:
    PYTHONPATH=. python services/orion-sql-writer/evals/run_clue_rates_eval.py [--since 2026-11-18] [--near 1.0]
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
STREET = "(street)"
# Detector kinds that are clues. A dog she named comes through ``named`` as ``with:<name>``.
CLUE_KINDS = ("stroller", "dog")

_SQL = """
WITH pos AS (
  SELECT DISTINCT e.crop_id, e.person_label AS who FROM vision_person_exemplar e
  WHERE e.polarity = 1 AND e.revoked_at IS NULL
    AND NOT EXISTS (SELECT 1 FROM vision_person_exemplar n WHERE n.crop_id = e.crop_id
                    AND n.person_label = e.person_label AND n.polarity = -1 AND n.revoked_at IS NULL)),
truth AS (SELECT crop_id, min(who) AS who FROM pos GROUP BY crop_id HAVING count(*) = 1),
p AS (
  SELECT c.crop_id, c.observed_at, c.stream_id, c.box_xyxy AS b,
         c.stream_id || '|' || coalesce(c.individual_id, c.crop_id) || '|'
           || to_char(date_trunc('minute', c.observed_at), 'YYYY-MM-DD"T"HH24:MI') AS passage,
         coalesce(t.who, CASE WHEN nullif(i.label, '') IS NULL THEN :street END) AS who
  FROM vision_crop_observation c
  LEFT JOIN truth t USING (crop_id)
  LEFT JOIN vision_individual i ON i.individual_id = c.individual_id
  WHERE c.label = 'person' AND c.observed_at >= CAST(:since AS timestamptz)),
o AS (
  SELECT label, observed_at, stream_id, individual_id AS oid, box_xyxy AS b FROM vision_crop_observation
  WHERE label = ANY(:kinds) AND observed_at >= CAST(:since AS timestamptz))
SELECT p.who, p.crop_id, p.passage, o.label AS kind, nullif(oi.label, '') AS named
FROM p
LEFT JOIN o ON o.stream_id = p.stream_id AND o.observed_at = p.observed_at
  AND abs(((o.b[1] + o.b[3]) / 2) - ((p.b[1] + p.b[3]) / 2)) < :near * (p.b[4] - p.b[2])
LEFT JOIN vision_individual oi ON oi.individual_id = o.oid
WHERE p.who IS NOT NULL
"""


def tally(rows: Iterable[Tuple[str, str, str | None, str | None]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """rows (who, unit, clue kind or None, named object or None) -> (units per person,
    units-with-clue per person per clue). The unit is a crop id or a passage key.
    A named object (a dog she named) is its own clue ``with:<name>``. A unit counts
    once per clue however many boxes or frames it has."""
    crops: Dict[str, set] = {}
    hits: Dict[str, Dict[str, set]] = {}
    for who, unit, kind, named in rows:
        crops.setdefault(who, set()).add(unit)
        for clue in ([kind] if kind else []) + ([f"with:{named}"] if kind and named else []):
            hits.setdefault(who, {}).setdefault(clue, set()).add(unit)
    return ({w: len(c) for w, c in crops.items()},
            {w: {k: len(v) for k, v in by.items()} for w, by in hits.items()})


def weight_of_evidence(hit: int, n: int, street_hit: int, street_n: int) -> float:
    """ln(P(clue | person) / P(clue | street)), add-0.5 smoothing on both."""
    return math.log(((hit + 0.5) / (n + 1.0)) / ((street_hit + 0.5) / (street_n + 1.0)))


def report(n: Dict[str, int], hits: Dict[str, Dict[str, int]], min_crops: int, unit: str = "crops") -> List[str]:
    street_n = n.get(STREET, 0)
    clues = sorted({c for by in hits.values() for c in by})
    out = [f"street: {street_n} {unit}; " + ", ".join(
        f"{c} {hits.get(STREET, {}).get(c, 0)}" for c in clues)]
    for who in sorted((w for w in n if w != STREET), key=lambda w: -n[w]):
        if n[who] < min_crops:
            continue
        cells = []
        for c in clues:
            h = hits.get(who, {}).get(c, 0)
            woe = weight_of_evidence(h, n[who], hits.get(STREET, {}).get(c, 0), street_n)
            cells.append(f"{c} {h}/{n[who]} ({h / n[who]:.1%}, woe {woe:+.2f})")
        out.append(f"{who}: " + "; ".join(cells))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pg", default=os.getenv("ORION_EVAL_PG_URI", DEFAULT_PG))
    ap.add_argument("--since", default="2026-11-18", help="stroller boxes start 2026-11-18")
    ap.add_argument("--near", type=float, default=1.0, help="sideways distance, in person-heights")
    ap.add_argument("--min-crops", type=int, default=20)
    ap.add_argument("--min-passages", type=int, default=10)
    args = ap.parse_args()
    from sqlalchemy import create_engine, text

    with create_engine(args.pg).connect() as conn:
        rows = conn.execute(text(_SQL), {"street": STREET, "since": args.since, "near": args.near,
                                         "kinds": list(CLUE_KINDS)}).fetchall()
    print(f"since {args.since}, near = {args.near} person-heights")
    n = {}
    for unit, idx, floor in (("passages", 2, args.min_passages), ("crops", 1, args.min_crops)):
        n, hits = tally((r[0], r[idx], r[3], r[4]) for r in rows)
        print(f"-- per {unit} (people with >= {floor}){' -- trust these' if unit == 'passages' else ''}")
        for line in report(n, hits, floor, unit):
            print(line)
    if n.get(STREET, 0) == 0:
        print("PROBLEM no street crops: nothing to compare against")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
