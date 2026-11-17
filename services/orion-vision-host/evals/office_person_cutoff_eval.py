"""Office camera (cam0) person cut-off eval (2026-11-17).

Question it answers: does the office's ``person`` cut-off keep a pile of boxes
from becoming "Juniper in the office", without losing the real people at the
desk?

Live 2026-11-15..16 a pile of boxes on the office floor read as ``person``
2,396 times in one place (scores 0.35-0.52, no face). The office sessions
named it "Juniper (probably)" or "Jacob (maybe)", the captioner described
"her" packing boxes, and Orion sent check-ins with nobody in the room.

Data: ``fixtures/office_person_boxes.csv``. These are every cam0 ``person``
crop stored from 2026-11-05 to 2026-11-16 (time, score, box, has_face). The
``truth`` column is ``prop`` for the boxes' one fixed spot and size (x1
1070-1120, y1 790-810, x2 <= 1350, y2 <= 950 on the 2560x1920 frame; the crop
thumbnail shows cardboard and plastic, and no box of that size sat there
before 11-15) and ``real`` for the rest. One real 11-14 box starts at the same
corner but is far larger (a person passing). No images are committed.

Check, replayed on the fixture with the shipped config: the cut-off
config/vision_profiles.yaml ``box_threshold_by_stream.cam0.person``. Reports
the prop boxes kept and the real boxes kept (all, and with a face). Exit 1
when any prop box is kept or under 85% of real boxes are kept.

The stationary-prop rule is deliberately NOT turned on for cam0: a marked
prop stays marked for 24 h across gaps, so one long desk session would make
Juniper's seat a prop on every later visit.

Usage (from repo root; stdlib + PyYAML, e.g. the repo .venv)::

  .venv/bin/python services/orion-vision-host/evals/office_person_cutoff_eval.py
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "fixtures" / "office_person_boxes.csv"
PROFILES = REPO / "config" / "vision_profiles.yaml"
STREAM = "cam0"
MIN_REAL_KEPT = 0.85


def load() -> List[dict]:
    rows = []
    with DATA.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": datetime.fromisoformat(r["observed_at"].replace("Z", "+00:00")).timestamp(),
                "score": float(r["score"]),
                "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                "face": r["has_face"] == "1",
                "truth": r["truth"],
            })
    return rows


def configured_cutoff() -> float:
    profiles = yaml.safe_load(PROFILES.read_text())["profiles"]
    for prof in profiles:
        by_stream = (prof.get("params") or {}).get("box_threshold_by_stream") or {}
        if STREAM in by_stream and "person" in by_stream[STREAM]:
            return float(by_stream[STREAM]["person"])
    raise SystemExit(f"no box_threshold_by_stream.{STREAM}.person in {PROFILES}")


def cutoff_check(rows: List[dict], cutoff: float) -> Tuple[bool, List[str]]:
    lines = [f"cut-off {STREAM} person = {cutoff}"]
    props = [r for r in rows if r["truth"] == "prop"]
    real = [r for r in rows if r["truth"] == "real"]
    face = [r for r in real if r["face"]]
    kept_props = sum(r["score"] >= cutoff for r in props)
    kept_real = sum(r["score"] >= cutoff for r in real)
    kept_face = sum(r["score"] >= cutoff for r in face)
    lines.append(f"  prop boxes kept: {kept_props}/{len(props)}")
    lines.append(f"  real boxes kept: {kept_real}/{len(real)} ({kept_real / max(1, len(real)):.0%})")
    lines.append(f"  real boxes with a face kept: {kept_face}/{len(face)}")
    ok = kept_props == 0 and kept_real >= MIN_REAL_KEPT * len(real)
    return ok, lines


def main() -> int:
    ok, lines = cutoff_check(load(), configured_cutoff())
    print("\n".join(lines))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
