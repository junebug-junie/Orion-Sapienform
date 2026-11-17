"""Office camera (cam0) person cut-off and prop-hold eval (2026-11-17).

Question it answers: does the office's ``person`` cut-off plus its stationary
hold keep a pile of boxes from becoming "Juniper in the office", without
losing the real people at the desk?

Live 2026-11-15..16 a pile of boxes on the office floor read as ``person``
2,397 times in one place (scores 0.35-0.52, no face). The office sessions
named it "Juniper (probably)" or "Jacob (maybe)", the captioner described
"her" packing boxes, and Orion sent check-ins with nobody in the room.

Data: ``fixtures/office_person_boxes.csv``. These are every cam0 ``person``
crop stored from 2026-11-05 to 2026-11-16 (time, score, box, has_face). The
``truth`` column is ``prop`` for the boxes' one fixed spot and size (x1
1070-1120, y1 790-810, x2 <= 1350, y2 <= 950 on the 2560x1920 frame; the crop
thumbnail shows cardboard and plastic, and no box of that size sat there
before 11-15) and ``real`` for the rest. One real 11-14 box starts at the same
corner but is far larger (a person passing). No images are committed.

Two checks, both replayed on the fixture with the shipped code and config:

1. cut-off: config/vision_profiles.yaml ``box_threshold_by_stream.cam0.person``.
   Reports the prop boxes kept and the real boxes kept (all, and with a face).
2. hold: orion-vision-window's FixtureTracker over the boxes left after the
   cut-off at a 0.35 floor (the old behaviour, so the hold is tested alone),
   with the shipped WINDOW_STATIONARY_STREAMS cam0 hold. Reports the longest
   still run of real boxes and of prop boxes, and whether each became a prop.

Exit 1 when a check fails: any prop box kept at the cut-off, under 85% of real
boxes kept, a real run reaching the hold, or the prop never reaching it.

Usage (from repo root; stdlib + PyYAML + pydantic-settings, e.g. the repo .venv)::

  .venv/bin/python services/orion-vision-host/evals/office_person_cutoff_eval.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "fixtures" / "office_person_boxes.csv"
PROFILES = REPO / "config" / "vision_profiles.yaml"
STREAM = "cam0"
MIN_REAL_KEPT = 0.85

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services" / "orion-vision-window"))


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


def hold_check(rows: List[dict]) -> Tuple[bool, List[str]]:
    from app.fixtures import FixtureTracker
    from app.settings import Settings
    from orion.vision.stationary import stream_min_secs

    s = Settings()
    holds = stream_min_secs(s.WINDOW_STATIONARY_STREAMS, s.VISION_STATIONARY_MIN_SEC)
    hold = holds.get(STREAM, 0.0)
    lines = [f"hold {STREAM} = {hold:.0f} s (WINDOW_STATIONARY_STREAMS={s.WINDOW_STATIONARY_STREAMS})"]
    if hold <= 0:
        return False, lines + ["  office camera has no prop hold"]
    longest: Dict[str, float] = {"real": 0.0, "prop": 0.0}
    became: Dict[str, bool] = {"real": False, "prop": False}
    for truth in ("real", "prop"):
        tr = FixtureTracker(streams={STREAM: hold}, min_sec=hold,
                            iou=s.VISION_STATIONARY_IOU, max_gap_sec=s.VISION_STATIONARY_MAX_GAP_SEC)
        for r in rows:
            if r["truth"] != truth:
                continue
            tr.observe_window(STREAM, [(r["t"], [r["box"]])])
            for run in tr._runs.get(STREAM, []):
                longest[truth] = max(longest[truth], run.held_sec())
            became[truth] = became[truth] or tr.fixture_only(STREAM)
    for truth in ("real", "prop"):
        lines.append(f"  {truth}: longest still run {longest[truth] / 60:.1f} min, became a prop: {became[truth]}")
    return (not became["real"]) and became["prop"], lines


def main() -> int:
    rows = load()
    ok1, l1 = cutoff_check(rows, configured_cutoff())
    ok2, l2 = hold_check(rows)
    print("\n".join(l1 + l2))
    print("PASS" if ok1 and ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
