"""Office camera (cam0) person eval: cut-off + room prop rule (2026-11-17, 11-24).

Question it answers: do the office's ``person`` cut-off and its room prop rule
keep a pile of boxes from becoming "Juniper in the office", without losing
the real people at the desk?

Live 2026-11-15..16 a pile of boxes on the office floor read as ``person``
2,396 times in one place (scores 0.35-0.52, no face). The office sessions
named it "Juniper (probably)" or "Jacob (maybe)", the captioner described
"her" packing boxes, and Orion sent check-ins with nobody in the room. After
the 0.55 cut-off shipped (PR #3616), one pile box still scored 0.554
(2026-11-22), so the cut-off alone has almost no margin (#3614).

Data: ``fixtures/office_person_boxes.csv``. These are every cam0 ``person``
crop stored from 2026-11-05 to 2026-11-23 (time, score, box, has_face). The
``truth`` column is ``prop`` for the boxes' one fixed spot and size (x1
1070-1120, y1 790-810, x2 <= 1350, y2 <= 950 on the 2560x1920 frame; the crop
thumbnail shows cardboard and plastic, and no box of that size sat there
before 11-15) and ``real`` for the rest. One real 11-14 box starts at the same
corner but is far larger (a person passing). No images are committed.

Two checks, replayed on the fixture with the shipped config:

1. cut-off (config/vision_profiles.yaml ``box_threshold_by_stream.cam0.person``):
   at least 85% of real boxes kept, at most 0.5% of prop boxes kept.
2. room prop rule (orion-vision-window's FixtureTracker with the
   VISION_STATIONARY_ROOM_* values from its .env_example), over ALL boxes
   regardless of score, as if the pile scored above the cut-off: the pile
   becomes a prop, and no real person ever does.

Usage (from repo root; stdlib + PyYAML, e.g. the repo .venv)::

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
WINDOW = REPO / "services" / "orion-vision-window"
STREAM = "cam0"
MIN_REAL_KEPT = 0.85
MAX_PROP_KEPT = 0.005


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
    ok = kept_props <= MAX_PROP_KEPT * len(props) and kept_real >= MIN_REAL_KEPT * len(real)
    return ok, lines


def window_env() -> Dict[str, str]:
    """The operator contract: orion-vision-window/.env_example key=value pairs."""
    out: Dict[str, str] = {}
    for line in (WINDOW / ".env_example").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def room_rule_check(rows: List[dict]) -> Tuple[bool, List[str]]:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(WINDOW))
    from app.fixtures import FixtureTracker

    env = window_env()
    rooms = env.get("VISION_STATIONARY_ROOM_STREAMS", "")
    hold = float(env.get("VISION_STATIONARY_ROOM_MIN_SEC", "0"))
    lines = [f"room prop rule: VISION_STATIONARY_ROOM_STREAMS={rooms} hold={hold:.0f} s"]
    if STREAM not in [x.strip() for x in rooms.split(",")] or hold <= 0:
        return False, lines + [f"  {STREAM} is not a room camera"]
    marked: Dict[str, float] = {}
    longest: Dict[str, float] = {}
    for truth in ("real", "prop"):
        tr = FixtureTracker(streams=(), min_sec=float(env.get("VISION_STATIONARY_MIN_SEC", "1800")),
                            iou=float(env.get("VISION_STATIONARY_IOU", "0.5")),
                            max_gap_sec=float(env.get("VISION_STATIONARY_MAX_GAP_SEC", "300")),
                            room_streams=rooms, room_min_sec=hold)
        first, best = None, 0.0
        for r in rows:
            if r["truth"] != truth:
                continue
            tr.observe_window(STREAM, [(r["t"], [r["box"]], [r["face"]])])
            if first is None and tr.fixture_only(STREAM):
                first = r["t"]
            best = max([best] + [run.held_sec() for run in tr._runs.get(STREAM, [])])
        longest[truth] = best
        if first is not None:
            marked[truth] = first
    for truth in ("real", "prop"):
        when = (datetime.fromtimestamp(marked[truth]).astimezone().isoformat(timespec="minutes")
                if truth in marked else "never")
        lines.append(f"  {truth}: longest still run {longest[truth] / 60:.1f} min, first marked a prop: {when}")
    return ("real" not in marked) and ("prop" in marked), lines


def main() -> int:
    rows = load()
    ok1, l1 = cutoff_check(rows, configured_cutoff())
    ok2, l2 = room_rule_check(rows)
    print("\n".join(l1 + l2))
    print("PASS" if ok1 and ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
