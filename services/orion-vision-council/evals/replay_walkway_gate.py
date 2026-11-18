"""Replay eval: how often would the council re-interpret the walkway? (#3387)

Reads two CSV exports of real data and replays them through the real code:
orion-vision-window's belief tracker and fixture tracker, and the council's
transition gate. Prints label-change interpretations per hour for three cases:

  recorded      the believed labels production actually wrote (old belief
                tracker), through the old gate rule;
  belief_fix    this repo's belief tracker, old gate rule;
  both          this repo's belief tracker and the fixture-only gate rule.

The refresh TTL (COUNCIL_TRANSITION_REFRESH_SEC, at most 6 an hour) is applied
but not counted: the numbers are label-change triggers only. The replay treats
every interpretation as instant, so it is an upper bound on rows (live, an LLM
call takes seconds and windows that arrive meanwhile are skipped).

Export (read-only; run on the database host):

  Q='docker exec -i -e PGOPTIONS=-cdefault_transaction_read_only=on orion-athena-sql-db psql -U postgres -d conjourney -At -c'
  $Q "copy (select extract(epoch from observed_at), artifact_id, label, score, array_to_string(box_xyxy, ' ')
            from vision_crop_observation where stream_id='walkway'
            and observed_at > now() - interval '96 hours' order by observed_at) to stdout with csv" > crops.csv
  $Q "copy (select window_end_ts, frame_count, counts::text, believed_labels::text
            from vision_scene_inventory where stream_id='walkway'
            and observed_at > now() - interval '96 hours' order by window_end_ts) to stdout with csv" > windows.csv

Run from the repo root:

  python services/orion-vision-council/evals/replay_walkway_gate.py crops.csv windows.csv

The first 48 h only warm the fixture tracker (a fixture needs 1800 s in one
place); the table covers the rest. Exit 1 if `both` is not below `recorded`.

Result on 2026-11-15 (96 h export, 48 h scored): recorded 3,949, belief_fix
1,174, both 300.
"""

from __future__ import annotations

import collections
import csv
import datetime
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _load(service: str, module: str):
    """Load one module of a service's ``app`` package without importing two
    different packages both named ``app``."""
    path = REPO / "services" / service / "app" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{service.replace('-', '_')}_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fixtures = _load("orion-vision-window", "fixtures")
scene_belief = _load("orion-vision-window", "scene_belief")
gate = _load("orion-vision-council", "evidence_transition")

LABELS = ["person", "dog", "bicycle", "stroller", "vehicle", "package", "mail truck"]  # WINDOW_CAPTION_GATE_LABELS
MIN_SCORE, WARMUP_SEC, TTL_SEC = 0.25, 48 * 3600, 600.0


def _interprets(seq, *, use_fixtures: bool) -> list[float]:
    """Times of label-change interpretations, through the real gate."""
    tracker = gate.EvidenceTransitionTracker()
    out = []
    for t, labels, fixture_only in seq:
        snap = gate.EvidenceSnapshot(hard_labels=labels, person_present="person" in labels,
                                     fixture_only_labels=fixture_only if use_fixtures else frozenset())
        decision = tracker.evaluate(stream_key="walkway", snapshot=snap, now=t, max_refresh_sec=TTL_SEC)
        if decision.interpret:
            tracker.record_interpretation(stream_key="walkway", snapshot=snap, now=t)
            if decision.reason != "refresh_ttl":
                out.append(t)
    return out


def main(crops_csv: str, windows_csv: str) -> int:
    frames: dict = collections.OrderedDict()
    for t, art, label, score, box in csv.reader(open(crops_csv)):
        frame = frames.setdefault(art, (float(t), collections.defaultdict(list)))
        if float(score) >= MIN_SCORE:
            frame[1][label.lower()].append([float(x) for x in box.split()])
    ordered = sorted(frames.values(), key=lambda f: f[0])
    windows = [(float(e), int(fc), json.loads(c), json.loads(b)) for e, fc, c, b in csv.reader(open(windows_csv))]
    if not windows:
        print("no windows in the export")
        return 1

    trackers = {l: fixtures.FixtureTracker(streams=["walkway"], min_sec=1800, iou=0.5, max_gap_sec=300, label=l)
                for l in LABELS}
    belief = scene_belief.SceneBeliefTracker(vote_n=3, enter_votes=3, exit_votes=0)
    recorded, replayed = [], []
    fi, scored_from = 0, windows[0][0] + WARMUP_SEC
    for end, frame_count, counts, believed in windows:
        per = {l: [] for l in LABELS}
        while fi < len(ordered) and ordered[fi][0] <= end:
            t, boxes = ordered[fi]
            fi += 1
            for l in LABELS:
                per[l].append((t, boxes.get(l, [])))
        for l in LABELS:
            if per[l]:
                trackers[l].observe_window("walkway", per[l])
        hard = frozenset(k.lower() for k in counts)
        if frame_count > 0:
            belief.observe(hard)
        fixture_only = frozenset(l for l in LABELS if trackers[l].fixture_only("walkway"))
        if end >= scored_from:
            old_believed = frozenset(x.lower() for x in believed)
            recorded.append((end, old_believed or hard, fixture_only))
            replayed.append((end, belief.believed_labels or hard, fixture_only))

    cases = {"recorded": _interprets(recorded, use_fixtures=False),
             "belief_fix": _interprets(replayed, use_fixtures=False),
             "both": _interprets(replayed, use_fixtures=True)}
    by_hour = {name: collections.Counter(
        datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%m-%d %H") for t in times)
        for name, times in cases.items()}
    print(f"{'hour (UTC)':<12}" + "".join(f"{name:>12}" for name in cases))
    for hour in sorted(set().union(*by_hour.values())):
        print(f"{hour:<12}" + "".join(f"{by_hour[name][hour]:>12}" for name in cases))
    totals = {name: len(times) for name, times in cases.items()}
    print(f"{'TOTAL':<12}" + "".join(f"{totals[name]:>12}" for name in cases))
    return 0 if totals["both"] < totals["recorded"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
