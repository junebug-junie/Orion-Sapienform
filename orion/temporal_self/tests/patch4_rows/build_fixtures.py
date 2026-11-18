"""Compact live temporal-self rows into the patch-4 test fixtures. Read-only source: orion-durable-runs
(:8124) GET /temporal-self/day/2026-10-09, /frame (2026-10-11 23:53:23Z) and /arcs?day_id=2026-10-11,
saved as day_<id>.json, frame.json, arcs_1011.json in a scratch dir.

    python3 build_fixtures.py <scratch dir> <this dir>
"""
import copy, gzip, json, sys
S, OUT = sys.argv[1], sys.argv[2]
KEEP_ARC = {"arc_id","day_id","kind","subject_ref","subject_label","began_at","ended_at","last_seen_at","status",
            "closed_reason","attention_returns","cumulative_dwell_sec","warnings"}
KEEP_EXP = {"event_id","source_kind","committed_at","resolved_at","verdict"}
def compact_frame(f, keep_arcs=True):
    f = copy.deepcopy(f)
    for k in ("unbound_context_event_ids","self_change_event_ids","source_cursors","entered_day_with"):
        f[k] = [] if isinstance(f.get(k), list) else {}
    f["expectations_pending"] = [{k: v for k, v in x.items() if k in KEEP_EXP} for x in f["expectations_pending"]]
    f["expectations_resolved_today"] = [{k: v for k, v in x.items() if k in KEEP_EXP} for x in f["expectations_resolved_today"]]
    return f
for d in ("2026-10-09",):
    day = json.load(open(f"{S}/day_{d}.json"))
    day["arcs"] = [{k: v for k, v in a.items() if k in KEEP_ARC} for a in day["arcs"]]
    day["frame"] = compact_frame(day["frame"])
    day["frame"]["arcs_today"] = []; day["frame"]["active_by_kind"] = {}
    with gzip.GzipFile(f"{OUT}/day_{d}.json.gz", "wb", mtime=0) as fh:
        fh.write(json.dumps(day, sort_keys=True, separators=(",", ":")).encode())
# The real 2026-10-11 23:53:23Z frame, with last_seen_at filled from the same day's stored arcs
# (the field patch 4 adds to ArcSummaryV1; the producer fills it once durable-runs is rebuilt).
frame = json.load(open(f"{S}/frame.json"))
arcs = {a["arc_id"]: a for a in json.load(open(f"{S}/arcs_1011.json"))["arcs"]}
def fill(s):
    if s and s["arc_id"] in arcs:
        s["last_seen_at"] = arcs[s["arc_id"]]["last_seen_at"]
    return s
frame = compact_frame(frame)
frame["active_arc"] = fill(frame["active_arc"]); frame["previous_arc"] = fill(frame["previous_arc"])
frame["arcs_today"] = [fill(s) for s in frame["arcs_today"]]
frame["active_by_kind"] = {k: fill(v) for k, v in frame["active_by_kind"].items()}
json.dump(frame, open(f"{OUT}/frame_2026-10-11_talking.json", "w"), sort_keys=True, separators=(",", ":"))
# Busy: the same day rebuilt at 17:20Z from its stored arcs (no frame history exists): attention on
# "Bus synaptic prediction error" (arc ae0a3cb57ae73ee4, 16:51-17:32Z, open at 17:20).
T = "2026-10-11T17:20:00Z"
def summ(a, open_=False):
    s = {k: a[k] for k in ("arc_id","kind","subject_ref","subject_label","began_at","ended_at","status",
                           "attention_returns","cumulative_dwell_sec","last_seen_at")}
    if open_:
        s.update(status="open", ended_at=None, last_seen_at="2026-10-11T17:19:30Z", attention_returns=1)
    return s
busy = copy.deepcopy(frame)
busy.update(as_of="2026-10-11T17:16:00Z", day_phase="morning", frame_id="busyfixture00000")
before = [a for a in arcs.values() if a["began_at"] <= T and a["kind"] != "reverie" and (a["ended_at"] or "9") <= T]
busy["active_arc"] = summ(arcs["ae0a3cb57ae73ee4"], open_=True)
busy["previous_arc"] = summ(arcs["0296c2551753e179"])
busy["arcs_today"] = sorted([summ(a) for a in before] + [busy["active_arc"]], key=lambda s: (s["began_at"], s["arc_id"]))[-64:]
busy["active_by_kind"] = {"attention": busy["active_arc"]}
busy["open_threads"] = [t for t in frame["open_threads"] if t["first_seen_today"] <= T]
busy["warnings"] = []
json.dump(busy, open(f"{OUT}/frame_2026-10-11_busy.json", "w"), sort_keys=True, separators=(",", ":"))
