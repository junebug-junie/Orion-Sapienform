#!/usr/bin/env python3
"""Mutation check for issue #3364's guards: a want ranks only against past events decided with the same grading,
and the camera-asks replay reproduces the live market (its reference is the market's ledger, read through the
market's 300 s snapshot).

Each mutant breaks one guard; the tests must FAIL on every one. Files are rewritten IN PLACE and restored in a
``finally`` (a SIGKILL mid-run would leave a mutant in the tree; ``git diff`` shows it). No database.

    PYTHONPATH=. python services/orion-durable-runs/evals/event_rank_grading_mutations.py
"""
import os
import subprocess
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
PY = sys.executable
TESTS = ["orion/physiology/tests/test_event_sources.py",
         "services/orion-durable-runs/tests/test_event_wants_menu.py",
         "services/orion-durable-runs/tests/test_camera_asks_replay_parity.py"]
APP = "orion/physiology/appetite.py"
MKT = "services/orion-durable-runs/app/physiology_market.py"
RPL = "services/orion-durable-runs/evals/event_wants_replay.py"
MUTANTS = {
    "pairs of the other grading kept": (APP, "            if len(e) != 2 or not isinstance(e[1], bool) or e[1] != graded:\n",
                                        "            if len(e) != 2:\n"),
    "unknown grading kept": (APP, "            if len(e) != 2 or not isinstance(e[1], bool) or e[1] != graded:\n",
                             "            if len(e) != 2 or (isinstance(e[1], bool) and e[1] != graded):\n"),
    "appetite strips the grading": (APP, "        event_values=same_grading_events(event_intensities, graded))",
                                    "        event_values=same_grading_events([e[0] if isinstance(e, tuple) else e "
                                    "for e in event_intensities], graded))"),
    "market read drops the grading": (MKT, '.append((float(r["intensity"]), g if isinstance(g, bool) else None))',
                                      '.append(float(r["intensity"]))'),
    "market SQL drops the grading": (MKT, "THEN (decision->>'graded')::boolean END AS graded",
                                     "THEN (decision->>'graded')::boolean END AS graded_unused"),
    "replay ignores the ledger": (RPL, '        ref: list[Any] = [(r["intensity"], r["graded"]) for r in led if '
                                      'snap - SOURCE_HISTORY <= r["at"] < snap]',
                                  "        ref: list[Any] = []"),
    "replay has no market snapshot lag": (RPL, "        snap = t - snapshot_lag\n", "        snap = t\n"),
    "replay ignores same_grading": (RPL, "        event_values = (same_grading_events(ref, app.graded) if same_grading",
                                    "        event_values = (same_grading_events([v for v, _ in ref], app.graded) "
                                    "if same_grading"),
    "replay stand-ins after the ledger starts": (RPL, "(ledger_from is None or t < ledger_from) and ", ""),
}


def run():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{W}:{W}/services/orion-durable-runs"}
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS], cwd=W, env=env,
                          capture_output=True, text=True)


def last(r):
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]


base = run()
print("baseline:", last(base))
assert base.returncode == 0, base.stdout[-2000:]
results = {}
for name, (rel, a, b) in MUTANTS.items():
    f = W / rel
    orig = f.read_text()
    try:
        assert orig.count(a) == 1, f"mutant {name!r} does not apply"
        f.write_text(orig.replace(a, b))
        r = run()
        results[name] = "KILLED" if r.returncode != 0 else "SURVIVED"
        print(f"{results[name]:9s} {name}  ({last(r)})")
    finally:
        f.write_text(orig)
sys.exit(0 if all(v == "KILLED" for v in results.values()) else 1)
