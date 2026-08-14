#!/usr/bin/env python3
"""Definition-change alert: tell Juniper when an agent edits a metric's meaning.

    python scripts/check_definition_drift.py            # report vs the lock
    python scripts/check_definition_drift.py --gate     # exit 1 on any drift
    python scripts/check_definition_drift.py --update   # re-lock, print deltas
    python scripts/check_definition_drift.py --json

WHY THIS EXISTS
---------------
R4 of docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md. Juniper's
ask, verbatim: bus streams and organ signals do not need a liveness verdict,
they need "a gate to flag it to me when an agent starts to fuck around in
there".

The failure it replaces is on record. `execution_load`, `bus_health` and
`transport_pressure` were renamed on 2026-07-24. Three weeks later
`execution_load` was still sitting in all four live node vectors, frozen at
0.2672 -- not zero, a plausible-looking reading with no producer behind it,
which any generic consumer iterating the vector reads as real. Nothing
announced the removal, and the PR that did the renaming looked like a routine
find-and-replace. Found by hand on 2026-08-14.

HOW THE ALERT REACHES JUNIPER
-----------------------------
Not a notification channel. The lock file itself:

    config/metrics/metric_definitions.lock.json

The gate goes red the moment a PR changes a resolved definition. The only way
to make it green is `--update`, which rewrites the lock AND records the
classified deltas into the lock's own `_last_change` block. So the PR diff
Juniper reads contains a plain-English line saying exactly what changed:

    "high  removed  metric://field_channel/orion-field-digester/execution_load"

An agent cannot re-lock quietly, because re-locking is what writes the
sentence. That is the whole mechanism -- there is no separate reporting path
to forget to call.

WHY IT IS A LOCK AND NOT A RATCHET
----------------------------------
`orphan_baseline.json` and `merge_domination_baseline.json` are ratchets: they
may shrink, never grow, because an orphan and a dominated merge are both
defects. A definition change is not a defect. It is an event. So this file is
a lock -- it tracks the current truth exactly, in both directions, and its diff
is the deliverable.

STATIC BY CONSTRUCTION
----------------------
Reads four registries and nothing else: no Postgres, no Redis, no bus. It
therefore runs in .github/workflows/orion-static-gates.yml alongside
check_metric_lineage.py, which already imports the same graph under the same
minimal dep set (pydantic, pydantic-settings, PyYAML).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# scripts/ on sys.path[0] shadows stdlib `platform` via scripts/platform/ and
# breaks pydantic -- same fix as check_metric_lineage.py.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metrics.definitions import (  # noqa: E402
    SEVERITY,
    build_lock,
    diff_locks,
    format_report,
)
from orion.metrics.lineage import build_graph  # noqa: E402

LOCK_PATH = REPO_ROOT / "config" / "metrics" / "metric_definitions.lock.json"

LOCK_COMMENT = (
    "Definition lock for scripts/check_definition_drift.py. Generated, never "
    "hand-edited. Regenerate with --update; the resulting diff IS the "
    "definition-change alert. Unlike the *_baseline.json ratchets this tracks "
    "current truth in both directions -- a definition change is an event, not "
    "a defect."
)


def _load_lock() -> tuple[dict, dict]:
    """Returns (definitions, whole_file). Missing file -> empty definitions."""
    if not LOCK_PATH.exists():
        return {}, {}
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return data.get("definitions", {}), data


def _write_lock(definitions: dict, diff, *, first_run: bool) -> None:
    if first_run:
        # Every metric is technically "added" against an absent lock. Recording
        # 595 additions would make the one file a reader is meant to scan for
        # real events open with 595 non-events.
        last_change = {
            "change_count": 0,
            "high_severity_count": 0,
            "changes": ["initial lock -- no prior state to diff against"],
        }
    else:
        last_change = {
            "change_count": len(diff.changes),
            "high_severity_count": len(diff.high),
            # Plain sentences, not structured deltas: this block exists to be
            # READ in a PR diff, and a nested object of before/after tuples is
            # not read, it is scrolled past.
            "changes": [
                f"{change.severity:<6} {change.describe()}" for change in diff.changes
            ],
        }
    payload = {
        "_comment": LOCK_COMMENT,
        "metric_count": len(definitions),
        "_last_change": last_change,
        "definitions": definitions,
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", action="store_true", help="Exit 1 if any definition drifted."
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the lock from the current registries and record the deltas.",
    )
    args = parser.parse_args(argv)

    graph = build_graph()
    current = build_lock(graph)
    locked, whole = _load_lock()

    first_run = not whole
    diff = diff_locks(locked, current)

    if args.update:
        _write_lock(current, diff, first_run=first_run)
        rel = LOCK_PATH.relative_to(REPO_ROOT)
        if first_run:
            print(f"{rel}: created, {len(current)} metric definitions locked")
            return 0
        print(f"{rel}: updated, {len(current)} metric definitions locked")
        for line in format_report(diff):
            print(line)
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    "locked_count": len(locked),
                    "current_count": len(current),
                    "lock_present": not first_run,
                    "change_count": len(diff.changes),
                    "high_severity_count": len(diff.high),
                    "severity_scale": SEVERITY,
                    "changes": [
                        {
                            "kind": c.kind,
                            "severity": c.severity,
                            "surface": c.surface,
                            "urn": c.urn,
                            "previous_urn": c.previous_urn,
                            "fields": {
                                k: {"before": b, "after": a}
                                for k, (b, a) in c.fields.items()
                            },
                            "describe": c.describe(),
                        }
                        for c in diff.changes
                    ],
                },
                indent=2,
            )
        )
    else:
        if first_run:
            print(
                f"no lock at {LOCK_PATH.relative_to(REPO_ROOT)} -- "
                f"run --update to create it ({len(current)} definitions)"
            )
        else:
            print(
                f"{len(current)} metric definitions "
                f"({len(diff.changes)} changed, {len(diff.high)} high severity)\n"
            )
            for line in format_report(diff):
                print(line)

    if args.gate and (diff or first_run):
        print("\ndefinition drift gate: FAIL", file=sys.stderr)
        if first_run:
            print(
                "  no committed lock to compare against; run --update and commit it",
                file=sys.stderr,
            )
        else:
            print(
                "  a metric definition changed. This is not automatically wrong --\n"
                "  re-lock with `python scripts/check_definition_drift.py --update`\n"
                "  and COMMIT the lock, so the change is stated in the PR diff.",
                file=sys.stderr,
            )
        return 1
    if args.gate:
        print("\ndefinition drift gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
