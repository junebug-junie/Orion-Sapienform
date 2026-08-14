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

WHY `_last_change` IS COMPUTED AGAINST THE MERGE BASE, NOT THE FILE ON DISK
---------------------------------------------------------------------------
The obvious implementation -- diff the current registries against whatever the
lock currently holds -- is last-write-wins, and it fails three ways that were
all found in review of this file's own first version:

1. Two `--update` calls on one branch: the first call's deltas are overwritten
   by the second call's, so a high-severity consumer removal can vanish from
   the alert while the gate stays green.
2. A lock clobbered to `{}` by a botched merge resolution reads as "no lock
   yet", so `--update` writes "initial lock" and discards every real delta.
3. It shipped a false alert. The first version of this lock was committed
   carrying two fabricated high-severity sentences -- residue of a mutation
   test that locked a mutated registry state, then reverted it and re-locked,
   so `_last_change` recorded the REVERT as though it were the change.

So `_last_change` answers one fixed question -- "what does this branch change
relative to the merge base?" -- which is idempotent under repeated `--update`,
immune to the state of the working copy's lock, and the question a PR reader
is actually asking.

`--gate` then RECOMPUTES that block and fails if the committed one disagrees.
That is what makes the sentence a constraint rather than a convention: hand-
editing `_last_change` to erase an alert, or resolving a merge conflict in it
by taking the other side, both fail the gate. Without that check the whole
mechanism was advisory, and this docstring previously claimed otherwise.

When the merge base cannot be resolved (no git, shallow clone with no
`origin/main`), the recomputation is SKIPPED WITH AN EXPLICIT PRINTED NOTE --
never silently, because a silently-skipped integrity check is the failure this
file exists to stop.

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


BASE_REFS = ("origin/main", "main")


def _rel(path: Path) -> str:
    """Repo-relative path, falling back to the absolute one.

    `Path.relative_to` RAISES for anything outside REPO_ROOT, which turned a
    redirected LOCK_PATH into a ValueError traceback rather than a report.
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)

NO_PRIOR_STATE = "initial lock -- no prior state to diff against"
BASE_UNAVAILABLE = "merge base unavailable -- deltas not computed"


def _load_lock() -> tuple[dict, dict, bool]:
    """Returns (definitions, whole_file, exists).

    `exists` is the FILE's existence, deliberately not `bool(whole_file)`: a
    lock clobbered to `{}` is falsy but is emphatically not a first run, and
    conflating the two made `--update` announce "initial lock" while dropping
    every real delta on the floor.
    """
    if not LOCK_PATH.exists():
        return {}, {}, False
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return data.get("definitions", {}), data, True


def _base_definitions() -> tuple[dict | None, str]:
    """The lock's `definitions` at the merge base with main.

    Returns (definitions, note). `None` means the base could not be resolved
    at all -- distinct from `{}`, which means the base genuinely predates this
    lock file (the commit that introduces it).
    """
    import subprocess

    rel = _rel(LOCK_PATH)
    for ref in BASE_REFS:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", ref],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if merge_base.returncode != 0:
            continue
        base = merge_base.stdout.strip()
        show = subprocess.run(
            ["git", "show", f"{base}:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if show.returncode != 0:
            # Ref resolved but the lock does not exist there -- this branch
            # introduces it. A real answer, not a failure.
            return {}, f"merge base {base[:9]} ({ref}) has no lock yet"
        try:
            data = json.loads(show.stdout)
        except json.JSONDecodeError:
            return None, f"merge base {base[:9]} ({ref}) has an unparsable lock"
        return data.get("definitions", {}), f"merge base {base[:9]} ({ref})"
    return None, f"no merge base against any of {', '.join(BASE_REFS)}"


def _base_is_head() -> bool:
    """True when the merge base IS this commit -- i.e. we are standing on the base.

    The recomputation below asks "what does this BRANCH change relative to the merge
    base?". On ``main`` itself that question has no meaning: the merge base is HEAD,
    so the diff is empty by construction and the derived block collapses to "no
    definition changes" -- while the committed block still states what the branch that
    last landed here actually changed. The two can never agree, so the gate failed on
    ``main`` after every definition change, by construction.

    That is not hypothetical. Both post-gate commits on main -- 4cca1eb5f (the PR that
    introduced this file) and b4a697a9d -- were red for exactly this reason, and any
    branch cut from a red main inherited the failure, so re-locking alone would have
    been a treadmill rather than a fix.

    Skipping the check here does not weaken it: on a real PR branch HEAD is never the
    merge base, so hand-editing ``_last_change`` to erase an alert is still caught
    where it matters -- in the PR that does the editing.
    """
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if head.returncode != 0:
        return False
    for ref in BASE_REFS:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", ref], cwd=REPO_ROOT, capture_output=True, text=True
        )
        if merge_base.returncode != 0:
            continue
        return merge_base.stdout.strip() == head.stdout.strip()
    return False


def _last_change_block(base: dict | None, current: dict, note: str) -> dict:
    """The alert block, derived from the merge base. Pure given its inputs."""
    if base is None:
        return {
            "base": note,
            "change_count": None,
            "high_severity_count": None,
            "changes": [BASE_UNAVAILABLE],
        }
    if not base:
        # Every metric is technically "added" against an absent base lock.
        # Recording 595 additions would make the one file a reader is meant to
        # scan for real events open with 595 non-events.
        return {
            "base": note,
            "change_count": 0,
            "high_severity_count": 0,
            "changes": [NO_PRIOR_STATE],
        }
    diff = diff_locks(base, current)
    return {
        "base": note,
        "change_count": len(diff.changes),
        "high_severity_count": len(diff.high),
        # Plain sentences, not structured deltas: this block exists to be READ
        # in a PR diff, and a nested object of before/after tuples is not read,
        # it is scrolled past.
        "changes": [
            f"{change.severity:<6} {change.describe()}" for change in diff.changes
        ]
        or ["no definition changes relative to the merge base"],
    }


def _write_lock(definitions: dict, last_change: dict) -> None:
    payload = {
        "_comment": LOCK_COMMENT,
        "metric_count": len(definitions),
        "_last_change": last_change,
        "definitions": definitions,
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


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

    if args.update and args.gate:
        # Not merely redundant: --update returns before the gate block, so this
        # combination would rewrite the lock and exit 0 with the gate silently
        # skipped -- an invocation that looks like it enforced and did not.
        parser.error("--update and --gate are mutually exclusive")

    graph = build_graph()
    current = build_lock(graph)
    locked, whole, lock_exists = _load_lock()
    rel = _rel(LOCK_PATH)

    diff = diff_locks(locked, current)
    base, base_note = _base_definitions()

    if args.update:
        block = _last_change_block(base, current, base_note)
        _write_lock(current, block)
        verb = "updated" if lock_exists else "created"
        print(f"{rel}: {verb}, {len(current)} metric definitions locked")
        print(f"  deltas computed against {base_note}")
        if base is None:
            print(f"  WARNING: {BASE_UNAVAILABLE} -- _last_change is not authoritative")
        # Echo the block's OWN sentences rather than re-deriving a report, so
        # the console and the committed file can never describe the same
        # update differently.
        for line in block["changes"]:
            print(f"  {line}")
        return 0

    # The committed alert block must match what the merge base says it should
    # be. Without this the sentence is a convention an agent can hand-edit
    # away; with it, erasing the alert fails the gate.
    stale_alert: str | None = None
    on_base_branch = _base_is_head()
    if lock_exists and base is not None and not on_base_branch:
        expected = _last_change_block(base, locked, base_note)
        committed = whole.get("_last_change") or {}
        if committed.get("changes") != expected["changes"]:
            stale_alert = (
                "committed _last_change does not match the merge-base diff\n"
                f"    committed: {committed.get('changes')}\n"
                f"    expected : {expected['changes']}"
            )

    if args.json:
        print(
            json.dumps(
                {
                    "locked_count": len(locked),
                    "current_count": len(current),
                    "lock_present": lock_exists,
                    "merge_base": base_note,
                    "merge_base_resolved": base is not None,
                    "stale_alert": stale_alert,
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
        if not lock_exists:
            print(f"no lock at {rel} -- run --update to create it ({len(current)} defs)")
        else:
            print(
                f"{len(current)} metric definitions "
                f"({len(diff.changes)} changed, {len(diff.high)} high severity)\n"
            )
            for line in format_report(diff):
                print(line)
        if base is None:
            print(f"\nNOTE: {BASE_UNAVAILABLE} ({base_note});")
            print("      committed _last_change was NOT verified this run.")
        elif on_base_branch:
            # Stated, never silent -- a silently-skipped integrity check is the
            # failure this file's own docstring calls out.
            print("\nNOTE: HEAD is the merge base (on the base branch, not a PR);")
            print("      _last_change describes the branch that landed here and is")
            print("      not recomputable from this vantage point. Not verified.")

    if args.gate and (diff or not lock_exists or stale_alert):
        print("\ndefinition drift gate: FAIL", file=sys.stderr)
        if not lock_exists:
            print(
                "  no committed lock to compare against; run --update and commit it",
                file=sys.stderr,
            )
        if diff:
            print(
                "  a metric definition changed. This is not automatically wrong --\n"
                "  re-lock with `python scripts/check_definition_drift.py --update`\n"
                "  and COMMIT the lock, so the change is stated in the PR diff.",
                file=sys.stderr,
            )
        if stale_alert:
            print(
                f"  {stale_alert}\n"
                "  The alert block is derived, not editable. Re-run --update.",
                file=sys.stderr,
            )
        return 1
    if args.gate:
        print("\ndefinition drift gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
