#!/usr/bin/env python3
"""Every newly added SQL migration must say which services cannot run without it.

``scripts/safe_docker_build.sh <svc> up`` refuses to bring a service up while a migration whose
header says ``-- ORION-MIGRATION-REQUIRED-BY: <svc>`` is unapplied (2026-10-10/11: PRs #2594 and
#2605 deployed orion-durable-runs before their migrations; every step failed with UndefinedTable).
That gate is only as good as the declarations, so this CI check makes them mandatory going forward:

1. Every ``manual_migration_*.sql`` ADDED relative to the merge base with ``--base`` (default
   ``origin/main``) must declare ``ORION-MIGRATION-REQUIRED-BY: <svc>[, <svc>...]`` or
   ``ORION-MIGRATION-REQUIRED-BY: none <reason>``. ``*_rollback.sql`` and NOT-A-MIGRATION files
   are exempt (they are never part of the expected schema).
2. Every declaration anywhere in the corpus must parse, name real ``services/<svc>/`` directories,
   and a ``-- DESTRUCTIVE`` file may only declare ``none`` (it must never be demanded by a deploy).

Static: needs git, no database. Exit 0 = clean, 1 = violations, 2 = could not resolve the base.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion import sql_migration_drift as drift  # noqa: E402

MIG_DIR = drift.MIGRATION_SUBDIR


def _exempt(name: str, text: str) -> bool:
    return drift.is_rollback_file(name) or bool(drift._NOT_A_MIGRATION.search(text))


def check_files(repo: Path, new_names: set[str]) -> list[str]:
    """Pure-ish core: the corpus under ``repo`` plus which basenames are new in this change."""
    errors: list[str] = []
    for p in sorted((repo / MIG_DIR).glob("*.sql")):
        text = p.read_text(errors="replace")
        rb = drift.parse_required_by(text)
        rel = f"{MIG_DIR}/{p.name}"
        if rb is None:
            if p.name in new_names and p.name.startswith("manual_migration_") and not _exempt(p.name, text):
                errors.append(
                    f"{rel}: new migration declares no consumer. Add a header line\n"
                    f"    -- ORION-MIGRATION-REQUIRED-BY: <service>[, <service>]   (services that fail without it)\n"
                    f"  or\n"
                    f"    -- ORION-MIGRATION-REQUIRED-BY: none <reason>"
                )
            continue
        for e in rb.errors:
            errors.append(f"{rel}: {e}")
        for svc in rb.services:
            if not (repo / "services" / svc / "docker-compose.yml").is_file():
                errors.append(f"{rel}: REQUIRED-BY names {svc!r}, but services/{svc}/docker-compose.yml "
                              "does not exist")
        if rb.services and not drift.is_destructive(text) and not drift.has_unconditional_effects(text):
            errors.append(f"{rel}: REQUIRED-BY on a data-only/fully guarded migration over-promises -- "
                          "the schema cannot show it ran, so the deploy gate could never enforce it; "
                          "declare 'ORION-MIGRATION-REQUIRED-BY: none <reason>'")
        if rb.services and drift.is_destructive(text):
            errors.append(f"{rel}: a DESTRUCTIVE migration can never be a deploy dependency; "
                          "declare 'ORION-MIGRATION-REQUIRED-BY: none <reason>'")
    return errors


def new_migration_names(repo: Path, base: str) -> set[str]:
    """Basenames added since the merge base with ``base``: committed, staged, unstaged, untracked."""
    mb = subprocess.run(["git", "-C", str(repo), "merge-base", "HEAD", base],
                        capture_output=True, text=True)
    if mb.returncode != 0:
        raise RuntimeError(f"cannot find a merge base with {base!r}: {mb.stderr.strip()}")
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", "--no-renames", "--diff-filter=A", mb.stdout.strip(),
         "--", str(MIG_DIR)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    out += subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "--", str(MIG_DIR)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return {Path(x).name for x in out if x.endswith(".sql")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main", help="ref whose merge base defines 'new' (default origin/main)")
    ap.add_argument("--repo", default=str(REPO_ROOT))
    args = ap.parse_args(argv)
    repo = Path(args.repo)
    try:
        new = new_migration_names(repo, args.base)
    except Exception as exc:  # noqa: BLE001
        print(f"check_migration_required_by: {exc}", file=sys.stderr)
        return 2
    errors = check_files(repo, new)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"\n{len(errors)} migration declaration problem(s). The deploy gate in "
              "scripts/safe_docker_build.sh reads these headers.", file=sys.stderr)
        return 1
    print(f"migration REQUIRED-BY declarations OK ({len(new)} new migration file(s) vs {args.base})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
