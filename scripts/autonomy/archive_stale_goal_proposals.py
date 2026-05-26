#!/usr/bin/env python3
"""Archive duplicate/stale ProposedGoal rows in the autonomy goals graph.

**Production:** goal archive is automated in ``orion-actions`` (startup drain + nightly) and
``orion-spark-concept-induction`` (post-publish trim). Use this CLI for operator dry-run/debug only.

Loads graph URLs from service ``.env`` files when unset (host-side ``orion-athena-fuseki`` → ``127.0.0.1`` rewrite).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def _rewrite_docker_graph_hosts() -> None:
    """When run on the host, Docker service names (orion-athena-fuseki) often do not resolve."""
    import socket

    docker_hosts = ("orion-athena-fuseki", "orion-athena-graphdb")
    for hostname in docker_hosts:
        try:
            socket.getaddrinfo(hostname, None)
            return
        except OSError:
            continue
    for key in (
        "AUTONOMY_GRAPH_QUERY_URL",
        "AUTONOMY_GRAPH_UPDATE_URL",
        "RDF_STORE_QUERY_URL",
        "RDF_STORE_UPDATE_URL",
        "RDF_STORE_BASE_URL",
        "GRAPHDB_URL",
    ):
        raw = os.environ.get(key, "")
        if not raw:
            continue
        for hostname in docker_hosts:
            if hostname in raw:
                os.environ[key] = raw.replace(hostname, "127.0.0.1")


for _env_path in (
    ROOT / "services" / "orion-cortex-exec" / ".env",
    ROOT / "services" / "orion-actions" / ".env",
    ROOT / "services" / "orion-cortex-exec" / ".env_example",
    ROOT / "scripts" / "autonomy" / ".env",
):
    _load_env_file(_env_path)
_rewrite_docker_graph_hosts()

from orion.autonomy.goal_archive import archive_subject_goals, archive_subjects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--subject", default="")
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Archive all subjects in AUTONOMY_GOAL_ARCHIVE_SUBJECTS (default orion,relationship)",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    if args.all_subjects:
        summaries = archive_subjects(dry_run=dry_run)
        for summary in summaries:
            print(summary)
        return 0

    subject = args.subject.strip() or "orion"
    summary = archive_subject_goals(subject, dry_run=dry_run)
    print(
        f"subject={summary.get('subject')} candidates={summary.get('candidates')} "
        f"applied={summary.get('applied')} dry_run={summary.get('dry_run')} "
        f"rows_scanned={summary.get('rows_scanned')}"
    )
    if summary.get("error"):
        print(f"error={summary['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
