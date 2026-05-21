#!/usr/bin/env python3
"""Archive duplicate/stale ProposedGoal rows in the autonomy goals graph.

Run once after Phase 0 deploy with ``--apply`` (default is dry-run).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from orion.autonomy.repository import AUTONOMY_GOALS_GRAPH

AUTONOMY_GOALS_GRAPH_URI = AUTONOMY_GOALS_GRAPH


def build_archive_candidates(
    rows: list[dict],
    *,
    max_active_per_subject: int,
    retention_days: int,
) -> list[str]:
    _ = datetime.now(timezone.utc) - timedelta(days=retention_days)
    by_origin: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("proposal_status") in {"archived", "superseded"}:
            continue
        by_origin[str(row["drive_origin"])].append(row)
    to_archive: list[str] = []
    for _origin, group in by_origin.items():
        by_base: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_base[str(row.get("goal_statement_base") or "")].append(row)
        survivors: list[dict] = []
        for base_rows in by_base.values():
            best = max(base_rows, key=lambda r: (float(r.get("priority", 0)), str(r["artifact_id"])))
            survivors.append(best)
        ranked = sorted(survivors, key=lambda r: (-float(r.get("priority", 0)), str(r["artifact_id"])))
        keep = ranked[:max_active_per_subject]
        keep_ids = {r["artifact_id"] for r in keep}
        for row in group:
            if row["artifact_id"] not in keep_ids:
                to_archive.append(str(row["artifact_id"]))
    return to_archive


def _resolve_graph_query_config() -> tuple[str, str | None, str | None, str | None]:
    """Return (query_endpoint, update_endpoint, user, password)."""
    from orion.graph.backend_config import resolve_autonomy_read_query_url, resolve_graph_update_url, resolve_rdf_store_auth

    query_url, _src = resolve_autonomy_read_query_url()
    if query_url:
        update_url = resolve_graph_update_url() or query_url
        user, password = resolve_rdf_store_auth()
        return query_url, update_url, user, password

    base = os.getenv("GRAPHDB_URL", "").strip()
    if not base:
        raise SystemExit("AUTONOMY_GRAPH_QUERY_URL or GRAPHDB_URL required")
    repo = (os.getenv("GRAPHDB_REPO") or "collapse").strip()
    endpoint = base if "/repositories/" in base else f"{base.rstrip('/')}/repositories/{repo}"
    user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
    password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None
    return endpoint, endpoint, user, password


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--subject", default="orion")
    args = parser.parse_args()
    dry_run = not args.apply
    max_active = int(os.getenv("AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT", "3"))
    retention = int(os.getenv("AUTONOMY_GOAL_RETENTION_DAYS", "30"))
    from orion.spark.concept_induction.graph_query import GraphQueryClient, GraphQueryConfig

    query_endpoint, update_endpoint, user, password = _resolve_graph_query_config()
    client = GraphQueryClient(
        GraphQueryConfig(
            endpoint=query_endpoint,
            update_endpoint=update_endpoint,
            graph_uri=AUTONOMY_GOALS_GRAPH_URI,
            timeout_sec=float(os.getenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "30")),
            user=user,
            password=password,
        )
    )
    select = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?drive_origin ?goal_statement ?priority ?proposal_status
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH_URI}> {{
    ?a a orion:ProposedGoal ; orion:subjectKey "{args.subject}" ;
      orion:artifactId ?artifact_id ; orion:driveOrigin ?drive_origin ;
      orion:goalStatement ?goal_statement ; orion:proposalPriority ?priority .
    OPTIONAL {{ ?a orion:proposalStatus ?proposal_status . }}
  }}
}}"""
    raw_rows = client.select(select)
    rows = [
        {
            "artifact_id": r["artifact_id"]["value"],
            "drive_origin": r["drive_origin"]["value"],
            "goal_statement_base": r["goal_statement"]["value"].split(" · ", 1)[0],
            "priority": r["priority"]["value"],
            "proposal_status": (r.get("proposal_status") or {}).get("value", "proposed"),
        }
        for r in raw_rows
    ]
    to_archive = build_archive_candidates(rows, max_active_per_subject=max_active, retention_days=retention)
    print(f"candidates={len(to_archive)} dry_run={dry_run}")
    if dry_run:
        for aid in to_archive[:20]:
            print(f"  would archive {aid}")
        return 0
    for aid in to_archive:
        update = f"""
PREFIX orion: <http://conjourney.net/orion#>
DELETE {{ ?a orion:proposalStatus ?old . }}
INSERT {{ ?a orion:proposalStatus "archived" . }}
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH_URI}> {{
    ?a orion:artifactId "{aid}" .
    OPTIONAL {{ ?a orion:proposalStatus ?old . }}
  }}
}}"""
        client.update(update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
