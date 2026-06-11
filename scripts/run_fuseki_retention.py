#!/usr/bin/env python3
"""Run SPARQL retention pruning against Fuseki (cron / operator use)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from orion.graph.rdf_retention import parse_retention_policies, run_retention_pass


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or not str(val).strip():
        return default
    return val.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune aged RDF graphs in Fuseki via SPARQL UPDATE")
    parser.add_argument("--dry-run", action="store_true", help="Log updates without executing")
    parser.add_argument(
        "--policies",
        default=_env("RDF_RETENTION_POLICIES"),
        help="JSON policy array (default: RDF_RETENTION_POLICIES env or built-in defaults)",
    )
    args = parser.parse_args()

    enabled = (_env("RDF_RETENTION_ENABLED", "true") or "true").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        print("RDF_RETENTION_ENABLED=false; skipping")
        return 0

    dry_run = args.dry_run or (_env("RDF_RETENTION_DRY_RUN", "false") or "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    base = (_env("RDF_STORE_BASE_URL") or "http://orion-athena-fuseki:3030").rstrip("/")
    dataset = (_env("RDF_STORE_DATASET") or "orion").strip("/")
    query_url = _env("RDF_STORE_QUERY_URL") or f"{base}/{dataset}/query"
    update_url = _env("RDF_STORE_UPDATE_URL") or f"{base}/{dataset}/update"
    user = _env("RDF_STORE_USER") or "admin"
    password = _env("RDF_STORE_PASS") or _env("FUSEKI_ADMIN_PASSWORD") or "orion"
    timeout = float(_env("RDF_RETENTION_TIMEOUT_SEC") or "300")

    policies = parse_retention_policies(args.policies)
    results = run_retention_pass(
        policies=policies,
        query_url=query_url,
        update_url=update_url,
        user=user,
        password=password,
        dry_run=dry_run,
        timeout_sec=timeout,
    )

    summary = [
        {
            "graph": r.graph,
            "deleted_by_age": r.deleted_by_age,
            "deleted_by_cap": r.deleted_by_cap,
            "dry_run": r.dry_run,
            "errors": r.errors,
        }
        for r in results
    ]
    print(json.dumps(summary, indent=2))
    return 1 if any(r.errors for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
