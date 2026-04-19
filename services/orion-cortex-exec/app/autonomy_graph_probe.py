from __future__ import annotations

import argparse
import json

from app.chat_stance import (
    resolve_autonomy_graph_timeout_sec,
    resolve_autonomy_graphdb_config,
    resolve_autonomy_subject_max_workers,
    resolve_autonomy_subquery_max_workers,
)
from orion.autonomy.repository import GraphAutonomyRepository


def run_probe(subjects: list[str]) -> list[dict[str, object]]:
    cfg = resolve_autonomy_graphdb_config()
    timeout_sec = resolve_autonomy_graph_timeout_sec()
    repository = GraphAutonomyRepository(
        endpoint=cfg.get("endpoint"),
        timeout_sec=timeout_sec,
        user=cfg.get("user"),
        password=cfg.get("password"),
        subject_max_workers=resolve_autonomy_subject_max_workers(),
        subquery_max_workers=resolve_autonomy_subquery_max_workers(),
    )
    lookups = repository.list_latest(subjects, observer={"consumer": "autonomy_graph_probe"})
    return [
        {
            "subject": item.subject,
            "availability": item.availability,
            "unavailable_reason": item.unavailable_reason,
            "subqueries": item.subquery_diagnostics or {},
        }
        for item in lookups
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomy GraphDB subquery diagnostics.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=["orion", "relationship", "juniper"],
        help="Subjects to probe (default: orion relationship juniper)",
    )
    args = parser.parse_args()
    results = run_probe(list(args.subjects))
    print(json.dumps({"subjects": args.subjects, "results": results}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
