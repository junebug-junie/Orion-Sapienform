"""Archive stale ProposedGoal rows in the autonomy goals graph."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from orion.autonomy.constants import AUTONOMY_GOALS_GRAPH
from orion.graph.backend_config import resolve_autonomy_read_query_url, resolve_graph_update_url, resolve_rdf_store_auth
from orion.graph.sparql_client import SparqlHttpClient

logger = logging.getLogger("orion.autonomy.goal_archive")

AUTONOMY_GOALS_GRAPH_URI = AUTONOMY_GOALS_GRAPH
DEFAULT_ARCHIVE_SUBJECTS = ("orion", "relationship")


class GoalArchiveError(RuntimeError):
    pass


def build_archive_candidates(
    rows: list[dict],
    *,
    max_active_per_subject: int,
    retention_days: int,
) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    by_origin: dict[str, list[dict]] = defaultdict(list)
    to_archive: list[str] = []
    for row in rows:
        if row.get("proposal_status") in {"archived", "superseded"}:
            continue
        created_at = row.get("created_at")
        if isinstance(created_at, datetime) and created_at < cutoff:
            to_archive.append(str(row["artifact_id"]))
            continue
        by_origin[str(row["drive_origin"])].append(row)
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


def _resolve_graph_endpoints() -> tuple[str, str, str | None, str | None]:
    query_url, _src = resolve_autonomy_read_query_url()
    if query_url:
        update_url = resolve_graph_update_url() or query_url
        user, password = resolve_rdf_store_auth()
        return query_url, update_url, user, password

    base = os.getenv("GRAPHDB_URL", "").strip()
    if not base:
        raise GoalArchiveError("AUTONOMY_GRAPH_QUERY_URL or GRAPHDB_URL required")
    repo = (os.getenv("GRAPHDB_REPO") or "collapse").strip()
    endpoint = base if "/repositories/" in base else f"{base.rstrip('/')}/repositories/{repo}"
    user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
    password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None
    return endpoint, endpoint, user, password


def _build_client(*, timeout_sec: float) -> SparqlHttpClient:
    query_endpoint, update_endpoint, user, password = _resolve_graph_endpoints()
    return SparqlHttpClient(
        query_endpoint,
        update_endpoint,
        timeout_sec=timeout_sec,
        user=user,
        password=password,
    )


def _binding_value(row: dict[str, Any], key: str) -> str | None:
    cell = row.get(key)
    if isinstance(cell, dict):
        raw = cell.get("value")
        return str(raw) if raw is not None else None
    return None


def _fetch_goal_rows(client: SparqlHttpClient, *, subject: str) -> list[dict]:
    select = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?drive_origin ?goal_statement ?priority ?proposal_status ?created_at
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH_URI}> {{
    ?a a orion:ProposedGoal ; orion:subjectKey "{subject}" ;
      orion:artifactId ?artifact_id ; orion:driveOrigin ?drive_origin ;
      orion:goalStatement ?goal_statement ; orion:proposalPriority ?priority .
    OPTIONAL {{ ?a orion:proposalStatus ?proposal_status . }}
    OPTIONAL {{ ?a orion:timestamp ?created_at . }}
  }}
}}"""
    raw_rows = client.select(select)
    rows: list[dict] = []
    for r in raw_rows:
        created_raw = _binding_value(r, "created_at")
        created_at = None
        if isinstance(created_raw, str) and created_raw.strip():
            try:
                created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            except ValueError:
                created_at = None
        artifact_id = _binding_value(r, "artifact_id")
        drive_origin = _binding_value(r, "drive_origin")
        goal_statement = _binding_value(r, "goal_statement") or ""
        priority = _binding_value(r, "priority") or "0"
        if not artifact_id or not drive_origin:
            continue
        rows.append(
            {
                "artifact_id": artifact_id,
                "drive_origin": drive_origin,
                "goal_statement_base": goal_statement.split(" · ", 1)[0],
                "priority": priority,
                "proposal_status": _binding_value(r, "proposal_status") or "proposed",
                "created_at": created_at,
            }
        )
    return rows


def archive_subject_goals(
    subject: str,
    *,
    dry_run: bool = True,
    max_active_per_subject: int | None = None,
    retention_days: int | None = None,
    max_updates: int | None = None,
) -> dict[str, object]:
    """Archive duplicate/stale goals for one subject. Returns summary dict."""
    max_active = max_active_per_subject if max_active_per_subject is not None else int(
        os.getenv("AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT", "3")
    )
    retention = retention_days if retention_days is not None else int(os.getenv("AUTONOMY_GOAL_RETENTION_DAYS", "30"))
    cap = max_updates if max_updates is not None else int(os.getenv("AUTONOMY_GOAL_ARCHIVE_MAX_UPDATES", "200"))
    timeout_sec = float(os.getenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "30"))

    client = _build_client(timeout_sec=timeout_sec)
    rows = _fetch_goal_rows(client, subject=subject)
    to_archive = build_archive_candidates(rows, max_active_per_subject=max_active, retention_days=retention)
    if cap > 0:
        to_archive = to_archive[:cap]

    applied = 0
    if not dry_run:
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
            applied += 1

    summary = {
        "subject": subject,
        "candidates": len(to_archive),
        "applied": applied,
        "dry_run": dry_run,
        "rows_scanned": len(rows),
    }
    logger.info(
        "autonomy_goal_archive subject=%s candidates=%s applied=%s dry_run=%s rows_scanned=%s",
        subject,
        summary["candidates"],
        summary["applied"],
        dry_run,
        summary["rows_scanned"],
    )
    return summary


def archive_subjects_drain(
    subjects: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
    max_rounds: int | None = None,
) -> dict[str, object]:
    """Repeated archive rounds for backlog drain (orion-actions startup / first deploy)."""
    rounds_cap = max_rounds if max_rounds is not None else int(
        os.getenv("AUTONOMY_GOAL_ARCHIVE_BOOTSTRAP_MAX_ROUNDS", "30")
    )
    rounds_cap = max(1, min(rounds_cap, 200))
    per_round_cap = int(os.getenv("AUTONOMY_GOAL_ARCHIVE_MAX_UPDATES", "200"))
    totals: dict[str, object] = {
        "dry_run": dry_run,
        "rounds": 0,
        "applied": 0,
        "candidates": 0,
        "subjects": [],
    }
    subject_summaries: list[dict[str, object]] = []
    for _round in range(rounds_cap):
        summaries = archive_subjects(subjects, dry_run=dry_run)
        totals["rounds"] = int(totals["rounds"]) + 1  # type: ignore[assignment]
        subject_summaries = summaries
        if any("error" in s for s in summaries):
            totals["error"] = "subject_errors"
            totals["subjects"] = summaries
            break
        round_candidates = sum(int(s.get("candidates") or 0) for s in summaries)
        round_applied = sum(int(s.get("applied") or 0) for s in summaries)
        totals["candidates"] = int(totals["candidates"]) + round_candidates  # type: ignore[assignment]
        totals["applied"] = int(totals["applied"]) + round_applied  # type: ignore[assignment]
        if dry_run or round_candidates == 0 or (not dry_run and round_applied == 0):
            totals["subjects"] = summaries
            break
        if round_candidates < per_round_cap:
            totals["subjects"] = summaries
            break
    else:
        totals["subjects"] = subject_summaries
        totals["truncated"] = True
    logger.info(
        "autonomy_goal_archive_drain rounds=%s applied=%s candidates=%s dry_run=%s truncated=%s",
        totals.get("rounds"),
        totals.get("applied"),
        totals.get("candidates"),
        dry_run,
        totals.get("truncated", False),
    )
    return totals


def archive_subjects(
    subjects: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    raw = os.getenv("AUTONOMY_GOAL_ARCHIVE_SUBJECTS", ",".join(DEFAULT_ARCHIVE_SUBJECTS))
    subject_list = list(subjects) if subjects is not None else [s.strip() for s in raw.split(",") if s.strip()]
    out: list[dict[str, object]] = []
    for subject in subject_list:
        try:
            out.append(archive_subject_goals(subject, dry_run=dry_run))
        except (GoalArchiveError, OSError) as exc:
            logger.warning("autonomy_goal_archive_failed subject=%s error=%s", subject, exc)
            out.append({"subject": subject, "error": str(exc), "dry_run": dry_run})
        except Exception as exc:
            logger.warning("autonomy_goal_archive_failed subject=%s error=%s", subject, exc)
            out.append({"subject": subject, "error": str(exc), "dry_run": dry_run})
    return out


def goal_archive_enabled() -> bool:
    return str(os.getenv("AUTONOMY_GOAL_ARCHIVE_ENABLED", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def maybe_archive_after_goal_publish(*, subject: str) -> None:
    """Lightweight post-publish trim when AUTONOMY_GOAL_ARCHIVE_ENABLED=true."""
    if not goal_archive_enabled():
        return
    max_updates = int(os.getenv("AUTONOMY_GOAL_ARCHIVE_MAX_UPDATES_PER_TICK", "25"))
    try:
        archive_subject_goals(subject, dry_run=False, max_updates=max_updates)
    except (GoalArchiveError, OSError) as exc:
        logger.warning("autonomy_goal_archive_tick_failed subject=%s error=%s", subject, exc)
