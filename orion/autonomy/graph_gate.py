"""V1 cutover: explicit gate for autonomy GraphDB SPARQL reads (stance / substrate adapter).

Global ``GRAPHDB_URL`` / ``GRAPHDB_REPO`` must **not** enable autonomy graph reads.
Only ``AUTONOMY_GRAPH_BACKEND=graphdb`` activates the graph path (with a resolved endpoint).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS

logger = logging.getLogger("orion.autonomy.graph_gate")

_AUTONOMY_SUBJECT_ORDER = ("orion", "relationship", "juniper")
_SUBQUERY_ORDER = ("identity", "drives", "goals")

AutonomyGraphReadMode = Literal["disabled", "graphdb_degraded", "graphdb"]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def autonomy_graph_backend_raw() -> str:
    return (os.getenv("AUTONOMY_GRAPH_BACKEND") or "").strip().lower()


def autonomy_graph_reads_explicitly_enabled() -> bool:
    return autonomy_graph_backend_raw() == "graphdb"


def is_quick_autonomy_graph_lane(ctx: Mapping[str, Any] | None) -> bool:
    """Aligned with ``autonomy_subject_fanout_from_runtime_ctx`` bounded quick lane."""
    if not ctx:
        return False
    verb = str(ctx.get("verb") or "").strip().lower()
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    hub_full = bool(opts.get("chat_quick_full_stance"))
    if verb not in FAST_SINGLE_PASS_CHAT_VERBS:
        return False
    if verb == "chat_quick" and hub_full:
        return False
    return True


def resolve_autonomy_graphdb_endpoint_url() -> tuple[str | None, str, str | None, str | None]:
    """Return ``(sparql_endpoint, repo, user, password)``; endpoint ``None`` when unconfigured."""
    endpoint_raw = (
        os.getenv("GRAPHDB_QUERY_ENDPOINT")
        or os.getenv("GRAPHDB_URL")
        or os.getenv("CONCEPT_PROFILE_GRAPHDB_ENDPOINT")
        or os.getenv("CONCEPT_PROFILE_GRAPHDB_URL")
        or ""
    ).strip()
    repo = (os.getenv("GRAPHDB_REPO") or os.getenv("CONCEPT_PROFILE_GRAPHDB_REPO") or "collapse").strip() or "collapse"
    user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
    password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None

    endpoint = endpoint_raw
    if endpoint and endpoint.rstrip("/").endswith("/repositories"):
        endpoint = f"{endpoint.rstrip('/')}/{repo}"
    elif endpoint and "/repositories/" not in endpoint:
        endpoint = f"{endpoint.rstrip('/')}/repositories/{repo}"

    return (endpoint or None, repo, user, password)


def _parse_subjects_csv(raw: str) -> list[str]:
    allowed = frozenset(_AUTONOMY_SUBJECT_ORDER)
    out: list[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower()
        if s in allowed and s not in out:
            out.append(s)
    return out


def _parse_subqueries_csv(raw: str) -> tuple[str, ...]:
    want = frozenset(p.strip().lower() for p in (raw or "").split(",") if p.strip())
    return tuple(sq for sq in _SUBQUERY_ORDER if sq in want)


@dataclass(frozen=True)
class AutonomyGraphReadPlan:
    mode: AutonomyGraphReadMode
    endpoint: str | None
    repo: str
    user: str | None
    password: str | None
    timeout_sec: float
    subjects: tuple[str, ...]
    active_subqueries: tuple[str, ...]
    skipped_reason: str | None
    explicit_backend: bool


def resolve_autonomy_graph_read_plan(ctx: Mapping[str, Any] | None) -> AutonomyGraphReadPlan:
    endpoint, repo, user, password = resolve_autonomy_graphdb_endpoint_url()
    explicit = autonomy_graph_reads_explicitly_enabled()

    if not explicit:
        return AutonomyGraphReadPlan(
            mode="disabled",
            endpoint=None,
            repo=repo,
            user=user,
            password=password,
            timeout_sec=0.0,
            subjects=tuple(),
            active_subqueries=tuple(),
            skipped_reason="v1_cutover_default_backend_not_graphdb",
            explicit_backend=False,
        )

    if not endpoint:
        return AutonomyGraphReadPlan(
            mode="graphdb_degraded",
            endpoint=None,
            repo=repo,
            user=user,
            password=password,
            timeout_sec=0.0,
            subjects=tuple(),
            active_subqueries=tuple(),
            skipped_reason="graphdb_endpoint_unconfigured",
            explicit_backend=True,
        )

    if is_quick_autonomy_graph_lane(ctx):
        timeout = max(0.25, _env_float("AUTONOMY_QUICK_GRAPH_TIMEOUT_SEC", 3.0))
        subs = _parse_subjects_csv(os.getenv("AUTONOMY_QUICK_GRAPH_SUBJECTS") or "orion")
        if not subs:
            subs = ["orion"]
        sq = _parse_subqueries_csv(os.getenv("AUTONOMY_QUICK_GRAPH_SUBQUERIES") or "identity")
        if not sq:
            sq = ("identity",)
        return AutonomyGraphReadPlan(
            mode="graphdb",
            endpoint=endpoint,
            repo=repo,
            user=user,
            password=password,
            timeout_sec=timeout,
            subjects=tuple(subs),
            active_subqueries=sq,
            skipped_reason=None,
            explicit_backend=True,
        )

    timeout = max(0.25, _env_float("AUTONOMY_GRAPH_TIMEOUT_SEC", _env_float("GRAPHDB_TIMEOUT_SEC", 4.5)))
    return AutonomyGraphReadPlan(
        mode="graphdb",
        endpoint=endpoint,
        repo=repo,
        user=user,
        password=password,
        timeout_sec=timeout,
        subjects=_AUTONOMY_SUBJECT_ORDER,
        active_subqueries=_SUBQUERY_ORDER,
        skipped_reason=None,
        explicit_backend=True,
    )


def log_autonomy_graph_backend_decision(
    *,
    plan: AutonomyGraphReadPlan,
    consumer: str,
    verb: str,
    mode: str,
) -> None:
    base: dict[str, Any] = {
        "consumer": consumer,
        "verb": verb,
        "mode": mode,
    }
    if plan.mode == "disabled":
        payload = {**base, "backend": "disabled", "reason": plan.skipped_reason or "v1_cutover_default"}
        logger.info("autonomy_graph_backend_decision %s", json.dumps(payload, sort_keys=True))
        return
    if plan.mode == "graphdb_degraded":
        payload = {
            **base,
            "backend": "graphdb",
            "explicit": True,
            "degraded": True,
            "reason": plan.skipped_reason or "graphdb_endpoint_unconfigured",
        }
        logger.info("autonomy_graph_backend_decision %s", json.dumps(payload, sort_keys=True))
        return

    payload = {
        **base,
        "backend": "graphdb",
        "explicit": True,
        "timeout_sec": plan.timeout_sec,
        "subjects": list(plan.subjects),
        "subqueries": list(plan.active_subqueries),
    }
    logger.info("autonomy_graph_backend_decision %s", json.dumps(payload, sort_keys=True))
