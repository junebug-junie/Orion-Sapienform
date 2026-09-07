"""Hub Surface: the decision-point dashboard for the attention-schema-surface /
goal-bridge / durable-runs arc (2026-09-04..07).

Two decision points this arc introduced, both otherwise invisible except by
hand SQL against the live containers:

1. **Does the goal producer's pick actually win Orion's attention?** Read from
   `substrate_attention_self_model.self_model_json` (`AttentionSelfModelV1`) --
   `voluntary_override_absent_reason` and `attention_reason`.
2. **Does a long investigation survive to actually finish?** Read from
   `substrate_durable_run_state` (`DurableRunStateV1`).

Everything here is a read. Two numbers describe a one-time historical moment
that cannot be recomputed after the fact -- the pre-deploy control window,
and "3 started, 0 finished" from before durable runs existed -- and are kept
as cited constants (`BRIDGE_BASELINE`, `DURABLE_RUN_BASELINE`) rather than
queried live. Everything else is live, so the page never goes stale.

Design: docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md,
docs/superpowers/specs/2026-09-06-durable-cognition-runs-from-cortex-design.md.
Mockup this was built from: https://claude.ai/code/artifact/8e8e2dbc-1b66-4bbf-ba90-c68475e1b6d7
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text

from app.settings import settings

# /api/hub-surface/* -- the data. `page_router` (below, no prefix) serves the
# page itself, matching how substrate.html / causal_geometry.html are served
# as bare paths in api_routes.py.
router = APIRouter(prefix="/api/hub-surface", tags=["hub-surface"])
page_router = APIRouter()

_engine_instance: Any = None


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
        _engine_instance = create_engine(uri, pool_pre_ping=True)
    return _engine_instance


ALLOWED_WINDOW_MINUTES: frozenset[int] = frozenset({60, 360, 1440})
DEFAULT_WINDOW_MINUTES: int = 1440


def normalize_window_minutes(raw: int) -> int:
    return raw if raw in ALLOWED_WINDOW_MINUTES else DEFAULT_WINDOW_MINUTES


# Frozen historical facts -- the pre-deploy world these two mechanisms
# replaced. Cannot be recomputed: the old code path is gone.
BRIDGE_BASELINE: dict[str, float] = {
    # Control window banked immediately before the bridge deployed 2026-09-06.
    # docs/superpowers/pr-reports/2026-09-06-attention-goal-bridge-pr.md
    "goal_matched_no_loop": 42.1,
    "top_down_override": 4.2,
    "goal_target_already_winning": 20.8,
    "bias_did_not_flip_winner": 0.0,
    "as_of": "2026-09-06",
}
DURABLE_RUN_BASELINE: dict[str, Any] = {
    "started": 3,
    "finished": 0,
    "as_of": "2026-09-06",
    "note": (
        "3 curiosity investigations started, 0 finished -- every crash or "
        "restart killed the run outright, burning the daily cap slot with "
        "nothing to show for it."
    ),
}

# Hand-maintained, not a table: one-line notes worth pinning to the history
# chart when something materially changes the picture. Extend by hand when
# it's genuinely worth remembering -- same spirit as a changelog entry, not
# a general event log.
HUB_SURFACE_MILESTONES: list[dict[str, str]] = [
    {
        "date": "2026-09-07",
        "label": (
            "orion:curiosity:turn:reply:* bus-catalog entry fixed -- durable "
            "runs could finish for the first time"
        ),
    },
]


# --------------------------------------------------------------------------
# Pure helpers (no I/O) -- separated so the parsing/aggregation logic is
# testable against fixed rows, matching this repo's established route-module
# convention (attention_organ_routes.summarize_history).
# --------------------------------------------------------------------------


def _parse_self_model(raw: Any) -> dict[str, Any] | None:
    payload = raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    return payload if isinstance(payload, dict) else None


def summarize_bridge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`rows` are `{generated_at, self_model_json}` dicts, any order.

    Returns the live share of each `voluntary_override_absent_reason` value
    plus the `top_down_override` share of `attention_reason`, as percentages
    of the parseable rows. Malformed rows are counted, not silently dropped,
    so an empty/broken window reads as `sample_count: 0`, never a false 0%.
    """
    absent_counts: dict[str, int] = {}
    override_fired = 0
    parseable = 0
    malformed = 0
    for row in rows:
        payload = _parse_self_model(row.get("self_model_json"))
        if payload is None:
            malformed += 1
            continue
        parseable += 1
        reason = payload.get("voluntary_override_absent_reason")
        if isinstance(reason, str) and reason:
            absent_counts[reason] = absent_counts.get(reason, 0) + 1
        if payload.get("attention_reason") == "top_down_override":
            override_fired += 1

    def pct(n: int) -> float:
        return round((n / parseable) * 100.0, 1) if parseable else 0.0

    branches = {k: pct(v) for k, v in absent_counts.items()}
    branches["top_down_override"] = pct(override_fired)
    return {
        "sample_count": parseable,
        "malformed_row_count": malformed,
        "branches": branches,
    }


def summarize_durable_lifecycle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`rows` are `{run_id, status}` dicts, one per state-transition event in
    the window. Counts DISTINCT run_ids that ever hit each status (a run can
    appear in more than one bucket -- failed then completed), plus the raw
    transition count for `resumed`/`failed` since those matter as event
    counts (how many retries happened), not just how many runs retried."""
    by_status_runs: dict[str, set[str]] = {}
    by_status_events: dict[str, int] = {}
    for row in rows:
        status = row.get("status")
        run_id = row.get("run_id")
        if not isinstance(status, str) or not run_id:
            continue
        by_status_runs.setdefault(status, set()).add(run_id)
        by_status_events[status] = by_status_events.get(status, 0) + 1
    return {
        "completed_runs": len(by_status_runs.get("completed", set())),
        "abandoned_runs": len(by_status_runs.get("abandoned", set())),
        "running_runs": len(by_status_runs.get("running", set())),
        "resumed_events": by_status_events.get("resumed", 0),
        "resumed_runs": len(by_status_runs.get("resumed", set())),
        "failed_events": by_status_events.get("failed", 0),
    }


def pick_example_run(rows: list[dict[str, Any]]) -> str | None:
    """The run with the most `resumed_from_node` transitions among `rows`
    (`{run_id, resumed_from_node}` dicts) -- the most illustrative instance
    of "survived a restart", not just whichever run is newest. `None` when
    no run has ever resumed."""
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("resumed_from_node"):
            run_id = row.get("run_id")
            if run_id:
                counts[run_id] = counts.get(run_id, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.get("/bridge")
def bridge(minutes: int = Query(DEFAULT_WINDOW_MINUTES)) -> dict[str, Any]:
    window = normalize_window_minutes(minutes)
    with _engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT generated_at, self_model_json
                    FROM substrate_attention_self_model
                    WHERE generated_at >= NOW() - (:minutes * INTERVAL '1 minute')
                    ORDER BY generated_at DESC
                    """
                ),
                {"minutes": window},
            )
            .mappings()
            .all()
        )
    return {
        "window_minutes": window,
        "baseline": BRIDGE_BASELINE,
        **summarize_bridge([dict(r) for r in rows]),
    }


@router.get("/bridge/trend")
def bridge_trend() -> dict[str, Any]:
    """Day-bucketed branch share for the last 7 days (bounded by the table's
    own 168h retention anyway, so this never silently truncates)."""
    with _engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT date_trunc('day', generated_at) AS day, self_model_json
                    FROM substrate_attention_self_model
                    WHERE generated_at >= NOW() - INTERVAL '7 days'
                    ORDER BY generated_at ASC
                    """
                )
            )
            .mappings()
            .all()
        )
    by_day: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        day = r["day"].date().isoformat()
        by_day.setdefault(day, []).append(dict(r))
    days = [
        {"day": day, **summarize_bridge(day_rows)}
        for day, day_rows in sorted(by_day.items())
    ]
    return {"days": days, "baseline": BRIDGE_BASELINE}


@router.get("/durable-runs")
def durable_runs(minutes: int = Query(DEFAULT_WINDOW_MINUTES)) -> dict[str, Any]:
    window = normalize_window_minutes(minutes)
    with _engine().connect() as conn:
        lifecycle_rows = (
            conn.execute(
                text(
                    """
                    SELECT run_id, status
                    FROM substrate_durable_run_state
                    WHERE generated_at >= NOW() - (:minutes * INTERVAL '1 minute')
                    """
                ),
                {"minutes": window},
            )
            .mappings()
            .all()
        )
        # Deliberately ALL-TIME, not scoped to `window` -- the example exists
        # to show the single most illustrative resume ever recorded, not
        # "whichever run happens to have a transition in the last hour" (a
        # short window would almost never show one at all). Kept honest in
        # the response as `example_scope: "all_time"` so a short-window
        # caller isn't left thinking the two numbers describe the same span.
        resume_rows = (
            conn.execute(
                text(
                    """
                    SELECT run_id, resumed_from_node
                    FROM substrate_durable_run_state
                    WHERE resumed_from_node IS NOT NULL
                    """
                )
            )
            .mappings()
            .all()
        )

    example_run_id = pick_example_run([dict(r) for r in resume_rows])
    example: dict[str, Any] | None = None
    if example_run_id:
        with _engine().connect() as conn:
            steps = (
                conn.execute(
                    text(
                        """
                        SELECT node, status, next_node, resumed_from_node, generated_at, detail
                        FROM substrate_durable_run_state
                        WHERE run_id = :run_id
                        ORDER BY generated_at ASC
                        """
                    ),
                    {"run_id": example_run_id},
                )
                .mappings()
                .all()
        )
        example = {
            "run_id": example_run_id,
            "steps": [
                {
                    "node": s["node"],
                    "status": s["status"],
                    "next_node": s["next_node"],
                    "resumed_from_node": s["resumed_from_node"],
                    "generated_at": _iso(s["generated_at"]),
                    "detail": s["detail"],
                }
                for s in steps
            ],
        }

    return {
        "window_minutes": window,
        "baseline": DURABLE_RUN_BASELINE,
        # Live, unlike the goal-bridge's kill switch (services/orion-attention-
        # runtime/.env) which Hub has no way to read without new RPC plumbing.
        "kickoff_via_cortex": settings.HUB_CURIOSITY_KICKOFF_VIA_CORTEX,
        **summarize_durable_lifecycle([dict(r) for r in lifecycle_rows]),
        "example": example,
        "example_scope": "all_time",
    }


@router.get("/durable-runs/trend")
def durable_runs_trend() -> dict[str, Any]:
    """Cumulative completed-run count per day since inception -- no fixed
    epoch to bound this at, the table itself is the record -- plus the
    hand-maintained milestone notes plotted as reference markers."""
    with _engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT date_trunc('day', generated_at) AS day, run_id
                    FROM substrate_durable_run_state
                    WHERE status = 'completed'
                    ORDER BY generated_at ASC
                    """
                )
            )
            .mappings()
            .all()
        )
    seen: set[str] = set()
    by_day: dict[str, int] = {}
    for r in rows:
        run_id = r["run_id"]
        if run_id in seen:
            continue
        seen.add(run_id)
        day = r["day"].date().isoformat()
        by_day[day] = by_day.get(day, 0) + 1
    cumulative = 0
    series = []
    for day in sorted(by_day.keys()):
        cumulative += by_day[day]
        series.append({"day": day, "cumulative_completed": cumulative})
    return {"series": series, "milestones": HUB_SURFACE_MILESTONES}


@router.get("/activity")
def activity(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    """Last `limit` rows across all attention-schema lanes, newest first --
    `reason_narrative` is already written to be a plain sentence, rendered
    directly with no per-lane translation logic."""
    with _engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT generated_at, process, reason_narrative, attention_reason
                    FROM substrate_attention_schema
                    ORDER BY generated_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
    return {
        "rows": [
            {
                "generated_at": _iso(r["generated_at"]),
                "process": r["process"],
                "reason_narrative": r["reason_narrative"],
                "attention_reason": r["attention_reason"],
            }
            for r in rows
        ]
    }


@page_router.get("/hub-surface")
def hub_surface_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "hub_surface.html").read_text(encoding="utf-8")
    rendered = template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version())
    return HTMLResponse(
        content=rendered,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
