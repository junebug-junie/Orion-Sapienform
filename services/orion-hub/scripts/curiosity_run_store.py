"""Store readers behind `/curiosity/api/runs` and `/curiosity/api/run/{id}`.

This module owns the I/O; `orion/curiosity/run_story.py` owns the join. Each
read is bounded by a window (`days`, default 14, max 90 -- the lifecycle
table's retention) or by one run id, never by a row cap that would truncate
silently. Every store is read best-effort and NAMED in the payload's
`stores` map: Postgres down with the graph up still yields graph-only stories
that say their start clock came from the graph; both down is `available:
false`, never an empty strip that reads as "Orion did nothing for two weeks".

Reads only. Hub never writes to Orion's graph and this module writes to
nothing at all.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from orion.curiosity.atlas import (
    RunNodeRows,
    read_run_ids_since,
    read_run_nodes,
    valid_run_id,
)
from orion.curiosity.run_story import (
    CURIOSITY_WORKFLOWS,
    LINES,
    RunStoryRows,
    build_stories,
    outreach_key,
    reach_out_totals,
    run_to_payload,
    story_to_payload,
    summaries,
)
from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable

logger = logging.getLogger("orion-hub.curiosity_run_store")

WINDOW_DAYS_DEFAULT = 14
WINDOW_DAYS_MAX = 90

_LIFECYCLE_COLS = (
    "run_id, workflow, node, next_node, status, resumed_from_node, "
    "correlation_id, created_at, detail::text AS detail"
)
LIFECYCLE_WINDOW_SQL = (
    f"SELECT {_LIFECYCLE_COLS} FROM substrate_durable_run_state "
    "WHERE workflow = ANY($1::text[]) AND created_at >= $2 ORDER BY created_at ASC"
)
LIFECYCLE_RUN_SQL = (
    f"SELECT {_LIFECYCLE_COLS} FROM substrate_durable_run_state "
    "WHERE run_id = $1 AND workflow = ANY($2::text[]) ORDER BY created_at ASC"
)
JOURNALS_SQL = (
    "SELECT entry_id, source_ref, title, body, created_at FROM journal_entries "
    "WHERE source_ref = ANY($1::text[])"
)
# PR #2290 ("fix(hub): record every curiosity outreach decision; stamp
# Juniper's reply") -- OPEN, NOT MERGED as of this branch. Once it lands,
# every curiosity outreach decision -- including a pre-check block, which
# today only logs a line and writes nothing -- gets a row keyed
# `correlation_id = uuid5(NAMESPACE_URL, f"curiosity_outreach:{run_id}")`
# (`orion.curiosity.run_story.outreach_key`, the same derivation the loop
# itself uses) with `result_json.source = "curiosity_outreach"` and
# `result_json.run_id` for a reverse lookup. This query is written against
# that (still-pending) contract now so the read side needs no change when
# #2290 merges; until then it simply finds no rows for any run, and the
# join degrades to `not_recorded` (`run_story.py`'s own contract, unaffected
# by whether #2290 has landed). The source filter is redundant with the
# correlation_id match in practice (uuid5 over a fixed namespace does not
# collide across sources) but is kept explicit: a decision row this query
# returns must be self-evidently a curiosity decision, not merely one that
# happened to share a key.
OUTREACH_SQL = (
    "SELECT decision_id, decided_at, reason, correlation_id, session_id, "
    "result_json::text AS result_json FROM endogenous_outreach_decisions "
    "WHERE correlation_id = ANY($1::text[]) AND result_json->>'source' = 'curiosity_outreach'"
)
# The sent message carries the outreach key as its own correlation id; a
# reply (PR #2290, not yet merged -- see above) will carry it as
# `client_meta.in_reply_to` on the next chat_history_log row equal to that
# same correlation_id. One query, both; both return no rows until #2290
# lands, same as OUTREACH_SQL above.
CHAT_SQL = (
    "SELECT correlation_id, session_id, prompt, response, "
    "client_meta::text AS client_meta, created_at FROM chat_history_log "
    "WHERE correlation_id = ANY($1::text[]) "
    "OR client_meta->>'in_reply_to' = ANY($1::text[])"
)
READINGS_SQL = (
    "SELECT hop_run_id, hop_n, hop_written_at, about_prior_id, kind, "
    "moved_the_claim, reading_confidence, reasoning FROM curiosity_hop_reading "
    "WHERE hop_run_id = ANY($1::text[])"
)
_SELF_SENSE_COLS = (
    "run_id, created_at, question_key, question, answer_text, answer_source, "
    "correlation_id, self_label_score, grounded_record_score, self_definition_version"
)
SELF_SENSE_WINDOW_SQL = (
    f"SELECT {_SELF_SENSE_COLS} FROM self_sense_eval_log "
    "WHERE created_at >= $1 ORDER BY created_at ASC"
)
SELF_SENSE_RUN_SQL = (
    f"SELECT {_SELF_SENSE_COLS} FROM self_sense_eval_log "
    "WHERE run_id = $1 ORDER BY created_at ASC"
)
# The admission path (since 2026-09-14; PR #2288): the acceptance row is the
# run's true start, `request->>'workflow'` its real workflow, and its events
# the lifecycle. The bridge table above only receives the terminal row.
_ADMISSION_COLS = "run_id, request::text AS request, created_at, control, terminal, updated_at"
ADMISSION_WINDOW_SQL = (
    f"SELECT {_ADMISSION_COLS} FROM durable_admission_runs "
    "WHERE created_at >= $1 OR updated_at >= $1 ORDER BY created_at ASC"
)
ADMISSION_RUN_SQL = f"SELECT {_ADMISSION_COLS} FROM durable_admission_runs WHERE run_id = $1"
# Row-by-row only for events the story renders; the noisy kinds are counted
# instead (3,482 `run.checkpoint_resume_failed` rows in 14 days, live).
RENDERED_EVENTS = (
    "run.accepted", "run.waiting_resource", "run.admitted", "run.lane_assigned", "run.started",
    "run.running", "run.resumed", "run.retrying", "run.completed", "run.failed", "run.cancelled",
    "resource.lease_released", "resource.lease_expired",
)
COUNTED_EVENTS = ("run.checkpoint_resume_failed",)
EVENTS_SQL = (
    "SELECT entry_id, run_id, event, generated_at, payload::text AS payload "
    "FROM durable_resource_events WHERE run_id = ANY($1::text[]) AND event = ANY($2::text[]) "
    "ORDER BY generated_at ASC, entry_id ASC"
)
EVENT_COUNTS_SQL = (
    "SELECT run_id, event, count(*) AS n FROM durable_resource_events "
    "WHERE run_id = ANY($1::text[]) AND event = ANY($2::text[]) GROUP BY run_id, event"
)


def clamp_days(value: Any) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return WINDOW_DAYS_DEFAULT
    return max(1, min(WINDOW_DAYS_MAX, days))


def clamp_line(value: Any) -> str:
    text = str(value or "all").strip()
    return text if text in LINES else "all"


def _dicts(rows: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


class _Stores:
    """What each store said. `None` is "not asked"; `"ok"` is a read that
    returned; anything else is the failure text the page shows."""

    def __init__(self) -> None:
        self.postgres: Optional[str] = None
        self.graph: Optional[str] = None

    def payload(self) -> dict[str, Optional[str]]:
        return {"postgres": self.postgres, "graph": self.graph}

    @property
    def any_ok(self) -> bool:
        return self.postgres == "ok" or self.graph == "ok"


async def _pg_primary(
    pool: Any, stores: _Stores, *, run_id: Optional[str], since: Optional[datetime]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(bridge lifecycle rows, self-sense rows, admission rows): the tables
    that DEFINE which runs exist on the Postgres side."""
    if pool is None:
        stores.postgres = "no_pool"
        return [], [], []
    try:
        async with pool.acquire() as conn:
            if run_id is not None:
                lifecycle = await conn.fetch(LIFECYCLE_RUN_SQL, run_id, list(CURIOSITY_WORKFLOWS))
                sense = await conn.fetch(SELF_SENSE_RUN_SQL, run_id)
                admission = await conn.fetch(ADMISSION_RUN_SQL, run_id)
            else:
                lifecycle = await conn.fetch(LIFECYCLE_WINDOW_SQL, list(CURIOSITY_WORKFLOWS), since)
                sense = await conn.fetch(SELF_SENSE_WINDOW_SQL, since)
                admission = await conn.fetch(ADMISSION_WINDOW_SQL, since)
        stores.postgres = "ok"
        return _dicts(lifecycle), _dicts(sense), _dicts(admission)
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("curiosity_run_store_pg_primary_failed err=%s", exc)
        stores.postgres = f"{type(exc).__name__}: {str(exc)[:160]}"
        return [], [], []


async def _pg_secondary(pool: Any, stores: _Stores, run_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Journal, outreach decision, chat rows and hop readings for known runs.
    A failure here degrades the story (journal missing, decision
    `not_recorded`) and is logged; it does not blank the strip."""
    empty: dict[str, list[dict[str, Any]]] = {"journals": [], "outreach": [], "chat": [], "readings": [],
                                              "events": [], "event_counts": []}
    if pool is None or not run_ids:
        return empty
    keys = [outreach_key(r) for r in run_ids]
    refs = [f"curiosity:{r}" for r in run_ids]
    try:
        async with pool.acquire() as conn:
            journals = await conn.fetch(JOURNALS_SQL, refs)
            outreach = await conn.fetch(OUTREACH_SQL, keys)
            chat = await conn.fetch(CHAT_SQL, keys)
            readings = await conn.fetch(READINGS_SQL, run_ids)
            events = await conn.fetch(EVENTS_SQL, run_ids, list(RENDERED_EVENTS))
            counts = await conn.fetch(EVENT_COUNTS_SQL, run_ids, list(COUNTED_EVENTS))
        return {
            "journals": _dicts(journals), "outreach": _dicts(outreach),
            "chat": _dicts(chat), "readings": _dicts(readings),
            "events": _dicts(events), "event_counts": _dicts(counts),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_run_store_pg_secondary_failed err=%s", exc)
        if stores.postgres == "ok":
            stores.postgres = f"partial: {type(exc).__name__}: {str(exc)[:120]}"
        return empty


async def _graph_ids(reader: Optional[WorldviewReader], stores: _Stores, since_ms: int) -> list[str]:
    if reader is None:
        stores.graph = "graph_not_configured"
        return []
    try:
        ids = await asyncio.to_thread(read_run_ids_since, reader, since_ms)
        stores.graph = "ok"
        return ids
    except WorldviewUnavailable as exc:
        stores.graph = str(exc)[:200]
        return []


async def _graph_nodes(reader: Optional[WorldviewReader], stores: _Stores, run_ids: list[str]) -> RunNodeRows:
    if reader is None:
        stores.graph = "graph_not_configured"
        return RunNodeRows()
    if not run_ids:
        stores.graph = stores.graph or "ok"
        return RunNodeRows()
    try:
        rows = await asyncio.to_thread(read_run_nodes, reader, run_ids)
        stores.graph = "ok"
        return rows
    except WorldviewUnavailable as exc:
        stores.graph = str(exc)[:200]
        return RunNodeRows()


def _rows(lifecycle, sense, admission, nodes: RunNodeRows, secondary) -> RunStoryRows:
    return RunStoryRows(
        lifecycle=lifecycle, roles=nodes.roles, hops=nodes.hops, findings=nodes.findings,
        revisions=nodes.revisions, outcomes=nodes.outcomes, priors=nodes.priors,
        journals=secondary["journals"], outreach=secondary["outreach"],
        chat=secondary["chat"], readings=secondary["readings"], self_sense=sense,
        self_writes=nodes.self_writes, admission=admission,
        resource_events=secondary["events"], event_counts=secondary["event_counts"],
    )


async def read_runs_payload(
    *,
    pool: Any,
    reader: Optional[WorldviewReader],
    days: int = WINDOW_DAYS_DEFAULT,
    line: str = "all",
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """`GET /curiosity/api/runs`: bounded summaries, newest first."""
    days = clamp_days(days)
    line = clamp_line(line)
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)
    stores = _Stores()

    lifecycle, sense, admission = await _pg_primary(pool, stores, run_id=None, since=since)
    graph_ids = await _graph_ids(reader, stores, since_ms)
    run_ids = sorted(
        {str(r["run_id"]) for r in lifecycle if r.get("run_id")}
        | {str(r["run_id"]) for r in admission if r.get("run_id")}
        | set(graph_ids)
    )
    graph_run_ids = [r for r in run_ids if valid_run_id(r)]
    nodes = await _graph_nodes(reader, stores, graph_run_ids)
    sense_ids = sorted({str(r["run_id"]) for r in sense if r.get("run_id")})
    secondary = await _pg_secondary(pool, stores, sorted(set(run_ids) | set(sense_ids)))

    if not stores.any_ok:
        return {
            "available": False,
            "reason": stores.postgres if stores.postgres != "ok" else stores.graph,
            "stores": stores.payload(),
            "window_days": days,
            "line": line,
        }
    stories = build_stories(_rows(lifecycle, sense, admission, nodes, secondary))
    runs = summaries(stories, line=line)
    all_runs = summaries(stories)
    return {
        "available": True,
        "window_days": days,
        "since": since.isoformat(),
        "line": line,
        "stores": stores.payload(),
        "runs": [run_to_payload(r) for r in runs],
        "reach_outs": reach_out_totals(all_runs),
        "totals": {ln: sum(1 for r in all_runs if r.line == ln) for ln in LINES},
    }


async def read_run_payload(
    *, pool: Any, reader: Optional[WorldviewReader], run_id: str
) -> dict[str, Any]:
    """`GET /curiosity/api/run/{run_id}`: one full story."""
    rid = valid_run_id(run_id)
    if rid is None:
        return {"available": True, "found": False, "reason": "bad_run_id"}
    stores = _Stores()
    lifecycle, sense, admission = await _pg_primary(pool, stores, run_id=rid, since=None)
    nodes = await _graph_nodes(reader, stores, [rid])
    secondary = await _pg_secondary(pool, stores, [rid])
    if not stores.any_ok:
        return {
            "available": False,
            "reason": stores.postgres if stores.postgres != "ok" else stores.graph,
            "stores": stores.payload(),
            "run_id": rid,
        }
    stories = build_stories(_rows(lifecycle, sense, admission, nodes, secondary))
    story = stories.get(rid)
    if story is None:
        return {"available": True, "found": False, "run_id": rid, "stores": stores.payload()}
    payload = story_to_payload(story)
    payload.update({"available": True, "found": True, "stores": stores.payload()})
    return payload
