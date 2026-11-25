"""Read-only feed for the Inner Sky page (static/inner-sky.html), the Hub's Cognitive EKG.

The page draws rows, not dials. Each light on it is one stored row:

* ``GET /api/inner-sky/events`` -- Orion's experiences (``temporal_self_event``, the experience contract every
  organ emits through) and the bindings between them (``experience_use``: one organ cited another organ's
  experience). Labels are ``orion.temporal_self.now.now_label``, the same code-written label every organ's
  now read shows, so the sky says what Orion is shown, in the same words.
* ``GET /api/inner-sky/state`` -- the services as the bus synaptic graph sees them (``orion_bus_synapse`` in
  FalkorDB: cumulative publish counts and live cause-and-effect links) plus the open attention arc and day
  phase from durable-runs' ``/temporal-self/frame``. The page turns count deltas between polls into real
  message rates.

Never reads through ``GET /temporal-self/now``: that call stores an ``experience_shown`` row, and a debug page
must not register as Orion seeing something. Pure reads only; every part that fails reports its own error
and the rest still answers.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

from orion.graph.falkor_client import RedisGraphQueryClient
from orion.integration.organs import load_organs
from orion.temporal_self.now import now_label

from app.settings import settings
from scripts.self_brain_routes import _engine as _pg_engine

logger = logging.getLogger("orion-hub.inner-sky")

router = APIRouter(prefix="/api/inner-sky", tags=["inner-sky"])

WINDOW_HOURS = 12
EVENT_PAGE = 3000
CITE_PAGE = 500
LINK_WINDOW_SEC = 24 * 3600
FRAME_TIMEOUT_SEC = 3.0
# Cursors re-read this far back. ingested_at/cited_at are transaction-start times, so a transaction that
# began earlier can commit after a later one was already polled; and a full page can end inside a group of
# equal timestamps. The page dedupes by id, so the overlap costs only a few repeated rows.
CURSOR_OVERLAP_SEC = 60
UNREGISTERED = "unregistered"


def _epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return float(value)


def _payload(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def shape_event(row: Mapping[str, Any], organ_of: Mapping[str, str]) -> dict[str, Any]:
    """One ``temporal_self_event`` row as the page draws it: when, whose, and the words Orion is shown."""
    kind = str(row["source_kind"])
    return {
        "id": str(row["event_id"]),
        "t": _epoch(row["occurred_at"]),
        "organ": organ_of.get(kind, UNREGISTERED),
        "kind": kind,
        "label": now_label(kind, row.get("label"), row.get("verdict"), _payload(row.get("payload_json"))),
        "salience": row.get("salience"),
    }


def shape_cite(row: Mapping[str, Any], organ_of: Mapping[str, str]) -> dict[str, Any]:
    """One ``experience_use`` row: organ ``by`` used an experience that belongs to organ ``from``."""
    event_id = str(row["event_id"])
    kind = str(row.get("source_kind") or event_id.split(":", 1)[0])
    label = (
        now_label(kind, row.get("label"), row.get("verdict"), _payload(row.get("payload_json")))
        if row.get("source_kind")
        else ""
    )
    return {
        "id": str(row.get("use_id") or f"{event_id}|{row['citing_organ']}|{_epoch(row['cited_at'])}"),
        "t": _epoch(row["cited_at"]),
        "event_id": event_id,
        "by": str(row["citing_organ"]),
        "from": organ_of.get(kind, UNREGISTERED),
        "kind": kind,
        "label": label,
    }


def _fetch_event_rows(since: float, limit: int) -> list[dict[str, Any]]:
    from sqlalchemy import text

    engine = _pg_engine()
    if engine is None:
        raise HTTPException(503, "postgres_uri_not_configured")
    sql = text(
        """
        SELECT event_id, occurred_at, source_kind, label, verdict, payload_json, salience, ingested_at
        FROM temporal_self_event
        WHERE ingested_at > to_timestamp(:since) - make_interval(secs => :overlap)
          AND occurred_at > now() - make_interval(hours => :hours)
        ORDER BY ingested_at, event_id
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql, {"since": since, "overlap": CURSOR_OVERLAP_SEC if since > 0 else 0, "hours": WINDOW_HOURS, "limit": limit})]


def _fetch_cite_rows(since: float, limit: int) -> list[dict[str, Any]]:
    from sqlalchemy import text

    engine = _pg_engine()
    if engine is None:
        raise HTTPException(503, "postgres_uri_not_configured")
    sql = text(
        """
        SELECT u.use_id, u.event_id, u.citing_organ, u.cited_at, e.source_kind, e.label, e.verdict, e.payload_json
        FROM experience_use u
        LEFT JOIN temporal_self_event e ON e.event_id = u.event_id
        WHERE u.cited_at > to_timestamp(:since) - make_interval(secs => :overlap)
          AND u.cited_at > now() - make_interval(hours => :hours)
        ORDER BY u.cited_at, u.use_id
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql, {"since": since, "overlap": CURSOR_OVERLAP_SEC if since > 0 else 0, "hours": WINDOW_HOURS, "limit": limit})]


@router.get("/events")
def events(
    since: float = Query(default=0.0, ge=0.0, description="ingested_at cursor (epoch s) from the last answer"),
    cite_since: float = Query(default=0.0, ge=0.0, description="cited_at cursor (epoch s) from the last answer"),
) -> dict[str, Any]:
    """Experiences stored after ``since`` and bindings after ``cite_since``, inside the last 12 hours.

    Cursors move on store time, not occurrence time, and every read re-covers the last CURSOR_OVERLAP_SEC,
    so a row whose transaction committed late is still delivered; rows may repeat, and callers dedupe by
    ``id``. ``more`` says a page was full: ask again at once with the returned cursors."""
    organ_of = load_organs().organ_of
    rows = _fetch_event_rows(since, EVENT_PAGE)
    cites = _fetch_cite_rows(cite_since, CITE_PAGE)
    return {
        "window_hours": WINDOW_HOURS,
        "now": time.time(),
        "organs": list(load_organs().organs),
        "events": [shape_event(r, organ_of) for r in rows],
        "cites": [shape_cite(r, organ_of) for r in cites],
        "cursor": max(since, _epoch(rows[-1]["ingested_at"])) if rows else since,
        "cite_cursor": max(cite_since, _epoch(cites[-1]["cited_at"])) if cites else cite_since,
        "more": len(rows) >= EVENT_PAGE or len(cites) >= CITE_PAGE,
    }


def _graph_client() -> RedisGraphQueryClient:
    uri = (settings.FALKORDB_URI or "").strip()
    if not uri:
        raise RuntimeError("falkordb_uri_not_configured")
    return RedisGraphQueryClient(uri=uri, graph_name=getattr(settings, "FALKORDB_BUS_GRAPH", "") or "orion_bus_synapse")


def read_body(client: Any, now: float) -> dict[str, Any]:
    """Every service with its lifetime publish count and last-seen time, and every cause-and-effect link seen
    in the last day with its lifetime count. Counts are cumulative: rates are the page's job (deltas)."""
    services = client.graph_query(
        """
        MATCH (o:Organ)
        OPTIONAL MATCH (o)-[r:PUBLISHES]->(:Channel)
        RETURN o.organ_id AS id, sum(r.count) AS n, max(r.last_seen_epoch) AS last
        """
    )
    links = client.graph_query(
        """
        MATCH (a:Organ)-[r:CAUSALLY_FOLLOWED_BY]->(b:Organ)
        WHERE r.last_seen_epoch > $cut
        RETURN a.organ_id AS a, b.organ_id AS b, r.count AS n, r.last_seen_epoch AS last
        """,
        {"cut": now - LINK_WINDOW_SEC},
    )
    return {
        "services": [
            {"id": str(r["id"]), "n": int(r["n"] or 0), "last": float(r["last"]) if r.get("last") is not None else None}
            for r in services
            if r.get("id")
        ],
        "links": [
            {"a": str(r["a"]), "b": str(r["b"]), "n": int(r["n"] or 0), "last": float(r["last"] or 0)}
            for r in links
            if r.get("a") and r.get("b") and r["a"] != r["b"]
        ],
    }


def focus_from_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    """The open attention arc and the day phase, verbatim from the day frame. No open attention arc reads as
    none: another kind's arc (curiosity, interoception) is never shown as what Orion attends to."""
    arc = (frame.get("active_by_kind") or {}).get("attention") or {}
    return {
        "attention": arc.get("subject_label") or arc.get("subject_ref") or None,
        "attention_since": arc.get("began_at"),
        "day_phase": frame.get("day_phase"),
        "as_of": frame.get("as_of"),
    }


@router.get("/state")
def state() -> dict[str, Any]:
    """The service body and the current focus. A part that cannot be read is null with its error beside it."""
    now = time.time()
    out: dict[str, Any] = {"now": now, "body": None, "body_error": None, "focus": None, "focus_error": None}
    try:
        out["body"] = read_body(_graph_client(), now)
    except Exception as exc:  # noqa: BLE001 - the page shows the error instead of a blank shell
        logger.warning("inner-sky body read failed: %s", exc)
        out["body_error"] = str(exc)[:200]
    base = (getattr(settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", "") or "").rstrip("/")
    if not base:
        out["focus_error"] = "durable_runs_url_not_configured"
        return out
    try:
        resp = httpx.get(f"{base}/temporal-self/frame", timeout=FRAME_TIMEOUT_SEC)
        resp.raise_for_status()
        out["focus"] = focus_from_frame(resp.json())
    except Exception as exc:  # noqa: BLE001
        logger.warning("inner-sky focus read failed: %s", exc)
        out["focus_error"] = str(exc)[:200]
    return out
