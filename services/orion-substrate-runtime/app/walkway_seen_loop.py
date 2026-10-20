"""Named walkway people get a "seen on <day>" link in the substrate (2026-10-18).

The named-people entity nodes (``entity:vision_individual:*``, written only when
Juniper answers "who is this?", app/ask_answered_listener.py) had 0 edges. This
clocked pass reads the COMPLETE walkway day rollups of the last few days
(``vision_rollup`` grain=day, written by orion-sql-writer) and, for each named
person seen that day whose entity node exists, materializes one day EvidenceNodeV1
plus one ``observed_in`` edge (orion/substrate/adapters/vision_individual.py
``map_walkway_day_seen_to_substrate``) through the same SubstrateGraphMaterializer
path the ask listener uses.

Least invasive seam: no new bus channel or schema. It reads a table the reducer
already writes, with the store and engine this service already has. Idempotent:
day node id from (stream, date), edge identity (source, predicate, target); a day
already linked in this process is skipped, and a restart re-applies the same
record harmlessly. A day is re-linked when its set of linkable people grows (someone
named after the first pass).

Which entity nodes: every individual in the day's ``named`` entry (the rollup lists
the individual ids carrying that name that day), plus individuals Juniper typed the
same name for on the same stream (``label_copied`` false): a copied label has no
entity node of its own, so its sightings link to the typed individual's node.
Entity nodes are never created here.

Trace: ``walkway_seen linked day=... stream=... people=... edges_created=...``.
Fail-open: a missing table, store or DB error is logged; the loop keeps going.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("orion.substrate.runtime.walkway_seen")

DAYS_SQL = (
    "SELECT stream_id, period_start, label, line, rollup->'named' AS named FROM vision_rollup "
    "WHERE grain = 'day' AND complete AND stream_id = ANY(:streams) AND period_start > :since "
    "ORDER BY period_start"
)
TYPED_SQL = (
    "SELECT individual_id FROM vision_individual WHERE stream_id = :s AND lower(label) = lower(:l) "
    "AND NOT label_copied"
)


def _named(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def link_recent_days(*, engine: Any, store: Any, streams: tuple[str, ...], lookback_days: int,
                     done: dict, now: Optional[datetime] = None) -> list[dict[str, Any]]:
    """One pass. Returns one summary per day linked. Blocking (DB + graph)."""
    from sqlalchemy import text

    from orion.core.schemas.cognitive_substrate import EntityNodeV1
    from orion.substrate.adapters.vision_individual import (
        entity_node_id_for_individual,
        map_walkway_day_seen_to_substrate,
    )
    from orion.substrate.materializer import SubstrateGraphMaterializer
    from orion.substrate.reconcile import SubstrateIdentityResolver

    ts = now or datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    with engine.connect() as conn:
        rows = conn.execute(text(DAYS_SQL), {"streams": list(streams),
                                             "since": ts - timedelta(days=lookback_days)}).mappings().all()
        for row in rows:
            day = date.fromisoformat(str(row["label"]))
            key = (row["stream_id"], day)
            seen: list[tuple[str, str, Optional[str]]] = []
            for label, entry in _named(row["named"]).items():
                ids = set((entry or {}).get("individual_ids") or [])
                ids.update(r[0] for r in conn.execute(text(TYPED_SQL), {"s": row["stream_id"], "l": label}).all())
                first = ((entry or {}).get("times") or [None])[0]
                for iid in sorted(ids):
                    nid = entity_node_id_for_individual(iid)
                    if isinstance(store.get_node_by_id(nid), EntityNodeV1):
                        seen.append((nid, label, first))
            # Skip only when this exact set of people was already linked for the day: a person
            # Juniper names later (a new entity node) still gets the day's link.
            linked = frozenset(nid for nid, _, _ in seen)
            if done.get(key) == linked:
                continue
            record = map_walkway_day_seen_to_substrate(
                stream_id=row["stream_id"], day=day, line=str(row["line"] or ""), seen=seen,
                observed_at=row["period_start"] if row["period_start"].tzinfo else
                row["period_start"].replace(tzinfo=timezone.utc))
            if record is None:
                continue
            result = SubstrateGraphMaterializer(
                store=store, identity_resolver=SubstrateIdentityResolver(store=store)).apply_record(record)
            done[key] = linked
            summary = {"day": day.isoformat(), "stream": row["stream_id"],
                       "people": sorted({label for _, label, _ in seen}), "edges": len(record.edges),
                       "edges_created": result.edges_created}
            logger.info("walkway_seen linked day=%s stream=%s people=%s edges=%d edges_created=%d",
                        summary["day"], summary["stream"], ",".join(summary["people"]), summary["edges"],
                        summary["edges_created"])
            out.append(summary)
    return out


async def run_walkway_seen_loop(
    *,
    get_store: Callable[[], Any],
    get_engine: Callable[[], Any],
    settings: Any,
    stop_event: asyncio.Event | None = None,
) -> None:
    if not bool(getattr(settings, "walkway_seen_enabled", False)):
        logger.info("walkway_seen loop disabled by config (WALKWAY_SEEN_ENABLED=false)")
        return
    interval = float(getattr(settings, "walkway_seen_interval_sec", 1800.0) or 1800.0)
    streams = tuple(s.strip() for s in str(getattr(settings, "walkway_seen_streams", "walkway")).split(",")
                    if s.strip())
    lookback = int(getattr(settings, "walkway_seen_lookback_days", 3) or 3)
    done: dict = {}
    logger.info("walkway_seen loop starting interval_sec=%.0f streams=%s lookback_days=%d", interval, streams, lookback)
    while stop_event is None or not stop_event.is_set():
        try:
            # Off the loop: the first get_store() hydrates the whole substrate graph.
            engine, store = await asyncio.to_thread(lambda: (get_engine(), get_store()))
            if engine is None or store is None:
                logger.warning("walkway_seen skipped engine=%s store=%s", engine is not None, store is not None)
            else:
                await asyncio.to_thread(link_recent_days, engine=engine, store=store, streams=streams,
                                        lookback_days=lookback, done=done)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- fail-open; a missing vision_rollup lands here
            logger.warning("walkway_seen pass failed err=%s: %s", type(exc).__name__, exc)
        await asyncio.sleep(interval)
