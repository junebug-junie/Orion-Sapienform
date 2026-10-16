"""Persistence for the personality graph (``manual_migration_personality_graph_v1.sql``).

Single writer: ``app.personality_driver``. One transaction per tick (``commit_tick``) writes, in
the migration's order: new evidence rows (``ON CONFLICT DO NOTHING``), changed dispositions,
weight-history rows (``ON CONFLICT DO NOTHING``), and the per-source cursors. A restart resumes from
the cursors: a tick that failed before commit wrote nothing, and is re-read in full next tick.

Reads for the routes: the frame (``project_frame`` over the stored nodes), dispositions with their
recent weight history, and evidence rows, newest first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.personality.graph import CHOICE_LANE, EvidenceRow, HistoryRow, Node, choice_style_from_rows, project_frame
from orion.schemas.personality import FRAME_MAX_DISPOSITIONS, PersonalityFrameV1

from app.personality_sources import Cursor

logger = logging.getLogger("orion-durable-runs.personality_store")

STATEMENT_TIMEOUT = "30s"
CHOICE_STYLE_WINDOW_DAYS = 14
HISTORY_POINTS = 60

FRAME_NODE_COLUMNS = ("disposition_id, origin, status, label, label_similarity, centroid_n, weight, evidence_count, "
                      "lane_counts, chosen_count, first_seen_at, last_reinforced_at, retired_at, updated_at, lane_recent")
NODE_COLUMNS = ("disposition_id, origin, status, label, label_similarity, centroid, centroid_n, weight, "
                "evidence_count, lane_counts, chosen_count, first_seen_at, last_reinforced_at, retired_at, updated_at, "
                "lane_recent")

INSERT_EVIDENCE_SQL = """
INSERT INTO personality_evidence (evidence_id, lane, source_ref, occurred_at, excerpt, disposition_id,
  similarity, weight_delta, meta)
VALUES (%(evidence_id)s, %(lane)s, %(source_ref)s, %(occurred_at)s, %(excerpt)s, %(disposition_id)s,
  %(similarity)s, %(weight_delta)s, %(meta)s::jsonb)
ON CONFLICT (evidence_id) DO NOTHING
"""
UPSERT_NODE_SQL = """
INSERT INTO personality_disposition (disposition_id, origin, status, label, label_similarity, centroid,
  centroid_n, weight, evidence_count, lane_counts, chosen_count, first_seen_at, last_reinforced_at,
  retired_at, updated_at, lane_recent)
VALUES (%(disposition_id)s, %(origin)s, %(status)s, %(label)s, %(label_similarity)s, %(centroid)s::real[],
  %(centroid_n)s, %(weight)s, %(evidence_count)s, %(lane_counts)s::jsonb, %(chosen_count)s, %(first_seen_at)s,
  %(last_reinforced_at)s, %(retired_at)s, %(updated_at)s, %(lane_recent)s::jsonb)
ON CONFLICT (disposition_id) DO UPDATE SET origin = EXCLUDED.origin, status = EXCLUDED.status,
  label = EXCLUDED.label, label_similarity = EXCLUDED.label_similarity, centroid = EXCLUDED.centroid,
  centroid_n = EXCLUDED.centroid_n, weight = EXCLUDED.weight, evidence_count = EXCLUDED.evidence_count,
  lane_counts = EXCLUDED.lane_counts, chosen_count = EXCLUDED.chosen_count,
  first_seen_at = EXCLUDED.first_seen_at, last_reinforced_at = EXCLUDED.last_reinforced_at,
  retired_at = EXCLUDED.retired_at, updated_at = EXCLUDED.updated_at, lane_recent = EXCLUDED.lane_recent
"""
# Fade-only change (no evidence folded into the node this tick): skip rewriting the 1024-float
# centroid. Every non-retired node fades every tick, so this is the common case. lane_recent fades
# with the weight, so it is written here too (a few keys of jsonb).
UPDATE_NODE_WEIGHT_SQL = """
UPDATE personality_disposition SET status = %(status)s, weight = %(weight)s, retired_at = %(retired_at)s,
  updated_at = %(updated_at)s, lane_recent = %(lane_recent)s::jsonb
WHERE disposition_id = %(disposition_id)s
"""
INSERT_HISTORY_SQL = """
INSERT INTO personality_weight_history (disposition_id, at, weight, status, evidence_count)
VALUES (%(disposition_id)s, %(at)s, %(weight)s, %(status)s, %(evidence_count)s)
ON CONFLICT (disposition_id, at) DO NOTHING
"""
UPSERT_CURSOR_SQL = """
INSERT INTO personality_cursor (source, cursor_at, cursor_ref, updated_at)
VALUES (%(source)s, %(at)s, %(ref)s, %(now)s)
ON CONFLICT (source) DO UPDATE SET cursor_at = EXCLUDED.cursor_at, cursor_ref = EXCLUDED.cursor_ref,
  updated_at = EXCLUDED.updated_at
"""


@dataclass
class TickWrite:
    now: datetime
    evidence: list[EvidenceRow] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    # disposition_id -> node_key() as loaded; a node whose key did not move only faded or changed
    # status, so only its weight/status columns are written.
    loaded: dict[str, tuple] = field(default_factory=dict)
    history: list[HistoryRow] = field(default_factory=list)
    cursors: dict[str, Cursor] = field(default_factory=dict)


def _lane_recent(raw: Any) -> dict[str, float]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    return {str(k): float(v) for k, v in (raw or {}).items()}


def _json_lane_recent(n: Node) -> str:
    return json.dumps({k: float(v) for k, v in (n.lane_recent or {}).items()}, sort_keys=True)


def node_from_row(r: Mapping[str, Any]) -> Node:
    lane_counts = r.get("lane_counts") or {}
    if isinstance(lane_counts, str):
        lane_counts = json.loads(lane_counts)
    return Node(
        disposition_id=r["disposition_id"], origin=r["origin"], status=r["status"], label=r["label"],
        label_similarity=float(r["label_similarity"] or 0.0), centroid=[float(x) for x in (r["centroid"] or [])],
        centroid_n=int(r["centroid_n"] or 0), weight=float(r["weight"] or 0.0),
        evidence_count=int(r["evidence_count"] or 0), lane_counts={str(k): int(v) for k, v in lane_counts.items()},
        chosen_count=int(r["chosen_count"] or 0), first_seen_at=r["first_seen_at"],
        last_reinforced_at=r["last_reinforced_at"], retired_at=r["retired_at"], updated_at=r["updated_at"],
        lane_recent=_lane_recent(r.get("lane_recent")),
    )


def node_row(n: Node) -> dict:
    return {
        "disposition_id": n.disposition_id, "origin": n.origin, "status": n.status, "label": n.label,
        "label_similarity": float(n.label_similarity), "centroid": [float(x) for x in n.centroid],
        "centroid_n": int(n.centroid_n), "weight": float(n.weight), "evidence_count": int(n.evidence_count),
        "lane_counts": json.dumps(n.lane_counts, sort_keys=True), "chosen_count": int(n.chosen_count),
        "first_seen_at": n.first_seen_at, "last_reinforced_at": n.last_reinforced_at, "retired_at": n.retired_at,
        "updated_at": n.updated_at, "lane_recent": _json_lane_recent(n),
    }


def evidence_row(e: EvidenceRow) -> dict:
    return {
        "evidence_id": e.evidence_id, "lane": e.lane, "source_ref": e.source_ref, "occurred_at": e.occurred_at,
        "excerpt": e.excerpt, "disposition_id": e.disposition_id, "similarity": e.similarity,
        "weight_delta": float(e.weight_delta), "meta": json.dumps(e.meta or {}, sort_keys=True, default=str),
    }


def history_row(h: HistoryRow) -> dict:
    return {"disposition_id": h.disposition_id, "at": h.at, "weight": float(h.weight), "status": h.status,
            "evidence_count": int(h.evidence_count)}


def cursor_rows(cursors: Mapping[str, Cursor], now: datetime) -> list[dict]:
    return [{"source": k, "at": c.at, "ref": c.ref or None, "now": now} for k, c in sorted(cursors.items())]


def node_key(n: Node) -> tuple:
    """Every column the weight/status-only UPDATE does NOT write (it writes status, weight,
    retired_at, updated_at, lane_recent). If any of these moved, the node gets a full upsert."""
    return (n.origin, n.label, n.label_similarity, n.centroid_n, n.evidence_count, n.chosen_count,
            json.dumps(n.lane_counts, sort_keys=True), n.first_seen_at, n.last_reinforced_at)


def split_nodes(nodes: Sequence[Node], loaded: Mapping[str, tuple]) -> tuple[list[dict], list[dict]]:
    """(full upserts, weight/status-only updates)."""
    full, light = [], []
    for n in nodes:
        prev = loaded.get(n.disposition_id)
        if prev is not None and prev == node_key(n):
            light.append({"disposition_id": n.disposition_id, "status": n.status, "weight": float(n.weight),
                          "retired_at": n.retired_at, "updated_at": n.updated_at,
                          "lane_recent": _json_lane_recent(n)})
        else:
            full.append(node_row(n))
    return full, light


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


class PersonalityStore:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def load_nodes(self) -> list[Node]:
        """Every non-retired node (retired is terminal and never matched again)."""
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                f"SELECT {NODE_COLUMNS} FROM personality_disposition WHERE status <> 'retired'")).fetchall()
        return [node_from_row(r) for r in rows]

    async def load_cursors(self) -> dict[str, Cursor]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute("SELECT source, cursor_at, cursor_ref FROM personality_cursor")).fetchall()
        return {r["source"]: Cursor(r["cursor_at"], r["cursor_ref"] or "") for r in rows}

    async def existing_evidence_ids(self, ids: Iterable[str]) -> set[str]:
        want = sorted(set(ids))
        if not want:
            return set()
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT evidence_id FROM personality_evidence WHERE evidence_id = ANY(%(ids)s)", {"ids": want})).fetchall()
        return {r["evidence_id"] for r in rows}

    async def commit_tick(self, w: TickWrite) -> None:
        full, light = split_nodes(w.nodes, w.loaded)
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                    if w.evidence:
                        await cur.executemany(INSERT_EVIDENCE_SQL, [evidence_row(e) for e in w.evidence])
                    if full:
                        await cur.executemany(UPSERT_NODE_SQL, full)
                    if light:
                        await cur.executemany(UPDATE_NODE_WEIGHT_SQL, light)
                    if w.history:
                        await cur.executemany(INSERT_HISTORY_SQL, [history_row(h) for h in w.history])
                    if w.cursors:
                        await cur.executemany(UPSERT_CURSOR_SQL, cursor_rows(w.cursors, w.now))

    # --- reads for the routes ------------------------------------------------------------------

    async def status_counts(self) -> dict[str, int]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT status, count(*) AS n FROM personality_disposition GROUP BY status")).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    async def frame(self, now: datetime, warnings: Sequence[str] = ()) -> PersonalityFrameV1:
        """One read-only snapshot (a tick committing mid-request cannot split nodes from counts).
        Only the strongest active nodes are loaded, without centroids; counts come from SQL."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                active = await (await conn.execute(
                    f"SELECT {FRAME_NODE_COLUMNS} FROM personality_disposition WHERE status = 'active' "
                    "ORDER BY weight DESC, disposition_id LIMIT %(n)s", {"n": FRAME_MAX_DISPOSITIONS})).fetchall()
                counts = {r["status"]: int(r["n"]) for r in await (await conn.execute(
                    "SELECT status, count(*) AS n FROM personality_disposition GROUP BY status")).fetchall()}
                choice_rows = await (await conn.execute(
                    "SELECT lane, occurred_at, meta FROM personality_evidence WHERE lane = %(lane)s "
                    "AND occurred_at > %(since)s",
                    {"lane": CHOICE_LANE, "since": now - timedelta(days=CHOICE_STYLE_WINDOW_DAYS)})).fetchall()
                n_ev = (await (await conn.execute("SELECT count(*) AS n FROM personality_evidence")).fetchone())["n"]
        style = choice_style_from_rows(choice_rows, now=now, window_days=CHOICE_STYLE_WINDOW_DAYS)
        nodes = [node_from_row({**r, "centroid": []}) for r in active]
        frame = project_frame(nodes, now=now, choice_style=style, evidence_rows=int(n_ev), warnings=warnings)
        return frame.model_copy(update={"active_count": counts.get("active", 0),
                                        "candidate_count": counts.get("candidate", 0),
                                        "retired_count": counts.get("retired", 0)})

    async def dispositions(self, status: str = "active", limit: int = 50) -> list[dict]:
        sql = ("SELECT disposition_id, origin, status, label, label_similarity, centroid_n, weight, evidence_count, "
               "lane_counts, lane_recent, chosen_count, first_seen_at, last_reinforced_at, retired_at, updated_at "
               "FROM personality_disposition")
        params: dict = {"limit": limit}
        if status != "all":
            sql += " WHERE status = %(status)s"
            params["status"] = status
        sql += " ORDER BY weight DESC, disposition_id LIMIT %(limit)s"
        async with self._pool.connection() as conn:
            rows = [dict(r) for r in await (await conn.execute(sql, params)).fetchall()]
            ids = [r["disposition_id"] for r in rows]
            hist = []
            if ids:
                hist = await (await conn.execute(
                    "SELECT disposition_id, at, weight, status, evidence_count FROM ("
                    " SELECT h.*, row_number() OVER (PARTITION BY disposition_id ORDER BY at DESC) AS rn"
                    " FROM personality_weight_history h WHERE disposition_id = ANY(%(ids)s)) x"
                    " WHERE rn <= %(n)s ORDER BY disposition_id, at", {"ids": ids, "n": HISTORY_POINTS})).fetchall()
        by_id: dict[str, list[dict]] = {}
        for h in hist:
            by_id.setdefault(h["disposition_id"], []).append(
                {"at": _iso(h["at"]), "weight": h["weight"], "status": h["status"], "evidence_count": h["evidence_count"]})
        return [{**{k: _iso(v) for k, v in r.items()}, "history": by_id.get(r["disposition_id"], [])} for r in rows]

    async def evidence(self, disposition_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        sql = ("SELECT evidence_id, lane, source_ref, occurred_at, excerpt, disposition_id, similarity, weight_delta, "
               "meta, created_at FROM personality_evidence")
        params: dict = {"limit": limit}
        if disposition_id:
            sql += " WHERE disposition_id = %(d)s"
            params["d"] = disposition_id
        sql += " ORDER BY occurred_at DESC, evidence_id LIMIT %(limit)s"
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return [{k: _iso(v) for k, v in dict(r).items()} for r in rows]
