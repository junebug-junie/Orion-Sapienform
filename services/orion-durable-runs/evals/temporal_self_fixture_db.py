"""Load export-shaped Temporal Self rows into real-typed source tables in a throwaway schema.

``fixtures/temporal_self_source_tables.sql`` is every source table's columns and exact types,
read from the live database (constraints dropped). Rows are the shape
``orion/temporal_self/evals/export_fixture_day.sql`` emits (the reducer adapters' input), and are
turned back into table rows here, so ``SourceReader``'s real SQL reads them the way it reads
production. Never point this at a real database: every caller creates a fresh schema.

Reverse mappings that are not one-to-one (stated so a mismatch is never a surprise):

* ``chat_turn``: ``has_prompt`` -> ``prompt`` 'x' or ''; ``unsolicited`` -> ``client_meta``.
* ``reverie_chain`` thoughts, ``oracle_thought_chain`` and ``expectation_verdict`` rows are merged
  by ``thought_id`` into ``substrate_reverie_thought`` (``thought_json.chain_id``).
* ``visual_run``'s joined attempt becomes a ``produced`` attempt carrying ``result_json.chain_id``.
* ``attention_loop_raised.chat_turn`` was an EXISTS over the whole live table. When it is true and
  no exported chat row has that correlation id, a stub chat row dated 2000-01-01 (never inside a
  read window, prompt '') makes the same EXISTS true here.
* ``metacog_observation.trigger_timestamp`` becomes a ``metacog_trigger`` row.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from psycopg.types.json import Jsonb

HERE = Path(__file__).resolve().parent
SOURCE_DDL = HERE / "fixtures" / "temporal_self_source_tables.sql"
REPO = HERE.parents[2]
MIGRATIONS = (
    REPO / "services/orion-sql-db/manual_migration_temporal_self_event_v1.sql",
    REPO / "services/orion-sql-db/manual_migration_temporal_self_v1.sql",
)


def load_jsonl(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r.pop("k")].append(r)
    return rows


def statements(sql: str) -> list[str]:
    """Split a migration the way psql runs it: one statement at a time (so CREATE INDEX
    CONCURRENTLY after COMMIT runs outside a transaction). Our migrations hold no ';' in strings."""
    body = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


async def create_schema(conn: Any) -> None:
    """Source tables, then the chronicle's own migrations (both idempotent). ``conn`` must be in
    autocommit mode, as psql is."""
    for sql in [SOURCE_DDL.read_text(), *(m.read_text() for m in MIGRATIONS)]:
        for stmt in statements(sql):
            await conn.execute(stmt, prepare=False)


def _v(value: Any) -> Any:
    return Jsonb(value) if isinstance(value, (dict, list)) else value


def table_rows(rows: dict[str, list[dict]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    thoughts: dict[str, dict] = {}

    def thought(tid: str, **fields: Any) -> None:
        t = thoughts.setdefault(tid, {"thought_id": tid})
        chain = fields.pop("chain_id", None)
        if chain:
            t["thought_json"] = {"chain_id": chain}
        t.update({k: v for k, v in fields.items() if v is not None})

    for r in rows.get("broadcast", []):
        out["substrate_attention_broadcast_log"].append({
            "log_id": r["log_id"], "generated_at": r["generated_at"], "created_at": r["generated_at"],
            "projection_json": r["projection_json"]})
    chat_corr = set()
    for r in rows.get("chat_turn", []):
        chat_corr.add(r.get("correlation_id"))
        out["chat_history_log"].append({
            "id": r["id"], "correlation_id": r.get("correlation_id"), "session_id": r.get("session_id"),
            "source": r.get("source"), "created_at": r["created_at"], "prompt": "x" if r.get("has_prompt") else "",
            "client_meta": {"unsolicited": bool(r.get("unsolicited"))}})
    for r in rows.get("curiosity_run", []):
        out["curiosity_offer_decisions"].append({k: r.get(k) for k in ("run_id", "decided_at", "turn_started_at", "arm", "offered")})
        if r.get("completed_at"):
            out["curiosity_run_outcomes"].append({k: r.get(k) for k in ("run_id", "completed_at", "turn_ok", "n_tested", "n_moved", "n_formed")})
    for r in rows.get("reverie_chain", []):
        out["substrate_reverie_chain"].append({k: r.get(k) for k in ("chain_id", "created_at", "theme_key", "terminal_reason")})
        for t in r.get("thoughts") or []:
            thought(t["thought_id"], created_at=t.get("created_at"), correlation_id=t.get("correlation_id"), chain_id=r["chain_id"])
    for r in rows.get("oracle_thought_chain", []):
        thought(r["thought_id"], chain_id=r.get("chain_id"))
    for r in rows.get("expectation_verdict", []):
        thought(r["thought_id"], created_at=r.get("created_at"), correlation_id=r.get("correlation_id"), chain_id=r.get("chain_id"),
                expectation_verdict=r.get("expectation_verdict"), expectation_scored_at=r.get("expectation_scored_at"))
    out["substrate_reverie_thought"].extend(thoughts.values())
    for r in rows.get("visual_run", []):
        out["reverie_visual_chain"].append({k: r.get(k) for k in ("chain_id", "created_at", "theme_key", "terminal_reason")})
        if r.get("attempt_started_at"):
            out["reverie_visual_attempt"].append({
                "attempt_id": f"fixture-run-{r['chain_id']}", "started_at": r["attempt_started_at"], "outcome": "produced",
                "result_json": {"chain_id": r["chain_id"], "detail": {"state": r.get("thermal_state")}}})
    for r in rows.get("visual_deferral", []):
        out["reverie_visual_attempt"].append({k: r.get(k) for k in ("attempt_id", "started_at", "outcome", "result_json")})
    for r in rows.get("gpu_wait", []):
        out["gpu_pool_events"].append(dict(r, created_at=r.get("generated_at")))
    out["dream_cycle"].extend(dict(r) for r in rows.get("dream_cycle", []))
    out["dream_hypothesis"].extend(dict(r) for r in rows.get("dream_hypothesis", []))
    out["substrate_action_outcomes"].extend(dict(r) for r in rows.get("action_outcome", []))
    for r in rows.get("metacog_observation", []):
        out["orion_metacog"].append({k: r.get(k) for k in ("id", "correlation_id", "severity", "trigger_kind", "timestamp")})
        if r.get("trigger_timestamp"):
            out["metacog_trigger"].append({"id": f"fixture-trigger-{r['id']}", "correlation_id": r.get("correlation_id"),
                                           "timestamp": r["trigger_timestamp"]})
    out["memory_consolidation_windows"].extend(dict(r) for r in rows.get("consolidation_window_close", []))
    for r in rows.get("attention_row", []):
        out["substrate_attention_schema"].append(dict(r, created_at=r.get("generated_at")))
    for r in rows.get("attention_loop_raised", []):
        out["attention_salience_trace"].append({k: r.get(k) for k in ("trace_id", "loop_id", "scope", "correlation_id", "created_at")})
        if r.get("chat_turn") and r.get("correlation_id") not in chat_corr:
            chat_corr.add(r.get("correlation_id"))
            out["chat_history_log"].append({"id": f"fixture-stub-{r['trace_id']}", "correlation_id": r["correlation_id"],
                                            "created_at": "2000-01-01T00:00:00", "prompt": ""})
    out["attention_loop_outcome"].extend(dict(r) for r in rows.get("attention_loop_verdict", []))
    out["field_dominance_run"].extend(dict(r) for r in rows.get("field_dominance_run", []))
    out["vision_events"].extend(dict(r) for r in rows.get("vision_percept", []))
    for r in rows.get("memory_episode", []):
        out["episode_memory"].append(dict(r, created_at=r.get("occurred_at")))
    out["orion_biometrics_cluster"].extend(dict(r) for r in rows.get("body_cluster", []))
    for r in rows.get("body_cabinet", []):
        out["orion_biometrics_summary"].append({"node": "athena", "timestamp": r["timestamp"],
                                                "measurements": {"cabinet_temp_c": r.get("cabinet_temp_c")}})
    out["cabinet_ambient_spike"].extend(dict(r) for r in rows.get("body_spike", []))
    return out


async def insert(conn: Any, table: str, rows: Iterable[dict]) -> int:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(sorted(r))].append(r)
    n = 0
    async with conn.cursor() as cur:
        for cols, batch in groups.items():
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('%s' for _ in cols)})"
            await cur.executemany(sql, [tuple(_v(r[c]) for c in cols) for r in batch])
            n += len(batch)
    return n


async def load(conn: Any, rows: dict[str, list[dict]]) -> dict[str, int]:
    return {t: await insert(conn, t, rs) for t, rs in table_rows(rows).items() if rs}
