"""Latest-value readers for metacog's evidence-cue "recent trend signals" block.

Tier 1 patch, per `docs/superpowers/specs/2026-07-28-metacog-turn-scoped-trend-
reducer-design.md`. That doc's own "Missing questions" §3 names the cheapest
real consumer for "how does trend data reach metacog": a new, additive
evidence-cue block in `MetacogContextService`
(`services/orion-cortex-exec/app/executor.py`), reading two already-computed,
already-persisted signal sources' *latest* value -- not a new reducer. This
module does not compute a window/trend itself -- both sources below are
windowed/reduced upstream by their own producers (`orion-substrate-runtime`
and `orion-biometrics`/`orion-field-digester`/`orion-sql-writer`
respectively). Building a new reducer, poll loop, or persistence layer here
would violate that doc's explicit scope boundary. Juniper's direct
authorization to implement this exact Tier-1 slice ("do it tier 1") and this
patch's own scope are recorded in the doc's "2026-07-29/30 update" section,
added alongside this patch.

Two Tier-1 signal sources, both independently verified real and non-degenerate
against live Postgres data on 2026-07-28/29/30 (see the design doc's "2026-07-28
update" section for the arbitration-layer measurement discipline this reuses,
the "2026-07-29/30 update" section for this patch's own numbers, and the PR
description for this patch):

1. `substrate_field_state`'s `prediction_error` node vectors for
   `node:substrate.execution` and `node:substrate.biometrics` -- real-time
   "surprise" scores, 0-1 range, ~2s write cadence. `prediction_error` was
   removed from every domain's decay set 2026-07-26
   (`services/orion-field-digester/app/digestion/decay.py::NODE_DECAY_CHANNELS`)
   so it is an undecayed raw snapshot for every domain, not just
   `node:substrate.bus_synaptic` -- the same `STALENESS_HORIZON_SEC` guard
   `orion.substrate.bus_synaptic_surprise.latest_bus_synaptic_prediction_error`
   already applies is reused here rather than re-derived.

2. `orion_biometrics_induction`'s per-channel windowed stats
   (`metrics.<channel>.{level, trend, spike_rate, volatility}`), one row per
   node per ~30-45s tick -- already-computed windowed statistics (the writer's
   own `window_sec` column), not something this module computes fresh.

Confirmed live 2026-07-30: a single `substrate_field_state` row's `field_json`
carries every live domain's node_vectors as of that tick (one row, not one row
per domain) -- so both prediction_error values are read in a single
round trip, not two.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine

from orion.substrate.bus_synaptic_surprise import STALENESS_HORIZON_SEC

# Fixed per this patch's Tier-1 scope (design doc's own two named candidates) --
# not operator-configurable, unlike the biometrics_induction node list below,
# because these two node ids are load-bearing to which signal this cue claims
# to be (a different node id would silently change what "recent_trend_signals"
# means without changing its name).
PREDICTION_ERROR_NODE_IDS: tuple[str, ...] = (
    "node:substrate.execution",
    "node:substrate.biometrics",
)


def latest_node_prediction_errors(
    engine: Engine, node_ids: Sequence[str] = PREDICTION_ERROR_NODE_IDS
) -> dict[str, float | None]:
    """Latest real `prediction_error` value per requested `substrate_field_state`
    node id, read off the single most recent row.

    Same undecayed-raw-snapshot staleness contract as
    `latest_bus_synaptic_prediction_error`: a row older than
    `STALENESS_HORIZON_SEC` returns `None` for every requested node (all-or-
    nothing on row age, not per-node -- there is only one row to be stale),
    so "no real signal available" stays distinguishable from "genuinely calm".

    Returns a dict with every requested node id as a key; values are `None`
    when the node is absent from that row, the row itself is missing, too
    stale, or unparseable.
    """
    result: dict[str, float | None] = {node_id: None for node_id in node_ids}
    if not node_ids:
        return result

    select_parts: list[str] = []
    params: dict[str, Any] = {}
    for i, node_id in enumerate(node_ids):
        key = f"node_{i}"
        select_parts.append(
            f"field_json -> 'node_vectors' -> :{key} ->> 'prediction_error' AS {key}"
        )
        params[key] = node_id

    query = text(
        f"""
        SELECT generated_at, {', '.join(select_parts)}
        FROM substrate_field_state
        ORDER BY generated_at DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, params).mappings().first()
    if not row:
        return result

    generated_at = row.get("generated_at")
    if generated_at is not None:
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - generated_at).total_seconds()
        if age_seconds > STALENESS_HORIZON_SEC:
            return result

    for i, node_id in enumerate(node_ids):
        raw = row.get(f"node_{i}")
        if raw is None:
            continue
        try:
            result[node_id] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def latest_biometrics_induction_by_node(
    engine: Engine,
    nodes: Sequence[str],
    *,
    max_age_sec: float = 180.0,
) -> dict[str, dict[str, Any]]:
    """Latest `orion_biometrics_induction` row per requested node.

    Real windowed level/trend/spike_rate/volatility per hardware channel,
    already computed by `orion-biometrics` and persisted by `orion-sql-writer`
    off the `channel_biometrics_induction` bus payload (the same payload
    `orion-state-service`'s in-memory `_biometrics_induction_by_node` cache
    ingests -- see that service's `store.py::ingest_biometrics_induction`).
    That in-memory cache is not reused here: `MetacogContextService`'s existing
    `state.get_latest` RPC requests `scope="global"`, which resolves via
    `orion-state-service`'s own `_pick_latest` to exactly one node (the
    configured `primary_node`, or whichever node's induction happens to be
    newest) -- not the per-node comparison this cue wants across the known
    field-topology hardware nodes. Reading the durable table directly is a
    fresh, small query, not a duplicate of that existing RPC path.

    `timestamp` on `orion_biometrics_induction` is a text column (see
    `services/orion-sql-writer/app/models/biometrics_induction.py`), not a
    native timestamp column -- confirmed live 2026-07-30 that a raw
    `timestamp > ...` comparison against a bound timestamptz errors ("operator
    does not exist: character varying > timestamp with time zone"), so the
    projected `ts` still casts explicitly for the max-age filter below.

    Shaped as a LATERAL "top-1 per node", not `DISTINCT ON (node)`, and two
    separate things forced that -- both measured live 2026-09-03 against the
    real 187k-row table, because neither is visible from the code alone:

    1. The old ORDER BY cast `timestamp::timestamptz`. `varchar::timestamptz`
       is not IMMUTABLE, so Postgres refuses to index that expression
       (`functions in index expression must be marked IMMUTABLE`) -- the cast
       made the ordering unindexable by construction.
    2. Removing the cast and adding `orion_biometrics_induction_node_ts_idx`
       was still not enough: `DISTINCT ON` has to consume its whole sorted
       input, and Postgres has no loose/skip index scan, so it kept choosing a
       parallel seq scan + external merge sort over all 138,927 matching rows
       to return two. Measured 911ms and ~150MB of temp spill WITH the index
       in place.

    The LATERAL form asks for exactly what is wanted -- one `ORDER BY
    timestamp DESC LIMIT 1` per requested node -- which the index answers in
    two lookups: 0.47ms and 8 shared buffer hits, versus 911ms and 30,015.
    The index is load-bearing for this shape too: with `enable_indexscan`,
    `enable_bitmapscan` AND `enable_indexonlyscan` all off (disabling only the
    first still leaves bitmap paths available, which under-measures), it is
    163ms for one node and 422ms for three. Neither half of the fix stands
    alone.

    Note what that means: losing the index does NOT raise or time out. 163ms
    is well inside the Hub's 2000ms statement_timeout, so the read simply
    sequential-scans a 247MB table forever, silently. orion-sql-writer's boot
    check (services/orion-sql-writer/app/main.py) is what reports that.

    Cost scales linearly in node count -- one index lookup per requested node
    -- which is right for the two callers today (1 and 3 nodes). A
    `METACOG_TREND_CUE_NODES` grown to many nodes would eventually want a
    different shape.

    Cost of getting this wrong, from pg_stat_statements before the fix: 418ms
    mean over 36,393 calls and 11.5 TB of cumulative temp spill -- 92% of all
    temp I/O on the instance -- once the Hub's Biometrics card began polling
    it every 10s per node.

    That is safe because the stored text sorts chronologically: every row is
    written by orion-sql-writer through psycopg2 against a server with
    `TimeZone=Etc/UTC` and `DateStyle=ISO`, so Postgres renders each value in
    fixed-width `YYYY-MM-DD HH:MM:SS[.ffffff]+00` form. Verified live
    2026-09-03 across all 187,563 rows: zero off-format values, and zero rows
    where text ordering and timestamptz ordering disagree.

    Two caveats worth stating rather than leaving implicit, both surfaced in
    review:

    - This database collates `en_US.utf8`, not `C`, so the ordering does NOT
      reduce to byte order and any argument of the form "'+' sorts below any
      digit" is reasoning about the wrong collation. The empirical check above
      is the evidence; the byte-order intuition is not. A glibc collation
      version change inside the Postgres image could in principle reorder a
      text index, which here would surface as `LIMIT 1` quietly returning an
      older row that the max-age filter then drops -- "no reading", no error.
    - The fixed-width guarantee is a property of the WRITER's session, not of
      this column. `services/orion-sql-writer/app/worker.py` passes a real
      `datetime` to psycopg2, and Postgres assignment-casts it to varchar
      using that session's DateStyle/TimeZone. Setting e.g. `PGTZ` on the
      sql-writer container would start writing `-06` offsets, at which point
      text order and chronological order genuinely diverge. Pinned by
      `services/orion-sql-writer/tests/test_biometrics_induction_timestamp_format.py`.

    Returns only nodes with a real row within `max_age_sec`; a missing or
    stale node is simply absent from the returned dict, not a zero-filled
    placeholder (same "honest absence over degenerate zero" contract as
    `latest_node_prediction_errors`).
    """
    if not nodes:
        return {}

    query = text(
        """
        SELECT n.node AS node, latest.metrics AS metrics,
               latest.timestamp::timestamptz AS ts
        FROM unnest(CAST(:nodes AS text[])) AS n(node)
        CROSS JOIN LATERAL (
            SELECT b.metrics, b.timestamp
            FROM orion_biometrics_induction b
            WHERE b.node = n.node
            ORDER BY b.timestamp DESC
            LIMIT 1
        ) AS latest
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"nodes": list(nodes)}).mappings().all()

    now = datetime.now(timezone.utc)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ts = row.get("ts")
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (now - ts).total_seconds() > max_age_sec:
            continue
        metrics = row.get("metrics")
        node = row.get("node")
        if isinstance(metrics, dict) and node:
            out[str(node)] = metrics
    return out


def most_notable_trend_channel(
    metrics: Mapping[str, Any],
) -> tuple[str, dict[str, float]] | None:
    """Pick the single channel whose `trend` deviates furthest from 0.5 (no
    directional change) out of one node's `orion_biometrics_induction.metrics`
    dict.

    Pure, no I/O. Exists so the evidence cue stays compact (one channel per
    node, not all ~14) while still always returning *something* -- including
    on a genuinely calm tick, where the "most notable" channel just reports a
    near-0.5 trend, honestly representing calm rather than being filtered out
    by a fixed threshold that could hide "nothing is moving" as easily as it
    hides noise.
    """
    best: tuple[str, dict[str, float]] | None = None
    best_dev = -1.0
    for channel, raw in metrics.items():
        if not isinstance(raw, Mapping):
            continue
        trend = raw.get("trend")
        if not isinstance(trend, (int, float)) or isinstance(trend, bool):
            continue
        dev = abs(float(trend) - 0.5)
        if dev > best_dev:
            best_dev = dev
            best = (str(channel), dict(raw))
    return best


def build_recent_trend_signals_cue(
    *,
    prediction_errors: Mapping[str, float | None],
    induction_by_node: Mapping[str, Mapping[str, Any]],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the compact, structured evidence cue rendered into
    `ctx["recent_trend_signals"]` -- a real evidence cue, not a raw telemetry
    dump: prediction_error is already a single scalar per node, and induction
    is reduced to each node's single most-notable channel via
    `most_notable_trend_channel` rather than all ~14 channels.
    """
    cue: dict[str, Any] = {
        "prediction_error": {
            node_id: value for node_id, value in prediction_errors.items()
        },
        "biometrics_induction": {},
    }
    for node, metrics in induction_by_node.items():
        notable = most_notable_trend_channel(metrics)
        if notable is None:
            continue
        channel, values = notable
        cue["biometrics_induction"][node] = {
            "channel": channel,
            "trend": values.get("trend"),
            "level": values.get("level"),
            "spike_rate": values.get("spike_rate"),
            "volatility": values.get("volatility"),
        }
    if generated_at is not None:
        cue["as_of"] = generated_at.isoformat()
    return cue
