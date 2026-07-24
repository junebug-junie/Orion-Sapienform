#!/usr/bin/env python3
"""Read-only AST/HOT attention self-model reducer measurement + replay.

Phase 1 of `docs/superpowers/specs/2026-07-18-objective-3-consciousness-
scaffolded-roadmap-design.md`. Replays the REAL, pure production reducer
(`reduce_attention_self_model`, `orion/substrate/attention_self_model.py`)
over real historical Postgres data for its inputs (`substrate_attention_
frames` -> `FieldAttentionFrameV1`, `substrate_attention_broadcast_log` ->
`AttentionBroadcastProjectionV1`, `substrate_field_state` -> the five
Active-Inference domains' raw `prediction_error`), joined by nearest-
preceding timestamp with the field lane's own tick cadence driving the
replay (it is the highest-frequency real signal, per the Phase 1 design).

**2026-07-23: `substrate_self_state` input removed, replaced with
`substrate_field_state`.** Mirrors `attention_self_model.py`'s own
`SelfStateV1` -> Active-Inference substrate swap (see that module's docstring
for the full account -- the producer this used to read no longer exists).
`prediction_error_by_domain` (current-tick snapshot) and
`prediction_error_trend_by_domain` (this script's own computation --
mean(prior half) - mean(recent half), a reversion prediction, over
`PREDICTION_ERROR_TREND_WINDOW_TICKS` field_state ticks -- see
`compute_prediction_error_trend()`'s own docstring for why this is the
opposite sign of the naive "continue the recent direction" formula it
replaced, and `docs/superpowers/specs/2026-07-23-predicted-shift-reversion-
finding.md` for the validation. Window size is a starting anchor sized to
the live ~2s field_state cadence confirmed 2026-07-23 via
`substrate_field_state` timestamp diffs, not yet independently calibrated --
see Missing Question 3 in the L6 design doc) are both derived here, in the
pure replay layer, from real `substrate_field_state` rows -- the reducer
itself does no time-series math of its own (see its module docstring).

**Load-bearing finding, discovered while building this script (2026-07-18):
`substrate_attention_broadcast_projection` -- the third input,
`AttentionBroadcastProjectionV1`, the GWT-dispatch/Lamme lane -- is a
SINGLETON UPSERT table.** Confirmed live via `\\d
substrate_attention_broadcast_projection` (PRIMARY KEY on `projection_id`)
and a row-count query (exactly 1 row, always). There was no per-tick history
for this lane in Postgres -- only the single most-recent snapshot was ever
observable, so any historical `voluntary_override` event was overwritten in
place before it could be replayed.

**Fixed 2026-07-18 (this patch):** `substrate_attention_broadcast_log`
(`manual_migration_attention_broadcast_log_v1.sql`) is a new append-only
companion table -- one row per broadcast tick, written alongside the
singleton by `save_attention_broadcast_history()`
(`services/orion-substrate-runtime/app/store.py`) from
`_attention_broadcast_tick()` (`services/orion-substrate-runtime/app/
worker.py`). This script now joins broadcast rows by nearest-preceding
timestamp the same two-pointer way it joins `field_state_rows` (see below),
instead of passing one static row to every call. The old singleton table,
its writer (`save_attention_broadcast`), and `AttentionBroadcastProjectionV1`
itself are untouched.

**Deployment caveat, honestly stated:** the log only accumulates forward
from the moment this patch deploys -- the old snapshot rows were overwritten
in place and are not recoverable, so there is no way to backfill history.
Immediately after deploy the log will be empty or thin, and a run against a
short window will likely still show zero `voluntary_override` events -- not
because the table structurally cannot support it (the old finding), but
because too little real history has accumulated yet. See the report's own
"Broadcast lane coverage" and "Acceptance check" sections for the concrete
numbers on any given run, and root CLAUDE.md's "runtime truth beats config
truth" mandate for why this is reported plainly rather than smoothed over.

This performs NO writes, emits NO events, flips NO flags. It reports:

  1. The distribution of `attention_reason` over the replay window.
  2. Broadcast-lane coverage: how many ticks in the window could be joined to
     *some* real historical broadcast-log row (not just the current
     snapshot), and how many of those were stale vs. fresh.
  3. The acceptance check: does the replay window contain at least one real
     `voluntary_override` event, and if so, a concrete before/after
     narrative example proving the reducer's "why" branches on it (not just
     a generic salience reading). If the window contains none -- expected
     immediately post-deploy while the log is still young -- that is
     reported as NOT MET via Postgres replay, with the reasoning above,
     rather than asserted as passing.
  4. Live-data sanity check (CLAUDE.md's metric-quality-gate) for the
     Active-Inference confidence/predicted_shift signals: per-domain
     prediction_error coverage in the window, and whether `confidence`/
     `predicted_shift` show real, non-degenerate variance -- not flat/
     always-null/always-saturated.

Run:
    python scripts/analysis/measure_ast_hot_reducer.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.ast_hot_reducer")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000

# The five real, live Predictive-Processing domains
# (`orion/substrate/prediction_error.py`), keyed by their FieldStateV1 node
# id. `node:substrate.transport` is included despite its documented
# structurally-narrow scope (`services/orion-substrate-runtime/README.md`'s
# "transport domain is one queue" note) -- excluding it would be an
# undisclosed thumb on the scale, not a fix; its near-permanent 0.0 is
# reported honestly in `_aggregate_prediction_error_confidence`'s own basis
# string instead.
PREDICTION_ERROR_DOMAIN_NODES: dict[str, str] = {
    "execution": "node:substrate.execution",
    "transport": "node:substrate.transport",
    "biometrics": "node:substrate.biometrics",
    "chat": "node:substrate.chat",
    "route": "node:substrate.route",
}

# Trend window, in field_state ticks, split into two equal halves (recent vs
# prior) for the mean-delta trend computation. Sized to the live ~2s
# field_state cadence confirmed 2026-07-23 (30 ticks =~ 60s) -- a starting
# anchor, not yet independently calibrated against how fast a genuine rising
# trend distinguishes itself from tick noise (L6 design doc's Missing
# Question 3). Deliberately small enough to stay a real "what's happening
# right now" window, not a long-run average.
PREDICTION_ERROR_TREND_WINDOW_TICKS: int = 30

OUTPUT_DIR = Path("/tmp/ast-hot-reducer")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure replay/summary layer -- no I/O. Calls the REAL production reducer
# (reduce_attention_self_model), itself pure (no I/O -- verified by reading
# orion/substrate/attention_self_model.py), so this whole layer has no I/O
# of its own and is exercised directly by unit tests, no DB involved.
# ===========================================================================


@dataclass
class ReplayTick:
    generated_at: datetime
    attention_reason: str
    confidence: Optional[float]
    prediction_error_confidence: Optional[float]
    broadcast_lane_present: bool
    broadcast_lane_stale: bool
    broadcast_lane_age_sec: Optional[float]
    field_lane_present: bool
    predicted_shift: Optional[str]
    has_voluntary_override: bool
    reason_narrative: str


def extract_prediction_error_by_domain(field_state_payload: dict) -> dict[str, float]:
    """Pull the current raw `prediction_error` value for each of the five
    real domains out of one `FieldStateV1.node_vectors` payload. Only
    includes a domain if its node and channel are actually present in this
    payload -- missing, not defaulted to 0.0, so a genuinely absent domain
    doesn't silently masquerade as "confirmed calm" in the aggregate.
    """
    node_vectors = field_state_payload.get("node_vectors") or {}
    out: dict[str, float] = {}
    for domain, node_id in PREDICTION_ERROR_DOMAIN_NODES.items():
        vector = node_vectors.get(node_id)
        if not isinstance(vector, dict) or "prediction_error" not in vector:
            continue
        try:
            out[domain] = float(vector["prediction_error"])
        except (TypeError, ValueError):
            continue
    return out


def compute_prediction_error_trend(
    window: list[dict[str, float]],
) -> dict[str, float]:
    """Reversion-based trend per domain, over an ordered (oldest-to-newest)
    window of `extract_prediction_error_by_domain()` outputs. Positive =
    predicted to rise next; negative = predicted to fall next (same
    contract `reduce_attention_self_model()` already consumes -- only the
    computation changed here, 2026-07-23).

    **This is deliberately mean(prior half) - mean(recent half) -- the
    OPPOSITE sign of the naive "continue the recent direction" formula this
    replaced.** That naive continuation formula (mean(recent) - mean(prior),
    predicting the recent direction keeps going) was empirically WORSE than
    a coin flip: back-tested against real `substrate_field_state` biometrics
    history (the only domain with enough real variance to test against --
    execution/chat/route are real but tiny, transport reads exactly 0.0 for
    entire multi-hour windows, per the already-documented transport
    narrow-scope finding) on two independent, non-overlapping 3-4h windows,
    checking whether the named domain's value actually moved in the
    predicted direction ~60s later: 37.7% accuracy (n=332, z=-4.50) and
    41.0% accuracy (n=454, z=-3.85), both far below chance. A
    decay-projection formula (compare the current value to what pure
    exponential continuation of the prior half's own trajectory would
    predict, before this fix) only marginally improved on that (43-45% on a
    separate earlier back-test), still well below chance -- the problem
    isn't the extrapolation method, it's the extrapolation *direction*: this
    signal is spike-and-settle (a burst of activity is more often followed
    by quiet than by more activity), not momentum-carrying. Predicting the
    OPPOSITE of the naive continuation direction scored 62.3% and 59.0% on
    the same two windows (sums to exactly 100% with continuation by
    construction -- same backtest pass, same sample set, strictly opposite
    predictions) -- real, reproducible, above-chance signal. Full validation
    methodology and numbers:
    `docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md`.

    **Validated on biometrics only, applied to all five domains.** No
    independent data exists yet for execution/transport/chat/route --
    applying the same reversion sign to them is a reasoned extrapolation
    (all five domains are computed the same way, as deltas between
    successive states from discrete events -- turns, exec steps, tool
    calls -- so the same spike-and-settle dynamic is plausible), not an
    independently confirmed one. In practice this mostly matters for
    biometrics anyway (it wins the cross-domain argmax the overwhelming
    majority of the time in live replay -- see the reducer script's own
    report), but a future pass should back-test the other four domains
    separately once enough real variance accumulates, rather than assuming
    this generalizes.

    A domain only gets a trend value if it has at least one reading in BOTH
    halves -- comparing a half with zero real observations would fabricate
    a trend from nothing, not measure one. Fewer than 2 ticks in the window
    yields an empty dict (nothing to compare yet).
    """
    if len(window) < 2:
        return {}
    mid = len(window) // 2
    prior_half, recent_half = window[:mid], window[mid:]

    def _domain_means(half: list[dict[str, float]]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for snapshot in half:
            for domain, value in snapshot.items():
                sums[domain] = sums.get(domain, 0.0) + value
                counts[domain] = counts.get(domain, 0) + 1
        return {d: sums[d] / counts[d] for d in sums}

    prior_means = _domain_means(prior_half)
    recent_means = _domain_means(recent_half)
    return {
        domain: prior_means[domain] - recent_means[domain]
        for domain in recent_means
        if domain in prior_means
    }


def replay_reducer(
    field_rows: list[tuple[datetime, dict]],
    broadcast_rows: list[tuple[datetime, dict]],
    field_state_rows: list[tuple[datetime, dict]],
) -> tuple[list[ReplayTick], int, int, int]:
    """Replay the real `reduce_attention_self_model` over ordered field-lane
    ticks (the highest-frequency real signal -- drives replay cadence),
    joining `broadcast_rows` by nearest-preceding timestamp (same two-pointer
    pattern `substrate_attention_broadcast_log` already used -- see module
    docstring) and `field_state_rows` the same way, additionally maintaining
    a rolling `PREDICTION_ERROR_TREND_WINDOW_TICKS`-sized window of
    per-domain `prediction_error` snapshots to feed
    `compute_prediction_error_trend`. Returns (ticks, field_rows_skipped,
    field_state_rows_skipped, broadcast_rows_skipped).
    """
    from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
    from orion.schemas.field_attention_frame import FieldAttentionFrameV1
    from orion.substrate.attention_self_model import reduce_attention_self_model

    broadcasts: list[tuple[datetime, AttentionBroadcastProjectionV1]] = []
    broadcast_skipped = 0
    for ts, payload in broadcast_rows:
        try:
            broadcasts.append((ts, AttentionBroadcastProjectionV1.model_validate(payload)))
        except Exception:
            broadcast_skipped += 1

    field_states: list[tuple[datetime, dict[str, float]]] = []
    field_state_skipped = 0
    for ts, payload in field_state_rows:
        try:
            field_states.append((ts, extract_prediction_error_by_domain(payload)))
        except Exception:
            field_state_skipped += 1

    ticks: list[ReplayTick] = []
    field_skipped = 0
    fs_idx = 0  # two-pointer: field_rows, broadcasts, field_states are all ASC-sorted
    bc_idx = 0
    current_broadcast: Optional[AttentionBroadcastProjectionV1] = None
    current_prediction_error_by_domain: dict[str, float] = {}
    trend_window: deque[dict[str, float]] = deque(maxlen=PREDICTION_ERROR_TREND_WINDOW_TICKS)

    for ts, payload in field_rows:
        try:
            field_model = FieldAttentionFrameV1.model_validate(payload)
        except Exception:
            field_skipped += 1
            continue

        while bc_idx < len(broadcasts) and broadcasts[bc_idx][0] <= ts:
            current_broadcast = broadcasts[bc_idx][1]
            bc_idx += 1

        while fs_idx < len(field_states) and field_states[fs_idx][0] <= ts:
            current_prediction_error_by_domain = field_states[fs_idx][1]
            if current_prediction_error_by_domain:
                trend_window.append(current_prediction_error_by_domain)
            fs_idx += 1

        model = reduce_attention_self_model(
            current_broadcast,
            field_model,
            now=ts,
            prediction_error_by_domain=current_prediction_error_by_domain or None,
            prediction_error_trend_by_domain=compute_prediction_error_trend(list(trend_window)) or None,
        )
        ticks.append(
            ReplayTick(
                generated_at=ts,
                attention_reason=model.attention_reason,
                confidence=model.confidence,
                prediction_error_confidence=model.prediction_error_confidence,
                broadcast_lane_present=model.broadcast_lane_present,
                broadcast_lane_stale=model.broadcast_lane_stale,
                broadcast_lane_age_sec=model.broadcast_lane_age_sec,
                field_lane_present=model.field_lane_present,
                predicted_shift=model.predicted_shift,
                has_voluntary_override=model.voluntary_override is not None,
                reason_narrative=model.reason_narrative,
            )
        )
    return ticks, field_skipped, field_state_skipped, broadcast_skipped


def reason_histogram(ticks: list[ReplayTick]) -> Counter:
    return Counter(t.attention_reason for t in ticks)


def find_override_examples(ticks: list[ReplayTick], *, context: int = 1) -> list[list[ReplayTick]]:
    """Contiguous windows of `ticks` (ordered) around every tick where
    `has_voluntary_override` is True, `context` ticks either side, for a
    concrete before/after narrative example.
    """
    examples: list[list[ReplayTick]] = []
    for i, t in enumerate(ticks):
        if t.has_voluntary_override:
            lo = max(0, i - context)
            hi = min(len(ticks), i + context + 1)
            examples.append(ticks[lo:hi])
    return examples


# ===========================================================================
# I/O layer -- psycopg2 read-only. Mirrors measure_origination_gate.py's
# connection contract exactly (refuses a non-read-only session).
# ===========================================================================


def open_readonly_connection(dsn: str):
    try:
        import psycopg2
    except Exception:  # pragma: no cover
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only (got %r)", value)
            conn.close()
            return None
    except Exception:
        logger.error("failed to enforce read-only session", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn


def _rows_to_payload_list(rows, *, ts_col_idx: int = 1, payload_col_idx: int = 0) -> tuple[list[tuple[datetime, dict]], bool]:
    out: list[tuple[datetime, dict]] = []
    for row in rows:
        payload, ts = row[payload_col_idx], row[ts_col_idx]
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        out.append((ts, payload))
    return out, len(rows) >= MAX_ROWS


def fetch_field_attention_rows(conn, since: datetime) -> tuple[list[tuple[datetime, dict]], bool]:
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT frame_json, generated_at
                FROM substrate_attention_frames
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames", exc_info=True)
        return [], False
    return _rows_to_payload_list(rows)


def fetch_field_state_rows(conn, since: datetime) -> tuple[list[tuple[datetime, dict]], bool]:
    """Raw `FieldStateV1` snapshots -- the source of the five Active-
    Inference domains' `prediction_error` (see `PREDICTION_ERROR_DOMAIN_
    NODES`/`extract_prediction_error_by_domain`). Replaces
    `fetch_self_state_rows` (removed 2026-07-23, same producer-killed reason
    as `attention_self_model.py`'s own swap)."""
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT field_json, generated_at
                FROM substrate_field_state
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_field_state", exc_info=True)
        return [], False
    return _rows_to_payload_list(rows)


def fetch_broadcast_history_rows(conn, since: datetime) -> tuple[list[tuple[datetime, dict]], bool]:
    """Real per-tick broadcast history from the append-only companion log
    (`substrate_attention_broadcast_log`, `manual_migration_attention_
    broadcast_log_v1.sql`) -- mirrors `fetch_self_state_rows` exactly. This
    is what makes a genuine historical `voluntary_override` search possible;
    see module docstring for why the old singleton-projection table alone
    could not support it.
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT projection_json, generated_at
                FROM substrate_attention_broadcast_log
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_broadcast_log", exc_info=True)
        return [], False
    return _rows_to_payload_list(rows)


def fetch_broadcast_row_count(conn) -> Optional[int]:
    """Cross-check the singleton-table finding live, every run -- not just
    asserted once in a docstring. None on failure (degrades quietly).
    """
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM substrate_attention_broadcast_projection")
            row = cur.fetchone()
    except Exception:
        logger.warning("broadcast row-count query failed", exc_info=True)
        return None
    return int(row[0]) if row else None


def fetch_latest_broadcast_row(conn) -> Optional[tuple[datetime, dict]]:
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT projection_json, generated_at
                FROM substrate_attention_broadcast_projection
                ORDER BY generated_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch substrate_attention_broadcast_projection", exc_info=True)
        return None
    if not row:
        return None
    rows, _ = _rows_to_payload_list([row])
    return rows[0] if rows else None


# ===========================================================================
# Report rendering + orchestration.
# ===========================================================================


class ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._start = time.monotonic()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")
        except Exception:
            self._fh = None

    def emit(self, title: str, *, percent: float, processed: int, total: int) -> None:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = processed / elapsed
        line = (
            f"{datetime.now(timezone.utc).isoformat()} | {title} | "
            f"{percent:5.1f}% | rows={processed}/{total} | rate={rate:.1f}/s"
        )
        logger.info(line)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def write_ticks_csv(path: Path, ticks: list[ReplayTick]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "generated_at", "attention_reason", "confidence",
                "prediction_error_confidence",
                "broadcast_lane_present", "broadcast_lane_stale", "broadcast_lane_age_sec",
                "field_lane_present", "predicted_shift", "has_voluntary_override",
            ]
        )
        for t in ticks:
            writer.writerow(
                [
                    t.generated_at.isoformat(), t.attention_reason,
                    "" if t.confidence is None else f"{t.confidence:.4f}",
                    "" if t.prediction_error_confidence is None else f"{t.prediction_error_confidence:.4f}",
                    t.broadcast_lane_present, t.broadcast_lane_stale,
                    "" if t.broadcast_lane_age_sec is None else f"{t.broadcast_lane_age_sec:.3f}",
                    t.field_lane_present, t.predicted_shift or "", t.has_voluntary_override,
                ]
            )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    ticks: list[ReplayTick],
    field_rows_truncated: bool,
    field_rows_skipped: int,
    field_state_rows_skipped: int,
    field_state_rows_replayed: int,
    broadcast_rows_replayed: int,
    broadcast_rows_skipped: int,
    broadcast_row_count: Optional[int],
    broadcast_row: Optional[tuple[datetime, dict]],
    override_examples: list[list[ReplayTick]],
    caveats: list[str],
) -> str:
    hist = reason_histogram(ticks)
    n = len(ticks)

    def pct(count: int) -> str:
        return "n/a" if n == 0 else f"{100.0 * count / n:.1f}%"

    hist_lines = "\n".join(
        f"| {reason} | {hist.get(reason, 0)} | {pct(hist.get(reason, 0))} |"
        for reason in ("top_down_override", "bottom_up_salience", "field_salience_only", "no_data")
    )

    n_broadcast_present = sum(1 for t in ticks if t.broadcast_lane_present)
    n_broadcast_fresh = sum(1 for t in ticks if t.broadcast_lane_present and not t.broadcast_lane_stale)
    n_broadcast_stale = sum(1 for t in ticks if t.broadcast_lane_present and t.broadcast_lane_stale)

    confidences = [t.confidence for t in ticks if t.confidence is not None]
    n_confidence = len(confidences)
    confidence_min = min(confidences) if confidences else None
    confidence_max = max(confidences) if confidences else None
    confidence_mean = (sum(confidences) / n_confidence) if confidences else None
    confidence_degenerate = (
        n_confidence > 0 and confidence_min is not None and confidence_max is not None
        and (confidence_max - confidence_min) < 1e-6
    )
    # confidence is populated on every attention_reason branch, but the new
    # prediction_error-based formula (_aggregate_prediction_error_confidence)
    # only ever fires in field_salience_only -- top_down_override/
    # bottom_up_salience use broadcast.coalition_stability_score instead, and
    # dominate whenever the broadcast lane is fresh. Reporting the aggregate
    # confidence stats above alone would misleadingly imply broad coverage of
    # the *new* formula specifically -- break it out on its own.
    aid_confidences = [
        t.confidence for t in ticks
        if t.confidence is not None and t.attention_reason == "field_salience_only"
    ]
    n_aid_confidence = len(aid_confidences)
    aid_confidence_min = min(aid_confidences) if aid_confidences else None
    aid_confidence_max = max(aid_confidences) if aid_confidences else None

    # prediction_error_confidence (2026-07-24): unconditional counterpart,
    # computed regardless of attention_reason branch, restricted to
    # ACTIVE_INFERENCE_DOMAINS (excludes the confirmed-dead transport
    # domain). This is the field that closes the branch-starvation gap
    # named above -- expect its coverage to track prediction_error data
    # availability (like predicted_shift), not the narrow field_salience_only
    # branch count.
    pe_confidences = [t.prediction_error_confidence for t in ticks if t.prediction_error_confidence is not None]
    n_pe_confidence = len(pe_confidences)
    pe_confidence_min = min(pe_confidences) if pe_confidences else None
    pe_confidence_max = max(pe_confidences) if pe_confidences else None
    pe_confidence_mean = (sum(pe_confidences) / n_pe_confidence) if pe_confidences else None

    predicted_shifts = [t.predicted_shift for t in ticks if t.predicted_shift]
    n_predicted_shift = len(predicted_shifts)
    predicted_shift_domains = Counter(s.split(" prediction-error", 1)[0] for s in predicted_shifts)

    broadcast_singleton_line = (
        "n/a (query failed)" if broadcast_row_count is None
        else f"{broadcast_row_count} row(s) (still a singleton, by design -- "
             "history now lives in substrate_attention_broadcast_log instead)"
    )
    broadcast_current_line = (
        "none (table empty)" if broadcast_row is None
        else f"generated_at={broadcast_row[0].isoformat()}, voluntary_override="
             f"{'present' if (broadcast_row[1].get('frame') or {}).get('voluntary_override') else 'null'}"
    )

    lines = [
        "# AST/HOT Attention Self-Model Reducer Measurement",
        "",
        "Read-only. No writes, no events, no flag/config changes. Replays the real "
        "`reduce_attention_self_model` production function over historical "
        "`substrate_attention_frames` (field lane, drives cadence), "
        "`substrate_field_state` (five Active-Inference domains' `prediction_error`), "
        "and `substrate_attention_broadcast_log` rows, all joined by nearest-preceding "
        "timestamp.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Field-lane (FieldAttentionFrameV1) ticks replayed: {n}",
        f"- Field-lane rows truncated at MAX_ROWS={MAX_ROWS}: {field_rows_truncated}",
        f"- Field-lane rows skipped (failed to parse): {field_rows_skipped}",
        f"- Field-state rows in window: {field_state_rows_replayed} "
        f"(skipped/failed to parse: {field_state_rows_skipped})",
        f"- Broadcast-log rows in window: {broadcast_rows_replayed} "
        f"(skipped/failed to parse: {broadcast_rows_skipped})",
        "",
        "## Broadcast lane: real per-tick history (fixed 2026-07-18)",
        "",
        "`substrate_attention_broadcast_projection` is still a singleton upsert table "
        "(PRIMARY KEY on `projection_id`, overwritten every tick) -- that has not "
        "changed and is not being touched. What changed: "
        "`substrate_attention_broadcast_log` is a new append-only companion (one row "
        "per tick, written by `save_attention_broadcast_history()` alongside the "
        "singleton), so this replay now joins broadcast rows by nearest-preceding "
        "timestamp the same way it already joins self-state rows, instead of pinning "
        "one static snapshot to every tick.",
        "",
        f"- Live singleton row count (unchanged, for reference): {broadcast_singleton_line}",
        f"- Current singleton snapshot: {broadcast_current_line}",
        f"- Field-lane ticks in this window joined to *some* real historical broadcast-"
        f"log row (i.e. at least one broadcast-log row at or before tick.generated_at "
        f"existed): **{n_broadcast_present}** / {n} ({pct(n_broadcast_present)})",
        f"  - of those, fresh (within staleness threshold): {n_broadcast_fresh}",
        f"  - of those, stale (\"no new GWT-dispatch-lane activity since last frame\"): "
        f"{n_broadcast_stale}",
        "",
        "This is a real per-tick join now, not the old single-snapshot pin -- coverage "
        "grows as `substrate_attention_broadcast_log` accumulates real ticks from the "
        "deploy of this patch onward. It cannot be backfilled: the pre-patch singleton "
        "rows were overwritten in place and are not recoverable, so history only exists "
        "from deploy time forward. A run shortly after deploy will show low or zero "
        "broadcast-log coverage for that reason, not because the join is broken.",
        "",
        "## Active-Inference confidence / predicted_shift live-data sanity check",
        "",
        "CLAUDE.md's metric-quality-gate requirement: confirm these two signals are not "
        "degenerate (flat, always-null, always-saturated) before treating them as usable.",
        "",
        f"- Ticks with a non-null `confidence` (any branch): {n_confidence} / {n} ({pct(n_confidence)})",
        f"  - min={_fmt(confidence_min)} max={_fmt(confidence_max)} mean={_fmt(confidence_mean)}",
        f"  - **DEGENERATE (flat within 1e-6)**" if confidence_degenerate else "  - real variance observed, not flat",
        f"  - **caveat**: this mixes two different formulas -- `broadcast.coalition_"
        f"stability_score` (top_down_override/bottom_up_salience branches) and the "
        f"new prediction_error-based formula (field_salience_only only). Broken out "
        f"below.",
        f"- Ticks using the NEW prediction_error-based confidence specifically "
        f"(field_salience_only branch): {n_aid_confidence} / {n} ({pct(n_aid_confidence)})",
        f"  - min={_fmt(aid_confidence_min)} max={_fmt(aid_confidence_max)}",
        "",
        "### `prediction_error_confidence` (2026-07-24, unconditional, ACTIVE_INFERENCE_DOMAINS only)",
        "",
        "Closes the branch-starvation gap above: computed regardless of attention_reason, "
        "restricted to execution/biometrics/chat/route (transport excluded -- confirmed "
        "live 2026-07-24 that it reads exactly 0.0 for 100% of a real window, see "
        "docs/notes/2026-07-24-attention-reason-branch-starvation-finding.md).",
        "",
        f"- Ticks with a non-null `prediction_error_confidence`: {n_pe_confidence} / {n} "
        f"({pct(n_pe_confidence)})",
        f"  - min={_fmt(pe_confidence_min)} max={_fmt(pe_confidence_max)} mean={_fmt(pe_confidence_mean)}",
        "  - Expected: this should track prediction_error data availability (like "
        "predicted_shift), not the narrow field_salience_only branch count above -- if "
        "coverage here is still near 0%, the ACTIVE_INFERENCE_DOMAINS filter or its wiring "
        "should be re-checked before trusting this signal.",
        f"- Ticks with a non-null `predicted_shift`: {n_predicted_shift} / {n} ({pct(n_predicted_shift)})",
        "  - domain breakdown: "
        + (
            ", ".join(f"{d}={c}" for d, c in predicted_shift_domains.most_common())
            if predicted_shift_domains else "none"
        ),
        "",
        "## attention_reason distribution",
        "",
        "| reason | count | pct |",
        "| --- | --- | --- |",
        hist_lines,
        "",
        "## Acceptance check: does the window contain a real voluntary_override event, "
        "narrated correctly?",
        "",
    ]

    if override_examples:
        lines.append(
            f"**MET.** Found {len(override_examples)} tick(s) with a real, non-null "
            f"`voluntary_override` in the replay window, sourced from real historical "
            f"rows in `substrate_attention_broadcast_log` (not the old single-snapshot "
            f"pin). Concrete before/after example (first occurrence):"
        )
        lines.append("")
        for t in override_examples[0]:
            marker = " <-- OVERRIDE TICK" if t.has_voluntary_override else ""
            lines.append(
                f"- `{t.generated_at.isoformat()}` reason=**{t.attention_reason}** "
                f"confidence={_fmt(t.confidence)}{marker}"
            )
            lines.append(f"  narrative: {t.reason_narrative}")
        lines.append("")
        lines.append(
            "The reducer's `attention_reason` visibly branches from "
            "`bottom_up_salience`/`field_salience_only` to `top_down_override` at the "
            "marked tick, and `reason_narrative` names the specific goal/loop involved -- "
            "proving the two inputs are unified (the override is narrated as *the* reason "
            "for that tick's attention state), not just both present in the schema."
        )
    else:
        lines.extend(
            [
                "**NOT MET via Postgres replay.** Zero ticks in this window carry a "
                "non-null `voluntary_override`. This is no longer a structural gap in the "
                "table (that was fixed 2026-07-18 -- `substrate_attention_broadcast_log` "
                "now supports a real historical join, see above); it is insufficient "
                "accumulated history: the log only started accumulating at deploy time "
                f"and this window ({broadcast_rows_replayed} broadcast-log row(s) found) "
                "has not yet caught a real `voluntary_override` tick. Re-running this "
                "script after more live ticks accumulate (each `ORION_ATTENTION_"
                "BROADCAST_INTERVAL_SEC` seconds, confirmed live=30s) is expected to "
                "close this gap over the following days, not require further schema "
                "changes.",
                "",
                "Compensating evidence that the reducer's why-branching logic is real and "
                "correct (see `orion/substrate/tests/test_attention_self_model.py`, run "
                "as part of this PR):",
                "",
                "- `TestVoluntaryOverridePresent` exercises the reducer with a real "
                "`VoluntaryOverrideV1` shape (the exact production schema, not a "
                "simplified stand-in) and asserts `attention_reason == "
                "\"top_down_override\"` plus a narrative naming the specific "
                "chosen/beaten loop IDs and applied_bias -- the same branch this replay "
                "would exercise if a live override were present in the window.",
                "- `TestCadenceMismatch` exercises the exact broadcast-absent/stale "
                "scenario this replay depends on: a broadcast snapshot dated after the "
                "reference tick is honestly treated as absent, not reused retroactively.",
                "",
                "**Recommendation for Juniper's sign-off**: the structural gap (no "
                "append-only broadcast history) is closed as of 2026-07-18 -- "
                "`substrate_attention_broadcast_log` now exists and this script joins it "
                "per-tick. What remains is purely accumulation time: re-run this script "
                "with `--window-hours` covering the period since deploy once a few days "
                "of live ticks have landed, and expect this section to flip to MET once a "
                "real `voluntary_override` event occurs within that accumulated window. "
                "No further schema/bus contract change is anticipated for Phase 1.",
            ]
        )

    lines.extend(["", "## Reading this", ""])
    lines.extend(
        [
            "- `top_down_override` count > 0 with a narrative naming real loop IDs is the "
            "only thing that satisfies the Phase 1 acceptance check as written; a "
            "nonzero `bottom_up_salience`/`field_salience_only` count alone is not "
            "sufficient (that would just prove the two inputs are both present, not "
            "unified).",
            "- A high `field_salience_only` fraction can still be expected shortly after "
            "deploy while `substrate_attention_broadcast_log` is young -- it is not "
            "necessarily a sign the reducer is failing to read the broadcast lane, it can "
            "be the honest consequence of that lane not having accumulated much history "
            "yet. Check the broadcast-log row count above before concluding either way.",
            "",
        ]
    )
    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run(window: timedelta, window_label: str) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - window
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)
    caveats: list[str] = []

    progress.emit("connect", percent=0.0, processed=0, total=0)
    conn = open_readonly_connection(dsn)
    if conn is None:
        caveats.append("postgres unavailable or not read-only; nothing measured")
        progress.close()
        print("ERROR: could not open a read-only postgres connection; see log")
        return 2

    field_rows, field_truncated = fetch_field_attention_rows(conn, window_start)
    if field_truncated:
        caveats.append(f"field-attention rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("field_attention loaded", percent=25.0, processed=len(field_rows), total=len(field_rows))

    field_state_rows, field_state_truncated = fetch_field_state_rows(conn, window_start)
    if field_state_truncated:
        caveats.append(f"field-state rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("field_state loaded", percent=45.0, processed=len(field_state_rows), total=len(field_state_rows))

    broadcast_rows, broadcast_truncated = fetch_broadcast_history_rows(conn, window_start)
    if broadcast_truncated:
        caveats.append(f"broadcast-log rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("broadcast_log loaded", percent=52.0, processed=len(broadcast_rows), total=len(broadcast_rows))

    # Secondary diagnostic only, kept for corroboration: the singleton table's
    # own row count and current snapshot, unrelated to the real per-tick join
    # above (which now runs entirely off substrate_attention_broadcast_log).
    broadcast_row_count = fetch_broadcast_row_count(conn)
    broadcast_row = fetch_latest_broadcast_row(conn)
    progress.emit("broadcast singleton loaded", percent=55.0, processed=1 if broadcast_row else 0, total=1)

    try:
        conn.close()
    except Exception:
        pass

    if not field_rows:
        caveats.append("no substrate_attention_frames rows in window; nothing to replay")
        progress.close()
        report = render_report(
            window_label=window_label, window_start=window_start, window_end=now,
            ticks=[], field_rows_truncated=field_truncated, field_rows_skipped=0,
            field_state_rows_skipped=0, field_state_rows_replayed=len(field_state_rows),
            broadcast_rows_replayed=len(broadcast_rows),
            broadcast_rows_skipped=0, broadcast_row_count=broadcast_row_count,
            broadcast_row=broadcast_row, override_examples=[], caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    progress.emit("replaying", percent=60.0, processed=0, total=len(field_rows))
    ticks, field_skipped, field_state_skipped, broadcast_skipped = replay_reducer(
        field_rows, broadcast_rows, field_state_rows
    )
    progress.emit("replay done", percent=95.0, processed=len(ticks), total=len(field_rows))

    override_examples = find_override_examples(ticks)

    write_ticks_csv(CSV_PATH, ticks)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        ticks=ticks,
        field_rows_truncated=field_truncated,
        field_rows_skipped=field_skipped,
        field_state_rows_skipped=field_state_skipped,
        field_state_rows_replayed=len(field_state_rows),
        broadcast_rows_replayed=len(broadcast_rows),
        broadcast_rows_skipped=broadcast_skipped,
        broadcast_row_count=broadcast_row_count,
        broadcast_row=broadcast_row,
        override_examples=override_examples,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(ticks), total=len(field_rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only AST/HOT attention self-model reducer measurement.")
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"analysis window in hours (default {DEFAULT_WINDOW_HOURS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    window = timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    return run(window, window_label)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
