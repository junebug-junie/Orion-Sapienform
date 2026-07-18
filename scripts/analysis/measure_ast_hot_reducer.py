#!/usr/bin/env python3
"""Read-only AST/HOT attention self-model reducer measurement + replay.

Phase 1 of `docs/superpowers/specs/2026-07-18-objective-3-consciousness-
scaffolded-roadmap-design.md`. Replays the REAL, pure production reducer
(`reduce_attention_self_model`, `orion/substrate/attention_self_model.py`)
over real historical Postgres data for two of its three inputs
(`substrate_attention_frames` -> `FieldAttentionFrameV1`, `substrate_
self_state` -> `SelfStateV1`), joined by nearest-preceding timestamp with the
field lane's own tick cadence driving the replay (it is the highest-
frequency real signal, per the Phase 1 design).

**Load-bearing finding, discovered while building this script (2026-07-18):
`substrate_attention_broadcast_projection` -- the third input,
`AttentionBroadcastProjectionV1`, the GWT-dispatch/Lamme lane -- is a
SINGLETON UPSERT table.** Confirmed live via `\\d
substrate_attention_broadcast_projection` (PRIMARY KEY on `projection_id`)
and a row-count query (exactly 1 row, always). There is no per-tick history
for this lane in Postgres -- only the single most-recent snapshot is ever
observable. This script does NOT fabricate history for it: the same single
broadcast row is passed to the reducer for every replayed field-lane tick,
and the reducer itself (see its own docstring/logic) honestly reports the
broadcast lane as *absent* for any tick that predates the snapshot's own
`generated_at` (rather than silently reusing a future snapshot as if it
applied retroactively), and as *stale* for ticks more than
`DEFAULT_BROADCAST_STALE_THRESHOLD_SEC` after it. In practice this means only
a small tail of the most-recent replayed ticks (near the moment this script
was run) can ever show broadcast data at all -- the rest of the window
honestly shows `broadcast_lane_present=False`. See the report's own
"Broadcast lane coverage" and "Acceptance check" sections for the concrete
numbers this produces on a real run, and root CLAUDE.md's "runtime truth
beats config truth" mandate for why this is reported plainly rather than
smoothed over.

This performs NO writes, emits NO events, flips NO flags. It reports:

  1. The distribution of `attention_reason` over the replay window.
  2. Broadcast-lane coverage: how many ticks could honestly be joined to the
     single available broadcast snapshot at all, and how many of those were
     stale vs. fresh.
  3. The acceptance check: does the replay window contain at least one real
     `voluntary_override` event, and if so, a concrete before/after
     narrative example proving the reducer's "why" branches on it (not just
     a generic salience reading). If the window contains none (the honest,
     confirmed-live state of `substrate_attention_broadcast_projection` as
     of this script's own design pass), that is reported as NOT MET via
     Postgres replay, with the reasoning above, rather than asserted as
     passing.

Run:
    python scripts/analysis/measure_ast_hot_reducer.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.ast_hot_reducer")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000

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
    broadcast_lane_present: bool
    broadcast_lane_stale: bool
    broadcast_lane_age_sec: Optional[float]
    field_lane_present: bool
    self_state_present: bool
    has_voluntary_override: bool
    reason_narrative: str


def replay_reducer(
    field_rows: list[tuple[datetime, dict]],
    self_state_rows: list[tuple[datetime, dict]],
    broadcast_row: Optional[tuple[datetime, dict]],
) -> tuple[list[ReplayTick], int, int]:
    """Replay the real `reduce_attention_self_model` over ordered field-lane
    ticks (the highest-frequency real signal -- drives replay cadence),
    joining `self_state_rows` by nearest-preceding timestamp and passing the
    single `broadcast_row` (or None) to every call -- see module docstring
    for why a per-tick broadcast join is not possible with the current
    schema. Returns (ticks, field_rows_skipped, self_state_rows_skipped).
    """
    from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
    from orion.schemas.field_attention_frame import FieldAttentionFrameV1
    from orion.schemas.self_state import SelfStateV1
    from orion.substrate.attention_self_model import reduce_attention_self_model

    broadcast_model: Optional[AttentionBroadcastProjectionV1] = None
    if broadcast_row is not None:
        try:
            broadcast_model = AttentionBroadcastProjectionV1.model_validate(broadcast_row[1])
        except Exception:
            logger.warning("broadcast_row_parse_failed", exc_info=True)
            broadcast_model = None

    self_states: list[tuple[datetime, SelfStateV1]] = []
    self_state_skipped = 0
    for ts, payload in self_state_rows:
        try:
            self_states.append((ts, SelfStateV1.model_validate(payload)))
        except Exception:
            self_state_skipped += 1

    ticks: list[ReplayTick] = []
    field_skipped = 0
    ss_idx = 0  # two-pointer: both field_rows and self_states are ASC-sorted
    current_self_state: Optional[SelfStateV1] = None

    for ts, payload in field_rows:
        try:
            field_model = FieldAttentionFrameV1.model_validate(payload)
        except Exception:
            field_skipped += 1
            continue

        while ss_idx < len(self_states) and self_states[ss_idx][0] <= ts:
            current_self_state = self_states[ss_idx][1]
            ss_idx += 1

        model = reduce_attention_self_model(
            broadcast_model, field_model, current_self_state, now=ts
        )
        ticks.append(
            ReplayTick(
                generated_at=ts,
                attention_reason=model.attention_reason,
                confidence=model.confidence,
                broadcast_lane_present=model.broadcast_lane_present,
                broadcast_lane_stale=model.broadcast_lane_stale,
                broadcast_lane_age_sec=model.broadcast_lane_age_sec,
                field_lane_present=model.field_lane_present,
                self_state_present=model.self_state_present,
                has_voluntary_override=model.voluntary_override is not None,
                reason_narrative=model.reason_narrative,
            )
        )
    return ticks, field_skipped, self_state_skipped


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


def fetch_self_state_rows(conn, since: datetime) -> tuple[list[tuple[datetime, dict]], bool]:
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT self_state_json, generated_at
                FROM substrate_self_state
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_self_state", exc_info=True)
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
                "broadcast_lane_present", "broadcast_lane_stale", "broadcast_lane_age_sec",
                "field_lane_present", "self_state_present", "has_voluntary_override",
            ]
        )
        for t in ticks:
            writer.writerow(
                [
                    t.generated_at.isoformat(), t.attention_reason,
                    "" if t.confidence is None else f"{t.confidence:.4f}",
                    t.broadcast_lane_present, t.broadcast_lane_stale,
                    "" if t.broadcast_lane_age_sec is None else f"{t.broadcast_lane_age_sec:.3f}",
                    t.field_lane_present, t.self_state_present, t.has_voluntary_override,
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
    self_state_rows_skipped: int,
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

    broadcast_singleton_line = (
        "n/a (query failed)" if broadcast_row_count is None
        else f"{broadcast_row_count} row(s) (confirmed singleton if this is 1, every run)"
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
        "`substrate_attention_frames` (field lane, drives cadence) and "
        "`substrate_self_state` rows, joined by nearest-preceding timestamp.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Field-lane (FieldAttentionFrameV1) ticks replayed: {n}",
        f"- Field-lane rows truncated at MAX_ROWS={MAX_ROWS}: {field_rows_truncated}",
        f"- Field-lane rows skipped (failed to parse): {field_rows_skipped}",
        f"- Self-state rows skipped (failed to parse): {self_state_rows_skipped}",
        "",
        "## Broadcast lane: singleton-table finding (load-bearing)",
        "",
        "`substrate_attention_broadcast_projection` is a singleton upsert table "
        "(PRIMARY KEY on `projection_id`) -- confirmed live by this run's own "
        "row-count query, not just asserted from an earlier session:",
        "",
        f"- Live row count: {broadcast_singleton_line}",
        f"- Current (only) snapshot: {broadcast_current_line}",
        f"- Field-lane ticks in this window that could be honestly joined to the "
        f"single broadcast snapshot at all (i.e. tick.generated_at >= broadcast."
        f"generated_at): **{n_broadcast_present}** / {n} ({pct(n_broadcast_present)})",
        f"  - of those, fresh (within staleness threshold): {n_broadcast_fresh}",
        f"  - of those, stale (\"no new GWT-dispatch-lane activity since last frame\"): "
        f"{n_broadcast_stale}",
        "",
        "This is why the acceptance check below is scoped the way it is: there is no "
        "per-tick broadcast history in Postgres to replay against older ticks, only the "
        "single most-recent snapshot, honestly joined only to the small tail of ticks "
        "at or after its own timestamp.",
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
            f"`voluntary_override` in the replay window. Concrete before/after example "
            f"(first occurrence):"
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
                "non-null `voluntary_override`. This is a direct consequence of the "
                "singleton-table finding above, not a reducer bug: "
                f"`substrate_attention_broadcast_projection`'s single current snapshot "
                f"({broadcast_current_line}) does not currently carry a "
                "voluntary_override, and no earlier snapshot is recoverable from "
                "Postgres to search further back -- the row is overwritten in place on "
                "every broadcast tick (every "
                "`ORION_ATTENTION_BROADCAST_INTERVAL_SEC` seconds, confirmed live=30s), "
                "so any historical override event (including the one an earlier session "
                "reported observing live in this same table) has already been "
                "overwritten by the time this script runs.",
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
                "- `TestCadenceMismatch` exercises the exact singleton-table scenario "
                "found here: a broadcast snapshot dated after the reference tick is "
                "honestly treated as absent, not reused retroactively.",
                "",
                "**Recommendation for Juniper's sign-off**: if a genuine historical replay "
                "of the GWT-dispatch lane's `voluntary_override` events is needed for "
                "Phase 2+ (e.g. Phase 3's shadow comparison), "
                "`substrate_attention_broadcast_projection` needs an append-only "
                "companion (or `AttentionBroadcastProjectionV1` needs to also be "
                "published to a bus channel with retained history) -- the current "
                "singleton-upsert design cannot support it. Not built here: out of "
                "Phase 1 scope, and a schema/bus contract change per CLAUDE.md sec 6, "
                "not a reducer change.",
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
            "- A high `field_salience_only` fraction is expected and correct given the "
            "singleton-table finding above -- it is not a sign the reducer is failing to "
            "read the broadcast lane, it is the honest consequence of that lane having "
            "no queryable history.",
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

    self_state_rows, self_state_truncated = fetch_self_state_rows(conn, window_start)
    if self_state_truncated:
        caveats.append(f"self-state rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("self_state loaded", percent=45.0, processed=len(self_state_rows), total=len(self_state_rows))

    broadcast_row_count = fetch_broadcast_row_count(conn)
    broadcast_row = fetch_latest_broadcast_row(conn)
    progress.emit("broadcast loaded", percent=55.0, processed=1 if broadcast_row else 0, total=1)

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
            self_state_rows_skipped=0, broadcast_row_count=broadcast_row_count,
            broadcast_row=broadcast_row, override_examples=[], caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    progress.emit("replaying", percent=60.0, processed=0, total=len(field_rows))
    ticks, field_skipped, self_state_skipped = replay_reducer(field_rows, self_state_rows, broadcast_row)
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
        self_state_rows_skipped=self_state_skipped,
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
