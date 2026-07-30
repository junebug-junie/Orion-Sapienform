#!/usr/bin/env python3
"""Read-only measurement: does GitHub issue #1512's redundancy premise hold,
and does chat-level attention salience have any ground-truth outcome data to
validate against?

Issue #1512 (Sentience Striving Program) asks whether chat-level/coalition
attention salience (`orion/substrate/attention/salience.py`'s hand-weighted
`SEED_WEIGHTS` blend, `weights_version="seed-v1"`) is redundant with Layer 5
field attention (`orion/attention/field_attention/`, Candidate A/B, PR
#1484/#1488) or the AST/HOT reducer (`AttentionSelfModelV1`), per the
charter's (`orion/sentience_striving_program/README.md`) sec 7 rule: "Reuse
the live pipeline, don't parallel it... before being built."

This script answers that with four real, computed checks against live
Postgres -- it does not re-derive the answer from source reading alone, and
it does not assert any of the following without a live query backing it:

1. **Target-universe disjointness.** Chat-level salience scores *open loops*
   (`attention_salience_trace.theme_key`, e.g. `"open-loop-e11d318a9036"` --
   a conversational-continuity object, `OpenLoopV1`). Layer 5's
   `FieldAttentionFrameV1` (`orion/schemas/field_attention_frame.py`) targets
   are `node:*`/`capability:*`/`field:*`/`system:*` (physical hosts,
   capabilities, system-level aggregates -- `FieldAttentionTargetV1.
   target_kind`). Confirmed live 2026-07-30: `attention_salience_trace` has
   284 rows / 5 distinct `theme_key` values since 2026-07-24 (`open-loop-
   e11d318a9036`, `open-loop-1c822db7221d`, `open-loop-8f8766373aba`,
   `open-loop-3be13c7644a0`, `open-loop-9d84d08cddf5`); `substrate_
   attention_frames` (125,658 rows since 2026-07-27) has 19 distinct
   `target_id` values across all five target lists (`dominant_targets`,
   `node_targets`, `capability_targets`, `system_targets`,
   `suppressed_targets`) -- `capability:graph`, `capability:llm_inference`,
   `capability:memory`, `capability:orchestration`, `capability:storage`,
   `capability:transport`, `capability:vision`, `field:recent_
   perturbations`, `node:athena`, `node:atlas`, `node:circe`, `node:
   prometheus`, `node:rpc_timeout`, `node:substrate.biometrics`, `node:
   substrate.bus_synaptic`, `node:substrate.chat`, `node:substrate.
   execution`, `node:substrate.route`, `node:substrate.transport`. Zero
   overlap. This script re-derives that intersection from a real query
   every run rather than trusting this docstring's own numbers going stale.

2. **AST/HOT winner traceability.** `AttentionSelfModelV1.
   broadcast_selected_open_loop_id` (`orion/schemas/attention_self_model.
   py`) is sourced from `AttentionBroadcastProjectionV1` (the GWT-dispatch
   lane) -- confirmed live 2026-07-30: of 4,647 `substrate_attention_self_
   model` rows in its own ~168h retention window (`SUBSTRATE_ATTENTION_
   SELF_MODEL_LOG_RETENTION_HOURS`, `services/orion-substrate-runtime/app/
   settings.py`), 342 name a non-null `broadcast_selected_open_loop_id`,
   and every one of those values is exactly one of the 5 real `theme_key`s
   above (`open-loop-1c822db7221d`: 213, `open-loop-e11d318a9036`: 129).
   This means AST/HOT does not independently re-score open loops -- it is a
   narrative wrapper around the coalition winner this module already
   decided, not a competing/duplicate scorer. This script checks the
   stronger claim -- that every named winner is a real theme_key that was
   ALREADY being scored (its earliest `attention_salience_trace` sighting
   is at or before the self-model tick that names it) -- not just that the
   id happens to look like a loop id.

3. **The ground-truth outcome gap (the headline).** `attention_loop_
   outcome` (written only by `services/orion-hub/scripts/attention_loops_
   store.py::persist_loop_outcome`, called only from `attention_loops_
   routes.py`'s Resolve/Dismiss operator routes -- confirmed via
   `grep -rl attention_loop_outcome`, no other caller) records human/
   automated verdicts on
   whether a surfaced loop's salience score was actually worth surfacing.
   Confirmed live 2026-07-30: **0 rows, ever** -- re-checked fresh every
   run below, never hardcoded. There is no recorded instance of anything
   confirming or refuting either salience formula's real-world accuracy.
   Structurally the same shape as `measure_autonomy_gate.py`'s finding that
   the autonomy origination signal never fired once in 84,511 ticks -- a
   mechanism that computes continuously but has never been checked against
   real ground truth.

4. **Which hand-weighted formula is actually live.** Two independent,
   unvalidated hand-weighted formulas exist for the same object type:
   `orion/substrate/attention/salience.py`'s `SEED_WEIGHTS` blend (used when
   `ORION_ATTENTION_SALIENCE_V2_ENABLED` is truthy) and `orion/substrate/
   attention/scoring.py::score_loop`'s own separate inline formula (the
   fallback when the flag is off). This script reads the SAME flag the
   SAME way `salience.py`'s own `salience_v2_enabled()` does (same env key,
   same truthy set, imported directly -- not reimplemented) rather than
   trusting this docstring's "confirmed `true` in `.env` 2026-07-30" claim.

**This performs NO writes, emits NO events, flips NO flags, changes NO
consumer.** It is a read-only measurement instrument -- exactly the
prerequisite step the charter's sec 7 rule and the metric-quality-gate
discipline (CLAUDE.md sec 0A) ask for before any redesign/retire decision,
not a cognition-loop change itself.

**Explicitly NOT decided here** (mirrors `measure_phase3_biometrics_drive_
shadow_comparison.py`'s own "explicitly NOT decided" framing): this script
does not decide whether to retire, merge, or redesign chat-level attention.
It only establishes (a) whether issue #1512's redundancy premise holds
(finding: no -- disjoint target universes, AST/HOT is a wrapper not a
competitor) and (b) whether ground-truth validation exists for either
salience formula (finding: no, re-checked live below). The recommended next
step -- closing the outcome-logging gap (e.g. wiring the operator Resolve/
Dismiss UI's real usage, or an automatic ground-truth proxy) -- is a
design/proposal-mode decision for Juniper, not something this patch does.

Run:
    python scripts/analysis/measure_chat_attention_ground_truth_gap.py --window-hours 168
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported directly, not reimplemented -- the exact live flag name and
# truthy set `salience.py`'s own `salience_v2_enabled()` uses, so this
# script's "which formula is live" finding can never silently drift from
# the real production check.
from orion.substrate.attention.salience import SALIENCE_V2_FLAG, _TRUTHY as SALIENCE_TRUTHY_VALUES  # noqa: E402

logger = logging.getLogger("orion.analysis.chat_attention_ground_truth_gap")

# 168h (7 days) -- covers `attention_salience_trace`'s full real history
# (284 rows, 2026-07-24 through 2026-07-30, confirmed live) with margin,
# and matches `substrate_attention_self_model`'s own real retention default
# (`SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS=168.0`,
# `services/orion-substrate-runtime/app/settings.py`) rather than picking an
# independent window size that would silently clip one side's real history.
DEFAULT_WINDOW_HOURS: float = 168.0
MAX_ROWS: int = 200_000

OUTPUT_DIR = Path("/tmp/chat-attention-ground-truth-gap")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "self_model_winner_traceability.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# The two hand-weighted formulas this script must correctly attribute --
# named here once so the report and the pure resolver function can't drift
# apart on the label text.
LIVE_FORMULA_V2 = "salience.py::compute_salience (SEED_WEIGHTS blend, weights_version=seed-v1, v2)"
LIVE_FORMULA_FALLBACK = "scoring.py::score_loop (dormant inline fallback formula, used only when the v2 flag is off)"


# ===========================================================================
# Pure layer -- no I/O. Deterministic, unit-testable without a DB.
# ===========================================================================


@dataclass(frozen=True)
class DisjointnessResult:
    """Real, computed target-universe overlap check -- never assumed zero."""

    theme_key_count: int
    target_id_count: int
    overlap: frozenset[str]

    @property
    def is_disjoint(self) -> bool:
        return len(self.overlap) == 0

    @property
    def is_conclusive(self) -> bool:
        """False when either side's universe is empty. A vacuous "0 overlap"
        computed from an empty set is not evidence of disjointness -- it is
        the metric-quality-gate's "suspiciously clean 0.0" failure class
        (CLAUDE.md sec 0A): degenerate, not confirmed-calm. The report must
        not read `is_disjoint` as a real finding unless both universes were
        actually populated this run.
        """
        return self.theme_key_count > 0 and self.target_id_count > 0


def compute_target_universe_disjointness(
    theme_keys: Iterable[str], target_ids: Iterable[str]
) -> DisjointnessResult:
    """Intersection of chat-level open-loop `theme_key`s and Layer 5's real
    `target_id` space. Blank/None values are dropped rather than counted, so
    a malformed row can't manufacture a spurious "overlap" or inflate either
    side's count.
    """
    tk = {str(k) for k in theme_keys if k}
    ti = {str(t) for t in target_ids if t}
    return DisjointnessResult(
        theme_key_count=len(tk), target_id_count=len(ti), overlap=frozenset(tk & ti)
    )


def compute_theme_first_seen(
    trace_rows: list[tuple[datetime, str, float]]
) -> dict[str, datetime]:
    """Earliest `created_at` per `theme_key`, from real
    `attention_salience_trace` rows -- the "was this loop already being
    scored" evidence the winner-traceability check needs. Pure aggregation,
    no I/O.
    """
    first_seen: dict[str, datetime] = {}
    for ts, theme_key, _salience in trace_rows:
        if not theme_key:
            continue
        existing = first_seen.get(theme_key)
        if existing is None or ts < existing:
            first_seen[theme_key] = ts
    return first_seen


@dataclass(frozen=True)
class WinnerTraceabilityResult:
    """Per-row verdict: does AST/HOT's named `broadcast_selected_open_loop_id`
    correspond to a real, already-scored loop -- not a dangling/unrelated id.
    """

    total_self_model_rows: int
    rows_with_winner: int
    traceable_count: int
    untraceable_loop_ids: tuple[str, ...]

    @property
    def all_traceable(self) -> bool:
        return self.rows_with_winner > 0 and self.traceable_count == self.rows_with_winner


def check_winner_traceability(
    self_model_rows: list[tuple[datetime, Optional[str]]],
    theme_first_seen: dict[str, datetime],
) -> WinnerTraceabilityResult:
    """For every AST/HOT row naming a `broadcast_selected_open_loop_id`,
    confirm that id is a real `attention_salience_trace.theme_key` whose
    earliest sighting is AT OR BEFORE this self-model tick's `generated_at`
    -- i.e. the loop was already being independently scored when AST/HOT
    narrated it as the winner, not a coincidentally-shaped id with no real
    backing. A loop id absent from `theme_first_seen`, or first sighted
    AFTER the self-model tick (which would mean AST/HOT is somehow ahead of
    its own real source), is honestly reported untraceable -- never assumed
    matched.
    """
    total = len(self_model_rows)
    with_winner = [(ts, lid) for ts, lid in self_model_rows if lid]
    traceable = 0
    untraceable: list[str] = []
    for ts, lid in with_winner:
        first_seen = theme_first_seen.get(lid)
        matched = False
        if first_seen is not None:
            fs = first_seen if first_seen.tzinfo else first_seen.replace(tzinfo=timezone.utc)
            t = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            matched = fs <= t
        if matched:
            traceable += 1
        else:
            untraceable.append(lid)
    return WinnerTraceabilityResult(
        total_self_model_rows=total,
        rows_with_winner=len(with_winner),
        traceable_count=traceable,
        untraceable_loop_ids=tuple(sorted(set(untraceable))[:10]),
    )


def resolve_live_salience_formula(raw_flag_value: Optional[str]) -> tuple[str, bool]:
    """Which of the two hand-weighted open-loop salience formulas is live,
    given the raw `ORION_ATTENTION_SALIENCE_V2_ENABLED` env value.

    Uses the SAME truthy set `salience.py`'s own `salience_v2_enabled()`
    checks against (`SALIENCE_TRUTHY_VALUES`, imported directly from that
    module) -- not an independently-chosen set of truthy strings that could
    silently diverge from the real check. Pure function (takes the raw
    value as an argument) so this is testable without mutating process env;
    the I/O layer below passes the real `os.environ` value it read.
    Returns (formula_label, is_v2).
    """
    is_v2 = str(raw_flag_value if raw_flag_value is not None else "false").strip().lower() in SALIENCE_TRUTHY_VALUES
    return (LIVE_FORMULA_V2 if is_v2 else LIVE_FORMULA_FALLBACK), is_v2


# ===========================================================================
# I/O layer -- psycopg2 read-only. Same connection contract as the other
# measure_*.py scripts in this directory (refuses a non-read-only session).
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


def fetch_salience_trace_rows(conn, since: datetime) -> tuple[list[tuple[datetime, str, float]], bool]:
    """(created_at, theme_key, salience) ASC, within window. Feeds BOTH the
    disjointness check's `theme_key` set and the winner-traceability check's
    per-theme first-seen map -- one real query, two consumers, so the two
    checks can never see inconsistent snapshots of this table.
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, theme_key, salience
                FROM attention_salience_trace
                WHERE created_at >= %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch attention_salience_trace", exc_info=True)
        return [], False
    out: list[tuple[datetime, str, float]] = []
    for created_at, theme_key, salience in rows:
        if created_at is None or not theme_key:
            continue
        ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
        out.append((ts, str(theme_key), float(salience) if salience is not None else 0.0))
    return out, len(rows) >= MAX_ROWS


def fetch_loop_outcome_count(conn, since: datetime) -> Optional[int]:
    """Real row count in `attention_loop_outcome` within the window -- the
    headline number. `None` (never `0`) on any fetch failure, so a DB hiccup
    can never masquerade as a confirmed-zero finding.
    """
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM attention_loop_outcome WHERE created_at >= %s",
                (since,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        logger.error("failed to fetch attention_loop_outcome count", exc_info=True)
        return None


def fetch_self_model_rows(conn, since: datetime) -> tuple[list[tuple[datetime, Optional[str]]], bool]:
    """(generated_at, broadcast_selected_open_loop_id) ASC, within window.
    `substrate_attention_self_model` has exactly one writer in the repo
    (`_attention_broadcast_tick()`'s self-model block,
    `services/orion-substrate-runtime/app/store.py::save_attention_self_
    model`) -- reading it here is a cross-service read of a table this
    script never writes to, same pattern as `get_latest_field_attention_
    frame()`'s own comment in that store module.
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, self_model_json ->> 'broadcast_selected_open_loop_id'
                FROM substrate_attention_self_model
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_self_model", exc_info=True)
        return [], False
    out: list[tuple[datetime, Optional[str]]] = []
    for generated_at, loop_id in rows:
        if generated_at is None:
            continue
        ts = generated_at if generated_at.tzinfo else generated_at.replace(tzinfo=timezone.utc)
        out.append((ts, str(loop_id) if loop_id else None))
    return out, len(rows) >= MAX_ROWS


_FRAME_TARGET_LIST_FIELDS = (
    "dominant_targets", "node_targets", "capability_targets",
    "system_targets", "suppressed_targets",
)


def fetch_frame_target_ids(conn, since: datetime) -> tuple[set[str], int, bool]:
    """Distinct `target_id` values across ALL five `FieldAttentionFrameV1`
    target lists (`orion/schemas/field_attention_frame.py`) -- Layer 5's
    real target-id space, within the window. Restricted to the most recent
    `MAX_ROWS` frames (matching every other fetcher's truncation
    convention) before extraction, rather than scanning the whole table
    unbounded.

    Returns (distinct target ids, frame-row count actually scanned for
    extraction, truncated-at-MAX_ROWS flag). `scanned_row_count` is the size
    of the SAME capped `recent_frames` set the target-id extraction reads
    from -- not a separate, uncapped `count(*)` over the full window (an
    earlier version of this function reported the uncapped total here,
    which meant "truncated" could fire while the displayed number was still
    the true, larger total -- silently misleading about how many rows the
    id extraction actually saw once this table exceeds MAX_ROWS rows in the
    window; not yet triggered live, fixed on review before it could bite).
    """
    if conn is None:
        return set(), 0, False
    union_parts = " UNION ALL ".join(
        f"SELECT jsonb_array_elements(frame_json->'{field}') AS t FROM recent_frames"
        for field in _FRAME_TARGET_LIST_FIELDS
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH recent_frames AS (
                    SELECT frame_json FROM substrate_attention_frames
                    WHERE generated_at >= %s
                    ORDER BY generated_at DESC
                    LIMIT %s
                )
                SELECT count(*) FROM recent_frames
                """,
                (since, MAX_ROWS),
            )
            scanned_row_count = int(cur.fetchone()[0])
            cur.execute(
                f"""
                WITH recent_frames AS (
                    SELECT frame_json FROM substrate_attention_frames
                    WHERE generated_at >= %s
                    ORDER BY generated_at DESC
                    LIMIT %s
                ),
                targets AS ( {union_parts} )
                SELECT DISTINCT t ->> 'target_id' FROM targets
                """,
                (since, MAX_ROWS),
            )
            id_rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames target ids", exc_info=True)
        return set(), 0, False
    target_ids = {str(r[0]) for r in id_rows if r[0]}
    return target_ids, scanned_row_count, scanned_row_count >= MAX_ROWS


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


def write_traceability_csv(
    path: Path,
    self_model_rows: list[tuple[datetime, Optional[str]]],
    theme_first_seen: dict[str, datetime],
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["generated_at", "broadcast_selected_open_loop_id", "loop_first_seen_at", "traceable"]
        )
        for ts, lid in self_model_rows:
            if not lid:
                writer.writerow([ts.isoformat(), "", "", ""])
                continue
            first_seen = theme_first_seen.get(lid)
            fs = None
            if first_seen is not None:
                fs = first_seen if first_seen.tzinfo else first_seen.replace(tzinfo=timezone.utc)
            t = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            traceable = fs is not None and fs <= t
            writer.writerow(
                [ts.isoformat(), lid, fs.isoformat() if fs else "", traceable]
            )


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    trace_row_count: int,
    trace_truncated: bool,
    disjointness: DisjointnessResult,
    frame_row_count: int,
    frame_truncated: bool,
    traceability: WinnerTraceabilityResult,
    self_model_truncated: bool,
    outcome_count: Optional[int],
    formula_label: str,
    formula_is_v2: bool,
    raw_flag_value: Optional[str],
    caveats: list[str],
) -> str:
    lines = [
        "# Chat-level attention ground-truth gap -- read-only measurement",
        "",
        "Answers GitHub issue #1512: is chat-level/coalition open-loop attention "
        "salience (`orion/substrate/attention/salience.py`) redundant with Layer 5 field "
        "attention or the AST/HOT reducer, and does either salience formula have real "
        "ground-truth outcome data to validate against? Read-only. No writes, no events, "
        "no flag/config changes.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        "",
        "## 1. Target-universe disjointness "
        "(chat-level open loops vs. Layer 5 field targets)",
        "",
        f"- `attention_salience_trace` rows in window: {trace_row_count}"
        + (" (truncated at MAX_ROWS)" if trace_truncated else ""),
        f"- Distinct `theme_key` values (chat-level open-loop ids): {disjointness.theme_key_count}",
        f"- `substrate_attention_frames` rows in window: {frame_row_count}"
        + (" (truncated at MAX_ROWS)" if frame_truncated else ""),
        f"- Distinct `target_id` values (Layer 5 field targets, all 5 target lists): "
        f"{disjointness.target_id_count}",
        f"- **Overlap: {len(disjointness.overlap)}**"
        + (f" -- {sorted(disjointness.overlap)}" if disjointness.overlap else ""),
        (
            "- **VERDICT: INCONCLUSIVE** -- one or both id universes were empty this "
            "window (theme_key_count={tk}, target_id_count={ti}); a 0-overlap result "
            "computed from an empty set is degenerate, not evidence of real "
            "disjointness. See coverage caveats.".format(
                tk=disjointness.theme_key_count, ti=disjointness.target_id_count
            )
            if not disjointness.is_conclusive else
            f"- **VERDICT: {'DISJOINT' if disjointness.is_disjoint else 'NOT DISJOINT'}** -- "
            + (
                "zero overlap, computed live, not assumed, over two real non-empty "
                "universes. Chat-level salience and Layer 5 field attention score "
                "different real object types; issue #1512's naive redundancy framing "
                "does not hold at the target-id level."
                if disjointness.is_disjoint else
                "a real overlap was found -- issue #1512's redundancy concern may be more "
                "founded than this script's docstring assumed; inspect the overlapping ids "
                "above before concluding anything."
            )
        ),
        "",
        "## 2. AST/HOT winner traceability "
        "(does `broadcast_selected_open_loop_id` name a real, already-scored loop?)",
        "",
        f"- `substrate_attention_self_model` rows in window: {traceability.total_self_model_rows}"
        + (" (truncated at MAX_ROWS)" if self_model_truncated else ""),
        f"- Rows naming a non-null `broadcast_selected_open_loop_id`: {traceability.rows_with_winner}",
        f"- Traceable to a real `theme_key` already being scored at or before that tick: "
        f"{traceability.traceable_count} / {traceability.rows_with_winner}",
    ]
    if traceability.rows_with_winner == 0:
        lines.append(
            "- No self-model row in this window named a winner -- cannot evaluate "
            "traceability (see coverage caveats)."
        )
    elif traceability.all_traceable:
        lines.append(
            "- **VERDICT: fully traceable.** Every named winner is a real, "
            "independently-tracked open loop this system already scored -- consistent "
            "with AST/HOT being a narrative wrapper around the coalition winner "
            "(`AttentionBroadcastProjectionV1`), not a second, competing scorer."
        )
    else:
        lines.append(
            f"- **VERDICT: NOT fully traceable.** {len(traceability.untraceable_loop_ids)} "
            f"distinct untraceable loop id(s), e.g. {list(traceability.untraceable_loop_ids)} "
            "-- AST/HOT named a winner this script could not find as an already-scored "
            "`theme_key`. Worth investigating before trusting the wrapper-not-competitor "
            "reading."
        )
    lines.extend(
        [
            "",
            "## 3. Ground-truth outcome gap (the headline finding)",
            "",
            f"- `attention_loop_outcome` rows in window: "
            f"{'ERROR (see caveats)' if outcome_count is None else outcome_count}",
            f"- `attention_salience_trace` rows in the SAME window (context, from section 1): "
            f"{trace_row_count}",
        ]
    )
    if outcome_count == 0:
        lines.append(
            "- **VERDICT: zero ground-truth outcomes, ever, in this window.** Neither "
            "`salience.py`'s SEED_WEIGHTS blend nor `scoring.py`'s dormant fallback has a "
            "single recorded human/automated verdict (via the operator Resolve/Dismiss "
            "API, `services/orion-hub/scripts/attention_loops_routes.py`) confirming or "
            "refuting that a high-salience loop was actually worth surfacing. This -- not "
            "the redundancy question in section 1/2 -- is the load-bearing open question "
            "this measurement surfaces."
        )
    elif outcome_count is not None and outcome_count > 0:
        lines.append(
            f"- **VERDICT: {outcome_count} real outcome(s) exist** -- re-run with the "
            "outcome rows themselves to check whether high-salience loops correlate with "
            "'resolved' verdicts (not attempted by this script; see 'what this does not "
            "decide' below)."
        )
    else:
        lines.append("- Could not determine (fetch failed; see coverage caveats).")

    lines.extend(
        [
            "",
            "## 4. Which hand-weighted salience formula is live",
            "",
            f"- Raw `{SALIENCE_V2_FLAG}` env value read by this process: "
            f"{raw_flag_value!r}",
            f"- Resolved (same truthy check as `salience.py::salience_v2_enabled()`): "
            f"{'v2 (SEED_WEIGHTS)' if formula_is_v2 else 'dormant fallback (scoring.py::score_loop)'}",
            f"- **Live formula: {formula_label}**",
            "",
            "## What this does NOT decide",
            "",
            "- Does not decide whether to retire, merge, or redesign chat-level attention "
            "salience.",
            "- Does not validate either salience formula's real-world accuracy -- there is "
            "no ground-truth data to validate against (section 3).",
            "- Does not itself close the outcome-logging gap. The recommended next step -- "
            "wiring the operator Resolve/Dismiss UI's real usage, or building an automatic "
            "ground-truth proxy -- is a design/proposal-mode decision for Juniper, not "
            "something this patch does.",
            "",
            "## Coverage caveats",
            "",
        ]
    )
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

    trace_rows, trace_truncated = fetch_salience_trace_rows(conn, window_start)
    if trace_truncated:
        caveats.append(f"attention_salience_trace rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("salience_trace loaded", percent=20.0, processed=len(trace_rows), total=len(trace_rows))

    target_ids, frame_row_count, frame_truncated = fetch_frame_target_ids(conn, window_start)
    if frame_truncated:
        caveats.append(f"substrate_attention_frames rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("frame target ids loaded", percent=45.0, processed=frame_row_count, total=frame_row_count)

    self_model_rows, self_model_truncated = fetch_self_model_rows(conn, window_start)
    if self_model_truncated:
        caveats.append(f"substrate_attention_self_model rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("self_model loaded", percent=65.0, processed=len(self_model_rows), total=len(self_model_rows))

    outcome_count = fetch_loop_outcome_count(conn, window_start)
    if outcome_count is None:
        caveats.append("attention_loop_outcome count fetch failed")
    progress.emit("outcome count loaded", percent=80.0, processed=0, total=0)

    raw_flag_value = os.environ.get(SALIENCE_V2_FLAG)

    try:
        conn.close()
    except Exception:
        pass

    if not trace_rows:
        caveats.append("no attention_salience_trace rows in window; theme_key universe is empty")
    if frame_row_count == 0:
        caveats.append("no substrate_attention_frames rows in window; target_id universe is empty")
    if not self_model_rows:
        caveats.append("no substrate_attention_self_model rows in window; cannot check winner traceability")

    theme_keys = {tk for _, tk, _ in trace_rows}
    disjointness = compute_target_universe_disjointness(theme_keys, target_ids)
    theme_first_seen = compute_theme_first_seen(trace_rows)
    traceability = check_winner_traceability(self_model_rows, theme_first_seen)
    formula_label, formula_is_v2 = resolve_live_salience_formula(raw_flag_value)

    write_traceability_csv(CSV_PATH, self_model_rows, theme_first_seen)

    progress.emit("done", percent=100.0, processed=len(self_model_rows), total=len(self_model_rows))
    progress.close()

    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        trace_row_count=len(trace_rows),
        trace_truncated=trace_truncated,
        disjointness=disjointness,
        frame_row_count=frame_row_count,
        frame_truncated=frame_truncated,
        traceability=traceability,
        self_model_truncated=self_model_truncated,
        outcome_count=outcome_count,
        formula_label=formula_label,
        formula_is_v2=formula_is_v2,
        raw_flag_value=raw_flag_value,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only measurement of chat-level attention salience's redundancy "
            "premise (issue #1512) and ground-truth outcome gap."
        )
    )
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
