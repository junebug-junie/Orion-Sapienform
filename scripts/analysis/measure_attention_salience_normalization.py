#!/usr/bin/env python3
"""Read-only measurement: would per-channel z-score normalization diversify
Layer 5 attention's top-1-winner selection over real history?

Session context (2026-07-28, Juniper conversation): tonight's investigation
found the same "noisiest wins" arbitration disease in three places in this
codebase -- old drives system (retired, 96% `dominant_drive=relational`
monoculture pre-fix), the proposal/policy pipeline (`orion/proposals/
scoring.py::proposal_priority()`, fixed-weight linear blend, never
calibrated against `orion/feedback/builder.py`'s real recorded outcomes),
and Layer 5 attention (`orion/attention/field_attention/{scoring,
selectors}.py`, live via `orion-attention-runtime`). The Sentience Striving
Program charter (`orion/sentience_striving_program/README.md`) §6 item 5 ran
`measure_emergent_clustering_probe.py` and found `select_system_targets`'s
`field:recent_perturbations` target (`orion/attention/field_attention/
selectors.py:128-140`, `salience = min(1.0, recent_perturbation_count /
10.0)`) wins top-1 in ~99.98% of ticks by saturating to ~1.0 under the
field's near-constant real perturbation rate -- structurally out-competing
every other target almost always, because nothing normalizes for one
channel being noisier/more-saturating than the others before comparing raw
magnitudes.

Juniper's converged direction: before writing a single line of live scoring
code, MEASURE whether per-channel normalization (z-scoring each channel
against its own real historical distribution before ranking) would actually
fix this, per the charter's own §7 rule ("Measure before minting. Every new
signal gets a read-only instrument and real historical replay before it
gates anything live.") This script is that instrument.

**Verified table (not assumed, re-checked fresh this run, not cited from a
prior write-up):** same `substrate_attention_frames` Postgres table (
`frame_json` jsonb column) that `measure_emergent_clustering_probe.py`
already reads, written by `services/orion-attention-runtime/app/
store.py::save_attention_frame()`. Row count and time range are queried
live at the top of every run -- see "real data span" in the report, not a
hardcoded prior number. CLAUDE.md's metric-quality-gate rule requires
re-verifying a claim every time a build depends on it, not trusting a past
write-up, even one from three days ago.

## Disclosed scoping decisions

1. **Data-access pattern, target parsing, and absence convention are reused
   from `measure_emergent_clustering_probe.py`** (same read-only-session
   contract, same `dominant_targets ∪ capability_targets` merge, same
   "absence in a tick's list == 0.0 salience" convention -- a target absent
   from a tick fell below the live policy's `min_salience=0.10`, not a
   missing observation). Duplicated here rather than imported, matching
   this directory's existing convention (`measure_emergent_clustering_
   probe.py`'s own docstring: "Connection contract mirrors
   measure_ast_hot_reducer.py / measure_capability_salience_coupling.py
   exactly" -- every sibling script in `scripts/analysis/` is self-
   contained, none import each other as library code; only their test
   files load a sibling module by path for unit testing).

2. **Single measurement window, not the emergent-clustering probe's A/B
   split.** That script's two-window design exists to check whether
   *clustering structure* recurs across time (a similarity-between-windows
   question). This script asks a different question -- "does normalizing
   against a channel's own historical distribution change who wins top-1"
   -- which only needs ONE historical distribution per channel to normalize
   against. Uses the full real available history in
   `substrate_attention_frames` (same table, same ~72h retention) as that
   one measurement window, since a larger sample gives a more stable
   per-channel mean/stddev estimate than an arbitrarily-shortened window
   would, and the emergent-clustering probe already establishes that a
   single global full-history read is exactly how the existing 99.98%
   baseline figure was computed (`full_history_top1` in that script). This
   script's own baseline table is computed fresh, independently, over
   whatever real span is available at run time -- not copy-pasted from
   that script's prior output.

3. **Normalization method: z-score against the channel's own full-history
   series (absence-inclusive), not min-max or an EWMA band.** Justification:
   this is the same normalization shape `orion-substrate-runtime`'s
   `_grammar_reducer_poll_loop()` already uses live for `gap_zscore` bus-
   synaptic anomaly detection (a real, working precedent for "z-score a
   channel against its own history before treating a raw magnitude as
   meaningful"), and it directly answers the question this measurement was
   commissioned to answer: does *this* channel's *this* tick's value stand
   out relative to *its own* normal range. An **existing-mechanism check**
   was run before choosing this (CLAUDE.md metric quality gate step 5):
   `orion/signals/normalization.py::EwmaBand` already implements a
   different per-metric normalizer (EWMA mean/deviation band mapped to
   [0,1]) used live by the organ-signal-gateway signal adapters -- a real,
   running alternative. Not reused here because (a) it is stateful/online
   (built for a live per-tick updating consumer, not a one-shot replay over
   already-stored history) and (b) this measurement needs the option to
   report "this channel could not be normalized at all" (near-zero
   variance) as an explicit, visible finding, which a bounded-to-[0,1] EWMA
   band would silently absorb rather than surface. Named here so the
   choice is a disclosed decision, not silence about an existing option.
   The per-channel z-score series is built the SAME way
   `measure_emergent_clustering_probe.py::align_series` already builds a
   per-target salience series (absence -> 0.0, aligned to tick order) --
   this script's z-score is computed on exactly that same series shape.

4. **Degenerate-variance handling is a first-class, reported finding, not a
   silently-produced number.** If a channel's full-history raw series has
   near-zero variance (same `DEGENERATE_VARIANCE_EPS = 1e-12` threshold
   `measure_emergent_clustering_probe.py::pearson_correlation` already
   uses), its z-score is undefined at every tick and it is EXCLUDED from
   the normalized top-1 ranking entirely -- it cannot win under normalized
   comparison, not because it "loses" a real z-score comparison, but
   because no z-score could be computed for it at all. This is reported as
   its own explicit table entry (`DEGENERATE (excluded)`), and if the
   channel that was previously the raw-ranking's dominant winner turns out
   to be exactly this degenerate channel, that is called out explicitly as
   its own finding: any subsequent share drop is a mechanical side-effect
   of disqualification-by-division-by-~0, not evidence that normalization
   is doing real calibration work. This script also checks, separately,
   whether the normalized ranking just relocates the monoculture to a
   *different* single channel (a `MONOCULTURE_SHIFTED` finding) rather than
   genuinely diversifying -- trading one "always wins" channel for another
   is not success.

5. **Classification** (`classify_normalization_effect`): explicit numeric
   bands, see that function's docstring for the exact thresholds and
   reasoning -- disclosed, not hidden in an if/else, same discipline as
   `measure_emergent_clustering_probe.py::classify_similarity`.

6. **This script is strictly read-only.** It opens a read-only Postgres
   session (refuses to proceed if the session cannot be forced read-only,
   same contract as every sibling `measure_*` script), performs SELECT-only
   queries against `substrate_attention_frames`, and writes only to
   `/tmp/measure-attention-salience-normalization/`. It does NOT import,
   read as executable, modify, or write back to
   `orion/attention/field_attention/scoring.py`, `selectors.py`, any other
   live-running code path, or any database table. If this measurement
   supports normalization as a real fix, implementing it is an explicit,
   separate, larger patch requiring its own proposal-mode sign-off per
   `orion/sentience_striving_program/README.md`'s charter and root
   CLAUDE.md §0A's cognition-loop rule -- not a next step folded into this
   script or this run.

Run:
    python scripts/analysis/measure_attention_salience_normalization.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.attention_salience_normalization")

DEGENERATE_VARIANCE_EPS: float = 1e-12
MIN_TOTAL_ROWS: int = 200
MAX_ROWS: int = 200_000
MONOCULTURE_SHARE_THRESHOLD: float = 0.5
MEANINGFUL_DIVERSITY_SHARE: float = 0.10
DIVERSIFICATION_DROP_THRESHOLD: float = 0.30

OUTPUT_DIR = Path("/tmp/measure-attention-salience-normalization")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O. Reused parsing shape from
# measure_emergent_clustering_probe.py (see module docstring #1), plus new
# per-channel z-score / normalized-ranking logic.
# ===========================================================================


def _parse_target_list(raw: Any) -> list[dict]:
    """Parse a raw `frame_json->'dominant_targets'` or `->'capability_targets'`
    value (possibly a JSON string, possibly malformed, possibly None) into a
    list of dict entries. Never raises."""
    try:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            return []
        return [e for e in raw if isinstance(e, dict)]
    except Exception:
        return []


def extract_target_salience_map(dominant_raw: Any, capability_raw: Any) -> dict[str, float]:
    """Merge one tick's `dominant_targets` and `capability_targets` raw JSON
    into a single {target_id: salience_score} map. Same convention as
    `measure_emergent_clustering_probe.py::extract_target_salience_map`.
    Never raises."""
    result: dict[str, float] = {}
    for entry in _parse_target_list(dominant_raw):
        tid = entry.get("target_id")
        if not isinstance(tid, str):
            continue
        try:
            score = float(entry.get("salience_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        result[tid] = score
    for entry in _parse_target_list(capability_raw):
        tid = entry.get("target_id")
        if not isinstance(tid, str) or tid in result:
            continue
        try:
            score = float(entry.get("salience_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        result[tid] = score
    return result


def build_target_universe(raw_maps: list[dict[str, float]]) -> list[str]:
    seen: set[str] = set()
    for m in raw_maps:
        seen.update(m.keys())
    return sorted(seen)


def align_series(raw_maps: list[dict[str, float]], target_ids: list[str]) -> dict[str, list[float]]:
    """Per-target salience_score series aligned to `raw_maps`' tick order.
    Absence in a tick's map -> 0.0."""
    series: dict[str, list[float]] = {t: [] for t in target_ids}
    for m in raw_maps:
        for t in target_ids:
            series[t].append(float(m.get(t, 0.0)))
    return series


def channel_mean_std(values: list[float]) -> tuple[float, float, bool]:
    """(mean, stddev, is_degenerate). Population stddev (matches
    `pearson_correlation`'s own variance convention -- no Bessel correction,
    since this is a description of the observed full-history population, not
    an inference from a sample of a larger unseen population). Degenerate iff
    variance <= DEGENERATE_VARIANCE_EPS (near-constant / flat series -- e.g.
    a saturating channel that sits at ~1.0 almost every tick)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, True
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance <= DEGENERATE_VARIANCE_EPS:
        return mean, 0.0, True
    return mean, math.sqrt(variance), False


def compute_zscore_series(
    target_ids: list[str], raw_series: dict[str, list[float]]
) -> tuple[dict[str, list[Optional[float]]], dict[str, tuple[float, float, bool]]]:
    """Per-channel z-score series, z[i] = (raw[i] - mean) / std, computed
    against that channel's OWN full-history raw series (see module docstring
    #3). A degenerate channel (near-zero variance) gets an all-`None` series
    -- never a fabricated 0.0 or an artificially clamped value, since there
    is no real deviation to measure. Returns (z_series_by_target,
    stats_by_target) where stats_by_target[t] = (mean, std, is_degenerate)."""
    z_series: dict[str, list[Optional[float]]] = {}
    stats: dict[str, tuple[float, float, bool]] = {}
    for t in target_ids:
        values = raw_series[t]
        mean, std, degenerate = channel_mean_std(values)
        stats[t] = (mean, std, degenerate)
        if degenerate:
            z_series[t] = [None] * len(values)
        else:
            z_series[t] = [(v - mean) / std for v in values]
    return z_series, stats


def top1_winner_raw(target_map: dict[str, float]) -> Optional[str]:
    """The single highest-raw-salience target_id for one tick. Deterministic
    tie-break: highest score, then alphabetical target_id. Same shape as
    `measure_emergent_clustering_probe.py::top1_winner`."""
    if not target_map:
        return None
    return sorted(target_map.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def top1_winner_distribution_raw(raw_maps: list[dict[str, float]]) -> Counter:
    counter: Counter = Counter()
    for m in raw_maps:
        winner = top1_winner_raw(m)
        if winner is not None:
            counter[winner] += 1
    return counter


def top1_winner_distribution_normalized(
    target_ids: list[str], z_series: dict[str, list[Optional[float]]], n_ticks: int
) -> tuple[Counter, int]:
    """Per-tick argmax over defined (non-None) z-scores only -- a degenerate
    channel is structurally excluded from ever winning here (module
    docstring #4), not scored as a real loser. Deterministic tie-break:
    highest z, then alphabetical target_id. Returns (winner_counter,
    n_ticks_with_no_defined_z) -- the second value is a real coverage gap
    (every channel degenerate at that tick), reported explicitly rather than
    silently dropped."""
    counter: Counter = Counter()
    n_no_winner = 0
    for i in range(n_ticks):
        candidates = [
            (t, z_series[t][i]) for t in target_ids if z_series[t][i] is not None
        ]
        if not candidates:
            n_no_winner += 1
            continue
        winner = sorted(candidates, key=lambda kv: (-kv[1], kv[0]))[0][0]
        counter[winner] += 1
    return counter, n_no_winner


def classify_normalization_effect(
    raw_top1: Counter,
    raw_n: int,
    norm_top1: Counter,
    norm_n: int,
) -> str:
    """Explicit classification bands for "would per-channel z-score
    normalization diversify Layer 5 attention's top-1 winner selection."
    These exact numbers are a disclosed scoping decision for THIS
    measurement (module docstring #5), not asserted as universally correct.

    Note: whether the raw winner is itself a degenerate (near-zero-variance)
    channel does NOT change which band this function returns -- these bands
    only classify the resulting winner-frequency shift. The "is the raw
    dominant channel structurally disqualified vs. genuinely out-competed"
    distinction is a separate, additional finding rendered in
    `render_report`'s "Honest findings" section, deliberately kept out of
    the classification verdict itself so degeneracy-driven relocation still
    reads as `NOT_MET_MONOCULTURE_SHIFTED`, not silently as `MET`.

    - `INSUFFICIENT_DATA`: fewer than MIN_TOTAL_ROWS real ticks, or the
      normalized ranking had zero valid (non-degenerate-only) ticks to rank.
      Too little real structure to judge either way.
    - `NOT_MET`: the normalized top-1 winner's share of valid ticks is still
      >= MONOCULTURE_SHARE_THRESHOLD (0.5) AND it is the SAME target_id that
      won the raw ranking. Normalization did not change who wins; more than
      half of ticks still go to one channel.
    - `NOT_MET_MONOCULTURE_SHIFTED`: the normalized top-1 winner's share is
      still >= MONOCULTURE_SHARE_THRESHOLD (0.5) but it is a DIFFERENT
      target_id than the raw winner. Normalization relocated the monoculture
      to a different single channel rather than diversifying it -- this is
      explicitly NOT treated as success, even though the raw dominant
      channel's own share dropped, because one channel still wins more than
      half of all ticks.
    - `MET`: the normalized top-1 winner's share drops by at least
      DIVERSIFICATION_DROP_THRESHOLD (0.30, absolute) relative to the raw
      top-1 winner's share, AND at least two distinct channels each hold
      >= MEANINGFUL_DIVERSITY_SHARE (0.10) of valid ticks under the
      normalized ranking. Both a real share drop AND real multi-channel
      diversity are required -- a share drop alone could still leave one new
      channel dominant (see MONOCULTURE_SHIFTED above).
    - `AMBIGUOUS`: falls in neither MET nor NOT_MET/NOT_MET_MONOCULTURE_
      SHIFTED bands (e.g. a real but partial share drop that doesn't clear
      the 0.30 threshold, or clears it without producing real multi-channel
      diversity) -- reported plainly rather than forced into a verdict.
    """
    if raw_n < MIN_TOTAL_ROWS or norm_n == 0:
        return "INSUFFICIENT_DATA"

    raw_winner, raw_count = raw_top1.most_common(1)[0] if raw_top1 else (None, 0)
    raw_share = raw_count / raw_n if raw_n else 0.0

    norm_winner, norm_count = norm_top1.most_common(1)[0] if norm_top1 else (None, 0)
    norm_share = norm_count / norm_n if norm_n else 0.0

    n_meaningful_normalized = sum(
        1 for _, c in norm_top1.items() if (c / norm_n) >= MEANINGFUL_DIVERSITY_SHARE
    )

    if norm_share >= MONOCULTURE_SHARE_THRESHOLD:
        if norm_winner == raw_winner:
            return "NOT_MET"
        return "NOT_MET_MONOCULTURE_SHIFTED"

    if (raw_share - norm_share) >= DIVERSIFICATION_DROP_THRESHOLD and n_meaningful_normalized >= 2:
        return "MET"

    return "AMBIGUOUS"


# ===========================================================================
# I/O layer -- psycopg2 read-only. Connection contract mirrors
# measure_emergent_clustering_probe.py / measure_ast_hot_reducer.py exactly
# (refuses a non-read-only session).
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


def fetch_frame_time_range(conn) -> tuple[Optional[datetime], Optional[datetime], int]:
    if conn is None:
        return None, None, 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT min(generated_at), max(generated_at), count(*) FROM substrate_attention_frames")
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames time range", exc_info=True)
        return None, None, 0
    if not row:
        return None, None, 0
    min_ts, max_ts, count = row
    if min_ts is not None and min_ts.tzinfo is None:
        min_ts = min_ts.replace(tzinfo=timezone.utc)
    if max_ts is not None and max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=timezone.utc)
    return min_ts, max_ts, int(count or 0)


def fetch_target_rows(
    conn, start: datetime, end: datetime, max_rows: int = MAX_ROWS
) -> tuple[list[tuple[datetime, Any, Any]], bool]:
    """Real historical rows in [start, end], ASC by generated_at. Each row:
    (generated_at, dominant_targets_raw, capability_targets_raw)."""
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, frame_json->'dominant_targets', frame_json->'capability_targets'
                FROM substrate_attention_frames
                WHERE generated_at >= %s AND generated_at <= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (start, end, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames rows", exc_info=True)
        return [], False
    out: list[tuple[datetime, Any, Any]] = []
    for ts, dom_raw, cap_raw in rows:
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append((ts, dom_raw, cap_raw))
    return out, len(rows) >= max_rows


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


def write_ticks_csv(
    path: Path,
    raw_maps: list[dict[str, float]],
    timestamps: list[datetime],
    target_ids: list[str],
    z_series: dict[str, list[Optional[float]]],
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["generated_at"]
        for t in target_ids:
            header.append(f"raw:{t}")
            header.append(f"z:{t}")
        writer.writerow(header)
        for i, (ts, m) in enumerate(zip(timestamps, raw_maps)):
            row = [ts.isoformat()]
            for t in target_ids:
                row.append(f"{m.get(t, 0.0):.4f}")
                z = z_series[t][i]
                row.append("n/a" if z is None else f"{z:.4f}")
            writer.writerow(row)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(
    *,
    global_window_label: str,
    total_rows: int,
    n_ticks: int,
    target_ids: list[str],
    stats: dict[str, tuple[float, float, bool]],
    raw_top1: Counter,
    norm_top1: Counter,
    norm_valid_n: int,
    n_no_winner: int,
    classification: str,
    caveats: list[str],
) -> str:
    lines = [
        "# Attention Salience Normalization Probe -- Would Z-Scoring Diversify Layer 5's Top-1 Winner?",
        "",
        "Read-only. No writes, no events, no flag/config changes, no edits to "
        "`orion/attention/field_attention/{scoring,selectors}.py` or any other live "
        "code path. Runs a fresh per-channel z-score normalization over real "
        "`substrate_attention_frames` history and asks: does normalizing each "
        "channel against its own historical distribution before ranking change who "
        "wins top-1, and does it actually diversify the winner distribution or just "
        "relocate the monoculture to a different channel? Session context: "
        "2026-07-28 Juniper conversation, same three-places-same-disease finding "
        "documented in `docs/superpowers/specs/2026-07-28-metacog-turn-scoped-"
        "trend-reducer-design.md`'s 2026-07-28 update section.",
        "",
        f"- Real data span (queried fresh this run): {global_window_label}",
        f"- Total `substrate_attention_frames` rows in span: {total_rows}",
        f"- Real ticks used (rows with a parseable target list): {n_ticks}",
        f"- Target universe observed (dominant_targets ∪ capability_targets): {len(target_ids)} "
        f"({', '.join(f'`{t}`' for t in target_ids)})",
        "",
        "## Per-channel full-history stats (the normalization baseline)",
        "",
        "| target_id | mean (raw salience) | stddev | status |",
        "| --- | --- | --- | --- |",
    ]
    for t in target_ids:
        mean, std, degenerate = stats[t]
        status = "DEGENERATE (near-zero variance -- excluded from normalized ranking)" if degenerate else "normal"
        lines.append(f"| `{t}` | {mean:.4f} | {std:.6f} | {status} |")

    degenerate_channels = {t for t, (_, _, d) in stats.items() if d}

    lines.extend(
        [
            "",
            "## Winner-frequency table: raw magnitude ranking (BEFORE normalization)",
            "",
            f"Full-history top-1 winner per tick, same closest-real-analog-to-`dominant_drive` "
            f"definition as `measure_emergent_clustering_probe.py::top1_winner` "
            f"(highest raw `salience_score`, tie-break alphabetical target_id).",
            "",
            "| target_id | ticks won (top-1) | share |",
            "| --- | --- | --- |",
        ]
    )
    for target_id, count in raw_top1.most_common():
        share = count / n_ticks if n_ticks else 0.0
        lines.append(f"| `{target_id}` | {count} | {share * 100:.2f}% |")
    raw_winner, raw_count = raw_top1.most_common(1)[0] if raw_top1 else (None, 0)
    raw_share = raw_count / n_ticks if n_ticks else 0.0

    lines.extend(
        [
            "",
            "## Winner-frequency table: per-channel z-score ranking (AFTER normalization)",
            "",
            f"Per-tick argmax over defined (non-degenerate) z-scores only. "
            f"`{n_no_winner}` of `{n_ticks}` ticks had NO defined z anywhere (every "
            f"channel degenerate at that tick) and are excluded from the "
            f"`{norm_valid_n}`-tick denominator below, not silently folded in.",
            "",
            "| target_id | ticks won (top-1, normalized) | share (of valid ticks) |",
            "| --- | --- | --- |",
        ]
    )
    for target_id, count in norm_top1.most_common():
        share = count / norm_valid_n if norm_valid_n else 0.0
        lines.append(f"| `{target_id}` | {count} | {share * 100:.2f}% |")
    norm_winner, norm_count = norm_top1.most_common(1)[0] if norm_top1 else (None, 0)
    norm_share = norm_count / norm_valid_n if norm_valid_n else 0.0

    lines.extend(
        [
            "",
            "## Honest findings",
            "",
        ]
    )
    if raw_winner is not None and raw_winner in degenerate_channels:
        lines.append(
            f"- **The raw-ranking dominant channel (`{raw_winner}`, {raw_share*100:.2f}% share) "
            f"is itself DEGENERATE (near-zero variance).** Any share drop reported below is a "
            f"mechanical side-effect of that channel being structurally disqualified from the "
            f"normalized ranking (divide-by-~0), NOT evidence that z-score normalization is doing "
            f"real calibration work on real variance. This is exactly the scenario module "
            f"docstring #4 warned to check for explicitly rather than silently produce a "
            f"misleading number."
        )
    else:
        lines.append(
            f"- The raw-ranking dominant channel (`{raw_winner}`, {raw_share*100:.2f}% share) has "
            f"real, non-degenerate variance in its own history -- any share drop below reflects a "
            f"real change in relative ranking under normalization, not a disqualification artifact."
        )

    if norm_winner is not None and norm_winner != raw_winner and norm_share >= MONOCULTURE_SHARE_THRESHOLD:
        lines.append(
            f"- **Monoculture relocated, not fixed.** Under normalization, a DIFFERENT single "
            f"channel (`{norm_winner}`) now wins {norm_share*100:.2f}% of valid ticks -- still "
            f"above the {MONOCULTURE_SHARE_THRESHOLD*100:.0f}% monoculture threshold used by this "
            f"script. Normalization changed WHO wins, not WHETHER one channel structurally "
            f"dominates."
        )
    elif norm_winner is not None and norm_winner == raw_winner and norm_share >= MONOCULTURE_SHARE_THRESHOLD:
        lines.append(
            f"- **Same channel still wins under normalization.** `{norm_winner}` remains the "
            f"top-1 winner at {norm_share*100:.2f}% of valid ticks even after z-scoring."
        )
    elif norm_winner is not None:
        lines.append(
            f"- Under normalization, the top channel (`{norm_winner}`) holds {norm_share*100:.2f}% "
            f"of valid ticks -- below the {MONOCULTURE_SHARE_THRESHOLD*100:.0f}% monoculture "
            f"threshold."
        )

    lines.extend(
        [
            "",
            f"## Classification: **{classification}**",
            "",
        ]
    )
    lines.extend(
        {
            "INSUFFICIENT_DATA": [
                f"Fewer than {MIN_TOTAL_ROWS} real ticks were available, or normalization "
                "produced zero valid (non-degenerate-only) ticks to rank. Too little real "
                "structure to judge either way.",
            ],
            "NOT_MET": [
                "The same channel wins under both raw and normalized ranking, at or above a "
                f"{MONOCULTURE_SHARE_THRESHOLD*100:.0f}% share. Z-score normalization, as "
                "specified here, does NOT diversify Layer 5's top-1 winner selection on real "
                "current data.",
            ],
            "NOT_MET_MONOCULTURE_SHIFTED": [
                "A different channel wins under normalization, but it still holds >= "
                f"{MONOCULTURE_SHARE_THRESHOLD*100:.0f}% of valid ticks. Normalization relocates "
                "the monoculture rather than fixing the underlying disease -- explicitly NOT "
                "treated as success by this script's classification bands.",
            ],
            "MET": [
                "The normalized top-1 winner's share drops by at least "
                f"{DIVERSIFICATION_DROP_THRESHOLD*100:.0f} percentage points relative to the raw "
                "winner's share, AND at least two distinct channels each hold real (>= "
                f"{MEANINGFUL_DIVERSITY_SHARE*100:.0f}%) share under the normalized ranking. Real "
                "evidence that per-channel z-score normalization would diversify who wins.",
            ],
            "AMBIGUOUS": [
                "The share drop and/or resulting diversity fall between the NOT_MET/"
                "NOT_MET_MONOCULTURE_SHIFTED and MET bands -- reported plainly rather than forced "
                "into either verdict.",
            ],
        }.get(classification, ["(no narrative for this classification)"])
    )
    lines.append("")

    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(
        [
            "## Explicit scope boundary",
            "",
            "This script is a measurement only. It did NOT modify "
            "`orion/attention/field_attention/scoring.py`, `selectors.py`, any other live "
            "scoring/selector code path, or any database table. If this measurement's "
            "classification is MET, implementing per-channel normalization in the live scoring "
            "path is a separate, larger patch requiring its own explicit proposal-mode sign-off "
            "per `orion/sentience_striving_program/README.md`'s charter and root CLAUDE.md "
            "§0A's cognition-loop rule -- not a next step taken by this run.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> int:
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

    min_ts, max_ts, total_rows = fetch_frame_time_range(conn)
    progress.emit("time range fetched", percent=5.0, processed=total_rows, total=total_rows)

    if min_ts is None or max_ts is None or total_rows < MIN_TOTAL_ROWS:
        caveats.append(
            f"insufficient real substrate_attention_frames history (rows={total_rows}, "
            f"min_required={MIN_TOTAL_ROWS}); cannot run a meaningful probe"
        )
        try:
            conn.close()
        except Exception:
            pass
        progress.close()
        report = render_report(
            global_window_label="n/a (insufficient data)",
            total_rows=total_rows, n_ticks=0, target_ids=[], stats={},
            raw_top1=Counter(), norm_top1=Counter(), norm_valid_n=0, n_no_winner=0,
            classification="INSUFFICIENT_DATA", caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    global_label = f"{min_ts.isoformat()} -> {max_ts.isoformat()} ({(max_ts - min_ts).total_seconds() / 3600.0:.1f}h)"

    all_rows, truncated = fetch_target_rows(conn, min_ts, max_ts, MAX_ROWS)
    if truncated:
        caveats.append(
            f"history truncated at MAX_ROWS={MAX_ROWS} -- fetch is ORDER BY generated_at ASC, "
            "so truncation keeps the OLDEST rows and drops the newest. Not triggered at current "
            "real data volume as of this run."
        )
    progress.emit("history fetched", percent=25.0, processed=len(all_rows), total=len(all_rows))

    try:
        conn.close()
    except Exception:
        pass

    all_timestamps = [ts for ts, _, _ in all_rows]
    all_raw_maps = [extract_target_salience_map(d, c) for _, d, c in all_rows]
    target_ids = build_target_universe(all_raw_maps)
    n_ticks = len(all_raw_maps)
    progress.emit("target universe built", percent=40.0, processed=n_ticks, total=n_ticks)

    if not target_ids or n_ticks == 0:
        caveats.append("no target_ids observed in real history; cannot compute normalization")
        report = render_report(
            global_window_label=global_label, total_rows=total_rows, n_ticks=n_ticks,
            target_ids=[], stats={}, raw_top1=Counter(), norm_top1=Counter(),
            norm_valid_n=0, n_no_winner=0, classification="INSUFFICIENT_DATA", caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        progress.close()
        print(report)
        return 2

    raw_series = align_series(all_raw_maps, target_ids)
    raw_top1 = top1_winner_distribution_raw(all_raw_maps)
    progress.emit("raw baseline computed", percent=55.0, processed=n_ticks, total=n_ticks)

    z_series, stats = compute_zscore_series(target_ids, raw_series)
    progress.emit("z-scores computed", percent=75.0, processed=n_ticks, total=n_ticks)

    norm_top1, n_no_winner = top1_winner_distribution_normalized(target_ids, z_series, n_ticks)
    norm_valid_n = n_ticks - n_no_winner
    progress.emit("normalized ranking computed", percent=90.0, processed=1, total=1)

    degenerate_channels = {t for t, (_, _, d) in stats.items() if d}
    if degenerate_channels:
        caveats.append(
            f"{len(degenerate_channels)} of {len(target_ids)} channel(s) had near-zero "
            f"full-history variance and were excluded from normalized ranking: "
            f"{sorted(degenerate_channels)}"
        )
    if n_no_winner:
        caveats.append(
            f"{n_no_winner} of {n_ticks} ticks had no defined z-score anywhere (every observed "
            f"channel was degenerate or absent at that tick) and were excluded from the "
            f"normalized-ranking denominator"
        )

    classification = classify_normalization_effect(raw_top1, n_ticks, norm_top1, norm_valid_n)

    write_ticks_csv(CSV_PATH, all_raw_maps, all_timestamps, target_ids, z_series)
    report = render_report(
        global_window_label=global_label, total_rows=total_rows, n_ticks=n_ticks,
        target_ids=target_ids, stats=stats, raw_top1=raw_top1, norm_top1=norm_top1,
        norm_valid_n=norm_valid_n, n_no_winner=n_no_winner, classification=classification,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=1, total=1)
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only probe: would per-channel z-score normalization diversify "
        "Layer 5 attention's top-1-winner selection over real history?"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_arg_parser().parse_args(argv)
    return run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
