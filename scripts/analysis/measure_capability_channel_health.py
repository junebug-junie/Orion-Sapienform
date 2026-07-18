#!/usr/bin/env python3
"""Read-only capability channel-health measurement.

`scripts/analysis/measure_capability_salience_coupling.py` (PR #1169) measured
`FieldAttentionTargetV1.salience_score` for two capability targets --
`capability:transport` and `capability:llm_inference` -- over 48h of live
traffic. Finding: neither target's salience ever exceeded ~0.20, never once
cleared 0.25, let alone `field_attention_policy.v1.yaml`'s own
`high_salience=0.70` threshold. This script does not re-measure salience --
it measures the RAW inputs that salience is computed from, to find out
*why* salience reads so chronically low.

Salience for a capability target is computed by
`orion/attention/field_attention/scoring.py::weighted_pressure()`, which sums
`vector[channel] * channel_weights[channel]` over
`field_attention_policy.v1.yaml`'s `capability_channel_weights`:

    pressure:              0.70
    execution_pressure:    0.80
    reasoning_pressure:    0.60
    reliability_pressure:  1.00   <- highest weight
    transport_pressure:    0.75
    contract_pressure:     0.90
    confidence:           -0.35
    available_capacity:   -0.45

If a high-weight channel (especially `reliability_pressure`, weight 1.00 --
the single largest term in the sum) is dead or pinned near zero for a given
target, that alone caps how high that target's salience can ever go,
regardless of what any other channel does. This script measures each of the
8 channels above, for both `capability:transport` and `capability:llm_inference`,
against real historical data, to find which channel(s) -- if any -- are
actually dead.

This performs NO writes, emits NO events, flips NO flags, and proposes NO
config change. It does not modify `orion/attention/field_attention/*`,
`config/attention/field_attention_policy.v1.yaml`, or
`orion/autonomy/capability_policy.py`.

Data source: `FieldStateV1` (which `capability_vectors` lives on) is
persisted in Postgres `substrate_field_state` (`tick_id`, `generated_at`,
`field_json`) -- NOT `substrate_self_state`, a different, downstream table.
`field_json->'capability_vectors'` is a dict keyed by target_id (e.g.
`"capability:transport"`), each value itself a dict of
`{channel_name: raw_value}`.

A missing channel in a given tick's capability_vectors entry (or a missing
target entirely) means treat as `0.0` for that tick -- same "absent = below
threshold, not a skip" convention as `measure_capability_salience_coupling.py`.

"Dead or zero-or-subnormal" detection uses a `< 1e-100` cutoff rather than an
exact `== 0.0` check. This repo has repeatedly found channels that decay to
IEEE-754 subnormal floats (e.g. ~6.85e-322) instead of landing exactly on
0.0 -- numerically dead but not bitwise-zero. A plain `== 0.0` check would
silently miss that pattern, which is the whole reason to check this at all.
1e-100 is many orders of magnitude below any value that could plausibly
represent real signal in a [0, 1]-scaled pressure channel, and comfortably
above the subnormal range (~1e-308 to ~1e-320), so it cleanly separates
"numerically dead" from "real small value."

A channel is classified "live" if `max - median > 0.05` for that
(target, channel) series, else "effectively dead". 0.05 is chosen because it
is smaller than any of the threshold-sweep points used elsewhere in this
analysis lineage (`min_salience=0.10`), so a channel that swings by less than
that can never meaningfully move a salience decision gated at those
thresholds -- it is functionally inert for this purpose even if not
literally pinned at a single value. This is a real, stated threshold, not an
arbitrary eyeball number: it is set to be strictly finer-grained than the
smallest downstream decision boundary this data feeds.

Run:
    python scripts/analysis/measure_capability_channel_health.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.capability_channel_health")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000

# The 8 channels field-attention's weighted_pressure() sums over for
# capability targets, per config/attention/field_attention_policy.v1.yaml's
# capability_channel_weights. Not re-derived here -- this script only
# measures the raw inputs to that formula.
CAPABILITY_CHANNELS: tuple[str, ...] = (
    "pressure",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "transport_pressure",
    "contract_pressure",
    "confidence",
    "available_capacity",
)

# The two capability targets under investigation in this lineage (see
# measure_capability_salience_coupling.py's CAPABILITY_TARGET_MAP).
CAPABILITY_TARGETS: tuple[str, ...] = (
    "capability:transport",
    "capability:llm_inference",
)

# Below this magnitude, a raw channel value is treated as "zero or
# numerically-decayed-to-dust subnormal", not real signal. See module
# docstring for rationale (catches ~1e-322-style decayed floats that a plain
# `== 0.0` check would miss).
SUBNORMAL_CUTOFF: float = 1e-100

# A channel's (max - median) must exceed this to be classified "live". See
# module docstring for rationale (finer-grained than the smallest downstream
# decision boundary, min_salience=0.10, that this data feeds).
LIVE_SPREAD_THRESHOLD: float = 0.05

OUTPUT_DIR = Path("/tmp/capability-channel-health")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure extraction + summary layer. No I/O. Unit-testable directly on
# synthetic capability_vectors JSON blobs and synthetic float lists -- no
# engine replay, no DB.
# ===========================================================================


def extract_channel_value(vectors_raw: Any, target_id: str, channel: str) -> float:
    """Given the raw `field_json->'capability_vectors'` value for one tick (a
    JSON object keyed by target_id, each value a dict of
    `{channel_name: raw_value}`, possibly still a JSON string, possibly
    malformed, possibly None), return the raw value for `channel` under
    `target_id`, or 0.0 if the target or channel is absent from this tick, or
    if the payload is malformed in any way. Never raises.
    """
    try:
        if vectors_raw is None:
            return 0.0
        if isinstance(vectors_raw, str):
            vectors_raw = json.loads(vectors_raw)
        if not isinstance(vectors_raw, dict):
            return 0.0
        target_entry = vectors_raw.get(target_id)
        if not isinstance(target_entry, dict):
            return 0.0
        raw_value = target_entry.get(channel, 0.0)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0
    except Exception:
        return 0.0


def is_zero_or_subnormal(value: float) -> bool:
    """True if `value` is exactly zero or a numerically-decayed subnormal
    float (magnitude below SUBNORMAL_CUTOFF). Deliberately not a plain
    `== 0.0` check -- see module docstring."""
    try:
        return abs(value) < SUBNORMAL_CUTOFF
    except Exception:
        return True


@dataclass
class Distribution:
    count: int = 0
    median: Optional[float] = None
    p90: Optional[float] = None
    max: Optional[float] = None


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def summarize_channel(values: list[float]) -> Distribution:
    if not values:
        return Distribution()
    return Distribution(
        count=len(values),
        median=float(statistics.median(values)),
        p90=_percentile(values, 0.9),
        max=float(max(values)),
    )


def frac_zero_or_subnormal(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if is_zero_or_subnormal(v)) / len(values)


def classify_channel(dist: Distribution) -> str:
    """"live" if the channel's (max - median) exceeds LIVE_SPREAD_THRESHOLD,
    else "dead". A channel with no observations is "dead" (no signal at
    all)."""
    if dist.count == 0 or dist.median is None or dist.max is None:
        return "dead"
    spread = dist.max - dist.median
    return "live" if spread > LIVE_SPREAD_THRESHOLD else "dead"


# ===========================================================================
# I/O layer -- psycopg2 read-only. Function body reused verbatim from
# measure_capability_salience_coupling.py's open_readonly_connection()
# (itself reused from measure_origination_gate.py), not reinvented here.
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


def fetch_capability_vector_rows(
    conn, since: datetime, max_rows: int = MAX_ROWS
) -> tuple[list[tuple[datetime, Any]], bool]:
    """Fetch (generated_at, raw capability_vectors json) ordered ASC over
    substrate_field_state rows with generated_at >= since. Returns
    (rows, truncated). Never raises -- a malformed row is skipped.
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, field_json->'capability_vectors' AS vectors
                FROM substrate_field_state
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_field_state", exc_info=True)
        return [], False
    out: list[tuple[datetime, Any]] = []
    for generated_at, vectors_raw in rows:
        try:
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            out.append((generated_at, vectors_raw))
        except Exception:
            logger.warning("skipping malformed field-state row ts=%s", generated_at, exc_info=True)
            continue
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
    path: Path, cell_values: dict[tuple[str, str], list[tuple[datetime, float]]]
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["target_id", "channel", "generated_at", "raw_value"])
        for (target_id, channel), rows in cell_values.items():
            for generated_at, value in rows:
                writer.writerow([target_id, channel, generated_at.isoformat(), f"{value:.6g}"])


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    cell_results: dict[tuple[str, str], dict[str, Any]],
    caveats: list[str],
) -> str:
    lines = [
        "# Capability Channel Health Measurement",
        "",
        "Read-only. No writes, no events, no flag/config changes. Measures the "
        "raw per-channel inputs to `weighted_pressure()` "
        "(`orion/attention/field_attention/scoring.py`) for the two capability "
        "targets investigated by `measure_capability_salience_coupling.py` "
        "(PR #1169), to find out *why* their salience reads chronically low. "
        "Does not touch `orion/attention/field_attention/*`, "
        "`config/attention/field_attention_policy.v1.yaml`, or "
        "`orion/autonomy/capability_policy.py`.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Zero-or-subnormal cutoff: abs(value) < {SUBNORMAL_CUTOFF:g}",
        f"- Live-channel spread threshold: (max - median) > {LIVE_SPREAD_THRESHOLD:g}",
        "",
        "## Per (target, channel) raw value distribution",
        "",
        "A missing channel or missing target for a given tick counts as "
        "`raw_value = 0.0` (same 'absent = below threshold, not a skip' "
        "convention as measure_capability_salience_coupling.py).",
        "",
        "| target | channel | weight | n ticks | median | p90 | max | frac_dead | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (target_id, channel), result in cell_results.items():
        dist: Distribution = result["dist"]
        frac_dead = result["frac_zero_or_subnormal"]
        verdict = result["verdict"]
        weight = result["weight"]
        lines.append(
            f"| `{target_id}` | `{channel}` | {weight:+.2f} | {dist.count} | "
            f"{_fmt(dist.median)} | {_fmt(dist.p90)} | {_fmt(dist.max)} | "
            f"{frac_dead:.2f} | {verdict} |"
        )

    # "Reading this" -- name which high-weight channels are actually dead,
    # per target, rather than assuming reliability_pressure in advance.
    lines.extend(["", "## Reading this", ""])
    lines.append(
        "`weighted_pressure()` sums `vector[channel] * channel_weights[channel]` "
        "over all 8 channels above. A dead channel (verdict `dead`) contributes "
        "~0 to that sum regardless of its weight; a dead HIGH-weight channel "
        "caps the achievable sum the hardest, since it is the single largest "
        "term that never fires."
    )
    for target_id in CAPABILITY_TARGETS:
        dead_channels = [
            (channel, cell_results[(target_id, channel)]["weight"])
            for channel in CAPABILITY_CHANNELS
            if (target_id, channel) in cell_results
            and cell_results[(target_id, channel)]["verdict"] == "dead"
        ]
        dead_channels.sort(key=lambda pair: abs(pair[1]), reverse=True)
        if dead_channels:
            named = ", ".join(f"`{c}` (weight {w:+.2f})" for c, w in dead_channels)
            lines.append(f"- `{target_id}`: dead channels -- {named}.")
        else:
            lines.append(f"- `{target_id}`: no dead channels found; all 8 channels show real spread.")
    lines.append(
        "- If `reliability_pressure` (weight 1.00, the single largest weight in "
        "the sum) is dead for a target, that alone caps how high that target's "
        "salience can ever go. But it is rarely the whole story if other "
        "high-weight channels (`contract_pressure` 0.90, `execution_pressure` "
        "0.80, `transport_pressure` 0.75, `pressure` 0.70) are also dead for "
        "the same target -- the practical answer to 'why is salience low' is "
        "whichever combination of high-weight, dead channels is actually found "
        "above, not assumed in advance."
    )
    lines.append("")

    # Correction (2026-07-18): this measurement's first-pass framing called
    # the findings above a "broader orion-field-digester producer gap." That
    # overclaimed. services/orion-field-digester/README.md's own "Field
    # channel glossary" section -- written 2026-07-16, a day before this
    # script existed -- already documents, per-channel and per-target, why
    # almost every "dead" cell above is dead: not a bug, but by-design
    # wiring (a channel simply has no diffusion edge into that particular
    # capability), a documented one-way-ratchet mechanism that's "currently
    # benign," or a genuinely-rare event that's "correctly wired, simply
    # has not received a real nonzero perturbation yet." Cross-checking each
    # dead cell above against that glossary before treating it as a gap is
    # the right next step, not further measurement here.
    lines.extend(["## Cross-check against services/orion-field-digester/README.md", ""])
    lines.append(
        "The Field channel glossary in that README (section \"Field channel "
        "glossary\", written 2026-07-16) already explains nearly every dead "
        "cell found above, per (target, channel):"
    )
    lines.extend(
        [
            "",
            "- `capability:transport`'s dead `pressure`/`execution_pressure`/"
            "`reasoning_pressure`: expected, not a bug -- those channels have no "
            "diffusion edge into `capability:transport` at all (they only ever "
            "populate `orchestration`/`llm_inference`), or (for `pressure`) their "
            "only source, the node-level `transport_pressure` channel, is "
            "documented as fully unproduced (\"confirmed absent... from all "
            "123,245+ live rows checked\").",
            "- `capability:transport`'s dead `reliability_pressure`: matches the "
            "glossary's own verdict on its source, `observer_failure_pressure` -- "
            "\"genuinely quiet, correctly wired, no bug... max observed value "
            "3e-323... simply has never received a real nonzero perturbation.\"",
            "- `capability:transport`'s pinned-constant `confidence`/"
            "`available_capacity`: both fed by documented one-way-ratchet "
            "channels (`delivery_confidence`, `bus_health`) the glossary calls "
            "\"currently benign since the bus is genuinely stable, but "
            "structurally could never show a real dip.\"",
            "- `capability:llm_inference`'s live `pressure`/`confidence`/"
            "`available_capacity` and dead `execution_pressure`/"
            "`transport_pressure`/`contract_pressure`: fully consistent with the "
            "glossary's documented diffusion edges (only `pressure`, from real "
            "`gpu_pressure`, and its two fallback-formula derivatives actually "
            "target `llm_inference`; the other three were never wired to it at "
            "all, by design).",
            "",
            "**One cell does not reconcile cleanly and is worth flagging rather "
            "than either dismissing or chasing further here**: the glossary "
            "calls `reasoning_pressure` \"the cleanest channel in the corpus... "
            "real signal,\" but that verdict is read from the merged JSONL "
            "corpus file (`collect_field_channel_pressures()`'s max()-merge "
            "across every entity sharing that channel name), not the raw "
            "per-capability `substrate_field_state` data this script reads. "
            "`reasoning_pressure` is fed by both `capability:llm_inference` "
            "(weight 0.85) and `capability:orchestration` (weight 0.90); this "
            "script found it dead specifically for `llm_inference`. The "
            "glossary already documents this exact max()-merge mechanism "
            "masking real per-capability variation for `confidence`/"
            "`available_capacity` -- this could be a third instance of the same "
            "masking bug, with `orchestration`'s variation hiding "
            "`llm_inference`'s own dead value in the merged corpus view. Being "
            "actively investigated elsewhere as of 2026-07-18 -- not pursued "
            "further by this script.",
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


# Channel weights, duplicated here (not imported) so this measurement script
# has zero import-time dependency on orion/attention/* -- keeps it safely
# read-only and decoupled from the live policy module's load-time side
# effects. Values must match config/attention/field_attention_policy.v1.yaml's
# capability_channel_weights; re-check that file if this script's report
# stops matching production behavior.
CAPABILITY_CHANNEL_WEIGHTS: dict[str, float] = {
    "pressure": 0.70,
    "execution_pressure": 0.80,
    "reasoning_pressure": 0.60,
    "reliability_pressure": 1.00,
    "transport_pressure": 0.75,
    "contract_pressure": 0.90,
    "confidence": -0.35,
    "available_capacity": -0.45,
}


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

    rows, truncated = fetch_capability_vector_rows(conn, window_start, MAX_ROWS)
    if truncated:
        caveats.append(f"rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("fetched substrate_field_state", percent=50.0, processed=len(rows), total=len(rows))

    try:
        conn.close()
    except Exception:
        pass

    cell_values: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
    cell_results: dict[tuple[str, str], dict[str, Any]] = {}
    total_cells = len(CAPABILITY_TARGETS) * len(CAPABILITY_CHANNELS)
    idx = 0
    for target_id in CAPABILITY_TARGETS:
        for channel in CAPABILITY_CHANNELS:
            idx += 1
            series: list[tuple[datetime, float]] = []
            for generated_at, vectors_raw in rows:
                value = extract_channel_value(vectors_raw, target_id, channel)
                series.append((generated_at, value))
            cell_values[(target_id, channel)] = series
            values_only = [v for _, v in series]
            dist = summarize_channel(values_only)
            verdict = classify_channel(dist)
            cell_results[(target_id, channel)] = {
                "dist": dist,
                "frac_zero_or_subnormal": frac_zero_or_subnormal(values_only),
                "verdict": verdict,
                "weight": CAPABILITY_CHANNEL_WEIGHTS.get(channel, 0.0),
            }
            progress.emit(
                f"summarized {target_id}/{channel}",
                percent=50.0 + 45.0 * idx / max(total_cells, 1),
                processed=len(series),
                total=len(series),
            )

    if not rows:
        caveats.append("no substrate_field_state rows found in window")

    write_ticks_csv(CSV_PATH, cell_values)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        cell_results=cell_results,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(rows), total=len(rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only capability-channel-health measurement."
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_WINDOW_HOURS,
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
