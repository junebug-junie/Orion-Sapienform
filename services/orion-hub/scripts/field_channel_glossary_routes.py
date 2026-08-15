"""Read-only Hub API for the Field Channel Glossary observability panel.

Serves two views over FieldStateV1's 29 raw digester channels
(orion.field.channel_glossary + services/orion-field-digester's
NODE_CHANNELS/CAPABILITY_CHANNELS):

- GET /channels: static glossary metadata (meaning, category, self-state
  dimension) from config/field/field_channel_glossary.v1.yaml.
- GET /health: LIVE per-channel liveness verdict computed from
  substrate_field_state over a rolling window, using the same
  collect_field_channel_pressures() merge every cognition consumer reads
  and the classifier orion.field.channel_glossary already validates against
  scripts/analysis/measure_capability_channel_health.py's thresholds.
  Deliberately NOT the README's frozen prose verdicts -- see that module's
  docstring for why.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from orion.field.channel_glossary import (
    CLEAN_VERDICTS,
    classify_channel_series,
    load_glossary,
)
from orion.field.pressure import collect_field_channel_pressures
from orion.field.regime import channel_regime
from orion.schemas.field_state import FieldStateV1

router = APIRouter(prefix="/api/field-channel-glossary", tags=["field-channel-glossary"])

ALLOWED_HOURS: frozenset[int] = frozenset({1, 6, 24})
DEFAULT_HOURS: int = 1
ROW_CAP: int = 6000

_engine_instance: Any = None


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
        _engine_instance = create_engine(uri, pool_pre_ping=True)
    return _engine_instance


def normalize_hours(hours: int | None) -> int:
    if hours in ALLOWED_HOURS:
        return int(hours)
    return DEFAULT_HOURS


def build_channel_series(
    rows: list[Any], known_channels: list[str] | None = None
) -> tuple[dict[str, list[float]], dict[str, list[Any]], int]:
    """Merged per-tick channel series from raw substrate_field_state rows.

    Pure w.r.t. FieldStateV1 payloads (already-decoded dicts); reuses the
    same collect_field_channel_pressures() merge every field-pressure
    consumer reads, rather than re-deriving polarity/merge logic here.

    collect_field_channel_pressures() only writes a channel into its merged
    output when `channel in PRESSURE_CHANNELS or value > 0`
    (orion.field.pressure) -- for the 5 channels in neither
    PRESSURE_CHANNELS nor HIGHER_IS_BETTER_CHANNELS (expected_offline_
    suppression, stream_backlog_pressure, contract_pressure,
    catalog_drift_pressure, observer_failure_pressure), a tick where every
    source reads exactly 0.0 means the channel is silently absent from the
    merge, even though DEFAULT_NODE_VECTOR/DEFAULT_CAPABILITY_VECTOR
    guarantee it's genuinely present on the raw FieldStateV1 payload. Without
    correcting for this, classify_channel_series() would call a channel that
    is present-and-correctly-quiet-at-zero "never_produced" (implying it was
    never wired up), which is a different and more alarming claim than
    "dead/no current signal". So: when a known channel is missing from the
    merge for a given tick, fall back to checking the raw node_vectors/
    capability_vectors keys directly (a structural presence check, not a
    re-derivation of merge polarity) and record 0.0 if the key is there.
    Only a channel absent from EVERY row's raw vectors for the whole window
    is genuinely never_produced.

    Returns (series, stamps, unparsable_row_count). `stamps` is the producer
    write time behind each merged sample, aligned 1:1 with `series`, so
    `channel_regime` can use its AUTHORITATIVE timestamp path instead of
    inferring liveness from value ratios. Discarding them (as this did until
    2026-08-14) forced ALL 38 channels onto the inference fallback, which is
    blind in the subnormal range where decay stops producing a 0.92 ratio.

    Measured on a live 1749-tick window: 26 of 38 channels move onto the
    timestamp path, and **7 verdicts change, in both directions** --
    `execution_friction` was claiming `producer_written` with no write in the
    window at all, `stream_backlog_pressure` was reported `static` while a
    producer really was writing it, and five more move from `static` (nothing
    changed) to `no_write_in_window` (nobody wrote), which is a different and
    more useful claim. The remaining 12 are capability-sourced or unstamped
    and correctly keep the fallback.

    `unparsable_row_count` lets callers distinguish "this channel is genuinely
    dead" from "every row in this window failed to parse" -- both would
    otherwise look identical (all channels empty).
    """
    series: dict[str, list[float]] = {ch: [] for ch in (known_channels or [])}
    stamps: dict[str, list[Any]] = {ch: [] for ch in (known_channels or [])}
    unparsable = 0
    for payload in rows:
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            state = FieldStateV1.model_validate(payload)
        except Exception:  # noqa: BLE001 - count and skip unparsable historical rows
            unparsable += 1
            continue
        merged, provenance = collect_field_channel_pressures(state)
        raw_keys: set[str] = set()
        for vector in state.node_vectors.values():
            raw_keys.update(vector.keys())
        for vector in state.capability_vectors.values():
            raw_keys.update(vector.keys())
        seen_channels = set(series.keys()) | set(merged.keys())
        for channel in seen_channels:
            if channel in merged:
                series.setdefault(channel, []).append(float(merged[channel]))
                stamps.setdefault(channel, []).append(
                    _winning_write_time(state, provenance.get(channel), channel)
                )
            elif channel in raw_keys:
                series.setdefault(channel, []).append(0.0)
                stamps.setdefault(channel, []).append(None)
    return series, stamps, unparsable


def _winning_write_time(state: FieldStateV1, source_id: str | None, channel: str):
    """The producer write time behind the MERGED value, or None.

    The merged value is one source's reading (R1's provenance dict names
    which), so the only honest timestamp for it is that winner's own
    `node_vector_updated_at` entry -- not the newest stamp across all sources,
    which would credit the merged value with a freshness no contributor had.

    Returns None for a capability winner: `collect_field_channel_pressures`
    resolves capability provenance through `capability_provenance`, so the
    string is an upstream name rather than a node key, and capability vectors
    carry no write timestamps at all. None is the correct answer there, and
    `_refresh_from_timestamps` degrades to "unknown" -> value inference rather
    than inventing one.
    """
    if not source_id:
        return None
    return state.node_vector_updated_at.get(source_id, {}).get(channel)


@router.get("/channels")
async def channels() -> dict[str, Any]:
    glossary = load_glossary()
    return {
        "categories": glossary["categories"],
        "channels": [
            {
                "channel": e.channel,
                "level": list(e.level),
                "category": e.category,
                "meaning": e.meaning,
                "self_state_dimension": e.self_state_dimension,
                "evidence_dimension": e.evidence_dimension,
            }
            for e in glossary["entries"]
        ],
    }


# Fraction of the fetched window used as the regime's OWN window; the earlier
# remainder becomes its baseline.
#
# Without a baseline, ChannelRegime reports level_percentile/dispersion_ratio/
# drift as None -- correct (absent beats a fabricated 0.0) but it throws away
# the relative half of the readout. Splitting gives all axes a real value at
# the cost of a shorter window, and 1/4 of an hour-long fetch is ~15 minutes,
# comfortably above MIN_REGIME_SAMPLES at the live ~2s cadence.
#
# Stated rather than silent because the split IS the window: a consumer
# reading `regime.window_seconds` gets the sliced span, not `hours`.
REGIME_WINDOW_FRACTION: float = 0.25


def _regime_for(channel: str, values: list[float], window_hours: int, stamps=None):
    """Regime readout for one channel over a declared sub-window.

    R2 (PR #1633) exists because `classify_channel_series()` collapses level
    and dispersion into one word: a channel at 0.81 with dispersion 0.0006 and
    one idling at 0.02 both return "quiet". This computes them separately, so
    "busy at near max but steady" is expressible.
    """
    n = len(values)
    split = int(n * (1.0 - REGIME_WINDOW_FRACTION))
    window = values[split:] if n else []
    baseline = values[:split] if split else None
    span = window_hours * 3600.0
    window_seconds = span * (len(window) / n) if n else span
    # Sliced identically to `values`, or refresh_state would be computed from a
    # different span than level/dispersion.
    updated_at = (stamps[split:] if stamps and len(stamps) == n else None)
    return channel_regime(
        channel,
        window,
        window_seconds=window_seconds,
        baseline=baseline,
        updated_at=updated_at,
    )


@router.get("/health")
async def health(hours: int = Query(DEFAULT_HOURS)) -> dict[str, Any]:
    window_hours = normalize_hours(hours)
    with _engine().connect() as conn:
        # DESC + reverse (not ASC + LIMIT) so that when the window's row
        # count exceeds ROW_CAP, the retained rows are the newest ticks, not
        # the oldest -- an ASC LIMIT would silently classify the panel from
        # the stalest slice of exactly the windows large enough to truncate
        # (hours=6/24 at ~1.8k rows/hour), the same staleness failure mode
        # this feature exists to replace.
        result = conn.execute(
            text(
                """
                SELECT field_json
                FROM substrate_field_state
                WHERE generated_at >= NOW() - (:hours * INTERVAL '1 hour')
                ORDER BY generated_at DESC
                LIMIT :row_cap
                """
            ),
            {"hours": window_hours, "row_cap": ROW_CAP},
        )
        payloads = [row[0] for row in result]
    payloads.reverse()

    row_count = len(payloads)
    glossary = load_glossary()
    known_channels = [e.channel for e in glossary["entries"]]
    series, stamps, unparsable_count = build_channel_series(payloads, known_channels)

    out = []
    for entry in glossary["entries"]:
        values = series.get(entry.channel, [])
        verdict = classify_channel_series(values)
        regime = _regime_for(entry.channel, values, window_hours, stamps.get(entry.channel))
        out.append(
            {
                "channel": entry.channel,
                "level": list(entry.level),
                "category": entry.category,
                "verdict": verdict,
                "clean": verdict in CLEAN_VERDICTS,
                "sample_count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "last": values[-1] if values else None,
                # R2 (PR #1633): the axes the single-word verdict collapses.
                # `verdict` stays authoritative for the panel's clean/dirty
                # colouring; this sits beside it, it does not replace it.
                "regime": {
                    "label": regime.regime,
                    "window_seconds": regime.window_seconds,
                    "level": regime.level,
                    "pressure_equivalent_level": regime.pressure_equivalent_level,
                    "polarity_inverted": regime.polarity_inverted,
                    "dispersion": regime.dispersion,
                    "level_percentile": regime.level_percentile,
                    "dispersion_ratio": regime.dispersion_ratio,
                    "drift": regime.drift,
                    "saturation_low": regime.saturation_low,
                    "saturation_high": regime.saturation_high,
                    "refresh_state": regime.refresh_state,
                    "refresh_evidence": regime.refresh_evidence,
                },
            }
        )

    return {
        "window_hours": window_hours,
        "row_count": row_count,
        "row_cap": ROW_CAP,
        "truncated": row_count >= ROW_CAP,
        "unparsable_count": unparsable_count,
        "channels": out,
    }
