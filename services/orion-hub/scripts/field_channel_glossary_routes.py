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
) -> tuple[dict[str, list[float]], int]:
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

    Returns (series, unparsable_row_count) so callers can distinguish "this
    channel is genuinely dead" from "every row in this window failed to
    parse" -- both would otherwise look identical (all channels empty).
    """
    series: dict[str, list[float]] = {ch: [] for ch in (known_channels or [])}
    unparsable = 0
    for payload in rows:
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            state = FieldStateV1.model_validate(payload)
        except Exception:  # noqa: BLE001 - count and skip unparsable historical rows
            unparsable += 1
            continue
        merged, _provenance = collect_field_channel_pressures(state)
        raw_keys: set[str] = set()
        for vector in state.node_vectors.values():
            raw_keys.update(vector.keys())
        for vector in state.capability_vectors.values():
            raw_keys.update(vector.keys())
        seen_channels = set(series.keys()) | set(merged.keys())
        for channel in seen_channels:
            if channel in merged:
                series.setdefault(channel, []).append(float(merged[channel]))
            elif channel in raw_keys:
                series.setdefault(channel, []).append(0.0)
    return series, unparsable


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
    series, unparsable_count = build_channel_series(payloads, known_channels)

    out = []
    for entry in glossary["entries"]:
        values = series.get(entry.channel, [])
        verdict = classify_channel_series(values)
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
