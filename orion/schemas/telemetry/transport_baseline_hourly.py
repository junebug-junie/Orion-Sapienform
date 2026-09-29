"""Hourly per-hop summary of the transport baseline gate's readings.

Why this exists: the gate (``orion/metacog/transport_baseline.py`` via
orion-equilibrium-service ``app/transport_baseline_gate.py``) wrote its per-window
readings only as ``transport_baseline_obs`` container log lines, which every
restart erases -- so the spec's log-only week (acceptance check 1 of
``docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-
design.md``) could not be graded. One row per (service, instance, hop, hour) is
~64 keys x 24 = ~1,500 rows/day instead of ~180,000 per-window lines.

Producer: orion-equilibrium-service, on ``orion:equilibrium:transport_baseline:hourly``,
at the end of each hour and best-effort on shutdown (``flush_reason``).
Consumer: orion-sql-writer -> ``transport_baseline_hourly``; read by
``scripts/analysis/grade_transport_baseline.py``.

A restart mid-hour produces two rows for that hour (``flush_reason="shutdown"``
then ``"hour_end"``). Counts add; percentiles are per row and a reader combines
them weighted by ``windows_evaluated`` (an approximation, stated here, not hidden).

Hour boundaries use the snapshot's own ``window_end`` (the producer's clock, same
clock the reducer runs on), in UTC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

TRANSPORT_BASELINE_HOURLY_KIND = "transport_baseline.hourly.v1"
TRANSPORT_BASELINE_HOURLY_CHANNEL = "orion:equilibrium:transport_baseline:hourly"


class TransportBaselineHourlyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_id: str
    service: str
    instance: Optional[str] = None
    key: str = Field(..., description="Hop key, e.g. a bus request channel, '<channel>#<label>', 'verb:<name>'.")
    hour_start: datetime
    flush_reason: Literal["hour_end", "shutdown"]
    flushed_at: datetime

    windows_seen: int = Field(..., ge=0, description="Snapshots folded for this key in the hour.")
    windows_evaluated: int = Field(..., ge=0, description="Folds that produced a latency evaluation (>= min_calls pooled).")
    success_count: int = Field(0, ge=0)
    timeout_count: int = Field(0, ge=0)

    z_p50: Optional[float] = Field(None, description="Median z over evaluated windows; None if none evaluated.")
    z_p90: Optional[float] = None
    saturation_ratio_p50: Optional[float] = None
    baseline_ms: Optional[float] = Field(None, description="exp(fast mean) at the hour's last fold.")
    floor_ms_start: Optional[float] = Field(None, description="Floor at the hour's first fold.")
    floor_ms: Optional[float] = Field(None, description="Floor at the hour's last fold.")
    calls_per_min_mean: float = 0.0

    conditions_opened: Dict[str, int] = Field(
        default_factory=dict,
        description="condition -> episodes opened this hour, excluded keys included (regime_shift is an open).",
    )
    open_at_hour_end: List[str] = Field(default_factory=list)
    would_emit_by_condition: Dict[str, int] = Field(
        default_factory=dict,
        description="'condition:phase' -> triggers the gate would publish with EMIT on (excluded keys never count).",
    )

    excluded: bool = False
    warm: bool = False
    emit_effective: bool = False
    config_fingerprint: str
