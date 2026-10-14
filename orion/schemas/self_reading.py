"""Self-readings: a body signal Orion can understand, against its own last 7 days.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md section C.

One ``SelfReadingV1`` per internal (body) signal the semantic layer lets reach a prompt
(``config/field/field_channel_glossary.v1.yaml`` entries whose ``prompt_sites`` name
``orion.attention.self_readings:render_self_readings``). The calibration is the attention seam's
own (``orion.attention.world_first.judge_candidate`` over ``PredictionErrorMagnitudeV1``), not a
second copy. ``SelfReadingsBlockV1.text`` is what stance_react / reverie see; it is "" whenever
nothing is unusual or nothing could be read (``skip_reason`` says which). Every number in ``text``
comes from a reading whose ``source_ref`` names the persisted rows it was computed from.

Not on the bus: the rendered text rides inside the existing cortex-exec plan context. Registered
so the typed view has one name across orion-thought and the evals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

SelfReadingState = Literal["live", "no_reading", "stale", "insufficient_history"]
SelfReadingsSkipReason = Literal[
    "disabled", "fetch_failed", "fetch_timeout", "busy", "no_history", "all_calm", "render_failed",
]
SELF_READINGS_MAX_CHARS = 600


class SelfReadingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_reading.v1"] = "self_reading.v1"
    source_id: str  # node:substrate.<name>
    label: str  # plain words
    value_kind: str  # level | count | trigger | binary (placeholder/bucket never get a reading)
    # level/count: the stored reading. trigger/binary: how many times it fired in rate_window_sec.
    value: float | None = None
    units: str = ""
    # Share of the last 7 days strictly below now, in the signal's BAD direction (the attention
    # seam's eligibility percentile). None when there is no reading or too little history.
    percentile_now: float | None = Field(default=None, ge=0.0, le=1.0)
    # Mid-rank percentile used for ranking only (ceiling ties).
    rank_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    band: str = "insufficient_history"  # quiet | usual | high | unusual | insufficient_history | absent
    score_0_10: int | None = Field(default=None, ge=0, le=10)  # round(percentile_now * 10)
    trend: str = "insufficient_history"
    state: SelfReadingState
    unusual: bool = False  # the attention seam's own eligibility verdict: high/unusual, fresh
    reason: str = ""  # the seam's stated reason, verbatim
    observed_at: AwareDatetime | None = None
    age_sec: float | None = None
    n_readings_7d: int = 0
    rate_window_sec: float | None = None  # trigger/binary only
    # Orion's own (LLM-assigned) estimate vs a measurement. Always False for these readings.
    own_estimate: bool = False
    # Persisted rows the numbers came from, e.g.
    # "substrate_node_prediction_error_history:node:substrate.chat@2026-10-14T03:11:18.254835+00:00".
    source_ref: str = ""
    ref: str = ""  # the short ref shown in text, e.g. "pe:chat@2026-10-14T03:11:18Z"
    plain: str = ""  # one deterministic sentence, no LLM


class SelfReadingsBlockV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_readings.block.v1"] = "self_readings.block.v1"
    rendered_at: AwareDatetime
    text: str = Field(default="", max_length=SELF_READINGS_MAX_CHARS)
    readings: list[SelfReadingV1] = Field(default_factory=list)
    shown: list[str] = Field(default_factory=list)  # source_ids rendered as unusual, in order
    lines_dropped: int = 0  # unusual readings left out to fit the count/char caps
    skip_reason: SelfReadingsSkipReason | None = None
