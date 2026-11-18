"""Self-signal -> outcome association: "when this is high, my next hour tends to ...".

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md section E ("Phase 2: what
it tends to lead to"). Reducer: ``orion/attention/self_signal_outcomes.py`` (pure, no LLM).

Two durable row kinds, both produced by orion-equilibrium-service and persisted by orion-sql-writer:

- ``SelfSignalHourV1`` (``self_signal_hour``): one row per completed UTC hour. For every internal
  ``node:substrate.*`` body signal, whether that hour was ``high`` (high/unusual for itself, by the
  attention seam's own rule, for more than 300 s within 600 s at some point in the hour: the
  self_signal_episode gate's open dwell), ``normal``, or ``unknown`` (no
  readable reading: never counted as normal). Plus that hour's outcome counts (failed harness
  turns, repaired replies, RPC timeouts) with their exposure. This row is what lets the
  association outlive ``substrate_node_prediction_error_history``'s 7-day prune.
- ``SelfSignalOutcomeAssociationV1`` (``self_signal_outcome_association``): one row per
  (signal, outcome) per refresh: next-hour outcome rate when the signal was high vs normal, the
  rate ratio and its 95% interval, and a verdict. Prompts cite ``association_id``.

Not a registry of concepts: every field is consumed by the self-reading line
(``orion.attention.self_readings``), the episode open row's evidence
(``orion.metacog.evidence_map.map_self_signal_episode``) or the truth-to-rows eval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SELF_SIGNAL_HOUR_KIND = "self_signal.hour.v1"
SELF_SIGNAL_HOUR_CHANNEL = "orion:equilibrium:self_signal:hour"
SELF_SIGNAL_ASSOCIATION_KIND = "self_signal.outcome_association.v1"
SELF_SIGNAL_ASSOCIATION_CHANNEL = "orion:equilibrium:self_signal:outcome_association"

SignalHourCondition = Literal["high", "normal", "unknown"]
OutcomeKind = Literal["turn_failed", "response_repair", "rpc_timeout"]
AssociationVerdict = Literal[
    "tends_more",  # 95% interval of the rate ratio entirely above 1
    "tends_less",  # entirely below 1
    "no_clear_difference",  # interval contains 1
    "below_min_n",  # too few high or normal hours to say anything
    "excluded_circular",  # the signal is computed from (or shares an upstream with) the outcome
]

# Caps (Juniper's standing rule: cap collections).
MAX_SIGNALS_PER_HOUR = 32


class SignalHourV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str
    condition: SignalHourCondition
    n_readings: int = Field(0, ge=0)  # rows of the signal observed in the hour
    n_judged: int = Field(0, ge=0)  # level/count: 1 when any judged reading spoke for the hour; trigger: rows
    n_hot: int = Field(0, ge=0)  # readings in the hour eligible as high/unusual (trigger: firings)
    # Level/count only, from the replayed 30 s polls: seconds of the hour a judged reading spoke
    # for, and how many of those it was high/unusual for itself. high <=> the episode gate's open
    # dwell (> 300 s hot within 600 s) was met at some poll in the hour.
    judged_sec: float = Field(0.0, ge=0.0)
    hot_sec: float = Field(0.0, ge=0.0)


class OutcomeHourV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: OutcomeKind
    exposure: int = Field(0, ge=0)  # turns run / RPC calls made in the hour
    events: int = Field(0, ge=0)  # failed turns / repaired replies / timed-out calls
    source: str = ""  # the table + predicate the counts came from


class SelfSignalHourV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_signal.hour.v1"] = "self_signal.hour.v1"
    hour_id: str  # "ssh:<hour_start ISO Z>": primary key, a recompute upserts the same row
    hour_start: datetime
    computed_at: datetime
    signals: list[SignalHourV1] = Field(default_factory=list, max_length=MAX_SIGNALS_PER_HOUR)
    outcomes: list[OutcomeHourV1] = Field(default_factory=list, max_length=8)
    # Hours of PE history before hour_start the percentiles had (< 168 = a shorter week than
    # live, e.g. a backfilled hour near the prune horizon).
    history_hours: float = Field(0.0, ge=0.0)


class SelfSignalOutcomeAssociationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_signal.outcome_association.v1"] = "self_signal.outcome_association.v1"
    association_id: str
    computed_at: datetime
    signal_id: str
    outcome: OutcomeKind
    # Hours (SelfSignalHourV1.hour_start) the counts were drawn from, inclusive.
    window_start: datetime | None = None
    window_end: datetime | None = None
    # Hour H (signal high / normal) paired with hour H+1's outcome; only pairs where H+1 had
    # exposure (a turn, an RPC call) count. "events" = next hours with >= 1 outcome event.
    n_high: int = Field(0, ge=0)
    events_high: int = Field(0, ge=0)
    n_normal: int = Field(0, ge=0)
    events_normal: int = Field(0, ge=0)
    rate_high: float | None = None
    rate_normal: float | None = None
    ratio: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # Mantel-Haenszel over next-hour activity strata, SE x se_inflation (serial correlation), at
    # the Bonferroni z for n_comparisons simultaneous pairs (family-wise 5%).
    ci_method: str = "mantel_haenszel_log_rr_bonferroni_ar1"
    ci_z: float = Field(1.959964, gt=0.0)
    se_inflation: float = Field(1.0, ge=1.0)
    n_comparisons: int = Field(1, ge=1)
    n_strata: int = Field(0, ge=0)
    min_hours: int = Field(0, ge=0)
    min_history_hours: float = Field(0.0, ge=0.0)
    verdict: AssociationVerdict
    excluded_reason: str = ""
    # The one deterministic clause a prompt may show ("" unless verdict is tends_more/less).
    clause: str = ""
