from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetacogTriggerV1(BaseModel):
    trigger_kind: str = Field(
        ...,
        description=(
            "baseline | dense | manual | pulse | relational | llm_surface_instability "
            "(advisory: language-surface instability from logprob summary, not factual confidence) "
            "| telemetry_anomaly | chat_turn | transport | insight | flow | repair_pressure_trend. "
            "relational = a live repair_pressure_v2 appraisal (see repair_pressure_metacog_gate.py). "
            "repair_pressure_trend = a sustained (3+ consecutive, EWMA-z-scored) elevated run across "
            "multiple repair_pressure_v2 appraisals over time -- 'has this kept happening', not one "
            "turn's reading (see repair_pressure_trend_gate.py, orion/metacog/trend_reducer.py). "
            "Distinct from relational: relational fires on a single appraisal crossing an absolute "
            "level/confidence floor; repair_pressure_trend fires on a real windowed baseline being "
            "exceeded repeatedly, and can fire even when no single appraisal alone would trip "
            "relational. Cold-start guarded (needs a configured minimum sample count before it can "
            "fire at all). Hop 0 of the stream-of-consciousness hop-chain design "
            "(docs/superpowers/specs/2026-07-29-stream-of-consciousness-hop-chain-design.md). "
            "telemetry_anomaly = a field_channel_corpus.v1 reconstruction-loss anomaly from a trained "
            "orion/mood_arc/fit_encoder.py encoder (see telemetry_anomaly_metacog_gate.py). "
            "chat_turn = a completed (or timed-out) chat turn whose correlated ThoughtEventV1 + "
            "HarnessRunV1 (or a governor/stance-react timeout GrammarEventV1) evidence trips a "
            "gate condition (see chat_turn_metacog_gate.py). "
            "transport = a real OrionBusAsync.rpc_request() timeout or degraded RpcHealthSnapshotV1 "
            "window trips a gate condition (see transport_metacog_gate.py). "
            "insight and flow are the two generative (non-rupture) kinds -- unlike every kind above "
            "them they fire on a positive/neutral state, not a failure. Both read one field, "
            "AttentionSelfModelV1.prediction_error_confidence (persisted to "
            "substrate_attention_self_model), as two different windowing functions: "
            "insight = a sustained low->high confidence recovery, i.e. a surprise got resolved "
            "(see insight_metacog_gate.py); flow = a sustained high-confidence, low-variance "
            "plateau (see flow_metacog_gate.py). These two are also the only kinds that map "
            "directly to a CollapseMirrorEntryV2.type (epiphany / flow respectively, in "
            "orion-cortex-exec's _fallback_metacog_draft); every other kind's type is guessed "
            "from phi bands."
        ),
    )
    reason: str
    zen_state: str = Field("unknown", description="zen | not_zen | unknown")
    pressure: float = Field(0.0, ge=0.0, le=1.0)
    window_sec: int = 15
    frame_refs: List[str] = Field(default_factory=list)
    signal_refs: List[str] = Field(default_factory=list)
    upstream: Dict[str, Any] = Field(default_factory=dict, description="Compact summary of upstream trigger event")
    recall_enabled: Optional[bool] = Field(
        default=None,
        description="Override for downstream recall usage (true/false), None to use defaults.",
    )
    timestamp: str = Field(default_factory=_utc_now_iso)
