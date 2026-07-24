from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.attention_frame import VoluntaryOverrideV1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


AttentionReasonV1 = Literal[
    "top_down_override",
    "bottom_up_salience",
    "field_salience_only",
    "no_data",
]


class AttentionSelfModelV1(BaseModel):
    """AST/HOT instrumentation — a single inspectable model *of* Orion's own
    current attention (Attention Schema Theory / Higher-Order Theory,
    `orion/sentience_striving_program/README.md` sec 9b items 2 and 4).

    Unifies the two real, previously-disconnected attention self-models found
    live during the 2026-07-18 Objective 3 roadmap scoping pass
    (`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`,
    Phase 1):

    - The GWT-dispatch lane: `AttentionBroadcastProjectionV1`
      (`orion/schemas/attention_frame.py`), which wraps `AttentionFrameV1` and
      carries Lamme's `voluntary_override` (top-down goal bias flipping the
      workspace-competition winner — `orion/substrate/attention/top_down.py`).
      Persisted as a *singleton* upsert row in
      `substrate_attention_broadcast_projection` (confirmed live 2026-07-18:
      PK on `projection_id`, exactly one row at any time) — there is no
      per-tick history for this lane, only the most recent snapshot.
    - The general field lane (Layer 5/6): `FieldAttentionFrameV1`, updated on
      a continuous ~2s tick by `orion-attention-runtime`, with real per-tick
      history in `substrate_attention_frames`. **2026-07-23: this lane's
      `SelfStateV1` co-input was removed** (that producer no longer exists,
      PR #1266) and replaced with the five real Active-Inference domains'
      `prediction_error` signal (`orion/substrate/prediction_error.py`) —
      see `orion/substrate/attention_self_model.py`'s module docstring for
      the full account.

    This is a read-only measurement artifact (Phase 1 scope). It is not
    published to any bus channel and not consumed by any live decision path —
    that wiring is explicitly future work (Phase 3+ of the roadmap doc).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.self_model.v1"] = "attention.self_model.v1"
    generated_at: datetime = Field(default_factory=_utc_now)

    # --- What's currently salient (field lane) -----------------------------
    field_lane_present: bool = False
    field_overall_salience: float | None = Field(default=None, ge=0.0, le=1.0)
    field_salient_target_ids: list[str] = Field(default_factory=list)

    # --- What was last dispatched (GWT-dispatch / broadcast lane) ----------
    broadcast_lane_present: bool = False
    broadcast_selected_action_type: str | None = None
    broadcast_selected_open_loop_id: str | None = None
    broadcast_selected_description: str | None = None
    broadcast_attended_node_ids: list[str] = Field(default_factory=list)

    # --- Why: the aboutness claim AST/HOT instrumentation must make honestly
    attention_reason: AttentionReasonV1 = "no_data"
    voluntary_override: VoluntaryOverrideV1 | None = None
    reason_narrative: str = ""

    # --- How confident, derived from a real signal, never invented ---------
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_basis: str = ""

    # --- Unconditional Active-Inference confidence, additive to the above --
    # 2026-07-24: `confidence` above only populates from the real
    # prediction-error formula inside the `field_salience_only` branch,
    # which live data showed fires on ~0.04% of ticks (structurally
    # preempted by the broadcast lane being fresh almost always -- see
    # `docs/notes/2026-07-24-attention-reason-branch-starvation-finding.md`).
    # These two fields are computed unconditionally (mirroring
    # `predicted_shift`'s existing positioning) and restricted to
    # `orion.substrate.attention_self_model.ACTIVE_INFERENCE_DOMAINS`
    # (excludes the confirmed-dead `transport` domain). Additive only --
    # `confidence`/`confidence_basis` above are unchanged.
    prediction_error_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction_error_confidence_basis: str = ""

    # --- What's predicted to shift next (modest, honestly scoped) ----------
    predicted_shift: str | None = None
    predicted_shift_basis: str = ""

    # --- Explicit, honest cadence-mismatch state ----------------------------
    # The broadcast lane updates far less often than the field lane (real
    # config confirmed live 2026-07-18: ORION_ATTENTION_BROADCAST_INTERVAL_SEC
    # =30s vs. the field lane's ~2s tick). "No new GWT-dispatch-lane activity
    # since last frame" must be a distinct, named state from "nothing
    # salient" — never silently collapsed into the same output.
    broadcast_lane_stale: bool = True
    broadcast_lane_age_sec: float | None = None
