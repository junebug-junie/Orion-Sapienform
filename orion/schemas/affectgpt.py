"""Bus payload schemas for the AffectGPT multimodal (face+voice) affect worker.

DELIBERATELY SEPARATE from ``orion.schemas.affective_state.JuniperAffectiveStateV1``
(``orion:substrate:juniper_affective_state``). That existing, approved signal
(docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md)
is narrowly scoped to a text-derived aggregate (swear-word frequency in typed
messages) precisely BECAUSE its proposal explicitly disclaims being "a general
mood-reading feature" or "a generic emotional surveillance capability" — scope
tied to co-creation cost, not Juniper's actual face/voice.

This module's schemas are a materially broader, more privacy-sensitive
capability: real webcam/microphone-derived facial and vocal affect via a 7B
VLM (AffectGPT). Juniper directly asked for this to be built (2026-08-22
session), which is CLAUDE.md §0A's own stated exception to proposal-mode
gating for cognition/self-modeling/social-continuity changes — but it does
NOT retroactively narrow the existing text-only signal's scope or authorize
reusing its name/channel. Keeping these on distinct schema_ids and distinct
channels (``orion:affectgpt:*`` vs ``orion:substrate:juniper_affective_state``)
is what keeps that boundary real instead of accidental.

No emotion taxonomy/classifier field is defined here on purpose (CLAUDE.md's
"no keyword cathedral" rule) -- ``raw_response`` (the model's own free-text
reasoning) is the only signal carried until a real, theory-anchored label set
is justified by actual accumulated data, not invented ahead of it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AffectGptAssessRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_path: str = Field(
        ...,
        description=(
            "Path readable inside the worker container. Used ONLY for "
            "Haar-cascade face-crop extraction -- no raw video frames are "
            "ever fed to the model (no frame-mode checkpoint exists)."
        ),
    )
    audio_path: str
    subtitle: str = Field(
        default="",
        description=(
            "Real transcript text if available; empty string if none. "
            "Confirmed live 2026-08-22: omitting real subtitle text produces "
            "materially worse-grounded output than supplying it."
        ),
    )
    user_message: Optional[str] = Field(
        default=None,
        description="Overrides the worker's default 'infer emotional state' prompt.",
    )
    meta: Optional[Dict[str, Any]] = None


class AffectGptAssessResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    raw_response: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    model_ckpt: Optional[str] = None
    face_or_frame_mode: Optional[str] = None
    face_detection: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Haar-cascade detection telemetry: frames_total/detected/carried_forward/fallback + detection_rate.",
    )
    timings: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class JuniperMultimodalAffectV1(BaseModel):
    """Domain event published by orion-juniper-affective-state after wrapping
    one worker assessment. See module docstring for why this is NOT the same
    schema/channel as the existing text-only JuniperAffectiveStateV1."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["juniper_multimodal_affect.v1"] = "juniper_multimodal_affect.v1"
    observed_at: datetime
    source: Literal["affectgpt"] = "affectgpt"
    ok: bool
    raw_response: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    model_ckpt: Optional[str] = None
    face_detection: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, Any]] = None
    # Paths only, never raw bytes/frames on the wire -- keeps this event
    # inspectable/traceable without becoming a raw-media exposure surface.
    input_ref: Optional[Dict[str, Any]] = None
