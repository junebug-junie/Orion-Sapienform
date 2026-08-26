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
    subtitle_source: Optional[Literal["caller", "transcribed", "none"]] = Field(
        default=None,
        description=(
            "Where the subtitle text actually fed to the model came from -- "
            "'caller' (the request already carried real text), 'transcribed' "
            "(request's subtitle was empty, Whisper produced real text from "
            "audio_path), or 'none' (empty request subtitle and either "
            "transcription is disabled/failed or the clip was judged "
            "near-silent). Added 2026-08-22 -- this service's own README "
            "already documented that an empty subtitle materially degrades "
            "output quality, but nothing surfaced WHICH case produced a "
            "given raw_response, so a caller reading a generic hedge (e.g. "
            "\"cannot infer emotional state\") had no way to tell whether "
            "that was a real model read on real speech or an artifact of "
            "silently getting no transcript at all."
        ),
    )
    transcript: Optional[str] = Field(
        default=None,
        description=(
            "The actual Whisper transcript text when subtitle_source=="
            "'transcribed' -- what the model was actually shown, not just "
            "whether transcription happened. Review finding, 2026-08-22: "
            "this is verbatim transcribed speech on the wire, a real "
            "widening of this module's privacy surface -- see module "
            "docstring's 'Paths only, never raw bytes/frames' principle, "
            "which this field is a deliberate, Juniper-approved exception "
            "to (not an oversight). Mitigating factor: raw_response "
            "already routinely paraphrases/quotes the same speech content "
            "indirectly (confirmed live -- e.g. 'the caption reads: "
            "\"...\"'), so this field mostly makes explicit what was "
            "already reachable, rather than opening a wholly new leak."
        ),
    )
    meta: Optional[Dict[str, Any]] = None


class AffectReadV1(BaseModel):
    """One structured affect read, produced by the ``vision`` backend.

    **Why this exists at all, given the module docstring above says "no
    emotion taxonomy field on purpose".** That rule was written when
    ``raw_response`` (AffectGPT's free-text reasoning) was the only signal,
    and it was the right call then: inventing a label set ahead of data is
    the keyword cathedral CLAUDE.md bans. What changed on 2026-08-26 is that
    free text turned out to be actively unusable *as a prompt input*, for a
    reason no taxonomy debate would have surfaced -- the read Orion actually
    received for chat turn ``ddddfe40`` was a 400-character essay opening
    "In the text, based on the provided information, it is not possible to
    infer the character's emotional state from the subtitle content."
    Schema-valid, ``ok=True``, and mirrored verbatim into Juniper's chat
    prompt.

    So this is NOT a taxonomy. ``primary_affect`` is a free string, not an
    enum, precisely because no label set has been earned yet -- whatever the
    model says goes in, capped for length. What IS structured here is only
    the machinery a consumer needs to decide *whether to believe the read at
    all*: a confidence the mirror gate can threshold on, an explicit
    ``cannot_tell`` list, and the ``cues`` the model claims to have used.
    Every field has a live consumer in the same changeset
    (``orion/situational/context.py``) or it would not be here.

    **``cues`` is the anti-confabulation field.** The replaced backend
    asserted "the acoustic characteristics of the voice indicate a negative
    emotion" about an audio track measured at -49.2 dB peak -- i.e. silence.
    Forcing the model to name the evidence it used makes that failure
    visible in the record instead of laundering it into a confident mood.
    """

    model_config = ConfigDict(extra="ignore")

    # -1.0 (strongly negative) .. +1.0 (strongly positive).
    valence: float = Field(..., ge=-1.0, le=1.0)
    # 0.0 (calm/still) .. 1.0 (highly activated).
    arousal: float = Field(..., ge=0.0, le=1.0)
    primary_affect: str = Field(
        ...,
        max_length=64,
        description=(
            "The model's own short label. Deliberately NOT an enum -- see "
            "class docstring. Capped only so a runaway generation cannot "
            "smuggle an essay through this field the way raw_response did."
        ),
    )
    cues: list[str] = Field(
        default_factory=list,
        description="Specific observations the read rests on, model's own words.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "The gate orion/situational/context.py thresholds on before this "
            "read is allowed to colour a chat turn. A low value here is a "
            "SUCCESSFUL read reporting honest uncertainty -- distinct from "
            "ok=False, which means the pipeline itself failed."
        ),
    )
    cannot_tell: list[str] = Field(
        default_factory=list,
        description="What the model explicitly declined to judge from this input.",
    )


class JuniperMultimodalAffectV1(BaseModel):
    """Domain event published by orion-juniper-affective-state after wrapping
    one worker assessment. See module docstring for why this is NOT the same
    schema/channel as the existing text-only JuniperAffectiveStateV1."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["juniper_multimodal_affect.v1"] = "juniper_multimodal_affect.v1"
    observed_at: datetime
    # Widened from Literal["affectgpt"] on 2026-08-26. Additive: "affectgpt"
    # remains the default, so every stored row and every existing consumer
    # keeps validating unchanged.
    source: Literal["affectgpt", "vision"] = "affectgpt"
    backend: Literal["affectgpt", "vision"] = Field(
        default="affectgpt",
        description=(
            "Which inference backend produced this read. Exists as a field, "
            "not just an env setting, because the two are not "
            "interchangeable in quality and a stored row must say which one "
            "it came from -- the affectgpt rows in "
            "juniper_multimodal_affect_log predating 2026-08-26 include "
            "three reads that confidently misgendered the subject, and a "
            "later analysis needs to be able to exclude them by provenance "
            "rather than by date arithmetic."
        ),
    )
    affect: Optional["AffectReadV1"] = Field(
        default=None,
        description=(
            "Structured read. Populated by backend='vision'; always None for "
            "backend='affectgpt', which only ever produced free text. "
            "Consumers must treat None as 'no structured read available' and "
            "fall back to raw_response, never as 'affect was neutral'."
        ),
    )
    frames_used: Optional[int] = Field(
        default=None,
        description="How many frames the vision backend actually sent to the model.",
    )
    ok: bool
    raw_response: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    model_ckpt: Optional[str] = None
    face_detection: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, Any]] = None
    subtitle_source: Optional[Literal["caller", "transcribed", "none"]] = Field(
        default=None,
        description="See AffectGptAssessResultPayload.subtitle_source -- threaded straight through by _wrap_event().",
    )
    transcript: Optional[str] = Field(
        default=None,
        description="See AffectGptAssessResultPayload.transcript -- threaded straight through by _wrap_event().",
    )
    # Paths only, never raw bytes/frames on the wire -- keeps this event
    # inspectable/traceable without becoming a raw-media exposure surface.
    input_ref: Optional[Dict[str, Any]] = None
    # Added 2026-08-22 for the ambient (recurring) capture toggle -- Juniper's
    # own ask: "ensure data model has good ability to be correlative with
    # other components in the mesh." Two distinct things:
    trigger: Literal["manual", "ambient", "chat_turn_pre", "chat_turn_post"] = Field(
        default="manual",
        description=(
            "Which entry point produced this event -- POST /trigger or "
            "/capture_and_assess called directly (manual) vs Hub's recurring "
            "toggle loop (ambient) vs the pair bracketing one Orion-mode "
            "chat turn (chat_turn_pre, fired once Whisper has a transcript "
            "and before the turn runs; chat_turn_post, fired once the turn's "
            "reply has been handed back). Now four producers of the same "
            "event type; a consumer needs this to tell them apart -- and the "
            "chat_turn_* pair specifically is only meaningful AS a pair, "
            "joined via chat_correlation_id below."
        ),
    )
    chat_correlation_id: Optional[str] = Field(
        default=None,
        description=(
            "The Orion-mode chat turn's OWN correlation_id (Hub's per-turn "
            "trace_id), present only on trigger=chat_turn_pre/chat_turn_post. "
            "Deliberately NOT reusing `correlation_id` above, which already "
            "means something else and must keep meaning it: that one joins "
            "the three legs of a single capture attempt (retina RPC, worker "
            "RPC, this event). This one joins a capture to the conversation "
            "that caused it, and joins the pre/post pair of one turn to each "
            "other -- two different join axes, two fields. A consumer asking "
            "'how did Juniper's affect move across this turn' needs exactly "
            "this key; observed_at-proximity cannot answer it, because "
            "concurrent ambient ticks land in the same time window."
        ),
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description=(
            "The ONE id threading through this entire capture attempt -- the "
            "retina clip-capture RPC, the worker assess RPC, and this event "
            "all share it (see capture_and_assess()'s single corr_id, not two "
            "independently-generated ones per RPC leg). Lets any mesh "
            "consumer join all three legs of one tick via a single id, on top "
            "of the existing observed_at-proximity join every other signal "
            "here already supports."
        ),
    )
