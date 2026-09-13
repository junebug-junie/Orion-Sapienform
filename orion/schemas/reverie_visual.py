"""Reverie visual chain schemas — Patch 1 of docs/superpowers/specs/
2026-08-20-reverie-visual-chain-design.md.

A second, parallel reverie chain alongside the text chain
(`orion.schemas.reverie.ReverieChainV1`): on a slow, capacity-gated cadence,
generate an image about gathered context via a diffusion model, run the image
back through vision captioning to get a description, and feed that
description into the *next* reverie's context — a real generate -> observe
-> interpret loop.

Naming: `reverie_visual_*`, no `substrate_` prefix (design doc §3) — that
namespace is scoped to the live attention-coalition rung; this chain pulls
from broader context (chats, dreams), so it doesn't belong there.

Continuity: `prior_description` on `ReverieVisualChainV1` is the enforced
column the context-builder for step N+1 actually reads (Patch 2), not a JSON
forward-pointer field left for someone to wire up later. The text chain's
`SpontaneousThoughtV1.next_focus`/`drift` fields are the counter-example this
deliberately avoids repeating (design doc §2): they carry stated producer
intent ("the LLM's forward pointer for the next step") and zero consumers —
a live keyword-cathedral instance. Do not add an analogous unread field here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from orion.schemas.thought import CoalitionSnapshotV1
from orion.schemas.reverie import MAX_EVIDENCE_REFS

# Cap on the verbatim thought/step window per chain (§ cap-all-collections),
# mirrored from orion.schemas.reverie.MAX_CHAIN_THOUGHTS.
MAX_VISUAL_CHAIN_STEPS = 50

# Same terminal-reason vocabulary as the text chain (orion.schemas.reverie.
# TerminalReason) plus a visual-specific reason for when generation itself
# fails and the chain has to stop rather than silently skip a step.
VisualTerminalReason = Literal[
    "pressure_discharged",
    "max_steps",
    "no_coalition",
    "refractory",
    "low_salience",
    "generation_failed",
    # Added 2026-08-31. The whole run exceeded ORION_VISUAL_CHAIN_RUN_DEADLINE_SEC
    # and was abandoned so the single-flight lock could be released. Distinct from
    # "generation_failed" (a hop returned an error) -- here a hop never returned at
    # all. Safe to add: the column is plain `text` with no CHECK constraint
    # (manual_migration_reverie_visual_chain.sql), and every cross-service reader
    # (orion-hub reverie_routes.py / reverie-tab.js, orion-cortex-exec
    # verb_adapters.py) passes the string through without validating it -- unlike
    # the closed vocabularies in policy_decision_frame.py, where a stale consumer
    # rejects the WHOLE frame (that failure was hit live 2026-08-30, PR #2004).
    "run_deadline_exceeded",
    # Orion declined to spend GPU watts because the room the GPU heats is too
    # warm for the person in it. Recorded as a terminal reason rather than a
    # silent skip on purpose: a refusal that leaves no row is indistinguishable
    # from a scheduler that stopped running, and this repo has been bitten by
    # that shape more than once. A refusal is a decision and should read like one.
    "thermal_refused",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReverieVisualContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    thought_id: str
    thought_correlation_id: str
    thought_created_at: datetime
    text_chain_id: str
    coalition: CoalitionSnapshotV1
    evidence_refs: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)


class VisualSourceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class VisualContextSelectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selection_method: Literal["round_robin"] = "round_robin"
    source_kind: Literal["reverie", "self_study", "memory", "prior_visual", "default_seed"]
    source: VisualSourceV1 | None = None
    reverie: ReverieVisualContextV1 | None = None
    continuity: VisualSourceV1 | None = None


class VisualProductionReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chain_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(gt=0)
    path: str = Field(min_length=1)
    produced_at: datetime


class VisualActivityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["reverie.visual.activity.v1"] = "reverie.visual.activity.v1"
    observed_at: datetime = Field(default_factory=_utc_now)
    history_status: Literal["ok", "unavailable"]
    last_success_at: datetime | None = None
    last_success_chain_id: str | None = None
    last_success_sha256: str | None = None
    last_attempt_at: datetime | None = None
    last_attempt_outcome: str | None = None
    active_attempt_id: str | None = None


class VisualBaselineEligibilityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    need_id: str = Field(min_length=1)
    observed_at: datetime
    due_at: datetime
    last_success_at: datetime | None = None
    last_success_chain_id: str | None = None
    last_success_sha256: str | None = None
    policy_id: str = Field(min_length=1)


class VisualRunRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dispatch_id: str | None = None
    proposal_id: str | None = None
    decision_id: str | None = None
    correlation_id: str | None = None
    visual_baseline: VisualBaselineEligibilityV1 | None = None


VisualRunOutcome = Literal["produced", "deferred_thermal", "deferred_busy", "already_satisfied", "failed", "unknown"]


class VisualExecutionReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: VisualRunRequestV1 = Field(default_factory=VisualRunRequestV1)
    attempt_id: str | None = None
    outcome: VisualRunOutcome
    gate_reason: str
    thermal_gate: dict = Field(default_factory=dict)
    source_selection_status: Literal["selected", "source_selection_not_reached"] = "source_selection_not_reached"
    source_refs: list[str] = Field(default_factory=list, max_length=4)
    source_kind: str | None = None
    artifact_persisted: bool = False
    production_receipt: VisualProductionReceiptV1 | None = None


class ReverieVisualChainV1(BaseModel):
    """Readout of one visual reverie chain — mirrors `reverie_visual_chain`.

    `prior_description` is the enforced continuity column (module docstring,
    design doc §2/§5) — read directly by the next step's context-builder, not
    threaded through `chain_json`.
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["reverie.visual.chain.v1"] = "reverie.visual.chain.v1"
    chain_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    theme_key: str | None = None
    terminal_reason: VisualTerminalReason = "max_steps"
    ema_salience: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_description: str | None = None
    # NOTE (2026-08-28): `chain_json["description"]` has a SECOND consumer
    # outside this service -- `services/orion-hub/scripts/endogenous_outreach.
    # py::_fetch_current_daydream` reads it straight out of Postgres to ground
    # Orion's unprompted outreach. It is an untyped key with no schema field
    # and no cross-service contract test, so renaming or dropping it makes
    # that lane go permanently silent with no error. The only thing that would
    # notice is `services/orion-hub/evals/test_daydream_caption_quality_eval.
    # py`'s liveness check, and only when run against the live host.
    # It reads this key rather than `prior_description` above on purpose:
    # `visual_chain.py` sets `prior_description = description or
    # continuity_fallback`, which carries the PREVIOUS run's caption forward
    # on a caption-failure row.
    context_selection: VisualContextSelectionV1 | None = None
    chain_json: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def restore_selection(self):
        if self.context_selection is None and self.chain_json.get("context_selection"):
            self.context_selection = VisualContextSelectionV1.model_validate(self.chain_json["context_selection"])
        return self

    stored_at: datetime = Field(default_factory=_utc_now)


class ReverieVisualArtifactV1(BaseModel):
    """One generated image — mirrors `reverie_visual_artifact`.

    Content-addressed by `sha256` (orion.reverie.visual_storage). Image bytes
    never live on this model or in `ReverieVisualChainV1.chain_json` — only
    the pointer (`path`) does (design doc §6).
    """

    schema_version: Literal["reverie.visual.artifact.v1"] = "reverie.visual.artifact.v1"
    sha256: str
    chain_id: str
    step_index: int = Field(ge=0)
    mime: str
    bytes: int = Field(ge=0)
    width: int | None = None
    height: int | None = None
    path: str
    description: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
