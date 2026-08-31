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

from pydantic import BaseModel, ConfigDict, Field

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
    chain_json: dict = Field(default_factory=dict)
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
