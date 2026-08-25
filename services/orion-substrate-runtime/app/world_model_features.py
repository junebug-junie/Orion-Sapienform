"""Assembles one ``WorldModelTrajectoryStepV1`` per tick from real state
``BiometricsSubstrateWorker`` already holds -- the feature-assembly half of
the world-model publish tick (``worker.py``'s ``_world_model_publish_tick``).

This is the FIRST real producer for ``orion:exec:request:WorldModelService``
(services/orion-world-model, PR #1775/#1861) -- previously only a manual
test-publish CLI (``services/orion-world-model/scripts/publish_test_task.py``)
issued requests.

Honesty split (CLAUDE.md "no empty-shell cognition" -- a schema-valid payload
with meaningless content is not a success state, so this must be flagged, not
hidden):

- ``execution_context``: REAL. Four of this worker's existing per-domain
  prediction-error scalars, cached on the worker instance by their own
  ``_*_tick()`` methods (``_execution_tick``, ``_chat_tick``, ``_route_tick``,
  ``_bus_synaptic_tick``) as of this patch. ``transport`` is deliberately
  EXCLUDED -- ``transport_prediction_error()`` is a retired metric (CLAUDE.md
  0A "kill means kill", excluded from ``ACTIVE_INFERENCE_DOMAINS`` since
  2026-07-26 because it measures a narrow 2-Redis-Stream census, not real bus
  traffic); wiring a known-dead signal into a brand-new consumer would repeat
  the exact mistake that section documents. The ``biometrics`` domain's own
  prediction-error SCALAR (node:substrate.biometrics, distinct from the raw
  biometrics feature-group VECTOR below) is also excluded here -- folding a
  different domain's scalar into this group would blur what "execution
  context" means; it stays inside its own node, not this trajectory step.
- ``vision_embedding``: REAL when a fresh, correctly-dimensioned embedding is
  available; explicitly zero-filled and flagged otherwise (see
  ``build_vision_embedding_group``'s docstring for the unresolved-dim
  reasoning -- this module never guesses a "corrected" dim).
- ``temporal``: REAL. A pure function over wall-clock time -- not a
  placeholder.
- ``biometrics``, ``affect``, ``memory_pointers``: ZERO-FILLED. No producer
  reachable from this process today: the real biometrics numeric vector lives
  in a different service (orion-field-digester, not cheaply reachable here
  per CLAUDE.md section 5's service-boundary rule); Orion's own affect has no
  producer anywhere in the repo as of this patch; memory-pointer vectors are
  an unbuilt design. Every tick's request carries
  ``meta.zero_filled_groups`` naming exactly which groups this is true for,
  so a downstream reader never mistakes a zero vector for a real "nothing is
  happening" reading.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orion.schemas.world_model import WorldModelFeatureGroupV1, WorldModelTrajectoryStepV1

logger = logging.getLogger("orion.substrate.runtime.world_model_features")

# Groups this producer has no real numeric-vector source for at all, this
# patch, regardless of any live state -- always zero-filled, always flagged.
# See module docstring. ``vision_embedding`` is NOT listed here even though it
# can also end up zero-filled -- its zero-fill is state-dependent (missing or
# wrong-dim this particular tick), not a permanent, patch-wide gap, so it is
# added to the meta list dynamically instead (see
# ``assemble_world_model_trajectory_step``).
ALWAYS_ZERO_FILLED_GROUPS: Tuple[str, ...] = ("biometrics", "affect", "memory_pointers")

# Execution-context domain order. Index in this tuple is the vector slot each
# scalar lands in -- kept fixed so a downstream consumer's slot->domain
# mapping stays stable across ticks even when a domain's cached scalar is
# `None` this particular tick (its slot is simply left at 0.0).
_EXECUTION_CONTEXT_DOMAINS: Tuple[str, ...] = ("execution", "chat", "route", "bus_synaptic")


@dataclass(frozen=True)
class WorldModelFeatureDims:
    """This producer's configured feature-group dims (Settings.world_model_
    dim_*). Kept as its own small dataclass rather than passing ``Settings``
    itself into every helper below -- these functions are pure and should not
    need a pydantic-settings object to be unit-testable."""

    biometrics: int
    affect: int
    execution_context: int
    memory_pointers: int
    temporal: int
    vision_embedding: int


@dataclass(frozen=True)
class ExecutionContextScalars:
    """Real per-domain prediction-error scalars, as cached by the worker's own
    ``_*_tick()`` methods. ``None`` means that tick has not produced a real
    value yet this process lifetime (cold start / that grammar-event stream
    has been empty) -- treated as "not available yet", never coerced to 0.0,
    so a genuine confirmed-calm 0.0 reading is never confused with "never
    ticked" (same distinction ``_perception_prediction_error_tick`` already
    makes for its own domain, worker.py's ``still_warming`` branch)."""

    execution: Optional[float]
    chat: Optional[float]
    route: Optional[float]
    bus_synaptic: Optional[float]

    def as_ordered_values(self) -> Tuple[Optional[float], ...]:
        return (self.execution, self.chat, self.route, self.bus_synaptic)


def temporal_features(
    now: datetime, *, dim: int, process_started_at: datetime
) -> List[float]:
    """Pure function over wall-clock time -- real data, not a placeholder.

    Layout (truncated/padded to ``dim``, so this stays correct for whatever
    ``WM_DIM_TEMPORAL``-equivalent dim is configured, not hardcoded to 8):

      [0] hour-of-day, sin (24h cycle)
      [1] hour-of-day, cos
      [2] day-of-week, sin (7-day cycle, Monday=0)
      [3] day-of-week, cos
      [4] minute-of-hour, sin (60min cycle)
      [5] minute-of-hour, cos
      [6] process session-elapsed seconds, tanh-squashed to (0, 1) at an
          ~hour scale (tanh(elapsed_sec / 3600.0)) -- deliberately not a raw
          unbounded seconds count, which would blow past any sane feature
          scale for a long-lived process.
      [7] reserved, 0.0 (no seventh real signal in this patch)

    Extra configured slots beyond index 7 are 0.0; a dim smaller than 8
    truncates from the front (hour-of-day first).
    """
    hour_frac = (now.hour + now.minute / 60.0 + now.second / 3600.0) / 24.0
    dow_frac = now.weekday() / 7.0
    minute_frac = (now.minute + now.second / 60.0) / 60.0
    elapsed_sec = max(0.0, (now - process_started_at).total_seconds())
    session_elapsed = math.tanh(elapsed_sec / 3600.0)

    full = [
        math.sin(2.0 * math.pi * hour_frac),
        math.cos(2.0 * math.pi * hour_frac),
        math.sin(2.0 * math.pi * dow_frac),
        math.cos(2.0 * math.pi * dow_frac),
        math.sin(2.0 * math.pi * minute_frac),
        math.cos(2.0 * math.pi * minute_frac),
        session_elapsed,
        0.0,
    ]
    if dim <= len(full):
        return full[:dim]
    return full + [0.0] * (dim - len(full))


def zero_feature_group(dim: int) -> WorldModelFeatureGroupV1:
    return WorldModelFeatureGroupV1(dim=dim, vector=[0.0] * dim)


def build_execution_context_group(
    dim: int, scalars: ExecutionContextScalars
) -> Tuple[WorldModelFeatureGroupV1, List[str]]:
    """Fixed-slot layout (see ``_EXECUTION_CONTEXT_DOMAINS``): whichever
    domains have a real cached scalar populate their slot; the rest -- both
    "not ticked yet" domains and any slot beyond the four real domains up to
    ``dim`` -- stay 0.0. Returns the group plus the list of domain names that
    were actually real this tick (for ``meta.real_execution_context_domains``,
    the honesty trail)."""
    vector = [0.0] * dim
    real_domains: List[str] = []
    for i, (name, value) in enumerate(
        zip(_EXECUTION_CONTEXT_DOMAINS, scalars.as_ordered_values())
    ):
        if i >= dim:
            break
        if value is not None:
            vector[i] = float(value)
            real_domains.append(name)
    return WorldModelFeatureGroupV1(dim=dim, vector=vector), real_domains


def build_vision_embedding_group(
    dim: int, *, raw_vector: Optional[Sequence[float]]
) -> Tuple[WorldModelFeatureGroupV1, Dict[str, Any]]:
    """Defensive vision-embedding assembly.

    ``WM_DIM_VISION_EMBEDDING=512`` (services/orion-world-model/app/
    settings.py, mirrored here as ``Settings.world_model_dim_vision_
    embedding``) was never verified against the real deployed SigLIP2 profile
    (``google/siglip2-so400m-patch14-384`` on orion-vision-host, profile
    ``embed_image``) -- as of this patch that profile has never actually been
    exercised on athena, so there is no live number to check yet. This
    function deliberately does NOT hardcode a "corrected" guess (e.g. 1152):
    that would just trade one unverified constant for another.

    Instead: compare the real observed vector's length to the configured
    ``dim`` at publish time.
      - Match -> use the real vector, ``vision_source="real"``.
      - Missing (no embedding cached yet, e.g. the P2 listener is disabled or
        has not seen a message yet this process lifetime) ->
        ``vision_source="unavailable"``.
      - Length mismatch -> zero-fill AND log a loud warning naming both the
        observed and configured dims, ``vision_source="dim_mismatch"``. This
        is the mechanism that lets the FIRST time a real embedding actually
        flows through either silently confirm the config is right (dims
        match) or surface the real number for a human to fix -- instead of
        crashing this tick or corrupting the request with a shape the
        world-model service's own ``trajectory_steps_to_tensors`` (app/
        main.py) would reject anyway (its own dim check is stricter: it
        raises ValueError on a mismatch rather than degrading).
    """
    if not raw_vector:
        return zero_feature_group(dim), {"vision_source": "unavailable"}
    observed_dim = len(raw_vector)
    if observed_dim != dim:
        logger.warning(
            "world_model_vision_dim_mismatch observed=%d configured=%d -- zero-filling "
            "this tick's vision_embedding group rather than guessing a corrected dim. "
            "See services/orion-substrate-runtime/README.md 'World-model publish tick'.",
            observed_dim,
            dim,
        )
        return zero_feature_group(dim), {
            "vision_source": "dim_mismatch",
            "vision_dim_observed": observed_dim,
            "vision_dim_configured": dim,
        }
    return (
        WorldModelFeatureGroupV1(dim=dim, vector=[float(x) for x in raw_vector]),
        {"vision_source": "real"},
    )


def assemble_world_model_trajectory_step(
    *,
    now: datetime,
    process_started_at: datetime,
    dims: WorldModelFeatureDims,
    execution_context: ExecutionContextScalars,
    vision_embedding_vector: Optional[Sequence[float]],
) -> Tuple[WorldModelTrajectoryStepV1, Dict[str, Any]]:
    """Build one real ``WorldModelTrajectoryStepV1`` plus a ``meta`` dict
    documenting exactly which groups were zero-filled and why -- assigned to
    ``WorldModelTaskRequestPayload.meta`` by the caller (worker.py). One step
    per request is valid: ``services/orion-world-model/app/main.py``'s
    ``WorldModelService`` holds no session/window state across requests, so
    this producer does not need to buffer a rolling trajectory window."""
    execution_context_group, real_domains = build_execution_context_group(
        dims.execution_context, execution_context
    )
    vision_group, vision_meta = build_vision_embedding_group(
        dims.vision_embedding, raw_vector=vision_embedding_vector
    )
    temporal_vector = temporal_features(
        now, dim=dims.temporal, process_started_at=process_started_at
    )

    zero_filled_groups = list(ALWAYS_ZERO_FILLED_GROUPS)
    if vision_meta.get("vision_source") != "real":
        zero_filled_groups.append("vision_embedding")

    step = WorldModelTrajectoryStepV1(
        ts=now.timestamp(),
        biometrics=zero_feature_group(dims.biometrics),
        affect=zero_feature_group(dims.affect),
        execution_context=execution_context_group,
        memory_pointers=zero_feature_group(dims.memory_pointers),
        temporal=WorldModelFeatureGroupV1(dim=dims.temporal, vector=temporal_vector),
        vision_embedding=vision_group,
    )
    meta: Dict[str, Any] = {
        "producer": "orion-substrate-runtime",
        "zero_filled_groups": zero_filled_groups,
        "real_execution_context_domains": real_domains,
        **vision_meta,
    }
    return step, meta
