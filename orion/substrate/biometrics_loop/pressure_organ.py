from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    ActiveNodePressureStateV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.organ_emission import OrganEmissionV1

from .candidate_events import PRESSURE_SOURCE_COMPONENT, build_pressure_candidate_events
from .ids import parse_biometrics_trace_id, parse_pressure_trace_id

ALLOWED_PRESSURE_ROLES = frozenset(
    {
        "node_pressure_detected",
        "node_pressure_reinforced",
        "node_pressure_decayed",
        "node_availability_concern",
        "node_availability_recovered",
        "node_pressure_suppressed",
        "node_capability_impact",
    }
)

DEFAULT_STALE_AFTER_SEC = 180
DEFAULT_MIN_CONFIDENCE = 0.60
HINT_THRESHOLD = 0.55
GPU_HINT_THRESHOLD = 0.60


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _node_id_from_trigger(trigger_event: GrammarEventV1, catalog: NodeCatalog) -> str | None:
    parsed = parse_biometrics_trace_id(trigger_event.trace_id)
    if parsed:
        return catalog.resolve(parsed).node_id
    if trigger_event.atom and trigger_event.atom.text_value:
        return catalog.resolve(trigger_event.atom.text_value).node_id
    return None


def _is_stale(
    *,
    last_seen_at: datetime | None,
    stale_after_sec: int,
    now: datetime,
) -> bool:
    if last_seen_at is None:
        return True
    seen = last_seen_at if last_seen_at.tzinfo else last_seen_at.replace(tzinfo=timezone.utc)
    return (now - seen).total_seconds() > stale_after_sec


def _has_pressure_hints(hints: dict) -> bool:
    return any(float(v) >= HINT_THRESHOLD for v in hints.values() if v is not None)


def _primary_hint_kind(hints: dict) -> str:
    if not hints:
        return "strain"
    ranked = sorted(
        ((k, float(v)) for k, v in hints.items() if v is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0] if ranked else "strain"


def sweep_absent_nodes(
    *,
    node_bio: NodeBiometricsProjectionV1,
    stale_after_sec: int = DEFAULT_STALE_AFTER_SEC,
    now: datetime | None = None,
) -> list[str]:
    """Node ids that are `expected_online` and have gone stale.

    This is the trigger `invoke_biometrics_pressure()` never had. That function
    resolves its subject node from the *incoming* event
    (`_node_id_from_trigger()`), so a node that stops reporting entirely is never
    evaluated at all -- Rule B ("expected online + stale -> availability
    concern") can only ever fire for a node that is still talking. A node that is
    genuinely gone is structurally invisible to it.

    Confirmed live 2026-08-29: circe (`expected_online: true`, 5 declared
    capabilities) went dark 00:01:16Z -> 00:47:04Z and its
    `last_accepted_at["availability"]` never moved off 2026-08-19T03:13:33Z. Four
    other detectors saw the outage inside two minutes; this one could not, by
    construction.

    Pure function over the projection that already carries `last_seen_at` and
    `expected_online` per node -- no new state, no new storage, no clock of its
    own beyond the injected `now`. Callers pair each returned id with the same
    `invoke_biometrics_pressure()` the event path uses, so absence and presence
    produce the same downstream shapes.

    A node whose `last_seen_at` is None but which IS present in the projection is
    treated as stale (`_is_stale` returns True for None).

    **Known gap, deliberate for phase 1:** this iterates `node_bio.nodes`, which is
    built from received biometrics events. A node that has never reported even once
    is not in that projection at all and therefore cannot be swept here. That is not
    hypothetical -- `prometheus` is catalogued `expected_online: true` with
    `monitoring/logs/metrics: true` and has never written a single `orion_biometrics`
    row; the live projection contains only `atlas`, `circe`, `athena`. Catching a
    never-reported node requires sweeping the catalog, not the projection, and is
    tracked as a phase-2 item in the design doc rather than silently implied here.
    """
    clock = _utc_now(now)
    absent: list[str] = []
    for node_id, state in (node_bio.nodes or {}).items():
        if state.expected_online is not True:
            continue
        if not _is_stale(
            last_seen_at=state.last_seen_at,
            stale_after_sec=stale_after_sec,
            now=clock,
        ):
            continue
        absent.append(node_id)
    return sorted(absent)


def invoke_biometrics_pressure(
    *,
    trigger_event: GrammarEventV1,
    node_bio: NodeBiometricsProjectionV1,
    active_pressure: ActiveNodePressureProjectionV1,
    catalog: NodeCatalog,
    stale_after_sec: int = DEFAULT_STALE_AFTER_SEC,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    now: datetime | None = None,
) -> OrganEmissionV1:
    clock = _utc_now(now)
    node_id = _node_id_from_trigger(trigger_event, catalog)
    if node_id is None:
        return OrganEmissionV1(
            emission_id=f"oem_{uuid4().hex[:12]}",
            organ_id="biometrics_pressure",
            invocation_id=f"inv_{uuid4().hex[:12]}",
            triggered_by_event_ids=[trigger_event.event_id],
            inspected_projection_ids=[node_bio.projection_id, active_pressure.projection_id],
            created_at=clock,
        )

    bio_state = node_bio.nodes.get(node_id)
    pressure_state = active_pressure.nodes.get(node_id)
    active_pressures = list(pressure_state.active_pressures) if pressure_state else []

    evidence = [trigger_event.event_id]
    if bio_state:
        for field in (
            "latest_sample_event_id",
            "latest_summary_event_id",
            "latest_induction_event_id",
        ):
            event_id = getattr(bio_state, field)
            if event_id:
                evidence.append(event_id)

    candidates: list[list[GrammarEventV1]] = []
    expected_online = bio_state.expected_online if bio_state else catalog.resolve(node_id).expected_online
    stale = _is_stale(
        last_seen_at=bio_state.last_seen_at if bio_state else None,
        stale_after_sec=stale_after_sec,
        now=clock,
    )
    hints = dict(bio_state.pressure_hints) if bio_state else {}
    capabilities = bio_state.capabilities if bio_state else []

    # Rule A: expected offline + stale/missing -> suppressed
    if expected_online is False and stale:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_pressure_suppressed",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, 0.85),
                observed_at=clock,
            )
        )

    # Rule B: expected online + stale -> availability concern
    elif expected_online is True and stale:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_availability_concern",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, 0.8),
                observed_at=clock,
            )
        )
        # Rule F: absence is a capability impact too. Rule E (below) is the only
        # other emitter of `node_capability_impact` and it fires on
        # `gpu_hint >= GPU_HINT_THRESHOLD` -- GPU *saturation*. A node that is gone
        # reports no hints at all, so before this branch existed capability impact
        # was structurally unreachable by absence: `grammar_atoms` held 0 rows with
        # this role, ever, and `capability_impacts` was `[]` in every
        # `substrate_active_node_pressure_projection` row ever written. The role,
        # the ROLE_TO_PRESSURE_KIND mapping and the reducer arm all already
        # existed; nothing could reach them.
        #
        # One event per stale node, not one per capability: `build_pressure_
        # candidate_events` derives `atom_id` from `{trace_id}:{semantic_role}` and
        # `trace_id` from `{node_id}:{ts}`, so per-capability events in the same
        # tick would collide on both. The reducer expands this single event into
        # the node's real declared capabilities from the catalog profile it has
        # already resolved (see pressure_reducer.py's node_capability_impact arm).
        # Repeat ticks during a long outage are absorbed by that reducer's
        # existing per-node+pressure_kind merge window, so a 45-minute outage does
        # not become a 45-minute event storm.
        if capabilities:
            candidates.append(
                build_pressure_candidate_events(
                    node_id=node_id,
                    semantic_role="node_capability_impact",
                    evidence_event_ids=evidence,
                    confidence=max(min_confidence, 0.8),
                    observed_at=clock,
                )
            )

    # Rule B': expected online + fresh + a prior availability concern is
    # still flagged -> recovered. The symmetric counterpart Rule B needed
    # and never had: "availability" in active_pressures was a one-way
    # ratchet -- the only removal path (Rule D / node_pressure_decayed) is
    # hardcoded in pressure_reducer.py's ROLE_TO_PRESSURE_KIND to clear
    # "strain" only, never "availability". Confirmed live 2026-07-22:
    # node:atlas had a transient staleness blip set "availability" once,
    # then stayed permanently flagged (field-digester's availability
    # channel pinned at 0.0) for hours after biometrics resumed reporting
    # fresh (last_seen_at continuously current) -- a real node, fully
    # reachable, misrepresented as unavailable.
    elif expected_online is True and not stale and "availability" in active_pressures:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_availability_recovered",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, 0.8),
                observed_at=clock,
            )
        )

    has_hints = _has_pressure_hints(hints)
    hint_kind = _primary_hint_kind(hints)

    # Rule C: hints + prior active -> reinforced
    if has_hints and active_pressures:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_pressure_reinforced",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, float(hints.get(hint_kind, min_confidence))),
                observed_at=clock,
            )
        )
    # New detection when hints present without prior active pressure
    elif has_hints and not active_pressures:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_pressure_detected",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, float(hints.get(hint_kind, min_confidence))),
                observed_at=clock,
            )
        )

    # Rule D: prior pressured + empty hints -> decayed
    if active_pressures and not has_hints:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_pressure_decayed",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, 0.7),
                observed_at=clock,
            )
        )

    # Rule E: llm inference capability + gpu hint -> capability impact
    has_llm = any(
        cap in capabilities
        for cap in ("local_llm_heavy", "local_llm_quick", "llm_inference")
    )
    gpu_hint = float(hints.get("gpu", 0.0))
    if has_llm and gpu_hint >= GPU_HINT_THRESHOLD:
        candidates.append(
            build_pressure_candidate_events(
                node_id=node_id,
                semantic_role="node_capability_impact",
                evidence_event_ids=evidence,
                confidence=max(min_confidence, gpu_hint),
                observed_at=clock,
            )
        )

    return OrganEmissionV1(
        emission_id=f"oem_{uuid4().hex[:12]}",
        organ_id="biometrics_pressure",
        invocation_id=f"inv_{uuid4().hex[:12]}",
        triggered_by_event_ids=[trigger_event.event_id],
        inspected_projection_ids=[node_bio.projection_id, active_pressure.projection_id],
        candidate_events=[event for trace in candidates for event in trace],
        created_at=clock,
    )
