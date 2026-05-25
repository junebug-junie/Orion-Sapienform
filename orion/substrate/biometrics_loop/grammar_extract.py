from __future__ import annotations

from datetime import datetime, timezone

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import AvailabilityStatus, NodeBiometricsStateV1
from orion.schemas.grammar import GrammarEventV1

from .ids import parse_biometrics_trace_id


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _payload_kind(payload_ref: str | None) -> str | None:
    if not payload_ref:
        return None
    if payload_ref.startswith("biometrics.sample:"):
        return "sample"
    if payload_ref.startswith("biometrics.summary:"):
        return "summary"
    if payload_ref.startswith("biometrics.induction:"):
        return "induction"
    return None


def _resolve_node_id(events: list[GrammarEventV1], catalog: NodeCatalog) -> str:
    for event in events:
        if event.trace_id:
            parsed = parse_biometrics_trace_id(event.trace_id)
            if parsed:
                return catalog.resolve(parsed).node_id
        atom = event.atom
        if atom and atom.semantic_role == "node_context" and atom.text_value:
            return catalog.resolve(atom.text_value).node_id
    return "unknown"


def _compute_availability(
    *,
    profile_expected_online: bool | None,
    last_seen_at: datetime | None,
    stale_after_sec: int,
    now: datetime,
    availability_summary: str | None,
) -> AvailabilityStatus:
    if profile_expected_online is False:
        return "offline_expected"

    if last_seen_at is None:
        return "missing_expected"

    age_sec = (now - last_seen_at).total_seconds()
    if age_sec > stale_after_sec:
        if "DEGRADED" in (availability_summary or "").upper():
            return "stale"
        return "stale"

    if "missing" in (availability_summary or "").lower():
        return "missing_expected"

    return "online"


def extract_node_state_from_events(
    events: list[GrammarEventV1],
    catalog: NodeCatalog,
    *,
    stale_after_sec: int = 180,
    now: datetime | None = None,
) -> NodeBiometricsStateV1:
    """Build node biometrics state from ordered grammar events (atoms only, no debug_trace)."""
    clock = _utc_now(now)
    node_id = _resolve_node_id(events, catalog)
    profile = catalog.resolve(node_id)

    last_seen_at: datetime | None = None
    latest_trace_id: str | None = None
    latest_payload_ref: str | None = None
    latest_sample_event_id: str | None = None
    latest_summary_event_id: str | None = None
    latest_induction_event_id: str | None = None
    availability_summary: str | None = None
    pressure_hints: dict[str, float] = {}

    for event in events:
        if event.trace_id:
            latest_trace_id = event.trace_id
        observed = event.observed_at or event.emitted_at
        if observed:
            observed = observed if observed.tzinfo else observed.replace(tzinfo=timezone.utc)
            if last_seen_at is None or observed > last_seen_at:
                last_seen_at = observed

        atom = event.atom
        if atom is None:
            continue

        if atom.payload_ref:
            latest_payload_ref = atom.payload_ref

        if atom.semantic_role == "telemetry_sample":
            latest_sample_event_id = event.event_id
        elif atom.semantic_role == "body_state":
            payload_kind = _payload_kind(atom.payload_ref)
            if payload_kind == "induction":
                latest_induction_event_id = event.event_id
            else:
                latest_summary_event_id = event.event_id
            if atom.salience is not None:
                pressure_hints["strain"] = float(atom.salience)
        elif atom.semantic_role == "node_availability":
            availability_summary = atom.summary
        elif atom.semantic_role == "capability_surface" and atom.salience is not None:
            if profile.capabilities.get("local_llm_heavy"):
                pressure_hints["gpu"] = float(atom.salience)

    availability = _compute_availability(
        profile_expected_online=profile.expected_online,
        last_seen_at=last_seen_at,
        stale_after_sec=stale_after_sec,
        now=clock,
        availability_summary=availability_summary,
    )

    capabilities = [k for k, enabled in profile.capabilities.items() if enabled]
    aliases = [a for a, canonical in catalog.aliases.items() if canonical == node_id and a != node_id]

    return NodeBiometricsStateV1(
        node_id=node_id,
        aliases=sorted(aliases),
        role=profile.role,
        capabilities=capabilities,
        expected_online=profile.expected_online,
        last_seen_at=last_seen_at,
        latest_trace_id=latest_trace_id,
        latest_payload_ref=latest_payload_ref,
        latest_sample_event_id=latest_sample_event_id,
        latest_summary_event_id=latest_summary_event_id,
        latest_induction_event_id=latest_induction_event_id,
        availability_status=availability,
        pressure_hints=pressure_hints,
    )
