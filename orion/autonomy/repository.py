from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

from orion.autonomy.models import AutonomyStateV1

logger = logging.getLogger("orion.autonomy.repository")

AvailabilityKind = Literal["available", "degraded", "empty", "unavailable"]


@dataclass(frozen=True)
class AutonomyRepositoryStatus:
    backend: str
    source_path: str
    source_available: bool


@dataclass(frozen=True)
class AutonomyLookupV1:
    subject: str
    state: AutonomyStateV1 | None
    availability: AvailabilityKind
    unavailable_reason: str | None = None
    subquery_diagnostics: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class SubjectBinding:
    model_layer: str
    entity_id: str


SUBJECT_BINDINGS: dict[str, SubjectBinding] = {
    "orion": SubjectBinding(model_layer="self-model", entity_id="self:orion"),
    "juniper": SubjectBinding(model_layer="user-model", entity_id="user:juniper"),
    "relationship": SubjectBinding(model_layer="relationship-model", entity_id="relationship:orion|juniper"),
}


class AutonomyRepository(Protocol):
    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        ...

    def list_latest(self, subjects: Sequence[str], *, observer: dict[str, str] | None = None) -> list[AutonomyLookupV1]:
        ...

    def status(self) -> AutonomyRepositoryStatus:
        ...


def _drives_facet_status(lookup: AutonomyLookupV1 | None) -> str:
    if lookup is None:
        return ""
    diags = (lookup.subquery_diagnostics or {}).get("drives")
    if not isinstance(diags, dict):
        return ""
    return str(diags.get("status") or "").strip().lower()


def _drives_facet_ok(lookup: AutonomyLookupV1 | None) -> bool:
    # AutonomyStateV1.dominant_drive/drive_pressures/active_drives were
    # removed 2026-07-30 (chore/delete-orion-drives Wave 2a) -- this function
    # crashed (AttributeError) on any lookup whose state was populated before
    # this fix, since it unconditionally read those attributes. The real
    # remaining signal is the "drives" subquery's own diagnostics (row_count),
    # which this already had as a fallback path -- promoted to the only path.
    if lookup is None or lookup.availability not in {"available", "degraded"}:
        return False
    status = _drives_facet_status(lookup)
    if status != "ok":
        return False
    diags = (lookup.subquery_diagnostics or {}).get("drives")
    if not isinstance(diags, dict):
        return False
    return int(diags.get("row_count") or 0) > 0


@dataclass(frozen=True)
class SelectedAutonomyLookup:
    lookup: AutonomyLookupV1 | None
    selected_subject: str | None
    contextual_fallback: bool
    orion_lookup: AutonomyLookupV1 | None = None


def select_preferred_autonomy_lookup(by_subject: Mapping[str, AutonomyLookupV1]) -> SelectedAutonomyLookup:
    """Prefer Orion when drives are usable; fall back to relationship drives with explicit provenance."""
    orion = by_subject.get("orion")
    relationship = by_subject.get("relationship")
    if _drives_facet_ok(orion):
        return SelectedAutonomyLookup(orion, "orion", False, orion)
    if _drives_facet_ok(relationship):
        return SelectedAutonomyLookup(relationship, "relationship", True, orion)
    _ORION_DRIVES_BAD = frozenset({"timeout", "error", "deferred", "query_error", "connection_error"})
    for subject in ("relationship", "orion", "juniper"):
        lookup = by_subject.get(subject)
        if lookup and lookup.availability in {"available", "degraded"} and lookup.state is not None:
            if subject == "orion" and _drives_facet_status(lookup) in _ORION_DRIVES_BAD:
                continue
            contextual = (
                subject == "relationship"
                and _drives_facet_ok(relationship)
                and not _drives_facet_ok(orion)
            )
            return SelectedAutonomyLookup(lookup, subject, contextual, orion)
    for subject in ("relationship", "orion", "juniper"):
        lookup = by_subject.get(subject)
        if lookup and lookup.availability not in {"empty"} and lookup.state is not None:
            if subject == "orion" and _drives_facet_status(lookup) in _ORION_DRIVES_BAD:
                continue
            contextual = (
                subject == "relationship"
                and _drives_facet_ok(relationship)
                and not _drives_facet_ok(orion)
            )
            return SelectedAutonomyLookup(lookup, subject, contextual, orion)
    # Last resort: Orion partial state (identity/goals) for proposal_only degraded stance.
    if orion and orion.availability in {"available", "degraded"} and orion.state is not None:
        return SelectedAutonomyLookup(orion, "orion", False, orion)
    return SelectedAutonomyLookup(None, None, False, orion)


class LocalAutonomyRepository:
    """Tiny local fallback seam for tests/dev; defaults to empty."""

    def __init__(self, *, source_path: str | None = None) -> None:
        self._source_path = source_path or ""

    def status(self) -> AutonomyRepositoryStatus:
        return AutonomyRepositoryStatus(
            backend="local",
            source_path=self._source_path or "autonomy:local:empty",
            source_available=bool(self._source_path and Path(self._source_path).exists()),
        )

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        del observer
        if subject not in SUBJECT_BINDINGS:
            return AutonomyLookupV1(subject=subject, state=None, availability="empty")
        return AutonomyLookupV1(subject=subject, state=None, availability="empty")

    def list_latest(self, subjects: Sequence[str], *, observer: dict[str, str] | None = None) -> list[AutonomyLookupV1]:
        return [self.get_latest(subject, observer=observer) for subject in subjects]


# GraphAutonomyRepository and ShadowAutonomyRepository DELETED 2026-07-30 (SSP §6
# Objective 6 follow-up, "kill means kill" -- CLAUDE.md §0A). Confirmed live before
# deletion: no Fuseki container, no GraphDB container running anywhere in this
# deployment (docker ps against the real host, checked directly, not assumed);
# AUTONOMY_GRAPH_BACKEND=disabled is the checked-in .env_example default and was
# confirmed live in the actual orion-athena-cortex-exec-chat container.
# chat_stance.py's own graph_gate.py read-plan resolver already routed every real
# call away from these classes before this deletion -- confirmed unreachable in
# production, not merely unused in theory. Neither Fuseki nor legacy GraphDB
# (AUTONOMY_GRAPH_BACKEND=graphdb) can succeed since neither backend exists; there
# is no live config that could re-enable this path. If a real graph-backed
# autonomy repository is wanted again, it needs a live target and a fresh design,
# not a resurrection of these classes -- see
# docs/superpowers/specs/2026-07-30-goal-system-remaining-gaps-design.md Part G.


def build_autonomy_repository(*, source_path: str | None = None) -> AutonomyRepository:
    """Real backend is LocalAutonomyRepository only -- the graph/shadow backends
    (GraphAutonomyRepository, ShadowAutonomyRepository) were deleted 2026-07-30;
    see the comment above this function for why. Kept as a function (not a bare
    LocalAutonomyRepository() call at each site) so a real backend can be
    reintroduced here later without touching every caller.
    """
    return LocalAutonomyRepository(source_path=source_path)
