from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from orion.core.schemas.reasoning import AnchorScope, ArtifactType, ContradictionV1, ReasoningStatus
from orion.core.schemas.reasoning_io import ReasoningArtifactV1, ReasoningWriteRequestV1, ReasoningWriteResultV1


class ReasoningRepository(Protocol):
    def write_artifacts(self, request: ReasoningWriteRequestV1) -> ReasoningWriteResultV1:
        ...

    def list_latest(self, *, limit: int = 20) -> list[ReasoningArtifactV1]:
        ...

    def list_by_scope(self, scope: AnchorScope, *, limit: int = 50) -> list[ReasoningArtifactV1]:
        ...

    def list_by_type(self, artifact_type: ArtifactType, *, limit: int = 50) -> list[ReasoningArtifactV1]:
        ...


class InMemoryReasoningRepository:
    """Deterministic in-memory seam for Phase 2 materialization and tests."""

    def __init__(self) -> None:
        self._artifacts: list[ReasoningArtifactV1] = []
        self._idempotency_seen: set[str] = set()
        self._by_scope: dict[str, list[ReasoningArtifactV1]] = defaultdict(list)
        self._by_type: dict[str, list[ReasoningArtifactV1]] = defaultdict(list)
        self._by_subject_ref: dict[str, list[ReasoningArtifactV1]] = defaultdict(list)

    def write_artifacts(self, request: ReasoningWriteRequestV1) -> ReasoningWriteResultV1:
        if request.idempotency_key and request.idempotency_key in self._idempotency_seen:
            return ReasoningWriteResultV1(
                request_id=request.request_id,
                accepted=True,
                stored_count=0,
                deduped=True,
                status="deduped",
                message="idempotency key already processed",
            )

        for artifact in request.artifacts:
            self._artifacts.append(artifact)
            self._by_scope[artifact.anchor_scope].append(artifact)
            self._by_type[artifact.artifact_type].append(artifact)
            if artifact.subject_ref:
                self._by_subject_ref[artifact.subject_ref].append(artifact)

        if request.idempotency_key:
            self._idempotency_seen.add(request.idempotency_key)

        return ReasoningWriteResultV1(
            request_id=request.request_id,
            accepted=True,
            stored_count=len(request.artifacts),
            deduped=False,
            status="stored",
            artifact_ids=[a.artifact_id for a in request.artifacts],
        )

    def list_latest(self, *, limit: int = 20) -> list[ReasoningArtifactV1]:
        return list(reversed(self._artifacts[-max(0, limit) :]))

    def list_by_scope(self, scope: AnchorScope, *, limit: int = 50) -> list[ReasoningArtifactV1]:
        return list(reversed(self._by_scope.get(scope, [])[-max(0, limit) :]))

    def list_by_type(self, artifact_type: ArtifactType, *, limit: int = 50) -> list[ReasoningArtifactV1]:
        return list(reversed(self._by_type.get(artifact_type, [])[-max(0, limit) :]))

    def list_by_status(self, status: ReasoningStatus, *, limit: int = 50) -> list[ReasoningArtifactV1]:
        matches = [a for a in self._artifacts if a.status == status]
        return list(reversed(matches[-max(0, limit) :]))


    def get_by_id(self, artifact_id: str) -> ReasoningArtifactV1 | None:
        for artifact in reversed(self._artifacts):
            if artifact.artifact_id == artifact_id:
                return artifact
        return None

    def update_status(self, artifact_id: str, status: ReasoningStatus) -> bool:
        artifact = self.get_by_id(artifact_id)
        if artifact is None:
            return False
        artifact.status = status
        return True

    def list_by_subject_ref(self, subject_ref: str | None, *, limit: int = 100) -> list[ReasoningArtifactV1]:
        if not subject_ref:
            return []
        return list(reversed(self._by_subject_ref.get(subject_ref, [])[-max(0, limit) :]))

    def list_contradictions_for(self, artifact_id: str, *, unresolved_only: bool = True) -> list[ContradictionV1]:
        matches: list[ContradictionV1] = []
        for artifact in self._artifacts:
            if isinstance(artifact, ContradictionV1) and artifact_id in artifact.involved_artifact_ids:
                if unresolved_only and artifact.resolution_status == "resolved":
                    continue
                matches.append(artifact)
        return matches
