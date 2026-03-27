from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from orion.core.schemas.concept_induction import ConceptProfile

from .settings import DEFAULT_CONCEPT_STORE_PATH, ConceptSettings, get_settings
from .store import LocalProfileStore


AvailabilityKind = Literal["available", "empty", "unavailable"]


@dataclass(frozen=True)
class ConceptProfileRepositoryStatus:
    backend: str
    source_path: str
    placeholder_default_in_use: bool
    source_available: bool


@dataclass(frozen=True)
class ConceptProfileLookupV1:
    subject: str
    profile: ConceptProfile | None
    availability: AvailabilityKind
    unavailable_reason: str | None = None


class ConceptProfileRepository(Protocol):
    def get_latest(self, subject: str) -> ConceptProfileLookupV1:
        ...

    def list_latest(self, subjects: Sequence[str]) -> list[ConceptProfileLookupV1]:
        ...

    def status(self) -> ConceptProfileRepositoryStatus:
        ...


class LocalConceptProfileRepository:
    """Repository seam adapter backed by LocalProfileStore."""

    def __init__(self, *, store_path: str) -> None:
        self._store_path = store_path
        self._store = LocalProfileStore(store_path)

    def status(self) -> ConceptProfileRepositoryStatus:
        return ConceptProfileRepositoryStatus(
            backend="local",
            source_path=self._store_path,
            placeholder_default_in_use=self._store_path == DEFAULT_CONCEPT_STORE_PATH,
            source_available=Path(self._store_path).exists(),
        )

    def get_latest(self, subject: str) -> ConceptProfileLookupV1:
        try:
            profile = self._store.load(subject)
        except Exception:
            return ConceptProfileLookupV1(
                subject=subject,
                profile=None,
                availability="unavailable",
                unavailable_reason="read_error",
            )

        if profile is not None:
            return ConceptProfileLookupV1(subject=subject, profile=profile, availability="available")

        if not Path(self._store_path).exists():
            return ConceptProfileLookupV1(
                subject=subject,
                profile=None,
                availability="unavailable",
                unavailable_reason="source_missing",
            )

        return ConceptProfileLookupV1(subject=subject, profile=None, availability="empty")

    def list_latest(self, subjects: Sequence[str]) -> list[ConceptProfileLookupV1]:
        return [self.get_latest(subject) for subject in subjects]


def build_concept_profile_repository(settings: ConceptSettings | None = None) -> ConceptProfileRepository:
    cfg = settings or get_settings()
    return LocalConceptProfileRepository(store_path=cfg.store_path)
