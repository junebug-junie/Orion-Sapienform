from __future__ import annotations

import json
from pathlib import Path

from orion.spark.concept_induction.profile_repository import (
    build_concept_profile_repository,
    LocalConceptProfileRepository,
)
from orion.spark.concept_induction.store import LocalProfileStore


def _write_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "orion": {
                        "profile_id": "profile-1",
                        "subject": "orion",
                        "revision": 2,
                        "created_at": "2026-03-23T00:00:00+00:00",
                        "window_start": "2026-03-22T00:00:00+00:00",
                        "window_end": "2026-03-23T00:00:00+00:00",
                        "concepts": [],
                        "clusters": [],
                        "state_estimate": None,
                        "metadata": {},
                    }
                }
            }
        )
    )


def test_local_repository_matches_local_profile_store(tmp_path) -> None:
    store_path = tmp_path / "concepts.json"
    _write_fixture(store_path)

    store = LocalProfileStore(str(store_path))
    repository = LocalConceptProfileRepository(store_path=str(store_path))

    from_store = store.load("orion")
    from_repository = repository.get_latest("orion").profile

    assert from_store is not None
    assert from_repository is not None
    assert from_repository.model_dump(mode="json") == from_store.model_dump(mode="json")


def test_repository_distinguishes_empty_and_unavailable(tmp_path) -> None:
    missing_repo = LocalConceptProfileRepository(store_path=str(tmp_path / "missing.json"))
    missing = missing_repo.get_latest("orion")
    assert missing.availability == "unavailable"
    assert missing.unavailable_reason == "source_missing"

    existing_path = tmp_path / "concepts.json"
    _write_fixture(existing_path)
    existing_repo = LocalConceptProfileRepository(store_path=str(existing_path))

    present = existing_repo.get_latest("orion")
    assert present.availability == "available"

    absent_subject = existing_repo.get_latest("juniper")
    assert absent_subject.availability == "empty"
    assert absent_subject.unavailable_reason is None


def test_repository_factory_returns_local_repository(tmp_path) -> None:
    class Settings:
        store_path = str(tmp_path / "concepts.json")

    repository = build_concept_profile_repository(Settings())
    assert isinstance(repository, LocalConceptProfileRepository)
    assert repository.status().backend == "local"
