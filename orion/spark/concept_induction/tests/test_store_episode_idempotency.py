from __future__ import annotations

from datetime import datetime, timezone

from orion.spark.concept_induction.store import LocalProfileStore


def test_episode_run_processed_roundtrip(tmp_path) -> None:
    store = LocalProfileStore(str(tmp_path / "state.json"))
    assert store.is_episode_run_processed("run-1") is False
    store.mark_episode_run_processed("run-1", processed_at=datetime.now(timezone.utc))
    assert store.is_episode_run_processed("run-1") is True
    # a different run is unaffected
    assert store.is_episode_run_processed("run-2") is False


def test_episode_run_processed_ignores_empty_run_id(tmp_path) -> None:
    store = LocalProfileStore(str(tmp_path / "state.json"))
    store.mark_episode_run_processed("", processed_at=datetime.now(timezone.utc))
    assert store.is_episode_run_processed("") is False


def test_episode_run_processed_persists_across_instances(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    LocalProfileStore(path).mark_episode_run_processed("run-9", processed_at=datetime.now(timezone.utc))
    assert LocalProfileStore(path).is_episode_run_processed("run-9") is True
