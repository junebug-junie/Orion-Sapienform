"""Regression: ConceptWorker._bus_synaptic_surprise_source's fail-open wiring.

`ORION_ACTION_OUTCOME_DB_URL` is optional for this service (it has no other
Postgres dependency) -- unconfigured must degrade to None on every call, not
raise or spin up a real connection attempt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings


async def _fetch_backend(*_, **__) -> dict:
    return {"success": False}


def _worker(action_outcome_db_url: str = "") -> ConceptWorker:
    cfg = ConceptSettings(action_outcome_db_url=action_outcome_db_url)
    return ConceptWorker(cfg, fetch_backend=_fetch_backend)


def test_surprise_source_none_when_db_url_unconfigured() -> None:
    worker = _worker(action_outcome_db_url="")
    assert worker._get_surprise_engine() is None
    assert worker._bus_synaptic_surprise_source() is None


def test_surprise_source_returns_real_value_when_configured() -> None:
    worker = _worker(action_outcome_db_url="postgresql://test:test@localhost/test")
    with patch(
        "orion.spark.concept_induction.bus_worker.latest_bus_synaptic_prediction_error",
        return_value=0.1675,
    ):
        assert worker._bus_synaptic_surprise_source() == 0.1675


def test_surprise_source_engine_cached_across_calls() -> None:
    worker = _worker(action_outcome_db_url="postgresql://test:test@localhost/test")
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        worker._get_surprise_engine()
        worker._get_surprise_engine()
    mock_create_engine.assert_called_once()


def test_surprise_source_fails_open_on_query_error() -> None:
    worker = _worker(action_outcome_db_url="postgresql://test:test@localhost/test")
    with patch(
        "orion.spark.concept_induction.bus_worker.latest_bus_synaptic_prediction_error",
        side_effect=RuntimeError("db unreachable"),
    ):
        assert worker._bus_synaptic_surprise_source() is None
