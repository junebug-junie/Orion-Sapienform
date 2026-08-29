"""Tests for the run lifecycle: never strand a run in a non-terminal status.

Live incident these pin (2026-08-29): six runs sat in ``running/enriching``,
the oldest 21 hours, and **zero** runs were ``complete`` for the Orion model.
``fetch_latest_completed_run`` filters on ``status='complete'``, so the
concept-atlas ingest returned ``topic_foundry_no_completed_run`` and the graph
had no source run at all. Two were stranded by ordinary redeploys.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.services import enrichment as enrichment_module
from app.services import run_recovery as recovery_module
from app.services.run_recovery import (
    NON_TERMINAL_STATUSES,
    TERMINAL_STATUSES,
    recovery_decision,
    terminal_status_for_enrichment,
)

RUN_ID = "11111111-1111-1111-1111-111111111111"


# A real RunSpecSnapshot payload -- the recovery path rebuilds a RunRecord
# from the stored row, so a stub `specs` would make these tests pass against
# an implementation that cannot round-trip a real one.
SPECS = {
    "dataset": {
        "dataset_id": "22222222-2222-2222-2222-222222222222",
        "name": "ds",
        "source_table": "chat_history_log",
        "id_column": "correlation_id",
        "time_column": "created_at",
        "text_columns": ["prompt", "response"],
        "created_at": "2026-08-29T00:00:00+00:00",
    },
    "windowing": {},
    "model": {"embedding_source_url": "http://vector-host:8320/embedding"},
}


def _run(**overrides):
    row = {
        "run_id": RUN_ID,
        "model_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "specs": SPECS,
        "spec_hash": "abc",
        "status": "running",
        "stage": "enriching",
        "stats": {},
        "artifact_paths": {},
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    row.update(overrides)
    return row


# --- recovery_decision ----------------------------------------------------


def test_a_stranded_run_with_segments_is_closed_as_complete() -> None:
    # This is the case that matters: d3adedab had 157 of 375 segments
    # enriched and was still unusable, because status='complete' is what
    # fetch_latest_completed_run filters on.
    decision = recovery_decision(_run(), segment_count=375, enriched_count=157)
    assert decision is not None
    assert decision.status == "complete"
    assert decision.stage == "enriched"


def test_stage_comes_from_the_segments_not_the_interrupted_stage_string() -> None:
    # The stage it was interrupted at is exactly what cannot be trusted after
    # a crash: this row says "enriching" but nothing was ever enriched.
    decision = recovery_decision(_run(stage="enriching"), segment_count=375, enriched_count=0)
    assert decision.stage == "trained"


def test_a_stranded_run_with_no_segments_is_failed_with_an_explicit_error() -> None:
    decision = recovery_decision(_run(stage="training"), segment_count=0, enriched_count=0)
    assert decision.status == "failed"
    assert decision.stage == "failed"
    assert "interrupted by a service restart" in decision.error


def test_a_recovered_complete_run_carries_no_error() -> None:
    # It is going to be served as a healthy run; an error string on it would
    # make every future reader wonder what is wrong with it.
    assert recovery_decision(_run(), segment_count=10, enriched_count=10).error is None


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_a_terminal_run_is_left_alone(status: str) -> None:
    assert recovery_decision(_run(status=status), segment_count=10, enriched_count=10) is None


@pytest.mark.parametrize("status", sorted(NON_TERMINAL_STATUSES))
def test_every_non_terminal_status_is_recovered(status: str) -> None:
    # `queued` strands just as permanently as `running` -- a run enqueued as
    # a BackgroundTask that never started has no worker either.
    assert recovery_decision(_run(status=status), segment_count=1, enriched_count=0) is not None


def test_recovery_decision_always_lands_on_a_terminal_status() -> None:
    for segments, enriched in ((0, 0), (1, 0), (5, 5)):
        decision = recovery_decision(_run(), segment_count=segments, enriched_count=enriched)
        assert decision.status in TERMINAL_STATUSES


# --- terminal_status_for_enrichment ---------------------------------------


def test_enrichment_never_restores_a_non_terminal_status() -> None:
    # The original latch: _run_enrichment read status at entry and wrote it
    # back as the terminal state, so a second pass starting during a first
    # one pinned the run at "running" forever.
    for status in ("running", "queued", "", None):
        assert terminal_status_for_enrichment({"status": status}) not in NON_TERMINAL_STATUSES


def test_enrichment_restores_complete_for_a_healthy_run() -> None:
    assert terminal_status_for_enrichment({"status": "complete"}) == "complete"
    assert terminal_status_for_enrichment({"status": "running"}) == "complete"


def test_enrichment_does_not_promote_a_failed_run_to_complete() -> None:
    assert terminal_status_for_enrichment({"status": "failed"}) == "failed"


# --- _run_enrichment lifecycle -------------------------------------------


def _stub_enrichment(monkeypatch, run_row, *, fetch_segments_raises=False):
    writes = []
    monkeypatch.setattr(enrichment_module, "fetch_run", lambda run_id: run_row)
    monkeypatch.setattr(
        enrichment_module, "update_run", lambda record: writes.append((record.status, record.stage))
    )
    monkeypatch.setattr(enrichment_module, "load_taxonomy", lambda spec: [])
    monkeypatch.setattr(enrichment_module, "_load_enrichment_spec", lambda row: _Spec())
    monkeypatch.setattr(enrichment_module, "_load_segment_text_map", lambda row: {})
    monkeypatch.setattr(enrichment_module, "_write_enrichment_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(enrichment_module, "_publish_enrich_complete", lambda *a, **k: None)
    monkeypatch.setattr(enrichment_module, "_generate_edges", lambda row: None)

    def _fetch_segments(run_id, has_enrichment=None):
        if fetch_segments_raises:
            raise RuntimeError("database went away mid-enrichment")
        return []

    monkeypatch.setattr(enrichment_module, "fetch_segments", _fetch_segments)
    return writes


class _Spec:
    aspect_taxonomy = "default"


def test_enrichment_restores_a_terminal_status_even_when_it_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole incident in one test: a raise between the "running" write and
    # the restore used to leave the run pinned at running forever.
    run_row = _run(status="complete", stage="trained")
    writes = _stub_enrichment(monkeypatch, run_row, fetch_segments_raises=True)
    with pytest.raises(RuntimeError):
        enrichment_module._run_enrichment(UUID(RUN_ID), force=False, enricher="heuristic", limit=None)
    assert writes[0] == ("running", "enriching")
    assert writes[-1][0] == "complete"
    assert writes[-1][0] not in NON_TERMINAL_STATUSES


def test_enrichment_restores_complete_on_the_success_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_row = _run(status="complete", stage="trained")
    writes = _stub_enrichment(monkeypatch, run_row)
    enrichment_module._run_enrichment(UUID(RUN_ID), force=False, enricher="heuristic", limit=None)
    assert writes[-1] == ("complete", "enriched")


def test_enrichment_refuses_a_second_pass_while_one_is_in_flight(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Reachable in normal operation: the Hub scheduler triggers enrichment
    # every tick. This is what wrote "running" back as the terminal status.
    run_row = _run(status="running", stage="enriching")
    writes = _stub_enrichment(monkeypatch, run_row)
    with caplog.at_level(logging.WARNING, logger="topic-foundry.enrichment"):
        enrichment_module._run_enrichment(UUID(RUN_ID), force=False, enricher="heuristic", limit=None)
    assert writes == []
    assert "enrichment_already_in_flight" in caplog.text


def test_enrichment_still_skips_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    run_row = _run(status="failed", stage="failed")
    writes = _stub_enrichment(monkeypatch, run_row)
    enrichment_module._run_enrichment(UUID(RUN_ID), force=False, enricher="heuristic", limit=None)
    assert writes == []


# --- the reaper -----------------------------------------------------------


def test_reaper_closes_every_stranded_run(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_run(run_id=RUN_ID), _run(run_id=str(uuid4()), stage="training")]
    updated = []
    monkeypatch.setattr(
        "app.storage.repository.list_non_terminal_runs", lambda: rows, raising=False
    )
    monkeypatch.setattr(
        "app.storage.repository.count_segments",
        lambda run_id, has_enrichment=None: (375 if has_enrichment is None else 157),
        raising=False,
    )
    monkeypatch.setattr(
        "app.storage.repository.update_run", lambda record: updated.append(record), raising=False
    )
    assert recovery_module.recover_stranded_runs() == 2
    assert [r.status for r in updated] == ["complete", "complete"]


def test_reaper_returns_zero_and_never_raises_when_the_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This runs in lifespan. It must never be the reason the service cannot
    # start, since the service is what lets an operator diagnose it.
    def _boom():
        raise RuntimeError("postgres unreachable")

    monkeypatch.setattr("app.storage.repository.list_non_terminal_runs", _boom, raising=False)
    assert recovery_module.recover_stranded_runs() == 0
