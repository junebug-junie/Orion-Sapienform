"""Tests for the run lifecycle: never strand a run in a non-terminal status.

Live incident these pin (2026-08-29): six runs sat in ``running/enriching``,
the oldest 21 hours, and **zero** runs were ``complete`` for the Orion model.
``fetch_latest_completed_run`` filters on ``status='complete'``, so the
concept-atlas ingest returned ``topic_foundry_no_completed_run`` and the graph
had no source run at all. Two were stranded by ordinary redeploys.
"""

from __future__ import annotations

import inspect
import logging
import pathlib
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
REAL_STATS = {"cluster_count": 7, "docs_generated": 394, "outlier_pct": 0.27}
REAL_ARTIFACTS = {"run_dir": "/data/runs/abc", "topics_summary": "/data/runs/abc/topics.json"}
ORIGINAL_COMPLETED_AT = datetime(2026, 8, 28, 5, 9, 31, tzinfo=timezone.utc)


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
        # Real values, not {}: update_run writes the WHOLE row, so a reaper
        # that forgot to carry these forward would silently destroy the
        # training stats and artifact paths of every run it recovered. With
        # empty fixtures nothing could ever catch that.
        "stats": REAL_STATS,
        "artifact_paths": REAL_ARTIFACTS,
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": ORIGINAL_COMPLETED_AT,
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


@pytest.mark.parametrize("status", ["running", "queued"])
def test_every_non_terminal_status_is_recovered(status: str) -> None:
    # Literals, deliberately. Parametrizing over NON_TERMINAL_STATUSES would
    # make this self-referential -- recovery_decision branches on that same
    # frozenset, so dropping "queued" from it would shrink the test with it
    # and still report green. `queued` strands just as permanently as
    # `running`: a run enqueued as a BackgroundTask that never started has no
    # worker either.
    assert recovery_decision(_run(status=status), segment_count=1, enriched_count=0) is not None


def test_the_policy_constant_and_the_reaper_sql_cannot_desynchronize() -> None:
    # list_non_terminal_runs builds its predicate from this set. If the two
    # ever disagree, every run in the missing status sits un-reaped forever
    # -- which is the incident, permanently.
    from app.storage import repository

    assert NON_TERMINAL_STATUSES == frozenset({"running", "queued"})
    assert TERMINAL_STATUSES == frozenset({"complete", "failed"})
    assert NON_TERMINAL_STATUSES.isdisjoint(TERMINAL_STATUSES)
    source = inspect.getsource(repository.list_non_terminal_runs)
    assert "NON_TERMINAL_STATUSES" in source, "the SQL no longer references the constant"
    # And the constant must be what the query actually uses. Merely importing
    # it while the SQL hardcodes the literals alongside is the desync this
    # guards against, so check for the literals directly.
    for literal in ('"running"', "'running'", '"queued"', "'queued'"):
        assert literal not in source, f"the SQL hardcodes {literal} again"


def test_recovery_decision_always_lands_on_a_terminal_status() -> None:
    for segments, enriched in ((0, 0), (1, 0), (5, 5)):
        decision = recovery_decision(_run(), segment_count=segments, enriched_count=enriched)
        assert decision.status in TERMINAL_STATUSES


# --- terminal_status_for_enrichment ---------------------------------------


def test_enrichment_restores_complete_for_a_healthy_run() -> None:
    # The original latch: _run_enrichment read status at entry and wrote it
    # back as the terminal state, pinning the run at "running" forever.
    assert terminal_status_for_enrichment({"status": "complete"}) == "complete"
    assert terminal_status_for_enrichment({"status": "running"}) == "complete"
    assert terminal_status_for_enrichment({"status": "queued"}) == "complete"
    assert terminal_status_for_enrichment({"status": None}) == "complete"


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
        # One real segment, so the success path actually enriches something.
        # With an empty list enriched_count stays 0 and the stage assertions
        # below would pass against an implementation that never ran at all.
        return [{"segment_id": str(uuid4()), "enriched_at": None}]

    monkeypatch.setattr(enrichment_module, "fetch_segments", _fetch_segments)
    monkeypatch.setattr(
        enrichment_module, "_enrich_segment", lambda seg, tax, enr, tmap: {"meaning": {}}
    )
    monkeypatch.setattr(
        enrichment_module, "update_segment_enrichment", lambda sid, **kw: None
    )
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
    # And it must not claim the pass succeeded: nothing was read at all, so
    # recording stage="enriched" would be the same class of lie as reporting
    # an empty projection as a success.
    assert writes[-1][1] != "enriched"


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


def _stub_reaper(monkeypatch, rows, *, segments=375, enriched=157):
    updated = []
    monkeypatch.setattr(
        "app.storage.repository.list_non_terminal_runs", lambda: rows, raising=False
    )
    # Keyword-only, matching the real count_segments(run_id, *, has_enrichment=...)
    # signature -- a looser stub would let a caller that passes it positionally
    # pass here and TypeError in production.
    monkeypatch.setattr(
        "app.storage.repository.count_segments",
        lambda run_id, *, has_enrichment=None, **kw: (enriched if has_enrichment else segments),
        raising=False,
    )
    monkeypatch.setattr(
        "app.storage.repository.update_run", lambda record: updated.append(record), raising=False
    )
    return updated


def test_reaper_closes_every_stranded_run(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_run(run_id=RUN_ID), _run(run_id=str(uuid4()), stage="training")]
    updated = _stub_reaper(monkeypatch, rows)
    assert recovery_module.recover_stranded_runs() == 2
    assert [r.status for r in updated] == ["complete", "complete"]


def test_reaper_preserves_everything_update_run_overwrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # update_run is a full-row SET, not a partial patch. A reaper that wrote
    # stats={} / artifact_paths={} / completed_at=now would look identical in
    # every other test here and would have permanently destroyed the training
    # artifacts and real completion times of all five runs recovered on
    # 2026-08-29 -- the rows the PR cites as proof it worked.
    updated = _stub_reaper(monkeypatch, [_run(run_id=RUN_ID)])
    assert recovery_module.recover_stranded_runs() == 1
    record = updated[0]
    assert record.stats == REAL_STATS
    assert record.artifact_paths == REAL_ARTIFACTS
    assert record.completed_at == ORIGINAL_COMPLETED_AT


def test_reaper_stamps_a_completion_time_only_when_there_was_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updated = _stub_reaper(monkeypatch, [_run(run_id=RUN_ID, completed_at=None)])
    recovery_module.recover_stranded_runs()
    assert updated[0].completed_at is not None


def test_reaper_writes_an_error_only_on_the_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A run served as `complete` must not carry an error string, or every
    # future reader wonders what is wrong with it.
    healthy = _stub_reaper(monkeypatch, [_run(run_id=RUN_ID)])
    recovery_module.recover_stranded_runs()
    assert healthy[0].error is None


def test_reaper_returns_zero_and_never_raises_when_the_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This runs in lifespan. It must never be the reason the service cannot
    # start, since the service is what lets an operator diagnose it.
    def _boom():
        raise RuntimeError("postgres unreachable")

    monkeypatch.setattr("app.storage.repository.list_non_terminal_runs", _boom, raising=False)
    assert recovery_module.recover_stranded_runs() == 0


def test_training_inline_enrichment_is_not_blocked_by_the_reentrancy_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _run_training sets status="running", stage="enriching" on its OWN run
    # and then calls run_enrichment_sync -- byte-for-byte the state the guard
    # rejects. Without owns_run, the guard silently turns the enrichment step
    # of EVERY training run into a no-op.
    run_row = _run(status="running", stage="enriching")
    writes = _stub_enrichment(monkeypatch, run_row)
    enrichment_module.run_enrichment_sync(
        UUID(RUN_ID), force=False, enricher="heuristic", limit=None
    )
    assert writes, "training's inline enrichment did no work at all"
    assert writes[0] == ("running", "enriching")
    assert writes[-1][1] == "enriched"


def test_the_background_task_path_does_not_set_owns_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The exemption must be the inline caller's alone. If enqueue_enrichment
    # ever passes owns_run=True the guard is dead for the path it was
    # written for -- the scheduler's every-tick trigger.
    captured = {}

    class _BG:
        def add_task(self, fn, *args, **kwargs):
            captured["kwargs"] = kwargs

    enrichment_module.enqueue_enrichment(
        _BG(), UUID(RUN_ID), force=False, enricher="heuristic", limit=None
    )
    assert captured["kwargs"].get("owns_run") is not True


def test_a_write_failure_in_the_finally_does_not_mask_the_real_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A raise inside the finally re-creates the original bug and hides it
    # better: it would replace the real exception AND leave the run pinned at
    # status="running", in code that now looks protected.
    run_row = _run(status="complete", stage="trained")
    _stub_enrichment(monkeypatch, run_row, fetch_segments_raises=True)

    # The FIRST update_run is the status="running" write, which happens
    # before the try. Only the second one -- the terminal restore inside the
    # finally -- is the case under test.
    calls = {"n": 0}

    def _boom(record):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("postgres went away during the terminal write")

    monkeypatch.setattr(enrichment_module, "update_run", _boom)
    with caplog.at_level(logging.ERROR, logger="topic-foundry.enrichment"):
        with pytest.raises(RuntimeError, match="database went away mid-enrichment"):
            enrichment_module._run_enrichment(
                UUID(RUN_ID), force=False, enricher="heuristic", limit=None
            )
    assert "enrichment_terminal_write_failed" in caplog.text


# --- POST /runs/{id}/enrich may only touch a completed run


@pytest.mark.parametrize("status", ["queued", "running", "failed", "", None])
def test_enrich_is_refused_for_a_run_that_is_not_complete(status) -> None:
    reason = recovery_module.enrich_refusal_reason({"status": status})
    assert reason is not None
    assert "not complete" in reason


def test_enrich_is_allowed_for_a_completed_run() -> None:
    assert recovery_module.enrich_refusal_reason({"status": "complete"}) is None


def test_the_enrich_route_actually_calls_the_predicate() -> None:
    # The predicate is only worth anything if the route uses it. Importing
    # app.routers.runs here would pull in the sklearn/joblib training stack,
    # so this reads the source instead -- the same trick the Hub scheduler
    # policy test uses, and it fails just as hard if the wiring is removed.
    source = pathlib.Path(__file__).resolve().parents[1] / "app" / "routers" / "runs.py"
    text = source.read_text()
    assert "enrich_refusal_reason(row)" in text
    assert "status_code=409" in text
