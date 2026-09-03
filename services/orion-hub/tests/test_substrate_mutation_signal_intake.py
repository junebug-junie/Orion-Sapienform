"""The scheduled mutation cycle must say WHY it got zero signals.

Live finding 2026-08-30: the cycle logged `signals_processed: 0` every 30
seconds for five weeks while 1,358 telemetry rows sat in the store, because its
required (invocation_surface, target_zone) pair was never jointly produced. All
20 `substrate_mutation_*` tables are at 0 rows as a direct result. An
unexplained zero is indistinguishable from a healthy idle cycle; these tests pin
the explanation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder
from scripts import api_routes


def _record(*, surface: str, zone: str, at: datetime) -> GraphReviewTelemetryRecordV1:
    return GraphReviewTelemetryRecordV1(
        invocation_surface=surface,  # type: ignore[arg-type]
        target_zone=zone,  # type: ignore[arg-type]
        anchor_scope="orion",
        subject_ref="entity:orion",
        selection_reason="signal-intake-test",
        selected_priority=50,
        execution_outcome="executed",
        runtime_duration_ms=1,
        selected_at=at,
    )


@pytest.fixture
def scheduler_env(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "false")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "false")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", "false")
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", SubstrateMutationStore())
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_SURFACES", {"routing": {"chat_reflective_lane_threshold": 0.5}})
    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (True, "supported"))
    monkeypatch.setattr(
        api_routes,
        "SUBSTRATE_MUTATION_SIGNAL_INTAKE",
        {"reason": "no_cycle_observed_yet", "consecutive_starved_cycles": 0},
    )

    def _install(store: GraphReviewTelemetryRecorder) -> None:
        monkeypatch.setattr(api_routes, "SUBSTRATE_REVIEW_TELEMETRY_STORE", store)

    return _install


def _live_shaped_store() -> GraphReviewTelemetryRecorder:
    """The live histogram in miniature. 6 operator_review/concept_graph rows and
    2 chat_reflective_lane/autonomy_graph rows: the intersection the scheduler
    needs is empty, exactly as in production."""
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    for i in range(6):
        store.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=i)))
    for i in range(2):
        store.record(_record(surface="chat_reflective_lane", zone="autonomy_graph", at=base + timedelta(minutes=10 + i)))
    return store


def test_live_starvation_is_named_zone_filter_rejected_all(scheduler_env) -> None:
    scheduler_env(_live_shaped_store())

    intake = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]

    assert intake["reason"] == "zone_filter_rejected_all"
    assert intake["starved"] is True
    # 8 rows exist, 6 clear the surface filter, 0 clear the zone filter.
    assert intake["store_total_records"] == 8
    assert intake["store_matched_surface"] == 6
    assert intake["before_zone_filter"] == 6
    assert intake["after_zone_filter"] == 0
    assert intake["allowed_zones"] == ["autonomy_graph"]
    # The report names the values that DO exist, so the mismatch is readable
    # without a database session.
    assert intake["zone_histogram"] == {"concept_graph": 6, "autonomy_graph": 2}
    assert intake["surface_histogram"] == {"operator_review": 6, "chat_reflective_lane": 2}


def test_an_empty_store_is_reported_as_empty_not_starved(scheduler_env) -> None:
    scheduler_env(GraphReviewTelemetryRecorder())

    intake = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]

    assert intake["reason"] == "store_empty"
    assert intake["starved"] is False
    assert intake["consecutive_starved_cycles"] == 0


def test_wrong_surface_only_is_distinguished_from_wrong_zone(scheduler_env) -> None:
    """Rows in the right zone but the wrong surface must not be blamed on the
    zone filter -- widening allowed_zones would not recover them."""
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    for i in range(3):
        store.record(_record(surface="chat_reflective_lane", zone="autonomy_graph", at=base + timedelta(minutes=i)))
    scheduler_env(store)

    intake = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]

    assert intake["reason"] == "surface_filter_rejected_all"
    assert intake["starved"] is True
    assert intake["store_total_records"] == 3
    assert intake["store_matched_surface"] == 0
    assert intake["before_zone_filter"] == 0


def test_an_autonomy_graph_row_alone_is_not_reported_as_healthy(scheduler_env) -> None:
    """As of 2026-09-03 "routing" (chat_reflective_lane_threshold) is parked:
    mutation_detectors.py filters every autonomy_graph-zoned signal out of
    from_review_telemetry() unconditionally (see PARKED_TELEMETRY_ZONES), so
    a row that clears every filter but sits in that zone can never produce a
    live signal any more. Formerly `test_a_satisfying_row_reports_healthy`,
    which asserted this exact row combination was "healthy" -- reporting that
    now would reproduce the same "unexplained zero looks fine" failure this
    whole module exists to prevent, just one level down (after_zone_filter
    positive, signals_processed still zero).
    """
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = _live_shaped_store()
    store.record(_record(surface="operator_review", zone="autonomy_graph", at=base + timedelta(minutes=20)))
    scheduler_env(store)

    intake = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]

    assert intake["reason"] == "matched_rows_only_in_parked_zones"
    assert intake["starved"] is False
    assert intake["after_zone_filter"] == 1
    assert intake["after_zone_filter_live"] == 0


def test_a_satisfying_row_in_a_live_zone_reports_healthy(scheduler_env, monkeypatch) -> None:
    """The positive case for "healthy" now needs a zone that isn't parked --
    self_relationship_graph (prompt_profile), reachable when the cognitive
    lane is enabled. autonomy_graph alone can never produce this any more
    (see the test above)."""
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "true")
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = _live_shaped_store()
    store.record(_record(surface="operator_review", zone="self_relationship_graph", at=base + timedelta(minutes=20)))
    scheduler_env(store)

    intake = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]

    assert intake["reason"] == "healthy"
    assert intake["starved"] is False
    assert intake["after_zone_filter"] == 1
    assert intake["after_zone_filter_live"] == 1


def test_consecutive_starved_cycles_accumulates_then_resets(scheduler_env, monkeypatch) -> None:
    """A lifetime counter cannot tell "starved since boot" from "starved once,
    months ago". The counter must climb while starved and reset on recovery."""
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "true")
    starved_store = _live_shaped_store()
    scheduler_env(starved_store)

    first = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]
    second = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]
    assert first["consecutive_starved_cycles"] == 1
    assert second["consecutive_starved_cycles"] == 2

    # self_relationship_graph, not autonomy_graph: the latter is a parked zone
    # now (see test_an_autonomy_graph_row_alone_is_not_reported_as_healthy)
    # and could never recover this cycle to "healthy".
    starved_store.record(
        _record(
            surface="operator_review",
            zone="self_relationship_graph",
            at=datetime(2026, 8, 30, 12, 30, tzinfo=timezone.utc),
        )
    )
    recovered = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]
    assert recovered["reason"] == "healthy"
    assert recovered["consecutive_starved_cycles"] == 0


def test_signal_intake_endpoint_serves_the_last_cycles_report(scheduler_env) -> None:
    scheduler_env(_live_shaped_store())
    api_routes.execute_substrate_mutation_scheduled_cycle()

    payload = api_routes.api_substrate_mutation_runtime_signal_intake()

    assert payload["data"]["reason"] == "zone_filter_rejected_all"
    assert payload["data"]["starved"] is True
    assert "source" in payload


def test_disabled_routing_proposals_is_not_reported_as_starvation(scheduler_env, monkeypatch) -> None:
    """max_signals drops to 0 when routing proposals are off. Building the store
    query with limit=0 raises ValidationError (the field is ge=1) and would take
    the cycle down inside the lock, so the zero budget is honoured directly --
    and a configured zero must never be labelled starvation."""
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", "false")
    scheduler_env(_live_shaped_store())

    summary = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]
    intake = summary["signal_intake"]

    assert summary["status"] == "completed"
    assert intake["reason"] == "signals_disabled"
    assert intake["starved"] is False
    assert intake["consecutive_starved_cycles"] == 0


def test_no_reason_is_unreachable(scheduler_env, monkeypatch) -> None:
    """Every reason the report can emit must be producible by some real cycle.
    A branch nothing can reach is exactly the dead scaffolding this patch exists
    to expose. "matched_rows_only_in_parked_zones" (added 2026-09-03 alongside
    the routing park) and "healthy" (now needing a non-parked zone -- see
    test_a_satisfying_row_in_a_live_zone_reports_healthy) are both covered."""
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    seen = set()

    scheduler_env(GraphReviewTelemetryRecorder())
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])

    wrong_surface = GraphReviewTelemetryRecorder()
    wrong_surface.record(_record(surface="chat_reflective_lane", zone="autonomy_graph", at=base))
    scheduler_env(wrong_surface)
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])

    scheduler_env(_live_shaped_store())
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])

    parked_only = _live_shaped_store()
    parked_only.record(_record(surface="operator_review", zone="autonomy_graph", at=base + timedelta(minutes=20)))
    scheduler_env(parked_only)
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])

    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "true")
    healthy = _live_shaped_store()
    healthy.record(_record(surface="operator_review", zone="self_relationship_graph", at=base + timedelta(minutes=20)))
    scheduler_env(healthy)
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "false")

    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", "false")
    scheduler_env(_live_shaped_store())
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", "true")

    seen.add(
        api_routes.execute_substrate_mutation_scheduled_cycle(telemetry_override=[])["summary"][
            "signal_intake"
        ]["reason"]
    )

    truncating = GraphReviewTelemetryRecorder()
    for i in range(4):
        truncating.record(_record(surface="operator_review", zone="autonomy_graph", at=base + timedelta(minutes=i)))
    for i in range(40):
        truncating.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=10 + i)))
    scheduler_env(truncating)
    seen.add(api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"])

    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "false")
    api_routes.execute_substrate_mutation_scheduled_cycle()
    seen.add(api_routes.SUBSTRATE_MUTATION_SIGNAL_INTAKE["reason"])
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "true")

    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (False, "mutation_store_degraded"))
    api_routes.execute_substrate_mutation_scheduled_cycle()
    seen.add(api_routes.SUBSTRATE_MUTATION_SIGNAL_INTAKE["reason"])

    assert seen == {
        "store_empty",
        "surface_filter_rejected_all",
        "zone_filter_rejected_all",
        "limit_truncated_usable_signals",
        "matched_rows_only_in_parked_zones",
        "healthy",
        "signals_disabled",
        "telemetry_override",
        "autonomy_disabled",
        "runtime_unsupported",
    }


def test_limit_truncation_is_not_blamed_on_the_zone_filter(scheduler_env) -> None:
    """40 usable signals sitting in the store while the cycle consumes zero, because
    the newest `limit` rows all landed in a zone this cycle rejects. Calling that a
    zone mismatch sends an operator to widen `allowed_zones`, which changes nothing.
    """
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    for i in range(40):  # usable, older
        store.record(_record(surface="operator_review", zone="autonomy_graph", at=base + timedelta(minutes=i)))
    for i in range(40):  # newer, wrong zone -- these win the 32-row slice
        store.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=100 + i)))
    scheduler_env(store)

    intake = api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]

    assert intake["reason"] == "limit_truncated_usable_signals"
    assert intake["starved"] is True
    assert intake["after_zone_filter"] == 0
    assert intake["usable_zone_rows_before_limit"] == 40


def test_a_bailing_cycle_does_not_leave_a_stale_healthy_report(scheduler_env, monkeypatch) -> None:
    """A healthy report must not outlive the cycle that produced it. Postgres
    degrading turns every later tick into a no-op that never reaches signal
    intake; the endpoint answering "healthy, not starved" for weeks after is the
    same silent-zero failure this patch exists to remove."""
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", "true")
    healthy = _live_shaped_store()
    healthy.record(
        _record(
            surface="operator_review",
            zone="self_relationship_graph",
            at=datetime(2026, 8, 30, 12, 30, tzinfo=timezone.utc),
        )
    )
    scheduler_env(healthy)
    assert api_routes.execute_substrate_mutation_scheduled_cycle()["summary"]["signal_intake"]["reason"] == "healthy"

    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (False, "mutation_store_degraded"))
    api_routes.execute_substrate_mutation_scheduled_cycle()

    published = api_routes.api_substrate_mutation_runtime_signal_intake()["data"]
    assert published["reason"] == "runtime_unsupported"
    assert published["detail"] == "mutation_store_degraded"
    assert published["starved"] is False


def test_an_override_cycle_does_not_clobber_the_live_report(scheduler_env) -> None:
    """Injected telemetry says nothing about live intake. Letting it publish would
    reset the starvation counter the report exists to maintain."""
    scheduler_env(_live_shaped_store())
    api_routes.execute_substrate_mutation_scheduled_cycle()
    api_routes.execute_substrate_mutation_scheduled_cycle()
    assert api_routes.SUBSTRATE_MUTATION_SIGNAL_INTAKE["consecutive_starved_cycles"] == 2

    api_routes.execute_substrate_mutation_scheduled_cycle(telemetry_override=[])

    live = api_routes.SUBSTRATE_MUTATION_SIGNAL_INTAKE
    assert live["reason"] == "zone_filter_rejected_all"
    assert live["consecutive_starved_cycles"] == 2


def test_the_report_is_stamped_with_the_tick_that_produced_it(scheduler_env) -> None:
    scheduler_env(_live_shaped_store())
    result = api_routes.execute_substrate_mutation_scheduled_cycle()
    intake = result["summary"]["signal_intake"]
    assert intake["tick_id"] == result["tick_id"]
    assert intake["observed_at"]
