"""The absence sweep must actually run, including when nothing is reporting.

Closes the gap the 2026-08-29 circe outage exposed: circe was gone 00:01:16Z ->
00:47:04Z with `expected_online: true`, and the node-availability rule never fired
because `invoke_biometrics_pressure()` resolves its subject from the incoming event.
PR #1935 made absence *expressible*; this makes it *reachable*.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.substrate.biometrics_loop.ids import parse_biometrics_trace_id
from orion.substrate.biometrics_loop.pipeline import process_biometrics_grammar_events
from orion.substrate.biometrics_loop.pressure_organ import (
    build_absence_trigger_event,
    sweep_absent_nodes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
NOW = datetime(2026, 8, 29, 0, 47, 0, tzinfo=timezone.utc)
CIRCE_CAPS = ["local_llm_heavy", "local_llm_quick", "training", "batch_inference", "dream_batch"]


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _bio(**states: NodeBiometricsStateV1) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="proj_node_bio", generated_at=NOW, nodes=dict(states)
    )


def _node(node_id: str, *, minutes_silent: float, caps: list[str]) -> NodeBiometricsStateV1:
    return NodeBiometricsStateV1(
        node_id=node_id,
        expected_online=True,
        availability_status="online",
        last_seen_at=NOW - timedelta(minutes=minutes_silent),
        capabilities=caps,
    )


def test_synthetic_trigger_resolves_to_the_absent_node() -> None:
    event = build_absence_trigger_event("circe", now=NOW)
    assert parse_biometrics_trace_id(event.trace_id) == "circe"
    # Must be distinguishable from a real reported sample, or an absence signal
    # would be indistinguishable from a report in the trace record.
    assert event.provenance.source_component == "biometrics_absence_sweep"


def test_synthetic_trigger_is_deterministic_for_one_tick() -> None:
    a = build_absence_trigger_event("circe", now=NOW)
    b = build_absence_trigger_event("circe", now=NOW)
    assert a.event_id == b.event_id
    assert a.event_id != build_absence_trigger_event("athena", now=NOW).event_id


def test_absent_node_produces_capability_impacts_end_to_end(catalog: NodeCatalog) -> None:
    """The whole point: a silent circe now reaches the pressure projection."""
    node_bio = _bio(circe=_node("circe", minutes_silent=45, caps=CIRCE_CAPS))
    pressure = ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure", generated_at=NOW, nodes={}
    )
    saved: dict = {}

    absent = sweep_absent_nodes(node_bio=node_bio, catalog=catalog, now=NOW)
    assert absent == ["circe"]

    process_biometrics_grammar_events(
        events=[build_absence_trigger_event(n, now=NOW) for n in absent],
        catalog=catalog,
        load_node_bio=lambda: node_bio,
        save_node_bio=lambda p: saved.__setitem__("node_bio", p),
        load_pressure=lambda: pressure,
        save_pressure=lambda p: saved.__setitem__("pressure", p),
        save_receipt=lambda _r: None,
        save_emission=lambda _e: None,
        publish_accepted=None,
        enable_node_reducer=False,
        enable_organ=True,
        enable_pressure_reducer=True,
        now=NOW,
    )

    state = saved["pressure"].nodes["circe"]
    assert "availability" in state.active_pressures
    assert state.capability_impacts == [f"capability:{c}" for c in sorted(CIRCE_CAPS)]
    # A synthetic trigger is not a report: it must never refresh last_seen_at, or the
    # sweep would mark the node fresh and stop detecting its own outage.
    assert "node_bio" not in saved


def test_sweep_runs_before_the_no_events_early_return() -> None:
    """A total outage produces NO biometrics events, so a sweep placed after
    `if not events: return` would be skipped in exactly the incident it exists to
    catch. Asserts the ordering in the real worker source."""
    src = (
        REPO_ROOT / "services" / "orion-substrate-runtime" / "app" / "worker.py"
    ).read_text()
    sweep_at = src.index("absence_published = self._absence_sweep(now)")
    early_return_at = src.index("if not events:\n            return None, absence_published")
    assert sweep_at < early_return_at, "sweep must run before the early return"


def test_sweep_is_gated_by_the_organ_flag() -> None:
    src = (
        REPO_ROOT / "services" / "orion-substrate-runtime" / "app" / "worker.py"
    ).read_text()
    body = src[src.index("def _absence_sweep") : src.index("def _tick")]
    assert "if not self._settings.enable_biometrics_pressure_organ:" in body
    assert "biometrics_node_stale_after_sec" in body, "must use the configured threshold"
    assert "except Exception:" in body, "must not take down the ordinary tick"
