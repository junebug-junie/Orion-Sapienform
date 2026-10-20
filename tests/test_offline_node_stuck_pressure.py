"""An expected-offline node that stopped reporting must lose ALL its pressure, once.

Regression cover for the loop PR #2714's report found on atlas: `node_pressure_suppressed`
cleared only `strain` (its ROLE_TO_PRESSURE_KIND bucket), so an `availability` pressure
written while the node was still believed online could never clear (Rule B' needs the
node to report again). The suppressible sweep then re-selected the node on every tick,
and each pass let Rule C re-add `strain` from the node's frozen pressure_hints. The
sweep could never go quiet.

This drives the same path the runtime's `_absence_sweep` drives -- sweep, synthetic
trigger, organ, pressure reducer -- over many ticks, with a fake node so it does not
depend on any real host's catalog entry.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    ActiveNodePressureStateV1,
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.substrate.biometrics_loop.pipeline import process_biometrics_grammar_events
from orion.substrate.biometrics_loop.pressure_organ import (
    build_absence_trigger_event,
    invoke_biometrics_pressure,
    sweep_suppressible_nodes,
)

NOW = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
LAST_SEEN = NOW - timedelta(days=5)
# Shape of hecate's real frozen hints (substrate_node_biometrics_projection, 2026-10-20):
# `stability` sits above HINT_THRESHOLD forever, which is what let Rule C fire.
FROZEN_HINTS = {
    "gpu": 0.0,
    "strain": 0.087,
    "stability": 0.9996,
    "power_pressure": 0.548,
    "memory_pressure": 0.061,
}


@pytest.fixture
def catalog(tmp_path: Path) -> NodeCatalog:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "nodes:\n"
        "  nyx:\n"
        "    role: gpu_server\n"
        "    expected_online: false\n"
        "    capabilities:\n"
        "      local_llm_heavy: true\n"
    )
    return NodeCatalog.load(path)


def _node_bio() -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="proj_node_bio",
        generated_at=NOW,
        nodes={
            "nyx": NodeBiometricsStateV1(
                node_id="nyx",
                expected_online=False,
                last_seen_at=LAST_SEEN,
                pressure_hints=dict(FROZEN_HINTS),
                capabilities=["local_llm_heavy"],
                availability_status="offline_expected",
            )
        },
    )


def _stuck_pressure() -> ActiveNodePressureProjectionV1:
    # What atlas carried: availability from when it was believed online, plus strain.
    return ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=LAST_SEEN,
        nodes={
            "nyx": ActiveNodePressureStateV1(
                node_id="nyx",
                availability_status="stale",
                active_pressures=["availability", "strain"],
                pressure_score=1.0,
                last_updated_at=LAST_SEEN,
            )
        },
    )


def _run_sweeps(
    catalog: NodeCatalog,
    ticks: int,
    *,
    spacing_sec: float = 400,
    start: ActiveNodePressureProjectionV1 | None = None,
    **kwargs,
) -> tuple[ActiveNodePressureProjectionV1, list[int]]:
    node_bio = _node_bio()
    state = {"pressure": start or _stuck_pressure()}
    selected_per_tick: list[int] = []
    for i in range(ticks):
        # 400 s apart: past the reducer's 300 s merge window, so no candidate is
        # held back by dedup and the loop shows up if it exists.
        now = NOW + timedelta(seconds=spacing_sec * i)
        targets = sweep_suppressible_nodes(
            node_bio=node_bio,
            active_pressure=state["pressure"],
            catalog=catalog,
            now=now,
        )
        selected_per_tick.append(len(targets))
        if not targets:
            continue
        process_biometrics_grammar_events(
            events=[build_absence_trigger_event(n, now=now) for n in targets],
            catalog=catalog,
            load_node_bio=lambda: node_bio,
            save_node_bio=lambda _p: None,
            load_pressure=lambda: state["pressure"],
            save_pressure=lambda p: state.__setitem__("pressure", p),
            save_receipt=lambda _r: None,
            save_emission=lambda _e: None,
            publish_accepted=None,
            enable_node_reducer=False,
            now=now,
            **kwargs,
        )
    return state["pressure"], selected_per_tick


def test_suppression_clears_every_pressure_and_the_sweep_goes_quiet(catalog: NodeCatalog) -> None:
    pressure, selected = _run_sweeps(catalog, ticks=10)
    node = pressure.nodes["nyx"]
    assert node.active_pressures == []
    assert set(node.suppressed_pressures) == {"availability", "strain"}
    assert node.availability_status == "suppressed"
    assert node.pressure_score == 0.0
    # Self-terminating: selected once, then never again.
    assert selected[0] == 1
    assert sum(selected[1:]) == 0


def test_live_hecate_shape_score_only_is_cleared(catalog: NodeCatalog) -> None:
    """Review finding: live hecate (2026-10-20) was active [], suppressed
    ["strain"], pressure_score 1.0. With empty lists the sweep never selected it,
    so the score never cleared. The score now counts as something to clear."""
    start = _stuck_pressure()
    node = start.nodes["nyx"]
    node.active_pressures = []
    node.suppressed_pressures = ["strain"]
    node.availability_status = "suppressed"
    node.pressure_score = 1.0
    pressure, selected = _run_sweeps(catalog, ticks=5, start=start)
    assert pressure.nodes["nyx"].pressure_score == 0.0
    assert selected[0] == 1 and sum(selected[1:]) == 0


def test_settles_at_the_runtime_sweep_rate_through_the_merge_window(catalog: NodeCatalog) -> None:
    """The runtime sweeps about every second. A recent strain acceptance holds the
    suppression in the 300 s merge window; it must still land once it opens."""
    start = _stuck_pressure()
    start.nodes["nyx"].last_accepted_at = {"strain": NOW}
    pressure, selected = _run_sweeps(catalog, ticks=330, spacing_sec=1, start=start)
    node = pressure.nodes["nyx"]
    assert node.active_pressures == []
    assert node.pressure_score == 0.0
    assert selected[-1] == 0


def test_frozen_hints_do_not_re_add_strain_after_suppression(catalog: NodeCatalog) -> None:
    pressure = _stuck_pressure()
    pressure.nodes["nyx"].active_pressures = []
    pressure.nodes["nyx"].suppressed_pressures = ["availability", "strain"]
    emission = invoke_biometrics_pressure(
        trigger_event=build_absence_trigger_event("nyx", now=NOW),
        node_bio=_node_bio(),
        active_pressure=pressure,
        catalog=catalog,
        now=NOW,
    )
    roles = {ev.atom.semantic_role for ev in emission.candidate_events if ev.atom}
    assert roles == {"node_pressure_suppressed"}


def test_kill_switch_restores_frozen_hint_rules(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=build_absence_trigger_event("nyx", now=NOW),
        node_bio=_node_bio(),
        active_pressure=_stuck_pressure(),
        catalog=catalog,
        now=NOW,
        offline_frozen_hint_guard=False,
    )
    roles = {ev.atom.semantic_role for ev in emission.candidate_events if ev.atom}
    assert "node_pressure_reinforced" in roles


def test_fresh_expected_offline_node_still_reads_its_hints(catalog: NodeCatalog) -> None:
    """The guard is for a node that stopped reporting. One that is reporting (hecate
    switched on while catalogued expected_online: false) keeps Rules C/D/E."""
    node_bio = _node_bio()
    node_bio.nodes["nyx"].last_seen_at = NOW - timedelta(seconds=10)
    emission = invoke_biometrics_pressure(
        trigger_event=build_absence_trigger_event("nyx", now=NOW),
        node_bio=node_bio,
        active_pressure=_stuck_pressure(),
        catalog=catalog,
        now=NOW,
    )
    roles = {ev.atom.semantic_role for ev in emission.candidate_events if ev.atom}
    assert "node_pressure_reinforced" in roles
    assert "node_pressure_suppressed" not in roles
