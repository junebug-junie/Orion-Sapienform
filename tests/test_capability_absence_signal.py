"""Absence -> capability impact (Rule F) and the sweep that makes it reachable.

Regression cover for the 2026-08-29 circe outage: circe went dark 00:01:16Z ->
00:47:04Z with `expected_online: true` and five declared capabilities, and the
node-availability rule never fired once because
`invoke_biometrics_pressure()` resolves its subject from the *incoming* event.
See docs/superpowers/specs/2026-08-29-capability-absence-signal-design.md.
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
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1
from orion.substrate.biometrics_loop.pressure_organ import (
    DEFAULT_STALE_AFTER_SEC,
    invoke_biometrics_pressure,
    sweep_absent_nodes,
)
from orion.substrate.biometrics_loop.ids import parse_pressure_trace_id
from orion.substrate.biometrics_loop.emission_validator import (
    group_candidate_events_by_trace,
)
from orion.substrate.biometrics_loop.pressure_reducer import reduce_node_pressure_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 8, 29, 0, 47, 0, tzinfo=timezone.utc)

# circe's real declared-true capabilities in config/biometrics/node_catalog.yaml.
# `graphdb`/`postgres`/`hub` are declared false there and must NOT appear.
CIRCE_CAPS = [
    "local_llm_heavy",
    "local_llm_quick",
    "training",
    "batch_inference",
    "dream_batch",
]


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _bio(nodes: dict[str, NodeBiometricsStateV1]) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="proj_node_bio", generated_at=FIXED_TS, nodes=nodes
    )


def _state(
    node_id: str,
    *,
    expected_online: bool | None,
    last_seen_at: datetime | None,
    capabilities: list[str] | None = None,
) -> NodeBiometricsStateV1:
    return NodeBiometricsStateV1(
        node_id=node_id,
        expected_online=expected_online,
        availability_status="online",
        last_seen_at=last_seen_at,
        capabilities=capabilities or [],
    )


def _trigger(node_id: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id="gev_trigger",
        event_kind="atom_emitted",
        trace_id=f"biometrics.node:{node_id}:2026-08-29T00:47:00Z",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        provenance=GrammarProvenanceV1(
            source_service="orion-biometrics",
            source_component="biometrics_grammar_emit",
        ),
    )


def _roles(emission) -> list[str]:
    return [e.atom.semantic_role for e in emission.candidate_events if e.atom is not None]


# ---------------------------------------------------------------- sweep


def test_sweep_finds_the_node_that_stopped_reporting() -> None:
    """The real circe shape: fresh athena, 45-min-silent circe."""
    fresh = FIXED_TS - timedelta(seconds=2)
    silent = FIXED_TS - timedelta(minutes=45)
    bio = _bio(
        {
            "athena": _state("athena", expected_online=True, last_seen_at=fresh),
            "circe": _state("circe", expected_online=True, last_seen_at=silent),
        }
    )
    assert sweep_absent_nodes(node_bio=bio, now=FIXED_TS) == ["circe"]


def test_sweep_ignores_expected_offline_nodes() -> None:
    """atlas is decommissioned (`expected_online: false`); its permanent silence
    must never be surfaced as an outage."""
    bio = _bio(
        {
            "atlas": _state(
                "atlas",
                expected_online=False,
                last_seen_at=FIXED_TS - timedelta(days=9),
            )
        }
    )
    assert sweep_absent_nodes(node_bio=bio, now=FIXED_TS) == []


def test_sweep_flags_a_projection_node_with_no_last_seen_at() -> None:
    """`last_seen_at is None` for a node that IS in the projection counts as stale.

    Deliberately NOT claiming to cover the `prometheus` case. prometheus is
    catalogued `expected_online: true` with monitoring/logs/metrics and has never
    written an orion_biometrics row -- so it is absent from the projection entirely
    (live check 2026-08-29: the projection holds only atlas, circe, athena) and this
    function cannot see it. Catching never-reported nodes needs a catalog sweep;
    phase 2. See sweep_absent_nodes()'s docstring.
    """
    bio = _bio({"circe": _state("circe", expected_online=True, last_seen_at=None)})
    assert sweep_absent_nodes(node_bio=bio, now=FIXED_TS) == ["circe"]


def test_sweep_is_quiet_just_under_the_threshold() -> None:
    """Guards the boundary in the direction that matters: a node one second inside
    the window must not be reported, or every ordinary tick becomes an outage."""
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(seconds=DEFAULT_STALE_AFTER_SEC - 1),
            )
        }
    )
    assert sweep_absent_nodes(node_bio=bio, now=FIXED_TS) == []
    bio_over = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(seconds=DEFAULT_STALE_AFTER_SEC + 1),
            )
        }
    )
    assert sweep_absent_nodes(node_bio=bio_over, now=FIXED_TS) == ["circe"]


# ---------------------------------------------------------------- Rule F


def test_absent_node_emits_capability_impact(catalog: NodeCatalog) -> None:
    """Rule F. Before it existed, `node_capability_impact` was only reachable via
    Rule E's `gpu_hint >= 0.60` -- GPU saturation -- which a dead node can never
    produce, because a dead node reports no hints at all."""
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(minutes=45),
                capabilities=CIRCE_CAPS,
            )
        }
    )
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=bio,
        active_pressure=ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )
    roles = _roles(emission)
    assert "node_availability_concern" in roles
    assert "node_capability_absent" in roles
    # Exactly one, not one per capability: atom_id is {trace_id}:{semantic_role},
    # so per-capability events in one tick would collide on their own ids. The
    # reducer expands this single event into the real names.
    assert roles.count("node_capability_absent") == 1


def test_absent_node_with_no_declared_capabilities_stays_quiet(catalog: NodeCatalog) -> None:
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(minutes=45),
                capabilities=[],
            )
        }
    )
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=bio,
        active_pressure=ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )
    assert "node_capability_absent" not in _roles(emission)


def test_fresh_node_emits_no_capability_impact(catalog: NodeCatalog) -> None:
    """The control arm: same node, same capabilities, still reporting."""
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(seconds=2),
                capabilities=CIRCE_CAPS,
            )
        }
    )
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=bio,
        active_pressure=ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )
    assert "node_capability_absent" not in _roles(emission)


# ---------------------------------------------------------------- reducer


def test_reducer_records_real_capability_names(catalog: NodeCatalog) -> None:
    """End-to-end: organ -> reducer -> projection.

    Pins the actual defect this replaces. The arm used to append
    `f"capability:{pressure_kind}"` where pressure_kind is the constant
    "capability", so the only value it could ever produce was the literal
    "capability:capability". Asserting the real names is what makes that a
    failing test rather than a passing one.
    """
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(minutes=45),
                capabilities=CIRCE_CAPS,
            )
        }
    )
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=bio,
        active_pressure=ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )
    # The organ groups its candidates into one flat event list; the reducer wants
    # them re-grouped per trace, the same way the pipeline does it.
    grouped = group_candidate_events_by_trace(emission.candidate_events)
    projection, _receipt = reduce_node_pressure_candidates(
        candidates=grouped,
        projection=ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )
    impacts = projection.nodes["circe"].capability_impacts
    assert "capability:capability" not in impacts, "the pre-fix constant label leaked through"
    assert impacts == [f"capability:{c}" for c in sorted(CIRCE_CAPS)]


# ------------------------------------------------- trace-id collision (pre-existing)


def test_two_rules_for_one_node_survive_as_separate_traces(catalog: NodeCatalog) -> None:
    """Regression for a pre-existing drop, found while building Rule F.

    `build_pressure_candidate_events` derived trace_id from node+timestamp only, so
    every rule firing for the same node in the same tick shared one trace_id,
    `group_candidate_events_by_trace()` merged them, and the reducer -- one atom per
    trace -- kept only the first. This uses SHIPPING rules only (no Rule F): a fresh
    node with `gpu` hint 0.7 and a prior active pressure fires Rule C
    (node_pressure_reinforced) and Rule E (node_capability_impact) together.
    Before the fix both arrived in a single trace and the capability impact was
    discarded, which is a second reason `node_capability_impact` had 0 rows in
    `grammar_atoms` for its whole lifetime.
    """
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(seconds=2),  # fresh: Rule F cannot fire
                capabilities=["local_llm_heavy"],
            )
        }
    )
    bio.nodes["circe"].pressure_hints = {"gpu": 0.7}
    prior = ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={
            "circe": ActiveNodePressureStateV1(
                node_id="circe",
                availability_status="online",
                last_updated_at=FIXED_TS,
                active_pressures=["strain"],
            )
        },
    )
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=bio,
        active_pressure=prior,
        catalog=catalog,
        now=FIXED_TS,
    )
    roles = _roles(emission)
    assert "node_pressure_reinforced" in roles
    assert "node_capability_impact" in roles

    grouped = group_candidate_events_by_trace(emission.candidate_events)
    assert len(grouped) == 2, "both rules collapsed into one trace; the second is dropped"
    trace_ids = {t[0].trace_id for t in grouped}
    assert len(trace_ids) == 2
    # The node must still be recoverable from every trace id, or the reducer
    # rejects the trace outright (parse_pressure_trace_id -> profile.known).
    for trace_id in trace_ids:
        assert parse_pressure_trace_id(trace_id) == "circe"


# ------------------------------------------------- review findings 1-4


def _reduce(emission, catalog, prior=None):
    grouped = group_candidate_events_by_trace(emission.candidate_events)
    return reduce_node_pressure_candidates(
        candidates=grouped,
        projection=prior
        or ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )


def _saturated_and_dead(gpu: float = 0.9):
    """The crash-under-load shape: pressure_hints are never cleared when a node
    goes silent (node_reducer merges hints and only ever adds), so a GPU-heavy node
    that dies stays stale AND keeps gpu >= GPU_HINT_THRESHOLD -- firing Rule E and
    Rule F in the same tick."""
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(minutes=45),
                capabilities=CIRCE_CAPS,
            )
        }
    )
    bio.nodes["circe"].pressure_hints = {"gpu": gpu}
    return bio


def test_saturated_and_dead_node_emits_no_duplicate_event_ids(catalog: NodeCatalog) -> None:
    """Review finding 1. Rule E and Rule F both used semantic_role
    'node_capability_impact', so putting the role in the trace_id did not separate
    them: they collided anyway, one atom was silently dropped, and pipeline.py's
    `event_id in accepted` filter matched the duplicate twice and republished the
    same event_id on the accepted grammar channel."""
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=_saturated_and_dead(),
        active_pressure=ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
        ),
        catalog=catalog,
        now=FIXED_TS,
    )
    event_ids = [e.event_id for e in emission.candidate_events]
    assert len(event_ids) == len(set(event_ids)), "duplicate event_ids in one emission"

    grouped = group_candidate_events_by_trace(emission.candidate_events)
    assert len({t[0].trace_id for t in grouped}) == len(grouped)
    # Every emitted atom must survive grouping -- one atom per trace, none swallowed.
    atoms = [e for e in emission.candidate_events if e.atom is not None]
    assert len(atoms) == len(grouped)


def test_saturation_alone_does_not_impact_non_llm_capabilities(catalog: NodeCatalog) -> None:
    """Review finding 3. Rule E gates on `has_llm and gpu_hint >= 0.60`. Expanding
    it to every declared capability meant one routine GPU spike on a healthy,
    still-reporting node marked `training`, `dream_batch` and `batch_inference`
    impacted -- a false signal phase 3's capability-aware routing would consume."""
    bio = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(seconds=2),  # healthy, not stale
                capabilities=CIRCE_CAPS,
            )
        }
    )
    bio.nodes["circe"].pressure_hints = {"gpu": 0.65}
    projection, _ = _reduce(
        invoke_biometrics_pressure(
            trigger_event=_trigger("circe"),
            node_bio=bio,
            active_pressure=ActiveNodePressureProjectionV1(
                projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
            ),
            catalog=catalog,
            now=FIXED_TS,
        ),
        catalog,
    )
    impacts = projection.nodes["circe"].capability_impacts
    assert "capability:local_llm_heavy" in impacts
    assert "capability:local_llm_quick" in impacts
    for not_gpu_gated in ("training", "dream_batch", "batch_inference"):
        assert f"capability:{not_gpu_gated}" not in impacts


def test_absence_impacts_every_declared_capability(catalog: NodeCatalog) -> None:
    """The counterpart: the provider is gone, so everything it declares is gone."""
    projection, _ = _reduce(
        invoke_biometrics_pressure(
            trigger_event=_trigger("circe"),
            node_bio=_bio(
                {
                    "circe": _state(
                        "circe",
                        expected_online=True,
                        last_seen_at=FIXED_TS - timedelta(minutes=45),
                        capabilities=CIRCE_CAPS,
                    )
                }
            ),
            active_pressure=ActiveNodePressureProjectionV1(
                projection_id="proj_active_pressure", generated_at=FIXED_TS, nodes={}
            ),
            catalog=catalog,
            now=FIXED_TS,
        ),
        catalog,
    )
    assert projection.nodes["circe"].capability_impacts == [
        f"capability:{c}" for c in sorted(CIRCE_CAPS)
    ]


def test_recovery_clears_capability_impacts(catalog: NodeCatalog) -> None:
    """Review finding 2. capability_impacts had no removal path, so labels persisted
    forever and flowed into the concept graph via biometrics_ctx.py:102 -- the exact
    one-way-ratchet failure Rule B' itself exists to undo for `availability`."""
    prior = ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={
            "circe": ActiveNodePressureStateV1(
                node_id="circe",
                availability_status="stale",
                last_updated_at=FIXED_TS,
                active_pressures=["availability"],
                capability_impacts=[f"capability:{c}" for c in sorted(CIRCE_CAPS)],
            )
        },
    )
    # Fresh again, with a prior availability flag -> Rule B' fires.
    recovered = _bio(
        {
            "circe": _state(
                "circe",
                expected_online=True,
                last_seen_at=FIXED_TS - timedelta(seconds=2),
                capabilities=CIRCE_CAPS,
            )
        }
    )
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger("circe"),
        node_bio=recovered,
        active_pressure=prior,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert "node_availability_recovered" in _roles(emission)
    projection, _ = _reduce(emission, catalog, prior=prior)
    assert projection.nodes["circe"].capability_impacts == []
    assert projection.nodes["circe"].availability_status == "online"


def test_absence_and_saturation_use_separate_merge_buckets() -> None:
    """Review finding 4. Both roles shared pressure_kind='capability', so a
    saturation impact within merge_window_sec (300s) silently swallowed the first
    absence impact -- and 'saturated, then died' is the ordinary crash sequence."""
    from orion.substrate.biometrics_loop.pressure_reducer import ROLE_TO_PRESSURE_KIND

    assert (
        ROLE_TO_PRESSURE_KIND["node_capability_absent"]
        != ROLE_TO_PRESSURE_KIND["node_capability_impact"]
    )
