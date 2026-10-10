"""Curiosity tick: link-accepted seeds (decision 2) + seed neighborhood attach (patch 1).

Real FrontierCuriosityEvaluator over an in-memory graph that holds one accepted reading
link; the Postgres store is a mock that returns the journal row the real SQL would
(the SQL itself is proven in orion/substrate/tests/test_link_accepted_seeds_pg.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.store import EndogenousCuriosityPersistResult  # noqa: E402
from app.worker import BiometricsSubstrateWorker  # noqa: E402
from orion.core.schemas.cognitive_substrate import AssertionNodeV1  # noqa: E402
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1  # noqa: E402
from orion.substrate.evals.neighborhood_fixture import PROVENANCE, TEMPORAL, concept, edge, graph  # noqa: E402
from orion.substrate.link_accepted_seeds import projection_edge_id  # noqa: E402

CLAIM = "assertion-reading-1"
EDGE = projection_edge_id(CLAIM)
LEGACY_KEYS = {
    "signal_id", "signal_type", "anchor_scope", "subject_ref", "target_zone", "task_type_candidate",
    "focal_node_refs", "focal_edge_refs", "signal_strength", "evidence_summary", "confidence", "notes",
}


def _proposed(node_id):
    return concept(node_id).model_copy(update={"promotion_state": "proposed"})


def _graph(*, claim_state="provisional", extra_nodes=(), extra_edges=()):
    claim = AssertionNodeV1(node_id=CLAIM, anchor_scope="world", promotion_state=claim_state, temporal=TEMPORAL,
                            provenance=PROVENANCE, predicate="associated_with", statement_key="k|1",
                            statement_text="a reading claim", revision=1)
    link = edge(EDGE, "read-a", "read-b", edge_role="semantic_projection", assertion_id=CLAIM,
                assertion_revision=1)
    return graph([_proposed("read-a"), _proposed("read-b"), claim, *extra_nodes], [link, *extra_edges])


def _row(assertion_id=CLAIM, subject="read-a", obj="read-b", revision=1):
    return {"assertion_id": assertion_id, "revision": revision, "state": "provisional",
            "subject_node_id": subject, "object_node_id": obj, "predicate": "associated_with",
            "statement_text": "the reading says a goes with b",
            "edge_ids": ["edge-structure", projection_edge_id(assertion_id)], "materialized_at": None}


def _scored(strength, node="node:substrate.chat"):
    return FrontierInvocationSignalV1(
        signal_type="curiosity_candidate", anchor_scope="orion", subject_ref="entity:orion",
        target_zone="concept_graph", task_type_candidate="evidence_gap_scan", focal_node_refs=[node],
        signal_strength=strength, confidence=0.7, evidence_summary=f"scored {strength}",
        notes=["endogenous_seed", "source:prediction_error"])


def _worker(monkeypatch, *, neighborhood=True, link_seeds=True, cap=None, rows=()):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_ENDOGENOUS_CURIOSITY_ENABLED", "true")
    monkeypatch.setenv("ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH", "false")
    monkeypatch.setenv("ORION_ENDOGENOUS_CURIOSITY_SEED_NEIGHBORHOOD_ENABLED", str(neighborhood).lower())
    monkeypatch.setenv("ORION_ENDOGENOUS_CURIOSITY_LINK_SEEDS_ENABLED", str(link_seeds).lower())
    if cap is not None:
        monkeypatch.setenv("ORION_ENDOGENOUS_CURIOSITY_LINK_SEED_CAP", str(cap))
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    worker._store = MagicMock()
    worker._store.load_attention_broadcast.return_value = None
    worker._store.load_chat_session_projection.return_value = None
    worker._store.load_latest_system_one_appraisal.return_value = None
    worker._store.load_unseeded_accepted_reading_links.return_value = list(rows)
    worker._store.save_endogenous_curiosity_candidates.return_value = EndogenousCuriosityPersistResult(
        candidate_set_id="curiosity-test", gate_lineage_persisted=True)
    return worker


def _tick(worker, graph_store, scored):
    with patch("orion.substrate.graphdb_store.build_substrate_store_from_env", return_value=graph_store), \
            patch("orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
                  return_value=list(scored)):
        worker._endogenous_curiosity_tick()
    call = worker._store.save_endogenous_curiosity_candidates.call_args
    assert call is not None, "tick stored nothing"
    signals = call.args[0]
    return [s.model_dump(mode="json") for s in signals], dict(call.kwargs.get("gate") or {})


def _link_rows(stored):
    return [s for s in stored if "source:reading_link_accepted" in s["notes"]]


def test_accepted_claim_seed_is_stored_with_the_link_as_its_focal_edge(monkeypatch):
    stored, gate = _tick(_worker(monkeypatch, rows=[_row()]), _graph(), [_scored(0.9)])
    (link,) = _link_rows(stored)
    assert link["focal_node_refs"] == ["read-a", "read-b"]
    assert link["focal_edge_refs"] == [EDGE]
    assert link["projection_endpoint_node_refs"] == ["read-a", "read-b"]
    assert "boundary_edge_refs" not in link  # empty -> omitted
    assert link["signal_strength"] == 0.0 and "strength:unscored_event" in link["notes"]
    assert f"link_assertion:{CLAIM}@1" in link["notes"]
    assert gate["link_seeds"]["minted"] == 1 and gate["link_seeds"]["keys"] == [f"link_assertion:{CLAIM}@1"]
    assert gate["neighborhood"]["nonempty"] == 1 and gate["neighborhood"]["internal_edges"] == 1
    # The scored seed on an edgeless organ node is stored exactly as before.
    assert set(stored[0]) == LEGACY_KEYS and stored[0]["focal_edge_refs"] == []


@pytest.mark.parametrize("state", ["rejected", "deprecated"])
def test_a_rejected_or_deprecated_claim_attaches_no_edge(monkeypatch, state):
    # The SQL never returns these (PG test); even if a stale row slipped through,
    # the graph read refuses an unaccepted projection.
    stored, gate = _tick(_worker(monkeypatch, rows=[_row()]), _graph(claim_state=state), [_scored(0.9)])
    (link,) = _link_rows(stored)
    assert link["focal_edge_refs"] == [] and "projection_endpoint_node_refs" not in link
    assert gate["neighborhood"]["nonempty"] == 0
    assert gate["neighborhood"]["degraded_reasons"].get("focal_unavailable_or_filtered", 0) >= 1


def test_no_journal_rows_mints_nothing(monkeypatch):
    stored, gate = _tick(_worker(monkeypatch, rows=[]), _graph(), [_scored(0.9)])
    assert _link_rows(stored) == [] and gate["link_seeds"]["minted"] == 0


def test_duplicate_rows_for_one_assertion_mint_one_seed(monkeypatch):
    stored, _ = _tick(_worker(monkeypatch, rows=[_row(), _row()]), _graph(), [_scored(0.9)])
    assert len(_link_rows(stored)) == 1


def test_per_tick_cap_and_hard_ceiling(monkeypatch):
    rows = [_row(assertion_id=f"assertion-{i}", subject=f"r{i}", obj=f"r{i + 1}") for i in range(8)]
    stored, gate = _tick(_worker(monkeypatch, rows=rows, cap=2), _graph(), [_scored(0.9)])
    assert len(_link_rows(stored)) == 2 and gate["link_seeds"]["cap"] == 2
    stored, gate = _tick(_worker(monkeypatch, rows=rows, cap=50), _graph(), [_scored(0.9)])
    assert len(_link_rows(stored)) == 4 and gate["link_seeds"]["cap"] == 4


def test_link_seeds_never_displace_scored_seeds(monkeypatch):
    scored = [_scored(0.95 - i * 0.05, node=f"node:substrate.n{i}") for i in range(8)]
    rows = [_row(assertion_id=f"assertion-{i}", subject=f"r{i}", obj=f"r{i + 1}") for i in range(2)]
    stored, _ = _tick(_worker(monkeypatch, rows=rows), _graph(), scored)
    assert [s["signal_id"] for s in stored[:8]] == [s.signal_id for s in scored]
    assert len(_link_rows(stored)) == 2 and len(stored) == 10


def test_ranking_and_decision_unchanged_with_both_features_on_vs_off(monkeypatch):
    scored = [_scored(0.8, "node:substrate.chat"), _scored(0.6, "node:substrate.execution")]
    off_rows, off_gate = _tick(_worker(monkeypatch, neighborhood=False, link_seeds=False), _graph(), scored)
    on_rows, on_gate = _tick(_worker(monkeypatch, rows=[_row()]), _graph(), scored)
    decision = ("evaluator_outcome", "evaluator_task", "bounded_context_reason", "seed_count")
    assert {k: off_gate[k] for k in decision[:3]} == {k: on_gate[k] for k in decision[:3]}
    assert off_gate["evaluator_outcome"] == "invoke"
    scored_ids = [s.signal_id for s in scored]
    assert [r["signal_id"] for r in off_rows if r["signal_id"] in scored_ids] == scored_ids
    assert [r["signal_id"] for r in on_rows if r["signal_id"] in scored_ids] == scored_ids
    # Same scored rows, key for key, with the features on (their organ nodes have no links).
    assert [r for r in on_rows if r["signal_id"] in scored_ids] == [r for r in off_rows if r["signal_id"] in scored_ids]


def test_neighborhood_alone_does_not_change_stored_order_or_decision(monkeypatch):
    scored = [_scored(0.8, "read-a"), _scored(0.6, "node:substrate.execution")]
    off_rows, off_gate = _tick(_worker(monkeypatch, neighborhood=False, link_seeds=False), _graph(), scored)
    on_rows, on_gate = _tick(_worker(monkeypatch, neighborhood=True, link_seeds=False), _graph(), scored)
    assert [r["signal_id"] for r in on_rows] == [r["signal_id"] for r in off_rows]
    assert off_gate["evaluator_outcome"] == on_gate["evaluator_outcome"]
    # A scored seed on a linked reading concept now carries the link as a boundary edge.
    assert on_rows[0]["boundary_edge_refs"] == [EDGE] and on_rows[0]["neighbor_node_refs"] == ["read-b"]
    assert on_rows[0]["projection_endpoint_node_refs"] == ["read-a", "read-b"]
    assert "boundary_edge_refs" not in off_rows[0]


def test_flags_off_rows_and_gate_have_the_pre_change_shape(monkeypatch):
    stored, gate = _tick(_worker(monkeypatch, neighborhood=False, link_seeds=False), _graph(),
                         [_scored(0.9, "read-a")])
    assert all(set(s) == LEGACY_KEYS for s in stored)
    assert "neighborhood" not in gate and "link_seeds" not in gate
    worker = _worker(monkeypatch, neighborhood=False, link_seeds=False)
    _tick(worker, _graph(), [_scored(0.9)])
    worker._store.load_unseeded_accepted_reading_links.assert_not_called()


def test_link_seed_alone_runs_the_evaluator_and_never_invokes(monkeypatch):
    stored, gate = _tick(_worker(monkeypatch, rows=[_row()]), _graph(), [])
    assert len(_link_rows(stored)) == 1
    assert gate["evaluator_outcome"] == "noop"  # strength 0.0 < invoke threshold


def test_legacy_edges_are_dropped_and_counted(monkeypatch):
    stored, gate = _tick(
        _worker(monkeypatch, link_seeds=False),
        _graph(extra_nodes=[concept("canon-1"), concept("canon-2")], extra_edges=[edge("legacy-1", "canon-1", "canon-2")]),
        [_scored(0.9, "canon-1")])
    assert "boundary_edge_refs" not in stored[0] and stored[0]["focal_edge_refs"] == []
    assert gate["neighborhood"]["legacy_edges_excluded"] == 1


def test_a_failing_read_leaves_the_seed_unchanged_and_the_tick_completes(monkeypatch):
    class Broken:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def read_neighborhood(self, request):
            raise RuntimeError("falkor down")

    stored, gate = _tick(_worker(monkeypatch, rows=[_row()]), Broken(_graph()), [_scored(0.9)])
    (link,) = _link_rows(stored)
    assert link["focal_edge_refs"] == [] and set(link) == LEGACY_KEYS
    assert gate["neighborhood"]["degraded_reasons"]["unavailable:RuntimeError"] == 2


def test_journal_read_failure_mints_nothing_and_says_why(monkeypatch):
    worker = _worker(monkeypatch)
    worker._store.load_unseeded_accepted_reading_links.side_effect = RuntimeError("pg down")
    stored, gate = _tick(worker, _graph(), [_scored(0.9)])
    assert _link_rows(stored) == [] and gate["link_seeds"]["error"] == "RuntimeError"


def test_system_one_veto_path_also_stores_link_seeds_with_edges(monkeypatch):
    worker = _worker(monkeypatch, rows=[_row()])
    veto = MagicMock(admit_evaluator=False, gate_result="system_one_noop", frame_id="f", selected_level=0)
    veto.to_telemetry.return_value = {"gate_result": "system_one_noop", "admit_evaluator": False}
    with patch("orion.substrate.system_one_access.decide_curiosity_admission", return_value=veto):
        stored, gate = _tick(worker, _graph(), [_scored(0.9)])
    (link,) = _link_rows(stored)
    assert link["focal_edge_refs"] == [EDGE]
    assert gate["neighborhood"]["nonempty"] == 1 and gate["link_seeds"]["minted"] == 1


def test_reading_actor_matches_the_reading_producer():
    from orion.substrate.link_accepted_seeds import READING_PROPOSAL_ACTORS
    from orion.world_pulse_read.assertions import READING_ACTOR

    assert READING_PROPOSAL_ACTORS == (READING_ACTOR,)
