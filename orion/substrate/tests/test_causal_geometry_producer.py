from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

import orion.substrate.causal_geometry_producer as producer_module
from orion.schemas.field_state import FieldEdgeV1
from orion.substrate.causal_geometry_engine import BUCKET_SECONDS
from orion.substrate.field_topology_learned_store import FieldTopologyLearnedWeightsStore

BASE_TS = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _points_for(values: np.ndarray, bucket_seconds: int = BUCKET_SECONDS) -> List[Tuple[datetime, float]]:
    return [(BASE_TS + timedelta(seconds=i * bucket_seconds), float(v)) for i, v in enumerate(values)]


def _divergent_channels(n_buckets: int = 200, seed: int = 7) -> Dict[str, List[Tuple[datetime, float]]]:
    rng = np.random.default_rng(seed)
    # Strongly correlated pair aliasing to a designed cap->cap edge's
    # channel_map, with observed_strength far from designed_weight (0.9 vs
    # 0.2) so it clears MIN_MEANINGFUL_DELTA and produces a real candidate.
    x = rng.normal(size=n_buckets)
    y = x * 0.95 + rng.normal(scale=0.05, size=n_buckets)
    return {
        "cap:transport": _points_for(x),
        "cap:orchestration": _points_for(y),
    }


def _fixture_topology() -> Dict:
    return {
        "schema_version": "field_lattice.v1.test",
        "edges": [
            {
                "source_id": "cap:transport",
                "target_id": "cap:orchestration",
                "edge_type": "capability_capability",
                "weight": 0.2,
                "channel_map": {"transport_pressure": "orchestration"},
            }
        ],
    }


def _field_edges() -> List[FieldEdgeV1]:
    return [
        FieldEdgeV1(
            source_id="cap:transport",
            target_id="cap:orchestration",
            edge_type="capability_capability",
            weight=0.2,
        )
    ]


def _run(
    monkeypatch,
    *,
    channels=None,
    topology=None,
    store=None,
    fetch_raises=False,
    propose_raises=False,
    persist_snapshot_mock=None,
):
    if fetch_raises:
        def _boom_fetch(*args, **kwargs):
            raise RuntimeError("postgres unreachable")

        monkeypatch.setattr(producer_module, "fetch_channels", _boom_fetch)
    else:
        monkeypatch.setattr(
            producer_module,
            "fetch_channels",
            lambda *a, **k: (channels if channels is not None else _divergent_channels(), {}),
        )
    monkeypatch.setattr(
        producer_module, "load_field_topology", lambda *a, **k: (topology if topology is not None else _fixture_topology())
    )

    if propose_raises:
        def _boom_propose(*args, **kwargs):
            raise RuntimeError("proposal construction failed")

        monkeypatch.setattr(producer_module, "propose_field_topology_patches", _boom_propose)

    # Persistence is a Postgres side effect unrelated to this module's own
    # measurement/proposal/dedup logic -- stub it out to a no-op success by
    # default so these tests don't need a real DB. Tests that care about the
    # persistence call itself pass their own mock via `persist_snapshot_mock`.
    mock_persist = persist_snapshot_mock or MagicMock(return_value={"ok": True, "error": None})
    monkeypatch.setattr(producer_module.causal_geometry_snapshot_store, "persist_snapshot", mock_persist)

    used_store = store if store is not None else FieldTopologyLearnedWeightsStore()
    result = producer_module.run_causal_geometry_production_cycle(
        postgres_uri="postgresql://unused/unused",
        topology_path="/unused/path.yaml",
        field_edges=_field_edges(),
        store=used_store,
        now=BASE_TS + timedelta(seconds=250 * BUCKET_SECONDS),
    )
    return result, used_store, mock_persist


def test_successful_cycle_creates_a_new_pending_proposal(monkeypatch) -> None:
    result, store, _mock = _run(monkeypatch)

    assert result["ok"] is True
    assert result["stage"] == "completed"
    assert result["proposals_created"] == 1
    assert result["proposals_skipped_pending_duplicate"] == 0
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].patch.target_ref == "cap:transport->cap:orchestration"


def test_dedup_skips_edge_with_existing_pending_proposal(monkeypatch) -> None:
    result_first, store, _mock = _run(monkeypatch)
    assert result_first["proposals_created"] == 1

    result_second, _store, _mock2 = _run(monkeypatch, store=store)

    assert result_second["ok"] is True
    assert result_second["proposals_created"] == 0
    assert result_second["proposals_skipped_pending_duplicate"] == 1
    # Still exactly one pending proposal for the edge, not two.
    assert len(store.list_pending()) == 1


def test_measurement_failure_degrades_without_raising(monkeypatch) -> None:
    result, _store, _mock = _run(monkeypatch, fetch_raises=True)

    assert result["ok"] is False
    assert result["stage"] == "measurement"
    assert "postgres unreachable" in result["error"]
    assert result["proposals_created"] == 0


def test_proposal_stage_failure_degrades_without_raising(monkeypatch) -> None:
    result, _store, _mock = _run(monkeypatch, propose_raises=True)

    assert result["ok"] is False
    assert result["stage"] == "proposal"
    assert "proposal construction failed" in result["error"]
    # Measurement stage did succeed, so snapshot_id is populated even though
    # the proposal stage failed.
    assert result["snapshot_id"] is not None


def test_insufficient_data_snapshot_yields_zero_candidates_not_an_error(monkeypatch) -> None:
    # Independent random series -> no significant edges -> insufficient_data.
    rng = np.random.default_rng(99)
    channels = {
        "cap:transport": _points_for(rng.normal(size=200)),
        "cap:orchestration": _points_for(rng.normal(size=200)),
    }
    result, _store, _mock = _run(monkeypatch, channels=channels)

    assert result["ok"] is True
    assert result["insufficient_data"] is True
    assert result["candidates_found"] == 0
    assert result["proposals_created"] == 0


def test_intra_cycle_duplicate_candidates_for_the_same_edge_enqueue_only_one_proposal(monkeypatch) -> None:
    """Regression: two different capability channels can alias to the same physical
    (source_id, target_id) edge (causal_geometry_engine.build_divergence()'s
    `#<capability_channel>` disambiguation suffix exists exactly for this case). If
    both aliased channels diverge meaningfully in the same cycle,
    propose_field_topology_patches() legitimately returns two candidates with the
    identical (stripped) target_ref -- the producer's dedup must catch the second
    one against the first *within* the same cycle, not just across cycles."""
    rng = np.random.default_rng(11)
    transport = rng.normal(size=200)
    orch_a = transport * 0.95 + rng.normal(scale=0.05, size=200)
    orch_b = transport * 0.9 + rng.normal(scale=0.08, size=200)
    channels = {
        "cap:transport": _points_for(transport),
        "cap:orch_a": _points_for(orch_a),
        "cap:orch_b": _points_for(orch_b),
    }
    topology = {
        "schema_version": "field_lattice.v1.test",
        "edges": [
            {
                "source_id": "cap:transport",
                "target_id": "cap:orchestration",
                "edge_type": "capability_capability",
                "weight": 0.1,
                "channel_map": {"orch_a": "orch_a", "orch_b": "orch_b"},
            }
        ],
    }

    result, store, _mock = _run(monkeypatch, channels=channels, topology=topology)

    assert result["ok"] is True
    pending = store.list_pending()
    assert len(pending) == 1, f"expected exactly one pending proposal, got {len(pending)}: {[p.patch.target_ref for p in pending]}"
    assert pending[0].patch.target_ref == "cap:transport->cap:orchestration"


def test_successful_cycle_persists_the_built_snapshot(monkeypatch) -> None:
    mock_persist = MagicMock(return_value={"ok": True, "error": None})
    result, _store, mock_persist = _run(monkeypatch, persist_snapshot_mock=mock_persist)

    assert result["ok"] is True
    mock_persist.assert_called_once()
    call_args = mock_persist.call_args.args
    assert call_args[0] == "postgresql://unused/unused"
    assert call_args[1].snapshot_id == result["snapshot_id"]


def test_summary_dict_reports_snapshot_persisted_true_on_success(monkeypatch) -> None:
    result, _store, _mock = _run(
        monkeypatch, persist_snapshot_mock=MagicMock(return_value={"ok": True, "error": None})
    )

    assert result["ok"] is True
    assert result["snapshot_persisted"] is True


def test_summary_dict_reports_snapshot_persisted_false_when_persist_fails(monkeypatch) -> None:
    result, _store, _mock = _run(
        monkeypatch, persist_snapshot_mock=MagicMock(return_value={"ok": False, "error": "boom"})
    )

    assert result["ok"] is True
    assert result["snapshot_persisted"] is False


def test_summary_dict_reports_snapshot_persisted_false_on_measurement_failure(monkeypatch) -> None:
    result, _store, _mock = _run(monkeypatch, fetch_raises=True)

    assert result["ok"] is False
    assert result["snapshot_persisted"] is False


def test_proposal_notes_contain_trial_status_after_successful_cycle(monkeypatch) -> None:
    result, store, _mock = _run(monkeypatch)

    assert result["ok"] is True
    pending = store.list_pending()
    assert len(pending) == 1
    notes = pending[0].notes
    assert any(note.startswith("trial_status:") for note in notes)
    # No replay corpus is registered for field_topology_weight_patch, so this
    # always short-circuits to inconclusive -- that is expected, not a bug.
    assert "trial_status:inconclusive" in notes


def test_module_never_imports_patch_applier() -> None:
    source = Path(producer_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "orion.substrate.mutation_apply"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "orion.substrate.mutation_apply"
