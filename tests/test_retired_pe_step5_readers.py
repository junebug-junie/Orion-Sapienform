"""PE suite rebuild step 5 (2026-11): node:substrate.bus_synaptic and
node:substrate.route are retired prediction-error nodes; step 6 (2026-11-13):
node:substrate.chat (successor node:prediction.chat_turn).

Killing the producer is not enough: the FalkorDB node keeps its last value
(and can be re-touched by the materializer), and history rows linger 7 days.
Every generic prediction-error reader skips the shared deny-list
(orion.substrate.prediction_error_freshness.RETIRED_PREDICTION_ERROR_NODE_IDS).
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.schemas.attention_candidate import world_first_fenced
from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS
from orion.substrate.prediction_error_freshness import (
    RETIRED_PREDICTION_ERROR_NODE_IDS,
    is_retired_prediction_error_node,
)

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
RETIRED = sorted(RETIRED_PREDICTION_ERROR_NODE_IDS)
LIVE = "node:substrate.execution"


def test_deny_list_is_exactly_the_retired_nodes() -> None:
    assert RETIRED_PREDICTION_ERROR_NODE_IDS == {
        "node:substrate.bus_synaptic", "node:substrate.route", "node:substrate.chat",
        "node:substrate.codebase",  # 2026-11-13: churn z, not a forecast
    }
    assert not is_retired_prediction_error_node(LIVE)
    assert not is_retired_prediction_error_node("node:prediction.chat_turn")
    assert not is_retired_prediction_error_node(None)


def test_active_inference_domains_are_the_two_live_ones() -> None:
    assert set(ACTIVE_INFERENCE_DOMAINS) == {"execution", "biometrics"}


@pytest.mark.parametrize("node_id", RETIRED)
def test_world_first_contests_fence_retired_nodes(node_id: str) -> None:
    assert world_first_fenced(node_id)
    assert not world_first_fenced(LIVE)


def test_pe_history_cache_drops_retired_rows() -> None:
    rows = [(nid, NOW - timedelta(minutes=5), 0.9) for nid in RETIRED] + [
        (LIVE, NOW - timedelta(minutes=5), 0.2)
    ]
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=NOW)
    assert set(cache.current()) == {LIVE}
    assert set(cache.magnitudes(now=NOW)) == {LIVE}
    for nid in RETIRED:
        assert cache.history(nid) == []


def _node(node_id: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        metadata={"prediction_error": value},
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=10)),
    )


def test_endogenous_curiosity_never_seeds_a_retired_node() -> None:
    from orion.substrate.endogenous_curiosity import _prediction_error_candidates

    nodes = [_node(nid, 1.0) for nid in RETIRED] + [_node(LIVE, 1.0)]
    out = _prediction_error_candidates(
        nodes, anchor_scope="orion", subject_ref=None, min_error=0.1, now=NOW
    )
    refs = {ref for c in out for ref in c.focal_node_refs}
    assert refs == {LIVE}


def test_dynamics_pressure_seed_ignores_retired_nodes() -> None:
    from orion.substrate.pressure import PressureConfig, prediction_error_pressure

    cfg = PressureConfig()
    for nid in RETIRED:
        assert prediction_error_pressure(_node(nid, 1.0), cfg, now=NOW) == 0.0
    assert prediction_error_pressure(_node(LIVE, 1.0), cfg, now=NOW) > 0.0


def test_retired_nodes_left_every_live_map() -> None:
    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & set(PREDICTION_ERROR_NATIVE_TARGETS)
    from orion.attention.self_readings import _WORDS
    from orion.attention.self_signal_outcomes import CIRCULAR_PAIRS

    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & set(_WORDS)
    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & {sig for sig, _ in CIRCULAR_PAIRS}
    from orion.grammar.act_ledger import ACT_ANSWERS_WORKSPACE_REFS

    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & {r for refs in ACT_ANSWERS_WORKSPACE_REFS.values() for r in refs}


def test_glossary_and_topology_have_no_retired_node_entries() -> None:
    glossary = yaml.safe_load((REPO / "config/field/field_channel_glossary.v1.yaml").read_text())
    nodes = {e.get("node") for e in glossary.get("channels") or [] if e.get("node")}
    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & nodes
    topology = yaml.safe_load((REPO / "config/field/orion_field_topology.v1.yaml").read_text())
    sources = {e["source_id"] for e in topology.get("edges") or []}
    assert not RETIRED_PREDICTION_ERROR_NODE_IDS & sources


def test_field_digester_prunes_the_same_ids() -> None:
    """RETIRED_PSEUDO_NODES lives behind a service boundary; parse it rather
    than import the digester app (Trap A: the prune is what keeps a retired
    node out of select_host_targets' 'any other node_vectors key' branch)."""
    src = (REPO / "services/orion-field-digester/app/tensor/channels.py").read_text()
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "RETIRED_PSEUDO_NODES":
            keys = {k.value for k in node.value.keys}
    assert RETIRED_PREDICTION_ERROR_NODE_IDS <= keys


def test_host_targets_would_admit_an_unpruned_retired_node() -> None:
    """Why the field-side prune matters (Trap A): select_host_targets treats
    any node_vectors key outside PREDICTION_ERROR_NATIVE_TARGETS as a host."""
    from orion.attention.field_attention.policy import load_attention_policy
    from orion.attention.field_attention.selectors import select_host_targets
    from orion.schemas.field_state import FieldStateV1

    policy = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="t",
        node_vectors={"node:athena": {"cpu_pressure": 0.5}, "node:substrate.route": {"prediction_error": 0.5}},
    )
    ids = {t.target_id for t in select_host_targets(field, policy, None)}
    assert "node:substrate.route" in ids  # hence RETIRED_PSEUDO_NODES must prune it
