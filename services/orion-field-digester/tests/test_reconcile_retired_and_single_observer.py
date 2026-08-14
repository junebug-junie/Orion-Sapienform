"""Retired channel names and misplaced single-observer channels must not
survive a reconcile tick.

Both were found live on 2026-08-14, three weeks after the 2026-07-24 renames.
"""
from datetime import datetime, timedelta, timezone

from app.graph.lattice import load_lattice
from app.tensor.channels import RETIRED_NODE_CHANNELS, SINGLE_OBSERVER_NODE_CHANNELS
from app.tensor.reconcile import reconcile_field_state_with_lattice
from orion.schemas.field_state import FieldStateV1

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


def _lattice():
    return load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")


def _state(node_vectors, updated_at=None):
    return FieldStateV1(
        generated_at=NOW,
        tick_id="t",
        node_vectors=node_vectors,
        node_vector_updated_at=updated_at or {},
    )


def test_retired_names_are_pruned_from_a_lattice_node() -> None:
    """Live: all four lattice nodes carried execution_load frozen at 0.2672 --
    a plausible-looking reading with no producer behind it."""
    state = _state({"node:athena": {"cpu_pressure": 0.4, "execution_load": 0.2672,
                                    "transport_pressure": 0.00093}})
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    vec = out.node_vectors["node:athena"]
    assert "execution_load" not in vec
    assert "transport_pressure" not in vec
    assert vec["cpu_pressure"] == 0.4


def test_retired_names_are_pruned_from_an_OFF_LATTICE_node() -> None:
    """node:rpc_timeout is not in the lattice, so _ensure_node_vector never
    runs for it. This is the gap the old pruning had."""
    state = _state({"node:rpc_timeout": {"bus_health": 1.0, "transport_pressure": 0.5}})
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    assert out.node_vectors["node:rpc_timeout"] == {}


def test_single_observer_channels_are_pruned_from_an_off_lattice_node() -> None:
    """The live bug: rpc_timeout held delivery_confidence/stream_backlog_health
    at 0.5 with a 774s-old write while node:athena reported 1.0 fresh. Both are
    HIGHER_IS_BETTER, so min() let the stale 0.5 win.
    """
    state = _state({
        "node:rpc_timeout": {"delivery_confidence": 0.5, "stream_backlog_health": 0.5},
        "node:athena": {"delivery_confidence": 1.0, "stream_backlog_health": 1.0},
    })
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    assert out.node_vectors["node:rpc_timeout"] == {}
    # The declared owner keeps them.
    assert out.node_vectors["node:athena"]["delivery_confidence"] == 1.0
    assert out.node_vectors["node:athena"]["stream_backlog_health"] == 1.0


def test_pruning_also_drops_the_orphaned_write_timestamp() -> None:
    """A stamp for a channel that no longer exists is a dangling fact -- and
    the staleness rule reads these."""
    state = _state(
        {"node:rpc_timeout": {"delivery_confidence": 0.5, "execution_load": 0.2672}},
        {"node:rpc_timeout": {"delivery_confidence": NOW - timedelta(seconds=774),
                              "execution_load": NOW - timedelta(seconds=900)}},
    )
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    assert out.node_vector_updated_at["node:rpc_timeout"] == {}


def test_undeclared_but_not_retired_keys_still_survive() -> None:
    """_ensure_node_vector's preserve-undeclared-keys loop is load-bearing --
    it is how a channel survives between being written and being declared.
    Pruning is a NAMED set, never "anything not in NODE_CHANNELS"."""
    state = _state({"node:athena": {"some_brand_new_channel": 0.42}})
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    assert out.node_vectors["node:athena"]["some_brand_new_channel"] == 0.42


def test_retired_set_and_live_channels_do_not_overlap() -> None:
    """A name cannot be both retired and current. Guards against re-adding a
    retired name to NODE_CHANNELS and having it silently pruned every tick."""
    from app.tensor.channels import CAPABILITY_CHANNELS, NODE_CHANNELS

    live = set(NODE_CHANNELS) | set(CAPABILITY_CHANNELS)
    assert not (set(RETIRED_NODE_CHANNELS) & live)
    # Every replacement named in the map must itself be a real channel.
    for old, new in RETIRED_NODE_CHANNELS.items():
        assert new in live, f"{old} claims to be replaced by {new}, which does not exist"


def test_single_observer_owners_are_real_and_not_retired() -> None:
    for channel, owner in SINGLE_OBSERVER_NODE_CHANNELS.items():
        assert channel not in RETIRED_NODE_CHANNELS
        assert owner.startswith("node:")
