from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.transport_metacog_gate import (
    build_transport_metacog_trigger_from_grammar_atom,
)
import app.transport_metacog_gate as transport_metacog_gate


def _grammar_atom(**overrides) -> dict:
    base = {
        "semantic_role": "rpc_transport_timeout",
        "text_value": "orion:cortex:exec:request:background",
        "summary": "RPC timeout: orion:cortex:exec:request:background -> reply after 60.0s",
    }
    base.update(overrides)
    return base


# --- Option A (pooled rpc_health timeouts): retired 2026-09-29 -------------


def test_pooled_snapshot_timeout_builder_is_gone():
    """Kill means kill: the pooled rpc_health timeout trigger was a coarser copy
    of the per-call rpc_transport_timeout atom (live: 675 of 677 timeouts it
    counted had a matching atom in the same 30 s window). It must not come back
    as a fallback; the atom owns timeouts while the baseline gate is log-only."""
    assert not hasattr(transport_metacog_gate, "build_transport_metacog_trigger_from_snapshot")


# --- Option C: grammar-atom-driven -----------------------------------------


def test_grammar_atom_wrong_role_fires_nothing():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        _grammar_atom(semantic_role="exec_turn_timeout"),
        correlation_id="corr-1",
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
    )
    assert trigger is None


def test_grammar_atom_non_dict_fires_nothing():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        None,  # type: ignore[arg-type]
        correlation_id="corr-1",
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
    )
    assert trigger is None


def test_grammar_atom_rpc_timeout_always_fires():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        _grammar_atom(),
        correlation_id="corr-1",
        zen_state="not_zen",
        pressure=0.3,
        recall_enabled=True,
    )
    assert trigger is not None
    assert trigger.trigger_kind == "transport"
    assert trigger.upstream["evidence_source"] == "rpc_transport_timeout_grammar"
    assert trigger.upstream["fired_conditions"] == ["rpc_timeout"]
    assert trigger.upstream["request_channel"] == "orion:cortex:exec:request:background"
    assert trigger.signal_refs == ["corr-1"]


def test_grammar_atom_no_correlation_id_still_fires():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        _grammar_atom(),
        correlation_id="",
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
    )
    assert trigger is not None
    assert trigger.signal_refs == []
