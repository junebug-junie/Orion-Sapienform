from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from causal_dag_empirical_verification import (  # type: ignore
    _live_organ_id_for_service,
    compute_causal_dag_diff,
)


def _entry(service: str, causal_parent_organs: list[str]) -> dict:
    return {"service": service, "causal_parent_organs": causal_parent_organs}


def test_live_organ_id_strips_orion_prefix() -> None:
    assert _live_organ_id_for_service("orion-cortex-exec") == "cortex-exec"
    assert _live_organ_id_for_service("orion-recall") == "recall"


def test_live_organ_id_returns_none_for_non_service_strings() -> None:
    assert _live_organ_id_for_service("orion library (orion/journaler/)") is None
    assert _live_organ_id_for_service("") is None


def test_registry_edge_confirmed_by_real_live_evidence() -> None:
    registry = {
        "biometrics": _entry("orion-biometrics", []),
        "equilibrium": _entry("orion-equilibrium-service", ["biometrics"]),
    }
    # Deliberately mismatched (wrong target organ) to prove the confirmed
    # bucket requires a real match, not just presence of any live edges.
    result = compute_causal_dag_diff(registry, live_edges=[("biometrics", "landing-pad")])
    assert result["confirmed"] == []
    assert result["missing_from_live"] == [("biometrics", "equilibrium-service")]

    result2 = compute_causal_dag_diff(registry, live_edges=[("biometrics", "equilibrium-service")])
    assert result2["confirmed"] == [("biometrics", "equilibrium-service")]
    assert result2["missing_from_live"] == []


def test_same_service_edges_are_not_counted_as_missing() -> None:
    # graph_cognition and autonomy both really live under orion-cortex-exec --
    # a registry edge between them can never appear as a real
    # CAUSALLY_FOLLOWED_BY edge (passive wiretap only sees one organ_id per
    # physical service), so it must land in its own bucket, not "missing."
    registry = {
        "graph_cognition": _entry("orion-cortex-exec", []),
        "autonomy": _entry("orion-cortex-exec", ["graph_cognition"]),
    }
    result = compute_causal_dag_diff(registry, live_edges=[])
    assert result["same_service_internal_edges"] == [("cortex-exec", "cortex-exec")]
    assert result["missing_from_live"] == []
    assert result["confirmed"] == []


def test_unmapped_entries_are_reported_not_force_matched() -> None:
    registry = {
        "journaler": _entry("orion library (orion/journaler/)", ["collapse_mirror"]),
        "collapse_mirror": _entry("orion-collapse-mirror", []),
    }
    result = compute_causal_dag_diff(registry, live_edges=[])
    assert result["unmapped_registry_entries"] == ["journaler"]
    # journaler's edge can't be evaluated at all (parent side unmappable) --
    # must not silently appear in any bucket.
    assert result["missing_from_live"] == []
    assert result["confirmed"] == []


def test_observed_edge_with_no_registry_counterpart_is_surfaced() -> None:
    registry = {"biometrics": _entry("orion-biometrics", [])}
    live_edges = [("biometrics", "landing-pad")]
    result = compute_causal_dag_diff(registry, live_edges)
    assert result["observed_not_in_registry"] == [("biometrics", "landing-pad")]


def test_empty_registry_and_empty_live_edges_produce_empty_result() -> None:
    result = compute_causal_dag_diff({}, [])
    assert result["confirmed"] == []
    assert result["missing_from_live"] == []
    assert result["observed_not_in_registry"] == []
    assert result["same_service_internal_edges"] == []
    assert result["unmapped_registry_entries"] == []
