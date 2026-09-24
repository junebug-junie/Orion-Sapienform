from __future__ import annotations

import pytest

import orion.metacog.service as service
from orion.metacog.evidence_map import map_trigger
from orion.metacog.service import (
    IS_CAUSALLY_DENSE_THRESHOLD,
    compute_causal_density,
    compute_provenance,
)


def test_retired_scorers_are_gone():
    """2026-09-24: severity/density/touches from the writer's logprobs and the
    global-state blend were retired outright, not kept as fallbacks."""
    for name in ("compute_severity", "compute_touches", "turn_effect_severity"):
        assert not hasattr(service, name)


def test_causal_density_is_event_magnitude():
    m = map_trigger(
        "telemetry_anomaly",
        "r",
        {"recon_loss": 0.03, "threshold": 0.01, "top_channels": [], "deviation_direction": "elevated"},
    )
    density = compute_causal_density(m)
    assert density.score == m.magnitude
    assert density.score >= IS_CAUSALLY_DENSE_THRESHOLD
    assert density.label in ("dense", "critical")
    assert density.rationale.startswith("event_magnitude[telemetry_anomaly]")


def test_causal_density_zero_only_without_evidence():
    density = compute_causal_density(map_trigger("baseline", "scheduled_check", {}))
    assert density.score == 0.0
    assert density.label == "ambient"


def test_dense_threshold_matches_critical_band():
    assert IS_CAUSALLY_DENSE_THRESHOLD == pytest.approx(0.6)


def test_provenance_source_is_dynamic_per_trigger_kind():
    assert compute_provenance(trigger_kind="relational", touches=[]).source == (
        "cortex_exec.metacog_pipeline.relational"
    )


def test_provenance_impacts_are_event_touches_and_carries_pipeline_steps():
    steps = [f"step {i}" for i in range(30)] + ["x" * 500]
    p = compute_provenance(trigger_kind="transport", touches=["cortex-exec", "orion:state:request"], pipeline_steps=steps)
    assert p.impacts == ["cortex-exec", "orion:state:request"]
    assert len(p.pipeline_steps) == 20
    assert p.pipeline_steps[-1] == "x" * 200
