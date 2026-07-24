"""Regression test for a live-caught gap: DEFAULT_BACKEND_WEIGHTS must give
bus_synaptic_anomaly an explicit weight, not fusion.py's generic 0.5
fallback for unrecognized sources. Same class of bug the repo's own
falkor_neighborhood comment describes -- found live 2026-07-24 after
RECALL_BUS_SYNAPTIC_ANOMALY_IN_CHAT was flipped on in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.fusion import DEFAULT_BACKEND_WEIGHTS, _backend_weights


def test_bus_synaptic_anomaly_has_an_explicit_default_weight() -> None:
    assert "bus_synaptic_anomaly" in DEFAULT_BACKEND_WEIGHTS
    assert DEFAULT_BACKEND_WEIGHTS["bus_synaptic_anomaly"] == 0.3


def test_bus_synaptic_anomaly_weight_survives_into_a_real_profile() -> None:
    # A profile with no explicit backend_weights override (the common case --
    # no on-disk profile YAML mentions this source) must still resolve to the
    # deliberate default, not the 0.5 fallback reserved for truly unknown sources.
    profile = {"relevance": {"backend_weights": {"sql_chat": 0.6}}}
    weights = _backend_weights(profile)
    assert weights["bus_synaptic_anomaly"] == 0.3
