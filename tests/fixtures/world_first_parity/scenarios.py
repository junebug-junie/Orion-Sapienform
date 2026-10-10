"""Deterministic inputs for the ATTENTION_WORLD_FIRST_ENABLED=false parity test.

The golden outputs next to this file were produced by running ``run_all()``
against origin/main at dc8eab06c (before world-first existed):

    PYTHONPATH=<origin/main checkout> python tests/fixtures/world_first_parity/scenarios.py

tests/test_attention_world_first_parity.py runs the same scenarios against the
current tree with world-first OFF and asserts byte-identical output, so
"false restores the previous ranking exactly" is a checked claim, not a hope.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _field_scenarios():
    from orion.attention.field_attention.candidate_precision_weighted import (
        NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
        NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
        PrecisionEwmaBaseline,
        advance_precision_baseline,
    )
    from orion.schemas.field_state import FieldStateV1

    def base(values, observed_at=None):
        b = advance_precision_baseline(
            PrecisionEwmaBaseline(),
            values,
            alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
            min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
        )
        if observed_at is not None:
            from dataclasses import replace

            b = replace(b, last_observed_at=observed_at)
        return b

    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_parity_1",
        node_vectors={
            "node:athena": {"cortex_exec_step_load": 0.8, "availability": 0.9},
            "node:circe": {"gpu_pressure": 0.4},
            "node:substrate.execution": {"prediction_error": 0.6},
            "node:substrate.perception": {"prediction_error": 0.2},
        },
        capability_vectors={
            "capability:orchestration": {"execution_pressure": 0.7},
            "capability:vision": {"pressure": 0.2},
        },
        recent_perturbations=["state_delta:a"],
    )
    calm = [0.01, 0.02, 0.015, 0.02, 0.01] * 6
    yield "calm_and_spike", field, {
        "node:substrate.biometrics": base(calm + [0.03], NOW - timedelta(seconds=20)),
        "node:substrate.execution": base(calm + [0.6], NOW - timedelta(seconds=40)),
        "node:substrate.chat": base([0.0] * 25 + [0.4], NOW - timedelta(hours=3)),
        "node:substrate.route": base([0.0, 0.25] * 12, NOW - timedelta(minutes=5)),
        "node:substrate.bus_synaptic": base([0.03, 0.04] * 15, NOW - timedelta(seconds=30)),
    }
    yield "all_calm", field.model_copy(update={"tick_id": "tick_parity_2"}), {
        "node:substrate.biometrics": base(calm, NOW),
        "node:substrate.bus_synaptic": base([0.03, 0.031] * 15, NOW),
    }


def run_field() -> dict:
    from orion.attention.field_attention.builder import build_attention_frame
    from orion.attention.field_attention.policy import load_attention_policy

    repo = Path(__file__).resolve().parents[3]
    policy = load_attention_policy(repo / "config" / "attention" / "field_attention_policy.v1.yaml")
    out = {}
    previous = None
    for name, field, baselines in _field_scenarios():
        frame = build_attention_frame(
            field=field,
            policy=policy,
            prediction_error_baselines=baselines,
            previous_frame=previous,
            now=NOW,
        )
        out[name] = frame.model_dump(mode="json")
        previous = frame
    return out


def _node(node_id, label, pressure, *, pe=None, reason=None):
    md = {"dynamic_pressure": pressure}
    if pe is not None:
        md["prediction_error"] = pe
    if reason is not None:
        md["dynamic_pressure_reason"] = reason
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        node_kind="concept",
        metadata=md,
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=30)),
    )


def run_broadcast() -> dict:
    os.environ["ORION_ATTENTION_TOPDOWN_ENABLED"] = "false"
    from orion.substrate.attention_broadcast import build_substrate_attention_frame

    nodes = [
        _node("node:substrate.execution", "Execution prediction error", 0.31, pe=0.6, reason="prediction_error_seed"),
        _node("node:substrate.biometrics", "Biometrics prediction error", 0.08, pe=0.03, reason="prediction_error_seed"),
        _node("node:substrate.perception", "Perception prediction error", 0.12, pe=0.2, reason="prediction_error_seed"),
        _node("node:concept.x", "unresolved contradiction", 0.2, reason="contradiction_unresolved"),
        _node("node:concept.calm", "calm concept", 0.01),
    ]
    out = {}
    with patch(
        "orion.substrate.attention_broadcast.load_terminal_verdict_loop_ids",
        return_value=set(),
    ):
        frame = build_substrate_attention_frame(nodes=nodes, min_salience=0.05, now=NOW)
    dumped = frame.model_dump(mode="json")
    out["mixed"] = dumped
    return out


def run_all() -> dict:
    return {"field": run_field(), "broadcast": run_broadcast()}


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(os.pathsep)[0] or ".")
    (HERE / "golden.json").write_text(json.dumps(run_all(), indent=1, sort_keys=True) + "\n")
    print("wrote", HERE / "golden.json")
