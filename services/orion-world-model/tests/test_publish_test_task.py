"""Regression test for scripts/publish_test_task.py's envelope construction.

Code review (2026-08-20) caught this smoke script's `BaseEnvelope(...)` call
crashing on construction -- twice, in two different ways (a bad `schema_id=`
kwarg, then a bare-string `source=`) -- before it was ever run successfully.
This test constructs the exact same envelope the script builds and asserts
it validates, so that regression can't silently return.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from orion.schemas.world_model import WorldModelTaskRequestPayload

# Loaded by explicit file path, not `from scripts.publish_test_task import
# ...` -- this repo's top-level `scripts/` is a *regular* package (has
# scripts/__init__.py), so it wins any `import scripts` lookup outright and
# shadows this service's own services/orion-world-model/scripts/ directory
# (verified live: the natural `sys.path.insert` + package-import approach
# resolved to the wrong "scripts" and raised ModuleNotFoundError for this
# submodule).
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "publish_test_task.py"
_spec = importlib.util.spec_from_file_location("world_model_publish_test_task", _SCRIPT_PATH)
publish_test_task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish_test_task)  # type: ignore[union-attr]

build_request_envelope = publish_test_task.build_request_envelope
build_trajectory = publish_test_task.build_trajectory


def test_build_trajectory_shapes():
    traj = build_trajectory(
        3,
        dim_biometrics=4,
        dim_affect=2,
        dim_execution_context=2,
        dim_memory_pointers=4,
        dim_temporal=1,
        dim_vision_embedding=8,
    )
    assert len(traj) == 3
    assert traj[0].biometrics.dim == 4
    assert len(traj[0].vision_embedding.vector) == 8


def test_build_request_envelope_constructs_and_validates():
    traj = build_trajectory(
        2,
        dim_biometrics=4,
        dim_affect=2,
        dim_execution_context=2,
        dim_memory_pointers=4,
        dim_temporal=1,
        dim_vision_embedding=8,
    )
    envelope = build_request_envelope(traj, correlation_id="11111111-1111-1111-1111-111111111111", reply_channel="orion:worldmodel:reply:test")

    assert envelope.kind == "world_model.task.request"
    assert envelope.schema_id == "orion.envelope"  # the fixed envelope-format marker, not the message kind
    assert envelope.source.name == "cli-tester"
    assert envelope.reply_to == "orion:worldmodel:reply:test"
    assert isinstance(envelope.payload, dict)

    # The payload dict round-trips into the real request schema.
    payload = WorldModelTaskRequestPayload(**envelope.payload)
    assert payload.task_type == "predict_next_state"
    assert len(payload.trajectory) == 2
