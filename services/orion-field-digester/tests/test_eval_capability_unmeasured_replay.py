"""Gate for scripts/eval_capability_unmeasured_replay.py's headline claim.

The PR that dropped unmeasured capability channels (2026-10-07) was chosen
because, replayed on real ticks, it changed no downstream reading while the
"confidence = 0" alternative made an unmeasured eye the attention selector's
most urgent capability. This pins both halves on a synthetic tick built from
the live topology, so a later change to the digester or a consumer that breaks
either half fails here instead of silently.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "eval_capability_unmeasured_replay", REPO / "scripts" / "eval_capability_unmeasured_replay.py"
)
replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = replay
_spec.loader.exec_module(replay)

from app.tensor.field_state import empty_field_state  # noqa: E402  (path set by the eval module)

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def _tick_jsonl(tmp_path: Path) -> Path:
    lattice = replay.load_lattice(replay.TOPOLOGY)
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    state.node_vectors.setdefault("node:athena", {}).update({"cpu_pressure": 0.3, "disk_pressure": 0.2, "observer_failure_pressure": 0.0})
    state.node_vectors.setdefault("node:circe", {}).update({"gpu_pressure": 0.4, "inference_failure_pressure": 0.0})
    state.node_vectors["node:substrate.vision_organ"] = {"vision_frame_staleness": 0.0, "vision_processing_failure_pressure": 0.0}
    state.node_vectors["node:substrate.storage_write"] = {"write_failure_pressure": 0.0}
    state.node_vectors["node:substrate.rpc_delivery"] = {"rpc_timeout_pressure": 0.0}
    state.node_vectors["node:substrate.bus_synaptic"] = {"prediction_error": 0.1}
    for node, vec in state.node_vectors.items():
        state.node_vector_updated_at.setdefault(node, {}).update({ch: NOW for ch in vec})
    p = tmp_path / "ticks.jsonl"
    p.write_text(state.model_dump_json() + "\n")
    return p


def test_dropping_unmeasured_changes_no_downstream_reading(tmp_path: Path) -> None:
    out = replay.run(_tick_jsonl(tmp_path))
    for scen, c in out["scenarios"].items():
        diffs = {k: v for k, v in c.items() if k.startswith("a:") and k.endswith("differs") and v}
        assert diffs == {}, (scen, diffs)
        # legacy is main's real apply_diffusion; every channel both wrote must agree
        assert c["a_measured_channel_mismatch"] == 0, scen
    assert out["scenarios"]["outage:vision"]["ticks_with_unmeasured_capability_channel"] == 1
    assert out["scenarios"]["natural"]["ticks_with_unmeasured_capability_channel"] == 0


def test_confidence_zero_would_hand_attention_to_the_unmeasured_eye(tmp_path: Path) -> None:
    c = replay.run(_tick_jsonl(tmp_path))["scenarios"]["outage:vision"]
    assert c["b0:outage_cap_is_attention_top"] == 1
    assert c.get("a:outage_cap_is_attention_top", 0) == 0
    assert c.get("legacy:outage_cap_is_attention_top", 0) == 0
