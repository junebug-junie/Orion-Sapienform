"""Pins the static body facts corrected 2026-10-14 to the files Orion reads them from.

Live truth checked that day: circe `nvidia-smi -L` -> 4 GPUs (2x V100-PCIE-32GB, PG500-216,
V100-SXM2-32GB) carrying chat/agent/metacog/fast/diffusion; the chat container runs with
CUDA_VISIBLE_DEVICES_OVERRIDE=0 (one card); hecate does not answer ping (on-demand).
These files stay static until the live body map (PR #2648) replaces them.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CHAT_PROFILE = "qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition"


def _load(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))


def test_circe_is_not_called_a_burst_gpu() -> None:
    circe = _load("config/biometrics/node_catalog.yaml")["nodes"]["circe"]
    assert circe["role"] == "primary_cognition_gpu"
    assert circe["expected_online"] is True


def test_hecate_is_in_the_field_topology_and_on_demand() -> None:
    topo = _load("config/field/orion_field_topology.v1.yaml")
    node_ids = [n["node_id"] for n in topo["nodes"]]
    assert "hecate" in node_ids
    # No edge: an offline node must not feed any capability with seeded defaults.
    assert not [e for e in topo["edges"] if "node:hecate" in (e["source_id"], e["target_id"])]
    hecate = _load("config/biometrics/node_catalog.yaml")["nodes"]["hecate"]
    assert hecate["expected_online"] is False


def test_hecate_reaches_field_digester_and_self_sense_loaders() -> None:
    """Both parse the file the same way: `node_id` from `nodes`. self_study's
    _hardware_items uses the same rule (covered by its own suite)."""
    import importlib.util
    import sys

    path = REPO / "services/orion-field-digester/app/graph/lattice.py"
    spec = importlib.util.spec_from_file_location("_fd_lattice_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        lattice = mod.load_lattice(REPO / "config/field/orion_field_topology.v1.yaml")
    finally:
        sys.modules.pop(spec.name, None)
    assert "node:hecate" in lattice.nodes

    from orion.evals.self_sense import load_field_node_ids

    assert "hecate" in load_field_node_ids()


def test_chat_profile_label_says_one_card_and_key_is_kept() -> None:
    profiles = _load("config/llm_profiles.yaml")["profiles"]
    chat = profiles[CHAT_PROFILE]  # key referenced by LLM_PROFILE_NAME on circe; must not move
    assert chat["gpu"]["num_gpus"] == 1
    assert chat["gpu"]["device_ids"] == [0]
    assert "1x V100 32GB" in chat["display_name"]
    assert "2x" not in chat["display_name"]
