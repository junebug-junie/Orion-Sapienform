"""Tests for delivery verbs (answer depth overhaul)."""

from __future__ import annotations

import yaml
from pathlib import Path


def test_delivery_pack_exists_and_has_verbs():
    """Delivery pack exists and lists all expected verbs."""
    base = Path(__file__).resolve().parents[3] / "orion" / "cognition"
    pack_path = base / "packs" / "delivery_pack.yaml"
    assert pack_path.exists(), f"delivery_pack.yaml not found at {pack_path}"
    data = yaml.safe_load(pack_path.read_text())
    expected = {
        "answer_direct",
        "finalize_response",
        "write_guide",
        "write_tutorial",
        "write_runbook",
        "write_recommendation",
        "compare_options",
        "synthesize_patterns",
        "generate_code_scaffold",
    }
    verbs = set(data.get("verbs") or [])
    for name in expected:
        assert name in verbs, f"delivery verb {name} not in pack; have {verbs}"


def test_delivery_verb_yamls_exist():
    """All delivery verb YAML files exist."""
    base = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "verbs"
    expected = [
        "answer_direct",
        "finalize_response",
        "write_guide",
        "write_tutorial",
        "write_runbook",
        "write_recommendation",
        "compare_options",
        "synthesize_patterns",
        "generate_code_scaffold",
    ]
    for name in expected:
        path = base / f"{name}.yaml"
        assert path.exists(), f"verb {name}.yaml not found at {path}"
