from __future__ import annotations

from pathlib import Path

import yaml

from orion.autonomy.models import CapabilityPolicyV1

POLICY_YAML = (
    Path(__file__).resolve().parents[3] / "config" / "autonomy" / "capability_policy.v1.yaml"
)


def test_capability_policy_v1_yaml_loads_and_validates() -> None:
    raw = yaml.safe_load(POLICY_YAML.read_text(encoding="utf-8"))
    policy = CapabilityPolicyV1.model_validate(raw)
    assert policy.version == "v1"
    assert len(policy.rules) == 4
    assert policy.rules[0].capability_id == "web.fetch.readonly"
    assert policy.rules[0].auto_execute is True
    assert policy.rules[3].capability_id == "web.fetch.write"
    assert policy.rules[3].auto_execute is False
