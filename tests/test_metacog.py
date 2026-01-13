
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import pytest
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, normalize_collapse_entry

def test_metacog_trigger_schema():
    t = MetacogTriggerV1(
        trigger_kind="baseline",
        reason="test",
        zen_state="zen",
        pressure=0.5
    )
    assert t.trigger_kind == "baseline"
    assert t.pressure == 0.5
    assert t.timestamp is not None

def test_normalize_collapse_entry_from_llm_json():
    # Simulate LLM output which might be V1-like or partial V2
    llm_output = {
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": ["idle"],
        "field_resonance": "calm",
        "type": "flow",
        "emergent_entity": "self",
        "summary": "all good",
        "mantra": "om",
        "causal_echo": None,
    }

    v2 = normalize_collapse_entry(llm_output)
    assert isinstance(v2, CollapseMirrorEntryV2)
    assert v2.snapshot_kind == "baseline"
    assert v2.event_id.startswith("collapse_")
    assert v2.numeric_sisters.valence is None  # Defaults

def test_normalize_collapse_entry_v2_full():
    v2_dict = {
        "event_id": "collapse_123",
        "observer": "orion",
        "trigger": "test",
        "observer_state": ["test"],
        "type": "test",
        "emergent_entity": "test",
        "summary": "test",
        "mantra": "test",
        "snapshot_kind": "metacog",
        "numeric_sisters": {"valence": 0.9}
    }
    v2 = normalize_collapse_entry(v2_dict)
    assert v2.event_id == "collapse_123"
    assert v2.numeric_sisters.valence == 0.9


def test_normalize_observer_state_and_tags_string_sets():
    v2_dict = {
        "event_id": "collapse_abc",
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": "{running_metacognition,scheduled_check,zen_state_not_achieved,pressure_level_low}",
        "type": "flow",
        "emergent_entity": "Synaptic Spark",
        "summary": "Focus state.",
        "mantra": "Effortless efficiency",
        "tags": "{metacognition, self-awareness, awareness}",
    }
    v2 = normalize_collapse_entry(v2_dict)
    assert v2.observer_state == [
        "running_metacognition",
        "scheduled_check",
        "zen_state_not_achieved",
        "pressure_level_low",
    ]
    assert v2.tags == ["metacognition", "self-awareness", "awareness"]


@pytest.mark.parametrize(
    "raw_state,expected",
    [
        (
            "{\"scheduled_ check\",metacognition,\"Zen State: not_zen\"}",
            ["scheduled_check", "metacognition", "Zen State: not_zen"],
        ),
        (
            "scheduled_check\nmetacognition\nZen State: not_zen",
            ["scheduled_check", "metacognition", "Zen State: not_zen"],
        ),
    ],
)
def test_normalize_observer_state_variants(raw_state, expected):
    v2_dict = {
        "event_id": "collapse_variant",
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": raw_state,
        "type": "idle",
        "emergent_entity": "Unknown",
        "summary": "Fallback",
        "mantra": "Observe.",
    }
    v2 = normalize_collapse_entry(v2_dict)
    assert v2.observer_state == expected


def test_v1_and_v2_shape_detection():
    v1_like = {
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": "idle",
        "field_resonance": "calm",
        "type": "flow",
        "emergent_entity": "self",
        "summary": "all good",
        "mantra": "om",
        "causal_echo": None,
        "timestamp": "2025-01-01T00:00:00Z",
        "environment": "dev",
    }
    v1 = normalize_collapse_entry(v1_like)
    assert v1.snapshot_kind == "baseline"
    assert v1.event_id.startswith("collapse_")

    v2_like = {
        "event_id": "collapse_keep",
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": ["idle"],
        "field_resonance": "calm",
        "type": "flow",
        "emergent_entity": "self",
        "summary": "all good",
        "mantra": "om",
        "snapshot_kind": "baseline",
        "numeric_sisters": {"valence": 0.4},
    }
    v2 = normalize_collapse_entry(v2_like)
    assert v2.event_id == "collapse_keep"
    assert v2.numeric_sisters.valence == 0.4


def test_state_snapshot_tags_follow_entry_tags():
    v2_dict = {
        "event_id": "collapse_tags",
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": ["idle"],
        "type": "flow",
        "emergent_entity": "self",
        "summary": "all good",
        "mantra": "om",
        "tags": ["metacognition", "flow_state"],
    }
    v2 = normalize_collapse_entry(v2_dict)
    assert v2.state_snapshot.tags == ["metacognition", "flow_state"]
