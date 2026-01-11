
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
