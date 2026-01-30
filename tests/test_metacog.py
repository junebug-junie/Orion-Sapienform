
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import importlib.util
import types
from pathlib import Path

import pytest
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.collapse_mirror import (
    CollapseMirrorEntryV2,
    normalize_collapse_entry,
    should_route_to_triage,
)
def _load_executor_module():
    repo_root = Path(__file__).resolve().parents[1]
    app_dir = repo_root / "services" / "orion-cortex-exec" / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

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


def test_change_type_dict_coercion():
    v2_dict = {
        "event_id": "collapse_change_type",
        "observer": "orion",
        "trigger": "baseline",
        "observer_state": ["idle"],
        "type": "flow",
        "emergent_entity": "self",
        "summary": "all good",
        "mantra": "om",
        "change_type": {"label": "escalating", "signal": 0.7},
    }
    v2 = CollapseMirrorEntryV2.model_validate(v2_dict)
    assert v2.change_type == "escalating"
    assert v2.change_type_scores["signal"] == 0.7


def test_mirror_split_gate():
    assert should_route_to_triage({"observer": "juniper"}) is True
    assert should_route_to_triage({"observer": "orion"}) is False


def test_metacog_source_service_is_stamped():
    executor_module = _load_executor_module()
    entry = executor_module._fallback_metacog_draft({"trigger": {"trigger_kind": "baseline"}})
    assert entry.source_service == "metacog"


def test_turn_effect_telemetry_guard():
    executor_module = _load_executor_module()
    turn_effect = {"user": {"valence": 0.1}}
    summary = executor_module.summarize_turn_effect(turn_effect)
    telemetry_base = {"turn_effect": {"user": {"valence": 0.2}}, "note": "keep"}
    telemetry_patch = {"turn_effect": {"user": {"valence": -1.0}}, "note": "override"}
    merged = executor_module._merge_telemetry_system_owned(telemetry_base, telemetry_patch)
    merged["turn_effect"] = turn_effect
    merged["turn_effect_summary"] = summary
    assert merged["turn_effect"]["user"]["valence"] == 0.1
    assert merged["turn_effect_summary"] == summary
    assert merged["note"] == "override"


def test_metacog_entry_id_differs_from_trigger_corr():
    executor_module = _load_executor_module()
    ctx = {"trigger_correlation_id": "corr-1"}
    entry = executor_module._fallback_metacog_draft(ctx)
    assert entry.id != "corr-1"
    assert entry.event_id == entry.id


def test_metacog_system_fields_override_llm_ids():
    executor_module = _load_executor_module()
    ctx = {
        "metacog_entry_id": "base-id",
        "trigger_correlation_id": "corr-1",
        "trigger_trace_id": "trace-1",
    }
    entry_dict = {
        "id": "llm-id",
        "event_id": "llm-id",
        "correlation_id": "llm-corr",
        "state_snapshot": {
            "telemetry": {
                "trigger_kind": "llm",
                "trigger_correlation_id": "llm",
                "trigger_trace_id": "llm",
            }
        },
    }
    updated = executor_module._apply_metacog_system_fields(entry_dict, ctx)
    assert updated["id"] == "base-id"
    assert updated["event_id"] == "base-id"
    assert "correlation_id" not in updated
    telemetry = updated["state_snapshot"]["telemetry"]
    assert telemetry["trigger_correlation_id"] == "corr-1"
    assert telemetry["trigger_trace_id"] == "trace-1"
