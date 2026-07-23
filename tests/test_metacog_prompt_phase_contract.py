import importlib.util
import json
import sys
import types
from pathlib import Path

from orion.schemas.metacog_patches import (
    MetacogDraftTextPatchV1,
    MetacogDraftWhatChangedV1,
    MetacogEnrichScorePatchV1,
    MetacogNumericSistersV1,
)


def _allowed_keys(model_cls):
    return set(model_cls.model_fields.keys())


def _read_template(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "orion" / "cognition" / "prompts" / name).read_text()


def _extract_example_json(text: str) -> str:
    start_tag = "<example_json>"
    end_tag = "</example_json>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    assert start != -1, "example_json start tag missing"
    assert end != -1, "example_json end tag missing"
    return text[start + len(start_tag):end].strip()


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


def test_draft_prompt_requires_patch_only():
    text = _read_template("log_orion_metacognition_draft.j2")
    example = _extract_example_json(text)
    assert "FULL CollapseMirrorEntryV2" not in text
    example_payload = json.loads(example)
    assert set(example_payload.keys()) <= _allowed_keys(MetacogDraftTextPatchV1)
    what_changed = example_payload.get("what_changed") or {}
    assert set(what_changed.keys()) <= _allowed_keys(MetacogDraftWhatChangedV1)


def test_enrich_prompt_requires_patch_only():
    text = _read_template("log_orion_metacognition_enrich.j2")
    example = _extract_example_json(text)
    assert "FULL CollapseMirrorEntryV2" not in text
    example_payload = json.loads(example)
    assert set(example_payload.keys()) <= _allowed_keys(MetacogEnrichScorePatchV1)
    numeric_sisters = example_payload.get("numeric_sisters") or {}
    assert set(numeric_sisters.keys()) <= _allowed_keys(MetacogNumericSistersV1)


def test_metacog_wrapper_instruction_is_patch_only():
    executor_module = _load_executor_module()
    messages = executor_module._metacog_messages(
        "prompt",
        allowed_keys=_allowed_keys(MetacogDraftTextPatchV1),
        phase="draft",
    )
    instruction = messages[1]["content"]
    assert "CollapseMirrorEntryV2" not in instruction
    assert "MetacogDraftTextPatchV1" not in instruction


# --- trigger_upstream_json evidence-cue block --------------------------------
#
# trigger.upstream carries the real per-trigger-kind evidence (chat_turn's
# fired_conditions/alignment_notes/surprise_level, telemetry_anomaly's
# recon_loss/top_channels, relational's evidence/behavior_applied) built into
# every gate in orion-equilibrium-service, but until this patch neither prompt
# rendered it -- only trigger.reason's compact string reached the LLM. See
# docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md's
# chat_turn spec, question 5.


def test_draft_prompt_renders_trigger_upstream_json():
    text = _read_template("log_orion_metacognition_draft.j2")
    assert "{{ trigger_upstream_json }}" in text
    assert "do NOT paste into output" in text.split("{{ trigger_upstream_json }}")[0][-200:]


def test_enrich_prompt_renders_trigger_upstream_json():
    text = _read_template("log_orion_metacognition_enrich.j2")
    assert "{{ trigger_upstream_json }}" in text
    assert "trigger_upstream_json" in text.split("ANTI-LOG CONSTRAINTS")[1].split("TASK")[0]


def test_trigger_upstream_json_ctx_value_matches_executor_convention():
    """Same expression executor.py uses to build ctx["trigger_upstream_json"]
    -- covers the shapes MetacogTriggerV1.upstream can actually take: nested
    dicts (chat_turn's grounding_capsule), an empty dict (baseline/manual
    triggers, which never set upstream), and a non-JSON-native value
    (default=str fallback, in case a future upstream field carries one)."""
    import json as _json

    upstream = {
        "fired_conditions": ["disposition=defer", "boundary_register=true"],
        "grounding_capsule": {"identity_summary": ["x"], "provenance": {}},
        "surprise_level": 0.82,
    }
    rendered = _json.dumps(upstream or {}, indent=2, default=str)
    parsed = _json.loads(rendered)
    assert parsed == upstream

    assert _json.dumps({} or {}, indent=2, default=str) == "{}"

    from datetime import datetime, timezone

    non_native = {"as_of": datetime(2026, 7, 23, tzinfo=timezone.utc)}
    rendered_non_native = _json.dumps(non_native or {}, indent=2, default=str)
    assert "2026-07-23" in rendered_non_native


def test_draft_and_enrich_templates_render_trigger_upstream_json_with_jinja():
    from jinja2 import Environment

    env = Environment(autoescape=False)

    draft_text = _read_template("log_orion_metacognition_draft.j2")
    ctx = {
        "trigger": {"trigger_kind": "chat_turn", "reason": "chat_turn:disposition=defer", "pressure": 0.4, "zen_state": "not_zen"},
        "trigger_upstream_json": '{\n  "fired_conditions": [\n    "disposition=defer"\n  ]\n}',
        "context_summary": "",
        "spark_state_json": "{}",
        "turn_effect_json": "null",
        "recent_turn_effect_alerts_json": "[]",
        "turn_effect_policy_json": "{}",
        "turn_effect_explanations_json": "{}",
        "spark_embodiment_narrative": "",
        "metacog_biometrics_cue": "",
        "metacog_substrate_cue": "",
    }
    rendered = env.from_string(draft_text).render(**ctx)
    assert '"disposition=defer"' in rendered

    enrich_text = _read_template("log_orion_metacognition_enrich.j2")
    enrich_ctx = dict(ctx)
    enrich_ctx["collapse_json"] = "{}"
    rendered_enrich = env.from_string(enrich_text).render(**enrich_ctx)
    assert '"disposition=defer"' in rendered_enrich
