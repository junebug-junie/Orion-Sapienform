from __future__ import annotations

from scripts.substrate_effect_cache import SubstrateEffectCache
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline


def test_high_pressure_prompt_yields_repair_concrete_summary():
    cache = SubstrateEffectCache(max_entries=8)
    summary, snapshot = run_substrate_effect_pipeline(
        turn_id="turn-A",
        message_id=None,
        user_text=(
            "you gave me garbage directions, again. stop being vague — "
            "build me a design spec for claude, nuts and bolts, "
            "arsonist pov only"
        ),
        source_id="conv-A",
        contract_before={"mode": "default"},
        cache=cache,
    )
    assert summary is not None
    assert summary["appraisal_kind"] == "repair_pressure"
    assert summary["level_label"] in {"HIGH", "MEDIUM"}
    assert summary["evidence_count"] >= 3
    assert summary["changed_behavior"] is True
    assert summary["behavior_applied"] in {"repair_concrete", "concrete_bias"}
    assert cache.get("turn-A") is snapshot
    assert snapshot.contract_after["mode"] in {"repair_concrete", "concrete_bias"}


def test_benign_prompt_yields_no_behavior_change():
    cache = SubstrateEffectCache(max_entries=8)
    summary, snapshot = run_substrate_effect_pipeline(
        turn_id="turn-B",
        message_id=None,
        user_text="what's the weather like in Paris?",
        source_id="conv-B",
        contract_before={"mode": "default"},
        cache=cache,
    )
    assert summary is not None
    assert summary["changed_behavior"] is False
    assert summary["behavior_applied"] is None
    assert snapshot.contract_after.get("mode") == "default"


def test_pipeline_handles_internal_failure_without_raising(monkeypatch):
    import scripts.substrate_effect_pipeline as mod

    def boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("forced")

    monkeypatch.setattr(mod, "appraise_repair_pressure", boom)
    cache = SubstrateEffectCache(max_entries=8)
    summary, snapshot = run_substrate_effect_pipeline(
        turn_id="turn-C",
        message_id=None,
        user_text="anything",
        source_id="conv-C",
        contract_before={"mode": "default"},
        cache=cache,
    )
    assert summary is None
    assert snapshot is None
    assert cache.get("turn-C") is None
