"""The real finalization chain must retain one owner lane at every LLM hop.

An admitted turn's owner is its GPU pool hold (test_gpu_lease_transport.py pins that every hop
carries it); this file pins the unheld owner rule."""
import json

import pytest

from orion.harness.finalize import run_harness_finalize_chain
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought


@pytest.mark.asyncio
async def test_no_lease_sonnet_finalize_chain_keeps_chat_owner_through_repair(monkeypatch):
    """An unstamped Sonnet turn keeps the chat owner across finalize and repair."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "false")
    thought = make_thought()
    overlay = make_repair_overlay()
    draft = "A draft that needs repair."
    molecule = build_draft_molecule(
        correlation_id="no-lease-sonnet",
        thought=thought,
        draft_text=draft,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=overlay,
    )
    calls = []

    async def substrate_client(_molecule):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(request):
        calls.append(request.plan.verb_name)
        assert request.context["llm_route"] == "chat"
        assert request.context["llm_lane"] == "chat"
        assert request.context["allow_chat_fallback"] is False
        assert "resource_lease" not in request.context
        if request.plan.verb_name == "harness_finalize_reflect":
            reflection = make_reflection(alignment_verdict="misaligned")
            return {"final_text": json.dumps(reflection.model_dump(mode="json"))}
        assert request.plan.verb_name == "orion_response_repair"
        return {"final_text": "The repaired final answer."}

    result = await run_harness_finalize_chain(
        correlation_id="no-lease-sonnet",
        draft_text=draft,
        draft_molecule=molecule,
        thought=thought,
        grammar_receipts=[],
        repair_overlay=overlay,
        user_message="Please answer directly.",
        voice_contract=None,
        cortex_client=cortex_client,
        substrate_client=substrate_client,
        fcc_model_label="MODEL_SONNET",
    )

    assert result.final_text == "The repaired final answer."
    assert result.response_repair_ran is True
    assert calls == ["harness_finalize_reflect", "orion_response_repair"]
