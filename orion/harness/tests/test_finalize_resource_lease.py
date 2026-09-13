"""The real finalization chain must retain one admission owner at every LLM hop."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from orion.harness.finalize import run_harness_finalize_chain
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.schemas.resource_admission import ResourceLeaseV1


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["agent", "metacog"])
async def test_reflection_retry_and_voice_keep_the_owning_lease(monkeypatch, lane):
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    now = datetime.now(timezone.utc)
    lease = ResourceLeaseV1(
        run_id="admitted-study", demand_id="admitted-study:turn", lease_id="owner",
        resource_key=f"llm.route.{lane}", lane=lane, backend_key="http://worker:8000",
        generation=7, granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60),
    )
    thought = make_thought()
    overlay = make_repair_overlay()
    draft = "A grounded internal draft."
    molecule = build_draft_molecule(
        correlation_id="attempt-seven", thought=thought, draft_text=draft,
        grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=overlay,
    )
    calls = []
    reflection_count = 0

    async def substrate_client(_molecule):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(request):
        nonlocal reflection_count
        verb = request.plan.verb_name
        calls.append(verb)
        if verb == "look_at_camera":
            return {"final_text": "The camera shows a table."}
        # At capacity one, any missing/wrong owner would wait behind this
        # turn. Inspect actual requests produced by every chain builder.
        assert request.context["resource_lease"] == lease.model_dump(mode="json")
        assert request.context["llm_route"] == lane
        assert request.context["llm_lane"] == lane
        assert request.context["allow_chat_fallback"] is False
        if verb == "harness_finalize_reflect":
            reflection_count += 1
            reflection = make_reflection(
                alignment_verdict="misaligned" if reflection_count == 1 else "aligned",
                recommended_tool="look_at_camera" if reflection_count == 1 else None,
            )
            return {"final_text": json.dumps(reflection.model_dump(mode="json"))}
        assert verb == "orion_voice_finalize"
        return {"final_text": "The grounded final answer."}

    result = await run_harness_finalize_chain(
        correlation_id="attempt-seven", draft_text=draft, draft_molecule=molecule,
        thought=thought, grammar_receipts=[], repair_overlay=overlay,
        user_message="What does the camera show?", voice_contract=None,
        cortex_client=cortex_client, substrate_client=substrate_client, resource_lease=lease,
    )
    assert result.final_text == "The grounded final answer."
    assert calls == ["harness_finalize_reflect", "look_at_camera", "harness_finalize_reflect", "orion_voice_finalize"]

