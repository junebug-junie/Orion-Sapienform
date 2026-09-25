"""A held run's whole auxiliary LLM chain -- stance, finalize reflect, the bounded tool retry and the
conditional repair -- runs INSIDE the run's GPU pool hold (stage 4.5).

The production stance and finalize builders build each call's context; a cortex client stands in
for cortex-exec -> gateway and places every call on the REAL in-process pool exactly the way the
gateway does: a call carrying ``gpu_lease`` is an ``attach``. Evidence: every call is a child of
the run's hold, never a second agent lease (which would queue behind the run itself), and a foreign
agent call meanwhile waits in line. Replaces the pre-4.5 durable capacity-permit version.
"""
import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

from orion.harness.finalize import run_harness_finalize_chain
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.schemas.gpu_pool import GpuLeaseRefV1, GpuLeaseRequestV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_fixture import InProcessPool


@pytest.mark.parametrize("repair_required", [False, True])
def test_capacity_one_stance_retry_and_conditional_repair_share_owner(monkeypatch, repair_required):
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "true")
    # Give Thought a distinct package name: these service integration tests
    # already loaded durable-runs' unrelated `app` package.
    package_name = "capacity_test_thought"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "orion-thought" / "app")]
    monkeypatch.setitem(sys.modules, package_name, package)
    builder = importlib.import_module(f"{package_name}.bus_listener").build_stance_react_plan_request

    async def scenario():
        gpu = await InProcessPool().boot()
        run_id, correlation_id = "auxiliary-chain", str(uuid4())
        held = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id=f"{run_id}:1", kind="hold",
            holder=f"durable-runs:{run_id}", work_class="agent", priority="background", retryable=True))
        assert held.status == "granted"
        ref = GpuLeaseRefV1(lease_id=held.lease_id, generation=held.grant.generation, role=held.grant.role,
                            holder=f"durable-runs:{run_id}")
        seen = []
        reflection_count = 0

        async def cortex_client(plan_request):
            nonlocal reflection_count
            verb = plan_request.plan.verb_name
            seen.append(verb)
            if verb == "look_at_camera":
                # This tool reads a passive projection; it performs no LLM call.
                return {"final_text": "The camera shows a table."}
            context = plan_request.context
            # The gateway's rule: a carried ref is an attach, never a lease of its own.
            carried = GpuLeaseRefV1.model_validate(context["gpu_lease"])
            assert carried == ref, f"{verb} dropped or changed the run's hold"
            assert context["llm_route"] == "agent", f"{verb} named a pool role as its route"
            child = await gpu.dispatch(GpuLeaseRequestV1(verb="attach", request_id=f"{verb}:{uuid4().hex}",
                holder="orion-llm-gateway", work_class="agent", priority="background",
                hold_lease_id=carried.lease_id, hold_generation=carried.generation))
            assert child.status == "granted", f"{verb} blocked behind its own hold: {child}"
            try:
                # Meanwhile a foreign agent call waits: the hold owns the card between these calls.
                foreign = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id=uuid4().hex,
                    holder="someone-else", work_class="agent", priority="background"))
                assert foreign.status == "queued"
                await gpu.dispatch(GpuLeaseRequestV1(verb="cancel", lease_id=foreign.lease_id))
                if verb == "harness_finalize_reflect":
                    reflection_count += 1
                    reflection = make_reflection(
                        alignment_verdict="misaligned" if reflection_count == 1 or repair_required else "aligned",
                        recommended_tool="look_at_camera" if reflection_count == 1 else None,
                    )
                    return {"final_text": json.dumps(reflection.model_dump(mode="json"))}
                return {"final_text": "A grounded final answer."}
            finally:
                await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=child.lease_id, outcome="ok"))

        stance = StanceReactRequestV1(
            correlation_id=correlation_id, session_id="study", user_message="study",
            association=HubAssociationBundleV1(
                correlation_id=correlation_id, broadcast=None, broadcast_stale=True,
                read_source="felt_state_reader",
            ), repair_bundle=None, stance_inputs={}, gpu_lease=ref,
        )
        await cortex_client(builder(stance))
        thought = make_thought()
        overlay = make_repair_overlay()
        draft = "A grounded internal draft."
        molecule = build_draft_molecule(
            correlation_id=correlation_id, thought=thought, draft_text=draft,
            grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=overlay,
        )

        async def substrate_client(_molecule):
            return make_appraisal(surprise_level=0.5)

        result = await run_harness_finalize_chain(
            correlation_id=correlation_id, draft_text=draft, draft_molecule=molecule,
            thought=thought, grammar_receipts=[], repair_overlay=overlay,
            user_message="What does the camera show?", voice_contract=None,
            cortex_client=cortex_client, substrate_client=substrate_client, gpu_lease=ref,
        )
        assert result.final_text == ("A grounded final answer." if repair_required else draft)
        assert result.response_repair_ran is repair_required
        assert seen == ["stance_react", "harness_finalize_reflect", "look_at_camera",
                        "harness_finalize_reflect"] + (["orion_response_repair"] if repair_required else [])
        children = gpu.leases(hold_lease_id=held.lease_id)
        llm_calls = [v for v in seen if v != "look_at_camera"]
        assert len(children) == len(llm_calls) and all(c["status"] == "released" for c in children)
        # No agent-class request lease outside the hold (acceptance check 2's rule, here in miniature).
        assert not [r for r in gpu.leases(kind="request", work_class="agent", hold_lease_id=None)
                    if r["holder"] != "someone-else"]
        assert (await gpu.lease(held.lease_id))["status"] == "granted"

    asyncio.run(scenario())
