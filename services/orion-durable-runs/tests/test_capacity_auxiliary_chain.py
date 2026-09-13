"""Real Postgres reservation plus the production stance and finalization builders."""
import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.harness.finalize import run_harness_finalize_chain
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.schemas.resource_admission import CapacityAcquireV1, CapacityTokenV1, ResourceLeaseV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_admission_runtime_postgres import DSN, request, with_database

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


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

    async def scenario(pool, saver, store):
        capacity = PostgresCapacityStore(store, ttl_seconds=90)
        backend = "http://test-backend"
        broker = ResourceBroker(store, lanes={"agent": {
            "backend_key": backend, "healthy": True, "configured": True, "capabilities": {},
        }}, capacity=capacity)
        durable = request("auxiliary-chain")
        await store.submit(durable.model_dump(mode="json"))
        await store.register_demand(durable.run_id, durable.admission.model_dump(mode="json"))
        assert len(await broker.tick()) == 1
        lease = ResourceLeaseV1.model_validate(await store.get_lease(durable.run_id))
        seen = []
        reflection_count = 0

        def acquisition(stage, owner=None):
            return CapacityAcquireV1(
                request_id=f"{stage}:{uuid4().hex}", correlation_id=durable.correlation_id,
                lane="agent", backend_key=backend, max_inflight=1, budget_sec=30, lease=owner,
            )

        async def cortex_client(plan_request):
            nonlocal reflection_count
            verb = plan_request.plan.verb_name
            seen.append(verb)
            if verb == "look_at_camera":
                # This tool reads a passive projection; it performs no LLM call.
                return {"final_text": "The camera shows a table."}
            context = plan_request.context
            claim = acquisition(verb, context.get("resource_lease"))
            claim = claim.model_copy(update={"lane": context["llm_route"]})
            admitted = await capacity.acquire(claim)
            assert admitted["acquired"], f"{verb} blocked on its own reservation: {admitted}"
            permit = admitted["permit"]
            assert permit["lease_id"] == lease.lease_id
            assert permit["generation"] == lease.generation
            assert context["llm_lane"] == lease.lane
            try:
                # One owner request at a time, while unleased work remains fenced.
                assert (await capacity.acquire(acquisition("parallel", lease)))["reason"] == "owner_request_active"
                assert (await capacity.acquire(acquisition("foreign")))["reason"] == "durable_lease_active"
                if verb == "harness_finalize_reflect":
                    reflection_count += 1
                    reflection = make_reflection(
                        alignment_verdict="misaligned" if reflection_count == 1 or repair_required else "aligned",
                        recommended_tool="look_at_camera" if reflection_count == 1 else None,
                    )
                    return {"final_text": json.dumps(reflection.model_dump(mode="json"))}
                return {"final_text": "A grounded final answer."}
            finally:
                assert (await capacity.release(CapacityTokenV1(
                    request_id=permit["request_id"], permit_id=permit["permit_id"],
                )))["released"]

        stance = StanceReactRequestV1(
            correlation_id=durable.correlation_id, session_id="study", user_message="study",
            association=HubAssociationBundleV1(
                correlation_id=durable.correlation_id, broadcast=None, broadcast_stale=True,
                read_source="felt_state_reader",
            ), repair_bundle=None, stance_inputs={}, resource_lease=lease,
        )
        await cortex_client(builder(stance))
        thought = make_thought()
        overlay = make_repair_overlay()
        draft = "A grounded internal draft."
        molecule = build_draft_molecule(
            correlation_id=durable.correlation_id, thought=thought, draft_text=draft,
            grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=overlay,
        )

        async def substrate_client(_molecule):
            return make_appraisal(surprise_level=0.5)

        result = await run_harness_finalize_chain(
            correlation_id=durable.correlation_id, draft_text=draft, draft_molecule=molecule,
            thought=thought, grammar_receipts=[], repair_overlay=overlay,
            user_message="What does the camera show?", voice_contract=None,
            cortex_client=cortex_client, substrate_client=substrate_client, resource_lease=lease,
        )
        assert result.final_text == ("A grounded final answer." if repair_required else draft)
        assert result.response_repair_ran is repair_required
        assert seen == ["stance_react", "harness_finalize_reflect", "look_at_camera",
                        "harness_finalize_reflect"] + (["orion_response_repair"] if repair_required else [])
        assert (await capacity.snapshot())["active_permits"] == []
        assert await store.validate(lease.model_dump(mode="json"))
        assert (await capacity.acquire(acquisition("still-foreign")))["reason"] == "durable_lease_active"

    asyncio.run(with_database(scenario))
