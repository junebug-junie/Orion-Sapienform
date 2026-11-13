from __future__ import annotations

import json
import time
import uuid

import pytest

from orion.harness.finalize import run_finalize_reflection
from orion.harness import finalize
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought


JSON_502 = {
    "status": "fail",
    "steps": [{"error": "upstream_http_5xx:http_502: model did not return valid JSON: "
                         "Expecting property name enclosed in double quotes"}],
}
OTHER_502 = {"status": "fail", "steps": [{"error": "upstream_http_5xx:http_502: worker unavailable"}]}
VALID = {"final_text": json.dumps({
    "imperative": "answer carefully", "tone": "steady", "strain_refs": [],
    "alignment_notes": ["draft is supported"], "alignment_verdict": "aligned",
    "strain_unresolved": False,
}), "trace_id": "retry-trace"}


async def _reflect(client, *, deadline_monotonic=None):
    return await run_finalize_reflection(
        correlation_id="judge-json-502", draft_text="A supported answer.",
        thought=make_thought(repair_pressure_level=0.9),
        substrate_appraisal=make_appraisal(surprise_level=0.5),
        repair_overlay=make_repair_overlay(), cortex_client=client,
        deadline_monotonic=deadline_monotonic,
    )


@pytest.mark.asyncio
async def test_json_502_retries_only_judge_step_and_keeps_real_verdict():
    requests = []

    async def client(req):
        requests.append(req)
        return JSON_502 if len(requests) == 1 else VALID

    reflection, quick, trace = await _reflect(client)

    assert len(requests) == 2
    assert requests[0].plan.verb_name == requests[1].plan.verb_name == "harness_finalize_reflect"
    assert len(requests[1].plan.steps) == 1
    assert requests[1].args.request_id != requests[0].args.request_id
    assert uuid.UUID(requests[1].args.request_id)
    assert requests[1].context["metadata"]["correlation_id"] == "judge-json-502"
    assert requests[1].plan.timeout_ms <= 60_000
    assert requests[1].plan.steps[0].timeout_ms <= 60_000
    assert quick is False and trace == "retry-trace"
    assert reflection.reflection_source == "substrate_informed_pass"
    assert reflection.alignment_verdict == "aligned"


@pytest.mark.asyncio
async def test_json_502_repeated_failure_degrades_after_one_retry():
    requests = []

    async def client(req):
        requests.append(req)
        return JSON_502

    reflection, quick, _ = await _reflect(client)

    assert len(requests) == 2
    assert quick is False
    assert reflection.reflection_source == "degraded_llm_failure_fallback"
    assert reflection.alignment_verdict == "misaligned"


@pytest.mark.asyncio
async def test_unrelated_502_does_not_retry(caplog):
    requests = []

    async def client(req):
        requests.append(req)
        return OTHER_502

    reflection, _, _ = await _reflect(client)

    assert len(requests) == 1
    assert reflection.reflection_source == "degraded_llm_failure_fallback"
    assert "harness_finalize_reflect_llm_failed_using_degraded_reflection" in caplog.text
    assert "harness_finalize_reflect_payload_unparseable_using_degraded_reflection" not in caplog.text


@pytest.mark.asyncio
async def test_json_502_does_not_retry_when_reply_budget_is_low():
    requests = []

    async def client(req):
        requests.append(req)
        return JSON_502

    reflection, _, _ = await _reflect(client, deadline_monotonic=time.monotonic() + 60)

    assert len(requests) == 1
    assert reflection.reflection_source == "degraded_llm_failure_fallback"


@pytest.mark.asyncio
async def test_json_502_does_not_replay_a_plan_with_more_than_one_step(monkeypatch):
    original = finalize.build_finalize_reflect_plan_request

    def with_extra_step(**kwargs):
        request = original(**kwargs)
        request.plan.steps.append(request.plan.steps[0].model_copy(deep=True))
        return request

    monkeypatch.setattr(finalize, "build_finalize_reflect_plan_request", with_extra_step)
    requests = []

    async def client(req):
        requests.append(req)
        return JSON_502

    reflection, _, _ = await _reflect(client)

    assert len(requests) == 1
    assert reflection.reflection_source == "degraded_llm_failure_fallback"
