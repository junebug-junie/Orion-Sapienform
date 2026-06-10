import asyncio

from app.decision_router import DecisionRouter, _trace_autopsy_mode_for_text
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest


def _req(text: str, **options) -> CortexClientRequest:
    opts = {"route_intent": "auto", **options}
    return CortexClientRequest.model_validate(
        {
            "mode": "auto",
            "route_intent": "auto",
            "packs": [],
            "options": opts,
            "recall": {"enabled": True, "required": False, "mode": "hybrid", "profile": None},
            "context": {
                "messages": [{"role": "user", "content": text}],
                "raw_user_text": text,
                "user_message": text,
                "metadata": {},
            },
        }
    )


class _FakeBus:
    codec = None


def test_trace_autopsy_heuristic_corr_fail_open():
    assert _trace_autopsy_mode_for_text("Why did corr 123 fail open?") is True


def test_trace_autopsy_heuristic_runtime_failed():
    assert _trace_autopsy_mode_for_text("Orion cortex run failed during agent-chain") is True


def test_docker_build_failed_not_trace_autopsy():
    assert _trace_autopsy_mode_for_text("My Docker build failed") is False


def test_pytest_failed_not_trace_autopsy():
    assert _trace_autopsy_mode_for_text("this test failed on CI") is False


def test_docker_build_failed_no_context_exec_trace_autopsy_route():
    router = DecisionRouter(_FakeBus())
    req = _req("My Docker build failed during compose up")
    routed = asyncio.run(
        router.route(req, correlation_id="c-docker-fail", source=ServiceRef(name="orch", version="0", node="n"))
    )
    assert routed.request.options.get("context_exec_mode") != "trace_autopsy"
