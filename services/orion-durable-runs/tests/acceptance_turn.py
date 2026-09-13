"""Production Curiosity/Hub/Thought/Governor/Exec/Gateway adapters for acceptance.

Only external cognition is fixture-backed: association/situation reads, substrate
appraisal, FCC's subprocess and model output. The fixture FCC emits three explicit
fixture lookup steps; this is transport/lifecycle evidence, not a cognitive eval.
Redis is supplied by acceptance_bus.TypedBus and HTTP is routed through ASGI.
No socket transport or production bus is reachable from this helper.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import dotenv
import httpx
from pydantic_settings.sources import DotEnvSettingsSource

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.harness.fcc_motor import _build_subprocess_env
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_thought
from orion.llm.resource_lease import LEASE_HEADER, decode_lease_header
from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.harness_finalize import HarnessRunRequestV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ServiceRef(name="durable-acceptance-fixture")
DRAFT = "The isolated fixture ledger contains one measured observation and its provenance."
REPAIRED = "The fixture evidence supports one bounded observation; broader conclusions remain unverified."


def _package(monkeypatch, name, path):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, package)
    return package


def build_turn_adapter(monkeypatch, bus, capacity, store, repair_required, *, authority_app):
    """Install the real turn adapters on a TypedBus; caller owns DB and authority.

    Return has handle_turn(envelope), stages (actual dispatched lease identities),
    runs (real Governor artifacts), gateway_app, and async close(). All monkeypatch
    changes must remain active until outstanding turn tasks and close() complete.
    """
    hub_root = ROOT / "services/orion-hub"
    monkeypatch.setenv("SUBSTRATE_CONTROL_PLANE_DETACHED", "1")
    for key in ("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "SUBSTRATE_POLICY_POSTGRES_URL", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    policy_directory = tempfile.TemporaryDirectory(prefix="durable-acceptance-policy-")
    isolated_root = Path(policy_directory.name)
    monkeypatch.setenv("SUBSTRATE_POLICY_SQL_DB_PATH", str(isolated_root / "policy.sqlite3"))
    monkeypatch.setenv("SUBSTRATE_POLICY_LEGACY_JSON_PATH", str(isolated_root / "policy.json"))
    # The real FCC environment builder may create its configuration directory.
    # Pin both the effective directory and the Harness/Hub fallback aliases so
    # running acceptance from an operator shell cannot touch their Claude state.
    claude_config = isolated_root / "claude"
    claude_config.mkdir()
    for key in ("CLAUDE_CONFIG_DIR", "HARNESS_FCC_CLAUDE_CONFIG_DIR", "HUB_AGENT_CLAUDE_CONFIG_DIR"):
        monkeypatch.setenv(key, str(claude_config))
    monkeypatch.setenv("HARNESS_FCC_WORKSPACE", str(isolated_root))
    monkeypatch.setenv("HUB_AGENT_CLAUDE_WORKSPACE", str(isolated_root))
    monkeypatch.setenv("HARNESS_FCC_ENV_PATH", str(isolated_root / "fixture-fcc.env"))
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_MODE_DIR", str(isolated_root / "context-mode"))
    monkeypatch.setenv("CONTEXT_MODE_DIR", str(isolated_root / "context-mode"))
    monkeypatch.setenv("CONTEXT_MODE_PROJECT_DIR", str(isolated_root))
    for key in ("HARNESS_FCC_MCP_ENABLED", "HARNESS_FCC_CONTEXT_MODE_ENABLED",
                "HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "HARNESS_FCC_GITNEXUS_ENABLED"):
        monkeypatch.setenv(key, "false")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    for key in ("CHANNEL_VOICE_TRANSCRIPT", "CHANNEL_VOICE_LLM", "CHANNEL_VOICE_TTS",
                "CHANNEL_COLLAPSE_INTAKE", "CHANNEL_COLLAPSE_TRIAGE"):
        monkeypatch.setenv(key, "acceptance:" + key.lower())
    # Hub's historical absolute `scripts.settings -> app.settings` import needs
    # a service namespace beside the runner's already imported `app` package.
    # Service settings normally read local operator .env files on import. The
    # acceptance adapters take their configuration only from this fixture.
    with monkeypatch.context() as settings_import:
        settings_import.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
        settings_import.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
        _package(monkeypatch, "acceptance_hub", hub_root / "app")
        hub_settings = importlib.import_module("acceptance_hub.settings")
        _package(monkeypatch, "scripts", hub_root / "scripts")
        monkeypatch.setitem(sys.modules, "scripts.settings", hub_settings)
        for service, alias in (("orion-thought", "acceptance_thought"),
                               ("orion-harness-governor", "acceptance_governor"),
                               ("orion-cortex-exec", "acceptance_exec"),
                               ("orion-llm-gateway", "acceptance_gateway")):
            _package(monkeypatch, alias, ROOT / "services" / service / "app")

        curiosity = importlib.import_module("scripts.curiosity_investigation")
        thought = importlib.import_module("acceptance_thought.bus_listener")
        governor = importlib.import_module("acceptance_governor.bus_listener")
        with monkeypatch.context() as service_import:
            service_import.setitem(sys.modules, "app", sys.modules["acceptance_exec"])
            service_import.setitem(sys.modules, "app.settings", importlib.import_module("acceptance_exec.settings"))
            executor = importlib.import_module("acceptance_exec.executor")
        with monkeypatch.context() as service_import:
            service_import.setitem(sys.modules, "app", sys.modules["acceptance_gateway"])
            service_import.setitem(sys.modules, "app.settings", importlib.import_module("acceptance_gateway.settings"))
            gateway = importlib.import_module("acceptance_gateway.main")
        backend = importlib.import_module("acceptance_gateway.llm_backend")
        upstream = importlib.import_module("acceptance_gateway.upstream_admission")
        from orion.hub import turn_orchestrator
        import orion.harness.runner as runner_module
        import orion.mind.substrate_emit as substrate_emit

    monkeypatch.setenv("HARNESS_LLM_GATEWAY_URL", "http://fixture-gateway")
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "false")
    monkeypatch.setenv("EMBODIMENT_D_FINALIZE_ENABLED", "false")
    monkeypatch.setattr(hub_settings.settings, "ENABLE_PRE_TURN_APPRAISAL", False)
    monkeypatch.setattr(thought.settings, "mind_enrichment_enabled", False)
    monkeypatch.setattr(substrate_emit, "emit_observation", lambda **kwargs: None)
    monkeypatch.setattr(turn_orchestrator, "build_hub_association_bundle", lambda **kw: HubAssociationBundleV1(
        correlation_id=kw["correlation_id"], broadcast=None, broadcast_stale=True,
        read_source="felt_state_reader"))
    monkeypatch.setattr(turn_orchestrator, "_build_situation_prompt_fragment", AsyncMock(return_value={
        "status": "skipped", "compact_text": None, "provider_status": {"fixture": "isolated"}}))
    monkeypatch.setattr(runner_module, "read_last_tool_fetch", AsyncMock(return_value=None))

    for name, value in {
        "llm_gateway_capacity_enabled": True,
        "llm_gateway_capacity_url": "http://fixture-authority/capacity",
        "llm_gateway_lease_validation_enabled": True,
        "llm_gateway_lease_validation_url": "http://fixture-authority/leases/validate",
        "llm_gateway_upstream_max_inflight": 1,
        "llm_gateway_background_poll_interval_sec": 0.001,
        "llm_lane_routing_enabled": False,
        "llm_gateway_anthropic_passthrough_enabled": True,
        "llm_gateway_openai_passthrough_enabled": True,
        "llm_route_table_json": json.dumps({lane: {
            "url": url, "backend": "llamacpp", "model": "fixture-model"}
            for lane, url in (("agent", "http://fixture-backend"), ("metacog", "http://fixture-metacog"))}),
    }.items():
        monkeypatch.setattr(gateway.settings, name, value)
    backend._load_route_targets.cache_clear()
    upstream.reset_upstream_admission_for_tests()
    gateway.reset_executor_for_tests()

    stages = []
    runs = []
    requests = []
    http_owners = {}

    def observe(stage, lease):
        assert lease is not None, f"{stage} dropped the durable owner"
        stages.append({"stage": stage, **{key: lease[key] for key in (
            "run_id", "lease_id", "generation", "lane", "backend_key")}})

    def model_text(stage, correlation_id):
        if stage == "stance_react":
            return make_thought(correlation_id=correlation_id, disposition="proceed").model_dump_json()
        if stage == "harness_finalize_reflect":
            return make_reflection(correlation_id=correlation_id,
                alignment_verdict="misaligned" if repair_required else "aligned").model_dump_json()
        assert stage == "orion_response_repair", f"unexpected model call: {stage}"
        return REPAIRED

    def model_dispatch(body, plan):
        lease = (body.options or {}).get("resource_lease")
        stage = (body.options or {}).get("verb")
        observe(stage, lease)
        text = model_text(stage, body.trace_id)
        return {"text": text, "content": text, "route": plan.route, "model": "fixture-model"}

    monkeypatch.setattr(gateway, "run_llm_chat", model_dispatch)

    class IsolatedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.host == "fixture-authority":
                return await httpx.ASGITransport(app=authority_app).handle_async_request(request)
            if request.url.host == "fixture-gateway":
                http_owners[request.headers["x-request-id"]] = decode_lease_header(request.headers[LEASE_HEADER])
                return await httpx.ASGITransport(app=gateway.app).handle_async_request(request)
            if request.url.host in {"fixture-backend", "fixture-metacog"}:
                assert request.url.path == "/v1/messages"
                assert LEASE_HEADER not in request.headers, "Gateway must not leak its authority token upstream"
                token = http_owners[request.headers["x-request-id"]]
                observe("fcc_primary", token)
                active = (await capacity.snapshot())["active_permits"]
                assert any(row["lease_id"] == token["lease_id"] for row in active)
                return httpx.Response(200, request=request, json={
                    "id": "fixture-message", "type": "message", "role": "assistant",
                    "model": "fixture-model", "content": [{"type": "text", "text": DRAFT}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
            raise AssertionError(f"Acceptance attempted external HTTP: {request.url.host}")

    real_client = httpx.AsyncClient
    def isolated_client(*args, **kwargs):
        kwargs.setdefault("transport", IsolatedTransport())
        return real_client(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", isolated_client)

    async def fcc_fixture(**kwargs):
        # Use the actual FCC environment builder to exercise owner header
        # transport; only the subprocess/model is replaced by this HTTP call.
        env = _build_subprocess_env(fcc_server_url="http://unreachable-proxy", auth_token="fixture",
                                    resource_lease=kwargs["resource_lease"])
        assert Path(env["CLAUDE_CONFIG_DIR"]).resolve() == claude_config.resolve()
        assert Path(env["HARNESS_FCC_WORKSPACE"]).resolve() == isolated_root.resolve()
        headers = {line.partition(":")[0]: line.partition(":")[2].strip()
                   for line in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()}
        headers["x-request-id"] = kwargs["correlation_id"]
        async with httpx.AsyncClient() as client:
            response = await client.post(env["ANTHROPIC_BASE_URL"] + "/v1/messages", headers=headers, json={
                "model": kwargs["fcc_model_label"], "max_tokens": 256,
                "messages": [{"role": "user", "content": kwargs["prompt"]}]})
            assert response.status_code == 200, response.text
        for fixture_read in ("observation", "provenance", "prior"):
            yield {"type": "step", "step": {"type": "tool_result", "raw": {
                "type": "tool_result", "tool_name": "read_fixture_ledger",
                "content": f"Isolated fixture read: {fixture_read}"}}}
        yield {"type": "final", "llm_response": response.json()["content"][0]["text"],
               "metadata": {"exit_code": 0, "fcc_served_model": "fixture-model"}}

    async def reply(env, kind, payload):
        await bus.publish(env.reply_to, BaseEnvelope(kind=kind, source=SOURCE,
            correlation_id=env.correlation_id, payload=payload))

    async def handle_thought(env):
        await thought.handle_stance_react_request(bus, StanceReactRequestV1.model_validate(env.payload),
            reply_to=env.reply_to, correlation_id=str(env.correlation_id))

    async def handle_harness(env):
        request = HarnessRunRequestV1.model_validate(env.payload)
        requests.append(request)
        substrate = SimpleNamespace(finalize_appraisal=AsyncMock(side_effect=lambda molecule, **kw:
            make_appraisal(correlation_id=request.correlation_id, surprise_level=0.5)))
        run = await governor.handle_harness_run_request(bus, request, reply_to=env.reply_to,
            correlation_id=str(env.correlation_id), substrate_client=substrate,
            runner=HarnessRunner(bus, fcc_runner=fcc_fixture, served_model_probe=AsyncMock(return_value="fixture-model")))
        runs.append(run)

    async def handle_plan(env):
        request = PlanExecutionRequest.model_validate(env.payload)
        # External recall/graph providers are not started. Execute the actual
        # plan's LLM step through Exec's production routing and typed client.
        steps = [step for step in request.plan.steps if "LLMGatewayService" in step.services]
        assert len(steps) == 1, request.plan.verb_name
        result = await executor.call_step_services(bus=bus, source=SOURCE, step=steps[0],
            ctx=dict(request.context), correlation_id=str(env.correlation_id))
        assert result.status == "success", result.model_dump(mode="json")
        final_text = result.result["LLMGatewayService"]["content"]
        await reply(env, "cortex.exec.result", {"result": {"final_text": final_text,
            "steps": [result.model_dump(mode="json")]}})

    async def handle_llm(env):
        response = await gateway.handle_chat(env)
        await bus.publish(env.reply_to, response)

    bus.handlers[hub_settings.settings.CHANNEL_THOUGHT_REQUEST] = handle_thought
    bus.handlers[hub_settings.settings.CHANNEL_HARNESS_RUN_REQUEST] = handle_harness
    bus.handlers[hub_settings.settings.CHANNEL_HARNESS_RUN_REQUEST_AGENT] = handle_harness
    bus.handlers[thought.settings.channel_cortex_exec_request] = handle_plan
    bus.handlers[governor.settings.channel_cortex_exec_request] = handle_plan
    bus.handlers[executor.settings.channel_llm_intake] = handle_llm

    loop = curiosity.CuriosityInvestigation(enabled=True, tick_interval_sec=60, min_cooldown_sec=0,
        daily_cap=10, timeout_sec=30, session_id="isolated-durable-acceptance", llm_route="agent",
        pool_provider=lambda: None, source_ref=SOURCE, kickoff_via_cortex=True,
        durable_admission_enabled=True, lease_validation_url="http://fixture-authority/leases/validate")
    loop._bus = bus
    loop._harness_rpc_bus = bus

    async def handle_turn(env):
        await loop._handle_turn_request({"data": bus.codec.encode(env)})

    async def close():
        gateway.reset_executor_for_tests()
        upstream.reset_upstream_admission_for_tests()
        backend._load_route_targets.cache_clear()
        policy_directory.cleanup()

    return SimpleNamespace(handle_turn=handle_turn, stages=stages, runs=runs, requests=requests,
                           loop=loop, gateway_app=gateway.app, close=close)
