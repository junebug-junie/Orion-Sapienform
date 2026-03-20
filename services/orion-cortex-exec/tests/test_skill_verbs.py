import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg
spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.verb_adapters", APP_DIR / "verb_adapters.py")
verb_adapters = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = verb_adapters
spec.loader.exec_module(verb_adapters)

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.verbs.base import VerbContext  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest  # noqa: E402
from orion.schemas.pad.v1 import PadRpcResponseV1  # noqa: E402


class _Codec:
    @staticmethod
    def decode(data):
        return SimpleNamespace(ok=True, error=None, envelope=BaseEnvelope.model_validate(data))


class _FakeBus:
    def __init__(self, result_payload: dict) -> None:
        self.codec = _Codec()
        self.result_payload = result_payload
        self.calls = []

    async def rpc_request(self, channel: str, envelope: BaseEnvelope, *, reply_channel: str, timeout_sec: float):
        self.calls.append((channel, envelope, reply_channel, timeout_sec))
        return {
            "data": BaseEnvelope(
                kind="PadRpcResponseV1",
                source=ServiceRef(name="pad"),
                correlation_id=str(envelope.correlation_id),
                payload=self.result_payload,
            ).model_dump(mode="json")
        }


def _plan_request(verb_name: str, *, skill_args: dict | None = None) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name=verb_name, steps=[]),
        args=PlanExecutionArgs(request_id=str(uuid4()), extra={"skill_args": skill_args or {}}),
        context={"metadata": {}},
    )


def test_safe_runner_blocks_non_allowlisted_commands():
    runner = verb_adapters.SafeCommandRunner(allowed_commands={"nvidia-smi"}, timeout_sec=1)

    try:
        runner.run(["bash", "-lc", "echo nope"])
    except PermissionError as exc:
        assert "command_not_allowlisted:bash" in str(exc)
    else:
        raise AssertionError("expected allowlist block")


def test_nvidia_smi_parser_parses_sample_output():
    rows = verb_adapters._parse_nvidia_smi_csv(
        "0, NVIDIA RTX 4090, GPU-123, 44, 67, 8192, 24564, 210.50, P2\n"
        "1, NVIDIA RTX 4080, GPU-456, 39, 12, 1024, 16384, 90.00, P8\n"
    )

    assert len(rows) == 2
    assert rows[0]["index"] == 0
    assert rows[0]["memory_used_ratio"] == 8192 / 24564
    assert rows[1]["pstate"] == "P8"


def test_docker_engine_mapping_parses_sample_response():
    mapped = verb_adapters._map_docker_engine_containers(
        [
            {
                "Id": "abc123",
                "Names": ["/orion-api"],
                "Image": "orion/api:latest",
                "State": "running",
                "Status": "Up 3 minutes",
                "Command": "python app.py",
                "Ports": [{"PrivatePort": 8000, "PublicPort": 18000, "Type": "tcp"}],
            }
        ]
    )

    assert mapped == [
        {
            "id": "abc123",
            "name": "orion-api",
            "image": "orion/api:latest",
            "state": "running",
            "status": "Up 3 minutes",
            "command": "python app.py",
            "ports": [{"private_port": 8000, "public_port": 18000, "type": "tcp"}],
        }
    ]


def test_docker_ps_parser_parses_json_lines():
    rows = verb_adapters._parse_docker_ps_lines('{"ID":"abc","Image":"img","Names":"svc","State":"running","Status":"Up","Command":"python","Ports":"0.0.0.0:1->1/tcp"}\n')

    assert rows[0]["id"] == "abc"
    assert rows[0]["name"] == "svc"


def test_time_now_uses_requested_timezone():
    req = _plan_request("skills.system.time_now.v1", skill_args={"timezone": "UTC"})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})

    out, effects = asyncio.run(verb_adapters.TimeNowVerb().execute(ctx, req))

    assert effects == []
    data = json.loads(out.final_text)
    assert data["timezone"] == "UTC"
    assert data["local_iso"].endswith("+00:00")
    assert data["utc_iso"].endswith("+00:00")


def test_biometrics_snapshot_maps_mock_http(monkeypatch):
    payload = {
        "status": "OK",
        "reason": "fresh",
        "as_of": "2026-03-18T12:00:00+00:00",
        "freshness_s": 4.2,
        "constraint": "GPU_MEM",
        "cluster": {"composite": {"strain": 0.62, "stability": 0.44}, "trend": {"strain": {"trend": 0.6}}},
        "nodes": {"athena": {"summary": {"composites": {"strain": 0.62}}, "status": "OK"}},
    }
    monkeypatch.setattr(verb_adapters, "_http_json_get", lambda url, timeout_sec: payload)
    req = _plan_request("skills.biometrics.snapshot.v1")
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})

    out, _ = asyncio.run(verb_adapters.BiometricsSnapshotVerb().execute(ctx, req))

    data = json.loads(out.final_text)
    assert data["status"] == "OK"
    assert data["constraint"] == "GPU_MEM"
    assert data["cluster"]["composite"]["strain"] == 0.62


def test_landing_pad_metrics_snapshot_calls_rpc_get_stats():
    rpc_payload = PadRpcResponseV1(request_id="req-1", ok=True, result={"stats": {"ingested": 9}}).model_dump(mode="json")
    bus = _FakeBus(rpc_payload)
    req = _plan_request("skills.landing_pad.metrics_snapshot.v1")
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="exec"), "correlation_id": str(uuid4())})

    out, _ = asyncio.run(verb_adapters.LandingPadMetricsSnapshotVerb().execute(ctx, req))

    assert bus.calls
    channel, env, _, _ = bus.calls[0]
    assert channel == "orion:pad:rpc:request"
    assert env.payload["method"] == "get_stats"
    data = json.loads(out.final_text)
    assert data["stats"]["ingested"] == 9
