import asyncio
import importlib.util
import json
import os
import shutil
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

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


def test_tailscale_json_parsing_and_active_nodes():
    parsed = verb_adapters._parse_tailscale_status_json(
        {
            "BackendState": "Running",
            "Self": {"HostName": "athena", "TailscaleIPs": ["100.64.0.1"], "OS": "linux"},
            "Peer": {
                "p1": {"HostName": "zeus", "Online": True, "TailscaleIPs": ["100.64.0.2"], "OS": "linux"},
                "p2": {"HostName": "hera", "Online": False, "TailscaleIPs": ["100.64.0.3"], "OS": "linux"},
            },
        }
    )
    active = verb_adapters._derive_active_nodes(parsed)
    assert "athena" in active
    assert "zeus" in active
    assert "hera" not in active


def test_smartctl_json_normalization():
    normalized = verb_adapters._normalize_smartctl_device(
        node_name="athena",
        device="/dev/sda",
        payload={
            "device": {"protocol": "ATA"},
            "model_name": "Samsung",
            "serial_number": "SN-1",
            "smart_status": {"passed": True},
            "temperature": {"current": 31},
            "power_on_time": {"hours": 100},
        },
        exit_status=0,
    )
    assert normalized["protocol"] == "ata"
    assert normalized["overall_health"] == "passed"
    assert normalized["temperature_c"] == 31.0


def test_nvme_json_normalization():
    normalized = verb_adapters._normalize_nvme_smart_log(
        node_name="athena",
        device="/dev/nvme0n1",
        payload={"temperature": 36, "percentage_used": 12, "media_errors": 0},
    )
    assert normalized["protocol"] == "nvme"
    assert normalized["temperature_c"] == 36.0
    assert normalized["percentage_used"] == 12


def test_changed_file_to_service_inference_and_group_summary():
    paths = ["services/orion-actions/app/main.py", "orion/schemas/registry.py"]
    inferred = verb_adapters._infer_services_from_paths(paths)
    assert "orion-actions" in inferred
    assert "orion.schemas" in inferred
    grouped = verb_adapters._summarize_prs_by_service(
        [{"number": 12, "inferred_services": inferred}, {"number": 15, "inferred_services": ["orion-actions"]}]
    )
    assert any(item["service"] == "orion-actions" and item["pr_numbers"] == [12, 15] for item in grouped)


def _docker_prune_runner_factory():
    created = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    inspect_body = [
        {
            "Id": "abcd0000000000000000000000000000000000000000000000000000000000",
            "Created": created,
            "SizeRootFs": 2048,
            "Config": {"Labels": {}},
        }
    ]

    class _Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, command):
            if command[:2] == ["docker", "ps"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"ID":"abcd00000000","Names":"/stopped","Image":"img:v1","State":"exited"}\n',
                    stderr="",
                )
            if command[:3] == ["docker", "container", "inspect"]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(inspect_body), stderr="")
            if command[:3] == ["docker", "rm", "-f"]:
                return SimpleNamespace(returncode=0, stdout="abcd0000000000000000000000000000000000000000000000000000000000\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=f"unexpected:{command!r}")

    return _Runner


def test_docker_prune_dry_run_behavior(monkeypatch):
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _docker_prune_runner_factory())
    monkeypatch.setattr(verb_adapters.settings, "docker_prune_default_until", "")
    monkeypatch.setattr(verb_adapters.settings, "docker_protected_labels", "")
    req = _plan_request("skills.runtime.docker_prune_stopped_containers.v1", skill_args={"dry_run": True})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.DockerPruneStoppedContainersVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["status"] == "preview"
    assert data["run_mode"] == "preview"
    assert data["pruned_container_count"] == 0
    assert data["would_prune_count"] == 1
    assert "PREVIEW (no changes made)" in data["user_facing_summary"]


def test_docker_prune_execute_policy_gate(monkeypatch):
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _docker_prune_runner_factory())
    monkeypatch.setattr(verb_adapters.settings, "docker_prune_default_until", "")
    monkeypatch.setattr(verb_adapters.settings, "docker_protected_labels", "")
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_mutating_runtime_housekeeping", False)
    req = _plan_request("skills.runtime.docker_prune_stopped_containers.v1", skill_args={"execute": True})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.DockerPruneStoppedContainersVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["status"] == "blocked"
    assert data["run_mode"] == "execute"
    assert data["policy_blocked"] is True
    assert "SKILLS_ALLOW_MUTATING_RUNTIME_HOUSEKEEPING=false" in data["user_facing_summary"]


def test_docker_prune_natural_language_preview_phrases():
    prev, _ = verb_adapters._resolve_docker_prune_run_mode({"text": "Dry-run cleanup of stopped containers."})
    assert prev == "preview"
    prev2, _ = verb_adapters._resolve_docker_prune_run_mode({"text": "Show me which stopped containers would be pruned."})
    assert prev2 == "preview"


def test_docker_prune_natural_language_execute_prune():
    ex, _ = verb_adapters._resolve_docker_prune_run_mode({"text": "Prune stopped containers."})
    assert ex == "execute"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_docker_prune_live_preview_phrase_no_mutation(monkeypatch):
    """Live host Docker: preview only (natural-language dry-run cleanup phrase)."""
    monkeypatch.setattr(verb_adapters.settings, "docker_prune_default_until", "")
    monkeypatch.setattr(verb_adapters.settings, "docker_protected_labels", "")
    req = _plan_request(
        "skills.runtime.docker_prune_stopped_containers.v1",
        skill_args={"text": "Dry-run cleanup of stopped containers."},
    )
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.DockerPruneStoppedContainersVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["run_mode"] == "preview"
    assert data["mutated"] is False
    assert "PREVIEW (no changes made)" in data["user_facing_summary"]
    assert data["status"] == "preview"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_docker_prune_live_execute_phrase_policy_blocked_without_opt_in(monkeypatch):
    """Live host Docker: execute intent hits policy gate — must not call docker rm."""
    rm_called = {"n": 0}
    _RealSafe = verb_adapters.SafeCommandRunner

    class _CountRm:
        def __init__(self, **kwargs):
            self._inner = _RealSafe(**kwargs)

        def run(self, command):
            if len(command) >= 2 and command[0] == "docker" and command[1] == "rm":
                rm_called["n"] += 1
            return self._inner.run(command)

    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _CountRm)
    monkeypatch.setattr(verb_adapters.settings, "docker_prune_default_until", "")
    monkeypatch.setattr(verb_adapters.settings, "docker_protected_labels", "")
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_mutating_runtime_housekeeping", False)
    req = _plan_request(
        "skills.runtime.docker_prune_stopped_containers.v1",
        skill_args={"text": "Prune stopped containers."},
    )
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.DockerPruneStoppedContainersVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["run_mode"] == "execute"
    assert data["status"] == "blocked"
    assert rm_called["n"] == 0


def test_docker_prune_execute_runs_rm_when_policy_allows(monkeypatch):
    monkeypatch.setattr(verb_adapters, "SafeCommandRunner", _docker_prune_runner_factory())
    monkeypatch.setattr(verb_adapters.settings, "docker_prune_default_until", "")
    monkeypatch.setattr(verb_adapters.settings, "docker_protected_labels", "")
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_mutating_runtime_housekeeping", True)
    req = _plan_request("skills.runtime.docker_prune_stopped_containers.v1", skill_args={"execute": True})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.DockerPruneStoppedContainersVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["run_mode"] == "execute"
    assert data["status"] == "success"
    assert "EXECUTE:" in data["user_facing_summary"]
    assert data["pruned_container_count"] >= 1


def test_mesh_ops_round_happy_path_with_journal_write():
    class _Bus:
        def __init__(self):
            self.published = []

        async def publish(self, channel, envelope):
            self.published.append((channel, envelope.kind))

    async def _mesh(*args, **kwargs):
        return verb_adapters._skill_result_output(
            skill_name="skills.mesh.tailscale_mesh_status.v1",
            result={"nodes": [{"node_name": "athena", "peer_status_classification": "active"}]},
        ), []

    async def _disk(*args, **kwargs):
        return verb_adapters._skill_result_output(skill_name="skills.storage.disk_health_snapshot.v1", result={"summary": {"healthy": 1}}), []

    async def _prs(*args, **kwargs):
        return verb_adapters._skill_result_output(skill_name="skills.repo.github_recent_prs.v1", result={"available": True, "items": []}), []

    async def _docker(*args, **kwargs):
        return verb_adapters._skill_result_output(skill_name="skills.runtime.docker_prune_stopped_containers.v1", result={"status": "preview"}), []

    verb_adapters.TailscaleMeshStatusVerb.execute = _mesh
    verb_adapters.DiskHealthSnapshotVerb.execute = _disk
    verb_adapters.GithubRecentPullRequestsVerb.execute = _prs
    verb_adapters.DockerPruneStoppedContainersVerb.execute = _docker

    req = _plan_request("skills.mesh.mesh_ops_round.v1", skill_args={"write_journal": True, "include_docker_housekeeping": True})
    bus = _Bus()
    ctx = VerbContext(meta={"bus": bus, "source": ServiceRef(name="exec"), "correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.MeshOpsRoundVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["overall_health"] == "ok"
    assert data["journal_write"]["status"] == "published"
    assert bus.published and bus.published[0][0] == "orion:journal:write"


def test_mesh_ops_round_partial_failure_without_journal():
    async def _mesh(*args, **kwargs):
        return verb_adapters._skill_result_output(
            skill_name="skills.mesh.tailscale_mesh_status.v1",
            result={"nodes": []},
            ok=False,
            status="fail",
            error={"message": "no_mesh"},
        ), []

    async def _prs(*args, **kwargs):
        return verb_adapters._skill_result_output(skill_name="skills.repo.github_recent_prs.v1", result={"available": False}), []

    verb_adapters.TailscaleMeshStatusVerb.execute = _mesh
    verb_adapters.GithubRecentPullRequestsVerb.execute = _prs
    req = _plan_request("skills.mesh.mesh_ops_round.v1", skill_args={"write_journal": False})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.MeshOpsRoundVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["overall_health"] == "degraded"
    assert "mesh_presence_failed" in data["partial_failures"]
