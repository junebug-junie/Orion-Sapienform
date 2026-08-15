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

_GITHUB_RECENT_PRS_EXECUTE = verb_adapters.GithubRecentPullRequestsVerb.execute

spec = importlib.util.spec_from_file_location(
    f"{APP_PACKAGE_NAME}.actions_skill_registry", APP_DIR / "actions_skill_registry.py"
)
actions_skill_registry = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = actions_skill_registry
spec.loader.exec_module(actions_skill_registry)

spec = importlib.util.spec_from_file_location(
    f"{APP_PACKAGE_NAME}.capability_bridge", APP_DIR / "capability_bridge.py"
)
capability_bridge = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = capability_bridge
spec.loader.exec_module(capability_bridge)

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402
from orion.core.verbs.base import VerbContext  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest  # noqa: E402


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


def _vision_window_envelope(**overrides):
    envelope = {
        "window_id": "w-1",
        "stream_id": "cam-front",
        "start_ts": 1000.0,
        "end_ts": 1005.0,
        "summary": {
            "object_counts": {"door": 1, "chair": 3},
            "top_labels": [["chair", 3], ["door", 1]],
            "item_count": 4,
            "captions": ["a room with chairs and a door"],
            "label_counts": {"door": 1, "chair": 3},
            "detection_count": 4,
            "evidence": {"hard_labels": ["door", "chair"]},
        },
        "artifact_ids": ["art-1"],
        "artifact_uris": ["/mnt/telemetry/vision/frames/art-1.jpg"],
        "upstream_event_ids": ["evt-1"],
        "meta": {"internal": "plumbing"},
    }
    envelope.update(overrides)
    return envelope


def test_normalize_vision_window_current_drops_raw_frame_paths():
    """Privacy contract: artifact_uris/upstream_event_ids/meta (raw frame paths,
    bus correlation plumbing) must never survive normalization -- same contract
    PerceptionContextV1 already establishes for the situation brief."""
    payload = {
        "status": "ok",
        "source": "live_state",
        "snapshot_id": "w-1",
        "stream_id": "cam-front",
        "generated_at": 1005.0,
        "age_ms": 250,
        "envelope": _vision_window_envelope(),
    }

    result = verb_adapters._normalize_vision_window_current(payload)

    assert result["available"] is True
    assert result["status"] == "ok"
    assert result["window_id"] == "w-1"
    assert result["stream_id"] == "cam-front"
    assert result["item_count"] == 4
    assert result["detection_count"] == 4
    assert result["top_labels"] == [["chair", 3], ["door", 1]]
    assert result["captions"] == ["a room with chairs and a door"]
    assert "artifact_uris" not in result
    assert "upstream_event_ids" not in result
    assert "meta" not in result
    assert "/mnt/telemetry" not in json.dumps(result)


def test_normalize_vision_window_current_caps_captions_and_labels():
    envelope = _vision_window_envelope(
        summary={
            "object_counts": {},
            "top_labels": [["a", 5], ["b", 4], ["c", 3], ["d", 2], ["e", 1], ["f", 1]],
            "item_count": 1,
            "captions": ["one", "two", "three", "four"],
            "label_counts": {},
            "detection_count": 0,
            "evidence": {},
        }
    )
    payload = {
        "status": "ok",
        "snapshot_id": "w-2",
        "stream_id": None,
        "generated_at": 1005.0,
        "age_ms": 0,
        "envelope": envelope,
    }

    result = verb_adapters._normalize_vision_window_current(payload)

    assert len(result["top_labels"]) == 5
    assert len(result["captions"]) == 3


def test_normalize_vision_window_current_empty_status():
    result = verb_adapters._normalize_vision_window_current(
        {"status": "empty", "source": "none", "snapshot_id": None, "stream_id": None}
    )
    assert result["available"] is False
    assert result["status"] == "empty"
    assert result["captions"] == []


def test_normalize_vision_window_current_stale_still_available():
    """Regression: 'stale' means aged, not absent -- orion-vision-window's own
    http_current_stale_check only flips the status field, the envelope's real
    captions/labels stay populated. A caller must not read stale as no-data."""
    payload = {
        "status": "stale",
        "snapshot_id": "w-3",
        "stream_id": "cam-front",
        "generated_at": 1005.0,
        "age_ms": 400000,
        "envelope": _vision_window_envelope(),
    }

    result = verb_adapters._normalize_vision_window_current(payload)

    assert result["available"] is True
    assert result["status"] == "stale"
    assert result["captions"] == ["a room with chairs and a door"]


def test_normalize_vision_window_current_skips_malformed_top_label_entries():
    """Schema-drift defense: a top_labels entry that isn't a [label, count] pair
    (e.g. legacy recovery-store record, or a plain string) must be dropped, not
    raise -- this function runs outside LookAtCameraVerb's HTTP try/except."""
    envelope = _vision_window_envelope(
        summary={
            "object_counts": {},
            "top_labels": ["not_a_pair", ["door", 2], 42, None, ["chair"]],
            "item_count": 1,
            "captions": [],
            "label_counts": {},
            "detection_count": 2,
            "evidence": {},
        }
    )
    payload = {"status": "ok", "snapshot_id": "w-4", "stream_id": None, "envelope": envelope}

    result = verb_adapters._normalize_vision_window_current(payload)

    assert result["top_labels"] == [["door", 2]]


def test_look_at_camera_stream_id_is_url_quoted(monkeypatch):
    """A stream_id containing reserved URL characters must be percent-encoded,
    not interpolated raw into the path."""
    seen_urls = []

    def _fake_get(url, timeout_sec):
        seen_urls.append(url)
        return {"status": "empty", "source": "none", "snapshot_id": None, "stream_id": None}

    monkeypatch.setattr(verb_adapters, "_http_json_get", _fake_get)
    req = _plan_request(
        "skills.perception.look_at_camera.v1", skill_args={"stream_id": "cam/front?x=1"}
    )
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})

    asyncio.run(verb_adapters.LookAtCameraVerb().execute(ctx, req))

    assert seen_urls == [
        "http://orion-athena-vision-window:8000/api/vision-window/streams/cam%2Ffront%3Fx%3D1/current"
    ]


def test_look_at_camera_semantic_verb_resolves_via_capability_bridge():
    """End-to-end: the look_at_camera semantic verb's preferred_skill_families
    actually resolves to skills.perception.look_at_camera.v1 through the same
    resolve_capability_decision() path bound_capability_exec.py uses -- not
    just that the classifier functions agree in isolation."""
    registry = actions_skill_registry.ActionsSkillRegistry(verbs_dir=verb_adapters.VERBS_DIR)

    decision = capability_bridge.resolve_capability_decision(
        verb="look_at_camera",
        preferred_skill_families=["perception"],
        registry=registry,
    )

    assert decision.selected_skill == "skills.perception.look_at_camera.v1"
    assert decision.skill_family == "perception"
    assert decision.observational is True


def test_look_at_camera_maps_mock_http(monkeypatch):
    payload = {
        "status": "ok",
        "source": "live_state",
        "snapshot_id": "w-1",
        "stream_id": "cam-front",
        "generated_at": 1005.0,
        "age_ms": 250,
        "envelope": _vision_window_envelope(),
    }
    seen_urls = []

    def _fake_get(url, timeout_sec):
        seen_urls.append(url)
        return payload

    monkeypatch.setattr(verb_adapters, "_http_json_get", _fake_get)
    req = _plan_request("skills.perception.look_at_camera.v1")
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})

    out, effects = asyncio.run(verb_adapters.LookAtCameraVerb().execute(ctx, req))

    assert effects == []
    assert out.ok is True
    data = json.loads(out.final_text)
    assert data["available"] is True
    assert data["captions"] == ["a room with chairs and a door"]
    assert seen_urls == ["http://orion-athena-vision-window:8000/api/vision-window/current"]


def test_look_at_camera_uses_stream_scoped_endpoint_when_requested(monkeypatch):
    seen_urls = []

    def _fake_get(url, timeout_sec):
        seen_urls.append(url)
        return {"status": "empty", "source": "none", "snapshot_id": None, "stream_id": "cam-back"}

    monkeypatch.setattr(verb_adapters, "_http_json_get", _fake_get)
    req = _plan_request("skills.perception.look_at_camera.v1", skill_args={"stream_id": "cam-back"})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})

    out, _ = asyncio.run(verb_adapters.LookAtCameraVerb().execute(ctx, req))

    assert out.ok is False  # empty window -- honest failure, not empty-shell success
    assert seen_urls == [
        "http://orion-athena-vision-window:8000/api/vision-window/streams/cam-back/current"
    ]


def test_look_at_camera_never_raises_on_http_failure(monkeypatch):
    def _boom(url, timeout_sec):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(verb_adapters, "_http_json_get", _boom)
    req = _plan_request("skills.perception.look_at_camera.v1")
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})

    out, effects = asyncio.run(verb_adapters.LookAtCameraVerb().execute(ctx, req))

    assert effects == []
    assert out.ok is False
    assert out.error is not None


def test_look_at_camera_skill_family_and_risk_class():
    assert actions_skill_registry._family_for_skill("skills.perception.look_at_camera.v1") == "perception"
    risk_class, read_only, idempotent = actions_skill_registry._risk_for_skill(
        "skills.perception.look_at_camera.v1"
    )
    assert risk_class == "read_only"
    assert read_only is True
    assert idempotent is True


def test_look_at_camera_registered_in_manifest_with_correct_family():
    """End-to-end: the real skills.perception.look_at_camera.v1.yaml is
    auto-discovered by ActionsSkillRegistry's glob and classified correctly --
    not just that the classifier functions work in isolation."""
    registry = actions_skill_registry.ActionsSkillRegistry(verbs_dir=verb_adapters.VERBS_DIR)
    entry = next(
        (e for e in registry.list() if e.skill_id == "skills.perception.look_at_camera.v1"), None
    )
    assert entry is not None
    assert entry.family == "perception"
    assert entry.read_only is True
    assert entry.observational is True
    assert entry.requires_confirmation is False
    assert entry.requires_execute_opt_in is False


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


def test_github_recent_prs_includes_truncated_body(monkeypatch):
    monkeypatch.setattr(
        verb_adapters.GithubRecentPullRequestsVerb,
        "execute",
        _GITHUB_RECENT_PRS_EXECUTE,
    )
    sample_prs = [
        {
            "number": 42,
            "title": "Add compactor",
            "user": {"login": "juniper"},
            "state": "closed",
            "merged_at": "2026-07-08T12:00:00Z",
            "created_at": "2026-07-07T12:00:00Z",
            "updated_at": "2026-07-08T12:00:00Z",
            "labels": [],
            "base": {"ref": "main"},
            "head": {"ref": "feat/compactor"},
            "html_url": "https://github.com/acme/widgets/pull/42",
            "changed_files": 3,
            "body": "x" * 2500,
        }
    ]

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _urlopen(request, timeout=0):
        url = request.full_url
        if url.endswith("/pulls?state=closed&sort=updated&direction=desc&per_page=20"):
            return _Resp(sample_prs)
        if url.endswith("/files?per_page=100"):
            return _Resp([{"filename": "services/orion-hub/app/main.py"}])
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(verb_adapters, "urlopen", _urlopen)
    monkeypatch.setattr(verb_adapters.settings, "github_owner", "acme")
    monkeypatch.setattr(verb_adapters.settings, "github_repo", "widgets")
    monkeypatch.setattr(verb_adapters.settings, "github_token", "test-token")
    monkeypatch.setattr(verb_adapters.settings, "mesh_default_lookback_days", 7)

    req = _plan_request("skills.repo.github_recent_prs.v1", skill_args={"lookback_days": 7})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.GithubRecentPullRequestsVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["available"] is True
    assert len(data["items"]) == 1
    # Word-boundary truncation: an unbroken run of "x" has no whitespace to
    # break on, so it hard-cuts at the cap and appends the truncation marker
    # (rather than silently returning a body that looks complete).
    assert data["items"][0]["body"] == ("x" * 2000) + "…"


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


def test_mesh_up_all_services_policy_blocked(monkeypatch):
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_mesh_service_scripts", False)
    req = _plan_request("skills.mesh.up_all_services.v1")
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.MeshUpAllServicesVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["status"] == "blocked"
    assert data["policy_blocked"] is True
    assert out.ok is False


def test_mesh_up_all_services_runs_allowlisted_script(monkeypatch, tmp_path):
    script_dir = tmp_path / "mesh-utilities" / "common"
    script_dir.mkdir(parents=True)
    script = script_dir / "up_all_services.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -u\necho \"EXCLUDE_SERVICES_ADD=${EXCLUDE_SERVICES_ADD:-}\"\necho mesh_up_ok\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setattr(verb_adapters.self_study_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_mesh_service_scripts", True)
    monkeypatch.setattr(verb_adapters.settings, "skills_mesh_service_script_timeout_sec", 15.0)
    req = _plan_request("skills.mesh.up_all_services.v1")
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.MeshUpAllServicesVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert out.ok is True
    assert data["status"] == "success"
    tail = data.get("stdout_stderr_tail") or ""
    assert "orion-hub" in tail
    assert "mesh_up_ok" in tail


def test_mesh_refresh_service_envs_runs_allowlisted_script(monkeypatch, tmp_path):
    script_dir = tmp_path / "mesh-utilities" / "common"
    script_dir.mkdir(parents=True)
    script = script_dir / "refresh_service_envs.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -u\necho \"EXCLUDE_SERVICES_ADD=${EXCLUDE_SERVICES_ADD:-}\"\necho env_refresh_ok\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setattr(verb_adapters.self_study_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verb_adapters.settings, "skills_allow_mesh_service_scripts", True)
    monkeypatch.setattr(verb_adapters.settings, "skills_mesh_service_script_timeout_sec", 15.0)
    req = _plan_request("skills.mesh.refresh_service_envs.v1")
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.MeshRefreshServiceEnvsVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert out.ok is True
    assert data["status"] == "success"
    tail = data.get("stdout_stderr_tail") or ""
    assert "orion-hub" in tail
    assert "env_refresh_ok" in tail
