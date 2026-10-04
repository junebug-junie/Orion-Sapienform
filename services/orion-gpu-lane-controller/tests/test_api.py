"""app/main.py's HTTP surface: /health only.

GPU pool stage 5.6 deleted every HTTP control/status route (the GPU1 affect/agent flip, its
GPU_LANE_CONTROLLER_TOKEN, /v1/gpu-lane/status and /v1/gpu-slots/*): the pool's bus actuation
(app/actuator_bus.py) is the only way to move a card. Also the shared module loader for this suite.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_gpu_lane_controller"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.{name}", APP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


settings_module = _load("settings")
sys.modules[f"{APP_PACKAGE_NAME}.settings"] = settings_module
main_module = _load("main")


@pytest.fixture
def client(monkeypatch):
    # The bus chassis talks to a real bus -- irrelevant to these HTTP
    # contract tests and not something to stand a real Redis up for.
    monkeypatch.setattr(main_module.settings, "ORION_BUS_ENABLED", False)
    with TestClient(main_module.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@pytest.mark.parametrize("method,path", [
    ("post", "/v1/gpu-lane/flip"), ("get", "/v1/gpu-lane/status"),
    ("get", "/v1/gpu-slots/circe-gpu1/status"), ("get", "/v1/gpu-slots/circe-gpu2/status"),
    ("post", "/v1/gpu-slots/activate"), ("post", "/v1/gpu-lanes/flip"),
])
def test_stage5_6_legacy_http_routes_are_gone(client, method, path):
    resp = getattr(client, method)(path, json={"target": "agent"}) if method == "post" else client.get(path)
    assert resp.status_code in (404, 405)


def test_health_is_the_only_route():
    paths = {getattr(r, "path", "") for r in main_module.app.routes}
    assert {p for p in paths if p.startswith("/v1")} == set()


def test_stage5_6_gpu2_bridge_and_flip_modules_and_keys_are_gone():
    app_dir = SERVICE_DIR / "app"
    assert not (app_dir / "gpu2.py").exists() and not (app_dir / "lane_control.py").exists()
    fields = set(type(main_module.settings).model_fields)
    gone = {"GPU2_ENABLED", "GPU2_DIFFUSION_URL", "GPU2_AGENT_URL", "GPU2_DRAIN_TIMEOUT_SEC",
            "GPU2_MODEL_READY_TIMEOUT_SEC", "GPU2_POOL_FENCE_STATE_PATH", "GPU2_AUTHORITY", "GPU2_AUTHORITY_URL",
            "GPU_LANE_CONTROLLER_TOKEN", "GPU_LANE_HEALTH_POLL_SEC", "AGENT_GPU1_CUDA_VISIBLE_DEVICES",
            "AFFECT_COMPOSE_RELPATH", "AGENT_COMPOSE_RELPATH"}
    assert fields & gone == set()
    assert {"GPU_POOL_FENCE_STATE_PATH", "GPU_LANE_DRAIN_TIMEOUT_SEC"} <= fields
    # Renamed, not reset: the same file on the same volume, so the fence's last generation carries over.
    assert type(main_module.settings)(_env_file=None).GPU_POOL_FENCE_STATE_PATH == "/state/gpu2_pool_fence.json"
    assert type(main_module.settings)(_env_file=None).GPU_LANE_DRAIN_TIMEOUT_SEC == 300.0
    for rel in (".env_example", "docker-compose.yml"):
        text = (SERVICE_DIR / rel).read_text()
        live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        for key in ("GPU2_", "GPU_LANE_CONTROLLER_TOKEN"):
            assert key not in live, (rel, key)
    # No literal CUDA device index anywhere in the actuator: the card index comes from the YAML.
    for path in app_dir.glob("*.py"):
        assert "CUDA_VISIBLE_DEVICES" not in path.read_text(), path.name
