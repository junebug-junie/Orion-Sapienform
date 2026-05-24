# Orion Vision Frame Router — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `services/orion-vision-frame-router` — a bounded policy bridge that subscribes to `orion:vision:frames`, applies sampling/backpressure, and dispatches `VisionTaskRequestPayload` tasks to `orion:exec:request:VisionHostService` with preserved causality and reply tracking.

**Architecture:** Pure policy/dispatch service (no capture, no inference). Split into `settings`, `policy` (YAML + `FrameDispatchDecision`), `state` (per-camera counters + pending tasks), `envelopes` (`derive_child` host task builder), `metrics` (`RouterMetrics` + `SystemHealthV1`), `dispatcher` (frame + reply consumers + timeout sweeper), and `main` (FastAPI lifespan). Replies consumed via Redis `PSUBSCRIBE` on `orion:vision:reply:*`.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pydantic-settings v2, PyYAML, Redis bus (`OrionBusAsync`), shared `orion.schemas.vision`, `orion.schemas.telemetry.system_health`, pytest.

**Spec source:** Operator brief “orion-vision-frame-router” (2026-05-23).

**Execution isolation (mandatory before Task 1):**

- **Do not touch** the main workspace branch `feat/repair-pressure-v1` (Claude is building there).
- Create an isolated worktree on a **new branch** (does not exist yet):

```bash
cd /mnt/scripts/Orion-Sapienform
test "$(git branch --show-current)" = "feat/repair-pressure-v1" && echo "main checkout OK — do not implement here"
git fetch origin main 2>/dev/null || true
git worktree add .worktrees/orion-vision-frame-router -b feat/orion-vision-frame-router-v1 origin/main
cd .worktrees/orion-vision-frame-router
```

- All implementation, commits, and PR push happen **only** inside `.worktrees/orion-vision-frame-router`.
- After updating `services/orion-vision-frame-router/.env_example`, copy to `services/orion-vision-frame-router/.env` in the worktree (gitignored, local sync). **Do not** copy other changed files back to the main checkout — only `.env` may be synced locally.
- Post-implementation: run **requesting-code-review** subagent, fix all findings, write `docs/superpowers/pr-reports/2026-05-23-orion-vision-frame-router-pr.md`, `git push -u origin feat/orion-vision-frame-router-v1`, `gh pr create`.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-vision-frame-router/app/settings.py` | Env-backed router config |
| `services/orion-vision-frame-router/app/policy.py` | YAML policy load + `decide()` + task request builder |
| `services/orion-vision-frame-router/app/state.py` | `CameraState`, `PendingTask`, `RouterState` |
| `services/orion-vision-frame-router/app/envelopes.py` | `make_host_task_envelope()` via `derive_child` |
| `services/orion-vision-frame-router/app/metrics.py` | `RouterMetrics`, health envelope builder |
| `services/orion-vision-frame-router/app/dispatcher.py` | Frame/reply handling, timeout sweeper |
| `services/orion-vision-frame-router/app/main.py` | `FrameRouterService`, FastAPI lifespan |
| `services/orion-vision-frame-router/app/__init__.py` | Package marker |
| `services/orion-vision-frame-router/requirements.txt` | Service deps (no opencv/torch) |
| `services/orion-vision-frame-router/Dockerfile` | Repo-root context image |
| `services/orion-vision-frame-router/docker-compose.yml` | Frame-router only; mesh Redis via `ORION_BUS_URL` |
| `services/orion-vision-frame-router/.env_example` | Full env contract |
| `services/orion-vision-frame-router/.env` | Copy of example (gitignored) |
| `services/orion-vision-frame-router/README.md` | Role, channels, smoke test |
| `config/vision_frame_router.yaml` | Default policy file |
| `orion/bus/channels.yaml` | Add frame-router to producers/consumers |
| `services/orion-vision-frame-router/tests/test_settings.py` | Settings defaults |
| `services/orion-vision-frame-router/tests/test_policy.py` | Policy decisions |
| `services/orion-vision-frame-router/tests/test_state.py` | State mutations |
| `services/orion-vision-frame-router/tests/test_envelopes.py` | Envelope contract |
| `services/orion-vision-frame-router/tests/test_dispatcher.py` | Dispatch + reply + timeout (fake bus) |
| `services/orion-vision-frame-router/tests/test_no_contamination.py` | Import boundary guard |

**Explicit non-goals:** camera capture, frame storage, cv2/YOLO/detector_worker, LLM calls, substrate emitters, SQL/RDF/vector writes, changes to `orion-vision-host` behavior (unless a one-line catalog/doc fix).

**Envelope contract notes:**

- Frame intake kind: `vision.frame.pointer` (from Retina/Edge).
- Host dispatch kind: `vision.task.request` (see `services/orion-vision-host/scripts/publish_test_task.py`).
- Host reply kind: `vision.task.result` on `reply_to` channel `orion:vision:reply:<correlation_id>`.
- Use `BaseEnvelope.derive_child()` so `correlation_id` and `causality_chain` carry from frame → task (`orion/core/bus/bus_schemas.py`).
- `BaseEnvelope.schema_id` default is `"orion.envelope"` — do not override unless bus enforcement requires it.

---

### Task 0: Worktree bootstrap + service skeleton

**Files:**
- Create: `services/orion-vision-frame-router/app/__init__.py`
- Create: `services/orion-vision-frame-router/requirements.txt`

- [ ] **Step 1: Create worktree and branch**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
git worktree add .worktrees/orion-vision-frame-router -b feat/orion-vision-frame-router-v1 origin/main
cd .worktrees/orion-vision-frame-router
git branch --show-current
```

Expected: `feat/orion-vision-frame-router-v1`.

- [ ] **Step 2: Create package skeleton**

Create `services/orion-vision-frame-router/app/__init__.py`:

```python
"""Orion vision frame router — policy bridge between Retina and Host."""
```

Create `services/orion-vision-frame-router/requirements.txt`:

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
redis[hiredis]==5.0.7
pydantic==2.9.2
pydantic-settings==2.5.2
loguru==0.7.2
PyYAML==6.0.2
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-vision-frame-router/
git commit -m "chore(vision-frame-router): bootstrap service package"
```

---

### Task 1: Settings

**Files:**
- Create: `services/orion-vision-frame-router/app/settings.py`
- Create: `services/orion-vision-frame-router/tests/test_settings.py`
- Create: `services/orion-vision-frame-router/.env_example`
- Create: `services/orion-vision-frame-router/.env` (copy, not committed)

- [ ] **Step 1: Write failing settings test**

Create `services/orion-vision-frame-router/tests/test_settings.py`:

```python
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))


def _reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    for key in list(os.environ):
        if key.startswith(
            (
                "SERVICE_",
                "LOG_",
                "ORION_BUS_",
                "CHANNEL_",
                "ROUTER_",
                "DEFAULT_",
                "MAX_",
                "TASK_",
                "REQUIRE_",
                "DRY_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.settings as mod

    importlib.reload(mod)
    return mod.Settings()


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch)
    assert s.SERVICE_NAME == "vision-frame-router"
    assert s.SERVICE_VERSION == "0.1.0"
    assert s.CHANNEL_FRAMES_IN == "orion:vision:frames"
    assert s.CHANNEL_HOST_INTAKE == "orion:exec:request:VisionHostService"
    assert s.CHANNEL_HOST_ARTIFACTS == "orion:vision:artifacts"
    assert s.CHANNEL_REPLY_PREFIX == "orion:vision:reply"
    assert s.CHANNEL_SYSTEM_HEALTH == "orion:system:health"
    assert s.ROUTER_ENABLED is True
    assert s.DEFAULT_TASK_TYPE == "retina_fast"
    assert s.DEFAULT_EVERY_N_FRAMES == 10
    assert s.MAX_INFLIGHT_TOTAL == 2
    assert s.DRY_RUN is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-frame-router
PYTHONPATH=../../orion:../../services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests/test_settings.py -v
```

Expected: FAIL (`ModuleNotFoundError: app.settings`).

- [ ] **Step 3: Implement settings**

Create `services/orion-vision-frame-router/app/settings.py`:

```python
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-frame-router"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    CHANNEL_FRAMES_IN: str = "orion:vision:frames"
    CHANNEL_HOST_INTAKE: str = "orion:exec:request:VisionHostService"
    CHANNEL_HOST_ARTIFACTS: str = "orion:vision:artifacts"
    CHANNEL_REPLY_PREFIX: str = "orion:vision:reply"
    CHANNEL_SYSTEM_HEALTH: str = "orion:system:health"

    ROUTER_ENABLED: bool = True
    ROUTER_POLICY_PATH: str = "/app/config/vision_frame_router.yaml"

    DEFAULT_TASK_TYPE: str = "retina_fast"
    DEFAULT_EVERY_N_FRAMES: int = 10
    DEFAULT_MIN_SECONDS_PER_CAMERA: float = 5.0

    MAX_INFLIGHT_TOTAL: int = 2
    MAX_INFLIGHT_PER_CAMERA: int = 1
    TASK_TIMEOUT_SECONDS: float = 30.0

    REQUIRE_IMAGE_PATH_EXISTS: bool = True
    DRY_RUN: bool = False

    HEALTH_INTERVAL_SECONDS: float = 10.0
```

Create `services/orion-vision-frame-router/.env_example`:

```bash
SERVICE_NAME=vision-frame-router
SERVICE_VERSION=0.1.0
LOG_LEVEL=INFO

ORION_BUS_URL=redis://localhost:6379/0
ORION_BUS_ENFORCE_CATALOG=false

CHANNEL_FRAMES_IN=orion:vision:frames
CHANNEL_HOST_INTAKE=orion:exec:request:VisionHostService
CHANNEL_HOST_ARTIFACTS=orion:vision:artifacts
CHANNEL_REPLY_PREFIX=orion:vision:reply
CHANNEL_SYSTEM_HEALTH=orion:system:health

ROUTER_ENABLED=true
ROUTER_POLICY_PATH=/app/config/vision_frame_router.yaml

DEFAULT_TASK_TYPE=retina_fast
DEFAULT_EVERY_N_FRAMES=10
DEFAULT_MIN_SECONDS_PER_CAMERA=5.0

MAX_INFLIGHT_TOTAL=2
MAX_INFLIGHT_PER_CAMERA=1
TASK_TIMEOUT_SECONDS=30.0

REQUIRE_IMAGE_PATH_EXISTS=true
DRY_RUN=false

HEALTH_INTERVAL_SECONDS=10.0
```

Copy: `cp services/orion-vision-frame-router/.env_example services/orion-vision-frame-router/.env`

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-frame-router
PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests/test_settings.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-frame-router/app/settings.py \
  services/orion-vision-frame-router/tests/test_settings.py \
  services/orion-vision-frame-router/.env_example
git commit -m "feat(vision-frame-router): add settings and env contract"
```

---

### Task 2: Policy config YAML

**Files:**
- Create: `config/vision_frame_router.yaml`

- [ ] **Step 1: Add default policy file**

Create `config/vision_frame_router.yaml`:

```yaml
version: 1

defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 10
  min_seconds_between_tasks_per_camera: 5
  max_inflight_per_camera: 1
  request: {}

global:
  max_inflight_total: 2
  drop_when_busy: true
  require_image_path_exists: true

cameras:
  mock-cam-01:
    enabled: true
    task_type: retina_fast
    every_n_frames: 3
    min_seconds_between_tasks_per_camera: 2

  porch_eye:
    enabled: true
    task_type: detect_open_vocab
    every_n_frames: 5
    min_seconds_between_tasks_per_camera: 3
    request:
      prompts:
        - person
        - package
        - vehicle
        - animal

  kitchen_eye:
    enabled: false
```

- [ ] **Step 2: Commit**

```bash
git add config/vision_frame_router.yaml
git commit -m "feat(vision-frame-router): add default dispatch policy config"
```

---

### Task 3: Router state

**Files:**
- Create: `services/orion-vision-frame-router/app/state.py`
- Create: `services/orion-vision-frame-router/tests/test_state.py`

- [ ] **Step 1: Write failing state tests**

Create `services/orion-vision-frame-router/tests/test_state.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.state import RouterState


def test_mark_dispatched_tracks_inflight() -> None:
    rs = RouterState()
    rs.mark_dispatched(
        correlation_id="abc",
        camera_id="cam1",
        image_path="/tmp/x.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:abc",
        now=100.0,
        frame_ts=99.0,
    )
    assert rs.inflight_total() == 1
    assert "abc" in rs.cameras["cam1"].inflight
    assert rs.pending["abc"].task_type == "retina_fast"


def test_clear_pending_drops_inflight() -> None:
    rs = RouterState()
    rs.mark_dispatched(
        correlation_id="abc",
        camera_id="cam1",
        image_path="/tmp/x.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:abc",
        now=100.0,
        frame_ts=None,
    )
    rs.clear_pending("abc", now=110.0)
    assert rs.inflight_total() == 0
    assert "abc" not in rs.pending
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests/test_state.py -v
```

- [ ] **Step 3: Implement state**

Create `services/orion-vision-frame-router/app/state.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class CameraState(BaseModel):
    frames_seen: int = 0
    frames_dispatched: int = 0
    last_dispatch_ts: float | None = None
    inflight: set[str] = Field(default_factory=set)
    last_skip_reason: str | None = None


class PendingTask(BaseModel):
    correlation_id: str
    camera_id: str
    frame_ts: float | None
    image_path: str
    task_type: str
    dispatched_at: float
    reply_to: str


class RouterState:
    def __init__(self) -> None:
        self.cameras: dict[str, CameraState] = {}
        self.pending: dict[str, PendingTask] = {}

    def camera(self, camera_id: str) -> CameraState:
        if camera_id not in self.cameras:
            self.cameras[camera_id] = CameraState()
        return self.cameras[camera_id]

    def inflight_total(self) -> int:
        return len(self.pending)

    def mark_seen(self, camera_id: str) -> CameraState:
        cam = self.camera(camera_id)
        cam.frames_seen += 1
        return cam

    def mark_dispatched(
        self,
        *,
        correlation_id: str,
        camera_id: str,
        image_path: str,
        task_type: str,
        reply_to: str,
        now: float,
        frame_ts: float | None,
    ) -> None:
        cam = self.camera(camera_id)
        cam.frames_dispatched += 1
        cam.last_dispatch_ts = now
        cam.inflight.add(correlation_id)
        self.pending[correlation_id] = PendingTask(
            correlation_id=correlation_id,
            camera_id=camera_id,
            frame_ts=frame_ts,
            image_path=image_path,
            task_type=task_type,
            dispatched_at=now,
            reply_to=reply_to,
        )

    def clear_pending(self, correlation_id: str, *, now: float) -> PendingTask | None:
        task = self.pending.pop(correlation_id, None)
        if not task:
            return None
        cam = self.camera(task.camera_id)
        cam.inflight.discard(correlation_id)
        return task

    def expired_correlation_ids(self, *, now: float, timeout_s: float) -> list[str]:
        out: list[str] = []
        for cid, task in list(self.pending.items()):
            if now - task.dispatched_at >= timeout_s:
                out.append(cid)
        return out
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-frame-router/app/state.py \
  services/orion-vision-frame-router/tests/test_state.py
git commit -m "feat(vision-frame-router): add in-memory router state"
```

---

### Task 4: Policy engine

**Files:**
- Create: `services/orion-vision-frame-router/app/policy.py`
- Create: `services/orion-vision-frame-router/tests/test_policy.py`

- [ ] **Step 1: Write failing policy tests**

Create `services/orion-vision-frame-router/tests/test_policy.py`:

```python
from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload

from app.policy import FrameDispatchPolicy, load_policy_file
from app.settings import Settings
from app.state import RouterState


@pytest.fixture
def policy_path(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 2
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 1
  request: {}
global:
  max_inflight_total: 2
  drop_when_busy: true
  require_image_path_exists: false
cameras:
  porch_eye:
    enabled: true
    task_type: detect_open_vocab
    every_n_frames: 1
    min_seconds_between_tasks_per_camera: 0
    request:
      prompts: [person, package]
  kitchen_eye:
    enabled: false
""",
        encoding="utf-8",
    )
    return p


def _frame_env(camera_id: str = "cam-a", image_path: str = "/tmp/f.jpg") -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path=image_path,
        camera_id=camera_id,
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.1.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def test_every_nth_frame_sampling(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    env = _frame_env("cam-a")
    d1 = policy.decide(env, state, now=1.0)
    assert d1.should_dispatch is False
    assert d1.reason == "frame_sampled_out"
    d2 = policy.decide(env, state, now=2.0)
    assert d2.should_dispatch is True
    assert d2.task_type == "retina_fast"


def test_disabled_camera_skips(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("kitchen_eye"), state, now=1.0)
    assert decision.should_dispatch is False
    assert decision.reason == "camera_disabled"


def test_camera_override_task_type(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("porch_eye"), state, now=1.0)
    assert decision.should_dispatch is True
    assert decision.task_type == "detect_open_vocab"
    assert decision.request_overrides.get("prompts") == ["person", "package"]


def test_global_inflight_limit(policy_path: Path) -> None:
    settings = Settings(
        ROUTER_POLICY_PATH=str(policy_path),
        REQUIRE_IMAGE_PATH_EXISTS=False,
        MAX_INFLIGHT_TOTAL=1,
    )
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    state.mark_dispatched(
        correlation_id="x",
        camera_id="cam-a",
        image_path="/tmp/a.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:x",
        now=1.0,
        frame_ts=None,
    )
    decision = policy.decide(_frame_env("cam-b"), state, now=2.0)
    assert decision.should_dispatch is False
    assert decision.reason == "global_inflight_limit"


def test_missing_image_path_skips(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=True)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("cam-a", image_path=""), state, now=1.0)
    assert decision.should_dispatch is False
    assert decision.reason == "missing_image_path"
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement policy**

Create `services/orion-vision-frame-router/app/policy.py` with:

- `FrameDispatchDecision` pydantic model (`should_dispatch`, `task_type`, `request_overrides`, `policy_name`, `reason`)
- `SUPPORTED_TASK_TYPES = {"retina_fast", "embed_image", "detect_open_vocab", "caption_frame"}`
- `load_policy_file(path) -> dict`
- `FrameDispatchPolicy` with `load(settings)`, `resolve_camera_policy(camera_id)`, `decide(env, state, now) -> FrameDispatchDecision`
- `build_task_request(frame, decision) -> VisionTaskRequestPayload`

Core `decide` logic (implement exactly):

```python
def decide(self, env: BaseEnvelope, state: RouterState, *, now: float) -> FrameDispatchDecision:
    if not self.settings.ROUTER_ENABLED:
        return FrameDispatchDecision(should_dispatch=False, policy_name="env", reason="router_disabled")

    frame = VisionFramePointerPayload.model_validate(env.payload)
    camera_id = frame.camera_id or "unknown"
    cam_state = state.mark_seen(camera_id)
    cam_policy, policy_name = self.resolve_camera_policy(camera_id)

    if not cam_policy.get("enabled", True):
        cam_state.last_skip_reason = "camera_disabled"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="camera_disabled")

    image_path = (frame.image_path or "").strip()
    require_exists = bool(
        cam_policy.get("require_image_path_exists", self.global_cfg.get("require_image_path_exists", self.settings.REQUIRE_IMAGE_PATH_EXISTS))
    )
    if not image_path:
        cam_state.last_skip_reason = "missing_image_path"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="missing_image_path")
    if require_exists and not os.path.isfile(image_path):
        cam_state.last_skip_reason = "image_path_not_visible"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="image_path_not_visible")

    every_n = int(cam_policy.get("every_n_frames", self.settings.DEFAULT_EVERY_N_FRAMES))
    if every_n > 1 and (cam_state.frames_seen % every_n) != 0:
        cam_state.last_skip_reason = "frame_sampled_out"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="frame_sampled_out")

    min_s = float(cam_policy.get("min_seconds_between_tasks_per_camera", self.settings.DEFAULT_MIN_SECONDS_PER_CAMERA))
    if cam_state.last_dispatch_ts is not None and (now - cam_state.last_dispatch_ts) < min_s:
        cam_state.last_skip_reason = "camera_rate_limited"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="camera_rate_limited")

    max_cam = int(cam_policy.get("max_inflight_per_camera", self.settings.MAX_INFLIGHT_PER_CAMERA))
    if len(cam_state.inflight) >= max_cam:
        cam_state.last_skip_reason = "camera_inflight_limit"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="camera_inflight_limit")

    max_total = int(self.global_cfg.get("max_inflight_total", self.settings.MAX_INFLIGHT_TOTAL))
    if state.inflight_total() >= max_total:
        cam_state.last_skip_reason = "global_inflight_limit"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="global_inflight_limit")

    task_type = str(cam_policy.get("task_type", self.settings.DEFAULT_TASK_TYPE))
    if task_type not in SUPPORTED_TASK_TYPES:
        cam_state.last_skip_reason = "unsupported_task_type"
        return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="unsupported_task_type")

    request_overrides = dict(cam_policy.get("request") or {})
    return FrameDispatchDecision(
        should_dispatch=True,
        task_type=task_type,
        request_overrides=request_overrides,
        policy_name=policy_name,
        reason="dispatch",
    )
```

`build_task_request`:

```python
def build_task_request(
    self,
    frame: VisionFramePointerPayload,
    env: BaseEnvelope,
    decision: FrameDispatchDecision,
) -> VisionTaskRequestPayload:
    base_request: dict[str, Any] = {"image_path": frame.image_path}
    base_request.update(decision.request_overrides or {})
    meta = {
        "camera_id": frame.camera_id,
        "stream_id": frame.stream_id,
        "frame_ts": frame.frame_ts,
        "source_frame_envelope_id": str(env.id),
        "source_frame_correlation_id": str(env.correlation_id),
        "router_policy": decision.policy_name,
    }
    return VisionTaskRequestPayload(
        task_type=decision.task_type or "retina_fast",
        request=base_request,
        meta=meta,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-frame-router/app/policy.py \
  services/orion-vision-frame-router/tests/test_policy.py
git commit -m "feat(vision-frame-router): add YAML dispatch policy engine"
```

---

### Task 5: Host task envelopes

**Files:**
- Create: `services/orion-vision-frame-router/app/envelopes.py`
- Create: `services/orion-vision-frame-router/tests/test_envelopes.py`

- [ ] **Step 1: Write failing envelope tests**

```python
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload

from app.envelopes import make_host_task_envelope


def test_host_task_envelope_contract() -> None:
    corr = uuid4()
    frame = VisionFramePointerPayload(image_path="/mnt/x.jpg", camera_id="cam1", frame_ts=1.0)
    frame_env = BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.2.0"),
        correlation_id=corr,
        payload=frame.model_dump(mode="json"),
    )
    task = VisionTaskRequestPayload(
        task_type="retina_fast",
        request={"image_path": "/mnt/x.jpg"},
        meta={"camera_id": "cam1"},
    )
    reply_to = f"orion:vision:reply:{corr}"
    out = make_host_task_envelope(
        frame_env=frame_env,
        frame=frame,
        task=task,
        service_name="vision-frame-router",
        service_version="0.1.0",
        reply_to=reply_to,
    )
    assert out.kind == "vision.task.request"
    assert out.reply_to == reply_to
    assert out.correlation_id == corr
    assert len(out.causality_chain) == 1
    assert out.causality_chain[0].kind == "vision.frame.pointer"
    validated = VisionTaskRequestPayload.model_validate(out.payload)
    assert validated.task_type == "retina_fast"
```

- [ ] **Step 2: Run — FAIL, implement, PASS**

Create `services/orion-vision-frame-router/app/envelopes.py`:

```python
from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload


def make_host_task_envelope(
    *,
    frame_env: BaseEnvelope,
    frame: VisionFramePointerPayload,
    task: VisionTaskRequestPayload,
    service_name: str,
    service_version: str,
    reply_to: str,
) -> BaseEnvelope:
    _ = frame  # lineage anchor; payload lives in task
    return frame_env.derive_child(
        kind="vision.task.request",
        source=ServiceRef(name=service_name, version=service_version),
        payload=task,
        reply_to=reply_to,
    )
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-vision-frame-router/app/envelopes.py \
  services/orion-vision-frame-router/tests/test_envelopes.py
git commit -m "feat(vision-frame-router): derive host task envelopes from frame lineage"
```

---

### Task 6: Metrics + health

**Files:**
- Create: `services/orion-vision-frame-router/app/metrics.py`

- [ ] **Step 1: Implement metrics module**

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from .state import RouterState

BOOT_ID = str(uuid.uuid4())


@dataclass
class RouterMetrics:
    frames_seen_total: int = 0
    frames_dispatched_total: int = 0
    frames_skipped_total: int = 0
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    host_replies_total: int = 0
    host_errors_total: int = 0
    host_timeouts_total: int = 0
    last_error: str | None = None

    def record_skip(self, reason: str) -> None:
        self.frames_skipped_total += 1
        self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1

    def record_dispatch(self) -> None:
        self.frames_dispatched_total += 1

    def record_seen(self) -> None:
        self.frames_seen_total += 1


def make_health_envelope(
    *,
    service_name: str,
    service_version: str,
    router_enabled: bool,
    dry_run: bool,
    policy_path: str,
    metrics: RouterMetrics,
    state: RouterState,
    status: str = "ok",
) -> BaseEnvelope:
    payload = SystemHealthV1(
        service=service_name,
        version=service_version,
        boot_id=BOOT_ID,
        last_seen_ts=datetime.now(timezone.utc),
        status=status,  # type: ignore[arg-type]
        details={
            "frames_seen_total": metrics.frames_seen_total,
            "frames_dispatched_total": metrics.frames_dispatched_total,
            "frames_skipped_total": metrics.frames_skipped_total,
            "inflight_total": state.inflight_total(),
            "pending_count": len(state.pending),
            "policy_path": policy_path,
            "router_enabled": router_enabled,
            "dry_run": dry_run,
            "skip_reason_counts": dict(metrics.skip_reason_counts),
            "last_error": metrics.last_error,
        },
    )
    return BaseEnvelope(
        kind="system.health.v1",
        source=ServiceRef(name=service_name, version=service_version),
        payload=payload.model_dump(mode="json"),
    )
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-vision-frame-router/app/metrics.py
git commit -m "feat(vision-frame-router): add router metrics and health envelope"
```

---

### Task 7: Dispatcher (frame in, reply out, timeouts)

**Files:**
- Create: `services/orion-vision-frame-router/app/dispatcher.py`
- Create: `services/orion-vision-frame-router/tests/test_dispatcher.py`

- [ ] **Step 1: Write failing dispatcher tests with fake bus**

Create `services/orion-vision-frame-router/tests/test_dispatcher.py` using a minimal fake:

```python
class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope: object) -> None:
        self.published.append((channel, envelope))
```

Tests to include:

1. `test_valid_frame_publishes_host_task` — one publish to `CHANNEL_HOST_INTAKE`, kind `vision.task.request`
2. `test_sampled_out_frame_publishes_nothing`
3. `test_dry_run_records_without_publish`
4. `test_invalid_payload_increments_error` — bad payload dict, no publish
5. `test_reply_clears_pending` — simulate `handle_reply_envelope`
6. `test_timeout_clears_pending` — `sweep_timeouts(now=...)`

- [ ] **Step 2: Implement dispatcher**

Create `services/orion-vision-frame-router/app/dispatcher.py` with `FrameDispatcher` class:

```python
class FrameDispatcher:
    def __init__(self, *, settings: Settings, policy: FrameDispatchPolicy, state: RouterState, metrics: RouterMetrics, bus: OrionBusAsync | None):
        ...

    async def handle_frame_envelope(self, env: BaseEnvelope) -> None:
        try:
            frame = VisionFramePointerPayload.model_validate(env.payload)
        except Exception as exc:
            self.metrics.last_error = f"invalid_frame_payload: {exc}"
            return

        self.metrics.record_seen()
        decision = self.policy.decide(env, self.state, now=time.time())
        if not decision.should_dispatch:
            self.metrics.record_skip(decision.reason)
            return

        task = self.policy.build_task_request(frame, env, decision)
        corr = str(env.correlation_id)
        reply_to = f"{self.settings.CHANNEL_REPLY_PREFIX}:{corr}"
        task_env = make_host_task_envelope(
            frame_env=env,
            frame=frame,
            task=task,
            service_name=self.settings.SERVICE_NAME,
            service_version=self.settings.SERVICE_VERSION,
            reply_to=reply_to,
        )

        if not self.settings.DRY_RUN and self.bus:
            await self.bus.publish(self.settings.CHANNEL_HOST_INTAKE, task_env)

        self.state.mark_dispatched(
            correlation_id=corr,
            camera_id=frame.camera_id or "unknown",
            image_path=frame.image_path or "",
            task_type=task.task_type,
            reply_to=reply_to,
            now=time.time(),
            frame_ts=frame.frame_ts,
        )
        self.metrics.record_dispatch()

    async def handle_reply_envelope(self, env: BaseEnvelope) -> None:
        from orion.schemas.vision import VisionTaskResultPayload

        try:
            result = VisionTaskResultPayload.model_validate(env.payload)
        except Exception as exc:
            self.metrics.last_error = f"invalid_reply_payload: {exc}"
            return

        corr = str(env.correlation_id)
        cleared = self.state.clear_pending(corr, now=time.time())
        if not cleared:
            return
        self.metrics.host_replies_total += 1
        if not result.ok:
            self.metrics.host_errors_total += 1

    def sweep_timeouts(self, *, now: float) -> int:
        expired = self.state.expired_correlation_ids(now=now, timeout_s=self.settings.TASK_TIMEOUT_SECONDS)
        for cid in expired:
            self.state.clear_pending(cid, now=now)
            self.metrics.host_timeouts_total += 1
        return len(expired)
```

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

```bash
git add services/orion-vision-frame-router/app/dispatcher.py \
  services/orion-vision-frame-router/tests/test_dispatcher.py
git commit -m "feat(vision-frame-router): add frame dispatcher with reply and timeout handling"
```

---

### Task 8: Main service loops

**Files:**
- Create: `services/orion-vision-frame-router/app/main.py`

- [ ] **Step 1: Implement `FrameRouterService`**

`main.py` responsibilities:

- Load `FrameDispatchPolicy.load(settings)` at start; reload optional (skip for MVP).
- `OrionBusAsync` connect.
- Task A: subscribe `CHANNEL_FRAMES_IN`, decode, `create_task(dispatcher.handle_frame_envelope)`.
- Task B: `subscribe(f"{CHANNEL_REPLY_PREFIX}:*", patterns=True)` — use `OrionBusAsync.subscribe(..., patterns=True)` per `orion/core/bus/async_service.py`.
- Task C: every 1s `dispatcher.sweep_timeouts(now=time.time())`.
- Task D: every `HEALTH_INTERVAL_SECONDS` publish `make_health_envelope(...)` to `CHANNEL_SYSTEM_HEALTH`.
- FastAPI `/healthz` returns metrics snapshot JSON.

Pattern for consumer (mirror host):

```python
async with self.bus.subscribe(settings.CHANNEL_FRAMES_IN) as pubsub:
    async for msg in self.bus.iter_messages(pubsub):
        decoded = self.bus.codec.decode(msg.get("data"))
        if decoded.ok and decoded.envelope:
            asyncio.create_task(self.dispatcher.handle_frame_envelope(decoded.envelope))
```

Reply loop uses `patterns=True` on `f"{settings.CHANNEL_REPLY_PREFIX}:*"`.

- [ ] **Step 2: Manual smoke (optional in worktree)**

```bash
cd services/orion-vision-frame-router
# ORION_BUS_URL must point at existing mesh Redis (see .env)
docker compose up -d --build orion-vision-frame-router
# with host/retina running on shared app-net + frame volume
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-vision-frame-router/app/main.py
git commit -m "feat(vision-frame-router): wire bus consumers and health loop"
```

---

### Task 9: Docker, README, env parity

**Files:**
- Create: `services/orion-vision-frame-router/Dockerfile`
- Create: `services/orion-vision-frame-router/docker-compose.yml`
- Create: `services/orion-vision-frame-router/README.md`

- [ ] **Step 1: Dockerfile (repo-root context)**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/orion-vision-frame-router/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-vision-frame-router/app ./app
COPY orion ./orion
COPY config/vision_frame_router.yaml /app/config/vision_frame_router.yaml

ENV PYTHONPATH=/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
```

- [ ] **Step 2: docker-compose.yml**

Do not define a local `redis` service in this compose file. The frame router must connect to the existing Orion mesh Redis via `ORION_BUS_URL`. This service should never publish host port `6379`; Redis is shared infrastructure, not owned by the frame-router compose project.

```yaml
services:
  orion-vision-frame-router:
    build:
      context: ../..
      dockerfile: services/orion-vision-frame-router/Dockerfile
    container_name: orion-${PROJECT:-dev}-vision-frame-router
    env_file:
      - .env
    environment:
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENFORCE_CATALOG=${ORION_BUS_ENFORCE_CATALOG:-false}
      - ROUTER_POLICY_PATH=/app/config/vision_frame_router.yaml
      - DRY_RUN=${DRY_RUN:-false}
    volumes:
      - ./app:/app/app
      - ../../orion:/app/orion
      - ../../config/vision_frame_router.yaml:/app/config/vision_frame_router.yaml:ro
      - /mnt/telemetry/vision/frames:/mnt/telemetry/vision/frames:ro
    networks:
      - app-net

networks:
  app-net:
    external: true
```

- [ ] **Step 3: README** — document role diagram, channels, policy file, smoke redis-cli commands from spec.

- [ ] **Step 4: Refresh `.env` from `.env_example`**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-frame-router/Dockerfile \
  services/orion-vision-frame-router/docker-compose.yml \
  services/orion-vision-frame-router/README.md
git commit -m "chore(vision-frame-router): add docker packaging and operator docs"
```

---

### Task 10: Bus catalog updates

**Files:**
- Modify: `orion/bus/channels.yaml`

- [ ] **Step 1: Update channel producers/consumers**

Under `orion:vision:frames` (~line 680):

```yaml
    producer_services: ["orion-vision-edge", "orion-vision-retina"]
    consumer_services:
      - "orion-vision-frame-router"
      - "orion-vision-edge"
      - "*"
```

Under `orion:exec:request:VisionHostService` (~line 424):

```yaml
    producer_services: ["orion-cortex-exec", "orion-hub", "orion-vision-host", "orion-vision-frame-router"]
```

Under `orion:vision:reply:*` (~line 432):

```yaml
    consumer_services:
      - "orion-vision-frame-router"
      - "*"
```

Verify `orion:vision:artifacts` already lists host producer (no change required unless consumer list should include frame-router for future metrics — **skip for MVP**).

- [ ] **Step 2: Commit**

```bash
git add orion/bus/channels.yaml
git commit -m "chore(bus): register orion-vision-frame-router on vision channels"
```

---

### Task 11: No-contamination guard

**Files:**
- Create: `services/orion-vision-frame-router/tests/test_no_contamination.py`

- [ ] **Step 1: Add import boundary test**

```python
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("cv2", "ultralytics", "yolo", "detector_worker")

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def test_router_does_not_import_forbidden_modules() -> None:
    hits: list[str] = []
    for path in APP_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN:
                        hits.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN:
                    hits.append(f"{path.name}: from {node.module}")
    assert hits == []
```

- [ ] **Step 2: Run full test suite**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-frame-router
PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests -q
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add services/orion-vision-frame-router/tests/test_no_contamination.py
git commit -m "test(vision-frame-router): guard against inference/capture imports"
```

---

### Task 12: Code review, PR report, push

- [ ] **Step 1: Run requesting-code-review subagent** on diff `origin/main...HEAD`. Fix every issue before proceeding.

- [ ] **Step 2: Write PR report** to `docs/superpowers/pr-reports/2026-05-23-orion-vision-frame-router-pr.md`:

```markdown
# PR: orion-vision-frame-router

## Summary
- New policy bridge: orion:vision:frames → sampled VisionTaskRequestPayload → orion:exec:request:VisionHostService
- Preserves causality via BaseEnvelope.derive_child; tracks pending via orion:vision:reply:<correlation_id>
- YAML policy with per-camera overrides; health on orion:system:health

## Test plan
- [ ] pytest services/orion-vision-frame-router/tests -q
- [ ] docker compose up orion-vision-frame-router (mesh Redis via ORION_BUS_URL); SUBSCRIBE frames/intake; PSUBSCRIBE replies
- [ ] Confirm host receives retina_fast / detect_open_vocab tasks only for sampled frames

## Risk
- Shared frame volume mount must match Retina output paths
```

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/orion-vision-frame-router-v1
gh pr create --title "feat(vision): add orion-vision-frame-router policy bridge" --body "$(cat <<'EOF'
## Summary
- Adds `orion-vision-frame-router` to bridge Retina frame pointers to Vision Host explicit tasks.
- Configurable sampling/backpressure via `config/vision_frame_router.yaml`.
- Preserves envelope causality and tracks host replies/timeouts.

## Test plan
- [x] `pytest services/orion-vision-frame-router/tests -q`
- [ ] Smoke: retina → frames → router → host intake → reply channel

EOF
)"
```

---

## Self-review (plan author checklist)

| Spec requirement | Task |
|------------------|------|
| Subscribe `orion:vision:frames` | Task 8 |
| Validate `VisionFramePointerPayload` | Task 7 |
| Configurable policy / backpressure | Tasks 2–4 |
| Publish to Host intake | Tasks 5, 7, 8 |
| `reply_to` = `orion:vision:reply:<correlation_id>` | Tasks 5, 7 |
| `derive_child` causality | Task 5 |
| Pending clear on reply/timeout | Tasks 3, 7, 8 |
| Health/metrics | Tasks 6, 8 |
| No capture/inference | Tasks 11 + non-goals |
| Bus catalog | Task 10 |
| Tests for policy/envelope/dispatcher/timeout/contamination | Tasks 1–7, 11 |
| Worktree isolation | Header + Task 0 |
| `.env` sync from example | Tasks 1, 9 |
| Code review + PR | Task 12 |

**Placeholder scan:** No TBD steps. All file paths and core code blocks are specified.

---

## Acceptance criteria mapping

| Criterion | Evidence |
|-----------|----------|
| Frame pointer → host task | `test_dispatcher.py` + smoke |
| `reply_to` channel | `test_envelopes.py` |
| Pending cleared | `test_dispatcher.py` timeout/reply |
| `dry_run` | `test_dispatcher.py` |
| Tests pass | Task 11 pytest |
| No cv2/YOLO | `test_no_contamination.py` |
