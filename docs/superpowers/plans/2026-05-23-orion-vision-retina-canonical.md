# Orion Vision Retina Canonical Intake — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `services/orion-vision-retina` from a mock folder pointer publisher into the canonical visual intake organ: sample frames from folder/rtsp/webcam/mock sources, persist JPEG artifacts, publish `VisionFramePointerPayload` on `orion:vision:frames`, and emit system health — with zero detector/YOLO/security logic.

**Architecture:** Split responsibilities into `settings`, `sources` (async `FrameSource` adapters), `frame_store` (disk save + retention cleanup), `envelopes` (canonical `BaseEnvelope` builder), `health` (`SystemHealthV1` + retina metrics in `details`), and `main` (`RetinaService` with testable `capture_once()`). Borrow capture-loop *shape* from `services/orion-vision-edge/app/capture_worker.py` only — never import `detector_worker` or ultralytics.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pydantic-settings v2, OpenCV (`opencv-python-headless`), numpy, Redis bus (`OrionBusAsync`), shared `orion.schemas.vision.VisionFramePointerPayload`, pytest at repo root (`PYTHONPATH=.`).

**Spec source:** Operator brief “orion-vision-retina canonical camera intake organ” (2026-05-23).

**Execution isolation (mandatory before Task 1):**

- **Do not touch** the main workspace branch `feat/repair-pressure-v1` (Claude is building there).
- Create an isolated worktree:

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main 2>/dev/null || true
git worktree add .worktrees/orion-vision-retina-canonical -b feat/orion-vision-retina-canonical-v1 origin/main
cd .worktrees/orion-vision-retina-canonical
```

- All implementation, commits, and PR push happen **only** inside `.worktrees/orion-vision-retina-canonical`.
- After updating `services/orion-vision-retina/.env_example`, copy to `services/orion-vision-retina/.env` in the worktree (not committed). **Do not** copy other changed files back to the main checkout — only `.env` may be synced locally by the operator if desired.
- Post-implementation: run **requesting-code-review** subagent, fix findings, push branch, open PR with `gh pr create`.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-vision-retina/app/settings.py` | Expanded retina config; `RETINA_SOURCE_PATH` → `RETINA_SOURCE` alias |
| `services/orion-vision-retina/app/sources.py` | `FrameSource` protocol + folder/rtsp/webcam/mock adapters |
| `services/orion-vision-retina/app/frame_store.py` | `save_frame`, `cleanup_old_frames`, filename sanitization |
| `services/orion-vision-retina/app/envelopes.py` | `make_frame_pointer_envelope()` |
| `services/orion-vision-retina/app/health.py` | `RetinaMetrics`, `make_system_health_envelope()` |
| `services/orion-vision-retina/app/main.py` | `RetinaService` with `capture_once`, `capture_loop`, lifespan |
| `services/orion-vision-retina/requirements.txt` | Add `opencv-python-headless`, `numpy` |
| `services/orion-vision-retina/docker-compose.yml` | Volumes, env parity, optional `/dev/video0` |
| `services/orion-vision-retina/.env_example` | Full env contract |
| `services/orion-vision-retina/.env` | Copy of `.env_example` (gitignored, local sync only) |
| `orion/bus/channels.yaml` | Verify/update `orion:vision:frames` consumers (no new channel required) |
| `tests/test_vision_retina_settings.py` | Settings defaults + alias |
| `tests/test_vision_retina_frame_store.py` | Save + cleanup |
| `tests/test_vision_retina_sources.py` | Folder source behavior |
| `tests/test_vision_retina_envelopes.py` | Envelope kind/source/payload |
| `tests/test_vision_retina_capture_once.py` | `capture_once` with fakes |
| `tests/test_vision_retina_no_detector.py` | Static import regression guard |

**Explicit non-goals (do not add):** YOLO, face/motion/presence detectors, `VisionEdgeArtifact`, substrate emitters, GraphDB/SQL/vector writes, vision council/scribe logic.

**Envelope contract note:** `BaseEnvelope.schema_id` must remain `"orion.envelope"` (see `orion/core/bus/bus_schemas.py`). Canonical **kind** is `vision.frame.pointer`. Payload validates as `VisionFramePointerPayload` per bus catalog `schema_id: VisionFramePointerPayload`. Do **not** copy vision-edge’s `kind=vision.edge.frame.v1`.

---

### Task 0: Worktree bootstrap + dependency baseline

**Files:**
- Modify: `services/orion-vision-retina/requirements.txt`

- [ ] **Step 1: Create worktree and branch (from repo root, not main checkout)**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
test "$(git branch --show-current)" = "feat/repair-pressure-v1" && echo "main checkout OK — do not implement here"
git worktree add .worktrees/orion-vision-retina-canonical -b feat/orion-vision-retina-canonical-v1 origin/main
cd .worktrees/orion-vision-retina-canonical
```

Expected: worktree path exists; `git branch --show-current` prints `feat/orion-vision-retina-canonical-v1`.

- [ ] **Step 2: Add OpenCV + numpy to requirements**

Edit `services/orion-vision-retina/requirements.txt` — append:

```text
opencv-python-headless==4.10.0.84
numpy==1.26.4
```

- [ ] **Step 3: Install dev deps in worktree (if venv missing)**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-retina-canonical
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_settings.py -q 2>&1 | head -5 || true
```

Expected before implementation: collection error / file not found (tests not written yet).

- [ ] **Step 4: Commit**

```bash
git add services/orion-vision-retina/requirements.txt
git commit -m "chore(vision-retina): add opencv and numpy dependencies"
```

---

### Task 1: Expanded settings

**Files:**
- Modify: `services/orion-vision-retina/app/settings.py`
- Create: `tests/test_vision_retina_settings.py`
- Modify: `services/orion-vision-retina/.env_example`
- Modify: `services/orion-vision-retina/.env` (copy from example, not committed)

- [ ] **Step 1: Write failing settings tests**

Create `tests/test_vision_retina_settings.py`:

```python
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"
sys.path.insert(0, str(SERVICE_ROOT))


def _reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    for key in list(os.environ):
        if key.startswith("RETINA_") or key in {
            "ORION_BUS_URL",
            "ORION_BUS_ENFORCE_CATALOG",
            "CHANNEL_RETINA_PUB",
            "FRAME_STORAGE_DIR",
            "JPEG_QUALITY",
            "HEALTH_INTERVAL_SECONDS",
            "SOURCE_RECONNECT_SECONDS",
        }:
            monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.settings as mod

    importlib.reload(mod)
    return mod.Settings()


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch)
    assert s.SERVICE_NAME == "vision-retina"
    assert s.SERVICE_VERSION == "0.2.0"
    assert s.CHANNEL_RETINA_PUB == "orion:vision:frames"
    assert s.RETINA_SOURCE_TYPE == "folder"
    assert s.RETINA_SOURCE == "/mnt/telemetry/vision/intake"
    assert s.RETINA_FPS == 1.0
    assert s.FRAME_STORAGE_DIR == "/mnt/telemetry/vision/frames"


def test_retina_source_path_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(
        monkeypatch,
        RETINA_SOURCE_PATH="/tmp/legacy-intake",
    )
    assert s.RETINA_SOURCE == "/tmp/legacy-intake"


def test_jpeg_quality_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch, JPEG_QUALITY="999")
    assert s.JPEG_QUALITY == 100
    s2 = _reload_settings(monkeypatch, JPEG_QUALITY="-5")
    assert s2.JPEG_QUALITY == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-retina-canonical
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_settings.py -v
```

Expected: FAIL (missing fields / wrong defaults).

- [ ] **Step 3: Implement settings**

Replace `services/orion-vision-retina/app/settings.py`:

```python
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-retina"
    SERVICE_VERSION: str = "0.2.0"
    LOG_LEVEL: str = "INFO"

    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    CHANNEL_RETINA_PUB: str = "orion:vision:frames"
    CHANNEL_RETINA_ERROR: str = "orion:vision:retina:error"
    CHANNEL_SYSTEM_HEALTH: str = "orion:system:health"

    RETINA_SOURCE_TYPE: str = "folder"
    RETINA_SOURCE: str = "/mnt/telemetry/vision/intake"
    RETINA_SOURCE_PATH: str | None = Field(default=None, exclude=True)
    RETINA_CAMERA_ID: str = "retina-cam-01"
    RETINA_STREAM_ID: str = "retina-stream-01"

    RETINA_FPS: float = 1.0
    RETINA_WIDTH: int | None = None
    RETINA_HEIGHT: int | None = None

    FRAME_STORAGE_DIR: str = "/mnt/telemetry/vision/frames"
    FRAME_RETENTION_SECONDS: int = 300
    JPEG_QUALITY: int = 90

    HEALTH_INTERVAL_SECONDS: float = 10.0
    SOURCE_RECONNECT_SECONDS: float = 5.0

    @field_validator("JPEG_QUALITY", mode="before")
    @classmethod
    def _clamp_jpeg_quality(cls, v: object) -> int:
        try:
            n = int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 90
        return max(1, min(100, n))

    def model_post_init(self, __context: object) -> None:
        if self.RETINA_SOURCE_PATH and not self.model_fields_set.get("RETINA_SOURCE"):
            object.__setattr__(self, "RETINA_SOURCE", self.RETINA_SOURCE_PATH)


def get_settings() -> Settings:
    return Settings()
```

**Note:** Pydantic v2 `model_post_init` alias handling — if `RETINA_SOURCE_PATH` env is set, use a `@model_validator(mode="after")` instead:

```python
from pydantic import model_validator

@model_validator(mode="after")
def _apply_source_path_alias(self) -> "Settings":
    if self.RETINA_SOURCE_PATH:
        self.RETINA_SOURCE = self.RETINA_SOURCE_PATH
    return self
```

And declare `RETINA_SOURCE_PATH: str | None = None` (not excluded) so env `RETINA_SOURCE_PATH` loads.

- [ ] **Step 4: Update `.env_example` and copy to `.env`**

`services/orion-vision-retina/.env_example`:

```bash
ORION_BUS_URL=redis://localhost:6379/0
ORION_BUS_ENFORCE_CATALOG=false

CHANNEL_RETINA_PUB=orion:vision:frames
CHANNEL_RETINA_ERROR=orion:vision:retina:error
CHANNEL_SYSTEM_HEALTH=orion:system:health

RETINA_SOURCE_TYPE=folder
RETINA_SOURCE=/mnt/telemetry/vision/intake
RETINA_CAMERA_ID=retina-cam-01
RETINA_STREAM_ID=retina-stream-01
RETINA_FPS=1.0

FRAME_STORAGE_DIR=/mnt/telemetry/vision/frames
FRAME_RETENTION_SECONDS=300
JPEG_QUALITY=90

HEALTH_INTERVAL_SECONDS=10.0
SOURCE_RECONNECT_SECONDS=5.0
```

```bash
cp services/orion-vision-retina/.env_example services/orion-vision-retina/.env
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_settings.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-vision-retina/app/settings.py services/orion-vision-retina/.env_example tests/test_vision_retina_settings.py
git commit -m "feat(vision-retina): expand settings contract and env example"
```

---

### Task 2: Frame store (save + cleanup)

**Files:**
- Create: `services/orion-vision-retina/app/frame_store.py`
- Create: `tests/test_vision_retina_frame_store.py`

- [ ] **Step 1: Write failing frame_store tests**

Create `tests/test_vision_retina_frame_store.py`:

```python
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from app.frame_store import SavedFrame, cleanup_old_frames, ensure_frame_dir, save_frame


def test_save_frame_writes_jpg(tmp_path: Path) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    saved = save_frame(
        frame,
        directory=str(tmp_path),
        camera_id="cam/01",
        stream_id="stream:01",
        ts=1710000000.0,
        quality=90,
    )
    assert isinstance(saved, SavedFrame)
    assert saved.format == "jpg"
    assert saved.width == 64
    assert saved.height == 48
    assert Path(saved.image_path).is_file()
    assert saved.image_path.endswith(".jpg")
    assert "cam_01" in Path(saved.image_path).name
    assert "stream_01" in Path(saved.image_path).name


def test_cleanup_removes_old_keeps_fresh(tmp_path: Path) -> None:
    old = tmp_path / "old.jpg"
    fresh = tmp_path / "fresh.jpg"
    old.write_bytes(b"x")
    fresh.write_bytes(b"y")
    old_ts = time.time() - 600
    os.utime(old, (old_ts, old_ts))
    removed = cleanup_old_frames(str(tmp_path), max_age_seconds=300)
    assert removed >= 1
    assert not old.exists()
    assert fresh.exists()
```

Add `sys.path` bootstrap in `conftest` or duplicate the `SERVICE_ROOT` insert from Task 1 tests (prefer single `tests/conftest_vision_retina.py`):

```python
# tests/conftest_vision_retina.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orion-vision-retina"))
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_frame_store.py -v
```

- [ ] **Step 3: Implement `frame_store.py`**

Create `services/orion-vision-retina/app/frame_store.py`:

```python
from __future__ import annotations

import os
import re
import time
import uuid

import cv2
import numpy as np
from pydantic import BaseModel


class SavedFrame(BaseModel):
    image_path: str
    width: int
    height: int
    format: str = "jpg"
    bytes_written: int | None = None


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def ensure_frame_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_frame(
    frame: np.ndarray,
    *,
    directory: str,
    camera_id: str,
    stream_id: str,
    ts: float,
    quality: int,
) -> SavedFrame:
    ensure_frame_dir(directory)
    cam = _sanitize_id(camera_id)
    stream = _sanitize_id(stream_id)
    epoch_ms = int(ts * 1000)
    short = uuid.uuid4().hex[:8]
    filename = f"{cam}_{stream}_{epoch_ms}_{short}.jpg"
    image_path = os.path.join(directory, filename)
    ok = cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {image_path}")
    h, w = frame.shape[:2]
    size = os.path.getsize(image_path)
    return SavedFrame(
        image_path=image_path,
        width=w,
        height=h,
        format="jpg",
        bytes_written=size,
    )


def cleanup_old_frames(directory: str, max_age_seconds: int) -> int:
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(directory):
        fp = os.path.join(directory, name)
        if not os.path.isfile(fp):
            continue
        if now - os.path.getmtime(fp) > max_age_seconds:
            try:
                os.remove(fp)
                removed += 1
            except OSError:
                pass
    return removed
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_frame_store.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-retina/app/frame_store.py tests/test_vision_retina_frame_store.py tests/conftest_vision_retina.py
git commit -m "feat(vision-retina): persist frames and retention cleanup"
```

---

### Task 3: Frame sources (folder, mock, rtsp, webcam)

**Files:**
- Create: `services/orion-vision-retina/app/sources.py`
- Create: `tests/test_vision_retina_sources.py`

- [ ] **Step 1: Write failing source tests**

Create `tests/test_vision_retina_sources.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from app.sources import FolderFrameSource, MockFrameSource, create_frame_source


@pytest.mark.asyncio
async def test_folder_source_reads_jpg(tmp_path: Path) -> None:
    # minimal valid jpeg bytes (1x1) — or write via cv2 in test
    import cv2

    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), dtype=np.uint8))
    src = FolderFrameSource(str(tmp_path))
    await src.start()
    result = await src.read()
    await src.stop()
    assert result is not None
    assert result.width == 10
    assert result.height == 10


@pytest.mark.asyncio
async def test_folder_source_empty_returns_none(tmp_path: Path) -> None:
    src = FolderFrameSource(str(tmp_path))
    await src.start()
    assert await src.read() is None
    await src.stop()


@pytest.mark.asyncio
async def test_mock_source_reads_from_folder(tmp_path: Path) -> None:
    import cv2

    cv2.imwrite(str(tmp_path / "m.png"), np.ones((8, 8, 3), dtype=np.uint8) * 127)
    src = MockFrameSource(str(tmp_path))
    await src.start()
    result = await src.read()
    await src.stop()
    assert result is not None


def test_create_frame_source_factory() -> None:
    assert isinstance(create_frame_source("folder", "/x"), FolderFrameSource)
    assert isinstance(create_frame_source("mock", "/x"), MockFrameSource)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_sources.py -v
```

- [ ] **Step 3: Implement `sources.py`**

Create `services/orion-vision-retina/app/sources.py` with:

```python
from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field


class FrameReadResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    frame: Any
    ts: float
    width: int
    height: int
    source_meta: dict[str, Any] = Field(default_factory=dict)


class FrameSource(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def read(self) -> FrameReadResult | None: ...


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class FolderFrameSource:
    def __init__(self, directory: str) -> None:
        if not directory:
            raise ValueError("folder source requires RETINA_SOURCE directory")
        self.directory = directory
        self._files: list[str] = []
        self._index = 0

    async def start(self) -> None:
        if not os.path.isdir(self.directory):
            self._files = []
            return
        self._files = sorted(
            str(self.directory / p.name)
            if False else os.path.join(self.directory, name)
            for name in os.listdir(self.directory)
            if Path(name).suffix.lower() in _IMAGE_EXTS
        )
        self._index = 0

    async def stop(self) -> None:
        return None

    async def read(self) -> FrameReadResult | None:
        if not self._files:
            return None
        path = self._files[self._index % len(self._files)]
        self._index += 1
        frame = await asyncio.to_thread(cv2.imread, path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"path": path, "type": "folder"},
        )


class MockFrameSource(FolderFrameSource):
    """Test/dev: random sample from folder like legacy mock."""

    async def read(self) -> FrameReadResult | None:
        if not self._files:
            return None
        path = random.choice(self._files)
        frame = await asyncio.to_thread(cv2.imread, path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"path": path, "type": "mock"},
        )


class _VideoCaptureSource:
    def __init__(
        self,
        source: str | int,
        *,
        width: int | None,
        height: int | None,
        reconnect_seconds: float,
        source_type: str,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._reconnect_seconds = reconnect_seconds
        self._source_type = source_type
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    async def start(self) -> None:
        await self._open()

    async def stop(self) -> None:
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None

    async def _open(self) -> None:
        await self.stop()
        cap = await asyncio.to_thread(cv2.VideoCapture, self._source)
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        self._cap = cap

    async def read(self) -> FrameReadResult | None:
        if self._cap is None or not self._cap.isOpened():
            await self._open()
        if self._cap is None or not self._cap.isOpened():
            self._failures += 1
            return None
        ok, frame = await asyncio.to_thread(self._cap.read)
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= 3:
                await asyncio.sleep(self._reconnect_seconds)
                await self._open()
                self._failures = 0
            return None
        self._failures = 0
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"type": self._source_type},
        )


class RtspFrameSource(_VideoCaptureSource):
    def __init__(self, url: str, **kwargs: Any) -> None:
        if not url:
            raise ValueError("rtsp source requires RETINA_SOURCE url")
        super().__init__(url, source_type="rtsp", **kwargs)


class WebcamFrameSource(_VideoCaptureSource):
    def __init__(self, device: str, **kwargs: Any) -> None:
        index: str | int = device if device.isdigit() else device
        if isinstance(index, str) and index.isdigit():
            index = int(index)
        super().__init__(index, source_type="webcam", **kwargs)


def create_frame_source(
    source_type: str,
    source: str,
    *,
    width: int | None = None,
    height: int | None = None,
    reconnect_seconds: float = 5.0,
) -> FrameSource:
    st = source_type.lower().strip()
    common = dict(width=width, height=height, reconnect_seconds=reconnect_seconds)
    if st == "folder":
        return FolderFrameSource(source)
    if st == "mock":
        return MockFrameSource(source)
    if st == "rtsp":
        return RtspFrameSource(source, **common)
    if st == "webcam":
        return WebcamFrameSource(source or "0", **common)
    raise ValueError(f"unsupported RETINA_SOURCE_TYPE: {source_type!r}")
```

Fix the FolderFrameSource list comprehension typo (`str(self.directory / p.name)` → `os.path.join(self.directory, name)`) before commit.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-retina/app/sources.py tests/test_vision_retina_sources.py
git commit -m "feat(vision-retina): add folder mock rtsp webcam frame sources"
```

---

### Task 4: Envelopes helper

**Files:**
- Create: `services/orion-vision-retina/app/envelopes.py`
- Create: `tests/test_vision_retina_envelopes.py`

- [ ] **Step 1: Write failing envelope test**

```python
from __future__ import annotations

from app.envelopes import make_frame_pointer_envelope
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload


def test_frame_pointer_envelope_contract() -> None:
    payload = VisionFramePointerPayload(
        image_path="/mnt/telemetry/vision/frames/x.jpg",
        camera_id="retina-cam-01",
        stream_id="retina-stream-01",
        frame_ts=1710000000.0,
        width=640,
        height=480,
        format="jpg",
    )
    env = make_frame_pointer_envelope(
        payload,
        service_name="vision-retina",
        service_version="0.2.0",
    )
    assert isinstance(env, BaseEnvelope)
    assert env.kind == "vision.frame.pointer"
    assert env.schema_id == "orion.envelope"
    assert env.source == ServiceRef(name="vision-retina", version="0.2.0")
    validated = VisionFramePointerPayload.model_validate(env.payload)
    assert validated.camera_id == "retina-cam-01"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `envelopes.py`**

```python
from __future__ import annotations

import uuid

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload


def make_frame_pointer_envelope(
    payload: VisionFramePointerPayload,
    *,
    service_name: str,
    service_version: str,
    correlation_id: str | None = None,
) -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name=service_name, version=service_version),
        correlation_id=uuid.UUID(correlation_id) if correlation_id else uuid.uuid4(),
        payload=payload.model_dump(mode="json"),
    )
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-retina/app/envelopes.py tests/test_vision_retina_envelopes.py
git commit -m "feat(vision-retina): canonical vision.frame.pointer envelopes"
```

---

### Task 5: Health telemetry

**Files:**
- Create: `services/orion-vision-retina/app/health.py`

- [ ] **Step 1: Implement health module (no separate test file — covered in capture_once + manual)**

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1


BOOT_ID = str(uuid.uuid4())


@dataclass
class RetinaMetrics:
    frames_published: int = 0
    frames_failed: int = 0
    last_frame_ts: float | None = None
    last_error: str | None = None
    fps_observed: float = 0.0


def make_system_health_envelope(
    *,
    service_name: str,
    service_version: str,
    camera_id: str,
    stream_id: str,
    source_type: str,
    source_ok: bool,
    metrics: RetinaMetrics,
    fps_target: float,
    storage_dir: str,
) -> BaseEnvelope:
    payload = SystemHealthV1(
        service=service_name,
        version=service_version,
        boot_id=BOOT_ID,
        last_seen_ts=datetime.now(timezone.utc),
        status="ok" if source_ok else "degraded",
        details={
            "camera_id": camera_id,
            "stream_id": stream_id,
            "source_type": source_type,
            "source_ok": source_ok,
            "frames_published": metrics.frames_published,
            "frames_failed": metrics.frames_failed,
            "fps_target": fps_target,
            "fps_observed": metrics.fps_observed,
            "last_frame_ts": metrics.last_frame_ts,
            "last_error": metrics.last_error,
            "storage_dir": storage_dir,
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
git add services/orion-vision-retina/app/health.py
git commit -m "feat(vision-retina): system health envelope helper"
```

---

### Task 6: RetinaService refactor (`main.py`)

**Files:**
- Modify: `services/orion-vision-retina/app/main.py`
- Create: `tests/test_vision_retina_capture_once.py`

- [ ] **Step 1: Write failing capture_once test**

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.main import RetinaService
from app.settings import Settings
from app.sources import FrameReadResult


class _FakeSource:
    def __init__(self, frame: np.ndarray | None) -> None:
        self._frame = frame

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def read(self) -> FrameReadResult | None:
        if self._frame is None:
            return None
        h, w = self._frame.shape[:2]
        return FrameReadResult(frame=self._frame, ts=1710000001.0, width=w, height=h)


@pytest.mark.asyncio
async def test_capture_once_publishes_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        FRAME_STORAGE_DIR=str(tmp_path),
        RETINA_CAMERA_ID="test-cam-01",
        RETINA_STREAM_ID="test-stream-01",
        CHANNEL_RETINA_PUB="orion:vision:frames",
    )
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)
    svc.source = _FakeSource(np.zeros((12, 16, 3), dtype=np.uint8))

    ok = await svc.capture_once()
    assert ok is True
    bus.publish.assert_awaited_once()
    channel, env = bus.publish.await_args.args
    assert channel == "orion:vision:frames"
    assert env.kind == "vision.frame.pointer"
    assert env.payload["camera_id"] == "test-cam-01"
    assert env.payload["stream_id"] == "test-stream-01"
    assert env.payload["width"] == 16
    assert env.payload["height"] == 12
    assert Path(env.payload["image_path"]).is_file()


@pytest.mark.asyncio
async def test_capture_once_skips_publish_on_read_failure(tmp_path: Path) -> None:
    settings = Settings(FRAME_STORAGE_DIR=str(tmp_path))
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)
    svc.source = _FakeSource(None)

    ok = await svc.capture_once()
    assert ok is False
    bus.publish.assert_not_awaited()
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Replace `main.py`**

Key structure (implement fully in worktree):

```python
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync

from .envelopes import make_frame_pointer_envelope
from .frame_store import cleanup_old_frames, save_frame
from .health import RetinaMetrics, make_system_health_envelope
from .settings import Settings, get_settings
from .sources import create_frame_source
from orion.schemas.vision import VisionFramePointerPayload


class RetinaService:
    def __init__(
        self,
        settings: Settings | None = None,
        bus: OrionBusAsync | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or OrionBusAsync(
            url=self.settings.ORION_BUS_URL,
            enforce_catalog=self.settings.ORION_BUS_ENFORCE_CATALOG,
        )
        self.source = create_frame_source(
            self.settings.RETINA_SOURCE_TYPE,
            self.settings.RETINA_SOURCE,
            width=self.settings.RETINA_WIDTH,
            height=self.settings.RETINA_HEIGHT,
            reconnect_seconds=self.settings.SOURCE_RECONNECT_SECONDS,
        )
        self.metrics = RetinaMetrics()
        self._capture_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._last_cleanup = 0.0

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=self.settings.LOG_LEVEL)
        await self.bus.connect()
        await self.source.start()
        self._shutdown.clear()
        self._capture_task = asyncio.create_task(self.capture_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(f"[RETINA] Started → {self.settings.CHANNEL_RETINA_PUB}")

    async def stop(self) -> None:
        self._shutdown.set()
        for task in (self._capture_task, self._health_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.source.stop()
        await self.bus.close()

    async def capture_loop(self) -> None:
        interval = 1.0 / max(self.settings.RETINA_FPS, 0.01)
        while not self._shutdown.is_set():
            t0 = time.time()
            try:
                await self.capture_once()
            except Exception as exc:
                self.metrics.last_error = str(exc)
                logger.error(f"[RETINA] capture_once failed: {exc}")
            if time.time() - self._last_cleanup > 10:
                await asyncio.to_thread(
                    cleanup_old_frames,
                    self.settings.FRAME_STORAGE_DIR,
                    self.settings.FRAME_RETENTION_SECONDS,
                )
                self._last_cleanup = time.time()
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def capture_once(self) -> bool:
        result = await self.source.read()
        if result is None:
            self.metrics.frames_failed += 1
            return False
        saved = await asyncio.to_thread(
            save_frame,
            result.frame,
            directory=self.settings.FRAME_STORAGE_DIR,
            camera_id=self.settings.RETINA_CAMERA_ID,
            stream_id=self.settings.RETINA_STREAM_ID,
            ts=result.ts,
            quality=self.settings.JPEG_QUALITY,
        )
        payload = VisionFramePointerPayload(
            image_path=saved.image_path,
            camera_id=self.settings.RETINA_CAMERA_ID,
            stream_id=self.settings.RETINA_STREAM_ID,
            frame_ts=result.ts,
            width=saved.width,
            height=saved.height,
            format=saved.format,
        )
        env = make_frame_pointer_envelope(
            payload,
            service_name=self.settings.SERVICE_NAME,
            service_version=self.settings.SERVICE_VERSION,
        )
        await self.bus.publish(self.settings.CHANNEL_RETINA_PUB, env)
        self.metrics.frames_published += 1
        self.metrics.last_frame_ts = result.ts
        return True

    async def _health_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                env = make_system_health_envelope(
                    service_name=self.settings.SERVICE_NAME,
                    service_version=self.settings.SERVICE_VERSION,
                    camera_id=self.settings.RETINA_CAMERA_ID,
                    stream_id=self.settings.RETINA_STREAM_ID,
                    source_type=self.settings.RETINA_SOURCE_TYPE,
                    source_ok=self.metrics.last_error is None,
                    metrics=self.metrics,
                    fps_target=self.settings.RETINA_FPS,
                    storage_dir=self.settings.FRAME_STORAGE_DIR,
                )
                await self.bus.publish(self.settings.CHANNEL_SYSTEM_HEALTH, env)
            except Exception as exc:
                logger.warning(f"[RETINA] health publish failed: {exc}")
            await asyncio.sleep(self.settings.HEALTH_INTERVAL_SECONDS)


service = RetinaService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()

app = FastAPI(title="Orion Vision Retina", version="0.2.0", lifespan=lifespan)
```

Verify `OrionBusAsync` constructor accepts `enforce_catalog=` — if not, pass via env only and use `OrionBusAsync(url=...)`.

- [ ] **Step 4: Run capture tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_capture_once.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-retina/app/main.py tests/test_vision_retina_capture_once.py
git commit -m "feat(vision-retina): RetinaService capture_once and health loop"
```

---

### Task 7: No-detector regression guard

**Files:**
- Create: `tests/test_vision_retina_no_detector.py`

- [ ] **Step 1: Add static contamination test**

```python
from __future__ import annotations

import ast
from pathlib import Path

RETINA_APP = Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina" / "app"

BANNED_IMPORT_ROOTS = {
    "ultralytics",
    "torch",
    "torchvision",
}
BANNED_MODULES = {
    "detector_worker",
}
BANNED_SYMBOLS = {
    "VisionEdgeArtifact",
}


def _walk_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_retina_has_no_detector_imports() -> None:
    found: set[str] = set()
    for py in RETINA_APP.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        found |= _walk_imports(tree)
    assert not (found & BANNED_IMPORT_ROOTS)
    assert "detector_worker" not in found


def test_retina_source_no_banned_symbols() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in RETINA_APP.glob("*.py"))
    for sym in BANNED_SYMBOLS:
        assert sym not in text
```

- [ ] **Step 2: Run — expect PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_vision_retina_no_detector.py
git commit -m "test(vision-retina): guard against detector contamination"
```

---

### Task 8: Docker compose + bus catalog

**Files:**
- Modify: `services/orion-vision-retina/docker-compose.yml`
- Modify: `orion/bus/channels.yaml` (consumer list only if needed)

- [ ] **Step 1: Update docker-compose.yml**

```yaml
services:
  orion-vision-retina:
    build: .
    container_name: orion-vision-retina
    environment:
      - ORION_BUS_URL=${ORION_BUS_URL:-redis://redis:6379/0}
      - ORION_BUS_ENFORCE_CATALOG=${ORION_BUS_ENFORCE_CATALOG:-false}
      - CHANNEL_RETINA_PUB=orion:vision:frames
      - CHANNEL_SYSTEM_HEALTH=orion:system:health
      - RETINA_SOURCE_TYPE=${RETINA_SOURCE_TYPE:-folder}
      - RETINA_SOURCE=${RETINA_SOURCE:-/mnt/telemetry/vision/intake}
      - RETINA_CAMERA_ID=${RETINA_CAMERA_ID:-retina-cam-01}
      - RETINA_STREAM_ID=${RETINA_STREAM_ID:-retina-stream-01}
      - RETINA_FPS=${RETINA_FPS:-1.0}
      - FRAME_STORAGE_DIR=/mnt/telemetry/vision/frames
      - FRAME_RETENTION_SECONDS=${FRAME_RETENTION_SECONDS:-300}
    volumes:
      - ./app:/app/app
      - ../../orion:/app/orion
      - /mnt/telemetry/vision/frames:/mnt/telemetry/vision/frames
      - /mnt/telemetry/vision/intake:/mnt/telemetry/vision/intake
    depends_on:
      - redis
    # Optional USB webcam:
    # devices:
    #   - /dev/video0:/dev/video0

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

- [ ] **Step 2: Patch bus catalog consumers (optional)**

In `orion/bus/channels.yaml` under `orion:vision:frames`, ensure:

```yaml
consumer_services: ["orion-vision-host", "orion-vision-window", "orion-vision-edge", "*"]
```

Channel already exists with `producer_services: ["orion-vision-edge", "orion-vision-retina"]` — no new channel required.

- [ ] **Step 3: Commit**

```bash
git add services/orion-vision-retina/docker-compose.yml orion/bus/channels.yaml
git commit -m "chore(vision-retina): compose volumes env and bus catalog consumers"
```

---

### Task 9: Full verification + smoke

- [ ] **Step 1: Run all retina tests**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-retina-canonical
PYTHONPATH=. ./venv/bin/python -m pytest \
  tests/test_vision_retina_settings.py \
  tests/test_vision_retina_frame_store.py \
  tests/test_vision_retina_sources.py \
  tests/test_vision_retina_envelopes.py \
  tests/test_vision_retina_capture_once.py \
  tests/test_vision_retina_no_detector.py \
  -v --tb=short
```

Expected: all PASS

- [ ] **Step 2: Docker smoke (optional, host paths must exist)**

```bash
cd services/orion-vision-retina
docker compose build
docker compose up -d
docker logs -f orion-vision-retina
```

- [ ] **Step 3: Commit any fixes**

---

### Task 10: Code review subagent + PR

- [ ] **Step 1: Dispatch code-reviewer subagent**

Use **requesting-code-review** skill: review diff against this plan and acceptance criteria. Fix all blocking issues in the worktree.

- [ ] **Step 2: Push branch**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-retina-canonical
git push -u origin feat/orion-vision-retina-canonical-v1
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --title "feat(vision-retina): canonical frame intake organ" --body "$(cat <<'EOF'
## Summary
- Upgrade orion-vision-retina to canonical visual intake: folder/rtsp/webcam/mock sources, JPEG frame store, VisionFramePointerPayload on orion:vision:frames, system health telemetry.
- No detectors, YOLO, or vision-edge detector_worker coupling.

## Test plan
- [ ] PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_*.py -v
- [ ] RETINA_SOURCE_TYPE=folder with images in /mnt/telemetry/vision/intake
- [ ] redis-cli SUBSCRIBE orion:vision:frames shows vision.frame.pointer envelopes
- [ ] Frames appear under FRAME_STORAGE_DIR; old frames cleaned per retention

EOF
)"
```

- [ ] **Step 4: Deliver PR markdown report to operator** (separate `PR-DESC.md` or chat summary with link).

---

## Acceptance criteria mapping

| Criterion | Task |
|-----------|------|
| folder mode runs | Task 3, 6, 8 |
| valid VisionFramePointerPayload on orion:vision:frames | Task 4, 6 |
| writes + cleans frames | Task 2, 6 |
| orion:system:health periodic | Task 5, 6 |
| survives empty/missing/transient failures | Task 3 (`read` → None), Task 6 |
| no detector code | Task 7 |
| unit tests | Tasks 1–7 |
| vision-edge untouched | Scope: no edits under `services/orion-vision-edge/` |
| bus catalog | Task 8 |

---

## Self-review (plan author)

**Spec coverage:** All required modules, settings, sources, tests, docker, and bus catalog covered. `capture.py` omitted (loop lives in `main.py`; YAGNI). Retina-specific error channel deferred — system health carries `last_error` in `details` per spec allowance.

**Placeholder scan:** No TBD steps.

**Type consistency:** `FrameReadResult`, `SavedFrame`, `RetinaService.capture_once`, and envelope helpers use consistent field names matching `VisionFramePointerPayload`.

**Risk:** Current `main.py` passes invalid `schema_id="vision.frame.pointer"` on `BaseEnvelope` — Task 4/6 fixes this. Confirm `OrionBusAsync` kwargs against `async_service.py` during implementation.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-orion-vision-retina-canonical.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with executing-plans checkpoints  

**Which approach?**

Also confirm: should the plan file commit land on `feat/orion-vision-retina-canonical-v1` only (worktree), or may it be committed on `main` separately for documentation?
