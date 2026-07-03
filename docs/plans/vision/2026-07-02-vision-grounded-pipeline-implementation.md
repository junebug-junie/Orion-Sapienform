# Vision Grounded Pipeline Implementation Plan

> **Superseded (2026-07-03):** Host pipe decoupled from edge activity. See `docs/superpowers/plans/2026-07-03-vision-host-pipe-edge-decouple.md` for the current architecture.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire edge YOLO/motion triggers into the vision mesh, gate host VLM caption on those triggers, sanitize captions, and ground Council events on structured evidence so Eye-Ball-1 walk-bys produce `person_presence` without YouTube hallucinations.

**Architecture:** Edge publishes slim artifacts + compact activity signals on a new bus channel. Frame router maintains per-`stream_id` trigger TTL and dispatches baseline (detect-only) vs triggered (caption+embed) `retina_fast` tasks. Window aggregates evidence tiers from edge/host artifacts. Council runs `enforce_evidence_grounding()` after LLM parse and falls back to deterministic `person_presence` when parse fails but edge person evidence exists.

**Tech stack:** Python 3, Pydantic v2, Redis pub/sub (`OrionBusAsync`), pytest via `./venv/bin/python` or `./orion_dev/bin/python` with `PYTHONPATH=.` from repo root.

**Design spec:** [`docs/plans/vision/2026-07-02-vision-grounded-pipeline-design.md`](../../plans/vision/2026-07-02-vision-grounded-pipeline-design.md)

**Worktree:** Implement on a dedicated branch/worktree (e.g. `feat/vision-grounded-pipeline`).

---

## File map

| Path | Role |
|------|------|
| `orion/schemas/vision.py` | Add `VisionEdgeActivityPayload` |
| `orion/schemas/registry.py` | Register schema + kind `vision.edge.activity.v1` |
| `orion/bus/channels.yaml` | Register `orion:vision:edge:activity` |
| `services/orion-vision-edge/app/activity.py` | Rate-limited activity payload builder + publish helper |
| `services/orion-vision-edge/app/detector_worker.py` | Call activity publish + slim artifacts |
| `services/orion-vision-edge/app/settings.py` | `CHANNEL_VISION_EDGE_ACTIVITY`, rate limit env |
| `services/orion-vision-edge/tests/test_activity.py` | Activity rate limit + label extraction tests |
| `services/orion-vision-frame-router/app/state.py` | Per-stream trigger TTL state |
| `services/orion-vision-frame-router/app/policy.py` | `streams` lookup, baseline/triggered tiers |
| `services/orion-vision-frame-router/app/activity.py` | Decode + record activity envelopes |
| `services/orion-vision-frame-router/app/main.py` | Activity consumer loop |
| `services/orion-vision-frame-router/app/settings.py` | `CHANNEL_EDGE_ACTIVITY_IN` |
| `services/orion-vision-frame-router/tests/test_policy_triggers.py` | Stream fallback + trigger gating tests |
| `services/orion-vision-window/app/projection.py` | `evidence` block in summary |
| `services/orion-vision-window/tests/test_projection_evidence.py` | Evidence tier tests |
| `services/orion-vision-host/app/caption_sanitize.py` | Prompt + reject rules |
| `services/orion-vision-host/app/runner.py` | Use sanitizer in `_run_caption_frame` |
| `services/orion-vision-host/tests/test_caption_sanitize.py` | Sanitizer unit tests |
| `services/orion-vision-council/app/evidence_grounding.py` | `enforce_evidence_grounding`, fallback builder |
| `services/orion-vision-council/app/interpretation.py` | Prompt rules + call grounding |
| `services/orion-vision-council/app/main.py` | Parse-failure fallback wiring |
| `services/orion-vision-council/tests/test_evidence_grounding.py` | Grounding + YouTube rejection tests |
| `config/vision_frame_router.yaml` | Baseline/triggered + `streams.cam0` |
| `docs/vision_services.md` | Document new channel + tiers |

---

### Task 1: Shared schema and bus channel

**Files:**
- Modify: `orion/schemas/vision.py`
- Modify: `orion/schemas/registry.py`
- Modify: `orion/bus/channels.yaml`
- Test: `tests/test_vision_edge_activity_schema.py` (repo root tests — matches other vision schema tests)

- [ ] **Step 1: Write the failing test**

Create `tests/test_vision_edge_activity_schema.py`:

```python
from orion.schemas.registry import SCHEMA_REGISTRY
from orion.schemas.vision import VisionEdgeActivityPayload


def test_vision_edge_activity_payload_roundtrip() -> None:
    payload = VisionEdgeActivityPayload(
        stream_id="cam0",
        camera_id="rtsp://192.168.1.21/Preview_01_sub",
        labels=["person", "motion"],
        max_score=0.82,
        frame_ts=1783025641.853,
        image_path="/mnt/telemetry/vision/frames/frame_x.jpg",
        artifact_id="art-1",
    )
    data = payload.model_dump(mode="json")
    restored = VisionEdgeActivityPayload.model_validate(data)
    assert restored.labels == ["person", "motion"]
    assert restored.stream_id == "cam0"


def test_registry_has_edge_activity_kind() -> None:
    assert "VisionEdgeActivityPayload" in SCHEMA_REGISTRY
    assert SCHEMA_REGISTRY["VisionEdgeActivityPayload"].kind == "vision.edge.activity.v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_edge_activity_schema.py -q --tb=short`  
Expected: **FAIL** — `ImportError: cannot import name 'VisionEdgeActivityPayload'`

- [ ] **Step 3: Implement schema, registry, channel**

In `orion/schemas/vision.py` after `VisionFramePointerPayload`:

```python
class VisionEdgeActivityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    camera_id: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    max_score: float = 1.0
    frame_ts: Optional[float] = None
    image_path: Optional[str] = None
    artifact_id: Optional[str] = None
```

In `orion/schemas/registry.py`, import `VisionEdgeActivityPayload` and register:

```python
"VisionEdgeActivityPayload": SchemaRegistration(
    model=VisionEdgeActivityPayload,
    kind="vision.edge.activity.v1",
),
```

In `orion/bus/channels.yaml` after `orion:vision:edge:error`:

```yaml
  - name: "orion:vision:edge:activity"
    kind: "event"
    schema_id: "VisionEdgeActivityPayload"
    message_kind: "vision.edge.activity.v1"
    producer_services: ["orion-vision-edge"]
    consumer_services: ["orion-vision-frame-router"]
    stability: "experimental"
    since: "2026-07-02"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_edge_activity_schema.py -q --tb=short`  
Expected: **PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/vision.py orion/schemas/registry.py orion/bus/channels.yaml tests/test_vision_edge_activity_schema.py
git commit -m "feat(vision): add VisionEdgeActivityPayload schema and bus channel"
```

---

### Task 2: Edge activity publisher with rate limiting

**Files:**
- Create: `services/orion-vision-edge/app/activity.py`
- Modify: `services/orion-vision-edge/app/settings.py`
- Modify: `services/orion-vision-edge/.env_example`
- Test: `services/orion-vision-edge/tests/test_activity.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-vision-edge/tests/test_activity.py`:

```python
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.activity import ActivityRateLimiter, labels_from_detections


def test_labels_from_detections_person_and_motion() -> None:
    detections = [
        {"kind": "yolo", "label": "person", "score": 0.9},
        {"kind": "motion", "label": "motion", "score": 1.0},
    ]
    assert labels_from_detections(detections) == ["person", "motion"]


def test_rate_limiter_blocks_duplicate_within_one_second() -> None:
    limiter = ActivityRateLimiter(min_interval_s=1.0)
    assert limiter.allow("cam0", "person", now=100.0) is True
    assert limiter.allow("cam0", "person", now=100.5) is False
    assert limiter.allow("cam0", "person", now=101.1) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=services/orion-vision-edge:. ./venv/bin/python -m pytest services/orion-vision-edge/tests/test_activity.py -q --tb=short`  
Expected: **FAIL** — `ModuleNotFoundError: app.activity`

- [ ] **Step 3: Implement activity module + settings**

Create `services/orion-vision-edge/app/activity.py`:

```python
from __future__ import annotations

import time
import uuid
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionEdgeActivityPayload

VISION_EDGE_ACTIVITY_KIND = "vision.edge.activity.v1"
TRIGGER_LABELS = frozenset({"person", "motion"})


def labels_from_detections(detections: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for det in detections:
        kind = str(det.get("kind") or "")
        label = str(det.get("label") or kind or "").strip().lower()
        if not label:
            continue
        if kind == "motion" or label == "motion":
            label = "motion"
        if label not in TRIGGER_LABELS:
            continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


class ActivityRateLimiter:
    def __init__(self, min_interval_s: float = 1.0) -> None:
        self.min_interval_s = min_interval_s
        self._last: dict[tuple[str, str], float] = {}

    def allow(self, stream_id: str, label: str, *, now: float) -> bool:
        key = (stream_id, label)
        prev = self._last.get(key)
        if prev is not None and (now - prev) < self.min_interval_s:
            return False
        self._last[key] = now
        return True


def build_activity_payload(
    *,
    stream_id: str,
    camera_id: str | None,
    labels: list[str],
    max_score: float,
    frame_ts: float | None,
    image_path: str | None,
    artifact_id: str | None,
) -> VisionEdgeActivityPayload:
    return VisionEdgeActivityPayload(
        stream_id=stream_id,
        camera_id=camera_id,
        labels=labels,
        max_score=max_score,
        frame_ts=frame_ts,
        image_path=image_path,
        artifact_id=artifact_id,
    )


async def publish_activity_if_allowed(
    bus,
    settings,
    *,
    stream_id: str,
    camera_id: str | None,
    labels: list[str],
    max_score: float,
    frame_ts: float | None,
    image_path: str | None,
    artifact_id: str | None,
    limiter: ActivityRateLimiter,
    parent_env: BaseEnvelope,
) -> None:
    if not labels or not bus.enabled:
        return
    now = time.time()
    allowed = [lb for lb in labels if limiter.allow(stream_id, lb, now=now)]
    if not allowed:
        return
    payload = build_activity_payload(
        stream_id=stream_id,
        camera_id=camera_id,
        labels=allowed,
        max_score=max_score,
        frame_ts=frame_ts,
        image_path=image_path,
        artifact_id=artifact_id or str(uuid.uuid4()),
    )
    out = parent_env.derive_child(
        kind=VISION_EDGE_ACTIVITY_KIND,
        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
        payload=payload.model_dump(mode="json"),
    )
    await bus.publish(settings.CHANNEL_VISION_EDGE_ACTIVITY, out)
```

Add to `services/orion-vision-edge/app/settings.py`:

```python
CHANNEL_VISION_EDGE_ACTIVITY: str = Field(
    "orion:vision:edge:activity", alias="CHANNEL_VISION_EDGE_ACTIVITY"
)
EDGE_ACTIVITY_MIN_INTERVAL_S: float = Field(1.0, alias="EDGE_ACTIVITY_MIN_INTERVAL_S")
```

Update `services/orion-vision-edge/.env_example`:

```bash
CHANNEL_VISION_EDGE_ACTIVITY=orion:vision:edge:activity
EDGE_ACTIVITY_MIN_INTERVAL_S=1.0
EDGE_PUBLISH_ARTIFACTS=true
```

Run: `python scripts/sync_local_env_from_example.py` from repo root.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=services/orion-vision-edge:. ./venv/bin/python -m pytest services/orion-vision-edge/tests/test_activity.py -q --tb=short`  
Expected: **PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-edge/app/activity.py services/orion-vision-edge/app/settings.py services/orion-vision-edge/.env_example services/orion-vision-edge/tests/test_activity.py
git commit -m "feat(vision-edge): add rate-limited edge activity publisher"
```

---

### Task 3: Wire edge detector to publish activity + slim artifacts

**Files:**
- Modify: `services/orion-vision-edge/app/detector_worker.py`

- [ ] **Step 1: Write failing integration-style test**

Add to `services/orion-vision-edge/tests/test_activity.py`:

```python
from app.activity import labels_from_detections


def test_empty_detections_yields_no_labels() -> None:
    assert labels_from_detections([]) == []
```

- [ ] **Step 2: Wire detector_worker**

In `services/orion-vision-edge/app/detector_worker.py`:

1. Module-level: `_activity_limiter = ActivityRateLimiter(...)` initialized lazily or at loop start from settings.
2. After `_run_detectors` builds `vision_objects` and `det_dicts`, compute:
   - `labels = labels_from_detections(det_dicts)`
   - `max_score = max((float(d.get("score", 0)) for d in det_dicts), default=0.0)`
3. Call `await publish_activity_if_allowed(...)` with `stream_id=pointer.stream_id or settings.STREAM_ID`, `camera_id=pointer.camera_id`, `image_path=pointer.image_path`, `frame_ts=pointer.frame_ts`.
4. When publishing artifact (`EDGE_PUBLISH_ARTIFACTS`), set `task_type="edge_detection"`, include `inputs={"stream_id": ..., "camera_id": ..., "image_path": ...}`, **no caption**.

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=services/orion-vision-edge:. ./venv/bin/python -m pytest services/orion-vision-edge/tests/test_activity.py -q --tb=short`  
Expected: **PASS**

- [ ] **Step 4: Compile check**

Run: `python3 -m compileall services/orion-vision-edge/app`  
Expected: exit **0**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-edge/app/detector_worker.py services/orion-vision-edge/tests/test_activity.py
git commit -m "feat(vision-edge): publish activity signals and slim detection artifacts"
```

---

### Task 4: Frame router trigger state + stream policy lookup

**Files:**
- Modify: `services/orion-vision-frame-router/app/state.py`
- Create: `services/orion-vision-frame-router/app/activity.py`
- Modify: `services/orion-vision-frame-router/app/policy.py`
- Modify: `services/orion-vision-frame-router/app/settings.py`
- Modify: `services/orion-vision-frame-router/.env_example`
- Test: `services/orion-vision-frame-router/tests/test_policy_triggers.py`

- [ ] **Step 1: Write failing tests**

Create `services/orion-vision-frame-router/tests/test_policy_triggers.py`:

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

from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState


@pytest.fixture
def trigger_policy_path(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  baseline:
    task_type: retina_fast
    every_n_frames: 1
    min_seconds_between_tasks_per_camera: 0
    request:
      want_caption: false
      want_embeddings: false
  triggered:
    task_type: retina_fast
    trigger_labels: [person, motion]
    trigger_ttl_seconds: 8
    min_seconds_between_tasks_per_camera: 0
    request:
      want_caption: true
      want_embeddings: true
global:
  max_inflight_total: 4
  require_image_path_exists: false
streams:
  cam0:
    enabled: true
cameras: {}
""",
        encoding="utf-8",
    )
    return p


def _frame_env(camera_id: str = "rtsp://cam", stream_id: str = "cam0") -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path="/tmp/f.jpg",
        camera_id=camera_id,
        stream_id=stream_id,
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def test_resolve_policy_by_stream_id_fallback(trigger_policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(trigger_policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    merged, name = policy.resolve_stream_policy("rtsp://unknown", "cam0")
    assert name == "cam0"
    assert merged["baseline"]["request"]["want_caption"] is False


def test_triggered_dispatch_when_person_active(trigger_policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(trigger_policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    state.record_activity("cam0", ["person"], now=100.0)
    decision = policy.decide(_frame_env(), state, now=100.5, image_path_exists=True)
    assert decision.should_dispatch is True
    assert decision.dispatch_tier == "triggered"
    assert decision.request_overrides.get("want_caption") is True


def test_baseline_dispatch_without_trigger(trigger_policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(trigger_policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env(), state, now=100.0, image_path_exists=True)
    assert decision.should_dispatch is True
    assert decision.dispatch_tier == "baseline"
    assert decision.request_overrides.get("want_caption") is False
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `PYTHONPATH=services/orion-vision-frame-router:. ./venv/bin/python -m pytest services/orion-vision-frame-router/tests/test_policy_triggers.py -q --tb=short`  
Expected: **FAIL** — missing `resolve_stream_policy`, `record_activity`, `dispatch_tier`

- [ ] **Step 3: Implement state + policy extensions**

`state.py` — add:

```python
class StreamTriggerState(BaseModel):
    label_ts: dict[str, float] = Field(default_factory=dict)

class RouterState:
    def __init__(self) -> None:
        ...
        self.stream_triggers: dict[str, StreamTriggerState] = {}

    def record_activity(self, stream_id: str, labels: list[str], *, now: float) -> None:
        st = self.stream_triggers.setdefault(stream_id, StreamTriggerState())
        for label in labels:
            st.label_ts[label.lower()] = now

    def active_labels(self, stream_id: str, trigger_labels: list[str], ttl_s: float, *, now: float) -> list[str]:
        st = self.stream_triggers.get(stream_id)
        if not st:
            return []
        out = []
        for label in trigger_labels:
            ts = st.label_ts.get(label.lower())
            if ts is not None and (now - ts) <= ttl_s:
                out.append(label.lower())
        return out
```

`policy.py` — extend `FrameDispatchDecision`:

```python
dispatch_tier: str = "baseline"  # baseline | triggered
```

Add methods:

- `resolve_stream_policy(camera_id, stream_id) -> tuple[dict, str]` — order: `cameras[camera_id]` → `streams[stream_id]` → merge into `defaults`; if legacy flat keys (`every_n_frames` at top level), wrap as `baseline` for backward compat.
- In `decide()`: read `triggered.trigger_labels` + TTL from merged policy; if `state.active_labels(stream_id, ...)` non-empty → use triggered tier config (shorter rate limit, caption on); else baseline tier.

Add `services/orion-vision-frame-router/app/activity.py`:

```python
from orion.schemas.vision import VisionEdgeActivityPayload

def handle_activity_envelope(env, state: RouterState, *, now: float) -> None:
    payload = VisionEdgeActivityPayload.model_validate(env.payload)
    if payload.stream_id:
        state.record_activity(payload.stream_id, payload.labels, now=now)
```

Settings + `.env_example`:

```python
CHANNEL_EDGE_ACTIVITY_IN: str = "orion:vision:edge:activity"
```

- [ ] **Step 4: Run new + existing policy tests**

Run: `PYTHONPATH=services/orion-vision-frame-router:. ./venv/bin/python -m pytest services/orion-vision-frame-router/tests/test_policy_triggers.py services/orion-vision-frame-router/tests/test_policy.py -q --tb=short`  
Expected: **PASS** (update legacy `test_policy.py` fixtures if flat defaults need baseline wrapper — keep backward compat)

- [ ] **Step 5: Wire activity consumer in main.py**

Add `_activity_loop()` subscribing to `CHANNEL_EDGE_ACTIVITY_IN`, decode, call `handle_activity_envelope`.

- [ ] **Step 6: Commit**

```bash
git add services/orion-vision-frame-router/app/state.py services/orion-vision-frame-router/app/policy.py services/orion-vision-frame-router/app/activity.py services/orion-vision-frame-router/app/main.py services/orion-vision-frame-router/app/settings.py services/orion-vision-frame-router/.env_example services/orion-vision-frame-router/tests/test_policy_triggers.py
git commit -m "feat(vision-frame-router): trigger-gated dispatch with stream policy lookup"
```

---

### Task 5: Window evidence summary

**Files:**
- Modify: `services/orion-vision-window/app/projection.py`
- Create: `services/orion-vision-window/tests/test_projection_evidence.py`

- [ ] **Step 1: Write failing test**

Create `services/orion-vision-window/tests/test_projection_evidence.py`:

```python
import time

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionCaption, VisionObject

from app.projection import summarize_items


def _art(task_type: str, label: str, score: float, caption: str | None = None) -> VisionArtifactPayload:
    cap = VisionCaption(text=caption, confidence=0.5) if caption else None
    return VisionArtifactPayload(
        artifact_id=f"{task_type}-{label}",
        correlation_id="c1",
        task_type=task_type,
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label=label, score=score, box_xyxy=[0, 0, 1, 1])],
            caption=cap,
        ),
        timing={},
        model_fingerprints={},
    )


def test_evidence_counts_edge_and_host_person() -> None:
    items = [
        (_art("edge_detection", "person", 0.9), time.time()),
        (_art("retina_fast", "door", 0.8), time.time()),
        (_art("retina_fast", "person", 0.7), time.time()),
        (_art("retina_fast", "screen", 0.6, caption="describe this image. youtube"), time.time()),
    ]
    summary = summarize_items(items)
    ev = summary["evidence"]
    assert ev["edge_person_hits"] == 1
    assert ev["host_person_hits"] == 1
    assert "person" in ev["hard_labels"]
    assert "door" in ev["hard_labels"]
    assert "youtube" in ev["soft_labels"]
    assert ev["caption_count"] == 1
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `PYTHONPATH=services/orion-vision-window:. ./venv/bin/python -m pytest services/orion-vision-window/tests/test_projection_evidence.py -q --tb=short`  
Expected: **FAIL** — `KeyError: 'evidence'`

- [ ] **Step 3: Implement evidence block in summarize_items**

Add helpers in `projection.py`:

```python
HARD_SCORE_THRESHOLD = 0.25
CAPTION_STOPLIST = frozenset({"youtube", "google", "video", "watching", "describe", "image"})

def _caption_soft_tokens(text: str) -> list[str]:
    tokens = [t.strip(".,!?").lower() for t in text.split() if t.strip()]
    return [t for t in tokens if t in CAPTION_STOPLIST]


def _build_evidence(items: list[tuple[VisionArtifactPayload, float]]) -> dict:
    hard_counts: dict[str, int] = {}
    soft_labels: list[str] = []
    edge_person_hits = 0
    host_person_hits = 0
    caption_count = 0
    for art, _ts in items:
        is_edge = art.task_type == "edge_detection"
        if art.outputs.caption and art.outputs.caption.text:
            caption_count += 1
            soft_labels.extend(_caption_soft_tokens(art.outputs.caption.text))
        for obj in art.outputs.objects or []:
            if obj.score < HARD_SCORE_THRESHOLD:
                continue
            label = obj.label.lower()
            hard_counts[label] = hard_counts.get(label, 0) + 1
            if label == "person":
                if is_edge:
                    edge_person_hits += 1
                else:
                    host_person_hits += 1
    hard_labels = sorted(hard_counts.keys())
    return {
        "hard_labels": hard_labels,
        "soft_labels": sorted(set(soft_labels)),
        "edge_person_hits": edge_person_hits,
        "host_person_hits": host_person_hits,
        "caption_count": caption_count,
    }
```

Merge into `summarize_items` return dict as `"evidence": _build_evidence(items)`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=services/orion-vision-window:. ./venv/bin/python -m pytest services/orion-vision-window/tests/test_projection_evidence.py services/orion-vision-window/tests/test_projection.py -q --tb=short`  
Expected: **PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-window/app/projection.py services/orion-vision-window/tests/test_projection_evidence.py
git commit -m "feat(vision-window): add evidence tiers to window summaries"
```

---

### Task 6: Host caption sanitizer + prompt + env defaults

**Files:**
- Create: `services/orion-vision-host/app/caption_sanitize.py`
- Modify: `services/orion-vision-host/app/runner.py`
- Modify: `services/orion-vision-host/.env_example`
- Create: `services/orion-vision-host/tests/test_caption_sanitize.py`

- [ ] **Step 1: Write failing tests**

Create `services/orion-vision-host/tests/test_caption_sanitize.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.caption_sanitize import CAPTION_PROMPT, sanitize_caption


def test_sanitize_rejects_youtube_slop() -> None:
    text, ok, reason = sanitize_caption("describe this image. youtube")
    assert ok is False
    assert text is None
    assert reason == "stoplist_ratio"


def test_sanitize_accepts_plain_scene() -> None:
    text, ok, reason = sanitize_caption("A desk with two monitors and an open door.")
    assert ok is True
    assert "monitors" in text
    assert reason is None


def test_caption_prompt_is_factual() -> None:
    assert "directly visible" in CAPTION_PROMPT.lower()
```

- [ ] **Step 2: Run — expect FAIL**

Run: `PYTHONPATH=services/orion-vision-host:. ./venv/bin/python -m pytest services/orion-vision-host/tests/test_caption_sanitize.py -q --tb=short`  
Expected: **FAIL**

- [ ] **Step 3: Implement caption_sanitize.py**

```python
from __future__ import annotations

import re

CAPTION_PROMPT = (
    "List visible objects and people. "
    "State only what is directly visible. "
    "No guesses about activity."
)
_PROMPT_ECHO = re.compile(r"describe this image", re.I)
_STOPLIST = frozenset({"youtube", "google", "video", "watching", "webcam", "com"})


def sanitize_caption(raw: str) -> tuple[str | None, bool, str | None]:
    text = (raw or "").strip()
    if _PROMPT_ECHO.search(text):
        return None, False, "prompt_echo"
    if len(text) < 12:
        return None, False, "too_short"
    tokens = [t.strip(".,!?").lower() for t in text.split() if t.strip()]
    if not tokens:
        return None, False, "empty"
    stop_hits = sum(1 for t in tokens if t in _STOPLIST)
    if stop_hits / len(tokens) > 0.4:
        return None, False, "stoplist_ratio"
    return text, True, None
```

In `runner._run_caption_frame`:
- Use `CAPTION_PROMPT` instead of `"Describe this image."`
- After decode: `caption_text, ok, reason = sanitize_caption(cleaned)`
- If not ok: set caption text empty/null, append warning `"caption_rejected:{reason}"`

Update `services/orion-vision-host/.env_example`:

```bash
VISION_VLM_MODEL_ID=Salesforce/blip2-opt-2.7b
VISION_VLM_TEMPERATURE=0.2
```

Run: `python scripts/sync_local_env_from_example.py`

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=services/orion-vision-host:. ./venv/bin/python -m pytest services/orion-vision-host/tests/test_caption_sanitize.py -q --tb=short`  
Expected: **PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-host/app/caption_sanitize.py services/orion-vision-host/app/runner.py services/orion-vision-host/.env_example services/orion-vision-host/tests/test_caption_sanitize.py
git commit -m "feat(vision-host): factual caption prompt and garbage sanitizer"
```

---

### Task 7: Council evidence grounding + parse-failure fallback

**Files:**
- Create: `services/orion-vision-council/app/evidence_grounding.py`
- Modify: `services/orion-vision-council/app/interpretation.py`
- Modify: `services/orion-vision-council/app/main.py`
- Create: `services/orion-vision-council/tests/test_evidence_grounding.py`

- [ ] **Step 1: Write failing tests**

Create `services/orion-vision-council/tests/test_evidence_grounding.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionSceneInterpretationV1, VisionWindowPayload

from app.evidence_grounding import (
    ACTIVITY_PATTERN,
    build_person_presence_fallback,
    enforce_evidence_grounding,
)


def _window(**summary) -> VisionWindowPayload:
    base = {
        "object_counts": {},
        "top_labels": [],
        "captions": summary.get("captions", []),
        "item_count": 1,
        "detection_count": 0,
        "evidence": summary.get("evidence", {}),
    }
    return VisionWindowPayload(
        window_id="w1",
        start_ts=1.0,
        end_ts=2.0,
        summary=base,
        artifact_ids=["art-edge-1"],
    )


def test_enforce_drops_youtube_activity_without_hard_person() -> None:
    window = _window(
        captions=["describe this image. youtube"],
        evidence={"hard_labels": ["door", "screen"], "edge_person_hits": 0, "soft_labels": ["youtube"]},
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Someone watching YouTube",
        event_candidates=[
            {
                "event_type": "human_activity",
                "narrative": "A person is watching a YouTube video on a screen.",
                "entities": ["person"],
                "tags": [],
                "confidence": 0.9,
                "salience": 0.8,
                "evidence_refs": ["art-edge-1"],
            }
        ],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert grounded.event_candidates == []
    assert notes


def test_person_presence_fallback_on_edge_hits() -> None:
    window = _window(evidence={"hard_labels": ["person"], "edge_person_hits": 2, "host_person_hits": 0})
    fb = build_person_presence_fallback(window)
    assert fb.event_candidates[0].event_type == "person_presence"
    assert "youtube" not in fb.event_candidates[0].narrative.lower()


def test_activity_pattern_matches_watching() -> None:
    assert ACTIVITY_PATTERN.search("watching a video")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `PYTHONPATH=services/orion-vision-council:. ./venv/bin/python -m pytest services/orion-vision-council/tests/test_evidence_grounding.py -q --tb=short`  
Expected: **FAIL**

- [ ] **Step 3: Implement evidence_grounding.py**

```python
from __future__ import annotations

import re
import uuid

from orion.schemas.vision import (
    VisionEventCandidateV1,
    VisionSceneInterpretationV1,
    VisionWindowPayload,
)

ACTIVITY_PATTERN = re.compile(
    r"\b(watching|reading|using|talking|listening|playing|browsing)\b", re.I
)
PERSON_PATTERN = re.compile(r"\bperson|someone|human\b", re.I)


def _hard_labels(window: VisionWindowPayload) -> set[str]:
    ev = (window.summary or {}).get("evidence") or {}
    return {str(x).lower() for x in (ev.get("hard_labels") or [])}


def _edge_person_hits(window: VisionWindowPayload) -> int:
    ev = (window.summary or {}).get("evidence") or {}
    try:
        return int(ev.get("edge_person_hits") or 0)
    except (TypeError, ValueError):
        return 0


def enforce_evidence_grounding(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1, list[str]]:
    hard = _hard_labels(window)
    notes: list[str] = []
    kept: list[VisionEventCandidateV1] = []
    for cand in interpretation.event_candidates:
        narrative = cand.narrative or ""
        mentions_person = bool(PERSON_PATTERN.search(narrative))
        mentions_activity = bool(ACTIVITY_PATTERN.search(narrative))
        if mentions_person and "person" not in hard:
            notes.append(f"dropped:{cand.event_type}:person_not_in_hard_labels")
            continue
        if mentions_activity and "person" not in hard:
            notes.append(f"dropped:{cand.event_type}:activity_without_person")
            continue
        updated = cand
        if mentions_activity and "person" in hard and cand.confidence > 0.4:
            if not (window.summary or {}).get("captions"):
                updated = cand.model_copy(update={"confidence": 0.4, "tags": [*cand.tags, "caption_inferred"]})
                notes.append(f"capped_confidence:{cand.event_type}")
        kept.append(updated)
    return interpretation.model_copy(update={"event_candidates": kept}), notes


def build_person_presence_fallback(window: VisionWindowPayload) -> VisionSceneInterpretationV1:
    refs = list(window.artifact_ids or [])
    return VisionSceneInterpretationV1(
        window_id=window.window_id,
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        scene_summary="A person was detected on camera.",
        event_candidates=[
            VisionEventCandidateV1(
                event_type="person_presence",
                narrative="A person was detected on camera.",
                entities=["person"],
                tags=["edge_yolo"],
                confidence=0.85,
                salience=0.7,
                evidence_refs=refs,
            )
        ],
        evidence_refs=refs,
    )
```

- [ ] **Step 4: Update interpretation prompt**

In `build_interpretation_prompt`, add to context summary the `evidence` block passthrough and append rules:

```python
"- Treat summary.evidence.hard_labels as factual detection evidence.\n"
"- Treat summary.captions as soft hints only; never sole basis for activity claims.\n"
"- Activity verbs require person in hard_labels.\n"
```

Add function `apply_evidence_pipeline(interpretation, window)` in `interpretation.py` calling `enforce_evidence_grounding`.

- [ ] **Step 5: Wire main.py**

In `_process_window` after `_generate_interpretation`:

```python
if interpretation is not None:
    interpretation, grounding_notes = enforce_evidence_grounding(interpretation, payload)
    for note in grounding_notes:
        logger.info(f"[COUNCIL] grounding {note}")
elif _edge_person_hits(payload) > 0:
    interpretation = build_person_presence_fallback(payload)
    parse_outcome = InterpretationParseOutcome(interpretation=interpretation, parse_mode="edge_fallback")
```

- [ ] **Step 6: Run council tests**

Run: `PYTHONPATH=services/orion-vision-council:. ./venv/bin/python -m pytest services/orion-vision-council/tests/test_evidence_grounding.py services/orion-vision-council/tests/test_interpretation_v2.py -q --tb=short`  
Expected: **PASS**

- [ ] **Step 7: Commit**

```bash
git add services/orion-vision-council/app/evidence_grounding.py services/orion-vision-council/app/interpretation.py services/orion-vision-council/app/main.py services/orion-vision-council/tests/test_evidence_grounding.py
git commit -m "feat(vision-council): enforce evidence grounding and person_presence fallback"
```

---

### Task 8: Config, docs, env sync, rollout

**Files:**
- Modify: `config/vision_frame_router.yaml`
- Modify: `docs/vision_services.md`
- Modify: `services/orion-vision-frame-router/.env_example` (if not done in Task 4)

- [ ] **Step 1: Update vision_frame_router.yaml**

Replace flat defaults with baseline/triggered structure from design §6.1. Add:

```yaml
streams:
  cam0:
    enabled: true
```

Keep existing `cameras:` entries unchanged.

- [ ] **Step 2: Update docs/vision_services.md**

Add rows:

| Service | Channel | Kind |
| orion-vision-edge | `orion:vision:edge:activity` | `vision.edge.activity.v1` |
| orion-vision-frame-router | subscribes to activity + frames | — |

Document baseline vs triggered dispatch.

- [ ] **Step 3: Sync env files**

Run: `python scripts/sync_local_env_from_example.py`  
Restart checklist (operator): `orion-athena-vision-edge`, `orion-orion-athena-vision-frame-router`, `orion-athena-vision-host`, `orion-athena-vision-window`, `orion-athena-vision-council`

- [ ] **Step 4: Run targeted test sweep**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_edge_activity_schema.py -q --tb=short
PYTHONPATH=services/orion-vision-edge:. ./venv/bin/python -m pytest services/orion-vision-edge/tests -q --tb=short
PYTHONPATH=services/orion-vision-frame-router:. ./venv/bin/python -m pytest services/orion-vision-frame-router/tests -q --tb=short
PYTHONPATH=services/orion-vision-window:. ./venv/bin/python -m pytest services/orion-vision-window/tests -q --tb=short
PYTHONPATH=services/orion-vision-host:. ./venv/bin/python -m pytest services/orion-vision-host/tests -q --tb=short
PYTHONPATH=services/orion-vision-council:. ./venv/bin/python -m pytest services/orion-vision-council/tests -q --tb=short
```

Expected: all **PASS**

- [ ] **Step 5: Live acceptance script (operator)**

Walk past Eye-Ball-1, then:

```bash
docker logs orion-athena-vision-edge --since 2m 2>&1 | rg 'Found.*person|activity'
docker logs orion-orion-athena-vision-frame-router --since 2m 2>&1 | rg 'triggered|want_caption'
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c \
  "SELECT created_at, event_type, narrative FROM vision_events ORDER BY created_at DESC LIMIT 5;"
```

Pass: `person_presence` or grounded narrative; no YouTube activity unless hard person + corroborating caption.

- [ ] **Step 6: Commit**

```bash
git add config/vision_frame_router.yaml docs/vision_services.md
git commit -m "docs(vision): frame router trigger config and edge activity channel"
```

---

## Spec coverage self-review

| Spec section | Task |
|--------------|------|
| §5 Item 1 edge detectors + artifacts | Task 2–3 |
| §5.2 edge activity channel | Task 1–3 |
| §6 trigger-gated host | Task 4 |
| §7 caption quality | Task 6 |
| §8 evidence + council grounding | Task 5, 7 |
| §4 identifier conventions (stream policy, keep RTSP camera_id) | Task 4 |
| §11 tests | All tasks |
| §12 rollout | Task 8 |

No TBD placeholders. Type names consistent: `VisionEdgeActivityPayload`, `dispatch_tier`, `edge_person_hits`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-vision-grounded-pipeline-implementation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** — run tasks in this session via executing-plans with batch checkpoints

Which approach?
