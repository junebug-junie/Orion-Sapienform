# Vision Host Pipe — Edge Decouple Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a self-contained host vision pipe (`frames → router → host → window → council`) that never gates on edge; host GroundingDINO replies feed caption triggers and person fallback authority.

**Architecture:** Remove frame-router subscription to `orion:vision:edge:activity`. New `host_trigger.py` extracts person labels from host task reply artifacts and refreshes per-stream trigger TTL (reuse `RouterState.record_activity`). Window skips `edge_detection` artifacts in evidence. Council fallback uses `host_person_hits` with `host_fallback` / `host_detect` tags. Edge unchanged on its own channels.

**Tech Stack:** Python 3.10+, Pydantic v2, Redis bus (`OrionBusAsync`), pytest via `./orion_dev/bin/python` or `./venv/bin/python` with `PYTHONPATH=.` from repo root.

**Spec:** [`docs/superpowers/specs/2026-07-03-vision-host-pipe-edge-decouple-design.md`](../specs/2026-07-03-vision-host-pipe-edge-decouple-design.md)

---

## File map

| Path | Responsibility |
|------|----------------|
| `services/orion-vision-frame-router/app/host_trigger.py` | **Create** — extract labels from host reply artifacts |
| `services/orion-vision-frame-router/app/state.py` | **Modify** — add `stream_id` to `PendingTask` |
| `services/orion-vision-frame-router/app/dispatcher.py` | **Modify** — call host trigger on reply; store stream_id on dispatch |
| `services/orion-vision-frame-router/app/main.py` | **Modify** — remove `_activity_loop` |
| `services/orion-vision-frame-router/app/activity.py` | **Delete** — edge-only handler |
| `services/orion-vision-frame-router/app/settings.py` | **Modify** — remove `CHANNEL_EDGE_ACTIVITY_IN` |
| `services/orion-vision-frame-router/app/metrics.py` | **Modify** — add `host_trigger_updates_total` |
| `services/orion-vision-frame-router/tests/test_host_trigger.py` | **Create** — unit tests for label extraction |
| `services/orion-vision-frame-router/tests/test_dispatcher.py` | **Modify** — reply → trigger integration test |
| `services/orion-vision-frame-router/tests/test_activity_envelope.py` | **Delete** |
| `services/orion-vision-frame-router/tests/test_settings.py` | **Modify** — drop edge channel assert |
| `services/orion-vision-window/app/projection.py` | **Modify** — skip `edge_detection` in `_build_evidence` |
| `services/orion-vision-window/tests/test_projection_evidence.py` | **Modify** — host-only evidence test |
| `services/orion-vision-council/app/evidence_grounding.py` | **Modify** — `host_person_hits`, fallback tag |
| `services/orion-vision-council/app/main.py` | **Modify** — fallback gate on host hits |
| `services/orion-vision-council/app/interpretation.py` | **Modify** — `host_fallback` parse mode |
| `services/orion-vision-council/tests/test_evidence_grounding.py` | **Modify** — host fallback tests |
| `services/orion-vision-council/tests/test_main_grounding_wiring.py` | **Modify** — host fallback wiring |
| `config/vision_frame_router.yaml` | **Modify** — `trigger_labels: [person]` |
| `orion/bus/channels.yaml` | **Modify** — remove router from edge activity consumers |
| `tests/test_vision_edge_activity_bus_catalog.py` | **Modify** — consumer list assertion |
| `docs/vision_services.md` | **Modify** — decouple narrative |
| `services/orion-vision-frame-router/README.md` | **Modify** |
| `services/orion-vision-council/README.md` | **Modify** |
| `services/orion-vision-edge/README.md` | **Modify** — edge-local activity note |
| `services/orion-vision-edge/.env_example` | **Modify** — optional `EDGE_PUBLISH_ARTIFACTS=false` default |

---

### Task 1: Host trigger helper (TDD)

**Files:**
- Create: `services/orion-vision-frame-router/app/host_trigger.py`
- Create: `services/orion-vision-frame-router/tests/test_host_trigger.py`

- [ ] **Step 1: Write failing tests**

Create `services/orion-vision-frame-router/tests/test_host_trigger.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import (
    VisionArtifactOutputs,
    VisionArtifactPayload,
    VisionObject,
    VisionTaskResultPayload,
)

from app.host_trigger import extract_host_trigger_labels, stream_id_from_host_result


def _result(*, stream_id: str = "cam0", labels: list[tuple[str, float]]) -> VisionTaskResultPayload:
    objects = [
        VisionObject(label=label, score=score, box_xyxy=[0, 0, 1, 1])
        for label, score in labels
    ]
    artifact = VisionArtifactPayload(
        artifact_id="a1",
        correlation_id="c1",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": stream_id, "camera_id": "rtsp://cam"},
        outputs=VisionArtifactOutputs(objects=objects),
        timing={},
        model_fingerprints={},
    )
    return VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=artifact)


def test_extract_host_trigger_labels_person_above_threshold() -> None:
    result = _result(labels=[("person", 0.9), ("door", 0.8)])
    labels = extract_host_trigger_labels(result, allowed={"person"}, score_threshold=0.25)
    assert labels == ["person"]


def test_extract_host_trigger_labels_ignores_low_score() -> None:
    result = _result(labels=[("person", 0.1)])
    labels = extract_host_trigger_labels(result, allowed={"person"}, score_threshold=0.25)
    assert labels == []


def test_extract_host_trigger_labels_empty_when_no_artifact() -> None:
    result = VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=None)
    assert extract_host_trigger_labels(result, allowed={"person"}) == []


def test_stream_id_from_host_result_prefers_artifact_inputs() -> None:
    result = _result(stream_id="cam0", labels=[("person", 0.9)])
    assert stream_id_from_host_result(result, fallback_stream_id="cam99") == "cam0"


def test_stream_id_from_host_result_uses_fallback() -> None:
    result = VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=None)
    assert stream_id_from_host_result(result, fallback_stream_id="cam99") == "cam99"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=.:services/orion-vision-frame-router ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests/test_host_trigger.py -v
```

Expected: FAIL — `ModuleNotFoundError: app.host_trigger`

- [ ] **Step 3: Implement minimal module**

Create `services/orion-vision-frame-router/app/host_trigger.py`:

```python
from __future__ import annotations

from orion.schemas.vision import VisionTaskResultPayload

HOST_TRIGGER_SCORE_THRESHOLD = 0.25


def stream_id_from_host_result(
    result: VisionTaskResultPayload,
    *,
    fallback_stream_id: str | None,
) -> str | None:
    if result.artifact and result.artifact.inputs:
        sid = result.artifact.inputs.get("stream_id")
        if sid is not None and str(sid).strip():
            return str(sid).strip()
    if fallback_stream_id and str(fallback_stream_id).strip():
        return str(fallback_stream_id).strip()
    return None


def extract_host_trigger_labels(
    result: VisionTaskResultPayload,
    *,
    allowed: set[str],
    score_threshold: float = HOST_TRIGGER_SCORE_THRESHOLD,
) -> list[str]:
    if not result.ok or result.artifact is None:
        return []
    objects = result.artifact.outputs.objects or []
    seen: set[str] = set()
    out: list[str] = []
    allowed_lower = {a.lower() for a in allowed}
    for obj in objects:
        label = str(obj.label or "").strip().lower()
        if not label or label not in allowed_lower:
            continue
        if float(obj.score) < score_threshold:
            continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=.:services/orion-vision-frame-router ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests/test_host_trigger.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-frame-router/app/host_trigger.py \
  services/orion-vision-frame-router/tests/test_host_trigger.py
git commit -m "feat(vision-frame-router): add host reply trigger label extraction"
```

---

### Task 2: PendingTask stream_id + reply wiring

**Files:**
- Modify: `services/orion-vision-frame-router/app/state.py`
- Modify: `services/orion-vision-frame-router/app/dispatcher.py`
- Modify: `services/orion-vision-frame-router/tests/test_state.py`
- Modify: `services/orion-vision-frame-router/tests/test_dispatcher.py`

- [ ] **Step 1: Write failing dispatcher integration test**

Add to `services/orion-vision-frame-router/tests/test_dispatcher.py`:

```python
from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject

# ... inside test file, add helper and test:

def _host_reply_with_person(*, corr, stream_id: str = "cam0") -> BaseEnvelope:
    artifact = VisionArtifactPayload(
        artifact_id="a1",
        correlation_id=str(corr),
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": stream_id},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="person", score=0.9, box_xyxy=[0, 0, 1, 1])]
        ),
        timing={},
        model_fingerprints={},
    )
    payload = VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=artifact)
    return BaseEnvelope(
        kind="vision.task.result",
        source=ServiceRef(name="vision-host", version="0.1.0"),
        correlation_id=corr,
        payload=payload.model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_host_reply_records_person_trigger_for_stream(tmp_path: Path) -> None:
    policy_yaml = tmp_path / "policy.yaml"
    policy_yaml.write_text(
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
  triggered:
    task_type: retina_fast
    trigger_labels: [person]
    trigger_ttl_seconds: 8
    min_seconds_between_tasks_per_camera: 0
    request:
      want_caption: true
global:
  max_inflight_total: 2
  require_image_path_exists: false
streams:
  cam0:
    enabled: true
cameras: {}
""",
        encoding="utf-8",
    )
    settings = Settings(ROUTER_POLICY_PATH=str(policy_yaml), REQUIRE_IMAGE_PATH_EXISTS=False)
    dispatcher = FrameDispatcher(
        settings=settings,
        policy=FrameDispatchPolicy.load(settings),
        state=RouterState(),
        metrics=RouterMetrics(),
        bus=FakeBus(),
    )
    corr = uuid4()
    frame = VisionFramePointerPayload(
        image_path="/tmp/f.jpg",
        camera_id="rtsp://cam",
        stream_id="cam0",
        frame_ts=time.time(),
    )
    frame_env = BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=corr,
        payload=frame.model_dump(mode="json"),
    )
    await dispatcher.handle_frame_envelope(frame_env)
    await dispatcher.handle_reply_envelope(_host_reply_with_person(corr=corr))
    active = dispatcher.state.active_labels("cam0", ["person"], ttl_s=8.0, now=time.time())
    assert active == ["person"]
    assert dispatcher.metrics.host_trigger_updates_total == 1
```

Also add `stream_id: str | None = None` to `PendingTask` in `state.py` and extend `mark_dispatched(..., stream_id: str | None = None)`.

Update `test_mark_dispatched_tracks_inflight` in `test_state.py` to pass `stream_id="cam0"` and assert `rs.pending["abc"].stream_id == "cam0"`.

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=.:services/orion-vision-frame-router ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests/test_dispatcher.py::test_host_reply_records_person_trigger_for_stream -v
```

Expected: FAIL — `host_trigger_updates_total` missing or trigger not recorded

- [ ] **Step 3: Implement state + dispatcher + metrics**

In `state.py`, add to `PendingTask`:

```python
stream_id: str | None = None
```

Extend `mark_dispatched` signature with `stream_id: str | None = None` and store on `PendingTask`.

In `metrics.py`, add to `RouterMetrics`:

```python
host_trigger_updates_total: int = 0
```

In `dispatcher.py`:

1. Import `extract_host_trigger_labels`, `stream_id_from_host_result` from `.host_trigger`
2. In `_handle_frame_envelope_inner`, pass `stream_id=frame.stream_id` to `mark_dispatched`
3. Replace `_handle_reply_envelope_inner` body after `clear_pending`:

```python
if not result.ok:
    self.metrics.host_errors_total += 1
    return

allowed = set(self.policy.default_trigger_labels())
labels = extract_host_trigger_labels(result, allowed=allowed)
stream_id = stream_id_from_host_result(result, fallback_stream_id=cleared.stream_id)
if labels and stream_id:
    self.state.record_activity(stream_id, labels, now=time.time())
    self.metrics.host_trigger_updates_total += 1
    logger.info("[ROUTER] host_trigger stream={} labels={}", stream_id, labels)
```

Add method to `FrameDispatchPolicy` in `policy.py`:

```python
def default_trigger_labels(self) -> list[str]:
    normalized = _normalize_tiers(dict(self.defaults))
    triggered = dict(normalized.get("triggered") or {})
    return list(triggered.get("trigger_labels") or [])
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=.:services/orion-vision-frame-router ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests/test_state.py \
  services/orion-vision-frame-router/tests/test_dispatcher.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-frame-router/app/state.py \
  services/orion-vision-frame-router/app/dispatcher.py \
  services/orion-vision-frame-router/app/metrics.py \
  services/orion-vision-frame-router/app/policy.py \
  services/orion-vision-frame-router/tests/test_state.py \
  services/orion-vision-frame-router/tests/test_dispatcher.py
git commit -m "feat(vision-frame-router): feed trigger TTL from host task replies"
```

---

### Task 3: Remove edge activity loop from router

**Files:**
- Modify: `services/orion-vision-frame-router/app/main.py`
- Delete: `services/orion-vision-frame-router/app/activity.py`
- Delete: `services/orion-vision-frame-router/tests/test_activity_envelope.py`
- Modify: `services/orion-vision-frame-router/app/settings.py`
- Modify: `services/orion-vision-frame-router/.env_example`
- Modify: `services/orion-vision-frame-router/tests/test_settings.py`

- [ ] **Step 1: Remove activity loop and imports from `main.py`**

Delete:
- `from .activity import handle_activity_envelope`
- `self._activity_task` field and all references
- entire `_activity_loop` method
- `_activity_task` from `stop()` task tuple

- [ ] **Step 2: Remove `CHANNEL_EDGE_ACTIVITY_IN` from settings**

In `settings.py`, delete the line:

```python
CHANNEL_EDGE_ACTIVITY_IN: str = "orion:vision:edge:activity"
```

Remove matching line from `.env_example`.

In `test_settings.py`, remove any assertion on `CHANNEL_EDGE_ACTIVITY_IN` if added later; no change needed if absent.

- [ ] **Step 3: Delete edge activity files**

```bash
rm services/orion-vision-frame-router/app/activity.py
rm services/orion-vision-frame-router/tests/test_activity_envelope.py
```

- [ ] **Step 4: Run frame-router test suite**

```bash
PYTHONPATH=.:services/orion-vision-frame-router ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -u services/orion-vision-frame-router/
git commit -m "refactor(vision-frame-router): remove edge activity subscription"
```

---

### Task 4: Router policy config — person-only triggers

**Files:**
- Modify: `config/vision_frame_router.yaml`
- Modify: `services/orion-vision-frame-router/tests/test_default_request_config.py` (if asserts motion in defaults)

- [ ] **Step 1: Update YAML**

In `config/vision_frame_router.yaml`, change:

```yaml
    trigger_labels: [person, motion]
```

to:

```yaml
    trigger_labels: [person]
```

- [ ] **Step 2: Run policy tests**

```bash
PYTHONPATH=.:services/orion-vision-frame-router ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests/test_policy_triggers.py -q
```

Expected: PASS (tests use inline YAML fixtures; repo default change is config-only)

- [ ] **Step 3: Commit**

```bash
git add config/vision_frame_router.yaml
git commit -m "chore(vision-frame-router): default trigger_labels to host person only"
```

---

### Task 5: Window — ignore edge artifacts in evidence

**Files:**
- Modify: `services/orion-vision-window/app/projection.py`
- Modify: `services/orion-vision-window/tests/test_projection_evidence.py`

- [ ] **Step 1: Write failing test**

Replace `test_evidence_counts_edge_and_host_person` with:

```python
def test_evidence_ignores_edge_detection_artifacts() -> None:
    items = [
        (_art("edge_detection", "person", 0.9), time.time()),
        (_art("retina_fast", "door", 0.8), time.time()),
        (_art("retina_fast", "person", 0.7), time.time()),
    ]
    summary = summarize_items(items)
    ev = summary["evidence"]
    assert ev["edge_person_hits"] == 0
    assert ev["host_person_hits"] == 1
    assert "person" in ev["hard_labels"]
    assert "door" in ev["hard_labels"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=.:services/orion-vision-window ./orion_dev/bin/python -m pytest \
  services/orion-vision-window/tests/test_projection_evidence.py::test_evidence_ignores_edge_detection_artifacts -v
```

Expected: FAIL — `edge_person_hits == 1`

- [ ] **Step 3: Implement skip in `_build_evidence`**

At start of loop in `_build_evidence`:

```python
for art, _ts in items:
    if art.task_type == "edge_detection":
        continue
    # ... existing body unchanged
```

- [ ] **Step 4: Run test**

```bash
PYTHONPATH=.:services/orion-vision-window ./orion_dev/bin/python -m pytest \
  services/orion-vision-window/tests/test_projection_evidence.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-window/app/projection.py \
  services/orion-vision-window/tests/test_projection_evidence.py
git commit -m "fix(vision-window): exclude edge_detection from pipeline evidence"
```

---

### Task 6: Council — host person fallback authority

**Files:**
- Modify: `services/orion-vision-council/app/evidence_grounding.py`
- Modify: `services/orion-vision-council/app/main.py`
- Modify: `services/orion-vision-council/app/interpretation.py`
- Modify: `services/orion-vision-council/tests/test_evidence_grounding.py`
- Modify: `services/orion-vision-council/tests/test_main_grounding_wiring.py`

- [ ] **Step 1: Write failing tests**

In `test_evidence_grounding.py`, rename/update:

```python
def test_person_presence_fallback_uses_host_detect_tag() -> None:
    window = _window(evidence={"hard_labels": ["person"], "host_person_hits": 2, "edge_person_hits": 0})
    fb = build_person_presence_fallback(window)
    assert fb.event_candidates[0].tags == ["host_detect"]
```

In `test_main_grounding_wiring.py`, update:

```python
def test_finalize_host_fallback_on_parse_failure() -> None:
    svc = CouncilService()
    window = _window(evidence={"hard_labels": ["person"], "host_person_hits": 2, "edge_person_hits": 0})
    interpretation, outcome = svc._finalize_interpretation(
        None,
        InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed"),
        window,
    )
    assert interpretation is not None
    assert interpretation.event_candidates[0].event_type == "person_presence"
    assert outcome.parse_mode == "host_fallback"
```

Rename `test_finalize_preserves_strict_v2_when_edge_fallback_after_grounding` to use `host_person_hits=2` and assert warning `host_fallback_after_grounding`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=.:services/orion-vision-council ./orion_dev/bin/python -m pytest \
  services/orion-vision-council/tests/test_evidence_grounding.py::test_person_presence_fallback_uses_host_detect_tag \
  services/orion-vision-council/tests/test_main_grounding_wiring.py::test_finalize_host_fallback_on_parse_failure -v
```

Expected: FAIL

- [ ] **Step 3: Implement council changes**

In `evidence_grounding.py`, add:

```python
def host_person_hits(window: VisionWindowPayload) -> int:
    ev = (window.summary or {}).get("evidence") or {}
    try:
        return int(ev.get("host_person_hits") or 0)
    except (TypeError, ValueError):
        return 0
```

Change `build_person_presence_fallback` tag to `["host_detect"]`.

In `main.py`:
- Replace `from app.evidence_grounding import edge_person_hits` with `host_person_hits`
- Replace all `edge_person_hits(window) > 0` with `host_person_hits(window) > 0`
- Replace `parse_mode="edge_fallback"` with `parse_mode="host_fallback"`
- Replace warning `"edge_fallback_after_grounding"` with `"host_fallback_after_grounding"`

In `interpretation.py`, update `ParseMode`:

```python
ParseMode = Literal[
    "strict_v2",
    "salvaged_v2",
    "legacy_events_dict",
    "legacy_events_list",
    "parse_failed",
    "host_fallback",
]
```

Remove `"edge_fallback"` from the literal (update any `_PARSE_MODE` validators in same file if present).

- [ ] **Step 4: Run council tests**

```bash
PYTHONPATH=.:services/orion-vision-council ./orion_dev/bin/python -m pytest \
  services/orion-vision-council/tests/test_evidence_grounding.py \
  services/orion-vision-council/tests/test_main_grounding_wiring.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-council/app/evidence_grounding.py \
  services/orion-vision-council/app/main.py \
  services/orion-vision-council/app/interpretation.py \
  services/orion-vision-council/tests/
git commit -m "fix(vision-council): person fallback authority from host detections"
```

---

### Task 7: Bus catalog + docs

**Files:**
- Modify: `orion/bus/channels.yaml`
- Modify: `tests/test_vision_edge_activity_bus_catalog.py`
- Modify: `docs/vision_services.md`
- Modify: `services/orion-vision-frame-router/README.md`
- Modify: `services/orion-vision-council/README.md`
- Modify: `services/orion-vision-edge/README.md`
- Modify: `services/orion-vision-window/README.md`
- Modify: `services/orion-vision-edge/.env_example`

- [ ] **Step 1: Update bus catalog**

In `orion/bus/channels.yaml`, change `orion:vision:edge:activity` entry:

```yaml
    consumer_services: []
```

(or remove `orion-vision-frame-router` from the list)

- [ ] **Step 2: Update catalog test**

In `tests/test_vision_edge_activity_bus_catalog.py`, replace:

```python
    assert "orion-vision-frame-router" in entry["consumer_services"]
```

with:

```python
    assert "orion-vision-frame-router" not in entry.get("consumer_services", [])
```

- [ ] **Step 3: Update READMEs**

Frame router README: diagram shows host reply → trigger TTL; remove edge activity subscribe section.

Council README: replace `edge_fallback` / `edge_yolo` / `edge_person_hits` references with host equivalents.

Edge README: state `orion:vision:edge:activity` is for edge-local consumers, not host pipe.

Window README: note `edge_detection` artifacts are ignored for evidence.

`docs/vision_services.md`: remove frame-router from edge activity consumer table row.

- [ ] **Step 4: Optional edge env default**

In `services/orion-vision-edge/.env_example`, set:

```
EDGE_PUBLISH_ARTIFACTS=false
```

Run sync if operator policy requires:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 5: Run catalog test**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_vision_edge_activity_bus_catalog.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/bus/channels.yaml tests/test_vision_edge_activity_bus_catalog.py \
  docs/vision_services.md services/orion-vision-frame-router/README.md \
  services/orion-vision-council/README.md services/orion-vision-edge/README.md \
  services/orion-vision-window/README.md services/orion-vision-edge/.env_example
git commit -m "docs(vision): document host-pipe decouple from edge"
```

---

### Task 8: Full verification

- [ ] **Step 1: Run combined unit suite**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests \
  services/orion-vision-window/tests/test_projection_evidence.py \
  services/orion-vision-council/tests/test_evidence_grounding.py \
  services/orion-vision-council/tests/test_main_grounding_wiring.py \
  tests/test_vision_edge_activity_bus_catalog.py -q
```

Expected: all PASS

- [ ] **Step 2: Compile touched services**

```bash
python3 -m compileall services/orion-vision-frame-router services/orion-vision-window services/orion-vision-council
```

Expected: exit 0

- [ ] **Step 3: Post-deploy runtime checks (operator)**

After restarting `orion-vision-frame-router`, `orion-vision-window`, `orion-vision-council`:

```bash
redis-cli -u "$ORION_BUS_URL" PUBSUB NUMSUB orion:vision:edge:activity
# Expected: 0 subscribers from frame-router (edge may still publish)

docker logs orion-orion-athena-vision-frame-router --since 2m 2>&1 | rg 'host_trigger|tier=triggered'
# Expected: host_trigger lines after host replies with person; triggered dispatches follow
```

- [ ] **Step 4: Final commit if env sync changed local files**

Only if `sync_local_env_from_example.py` modified gitignored `.env` — no commit needed for `.env`.

---

## Plan self-review

| Spec requirement | Task |
|------------------|------|
| Remove router edge activity subscription | Task 3 |
| Host reply → trigger TTL | Tasks 1–2 |
| Window ignores edge_detection | Task 5 |
| Council host_person_hits fallback | Task 6 |
| Edge unchanged (docs only) | Task 7 |
| trigger_labels `[person]` | Task 4 |
| Bus catalog update | Task 7 |
| Acceptance tests A–G | Tasks 1–2, 5–6, 8 |

**Placeholder scan:** No TBD steps. All code blocks are complete.

**Type consistency:** `host_person_hits`, `host_fallback`, `host_detect`, `host_trigger_updates_total` used consistently across tasks.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-03-vision-host-pipe-edge-decouple.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
