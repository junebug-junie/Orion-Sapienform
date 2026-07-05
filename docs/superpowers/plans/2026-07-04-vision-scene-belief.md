# Vision Scene Belief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate council metacog flicker on stable office scenes by habituating `hard_labels` in `orion-vision-window` and gating council transitions on `believed_hard_labels` only.

**Architecture:** `SceneBeliefTracker` maintains a per-stream observation ring (last N flushes) and vote-based enter/exit for labels. Window emits both observed and believed tiers in `summary.evidence`. Council `snapshot_from_window()` prefers `believed_hard_labels`; refresh TTL defaults to disabled (contract A: silence on stable scenes).

**Tech Stack:** Python 3.10+, Pydantic v2 settings, FastAPI, Redis bus (`OrionBusAsync`), pytest with `PYTHONPATH=services/<service>:.` from repo root.

**Spec:** [`docs/superpowers/specs/2026-07-04-vision-scene-belief-design.md`](../specs/2026-07-04-vision-scene-belief-design.md)

---

## File map

| Path | Responsibility |
|------|----------------|
| `services/orion-vision-window/app/scene_belief.py` | **Create** — `SceneBeliefTracker`, `SceneBeliefRegistry`, belief enrichment |
| `services/orion-vision-window/app/settings.py` | **Modify** — `WINDOW_BELIEF_*` env keys |
| `services/orion-vision-window/app/main.py` | **Modify** — wire belief registry in `_flush_and_publish` |
| `services/orion-vision-window/.env_example` | **Modify** — document belief keys |
| `services/orion-vision-window/docker-compose.yml` | **Modify** — pass belief env vars |
| `services/orion-vision-window/README.md` | **Modify** — evidence tiers table |
| `services/orion-vision-window/tests/test_scene_belief.py` | **Create** — unit tests for tracker algorithm |
| `services/orion-vision-window/tests/test_belief_flush_wiring.py` | **Create** — integration test for payload enrichment |
| `services/orion-vision-council/app/evidence_transition.py` | **Modify** — prefer `believed_hard_labels`; rename `labels_changed` → `salient_labels_changed` |
| `services/orion-vision-council/app/settings.py` | **Modify** — `COUNCIL_TRANSITION_REFRESH_SEC` default `0` |
| `services/orion-vision-council/.env_example` | **Modify** — refresh default 0, document belief gate |
| `services/orion-vision-council/docker-compose.yml` | **Modify** — compose default refresh `0` |
| `services/orion-vision-council/README.md` | **Modify** — belief tier + reason rename |
| `services/orion-vision-council/tests/test_scene_belief.py` | **Create** — council consumer tests |
| `services/orion-vision-council/tests/test_evidence_transition.py` | **Modify** — update `labels_changed` → `salient_labels_changed` |
| `services/orion-vision-council/tests/test_transition_settings.py` | **Modify** — assert refresh default 0 |
| `docs/vision_services.md` | **Modify** — document believed evidence tier |

**Non-goals (do not implement):** Redis-durable belief, embedding familiarity, `host_fallback` heartbeat, edge pipeline changes.

---

### Task 1: Window belief settings

**Files:**
- Modify: `services/orion-vision-window/app/settings.py`
- Modify: `services/orion-vision-window/.env_example`
- Modify: `services/orion-vision-window/docker-compose.yml`

- [ ] **Step 1: Add settings fields**

In `services/orion-vision-window/app/settings.py`, after `STALE_AFTER_MS`:

```python
    # Scene belief habituation (per-stream; ephemeral like live_state)
    WINDOW_BELIEF_ENABLED: bool = True
    WINDOW_BELIEF_VOTE_N: int = 3
    WINDOW_BELIEF_ENTER_VOTES: int = 2
    WINDOW_BELIEF_EXIT_VOTES: int = 1
```

- [ ] **Step 2: Update `.env_example`**

Append after `STALE_AFTER_MS=120000`:

```bash
# Scene belief habituation (observed → believed tier in summary.evidence)
WINDOW_BELIEF_ENABLED=true
WINDOW_BELIEF_VOTE_N=3
WINDOW_BELIEF_ENTER_VOTES=2
WINDOW_BELIEF_EXIT_VOTES=1
```

- [ ] **Step 3: Update docker-compose environment block**

In `services/orion-vision-window/docker-compose.yml`, add after `CHANNEL_WINDOW_REQUEST` line:

```yaml
      - WINDOW_BELIEF_ENABLED=${WINDOW_BELIEF_ENABLED:-true}
      - WINDOW_BELIEF_VOTE_N=${WINDOW_BELIEF_VOTE_N:-3}
      - WINDOW_BELIEF_ENTER_VOTES=${WINDOW_BELIEF_ENTER_VOTES:-2}
      - WINDOW_BELIEF_EXIT_VOTES=${WINDOW_BELIEF_EXIT_VOTES:-1}
```

- [ ] **Step 4: Sync local env**

Run from repo root:

```bash
python scripts/sync_local_env_from_example.py
```

Expected: keys added to `services/orion-vision-window/.env` (not committed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-window/app/settings.py \
  services/orion-vision-window/.env_example \
  services/orion-vision-window/docker-compose.yml
git commit -m "feat(vision-window): add scene belief configuration keys"
```

---

### Task 2: SceneBeliefTracker core (TDD)

**Files:**
- Create: `services/orion-vision-window/app/scene_belief.py`
- Create: `services/orion-vision-window/tests/test_scene_belief.py`

- [ ] **Step 1: Write failing tests**

Create `services/orion-vision-window/tests/test_scene_belief.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scene_belief import SceneBeliefRegistry, SceneBeliefTracker


def test_belief_ignores_single_empty_observation() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    tracker.observe(frozenset({"door", "screen"}))
    tracker.observe(frozenset())
    result = tracker.observe(frozenset({"door", "screen"}))
    assert result.believed_labels == frozenset({"door", "screen"})
    assert result.added == frozenset()
    assert result.removed == frozenset()


def test_belief_requires_enter_votes() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    tracker.observe(frozenset({"door", "screen"}))
    tracker.observe(frozenset({"door", "screen"}))
    once = tracker.observe(frozenset({"door", "screen", "package"}))
    assert "package" not in once.believed_labels
    twice = tracker.observe(frozenset({"door", "screen", "package"}))
    assert "package" in twice.believed_labels
    assert "package" in twice.added


def test_belief_requires_exit_votes() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    for _ in range(3):
        tracker.observe(frozenset({"door", "screen", "package"}))
    assert "package" in tracker.believed_labels
    tracker.observe(frozenset({"door", "screen"}))
    tracker.observe(frozenset({"door", "screen"}))
    result = tracker.observe(frozenset({"door", "screen"}))
    assert "package" not in result.believed_labels
    assert result.removed == frozenset({"package"})


def test_enrich_evidence_includes_believed_tier() -> None:
    tracker = SceneBeliefTracker(vote_n=3, enter_votes=2, exit_votes=1)
    tracker.observe(frozenset({"door"}))
    tracker.observe(frozenset({"door"}))
    evidence = {
        "hard_labels": ["door"],
        "soft_labels": [],
        "host_person_hits": 0,
        "edge_person_hits": 0,
        "caption_count": 0,
    }
    enriched = tracker.enrich_evidence(evidence)
    assert enriched["hard_labels"] == ["door"]
    assert enriched["believed_hard_labels"] == ["door"]
    assert enriched["belief"]["schema"] == "scene_belief.v1"
    assert enriched["belief"]["vote_n"] == 3
    assert enriched["belief"]["enter_votes"] == 2
    assert enriched["belief"]["exit_votes"] == 1
    assert enriched["belief"]["observation_count"] == 2


def test_registry_isolates_streams() -> None:
    registry = SceneBeliefRegistry(vote_n=3, enter_votes=2, exit_votes=1)
    registry.observe("cam0", frozenset({"door"}))
    registry.observe("cam0", frozenset({"door"}))
    registry.observe("cam1", frozenset({"chair"}))
    cam0 = registry.observe("cam0", frozenset({"door"}))
    cam1 = registry.observe("cam1", frozenset({"chair"}))
    assert cam0.believed_labels == frozenset({"door"})
    assert cam1.believed_labels == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests/test_scene_belief.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.scene_belief'`

- [ ] **Step 3: Implement scene_belief.py**

Create `services/orion-vision-window/app/scene_belief.py`:

```python
"""Per-stream scene label habituation (observed → believed tier)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeliefObserveResult:
    believed_labels: frozenset[str]
    added: frozenset[str]
    removed: frozenset[str]
    observation_count: int


class SceneBeliefTracker:
    def __init__(self, *, vote_n: int = 3, enter_votes: int = 2, exit_votes: int = 1) -> None:
        self._vote_n = vote_n
        self._enter_votes = enter_votes
        self._exit_votes = exit_votes
        self._ring: deque[frozenset[str]] = deque(maxlen=vote_n)
        self._believed: frozenset[str] = frozenset()

    @property
    def believed_labels(self) -> frozenset[str]:
        return self._believed

    def observe(self, observed: frozenset[str]) -> BeliefObserveResult:
        self._ring.append(observed)
        candidates = set(self._believed)
        for entry in self._ring:
            candidates.update(entry)

        new_believed = set(self._believed)
        for label in candidates:
            count = sum(1 for entry in self._ring if label in entry)
            if label not in self._believed and count >= self._enter_votes:
                new_believed.add(label)
            elif label in self._believed and count <= self._exit_votes:
                new_believed.discard(label)

        prev = self._believed
        self._believed = frozenset(new_believed)
        added = self._believed - prev
        removed = prev - self._believed
        return BeliefObserveResult(
            believed_labels=self._believed,
            added=added,
            removed=removed,
            observation_count=len(self._ring),
        )

    def enrich_evidence(self, evidence: dict[str, Any]) -> dict[str, Any]:
        out = dict(evidence)
        out["believed_hard_labels"] = sorted(self._believed)
        out["belief"] = {
            "schema": "scene_belief.v1",
            "vote_n": self._vote_n,
            "enter_votes": self._enter_votes,
            "exit_votes": self._exit_votes,
            "observation_count": len(self._ring),
        }
        return out


class SceneBeliefRegistry:
    def __init__(self, *, vote_n: int = 3, enter_votes: int = 2, exit_votes: int = 1) -> None:
        self._vote_n = vote_n
        self._enter_votes = enter_votes
        self._exit_votes = exit_votes
        self._trackers: dict[str, SceneBeliefTracker] = {}

    def _tracker(self, stream_id: str) -> SceneBeliefTracker:
        if stream_id not in self._trackers:
            self._trackers[stream_id] = SceneBeliefTracker(
                vote_n=self._vote_n,
                enter_votes=self._enter_votes,
                exit_votes=self._exit_votes,
            )
        return self._trackers[stream_id]

    def observe(self, stream_id: str, observed: frozenset[str]) -> BeliefObserveResult:
        return self._tracker(stream_id).observe(observed)

    def enrich_evidence(self, stream_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return self._tracker(stream_id).enrich_evidence(evidence)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests/test_scene_belief.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-window/app/scene_belief.py \
  services/orion-vision-window/tests/test_scene_belief.py
git commit -m "feat(vision-window): add SceneBeliefTracker with vote-based habituation"
```

---

### Task 3: Wire belief into window flush path

**Files:**
- Modify: `services/orion-vision-window/app/main.py`
- Create: `services/orion-vision-window/tests/test_belief_flush_wiring.py`

- [ ] **Step 1: Write failing integration test**

Create `services/orion-vision-window/tests/test_belief_flush_wiring.py`:

```python
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject

from app.main import WindowService


def _artifact(label: str, score: float = 0.8) -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id=f"art-{label}",
        correlation_id="c1",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label=label, score=score, box_xyxy=[0, 0, 1, 1])]
        ),
        timing={},
        model_fingerprints={},
    )


@pytest.mark.asyncio
async def test_flush_payload_includes_believed_hard_labels() -> None:
    svc = WindowService()
    now = time.time()
    buffered = [{"artifact": _artifact("door"), "ts": now, "env": None}]

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=buffered,
            correlation_id=None,
            causality_chain=[],
        )

    payload = svc._live_by_stream["cam0"]
    evidence = payload.summary["evidence"]
    assert "door" in evidence["hard_labels"]
    assert "believed_hard_labels" in evidence
    assert evidence["belief"]["schema"] == "scene_belief.v1"


@pytest.mark.asyncio
async def test_belief_survives_single_empty_flush() -> None:
    svc = WindowService()
    now = time.time()

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for label in ("door", "door"):
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _artifact(label), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[],
            correlation_id=None,
            causality_chain=[],
        )

    # Empty flush should not clear live state; last non-empty window still has belief
    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert "door" in evidence.get("believed_hard_labels", [])
```

Note: the second test's empty-buffer case is invalid for `_flush_and_publish` (early return). Replace with a flush that has artifacts producing no hard labels (all scores below threshold):

```python
@pytest.mark.asyncio
async def test_belief_survives_single_empty_observation() -> None:
    svc = WindowService()
    now = time.time()

    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for _ in range(2):
            await svc._flush_and_publish(
                stream_id="cam0",
                buffered=[{"artifact": _artifact("door"), "ts": now, "env": None}],
                correlation_id=None,
                causality_chain=[],
            )
        await svc._flush_and_publish(
            stream_id="cam0",
            buffered=[{"artifact": _artifact("door", score=0.01), "ts": now, "env": None}],
            correlation_id=None,
            causality_chain=[],
        )

    evidence = svc._live_by_stream["cam0"].summary["evidence"]
    assert evidence["hard_labels"] == []
    assert "door" in evidence["believed_hard_labels"]
```

Use the corrected version (low-score artifact), not the empty-buffer version.

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests/test_belief_flush_wiring.py -v
```

Expected: FAIL — `believed_hard_labels` KeyError or missing field

- [ ] **Step 3: Wire belief in WindowService**

In `services/orion-vision-window/app/main.py`:

Add import:

```python
from .scene_belief import SceneBeliefRegistry
```

In `WindowService.__init__`, after `self._cursor_i = 0`:

```python
        self._belief_registry = SceneBeliefRegistry(
            vote_n=settings.WINDOW_BELIEF_VOTE_N,
            enter_votes=settings.WINDOW_BELIEF_ENTER_VOTES,
            exit_votes=settings.WINDOW_BELIEF_EXIT_VOTES,
        )
```

In `_flush_and_publish`, after `build_window_payload(...)` and before `env_dump = payload.model_dump(...)`:

```python
        if settings.WINDOW_BELIEF_ENABLED:
            summary = dict(payload.summary or {})
            evidence = dict(summary.get("evidence") or {})
            observed = frozenset(str(x).strip().lower() for x in evidence.get("hard_labels") or [] if str(x).strip())
            result = self._belief_registry.observe(stream_id, observed)
            summary["evidence"] = self._belief_registry.enrich_evidence(stream_id, evidence)
            payload = payload.model_copy(update={"summary": summary})
            if result.added or result.removed:
                added = ",".join(sorted(result.added)) or "-"
                removed = ",".join(sorted(result.removed)) or "-"
                believed = ",".join(sorted(result.believed_labels)) or "-"
                logger.info(
                    f"[WINDOW] belief_transition stream={stream_id} "
                    f"added={added} removed={removed} believed={believed}"
                )
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests/test_belief_flush_wiring.py services/orion-vision-window/tests/test_scene_belief.py -v
```

Expected: all passed

- [ ] **Step 5: Run full window test suite**

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests -q
```

Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add services/orion-vision-window/app/main.py \
  services/orion-vision-window/tests/test_belief_flush_wiring.py
git commit -m "feat(vision-window): emit believed_hard_labels on flush"
```

---

### Task 4: Council snapshot prefers believed labels (TDD)

**Files:**
- Modify: `services/orion-vision-council/app/evidence_transition.py`
- Create: `services/orion-vision-council/tests/test_scene_belief.py`

- [ ] **Step 1: Write failing council tests**

Create `services/orion-vision-council/tests/test_scene_belief.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionWindowPayload

from app.evidence_transition import EvidenceTransitionTracker, snapshot_from_window


def _window(
    *,
    hard_labels: list[str] | None = None,
    believed_hard_labels: list[str] | None = None,
    stream_id: str = "cam0",
) -> VisionWindowPayload:
    evidence: dict = {
        "hard_labels": hard_labels or [],
        "soft_labels": [],
        "host_person_hits": 0,
        "caption_count": 0,
    }
    if believed_hard_labels is not None:
        evidence["believed_hard_labels"] = believed_hard_labels
        evidence["belief"] = {"schema": "scene_belief.v1"}
    return VisionWindowPayload(
        window_id="w1",
        start_ts=1.0,
        end_ts=2.0,
        stream_id=stream_id,
        summary={
            "object_counts": {},
            "top_labels": [],
            "captions": [],
            "item_count": 1,
            "detection_count": 0,
            "evidence": evidence,
        },
        artifact_ids=["art-1"],
    )


def test_snapshot_prefers_believed_hard_labels() -> None:
    window = _window(hard_labels=[], believed_hard_labels=["door", "screen"])
    snap = snapshot_from_window(window)
    assert snap.hard_labels == frozenset({"door", "screen"})
    assert snap.person_present is False


def test_snapshot_falls_back_to_hard_labels_without_belief() -> None:
    window = _window(hard_labels=["chair"])
    snap = snapshot_from_window(window)
    assert snap.hard_labels == frozenset({"chair"})


def test_stable_scene_when_observed_flickers_but_belief_stable() -> None:
    tracker = EvidenceTransitionTracker()
    stable = snapshot_from_window(
        _window(hard_labels=["door", "screen"], believed_hard_labels=["door", "screen"])
    )
    tracker.record_interpretation(stream_key="cam0", snapshot=stable, now=100.0)
    flicker = snapshot_from_window(
        _window(hard_labels=[], believed_hard_labels=["door", "screen"])
    )
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=flicker,
        now=110.0,
        max_refresh_sec=0.0,
    )
    assert decision.interpret is False
    assert decision.reason == "stable_scene"


def test_salient_labels_changed_on_belief_transition() -> None:
    tracker = EvidenceTransitionTracker()
    door = snapshot_from_window(_window(hard_labels=["door"], believed_hard_labels=["door"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=door, now=100.0)
    chair = snapshot_from_window(_window(hard_labels=["chair"], believed_hard_labels=["chair"]))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=chair,
        now=110.0,
        max_refresh_sec=0.0,
    )
    assert decision.interpret is True
    assert decision.reason == "salient_labels_changed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests/test_scene_belief.py -v
```

Expected: FAIL on `salient_labels_changed` (still returns `labels_changed`) or belief preference

- [ ] **Step 3: Update evidence_transition.py**

Replace `_hard_labels_from_window` with gate helper and update evaluate reason:

```python
def _hard_labels_from_evidence(evidence: dict) -> frozenset[str]:
    raw = evidence.get("hard_labels") or []
    return frozenset(str(label).strip().lower() for label in raw if str(label).strip())


def _labels_for_gate(window: VisionWindowPayload) -> frozenset[str]:
    summary = window.summary or {}
    evidence = summary.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    believed = evidence.get("believed_hard_labels")
    belief_meta = evidence.get("belief") or {}
    if (
        isinstance(believed, list)
        and believed
        and isinstance(belief_meta, dict)
        and belief_meta.get("schema") == "scene_belief.v1"
    ):
        return frozenset(str(label).strip().lower() for label in believed if str(label).strip())
    return _hard_labels_from_evidence(evidence)


def snapshot_from_window(window: VisionWindowPayload) -> EvidenceSnapshot:
    hard_labels = _labels_for_gate(window)
    return EvidenceSnapshot(
        hard_labels=hard_labels,
        person_present="person" in hard_labels,
    )
```

In `evaluate()`, change the generic label-change branch:

```python
            return EvidenceTransitionDecision(interpret=True, reason="salient_labels_changed")
```

(was `labels_changed`)

- [ ] **Step 4: Run council belief tests**

```bash
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests/test_scene_belief.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/orion-vision-council/app/evidence_transition.py \
  services/orion-vision-council/tests/test_scene_belief.py
git commit -m "feat(vision-council): gate transitions on believed_hard_labels"
```

---

### Task 5: Disable refresh TTL by default

**Files:**
- Modify: `services/orion-vision-council/app/settings.py`
- Modify: `services/orion-vision-council/.env_example`
- Modify: `services/orion-vision-council/docker-compose.yml`
- Modify: `services/orion-vision-council/tests/test_evidence_transition.py`
- Modify: `services/orion-vision-council/tests/test_transition_settings.py`

- [ ] **Step 1: Write failing settings test**

Add to `services/orion-vision-council/tests/test_transition_settings.py`:

```python
def test_settings_refresh_ttl_default_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COUNCIL_TRANSITION_REFRESH_SEC", raising=False)
    monkeypatch.delenv("COUNCIL_EVIDENCE_SKIP_MAX_SEC", raising=False)
    from app.settings import Settings

    assert Settings().COUNCIL_TRANSITION_REFRESH_SEC == 0.0
```

Add refresh-disabled integration test to `test_scene_belief.py`:

```python
def test_refresh_ttl_disabled_by_default() -> None:
    tracker = EvidenceTransitionTracker()
    snap = snapshot_from_window(
        _window(hard_labels=["door"], believed_hard_labels=["door"])
    )
    tracker.record_interpretation(stream_key="cam0", snapshot=snap, now=100.0)
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=snap,
        now=1000.0,
        max_refresh_sec=0.0,
    )
    assert decision.interpret is False
    assert decision.reason == "stable_scene"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests/test_transition_settings.py::test_settings_refresh_ttl_default_zero -v
```

Expected: FAIL — assert 120.0 == 0.0

- [ ] **Step 3: Change default and env surfaces**

In `services/orion-vision-council/app/settings.py`, change:

```python
    COUNCIL_TRANSITION_REFRESH_SEC: float = Field(
        default=0.0,
```

In `.env_example`, change comment and value:

```bash
# Force refresh at least every N seconds when labels are stable (0 = disable force; contract A).
COUNCIL_TRANSITION_REFRESH_SEC=0
```

In `docker-compose.yml`, change:

```yaml
      - COUNCIL_TRANSITION_REFRESH_SEC=${COUNCIL_TRANSITION_REFRESH_SEC:-0}
```

- [ ] **Step 4: Update existing test for reason rename**

In `services/orion-vision-council/tests/test_evidence_transition.py`, change `test_tracker_labels_changed`:

```python
def test_tracker_salient_labels_changed() -> None:
    ...
    assert decision.reason == "salient_labels_changed"
```

(function rename optional; reason string is required)

- [ ] **Step 5: Sync local env and run council tests**

```bash
python scripts/sync_local_env_from_example.py
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q
```

Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add services/orion-vision-council/app/settings.py \
  services/orion-vision-council/.env_example \
  services/orion-vision-council/docker-compose.yml \
  services/orion-vision-council/tests/test_evidence_transition.py \
  services/orion-vision-council/tests/test_transition_settings.py \
  services/orion-vision-council/tests/test_scene_belief.py
git commit -m "feat(vision-council): disable refresh TTL by default (contract A)"
```

---

### Task 6: Documentation

**Files:**
- Modify: `services/orion-vision-window/README.md`
- Modify: `services/orion-vision-council/README.md`
- Modify: `docs/vision_services.md`

- [ ] **Step 1: Update window README evidence table**

Add rows after `hard_labels`:

```markdown
| `believed_hard_labels` | Habituated labels (vote debounce over last N flushes; council gate input) |
| `belief` | Metadata: `schema`, `vote_n`, `enter_votes`, `exit_votes`, `observation_count` |
```

Add note: Council gates on `believed_hard_labels` when `belief.schema == scene_belief.v1`.

- [ ] **Step 2: Update council README**

Replace transition gate bullet:

```markdown
2. **Evidence transition gate** — `evidence_transition.py` compares **believed** host labels (`summary.evidence.believed_hard_labels` when present) and person presence per stream. Council runs metacog only on `first_window`, `person_entered`, `person_exited`, `salient_labels_changed`, or `refresh_ttl` when explicitly enabled (`COUNCIL_TRANSITION_REFRESH_SEC > 0`). Stable scenes skip LLM and publish nothing.
```

- [ ] **Step 3: Update docs/vision_services.md**

Under `VisionWindowPayload` summary bullet, extend:

```markdown
-   `summary.evidence`: `hard_labels` (observed), `believed_hard_labels` (habituated gate input), `belief` metadata, soft labels, person hit counts.
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-vision-window/README.md \
  services/orion-vision-council/README.md \
  docs/vision_services.md
git commit -m "docs: document vision scene belief evidence tier"
```

---

### Task 7: Verification gate

- [ ] **Step 1: Run deterministic checks from repo root**

```bash
git diff --check
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
pytest services/orion-vision-window/tests -q
pytest services/orion-vision-council/tests -q
```

Expected: no whitespace errors; env parity OK; all tests pass.

- [ ] **Step 2: Docker compose config smoke (optional if Docker available)**

```bash
docker compose --env-file .env --env-file services/orion-vision-window/.env \
  -f services/orion-vision-window/docker-compose.yml config > /dev/null
docker compose --env-file .env --env-file services/orion-vision-council/.env \
  -f services/orion-vision-council/docker-compose.yml config > /dev/null
```

- [ ] **Step 3: Code review subagent**

Run code review skill on the branch; fix material findings.

---

## Deploy order

1. Deploy **orion-vision-window** first (starts emitting `believed_hard_labels`).
2. Deploy **orion-vision-council** second (starts gating on belief; refresh TTL off).
3. Runtime acceptance (30 min cam0 sample):
   - `evidence_transition interpret` ≤ 2% of window count
   - No interpret lines triggered by observed-only flicker (`hard_labels=-` while belief stable)
   - `refresh_ttl` interpret count = 0

## Restart commands

```bash
docker compose --env-file .env --env-file services/orion-vision-window/.env \
  -f services/orion-vision-window/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-vision-council/.env \
  -f services/orion-vision-council/docker-compose.yml up -d --build
```

---

## Self-review (spec coverage)

| Spec section | Task |
|--------------|------|
| §4 SceneBeliefTracker algorithm | Task 2 |
| §4.1 Wire `_flush_and_publish` | Task 3 |
| §5 Belief fields in evidence | Tasks 2–3 |
| §6 Council snapshot + reasons | Task 4 |
| §6.3 Disable refresh TTL | Task 5 |
| §8 Schema/docs (additive) | Task 6 |
| §9 Configuration | Tasks 1, 5 |
| §10 Testing | Tasks 2–5 |
| §11 Belief transition logging | Task 3 |
| §13 Implementation order | Tasks 1–6 |
| Runtime acceptance (§10) | Task 7 deploy notes |

**Non-goals confirmed out of scope:** Redis belief, embeddings, host_fallback heartbeat, edge changes.
