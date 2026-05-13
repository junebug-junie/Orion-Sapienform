# Actions Scheduler — Periodic Docker Health — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED=true`, each periodic skills tick in `orion-actions` also runs `skills.docker.ps_status.v1`, derives findings for running unhealthy containers, merges them with existing GPU/biometrics threshold findings, and reuses the same audit + notify path when `ACTIONS_SKILLS_NOTIFY_ENABLED=true`.

**Architecture:** Add a pure `scheduler_docker_findings(skill_result)` helper and `SKILL_DOCKER_PS_STATUS_V1` constant in `services/orion-actions/app/logic.py`. Extend `Settings` and env wiring. In `main.py` `_scheduler_loop`, after biometrics and GPU dispatches, conditionally dispatch the docker verb with the same `wait_for_result` flag, merge lists, leave `_threshold_notify_skill_args` and dedupe keys unchanged so behavior stays predictable.

**Tech Stack:** Python 3, Pydantic settings, FastAPI lifespan scheduler (existing), pytest, existing Cortex orch RPC path.

---

## File structure

| File | Responsibility |
|------|------------------|
| `services/orion-actions/app/logic.py` | `SKILL_DOCKER_PS_STATUS_V1`; pure `scheduler_docker_findings()` + tiny `_docker_short_id()` private helper in same module |
| `services/orion-actions/app/settings.py` | `actions_skills_docker_health_enabled: bool` default `False`, alias `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED` |
| `services/orion-actions/app/main.py` | Import constant + helper; extend periodic-skills block to dispatch docker and merge findings |
| `services/orion-actions/.env_example` | Document new env var next to other `ACTIONS_SKILLS_*` keys |
| `services/orion-actions/docker-compose.yml` | Pass `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED` into container environment |
| `services/orion-actions/README.md` | One short paragraph under periodic skills: optional docker health, requires cortex-exec Docker access |
| `services/orion-actions/tests/test_scheduler_docker_findings.py` | Unit tests for `scheduler_docker_findings` |

No changes to `orion-cortex-exec` or the skill verb for v1.

---

### Task 1: Pure findings helper + unit tests (TDD)

**Files:**
- Modify: `services/orion-actions/app/logic.py`
- Create: `services/orion-actions/tests/test_scheduler_docker_findings.py`

**Behavior (must match design spec):**

- Return `[]` if `skill_result` is not a dict, or `available` is not truthy, or `containers` is missing / not a list / empty.
- For each dict in `containers`:
  - Require `status` to be a string containing **`(unhealthy)`** case-insensitively.
  - If `state` is a non-empty string: include only when `state.strip().lower() == "running"`.
  - If `state` is missing, empty, or not a string: include only when `status` starts with **`Up `** (case-insensitive).
- Emit `docker_unhealthy:<name>:<short_id>` where `name` is stripped string or `unnamed`; `short_id` is first 12 hex characters taken from `id` in order scanned (typical Docker id), or `unknown` if fewer than 12 hex digits found.

- [ ] **Step 1: Add failing tests**

Create `services/orion-actions/tests/test_scheduler_docker_findings.py`:

```python
from __future__ import annotations

import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import scheduler_docker_findings  # noqa: E402


def test_empty_not_dict() -> None:
    assert scheduler_docker_findings([]) == []  # type: ignore[arg-type]
    assert scheduler_docker_findings({}) == []


def test_unavailable_no_findings() -> None:
    assert scheduler_docker_findings({"available": False, "containers": []}) == []
    assert scheduler_docker_findings({"available": True, "containers": None}) == []  # type: ignore[dict-item]


def test_healthy_running_no_findings() -> None:
    skill = {
        "available": True,
        "containers": [
            {"id": "abc123def456", "name": "svc", "state": "running", "status": "Up 2 hours (healthy)"},
        ],
    }
    assert scheduler_docker_findings(skill) == []


def test_running_unhealthy_finding() -> None:
    skill = {
        "available": True,
        "containers": [
            {
                "id": "deadbeefc0ffee00000000000000000000000000000000000000000000000001",
                "name": "bad-svc",
                "state": "running",
                "status": "Up 1 minute (unhealthy)",
            },
        ],
    }
    assert scheduler_docker_findings(skill) == ["docker_unhealthy:bad-svc:deadbeefc0ff"]


def test_exited_with_unhealthy_text_skipped() -> None:
    skill = {
        "available": True,
        "containers": [
            {"id": "x", "name": "gone", "state": "exited", "status": "Exited (1) ... (unhealthy)"},
        ],
    }
    assert scheduler_docker_findings(skill) == []


def test_missing_state_requires_up_prefix() -> None:
    skill = {
        "available": True,
        "containers": [
            {"id": "aaaabbbbccccddddeeeeffff0000111122223333444455556666777788889999", "name": "n", "status": "Up 5s (unhealthy)"},
        ],
    }
    assert scheduler_docker_findings(skill) == ["docker_unhealthy:n:aaaabbbbcccc"]


def test_unnamed_container() -> None:
    skill = {
        "available": True,
        "containers": [
            {"id": "1111111111111111111111111111111111111111111111111111111111111111", "state": "running", "status": "Up (unhealthy)"},
        ],
    }
    assert scheduler_docker_findings(skill) == ["docker_unhealthy:unnamed:111111111111"]
```

- [ ] **Step 2: Run tests — expect import or assertion failures**

Run from repo root:

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-actions/tests/test_scheduler_docker_findings.py -q --tb=short
```

Expected: `ImportError` or `AttributeError` for missing `scheduler_docker_findings`.

- [ ] **Step 3: Implement `scheduler_docker_findings` in `logic.py`**

Add after the existing `SKILL_*` constants:

```python
SKILL_DOCKER_PS_STATUS_V1 = "skills.docker.ps_status.v1"


def _docker_short_id(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    hex_chars: list[str] = []
    for ch in s:
        if ch in "0123456789abcdefABCDEF":
            hex_chars.append(ch.lower())
            if len(hex_chars) >= 12:
                return "".join(hex_chars[:12])
    return "unknown"


def scheduler_docker_findings(skill_result: Dict[str, Any]) -> list[str]:
    if not isinstance(skill_result, dict) or not skill_result.get("available"):
        return []
    containers = skill_result.get("containers")
    if not isinstance(containers, list) or not containers:
        return []
    out: list[str] = []
    for item in containers:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if not isinstance(status, str) or "(unhealthy)" not in status.lower():
            continue
        state_val = item.get("state")
        if isinstance(state_val, str) and state_val.strip():
            if state_val.strip().lower() != "running":
                continue
        else:
            if not status.lower().startswith("up "):
                continue
        name_raw = item.get("name")
        name = str(name_raw).strip() if name_raw is not None else ""
        if not name:
            name = "unnamed"
        short_id = _docker_short_id(item.get("id"))
        out.append(f"docker_unhealthy:{name}:{short_id}")
    return out
```

Add `Any` to the `typing` import in `logic.py` if not already present.

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-actions/tests/test_scheduler_docker_findings.py -q --tb=short
```

Expected: all tests passed.

- [ ] **Step 5: Commit**

```bash
git add services/orion-actions/app/logic.py services/orion-actions/tests/test_scheduler_docker_findings.py
git commit -m "feat(orion-actions): add scheduler_docker_findings helper for periodic health"
```

---

### Task 2: Settings + env example + docker-compose

**Files:**
- Modify: `services/orion-actions/app/settings.py`
- Modify: `services/orion-actions/.env_example`
- Modify: `services/orion-actions/docker-compose.yml`

- [ ] **Step 1: Add setting**

In `Settings` next to `actions_skills_biometrics_stability_threshold`:

```python
    actions_skills_docker_health_enabled: bool = Field(False, alias="ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED")
```

- [ ] **Step 2: `.env_example`**

After `ACTIONS_SKILLS_BIOMETRICS_STABILITY_THRESHOLD=0.3` add:

```bash
# When true, periodic skills also dispatch skills.docker.ps_status.v1 (requires cortex-exec Docker access).
ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED=false
```

- [ ] **Step 3: `docker-compose.yml`**

After the line for `ACTIONS_SKILLS_BIOMETRICS_STABILITY_THRESHOLD`, add:

```yaml
      - ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED=${ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED}
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-actions/app/settings.py services/orion-actions/.env_example services/orion-actions/docker-compose.yml
git commit -m "feat(orion-actions): env flag for scheduled docker health snapshot"
```

---

### Task 3: Wire scheduler in `main.py`

**Files:**
- Modify: `services/orion-actions/app/main.py`

- [ ] **Step 1: Extend imports from `.logic`**

Change the import block to include:

```python
    SKILL_DOCKER_PS_STATUS_V1,
    scheduler_docker_findings,
```

- [ ] **Step 2: Replace periodic-skills block**

Locate the block (approximately lines 1826–1841) that dispatches biometrics and GPU and builds `findings`. Replace the inner part so that:

1. `docker_result: dict[str, Any] = {}` before the `if wait_for_result` or initialize inside branch — pattern: always define `docker_result` when you need merge.

Concrete pattern:

```python
                    skill_parent = BaseEnvelope(kind="orion.actions.trigger.skills.v1", source=src, payload={"scheduled": True})
                    wait_for_result = bool(settings.actions_skills_notify_enabled)
                    biometrics_result = await _dispatch_scheduled_skill(skill_parent, verb_name=SKILL_BIOMETRICS_SNAPSHOT_V1, wait_for_result=wait_for_result, metadata={"schedule": "periodic_skills"})
                    gpu_result = await _dispatch_scheduled_skill(skill_parent, verb_name=SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1, wait_for_result=wait_for_result, metadata={"schedule": "periodic_skills"})
                    docker_result: dict[str, Any] = {}
                    if settings.actions_skills_docker_health_enabled and wait_for_result:
                        docker_result = await _dispatch_scheduled_skill(
                            skill_parent,
                            verb_name=SKILL_DOCKER_PS_STATUS_V1,
                            wait_for_result=True,
                            metadata={"schedule": "periodic_skills"},
                        )
                    if wait_for_result:
                        findings = _scheduler_threshold_findings(biometrics_snapshot=biometrics_result, gpu_snapshot=gpu_result)
                        findings = [*findings, *scheduler_docker_findings(docker_result)]
                        if findings:
```

The remainder of the block (`_audit`, `notify_parent`, `_dispatch_scheduled_skill` for notify) stays unchanged.

**Note:** Docker skill runs only when **both** `actions_skills_docker_health_enabled` and `wait_for_result` (`ACTIONS_SKILLS_NOTIFY_ENABLED`) are true, because findings are only consumed in that branch. This avoids a useless Cortex RPC when notify is off. Document in README: enabling docker health without notify has no effect until notify is enabled.

- [ ] **Step 3: Run service tests**

```bash
./scripts/test_service.sh orion-actions services/orion-actions/tests/test_scheduler_docker_findings.py services/orion-actions/tests/test_skill_scheduler.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add services/orion-actions/app/main.py
git commit -m "feat(orion-actions): merge docker health findings into periodic skills notify"
```

---

### Task 4: README

**Files:**
- Modify: `services/orion-actions/README.md`

- [ ] **Step 1: Add documentation**

In the section that describes periodic skills / `ACTIONS_SKILLS_*` (near GPU and biometrics scheduler text), add 2–3 sentences:

- `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED` (default `false`) adds `skills.docker.ps_status.v1` on the periodic tick **only when** `ACTIONS_SKILLS_NOTIFY_ENABLED` is also true (same tick as GPU/biometrics wait path); findings merge into the same threshold notify. If notify is off, the docker snapshot is not requested (no merge path).
- Requires `orion-cortex-exec` to have Docker engine access, same as manual Hub skill runs for `skills.docker.ps_status.v1`.

- [ ] **Step 2: Commit**

```bash
git add services/orion-actions/README.md
git commit -m "docs(orion-actions): document scheduled docker health env flag"
```

---

### Task 5: Verification note (optional live stack)

**Files:** none required

- [ ] **Step 1 (optional):** On a machine with compose healthchecks, set `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED=true`, `ACTIONS_SKILLS_NOTIFY_ENABLED=true`, shorten `ACTIONS_SKILLS_INTERVAL_SECONDS` for a test, break one service health endpoint, confirm one notify includes `docker_unhealthy:...`.

Evidence is optional for merge; unit tests are the required bar per repo AGENTS.md for this change class.

---

## Spec coverage (self-review)

| Spec section | Task |
|--------------|------|
| Dispatch `skills.docker.ps_status.v1` when enabled | Task 3 |
| `scheduler_docker_findings` rules + finding format | Task 1 |
| Merge with biometrics/GPU + same notify/audit | Task 3 |
| Default off + env example + compose | Task 2 |
| README operator note | Task 4 |
| Constant in `logic.py` | Task 1 + Task 3 import |

**Plan vs design doc:** Design said “same `wait_for_result` as today” on the dispatch call; implementation only invokes docker when `wait_for_result` is true so there is no orphan RPC. Update the design spec if you want strict literal parity with biometrics always-on dispatch; product-wise this is preferable.

**Placeholder scan:** None.

**Type consistency:** `docker_result` is always a `dict` from `_dispatch_scheduled_skill` return type; `scheduler_docker_findings` accepts dict.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-09-actions-scheduler-docker-health-implementation.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — Fresh subagent per task, review between tasks. **Required sub-skill:** `superpowers:subagent-driven-development`.

2. **Inline execution** — Run tasks in this session with checkpoints. **Required sub-skill:** `superpowers:executing-plans`.

Which approach do you want for implementation?
