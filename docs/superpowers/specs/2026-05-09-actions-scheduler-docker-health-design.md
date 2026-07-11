# Actions Scheduler — Periodic Docker Health — Design

**Date:** 2026-05-09  
**Status:** Design approved; committed spec pending operator review before implementation plan  
**Primary goal:** Let `orion-actions` detect Docker `HEALTHCHECK` failures on the node on the same cadence as existing periodic skills, and surface them through the same threshold notify and audit path when enabled.

---

## Problem

Docker Compose services define `healthcheck`; the engine marks containers healthy or unhealthy. Orion has `skills.docker.ps_status.v1` (on-demand inventory) but no **automatic** polling in the actions periodic skills loop. Operators who already use `ACTIONS_SKILLS_*` for GPU/biometrics thresholds have no first-class hook for container health without external monitoring or manual skill runs.

---

## Goal

1. When periodic skills run and configuration allows it, dispatch `skills.docker.ps_status.v1` through the existing Cortex orch request path (`_dispatch_scheduled_skill`).
2. Derive **findings** from the skill result for containers that are running but **unhealthy** per Docker’s reported `status` text.
3. Merge those findings with existing biometrics/GPU threshold findings and reuse the existing notify + audit behavior when `ACTIONS_SKILLS_NOTIFY_ENABLED` is true.
4. Default **off** so environments without Docker socket access to cortex-exec do not pay RPC cost or noise.

---

## Non-goals (v1)

- Separate interval/cursor for Docker-only ticks (defer unless ops require faster cadence than `ACTIONS_SKILLS_INTERVAL_SECONDS`).
- Allowlist or prefix filters for container names (operator chose scope **A**: any unhealthy running container).
- Structured `Health` object from the Docker Engine API in `orion-cortex-exec` (optional follow-up; v1 uses `status` string heuristics only).
- Dedicated “recovered” notify when health returns to healthy (silent recovery is acceptable for v1).
- Multi-replica coordination for actions beyond existing single-writer assumptions documented for the scheduler.

---

## Selected approach

**Extend the existing periodic skills tick** in `services/orion-actions/app/main.py` (`_scheduler_loop`):

- After biometrics and GPU snapshots (same `wait_for_result` as today), if `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED` is true, dispatch `skills.docker.ps_status.v1`.
- Parse the decoded skill result with a pure helper `_scheduler_docker_findings(result: dict) -> list[str]`.
- Concatenate docker findings with `_scheduler_threshold_findings(...)` before the existing `if findings:` block that audits and dispatches `SKILL_NOTIFY_CHAT_MESSAGE_V1`.

**Verb constant:** add `SKILL_DOCKER_PS_STATUS_V1 = "skills.docker.ps_status.v1"` in `services/orion-actions/app/logic.py` next to other skill constants.

---

## Detection rules

**Input:** dict returned by `_extract_skill_result_from_orch` for the docker skill (same shape as other skills: top-level keys from skill result payload).

**No findings** when any of:

- `available` is false or missing and treated as unavailable.
- `containers` is missing or not a non-empty list. The helper consumes the same object `_extract_skill_result_from_orch` returns for other skills (Cortex metadata `skill_result`); for `skills.docker.ps_status.v1` that is the verb’s `result` dict with top-level `available`, `containers`, etc.
- Skill dispatch failed before a dict is returned (existing error handling; no new behavior required beyond not crashing).

**Per container:** consider an item a candidate only if it represents a **running** container:

- Prefer: `state` equals `"running"` (case-insensitive) when `state` is present.
- If `state` is absent (some `docker ps` JSON paths), treat as candidate only if `status` is a string and starts with `"Up "` (case-insensitive) to avoid alerting on exited containers whose status text might contain other words.

**Unhealthy:** `status` is a string and contains the substring **`(unhealthy)`** (case-insensitive). Rationale: Docker’s human-readable status line uses this form for failed healthchecks; requiring the parentheses reduces false positives from unrelated text.

**Finding string format:** stable, one per unhealthy container, e.g. `docker_unhealthy:<container_name>:<short_id>` where `short_id` is first 12 hex chars of id if available, else `unknown`. Empty `name` may use `unnamed`.

---

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `ACTIONS_SKILLS_DOCKER_HEALTH_ENABLED` | `false` | When `true` and periodic skills branch runs, also dispatch docker ps status skill. |

Document in `services/orion-actions/.env_example` adjacent to other `ACTIONS_SKILLS_*` keys. Add a one-line note in `services/orion-actions/README.md` under the periodic skills / scheduler section: docker check requires cortex-exec runtime with Docker engine access (socket or CLI) as already required by `skills.docker.ps_status.v1`.

**Interaction with notify:** Docker findings participate in the **same** merged `findings` list and the same `ACTIONS_SKILLS_NOTIFY_ENABLED` gate as GPU/biometrics. No separate `ACTIONS_SKILLS_DOCKER_NOTIFY_ENABLED` in v1.

---

## Audit and dedupe

- Reuse `_audit(..., action_name="skills.periodic.thresholds", extra={"findings": findings})` when any finding is present (including docker-only).
- Reuse `_threshold_notify_skill_args` / existing dedupe window on the notify dispatch; body text remains a semicolon-separated list of findings. Optional minor prefix tweak (e.g. first line “Periodic threshold findings:”) is acceptable if it clarifies mixed sources without changing dedupe keys semantics — **dedupe key must remain stable** for the same correlation id pattern already used for threshold notify.

---

## Testing

1. **Unit tests** for `_scheduler_docker_findings` in a new test module or next to `test_skill_scheduler.py`: cases for healthy running, `(unhealthy)`, unavailable skill, empty containers, stopped container (no alert), missing state with `Up ... (unhealthy)`.
2. **Integration-style** (optional v1): if extracting the findings function to a small `logic` or `scheduler_findings` module improves testability without large refactors, do so; otherwise test via imported private function from `main` only if the repo already does that for similar helpers — prefer a dedicated pure function in `logic.py` or `docker_health.py` under `app/` to keep `main.py` thinner only if the change stays minimal.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| False positives from substring match | Require `(unhealthy)` not bare word `unhealthy`. |
| False negatives if Docker changes status format | Document dependency; follow-up: structured health in cortex-exec. |
| Extra RPC load every tick when enabled | Default off; same interval as existing periodic skills. |
| Stopped/unhealthy wording edge cases | Running/`Up` gate before marking unhealthy. |

---

## Implementation checklist (for planning phase)

- [ ] `Settings` field + `.env_example` + README blurb  
- [ ] `SKILL_DOCKER_PS_STATUS_V1` + scheduler dispatch + merge findings  
- [ ] `_scheduler_docker_findings` + tests  
- [ ] Manual or scripted verification on a stack with a failing healthcheck (optional evidence in PR)
