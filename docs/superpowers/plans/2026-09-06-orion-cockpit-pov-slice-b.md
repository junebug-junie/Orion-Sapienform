# Orion Cockpit POV (Slice B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Slice A gap beads for `association`, `stance_inputs`, and `motor_boot` with real Turn Sighting hops that carry the actual payloads Orion saw (association bundle, stance request inputs, exact FCC motor prompt).

**Architecture:** Hub keeps owning monotonic `seq` and WS/bus publish. Pre-motor emit writes real `association` + `stance_inputs` hops (ingress stays an honest gap for Slice C). After `build_harness_prompt` in `HarnessRunner.run`, the governor publishes one synthetic `harness.run.step.v1` marked `_cockpit: "cockpit.motor_boot.v1"` carrying the exact prompt string; Hub’s existing step drain converts it to a `motor_boot` cockpit hop before any `motor_hop`. Soft HUD inspector gains a Prompt/Prefix section when `raw.prompt` is present.

**Tech Stack:** Existing `CockpitHopV1` + `orion/cockpit/*`, Hub `cockpit_emit` / `turn_orchestrator`, harness `runner.py` + `step_stream.py`, Hub static Soft HUD JS, pytest (no new services).

**Spec:** `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md`  
**Prerequisite:** Slice A landed (`docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-a.md`). Work from the Slice A branch/worktree (or a fresh `feat/cockpit-pov-slice-b` branched from it).

**Out of this plan:** Slice C (ingress thickness + remaining offboarding). Mind-enrichment coloring added inside `orion-thought` after the Hub request is sent is **not** required for Slice B — Hub records the `StanceReactRequestV1` inputs it actually sent.

**Plan artifacts already written with this doc:** Slice C stub + design status line update. Task 5 still verifies gates and commits those docs with the implementation branch.

## Global Constraints

- Unified Orion turn path only — not classic PlanRunner
- Never fabricate hops; missing stages stay `status="gap"`
- `motor_boot.raw.prompt` must be the exact string returned by `build_harness_prompt` at the assembly site — no re-compile later for the hop
- Hub owns `seq` via `orion.cockpit.sequencer` — harness does **not** assign cockpit seq
- Keep Turn Trace panel mounted and working
- Soft HUD aesthetic unchanged
- Work only in a git worktree; commit per task
- Do not shrink `graphify-out/`; run `scripts/safe_graphify_update.sh` once before PR, not mid-task
- No new bus channel unless a task explicitly requires it (Slice B reuses `orion:harness:run:step` + `orion:cockpit:hop`)

## File structure (locked)

| File | Responsibility |
|------|----------------|
| `orion/cockpit/markers.py` | `COCKPIT_MOTOR_BOOT_MARKER` shared by harness + Hub (no hub←harness import) |
| `orion/cockpit/builders.py` | Add `hop_from_association`, `hop_from_stance_inputs`, `hop_from_motor_boot` |
| `orion/cockpit/tests/test_builders.py` | Builder unit tests for the three new hops |
| `orion/hub/cockpit_emit.py` | Pre-motor emit takes association + stance inputs; stop gap for association/stance_inputs/motor_boot; drain recognizes motor_boot step marker |
| `orion/hub/turn_orchestrator.py` | Pass association + stance request fields into pre-motor emit |
| `orion/harness/runner.py` | After `build_harness_prompt`, publish synthetic motor_boot step |
| `orion/harness/step_stream.py` | Optional tiny helper `publish_harness_motor_boot` (or inline in runner) |
| `orion/harness/tests/test_harness_runner_motor_boot_cockpit.py` | Assert prompt publish happens once with exact string |
| `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py` | Update stage order expectations; motor_boot from drain |
| `services/orion-hub/static/js/cockpit-hud.js` | Inspector Prompt/Prefix section when `raw.prompt` exists |
| `services/orion-hub/tests/test_cockpit_hud_ui.py` | UI assertion for prompt section |
| `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-c-stub.md` | Replace B/C stub — C-only checklist |
| `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md` | Status line: Slice B planned |

### Locked marker / wire contract

```python
COCKPIT_MOTOR_BOOT_MARKER = "cockpit.motor_boot.v1"
# Published as HarnessRunStepV1.step:
# {
#   "_cockpit": COCKPIT_MOTOR_BOOT_MARKER,
#   "prompt": "<exact build_harness_prompt return value>",
#   "prompt_char_len": <int>,
# }
# step_index: -1  (never a real FCC tool step index)
```

Hub drain: if `step.get("_cockpit") == COCKPIT_MOTOR_BOOT_MARKER` → `stage="motor_boot"`, else existing `motor_hop` path.

### Locked pre-motor stage order after Slice B

| seq order | stage | status |
|-----------|--------|--------|
| 0 | `ingress` | `gap` (`deferred_to="slice_c"`) |
| 1 | `association` | `ok` (real bundle) |
| 2 | `stance_inputs` | `ok` (real Hub request inputs) |
| 3 | `stance_decision` | `ok` (unchanged) |
| …later… | `motor_boot` | `ok` (from harness step drain, before motor hops) |
| … | `motor_hop` | `ok` |

---

### Task 1: Builders for association, stance_inputs, motor_boot

**Files:**
- Create: `orion/cockpit/markers.py`
- Modify: `orion/cockpit/builders.py`
- Modify: `orion/cockpit/tests/test_builders.py`

**Interfaces:**
- Consumes: `CockpitHopV1`, `_base_hop`
- Produces:
  - `COCKPIT_MOTOR_BOOT_MARKER = "cockpit.motor_boot.v1"` in `orion/cockpit/markers.py`
  - `hop_from_association(*, correlation_id: str, seq: int, association: dict[str, Any]) -> CockpitHopV1`
  - `hop_from_stance_inputs(*, correlation_id: str, seq: int, stance_inputs: dict[str, Any]) -> CockpitHopV1`
  - `hop_from_motor_boot(*, correlation_id: str, seq: int, prompt: str, *, producer: str = "orion-harness-governor") -> CockpitHopV1`
  - `gap_hop(..., deferred_to: str = "slice_b")` — callers for ingress must pass `deferred_to="slice_c"`

- [ ] **Step 1: Write the failing tests**

Create `orion/cockpit/markers.py` only after the test fails on import — first append tests:

Append to `orion/cockpit/tests/test_builders.py`:

```python
from orion.cockpit.builders import (
    hop_from_association,
    hop_from_motor_boot,
    hop_from_stance_inputs,
)


def test_hop_from_association_ok():
    association = {
        "schema_version": "hub.association.bundle.v1",
        "correlation_id": "c1",
        "broadcast_stale": True,
        "broadcast": None,
        "execution_trajectory_slice": {"tick": 1},
        "repair_bundle": {"status": "ok"},
        "read_source": "felt_state_reader",
    }
    hop = hop_from_association(
        correlation_id="c1",
        seq=1,
        association=association,
    )
    assert hop.stage == "association"
    assert hop.status == "ok"
    assert hop.raw["broadcast_stale"] is True
    assert hop.raw["execution_trajectory_slice"]["tick"] == 1
    assert "association" in hop.visor_line
    assert hop.producer == "orion-hub"


def test_hop_from_stance_inputs_ok():
    payload = {
        "user_message": "hello there",
        "session_id": "s1",
        "llm_profile": "brain",
        "stance_inputs": {"user_message": "hello there", "surface_context": {"k": 1}},
    }
    hop = hop_from_stance_inputs(
        correlation_id="c1",
        seq=2,
        stance_inputs=payload,
    )
    assert hop.stage == "stance_inputs"
    assert hop.status == "ok"
    assert hop.raw["user_message"] == "hello there"
    assert hop.raw["stance_inputs"]["surface_context"]["k"] == 1
    assert hop.summary["user_message_len"] == len("hello there")
    assert hop.producer == "orion-hub"


def test_hop_from_motor_boot_carries_exact_prompt():
    prompt = "WHO YOU ARE\n...\nUSER: hi"
    hop = hop_from_motor_boot(
        correlation_id="c1",
        seq=4,
        prompt=prompt,
    )
    assert hop.stage == "motor_boot"
    assert hop.status == "ok"
    assert hop.raw["prompt"] == prompt
    assert hop.raw["prompt_char_len"] == len(prompt)
    assert hop.summary["prompt_char_len"] == len(prompt)
    assert str(len(prompt)) in hop.visor_line
    assert hop.producer == "orion-harness-governor"


def test_gap_hop_ingress_deferred_to_slice_c():
    hop = gap_hop(
        correlation_id="c1",
        seq=0,
        stage="ingress",
        deferred_to="slice_c",
    )
    assert hop.status == "gap"
    assert hop.summary["deferred_to"] == "slice_c"
    assert "slice_c" in hop.visor_line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/cockpit/tests/test_builders.py::test_hop_from_association_ok orion/cockpit/tests/test_builders.py::test_hop_from_stance_inputs_ok orion/cockpit/tests/test_builders.py::test_hop_from_motor_boot_carries_exact_prompt orion/cockpit/tests/test_builders.py::test_gap_hop_ingress_deferred_to_slice_c -v`  
Expected: FAIL (import / function not defined)

- [ ] **Step 3: Write minimal implementation**

Create `orion/cockpit/markers.py`:

```python
COCKPIT_MOTOR_BOOT_MARKER = "cockpit.motor_boot.v1"
```

In `orion/cockpit/builders.py`, add (keep existing helpers; extend `_base_hop` with optional `producer` override):

```python
def _base_hop(
    *,
    correlation_id: str,
    seq: int,
    stage: CockpitStageV1,
    visor_line: str,
    status: str,
    summary: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    producer: str = _HUB_PRODUCER,
) -> CockpitHopV1:
    return CockpitHopV1(
        correlation_id=correlation_id,
        seq=seq,
        stage=stage,
        visor_line=visor_line,
        status=status,  # type: ignore[arg-type]
        summary=summary or {},
        raw=raw or {},
        producer=producer,
    )


def hop_from_association(
    *,
    correlation_id: str,
    seq: int,
    association: dict[str, Any],
) -> CockpitHopV1:
    stale = bool(association.get("broadcast_stale"))
    label = "stale" if stale else "fresh"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="association",
        visor_line=f"association · {label}",
        status="ok",
        summary={
            "broadcast_stale": stale,
            "read_source": association.get("read_source"),
        },
        raw=dict(association),
    )


def hop_from_stance_inputs(
    *,
    correlation_id: str,
    seq: int,
    stance_inputs: dict[str, Any],
) -> CockpitHopV1:
    user_message = str(stance_inputs.get("user_message") or "")
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="stance_inputs",
        visor_line=f"stance inputs · {len(user_message)} chars",
        status="ok",
        summary={"user_message_len": len(user_message)},
        raw=dict(stance_inputs),
    )


def hop_from_motor_boot(
    *,
    correlation_id: str,
    seq: int,
    prompt: str,
    producer: str = "orion-harness-governor",
) -> CockpitHopV1:
    text = prompt if isinstance(prompt, str) else ""
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="motor_boot",
        visor_line=f"motor_boot · {len(text)} chars",
        status="ok",
        summary={"prompt_char_len": len(text)},
        raw={"prompt": text, "prompt_char_len": len(text)},
        producer=producer,
    )
```

Leave `gap_hop` default `deferred_to="slice_b"`; ingress callers pass `"slice_c"` explicitly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/cockpit/tests/test_builders.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cockpit/markers.py orion/cockpit/builders.py orion/cockpit/tests/test_builders.py
git commit -m "$(cat <<'EOF'
feat(cockpit): builders for association, stance_inputs, motor_boot hops

Slice B thick-input hop constructors; ingress gaps defer to slice_c.
EOF
)"
```

---

### Task 2: Pre-motor emit — real association + stance_inputs; ingress gap only

**Files:**
- Modify: `orion/hub/cockpit_emit.py`
- Modify: `orion/hub/turn_orchestrator.py` (call site only — signature + args)
- Modify: `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py`

**Interfaces:**
- Consumes: Task 1 builders; `reset_seq` / `next_seq`; `hop_from_thought`
- Produces:
  - `emit_pre_motor_hops(correlation_id, thought, *, association: dict, stance_inputs: dict) -> list[dict]`  
    (rename from `emit_slice_a_pre_motor_hops`; keep a thin alias that raises or redirects during migration — prefer rename + update all call sites in the same task)
  - Stages: `ingress`(gap) → `association`(ok) → `stance_inputs`(ok) → `stance_decision`(ok)
  - **Does not** emit `motor_boot` (harness drain owns that)

- [ ] **Step 1: Write the failing test updates**

Replace `test_orchestrator_emits_gap_then_stance_cockpit_hops` body expectations:

```python
def test_orchestrator_emits_thick_pre_motor_hops():
    from orion.hub.cockpit_emit import emit_pre_motor_hops

    frames = emit_pre_motor_hops(
        "corr-1",
        _THOUGHT,
        association={
            "schema_version": "hub.association.bundle.v1",
            "correlation_id": "corr-1",
            "broadcast_stale": True,
            "broadcast": None,
            "execution_trajectory_slice": None,
            "repair_bundle": None,
            "read_source": "felt_state_reader",
        },
        stance_inputs={
            "user_message": "hi",
            "session_id": None,
            "llm_profile": "brain",
            "stance_inputs": {"user_message": "hi"},
        },
    )
    hops = [f["hop"] for f in frames if f["kind"] == "cockpit_hop"]
    stages = [h["stage"] for h in hops]
    assert stages == ["ingress", "association", "stance_inputs", "stance_decision"]
    assert hops[0]["status"] == "gap"
    assert hops[0]["summary"]["deferred_to"] == "slice_c"
    assert hops[1]["status"] == "ok"
    assert hops[1]["raw"]["broadcast_stale"] is True
    assert hops[2]["status"] == "ok"
    assert hops[2]["raw"]["user_message"] == "hi"
    assert hops[3]["stage"] == "stance_decision"
    assert hops[3]["seq"] == 3
```

Update every other test in this file that calls `emit_slice_a_pre_motor_hops` to call `emit_pre_motor_hops` with the two new kwargs (can use empty/`{}` minimal dicts when the test only cares about seq). Recalculate expected `seq` for motor/finalize tests: after pre-motor, next seq is **4** (was 5 when motor_boot was a gap).

Example seq after Task 2 pre-motor (4 hops, seq 0..3): first motor_hop → seq 4; if motor_boot from Task 3 lands first → motor_boot seq 4, motor_hop seq 5.

For tests that only simulate motor_hop (no motor_boot yet), expect motor_hop `seq == 4`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py::test_orchestrator_emits_thick_pre_motor_hops -v`  
Expected: FAIL (`emit_pre_motor_hops` missing or wrong stages)

- [ ] **Step 3: Write minimal implementation**

`orion/hub/cockpit_emit.py`:

```python
from orion.cockpit.builders import (
    gap_hop,
    hop_from_association,
    hop_from_closure,
    hop_from_motor_step,
    hop_from_outcome,
    hop_from_run_artifact,
    hop_from_stance_inputs,
    hop_from_thought,
)

# Delete PRE_MOTOR_GAP_STAGES and emit_slice_a_pre_motor_hops.


def emit_pre_motor_hops(
    correlation_id: str,
    thought: dict[str, Any],
    *,
    association: dict[str, Any],
    stance_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    reset_seq(correlation_id)
    frames: list[dict[str, Any]] = []
    frames.append(
        _hop_frame(
            correlation_id,
            gap_hop(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                stage="ingress",
                deferred_to="slice_c",
            ),
        )
    )
    frames.append(
        _hop_frame(
            correlation_id,
            hop_from_association(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                association=association if isinstance(association, dict) else {},
            ),
        )
    )
    frames.append(
        _hop_frame(
            correlation_id,
            hop_from_stance_inputs(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                stance_inputs=stance_inputs if isinstance(stance_inputs, dict) else {},
            ),
        )
    )
    frames.append(
        _hop_frame(
            correlation_id,
            hop_from_thought(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                thought=thought,
            ),
        )
    )
    return frames


# Delete PRE_MOTOR_GAP_STAGES and emit_slice_a_pre_motor_hops.
```

In `turn_orchestrator.py`, change `_emit_pre_motor` to pass dumps:

```python
await _deliver_cockpit_frames(
    emit_pre_motor_hops(
        correlation_id,
        _thought_as_cockpit_dict(
            thought_obj,
            fallback_disposition=disposition,
            fallback_reasons=reasons,
        ),
        association=(
            association.model_dump(mode="json")
            if hasattr(association, "model_dump")
            else dict(association or {})
        ),
        stance_inputs={
            "user_message": user_message,
            "session_id": session_id,
            "llm_profile": getattr(stance_req, "llm_profile", "brain"),
            "stance_inputs": dict(stance_req.stance_inputs or {}),
        },
    ),
    bus=bus,
    cockpit_sink=cockpit_sink,
)
```

Ensure `stance_req` is in scope at every `_emit_pre_motor` call (it is built before react). If `_emit_pre_motor` is nested and `stance_req` is always defined before first call, capture it in the closure — do not rebuild a thinner dict later.

Also update imports: `emit_pre_motor_hops` instead of `emit_slice_a_pre_motor_hops`.

`rg "emit_slice_a_pre_motor_hops"` and fix every hit in the same commit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py orion/cockpit/tests/test_builders.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/hub/cockpit_emit.py orion/hub/turn_orchestrator.py services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py
git commit -m "$(cat <<'EOF'
feat(hub): emit real association and stance_inputs cockpit hops

Replace Slice A gaps; ingress remains an explicit slice_c gap.
EOF
)"
```

---

### Task 3: Harness publishes motor_boot step; Hub drain converts it

**Files:**
- Modify: `orion/hub/cockpit_emit.py` (`emit_motor_hop_from_claude_step` + export marker)
- Modify: `orion/harness/runner.py`
- Create: `orion/harness/tests/test_harness_runner_motor_boot_cockpit.py`
- Modify: `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py`

**Interfaces:**
- Consumes: `build_harness_prompt` return value; `publish_harness_run_step`; `hop_from_motor_boot`; `COCKPIT_MOTOR_BOOT_MARKER`
- Produces: one `harness.run.step.v1` with `step_index=-1` and marker **before** FCC loop; Hub emits `motor_boot` cockpit hop with Hub-assigned `seq`

- [ ] **Step 1: Write the failing tests**

Create `orion/harness/tests/test_harness_runner_motor_boot_cockpit.py`:

```python
from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


@pytest.mark.asyncio
async def test_harness_runner_publishes_motor_boot_step_with_exact_prompt() -> None:
    thought = make_thought(imperative="Check logs first.", tone="direct")
    captured_prompt: dict[str, str] = {}
    captured_steps: list[dict[str, Any]] = []

    async def _capture_prompt(*, prompt: str, **__: Any) -> AsyncIterator[dict[str, Any]]:
        captured_prompt["prompt"] = prompt
        yield {"type": "final", "llm_response": "done", "metadata": {"exit_code": 0}}

    async def capture_step(
        bus: Any,
        *,
        correlation_id: str,
        step_index: int,
        step: dict[str, Any],
        channel: str,
        source_name: str = "orion-harness-governor",
    ) -> None:
        captured_steps.append(
            {
                "correlation_id": correlation_id,
                "step_index": step_index,
                "step": step,
                "channel": channel,
                "source_name": source_name,
            }
        )

    request = HarnessRunRequestV1(
        correlation_id="c-motor-boot",
        thought_event=thought,
        user_message="what broke?",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    runner = HarnessRunner(AsyncMock(), fcc_runner=_capture_prompt)

    with (
        patch("orion.harness.runner.publish_harness_run_step", capture_step),
        patch("orion.harness.runner.read_last_tool_fetch", AsyncMock(return_value=None)),
    ):
        await runner.run(request)

    boot_steps = [
        s
        for s in captured_steps
        if isinstance(s.get("step"), dict)
        and s["step"].get("_cockpit") == COCKPIT_MOTOR_BOOT_MARKER
    ]
    assert len(boot_steps) == 1
    boot = boot_steps[0]
    assert boot["step_index"] == -1
    assert boot["correlation_id"] == "c-motor-boot"
    assert boot["step"]["prompt"] == captured_prompt["prompt"]
    assert boot["step"]["prompt_char_len"] == len(captured_prompt["prompt"])
    assert boot["step"]["prompt_char_len"] > 0
```

Hub drain test — add to `test_turn_orchestrator_cockpit_hops.py`:

```python
def test_motor_boot_from_drained_claude_step_before_motor_hop():
    from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_pre_motor_hops,
    )

    emit_pre_motor_hops(
        "corr-boot",
        _THOUGHT,
        association={"broadcast_stale": True, "read_source": "felt_state_reader"},
        stance_inputs={"user_message": "hi", "stance_inputs": {"user_message": "hi"}},
    )
    boot = emit_motor_hop_from_claude_step(
        "corr-boot",
        {
            "kind": "claude_step",
            "step_index": -1,
            "step": {
                "_cockpit": COCKPIT_MOTOR_BOOT_MARKER,
                "prompt": "EXACT PREFIX\nUSER: hi",
                "prompt_char_len": len("EXACT PREFIX\nUSER: hi"),
            },
        },
    )
    assert boot is not None
    assert boot["hop"]["stage"] == "motor_boot"
    assert boot["hop"]["raw"]["prompt"] == "EXACT PREFIX\nUSER: hi"
    assert boot["hop"]["seq"] == 4
    assert boot["hop"]["producer"] == "orion-harness-governor"

    hop = emit_motor_hop_from_claude_step(
        "corr-boot",
        {
            "kind": "claude_step",
            "step_index": 0,
            "step": {"type": "tool_use", "name": "Read"},
        },
    )
    assert hop["hop"]["stage"] == "motor_hop"
    assert hop["hop"]["seq"] == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py::test_motor_boot_from_drained_claude_step_before_motor_hop -v`  
Expected: FAIL (marker treated as motor_hop or AttributeError)

- [ ] **Step 3: Implement Hub drain recognition**

In `orion/hub/cockpit_emit.py` (import marker from `orion.cockpit.markers`):

```python
from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER


def emit_motor_hop_from_claude_step(
    correlation_id: str,
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if item.get("kind") != "claude_step":
        return None
    step = item.get("step")
    if not isinstance(step, dict):
        step = {}
    if step.get("_cockpit") == COCKPIT_MOTOR_BOOT_MARKER:
        prompt = step.get("prompt")
        hop = hop_from_motor_boot(
            correlation_id=correlation_id,
            seq=next_seq(correlation_id),
            prompt=prompt if isinstance(prompt, str) else "",
        )
        return _hop_frame(correlation_id, hop)
    hop = hop_from_motor_step(
        correlation_id=correlation_id,
        seq=next_seq(correlation_id),
        step_index=int(item.get("step_index") or 0),
        step=step,
    )
    return _hop_frame(correlation_id, hop)
```

- [ ] **Step 4: Implement harness publish after prompt assembly**

In `orion/harness/runner.py`, add import at module top:

```python
from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
```

Immediately after `prompt = build_harness_prompt(...)`:

```python
        try:
            await publish_harness_run_step(
                self.bus,
                correlation_id=request.correlation_id,
                step_index=-1,
                step={
                    "_cockpit": COCKPIT_MOTOR_BOOT_MARKER,
                    "prompt": prompt,
                    "prompt_char_len": len(prompt),
                },
                channel=self.step_channel,
            )
        except Exception:
            logger.warning(
                "harness motor_boot cockpit step publish failed corr=%s",
                request.correlation_id,
                exc_info=True,
            )
```

Fail-open: motor publish failure must not abort the FCC turn.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py orion/harness/tests/test_harness_runner_motor_boot_cockpit.py orion/cockpit/tests -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/hub/cockpit_emit.py orion/harness/runner.py orion/harness/tests/test_harness_runner_motor_boot_cockpit.py services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py
git commit -m "$(cat <<'EOF'
feat(harness): publish exact motor_boot prompt onto cockpit drain

Hub converts the marked harness step into a motor_boot sighting hop.
EOF
)"
```

---

### Task 4: Soft HUD inspector Prompt/Prefix section

**Files:**
- Modify: `services/orion-hub/static/js/cockpit-hud.js`
- Modify: `services/orion-hub/tests/test_cockpit_hud_ui.py`
- Modify: `services/orion-hub/static/css/cockpit-hud.css` (only if a new class needs Soft HUD styling)

**Interfaces:**
- Consumes: hop.raw.prompt string when present
- Produces: inspector HTML with a labeled Prompt/Prefix block (copyable via existing raw JSON still present)

- [ ] **Step 1: Write the failing UI test**

Add to `services/orion-hub/tests/test_cockpit_hud_ui.py`:

```python
def test_inspector_shows_prompt_section_for_motor_boot():
    html = _node_call(
        "renderFixture",
        {
            "hops": [
                {
                    "seq": 4,
                    "stage": "motor_boot",
                    "status": "ok",
                    "visor_line": "motor_boot · 12 chars",
                    "summary": {"prompt_char_len": 12},
                    "raw": {"prompt": "HELLO PREFIX", "prompt_char_len": 12},
                },
            ],
            "selectedSeq": 4,
        },
    )
    assert 'data-cockpit-section="prompt"' in html or "Prompt/Prefix" in html
    assert "HELLO PREFIX" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-hub/tests/test_cockpit_hud_ui.py::test_inspector_shows_prompt_section_for_motor_boot -v`  
Expected: FAIL (section missing)

- [ ] **Step 3: Minimal JS change**

In `cockpit-hud.js` inspector render (where `raw` is stringified today), before the raw `<pre>`:

```javascript
    const promptText = (raw && typeof raw.prompt === 'string') ? raw.prompt : '';
    const promptSection = promptText
      ? (
          '<div class="cockpit-inspector-section" data-cockpit-section="prompt">' +
            '<div class="cockpit-inspector-section-title">Prompt/Prefix</div>' +
            '<pre class="cockpit-inspector-prompt">' + escapeText(promptText) + '</pre>' +
          '</div>'
        )
      : '';
```

Insert `promptSection` into the inspector HTML assembly between summary and full raw JSON. Keep the full raw JSON block — do not drop `prompt` from raw.

Optional CSS (Soft HUD): frosted panel already applies; add `.cockpit-inspector-prompt { max-height: 40vh; overflow: auto; white-space: pre-wrap; }` if needed for readability.

- [ ] **Step 4: Run UI tests**

Run: `pytest services/orion-hub/tests/test_cockpit_hud_ui.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/static/js/cockpit-hud.js services/orion-hub/static/css/cockpit-hud.css services/orion-hub/tests/test_cockpit_hud_ui.py
git commit -m "$(cat <<'EOF'
feat(hub): show motor_boot Prompt/Prefix in Soft HUD inspector

EOF
)"
```

---

### Task 5: Docs, gates, Slice C stub

**Files:**
- Modify: `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md` (status line only)
- Delete or replace: `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b-stub.md`
- Create: `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-c-stub.md`
- Mark this plan’s tasks complete in the PR description (checkboxes as you go)

- [ ] **Step 1: Confirm Slice C stub + design status; commit plan docs with the branch**

Confirm these already exist from writing-plans (create/fix if missing):

- `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-c-stub.md`
- Design status line mentions Slice B planned / C stubbed
- `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b-stub.md` is gone (superseded)

Slice C stub body (reference):

```markdown
# Orion Cockpit POV — Slice C follow-on stub

**Spec:** `docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md`
**Slice B plan:** `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b.md`

Invoke **writing-plans** before implementation.

## Remaining thickness

| Stage | Deferred work | Likely capture site |
|-------|---------------|---------------------|
| **ingress** | User message, attachments, observation molecule as a real hop (today: gap) | `orion/hub/turn_orchestrator.py` ingress / `emit_observation` |
| **closure** extras | Any post-motor side-effects beyond outcome/closure already emitted in Slice A | `orion/harness/finalize.py`, offboarding paths |
| **stance_inputs** enrichment | Optional: mind_coloring / Thought-assembled `build_stance_react_context` echo if Hub-sent dict proves too thin live | `services/orion-thought/app/bus_listener.py` |

**Acceptance:** every canonical stage has a real hop or an owned gap with no silent omission; live + rewind still work.
```

Update design status line to: `Status: Slice A landed; Slice B planned; C stubbed`.

- [ ] **Step 2: Run focused gates**

```bash
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
pytest orion/cockpit/tests -q
pytest orion/harness/tests/test_harness_runner_motor_boot_cockpit.py -q
pytest services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py services/orion-hub/tests/test_chat_cockpit_routes.py services/orion-hub/tests/test_cockpit_hud_ui.py services/orion-hub/tests/test_turn_trace_panel_ui.py -q
```

Expected: all PASS.  
Note: `test_chat_cockpit_routes.py` previously asserted `"motor_boot" in body["gaps"]` for empty timelines — empty still lists canonical stages missing from hops; a timeline that includes a real motor_boot must **not** list it in `gaps`. Update that test if it uses a fixture hop list.

- [ ] **Step 3: safe graphify update once**

```bash
scripts/safe_graphify_update.sh
```

- [ ] **Step 4: Commit docs**

```bash
git add docs/superpowers/specs/2026-09-06-orion-cockpit-pov-design.md \
  docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b.md \
  docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-c-stub.md
git add -u docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-b-stub.md
git commit -m "$(cat <<'EOF'
docs(cockpit): Slice B thick-inputs plan; stub Slice C

EOF
)"
```

- [ ] **Step 5: Manual smoke (operator, after deploy)**

1. Orion-mode Hub chat turn  
2. Open Cockpit mid-turn — association / stance_inputs / stance_decision appear; motor_boot arrives when motor starts  
3. After complete, scrub to `motor_boot` — Prompt/Prefix shows real system/prefix text  
4. Confirm `ingress` still shows as gap bead  
5. Turn Trace still opens  

Restart (print for Juniper, do not sudo):

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# sql-writer only if schema/table changed (Slice B should not need it)
```

---

## Spec coverage self-check

| Spec Slice B requirement | Task |
|--------------------------|------|
| association reads as real hop | 1–2 |
| stance_inputs full Hub-fed bundle | 1–2 |
| motor_boot exact prefix at assembly site | 1, 3 |
| Soft HUD inspector shows prefix | 4 |
| Gaps honest for deferred (ingress → C) | 2, 5 |
| Live + rewind still work | 2–3 (same WS/API) |
| No fabricate / no empty-shell | 2–3 fail-open |

## Placeholder scan

None intentional. Task 3/4 tests are fully specified. Gap hops remain the only honest deferral (ingress → Slice C).

## Type consistency

- Marker: `COCKPIT_MOTOR_BOOT_MARKER = "cockpit.motor_boot.v1"` in `orion/cockpit/markers.py`
- Emit rename: `emit_pre_motor_hops` (not `emit_slice_a_pre_motor_hops`)
- motor_boot producer: `orion-harness-governor`
- association / stance_inputs producer: `orion-hub`
- WS kind unchanged: `cockpit_hop` / `cockpit_timeline_complete`

## Follow-on

- **Slice C:** `docs/superpowers/plans/2026-09-06-orion-cockpit-pov-slice-c-stub.md` — invoke writing-plans before coding
- Optional later: page fat `raw` via `GET .../cockpit/hops/{seq}` if motor_boot WS payloads stress browsers (spec risk mitigation); not required to close Slice B if smoke is fine
