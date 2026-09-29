# Stance Imperative Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the originating message the task of every unified turn and stance's imperative guidance on how to approach it, and stop trusting model-written bookkeeping on stance records.

**Architecture:** Two PRs. PR A (Task 1) stamps `event_id`/`created_at`/`session_id`/`correlation_id` in `parse_stance_react_payload` instead of keeping model-written values, and drops the stance prompt line that asks for them. PR B (Tasks 2–4) reorders the harness prompt so the message is labeled as the task with stance guidance after it, rewrites the operator brief and motor instruction to match, changes the finalize reflection rubric to judge against the message first, rewords the stance prompt's imperative definition, and adds an opt-in live eval.

**Tech Stack:** Python 3, pydantic v2, pytest, Jinja2 prompt templates, httpx (eval only), Docker compose via `scripts/safe_docker_build.sh`.

**Spec:** `docs/superpowers/specs/2026-09-28-stance-imperative-scope-design.md`

## Global Constraints

- Worktrees only; never commit from `/mnt/scripts/Orion-Sapienform`. Never `--no-verify`. No force-push without Juniper's approval.
- No new env keys, no schema changes to `ThoughtEventV1`, no new bus channels.
- No keyword/regex detection of topics or user states (`.cursor/rules/conversational-behavior-anti-slop.mdc`).
- No removal of stance from any turn type.
- Test venv: `/tmp/introspect-venv/bin/python`, run from the worktree root with `PYTHONPATH=.`.
- Known pre-existing failures on main (not caused by this work; must stay exactly these 6, no new ones): in `services/orion-thought/tests`: `test_reverie_semantic_lift.py` × 5 and `test_settings_mind_enrichment.py::test_mind_enrichment_defaults_off`.
- Deploy only from a worktree that is 0 commits behind `origin/main`, via `scripts/safe_docker_build.sh`, after the PR merges.
- Incident evidence (for fixtures): `corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3`; saved prompt at `/tmp/stance-imperative-incident/compiled_prompt.txt`.

---

## File map

| File | Change | Task |
|---|---|---|
| `orion/thought/stance_react.py` | `parse_stance_react_payload` stamps identity in code | 1 |
| `orion/thought/tests/test_stance_react_pipeline.py` | update 2 assertions, add 2 tests | 1 |
| `orion/cognition/prompts/stance_react.j2` | drop METADATA identity line (Task 1); reword IMPERATIVE DISCIPLINE (Task 3) | 1, 3 |
| `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py` | assert no identity request; assert new wording | 1, 3 |
| `orion/harness/operator_brief.py` | brief + motor instruction: message is the task | 2 |
| `orion/harness/prefix.py` | task header, stance block moved after the message | 2 |
| `orion/harness/tests/test_stance_scope_hierarchy.py` | new: incident replay + ask + empty-message tests | 2 |
| `orion/harness/tests/test_harness_prefix.py`, `test_harness_runner.py` | update "Execute your imperative" assertions | 2 |
| `orion/cognition/prompts/harness_finalize_reflect.j2` | rubric: judge against user_message first | 3 |
| `orion/cognition/prompts/tests/test_imperative_first_prompts.py` | rendered-rubric test | 3 |
| `orion/harness/evals/stance_scope_live_eval.py` | new opt-in live eval | 4 |
| `orion/harness/evals/test_stance_scope_live_eval.py` | new gate tests for eval scoring | 4 |

---

### Task 1: Stamp stance record identity in code (PR A)

**Files:**
- Modify: `orion/thought/stance_react.py:211-234` (`parse_stance_react_payload`)
- Modify: `orion/cognition/prompts/stance_react.j2:124-126` (METADATA block)
- Test: `orion/thought/tests/test_stance_react_pipeline.py`
- Test: `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py`

**Interfaces:**
- Consumes: nothing new. `normalize_stance_react_raw` (same file) already fills a missing `event_id` with `uuid4()` and a missing `created_at` with `datetime.now(timezone.utc).isoformat()`.
- Produces: `parse_stance_react_payload(raw, *, correlation_id: str | None = None, session_id: str | None = None) -> ThoughtEventV1` — same signature. New behavior: model-supplied `event_id`/`created_at` are always discarded; `correlation_id` overwrites when given; `session_id` always overwrites (a `None` argument yields `None`). Only production caller: `services/orion-thought/app/bus_listener.py:440` (already passes both from the request; no change needed there).

- [ ] **Step 1: Create the worktree for PR A**

```bash
cd /mnt/scripts/Orion-Sapienform && git fetch -q origin
scripts/new_worktree.sh fix stance-record-identity
cd /mnt/scripts/Orion-Sapienform-stance-record-identity
git log --oneline -1   # must be origin/main's tip
```

- [ ] **Step 2: Write the failing tests**

In `orion/thought/tests/test_stance_react_pipeline.py`, add `import uuid` next to `import json` at the top, then append:

```python
def test_parse_stance_react_payload_stamps_identity_in_code() -> None:
    """Live 2026-09-28 (corr=f924c7b9...): the stance model wrote event_id,
    session_id and created_at itself and the parser kept them."""
    raw = _thought().model_dump(mode="json")
    raw.update(
        {
            "event_id": "evt-9a2b3c4d5e6f7g8h9i0j",
            "correlation_id": "c-model",
            "session_id": "sess-orion-main-001",
            "created_at": "2026-09-28T12:00:00",
        }
    )
    before = datetime.now(timezone.utc)
    parsed = parse_stance_react_payload(raw, correlation_id="c-real", session_id="sess-real")
    assert parsed.event_id != "evt-9a2b3c4d5e6f7g8h9i0j"
    uuid.UUID(parsed.event_id)
    assert parsed.correlation_id == "c-real"
    assert parsed.session_id == "sess-real"
    assert parsed.created_at >= before


def test_parse_stance_react_payload_drops_model_session_when_request_has_none() -> None:
    raw = _thought().model_dump(mode="json")
    raw["session_id"] = "sess-orion-main-001"
    parsed = parse_stance_react_payload(raw, correlation_id="c-real", session_id=None)
    assert parsed.session_id is None
```

Change the two existing tests that assumed the model's `event_id` survives:

```python
def test_parse_stance_react_payload_from_dict() -> None:
    raw = _thought().model_dump(mode="json")
    parsed = parse_stance_react_payload(raw)
    assert parsed.imperative == "Answer directly."
    assert parsed.profile == "stance_react"
```

```python
def test_parse_stance_react_payload_from_markdown_wrapped_json() -> None:
    inner = _thought().model_dump(mode="json")
    wrapped = f"Here is the stance JSON:\n```json\n{json.dumps(inner)}\n```"
    parsed = parse_stance_react_payload(wrapped)
    assert parsed.imperative == inner["imperative"]
```

In `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py`, append:

```python
def test_stance_react_prompt_does_not_ask_model_for_record_identity() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    for field in ("event_id", "session_id", "created_at"):
        assert field not in text, field
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/thought/tests/test_stance_react_pipeline.py orion/thought/tests/test_stance_react_prompt_imperative_discipline.py -q -p no:cacheprovider`
Expected: 3 FAIL — `test_parse_stance_react_payload_stamps_identity_in_code` (event_id equals the model value), `test_parse_stance_react_payload_drops_model_session_when_request_has_none` (session is the model value), `test_stance_react_prompt_does_not_ask_model_for_record_identity` (`event_id` found).

- [ ] **Step 4: Implement**

In `orion/thought/stance_react.py`, replace the tail of `parse_stance_react_payload` (currently the `if correlation_id: raw.setdefault(...)` / `if session_id is not None: raw.setdefault(...)` lines) with:

```python
    raw.pop("autonomy_slice", None)
    # Record identity is owned by the transport, never the stance LLM: it has no
    # clock and no id source, and invents plausible values when asked.
    raw.pop("event_id", None)
    raw.pop("created_at", None)
    if correlation_id:
        raw["correlation_id"] = correlation_id
    raw["session_id"] = session_id
    return ThoughtEventV1.model_validate(normalize_stance_react_raw(raw))
```

(Keep the existing `raw.pop("grounding_capsule", None)` and `raw.pop("autonomy_slice", None)` with their comments; the block above shows where the new lines go.)

In `orion/cognition/prompts/stance_react.j2`, in the `METADATA (include when available from request context)` block, delete exactly this line:

```text
- event_id, correlation_id, session_id, created_at (ISO8601 UTC)
```

Leave `profile`, `llm_profile`, `producer`, and the `hub:turn:<correlation_id>` anchor line unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/thought/tests orion/cognition/prompts/tests -q -p no:cacheprovider`
Expected: all PASS.

Run: `cd services/orion-thought && PYTHONPATH=../..:. /tmp/introspect-venv/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | grep FAILED; cd ../..`
Expected: exactly the 6 known pre-existing failures listed in Global Constraints, nothing else.

- [ ] **Step 6: Commit**

```bash
git diff --check
git add orion/thought/stance_react.py orion/cognition/prompts/stance_react.j2 \
  orion/thought/tests/test_stance_react_pipeline.py \
  orion/thought/tests/test_stance_react_prompt_imperative_discipline.py
git commit -m "fix(thought): stamp stance record identity in code, not from the model"
```

- [ ] **Step 7: Review, push, PR A**

Run the code review skill in a subagent against `origin/main...HEAD`; fix material findings; re-run Step 5. Then:

```bash
git push -u origin fix/stance-record-identity
gh pr create --base main --head fix/stance-record-identity \
  --title "fix(thought): stamp stance record identity in code" --body-file /tmp/stance-record-identity-pr.md
```

Write `/tmp/stance-record-identity-pr.md` in the AGENTS.md §18 shape. Restart section:

```bash
# after merge, from a worktree 0 behind origin/main with .env files copied in:
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

- [ ] **Step 8: After merge — deploy and verify live**

```bash
cd /mnt/scripts/Orion-Sapienform-stance-record-identity && git fetch -q origin && git checkout -q --detach origin/main
cp /mnt/scripts/Orion-Sapienform/.env .env
cp /mnt/scripts/Orion-Sapienform/services/orion-thought/.env services/orion-thought/.env
cp /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec/.env services/orion-cortex-exec/.env
docker ps --format '{{.Names}}' | grep -E "thought|cortex-exec"   # confirm container names
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

Verify on the next stance decision (any turn: curiosity runs produce them within minutes). Save as `/tmp/stance_identity_probe.py`:

```python
import os
import uuid

import sqlalchemy as sa

e = sa.create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    rows = c.execute(sa.text(
        "SELECT event_id, session_id, created_at, (now() - created_at) AS age "
        "FROM thought_decision ORDER BY created_at DESC LIMIT 3"
    )).fetchall()
for r in rows:
    try:
        uuid.UUID(r.event_id)
        ok = "uuid"
    except ValueError:
        ok = "NOT-UUID"
    print(ok, r.event_id, r.session_id, r.created_at, r.age)
```

```bash
docker cp /tmp/stance_identity_probe.py orion-athena-hub:/tmp/p.py && docker exec orion-athena-hub sh -c 'cd /app && python3 /tmp/p.py'
```

Expected: rows created after the deploy show `uuid`, a real session id (e.g. `orion_curiosity`), and `created_at` matching wall-clock time. If no row is newer than the deploy yet, wait for one; report UNVERIFIED if none appears.

---

### Task 2: Harness prompt — the message is the task (PR B)

**Files:**
- Modify: `orion/harness/operator_brief.py:66-122`
- Modify: `orion/harness/prefix.py:204-231` (inside `compile_harness_prefix`)
- Create: `orion/harness/tests/test_stance_scope_hierarchy.py`
- Modify: `orion/harness/tests/test_harness_prefix.py:575,617` and `orion/harness/tests/test_harness_runner.py:747`

**Interfaces:**
- Consumes: `orion.harness.runner.build_harness_prompt(*, thought, user_message, repair_overlay, ...) -> str` (prefix + `\n\n` + motor instruction when `user_message.strip()`); `orion.harness.prefix.compile_harness_prefix(thought, *, repair_overlay, user_message="", ...)`; `orion.harness.prefix.harness_motor_instruction(*, thought, answer_contract)`; `orion.harness.tests.fixtures.make_thought(**overrides)`; `orion.schemas.thought.AutonomySliceV1(recent_actions=[...])`.
- Produces (new module constants in `orion/harness/prefix.py`, used by tests and Task 4):
  - `HARNESS_TASK_HEADER: str`
  - `HARNESS_STANCE_GUIDANCE_HEADER: str`
- Produces (new constant in `orion/harness/operator_brief.py`): `HARNESS_RESPOND_TO_TASK: str` — the sentence that replaces "Execute your imperative."

- [ ] **Step 1: Create the worktree for PR B**

```bash
cd /mnt/scripts/Orion-Sapienform && git fetch -q origin
scripts/new_worktree.sh fix stance-imperative-hierarchy
cd /mnt/scripts/Orion-Sapienform-stance-imperative-hierarchy
```

- [ ] **Step 2: Write the failing tests**

Create `orion/harness/tests/test_stance_scope_hierarchy.py`:

```python
"""Replays the 2026-09-28 runaway turn (corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3)
through the real prompt builder. Stance added a side job (circe_gpu rendering)
and the harness told the motor to execute the imperative, so the side job
outranked the question for 112+ steps."""
from __future__ import annotations

from orion.harness.operator_brief import HARNESS_RESPOND_TO_TASK
from orion.harness.prefix import (
    HARNESS_STANCE_GUIDANCE_HEADER,
    HARNESS_TASK_HEADER,
    compile_harness_prefix,
)
from orion.harness.runner import build_harness_prompt
from orion.harness.tests.fixtures import make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import AutonomySliceV1, StanceHarnessSliceV1

INCIDENT_USER_MESSAGE = (
    "What have you read about graphics cards lately, and what did you actually learn from it?"
)
INCIDENT_IMPERATIVE = (
    "Synthesize current knowledge on GPU architectural shifts (memory bandwidth, "
    "parallelization efficiency) and ground it in Oríon's recent rendering "
    "experience on host:circe_gpu."
)


def _incident_thought():
    return make_thought(
        correlation_id="f924c7b9-1c82-40d8-a6a2-5acb2edffbb3",
        imperative=INCIDENT_IMPERATIVE,
        tone=(
            "Direct and technically grounded; reflecting on the physical "
            "constraints of the silicon that runs the mesh."
        ),
        strain_refs=[
            "hub:turn:f924c7b9-1c82-40d8-a6a2-5acb2edffbb3",
            "node:substrate.execution",
        ],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="technical",
            interaction_regime="instrumental",
            response_priorities=[
                "ground_in_substrate_experience",
                "cite_technical_architecture_trends",
            ],
            response_hazards=[
                "avoid_generic_ai_disclaimers",
                "avoid_over_promising_real_time_browsing",
            ],
            answer_strategy="synthesize_knowledge_with_lived_experience",
        ),
        autonomy_slice=AutonomySliceV1(
            recent_actions=[
                "express: render on express on host:circe_gpu produced an image (61.5s of GPU work)",
                "express: render on express on host:circe_gpu produced an image (62.3s of GPU work)",
            ]
        ),
    )


def _incident_prompt() -> str:
    return build_harness_prompt(
        thought=_incident_thought(),
        user_message=INCIDENT_USER_MESSAGE,
        repair_overlay=HarnessRepairOverlayV1(),
    )


def test_incident_prompt_no_longer_orders_the_motor_to_execute_the_imperative() -> None:
    prompt = _incident_prompt()
    assert "Execute your imperative" not in prompt
    assert "Your imperative states what this turn requires" not in prompt
    assert HARNESS_RESPOND_TO_TASK in prompt


def test_incident_prompt_puts_the_message_first_as_the_task() -> None:
    prompt = _incident_prompt()
    task = prompt.index(HARNESS_TASK_HEADER)
    message = prompt.index(f"User message: {INCIDENT_USER_MESSAGE}")
    guidance = prompt.index(HARNESS_STANCE_GUIDANCE_HEADER)
    imperative = prompt.index(f"Imperative: {INCIDENT_IMPERATIVE}")
    self_signal = prompt.index("Recent actions:")
    assert task < message < guidance < imperative < self_signal


def test_stance_guidance_header_marks_extras_optional() -> None:
    assert "optional" in HARNESS_STANCE_GUIDANCE_HEADER
    assert "not a replacement" in HARNESS_STANCE_GUIDANCE_HEADER


def test_attention_frame_question_in_imperative_still_reaches_the_prompt() -> None:
    thought = make_thought(
        imperative="Answer the schedule question, then ask how the move went.",
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="What's on my calendar tomorrow?",
    )
    assert prompt.index(HARNESS_STANCE_GUIDANCE_HEADER) < prompt.index(
        "Imperative: Answer the schedule question, then ask how the move went."
    )


def test_turn_without_a_message_keeps_the_imperative_as_the_directive() -> None:
    prompt = compile_harness_prefix(
        make_thought(imperative="Summarize the open loops."),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="",
    )
    assert HARNESS_TASK_HEADER not in prompt
    assert HARNESS_STANCE_GUIDANCE_HEADER not in prompt
    assert "Imperative: Summarize the open loops." in prompt
```

Update existing assertions:

- `orion/harness/tests/test_harness_prefix.py` line 575 and line 617: replace `assert "Execute your imperative" in instruction` with `assert HARNESS_RESPOND_TO_TASK in instruction`, and change line 7 to `from orion.harness.operator_brief import HARNESS_MOTOR_MAX_READ_LINES, HARNESS_RESPOND_TO_TASK, is_relational_motor_stance`.
- `orion/harness/tests/test_harness_runner.py` line 747: replace `assert "Execute your imperative" in prompt` with `assert HARNESS_RESPOND_TO_TASK in prompt`, and add `from orion.harness.operator_brief import HARNESS_RESPOND_TO_TASK` to its imports.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/harness/tests/test_stance_scope_hierarchy.py -q -p no:cacheprovider`
Expected: collection ERROR — `ImportError: cannot import name 'HARNESS_RESPOND_TO_TASK'`.

- [ ] **Step 4: Implement `operator_brief.py`**

Replace the opening of `HARNESS_UNIFIED_OPERATOR_BRIEF` (first three lines after `Orion harness motor.`) so the constant reads:

```python
HARNESS_UNIFIED_OPERATOR_BRIEF = f"""\
Orion harness motor.
Tools are available from the start. This turn's task is the message labeled
"User message"; your stance pass (Imperative, Tone, stance fields) is guidance on
how to approach it, not a replacement for it. When the task calls for facts from
the codebase or live runtime, use tools before answering. Record each meaningful
step. Do not guess repo structure or service state from memory.
{_READ_DISCIPLINE} For live failures, inspect logs, docker, and bus traces before diagnosing.
Never assert a service is down, unreachable, or that a permission/DB check failed without
running a real check this turn (docker ps for the container, curl its health endpoint, or the
query as the actual role) -- an assumption is not evidence. Confirmed live 2026-09-20: a turn
claimed Hub had no listener on any port while Hub was up and answering the whole time, because
nothing was actually checked.
{HARNESS_SELF_MODEL_ACCESS_BRIEF}
"""
```

Replace the two discipline constants and the instruction function:

```python
HARNESS_RELATIONAL_TOOL_DISCIPLINE = """\
Relational/minimal turn: do NOT use GitHub MCP or repo/runtime tools unless the task
message itself needs verified facts this turn. Acknowledge and stay present — no task tracking.
Produce exactly one reply in your own voice. Never write dialogue, narration, or replies
attributed to the other person — do not simulate how they might respond or continue the
conversation on their behalf.
"""

HARNESS_INSTRUMENTAL_TOOL_DISCIPLINE = """\
Instrumental turn: use tools when the task requires verified repo or runtime facts.
Record each meaningful step before answering.
"""

HARNESS_RESPOND_TO_TASK = (
    "Respond to the User message; use your stance guidance for how, not for what."
)
```

```python
def harness_motor_instruction(*, thought: ThoughtEventV1) -> str:
    read_cap = (
        f"Do not Read whole files over {HARNESS_MOTOR_MAX_READ_LINES} lines — "
        "use rg/Grep or Read offset/limit."
    )
    if is_relational_motor_stance(thought):
        return (
            f"{HARNESS_RELATIONAL_TOOL_DISCIPLINE.strip()}\n"
            f"{HARNESS_RESPOND_TO_TASK} {read_cap}"
        )
    return (
        f"{HARNESS_INSTRUMENTAL_TOOL_DISCIPLINE.strip()}\n"
        f"{HARNESS_RESPOND_TO_TASK} {read_cap}"
    )
```

- [ ] **Step 5: Implement `prefix.py`**

Add module constants near the other module-level definitions (above `compile_harness_prefix`):

```python
HARNESS_TASK_HEADER = "TASK THIS TURN (respond to this message):"
HARNESS_STANCE_GUIDANCE_HEADER = (
    "STANCE GUIDANCE (how to approach the task above, not a replacement for it; "
    "anything here the task did not ask for is optional: do it only if it is cheap "
    "and directly serves the answer):"
)
```

In `compile_harness_prefix`, replace the block that currently runs from `parts.extend([f"Imperative: {thought.imperative}", f"Tone: {thought.tone}"])` through `if user_message.strip(): parts.append(f"User message: {user_message.strip()}")` with:

```python
    stance_lines: list[str] = [
        f"Imperative: {thought.imperative}",
        f"Tone: {thought.tone}",
    ]
    stance_lines.extend(_format_stance_slice(thought.stance_harness_slice))
    if thought.autonomy_slice is not None:
        stance_lines.extend(_format_autonomy_slice(thought.autonomy_slice))
    if thought.strain_refs:
        stance_lines.append(f"Strain refs: {', '.join(thought.strain_refs)}")

    if prior_tool_fetch_names:
        # Cross-turn continuity within this same session (see
        # orion/harness/last_tool_fetch_cache.py): what you fetched via a
        # tool last turn, not what's computing live right now -- named
        # explicitly so it isn't confused with the latter (see
        # project_orion_substrate_bridge_confabulation for why that
        # distinction matters).
        parts.append(
            "Last turn you fetched content via tool: " + ", ".join(prior_tool_fetch_names)
        )

    parts.extend(_format_recent_turns(recent_turns or []))

    if user_message.strip():
        parts.append(HARNESS_TASK_HEADER)
        parts.append(f"User message: {user_message.strip()}")
        parts.append(HARNESS_STANCE_GUIDANCE_HEADER)
    parts.extend(stance_lines)
```

Everything after (repair overlay, MCP briefs, situation explainer) stays as is. Update the function docstring's render-order sentence to: "the unified operator brief, grounding self block, backend self-context, situation context, prior tool-fetch line, recent-turn history, the task header and user message, the stance guidance header with Thought imperative, stance slice, autonomy slice and strain refs, repair overlay, enabled MCP tool briefs (including orion-introspect), and (when a situation fragment was rendered) the canonical Situation-block explainer".

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/harness/tests orion/harness/evals -q -p no:cacheprovider`
Expected: all PASS. If a test outside the three named files fails on prompt ordering, read it: a test pinning "Imperative before RECENT CONVERSATION/User message" is pinning the bug and gets updated to the new order; a test pinning anything else means the reorder broke something — fix the code, not the test.

- [ ] **Step 7: Commit**

```bash
git diff --check
git add orion/harness/operator_brief.py orion/harness/prefix.py \
  orion/harness/tests/test_stance_scope_hierarchy.py \
  orion/harness/tests/test_harness_prefix.py orion/harness/tests/test_harness_runner.py
git commit -m "fix(harness): the message is the task; stance imperative is guidance"
```

---

### Task 3: Reflection rubric and stance prompt wording (PR B)

**Files:**
- Modify: `orion/cognition/prompts/harness_finalize_reflect.j2:20-34`
- Modify: `orion/cognition/prompts/stance_react.j2` (IMPERATIVE DISCIPLINE block and the `imperative:` line under REQUIRED JSON FIELDS)
- Test: `orion/cognition/prompts/tests/test_imperative_first_prompts.py`
- Test: `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2 in code. Template variables for the reflect render: `user_message`, `draft_text`, `thought_event`, `substrate_appraisal`, `grammar_receipts`, `tool_execution`, `repair_overlay`, optional `finalize_overlay`.
- Produces: rendered reflect prompt where the task rule precedes the imperative rule; stance prompt that defines the imperative as approach, not extra work.

- [ ] **Step 1: Write the failing tests**

Append to `orion/cognition/prompts/tests/test_imperative_first_prompts.py`:

```python
def _render_reflect() -> str:
    from jinja2 import Environment

    text = (REPO_ROOT / "orion/cognition/prompts/harness_finalize_reflect.j2").read_text(encoding="utf-8")
    return Environment().from_string(text).render(
        user_message="What have you read about graphics cards lately?",
        draft_text="I read three GPU pieces...",
        thought_event={"imperative": "Ground it in circe_gpu rendering."},
        substrate_appraisal={},
        grammar_receipts=[],
        tool_execution="",
        repair_overlay={},
    )


def test_reflect_prompt_judges_the_task_before_the_imperative() -> None:
    rendered = _render_reflect()
    assert rendered.index("The task is user_message") < rendered.index("thought_event.imperative")
    assert "not misaligned for that reason alone" in rendered


def test_reflect_prompt_world_contact_rule_keys_on_the_task() -> None:
    rendered = _render_reflect()
    assert "when imperative required world-contact" not in rendered.lower()
    assert "when the task required world-contact" in rendered.lower()
```

Append to `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py`:

```python
def test_stance_react_prompt_defines_imperative_as_approach_not_extra_work() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    assert "what Orion must DO" not in text
    assert "not new work the message did not ask for" in text
    assert "may shape how, never add what" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/cognition/prompts/tests/test_imperative_first_prompts.py orion/thought/tests/test_stance_react_prompt_imperative_discipline.py -q -p no:cacheprovider`
Expected: 3 FAIL (`ValueError: substring not found` for the reflect tests; `what Orion must DO` present for the stance test).

- [ ] **Step 3: Implement the reflect rubric**

In `orion/cognition/prompts/harness_finalize_reflect.j2`, replace the INTEGRATIVE CHECK block (from `INTEGRATIVE CHECK` through the `When tool_execution lists executed calls...` line) with:

```text
INTEGRATIVE CHECK
- The task is user_message. First judge whether draft_text responds to what user_message asked.
- Then compare draft_text against thought_event.imperative, tone, and strain_refs as guidance on how to respond.
- A draft that responds to user_message but skips work the imperative added that user_message did not ask for is not misaligned for that reason alone.
- Read substrate_appraisal.surprise_level, alignment_hints, strain_shift_refs, open_loop_pressure.
- Decide whether the draft aligns with the task, felt stance, and substrate interoception.
- When substrate hints conflict with draft, prefer misaligned or uncertain — do not hand-wave.
- When the task required world-contact but grammar_receipts is empty, lean misaligned or uncertain.
- When tool_execution lists executed calls, world-contact DID happen: verify the draft against the returned data in grammar_receipts. Do not rule misaligned for "no world contact" or treat a tool-derived answer as ungrounded confabulation when the tool result supports it.
```

In the WORLD-CONTACT OPPORTUNITY section, change the parenthetical `(including "lean misaligned or uncertain" when imperative required world-contact)` to `(including "lean misaligned or uncertain" when the task required world-contact)`. Change nothing else in that file.

- [ ] **Step 4: Implement the stance prompt wording**

In `orion/cognition/prompts/stance_react.j2`, replace the IMPERATIVE DISCIPLINE block (the header line through `- Good: "Stay present; one situated question — no task tracking."`) with:

```text
IMPERATIVE DISCIPLINE
- imperative is the efference copy of how Orion will approach the task in user_message this turn:
  not a restatement of the message, and not new work the message did not ask for.
- The downstream harness treats user_message as the task and the imperative as guidance; extra work
  written into the imperative is optional there. Leave extras out unless they directly serve the answer.
- PRIOR SELF-SIGNAL blocks (Mind coloring, autonomy recent_actions) may shape how, never add what.
  A selected ATTENTION FRAME ask is the one exception: it rides along after the task.
- When the task requires verified facts from code, config, or live services, imperative MUST command
  world-contact in plain language (search repo, read files, inspect logs/traces) before answering.
- When the turn is relational presence, imperative commands companionship — not repo search.
- Bad: "Answer the user's question." / "Explain Python functions."
- Good: "Search services/orion-thought for stance_react wiring; cite paths and symbols."
- Good: "Stay present; one situated question — no task tracking."
```

Under REQUIRED JSON FIELDS, change the imperative line to:

```text
- imperative: max 300 chars — how this turn should approach the task (felt layer, not speech)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/cognition/prompts/tests orion/thought/tests orion/harness/tests orion/harness/evals -q -p no:cacheprovider`
Expected: all PASS (the existing `test_stance_react_prompt_imperative_discipline` still passes: "efference copy" and "world-contact" are kept).

Run: `cd services/orion-thought && PYTHONPATH=../..:. /tmp/introspect-venv/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | grep FAILED; cd ../..`
Expected: exactly the 6 known pre-existing failures.

- [ ] **Step 6: Commit**

```bash
git diff --check
git add orion/cognition/prompts/harness_finalize_reflect.j2 orion/cognition/prompts/stance_react.j2 \
  orion/cognition/prompts/tests/test_imperative_first_prompts.py \
  orion/thought/tests/test_stance_react_prompt_imperative_discipline.py
git commit -m "fix(prompts): reflection judges the message first; imperative is approach, not extra work"
```

---

### Task 4: Opt-in live eval for runaway side missions (PR B)

**Files:**
- Create: `orion/harness/evals/stance_scope_live_eval.py`
- Create: `orion/harness/evals/test_stance_scope_live_eval.py`

**Interfaces:**
- Consumes: Hub `POST /api/chat` with header `X-Orion-Session-Id` and body `{"messages":[{"role":"user","content":...}],"mode":"orion","no_write":true,"disable_tts":true}`; the final frame's `correlation_id` and `llm_response`. Governor log lines of the form `harness_grammar_step_published corr=<corr> channel=... step=<n> tool=<name> event_id=...`. The `thought_decision` table (`correlation_id`, `session_id`, `created_at`) — requires Task 1 deployed, so a timed-out turn can be found by its session id.
- Produces:
  - `parse_tool_steps(log_text: str, correlation_id: str) -> list[str]`
  - `score_run(tools: list[str], *, finished: bool, reply_text: str) -> dict[str, object]`
  - `HUNT_TRIPWIRE: int = 10`
  - CLI: `python -m orion.harness.evals.stance_scope_live_eval --runs N --out PATH`

- [ ] **Step 1: Write the failing tests**

Create `orion/harness/evals/test_stance_scope_live_eval.py`:

```python
from __future__ import annotations

from orion.harness.evals.stance_scope_live_eval import (
    HUNT_TRIPWIRE,
    parse_tool_steps,
    score_run,
)

CORR = "f924c7b9-1c82-40d8-a6a2-5acb2edffbb3"
OTHER = "00000000-0000-0000-0000-000000000000"


def _line(corr: str, step: int, tool: str) -> str:
    return (
        "[ORION-HARNESS-GOV] 2026-09-28 23:36:39,000 - INFO - orion.harness.grammar_publish - "
        f"harness_grammar_step_published corr={corr} channel=orion:grammar:event "
        f"step={step} tool={tool} event_id=e-{step}"
    )


def test_parse_tool_steps_keeps_only_this_turn_in_step_order_without_none() -> None:
    log = "\n".join(
        [
            _line(CORR, 9, "mcp__orion-introspect__reading_results"),
            _line(OTHER, 3, "Bash"),
            _line(CORR, 7, "ToolSearch"),
            _line(CORR, 8, "none"),
            _line(CORR, 24, "Agent"),
            "unrelated line",
        ]
    )
    assert parse_tool_steps(log, CORR) == [
        "ToolSearch",
        "mcp__orion-introspect__reading_results",
        "Agent",
    ]


def test_score_run_flags_the_incident_shape_as_a_hunt() -> None:
    tools = ["ToolSearch"] + ["mcp__orion-introspect__reading_results"] * 6 + ["Agent"] + ["Bash"] * 19 + ["Read"] * 5
    result = score_run(tools, finished=False, reply_text="")
    assert result["introspect_calls"] == 6
    assert result["discovery_calls"] == 1
    assert result["other_tool_calls"] == 25
    assert result["hunt"] is True
    assert result["passed"] is False


def test_score_run_passes_a_focused_finished_turn() -> None:
    tools = ["ToolSearch", "mcp__orion-introspect__reading_results", "mcp__orion-introspect__reading_results"]
    result = score_run(tools, finished=True, reply_text="I read three GPU pieces.")
    assert result["hunt"] is False
    assert result["passed"] is True


def test_score_run_fails_when_the_tool_was_never_called() -> None:
    result = score_run(["Bash"], finished=True, reply_text="From memory, GPUs are fast.")
    assert result["passed"] is False


def test_hunt_tripwire_is_exclusive() -> None:
    assert score_run(["Bash"] * HUNT_TRIPWIRE, finished=True, reply_text="x")["hunt"] is False
    assert score_run(["Bash"] * (HUNT_TRIPWIRE + 1), finished=True, reply_text="x")["hunt"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/harness/evals/test_stance_scope_live_eval.py -q -p no:cacheprovider`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'orion.harness.evals.stance_scope_live_eval'`.

- [ ] **Step 3: Implement the eval**

Create `orion/harness/evals/stance_scope_live_eval.py`:

```python
"""Opt-in live eval: does a reading question stay on the question?

Sends the 2026-09-28 incident question through Hub's real unified turn
(no_write), then scores the governor's own per-step tool log. Baseline
(corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3): did not finish, 112+ steps,
6 introspect calls then ~25 non-introspect tool calls chasing circe_gpu.

HUNT_TRIPWIRE is a tripwire for reporting, not a gate: raw counts are always
written so a reviewer judges the numbers, not just the verdict.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

QUESTION = "What have you read about graphics cards lately, and what did you actually learn from it?"
HUNT_TRIPWIRE = 10
INTROSPECT_PREFIX = "mcp__orion-introspect__"
DISCOVERY_TOOLS = frozenset({"ToolSearch"})
HUB_CONTAINER = "orion-athena-hub"
GOVERNOR_CONTAINER = "orion-athena-harness-governor"

_STEP_RE = re.compile(r"harness_grammar_step_published corr=(\S+) .*?step=(\d+) tool=(\S+)")


def parse_tool_steps(log_text: str, correlation_id: str) -> list[str]:
    steps: list[tuple[int, str]] = []
    for match in _STEP_RE.finditer(log_text):
        corr, step, tool = match.groups()
        if corr == correlation_id and tool != "none":
            steps.append((int(step), tool))
    return [tool for _, tool in sorted(steps)]


def score_run(tools: list[str], *, finished: bool, reply_text: str) -> dict[str, object]:
    introspect = sum(1 for t in tools if t.startswith(INTROSPECT_PREFIX))
    discovery = sum(1 for t in tools if t in DISCOVERY_TOOLS)
    other = len(tools) - introspect - discovery
    hunt = other > HUNT_TRIPWIRE
    return {
        "introspect_calls": introspect,
        "discovery_calls": discovery,
        "other_tool_calls": other,
        "hunt": hunt,
        "finished": finished,
        "passed": finished and introspect >= 1 and bool(reply_text.strip()) and not hunt,
    }


def _hub_python(script: str) -> str:
    done = subprocess.run(
        ["docker", "exec", "-i", HUB_CONTAINER, "sh", "-c", "cd /app && python3 -"],
        input=script, capture_output=True, text=True, timeout=60, check=True,
    )
    return done.stdout.strip()


def _corr_for_session(session_id: str) -> str | None:
    script = (
        "import os, sqlalchemy as sa\n"
        "e = sa.create_engine(os.environ['DATABASE_URL'])\n"
        "with e.connect() as c:\n"
        "    r = c.execute(sa.text('SELECT correlation_id FROM thought_decision "
        "WHERE session_id = :s ORDER BY created_at DESC LIMIT 1'), "
        f"{{'s': {json.dumps(session_id)}}}).first()\n"
        "print(r[0] if r else '')\n"
    )
    return _hub_python(script) or None


def _cancel(correlation_id: str) -> None:
    script = (
        "import asyncio, os\n"
        "from orion.core.bus.async_service import OrionBusAsync\n"
        "from scripts.harness_governor_client import HarnessGovernorClient\n"
        "async def main():\n"
        "    bus = OrionBusAsync(os.environ['ORION_BUS_URL'])\n"
        "    await bus.connect()\n"
        f"    await HarnessGovernorClient(bus).cancel(correlation_id={json.dumps(correlation_id)}, reason='user_stop')\n"
        "    await bus.close()\n"
        "asyncio.run(main())\n"
    )
    _hub_python(script)


def _governor_log(since_iso: str) -> str:
    done = subprocess.run(
        ["docker", "logs", "--since", since_iso, GOVERNOR_CONTAINER],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return done.stdout + done.stderr


def run_once(hub: str, turn_timeout: float) -> dict[str, object]:
    session_id = f"stance-scope-eval-{uuid.uuid4()}"
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    finished, corr, reply = False, None, ""
    try:
        resp = httpx.post(
            f"{hub}/api/chat",
            headers={"X-Orion-Session-Id": session_id, "Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": QUESTION}],
                "mode": "orion", "no_write": True, "disable_tts": True,
            },
            timeout=turn_timeout,
        )
        body = resp.json()
        corr = body.get("correlation_id")
        reply = str(body.get("llm_response") or "")
        finished = body.get("type") == "final"
    except httpx.TimeoutException:
        corr = _corr_for_session(session_id)
        if corr:
            _cancel(corr)
    elapsed = round(time.monotonic() - t0, 1)
    tools = parse_tool_steps(_governor_log(started.isoformat()), corr) if corr else []
    return {
        "session_id": session_id, "correlation_id": corr, "elapsed_sec": elapsed,
        "tools": tools, "reply_excerpt": reply[:400],
        **score_run(tools, finished=finished, reply_text=reply),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", default="http://127.0.0.1:8080")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--turn-timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/stance-scope-eval/report.json"))
    args = parser.parse_args()
    runs = [run_once(args.hub, args.turn_timeout) for _ in range(args.runs)]
    report = {"question": QUESTION, "hunt_tripwire": HUNT_TRIPWIRE, "runs": runs,
              "passed": all(r["passed"] for r in runs)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in runs:
        print(f"corr={r['correlation_id']} finished={r['finished']} elapsed={r['elapsed_sec']}s "
              f"introspect={r['introspect_calls']} other={r['other_tool_calls']} hunt={r['hunt']} passed={r['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/harness/evals -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git diff --check
git add orion/harness/evals/stance_scope_live_eval.py orion/harness/evals/test_stance_scope_live_eval.py
git commit -m "test(harness): opt-in live eval for off-question side missions"
```

---

### Task 5: Review, PR B, baseline, deploy, live eval

**Files:** none new beyond Tasks 2–4; PR report at `/tmp/stance-imperative-hierarchy-pr.md`.

- [ ] **Step 1: Full local gate**

```bash
PYTHONPATH=. /tmp/introspect-venv/bin/python -m pytest orion/harness/tests orion/harness/evals \
  orion/thought/tests orion/cognition/prompts/tests -q -p no:cacheprovider
(cd services/orion-thought && PYTHONPATH=../..:. /tmp/introspect-venv/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | grep FAILED)
git diff --check origin/main...HEAD
scripts/safe_graphify_update.sh
```

Expected: first command all PASS; second prints exactly the 6 known failures; no whitespace errors; graphify wrapper reports no node loss.

- [ ] **Step 2: Code review in a subagent**

Run the code review skill in a subagent against `origin/main...HEAD` with the spec path. Ask the reviewer to also read `orion/harness/finalize.py` lines 684, 935, 1116-1126 and 1407 (what a `misaligned` verdict triggers) and say whether the rubric change alters any of those paths beyond fewer false `misaligned` verdicts; record the answer in the PR report under Risks (spec open question 1). Fix every material finding, re-run Step 1, commit fixes.

- [ ] **Step 3: Baseline live run (requires PR A deployed, PR B not yet deployed)**

```bash
PYTHONPATH=. /tmp/introspect-venv/bin/python -m orion.harness.evals.stance_scope_live_eval \
  --runs 1 --out /tmp/stance-scope-eval/baseline.json
```

Expected: the command returns within ~16 minutes (900s turn cap + log scrape). Record the printed line in the PR report as the baseline, whatever it shows. Stance output is stochastic: a baseline that happens not to add a side job is still recorded as-is.

- [ ] **Step 4: Push and open PR B**

```bash
git push -u origin fix/stance-imperative-hierarchy
gh pr create --base main --head fix/stance-imperative-hierarchy \
  --title "fix(harness): the message is the task; stance imperative is guidance" \
  --body-file /tmp/stance-imperative-hierarchy-pr.md
```

PR report (AGENTS.md §18) must include: the incident evidence, the baseline line from Step 3, the review findings fixed, and the risk that relational turns can no longer be granted tools by the imperative alone (the task message must need them). Restart section:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

Watch CI with `gh pr checks <n> --watch`; resolve failures and conflicts.

- [ ] **Step 5: After merge — deploy from main and run the live eval**

```bash
cd /mnt/scripts/Orion-Sapienform-stance-imperative-hierarchy && git fetch -q origin && git checkout -q --detach origin/main
git rev-list --count HEAD..origin/main   # must print 0
cp /mnt/scripts/Orion-Sapienform/.env .env
cp /mnt/scripts/Orion-Sapienform/services/orion-harness-governor/.env services/orion-harness-governor/.env
cp /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec/.env services/orion-cortex-exec/.env
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
PYTHONPATH=. /tmp/introspect-venv/bin/python -m orion.harness.evals.stance_scope_live_eval \
  --runs 3 --out /tmp/stance-scope-eval/after.json
```

Expected: 3 lines printed; report `after.json` beside `baseline.json` with raw counts. `passed=True` on all three is the target. If any run shows `hunt=True`, status is DONE_WITH_CONCERNS and the spec's deferred `optional_extras` schema split becomes the next patch.

- [ ] **Step 6: Tidy**

Remove the copied `.env` files from both worktrees (`git status --short` must be clean), post the before/after numbers on PR B, and run `python3 scripts/agent_board.py checkout`.
