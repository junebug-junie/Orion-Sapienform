# Hire Mind → role teach disclosure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After Mind runs on an Orion-origin curiosity/self-inquiry turn, disclose allow-listed work-shape (and optional resume progress) into the role teach the harness sees — advisory only; Orion still authors `:InvestigationRole` / HelpRequest.

**Architecture:** Mind coloring already happens inside Thought before FCC. Carry a thin work-shape subset on `ThoughtEventV1`, then Hub splices `format_role_teach_disclosure(...)` into the frozen kickoff’s role section (or prepends a short advisory block when the marker is absent) before `build_harness_prompt`. Resume path (`_prompt_for_attempt`) may append hop-progress lines from existing `#2244` preamble data. No auto-hire.

**Tech Stack:** Python, Pydantic schemas, Hub turn orchestrator, Thought bus_listener, pytest.

## Global Constraints

- Orion-authored role/HelpRequest only — Python never MERGEs hire choice.
- No keyword cathedrals / no auto-hire on `expected_depth=deep`.
- Origin guard: disclosure only when `utterance_origin=orion` (curiosity path).
- Fail-open: missing Mind/coloring → prompt unchanged.
- Omit disclosure when all work-shape values are absent/`unknown` and foresight empty.
- Collision surface: `curiosity_investigation.py` also owns #2247 self-sense-eval — touch turn/resume only.
- Env escape: `HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE` default `true`.
- Spec: `docs/superpowers/specs/2026-09-19-hire-mind-role-disclosure-design.md`

## File map

| File | Responsibility |
| --- | --- |
| `orion/curiosity/role_teach_disclosure.py` | Pure formatter + splice helper |
| `orion/schemas/thought.py` | Optional `mind_work_shape` on ThoughtEventV1 |
| `services/orion-thought/app/bus_listener.py` | Attach work-shape from coloring onto ThoughtEvent |
| `orion/hub/turn_orchestrator.py` | After stance proceed, splice disclosure into motor `user_message` when orion-origin + flag |
| `services/orion-hub/scripts/curiosity_investigation.py` | Optional: progress lines on resume via formatter |
| `services/orion-hub/app/settings.py` + `.env_example` | Flag |
| Tests | Formatter, splice, Thought attach, Hub wiring |

---

### Task 1: Disclosure formatter (TDD)

**Files:**
- Create: `orion/curiosity/role_teach_disclosure.py`
- Test: `tests/test_role_teach_disclosure.py`

**Interfaces:**
- Produces: `format_role_teach_disclosure(mind_work_shape: Mapping[str, Any] | None, *, progress_lines: Sequence[str] = ()) -> list[str]`
- Produces: `splice_role_teach_disclosure(prompt: str, extra_lines: Sequence[str]) -> str`

- [ ] **Step 1:** Failing tests — present shape → lines; all-unknown+empty foresight → `[]`; splice inserts before `ASKING FOR CONTRACTOR HELP` (or after role header); empty extra → identity
- [ ] **Step 2:** Implement formatter + splice
- [ ] **Step 3:** Tests pass; commit

---

### Task 2: ThoughtEvent carries mind_work_shape

**Files:**
- Modify: `orion/schemas/thought.py`
- Modify: `services/orion-thought/app/bus_listener.py` (`run_stance_react`)
- Test: `services/orion-thought/tests/test_mind_work_shape_on_thought.py` (new)

**Interfaces:**
- Consumes: coloring dict from `select_mind_coloring`
- Produces: `ThoughtEventV1.mind_work_shape: dict[str, str] | None` with only `expected_depth` / `cross_cutting` / `foresight_note` / optional `user_intent` when present

- [ ] **Step 1:** Failing test — coloring with work-shape → thought event field set; juniper/no coloring → None
- [ ] **Step 2:** Schema field + attach in `run_stance_react` after coloring
- [ ] **Step 3:** Tests pass; commit

---

### Task 3: Hub turn splice + flag

**Files:**
- Modify: `orion/hub/turn_orchestrator.py` (after proceed, before harness)
- Modify: `services/orion-hub/app/settings.py`, `.env_example`, sync local `.env`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` — resume: pass progress hint lines into splice if cheap (reuse hop summary already in preamble — do not duplicate whole preamble)
- Test: `services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py` (new)
- Docs: short README note + parent spec acceptance pointer if needed

**Interfaces:**
- Consumes: `thought.mind_work_shape`, `utterance_origin`, settings flag
- Produces: mutated `user_message` for harness only

- [ ] **Step 1:** Failing tests — orion + shape + flag → prompt contains foresight; juniper → unchanged; flag off → unchanged
- [ ] **Step 2:** Wire splice; env parity
- [ ] **Step 3:** Tests pass; `python scripts/sync_local_env_from_example.py`; commit

---

### Task 4: Gates, review, PR

- [ ] Focused pytest for new tests
- [ ] Code review subagent; fix material findings
- [ ] Push + PR report (Markdown); note live verification UNVERIFIED until deploy
- [ ] Do **not** require a live hire for merge acceptance
