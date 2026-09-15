# Finalize Owner-Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make harness finalize (reflect + response repair) use the turn-owner LLM lane — chat for chat-owned Hub turns, agent for agent-owned turns, lease lane when admitted — so unleashed chat finalization no longer queues behind curiosity on gateway `agent`.

**Architecture:** Add one pure resolver (`resolve_finalize_llm_lane`) in `orion/harness/finalize.py`. Context builders stamp both `llm_lane` and `llm_route` from that resolver. Thread `fcc_model_label` from `HarnessRunRequestV1` through governor `bus_listener` → `run_harness_finalize_chain` → reflection / tool-retry / repair builders. Cortex-exec already honors top-level `ctx["llm_route"]` / `ctx["llm_lane"]` overrides; keep its verb-level `agent` default only as a missing-override fallback.

**Tech Stack:** Python 3, Pydantic `ResourceLeaseV1` / `HarnessRunRequestV1`, pytest (asyncio), existing `orion.llm.routes.is_agent_route_model_label`.

**Spec:** `docs/superpowers/specs/2026-09-15-finalize-owner-lane-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-finalize-owner-lane` — rename branch `docs/finalize-owner-lane` → `fix/finalize-owner-lane` at Task 1’s first code commit.

## Global Constraints

- Finalize LLM route = turn owner lane (lease wins; else agent FCC label → `agent`; else `chat`).
- Reuse `orion.llm.routes.is_agent_route_model_label` — do not invent a second classifier.
- Keep `allow_chat_fallback=False` on finalize contexts (no silent cross-lane fallback).
- No prompt edits, no keyword lists on user text, no admission rollout, no timeout changes, no FCC / `MODEL_SONNET` alias changes.
- Do not point finalize at gateway `harness` as a third identity; chat-owned → `chat`.
- Cortex-exec verb default `_default_llm_route_for_step(... harness_finalize_reflect ...) == "agent"` stays as missing-override fallback only; live path must always stamp owner via finalize context.
- Follow TDD: failing test → minimal impl → pass → commit per task.
- Work only in this worktree; never commit from the shared checkout.

## File map

| File | Responsibility |
|------|----------------|
| `orion/harness/finalize.py` | `resolve_finalize_llm_lane`; stamp builders; thread `fcc_model_label` through chain |
| `services/orion-harness-governor/app/bus_listener.py` | Pass `request.fcc_model_label` into `run_harness_finalize_chain` |
| `orion/harness/tests/test_finalize_owner_lane.py` | Pure resolver unit tests (new) |
| `orion/harness/tests/test_finalize_reflect_lane.py` | Invert no-lease agent-only assertions; chat vs agent ownership |
| `orion/harness/tests/test_finalize_resource_lease.py` | Lease still wins over conflicting model label |
| `services/orion-harness-governor/tests/test_harness_governor_rpc.py` | Assert `fcc_model_label` reaches finalize kwargs |
| `services/orion-cortex-exec/tests/test_llm_lane_propagation.py` | Add chat override case; retire “agent is idle isolation” docstring |
| `services/orion-harness-governor/README.md` | Owner-lane rule replaces “ordinary finalization keeps agent” |
| `services/orion-cortex-exec/app/executor.py` | Docstring only on `_default_llm_route_for_step` finalize bullets |

**Not in scope (despite older comments naming them):** `orion/harness/runner.py` does not call finalize — governor `bus_listener` does. Do not invent a runner change.

---

### Task 1: Pure owner-lane resolver

**Files:**
- Create: `orion/harness/tests/test_finalize_owner_lane.py`
- Modify: `orion/harness/finalize.py` (add `resolve_finalize_llm_lane` near the context builders)

**Interfaces:**
- Produces:
  ```python
  def resolve_finalize_llm_lane(
      *,
      resource_lease: ResourceLeaseV1 | None = None,
      fcc_model_label: str | None = None,
  ) -> str:
      ...
  ```
  Returns the gateway route/lane string: lease `.lane` when present; else `"agent"` iff `is_agent_route_model_label(fcc_model_label)`; else `"chat"` (including `None` / `MODEL_SONNET` / any non-agent label).
- Consumes: `ResourceLeaseV1`, `orion.llm.routes.is_agent_route_model_label`, `AGENT_ROUTE_FCC_MODEL_LABEL`

- [ ] **Step 1: Rename branch for code work**

```bash
cd /mnt/scripts/Orion-Sapienform-finalize-owner-lane
git branch -m fix/finalize-owner-lane
```

- [ ] **Step 2: Write the failing resolver tests**

Create `orion/harness/tests/test_finalize_owner_lane.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.harness.finalize import resolve_finalize_llm_lane
from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL
from orion.schemas.resource_admission import ResourceLeaseV1


def _lease(lane: str) -> ResourceLeaseV1:
    now = datetime.now(timezone.utc)
    return ResourceLeaseV1(
        run_id="r1",
        demand_id="d1",
        lease_id="L1",
        resource_key=f"llm.route.{lane}",
        lane=lane,
        backend_key="http://worker:8000",
        generation=1,
        granted_at=now,
        heartbeat_at=now,
        expires_at=now + timedelta(seconds=60),
    )


def test_no_lease_chat_owned_model_sonnet_resolves_chat() -> None:
    assert resolve_finalize_llm_lane(fcc_model_label="MODEL_SONNET") == "chat"


def test_no_lease_missing_label_defaults_chat() -> None:
    assert resolve_finalize_llm_lane(fcc_model_label=None) == "chat"


def test_no_lease_agent_fcc_label_resolves_agent() -> None:
    assert (
        resolve_finalize_llm_lane(fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL) == "agent"
    )


@pytest.mark.parametrize("lane", ["chat", "agent", "metacog"])
def test_lease_lane_wins_over_conflicting_model_label(lane: str) -> None:
    # Agent FCC label must not override an admitted chat (or other) lease.
    assert (
        resolve_finalize_llm_lane(
            resource_lease=_lease(lane),
            fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
        )
        == lane
    )
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /mnt/scripts/Orion-Sapienform-finalize-owner-lane
pytest orion/harness/tests/test_finalize_owner_lane.py -v
```

Expected: FAIL with `ImportError` / `cannot import name 'resolve_finalize_llm_lane'`.

- [ ] **Step 4: Implement the resolver**

In `orion/harness/finalize.py`, add import and function **above** `build_finalize_reflect_context`:

```python
from orion.llm.routes import is_agent_route_model_label
```

```python
def resolve_finalize_llm_lane(
    *,
    resource_lease: ResourceLeaseV1 | None = None,
    fcc_model_label: str | None = None,
) -> str:
    """Gateway llm_route/llm_lane for harness finalize (reflect + repair).

    Owner rule (2026-09-15): admitted lease wins; else agent FCC model label
    → agent; else chat (default unified Hub chat / non-agent labels including
    MODEL_SONNET). Do not hardcode ordinary finalize to agent — that stranded
    chat turns behind curiosity (corr 60f0e051).
    """
    if resource_lease is not None:
        return str(resource_lease.lane)
    if is_agent_route_model_label(fcc_model_label):
        return "agent"
    return "chat"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest orion/harness/tests/test_finalize_owner_lane.py -v
```

Expected: PASS (4 tests / parametrize expansions).

- [ ] **Step 6: Commit**

```bash
git add orion/harness/finalize.py orion/harness/tests/test_finalize_owner_lane.py
git commit -m "$(cat <<'EOF'
feat(harness): add resolve_finalize_llm_lane owner helper

Lease wins; agent FCC label → agent; otherwise chat for unleashed finalize.
EOF
)"
```

---

### Task 2: Context builders use owner lane (invert legacy tests)

**Files:**
- Modify: `orion/harness/tests/test_finalize_reflect_lane.py`
- Modify: `orion/harness/finalize.py` (`build_finalize_reflect_context`, `build_response_repair_context`, and their plan-request wrappers)

**Interfaces:**
- Consumes: `resolve_finalize_llm_lane`
- Produces: both builders accept `fcc_model_label: str | None = None` and set:
  - `llm_route` / `llm_lane` = `resolve_finalize_llm_lane(...)`
  - `allow_chat_fallback` remains `False`
- Plan-request builders pass `fcc_model_label` through to context builders

- [ ] **Step 1: Rewrite failing lane tests**

Replace `orion/harness/tests/test_finalize_reflect_lane.py` entirely with:

```python
from __future__ import annotations

from orion.harness.finalize import (
    build_finalize_reflect_context,
    build_response_repair_context,
)
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL


def test_finalize_reflect_context_chat_owned_no_lease_routes_chat() -> None:
    """Unleashed Hub chat (MODEL_SONNET / non-agent) finalizes on gateway chat."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="night night",
        fcc_model_label="MODEL_SONNET",
    )
    assert ctx["llm_lane"] == "chat"
    assert ctx["llm_route"] == "chat"
    assert ctx["allow_chat_fallback"] is False


def test_finalize_reflect_context_agent_owned_no_lease_routes_agent() -> None:
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="investigate",
        fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
    )
    assert ctx["llm_lane"] == "agent"
    assert ctx["llm_route"] == "agent"
    assert ctx["allow_chat_fallback"] is False


def test_finalize_reflect_context_missing_label_defaults_chat() -> None:
    """No lease + no label → chat (unified chat default), not agent."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="How are you?",
    )
    assert ctx["llm_lane"] == "chat"
    assert ctx["llm_route"] == "chat"
    assert ctx["allow_chat_fallback"] is False


def test_finalize_reflect_context_lane_is_top_level_for_cortex_ctx_merge() -> None:
    """cortex-exec spreads request.context into ctx; resolve_llm_lane_for_step
    reads ctx.get("llm_lane"). Guard key placement."""
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=make_thought(),
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="",
        fcc_model_label="MODEL_SONNET",
    )
    assert "llm_lane" in ctx
    assert "options" not in ctx or "llm_lane" not in ctx.get("options", {})


def test_response_repair_context_chat_owned_no_lease_routes_chat() -> None:
    ctx = build_response_repair_context(
        correlation_id="c-repair",
        draft_text="draft",
        reflection=make_reflection(),
        user_message="night night",
        fcc_model_label="MODEL_SONNET",
    )
    assert ctx["llm_route"] == "chat"
    assert ctx["llm_lane"] == "chat"
    assert ctx["allow_chat_fallback"] is False


def test_response_repair_context_agent_owned_no_lease_routes_agent() -> None:
    ctx = build_response_repair_context(
        correlation_id="c-repair",
        draft_text="draft",
        reflection=make_reflection(),
        user_message="investigate",
        fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
    )
    assert ctx["llm_route"] == "agent"
    assert ctx["llm_lane"] == "agent"
    assert ctx["allow_chat_fallback"] is False
```

- [ ] **Step 2: Run tests — expect FAIL on chat assertions**

```bash
pytest orion/harness/tests/test_finalize_reflect_lane.py -v
```

Expected: FAIL — current builders still hardcode `"agent"` when no lease (e.g. `assert ctx["llm_lane"] == "chat"` gets `"agent"`). Also FAIL on unexpected `fcc_model_label` kwarg until Step 3.

- [ ] **Step 3: Update context + plan-request builders**

In `build_finalize_reflect_context`, add `fcc_model_label: str | None = None` and replace the hardcoded agent default + the long “agent isolation” comment block with:

```python
def build_finalize_reflect_context(
    *,
    correlation_id: str,
    draft_text: str,
    thought: ThoughtEventV1,
    substrate_appraisal: SubstrateFinalizeAppraisalV1,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
    resource_lease: ResourceLeaseV1 | None = None,
    fcc_model_label: str | None = None,
) -> dict[str, Any]:
    lane = resolve_finalize_llm_lane(
        resource_lease=resource_lease,
        fcc_model_label=fcc_model_label,
    )
    return {
        "draft_text": draft_text,
        "thought_event": thought.model_dump(mode="json"),
        "substrate_appraisal": substrate_appraisal.model_dump(mode="json"),
        "grammar_receipts": grammar_receipt_summaries(grammar_receipts),
        "tool_execution": format_tool_execution_digest(grammar_receipts),
        "repair_overlay": repair_overlay.model_dump(mode="json"),
        "finalize_overlay": "",
        "user_message": user_message,
        # Owner-lane finalize: lease lane when admitted; else agent FCC label
        # → agent; else chat. Cortex-exec honors top-level llm_route/llm_lane.
        "llm_route": lane,
        "llm_lane": lane,
        **({"resource_lease": resource_lease.model_dump(mode="json")} if resource_lease else {}),
        "allow_chat_fallback": False,
        "metadata": {
            "correlation_id": correlation_id,
            "mode": "brain",
        },
    }
```

Update `build_finalize_reflect_plan_request` the same way — add `fcc_model_label` and pass it into `build_finalize_reflect_context(...)`.

In `build_response_repair_context`:

```python
def build_response_repair_context(
    *,
    correlation_id: str,
    draft_text: str,
    reflection: FinalizeReflectionV1,
    user_message: str,
    grammar_receipts: list[GrammarReceiptV1] | None = None,
    resource_lease: ResourceLeaseV1 | None = None,
    fcc_model_label: str | None = None,
) -> dict[str, Any]:
    lane = resolve_finalize_llm_lane(
        resource_lease=resource_lease,
        fcc_model_label=fcc_model_label,
    )
    return {
        "draft_text": draft_text,
        "reflection": reflection.model_dump(mode="json"),
        "grammar_receipts": grammar_receipt_summaries(grammar_receipts),
        "tool_execution": format_tool_execution_digest(grammar_receipts),
        "user_message": user_message,
        # Same owner as reflect (5b).
        "llm_route": lane,
        "llm_lane": lane,
        **({"resource_lease": resource_lease.model_dump(mode="json")} if resource_lease else {}),
        "allow_chat_fallback": False,
        "metadata": {
            "correlation_id": correlation_id,
            "mode": "brain",
        },
    }
```

Update `build_response_repair_plan_request` to accept and forward `fcc_model_label`.

- [ ] **Step 4: Run lane tests — expect PASS**

```bash
pytest orion/harness/tests/test_finalize_reflect_lane.py orion/harness/tests/test_finalize_owner_lane.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orion/harness/finalize.py orion/harness/tests/test_finalize_reflect_lane.py
git commit -m "$(cat <<'EOF'
fix(harness): stamp finalize contexts with owner lane

Chat-owned unleashed turns finalize on chat; agent FCC label still uses agent.
EOF
)"
```

---

### Task 3: Thread `fcc_model_label` through the finalize chain

**Files:**
- Modify: `orion/harness/finalize.py` — `run_finalize_reflection`, `maybe_run_finalize_tool_retry`, `run_orion_response_repair`, `run_harness_finalize_chain`
- Modify: `orion/harness/tests/test_finalize_resource_lease.py` (lease still wins when label conflicts)

**Interfaces:**
- Each of these gains `fcc_model_label: str | None = None` and forwards it beside `resource_lease` into the next builder/call.
- `run_harness_finalize_chain(..., resource_lease=..., fcc_model_label=...)` is the public seam governor will call.

Call sites inside `finalize.py` that must forward the kwarg (search `resource_lease=resource_lease` and add the twin):

1. `run_finalize_reflection` → `build_finalize_reflect_plan_request`
2. `maybe_run_finalize_tool_retry` → `run_finalize_reflection` (re-reflect after tool)
3. `run_orion_response_repair` → `build_response_repair_plan_request`
4. `run_harness_finalize_chain` → `run_finalize_reflection`, `maybe_run_finalize_tool_retry`, `run_orion_response_repair`

Update `run_harness_finalize_chain` docstring: owner identity is lease **or** `fcc_model_label`, not “lease only / agent otherwise.”

- [ ] **Step 1: Extend lease regression test so label cannot override lease**

Append to `orion/harness/tests/test_finalize_resource_lease.py`:

```python
@pytest.mark.asyncio
async def test_chat_lease_wins_over_agent_fcc_label(monkeypatch):
    """Admitted chat lease must win even if request carries agent FCC label."""
    monkeypatch.setenv("HARNESS_FINALIZE_TOOL_LOOP_ENABLED", "false")
    now = datetime.now(timezone.utc)
    lease = ResourceLeaseV1(
        run_id="admitted-chat",
        demand_id="admitted-chat:turn",
        lease_id="owner",
        resource_key="llm.route.chat",
        lane="chat",
        backend_key="http://worker:8000",
        generation=3,
        granted_at=now,
        heartbeat_at=now,
        expires_at=now + timedelta(seconds=60),
    )
    from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL

    thought = make_thought()
    overlay = make_repair_overlay()
    draft = "A grounded internal draft."
    molecule = build_draft_molecule(
        correlation_id="lease-chat",
        thought=thought,
        draft_text=draft,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=overlay,
    )

    async def substrate_client(_molecule):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(request):
        assert request.context["llm_route"] == "chat"
        assert request.context["llm_lane"] == "chat"
        assert request.context["allow_chat_fallback"] is False
        reflection = make_reflection(alignment_verdict="aligned")
        return {"final_text": json.dumps(reflection.model_dump(mode="json"))}

    result = await run_harness_finalize_chain(
        correlation_id="lease-chat",
        draft_text=draft,
        draft_molecule=molecule,
        thought=thought,
        grammar_receipts=[],
        repair_overlay=overlay,
        user_message="hi",
        voice_contract=None,
        cortex_client=cortex_client,
        substrate_client=substrate_client,
        resource_lease=lease,
        fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
    )
    assert result.final_text == draft
    assert result.response_repair_ran is False
```

- [ ] **Step 2: Run the new test — expect FAIL (fcc_model_label unexpected or lease ignored once threaded wrongly)**

```bash
pytest orion/harness/tests/test_finalize_resource_lease.py::test_chat_lease_wins_over_agent_fcc_label -v
```

Expected before threading: `TypeError: ... unexpected keyword argument 'fcc_model_label'`.

- [ ] **Step 3: Thread `fcc_model_label` through the four functions**

Minimal pattern for each signature and call:

```python
fcc_model_label: str | None = None,
```

and at every internal call that already passes `resource_lease=resource_lease`, also pass `fcc_model_label=fcc_model_label`.

For `run_finalize_reflection` plan build:

```python
    plan_request = build_finalize_reflect_plan_request(
        correlation_id=correlation_id,
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=substrate_appraisal,
        repair_overlay=overlay,
        user_message=user_message,
        grammar_receipts=grammar_receipts,
        resource_lease=resource_lease,
        fcc_model_label=fcc_model_label,
    )
```

Same for repair plan request and chain orchestration calls.

- [ ] **Step 4: Run lease + owner + reflect lane suites**

```bash
pytest \
  orion/harness/tests/test_finalize_resource_lease.py \
  orion/harness/tests/test_finalize_owner_lane.py \
  orion/harness/tests/test_finalize_reflect_lane.py \
  -v
```

Expected: PASS. Existing parametrized lease test (`lane in {agent, metacog}`) must still pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add orion/harness/finalize.py orion/harness/tests/test_finalize_resource_lease.py
git commit -m "$(cat <<'EOF'
feat(harness): thread fcc_model_label through finalize chain

Reflect, tool-retry re-reflect, and repair share one owner-lane signal.
EOF
)"
```

---

### Task 4: Governor wires `request.fcc_model_label` into finalize

**Files:**
- Modify: `services/orion-harness-governor/app/bus_listener.py` (~line 305 `run_harness_finalize_chain(...)`)
- Modify: `services/orion-harness-governor/tests/test_harness_governor_rpc.py`

**Interfaces:**
- Consumes: `HarnessRunRequestV1.fcc_model_label`
- Produces: `run_harness_finalize_chain(..., fcc_model_label=request.fcc_model_label)`

- [ ] **Step 1: Write failing governor assertion**

In `services/orion-harness-governor/tests/test_harness_governor_rpc.py`, add (near the existing `resource_lease` capture test):

```python
@pytest.mark.asyncio
async def test_finalize_chain_receives_fcc_model_label() -> None:
    from app import bus_listener

    thought = make_thought()
    req = HarnessRunRequestV1(
        correlation_id="c-owner-lane",
        thought_event=thought,
        user_message="night night",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        fcc_model_label="MODEL_SONNET",
        reply_to="orion:harness:run:result:c-owner-lane",
    )
    motor = _motor_result(thought)
    seen: dict[str, object] = {}

    async def _fake_finalize_chain(**kwargs: object) -> HarnessFinalizeChainResult:
        seen.update(kwargs)
        raise RuntimeError("stop after argument capture")

    with patch.object(
        bus_listener,
        "HarnessRunner",
        return_value=AsyncMock(run=AsyncMock(return_value=motor)),
    ), patch.object(bus_listener, "run_harness_finalize_chain", _fake_finalize_chain):
        await bus_listener.handle_harness_run_request(
            AsyncMock(),
            req,
            reply_to="orion:harness:run:result:c-owner-lane",
        )

    assert seen["fcc_model_label"] == "MODEL_SONNET"
```

Reuse the same imports / `_motor_result` / `HarnessFinalizeChainResult` helpers already in that file. If `reply_to` is not a `HarnessRunRequestV1` field in this tree, omit it and match neighboring tests’ constructor style exactly.

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd /mnt/scripts/Orion-Sapienform-finalize-owner-lane
pytest services/orion-harness-governor/tests/test_harness_governor_rpc.py::test_finalize_chain_receives_fcc_model_label -v
```

Expected: `AssertionError` / `KeyError` — `fcc_model_label` missing from kwargs.

- [ ] **Step 3: Wire the kwarg in bus_listener**

In `handle_harness_run_request`’s `run_harness_finalize_chain(...)` call, add:

```python
            fcc_model_label=request.fcc_model_label,
```

next to the existing `resource_lease=request.resource_lease` line.

- [ ] **Step 4: Re-run governor test — expect PASS**

```bash
pytest services/orion-harness-governor/tests/test_harness_governor_rpc.py::test_finalize_chain_receives_fcc_model_label -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  services/orion-harness-governor/app/bus_listener.py \
  services/orion-harness-governor/tests/test_harness_governor_rpc.py
git commit -m "$(cat <<'EOF'
feat(harness-governor): pass fcc_model_label into finalize chain

Unleashed Hub chat turns carry owner identity through post-draft finalize.
EOF
)"
```

---

### Task 5: Docs + cortex-exec comment/propagation alignment

**Files:**
- Modify: `services/orion-harness-governor/README.md` (Durable admission owner section ~line 231)
- Modify: `services/orion-cortex-exec/app/executor.py` (`_default_llm_route_for_step` docstring for finalize verbs)
- Modify: `services/orion-cortex-exec/tests/test_llm_lane_propagation.py` (add chat case; rewrite agent docstring)

**Interfaces:**
- No runtime API change. Cortex-exec `_default_llm_route_for_step` still returns `"agent"` for finalize verbs when **no** `ctx` override — that remains a missing-stamp fallback. Live path always stamps via Task 2 contexts; `_resolve_llm_route_override` already wins.

- [ ] **Step 1: Add cortex-exec chat override test (failing only if resolve regresses)**

In `services/orion-cortex-exec/tests/test_llm_lane_propagation.py`, keep the existing agent resolve test but replace its isolation docstring, and add:

```python
def test_finalize_reflect_ctx_llm_lane_resolves_chat() -> None:
    """Owner-stamped chat finalize must resolve to chat lane (not agent)."""
    step = SimpleNamespace(
        verb_name="harness_finalize_reflect",
        step_name="llm_harness_finalize_reflect",
    )
    out = resolve_llm_lane_for_step(
        step=step,
        ctx={
            "llm_lane": "chat",
            "allow_chat_fallback": False,
            "metadata": {"mode": "brain"},
        },
        settings=_settings(),
    )
    assert out["llm_lane"] == "chat"
    assert out["allow_chat_fallback"] is False
```

Rewrite `test_finalize_reflect_ctx_llm_lane_resolves_agent` docstring to: agent-owned / agent-stamped finalize still resolves to agent; do **not** claim agent is “unused by any other verb” or reject chat.

- [ ] **Step 2: Run cortex-exec lane tests**

```bash
pytest services/orion-cortex-exec/tests/test_llm_lane_propagation.py -v
```

Expected: PASS (chat case exercises existing `resolve_llm_lane_for_step` behavior).

- [ ] **Step 3: Update README + executor docstring**

In `services/orion-harness-governor/README.md`, replace:

> Ordinary finalization keeps the existing agent route.

with:

> Unleashed finalization uses the turn owner lane: chat for non-agent FCC
> labels (default Hub chat / `MODEL_SONNET`), agent for the agent FCC model
> label. Admitted leases still force their assigned lane for every finalize
> LLM hop.

In `services/orion-cortex-exec/app/executor.py` `_default_llm_route_for_step` docstring, change the finalize bullet from “AGENT lane… context builders also stamp route/lane=agent” to:

> harness_finalize_reflect / orion_response_repair: fallback AGENT only when
> the caller did not stamp `llm_route`. Live harness finalize always stamps
> owner lane via `orion.harness.finalize.resolve_finalize_llm_lane` (chat for
> chat-owned Hub turns, agent for agent FCC label, lease lane when admitted).
> `_resolve_llm_route_override` wins over this default.

Do **not** change the return value `"agent"` in the function body.

Leave `services/orion-cortex-exec/tests/test_harness_finalize_route.py` and `test_default_llm_route_for_step.py` asserting the empty-ctx fallback as-is (they are not “unleashed ordinary finalize always agent” from the harness builders). Optionally add a one-line comment above those tests: “fallback when finalize context omitted owner stamp.”

- [ ] **Step 4: Regression grep — no harness test asserts unleashed ordinary → agent**

```bash
cd /mnt/scripts/Orion-Sapienform-finalize-owner-lane
rg -n 'routes_to_agent|ordinary finaliz|llm_lane.*=.*\"agent\"' \
  orion/harness/tests/test_finalize_reflect_lane.py \
  orion/harness/finalize.py \
  services/orion-harness-governor/README.md
```

Expected: no leftover “ordinary finalize = agent” claim; harness reflect-lane tests must not assert unleashed default `"agent"` without an agent FCC label.

- [ ] **Step 5: Commit**

```bash
git add \
  services/orion-harness-governor/README.md \
  services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_llm_lane_propagation.py
git commit -m "$(cat <<'EOF'
docs: describe finalize owner-lane routing

Align governor README and cortex-exec fallback docs with chat-owned default.
EOF
)"
```

---

### Task 6: Full focused gate + PR handoff notes

**Files:** none new (verification only)

- [ ] **Step 1: Run focused harness + governor + cortex-exec suites**

```bash
cd /mnt/scripts/Orion-Sapienform-finalize-owner-lane
pytest \
  orion/harness/tests/test_finalize_owner_lane.py \
  orion/harness/tests/test_finalize_reflect_lane.py \
  orion/harness/tests/test_finalize_resource_lease.py \
  orion/harness/tests/test_harness_finalize_chain.py \
  orion/harness/tests/test_response_repair_gate.py \
  services/orion-harness-governor/tests/test_harness_governor_rpc.py \
  services/orion-cortex-exec/tests/test_llm_lane_propagation.py \
  services/orion-cortex-exec/tests/test_default_llm_route_for_step.py \
  services/orion-cortex-exec/tests/test_harness_finalize_route.py \
  -q
```

Expected: all PASS.

- [ ] **Step 2: Safe graphify update (AST only)**

```bash
scripts/safe_graphify_update.sh
```

- [ ] **Step 3: Push + open PR (when Juniper asks)**

PR body must include the design acceptance checks and:

```text
Live (post-deploy): UNVERIFIED until a short Hub chat turn while curiosity/
agent work is busy shows finalize llm_route=chat in cortex-exec / runtime
activity without a multi-minute agent wait (incident shape: corr 60f0e051).
```

Restart after merge/deploy (print for Juniper; do not sudo):

```bash
# From a worktree, via safe_docker_build.sh — exact service names as operated locally:
scripts/safe_docker_build.sh orion-harness-governor up -d --build
# cortex-exec unchanged at runtime for this patch (docs/tests only), but rebuild
# if this branch also carries unrelated cortex-exec code.
```

No `.env_example` changes → no env sync required.

---

## Self-review

1. **Spec coverage:** Owner rule / resolver → Task 1. Context builders + inverted harness tests → Task 2. Thread label through chain → Task 3. Governor wire → Task 4. README + comments + optional cortex propagation → Task 5. Acceptance unit checks 1–5 → Tasks 1–5; live check 6 → Task 6 UNVERIFIED note. Non-goals (admission, timeouts, FCC alias, harness third route) not tasked.
2. **Placeholder scan:** No TBD/TODO/“similar to Task N” without code.
3. **Type consistency:** `fcc_model_label: str | None` everywhere; resolver returns `str` lane; builders set both `llm_lane` and `llm_route` to the same value.
4. **Correction vs spec file list:** Spec named `runner.py`; live finalize call site is `bus_listener.py` — plan uses governor. Cortex-exec verb default left as fallback on purpose.
