# PREFLIGHT: Decision Contract Audit (PlannerReact / AgentChain / Council)

## Scope and method
Read-only audit via `rg` + source inspection only (no runtime/tests).

---

## Phase 1 — Hub button/path mapping to request envelope

### Hub UI mode buttons
**File:** `services/orion-hub/templates/index.html`  
**Evidence (buttons):**
```html
<button class="mode-btn ..." data-mode="brain">Brain</button>
<button class="mode-btn ..." data-mode="agent">Agent</button>
<button class="mode-btn ..." data-mode="council">Council</button>
```

### Hub frontend payload differences (brain vs agent vs council)
**File:** `services/orion-hub/static/js/app.js`  
**Functionality:** mode button click sets `currentMode`; send payload uses `mode: currentMode`.

**Evidence (mode binding):**
```javascript
const modeButtons = document.querySelectorAll('.mode-btn');
modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    currentMode = btn.dataset.mode || 'brain';
```

**Evidence (send payload):**
```javascript
const payload = {
   text_input: text,
   mode: currentMode,
   session_id: orionSessionId,
```

### Hub backend WS handler mapping
**File:** `services/orion-hub/scripts/websocket_handler.py`  
**Function:** websocket receive loop (inline in handler).

**Evidence (mode intake + trace verb mapping):**
```python
mode = data.get("mode", "brain")
trace_verb = "chat_general"
if mode == "agent":
    trace_verb = "task_execution"
elif mode == "council":
    trace_verb = "council_deliberation"
```

**Evidence (request to gateway):**
```python
chat_req = CortexChatRequest(
    prompt=prompt_with_ctx,
    mode=mode,
    ...
    metadata={"source": "hub_ws", "trace_verb": trace_verb}
)
```

### Hub HTTP handler mapping
**File:** `services/orion-hub/scripts/api_routes.py`  
**Function:** `handle_chat_request(...)`.

**Evidence (mode forwarded, optional explicit verb override only from UI verb chooser):**
```python
mode = payload.get("mode", "brain")
...
verb_override = None
...
req = CortexChatRequest(
    prompt=user_prompt,
    mode=mode,
    ...
    verb=verb_override,
```

**Key delta by path:**
- Brain/Agent/Council differs primarily by `mode`.
- `verb` is normally unset unless user picks exactly one custom verb in UI; then `verb_override` is set.

---

## Phase 2 — Orch branching and forwarding to Exec

### Orch entrypoint
**File:** `services/orion-cortex-orch/app/main.py`  
**Function:** `handle(env: BaseEnvelope)`.

**Evidence (ingress contract):**
```python
req = CortexClientRequest.model_validate(raw_payload)
logger.info(
    "Validated orch request: corr=%s mode=%s verb=%s ...",
    ..., req.mode, req.verb,
)
```

### Mode→plan branching
**File:** `services/orion-cortex-orch/app/orchestrator.py`  
**Function:** `_build_plan_for_mode(req)`.

**Evidence:**
```python
def _build_plan_for_mode(req: CortexClientRequest) -> ExecutionPlan:
    if req.mode == "agent":
        return build_agent_plan(req.verb)
    if req.mode == "council":
        return build_council_plan(req.verb)
    return build_plan_for_verb(req.verb, mode=req.mode)
```

### Agent/Council plan composition
**File:** `services/orion-cortex-orch/app/orchestrator.py`  
**Functions:** `build_agent_plan`, `build_council_plan`.

**Evidence (agent):**
```python
steps=[
  ExecutionStep(... step_name="planner_react", services=["PlannerReactService"], ...),
  ExecutionStep(... step_name="agent_chain", services=["AgentChainService"], ...)
],
metadata={"mode": "agent"},
```

**Evidence (council):**
```python
steps=[
  ExecutionStep(... step_name="council_supervisor", services=["CouncilService"], ...)
],
metadata={"mode": "council"},
```

### Where default `chat_general` is pinned before Orch
**File:** `services/orion-cortex-gateway/app/main.py` (also bus path mirror in `app/bus_client.py`).

**Evidence:**
```python
# If verb is not provided, default to chat_general
verb = req.verb if req.verb else "chat_general"
```

**Consequence:** even `mode=agent` or `mode=council` commonly enter Orch with `verb="chat_general"` unless UI explicitly overrides verb.

---

## Phase 3 — Core contract audit in Exec (parse/enforce vs prose)

### Exec entrypoint
**File:** `services/orion-cortex-exec/app/main.py`  
**Function:** `handle(env)`.

**Evidence (exec request parsing + router):**
```python
req_env = CortexExecRequest.model_validate(env.model_dump(mode="json"))
...
res = await router.run_plan(... req=req_env.payload, ... ctx=ctx)
```

### Router mode split (critical)
**File:** `services/orion-cortex-exec/app/router.py`  
**Function:** `PlanRunner.run_plan(...)`.

**Evidence:**
```python
ctx["verb"] = plan.verb_name
if mode in {"agent", "council"} or extra.get("supervised"):
    supervisor = Supervisor(bus)
    return await supervisor.execute(...)
```

### Generic non-supervised final extraction is text-only
**File:** `services/orion-cortex-exec/app/router.py`  
**Function:** `_extract_final_text(...)`.

**Evidence:**
```python
text = payload.get("content") or payload.get("text")
if text:
    return str(text)
```

No directive parsing here (`next_verb`, `action`, etc. ignored in this path).

### Supervisor planner decision handling (agent/council path)
**File:** `services/orion-cortex-exec/app/supervisor.py`  
**Functions:** `_planner_step`, `execute`.

**Evidence (PlannerReact response parsed as typed model, then only last `action` considered):**
```python
planner_res = await self.planner_client.plan(...)
...
final = planner_res.final_answer
action = None
if planner_res.trace:
    last = planner_res.trace[-1]
    action = last.action
```

**Evidence (execution rule):**
```python
if planner_final and planner_final.content:
    final_text = planner_final.content
...
if planner_step.status != "success" or not action:
    break
```

This is *partially binding* only when `action` exists and maps to a verb/tool. If planner returns prose/final answer or missing action, supervisor stops ReAct and later escalates.

### Escalation behavior when no binding action/final
**File:** `services/orion-cortex-exec/app/supervisor.py`  
**Function:** `execute`.

**Evidence:**
```python
if not final_text:
    agent_step = await self._agent_chain_escalation(...)
    ...
    final_text = agent_payload.get("text") or final_text
```

This converts to prose-return path (AgentChain `text`) instead of enforced verb execution.

### Council result handling in supervisor
**File:** `services/orion-cortex-exec/app/supervisor.py`  
**Function:** `execute`.

**Evidence:**
```python
council_payload = council_step.result.get("CouncilService", {})
if isinstance(council_payload, dict):
    final_text = council_payload.get("final_text") or final_text
```

`verdict.action` is not consumed as an executable directive.

---

## Phase 4 — Worker output contracts (what they actually return)

### PlannerReact service output
**Schema:** `orion/schemas/agents/schemas.py::PlannerResponse`
```python
class PlannerResponse(BaseModel):
    status: Literal["ok", "error", "timeout"] = "ok"
    final_answer: Optional[FinalAnswer] = None
    trace: List[TraceStep] = Field(default_factory=list)
```

**Service publication:** `services/orion-planner-react/app/bus_listener.py`
```python
resp = await run_react_loop(planner_req)
out_env = BaseEnvelope(kind="agent.planner.result", ... payload=resp.model_dump(...))
```

**Planner internals:** `services/orion-planner-react/app/api.py::run_react_loop`
- Emits `trace[].action` (tool_id/input) and/or `final_answer.content`.
- Bounded by `limits.max_steps`.

### AgentChain service output
**Schema:** `orion/schemas/agents/schemas.py::AgentChainResult`
```python
class AgentChainResult(BaseModel):
    mode: str
    text: str
    structured: Dict[str, Any] = Field(default_factory=dict)
    planner_raw: Dict[str, Any] = Field(default_factory=dict)
```

**Service production:** `services/orion-agent-chain/app/api.py::execute_agent_chain`
```python
final = raw_resp.get("final_answer") or {}
text = final.get("content") or ""
...
return AgentChainResult(mode=body.mode, text=text, ...)
```

This is fundamentally prose-first (`text`) with optional structure, no mandatory executable directive.

### Council service output
**Schema:** `orion/schemas/agents/schemas.py::CouncilResult`
```python
class CouncilResult(BaseModel):
    final_text: str
    ...
    verdict: AuditVerdict
```

**Service publication:** `services/orion-agent-council/app/publisher.py`
```python
env = BaseEnvelope(kind="council.result", ... payload=result.model_dump(...))
```

Council includes structured `verdict.action`, but Exec supervisor currently consumes only `final_text`.

---

## Phase 5 — All `chat_general` fallback injection points

Detected injection points:
1. `services/orion-cortex-gateway/app/main.py` — `verb = req.verb if req.verb else "chat_general"`
2. `services/orion-cortex-gateway/app/bus_client.py` — same fallback for bus ingress
3. `services/orion-hub/scripts/websocket_handler.py` — `trace_verb = "chat_general"` (telemetry label only)
4. `orion/cognition/hub_gateway/bus_harness.py` — CLI default `--verb chat_general`

### Which fallback affects agent/council collapse?
**Primary functional injector:** cortex-gateway defaulting `verb` to `chat_general` when hub sends mode-only payload.  
That means agent/council plans are often built with `verb_name=chat_general` as their target verb.

### Is there explicit fallback *inside* supervisor from agent/council to `chat_general`?
No explicit mode switch to `chat_general` in Exec supervisor. The observed degeneration is mostly **emergent** from:
- upstream default verb pinning to `chat_general`, and
- non-binding worker outputs being consumed as prose (`final_text` / `text`) without hard directive enforcement.

---

## Phase 6 — Stopping, budgets, and model routing

### Stop/loop control ownership
- **Supervisor loop cap:** `services/orion-cortex-exec/app/supervisor.py` uses `max_steps = int(ctx.get("max_steps") or 3)`.
- **Planner loop cap:** `services/orion-planner-react/app/api.py` loops `for step_index in range(payload.limits.max_steps)`.
- **Council loop cap:** `services/orion-agent-council/app/pipeline.py` loops while `ctx.round_index < settings.max_rounds and not ctx.stop`.

### Timeouts/budgets
- Step-level timeout in Exec step caller from step timeout (`step.timeout_ms`).
- Planner/Council RPC calls use configured timeout knobs.
- Council has `COUNCIL_ROUND_TIMEOUT_SEC`, `COUNCIL_LLM_TIMEOUT_SEC`, `COUNCIL_MAX_ROUNDS` settings.

### Model routing points
- PlannerReact picks model via env in planner service: `PLANNER_MODEL` defaulting to `llama-3-8b-instruct-q4_k_m`.
- LLM-Gateway has profile/route table knobs (`LLM_PROFILES_CONFIG_PATH`, `LLM_ROUTE_*`, default backend/model).
- Council relies on LLM client through gateway channels; routing/model choice is indirect via gateway configs.

---

## Verdict matrix

| Worker | Output structured? | Exec enforces as decision? | If not, what happens? | Evidence |
|---|---|---|---|---|
| PlannerReact | **Yes, mixed** (`trace[].action` + optional `final_answer`) | **Partially** (Supervisor executes only when `action` exists) | If no action / planner final prose, ReAct loop stops and flow escalates to AgentChain prose path. | `supervisor._planner_step` and `supervisor.execute`; `PlannerResponse` schema. |
| AgentChain | **Mostly prose contract** (`text`, optional `structured`) | **No** (used as terminal text) | Exec sets `final_text = agent_payload.text`; no next-verb/tool directive enforcement. | `AgentChainResult` schema + `supervisor.execute` escalation block. |
| Council | **Yes** (`verdict.action` + `final_text`) | **No (for execution)** | Exec reads `final_text` only; verdict not interpreted into executable steps. | `CouncilResult` schema + `supervisor.execute` council payload consumption. |

---

## Primary answer (why Agent/Council degenerate)
1. **Default verb pinning:** gateway injects `verb="chat_general"` whenever Hub does not explicitly set verb, including Agent/Council mode traffic.
2. **Weak binding in supervisor:** PlannerReact action is optional; absent action leads to fallback path.
3. **Prose terminalization:** AgentChain/Council are consumed primarily as `text`/`final_text`; structured directives are not a required, enforced contract at Exec boundary.

Net: degeneration is mostly from **non-binding decision contracts + prose finalization**, with `chat_general` default pinning amplifying it.

---

## “Append new models” diagnosis (NO implementation)

### Insertion point A — Cheap router model
- **Where:** `services/orion-cortex-exec/app/router.py::PlanRunner.run_plan` just before supervised branch dispatch.
- **Needed contract:** input = `{mode, verb, last_user_message, recall/debug context}`; output = strict enum `{path: direct|react_lite|agent_chain|council, confidence, budget_class}` that Exec must validate and enforce.

### Insertion point B — Supervisor model
- **Where:** `services/orion-cortex-exec/app/supervisor.py::execute` before/inside each loop iteration (after planner output, before tool execution/escalation).
- **Needed contract:** input = `{planner trace, pending action, observations, budgets}`; output = strict decision `{execute_action|request_replan|escalate_agent_chain|checkpoint_council|finish}` with hard schema validation.

### Insertion point C — ReAct-lite bounded loop
- **Where:** existing loop in `supervisor.execute` (currently `max_steps` + soft semantics).
- **Needed contract:** each iteration must emit validated step directive `{tool_id,input}` or explicit `{finish:true,final_text}`; absence/invalid = deterministic fail state (not silent prose fallback).

### Insertion point D — Council enforcement bridge
- **Where:** council handling block in `supervisor.execute` after `council_step` returns.
- **Needed contract:** map `CouncilResult.verdict.action` to explicit Exec action enum (`accept_text`, `revise`, `delegate_chain`, `new_round`) and enforce branch transitions; do not treat `final_text` alone as sole signal.

### Insertion point E — Hub/Gateway ingress contract hardening
- **Where:** `services/orion-cortex-gateway/app/main.py` / `app/bus_client.py` default verb section.
- **Needed contract:** if `mode in {agent,council}` and verb absent, route to mode-specific canonical entry verbs (or require explicit) rather than hard-pinning to `chat_general`.

---

## Commands run
```bash
rg -n --hidden --no-ignore-vcs "chat_general|agent\b|council\b|mode\b|button\b|/agent|/council" services hub orion app ui
rg -n --hidden --no-ignore-vcs "cortex\.orch\.request|orch\.request|Cortex-Orch" services hub orion app ui
rg -n --hidden --no-ignore-vcs "verb\b|verb_name\b|default.*chat_general|\"chat_general\"" services hub orion app ui

rg -n --hidden --no-ignore-vcs "Cortex-?Orch|orch.*handler|handle_.*orch|cortex\.orch" services/orion-cortex-orch* services/*orch* orion
rg -n --hidden --no-ignore-vcs "mode\b|agent\b|council\b|chat_general|default.*verb|fallback" services/orion-cortex-orch* services/*orch* orion

rg -n --hidden --no-ignore-vcs "Cortex-?Exec|exec.*handler|handle_.*exec|cortex\.exec" services/orion-cortex-exec* services/*exec* orion
rg -n --hidden --no-ignore-vcs "PlannerReact|react_planner|agent\.planner|planner.*request|planner.*result" services/orion-cortex-exec* services/*exec* orion
rg -n --hidden --no-ignore-vcs "AgentChain|agent_chain|agent\.chain|chain.*request|chain.*result" services/orion-cortex-exec* services/*exec* orion
rg -n --hidden --no-ignore-vcs "Council|council\.request|council\.result" services/orion-cortex-exec* services/*exec* orion
rg -n --hidden --no-ignore-vcs "model_validate|parse_obj|json\.loads|directive|verb_name|next_verb|action|tool_call" services/orion-cortex-exec* services/*exec* orion

rg -n --hidden --no-ignore-vcs "class .*Planner|PlannerReact|planner.*result|planner_response|return .*result|BaseModel" services/*planner* services orion
rg -n --hidden --no-ignore-vcs "class .*AgentChain|AgentChain|tool_calls|tools|observations|max_steps|stop|BaseModel|return" services/*agent* services orion
rg -n --hidden --no-ignore-vcs "class .*Council|Council|directive|finish|continue|next_step|BaseModel|return" services/*council* services orion

rg -n --hidden --no-ignore-vcs "\"chat_general\"|default.*chat_general|fallback.*chat_general|or \"chat_general\"" services orion

rg -n --hidden --no-ignore-vcs "max_steps|max_loops|stop|done|finish|budget|timeout|deadline|token" services/orion-cortex-exec* services/*planner* services/*agent* services/*council* orion
rg -n --hidden --no-ignore-vcs "MODEL|model_name|route|profile|cheap|fast|small" services/orion-llm-gateway* services/*planner* services/*council* orion
```
