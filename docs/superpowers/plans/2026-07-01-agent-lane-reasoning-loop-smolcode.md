# Agent-lane reasoning loop (smolcode REPL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Hub `mode: agent` run the existing smolagents `CodeAgent` reasoning loop as a first-class context-exec mode (`agent_repl`) on a loop-sized budget, with live step streaming into the Hub UI, proven against the live local model before retiring `investigation_v2` from that lane.

**Architecture:** A new context-exec mode `agent_repl` is dispatched straight to `SmolagentsCodeEngine` from `ContextExecRunner.run`, bypassing the RLM engine registry, the `alexzhang` default, `investigation_v2`, and all keyword classification. The Hub bridge builds `mode="agent_repl"` for the agent lane (behind `HUB_AGENT_REPL_ENABLED`). The smolagents model wrapper is fixed to preserve message roles and honor stop sequences. Budgets are raised for a multi-step loop (~10 min). Per-step reasoning is emitted as bus events and relayed live to the browser WebSocket. The loop's own `final_answer` is the chat response — no synthesis/finalize pass, no canned apology.

**Tech Stack:** Python 3.12, FastAPI, smolagents 1.26, OrionBus (Redis pub/sub), pydantic v2, aiohttp; Hub static JS (vanilla).

**Gating:** Gate 1 (Tasks 1–9) ships routing + permissions + wrapper fix + budgets + server-side step events, and must be proven live (Task 9 `GATE1_PASS`) before Gate 2. Gate 2 (Tasks 10–15) wires step streaming into the Hub UI and retires `investigation_v2` from the agent lane.

---

## Repo conventions (read before starting)

- **Context-exec tests** run from repo root with the repo on `PYTHONPATH`. The test files insert repo root + service root into `sys.path` themselves and import as `from app.<module> import ...`.
  - Command form: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/<file> -q`
  - Confirmed working (baseline): `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_smolcode_engine.py -q` → `6 passed`.
- **Hub tests** run via the shared runner: `./scripts/test_service.sh orion-hub services/orion-hub/tests/<file> -q`. Hub source imports use the `scripts.*` / `app.*` namespace (`services/orion-hub/scripts/settings.py` re-exports `from app.settings import settings`).
- If `./venv` is missing, bootstrap with `./scripts/bootstrap_test_envs.sh` (installs `requirements-dev.txt`, i.e. pytest).
- **Never** add keyword/phrase trigger lists to select conversational mode (`.cursor/rules/conversational-behavior-anti-slop.mdc`). The `agent_repl` lane carries **zero** keyword classification.
- **Env parity is non-negotiable** (`AGENTS.md`): every `.env` key change is mirrored in the sibling `.env_example` in the same change set, then `python scripts/sync_local_env_from_example.py` is run from repo root.

## File Structure

Files created or modified, and their responsibility:

- `orion/schemas/context_exec.py` — add `agent_repl` to the `ContextExecMode` literal (shared contract). *Modify.*
- `services/orion-context-exec/app/llm_tools.py` — extend `llm_chat_route` to accept explicit `messages` and `stop` sequences. *Modify.*
- `services/orion-context-exec/app/organ_runtime.py` — extend `OrganRuntime.llm_chat` to pass `messages` and `stop` through. *Modify.*
- `services/orion-context-exec/app/smolcode_engine.py` — fix the model wrapper (roles + stop), thread per-step timeout + max_steps from settings, accept `step_callbacks`. *Modify.*
- `services/orion-context-exec/app/events.py` — add `agent_step` event emitter method. *Modify.*
- `services/orion-context-exec/app/runner.py` — add `_run_agent_repl` dispatch branch. *Modify.*
- `services/orion-context-exec/app/settings.py` — add `context_exec_agent_repl_max_steps`, raise budget defaults. *Modify.*
- `services/orion-context-exec/.env` + `.env_example` — budget keys. *Modify.*
- `services/orion-hub/scripts/context_exec_agent_bridge.py` — route agent lane to `agent_repl` behind `HUB_AGENT_REPL_ENABLED`; no keyword matcher on this lane. *Modify.*
- `services/orion-hub/app/settings.py` — add `HUB_AGENT_REPL_ENABLED`, `HUB_CONTEXT_EXEC_EVENT_CHANNEL`, raise `HUB_CONTEXT_EXEC_TIMEOUT_SEC` default. *Modify.*
- `services/orion-hub/.env` + `.env_example` + `docker-compose.yml` — hub keys. *Modify.*
- `services/orion-hub/scripts/agent_step_relay.py` — **new**: long-lived bus subscriber fanning agent step events to per-correlation queues. *Create.*
- `services/orion-hub/scripts/main.py` — start/stop the relay. *Modify.*
- `services/orion-hub/scripts/websocket_handler.py` — per-turn queue drain to the browser WS during an `agent_repl` run. *Modify.*
- `services/orion-hub/static/js/app.js` — handle `agent_step` WS frames. *Modify.*
- `services/orion-hub/static/js/agent-trace.js` — render live steps into the trace panel. *Modify.*
- Tests: `services/orion-context-exec/tests/test_smolcode_engine.py`, `services/orion-context-exec/tests/test_agent_repl_runner.py` (new), `services/orion-hub/tests/test_agent_repl_bridge.py` (new), `services/orion-hub/tests/test_agent_step_relay.py` (new).
- Verification scripts: `services/orion-context-exec/scripts/verify_agent_repl_live.py` (new, Gate 1), `services/orion-hub/scripts/verify_agent_repl_stream_live.py` (new, Gate 2).

---

# GATE 1 — prove it reasons live

## Task 1: Add `agent_repl` mode to the shared contract

**Files:**
- Modify: `orion/schemas/context_exec.py:16-27`
- Test: `services/orion-context-exec/tests/test_agent_repl_runner.py` (create)

- [ ] **Step 1: Write the failing test**

Create `services/orion-context-exec/tests/test_agent_repl_runner.py`:

```python
"""Tests for the agent_repl mode + runner dispatch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_agent_repl_is_a_valid_mode():
    from orion.schemas.context_exec import ContextExecRequestV1

    req = ContextExecRequestV1(text="what does orion-hub do?", mode="agent_repl")
    assert req.mode == "agent_repl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_agent_repl_is_a_valid_mode -q`
Expected: FAIL — pydantic `ValidationError` (`agent_repl` not in allowed literal).

- [ ] **Step 3: Add the mode to the literal**

In `orion/schemas/context_exec.py`, add `"agent_repl"` to `ContextExecMode`:

```python
ContextExecMode = Literal[
    "belief_provenance",
    "trace_autopsy",
    "repo_impact_analysis",
    "patch_proposal",
    "memory_correction_proposal",
    "runtime_debug",
    "grammar_collision_audit",
    "memory_contradiction_review",
    "general_investigation",
    "investigation_v2",
    "agent_repl",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_agent_repl_is_a_valid_mode -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/context_exec.py services/orion-context-exec/tests/test_agent_repl_runner.py
git commit -m "feat(context-exec): add agent_repl mode to ContextExecMode contract"
```

---

## Task 2: Let the LLM RPC carry explicit messages + stop sequences

The current `llm_chat_route` wraps a single `prompt` string into one user message and never forwards stop sequences. The smolagents wrapper needs both to preserve roles and terminate code blocks.

**Files:**
- Modify: `services/orion-context-exec/app/llm_tools.py:32-63`
- Test: `services/orion-context-exec/tests/test_agent_repl_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-context-exec/tests/test_agent_repl_runner.py`:

```python
@pytest.mark.asyncio
async def test_llm_chat_route_forwards_messages_and_stop(monkeypatch):
    from app import llm_tools
    from orion.core.bus.bus_schemas import LLMMessage

    captured = {}

    class FakeCodec:
        def decode(self, data):
            class D:
                ok = True

                class envelope:
                    payload = {"content": "ok"}

            return D()

    class FakeBus:
        codec = FakeCodec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            captured["payload"] = env.payload
            return {"data": b"x"}

    monkeypatch.setattr(llm_tools.settings, "orion_bus_enabled", True, raising=False)

    msgs = [
        LLMMessage(role="system", content="you are an agent"),
        LLMMessage(role="user", content="find the bug"),
    ]
    result = await llm_tools.llm_chat_route(
        FakeBus(),
        prompt="find the bug",
        route="agent",
        messages=msgs,
        stop=["<end_code>"],
    )
    assert result["ok"] is True
    payload = captured["payload"]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["options"]["stop"] == ["<end_code>"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_llm_chat_route_forwards_messages_and_stop -q`
Expected: FAIL — `TypeError: llm_chat_route() got an unexpected keyword argument 'messages'`.

- [ ] **Step 3: Extend `llm_chat_route`**

In `services/orion-context-exec/app/llm_tools.py`, change the signature and message/options build. Replace the signature block and the `messages = [...]` / `req = ChatRequestPayload(...)` section:

```python
async def llm_chat_route(
    bus: OrionBusAsync | None,
    *,
    prompt: str,
    route: str,
    correlation_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    context: Any = None,
    schema: str | None = None,
    messages: list[LLMMessage] | None = None,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke LLM gateway bus RPC with an explicit trusted route id."""
    route_key = str(route or "").strip().lower()
    if not settings.orion_bus_enabled or bus is None:
        return {
            "ok": False,
            "route": route_key,
            "summary": "llm bus unavailable",
            "error": "bus_disabled",
        }

    reply_channel = f"orion:exec:result:LLMGatewayService:{uuid.uuid4().hex}"
    corr = _corr_uuid(correlation_id)
    chat_messages = messages if messages else [LLMMessage(role="user", content=prompt)]
    options: dict[str, Any] = {}
    if schema:
        options["schema"] = schema
    if stop:
        options["stop"] = list(stop)
    if context is not None:
        options["context"] = context
    req = ChatRequestPayload(
        messages=chat_messages,
        route=route_key,
        raw_user_text=prompt,
        session_id=session_id,
        user_id=user_id,
        options=options,
    )
```

Delete the now-redundant `if context is not None:` block that rebuilt `req` (context is folded into `options` above).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_llm_chat_route_forwards_messages_and_stop -q`
Expected: PASS

- [ ] **Step 5: Run the module's existing tests (regression)**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_smolcode_engine.py -q`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add services/orion-context-exec/app/llm_tools.py services/orion-context-exec/tests/test_agent_repl_runner.py
git commit -m "feat(context-exec): llm_chat_route forwards explicit messages and stop sequences"
```

---

## Task 3: Thread messages + stop through OrganRuntime.llm_chat

**Files:**
- Modify: `services/orion-context-exec/app/organ_runtime.py:69-89`
- Test: `services/orion-context-exec/tests/test_agent_repl_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-context-exec/tests/test_agent_repl_runner.py`:

```python
@pytest.mark.asyncio
async def test_organ_runtime_llm_chat_passes_messages_and_stop(monkeypatch):
    from app import organ_runtime as orm
    from orion.core.bus.bus_schemas import LLMMessage
    from orion.schemas.context_exec import ContextExecRequestV1

    seen = {}

    async def fake_route(bus, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "content": "hi"}

    monkeypatch.setattr(orm.llm_tools, "llm_chat_route", fake_route)

    rt = orm.OrganRuntime(
        bus=object(),
        request=ContextExecRequestV1(text="q", mode="agent_repl"),
        run_id="r1",
        llm_route="agent",
    )
    msgs = [LLMMessage(role="user", content="q")]
    await rt.llm_chat("q", route="agent", messages=msgs, stop=["<end_code>"])
    assert seen["messages"] == msgs
    assert seen["stop"] == ["<end_code>"]
    assert seen["route"] == "agent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_organ_runtime_llm_chat_passes_messages_and_stop -q`
Expected: FAIL — `TypeError: llm_chat() got an unexpected keyword argument 'messages'`.

- [ ] **Step 3: Extend `OrganRuntime.llm_chat`**

In `services/orion-context-exec/app/organ_runtime.py`, add imports and update the method. At the top, add to the existing bus_schemas usage (the module currently imports only `ContextExecRequestV1`); add a local import inside the method to avoid a new top-level dependency:

```python
    async def llm_chat(
        self,
        prompt: str,
        *,
        route: str | None = None,
        context: Any = None,
        schema: str | None = None,
        messages: Any = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        route_key = (route or self.llm_route).strip().lower()
        result = await llm_tools.llm_chat_route(
            self.bus,
            prompt=prompt,
            route=route_key,
            correlation_id=self.request.correlation_id,
            session_id=self.request.session_id,
            user_id=self.request.user_id,
            context=context,
            schema=schema,
            messages=messages,
            stop=stop,
        )
        self.llm_rpc_calls.append({"route": route_key, "prompt": prompt, "result": result})
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_organ_runtime_llm_chat_passes_messages_and_stop -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-context-exec/app/organ_runtime.py services/orion-context-exec/tests/test_agent_repl_runner.py
git commit -m "feat(context-exec): OrganRuntime.llm_chat forwards messages and stop"
```

---

## Task 4: Fix the smolagents model wrapper (roles + stop) and thread budgets

`OrionSmolagentsModel.generate` currently flattens messages to a single string, hardcodes `timeout=120`, and drops `stop_sequences`. Fix it to preserve roles, forward stop sequences, and honor a configurable per-step timeout. Also make `SmolagentsCodeEngine` read `max_steps` and per-step timeout from settings and accept `step_callbacks`.

**Files:**
- Modify: `services/orion-context-exec/app/settings.py:63,166` (budget defaults + new setting)
- Modify: `services/orion-context-exec/app/smolcode_engine.py:20-193`
- Test: `services/orion-context-exec/tests/test_smolcode_engine.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-context-exec/tests/test_smolcode_engine.py`:

```python
@pytest.mark.asyncio
async def test_model_preserves_roles_and_forwards_stop():
    from app.smolcode_engine import OrionSmolagentsModel

    runtime = _make_runtime()
    loop = asyncio.get_running_loop()
    model = OrionSmolagentsModel(runtime, loop, per_step_timeout=7.0)

    messages = [
        {"role": "system", "content": "You are a codebase agent."},
        {"role": "user", "content": "find the bug"},
    ]
    await loop.run_in_executor(
        None, lambda: model.generate(messages, stop_sequences=["<end_code>"])
    )

    runtime.llm_chat.assert_awaited_once()
    _, kwargs = runtime.llm_chat.call_args
    assert kwargs.get("route") == "agent"
    sent = kwargs.get("messages")
    assert [m.role for m in sent] == ["system", "user"]
    assert kwargs.get("stop") == ["<end_code>"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_smolcode_engine.py::test_model_preserves_roles_and_forwards_stop -q`
Expected: FAIL — `OrionSmolagentsModel.__init__` takes no `per_step_timeout`; `llm_chat` called without `messages`/`stop`.

- [ ] **Step 3: Add budget settings**

In `services/orion-context-exec/app/settings.py`, change the default for `context_exec_max_seconds` and `context_exec_llm_timeout_sec`, and add a max-steps setting:

Change line 63:

```python
    context_exec_max_seconds: float = Field(600.0, alias="CONTEXT_EXEC_MAX_SECONDS")
```

Change line 166:

```python
    context_exec_llm_timeout_sec: float = Field(120.0, alias="CONTEXT_EXEC_LLM_TIMEOUT_SEC")
```

Add after line 196 (`context_exec_rlm_fallback_enabled`):

```python
    context_exec_agent_repl_max_steps: int = Field(12, alias="CONTEXT_EXEC_AGENT_REPL_MAX_STEPS")
```

- [ ] **Step 4: Rewrite the model wrapper + engine**

In `services/orion-context-exec/app/smolcode_engine.py`:

Replace the `_messages_to_prompt` helper with a role-preserving converter to `LLMMessage`:

```python
from orion.core.bus.bus_schemas import LLMMessage


def _to_llm_messages(messages: list) -> tuple[list[LLMMessage], str]:
    """Convert smolagents messages to Orion LLMMessage list, preserving roles.

    Returns (messages, last_user_text) — last_user_text feeds raw_user_text/telemetry.
    """
    out: list[LLMMessage] = []
    last_user = ""
    for msg in messages:
        if hasattr(msg, "role"):
            role = str(msg.role.value if hasattr(msg.role, "value") else msg.role)
            raw = msg.content or ""
        elif isinstance(msg, dict):
            role = str(msg.get("role", "user"))
            raw = msg.get("content", "")
        else:
            continue
        content = raw if isinstance(raw, str) else " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        )
        role = role if role in {"system", "user", "assistant", "tool"} else "user"
        out.append(LLMMessage(role=role, content=content))
        if role == "user":
            last_user = content
    if not out:
        out = [LLMMessage(role="user", content="")]
    return out, last_user
```

Replace `OrionSmolagentsModel`:

```python
class OrionSmolagentsModel(Model):
    """smolagents Model wrapper that calls organ_runtime.llm_chat via agent lane."""

    def __init__(
        self,
        runtime: OrganRuntime,
        loop: asyncio.AbstractEventLoop,
        *,
        per_step_timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._loop = loop
        self._per_step_timeout = float(
            per_step_timeout if per_step_timeout is not None else settings.context_exec_llm_timeout_sec
        )

    def generate(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        response_format: object = None,
        tools_to_call_from: object = None,
        **kwargs: object,
    ) -> ChatMessage:
        llm_messages, last_user = _to_llm_messages(messages)
        future = asyncio.run_coroutine_threadsafe(
            self._runtime.llm_chat(
                last_user,
                route="agent",
                messages=llm_messages,
                stop=list(stop_sequences) if stop_sequences else None,
            ),
            self._loop,
        )
        result = future.result(timeout=self._per_step_timeout)
        content = result.get("content") or ""
        return ChatMessage(role="assistant", content=content)
```

Replace `SmolagentsCodeEngine.run` to accept `step_callbacks`, `max_steps`, and `per_step_timeout`:

```python
class SmolagentsCodeEngine(RLMEngine):
    """REPL-based reasoning engine using smolagents CodeAgent + local coder model."""

    engine_name = "smolcode"

    async def run(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        *,
        organ_runtime: OrganRuntime | None = None,
        step_callbacks: list | None = None,
        max_steps: int | None = None,
        per_step_timeout: float | None = None,
    ) -> Any:
        if organ_runtime is None:
            return {
                "error": "organ_runtime required for smolcode engine",
                "engine": "smolcode",
                "mode": request.mode,
            }

        from smolagents import CodeAgent  # lazy import — only loaded when engine is selected

        loop = asyncio.get_running_loop()
        tools = _make_tools(organ_runtime, loop)
        model = OrionSmolagentsModel(organ_runtime, loop, per_step_timeout=per_step_timeout)
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=int(max_steps or settings.context_exec_agent_repl_max_steps),
            step_callbacks=list(step_callbacks) if step_callbacks else None,
        )

        try:
            result = await loop.run_in_executor(None, agent.run, request.text)
            return {
                "summary": str(result),
                "mode": request.mode,
                "engine": "smolcode",
            }
        except Exception as exc:
            logger.error("smolcode engine failed: %s", exc, exc_info=True)
            return {
                "error": str(exc),
                "mode": request.mode,
                "engine": "smolcode",
            }
```

- [ ] **Step 5: Run the failing test + the full module**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_smolcode_engine.py -q`
Expected: PASS — the new test plus the 6 existing (the existing `test_smolcode_model_calls_agent_lane` asserts `kwargs.get("route") == "agent"`, which still holds).

Note: if `test_smolcode_model_calls_agent_lane` asserts `result.content == "agent response"`, it still passes because `_make_runtime()` returns `{"content": "agent response"}`.

- [ ] **Step 6: Commit**

```bash
git add services/orion-context-exec/app/settings.py services/orion-context-exec/app/smolcode_engine.py services/orion-context-exec/tests/test_smolcode_engine.py
git commit -m "feat(context-exec): smolagents wrapper preserves roles, honors stop, uses configurable budgets"
```

---

## Task 5: Add the `agent_step` event emitter

Emit one bus event per reasoning step on the existing `CHANNEL_CONTEXT_EXEC_EVENT`, tagged with `correlation_id`. This is the server-side visibility Gate 1 requires (readable in logs/trace) and the source the Hub relay subscribes to in Gate 2.

**Files:**
- Modify: `services/orion-context-exec/app/events.py:101-107`
- Test: `services/orion-context-exec/tests/test_agent_repl_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-context-exec/tests/test_agent_repl_runner.py`:

```python
@pytest.mark.asyncio
async def test_event_emitter_agent_step_publishes():
    from app.events import ContextExecEventEmitter

    published = []

    class FakeBus:
        async def publish(self, channel, env):
            published.append((channel, env.kind, env.payload))

    import app.events as ev
    ev.settings.orion_bus_enabled = True

    emitter = ContextExecEventEmitter(FakeBus(), correlation_id="corr-1")
    await emitter.agent_step(
        run_id="r1",
        mode="agent_repl",
        step_index=0,
        thought="I will grep the repo",
        tool_id="python_interpreter",
        tool_args="repo_grep('runtime')",
        observation="services/orion-hub/... matched",
        duration_ms=1234,
        is_final=False,
    )
    assert published, "no event published"
    channel, kind, payload = published[0]
    assert kind == "context.exec.agent_step.v1"
    assert payload["step_index"] == 0
    assert payload["tool_id"] == "python_interpreter"
    assert payload["correlation_id"] == "corr-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_event_emitter_agent_step_publishes -q`
Expected: FAIL — `AttributeError: 'ContextExecEventEmitter' object has no attribute 'agent_step'`.

- [ ] **Step 3: Add the `agent_step` method**

In `services/orion-context-exec/app/events.py`, add after `verb_step` (line 107):

```python
    async def agent_step(
        self,
        *,
        run_id: str,
        mode: ContextExecMode,
        step_index: int,
        thought: str,
        tool_id: str,
        tool_args: str,
        observation: str,
        duration_ms: int,
        is_final: bool,
    ) -> None:
        await self.publish(
            "context.exec.agent_step.v1",
            run_id=run_id,
            mode=mode,
            payload={
                "step_index": step_index,
                "thought": (thought or "")[:2000],
                "tool_id": tool_id,
                "tool_args": (tool_args or "")[:2000],
                "observation": (observation or "")[:2000],
                "duration_ms": int(duration_ms or 0),
                "is_final": bool(is_final),
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_event_emitter_agent_step_publishes -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-context-exec/app/events.py services/orion-context-exec/tests/test_agent_repl_runner.py
git commit -m "feat(context-exec): emit context.exec.agent_step.v1 events per reasoning step"
```

---

## Task 6: Runner dispatch — `_run_agent_repl`

Add a dedicated dispatch branch that constructs `OrganRuntime`/namespace, runs `SmolagentsCodeEngine` directly (not via the RLM registry / `alexzhang` default), wires a step callback that emits events + builds `verb_trace`, bounds the whole episode by `context_exec_max_seconds`, and puts the loop's `final_answer` straight into `final_text`. No synthesis, no finalize, no canned apology.

**Files:**
- Modify: `services/orion-context-exec/app/runner.py:354-363` (dispatch) and add method near `_run_investigation_v2`
- Test: `services/orion-context-exec/tests/test_agent_repl_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-context-exec/tests/test_agent_repl_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_agent_repl_returns_final_answer_as_final_text(monkeypatch):
    from app import runner as runner_mod
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None):
            # Simulate one emitted step via the callback, then a final answer.
            if step_callbacks:
                class _Step:
                    step_number = 0
                    is_final_answer = False
                    model_output = "let me look"
                    code_action = "repo_list('services')"
                    observations = "orion-hub/"
                    error = None

                    class timing:
                        duration = 0.5
                step_callbacks[0](_Step())
            return {"summary": "orion-hub is the operator UI + chat gateway.",
                    "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)

    req = ContextExecRequestV1(
        text="what does orion-hub do?",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.mode == "agent_repl"
    assert run.status == "ok"
    assert run.final_text == "orion-hub is the operator UI + chat gateway."
    # step callback populated the visible trace
    assert any(s.callable == "python_interpreter" for s in run.verb_trace)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py::test_run_agent_repl_returns_final_answer_as_final_text -q`
Expected: FAIL — runner has no `agent_repl` dispatch; falls through to the generic RLM path and does not return the stub's summary as `final_text`.

- [ ] **Step 3: Add the dispatch branch**

In `services/orion-context-exec/app/runner.py`, in `run()`, immediately after the `investigation_v2` branch (after line 363), add:

```python
        if request.mode == "agent_repl":
            return await self._run_agent_repl(
                request=request,
                run_id=run_id,
                started=started,
                events=events,
                verb_trace=verb_trace,
                failure_modes=failure_modes,
                workspace_info=workspace_info,
            )
```

- [ ] **Step 4: Add the `_run_agent_repl` method**

Add this method to `ContextExecRunner` (e.g. after `_run_investigation_v2`). It reuses the engine passed to the runner when it is the smolcode engine, otherwise builds one:

```python
    async def _run_agent_repl(
        self,
        *,
        request: ContextExecRequestV1,
        run_id: str,
        started: float,
        events: ContextExecEventEmitter,
        verb_trace: list[ContextExecVerbStepV1],
        failure_modes: list[str],
        workspace_info: dict[str, Any] | None = None,
    ) -> ContextExecRunV1:
        from .smolcode_engine import SmolagentsCodeEngine

        try:
            profile_selection = await resolve_llm_profile(request.llm_profile)
        except LLMProfileUnavailableError as exc:
            failure_modes.append(str(exc))
            profile_selection = LLMProfileSelection(
                requested=request.llm_profile,
                selected="agent",
                route_used="agent",
            )

        request = request.model_copy(update={"llm_profile": profile_selection.selected})
        organ_runtime = OrganRuntime(
            bus=self.rpc_bus,
            request=request,
            run_id=run_id,
            llm_route=profile_selection.route_used,
        )
        namespace = self._build_namespace(organ_runtime)
        organ_status = getattr(namespace, "_organ_status", {}) or {}

        loop = asyncio.get_running_loop()

        def _step_callback(memory_step: Any) -> None:
            # Runs in the executor thread while the main loop awaits run_in_executor.
            try:
                idx = int(getattr(memory_step, "step_number", len(verb_trace)))
                thought = str(getattr(memory_step, "model_output", "") or "")
                code = str(getattr(memory_step, "code_action", "") or "")
                obs = str(getattr(memory_step, "observations", "") or "")
                err = getattr(memory_step, "error", None)
                is_final = bool(getattr(memory_step, "is_final_answer", False))
                timing = getattr(memory_step, "timing", None)
                dur_ms = int(float(getattr(timing, "duration", 0.0) or 0.0) * 1000)
                status = "error" if err else "ok"
                step = ContextExecVerbStepV1(
                    step_index=idx,
                    verb="agent_step",
                    callable="python_interpreter",
                    input_summary=code[:2000] or thought[:2000],
                    output_summary=(str(err) if err else obs)[:2000],
                    status=status,
                    duration_ms=dur_ms,
                )
                verb_trace.append(step)
                asyncio.run_coroutine_threadsafe(
                    events.agent_step(
                        run_id=run_id,
                        mode=request.mode,
                        step_index=idx,
                        thought=thought,
                        tool_id="python_interpreter",
                        tool_args=code,
                        observation=str(err) if err else obs,
                        duration_ms=dur_ms,
                        is_final=is_final,
                    ),
                    loop,
                )
            except Exception:  # never break the loop on telemetry failure
                logger.warning("agent_repl step_callback failed run_id=%s", run_id, exc_info=True)

        engine = self.engine if getattr(self.engine, "engine_name", "") == "smolcode" else SmolagentsCodeEngine()

        status = "ok"
        result: dict[str, Any]
        try:
            result = await asyncio.wait_for(
                engine.run(
                    request,
                    namespace,
                    organ_runtime=organ_runtime,
                    step_callbacks=[_step_callback],
                    max_steps=settings.context_exec_agent_repl_max_steps,
                    per_step_timeout=settings.context_exec_llm_timeout_sec,
                ),
                timeout=settings.context_exec_max_seconds,
            )
        except asyncio.TimeoutError:
            status = "timeout"
            failure_modes.append("timeout")
            result = {"error": "agent reasoning loop exceeded time budget", "engine": "smolcode"}

        if isinstance(result, dict) and result.get("error") and status == "ok":
            status = "error"
            failure_modes.append("agent_repl_error")

        if status == "ok":
            final_text = str(result.get("summary") or "").strip() or "The agent completed without a final answer."
        else:
            # No canned relational apology on this lane; plain diagnostic text only.
            final_text = f"Agent reasoning loop did not complete: {result.get('error') or status}"

        runtime_debug = self._engine_runtime_debug(
            engine_used="smolcode",
            mode=request.mode,
            subcalls=len(organ_runtime.llm_rpc_calls),
        )
        runtime_debug.update(selection_runtime_debug(profile_selection))
        self._apply_workspace_debug(runtime_debug, workspace_info)
        runtime_debug["organ_status"] = organ_status
        runtime_debug["correlation_id"] = request.correlation_id
        runtime_debug["agent_repl"] = True
        runtime_debug["step_count"] = len(verb_trace)

        operator_summary = ContextExecOperatorSummaryV1(
            title="Agent reasoning loop",
            summary=final_text,
            agent_mode="agent_repl",
            route_used=profile_selection.route_used,
            model_synthesis_used=False,
            safety=ContextExecSafetySummaryV1(),
        )

        await events.finished(
            run_id=run_id,
            mode=request.mode,
            status=status,
            artifact_type=None,
            schema_valid=True,
            failure_modes=failure_modes,
        )
        run = ContextExecRunV1(
            run_id=run_id,
            status=status,  # type: ignore[arg-type]
            mode=request.mode,
            text=request.text,
            answer_contract=request.answer_contract.model_dump(mode="json") if request.answer_contract else None,
            findings_bundle=None,
            artifact_type=None,
            artifact={},
            final_text=final_text,
            verb_trace=verb_trace,
            operator_summary=operator_summary,
            runtime_debug=runtime_debug,
            failure_modes=failure_modes,
        )
        self._persist_run_ledger(run, request)
        return run
```

- [ ] **Step 5: Run the test + regression**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_agent_repl_runner.py -q`
Expected: PASS (all tests in the file)

Run: `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-context-exec/tests/test_smolcode_engine.py -q`
Expected: PASS (regression)

- [ ] **Step 6: Compile check**

Run: `./venv/bin/python -m compileall services/orion-context-exec/app/runner.py`
Expected: exit 0

- [ ] **Step 7: Commit**

```bash
git add services/orion-context-exec/app/runner.py services/orion-context-exec/tests/test_agent_repl_runner.py
git commit -m "feat(context-exec): add _run_agent_repl dispatch running smolcode loop directly"
```

---

## Task 7: Hub bridge routes the agent lane to `agent_repl` (no keyword matcher)

The agent lane must build `mode="agent_repl"` with `agent` permissions and a loop-sized budget. It must NOT call `_infer_context_exec_mode` or `investigation_v2`. A boolean flag `HUB_AGENT_REPL_ENABLED` (default true) selects the new path and allows rollback to the legacy path during Gate 1 proof.

**Files:**
- Modify: `services/orion-hub/app/settings.py:92-108`
- Modify: `services/orion-hub/scripts/context_exec_agent_bridge.py:157-203`
- Test: `services/orion-hub/tests/test_agent_repl_bridge.py` (create)

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_agent_repl_bridge.py`:

```python
"""Agent lane builds an agent_repl request with no keyword classification."""
from __future__ import annotations

import pytest

from scripts.context_exec_agent_bridge import build_context_exec_request
from scripts.settings import settings
from orion.schemas.cortex.contracts import CortexChatRequest


def _req(text: str) -> CortexChatRequest:
    return CortexChatRequest(mode="agent", session_id="s1", user_id="u1", trace_id="t1")


def test_agent_repl_request_is_built_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    body = build_context_exec_request(req=_req("x"), prompt="what breaks if I change the runtime?", llm_profile="agent")
    assert body.mode == "agent_repl"
    # Ceiling permissions granted, write/network stay off.
    assert body.permissions.read_repo is True
    assert body.permissions.read_recall is True
    assert body.permissions.write_repo is False
    assert body.permissions.network_enabled is False
    # Loop-sized budget.
    assert body.budget.max_seconds >= 600


def test_agent_repl_does_not_use_keyword_mode(monkeypatch):
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    # A prompt full of legacy keyword triggers must still route to agent_repl.
    body = build_context_exec_request(
        req=_req("x"),
        prompt="trace corr fail open repo impact patch proposal memory correction",
        llm_profile="agent",
    )
    assert body.mode == "agent_repl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_repl_bridge.py -q`
Expected: FAIL — `AttributeError: ... HUB_AGENT_REPL_ENABLED` and/or `body.mode != "agent_repl"`.

- [ ] **Step 3: Add the Hub setting**

In `services/orion-hub/app/settings.py`, in the context-exec agent lane block (after line 104, before `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED`), add:

```python
    HUB_AGENT_REPL_ENABLED: bool = Field(
        default=True,
        alias="HUB_AGENT_REPL_ENABLED",
    )
    HUB_CONTEXT_EXEC_EVENT_CHANNEL: str = Field(
        default="orion:context_exec:event",
        alias="HUB_CONTEXT_EXEC_EVENT_CHANNEL",
    )
```

And raise the timeout default (line 101-104):

```python
    HUB_CONTEXT_EXEC_TIMEOUT_SEC: float = Field(
        default=600.0,
        alias="HUB_CONTEXT_EXEC_TIMEOUT_SEC",
    )
```

- [ ] **Step 4: Route the agent lane to `agent_repl`**

In `services/orion-hub/scripts/context_exec_agent_bridge.py`, add a helper and branch at the top of `build_context_exec_request` (before the `investigation_v2_enabled()` check). Add the import for the budget type at the top with the other schema imports:

```python
from orion.schemas.context_exec import (
    ContextExecBudgetV1,
    ContextExecPermissionV1,
    ContextExecRequestV1,
    ContextExecRunV1,
    context_exec_permissions_for_llm_profile,
)
```

Add a helper near `investigation_v2_enabled`:

```python
def agent_repl_enabled() -> bool:
    return bool(settings.HUB_AGENT_REPL_ENABLED)
```

At the start of `build_context_exec_request`, after `llm_profile_norm = normalize_llm_profile(llm_profile)`:

```python
    if agent_repl_enabled():
        # First-class agent reasoning loop. No keyword classification on this lane.
        permissions = context_exec_permissions_for_llm_profile("agent")
        return ContextExecRequestV1(
            text=prompt,
            mode="agent_repl",
            session_id=req.session_id,
            user_id=req.user_id,
            correlation_id=req.trace_id,
            messages=list(req.messages or []),
            packs=list(req.packs or []),
            permissions=permissions,
            budget=ContextExecBudgetV1(max_seconds=600.0),
            llm_profile="agent",
        )
```

Note: `llm_profile="agent"` binds the run's default route to the agent lane; the smolagents model wrapper additionally forces `route="agent"` on every step call (Task 4). `llm_profile_norm` remains used by the legacy fallback branch below.

Leave the existing `investigation_v2` and `_infer_context_exec_mode` code in place below it (dormant fallback when `HUB_AGENT_REPL_ENABLED=false`; fully retired in Task 14).

- [ ] **Step 5: Run test to verify it passes**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_repl_bridge.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/app/settings.py services/orion-hub/scripts/context_exec_agent_bridge.py services/orion-hub/tests/test_agent_repl_bridge.py
git commit -m "feat(hub): route agent lane to agent_repl with no keyword classification"
```

---

## Task 8: Budget/env parity + local sync + gateway verify (Gate 1 config)

**Files:**
- Modify: `services/orion-context-exec/.env:27,86` and `.env_example` (matching keys)
- Modify: `services/orion-hub/.env:341` and `.env_example:58` and `docker-compose.yml:83`
- Verify only: `services/orion-llm-gateway/.env:71` (`READ_TIMEOUT_SEC=700`)

- [ ] **Step 1: Raise context-exec budget keys in `.env`**

In `services/orion-context-exec/.env`:
- Change `CONTEXT_EXEC_MAX_SECONDS=45` → `CONTEXT_EXEC_MAX_SECONDS=600`
- Change `CONTEXT_EXEC_LLM_TIMEOUT_SEC=30` → `CONTEXT_EXEC_LLM_TIMEOUT_SEC=120`
- Add under the LLM profile block: `CONTEXT_EXEC_AGENT_REPL_MAX_STEPS=12`

- [ ] **Step 2: Mirror in context-exec `.env_example`**

Apply the identical three edits to `services/orion-context-exec/.env_example` (keep any surrounding comments in sync).

- [ ] **Step 3: Raise Hub timeout + add flags in `.env`**

In `services/orion-hub/.env`:
- Change `HUB_CONTEXT_EXEC_TIMEOUT_SEC=120` → `HUB_CONTEXT_EXEC_TIMEOUT_SEC=600`
- Add `HUB_AGENT_REPL_ENABLED=true`
- Add `HUB_CONTEXT_EXEC_EVENT_CHANNEL=orion:context_exec:event`

- [ ] **Step 4: Mirror in Hub `.env_example` + docker-compose**

- Apply the same three edits to `services/orion-hub/.env_example` (near line 56-58).
- In `services/orion-hub/docker-compose.yml`, change line 83 to `- HUB_CONTEXT_EXEC_TIMEOUT_SEC=${HUB_CONTEXT_EXEC_TIMEOUT_SEC:-600}` and add:

```yaml
      - HUB_AGENT_REPL_ENABLED=${HUB_AGENT_REPL_ENABLED:-true}
      - HUB_CONTEXT_EXEC_EVENT_CHANNEL=${HUB_CONTEXT_EXEC_EVENT_CHANNEL:-orion:context_exec:event}
```

- [ ] **Step 5: Sync local env**

Run: `python scripts/sync_local_env_from_example.py`
Expected: exit 0; reports keys synced into `services/*/.env`.

- [ ] **Step 6: Verify gateway per-step budget**

Run: `PYTHONPATH=. ./venv/bin/python -c "import re,sys; t=open('services/orion-llm-gateway/.env').read(); m=re.search(r'READ_TIMEOUT_SEC=(\d+)', t); v=int(m.group(1)) if m else 0; print('READ_TIMEOUT_SEC', v); sys.exit(0 if v>=120 else 1)"`
Expected: prints `READ_TIMEOUT_SEC 700`, exit 0 (≥ per-step 120s). If < 120, raise it and mirror to `.env_example`.

- [ ] **Step 7: Commit**

```bash
git add services/orion-context-exec/.env services/orion-context-exec/.env_example services/orion-hub/.env services/orion-hub/.env_example services/orion-hub/docker-compose.yml
git commit -m "chore(env): raise agent-lane budgets and add agent_repl flags (env parity)"
```

---

## Task 9: Gate 1 live-proof verification script

Per `AGENTS.md` closure: a mocked harness does not count. This script runs a real diagnostic question against the live context-exec `/context-exec/run` endpoint in `agent_repl` mode and prints the step trace, timings, parse success, and grounding evidence.

**Files:**
- Create: `services/orion-context-exec/scripts/verify_agent_repl_live.py`

- [ ] **Step 1: Write the verification script**

Create `services/orion-context-exec/scripts/verify_agent_repl_live.py`:

```python
"""Gate 1 live proof: run agent_repl against the live stack and print the step trace.

Usage:
  ./venv/bin/python services/orion-context-exec/scripts/verify_agent_repl_live.py \
      --url http://127.0.0.1:8096 \
      --text "what would happen if we changed the orion-hub runtime?"
Exit 0 iff: status==ok, >=2 steps, non-empty grounded final_text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--text", default="what would happen if we changed the orion-hub runtime?")
    ap.add_argument("--timeout", type=float, default=650.0)
    args = ap.parse_args()

    body = {
        "text": args.text,
        "mode": "agent_repl",
        "llm_profile": "agent",
        "permissions": {"read_repo": True, "read_recall": True},
    }
    started = time.time()
    resp = requests.post(f"{args.url}/context-exec/run", json=body, timeout=args.timeout)
    elapsed = time.time() - started
    resp.raise_for_status()
    run = resp.json()

    steps = run.get("verb_trace") or []
    print(f"status={run.get('status')} mode={run.get('mode')} elapsed={elapsed:.1f}s steps={len(steps)}")
    for s in steps:
        print(f"  [{s.get('step_index')}] {s.get('callable')} status={s.get('status')} "
              f"dur_ms={s.get('duration_ms')} in={str(s.get('input_summary'))[:80]!r} "
              f"out={str(s.get('output_summary'))[:80]!r}")
    print("final_text:")
    print(run.get("final_text"))
    print("runtime_debug:", json.dumps(run.get("runtime_debug", {}), indent=2)[:1500])

    ok = (
        run.get("status") == "ok"
        and len(steps) >= 2
        and bool(str(run.get("final_text") or "").strip())
    )
    print("GATE1_PASS" if ok else "GATE1_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Bring up / confirm the live stack**

Ensure `orion-context-exec`, `orion-llm-gateway`, and the `atlas-worker-1` backend are running with the new env. If using compose:

Run: `docker compose --env-file .env --env-file services/orion-context-exec/.env -f services/orion-context-exec/docker-compose.yml up -d`
Then confirm health: `curl -s http://127.0.0.1:8096/health | head -c 400`
Expected: `{"ok": true, ...}`

- [ ] **Step 3: Run the live proof**

Run: `./venv/bin/python services/orion-context-exec/scripts/verify_agent_repl_live.py`
Expected (Gate 1 acceptance): completes < 10 min, prints a multi-step trace (`steps >= 2`) with `python_interpreter` tool calls and observations, a grounded `final_text`, and `GATE1_PASS`.

Inspect specifically:
- **Code-format parse success:** steps show real `input_summary` code and non-empty `output_summary` observations (not repeated parse errors).
- **Per-step + total latency:** `dur_ms` per step and total `elapsed`.
- **Grounding:** `final_text` references content the tools actually returned.

**If GATE1_FAIL** (model won't drive the smolagents code format): STOP. Do not proceed to Gate 2. Remediation is model/prompt/GBNF grammar via the `PUBLISH_CORTEX_EXEC_GRAMMAR` hook — record findings and open a follow-up. This is the highest-risk gate.

- [ ] **Step 4: Commit the script**

```bash
git add services/orion-context-exec/scripts/verify_agent_repl_live.py
git commit -m "test(context-exec): Gate 1 live-proof script for agent_repl reasoning loop"
```

---

# GATE 2 — expose (live step streaming) + retire investigation_v2

> Only start Gate 2 after Task 9 prints `GATE1_PASS` against the live stack.

## Task 10: Hub agent-step relay (long-lived bus subscriber)

A long-lived subscriber to `HUB_CONTEXT_EXEC_EVENT_CHANNEL` that fans `context.exec.agent_step.v1` events out to per-correlation `asyncio.Queue`s. Mirrors the existing `NotificationCache` register/unregister pattern, avoiding the pub/sub subscribe race.

**Files:**
- Create: `services/orion-hub/scripts/agent_step_relay.py`
- Test: `services/orion-hub/tests/test_agent_step_relay.py` (create)

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_agent_step_relay.py`:

```python
"""AgentStepRelay fans agent_step events to per-correlation queues."""
from __future__ import annotations

import asyncio

import pytest

from scripts.agent_step_relay import AgentStepRelay


@pytest.mark.asyncio
async def test_relay_dispatches_to_registered_queue():
    relay = AgentStepRelay(channel="orion:context_exec:event")
    q: asyncio.Queue = asyncio.Queue()
    relay.register_queue("corr-1", q)

    await relay._dispatch_payload(
        kind="context.exec.agent_step.v1",
        payload={"correlation_id": "corr-1", "step_index": 0, "tool_id": "python_interpreter"},
    )
    item = q.get_nowait()
    assert item["kind"] == "agent_step"
    assert item["step"]["step_index"] == 0

    relay.unregister_queue("corr-1", q)


@pytest.mark.asyncio
async def test_relay_ignores_non_step_kinds_and_unknown_corr():
    relay = AgentStepRelay(channel="orion:context_exec:event")
    q: asyncio.Queue = asyncio.Queue()
    relay.register_queue("corr-1", q)

    await relay._dispatch_payload(kind="context.exec.finished.v1",
                                  payload={"correlation_id": "corr-1"})
    await relay._dispatch_payload(kind="context.exec.agent_step.v1",
                                  payload={"correlation_id": "other"})
    assert q.empty()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_step_relay.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.agent_step_relay`.

- [ ] **Step 3: Implement the relay**

Create `services/orion-hub/scripts/agent_step_relay.py`:

```python
"""Relay context-exec agent_step bus events to per-correlation WebSocket queues."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion-hub.agent_step_relay")

AGENT_STEP_KIND = "context.exec.agent_step.v1"


class AgentStepRelay:
    def __init__(self, *, channel: str) -> None:
        self.channel = channel
        self._bus: Optional[OrionBusAsync] = None
        self._task: Optional[asyncio.Task] = None
        self._queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    async def start(self, bus: OrionBusAsync) -> None:
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-agent-step-relay")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def register_queue(self, correlation_id: str, queue: asyncio.Queue) -> None:
        self._queues[str(correlation_id)].add(queue)

    def unregister_queue(self, correlation_id: str, queue: asyncio.Queue) -> None:
        cid = str(correlation_id)
        self._queues.get(cid, set()).discard(queue)
        if cid in self._queues and not self._queues[cid]:
            self._queues.pop(cid, None)

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info("Subscribing to context-exec agent steps: %s", self.channel)
        try:
            async with self._bus.subscribe(self.channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    decoded = self._bus.codec.decode(msg.get("data"))
                    if not decoded.ok:
                        continue
                    env = decoded.envelope
                    await self._dispatch_payload(kind=str(env.kind), payload=env.payload or {})
        except asyncio.CancelledError:
            logger.info("Agent step relay cancelled.")
        except Exception as exc:
            logger.error("Agent step relay loop failed: %s", exc, exc_info=True)

    async def _dispatch_payload(self, *, kind: str, payload: dict[str, Any]) -> None:
        if kind != AGENT_STEP_KIND:
            return
        cid = str(payload.get("correlation_id") or "")
        queues = self._queues.get(cid)
        if not queues:
            return
        item = {"kind": "agent_step", "correlation_id": cid, "step": payload}
        for q in list(queues):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_step_relay.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/agent_step_relay.py services/orion-hub/tests/test_agent_step_relay.py
git commit -m "feat(hub): AgentStepRelay fans context-exec step events to per-correlation queues"
```

---

## Task 11: Wire the relay into Hub startup

The Hub exposes caches to `websocket_handler` as module-level globals in `scripts/main.py` (declared at module scope ~line 132-133, assigned inside the lifespan via `global`, read as `scripts.main.<name>` in `websocket_endpoint`). Follow that exact pattern for the relay.

**Files:**
- Modify: `services/orion-hub/scripts/main.py` (module globals ~132, imports, lifespan startup ~211/245, shutdown ~435)

- [ ] **Step 1: Declare the module-level global + import**

In `services/orion-hub/scripts/main.py`, add near the other cache imports (top of file, where `NotificationCache` is imported):

```python
from scripts.agent_step_relay import AgentStepRelay
```

Add the module-level global after line 133 (`notification_cache: Optional[NotificationCache] = None`):

```python
agent_step_relay: Optional[AgentStepRelay] = None
```

- [ ] **Step 2: Add to the `global` declarations**

Append `agent_step_relay` to both `global` statements:
- Line 211 (`global bus, rpc_bus, ... notification_cache, signals_inspect_cache, cognition_trace_cache, ...`) → add `, agent_step_relay`.
- Line 435 (shutdown `global bus, rpc_bus, biometrics_cache, notification_cache, ...`) → add `, agent_step_relay`.

- [ ] **Step 3: Start the relay after the notification cache**

In the lifespan startup, immediately after the `notification_cache` block (line ~250), add:

```python
            agent_step_relay = AgentStepRelay(channel=settings.HUB_CONTEXT_EXEC_EVENT_CHANNEL)
            await agent_step_relay.start(bus)
```

- [ ] **Step 4: Stop on shutdown**

In the shutdown section (where `notification_cache` is stopped, near line 435+), add:

```python
            if agent_step_relay is not None:
                try:
                    await agent_step_relay.stop()
                except Exception:
                    pass
```

- [ ] **Step 5: Compile check**

Run: `./venv/bin/python -m compileall services/orion-hub/scripts/main.py`
Expected: exit 0

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/main.py
git commit -m "feat(hub): start/stop AgentStepRelay in app lifespan (module-global pattern)"
```

---

## Task 12: Relay live steps to the browser during an agent_repl turn

While `run_hub_agent_via_context_exec` awaits the HTTP run, drain the per-correlation queue and push `agent_step` frames to the browser WebSocket. The relay subscription is always live, so registering a queue before issuing the run avoids the pub/sub race.

**Files:**
- Modify: `services/orion-hub/scripts/websocket_handler.py:960-990`
- Test: manual/live (Task 15); structural compile check here.

- [ ] **Step 1: Resolve the relay global at the top of `websocket_endpoint`**

In `services/orion-hub/scripts/websocket_handler.py`, next to the existing `biometrics_cache = scripts.main.biometrics_cache` / `notification_cache = scripts.main.notification_cache` assignments (lines 501-502), add:

```python
    agent_step_relay = scripts.main.agent_step_relay
```

- [ ] **Step 2: Register a queue and drain concurrently around the run**

In the same file, replace the agent-lane block (the `if used_context_exec_lane:` body around line 965) so it wraps the run with a queue drain, using the `agent_step_relay` resolved above:

```python
                if used_context_exec_lane:
                    step_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
                    relay = agent_step_relay
                    drain_task = None
                    if relay is not None:
                        relay.register_queue(trace_id, step_queue)

                        async def _drain_steps() -> None:
                            try:
                                while True:
                                    item = await step_queue.get()
                                    await _safe_ws_send_json(websocket, item)
                            except asyncio.CancelledError:
                                pass

                        drain_task = asyncio.create_task(_drain_steps(), name=f"agent-steps-{trace_id}")
                    try:
                        ctx_out = await run_hub_agent_via_context_exec(
                            req=chat_req,
                            prompt=transcript or prompt_with_ctx,
                            correlation_id=trace_id,
                            route_debug=route_debug if isinstance(route_debug, dict) else {},
                        )
                    finally:
                        if drain_task is not None:
                            drain_task.cancel()
                            try:
                                await drain_task
                            except asyncio.CancelledError:
                                pass
                        if relay is not None:
                            relay.unregister_queue(trace_id, step_queue)
                    if ctx_out.get("error"):
                        # ... existing error branch unchanged ...
```

Keep the rest of the existing branch (the `ctx_out.get("error")` handling and the `orion_response_text = str(ctx_out.get("llm_response") or "")` assignments) exactly as-is after the `finally`.

- [ ] **Step 3: Ensure `asyncio` and `scripts.main` are imported**

Confirm `import asyncio` and `import scripts.main` (used for `scripts.main.biometrics_cache`) are present at the top of `websocket_handler.py` (grep). Both already exist given the current `scripts.main.biometrics_cache` usage; add only if missing.

- [ ] **Step 4: Compile check**

Run: `./venv/bin/python -m compileall services/orion-hub/scripts/websocket_handler.py`
Expected: exit 0

- [ ] **Step 5: Hub bridge/regression tests**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_repl_bridge.py services/orion-hub/tests/test_agent_step_relay.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/websocket_handler.py
git commit -m "feat(hub): stream agent_repl reasoning steps to the browser during the turn"
```

---

## Task 13: Front-end — render live steps

Handle `agent_step` WS frames and render them into the agent-trace panel as they arrive.

**Files:**
- Modify: `services/orion-hub/static/js/app.js:10481` (socket.onmessage) and the second handler near line 10889
- Modify: `services/orion-hub/static/js/agent-trace.js`

- [ ] **Step 1: Add a live-steps buffer + handler in `app.js`**

In `services/orion-hub/static/js/app.js`, inside `socket.onmessage` (after the `d.kind === 'notification'` branch near line 10556), add:

```javascript
          if (d.kind === 'agent_step' && d.step) {
            try { appendLiveAgentStep(d.correlation_id, d.step); } catch (err) { console.warn('agent_step render failed', err); }
            return;
          }
```

Apply the same branch to the second `onmessage` handler if the file wires two socket paths (search for the duplicate `agentTrace: d.agent_trace` block near line 10889 and add the branch in that handler too).

- [ ] **Step 2: Implement `appendLiveAgentStep` + panel updates in `agent-trace.js`**

In `services/orion-hub/static/js/agent-trace.js`, add an exported (global) function that maintains a per-correlation live-steps container and appends rows. Match the file's existing DOM/util conventions (e.g. how `createAgentTracePanel` builds nodes):

```javascript
const _liveAgentSteps = new Map(); // correlationId -> [step,...]

function appendLiveAgentStep(correlationId, step) {
  if (!correlationId || !step) return;
  const list = _liveAgentSteps.get(correlationId) || [];
  list.push(step);
  _liveAgentSteps.set(correlationId, list);

  let panel = document.getElementById(`agent-live-${correlationId}`);
  if (!panel) {
    panel = document.createElement('div');
    panel.id = `agent-live-${correlationId}`;
    panel.className = 'agent-live-trace';
    const heading = document.createElement('div');
    heading.className = 'agent-live-trace__heading';
    heading.textContent = 'Reasoning steps (live)';
    panel.appendChild(heading);
    // Anchor into the active chat message container; fall back to a known transcript node.
    const anchor = document.getElementById('chat-messages') || document.body;
    anchor.appendChild(panel);
  }

  const row = document.createElement('div');
  row.className = 'agent-live-trace__step' + (step.is_final ? ' is-final' : '');
  const idx = step.step_index != null ? step.step_index : (list.length - 1);
  const dur = step.duration_ms != null ? `${step.duration_ms}ms` : '';
  row.textContent = `#${idx} ${step.tool_id || 'step'} ${dur} — ${String(step.observation || step.thought || '').slice(0, 200)}`;
  panel.appendChild(row);
  panel.scrollTop = panel.scrollHeight;
}

if (typeof window !== 'undefined') {
  window.appendLiveAgentStep = appendLiveAgentStep;
}
```

- [ ] **Step 3: Manual smoke (served page)**

Load the Hub page, open the browser console, and run:

```javascript
appendLiveAgentStep('demo', { step_index: 0, tool_id: 'python_interpreter', duration_ms: 120, observation: 'matched services/orion-hub/...' });
```

Expected: a "Reasoning steps (live)" panel appears with one row. (This confirms the render path before the live end-to-end test in Task 15.)

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/static/js/app.js services/orion-hub/static/js/agent-trace.js
git commit -m "feat(hub-ui): render live agent_repl reasoning steps as they stream in"
```

---

## Task 14: Retire investigation_v2 from the agent lane

Now that `agent_repl` is proven and streaming, make it the sole agent-lane path. Turn the legacy `investigation_v2` branch into dead/dormant code (no other caller invokes it via the lane) and default its flag off for the lane. The `investigation_v2` runner code itself stays (dormant) per the spec.

**Files:**
- Modify: `services/orion-hub/scripts/context_exec_agent_bridge.py:157-203`
- Modify: `services/orion-hub/.env` + `.env_example` (flag)
- Test: `services/orion-hub/tests/test_agent_repl_bridge.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-hub/tests/test_agent_repl_bridge.py`:

```python
def test_agent_lane_never_builds_investigation_v2(monkeypatch):
    # Even if the legacy investigation_v2 flag is on, the agent lane must not use it.
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "CONTEXT_EXEC_INVESTIGATION_V2_ENABLED", True, raising=False)
    body = build_context_exec_request(req=_req("x"), prompt="anything", llm_profile="agent")
    assert body.mode == "agent_repl"
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_repl_bridge.py::test_agent_lane_never_builds_investigation_v2 -q`
Expected: PASS already if Task 7's early return precedes the `investigation_v2_enabled()` check. If it FAILS, the early return is misplaced — move the `agent_repl` block above the `investigation_v2` branch.

- [ ] **Step 3: Make agent_repl unconditional on the lane**

In `services/orion-hub/scripts/context_exec_agent_bridge.py`, remove the `HUB_AGENT_REPL_ENABLED` conditionality on the agent lane (the lane always uses `agent_repl`) OR keep the flag but document that `false` is a temporary rollback only. Recommended: keep the flag for rollback but ensure the `agent_repl` block is the first return in `build_context_exec_request`. Add a comment marking the `investigation_v2` + `_infer_context_exec_mode` code below as dormant/legacy (no agent-lane caller).

- [ ] **Step 4: Default the legacy flag off in Hub env**

In `services/orion-hub/.env` and `.env_example`, set `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED=false` (Hub side; this is the Hub's mirror of the flag). Run `python scripts/sync_local_env_from_example.py`.

- [ ] **Step 5: Confirm no other agent-lane caller uses investigation_v2**

Run: `rg -n "mode=\"investigation_v2\"|_infer_context_exec_mode" services/orion-hub/scripts`
Expected: only references inside the dormant branch of `context_exec_agent_bridge.py` (no live caller). Record the output as evidence.

- [ ] **Step 6: Run bridge tests**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_agent_repl_bridge.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add services/orion-hub/scripts/context_exec_agent_bridge.py services/orion-hub/.env services/orion-hub/.env_example
git commit -m "chore(hub): retire investigation_v2 from the agent lane (agent_repl is sole path)"
```

---

## Task 15: Gate 2 live end-to-end verification

Prove the full path: Hub `mode: agent` → `agent_repl` → live steps in the UI → grounded `final_text`.

**Files:**
- Create: `services/orion-hub/scripts/verify_agent_repl_stream_live.py`

- [ ] **Step 1: Write a WS proof script**

Create `services/orion-hub/scripts/verify_agent_repl_stream_live.py`:

```python
"""Gate 2 live proof: drive the Hub chat WS in agent mode and assert live step frames.

Usage:
  ./venv/bin/python services/orion-hub/scripts/verify_agent_repl_stream_live.py \
      --ws ws://127.0.0.1:8080/ws/chat \
      --text "what would happen if we changed the orion-hub runtime?"
Exit 0 iff: >=2 agent_step frames received AND a final llm_response arrives.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import websockets


async def run(ws_url: str, text: str, timeout: float) -> int:
    steps = 0
    final = ""
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"mode": "agent", "is_text_input": True, "transcript": text}))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                d = json.loads(raw)
                if d.get("kind") == "agent_step":
                    steps += 1
                    s = d.get("step", {})
                    print(f"  step #{s.get('step_index')} {s.get('tool_id')} {s.get('duration_ms')}ms")
                elif d.get("llm_response"):
                    final = str(d.get("llm_response"))
                    break
        except asyncio.TimeoutError:
            print("timeout waiting for frames")

    print(f"steps={steps} final_len={len(final)}")
    print("final_text:", final[:500])
    ok = steps >= 2 and bool(final.strip())
    print("GATE2_PASS" if ok else "GATE2_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8080/ws/chat")
    ap.add_argument("--text", default="what would happen if we changed the orion-hub runtime?")
    ap.add_argument("--timeout", type=float, default=650.0)
    args = ap.parse_args()
    return asyncio.run(run(args.ws, args.text, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
```

Note: confirm the actual Hub chat WS path/message shape by reading `services/orion-hub/scripts/websocket_handler.py` (the handler reads `data.get("transcript")`, `data.get("is_text_input")`, `mode`). Adjust `--ws` path and the send payload to match the served route.

- [ ] **Step 2: Bring up the full stack with new env**

Ensure Hub + context-exec + gateway + worker are running with the new env (rebuild Hub if needed):

Run: `docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build`
Confirm Hub health: `curl -s http://127.0.0.1:8080/health | head -c 300` (adjust port to the Hub's actual health port).

- [ ] **Step 3: Run the Gate 2 proof**

Run: `./venv/bin/python services/orion-hub/scripts/verify_agent_repl_stream_live.py`
Expected: prints `>= 2` `agent_step` lines during the turn, a non-empty `final_text`, and `GATE2_PASS`.

- [ ] **Step 4: Browser confirmation (acceptance check #3)**

Open the Hub UI, switch to `mode: agent`, ask the diagnostic question, and confirm reasoning steps appear live in the trace panel during the turn (not only after completion), and the final chat message is the loop's grounded answer.

- [ ] **Step 5: Assert no canned apology (acceptance check #4)**

Run: `rg -n "relational|apolog|I'm sorry|canned" services/orion-context-exec/app/runner.py services/orion-hub/scripts/context_exec_agent_bridge.py`
Expected: no apology/relational-repair string is reachable on the `agent_repl` path (the finalize canned-apology branch belongs to `_run_investigation_v2`, which the lane no longer calls).

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/verify_agent_repl_stream_live.py
git commit -m "test(hub): Gate 2 live-proof script for agent_repl step streaming"
```

---

## Acceptance checks (map to spec)

1. **Hub `mode: agent` reaches `mode="agent_repl"`; no `investigation_v2` / `_infer_context_exec_mode` on the path.** — Tasks 7, 14; evidence: `test_agent_repl_bridge.py`, `rg` output (Task 14 Step 5), Gate 1 run trace `runtime_debug.agent_repl=true`.
2. **Live-stack run of a real diagnostic question completes < 10 min, multi-step trace, grounded NL answer.** — Task 9 `GATE1_PASS`.
3. **Reasoning steps appear live in Hub UI during the turn.** — Tasks 10–13; Task 15 `GATE2_PASS` + browser confirmation.
4. **No canned-apology string on this lane.** — Task 6 (`_run_agent_repl` emits plain diagnostic text), Task 15 Step 5.
5. **`.env` / `.env_example` parity for every changed key; local env synced.** — Task 8 (+ Task 14 Step 4), `scripts/sync_local_env_from_example.py` runs.

## Non-goals (do not implement)

- Dedicated agent worker or VRAM changes; keep shared `atlas-worker-1`.
- Model swap; keep `Active-GGUF-Model`.
- New tools beyond the existing four (`repo_grep`, `repo_read`, `repo_list`, `recall_query`).
- Any keyword heuristics, repo writes, or network access.
- Cancellation of orphaned worker generation on timeout (accepted trade-off; note only).

## Risks

1. **Local model may not reliably drive smolagents code format** — highest risk; Gate 1 (Task 9) exposes it; remediation is model/prompt/GBNF grammar via `PUBLISH_CORTEX_EXEC_GRAMMAR`. Do not proceed to Gate 2 on `GATE1_FAIL`.
2. **Orphaned work on the shared worker** — a timed-out ~10-min loop leaves the worker generating; next chat turn briefly stalls. Accepted.
3. **Streaming wiring across two services** — most likely scope-creep point; isolated behind Gate 2 (Tasks 10–13). The long-lived relay (register/unregister queues) avoids the pub/sub subscribe race that a per-turn subscription would hit.
