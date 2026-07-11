# context-exec SmolagentsCodeEngine Design

**Date**: 2026-06-18
**Status**: approved for implementation

## Problem

`orion-context-exec` produces shallow, hallucinated answers because all current engines are fake:

- `FakeRLMEngine` — hardcoded strings, no LLM, no tool calls
- `AlexZhangRLMEngine` — zero LLM calls; regex token extraction → hardcoded response dicts
- `investigation_v2` — static probe plan decided before the LLM sees anything; single LLM synthesis pass at the end gets bad evidence and hallucinates

None of these engines let the LLM decide what to search based on what it finds. The reasoning never drives the investigation.

## Goal

Replace the default engine for Agent mode with `SmolagentsCodeEngine`: a real REPL-based reasoning loop where the coder model writes Python code to call read-only organ tools, observes the results, and reasons about what to look at next — until it has enough to answer.

This is the same pattern as Claude Code: think → write code → execute → observe → repeat.

## Non-Goals

- No mutation (no memory writes, no file writes, no repo writes)
- Not replacing `investigation_v2` or `agent_synthesis` — they remain available for non-agent modes
- Not replacing the gateway or bus architecture — the agent calls through the existing `agent` lane
- No streaming output (full result returned at end of loop)
- No multi-agent orchestration — single CodeAgent loop

## Architecture

### Entry Point

`CONTEXT_EXEC_RLM_ENGINE=smolcode` activates the new engine via `build_engine()` in `rlm_engine.py`.

### New File: `smolcode_engine.py`

```
SmolagentsCodeEngine(RLMEngine)
  engine_name = "smolcode"

  async run(request, namespace, *, organ_runtime):
    loop = asyncio.get_event_loop()
    tools = _make_tools(organ_runtime, loop)
    model = OrionSmolagentsModel(organ_runtime, loop)
    agent = CodeAgent(tools=tools, model=model, max_steps=12)
    result = await loop.run_in_executor(None, agent.run, request.text)
    return {"summary": str(result), "mode": request.mode, "engine": "smolcode"}
```

### OrionSmolagentsModel

Custom `smolagents.Model` subclass. Bridges the sync smolagents interface to the async `OrganRuntime.llm_chat()` via `asyncio.run_coroutine_threadsafe()`.

- Route: `"agent"` → llama-cola → Circe node → coder model
- Translates smolagents message list to a single prompt string

```python
def __call__(self, messages, stop_sequences=None, **kwargs):
    prompt = _messages_to_prompt(messages)
    future = asyncio.run_coroutine_threadsafe(
        self._runtime.llm_chat(prompt, route="agent"),
        self._loop,
    )
    result = future.result(timeout=120)
    return ChatMessage(role="assistant", content=result.get("content", ""))
```

### Async Bridge

`SmolagentsCodeEngine.run()` is async (FastAPI context). `CodeAgent.run()` is sync. The bridge:

1. Capture the running event loop before spawning the executor thread
2. Run `CodeAgent.run()` in a thread via `loop.run_in_executor(None, agent.run, request.text)`
3. Inside that thread, use `asyncio.run_coroutine_threadsafe(coro, loop)` to call async organ methods back on the original event loop

This is the correct primitive for "submit async work from a thread to a running event loop."

### Tools Exposed to the Agent

Four tools only. The coder model writes Python calling these by name.

#### `repo_grep(pattern: str, path: str = "") -> str`
Searches the repo with a regex. Returns matching file paths, line numbers, and snippets as formatted text. Wraps `OrganRuntime.repo_grep()` directly (already sync).

#### `repo_read(path: str) -> str`
Reads a file from the repo and returns its content. Wraps `OrganRuntime.repo_read()` directly (already sync).

#### `repo_list(path: str = "") -> str`
Lists files and directories under a repo path. **New tool** — requires adding `repo_list()` to `repo_tools.py`. Without this the agent must grep blindly to discover structure.

#### `recall_query(query: str) -> str`
Semantic search over persisted user context via the recall service. Async — bridged via `run_coroutine_threadsafe`.

**Not included**: `traces_search` — the in-process store is empty on most agent runs and the Redis fallback is just key pattern matching. Noise, not signal. Remains available on `OrganRuntime` for other modes.

### New Function: `repo_tools.repo_list()`

~10 lines added to `repo_tools.py`. Lists allowed files under a path, respecting `DENY_PATTERNS` and `ALLOW_PREFIXES`. Returns a sorted list of relative paths (files and dirs).

### Sandboxing

smolagents `LocalPythonExecutor` restricts available names to only the provided tools plus safe builtins. The coder model cannot `import os`, write files, or call anything outside the tool set. Read-only constraint is structural, not instructed.

## Files Changed

| File | Change |
|------|--------|
| `services/orion-context-exec/app/smolcode_engine.py` | New — `SmolagentsCodeEngine`, `OrionSmolagentsModel`, `_make_tools()` |
| `services/orion-context-exec/app/repo_tools.py` | Add `repo_list()` function |
| `services/orion-context-exec/app/rlm_engine.py` | Add `smolcode` case to `build_engine()` |
| `services/orion-context-exec/requirements.txt` | Add `smolagents` |

## Acceptance Checks

- `CONTEXT_EXEC_RLM_ENGINE=smolcode` activates the engine without error
- Agent makes at least 2 distinct tool calls before returning (not a single grep)
- Returned summary contains actual file paths or content sourced from tool output
- No hallucinated paths in the output (same grounding check `agent_synthesis` already applies)
- Falls back gracefully if the agent lane is down: returns `{"error": "...", "engine": "smolcode"}` not a crash
- `repo_list("")` returns the top-level allowed directories
- `repo_list("services/orion-context-exec")` returns the files in that service
