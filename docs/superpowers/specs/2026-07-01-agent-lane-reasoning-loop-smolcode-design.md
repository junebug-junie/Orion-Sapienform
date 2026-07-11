# Agent-lane reasoning loop (smolcode REPL) — design

**Date:** 2026-07-01
**Status:** Approved design; pending implementation plan
**Services:** `orion-context-exec`, `orion-hub` (+ `orion-llm-gateway` config verify)
**Related:** `.cursor/rules/conversational-behavior-anti-slop.mdc`, `AGENTS.md` (env parity, closure rules)

## Problem

Operator intent: ask Orion a natural-language question about its own codebase/runtime via Hub `mode: agent, compute: chat`, have it **introspect itself** (grep/read repo, query recall), reason across multiple steps, and return a grounded natural-language answer — the way a coding agent diagnoses and reports. Instead, the lane returns shallow, memory-grounded replies or a canned apology.

### Verified root cause (why it fails today)

1. **The real reasoning loop is orphaned by routing.** A working smolagents `CodeAgent` engine exists — `SmolagentsCodeEngine` (`services/orion-context-exec/app/smolcode_engine.py`), `max_steps=12`, tools `repo_grep`/`repo_read`/`repo_list`/`recall_query`, model calls routed through `llm_chat(route="agent")`. But it is only reachable via the RLM engine registry on the **legacy** path (`CONTEXT_EXEC_RLM_ENGINE=smolcode`, `runner._execute_rlm`). With `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED=true` (default), the Hub bridge forces `mode="investigation_v2"` (`services/orion-hub/scripts/context_exec_agent_bridge.py:165-179`) and the runner short-circuits before the engine registry (`services/orion-context-exec/app/runner.py:354-363`). The loop never runs.

2. **The loop has never run against a live model.** Of six tests in `services/orion-context-exec/tests/test_smolcode_engine.py`, four mock `CodeAgent`; the one "real loop" test stubs the LLM to return `final_answer` immediately. The local model has never actually driven the smolagents code-block format end to end.

3. **Keyword heuristics gate the fallback path.** The legacy bridge path classifies intent via `_infer_context_exec_mode` keyword matching (`context_exec_agent_bridge.py:142-154`) and `investigation_v2` uses a keyword `answer_contract`. This violates the anti-slop cardinal rule and misroutes diagnostic questions.

4. **Canned apology on failure.** `investigation_v2`'s finalize pass emits a hardcoded relational-repair string when the LLM pass fails/misclassifies — surfacing failure as a deflection.

5. **Budget/timeout chain is provisioned for single-shot, not a loop.** `CONTEXT_EXEC_MAX_SECONDS=45` bounds the *entire* engine episode; a multi-step loop is aborted after ~1–2 calls. Per-call bus RPC times out at 30s (`CONTEXT_EXEC_LLM_TIMEOUT_SEC`). The `agent` route maps to `atlas-worker-1` — the **same single-slot llama.cpp worker as `chat`** — behind a single-subscriber, blocking gateway, so an agent step head-of-line-blocks chat. There is **no cancellation**: on timeout the worker keeps generating and the reply is orphaned.

## Goal

Make the Hub agent lane run the smolcode reasoning loop as a first-class mode, on a budget/worker configuration that fits a multi-step loop, with live step visibility in Hub, and prove it reasons against the live local model before retiring the old pipeline.

## Operator decisions (fixed)

- **Scope:** productionize smolcode as *the* agent lane **and** retire `investigation_v2` from that lane (options B+C), sequenced behind an early live-proof gate.
- **Compute:** keep the shared `atlas-worker-1` slot; no dedicated agent worker; no VRAM cost. Accept that an agent turn serializes chat while running.
- **Model:** keep the currently loaded `Active-GGUF-Model`; no model swap in this effort.
- **Latency/UX:** allow up to **10 minutes** per turn; **stream reasoning steps live into Hub**.
- **Final text:** the loop's own `final_answer` is the chat response — no synthesis/finalize pass, no canned-apology fallback.

## Architecture

### 1. First-class agent reasoning mode

Introduce a dedicated context-exec mode (working name `agent_repl`) dispatched straight to `SmolagentsCodeEngine`, parallel to the existing `investigation_v2` guard in `ContextExecRunner.run`.

- Bridge: for Hub `mode: agent`, build `ContextExecRequestV1(mode="agent_repl", ...)`. Remove the `investigation_v2` branch and the `_infer_context_exec_mode` keyword matcher from this lane.
- Runner: add `if request.mode == "agent_repl": return await self._run_agent_repl(...)`, which constructs `OrganRuntime`/namespace and runs the smolcode engine directly (not via the `alexzhang`-defaulted RLM registry).
- **Rejected alternative:** `CONTEXT_EXEC_RLM_ENGINE=smolcode` + investigation_v2 off. That still runs `_infer_context_exec_mode` keyword routing and the `alexzhang` default — reintroducing keyword heuristics. The new-mode path carries **zero** keyword classification.

### 2. Permissions (ceiling, not warrant)

Agent-lane requests unconditionally grant `read_repo` + `read_recall`. Permissions remain the hard safety ceiling: `write_enabled=false`, `network_enabled=false`. No `answer_contract` keyword gate on this lane.

### 3. Model wrapper reliability

`OrionSmolagentsModel.generate` currently flattens messages to a single string and drops `stop_sequences` / `response_format`. Fix it to preserve message roles and honor stop sequences so the local model can emit and terminate smolagents code blocks reliably. This is the primary reliability surface; if the model cannot cooperate, remediation is model/prompt/grammar (llama.cpp GBNF via the `PUBLISH_CORTEX_EXEC_GRAMMAR` hook), not more pipeline.

### 4. Budgets (shared worker, 10-min ceiling)

Raise the full chain, with `.env` / `.env_example` parity in the same change set and a local sync run (`scripts/sync_local_env_from_example.py`):

- `CONTEXT_EXEC_MAX_SECONDS`: 45 → ~600.
- `CONTEXT_EXEC_LLM_TIMEOUT_SEC`: 30 → real per-step value (~120s); align smolcode's `future.result(timeout=...)`.
- `HUB_CONTEXT_EXEC_TIMEOUT_SEC`: → ~600s (and cortex-exec budget if it bounds this path).
- Verify the running `orion-llm-gateway` `READ_TIMEOUT_SEC` on `atlas-worker-1` is ≥ per-step budget (example shows 700s; confirm actual `.env`).

### 5. Live step streaming to Hub

- context-exec: register smolagents `step_callbacks`; per step emit one event (step index, thought, tool id + args, observation snippet, timing) on the existing `CHANNEL_CONTEXT_EXEC_EVENT`, tagged with `correlation_id`.
- Hub: subscribe to the context-exec event channel, correlate by `correlation_id`, and relay each step to the browser over the existing chat WebSocket so the operator watches tool calls/observations arrive live.
- Largest new surface; spans both services. Kept behind Gate 2 so Gate 1 can prove reasoning without UI work.

### 6. Final answer

The loop's `final_answer` populates `run.final_text` directly, bypassing the synthesis/grounding override (`runner.py:519-563`) and the finalize pass. The finalize canned-apology branch is removed from this lane. Hub already surfaces `run.final_text` as `llm_response`.

## Data flow (target)

```
Hub mode:agent
  -> bridge: ContextExecRequestV1(mode="agent_repl", read_repo+read_recall)
  -> runner._run_agent_repl
       -> SmolagentsCodeEngine.CodeAgent loop (<=12 steps, <=10 min)
            step: model -> tool(repo_grep|repo_read|repo_list|recall_query) -> observation
            step_callback -> emit context-exec step event (correlation_id)
       -> final_answer -> run.final_text
  -> Hub relays step events to browser WebSocket (live)
  -> Hub renders run.final_text as chat
```

## Sequencing

**Gate 1 — prove it reasons live.** Ship routing + permissions + wrapper fix + budgets + minimal server-side step logging. Run "what would happen if we changed the orion-hub runtime?" against the live stack. Inspect: real step trace, code-format parse success rate, per-step and total latency, and whether the final answer is grounded in tool observations. Proceed only if the loop actually reasons.

**Gate 2 — expose + retire.** Wire step streaming into the Hub UI. Then retire `investigation_v2` from the agent lane (leave its code dormant/off; no other caller uses it).

## Scope boundaries

- **Tools v1:** existing four (`repo_grep`, `repo_read`, `repo_list`, `recall_query`). Trace/health/memory tools are a fast-follow.
- **Non-goals:** dedicated agent worker; model swap; agent-chain revival; any new keyword heuristics; repo writes; network access.

## Acceptance checks

1. Hub `mode: agent` reaches `mode="agent_repl"` → smolcode; `investigation_v2` and `_infer_context_exec_mode` are not on this path (evidence: run trace / logs).
2. A live-stack run of a real diagnostic question completes within 10 min, shows a multi-step tool trace, and returns a grounded natural-language answer (per `AGENTS.md` closure: live verification, not a mock harness).
3. Reasoning steps appear live in the Hub UI during the turn (Gate 2).
4. No canned-apology string can be emitted on this lane.
5. `.env` / `.env_example` parity holds for every changed key; local env synced.

## Risks

1. **Local model may not reliably drive smolagents code format** — highest risk; Gate 1 exposes it; remediation is model/prompt/GBNF grammar.
2. **Orphaned work on the shared worker** — a timed-out 10-min loop leaves the worker generating and briefly stalls the next chat turn (accepted trade-off of shared-worker choice).
3. **Streaming wiring across two services** — most likely scope-creep point; isolated behind Gate 2.
