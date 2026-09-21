# Finalize owner-lane — design

**Date:** 2026-09-15  
**Status:** approved for planning  
**Incident:** Hub unified chat turn `60f0e051` (“night night”) — motor draft ready in ~1 min on chat (`MODEL_SONNET` → live Qwen GGUF via FCC chat alias); finalize reflect waited ~475s on gateway `agent`, returned empty, degraded to misaligned, then response repair waited up to 540s on the same agent worker while curiosity held that lane.

## Arsonist summary

Finalize must stay on the lane that owns the turn. Chat-owned unified turns finalize on gateway `chat`. Agent-owned turns finalize on `agent` (or the admitted lease lane). The hard-coded “ordinary finalize = agent” default is wrong and is what stranded a finished goodnight behind curiosity.

## Problem

`orion/harness/finalize.py` builds reflect/repair cortex context with:

- `llm_lane` / `llm_route` = `resource_lease.lane` when a lease is present
- else **`"agent"`** always

That no-lease default was locked in after a 2026-08-16 starvation incident and encoded in `test_finalize_reflect_lane.py`. It assumed “agent is idle isolation.” Agent is no longer idle: curiosity and other agent-lane work compete there. Chat turns then leave their own chat capacity and queue behind agent work, while Hub still shows the chat turn running until harness finalize returns.

Semantics (do not re-twist):

- Hub/runtime **chat lane** = FCC label `MODEL_SONNET` (and other non-agent FCC labels). Live, that alias hits the Qwen GGUF chat backend (`llamacpp/chat`), not Anthropic Sonnet.
- Hub/runtime **agent lane** = agent FCC model label / gateway `agent`.
- Post-draft finalize (reflect + repair) is a cortex-exec LLM call. “Stay on chat” means gateway route **`chat`**, same pool the FCC chat alias already used — not a third path, not Anthropic.

## Rule

**Finalize LLM route = turn owner lane.**

| Turn owner | Finalize `llm_lane` / `llm_route` |
|------------|-----------------------------------|
| Admitted lease present | `resource_lease.lane` (unchanged) |
| Agent-owned (agent FCC model label), no lease | `agent` |
| Chat-owned (default unified chat / non-agent FCC label), no lease | `chat` |

Lease handoff for curiosity/agent admission stays as designed. This patch only fixes the unleashed / chat-owned default and threads owner identity into finalize context builders.

## Choke point

1. **Context builders** — `build_finalize_reflect_context` and `build_response_repair_context` in `orion/harness/finalize.py` (also any sibling repair/re-reflect builders that hardcode `"agent"` the same way).
2. **Owner signal** — harness runner already has `HarnessRunRequestV1.fcc_model_label` and optional `resource_lease`. Pass an explicit finalize lane (or the model label + lease) into `run_harness_finalize_chain` / reflection / repair so builders do not guess.
3. **Lane predicate** — reuse `orion.llm.routes.is_agent_route_model_label` (same predicate Hub uses for chat-vs-agent runtime activity and governor dispatch). Non-agent → `chat`.

Do not solve this with prompt text or keyword lists on the user message.

## Proposed behavior

```text
if resource_lease is not None:
    lane = resource_lease.lane
elif is_agent_route_model_label(fcc_model_label):
    lane = "agent"
else:
    lane = "chat"   # chat-owned default, including MODEL_SONNET
```

Set both `llm_lane` and `llm_route` to that lane. Keep `allow_chat_fallback=False` unless an existing lease/admission path already requires otherwise (do not reintroduce silent cross-lane fallback that undoes owner isolation).

Wire `fcc_model_label` (or a precomputed `finalize_llm_lane`) from `HarnessRunner` into the finalize chain call sites so leased and unleashed turns both carry owner identity.

## Files likely to touch

- `orion/harness/finalize.py` — owner-lane default; comments that claim ordinary finalize must be agent
- `orion/harness/runner.py` — pass owner lane / model label into finalize chain
- `orion/harness/tests/test_finalize_reflect_lane.py` — invert no-lease default assertions; add chat-owned vs agent-owned cases
- `orion/harness/tests/test_finalize_resource_lease.py` — lease still wins
- `services/orion-harness-governor/README.md` — replace “ordinary finalization keeps the existing agent route” with owner-lane rule
- Optional: `orion/harness/tests/test_llm_lane_propagation.py` (or cortex-exec equivalents) if they assert the old agent-only default

## Non-goals

- Requiring durable admission for every Hub chat turn (Approach 2) — out of scope; lease path already correct when present
- Moving the FCC motor off chat / changing `MODEL_SONNET` alias meaning
- Pointing finalize at gateway `harness` as a third identity (same worker as chat today; owner rule is `chat` for chat-owned turns)
- Soft preference / agent escape hatches that re-allow unleashed chat finalize on agent
- Raising or lowering finalize timeouts (separate from lane ownership)
- Quick-gate eligibility changes (separate; may still force LLM reflect, but that reflect must stay on owner lane)

## Acceptance checks

1. **Unit:** chat-owned, no lease → reflect and repair contexts have `llm_lane == llm_route == "chat"`.
2. **Unit:** agent FCC label, no lease → both contexts `"agent"`.
3. **Unit:** lease with `lane=chat` or `lane=agent` → contexts use lease lane regardless of model label.
4. **Regression:** no test remains that asserts unleashed ordinary finalize always routes to `agent`.
5. **Docs:** harness-governor README and finalize comments describe owner-lane, not “agent isolation.”
6. **Live (post-deploy, report UNVERIFIED until done):** a short Hub chat turn while an agent/curiosity job is busy still completes finalize on chat without multi-minute agent wait; runtime activity / cortex-exec logs show `llm_route=chat` for that turn’s reflect/repair.

## Risks

- Chat finalize shares the chat worker with other chat/FCC chat traffic (`n_parallel` constraints). Accepted: better than stranding chat behind agent/curiosity.
- Aug 2026 comment feared 5b vs live chat contention on chat. Owner-lane means this turn’s finalize shares this turn’s pool by design; cross-turn chat contention remains a capacity issue, not a reason to steal agent.

## Rollback

Revert the finalize default and tests. Lease path behavior is unchanged in spirit; only the no-lease default and explicit owner threading reverse.

## Recommended next patch

Thin implementation of Approach 1 (owner-lane default + thread `fcc_model_label` / lease into finalize). No admission rollout. Plan → tests first → flip default → README.
