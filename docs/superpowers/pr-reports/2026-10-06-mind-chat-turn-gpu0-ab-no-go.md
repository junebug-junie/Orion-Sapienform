# orion-mind on gpu0 (chat, Qwen3.6-35B) for Juniper's Hub turns: A/B says no-go

Date: 2026-10-06. Status: evidence only, no implementation. The directive said: replay real Mind
prompts on both routes first, and stop if the 35B makes Mind worse. It does, on two independent
axes, so the route switch was not built.

## Arsonist summary

Moving Mind's three LLM calls from metacog (gpu3, Qwen3-8B) to the chat lane (gpu0, Qwen3.6-35B)
would turn every Hub-turn Mind run from "works, ~14 s" into "fails open, 10-60 s":

- **Quality:** the 35B returned parseable JSON on 7/7 semantic calls that answered, and **0/7 were
  schema-valid**. It invents enum values: `claim_kind` = `status_claim`, `event_claim`,
  `factual_claim`, `social_greeting`...; `anchor` = `work`, `project`, `user_state`...;
  `recommended_effect` = `acknowledge_progress`, `validate_frustration`... None of those are in
  `orion/mind/synthesis_v1.py`'s Literals. A schema-invalid semantic phase ends the Mind run
  (`semantic_schema_invalid`, fail-open legacy brief). The 8B gets through because it answers with a
  single bare claim that `try_wrap_singleton_semantic_claim` / legacy normalization coerce; the 35B
  answers with the full wrapper object, which goes straight to strict validation.
- **Latency:** 35B semantic median 24.6 s (max 33.9 s) vs 3.5 s on metacog; 3/10 runs hit the
  60 s Mind timeout outright. With metacog doing semantic and the 35B doing appraisal + stance,
  1/4 runs finished (appraisal 12.8 s, stance 10.2 s vs ~5.7 s / ~4.7 s on metacog) and 3/4 timed
  out in appraisal (`invalid_handoff`).
- **Thinking burn:** not the failure. No `<think>` text in any 35B response; semantic/stance ran
  with `enable_thinking=false`, appraisal with the model default (thinking on) used 272 completion
  tokens on the one run that finished. The token budget was not what broke.

## Why the latency is that bad (interplay, measured)

gpu0 is lent: during the whole replay the pool held a long-lived `agent`-class lease on role
`chat` (`holder=http:anthropic`, ~35k-token prompts, `gpu_pool_lease_granted route=agent ...
role=chat url=...:8011`). Mind's interactive `chat` leases were **granted immediately** (pool
priority worked), but the upstream is `llama-server --parallel 1`, so the granted request queued
inside llama-server behind the agent's prefill and ran out the gateway read timeout (44-55 s).
Pool priority does not preempt an in-flight upstream request. When Juniper is actively chatting
(gpu0 not lent) the queue would instead be her own final-answer generation plus the stance-prepare
probe (PR #2512): Mind's 3 serial calls would sit in front of / behind those on one slot. That
case was not measured (no live Hub turns while this ran; she had signed off at 08:07 UTC).

## Method

- 10 most recent `hub_orion` turns from `chat_history_log` (prompt + 3 prior turns of the same
  session as `messages_tail`). Mind's own prompts are not persisted (`mind_runs` keeps only the
  result + a request summary), so they were rebuilt by Mind's own code: a throwaway script run
  inside the live `orion-mind` container called `engine.run_mind_llm_synthesis` directly
  (`trigger="replay"`, `MIND_TURN_MODEL_ROUTE=""`, all three phase routes forced to the route under
  test), with `MindLLMClient._bus_chat` wrapped to record per-call latency, usage, raw content.
  Nothing was persisted; no HTTP route was hit. Recall fragments were absent from the rebuilt
  snapshot, so prompts were lighter than live ones.
- Calls went through the real bus -> llm-gateway -> gpu-pool path. Spacing 12-15 s between runs;
  44 runs total (~70 LLM calls), 08:16-08:40 UTC 2026-10-06, no Hub chat activity in that window.
- Schema failures were diagnosed by re-validating the stored 35B content against
  `SemanticSynthesisV1` inside the container.

## Results

Live baseline for comparison: the 20 most recent real Hub-turn `mind_runs` on metacog_turn took
6.0-27.1 s total (most 6-11 s); two in the last 24 h timed out at 60 s in semantic.

| sample | route (sem/appr/stance) | total ms | semantic ms (or per-call ms) | outcome |
|---|---|---|---|---|
| 0 | metacog | 13217 | 3094 | meaningful_synthesis |
| 0 | chat | 33941 | 33914 | semantic_schema_invalid |
| 1 | metacog | 14406 | 4018 | meaningful_synthesis |
| 1 | chat | 10685 | 10651 | semantic_schema_invalid |
| 2 | metacog | 15961 | 4360 | meaningful_synthesis |
| 2 | chat | 14270 | 14237 | semantic_schema_invalid |
| 3 | metacog | 15837 | 4606 | meaningful_synthesis |
| 3 | chat | 25158 | 25124 | semantic_schema_invalid |
| 4 | metacog | 14297 | 3656 | meaningful_synthesis |
| 4 | chat | 11191 | 11157 | semantic_schema_invalid |
| 5 | metacog | 15557 | 4599 | meaningful_synthesis |
| 5 | chat | 24639 | 24607 | semantic_schema_invalid |
| 6 | metacog | 13534 | 3421 | meaningful_synthesis |
| 6 | chat | 60085 | - | RPC timeout waiting on orion:mind:llm:re |
| 7 | metacog | 7100 | 1955 | meaningful_synthesis |
| 7 | chat | 60083 | - | RPC timeout waiting on orion:mind:llm:re |
| 8 | metacog | 8525 | 2386 | meaningful_synthesis |
| 8 | chat | 60046 | - | RPC timeout waiting on orion:mind:llm:re |
| 9 | metacog | 13677 | 3054 | meaningful_synthesis |
| 9 | chat | 32200 | 32171 | semantic_schema_invalid |
| 0 | metacog/chat/chat | 24881 | [1871, 12788, 10166] | meaningful_synthesis |
| 1 | metacog/chat/chat | 62577 | [2476] | invalid_handoff |
| 2 | metacog/chat/chat | 61810 | [1712] | invalid_handoff |
| 3 | metacog/chat/chat | 63087 | [2982] | invalid_handoff |
Summary: metacog 10/10 `meaningful_synthesis`, total median 14.0 s (7.1-16.0 s); semantic 3.5 s,
appraisal 5.7 s, stance 4.7 s (medians). chat 0/10 usable: 7 `semantic_schema_invalid`,
3 timeouts at 60 s; total median 28.7 s.

## Decision

No route switch. Shipping `ORION_MIND_CHAT_TURN_ON_GPU0` (ON) would make every Hub-turn Mind run
fail open. What would have to change first, each independently testable:

1. **Semantic prompt / schema contract for a larger model**: either list the allowed
   `claim_kind` / `anchor` / `recommended_effect` values in the semantic prompt, or have the
   validator coerce unknown enum values (as legacy normalization already does for the singleton
   shape). Then re-run this replay; it is the regression check.
2. **A reserved slot for the Hub turn on gpu0**: while gpu0 is lent to agent work, a
   `--parallel 1` upstream cannot serve an interactive Mind call in time. Either the lend must be
   clawed back before Mind dispatches, or chat needs `--parallel >= 2`.
3. Re-measure the stance-prepare (PR #2512) + Mind queueing with Juniper actually chatting.

## Not done (deliberately)

- No route added to `config/gpu_pool.yaml` / `orion/llm/routes.py`, no flag, no
  `check_chat_route_poachers.py` allow-list entry: nothing would use them yet, and the gate fails
  a stale allow entry by design.
- No deploy, restart, or merge. No restart required.
