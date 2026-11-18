# Stop shipping Orion's chain-of-thought as their reply; thinking off for repair and compactor digests

## Summary

- When Orion's draft reply got rejected by reflection, the step that rewrites it (response repair) ran with thinking on, burned its whole budget thinking, and came back with an empty answer. The harness then picked up the hidden reasoning ("We need answer user's request: repair Orion's draft reply to Juniper…") and shipped it as Orion's reply. That stops here.
- Repair now asks the model not to think (`chat_template_kwargs.enable_thinking=False`) and gets its own 3072-token budget instead of the 8000 general one.
- The repair reply is read with a new answer-only extractor. It never falls back to reasoning fields and strips inline `<think>` text. A reasoning-only result, or a reply cut off at the token limit, now takes the harness's existing failure path instead of being shipped.
- The GitHub and chat compactor digest calls had a "reasoning off" option (`reasoning: {effort: none}`) that nothing anywhere reads. That option is replaced with the `chat_template_kwargs` switch that cortex-exec actually forwards to llama.cpp.
- The shared extractor's behaviour is unchanged for every other caller (the new `include_reasoning` kwarg defaults to True).

## Outcome moved

Failure mode: chain-of-thought delivered as Orion's spoken reply. Live blast radius, read-only from `harness_turn_trace`:

- **Last 7 days:** 17 of 504 finalized turns (3.4%) shipped repair-prompt reasoning as `final_text` (marker `draft reply to juniper` / `repair orion's draft`). By day: 09-24 ×2, 09-26 ×1, 09-27 ×5, 09-29 ×2, 09-30 ×7.
- **All time:** 32 turns, the first on 2026-09-24.
- **Reasoning openers** (`We need|The user wants|Let me|…`): 18 hits, 17 of them this same leak. The 18th ("I need to be honest — …") is a false positive.
- **Separate defect, same fix:** corr a84fc74a shipped a 52-character fragment that had been cut off at the token limit.

Repair verb, 7 days: 10 calls hit the 8000 cap. Of the agent-model calls, 9 returned `content=""` with 26–36k characters of reasoning.

Compactor: 2af9b6ea (09-30) spent 16000 tokens, 48k characters of them reasoning, and finished with `finish_reason=length`. 0db311e7 (09-25) hit the 8000 cap with empty content. On 09-25, two calls with no reasoning in their traces (07f394ac, 383796de) finished in about 1200–1400 tokens.

## Current architecture

- `orion/harness/finalize.py` `build_response_repair_context` set no `max_tokens` and no `chat_template_kwargs`. So cortex-exec `_resolve_llm_chat_max_tokens` gave it `llm_chat_general_max_tokens` (8000), with thinking on.
- `extract_response_repair_text` used `extract_cortex_payload_text`. When the answer was empty, that function falls back to `reasoning_content` / `inline_think_content` / `raw.choices[].message.reasoning_content`.
- `orion/cognition/compactor/map_reduce.py` sent `reasoning: {effort: none}`. That key appears nowhere in cortex-orch, cortex-exec or the gateway.

## Architecture touched

- Shared extractor `orion/cognition/cortex_payload_extract.py`: adds `include_reasoning` (default True), `extract_cortex_answer_text`, `strip_inline_think` and `cortex_payload_truncated`.
- Harness finalize repair request and its extractor.
- The compactor digest request options.
- No bus, schema, channel or env changes.

## Callers of `extract_cortex_payload_text` (enumerated before changing shared behaviour)

| Caller | Output kind | Changed? |
|---|---|---|
| `orion/harness/finalize.py` `extract_response_repair_text` | prose shown as Orion's reply | **yes**, now answer-only |
| `orion/harness/finalize.py` `extract_finalize_reflection_payload` | strict JSON, parsed | no. Its parser rejects reasoning; caps were 2 of ~200 in 3 days; not prose |
| `orion/harness/finalize.py` tool-loop summary (~L731) | tool evidence excerpt | no (follow-up candidate) |
| `services/orion-embodiment/app/worker.py` town speech (×2) | prose spoken in town | no. **Same flaw class, follow-up** |
| `services/orion-cortex-exec/app/self_study.py` | JSON, parsed | no |
| `services/orion-durable-runs/app/runner.py` self_study reflect | JSON, parsed | no. Prose verbs there already use `strict_final_text` |
| `orion/memory_graph/cortex_suggest_extract.py`, `services/orion-hub/scripts/cortex_memory_graph_text.py` | private copies of the helpers | not affected |

## Files changed

- `orion/cognition/cortex_payload_extract.py`: answer-only extraction, inline-think stripping, truncation detection.
- `orion/harness/finalize.py`: repair `max_tokens=3072` + `enable_thinking=False`; repair extractor refuses reasoning-only and truncated results.
- `orion/cognition/compactor/map_reduce.py`: swaps the inert `reasoning.effort` option for `chat_template_kwargs`.
- `orion/harness/tests/test_response_repair_reasoning_leak.py`: new tests. A stored-payload fixture (content "" + ~30k reasoning), the 52-character truncation case, inline `<think>`, metadata truncation, the request shape, and an end-to-end chain run that must raise `HarnessFinalizeFailedError`.
- `orion/cognition/compactor/tests/test_map_reduce.py`: the digest request carries `enable_thinking False` and no `reasoning` key.
- `services/orion-cortex-exec/tests/test_harness_finalize_max_tokens.py`: the real repair context resolves to 3072 via `ctx.max_tokens`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed:
  - The repair request context now carries `max_tokens` and `chat_template_kwargs`.
  - A repair that comes back reasoning-only or truncated raises (failure path) instead of shipping.
- Compatibility notes: plan context already spreads into cortex-exec ctx (`verb_adapters.py` LegacyPlanVerb), and the `_fwd_key` loop already forwards `chat_template_kwargs` (precedent: PR #2445).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/harness orion/cognition                                  488 passed
services/orion-cortex-exec tests/test_harness_finalize_max_tokens.py  5 passed
services/orion-durable-runs tests (PYTHONPATH=repo, as CI)            246 passed, 71 skipped
services/orion-harness-governor tests                                 58 passed
services/orion-embodiment test_worker_speech*.py                      16 passed
services/orion-cortex-exec test_chat_general_route_mapping.py         1 failure (test_introspect_spark_uses_quick_route), also fails on main, unrelated
static gates: check_metric_lineage, check_definition_drift, check_async_routes_not_blocking, check_chat_route_poachers: PASS
```

Mutation checks: each one below breaks at least one test.

- Repair extractor back to the shared extractor: 3 fail
- Truncation gate removed: 1 fail
- `chat_template_kwargs` dropped from the repair context: 1 fail
- `max_tokens` dropped: 1 fail
- Compactor reverted to `reasoning.effort`: 1 fail
- Block-level reasoning kept in answer-only mode: 2 fail
- Inline-think stripping disabled: 1 fail

## Evals run

```text
No harness-finalize eval for this seam beyond orion/harness/evals (test_layer_attribution covered in the orion/harness run).
Live read-only blast-radius query above serves as the baseline; re-run it after deploy:
  select count(*) from harness_turn_trace where created_at > '<deploy time>'
   and run_artifact->>'final_text' ilike '%draft reply to juniper%';   -- expect 0
```

## Docker/build/smoke checks

```text
Not deployed (per task). No Docker build: no dependency, env or compose change.
UNVERIFIED live: that llama.cpp honours enable_thinking=False for these two verbs. The same mechanism is live for orion_day (PR #2445), curiosity supervisor and pre_turn_appraisal.
```

## Review findings fixed

- Finding: reasoning delivered as inline `<think>…</think>` inside `content` would still win in answer-only mode, since the raw content is the last candidate. Confirmed by the reviewer with a probe.
  - Fix: `strip_inline_think` (same rules as cortex-exec router `_strip_think_content`) is applied to every answer-only candidate, and empty results are dropped.
  - Evidence: `test_inline_think_in_content_is_never_the_reply`; disabling the strip fails it.
- Finding: the 3072 budget silently depends on the backend honouring `enable_thinking`. The gateway forwards it only for llamacpp / llama-cola.
  - Fix: the dependency is documented at `RESPONSE_REPAIR_MAX_TOKENS`.
  - Evidence: live 7-day repair traces that record a backend all say `llamacpp` (40). The rest (226) have no backend recorded.
- Finding: the `metadata.runtime_response_diagnostics.truncation_detected` branch was untested.
  - Fix: added `test_top_level_truncation_flag_is_refused`.
- Not fixed (nit): there is no executor-level test that `ctx.options.chat_template_kwargs` reaches `gateway_options`. The `_fwd_key` loop is shared with `gpu_lease` and is live for PR #2445.

## Restart required

Deploy order: harness-governor carries the repair change, since `orion/harness` is imported there. durable-runs carries the compactor change, since it imports `orion/cognition/compactor`. cortex-exec gets only a test change and does not need a rebuild. They are independent of each other.

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a repair that fails now means the user sees the hub's turn-error frame with the draft's first 2000 characters (`partial_draft`) instead of a reply. Before, they saw leaked reasoning or a fragment. Mitigation: that is strictly better than chain-of-thought, and repair only runs when reflection rejected the draft. A nicer "ship the draft on repair failure" path is a product decision, left as a follow-up.
- Severity: low. Concern: a legitimately long rewrite (more than about 10k characters) gets cut at 3072 tokens and is refused. Mitigation: the observed maximum repair output is 6602 characters (14 days).
- Severity: low. Concern: the same flaw class remains in embodiment town speech (`services/orion-embodiment/app/worker.py`), which uses the reasoning-fallback extractor for prose. The cortex-orch concept-induction journal synth also sends the same inert `reasoning.effort=none` (`workflow_runtime.py:1466`). Mitigation: follow-ups; outside this patch's seam.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2456

🤖 Generated with [Claude Code](https://claude.com/claude-code)
