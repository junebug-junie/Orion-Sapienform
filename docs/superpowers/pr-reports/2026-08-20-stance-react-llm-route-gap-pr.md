# PR report: stance_react missing from llm_route + max_tokens tables

## Summary

- **Production incident.** Every real chat turn was being deferred: `"stance_react_failed: stance_react exec result missing thought payload"`. Traced live via correlation-ID logs across `orion-thought` → `orion-cortex-exec` → `orion-llm-gateway` (not guessed).
- **Root cause, two independent gaps in `executor.py`:** `stance_react` (the real stance-evaluation step of every unified chat turn, wired in by PR #1739) was never added to two verb-based lookup tables that every other "fat prompt" verb already has entries in.
  1. The default `llm_route` mapping — `stance_react` fell through to `None`, and the gateway's own fallback for an unset route is the small `quick` lane (512-token budget). Real stance_react prompts run 16–19K chars, so every call hit `"overflow on route=quick and no larger lane exists"`.
  2. `_resolve_llm_chat_max_tokens` — also missing `stance_react`, so even after fixing #1 the completion budget stayed at 512, truncating the required strict-JSON `ThoughtEventV1` payload mid-object.
- Extracted the inline `llm_route` if/elif chain into a standalone `_default_llm_route_for_step()` (pure, behavior-preserving refactor — verified condition-by-condition against the original for every pre-existing verb) so this exact bug class is unit-testable going forward; added `stance_react` to both tables, matching the treatment 4 sibling verbs already get for the identical symptom.
- Both fixes deployed and **live-verified end-to-end** against the real running system, not just tests.

## Outcome moved

Chat was fully broken (every stance_react-driven turn deferred) → now working. Live before/after:

- **Before:** gateway `"[LLM-GW ctx] overflow on route=quick and no larger lane exists -- returning error"`.
- **After fix #1 alone:** turn still failed differently — `finish_reason=length`, `emitted_chars=0` (truncated mid-object at the 512-token default).
- **After both fixes:** real outreach-trigger turn succeeded (`corr=adc38554-e4cb-41f2-b008-9bb935fd4238`): `route=chat`, `effective_max_tokens=8000`, `finish_reason=stop`, 1167 real chars, `structured_output_rejected=False`. The outreach API returned a real, grounded 282-char message.

## Current architecture

`services/orion-cortex-exec/app/executor.py`'s main plan-execution loop resolves an `llm_route` (physical gateway lane: `quick`/`chat`/`metacog`/etc.) and an `effective_max_tokens` completion budget per step, both via verb-name-keyed lookup logic, before dispatching the LLM call to `LLMGatewayService`. A separate, orthogonal concept (`llm_lane`/`execution_lane`/`priority`, computed by `app/llm_lane.py::resolve_llm_lane_for_step`) exists for gateway Phase 3 priority/admission metadata — it already classified `stance_react` correctly (`llm_lane=chat`, `priority=high`), which is why the bug was easy to miss: the logs looked like the right lane was chosen, but that value never actually reached the wire.

## Architecture touched

- `services/orion-cortex-exec` only. No schema/bus/contract changes — this is a routing-table completeness fix in one service.

## Files changed

- `services/orion-cortex-exec/app/executor.py`: extracted `_default_llm_route_for_step()` (new standalone function, replaces the inline if/elif chain at the `llm_route` computation call site); added `stance_react` → `"chat"` there; added `stance_react` → `settings.llm_chat_general_max_tokens` in `_resolve_llm_chat_max_tokens`.
- `services/orion-cortex-exec/tests/test_default_llm_route_for_step.py` (new): 11 tests — the new `stance_react` case, plus every pre-existing branch individually (regression guard, not just a new-case test).
- `services/orion-cortex-exec/tests/test_harness_finalize_max_tokens.py`: new `test_stance_react_uses_general_max_tokens_budget`, mirrors the existing sibling tests' exact mock shape.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `stance_react` LLM calls now route to the `chat` (Circe) lane with an 8000-token completion budget instead of silently defaulting to `quick` (512 tokens, and previously `None`/undefined).
- Compatibility notes: pure verb-scoped addition; every other verb's routing/max-tokens resolution is unchanged (verified — see Review findings below).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable — no new env keys; this fix reuses `settings.llm_chat_general_max_tokens` (`LLM_CHAT_GENERAL_MAX_TOKENS`), already documented and already correctly read live.
- local `.env` synced: not applicable.
- skipped keys requiring operator action: none.
- **Disclosed, not fixed — separate anomaly found during live investigation:** `settings.llm_chat_max_tokens_default` reads `512` live inside the running container despite `LLM_CHAT_MAX_TOKENS_DEFAULT=8000` being set in the container's real environment (confirmed via `docker exec ... printenv` and a direct Python import of the live `Settings` object). Sibling fields in the exact same `Settings` class (`llm_chat_general_max_tokens`, `llm_chat_quick_max_tokens`) correctly read their own env vars in the same container. Root cause not found — deliberately not chased under incident-response time pressure. This patch sidesteps it entirely by routing `stance_react` through `llm_chat_general_max_tokens` (confirmed working) rather than the broken field. Worth its own follow-up investigation.

## Tests run

```text
cd services/orion-cortex-exec

# New + directly touched tests
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest tests/test_default_llm_route_for_step.py tests/test_harness_finalize_max_tokens.py tests/test_executor_llm_route_override.py tests/test_llm_lane_propagation.py -q
→ 32 passed

# Wider adjacent suite
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest tests/test_default_llm_route_for_step.py tests/test_harness_finalize_max_tokens.py tests/test_executor_llm_route_override.py tests/test_llm_lane_propagation.py tests/test_chat_general_route_mapping.py tests/test_memory_graph_suggest_final_text.py tests/test_attention_frame_integration.py tests/test_chat_attention_salience_trace.py -q
→ 62 passed, 2 failed
```

The 2 failures (`test_chat_general_route_mapping.py::test_introspect_spark_uses_quick_route`, `test_memory_graph_suggest_final_text.py::test_memory_graph_suggest_max_tokens_budget`) are **pre-existing and confirmed unrelated** — both fail identically on unmodified `origin/main`, in code paths this patch does not touch.

The code-review agent additionally ran a live before/after full-suite diff (branch vs. unmodified `origin/main`, same restricted env): **zero tests newly fail on the branch.**

## Evals run

No dedicated eval harness exists for this routing-table logic. The live end-to-end verification against the real deployed system (see Outcome moved) is the strongest available evidence for this class of change — a real production incident reproduced, then confirmed resolved against the actual running pipeline, not a synthetic harness.

## Docker/build/smoke checks

Deployed live (worktree `.env`/`services/orion-cortex-exec/.env` symlinked to the primary checkout, confirmed gitignored):

```text
./scripts/safe_docker_build.sh orion-cortex-exec up -d --build
→ all 4 containers (cortex-exec, -chat, -spark, -background) recreated and started, twice
  (once per fix, since fix #1 alone was insufficient and had to be diagnosed further)
```

Live verification, both fixes together:

```text
curl -X POST http://localhost:8080/api/debug/endogenous-outreach/trigger
→ {"ok":true,"result":{"outreach":true,"reason":"sent","correlation_id":"adc38554-e4cb-41f2-b008-9bb935fd4238",
   "chars":282,"generation":{"elapsed_sec":127.318,"harness_grounding_status":"grounded", ...}}}

docker logs orion-athena-cortex-exec | grep adc38554-e4cb-41f2-b008-9bb935fd4238
→ llm_route_selected ... route=chat
→ llm_chat_budget ... route=chat effective_max_tokens=8000 max_tokens_source=settings.llm_chat_general_max_tokens_stance_react
→ llm_chat_result ... route=chat completion_tokens=1188 finish_reason=stop emitted_chars=1167
→ final_text_assembly ... structured_output_rejected=False raw_len=1167 clean_len=1167 final_len=1167
```

## Review findings fixed

- No material findings. Code review (run against the staged diff before commit) confirmed:
  - The `_default_llm_route_for_step` extraction is condition-by-condition behavior-preserving for every pre-existing verb — only `stance_react` gained new behavior.
  - Both new lookup additions are correctly scoped (`stance_react` has exactly one step, `llm_stance_react`, per `orion/cognition/verbs/stance_react.yaml`, so verb-only gating is safe) and consistent with the 4 sibling fixes they're modeled on.
  - New tests are real regression guards covering every pre-existing branch, not just the new case.
  - A live before/after full-suite diff showed zero newly-broken tests.
  - The disclosed `LLM_CHAT_MAX_TOKENS_DEFAULT` anomaly and the two pre-existing test failures were independently re-verified as accurately described and correctly out of scope.
  - One procedural-only note (commit before calling it done) — addressed by this commit.

## Restart required

Already done — deployed and live-verified above. For reference:

```bash
./scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: Low
- Concern: `settings.llm_chat_max_tokens_default` env-read anomaly (disclosed above) remains unexplained. If any *other* verb that relies on that specific field's env override is added or changes in the future, it could silently get 512 instead of the intended larger budget, the same way `stance_react` did here.
- Mitigation: Worth a small, dedicated follow-up investigation (not urgent — no verb is currently known to depend on this specific field reading its env override correctly at the time of this patch). Flagged here so it isn't lost.
- Severity: Low
- Concern: Merge-order — `Orion-Sapienform-kill-dead-phi-hint-fallback` (a large, unmerged worktree from an earlier investigation this session) also touches `executor.py`. Not re-checked in this incident-response pass given the urgency.
- Mitigation: Whichever branch merges second will need conflict resolution; noting for visibility.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1758
