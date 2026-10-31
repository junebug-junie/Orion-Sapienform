## Summary

- Reading turns kept failing at the stance step because the agent GPU lanes "think out loud" before answering, and the stance step had no instruction to stop that.
- `stance_react` now asks every lane for thinking off (`chat_template_kwargs.enable_thinking=False`), the same setting the chat lane already uses by default.
- No timeout was raised. The evidence says the work is short (about 340 output tokens); the extra time was unread reasoning.
- Regression tests cover both sides of the hand-off (orion-thought builds the setting; cortex-exec forwards it to the gateway). Both run in CI.
- The verb file's comment now records the root cause, so the next slow stance gets checked by token count, not by a third timeout raise.

## Outcome moved

Failure mode `turn_deferred:stance_react_failed: timeout:caller_budget_exhausted`: 112 failed reading runs since 2026-09-28 (108 of 109 timed-out stance calls were on agent-gpu2). Expected after deploy: agent-lane stances in the 300-500 token range and no stance budget timeouts.

## Current architecture

Hub `execute_unified_turn` -> orion-thought (`STANCE_REACT_TIMEOUT_SEC=360`) -> cortex-exec (verb `timeout_ms=240000`, gateway budget ~235 s) -> llm-gateway -> GPU pool lease. Reading turns hold an agent-class lease, so their stance runs on `agent` (gpu1, Strata Qwen3.8-Flash-Next, thinking `reasoning_effort=high`) or `agent-gpu2` (Ternary Bonsai 2 27B, ~29 tok/s, `reasoning_effort=xhigh`). Juniper's chat turns run stance on `chat` (same Flash-Next model, thinking off).

### Evidence (live Postgres, 2026-10-31)

`llm_call_usage`, verb `stance_react`, since 2026-10-29 22:51 (ledger start):

| lane | served | median completion tokens | avg duration | timeouts |
|---|---|---|---|---|
| chat | 120 | 342 | 13.9 s | 0 |
| agent (Strata) | 76 | 2,918 | 95.5 s | 7 |
| agent-gpu2 (Bonsai) | 71 | 3,652 (max 6,285) | 157.9 s | 42 |

Prompt size is the same on all lanes (~9-10k tokens), and pool wait is small (1-7 s average), so it is not prompt size and not queueing. Not a dead listener: `RpcNoListener` hit only 4 runs, all during the 04:28Z restart. Gateway logs show `reasoning_len=0` returned for these calls, so the reasoning tokens are generated and then discarded.

`gpu_pool_events` released leases, first cortex-exec agent-class lease per reading turn (since 10-01): agent-gpu2 `ok` 253 (median 138 s, p90 205 s, max 234 s -- clipped at the budget) vs `caller_budget_exhausted` 108 (all ~235 s); agent `ok` 416 (median 104 s) vs 1 timeout. First budget timeout 2026-10-14, so this **predates** the 2026-10-30 Strata swap. The swap did not cause it.

Same-model control: Qwen3.8-Flash-Next on the chat lane writes 342 tokens for stance; on the agent lane it writes 2,918 tokens. The only difference is the lane's thinking default.

Per-request `enable_thinking=False` is honoured on the Strata agent lane: orion-mind (which sends it) has a median of 651 completion tokens there (max 1,272), not thousands. The Bonsai template checks `enable_thinking is false` (config/chat_templates/ternary-bonsai-2-27b.jinja:58,166).

## Architecture touched

orion-thought stance context only. cortex-exec already forwards `chat_template_kwargs` (executor.py `_fwd_key` loop); no change there.

## Files changed

- `services/orion-thought/app/bus_listener.py`: `build_stance_react_context` adds `chat_template_kwargs={"enable_thinking": False}`.
- `services/orion-thought/tests/test_stance_context_response_format.py`: regression test across default, agent, chat and hold routes (fails on main).
- `services/orion-cortex-exec/tests/test_stance_react_thinking_forwarding.py`: the key reaches the gateway request options on the agent route.
- `orion/cognition/verbs/stance_react.yaml`: comment only; records the root cause. `timeout_ms` unchanged.
- `.github/workflows/orion-gpu-pool-tests.yml`: runs both tests.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: stance_react LLM calls send `chat_template_kwargs.enable_thinking=false`. Agent-lane stances now behave like chat-lane stances already do.
- Compatibility notes: existing optional gateway option; no new fields.

## Env/config changes

- None. No `.env_example` change; no sync needed.

## Tests run

```text
services/orion-thought: pytest tests -q                         -> 783 passed, 32 skipped
services/orion-thought: test_stance_context_response_format.py on main code -> 1 failed (regression test catches it)
services/orion-cortex-exec: test_gpu_lease_forwarding.py test_orion_day_verbs.py test_stance_react_thinking_forwarding.py -> 16 passed
scripts/check_verb_physiology.py                               -> OK
```

Note: running cortex-exec's `test_chat_*stance*` files in one process with `test_gpu_lease_forwarding.py` fails 28 tests on main too (test-order pollution, pre-existing, not touched here).

## Evals run

```text
No eval run. A live check after deploy is the real proof (query below).
```

Post-deploy check (after ~1 day):

```sql
select pool_role, outcome, count(*), percentile_cont(0.5) within group (order by completion_tokens), round(avg(duration_ms)/1000,1)
from llm_call_usage where verb='stance_react' and created_at > '<deploy time>' group by 1,2;
```

Expect agent/agent-gpu2 medians in the hundreds of tokens and no `upstream_timeout`.

## Docker/build/smoke checks

```text
Not run. No deploy (as briefed).
```

## Review findings fixed

- Finding: no material issues. The key reaches the gateway on every stance path (builder, build_exec_ctx, stance_prepare delta, executor forward loop, gateway OPENAI_COMPAT backends incl. strata).
  - Fix: none needed.
  - Evidence: review traced exec_ctx.py:15-26, stance_prepare.py:315-318, executor.py:4447-4458, llm_backend.py:1262-1265.
- Finding (minor): CI did not trigger on stance_react.yaml, exec_ctx.py, stance_prepare.py.
  - Fix: added them to both path lists in orion-gpu-pool-tests.yml.
  - Evidence: workflow diff.
- Finding (minor): 240 s vs 235 s looked like a typo between files.
  - Fix: verb comment now says cortex-exec gives the gateway the ceiling minus 5 s.
  - Evidence: stance_react.yaml diff.
- Finding (minor, not fixed): reasoning-load signals drop for agent-lane stances; Strata per-request override is not proven by a direct stance call.
  - Fix: recorded under Risks with the post-deploy query.
  - Evidence: see Risks.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low. Concern: agent-lane stances lose hidden reasoning. Mitigation: the reasoning was never read, and Juniper's chat-lane stances have run thinking-off all along.
- Severity: low. Concern: `harness_finalize_reflect` shows the same pattern (agent median 3,354 tokens vs chat 322). Not changed here; follow-up.
- Severity: low. Concern: agent-lane stances stop producing `reasoning_content`, so `reasoning_char_count` / `reasoning_load` for those runs falls from thousands of characters to near 0. Those feed field pressure, prediction error and coherence metrics. Mitigation: expect a one-time drop while baselines re-learn. This is not Orion "thinking less" about anything it read; that reasoning was discarded unread.
- Severity: low. Concern: Strata honouring per-request `enable_thinking=False` over its `reasoning_effort: high` run default is inferred (orion-mind sends the same key and gets a median of 651 tokens on that lane), not proven on a stance call. Mitigation: post-deploy query above. UNVERIFIED until it runs.
- Severity: info. Concern: 56 reading runs failed with `turn_exception:TimeoutError` (Hub-side whole-turn ceiling). That is a separate failure and is not addressed here.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
