## Summary

- The finalize 5b judge (it grades each draft reply) now leaves the turn's GPU pool hold when the hold is on a role the judge was not checked on. Today that is the gpu2 seat (pool role `agent-gpu2`, Bonsai 2 27B).
- For that one call it uses a new pool route `judge`: gpu1 (Flash-Next agent lane) first, then chat's card only while Juniper lends it. It never lands on the gpu2 seat or hecate.
- A hold on `agent` or `chat`, and every turn without a hold, routes exactly as before. The 2026-09-15 owner rule (corr 60f0e051, chat turns not stranded behind curiosity) is unchanged.
- The repair call and every other call of the run stay on the hold. The judge prompt and `FinalizeReflectionV1` are untouched.

## Outcome moved

Judge calls that ran on Bonsai: 22 of 90 after PR #3195 (cognition_traces 11-11 05:29Z..11-12; 37 of 176 by 11-12 evening, all under a hold on `agent-gpu2`). Expected after deploy: 0. Why it matters: with thinking off (PR #3195), replay showed the Bonsai judge caught 2 of 5 false claims (5 of 5 before); Flash-Next agent caught 6 of 9 (5 of 9 before). Thinking stays off.

## Current architecture

`resolve_finalize_llm_lane` (`orion/harness/finalize.py`) returned route `agent` plus the hold ref whenever the turn held a GPU pool lease. The gateway turns a call carrying a hold ref into `attach`, which runs the call on the hold's role whatever route it names. A durable run whose hold landed on `agent-gpu2` was judged by Bonsai. The verb file's `lanes: [chat, agent]` is body-economy physiology only and does not route. A call without a hold gets a one-shot request lease from the pool by route -> class (`config/gpu_pool.yaml`); class `agent` lists `[agent, agent-gpu2, agent-deep, chat]`, so simply dropping the hold and naming `agent` could still land on Bonsai.

Live check of the cause (llm_call_usage joined to gpu_pool_leases, 11-11 05:29Z..11-12): every judge call on `agent-gpu2` (38 of 38) was an attach child of a hold.

## Architecture touched

- `orion/harness/finalize.py`: `resolve_finalize_judge_placement` + `FINALIZE_JUDGE_HOLD_ROLES = {agent, chat}` + `FINALIZE_JUDGE_ROUTE = "judge"`. `build_finalize_reflect_context` uses it. `resolve_finalize_llm_lane` is unchanged (repair still uses it).
- `config/gpu_pool.yaml`: class `judge: {roles: [agent, chat], on_unavailable: wait}`, route `judge: {class: judge, priority: system}`, and `judge` added to the `agent` role's owners (like `memory_distill`; as a borrower, rule 5 would hold every judge call behind any queued agent lease).
- `orion/llm/routes.py`: `judge` in ACCEPTED, display order, SYSTEM (never a human Compute choice) and CORTEX_OVERRIDE_SYSTEM_ROUTES (cortex-exec accepts it as `ctx["llm_route"]`). New `AGENT_LANE_ROUTES = {agent, judge}`.
- `services/orion-llm-gateway/app/lane_routes.py`: the gateway's lane routing (on live, `LLM_LANE_ROUTING_ENABLED=true`) rewrote every agent-lane call without a hold to route `agent`. It now keeps a caller route in `AGENT_LANE_ROUTES`. Without this the judge left its hold and was still planned on class `agent`, which lists the gpu2 seat.
- `scripts/check_gpu_pool_config.py`: `judge` is a big-model class (may not list an 8B role).

Admission path reused: the gateway's normal one-shot request lease for a call without a hold (the same path every unheld turn verb uses). No new queue, no retry layer.

## Files changed

- `orion/harness/finalize.py`: judge placement rule.
- `config/gpu_pool.yaml`: judge class, route, owner entry.
- `orion/llm/routes.py`: register the route.
- `scripts/check_gpu_pool_config.py`: judge in BIG_CLASSES.
- `services/orion-llm-gateway/app/lane_routes.py`: keep `judge` through agent-lane routing.
- `orion/harness/tests/test_finalize_judge_placement.py` (new): judge context per hold role; judge roles equal the YAML class.
- `services/orion-llm-gateway/tests/test_judge_route_plan.py` (new): the judge's wire shape planned by the real gateway planner on the real YAML with lane routing on. Fails on the old `lane_routes.py` with `('agent', 'agent') != ('judge', 'judge')`.
- `services/orion-llm-gateway/tests/test_lane_routes.py`: agent lane keeps `judge`, still rewrites other routes.
- `services/orion-cortex-exec/tests/test_executor_llm_route_override.py`: cortex-exec accepts `judge` without a hold.
- `services/orion-durable-runs/tests/acceptance_turn.py`, `test_durable_acceptance.py`: the full curiosity acceptance (real pool scheduler, real gateway) now asserts the gpu2-seat run's judge is one plain acquire on class `judge`, granted on `agent` (gpu1) in the idle background blocker hold's gap; the home-card run's judge stays attached.
- `orion/harness/tests/test_gpu_lease_transport.py`: hop-carries-ref test now covers a gpu2-seat hold (judge off the hold) and an agent hold (judge on it).
- `orion/gpu_pool/tests/test_route_view.py`, `orion/llm/tests/test_routes.py`, `services/orion-llm-gateway/tests/test_anthropic_passthrough.py`: route lists include `judge`.
- `services/orion-gpu-pool/README.md`: judge class paragraph, deploy order, rollback.

## Schema / bus / API changes

- Added: gateway/pool route `judge`, pool class `judge`.
- Removed: none.
- Renamed: none.
- Behavior changed: the 5b judge of a hold on a role outside `{agent, chat}` sends no `gpu_lease` and names route `judge` (lane `agent`).
- Compatibility notes: no bus schema change. Old images reading the new YAML load it (no new priority, no new field).

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no env template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/harness/tests orion/gpu_pool/tests orion/llm                         1357 passed (before review fixes); orion/llm + judge placement re-run 34 passed
services/orion-llm-gateway/tests                                           487 passed (after review fixes)
services/orion-cortex-exec override/lease/finalize-route tests             26 passed (after review fixes)
services/orion-gpu-pool/tests                                              201 passed, 15 skipped
services/orion-gpu-lane-controller/tests                                   110 passed
services/orion-durable-runs/tests (local Postgres 16, as CI)              998 passed, 1 skipped
services/orion-cortex-exec route/lane/lease/finalize tests                 202 passed, 1 failed (pre-existing on main: test_introspect_spark_uses_quick_route)
services/orion-harness-governor/tests                                      100 passed, 1 failed (pre-existing on main: test_start_passes_settings_into_pool_config)
services/orion-actions + orion-hub route tests                             78 passed
scripts/check_gpu_pool_config.py                                           ok
scripts/check_chat_route_poachers.py                                       PASS
scripts/check_definition_drift.py --gate                                   PASS
```

## Evals run

```text
services/orion-gpu-pool/evals/run_pool_day_eval.py                         VERDICT: PASS
```

The judge-quality replay that motivated this (PR #3195, judge_crash_replay_eval) is not re-run: this patch does not change what the judge sees or how it is asked, only where it runs. The live check below is the eval.

## Docker/build/smoke checks

```text
Not run: no Docker build in this worktree. Runtime claim is UNVERIFIED until the post-deploy SQL below.
```

## Post-deploy check

Replace `<deploy>` with the harness-governor restart time (UTC).

Judge model mix (expect no Bonsai rows; `route=judge` rows served by circe-worker-agent or circe-worker-chat):

```sql
select steps->0->'result'->'LLMGatewayService'->>'model_used'            as model,
       steps->0->'result'->'LLMGatewayService'->'meta'->>'served_by'    as served_by,
       steps->0->'result'->'LLMGatewayService'->'meta'->>'route'        as route,
       count(*)
from cognition_traces
where verb = 'harness_finalize_reflect' and timestamp > '<deploy>'
group by 1, 2, 3 order by 4 desc;
```

Same from the gateway's own ledger (pool role is a fact here):

```sql
select route, pool_role, outcome, count(*), round(avg(wait_ms)) as avg_wait_ms
from llm_call_usage
where verb = 'harness_finalize_reflect' and recorded_at > '<deploy>'
group by 1, 2, 3 order by 4 desc;
```

Judge failure rate (baseline 11-11 05:29Z..11-12: 13 of 215, 11 of them `gpu_pool_unavailable:hold_not_granted:released`):

```sql
select count(*) filter (where exists (
         select 1 from jsonb_array_elements_text(run_artifact->'reflection'->'alignment_notes') n
         where n like 'reflect_llm_failed:%')) as failed,
       count(*) as judged
from harness_turn_trace
where created_at > '<deploy>' and run_artifact ? 'reflection';

select left(n, 120) as note, count(*)
from harness_turn_trace, jsonb_array_elements_text(run_artifact->'reflection'->'alignment_notes') n
where created_at > '<deploy>' and n like 'reflect_llm_failed:%'
group by 1 order by 2 desc;
```

Verdict mix (LLM verdicts only; before #3195 83/11/6 % aligned/misaligned/uncertain, after 95/4/1):

```sql
select run_artifact->'reflection'->>'alignment_verdict' as verdict, count(*)
from harness_turn_trace
where created_at > '<deploy>'
  and run_artifact->'reflection'->>'reflection_source' = 'substrate_informed_pass'
group by 1 order by 2 desc;
```

Seat wait for the new route (the risk to watch):

```sql
select status, role, count(*),
       percentile_cont(0.5)  within group (order by extract(epoch from granted_at - created_at)) as p50_s,
       percentile_cont(0.95) within group (order by extract(epoch from granted_at - created_at)) as p95_s
from gpu_pool_leases
where work_class = 'judge' and created_at > '<deploy>'
group by 1, 2;

select reason, count(*) from gpu_pool_events
where work_class = 'judge' and event = 'unavailable' and created_at > '<deploy>'
group by 1;
```

## Review findings fixed

Code review ran in a subagent against commit aa3ee1317.

- Finding (must-fix): the gateway's lane routing rewrote route `judge` to `agent` for any agent-lane call without a hold (`lane_routes.py` agent branch ignored the caller's route). Class `agent` still lists the gpu2 seat, so the fix did nothing live and the tests passed only because they stopped at the harness context.
  - Fix: `AGENT_LANE_ROUTES = {agent, judge}` in `orion/llm/routes.py`; the agent branch keeps a caller route in that set.
  - Evidence: `test_judge_route_plan.py` fails on the old `lane_routes.py` (`('agent', 'agent') == ('judge', 'judge')`) and passes now; gateway suite 487 passed.
- Finding (should-fix): no test crossed the cortex-exec or gateway seam.
  - Fix: cortex-exec override test and the gateway planner test above.
  - Evidence: both pass; the gateway one is the reproduction above.
- Finding (should-fix): the judge can queue on gpu1 behind system holds.
  - Fix: measured instead of changed. Last 7 days, holds on gpu1 were 125.7 hold-hours background and 0.3 hold-hours system (memory_distill), so the system-priority judge can use almost every idle hold gap (H3). Seat-wait SQL is in the post-deploy check. Listed under risks.
- Finding (should-fix): deploy order.
  - Fix: explicit order pool, gateway, cortex-exec, governor in Restart required; the fix is live only when all four run the new code.
- Nits not changed: `/v1/models` lists `judge` like `prime` already does (system priority, no new exposure class); the off-hold judge's GPU time is no longer charged to the run's hold (listed under risks); hold roles are pool roles, drift with the YAML is pinned by test.

## Restart required

Deploy from the clean primary `main` checkout after merge, in this order (pool and gateway bake `config/gpu_pool.yaml`; they must know route `judge` before cortex-exec and the governor send it):

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build && scripts/safe_docker_build.sh orion-llm-gateway up -d --build && scripts/safe_docker_build.sh orion-cortex-exec up -d --build && scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: more `gpu_pool_unavailable` judge failures. Off the hold, the judge waits in gpu1's queue like any one-shot system call. It may use the gaps of idle background holds (H3) but not of system holds, and it is shed at once while the heat reflex (`cabinet_hot`) blocks system work. Last 7 days, one-shot agent-class system calls on gpu1 waited p50 3.4 s, p95 84 s, and 28 of 263 got no seat (those had a ~60 s wait; the judge's is up to 300 s). No heat sheds since 10-26; October had ~30,000 in one week.
  - Mitigation: a failed judge degrades to the quick-lane verdict or `reflect_llm_failed` (the reply still goes out). The seat-wait SQL above shows it. Rollback is a revert. If failures climb, the next lever is the route's priority, not a fallback to the hold.
- Severity: low
  - Concern: partial deploy. If cortex-exec is old and the governor new, cortex-exec rejects the override and falls back to route `agent` without a hold (works; may still land on gpu2). If the gateway is old with lane routing on, it rewrites `judge` to `agent` (same).
  - Mitigation: restart order above.
- Severity: low
  - Concern: the repair (`orion_response_repair`) still runs on the hold, so ~22% of repairs run on Bonsai (69 of 316 in 14 days, all served, thinking off). No catch-rate replay exists for the repair, so there is no measured quality gap. Left alone per the brief.
- Severity: low
  - Concern: a gpu2-held run's judge now uses a gpu1 slot while the run still holds its gpu2 slot (about 15 s per judge). That GPU time is no longer charged to the run's hold in pool accounting; it shows as `work_class=judge` leases.
- Severity: low
  - Concern: deploy order. The fix is live only when the pool, the gateway, cortex-exec and the governor all run the new code. A new gateway with an old pool asks for class `judge` and is refused.
  - Mitigation: restart order above, all four in one line.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3219

🤖 Generated with [Claude Code](https://claude.com/claude-code)
