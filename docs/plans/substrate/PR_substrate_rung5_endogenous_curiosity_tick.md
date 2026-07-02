# PR: Wire rung-5 endogenous curiosity tick into substrate-runtime

Branch: `feat/substrate-rung5-endogenous-curiosity-tick` → `main`

Open: https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/substrate-rung5-endogenous-curiosity-tick

## Summary

Closes the runtime gap for self-modeling loop **rung 5**. PR #776 shipped `endogenous_curiosity_candidates()` and the `FrontierCuriosityEvaluator.endogenous_signals` hook, and operator sign-off enabled `ORION_ENDOGENOUS_CURIOSITY_ENABLED=true` on main — but nothing in any service called the module on a schedule.

This PR adds a bounded, fail-open tick loop to `orion-substrate-runtime` that:

1. Reads intrinsic inputs each cycle:
   - substrate graph nodes (prediction error from rung 1)
   - latest rung-3 attention broadcast frame (`substrate_attention_broadcast_projection`)
   - chat repair pressure (from `active_chat_session` projection)
2. Builds bounded `curiosity_candidate` seeds via `endogenous_curiosity_candidates()`
3. Routes them through `FrontierCuriosityEvaluator.evaluate(..., operator_requested=False, endogenous_signals=seeds)` using neutral metacog inputs so endogenous seeds drive the decision path
4. Logs `substrate_endogenous_curiosity_tick_completed seeds=N outcome=...` — **no expansion, landing, or auto-apply**

## Changes

| Area | Change |
|------|--------|
| Worker | `_endogenous_curiosity_tick` + `_endogenous_curiosity_loop` registered in `start()` |
| Settings | `ORION_ENDOGENOUS_CURIOSITY_*` fields + `ORION_ENDOGENOUS_CURIOSITY_TICK_INTERVAL_SEC` (default 60s) |
| Compose / env | Tick interval wired through docker-compose and `.env_example` |
| Tests | `test_worker_endogenous_curiosity_tick.py` — disabled/kill-switch no-op, evaluator wiring, fail-open |

## Safety / governance

- Respects existing guardrails: enable flag + kill switch (kill switch wins), budget cap from settings (hard max 8 in library code unchanged)
- `operator_requested=False` — no operator trigger impersonation
- Does **not** call `FrontierCuriosityOrchestrator.run()` — decision/plan only, no graph mutation
- Fail-open throughout: store init, snapshot, broadcast load, repair load, and evaluator errors never crash the loop

## Prerequisites on Athena

Rung 5 consumes outputs from rungs 1 and 3. Before expecting non-zero seeds:

- `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` + sparql store reachable
- `ORION_ATTENTION_BROADCAST_ENABLED=true` + `substrate_attention_broadcast_projection` migration applied
- Optional: chat grammar reducer producing repair pressure in `active_chat_session`

## Test plan

- [x] `PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_endogenous_curiosity_tick.py -q` — 4 passed
- [x] `python3 -m compileall services/orion-substrate-runtime/app -q`
- [ ] After deploy: `docker logs orion-athena-substrate-runtime 2>&1 | grep endogenous_curiosity` shows `substrate_endogenous_curiosity_tick_completed` every ~60s when enabled and seeds exist

## Non-goals

- No rung-6 proposal auto-apply
- No frontier expansion/landing execution
- No new Postgres projection table (logging only in v1; persistence can follow if needed)
