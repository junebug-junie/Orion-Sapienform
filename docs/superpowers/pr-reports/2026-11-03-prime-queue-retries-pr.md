# Orion Prime: stop losing thoughts to the queue, stop waking on attention churn

## Summary

- Prime now waits up to 10 minutes for a seat on the model, and still gets its full 60 seconds to think once it has one. Before, one 60-second clock covered both, so a call that queued for a minute had nothing left.
- A try that never reached the model is no longer recorded as a failed thought. It does not start Prime's 5-minute rest, and the next tick tries again. After 3 such tries in a row, Prime backs off (5 min, 10 min, ... at most 1 hour). A heat-shed answer's own "retry after" is always respected.
- Prime's normal model calls move from the lowest pool priority (background) to ordinary Orion work (system). After a try that got no seat, the next one goes one level higher, on a new priority called `starved`: ahead of other Orion work, still behind Juniper's turns and Orion's senses, and still stopped by the 34 C heat reflex.
- An upstream server error (5xx) is retried once at once. Two in a row is a no-model try, not a failed thought.
- Attention moving to another topic no longer wakes Prime. Attention still shows in every frame Prime reads. A 6-hour quiet floor wakes Prime if nothing else does.

## Outcome moved

Live, 2026-10-30 to 11-03: 254 Prime thoughts, 96 failed. 78 of the failures were "RPC timeout": the call sat in the pool queue for its whole 60 seconds. The pool's own rows agree (holder `orion-mind`, chat class): 166 granted (mean wait 3.7 s), **74 cancelled after a mean 60.0 s queued**, 18 upstream failures, 4 ran out of generation time. The chat seat was held 15.9 hours by the harness (`http:anthropic`, 2,189 calls at system priority) while Prime sat at background behind all of it.

Concerns churned with attention: 22 of 38 concern drops gave "attention has moved ..." as the reason, and 1 of 37 graded concern acts settled.

Expected after this patch (replayed on the live trigger timeline, `orion/prime/evals/run_prime_wake_replay.py`): about **28 thoughts a day, down from 68 tries a day**, at least 13 every day, and no silence longer than 6 hours (12.3 hours without the floor).

## Metric gate: the hunger cut-off (not used)

Juniper's scope allowed Prime to go `starved` when its bid's hunger (`decision.intensity`) was above a cut-off. I checked the number first and did not wire it in.

1. **Where it comes from.** `orion/physiology/appetite.py` `hunger()`: intensity = 1 - exp(-t / tau), where t is the time since ANY `prime` act closed (thoughts, questions, concerns, signals) and tau is Prime's own median gap between closes (live: 314-358 s, about the 300 s rest interval).
2. **Independence.** It is a transform of elapsed time since Prime's last close. It is not independent of the "last try got no seat" condition either: failed tries closed acts and reset it.
3. **Theory.** A drive ("time since the last meal") is a fair reading of need only if not eating means being denied food. For Prime, not thinking mostly means nothing changed.
4. **Live data (1,039 `prime` bids, 268 funded).** It saturates: 146 of 268 funded bids were at or above 0.95, 129 at or above 0.99. Of those 146, **119 (82%) came right after a successful thought**, a median 22 minutes later. That is Prime being idle, not starved. Of 82 funded bids that came right after a queue timeout, only 19 (23%) crossed 0.95. So a high reading mostly means "Prime was quiet", and a low one does not mean "Prime was fed".
5. **Existing mechanism.** The direct fact already exists: the last try got no seat. That is the condition used.
6. **Reversibility.** Nothing was built on it.

**Verdict:** intensity fails steps 3 and 4 for this use. Prime goes `starved` only after a try that got no seat. With this patch, no-seat tries no longer close acts, so the market's hunger for Prime now grows honestly while Prime is denied the model. Re-check intensity against the new data before reconsidering it.

## Current architecture

- Prime (`services/orion-mind/app/prime_loop.py`) opened a thought act, then called the gateway on route `prime` (pool priority `background`) with one 60 s budget. The gateway used that budget for both the pool wait and generation (`_dispatch_on_pool`). On a timeout the act closed as failed, and the 300 s rest interval started.
- `orion/prime/state.py` `observe()` counted a change in Temporal Self's active arc as a reason to think.
- Pool priorities: `urgent > interactive > perception > system > background`.

## Architecture touched

- GPU pool contract: a new priority level `starved` (`orion/gpu_pool/config.py`, `orion/schemas/gpu_pool.py`, scheduler, shed reasons). An old YAML still loads with new code (the validator inserts `starved` above `system`, the same approach `perception` used).
- Pool routes: `prime` is now `system`, new `prime_starved` is `starved`. Both are system-only routes (hidden from the human picker, refused as overrides).
- llm-gateway: a new optional per-call bus option, `gateway_pool_wait_sec`.
- orion-mind: the Prime runtime, the LLM client (seat wait, gateway error passed up), settings.
- Prime state: new fields, all with defaults (older saved state still loads).

## Files changed

- `config/gpu_pool.yaml`: `starved` level; `prime` -> system; new `prime_starved`.
- `orion/gpu_pool/config.py`, `orion/schemas/gpu_pool.py`: the new level, with old-YAML insertion.
- `orion/gpu_pool/shed.py`: `starved` is sheddable; `cabinet_hot` and `cooling_incident` block it.
- `orion/gpu_pool/scheduler.py`: `starved` holds can be paused for urgent work, like system ones.
- `orion/llm/routes.py`: route sets (`prime` leaves background, both Prime routes are system-only).
- `services/orion-llm-gateway/app/pool_placement.py`, `app/main.py`: the `gateway_pool_wait_sec` option; every bus reply's `meta` now carries `pool_wait_ms`.
- `services/orion-mind/app/llm_client.py`: `pool_wait_sec`; the RPC waits for seat + generation + 15 s; the gateway's typed error goes into `meta["gateway_error"]`.
- `services/orion-mind/app/prime_loop.py`: the try / no-model / thought split, the 5xx retry, the route choice.
- `orion/prime/state.py`: try classification, backoff, quiet floor, attention removed as a trigger.
- `orion/prime/acts.py`: thought act docs and basis (the horizon now covers the seat wait).
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`: `PRIME_STARVED_LLM_ROUTE`, `PRIME_SEAT_WAIT_SEC`.
- `orion/prime/evals/run_prime_frame_eval.py`: 2 new checks. `orion/prime/evals/run_prime_wake_replay.py` + live fixture: new replay eval.
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`: counts "low-priority grants while shed" from `SHEDDABLE`, not a hard-coded pair.
- `.github/workflows/orion-prime-tests.yml`: runs the replay eval.
- Tests: `orion/gpu_pool/tests/test_starved_priority.py`, `orion/prime/tests/test_prime_retries.py`, `services/orion-mind/tests/test_llm_client_seat_wait.py`, plus updated route, shed and Prime tests.
- READMEs: orion-mind, orion-llm-gateway, orion-gpu-pool.

## Schema / bus / API changes

- Added: pool priority `starved`; pool route `prime_starved`; gateway bus option `gateway_pool_wait_sec`; reply `meta.pool_wait_ms` (additive, in a free-form dict); `PrimeStateV1` fields `no_model_streak`, `no_model_tries_total`, `last_try_kind`, `retry_not_before`.
- Removed: `PrimeSnapshotV1.active_arc_id`, `active_arc_label`, `temporal_self_ok` (internal, never stored). `PrimeStateV1.seen_active_arc_id` is retired but kept so older saved state loads.
- Behavior changed: route `prime` is pool priority `system`, not `background`. A Prime try that never reached the model opens no act.
- Compatibility: new code loads an old `gpu_pool.yaml`. An **old** image does **not** load the new YAML (the `prime_starved` route names an unknown priority). The pool, gateway, lane controller and mesh-guardian must be rebuilt with this change. An old gateway ignores `gateway_pool_wait_sec` (Prime then times out at 60 s as before).

## Env/config changes

- Added keys: `PRIME_STARVED_LLM_ROUTE=prime_starved`, `PRIME_SEAT_WAIT_SEC=600` (orion-mind).
- `.env_example` updated: yes. Compose updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (both keys are in the primary checkout's `services/orion-mind/.env`).
- Skipped keys: none from this change. The script's existing divergence list (other services) is unrelated.

## Tests run

```text
orion/prime + orion/gpu_pool/tests + orion/llm 626 passed
services/orion-gpu-pool/tests                  199 passed, 15 skipped
services/orion-llm-gateway/tests               483 passed (new: seat-wait option, pool_wait_ms in reply)
services/orion-mind/tests                      173 passed
services/orion-gpu-lane-controller/tests       110 passed
orion-actions / cortex-exec / hub / vision-council / mesh-guardian / hardware-watch route+priority tests: all passed
.github/workflows/orion-static-gates.yml       40 of 40 run steps passed locally (pip/npm install steps skipped)
scripts/check_gpu_pool_config.py               ok
```

## Evals run

```text
run_prime_frame_eval.py      11/11 PASS (new: attention_background, starved_try)
run_prime_wake_replay.py     3/3 PASS: 28.0/day vs 67.8 live; every day wakes (13, 34, 24, 26); longest gap 6.0 h (12.3 h without the floor); the floor alone woke 3
run_pool_day_eval.py         PASS
run_controller_stale_eval.py PASS
```

## Docker/build/smoke checks

```text
Not run. Not deployed (out of scope by instruction). Live path UNVERIFIED.
```

## Review findings fixed

A code-review subagent read the full diff. Every finding is fixed:

- Finding (should-fix): with a 10-minute seat wait, a Juniper turn that started while Prime was still WAITING threw away Prime's later answer and counted it as a failed thought.
  - Fix: the gateway now reports its seat wait in every reply (`meta.pool_wait_ms`). Prime counts "a Juniper turn overlapped me" from the grant, not from the request. If the gateway sends no wait figure, the old, stricter rule applies.
  - Evidence: `test_a_juniper_turn_during_the_seat_wait_does_not_discard_the_thought` and `test_a_juniper_turn_during_generation_still_preempts` (same setup, opposite outcome), and `test_bus_reply_reports_the_pool_seat_wait`.
- Finding (should-fix): a new gateway asking an OLD pool for `starved` can make the gateway briefly treat the pool as unreachable (a 5 s refusal window for every caller). An old mesh-guardian or lane-controller image cannot load the new YAML.
  - Fix: the restart order below is pool first, orion-mind last. `scripts/rebuild_affected_services.py --plan` confirms the post-pull rebuild covers all five services (athena: gpu-pool, llm-gateway, mesh-guardian, orion-mind; circe: lane controller). Only orion-mind ever sends `starved`, and only after a no-seat try.
  - Evidence: the plan output, quoted in "Restart required".
- Finding (nit): with `PRIME_REFRACTORY_SEC=0`, the backoff was 0, so Prime could retry every 15-second tick forever.
  - Fix: the backoff base has a 60 s floor.
  - Evidence: `test_the_no_model_backoff_has_a_floor_even_with_no_refractory`.
- Finding (nit): the backoff was counted from the start of a try that may have waited 10 minutes.
  - Fix: it is now counted from the end of the try.
- Finding (nit): `PrimeSnapshotV1.temporal_self_ok` was dead.
  - Fix: removed.
- Finding (nit): the replay eval reported a net difference as "thoughts the floor added", and did not say that its settled-act closes come from the old, busier regime.
  - Fix: it now counts the thoughts the floor alone woke (3), and its docstring states the bias (the rate check is conservative; the daily-wake check is easier than live).
- Finding (nit, accepted as designed): a pool recall counts as "no seat", even if generation had started.
  - Fix: this is now stated in the thought act's documentation.

## Restart required

Order matters: the pool must know `starved` before anyone asks for it.

The post-pull rebuild (`rebuild_affected_services.py --plan --base <pre-merge>`) selects all of these: athena `orion-gpu-pool`, `orion-llm-gateway`, `orion-mesh-guardian`, `orion-mind`; circe `orion-gpu-lane-controller`. It does not control their order, so if you run them by hand, use this order (each from the clean primary `main` checkout):

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build              # 1. athena: the pool learns `starved`
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build   # 2. circe, after git pull there (/health is 503 until rebuilt)
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build         # 3. athena
scripts/safe_docker_build.sh orion-llm-gateway up -d --build           # 4. athena
scripts/safe_docker_build.sh orion-mind up -d --build                  # 5. athena: the only sender of `starved`
```

## Risks / concerns

- Severity: medium. Concern: between `git pull` on a host and the rebuild of a service that reads the live checkout's YAML (lane controller on circe, mesh-guardian), that service cannot load the pool config. Mitigation: rebuild in the order above; the lane controller's `/health` turns 503 so the gap is visible.
- Severity: low. Concern: a Prime frame can now be up to 10 minutes old by the time the model reads it (the frame is built before the seat wait). Mitigation: the frame carries its own time.
- Severity: low. Concern: rolling orion-mind back after it has saved the new state fields would make the old code discard Prime's saved state (strict schema): concerns and open questions would be lost. Mitigation: roll back only together with a state reset, or leave it.
- Severity: low. Concern: Prime's tick is blocked up to ~11 minutes during a try (settling and observing wait). Mitigation: settling is every 5 minutes anyway; nothing is lost, only delayed.
- UNVERIFIED: every live claim about the new behaviour (fewer failed thoughts, `starved` grants, the wake rate). Check after deploy: `gpu_pool_leases` rows for holder `orion-mind` with priority `starved`/`system`, and `prime_try_no_model` / `prime_thought ... route=` log lines.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3065

🤖 Generated with [Claude Code](https://claude.com/claude-code)
