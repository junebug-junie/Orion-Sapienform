# Body economy Patch C3b (thought): shadow bid at the painting generate step

## Summary

- The `generate` step of a `reverie.visual` run now asks the body economy "would you fund this GPU work?", and ignores the answer. The answer is only logged, next to what the step really did.
- One site covers both producers: waking paintings (`thought.painting`) and dream-carry image hops (`thought.dream_hop`).
- The bid's `want_id` is the step's correlation id. That is the `make` act's correlation id when this step opens the act. A later generate of the same attempt names the act it ran under.
- The bid is a background task. The step never waits for it, and its result, GPU work and act events are the same whatever the economy answers.
- Switch `THOUGHT_PHYSIOLOGY_SHADOW_BID_ENABLED`, shipped on.

## Outcome moved

Patch D needs evidence of where the economy and the real painting gates (thermal reflex, single-flight lock) disagree. Each would-be GPU render now leaves one `physiology_shadow_bid ... shadow=<funded|deferred|none> real=<ran|blocked:reason>` line, and durable-runs writes the decision to `want_ledger`.

## Current architecture

Patch C1 built the market (`POST /physiology/bid` on durable-runs) and the client (`orion.physiology.client`). Nothing in orion-thought called it. A painting's `make` act opens at its first real generation (`_mark_generating` sets `act_opened_at`), on the generate step's per-RPC correlation id.

## Architecture touched

- orion-thought only, plus one contract registration and the env sync list.
- No new bus channel.

## Files changed

- `services/orion-thought/app/physiology_shadow.py`: new. Same `ShadowBid` helper as the hub patch (place in background, log beside the real outcome, release a funded-but-blocked reservation, swallow every failure).
- `services/orion-thought/app/visual_steps.py`:
  - `generate_step` places the bid after the replay, adoption and in-flight checks (these are not a new want) and settles it in a `finally`.
  - The rest of the step, from the thermal gate on, moved unchanged into `_generate_gated`.
  - `_generate_work` reports whether the GPU call started and whether this call opened the act.
  - `_mark_generating` stores the opening correlation id beside `act_opened_at` (`stage_json.act_correlation_id`), so a later generate can name the act.
- `services/orion-thought/app/settings.py`, `.env_example`: the switch. The client's default URL (`orion-athena-durable-runs:8121`) resolves on app-net; checked from the running container.
- `scripts/sync_local_env_from_example.py`: `THOUGHT_PHYSIOLOGY_SHADOW_BID_ENABLED` added to `SYNC_EXACT` (no `THOUGHT_PHYSIOLOGY_` prefix is synced).
- `config/physiology/contract.yaml`: the helper registered as `display`.
- `services/orion-thought/tests/conftest.py`: autouse fixture so tests never POST to the live market.
- `services/orion-thought/tests/test_physiology_shadow_bid.py`: new (11 tests).
- `services/orion-thought/README.md`: section.

## Correlation join

| family | want_id | act correlation id | verified by |
|---|---|---|---|
| `thought.painting` | generate step `req.correlation_id` | the `act_opened` event's `correlation_id` (same `req`) | `test_generate_is_identical_for_every_answer_and_joins_the_act` asserts want_id == act_opened correlation_id |
| `thought.dream_hop` | same | same | `test_dream_hop_bids_as_its_own_family` |
| either, re-generate in the same attempt | its own step id | `act_correlation_id=<opening step id>` on the log line | `test_a_later_generate_names_the_act_it_ran_under` |

## Real-gate field

`real=ran` once the thermal gate, the single-flight lock and the stage compare-and-set passed and the GPU call started, whatever the render's fate. Otherwise `real=blocked:<the step's reason>`: `thermal_refused`, `deferred_busy`, `stage_changed`, `generate_deadline_exceeded`, ... A step that returns before its work decides settles when the work ends.

## Schema / bus / API changes

- Added: `stage_json.act_correlation_id` on `reverie_visual_attempt` (an additive key in an existing JSON column; written once, with `act_opened_at`; read only by the shadow log).
- Removed / Renamed: none.
- Behavior changed: none for the step (shadow only).
- Compatibility notes: an attempt opened before this deploy has no `act_correlation_id`; its re-generate logs `act_correlation_id=-`.

## Env/config changes

- Added keys: `THOUGHT_PHYSIOLOGY_SHADOW_BID_ENABLED=true` (orion-thought).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-thought/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-thought: pytest tests -> 782 passed, 32 skipped
  (incl. test_physiology_shadow_bid.py 11, test_visual_steps.py, test_painting_act.py)
orion-static-gates.yml python steps (38 gates, incl. check_body_economy_contract) -> all OK
```

## Evals run

```text
None. The shadow-vs-real comparison needs live decisions (scripts/analysis/body_economy_shadow_vs_real.py, another agent).
```

## Docker/build/smoke checks

```text
docker exec orion-athena-thought: GET http://orion-athena-durable-runs:8121/health -> 200 (the client's default host resolves on app-net).
The live bid from a real generate step is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: a step that returned before its work started (deadline, cancel) settled `blocked`, and a funded reservation could be released for a render that then ran.
  - Fix: when the work task is still pending, the line settles from the task's done callback.
  - Evidence: `test_a_step_that_returns_before_its_work_starts_settles_when_the_work_does` (fails on the settle-in-finally version).
- Finding: a re-generate in the same attempt logged `ran` under an id no act carries.
  - Fix: the opening write stores the act's correlation id; a later generate logs it as `act_correlation_id`.
  - Evidence: `test_a_later_generate_names_the_act_it_ran_under`.
- Finding: shadow construction was outside any try.
  - Fix: `_painting_shadow` builds and places inside a try and returns None on failure.
- Finding: the invariance test compared field names only; no "never waits" test.
  - Fix: compare full results, act event payloads and GPU calls across the four answers plus a flag-off baseline; a slow-economy test.
- Finding: `work_started` shadowed a local of the same name in `_generate_work`.
  - Fix: renamed to `gpu_started`.
- Finding: thermal-retry volume and lost act opens were undocumented.
  - Fix: README notes (group by `object_ref=durable_run:<run_id>`; a lost open joins nothing).

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

Run it from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low. Concern: during a heat lockdown, a held painting bids once per thermal retry (every 300 s), so want_ledger gets about 288 rows per run per day. Mitigation: these are honest "I want GPU now" asks; the report groups by run.
- Severity: low. Concern: durable-runs admission may also bid for `reverie.visual` runs (another patch). Mitigation: that bid is at admission with the run id; this one is at the generate step with the step id, so they do not collide, but a report should not sum them as one family.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2927

🤖 Generated with [Claude Code](https://claude.com/claude-code)
