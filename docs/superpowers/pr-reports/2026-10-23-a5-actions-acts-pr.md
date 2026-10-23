## Summary

- Two orion-actions activities are now acts in the act ledger, one act per attempt (body economy Patch A5, spec `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`, "Patch A results" producer split item 3).
  - `actions.daily_metacog` / `infer`: one `daily_metacog_v1` attempt.
  - `actions.journal` / `make`: one in-process journal compose.
- Each act opens after the dedupe gate and the semaphore wait, so queue time is not counted as the act. It closes once: 1 when the artifact is delivered, 0 on a delivery failure, an exception, or a cancellation.
- Each act's correlation id is the id its GPU leases carry, so the ledger's close-time cost fill joins.
  - Daily metacog: the attempt's envelope id.
  - Journal: a new per-compose id, sent as the compose request's correlation id.
- Journal ownership rule (stated below) keeps a journal from being counted twice with orion-durable-runs.
- New flag `ACTIONS_ACT_EMIT_ENABLED`, on by default. `orion:grammar:event` now lists `orion-actions` as a producer. The definition-drift lock is refreshed.

## Journal ownership rule

**The process that runs a journal's compose owns its act.**

- orion-actions composes in-process in `_dispatch_journal` → `_run_journal` → cortex-orch `journal.compose`. It emits `actions.journal` for those composes.
- A journal submitted as a durable `journal.compose` run is composed by orion-durable-runs. It will be owned by that service's run-seam act producer, which is A5 producer 1 and **not shipped yet**. Until it lands, durable journals have no act. Today the only durable journal is world_pulse_digest, via `app/world_pulse_journal.py` and cortex-orch's durable ingress.
- The durable path has no in-process fallback and never calls `_dispatch_journal`, so no journal can be counted by both services.
- Pinned by `test_world_pulse_journal_goes_durable_and_emits_no_actions_act`.

## Outcome moved

The journal family gets graded acts with wall seconds. Once live, it also gets GPU slot-seconds.

Live rates (last 7 days to 2026-10-23):

- **actions.journal: about 128 acts/day.**
  - 788 journals in 7 days came from the in-process path: metacog 771, scheduler 16, notify 1. That is about 113 successes/day.
  - Failed attempts are not stored anywhere. To estimate them, I compared the actions logs since its 18:06 restart: 8 failures to 59 successes (mostly `gpu_pool_unavailable:deadline`). That is about 12% failures, so about 128 attempts/day.
  - Entries from other writers are excluded: author `juniper_primary`, `world_pulse`, `self_study`, `embodiment`, `orion_day`.
- **actions.daily_metacog: 0 acts/day.** `ACTIONS_DAILY_METACOG_ENABLED=false` live. Juniper turned it off on purpose on 10-11 because it only emailed telemetry. The last delivery was 2026-09-18. The emitter is passive: it emits nothing until the job runs. It does not re-enable anything.

## Current architecture

- `_execute_daily` and `_dispatch_journal` are closures in the `lifespan` of `services/orion-actions/app/main.py`.
- They call cortex-exec (daily plan) and cortex-orch (journal.compose) over RPC.
- They audit to `orion:actions:audit`.
- Neither published an act.

## Architecture touched

- orion-actions only, plus the `orion:grammar:event` producer list.
- The event is built in `services/orion-actions/app/act_emit.py`. `orion.grammar.acts.act_opened_event` was not usable: its `_provenance` only accepts orion-hub and orion-thought, and the shared module was out of scope. `build_act` is reused as is. `orion/grammar/act_ledger.py` is untouched.

## Files changed

- `services/orion-actions/app/act_emit.py`: new. Act builders, horizons, `ActTrace` (open once, close once, never raises), bus publisher.
- `services/orion-actions/app/main.py`: `_act_publish`. `ActTrace` in `_dispatch_journal` and `_execute_daily`. `_run_journal` takes `compose_correlation_id`. `_execute_daily` handles `daily_metacog_v1` only; `daily_pulse_v1` opens nothing. Adds `except BaseException` close-and-reraise for cancellation.
- `services/orion-actions/app/settings.py`, `.env_example`: `ACTIONS_ACT_EMIT_ENABLED=true`. The compose file uses `env_file` only, so no compose edit is needed.
- `services/orion-actions/README.md`: section "Act ledger producers".
- `services/orion-actions/tests/test_act_emit.py`: new, 16 tests. They drive the real lifespan closures.
- `orion/bus/channels.yaml`: adds `orion-actions` to the producers.
- `config/metrics/metric_definitions.lock.json`: re-locked. It reports "high removed metric://bus_channel/orion-biometrics/orion:grammar:event" and "added .../orion-actions/...". That is a rename, not a removal: the channel metric's URI is keyed on the alphabetically first producer, and no other file references the old URI.

## Act contract

- **Key:** `<dedupe key>|<attempt correlation id>`.
  - A redelivered trigger has the same correlation id, so it maps to the same act id. The writer is first-close-wins.
  - A scheduler retry is a new attempt with a new correlation id, so it is a new act. It spends its own GPU time.
  - A restart in the middle of an attempt leaves its act open. The writer expires it at 0 when its horizon passes.
- **Expect:** `artifact_produced`. The horizon is the job's own deadline:
  - journal: `ACTIONS_EXEC_TIMEOUT_SECONDS` + 60 s, which is 480 s live.
  - daily metacog: 4 × timeout + 60 s (2 generation attempts × 2 RPC attempts), which is 1740 s live.
- **Outcome:**
  - journal: 1 when the draft came back and `journal.entry.write.v1` was published.
  - daily metacog: 1 when the report was delivered (notify accepted).
- **Cost:** `ActCostV1(seconds=wall)`. No ActCostV2-only fields.
- **correlation_id:** the id the attempt's GPU leases carry.
  - Journal: `uuid5(dedupe_key|trigger correlation id)` is sent as the cortex-orch compose request's correlation id.
    - The trigger's own id cannot be used. Every walkway job in one scheduler tick shares it, and a collapse's chat reply shares it too. The ledger charges each act every lease under its id, so a shared id would bill each job for the whole tick.
    - The journal write keeps the trigger's id, as before.
    - Verified live 10-23 that compose leases carry the compose request's id. Two metacog-digest journals (`0c95a159…`, `525afc83…`) each had their cortex-exec metacog leases under their request's `turn_correlation_id`.
  - Daily metacog: the attempt envelope's id. The scheduler mints one id per attempt.
    - The path is `_run_plan` → `derive_child` → cortex-exec `corr_id = env.correlation_id` → LLM client → gateway lease.
    - UNVERIFIED live, because the job is off.

## Schema / bus / API changes

- Added: orion-actions publishes `grammar.event.v1` `act_opened`/`act_closed` on `orion:grammar:event`. The schema is unchanged.
- Behavior changed: the cortex-orch `journal.compose` request now carries a per-compose correlation id. The journal write, the audit rows and the notifications keep the trigger's id. The RPC reply is keyed on `(reply_channel, request corr)`, so it still matches.

## Env/config changes

- Added keys: `ACTIONS_ACT_EMIT_ENABLED` (default true).
- `.env_example` updated: yes.
- Local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys orion-actions` → `+ACTIONS_ACT_EMIT_ENABLED='true'` (primary checkout).
- Skipped keys: none.

## Tests run

```text
services/orion-actions: pytest tests -q -> 274 passed (16 new in tests/test_act_emit.py)
CI static gates run locally (orion-static-gates.yml subset: no_personal_terms, env sync/parity, env-compose parity,
compose logging, grammar producer catalog, ci_service_image_sources, metric_lineage --gate, definition_drift --gate,
inner_state_registry, body_economy_contract + test, scripts stdlib shadow, hostname refs, compose relative mounts,
journal_dispatch_registry, daily_schedule_collisions, sentience_instruments, system_health_producers,
async_routes_not_blocking, chat_route_poachers) -> 22/22 PASS
Mutation checks: removing the journal cancellation close fails test_journal_cancellation_closes_zero_and_propagates;
removing an except-path close fails the failure tests; a wrong act correlation id fails the outbound-envelope asserts.
```

## Evals run

```text
None. orion-actions has no evals/ directory. Follow-up: the live eval is graded closes per family in grammar_acts.
```

## Docker/build/smoke checks

```text
Not run (no dependency or image change).
```

## Review findings fixed

- **MUST.** All walkway journals in one scheduler tick shared one correlation id, so each act would have been charged the whole tick's GPU time.
  - Fix: a per-compose correlation id, `act_emit.compose_correlation_id`, sent as the compose request's id.
  - Evidence: `test_two_composes_sharing_a_trigger_id_get_distinct_cost_keys`.
- **SHOULD.** A collapse-response journal shared its id with the collapse chat reply.
  - Fix: the same per-compose id fixes it.
- **SHOULD.** A cancel during the failure audit left the journal act open.
  - Fix: the close now runs before the audit, as on the daily path.
- **SHOULD.** No test covered the cancellation branches.
  - Fix: one test per path. Mutation-checked.
- **SHOULD.** No test pinned "act correlation id = the outbound request id".
  - Fix: the success tests now compare against the captured outbound envelope.
- **SHOULD.** The docs read as if durable journals were already covered.
  - Fix: the wording now says the durable-runs producer is not shipped, so durable journals have no act until it lands. Updated in the docstring, the settings comment, channels.yaml and this report.
- **NIT.** The metrics lock records a "removed" line.
  - Fix: explained above as a rename.
- **NIT (not fixed).** A cancel-path close can add up to 5 s to shutdown per in-flight job.
  - Reason: accepted. At most a few jobs are in flight at once.

## Restart required

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
```
Run this from the clean primary `main` checkout after the merge.

## Risks / concerns

- **Severity: low.** The act count is high, about 128 per day, almost all of them metacog-digest journals. This is real activity, not noise, but it makes this family dominate the act ledger's row count. **Mitigation:** none needed for correctness. The economy ranks each family against its own history.
- **Severity: low.** The GPU cost fill runs once, when the writer processes the close. A lease released after that query is missed. **Mitigation:** check after deploy (query below).
- **Severity: low.** This PR touches the same files as the other A5 producer PRs: `channels.yaml` and the metrics lock. **Mitigation:** the producer entry sits at a different list position. Re-run `check_definition_drift.py --update` after rebasing.
- **Live verification: UNVERIFIED until deploy.** Check with: `SELECT actor, count(*), avg(outcome_value) FROM grammar_acts WHERE actor LIKE 'actions.%' AND opened_at > now()-interval '1 day' GROUP BY 1;`

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
