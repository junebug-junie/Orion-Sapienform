## Summary

- **New `situation.update` graph** in orion-durable-runs: one self-driven LangGraph thread per UTC day (`situation:juniper:<date>`) that keeps Orion's running situation and publishes it as `SituationStateV1` to Redis `orion:situation:latest` and bus `orion:situation:state`. This is step 2 of the approved Situation Graph design (#2526). **Shadow: nothing reads it yet.**
- **Inputs:** finished chat turns, finished episode distills, and a 15-minute clock. A single writer coalesces bursts.
- **Facts are re-derived from `episode_memory` on every step,** so a step is idempotent. The graph checkpoints once per step, not per node, and day threads are seeded from the previous day and then retired, so the runner's resume sweep stays bounded.
- **Current state only from Juniper's own end dates.** `whereabouts` and `doing` hold only memories whose end date Juniper stated (writer v5, #2535). Undated `happened` memories go into `recent`. The live check found a newest-place rule naming "sang karaoke at the Jackalope bar during the Austin trip" as her whereabouts.
- **Priming is recall by shared referent** (people, places, projects): memories ranked by strength decayed over their half-life, a 0.4 s budget, skipped when the cues haven't changed. No graph reads and no text similarity.

## Outcome moved

Orion now holds a durable, inspectable picture of the current situation, kept off the chat turn's critical path. Step 3 replaces the chat turn's three per-turn searches (about 2 s of search time) with one read of this state.

Live dry run against production episode memory at the incident time (10-06 23:24):
- `whereabouts` is empty. That is truthful: the trip was distilled before v5, so it has no stated end date.
- 3 `waiting_on` follow-ups and 3 `recent` events.
- 6 primed memories in 6.5 ms.

## Current architecture

Chat rebuilt context on every turn from three searches keyed on the raw message (continuity, belief, finalize-reflect). Nothing carried "what is true right now" between turns. durable-runs ran GPU-admitted, submitted workflows only.

## Architecture touched

- orion-durable-runs: a new self-driven workflow, two new subscriptions, and a Redis and bus projection.
- Shared contracts: the `SituationStateV1` schema, a new channel, and the `situation.update` workflow literal.
- Episode validator: `named_referents` factored out of the name check and reused for cues.
- The metric definition lock.

## Files changed

- `orion/schemas/situation_state.py`: `SituationStateV1` plus its fact, primed, lapsed and event models, and constants.
- `orion/schemas/registry.py`: registers `SituationStateV1`.
- `orion/schemas/durable_run.py`: adds `situation.update` to `DurableWorkflowV1`.
- `orion/bus/channels.yaml`: adds `orion:situation:state`; adds durable-runs as a consumer of `orion:chat:history:turn` and `orion:durable:run:state`.
- `config/metrics/metric_definitions.lock.json`: re-locked for those three bus-channel definition changes.
- `orion/memory/episode/validate.py`: `named_referents()` and `referent_name()`, factored from `ungrounded_names` (same matching).
- `services/orion-durable-runs/app/situation_graph.py`: the graph and its pure reducers (facts, lapsed, cues, ranking, content hash).
- `services/orion-durable-runs/app/situation_store.py`: two read-only SQL queries.
- `services/orion-durable-runs/app/situation_driver.py`: the single writer (queue, coalescing, day threads, seed, retention, tick, trace row).
- `services/orion-durable-runs/app/main.py`: builds and starts the driver, routes the two event kinds, health field.
- `services/orion-durable-runs/app/runner.py`: `SELF_DRIVEN_WORKFLOWS`; the resume sweep skips situation threads.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: the `SITUATION_*` keys and `CHANNEL_CHAT_HISTORY_TURN`.
- `services/orion-durable-runs/README.md`: the situation graph section and deploy order.
- `services/orion-durable-runs/tests/test_situation_graph.py`: 22 tests.

## Schema / bus / API changes

- **Added:** `SituationStateV1` (`situation.state.v1`), channel `orion:situation:state`, Redis key `orion:situation:latest`, workflow `situation.update`.
- **Behavior changed:** durable-runs now subscribes to `orion:chat:history:turn` and `orion:durable:run:state`.
- **Compatibility:** adding `situation.update` to the `Literal`/`extra="forbid"` workflow type means **orion-sql-writer must deploy first**, or it drops these trace rows. Hub and orion-actions already skip rows they can't parse and filter by workflow. Only steps that change the situation write a trace row, so ticks don't flood Hub's activity surface.

## Env/config changes

- **Added keys:**
  - `SITUATION_GRAPH_ENABLED=true`
  - `SITUATION_TICK_SEC=900.0`
  - `SITUATION_DEFAULT_TTL_HOURS=48.0`
  - `SITUATION_PRIME_TIMEOUT_SEC=0.4`
  - `SITUATION_RETENTION_DAYS=2`
  - `SITUATION_REDIS_TTL_SEC=604800`
  - `CHANNEL_CHAT_HISTORY_TURN=orion:chat:history:turn`
- **`.env_example` updated:** yes, with compose passthrough (`check_service_env_compose_parity.py orion-durable-runs` reports OK, 56 keys).
- **Local `.env` synced:** yes. `python scripts/sync_local_env_from_example.py --all-keys orion-durable-runs` added all 7 keys to the primary checkout's `services/orion-durable-runs/.env`. The default sync skipped them, because their prefix is outside its sync list.
- **Skipped keys:** none.

## Tests run

```text
services/orion-durable-runs: pytest tests -q           299 passed, 72 skipped
orion/memory/episode/tests + metric drift               348 passed, 22 skipped (lock --update)
registry/channel tests (63 files)                       same failures as main, no new ones
check_env_key_single_source.py                          OK
```

## Evals run

```text
Live read-only run of the real graph + real SQL against production episode_memory
(in-memory checkpointer, nothing published), at 2026-10-06T23:24Z and 2026-10-07T03:00Z:
  before the fix: whereabouts = "sang karaoke at the Jackalope bar during the Austin trip" (WRONG)
  after:          whereabouts = none (no stated end date yet), waiting_on 3, recent 3,
                  cues 4, primed 6, prime 6.5 ms
```

The step 3 replay eval (10-05 → 10-06 with the chat path reading the state) comes with step 3.

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-durable-runs/docker-compose.yml config   renders all SITUATION_* keys
Not built or deployed from this branch; deploy from the primary checkout on main after merge.
```

## Review findings fixed

- Finding: a failed or timed-out priming was not retried for up to 6 hours. The new cues were saved, so the next step's "cues unchanged" shortcut kept the stale primed set.
  - Fix: the state stores `primed_cues`, the cues the primed set was actually built from, and the shortcut compares against those. This replaces the redundant `cues_revision` (finding 6).
  - Evidence: `test_failed_priming_is_retried_on_the_next_step`.
- Finding: a redelivered event skipped the graph but reported the previous step's `changed=True`, so a stale trace row was published.
  - Fix: `ingest` resets the step flags, and the driver returns early on `skipped`.
  - Evidence: `test_duplicate_event_publishes_no_row`.
- Finding: a fact pushed out of a full 3-item slot was recorded as lapsed while still true.
  - Fix: lapsing is judged against `current_ids` (every current fact), and the caps apply only to display.
  - Evidence: `test_fact_pushed_out_of_a_full_slot_is_not_lapsed`.
- Finding: the facts query had `LIMIT 200` with no `ORDER BY`.
  - Fix: it orders newest first. 18 rows qualify today.
- Finding: old threads were deleted only on a seeded first step, and never retried.
  - Fix: deletion runs on every new day thread until it succeeds, covering a 31-day range.
  - Evidence: `test_retirement_is_retried_until_it_succeeds`.
- Noted, not changed: Hub's activity surface will show `situation.update` as a run that completes many times under one run_id (the day thread). Hub's hooks filter by workflow, and only changed steps publish.

## Restart required

After merge, from the primary checkout on main. sql-writer goes first:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-sql-writer orion-durable-runs; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done
```

Proof after deploy:
- `curl -s localhost:8124/health | jq .situation` shows `steps ≥ 1`.
- `redis-cli -u $ORION_BUS_URL get orion:situation:latest` returns a `situation.state.v1` document.
- `substrate_durable_run_state` has `workflow='situation.update'` rows.

## Risks / concerns

- **Medium: undated stays don't count as current.**
  - Concern: `whereabouts` stays empty until a v5 distill stores a stated end date. Older trips, and any stay Juniper never gives an end for, show only as `recent`.
  - Mitigation: this is deliberate. A guessed "now" was the bug. Step 3 renders `recent` facts as "as of".
- **Low: priming can show something Juniper hasn't confirmed.**
  - Concern: priming ranks by referent overlap only, so a pending (unconfirmed) memory can be primed. It carries its `confirmation` state for step 3 to render.
  - Mitigation: none needed in shadow, since nothing reads the state yet.
- **Low: the resume sweep still walks every checkpoint.**
  - Concern: that is the existing behaviour; this graph only adds about one row per changed step.
  - Mitigation: day threads are retired after `SITUATION_RETENTION_DAYS`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
