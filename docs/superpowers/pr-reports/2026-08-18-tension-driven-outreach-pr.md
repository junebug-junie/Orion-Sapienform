## Summary

- Replaces `endogenous_outreach.py`'s randomized-timer stub ("Orion speaks first") with a real, inspectable trigger: has the same node been winning `orion.attention.tension`'s live Borda competition for a sustained, unbroken run of real ticks, right now.
- New `services/orion-hub/scripts/tension_outreach_trigger.py::current_run()` walks recent `substrate_field_state` rows for a consecutive same-winner run, gated by `MIN_RUN_LENGTH` (derived from a real replay of live history, not guessed).
- `FieldStateV1.tension_borda_winner_target_id` (new field) persists the Borda winner identity PR #1699 deliberately left unpersisted (no consumer at the time). This patch is the consumer.
- `HUB_ENDOGENOUS_OUTREACH_PROBABILITY` removed (kill means kill); `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` added as the one trigger constant kept operator-tunable.
- Companion **design-only** doc for the level-aware half of the same problem (`orion.field.regime.channel_regime()`, unwired, zero consumers) — proposal-mode, no code, ends on an open question for Juniper.

## Outcome moved

Orion's unprompted outreach now fires on a real, honestly-scoped internal signal instead of a coin flip. It can only ever claim "I noticed X change" (the underlying gate is a change-detector, not a level-detector) — a distress-shaped ("I'm worried about X") trigger is real, separate follow-up work, not built here.

## Current architecture

`services/orion-hub/scripts/endogenous_outreach.py`'s `_should_roll()` was a stub: `random.random() < self.probability`. Its own docstring named itself the sanctioned replacement seam since 2026-08-14. `orion.attention.tension`'s deviation-tension package (PR #1699/#1701, merged) already computes a live Borda-winner per tick but didn't persist the winner's identity — only the aggregate `tension_deviation_pressure` scalar.

## Architecture touched

- `orion/schemas/field_state.py` — schema contract (new field)
- `services/orion-field-digester/app/digestion/tension.py` — producer
- `services/orion-hub/scripts/{tension_outreach_trigger.py (new), pg_engine.py (new), endogenous_outreach.py, main.py}` — consumer
- `services/orion-hub/app/settings.py`, `.env_example`, live `.env` — env contract

## Files changed

- `orion/schemas/field_state.py`: `tension_borda_winner_target_id` field.
- `services/orion-field-digester/app/digestion/tension.py`: writes the new field once per tick.
- `services/orion-field-digester/tests/test_tension_pressure_baseline.py`: 3 new tests for the new field.
- `services/orion-hub/scripts/tension_outreach_trigger.py` (new): the real trigger.
- `services/orion-hub/scripts/pg_engine.py` (new): one cached engine shared by this patch's two same-tick DB reads.
- `services/orion-hub/scripts/endogenous_outreach.py`: real trigger wiring, async `_should_roll()`, honest prompt block.
- `services/orion-hub/scripts/main.py`: constructs `EndogenousOutreach` with the real `trigger_evaluator`.
- `services/orion-hub/app/settings.py`, `.env_example`: `HUB_ENDOGENOUS_OUTREACH_PROBABILITY` removed, `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` added.
- `services/orion-hub/tests/test_tension_outreach_trigger.py` (new), `test_endogenous_outreach.py`: 12 + 88 tests.
- `services/orion-hub/README.md`: section 4.1 rewritten.
- `docs/superpowers/specs/2026-08-16-tension-driven-outreach-design.md` (new): full design + review-findings writeup.
- `docs/superpowers/specs/2026-08-16-level-aware-significance-design.md` (new): companion Path 2, design-only.
- `config/metrics/metric_definitions.lock.json`: merge-base annotation refresh (0 content changes).

## Schema / bus / API changes

- Added: `FieldStateV1.tension_borda_winner_target_id: str | None`.
- Removed: nothing on the schema.
- Behavior changed: `chat.history.message.v1` outreach messages now sometimes ground on a real tension reason (prompt content only, not the schema).
- Compatibility notes: new field defaults to `None`; existing consumers unaffected.

## Env/config changes

- Added keys: `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` (default 8).
- Removed keys: `HUB_ENDOGENOUS_OUTREACH_PROBABILITY`.
- `.env_example` updated: yes.
- local `.env` synced: yes, directly (removed the stale key, which had silently reappeared via an earlier sync run reading the still-old primary-checkout `.env_example`; added the new key).
- skipped keys requiring operator action: none.

## Tests run

```
services/orion-hub: 88 tests pass (test_tension_outreach_trigger.py + test_endogenous_outreach.py)
services/orion-field-digester: 12 tests pass (test_tension_pressure_baseline.py)
Run live inside the running orion-athena-hub / orion-athena-field-digester containers
(their real dependency sets -- host has neither pytest nor the app's runtime deps).
```

## Evals run

No dedicated eval harness for this seam; the design doc's metric-quality-gate write-up (live-data replay deriving `MIN_RUN_LENGTH`) is the closest equivalent and is documented there.

## Docker/build/smoke checks

```
python scripts/check_definition_drift.py --gate  -> PASS, 0 changes
python scripts/check_service_env_compose_parity.py orion-hub  -> N/A (env_file: wholesale)
python scripts/check_settings_defaults.py orion-hub  -> not in the checker's audited-service allowlist, skipped
```

## Review findings fixed

8 finder angles ran against the diff; top finding independently surfaced by 4 of 8.

- Finding: `_should_roll()` ran its synchronous Postgres-backed evaluator inline, with no `asyncio.to_thread` wrap, on Hub's single uvicorn worker.
  - Fix: `_should_roll()` is now `async`, runs the evaluator via `asyncio.to_thread`.
  - Evidence: `test_trigger_evaluator_does_not_block_the_event_loop`.
- Finding: `status()`'s `last_tension_reason` went stale when an earlier gate blocked a tick before the trigger ever ran.
  - Fix: the blocked branch now clears `_last_tension_reason`, mirroring the existing `force` branch.
  - Evidence: `test_status_does_not_report_a_stale_tension_reason_after_a_blocked_tick`.
- Finding: two independent connection pools against the same DB on one tick.
  - Fix: new `scripts/pg_engine.py`, one cached engine shared by this patch's two call sites.
  - Evidence: code inspection; 20+ other pre-existing call sites elsewhere left alone (disclosed debt).
- Finding: `latest_deviation_pressure` actually held the run's peak, not the latest value.
  - Fix: renamed to `peak_deviation_pressure` everywhere.
  - Evidence: `test_reported_magnitude_is_the_max_within_the_run_not_the_latest_value`.
- Finding: `MIN_RUN_LENGTH` was a hardcoded constant, no way to retune without a deploy.
  - Fix: `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` (settings.py), wired via `functools.partial`.
  - Evidence: `test_min_run_length_is_overridable_per_call`.
- Finding: no contract test on the raw `field_json->>'...'` SQL keys vs. the real schema fields.
  - Fix: added `test_raw_sql_json_keys_match_the_real_schema_fields`.
  - Evidence: test passes; would fail on a future silent rename.
- Finding: `LOOKBACK_MINUTES`'s comment overstated coupling to field-digester's actual poll cadence.
  - Fix: reworded to disclose this is an intentionally generous, decoupled margin.
  - Evidence: module docstring diff.
- Finding: stale `settings.py` section header still said "stub random trigger".
  - Fix: updated.
  - Evidence: settings.py diff.
- Finding: no evidence the live `.env` was kept in sync with `.env_example`'s removed key.
  - Fix: re-verified directly — `HUB_ENDOGENOUS_OUTREACH_PROBABILITY=0.15` had in fact silently reappeared in the shared checkout's live `.env` (an earlier sync run read the still-old primary-checkout `.env_example`). Removed it directly; added the new key.
  - Evidence: `grep` before/after on the live `.env`.

Not fixed, disclosed: ~20 pre-existing `create_engine`/`POSTGRES_URI` call sites elsewhere in `orion-hub/scripts/*.py` still don't use the new shared `pg_engine` module — an unrelated, invasive migration across files this patch has no other reason to touch. Real follow-up work.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub

docker compose \
  --env-file .env \
  --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml \
  up -d --build orion-field-digester
```

## Risks / concerns

- Severity: low
- Concern: `MIN_RUN_LENGTH=8` and `LOOKBACK_MINUTES=10.0` are derived from a single one-time 2-hour replay against one `FieldTensionCompetition` tuning snapshot, not yet validated against real post-deploy firing-rate data.
- Mitigation: `MIN_RUN_LENGTH` is now operator-tunable without a deploy; both constants are disclosed as revisit-later in the design doc, same discipline as every other constant in this arc.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1707
