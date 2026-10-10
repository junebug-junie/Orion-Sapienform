## Summary

- Orion's self-started curiosity was being handed the same old "a conversation went badly" reading over and over. One repair turn from days ago was re-offered as a fresh curiosity candidate on every tick. That happened about every 74 seconds, around the clock.
- Fix: a chat turn's repair level now fades with the turn's age. It uses the same 30-minute linear fade that prediction-error sources already use (`PressureConfig().prediction_error_decay_horizon_seconds` = 1800 s). The strongest *faded* level wins.
- A fresh repair turn still produces a candidate. A 0.913 spike stays above curiosity's 0.6 floor for about 10 minutes and then drops out.
- No new env keys, no new knob, no schema change.
- Approved by Juniper 2026-10-10 (spec `docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md`, PR #2528, "Real bugs found" #1).

## Outcome moved

Repair-sourced curiosity candidates, read-only replay of every persisted candidate set against the live chat projection:

| day (UTC) | candidate sets | with repair candidate, before | with repair candidate, after |
|---|---|---|---|
| 10-02 | 1240 | 1240 | 8 |
| 10-03 | 1262 | 1262 | 1 |
| 10-04 | 1346 | 1346 | 8 |
| 10-05 | 1352 | 1352 | 9 |
| 10-06 | 1223 | 1223 | 10 |
| 10-07 | 1184 | 1184 | 0 |
| 10-08 | 1189 | 1189 | 0 |
| 10-09 | 1170 | 1170 | 0 |

Before the fix, **every** candidate set carried a repair candidate. That was not just the 0.913 spikes. A 0.698 turn from 2026-09-13 had already been replayed, unchanged, for weeks before the first 0.913 turn arrived on 10-02. After the fix, only days with a real HIGH turn produce repair candidates, about 8-10 ticks per spike. The expected daily count on a calm day is 0.

## Current architecture

`BiometricsSubstrateWorker._repair_appraisal_from_chat` (`services/orion-substrate-runtime/app/worker.py`) read the chat-session projection. That projection holds every turn since 2026-07-24, 2,798 turns. The function took the max `repair_pressure_level` across all of them with no age term. `endogenous_curiosity_candidates` then emitted it whenever it was at or above `min_repair_level` (0.6). Because candidates sort strongest-first, that replay also took a budget slot on every tick.

## Architecture touched

- `orion/substrate/endogenous_curiosity.py`: new pure `repair_appraisal_from_chat_turns(turns, *, now)`. `_prediction_error_staleness_decay` now shares `_linear_staleness_decay`, with unchanged behaviour.
- `services/orion-substrate-runtime/app/worker.py`: `_repair_appraisal_from_chat` delegates to the pure function.

## Files changed

- `orion/substrate/endogenous_curiosity.py`: per-turn staleness decay for repair pressure; shared decay helper.
- `services/orion-substrate-runtime/app/worker.py`: use the decayed appraisal; one `now` per tick.
- `orion/substrate/chat_loop/reducer.py`: keep a turn's first `observed_at` on re-reduction (review fix).
- `tests/test_chat_substrate_reducer.py`: re-reduction keeps `observed_at`, advances `last_updated_at`.
- `orion/substrate/tests/test_endogenous_curiosity.py`: fresh spike emits, old spike stops, live-shape regression, missing timestamp, recency beats size.
- `services/orion-substrate-runtime/tests/test_worker_endogenous_curiosity_tick.py`: worker-level regression with real `ChatSessionProjectionV1`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the repair appraisal's `dimensions["level"]` is now the age-decayed level. Its `summary` now includes `raw=` and `age=`. Nothing parses that summary. It only flows into `evidence_summary`.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no key change)
- skipped keys requiring operator action: none

## Design decisions

- **Why the 30-minute horizon, not a new window setting.** The brief asked to reuse curiosity's existing staleness mechanism. Live data supports the same value. In `repair_pressure_appraisal_log`, every HIGH appraisal (10-04 03:25, 10-05 03:35, 10-06 02:00) is a single turn followed by NONE/LOW on the next turn, so a repair episode is one turn. 30 minutes covers it with margin.
- **Dedupe/refractory.** No separate refractory was added. The decay itself is the refractory: one spike re-emits a weakening candidate for about 10 minutes (~8 ticks), then stops. That matches how prediction-error sources already behave. A separate cooldown would add new persisted state for no measured gain.
- **Missing `observed_at` → skipped, not fresh.** The prediction-error path treats an unknown age as unaged. For repair, "unknown age counts as current" is exactly the replay this patch fixes. So it follows `prediction_error_freshness.py`'s omit-when-unknown rule instead. Real `ChatTurnStateV1` rows always have `observed_at` because it is a required field.
- Designed constants were not touched: repair rest value 0.087 and fallback confidence 0.65 (`repair_pressure_v2.py`), and `min_repair_level` 0.6.

## Evidence: metric semantic layer CLI

`scripts/check_metric_lineage.py --metric <token>`:

- `repair_pressure` → two URNs, and neither is this path. One is `metric://field_channel/orion-field-digester/repair_pressure`: "How much conversational repair (corrections, re-explaining) is happening in chat", feeding `social_pressure`. The other is `metric://organ_signal/graph_cognition/repair_pressure#coherence`, produced by cortex-exec. The field-digester channel is a separate path and is already in `NODE_DECAY_CHANNELS`, so this bug does not affect it.
- `repair_pressure_level` → `UNREGISTERED` (no URN in any registry).
- `signal_strength`, `endogenous_curiosity`, `curiosity_candidate` → `UNREGISTERED`.
- `--drift` / `--generic-consumers`: no curiosity/repair consumer surfaced beyond a declared hub consumer for curiosity progress lines, which this patch does not touch.
- None of this changed the fix. The chat-projection repair level feeding curiosity has no lineage entry. That gap is recorded here and left as is: no URN was hand-authored.

## Tests run

```text
pytest orion/substrate/tests/test_endogenous_curiosity.py services/orion-substrate-runtime/tests/test_worker_endogenous_curiosity_tick.py
  -> 36 passed
pytest services/orion-substrate-runtime/tests (excluding DB-needing test_grammar_consumer_integration.py)
  -> 416 passed, 17 failed; the 17 failures are identical on clean origin/main (diffed)
pytest orion/substrate/tests tests/test_metric_lineage.py
  -> 3 failed (test_felt_state_self_definition_lane.py), identical on clean origin/main
Static gates (orion-static-gates.yml): check_metric_lineage --gate, check_definition_drift --gate,
  check_inner_state_registry, check_scripts_dir_no_stdlib_shadow, check_sentience_instruments --static-only,
  check_system_health_producers, check_control_surface_store_parity, check_async_routes_not_blocking,
  check_chat_route_poachers, check_service_hostname_refs, check_journal_dispatch_registry -> all exit 0
  (no metric re-lock needed)
```

Mutation checks:
- Removing the decay (`level = raw`) fails 5 tests: all four original pure-function tests and the worker regression.
- Removing the reducer `observed_at` preservation fails `test_reducer_keeps_first_observed_at_on_re_reduction`.

`pytest tests/test_chat_substrate_reducer.py` passes after the review fixes. The service suite's failure set is unchanged from origin/main.

## Evals run

```text
No eval harness covers endogenous curiosity. The read-only replay above (persisted candidate sets x live
chat projection through the new function) is the behavioural eval for this patch.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-substrate-runtime build -> Image orion-substrate-runtime-substrate-runtime Built
Not deployed (prod deploys from primary checkout on main after merge).
```

## Review findings fixed

Code review ran in a subagent against `git diff origin/main...HEAD`. It found nothing that must be fixed, one should-fix and five nits:

- Finding (should-fix): a repair turn's age was measured from when the chat reducer *last* processed it. If a late event arrived or the turn was reprocessed, the reducer re-stamped `observed_at` to now. An old spike would then look fresh for about 10 more minutes. That is bounded, but it is the same kind of replay.
  - Fix: `orion/substrate/chat_loop/reducer.py` now keeps the existing turn's `observed_at` when it updates a turn. `last_updated_at` still advances; `chat_prediction_error` reads that field, through `_latest_run`, so it is unaffected.
  - Evidence: `tests/test_chat_substrate_reducer.py::test_reducer_keeps_first_observed_at_on_re_reduction` passes. Removing the fix makes it fail (mutation-checked).
- Finding (nit): the tick computed `now` twice.
  - Fix: one `tick_now` is now passed to both `_repair_appraisal_from_chat` and `endogenous_curiosity_candidates`.
- Finding (nits): the age math was duplicated; a naive `now` raised `TypeError`, and a bare `except` swallowed it; `SimpleNamespace` was imported inside the function.
  - Fix: added a shared `_age_seconds` helper that normalises both datetimes to UTC. The `except` is narrowed to `(TypeError, ValueError)`. The import moved to module scope.
  - Evidence: new `test_future_dated_turn_is_treated_as_age_zero_and_naive_is_utc` passes.
- Finding (nit, confirmed safe): the refactor does not change prediction-error decay. The body is line-for-line the same, and the existing prediction-error tests pass.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `observed_at` is the reducer's clock at a turn's *first* reduction, not when the user spoke. Live, that gap is about 2 minutes. A turn reduced for the first time long after it happened (for example, a backfill) would look fresh for one 30-minute window.
- Mitigation: re-reduction no longer re-stamps it (review fix). Possible follow-up: stamp `observed_at` from the grammar events' own time.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2577

🤖 Generated with [Claude Code](https://claude.com/claude-code)
