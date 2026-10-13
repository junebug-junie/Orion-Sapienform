## Summary

- The telemetry-anomaly metacog trigger now fires **once per anomaly episode** instead of once per flagged 60-second scoring window. It re-fires inside an episode only when the anomaly climbs into a higher severity band (nominal -> degraded -> critical, the same bands `orion/metacog/evidence_map.py` uses).
- Windows whose lead channel is a turn-execution channel (glossary category `task_execution`) are left to the chat_turn trigger while a harness turn is active. Once 15 flagged windows land within 30 minutes (counted across episode breaks), it fires anyway, so a real outage that hangs turns still gets through.
- The open episode is saved in Redis, so a restart in the middle of an anomaly does not fire it again.
- orion-sql-writer now stores **every** anomaly score, flagged or not, in a new `field_channel_anomaly_score` table. That gives us the calm side of the distribution needed to recalibrate the 3.0 multiplier later. The multiplier is unchanged.
- Replay over 15 days of real stored fires: **2,485 fires -> 140**. The 10-08 GPU outage still fires, at 14:54Z, about 14 minutes after it started.

## Outcome moved

Each telemetry_anomaly fire costs one LLM metacog entry. Replaying the shipped behaviour against the new gate on real data from 2026-09-28 to 2026-10-13:

| day | before | after | episodes | turn-explained windows |
|---|---|---|---|---|
| 09-29 | 322 | 22 | 98 | 253 |
| 09-30 | 393 | 22 | 122 | 222 |
| 10-01 | 93 | 10 | 49 | 80 |
| 10-02 | 77 | 16 | 41 | 57 |
| 10-03 | 167 | 3 | 37 | 65 |
| 10-04 | 177 | 7 | 42 | 90 |
| 10-05 | 73 | 3 | 38 | 68 |
| 10-06 | 187 | 11 | 33 | 63 |
| 10-07 | 104 | 4 | 54 | 91 |
| 10-08 | 396 | 11 | 49 | 390 |
| 10-09 | 118 | 9 | 54 | 108 |
| 10-10 | 164 | 10 | 64 | 121 |
| 10-11 | 74 | 4 | 37 | 64 |
| 10-12 | 66 | 3 | 34 | 60 |
| 10-13 (partial) | 72 | 4 | 34 | 64 |
| **total** | **2,485** | **140** | | |

Of the reduction, episode grouping alone takes 2,485 to 885. Turn suppression takes 885 to 140.

Caveat: the old shared 30s cooldown occasionally dropped a flagged window when another trigger kind held the lane. Those dropped windows read as calm gaps in this replay and can split an episode, so the "after" numbers are an upper bound.

**10-08 outage.** The outage ran from about 14:40Z, led by `failure_pressure`. It hung chat turns for 20-70 minutes, so turn brackets covered every flagged window. With pure suppression it **never fires**, which is what the eval control test pins. The sustained override (15 flagged windows in 30 min) fires it at 14:54Z and escalates to critical at 14:55Z. The override costs +19 fires across the 15 days (121 -> 140). Non-chat episodes still fire: the 10-03 availability outage fires at 21:16Z (degraded) and escalates to critical at 21:17Z. The 10-10 disk_capacity_pressure episodes fire at 03:12Z, 05:56Z and 18:51Z.

**Dream novelty pressure.** orion-dream builds its novelty key from the trigger reason with the numbers stripped, which leaves direction + lead channel. It only counts keys it has **not** seen in the prior 48h (degraded = 0.6, critical = 1.0). The `reason` format is unchanged here. Distinct telemetry keys per day drop from 5-9 to 1-5. The keys lost are chronic turn-coupled ones (`elevated:harness_step_load`, `elevated:egress_confidence_deficit`, ...) that already sit in the 48h lookback, so they were adding almost no pressure. Day-granular estimate of new-key pressure from telemetry, 10-01..10-13: **10.8 before vs 10.0 after**. Telemetry is about 25-30% of daily distinct metacog keys, but only first appearances move pressure. **Dreams should not slow materially.** This is a replay estimate (UNVERIFIED live).

## Current architecture

- `orion-field-digester/app/anomaly_scorer.py:score_latest()` scores the latest 30-row window of 37 channels against `mood-arc-encoder:v4` every 60s (`FIELD_CHANNEL_ANOMALY_CHECK_INTERVAL_SEC`). It publishes `FieldChannelAnomalyScoreV1` on `orion:field_channel:anomaly_score`.
- `orion-equilibrium-service` built a trigger for every score with `recon_loss > recon_error_p95 * 3.0` and published it on the shared 30s cooldown lane. That lane is shorter than the 60s interval, so it never held a sustained anomaly back.
- 88% of fires since 10-01 were led by `task_execution` channels. These are produced from harness/execution run deltas in `ingest/state_deltas.py`, the same turns `chat_turn_metacog_gate.py` already reads.
- Scores were not persisted. Only flagged windows survived, as `metacog_trigger` rows.

## Architecture touched

- **New pure module** `orion/metacog/anomaly_episode.py`. It holds the episode state machine (`step`, `mark_published`, persisted-state helpers) and `TurnActivityTracker`. It reuses `evidence_map.banded` and its telemetry bands.
- **orion-equilibrium-service.**
  - `telemetry_anomaly_metacog_gate.evaluate_telemetry_anomaly_episode` adapts a score to the state machine. It looks up the lead channel's category from the glossary, so the image now copies `config/field`.
  - `service.py` loads and persists episode state, tracks turn activity, subscribes to grammar and harness-run channels when suppression is on, and publishes with `bypass_cooldown=True`.
  - The fire is recorded only after a real publish, so a failed publish is retried on the next window.
- **orion-sql-writer.** New `FieldChannelAnomalyScoreSQL` model (insert-only, created by `create_all`). New route `field_channel_anomaly.score.v1`. The channel is force-subscribed in code.
- **Semantic layer.**
  - New `field_channel_anomaly_score.v1` entry in `orion/inner_state_registry.py`, with `recon_loss` / `mean_signed_deviation` semantics.
  - `orion/bus/channels.yaml` now names `orion-sql-writer` as a consumer.
  - Lock re-locked: 6 added URNs plus that routing change. These are real definition changes.

## Files changed

- `orion/metacog/anomaly_episode.py`: new pure episode and turn state machine.
- `services/orion-equilibrium-service/app/telemetry_anomaly_metacog_gate.py`: episode adapter, glossary lead-category lookup.
- `services/orion-equilibrium-service/app/service.py`: wiring, state load/persist, turn tracking, subscriptions.
- `services/orion-equilibrium-service/app/settings.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`, `README.md`: 6 new keys (all ON), glossary copied into the image, docs.
- `services/orion-equilibrium-service/tests/test_telemetry_anomaly_episode.py`: 20 tests covering the state machine, adapter and service wiring.
- `services/orion-equilibrium-service/evals/test_telemetry_anomaly_episode_replay_eval.py` and `evals/fixtures/telemetry_anomaly_replay_2026-09-28_2026-10-13.jsonl`: replay eval on real data.
- `scripts/analysis/replay_telemetry_anomaly_episodes.py`: read-only export SQL plus the replay.
- `services/orion-sql-writer/app/models/field_channel_anomaly_score.py`, `models/__init__.py`, `app/settings.py`, `app/worker.py`, `.env_example`, `README.md`, `tests/test_field_channel_anomaly_score_sql.py`: score persistence.
- `orion/inner_state_registry.py`, `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`: semantic layer.
- `.github/workflows/telemetry-anomaly-episode-tests.yml`: CI gate for all of the above.

## Schema / bus / API changes

- **Added:**
  - A sql-writer subscription to `orion:field_channel:anomaly_score`, and the table `field_channel_anomaly_score`.
  - Additive keys in the `telemetry_anomaly` `MetacogTriggerV1.upstream` (a free-form dict): `episode_id`, `episode_fire_reason`, `episode_severity`, `episode_windows`, `episode_turn_explained_windows`, `episode_duration_sec`, `episode_peak_ratio`, `lead_channel_category`, `turn_active`.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:**
  - `telemetry_anomaly` fires per episode, not per window.
  - It no longer takes the shared cooldown lane, so it no longer starves (or is starved by) baseline/manual/relational fires.
  - `reason` format is unchanged, so the dream novelty key and evidence_map are unchanged.
- **Compatibility notes:** `FieldChannelAnomalyScoreV1` is unchanged (it is `extra="forbid"`, and substrate-runtime validates it strictly). This is why classification happens in equilibrium, not in the producer: no substrate-runtime redeploy is needed.

## Env/config changes

- **Added keys** (orion-equilibrium-service):
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_EPISODE_ENABLE=true`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_EPISODE_CALM_GAP_SEC=150`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_EPISODE_SUSTAINED_WINDOWS=15`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_EPISODE_SUSTAINED_HORIZON_SEC=1800`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_EPISODE_STATE_KEY=equilibrium:telemetry_anomaly_episode:v1`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TURN_SUPPRESS_ENABLE=true`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TURN_CARRY_SEC=150`
  - `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TURN_OPEN_MAX_SEC=900`
- **orion-sql-writer:** the `SQL_WRITER_SUBSCRIBE_CHANNELS` / `SQL_WRITER_ROUTE_MAP_JSON` examples gained the anomaly-score channel and route.
- **Removed / renamed keys:** none.
- **`.env_example` updated:** yes, both services.
- **Local `.env` synced** (`python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service orion-sql-writer`): all 8 equilibrium keys are in the primary checkout's `services/orion-equilibrium-service/.env`. `SUSTAINED_WINDOWS` was first synced as 10; after its default moved to 15 during review, the sync reported it as diverged, so that single line was updated by hand.
- **Skipped keys requiring operator action:** none. The sql-writer local `.env` value for `SQL_WRITER_SUBSCRIBE_CHANNELS` was already diverged from the example, so it was not overwritten. No action is needed: the channel is force-added in code (`effective_subscribe_channels`), and the route map merges code defaults.

## Metric quality gate (score persistence + episode gate)

1. **Provenance.** `recon_loss` comes from `anomaly_scorer.py:score_latest` (`score_windows(...)[-1]`), published by `anomaly_bus_publish.publish_anomaly_score`. The turn signal comes from `orion-harness-governor` `GrammarEventV1.provenance.source_service` (live: a turn's events run from turn start to turn end) and `HarnessRunV1.fcc_elapsed_sec`.
2. **Independence.** The new gate deliberately removes the dependent part. Task_execution channels come from the same execution_run deltas that chat_turn and `prediction_error_execution` read, so windows they lead during a turn are dropped as redundant. Nothing new enters a model.
3. **Theory anchor.**
   - Episode = a run of flagged windows separated by less than 2 calm windows. The live gap histogram shows 60-89s gaps (consecutive) and 90-149s gaps (one skipped window, flicker); separation starts at 150s.
   - Turn carry (150s) = the digester's 90s hold (`FIELD_DECAY_STALENESS_THRESHOLD_SEC`) plus about 55s of 0.92-per-2s decay to 10%.
   - Sustained override (15 flagged windows within 30 min) is well past the p75 turn length of about 570s.
  - Unclosed-turn expiry (900s) comes from the live longest silence between harness grammar events inside a turn (p99 840s, max 855s, 182 turns over 3 days).
4. **Live-data check.**
   - Before this patch, unflagged scores did not exist anywhere, which is why they are now persisted. Calm rest is the v4 manifest's held-out p50 of 0.0016, never 0.
   - Episode and suppression parameters were read from 15 days of real `metacog_trigger` and `harness_turn_trace` rows.
5. **Existing mechanism.** `trend_reducer.py` is an EWMA "sustained run" detector over appraisals, not episode grouping on an already-thresholded stream, so it was not reused. The checkpoint-in-Redis pattern and `bypass_cooldown` (documented for episode-shaped sources) were reused. `evidence_map.banded` supplies the severity bands.
6. **Reversibility.** `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_EPISODE_ENABLE=false` restores the exact old path. `..._TURN_SUPPRESS_ENABLE=false` keeps episodes without suppression. The table is additive and insert-only.

## Tests run

```text
services/orion-equilibrium-service: pytest tests evals -> 223 passed (after review fixes)
services/orion-sql-writer: tests/test_field_channel_anomaly_score_sql.py -> 5 passed
services/orion-sql-writer: full suite -> 14 failed / 741 passed. The 14 are pre-existing failures that are not caused by this branch: an extracted origin/main tree fails the same tests, plus more, under the same venv.
orion/metacog/tests + tests/test_metric_prompt_semantics.py -> 287 passed
scripts/check_metric_lineage.py --gate -> PASS; --prompt-semantics -> PASS
scripts/check_definition_drift.py --gate -> FAIL before re-lock (6 added + routing changes: sql-writer bus consumer, substrate-runtime registry consumer; all real) -> re-locked with --update -> PASS
scripts/check_inner_state_registry.py -> OK (23 entries)
scripts/check_env_template_parity.py -> PASS
git diff --check -> clean
```

## Evals run

```text
services/orion-equilibrium-service/evals/test_telemetry_anomaly_episode_replay_eval.py -> 6 passed
  2485 -> 140 fires; 10-08 outage first fire < 14:55Z; no-override control: the outage never fires;
  10-03 availability outage fires; suppression is what removes most episodes (885 vs 140);
  telemetry new-key dream pressure within 25% (10.8 vs 10.0).
```

## Docker/build/smoke checks

```text
docker build -f services/orion-equilibrium-service/Dockerfile (throwaway tag, removed after)
  in-container: lead_channel_category(failure_pressure) -> task_execution, (availability) -> sensor_trust_liveness;
  episode + turn-suppress flags True; app.service imports.
Not deployed (per instruction). Live path UNVERIFIED until deploy.
```

## Review findings fixed

The review subagent covered `git diff origin/main...HEAD`. It found no blockers, four should-fix issues and four nits.

- **Finding (should-fix):** a flickering turn-coupled outage never fired. The sustained count reset on every episode close, so a pattern of 9 flagged / 3 calm windows, repeated, gave 0 fires over about 4h.
  - Fix: the override now counts flagged windows in a trailing 30-minute horizon (`recent_flagged`, persisted). That count survives episode closes.
  - Evidence: `test_flickering_turn_coupled_outage_still_fires_via_trailing_horizon`, plus `test_ordinary_turn_bursts_stay_suppressed` as the control.
- **Finding (should-fix):** if the digester clock stepped backwards, every anomaly was muted until the clock caught up, and the muting survived restarts.
  - Fix: a step back larger than the calm gap now closes the episode. A small step back is still treated as a duplicate.
  - Evidence: `test_large_backward_clock_step_does_not_mute_anomalies`.
- **Finding (should-fix):** an unclosed turn counted as active for a full hour after its last event, and that limit was hard-coded.
  - Fix: 900s, taken from the live longest in-turn gap between harness events. It is now a setting, `.env_example` key and compose key.
  - The replay now streams synthetic turn activity every 300s, so it mirrors the live tracker. This changed the numbers above, and is why pure suppression now never fires 10-08.
  - Evidence: `test_unclosed_turn_stops_counting_after_open_max`.
- **Finding (should-fix):** the replay export would replay its own output after deploy, and it silently overcounts.
  - Fix: the export excludes rows that carry `episode_id`. A new `--source scores` reads `field_channel_anomaly_score` (calm windows included, deduped per window). The cooldown caveat is now in the docstring and in this report.
- **Finding (nit):** the replay and the live gate used different time sources for turns.
  - Fix: the live tracker now uses the grammar event's `emitted_at` when it is within 0-600s of receipt, and receipt time otherwise.
- **Finding (nit):** the Hub downstream-triggers panel would silently change meaning (one row per episode instead of per window).
  - Fix: its docstring in `services/orion-hub/scripts/mood_arc_status_routes.py` now says so.
- **Finding (nit):** a stalled digester can store the same window twice.
  - Fix: deduped in analysis (`DISTINCT ON (encoder_id, window_end)` in the scores export). No unique index was added: insert-only with the envelope-stamped primary key is the sql-writer convention. Left as a note.
- **Finding (nit):** the registry did not declare the substrate-runtime brain-frame consumer.
  - Fix: it is now declared, and the lock was re-locked.

## Restart required

Consumer-first: sql-writer must be up before the table has a writer, then equilibrium. The field-digester itself is unchanged.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-sql-writer orion-equilibrium-service; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

Post-deploy evidence to check:
- `SELECT count(*), sum((anomalous)::int) FROM field_channel_anomaly_score WHERE created_at > now() - interval '10 minutes';` should be about 10 rows.
- `docker logs orion-athena-equilibrium | grep telemetry_anomaly_episode` should show `decision=turn_explained|within_episode|episode_open`.
- `metacog_trigger` telemetry_anomaly rows per hour should drop.

## Risks / concerns

- **Severity: medium.** If the old cooldown dropped windows, the replay overcounts, so "after" is an upper bound. The exact figure needs `--source scores` once the table has a few days of data.
- **Severity: medium.** The turn tracker is in memory. After an equilibrium restart, a turn already in flight is unknown until its next harness grammar event, so one turn-coupled window could fire right after boot. That is bounded to one fire, because the episode state is restored.
- **Severity: medium.** Episode semantics drop mid-episode lead-channel changes at the same band. Example: an episode opened by `disk_capacity_pressure` stays one fire when `availability` takes over at the same severity. Intentional (one LLM entry per episode), but it is a loss of detail.
- **Severity: low.** Dream novelty effect is a day-granular replay estimate, not measured live.
- **Severity: low.** `field_channel_anomaly_score` has no retention yet (about 1,440 rows/day).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2644

🤖 Generated with [Claude Code](https://claude.com/claude-code)
