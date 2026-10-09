# Memory episodes, Stage 1 PR 1: boundary fixes and the episode close event

## Correction (post-merge review, 2026-10-02, made in #2484)

This report said live behavior is unchanged apart from the shadow. That was wrong. **Fix 2 changes live inputs.**

- **What changed:** each turn is now judged once, against the previous turn. Before, the scores the live path used came from the second, self-comparing judgment.
- **Where it shows** (`orion/memory/consolidation_gate.py` reads them):
  - the novelty and `memory_significance_score` stored on every window turn, which the crystallization gate compares against `min_novelty` and `min_significance`;
  - the boundary, significance and novelty values that `intake_consolidation_window.py` copies into a crystallization's provenance;
  - the scores in `chat_history_log.spark_meta`;
  - the number of `orion:signals:memory_consolidation` turn-change signals (the duplicate pass also emitted them).
- **Measured** on the 107 window-closing turns of the last 30 days, where both judgments survive: mean novelty 0.930 (kept) vs 0.651 (old saved), mean significance 0.472 vs 0.554. For non-closing turns the first judgment was overwritten and cannot be recovered, so their shift is UNVERIFIED.
- **Expected effect:** more turns clear the novelty floor and slightly fewer clear the significance floor, so the legacy intake's propose/skip mix moves. Live window **closing** is unchanged: it always used the first judgment.
- **Outreach stamp:** removed in #2484. `client_meta.conversation_phase` had no reader (outreach never reaches consolidation, and consolidation reads `spark_meta`).
- **GIN index:** moved to a separate `CREATE INDEX CONCURRENTLY` migration, plus rollback scripts (#2484).

## Summary

- **Each chat turn now records the conversation clock (Fix 1).** When a turn is saved, it now carries how long it had been since Juniper last spoke, and which bucket that falls in (same breath, short pause, resumed thread, long gap, next day, stale). Before this, 0 of 3,586 turns carried it.
- **Each turn is judged once, not twice (Fix 2).** The database writer announces every turn twice. The memory service judged the second copy too, and that second judgment compared the turn with itself. Its low score overwrote the saved score, while the first, high score had already closed the window. The service now skips a turn it has already placed in a window. The degraded-classify retry also updates the window, so the two scores stay equal.
- **The new boundary rule (Rule 3) runs in shadow beside the live one.** Every turn is also placed into a shadow episode table under Rule 3. Both decisions are recorded per turn, and live window closing does not change.
- **A closed shadow episode announces itself.** It publishes `memory.episode.closed.v1` on `orion:memory:episode:closed`, carrying the close lag (how long the episode waited for the next turn to close it).
- **Live windows now record why they closed** (`close_reason`, `boundary_score_at_close`).
- **A 30-day replay eval** compares the old rule with Rule 3 on real turns.

## Outcome moved

- Before this PR, the score that closed a window and the score saved for the same turn differed on **86 of 87** closing turns in 30 days. The mean closing score was 0.93; the mean saved score for those same turns was 0.30. After the fix, a test against a real Postgres shows them equal for every turn.
- Rule 3 episodes are inspectable on live traffic before any GPU time is spent.

## Two findings Juniper should see before Rule 3 goes live

**1. With honest scores, Rule 3 splits the Austin morning into three episodes, not one.** The spec's "one episode, 06:26-09:56" replay used the saved scores. Fix 2 shows those are the self-comparison artifact. The judge's real scores for the two "resumed thread" turns that morning were 0.970 (08:45) and 0.999 (09:44). Both are above the 0.92 threshold, so Rule 3 splits at each of them. Across 30 days, the judge's real score clears 0.92 on **19 of 29** resumed-thread turns, against 2 of 29 with the artifact scores.
- The root cause is the classify prompt: it asks for `BOUNDARY: YES or NO` but never says what a boundary is (`orion/memory/turn_change_classify.py:139`). The judge says YES whenever the turn differs from the one before it.
- I did not change the threshold (it is a knob, not a fix), and I did not change the prompt (that would change live turn classification). Recommended next step: give the prompt a one-line definition of a conversation boundary, then re-run this replay. That is a decision for Juniper.
- Times are UTC. The Austin morning is 2026-09-28 06:26-09:54 UTC, which is 00:26-03:54 MDT. It has 11 turns that reach the memory service, not 12: the 09:56 "Run your dream cycle." row has an empty response and never reaches it.

**2. If the live window rule read the new clock stamp, live windows would change.** Over 30 days the legacy rule would make about 33-35 windows instead of 98. The classify prompt would also start seeing a real phase instead of `phase=unknown`. You said live window closing must not change. So `MEMORY_LEGACY_BOUNDARY_USE_PHASE=false` (the default) hides the stamp from the legacy rule and from the classify prompt. Rule 3 in shadow always reads it. Flipping it on is a separate decision.

Smaller findings:
- **Orion's unprompted messages never reach the memory service.** They have an empty prompt, and sql-writer only forwards rows that have both a prompt and a response. So the spec's "first boundary is the 15:19 outreach long_gap" cannot happen. The Austin episode closes at the next real turn (09-29 04:01 UTC), with an 18.1 h lag. Outreach rows are stamped anyway (on `client_meta`, since that message envelope has no `spark_meta`).
- **The wall clock has a gap between 12 h and 48 h on the same local day.** It reads `unknown` there (`classify_conversation_phase`; unchanged behavior). Rule 3 then falls back to the 90-minute gap rule. 22 of 56 Rule 3 closes go that way.
- **The situation cache would have produced wrong stamps.** It keeps a brief for 300 s, and on a cache hit the brief's phase belongs to the earlier turn. Juniper's second message 4 minutes after a `next_day` first message would have been stamped `next_day`. That reads as a new conversation under Rule 3. The stamp is now recomputed on a cache hit (`_phase_stamp_from_cached_brief`), with a regression test.

## Current architecture

- **Service:** orion-memory-consolidation, consuming `orion:memory:turn:persisted` from orion-sql-writer.
- **Per turn:** classify with an LLM judge (`app/classify.py`), publish a `spark_meta` patch back to `chat_history_log`, append the turn to the open window, and close the window under `should_close_window` / time-gap fallback (`app/boundary.py`, `app/window_fetch.py`).
- **Wall clock:** `orion/situational/context.py` `_build_conversation_phase`, built by the Hub's situation brief per unified turn. It went into the prompt only, never onto the persisted turn.
- **Defect 2 mechanism:** confirmed live on 2026-09-28. Window `ca973fcb` closed on turn `8541979d` at score 0.981 (classified 06:31:27). The same turn was classified again at 06:31:28 against window `d1c87869`, which by then held only that turn, and scored 0.006. `chat_history_log` holds 0.006 with `memory_classify_ts` 06:31:28.

## Architecture touched

- **Hub:** situation build → `spark_meta.conversation_phase` on unified turns. Read-only stamp for legacy-lane turns and outreach.
- **orion-memory-consolidation:** dedup, close audit, Rule 3 shadow tracker, close event, legacy-view flag.
- **Contracts:** new schema plus channel.
- **Postgres:** additive migration.

## Files changed

- `orion/situational/context.py`: `classify_conversation_phase` (pure; `_build_conversation_phase` now uses it), `conversation_phase_stamp`, `read_conversation_phase_stamp` (read-only), and `build_situation_for_ctx(..., phase_stamp_out=)` with a correct stamp on cache hits.
- `orion/hub/turn_orchestrator.py`: carries the stamp from the situation build into `spark_meta.conversation_phase` on the persisted unified turn.
- `services/orion-hub/scripts/chat_history.py`: `publish_chat_turn` adds a read-only stamp when the turn has none (legacy WS/HTTP lanes; live today only for workflow commands).
- `services/orion-hub/scripts/endogenous_outreach.py`: stamped outreach `client_meta.conversation_phase` (REMOVED in #2484: no reader).
- `services/orion-memory-consolidation/app/worker.py`: Fix 2 dedup, the shadow observe/publish (fail-open), the close audit, and the legacy view.
- `services/orion-memory-consolidation/app/boundary.py`: `rule3_boundary`, `legacy_close_reason` (names the existing branches; behavior unchanged), `legacy_view`.
- `services/orion-memory-consolidation/app/window_fetch.py`: `legacy_close_decision` returns the reason.
- `services/orion-memory-consolidation/app/window_state.py`: `find_windowed_turn`, `update_turn_scores`, `record_close_audit`; stores the stamp on the window entry.
- `services/orion-memory-consolidation/app/episode_shadow.py` (new): the Rule 3 shadow store and the close event builder.
- `services/orion-memory-consolidation/app/retry_degraded_classifies.py`: updates the window score after a retry.
- `services/orion-memory-consolidation/app/{settings.py,main.py}`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-sql-db/manual_migration_memory_episode_v1.sql` (new).
- `orion/schemas/memory_episode.py` (new), `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`.
- Tests: `orion/situational/tests/test_conversation_phase_stamp.py`, `services/orion-hub/tests/test_conversation_phase_persist.py`, `services/orion-memory-consolidation/tests/test_episode_boundary_{rule3,dedup_unit,fix2_pg}.py`. Three Hub test fakes accept the new keyword.
- Eval: `services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py`, its test, and fixture `episode_boundary_replay_30d.json`. The fixture holds ids, times, scores and a command flag; no text.

## Schema / bus / API changes

- **Added:** `MemoryEpisodeClosedV1` (`memory.episode.closed.v1`), registered in both `_REGISTRY` and `SCHEMA_REGISTRY`; verified with `resolve("MemoryEpisodeClosedV1")`. Channel `orion:memory:episode:closed` (producer orion-memory-consolidation, consumer orion-durable-runs).
- **Payload:** `episode_id, source_platform, started_at, ended_at, closed_at, turn_ids, juniper_turn_count, command_turn_count, close_reason, phase_at_close, boundary_score_at_close, close_lag_sec, closing_turn_id, episode_status (closed|skipped), skip_reason, boundary_rule`.
- **Chat turn contract:** `spark_meta.conversation_phase = {phase_change, delta_user_seconds, crossed_day, source}`. The extra `source` field records how the stamp was taken: `situation_build`, `situation_cache` or `session_state_read`.
- **Behavior changed:** `chat_history_log.spark_meta.conversation_boundary_score` now holds the score that actually decided the window, not the self-comparison.
- **Compatibility:**
  - The consumer of the new channel (orion-durable-runs) ships in the stacked PR 2. Until then, events go to no subscriber; `memory_episode_shadow` is the durable record, and unpublished closes are re-sent on later turns.
  - `MemoryTurnPersistedV1` is unchanged.

## Env/config changes

- **Added keys** (orion-memory-consolidation): `MEMORY_EPISODE_SHADOW_ENABLED=true` (kill switch), `CHANNEL_MEMORY_EPISODE_CLOSED=orion:memory:episode:closed`, `MEMORY_LEGACY_BOUNDARY_USE_PHASE=false`.
- `.env_example` updated: yes. `docker-compose.yml` updated: yes.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py --all-keys orion-memory-consolidation`: yes, written to the primary checkout. The one pre-existing divergence (`CONCEPT_RELATION_RESOLUTION_ENABLED` local=true) was not touched.
- **Deviation:** the spec's `MEMORY_EPISODE_BOUNDARY_RULE=legacy|v2` was not added. Rule 3 always runs in shadow, and there is no live switch until cutover. `MEMORY_EPISODE_SHADOW_ENABLED` is the kill switch.

## Tests run

```text
pytest services/orion-memory-consolidation/tests services/orion-memory-consolidation/evals
  (with ORION_MEMORY_EPISODE_TEST_DATABASE_URL -> disposable postgres:16-alpine)  -> 268 passed
  (without it, as CI runs)                                                       -> 263 passed, 5 skipped
  baseline on main (tests only): 222 passed
pytest orion/situational/tests                                                   -> 94 passed
pytest services/orion-hub (turn_orchestrator / chat_history / situation / unified / outreach tests) -> 560 passed, 1 failed
  the 1 failure (test_turn_orchestrator_utterance_origin::..._mind_appraisal_text...) also fails on main
Mutation check: disabling the Fix 2 dedup makes test_window_score_equals_chat_log_score_for_every_turn fail.
Repo channel/catalog/registry tests: 144 passed, 5 failed. The same 5 fail on main.
Static gates (orion-static-gates.yml steps, run locally): all pass after re-lock, including
  check_metric_lineage.py --gate (PASS) and check_definition_drift.py --gate (PASS).
check_env_template_parity.py: PASS. git diff --check: clean.
```

## Evals run

```text
python services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py --refresh   (read-only)
105 direct turns, 30 days, captured 2026-10-02 04:45 UTC

| rule                                      | episodes | /active day | turns/ep mean, median | close lag p50 / p95 |
| legacy, as it ran                         | 98       | 4.08        | 2.05                  | n/a                 |
| legacy + Fix 1 phase (if flag flipped)    | 33-35    | 1.38-1.46   | 4.15-3.97, 3          | n/a                 |
| Rule 3, real judge scores (what will run) | 57       | 2.38        | 1.84, 1               | 6.2 h / 44.8 h      |
| Rule 3, artifact scores (spec's method)   | 40       | 1.67        | 2.62, 2               | 13.5 h / 46.0 h     |

Austin morning: live windows 10.
  Rule 3 real scores: 3 episodes (06:26-06:59, 08:45-09:06, 09:44-09:54 UTC).
  Rule 3 artifact scores: 1 episode, 06:26-09:54 UTC, closed by 09-29 04:01, lag 18.1 h.
Over-split rate (consecutive episodes sharing an event: referent): needs PR 2's memories. NOT MEASURED.
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-memory-consolidation/docker-compose.yml config
  -> renders MEMORY_EPISODE_SHADOW_ENABLED, CHANNEL_MEMORY_EPISODE_CLOSED, MEMORY_LEGACY_BOUNDARY_USE_PHASE
Not built or deployed (instructed). Migration applied cleanly to a disposable Postgres in tests.
Live table size for the new GIN index: memory_consolidation_windows 3,681 rows, 8 MB.
```

## Review findings fixed

The orchestrator runs the review.

## Restart required

Deploy order:
1. Apply `services/orion-sql-db/manual_migration_memory_episode_v1.sql`. The service is fail-open without it, but the shadow and the audit stay empty.
2. Deploy orion-memory-consolidation.
3. Deploy orion-hub.

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_memory_episode_v1.sql
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity:** medium. **Concern:** Rule 3 with the real judge over-splits resumed threads (finding 1). **Mitigation:** shadow only. Fix the BOUNDARY prompt and re-run the replay before any cutover.
- **Severity:** low. **Concern:** the Fix 2 dedup looks back 72 h. A duplicate publish arriving later than that would be classified again. **Mitigation:** the duplicates arrive about 1-2 s apart live.
- **Severity:** low. **Concern:** the legacy-lane read-only stamp measures from the last unified-lane turn, because workflow commands do not advance the clock. **Mitigation:** commands are excluded from episodes as `command_only` anyway.
- **UNVERIFIED:**
  - live liveness of the Hub stamp;
  - the shadow table filling on real traffic;
  - the close event on the real bus. None of these are deployed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2479

🤖 Generated with [Claude Code](https://claude.com/claude-code)
