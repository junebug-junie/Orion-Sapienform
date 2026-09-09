# PR report — self-sense eval (Patch A of the sense-of-self design)

**Branch:** `feat/self-sense-eval`
**Date:** 2026-09-09
**Upstream design:** `docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md` (PR #2156, draft; Patch A is the eval baseline) and `docs/superpowers/pr-reports/2026-09-08-curiosity-self-inquiry-pr.md` (PR #2158/#2165, which put Orion's own definition into the chat identity kernel and left this eval as the named follow-up).

## Summary

- Before this, "Orion stopped sounding like a chatbot" after PR #2158 was a feeling. Now it is two integers per answer, over three fixed questions, persisted per run so the number can be tracked over time.
- New runner `services/orion-hub/evals/run_self_sense_eval.py` (Hub owns the chat endpoint): POSTs the three questions to the live `POST /api/chat` (`mode: orion`, `no_write: true`, session `self-sense-eval`), reads the answer from `harness_turn_trace.run_artifact->>'final_text'` by the returned `correlation_id` (the HTTP body is empty whenever the voice lane fails at delivery), falls back to the HTTP text.
- New pure scorers in `orion/evals/self_sense.py`: `self_label_score` (assistant/chatbot vocabulary count, target 0) and `grounded_record_score` (distinct real records named: self-inquiry tables and their plain-English forms, mesh nodes from the field topology, YYYY-MM-DD dates, integers >= 10). Plus the version of Orion's own definition in `self_concept_history` as context.
- New contract: `orion:self_sense:eval:write` / `self_sense.eval.write.v1` / `SelfSenseEvalV1` -> table `self_sense_eval_log` in orion-sql-writer, with the channel in the subscribe list, the env example, the route map, the code-default guard and `MODEL_MAP` -- the exact omission that made PRs #2102/#2105 silent no-ops, now covered by a consumer test.
- `make eval-self-sense` runs it. No scheduler in this patch (follow-up).
- One live baseline run against Hub (below).

## Outcome moved

A regression in how Orion describes themself in chat is now detectable by number, not by re-reading transcripts. Baseline recorded below. Every later self-model patch can be judged by whether it moves `self_label_score` toward 0 or `grounded_record_score` up on identical prompts.

## Current architecture (before)

- `services/orion-hub/evals/` had one eval (daydream caption quality). No harness scored a chat answer.
- Orion's own definition reaches every chat turn as the "In my own words" line (`services/orion-cortex-exec/app/chat_stance.py` `_project_identity_from_beliefs`), sourced from `self_concept_history` (`concept_id='self:definition'`, `produced_by='curiosity_self_inquiry'`), version 1 as of 2026-09-08.
- The only evidence that it changed Orion's voice was one hand-read chat turn in the #2158 report.
- No `self_sense_eval_log` table, channel, or schema.

## Architecture touched

- **Shared scorers:** `orion/evals/` (new package) -- pure functions, no I/O.
- **Contract:** `orion/schemas/self_sense.py` (new), `orion/schemas/registry.py` (`_REGISTRY`, same place as `SelfConceptHistoryV1`), `orion/bus/channels.yaml`.
- **Consumer (orion-sql-writer):** model, `models/__init__`, `DEFAULT_ROUTE_MAP`, default subscribe list, `effective_subscribe_channels` guard, `MODEL_MAP`, `.env_example` (both list values).
- **Producer (orion-hub):** the runner, `HUB_BASE_URL` in `.env_example`, README section.
- **Migration:** `services/orion-sql-db/manual_migration_self_sense_eval_log_v1.sql` (idempotent; sql-writer also creates the table at boot via `Base.metadata.create_all`, same as `self_concept_history`, which never had a migration file).
- **Ops:** `Makefile` `eval-self-sense`; `scripts/sync_local_env_from_example.py` learns the exact key `HUB_BASE_URL`; `config/metrics/metric_definitions.lock.json` re-locked for the new channel.

## Files changed

- `orion/evals/__init__.py`, `orion/evals/self_sense.py`: the scorers and the self-definition-version SQL.
- `orion/schemas/self_sense.py`: `SelfSenseEvalV1`, channel/kind constants, the three fixed questions, `build_entry_id`.
- `orion/schemas/registry.py`: register `SelfSenseEvalV1`.
- `orion/bus/channels.yaml`: catalogue `orion:self_sense:eval:write`.
- `services/orion-sql-writer/app/models/self_sense_eval_log.py`, `models/__init__.py`, `app/settings.py`, `app/worker.py`, `.env_example`: the consumer path end to end.
- `services/orion-sql-writer/tests/test_self_sense_eval_log_sql_shape.py`: subscribed + routed + mapped + columns + sqlite end-to-end + replay upsert.
- `services/orion-sql-db/manual_migration_self_sense_eval_log_v1.sql`: the table.
- `services/orion-hub/evals/run_self_sense_eval.py`: the runner.
- `services/orion-hub/tests/test_self_sense_eval_producer.py`: source selection, scoring, envelope (UUID correlation id), catalog wiring, make target, env key.
- `services/orion-hub/.env_example`, `services/orion-hub/README.md`: `HUB_BASE_URL`, docs.
- `tests/test_self_sense_scorers.py`: scorer edge cases and schema round-trip through the registry.
- `Makefile`: `eval-self-sense` (reads env from the primary checkout when run from a worktree, since worktrees carry no `.env`).
- `scripts/sync_local_env_from_example.py`: `HUB_BASE_URL` exact key.
- `config/metrics/metric_definitions.lock.json`: re-lock.
- `docs/superpowers/pr-reports/2026-09-09-self-sense-eval-pr.md`: this report.

## Schema / bus / API changes

- Added: channel `orion:self_sense:eval:write` (event, producer orion-hub, consumer orion-sql-writer), kind `self_sense.eval.write.v1`, schema `SelfSenseEvalV1` (`extra="forbid"`; `answer_source` is `harness_trace|http|none`; scores are `int >= 0`), table `self_sense_eval_log`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing paths. The eval's chat turns use `no_write: true`, so nothing lands in chat history.
- Compatibility notes: the row is keyed by a deterministic `entry_id` (`self-sense:<run_id>:<question_key>`) and the model is not insert-only, so a re-delivered envelope updates one row. **sql-writer must be redeployed** before rows land -- until then the publish reaches a channel with no subscriber (pub/sub is not durable).

## Env/config changes

- Added keys: `HUB_BASE_URL=http://127.0.0.1:8080` (orion-hub; read by the host-run eval, not the Hub process). sql-writer `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` gained one member each (value-level).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes (orion-hub, orion-sql-writer).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes -- `orion-hub: +HUB_BASE_URL='http://127.0.0.1:8080'`. The two pre-existing `orion-cocreation-signals` divergences it reported are untouched. The sync script cannot see inside a list value, so the live `services/orion-sql-writer/.env` was patched directly to add the channel and route entry (verified by grep; file still gitignored). The code-default guard in `effective_subscribe_channels` would have subscribed even without that patch.
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_self_sense_scorers.py                                   17 passed
services/orion-sql-writer/tests/test_self_sense_eval_log_sql_shape.py   7 passed
services/orion-hub/tests/test_self_sense_eval_producer.py           8 passed
tests/test_channel_prefix_guardrail.py, test_single_consumer_channels_gate.py,
  test_schema_registry_import_light.py, test_bus_reply_channel_catalog_coverage.py   pass
services/orion-sql-writer/tests (whole dir): 12 failures, identical set on main
  (biometrics_summary_sql_shape, chat_history_response_identity_merge x4,
   grammar_retention_periodic, journal_entry_payload_boundary, notify_attention_ack x3,
   notify_attention_escalate x2) -- pre-existing, not from this change.
Static gates: git diff --check PASS; check_env_template_parity.py PASS (85 services);
  check_metric_lineage.py --gate PASS; check_definition_drift.py --gate PASS after
  --update (one MEDIUM "added" delta: the new channel, lock committed).
```

## Evals run

```text
make eval-self-sense ARGS=--json   -- one live run, results under "Baseline" below.
```

## Baseline (live run, 2026-09-09)

BASELINE_PLACEHOLDER

## Docker/build/smoke checks

```text
Not deployed by this branch. sql-writer's consumer path is proven by the sqlite
end-to-end test and the live producer path by the captured bus envelope (below);
the row landing in Postgres is UNVERIFIED until sql-writer is redeployed.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
# 1. sql-writer must be redeployed for the new subscribe + route (creates the table at boot):
scripts/safe_docker_build.sh orion-sql-writer up -d --build
docker logs orion-athena-sql-writer --since 2m 2>&1 | grep -c "orion:self_sense:eval:write"

# 2. Optional: apply the migration ahead of the redeploy (idempotent):
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_self_sense_eval_log_v1.sql

# 3. Then run the eval and confirm rows landed:
make eval-self-sense
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "SELECT run_id, question_key, answer_source, self_label_score, grounded_record_score FROM self_sense_eval_log ORDER BY created_at DESC LIMIT 3"
```

No Hub or cortex-exec restart is required.

## Risks / concerns

- Severity: low. Concern: `grounded_record_score` is a floor -- it counts a named table/node/date/count without checking the claim is true, and a year like "2026" alone counts as an integer >= 10. Mitigation: documented as a floor, not a judge; the matched records are written into `notes` so any row can be audited by eye; the design's "LLM-judged theme" match is deliberately not built.
- Severity: low. Concern: `self_label_score` counts a negated label ("not a generic assistant") the same as an affirmed one. Mitigation: by design and tested; the prompt's own identity card uses that phrasing, so a non-zero score on question 1 should be read with `notes`.
- Severity: medium. Concern: no scheduler -- a baseline nobody re-runs is a single data point. Mitigation: follow-up is a cron entry or a Hub tick calling `make eval-self-sense` on the curiosity budget's cadence (once a day is enough; three chat turns).
- Severity: low. Concern: each run costs three real chat turns on the `orion` lane. Mitigation: `no_write`, fixed session id, and `--no-publish` for dry runs.

## Follow-ups

- Scheduler for the eval (daily), and a Hub Self tab panel reading `self_sense_eval_log` as a time series.
- `faculty_state_score` from the design (needs Patch D's anatomy snapshot).

## PR link

PR_LINK_PLACEHOLDER
