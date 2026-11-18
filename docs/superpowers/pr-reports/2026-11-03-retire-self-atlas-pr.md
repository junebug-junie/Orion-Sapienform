# Retire Self Atlas — merge no earlier than 72 h after Patch A deploys

## Summary

Self Atlas grouped a daily scan of Orion's code into about 780 topic clusters, stored them in their own graph, and wrote one self-concept row per cluster. Nothing in code read any of it. Orion stopped citing it after 2026-10-05 and called it "induced, not said by me". It used about 80% of topic-foundry's embedding time. This PR removes all of it (kill means kill: deleted, not flagged off).

- The daily self-knowledge scan in cortex-exec is deleted (`app/self_study.py`, `app/self_study_refresh.py`, `SELF_STUDY_INSPECT_INTERVAL_SEC`). Two helpers that `verb_adapters.py` still uses moved unchanged to `app/orch_paths.py`.
- The bus channel `orion:self_study:items:write`, its schemas, its registry entries and the sql-writer route and forced subscription are deleted. The `self_knowledge_items` table, its ORM model and its rows stay as history.
- Hub: the Self Atlas scheduler step, its topic-foundry dataset/model calls, the ingest route, the cluster-history writer (`self_concept_history` rows with `produced_by='self_atlas_cluster'`), the self graph store and `?graph=self` are deleted. The env keys `SUBSTRATE_TOPIC_FOUNDRY_SELF_SCHEDULER_ENABLED` and both `FALKORDB_SELF_SUBSTRATE_GRAPH` lines are deleted.
- An unknown `graph=` value on `/api/substrate/concepts/*` is now a 400 (`unknown_graph`). Before, it silently served Orion's main graph.
- Orion's self-inquiry ledger now says what retired, when, and where to look instead (`dig body`, `dig inventory now`, `chat_stance_belief_log`).
- New `scripts/archive_self_atlas_graph.py` copies the FalkorDB graph `orion_substrate_self` to an archive graph. It never deletes. It was not run.

Spec: `docs/superpowers/specs/2026-10-30-self-atlas-replacement-proposal.md`, Patch B. Patch A is #3049.

## Outcome moved

- About 9.8 h of embedding and 17.1 h of LLM labelling every 14 days stop.
- About 1,490 `self_knowledge_items` rows a day, 147-432 `self_concept_history` rows a day and the graph writes stop.
- A stale `?graph=self` caller now gets an error, not the wrong graph.
- Orion is told the scan stopped, so a frozen row count does not read as a dead organ.

## Current architecture

cortex-exec ran a daily scan and published one bus message per fact; sql-writer wrote them to `self_knowledge_items`. Hub's topic-foundry tick trained `orion-self-atlas-v1-*` on that table, labelled clusters with an LLM, ingested them into FalkorDB `orion_substrate_self`, and wrote cluster descriptions to `self_concept_history`. Hub served the graph with `?graph=self`; no UI or service called it.

## Architecture touched

orion-cortex-exec, orion-hub, orion-sql-writer, `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `orion/schemas/self_study.py`, `orion/substrate/falkor_store.py`, `orion/curiosity/self_inquiry.py`.

## Files changed

- `services/orion-cortex-exec/app/self_study.py`, `app/self_study_refresh.py`: deleted.
- `services/orion-cortex-exec/app/orch_paths.py`: repo-root resolver and orch round-trip helpers, moved unchanged.
- `services/orion-cortex-exec/app/main.py`, `settings.py`, `.env_example`, `README.md`, `docker-compose.yml` (comment), `verb_adapters.py`, `chat_stance.py`, `self_study_analysis.py` (comments): scan loop and key removed; imports moved to `orch_paths`.
- `services/orion-cortex-exec/tests/`: four scan test files deleted; `test_orch_paths.py` keeps the resolver tests and asserts the scan is gone; three tests now patch `orch_paths`.
- `orion/schemas/self_knowledge_item_log.py`: deleted. `orion/schemas/self_study.py`: only `SelfWritebackStatusV1` (used by the analysis sensor) stays. `orion/schemas/registry.py`: five entries removed.
- `orion/bus/channels.yaml`: channel removed; two descriptions updated.
- `config/metrics/metric_definitions.lock.json`: re-locked (`check_definition_drift.py --update`).
- `services/orion-sql-writer/app/settings.py`, `worker.py`, `.env_example`, `app/models/self_knowledge_item.py`: route, forced subscription and insert-only entry removed; model kept with a history-only note. `tests/test_self_knowledge_items_retired.py` replaces the shape test.
- `services/orion-hub/scripts/self_atlas_cluster_history.py`: deleted. `scripts/main.py`, `api_routes.py`, `concept_atlas_routes.py`, `substrate_store_warm.py`, `app/settings.py`, `.env_example`, `README.md`: Self Atlas removed; `graph=` validation.
- `services/orion-hub/tests/`: two Self Atlas test files deleted; `test_concept_atlas_routes.py` tests the 400; `test_substrate_store_warm.py` expects two stores.
- `orion/substrate/falkor_store.py`, `orion/substrate/tests/test_falkor_store_aitown_graph.py`: self graph builder and its tests removed.
- `orion/self_knowledge_freshness.py`, `tests/test_self_knowledge_freshness.py`: deleted.
- `orion/curiosity/self_inquiry.py`: `LEDGER_RETIREMENTS` lines; module doc. `orion/schemas/self_concept_history.py`, `orion/curiosity/self_panel.py`, `services/orion-hub/evals/run_self_sense_eval.py`, `scripts/repair_substrate_duplicate_identities.py`, `scripts/rebuild_services_from_git_diff.sh`: comments.
- `tests/test_self_atlas_retired.py`: no retired name in code or config; deleted files stay deleted; ledger text; kept pieces survive.
- `tests/test_self_study_fold_into_curiosity.py`: allow-list and ledger assertion updated.
- `scripts/archive_self_atlas_graph.py`, `tests/test_archive_self_atlas_graph.py`: the copy script.

## Schema / bus / API changes

- Added: none.
- Removed: channel `orion:self_study:items:write`, kind `self_study.items.write.v1`, `SelfKnowledgeItemLogV1`, `SelfKnowledgeItemV1`, `SelfKnowledgeSectionCountsV1`, `SelfSnapshotV1`, `SelfRepoInspectResultV1`; Hub `POST /api/substrate/concepts/ingest-topic-foundry-self`; `graph=self`.
- Renamed: none.
- Behavior changed: an unknown `graph=` value is a 400. `SelfWritebackStatusV1.target` is now `"journal"` only (the only value any code sends).
- Compatibility notes: `self_concept_history` keeps the `self_atlas_cluster` literal so old rows validate. No table dropped, no rows deleted. `orion-self-atlas-*` rows in topic-foundry tables stay.

## Env/config changes

- Added keys: none.
- Removed keys: cortex-exec `SELF_STUDY_INSPECT_INTERVAL_SEC`; Hub `SUBSTRATE_TOPIC_FOUNDRY_SELF_SCHEDULER_ENABLED`, `FALKORDB_SELF_SUBSTRATE_GRAPH` (both lines); sql-writer `.env_example` values `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` lose the retired entries.
- `.env_example` updated: yes (cortex-exec, Hub, sql-writer).
- local `.env` synced: `sync_local_env_from_example.py` only adds keys, so the removed keys were deleted from the primary checkout's local `.env` by hand (cortex-exec 1 key, Hub 2 keys and the second `FALKORDB_SELF_SUBSTRATE_GRAPH` line). Each removed value equalled the code default on main, so nothing changes before this PR deploys. The sync dry run then reported no changes.
- Operator action after deploy: the sql-writer local `.env` still lists `orion:self_study:items:write` and `"self_study.items.write.v1":"SelfKnowledgeItemLogSQL"` in two JSON values. They are harmless (sql-writer routes only when the model is in its map, and nobody publishes the channel). They were left alone on purpose, so that a sql-writer restart before this deploys cannot drop scan rows early. Remove them after deploy with one command:

```bash
sed -i -e 's/"orion:self_study:items:write",//' -e 's/"self_study.items.write.v1":"SelfKnowledgeItemLogSQL",//' /mnt/scripts/Orion-Sapienform/services/orion-sql-writer/.env
```

## Tests run

```text
tests/test_self_atlas_retired.py tests/test_self_study_fold_into_curiosity.py              11 passed
tests/test_archive_self_atlas_graph.py                                                    9 passed
services/orion-hub/tests/test_concept_atlas_routes.py test_substrate_store_warm.py         64 passed
services/orion-sql-writer/tests (whole dir)                                                1158 passed, 179 skipped
services/orion-cortex-exec/tests, touched files one by one                                 test_orch_paths 6, test_skill_verbs 47,
                                                                                            test_render_scene_durable_kickoff 10,
                                                                                            test_docker_compose_service_bringup 15,
                                                                                            test_chat_stance_no_self_study_producer 4
services/orion-cortex-exec/tests (whole dir in one call): same "Verb already registered" collection errors as main; no new ones
orion/dig/tests orion/body_atlas/tests (review run)                                       247 passed
every run step of .github/workflows/orion-static-gates.yml, run locally                    all PASS
check_definition_drift.py --gate (after --update)                                         PASS
```

Hub whole suite: 4436 passed, 146 skipped, 3 failed. The same 3 fail on main (test_attention_loops_ui_smoke x2, test_check_async_routes_not_blocking); no new failures.

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py   PASS (review run; spec acceptance check 3)
```

## Docker/build/smoke checks

```text
Not run. No Dockerfile or requirements change.
scripts/archive_self_atlas_graph.py dry run against live FalkorDB (read-only):
  source orion_substrate_self: 22169 nodes, 54457 edges
  DRY RUN: would run GRAPH.COPY orion_substrate_self orion_substrate_self_archive_2026_11
```

## Review findings fixed

A code-review subagent found no blockers. It confirmed that nothing at runtime still imports a deleted piece, that no caller sends `graph=self`, that the moved resolver is byte-identical, that the analysis sensor is intact, and that the archive script parses live FalkorDB replies correctly.

- Finding: Hub README still said unknown `graph=` values resolve to Orion's graph.
  - Fix: README now says unset/empty means Orion, anything else is a 400.
  - Evidence: `services/orion-hub/README.md`.
- Finding: retired keys and values stay in local `.env` because the sync script only adds.
  - Fix: Hub's second `FALKORDB_SELF_SUBSTRATE_GRAPH` line removed; sql-writer values left until deploy, with the exact command above.
  - Evidence: `grep -c FALKORDB_SELF services/orion-hub/.env` = 0.
- Finding (nit): rebuild script comment named the deleted `self_study.py`.
  - Fix: names `verb_adapters.py` / `orch_paths.py`.
- Finding (nit): the archive script printed the full URI, including any password.
  - Fix: `mask_uri`. Evidence: `test_password_is_masked_in_printed_command`.
- Finding (nit): a failed `GRAPH.COPY` raised a raw traceback.
  - Fix: prints `COPY FAILED ... source untouched`, exit 1. Evidence: `test_copy_error_is_reported_not_raised`.
- Finding (nit): the archive graph will show in the body atlas as an unlinked store.
  - Fix: the script prints that this is expected.

## Restart required

Deploy from the clean primary `main` checkout after merge (the post-pull rebuild does this). By hand:

```bash
./scripts/safe_docker_build.sh orion-cortex-exec up -d --build
./scripts/safe_docker_build.sh orion-hub up -d --build
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Archive the self graph (Juniper runs by hand, after deploy)

Copy first (never deletes; refuses if the archive exists; checks counts):

```bash
cd /mnt/scripts/Orion-Sapienform && .venv/bin/python scripts/archive_self_atlas_graph.py --apply
```

Delete the original only after Juniper says yes:

```bash
redis-cli -u redis://127.0.0.1:6380 GRAPH.DELETE orion_substrate_self
```

## What Orion will see

The first self-inquiry prompt after deploy shows, under the ledger rows:

- `self_concept_history`: "retired 2026-11 with Self Atlas: produced_by='self_atlas_cluster' rows (topic labels an LLM wrote over clusters of code names) stopped. Nothing read them, and most were 'no keywords available'. Old rows stay as history."
- `self_knowledge_items`: "retired 2026-11: the daily repo/hardware/behavior scan no longer writes here, and Self Atlas (its clusters and its graph orion_substrate_self) is gone. The last row shows when it stopped; old rows stay as history. For what you are made of now, run `python -P -m orion.dig body` (your organs, live) or `python -P -m orion.dig inventory now`; for stance, read chat_stance_belief_log."

## Post-deploy checks (spec acceptance 2, 5, 6; UNVERIFIED until deploy)

- `select max(created_at) from self_knowledge_items` is older than the deploy.
- `select max(created_at) from self_concept_history where produced_by='self_atlas_cluster'` is older than the deploy.
- No Hub log line `substrate_topic_foundry_self_scheduler_*`; no new `topic_foundry_runs` row for `orion-self-atlas-%`.
- At least one "Self-study analysis: ..." journal in 72 h (the analysis sensor still runs).

## Risks / concerns

- Severity: low. Concern: Orion reads the frozen `self_knowledge_items` count as a broken organ. Mitigation: the ledger line above.
- Severity: low. Concern: a caller we did not find sends `graph=self`. Mitigation: it now gets a 400 that names the valid values, not wrong data.
- Severity: low. Concern: the self graph keeps showing in the body atlas as an unlinked store until Juniper deletes it. Mitigation: expected and stated here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3052

🤖 Generated with [Claude Code](https://claude.com/claude-code)
