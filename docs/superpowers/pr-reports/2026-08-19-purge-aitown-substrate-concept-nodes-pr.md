## Summary

- Purged the one confirmed-polluted `orion-topic-foundry` run's concept/evidence nodes from the live `orion_substrate` FalkorDB graph -- the retroactive cleanup Track A's design doc deferred until the AI-Town dataset filter (PR #1721) and the Concept Atlas HDBSCAN fix (PR #1726) both shipped.
- Run `87e6539e-0962-4ef3-8dc1-568866c4c57d` (unfiltered dataset, `where_sql=None`, `min_cluster_size=15` -- the exact broken default #1726 fixed) trained on AI-Town NPC-roleplay chat and produced 22 concept labels like "Electrical Testing", "storm and memory", "Soldering Techniques" -- the literal topics named in the original pollution bug report.
- **Live-executed and live-verified**: `orion_substrate` went from 66 -> 22 nodes. Re-confirmed again just now via direct Cypher query: 22 nodes total (16 `Concept`, 6 `Evidence`), 0 nodes matching the polluted run's id prefix.
- Code review on this branch caught two real issues in the script itself (both fixed) and, because it examined live file state rather than a strict diff, surfaced 5 additional real findings in already-merged code from PR #1743/#1744 -- those are out of this branch's scope and are being fixed separately (`fix/aitown-routing-followup-fixes`).

## Outcome moved

`orion_substrate`'s Concept Atlas no longer surfaces AI-Town roleplay content as if it were real Orion/Juniper conversation topics. Zero collateral damage -- verified before the run that no edge pointed from a kept node into the polluted subgraph, and after the run that exactly the 22 correct nodes remain.

## Current architecture

`orion-topic-foundry` clusters `chat_history_log` text (BERTopic/HDBSCAN) and writes Concept/Evidence nodes into the `orion_substrate` FalkorDB graph, keyed by `run_id` in the node id (`sub-concept-topicfoundry-<run_id>-...` / `sub-evidence-topicfoundry-<run_id>-...`). Nothing retroactively cleans up nodes from a run made obsolete by a later dataset/config fix -- this was a one-off script for that gap, not a recurring job.

## Architecture touched

- `scripts/purge_aitown_polluted_substrate_concepts.py` only. No service, schema, or contract changes -- this purges already-ingested graph data, it doesn't change how new data is produced or consumed.

## Files changed

- `scripts/purge_aitown_polluted_substrate_concepts.py`: one-off cleanup script. Snapshot-first (`snapshot_before.json`), counts before/after, single `DETACH DELETE` Cypher for the match set, `--dry-run` support. Rewritten during review (see below) to use the repo's shared `RedisGraphQueryClient` instead of a hand-rolled Redis wrapper, and to write `progress.log`/`report.md`/`before_after.csv` per AGENTS.md section 14.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

No existing test suite covers one-off FalkorDB cleanup scripts (consistent with every sibling script of this shape in the repo -- `scripts/backfill_*`, `scripts/purge_*`). Correctness was established by:

```text
python3 -m py_compile scripts/purge_aitown_polluted_substrate_concepts.py   -> OK
```

## Evals run

No eval harness exists for one-off graph cleanup scripts.

## Docker/build/smoke checks

Live-run against the real `orion_substrate` FalkorDB graph (not a smoke/staging copy -- this is a one-off targeted purge, snapshot-first per AGENTS.md section 14):

```text
before: {"graph_total_nodes": 66, "matching_polluted_nodes": 44}
purge:  DETACH DELETE on the 44 matched nodes (22 Concept + 22 Evidence)
after:  {"graph_total_nodes": 22, "matching_polluted_nodes": 0}
verdict: ok
```

Re-verified again just now, independently of any script run, via a direct Cypher query against the live graph:

```text
MATCH (n) RETURN count(n) AS c                        -> 22
MATCH (n) RETURN labels(n) AS l, count(n) AS c         -> [Evidence: 6, Concept: 16]
```

16 Concept + 6 Evidence = 22, matching the pre-run expectation exactly (12 seed/substrate nodes + 4 clean topicfoundry concepts from the two post-fix runs `ece65e49`/`2032434f` + 6 clean evidence nodes). Re-ran the rewritten script's `--dry-run` mode against this now-clean state to validate the `RedisGraphQueryClient` rewrite is behaviorally identical to the original run:

```text
before: {"graph_total_nodes": 22, "matching_polluted_nodes": 0}
DRY RUN: would delete 0 nodes. No changes made.
```

## Review findings fixed

- Finding: script violated AGENTS.md section 14's backfill protocol -- no `progress.log`, no `report.md`/`before_after.csv`, only a `report.json`.
  - Fix: added `_log_progress()` (writes to stdout + `progress.log`), `_write_before_after_csv()`, `_write_report_md()`.
  - Evidence: re-run produced all four artifacts under `/tmp/aitown_substrate_concept_purge/`.
- Finding: script hand-rolled a raw `redis.Redis` + `GRAPH.QUERY` wrapper instead of reusing `orion/graph/falkor_client.py::RedisGraphQueryClient`, which every sibling FalkorDB script in the repo already uses.
  - Fix: rewritten to import and use `RedisGraphQueryClient`.
  - Evidence: `--dry-run` re-run against the now-clean graph confirms identical, correct behavior (0 matching nodes, no changes).
- Findings NOT fixed on this branch (real, but in already-merged code from a different PR, not this script -- being fixed on `fix/aitown-routing-followup-fixes` instead, per CLAUDE.md section 5's service-boundary/thin-patch discipline):
  - `_apply_spark_meta_patch` (PR #1743) never acquired `_lock_chat_history_row` despite its sibling function's docstring requiring it.
  - The `SparkTelemetrySQL` back-populate branch and `_chat_history_thought_for_merge` (PR #1743) both only ever queried the primary `ChatHistoryLogSQL` table, silently missing AI-Town-routed rows.
  - `orion-recall`'s `sql_chat.py` id-batch lookups share one try/except around two `_fetch_rows_from_table` calls, so a mirror-table query failure discards already-successful primary-query results (PR #1744).
- Finding ruled explicitly out of scope: an unrelated, already-merged PR (#1742, `feat/chat-history-response-identity`) has its own incomplete rollout (`response_identity` column referenced in code but not yet migrated onto the live database). Not touched -- unrelated to this branch's purpose.

## Restart required

```text
No restart required.
```

This is a one-off data purge against a live datastore, not a code deploy -- no service restart changes its behavior.

## Risks / concerns

- Severity: low
- Concern: this script is a one-off, hardcoded to the single confirmed-polluted `run_id`. A future polluted run needs its own provenance trace and its own script invocation with a new `_POLLUTED_RUN_ID`, not a rerun of this one with a loosened filter.
- Mitigation: documented explicitly in the script's own module docstring.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1748
