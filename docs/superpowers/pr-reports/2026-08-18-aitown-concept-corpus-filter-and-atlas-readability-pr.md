## Summary

Track A of `docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md` — the tactical, ships-fast fix while Track B (physically splitting AI Town chat data into its own table) stays a design-only, phased effort.

- `chat_history_log` is ~90% `source='orion-embodiment'` (AI Town) rows; topic-foundry's dataset definition had zero platform filter, so Orion's "organically clustered" concept graph god nodes have mostly been AI Town NPC dialogue topics since the pipeline first ran.
- Excludes AI Town via the canonical `client_meta.external_room.platform=='aitown'` tag (`services/orion-recall/app/chat_source_tagging.py`'s established signal), using topic-foundry's already-existing `where_sql` dataset field.
- Renamed dataset/model to `-v2`: topic-foundry's dataset/model routes are create-only (no update endpoint), so changing `where_sql` under the old name would have been a silent no-op.
- Fixes Concept Atlas's Cytoscape layout: `nodeDimensionsIncludeLabels` was unset, so labels overlapped directly once the graph got dense.
- **Code review (high effort, 2 sub-agents) caught a real correctness bug**: the "latest completed run" lookup wasn't scoped to the new model, so ingestion/enrichment could keep silently reading the *old*, unfiltered model's runs regardless of the rename — fixed in the same PR (see "Review findings fixed" below).

## Outcome moved

Orion's concept graph stops training on a corpus that's 90% AI Town dialogue. Verified live, not just by code inspection — see "Docker/build/smoke checks."

## Current architecture

Two independent topic-foundry consumers exist in `services/orion-hub/scripts/concept_atlas_routes.py`: a scheduler that trains/enriches/ingests on a tick, and this PR touches only the dataset/model definition and the "which run is latest" resolution — no schema or contract changes.

## Files changed

- `services/orion-hub/scripts/concept_atlas_routes.py`: `_TOPIC_FOUNDRY_WHERE_SQL`/`_AITOWN_PLATFORM_TAG` added, dataset/model renamed to `-v2`, where_sql-drift warning added, both `fetch_latest_completed_run`/`fetch_run_topics_and_keywords` call sites now pass `model_name`.
- `services/orion-hub/scripts/topic_foundry_client.py`: `fetch_latest_completed_run`/`fetch_run_topics_and_keywords` gain a `model_name` param, passed through to topic-foundry's existing `GET /runs?model_name=` filter.
- `services/orion-hub/static/js/concept-atlas.js`: Cytoscape `cose` layout gains `nodeDimensionsIncludeLabels: true` + `componentSpacing: 80`.
- `services/orion-hub/tests/test_topic_foundry_scheduler.py`, `test_concept_atlas_ingest_topic_foundry.py`: new/updated assertions covering `where_sql`, the drift warning, and `model_name` scoping.
- `docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md`: copied into this branch (also lives on a separate docs-only PR) so in-code comment references resolve here regardless of merge order.

## Schema / bus / API changes

None — internal Python function signatures only (`model_name` is a new optional kwarg, backward compatible). No bus/schema/contract changes.

## Env/config changes

None.

## Tests run

```
.venv/bin/python3 -m pytest tests/test_topic_foundry_scheduler.py tests/test_concept_atlas_routes.py tests/test_concept_atlas_ingest_topic_foundry.py -q
  → 77 passed
```

## Evals run

No eval harness exists for this pipeline.

## Docker/build/smoke checks

Rebuilt and redeployed `orion-athena-hub` live (twice — once before review, once after fixes) and verified end-to-end against the real `orion-topic-foundry` service, not mocked:

```
$ docker exec orion-athena-hub python3 -c "... car._ensure_topic_foundry_dataset_and_model(...)"
dataset: orion-hub-autonomous-dataset-v2
model: orion-hub-autonomous-v2
where_sql: (client_meta -> 'external_room' ->> 'platform') IS DISTINCT FROM 'aitown'
ensure result: ('e85f9a4e-...', 'cd1954a4-...')   # idempotent, no duplicate creation
```

Proved the `model_name` scoping fix is real (not a no-op) by comparing unscoped vs. per-model lookups — the old model and the new model return genuinely different `run_id`s:

```
unscoped:                     run 5ca664f6...
scoped to old model:          run 795139fa...   # different run
scoped to new -v2 model:      run 5ca664f6...
```

Confirmed the dataset filter is functionally correct, not just syntactically accepted — the new model's first real run processed **60 documents**, vs. **~1,743** in the unfiltered corpus:

```json
{"docs_generated": 60, "cluster_count": 0, "outlier_pct": 1.0}
```

**Known, honest, expected consequence**: at 60 documents, HDBSCAN's default `min_cluster_size=15` finds zero real clusters, so ingestion correctly reports `topic_foundry_no_usable_topics` rather than fabricating concepts:

```json
{"available": false, "reason": "topic_foundry_no_usable_topics", "concepts_written": 0}
```

Concept Atlas will show few/no "Orion" concepts for a while after this deploys — not a bug, but a real, foreseeable side effect worth knowing about rather than being surprised by (documented in the design spec). It'll fill back in as real chat volume accumulates in the 30-day rolling window, or `min_cluster_size` gets deliberately tuned for the smaller corpus as a follow-up.

## Review findings fixed

- Finding: `fetch_latest_completed_run()`/`fetch_run_topics_and_keywords()` resolved "the latest completed run" globally across every topic-foundry model, not scoped to the new model — ingestion/enrichment could keep silently reading the old, unfiltered model's runs regardless of the corpus-filter rename.
  - Fix: both functions take `model_name`; both call sites pass `_TOPIC_FOUNDRY_MODEL_NAME`.
  - Evidence: live comparison above showing scoped vs. unscoped resolve to different runs; new test assertion (`test_trigger_topic_foundry_enrichment_success`) and existing-test extension (`test_ingest_normal_run_writes_concepts_excludes_outlier_and_below_floor`) both assert `model_name` reaches the HTTP call.
- Finding: get-or-create for the dataset matches purely by name, so a future `where_sql` edit under the same name would silently keep the stale filter forever (no update endpoint exists).
  - Fix: logs a `topic_foundry_dataset_where_sql_drift` warning when the found dataset's `where_sql` doesn't match the expected constant.
  - Evidence: new test `test_ensure_dataset_and_model_warns_on_where_sql_drift`.
- Finding: the design doc this PR's comments cite didn't exist on this branch (created in a separate, unmerged docs-only PR).
  - Fix: copied the doc into this branch too.
  - Evidence: `docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md` present in this diff.
- Finding: the AI Town platform tag was hand-rewritten as an inline SQL string literal instead of referencing the canonical `chat_source_tagging.AITOWN_TAG` signal.
  - Fix: named local `_AITOWN_PLATFORM_TAG` constant (deliberately not cross-importing across the service boundary — would violate this repo's service-isolation convention); comment documents the must-match-by-hand requirement explicitly.
  - Evidence: `concept_atlas_routes.py`'s updated constant block.
- Finding (topic-foundry's own hardening gap, not fixed here — flagged): `where_sql` is interpolated as a raw SQL fragment with no server-side validation; this PR is the first caller to populate it with a real value in production. Not exploitable via this diff (hardcoded module constant, no user input), but worth a follow-up hardening item in `orion-topic-foundry` before `where_sql` is ever exposed to a less-trusted caller.

## Restart required

Already done live on this session's Athena host as part of verification — `orion-athena-hub` is running the new image now.

## Risks / concerns

- Severity: low
- Concern: Concept Atlas's Orion graph will look sparse/empty for a while post-deploy (see "Docker/build/smoke checks" above), which could read as a regression to someone who doesn't know why.
- Mitigation: documented in the design spec and this PR description; not a defect — the alternative (keeping the old behavior) means the graph would still be reading a corpus that's 90% wrong.
- Severity: low
- Concern: `min_cluster_size=15` may need tuning for the smaller, real corpus size to ever produce concepts at all.
- Mitigation: explicitly flagged as a follow-up, not guessed at unsupervised in this PR — a real judgment call about clustering quality tradeoffs.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1721 (this PR)

Design doc companion PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1719
