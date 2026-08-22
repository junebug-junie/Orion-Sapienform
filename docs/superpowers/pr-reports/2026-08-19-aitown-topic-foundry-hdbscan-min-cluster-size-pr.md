## Summary

Follow-up to PR #1721 (AI Town concept-corpus filter). That fix correctly excluded AI Town rows from Orion's concept-graph training corpus, but its own PR report flagged the honest consequence: the real, much smaller Orion-only corpus (60-160ish documents) produced **0 clusters** every run, because Hub's scheduler-owned topic-foundry model had hardcoded `min_cluster_size=15` since the pipeline first shipped -- the exact value topic-foundry's own 2026-07-21 incident note flags as producing degenerate clusters.

- Exposed `min_cluster_size`/`metric` as new Hub settings (`SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE`/`_METRIC`, defaults 8/euclidean) instead of hardcoded literals.
- **Live-verification caught a real bug in the first attempt**: copied `metric="cosine"` from topic-foundry's own `ModelSpec` field default without independently verifying it against the real training path -- a live training run failed outright with `ValueError("Unrecognized metric 'cosine'")`. Corrected to `"euclidean"` (topic-foundry's own `TOPIC_FOUNDRY_HDBSCAN_METRIC` setting, already this deployment's real working value) and re-verified: a fresh run produced `cluster_count=3` on 62 real documents, and full ingestion wrote 3 real concepts to the live substrate graph.
- Code review (subagent) caught 3 more issues, all fixed same PR: unvalidated metric string (fail-late risk), `.env` not synced, and a hand-bumped model-name-suffix convention that would silently reproduce the exact "forgot to bump it" bug class it exists to prevent.

## Outcome moved

Concept Atlas can now actually populate from Orion's real, AI-Town-filtered chat corpus instead of sitting permanently empty. Verified live end-to-end (training + ingestion), not just by code inspection.

## Current architecture

`services/orion-hub/scripts/concept_atlas_routes.py::_ensure_topic_foundry_dataset_and_model` does idempotent get-or-create for the scheduler's topic-foundry dataset+model. topic-foundry's model-create API is create-only (no update endpoint) -- model_spec is fixed forever at creation, so retuning it always requires a new model name.

## Files changed

- `services/orion-hub/app/settings.py`: two new settings (`SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE`=8, `_METRIC`=euclidean); `field_validator` on the metric field against a live-queried allow-list of `sklearn.neighbors.KDTree`/`BallTree.valid_metrics` (pinned to this deployment's `hdbscan==0.8.41`).
- `services/orion-hub/scripts/concept_atlas_routes.py`: model_spec now reads from settings instead of hardcoded `min_cluster_size=15, metric="euclidean"`; model name suffix changed from a hand-bumped version literal to a SHA-256 fingerprint of the settings-driven model_spec fields, so a real config change always produces a fresh model name automatically.
- `services/orion-hub/.env_example`: two new keys, documented with the live-verification story.
- `services/orion-hub/tests/test_topic_foundry_scheduler.py`: model-creation payload assertions extended to pin `min_cluster_size=8`/`metric="euclidean"`; 3 new tests for the metric validator (rejects `cosine`, accepts `euclidean`) and the fingerprint function (deterministic, changes with settings).

## Schema / bus / API changes

None. Internal settings + one internal function's model-name derivation.

## Env/config changes

- Added keys: `SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE=8`, `SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC=euclidean`.
- `.env_example` updated: yes.
- local `.env` synced: yes, added directly to `services/orion-hub/.env` in the primary checkout (same values, matches what was live-verified). `scripts/sync_local_env_from_example.py` was not sufficient here on its own -- known gotcha: it reads `.env_example` from the primary checkout, and this branch's `.env_example` change wasn't merged there yet.
- skipped keys requiring operator action: none.

## Tests run

```
.venv/bin/python3 -m pytest services/orion-hub/tests/test_topic_foundry_scheduler.py \
  services/orion-hub/tests/test_concept_atlas_routes.py \
  services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py -q
  → 80 passed
```

Also ran the full `services/orion-hub/tests/` suite (1292 passed, 59 failed, 2 errors, pre-existing and unrelated -- none in any topic_foundry/concept_atlas/settings-adjacent test file; same failures independent of this patch, consistent with running outside the container without full live infra).

## Evals run

No eval harness exists for this pipeline (same gap noted in PR #1721).

## Docker/build/smoke checks

Rebuilt and redeployed `orion-athena-hub` live twice (once for the initial fix, once after review fixes), verified against the real `orion-topic-foundry` service:

**First attempt (caught the cosine bug):**
```
$ docker exec orion-athena-hub python3 -c "... trigger_topic_foundry_training_run() ..."
model: orion-hub-autonomous-v3 (min_cluster_size=8, metric="cosine")
run result: status=failed, error="Unrecognized metric 'cosine'"
```

**Corrected (metric=euclidean):**
```
model: orion-hub-autonomous-v4
run result: status=complete, doc_count=62, cluster_count=3, outlier_pct=0.258
ingest result: available=true, concepts_written=3, evidence_nodes_written=3, edges_written=6
```

**After review fixes (fingerprinted naming + validator):**
```
$ docker exec orion-athena-hub python3 -c "from app.settings import Settings; Settings()" # with SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC=cosine
ValidationError: SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC='cosine' is not a metric sklearn's KDTree/BallTree recognize ...

model name: orion-hub-autonomous-v4-1d4bd814
ensure result (1st call): ('e85f9a4e-...', '7f92bab3-...')
ensure result (2nd call): ('e85f9a4e-...', '7f92bab3-...')  # identical -- idempotent, no duplicate creation
```

## Review findings fixed

- Finding: `SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC` was a bare unvalidated `str` -- could silently reproduce the exact incident this PR fixes (creates a model successfully, fails deep in a background training task).
  - Fix: `field_validator` against a live-queried allow-list of real sklearn/hdbscan-recognized metrics.
  - Evidence: live test above -- `cosine` rejected at `Settings()` construction; new tests `test_topic_foundry_hdbscan_metric_validator_rejects_unrecognized_metric`/`_accepts_euclidean`.
- Finding: new `.env_example` keys were not synced into local `services/orion-hub/.env`.
  - Fix: added directly.
  - Evidence: `grep SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN services/orion-hub/.env` now returns both keys.
- Finding: hand-bumped model-name version suffix (`-v3`/`-v4`) is a naming-discipline bandaid -- a future settings change without a matching manual bump would silently keep training on the stale model_spec forever, undetectable on the model side (unlike the dataset's where_sql case, topic-foundry's `GET /models` doesn't return `model_spec` to diff against).
  - Fix: model name suffix now derived from a SHA-256 fingerprint of the settings-driven model_spec fields -- a real config change always produces a new name automatically.
  - Evidence: live idempotency check above; new test `test_topic_foundry_model_spec_fingerprint_changes_with_settings`.

## Restart required

Already done live on this session's Athena host as part of verification -- `orion-athena-hub` is running the new image now.

## Risks / concerns

- Severity: low
- Concern: the `hdbscan`-recognized-metric allow-list validates that a value is a real distance metric name, not that it produces *good* clusters for text embeddings. Only `euclidean` is live-verified to cluster well for this corpus.
- Mitigation: documented explicitly in the settings.py comment; not a blocker since `euclidean` is the shipped default and the only value anyone has reason to change it from without first live-verifying the alternative the same way this PR did.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1726
