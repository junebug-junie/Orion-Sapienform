# PR: enforce the topic-foundry meaning/sentiment object contract

Branch: `fix/topic-foundry-enrichment-contract`
Spec: `docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md` (branch 4 of 4)

## Summary

- `GET /segments` on orion-topic-foundry returned **HTTP 500** for any run holding a
  segment whose `meaning`/`sentiment` jsonb columns held prose strings. Live count:
  552 of 554 enriched rows, both columns.
- That endpoint is the only source of `segment_speakers` for the concept-atlas
  participation edges shipped in PR #1932, and `concept_atlas_routes` degrades a
  segments-fetch failure to an empty map -- so a live ingest returned
  `available: true` with 19 concepts written and `participation_edges: 0`. The
  feature was dead on the live path with every test green.
- Root cause: `_llm_prompt` named the six JSON keys but never their shapes, so the
  model answered with prose; `_finalize_enrichment`'s `setdefault` is a no-op for a
  key that is present-but-wrong-typed.
- New `app/services/enrichment_contract.py` holds the declared shape as constants
  plus coercion. **The prompt's shape block is generated from the same constants the
  coercion validates against**, so the instruction and the validator cannot drift.
- `kg_edges` no longer swallows the resulting `JSONDecodeError` into `{}` -- that
  was the entire explanation for `topic_foundry_edges` holding 0 rows for all time.

## Outcome moved

| metric | before | after |
|---|---|---|
| `GET /segments` | HTTP 500 | 200, 394/394 segments with speakers |
| ingest `segments_fetched` | 0 | 375 |
| ingest `segments_with_speakers` | 0 | 255 |
| ingest `participation_edges` | 0 | 34 |
| ingest `edges_written` | 19 | 170 |
| ingest `segment_topic_map_buckets` | 0 | 19 |
| **A3** `topic_foundry_edges` rows | **0, all time** | **50 (14 `mentions`)** |
| **A4** newly enriched `jsonb_typeof` | 552 string / 2 object | **5/5 object, both columns** |
| atlas nodes / edges | 62 / 199 | **114 / 386** |
| atlas main component | 42 | **94** |
| Orion degree | 32.26 | **56.85** |
| Juniper degree | 26.74 | **44.22** |

## Current architecture

`enqueue_enrichment` -> `_enrich_segment` -> `_llm_enrich` (prompt naming six keys,
no shapes) -> `_finalize_enrichment` (`setdefault`) -> `update_segment_enrichment`
writes straight to two `jsonb` columns declared `Dict[str, Any]` on `SegmentRecord`.
Nothing between the LLM and Postgres checked the shape. On the read side,
`routers/segments.py` builds `SegmentRecord` at two independent call sites, and
pydantic raised `dict_type` on both.

## Architecture touched

One service (`orion-topic-foundry`). No bus, schema-registry, HTTP contract, or env
changes. `orion-hub` is an unmodified consumer that starts working again because the
endpoint it already called stops 500ing.

## Files changed

- `services/orion-topic-foundry/app/services/enrichment_contract.py` (new): declared
  shape constants, `coerce_meaning`/`coerce_sentiment`, and
  `describe_enrichment_shape()` which generates the prompt block from those same
  constants. Pure -- no DB, network, LLM, or settings.
- `services/orion-topic-foundry/app/models.py`: `SegmentRecord` field validators
  (`mode="before"`). Validators rather than a router-side fixup because there are two
  independent construction sites and a miss in either one is a 500 on the endpoint
  the participation edges depend on. This is what makes the 552 existing rows
  readable without paying to re-enrich them.
- `services/orion-topic-foundry/app/services/enrichment.py`: prompt states the shapes;
  `_finalize_enrichment` coerces instead of `setdefault`.
- `services/orion-topic-foundry/app/services/kg_edges.py`: uses the shared coercion
  and logs `kg_edges_segment_meaning_unstructured` for a segment that can never yield
  edges, instead of silently producing `[]`.
- `services/orion-topic-foundry/tests/test_enrichment_contract.py` (new): 23 tests.
- `docs/superpowers/specs/2026-08-28-...-design.md`: branch 4 results, the acceptance
  checks still open, and the follow-ups found while verifying.

## Design notes

- **Prose is preserved, not discarded.** Coercing to `{}` would also stop the 500 and
  would destroy every word the enricher produced for 552 rows. Prose lands under
  `summary` with `unstructured: true`, so a consumer can tell "the enricher wrote
  prose" from "the enricher wrote a real object" instead of both arriving as an
  indistinguishable dict.
- **A non-numeric sentiment scalar is dropped, not defaulted to 0.0.** A fabricated
  0.0 valence is indistinguishable from a measured neutral one, and
  `fetch_segments`'s `COALESCE(..., 0)` already renders absence as 0 for sorting
  without persisting a number nobody produced.
- **A bare string list key becomes a one-element list, never comma-split.**
  `"orion, juniper"` may be one entity or two, and this module does not get to guess.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `GET /segments` and `GET /segments/{id}` now return an object for
  `meaning`/`sentiment` where they previously returned HTTP 500. The declared
  response model is unchanged -- this makes the API match the schema it already
  published.
- Compatibility: strictly widening on the read side. A consumer that already handled
  `Dict[str, Any]` is unaffected. New enrichment writes the declared shape; existing
  rows are coerced on read and left untouched on disk.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable, no env surface touched.
- local `.env` synced: not applicable (no `.env_example` change; verified
  `git diff --name-only origin/main...HEAD | grep env_example` is empty).
- skipped keys requiring operator action: none.

## Tests run

```text
# new tests
$ pytest tests/test_enrichment_contract.py -q
23 passed in 4.99s

# full topic-foundry suite (collectible subset)
$ pytest tests -q --ignore=tests/test_drift_reducer_loading.py \
    --ignore=tests/test_heartbeat_chassis.py --ignore=tests/test_training_umap_reduction.py
1 failed, 52 passed in 5.48s

# hub consumer side
$ pytest tests -q -k "concept or atlas or topic_foundry"
160 passed, 1647 deselected in 41.08s
```

Pre-existing failures, unrelated to this change and not introduced by it:

- `test_drift_reducer_loading.py`, `test_heartbeat_chassis.py`,
  `test_training_umap_reduction.py` fail to **collect** -- `No module named 'sklearn'`
  / `'joblib'` in the local venv. They run inside the container only.
- `test_chat_corpus_builder_stages.py::test_stage_pipeline_outputs_expected_records`
  fails on `turns[0].has_commands is True`, a `COMMAND_RE` regex assertion in
  `app/pipelines/chat_corpus_builder/`. Verified that package imports nothing from
  any file this branch touches (`grep -rn "enrichment_contract\|from app.models\|
  coerce_meaning\|coerce_sentiment"` returns no hits there).

### Mutation testing

Each mutation was asserted to have actually landed in the file before the run, so a
no-op replacement cannot read as a passing test.

| mutation | result |
|---|---|
| delete both `SegmentRecord` field validators | fails with the **verbatim live error**: `Input should be a valid dictionary [type=dict_type, input_value='Negative', input_type=str]` |
| `_finalize_enrichment` back to `setdefault` | `test_finalize_coerces_a_present_but_wrong_typed_key` fails |
| drop `describe_enrichment_shape()` from the prompt | `test_llm_prompt_tells_the_model_the_two_fields_are_objects_not_strings` fails |
| remove the `kg_edges` unstructured warning | `test_kg_edges_logs_a_warning_for_a_segment_that_can_never_yield_edges` fails |

## Evals run

```text
none -- services/orion-topic-foundry has no evals/ directory
```

Coverage for this branch is gate tests plus the live verification below. Recorded as
a follow-up in the spec.

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-topic-foundry build
Image orion-topic-foundry-topic-foundry Built

$ scripts/safe_docker_build.sh orion-topic-foundry up -d topic-foundry
Container orion-athena-topic-foundry Started

$ curl http://localhost:8615/health                                   -> 200
$ curl ".../segments?run_id=f9443362-...&format=wrapped&limit=1000"    -> 200 (was 500)
   total 394, items 394, with speakers 394

$ curl -X POST http://localhost:8080/api/substrate/concepts/ingest-topic-foundry
   {"available": true, "run_id": "d3adedab-...", "topics_fetched": 19,
    "concepts_written": 18, "edges_written": 170, "segments_fetched": 375,
    "segment_topic_map_buckets": 19, "segments_with_speakers": 255,
    "participation_edges": 34}

$ curl -X POST http://localhost:8615/runs/d3adedab-.../enrich \
    -d '{"limit": 5, "force": true, "enricher": "llm"}'
   -> 5/5 rows jsonb_typeof(meaning)='object' AND jsonb_typeof(sentiment)='object'
   -> entities populated: ["Athena","sustained pressure","concept region"],
      ["Circe","10G pipe"], ["Athena","Concept field","Young person"]
   -> sentiment full numeric: {"stance":0.2,"arousal":0.3,"valence":0.5,
      "friction":0.4,"uncertainty":0.6}

$ psql -d conjourney -c "select count(*) from topic_foundry_edges"
   50   (was 0 for all time)
```

## Restart required

Already applied during verification. To reproduce from a clean checkout:

```bash
scripts/safe_docker_build.sh orion-topic-foundry build
scripts/safe_docker_build.sh orion-topic-foundry up -d topic-foundry
```

`orion-hub` needs no rebuild -- it is an unmodified HTTP consumer.

## Risks / concerns

- Severity: low. Concern: the 552 legacy rows are coerced on read, so their `meaning`
  carries `{"summary": ..., "unstructured": true}` and yields no entity edges. Only a
  re-enrichment produces real structure from them. Mitigation: that is an LLM-cost
  decision for Juniper, not something to run unasked; `POST /runs/{id}/enrich` with
  `force: true` is the mechanism, and `kg_edges` now logs every such segment by id.
- Severity: low. Concern: `unstructured: true` is a new key that consumers do not yet
  branch on. Mitigation: it is additive inside an already-`Dict[str, Any]` field, and
  the only consumer that needs it (`kg_edges`) reads it in this same patch.
