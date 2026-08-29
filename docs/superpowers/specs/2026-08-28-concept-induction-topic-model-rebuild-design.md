# Concept induction rebuild: topic model -> concept graph

Date: 2026-08-28
Status: design, approved to build
Branch: `feat/concept-induction-rebuild`

## Naming, first, because it caused real confusion

"Concept induction" names two different things in this repo, and conflating them
wasted a diagnostic pass during the investigation that produced this spec.

- **`orion/spark/concept_induction/`** -- a per-subject profile generator driven by
  regex/keyword extraction over bus envelopes. `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false`
  since 2026-07-12 (`b66f1e0a6`, reason: "weak, pronoun-diluted extraction signal").
  **Deliberately off. Stays off. Out of scope for this spec.** Note its
  `settings.py:57` still defaults to `True`, so reading the code suggests it is live.
- **topic-foundry -> substrate concept graph -> Concept Atlas** -- HDBSCAN topic
  modelling over the chat corpus, materialized as `ConceptNodeV1` records in
  FalkorDB (`orion_substrate`), rendered at `/concept-atlas`. **This is the real,
  intended concept induction path.** It is what this spec rebuilds.

The two share a name, a substrate node kind, and an adapter-shaped module layout
(`orion/substrate/adapters/concept_induction.py` vs `.../topic_foundry.py`), which is
why "concept induction is off" reads as "the concept graph is off" when it is not.
This spec does not rename either module -- renaming a live service to fix a
documentation problem is not a seam. It does require every doc, log line, and PR
report produced under it to say which one it means.

## Arsonist summary

The Concept Atlas renders 24 nodes, 15 edges and 12 connected components. Four of
those nodes are hand-authored golden seeds (Orion, Juniper, Claude, the
relationship) that connect only to each other. Ten are substrate telemetry nodes at
degree 0.0, each its own component. The remaining ten are topics and evidence from
two throwaway 62-segment runs on 2026-08-19.

Every layer above the producer is built and correct: decay, god-node ranking,
typed-relation classification, component analysis, an honest-label plumbing path, a
parity harness. All of it sits on a producer that is fed 254 rows, segments them
wrongly, never enriches them into anything usable, and reaches the graph via a
scheduler that has never ticked.

The headline defect is not any single bug. It is that the pipeline's only path from
"Juniper and Orion talked" to "these two concepts are connected" runs through LLM
entity extraction over text -- attempting to *infer* a fact that is already a
foreign key on every row. Measured ceiling of that approach on this corpus: 28%.
Measured actual: 0%.

## Current architecture

```text
chat_history_log (254 rows)
  -> topic_foundry dataset (30d window, dead aitown filter)
  -> windowing (block_mode=turn_pairs)          <-- BUG: pairs whole exchanges
  -> embeddings (vector-host:8320)
  -> HDBSCAN (min_cluster_size=8)               -> 3 clusters, 26% outliers
  -> topic_foundry_segments (62 rows)
  -> [enrichment]                               <-- produces prose, not structure
  -> [kg_edges: mentions]                       <-- 0 rows, ever
  -> Hub scheduler (sleep 86400 FIRST)          <-- never ticks
  -> map_topic_foundry_run_to_substrate()
  -> FalkorSubstrateStore (orion_substrate)
  -> /api/substrate/concepts/network
  -> concept-atlas.js                           <-- hides 19/24 labels
```

Live state, 2026-08-28:

| fact | value |
|---|---|
| concept nodes in `orion_substrate` | 18 (4 seeds, 10 substrate telemetry, 4 topics) |
| edges | 15 (8 seed<->seed, 7 within one topic component) |
| connected components | 12 |
| organic edges touching a golden seed | **0** |
| `topic_foundry_edges` rows, all time | **0** |
| `orion_substrate_aitown` nodes | 0 |
| newest ingested run | 2026-08-19 (newest available: 2026-08-24) |

### Evidence: the corpus

```sql
SELECT count(*) FROM chat_history_log;                    -- 254 (oldest 2026-07-29)
SELECT count(*) FROM chat_history_log
  WHERE created_at > now() - interval '30 days';           -- 228
```

254 rows is everything that survives the 2026-07-23 Postgres disk loss. Juniper has
scoped this work to `chat_history_log` only -- `journal_entries` (40,572 rows) is
explicitly **not** in scope.

### Evidence: participation is known, not inferrable

```text
                speaker in        name appears in
Orion           254/254 (100%)    72  (28%)
Juniper         254/254 (100%)    67  (26%)
Claude          1/254             20  (8%)
```

Every row is a Juniper prompt plus an Orion response. `user_id` is NULL on all 254
rows; the speaker is carried by *which column the text is in*, not by a field.
`source` is `hub_orion`/`hub_ws`/`hub_http` -- there is no third-party speaker lane.

The mention-edge design (2026-08-20 landmark-connection spec) recovers this via NER
over segment text. Its best possible recall is the "name appears in" column. Its
actual recall is zero, because the chain that feeds it produces nothing (below).

Claude is the exception that justifies keeping the mention path: in this corpus
Claude is a *subject of conversation*, not a participant. A mention edge is the
correct semantics for Claude and the wrong one for Orion and Juniper.

### Evidence: windowing destroys the speaker and fabricates roles

Live run spec (`topic_foundry_runs.specs`, run `02126502`):

```json
"windowing": {"block_mode": "turn_pairs", "include_roles": ["user", "assistant"], ...}
```

`_make_block_text` (`services/orion-topic-foundry/app/services/windowing.py:117`)
under `turn_pairs` emits, per document:

```text
User: <exchange N: Juniper prompt \n Orion response>
Assistant: <exchange N+1: Juniper prompt \n Orion response>
```

It pairs *consecutive rows* and labels one "user" and one "assistant", but each row
already contains both speakers. So the atomic document is two full exchanges / four
utterances, carrying two false role labels that are embedded into the vectorized
text.

`_role_of` (`windowing.py:110`) reads `row.get("role") or row.get("speaker")`.
Neither column exists on `chat_history_log`, so both are `None`, so the guard at
`windowing.py:71` (`if spec.include_roles and role_first and role_second`)
short-circuits and the `include_roles` filter never runs. It has never filtered
anything.

### Evidence: enrichment cannot produce entities

`services/orion-topic-foundry/app/services/enrichment.py:212`:

```python
"Enrich this segment. Provide JSON with keys: title, aspects, aspect_scores, sentiment, meaning, evidence_spans."
```

The prompt names `meaning` and never states its shape. The LLM returns prose:

```sql
SELECT jsonb_typeof(meaning), count(*) FROM topic_foundry_segments
  WHERE meaning IS NOT NULL AND meaning::text <> 'null' GROUP BY 1;
-- string | 271
-- object |   2
```

`_finalize_enrichment` (`enrichment.py:227`) only `setdefault`s `meaning`, so a
string passes through unvalidated. Then `kg_edges._edges_from_segment`
(`services/orion-topic-foundry/app/services/kg_edges.py:47`) does
`json.loads(<prose>)`, raises `JSONDecodeError`, swallows it, and assigns
`meaning = {}`. Zero edges, silently, on every enriched run.

The two `object` rows come from `_heuristic_enrich` (`enrichment.py:179`), which
hardcodes `"entities": []`. **No code path in the service can emit a non-empty
entity list.** Zero mention edges is structurally guaranteed, not a data accident.

The live run spec also carries `"enable_enrichment": false`.

### Evidence: the scheduler has never ticked

`services/orion-hub/scripts/main.py:767`:

```python
while True:
    await asyncio.sleep(topic_foundry_interval_sec)   # 86400, BEFORE any work
```

Hub must stay up 24 unbroken hours to tick once. Zero `substrate_topic_foundry_scheduler_*_tick`
lines in the current container's log; the graph holds 2026-08-19 topics while
2026-08-24 runs sit uningested. `orion_substrate_aitown` is 0 nodes despite
`SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED=true`.

### Evidence: the atlas hides most of what it has

`services/orion-hub/static/js/concept-atlas.js:322`:

```js
label: (ele) => showAllLabels || !hasGodNodes || ele.data("godNode") ? ele.data("label") : "",
```

5 god nodes of 24 -> 19 nodes render with an empty label. The `!hasGodNodes`
fallback never fires because the 4 canonical seeds are god nodes unconditionally
(`concept_atlas_routes.py:1105`, canonical bypasses degree ranking). The declutter
is gated on god-node *presence* rather than node count, so on a sparse graph it is
pure loss.

Underneath that, 16 of 24 nodes have no human label to show: 10 `node:substrate.*`
telemetry nodes whose label is literally `"substrate:" + node_id`, and 6
`sub-evidence-topicfoundry-*` nodes labelled by node id. The substrate nodes are
also not flagged `synthetic_label` (that check matches only topic-foundry origin
plus a `topic_` prefix), so they paint as ordinary named concepts.

## Decisions taken (and the ones rejected)

### D1. Split prompt and response into separate documents. Rejected: keep fused; rejected: build both.

Juniper proposed running both a split (utterance) model and a fused (exchange)
model as two levels of detail. Rejected, for four reasons:

1. **The hierarchy is inverted.** What runs today (`turn_pairs`) is *two exchanges*
   per doc. Fused (`block_mode=rows`) is one exchange -- finer than live, not a
   coarser overview. Both candidates sit near the bottom of the scale.
2. **Fused is a mixture, not a scale.** Coarsening means more of the same kind of
   thing. Fusing prompt and response averages two different speech acts into one
   vector. When the answer is on-topic it adds nothing split does not give; when it
   diverges the embedding represents neither, and nothing downstream can tell the
   two cases apart. A level of detail needs consistent per-document semantics.
3. **n=254 cannot support two models.** Live today: 62 segments -> 3 clusters, 26%
   outliers. Split reaches ~508 documents, the first time this corpus has a shot at
   stable density. Multi-scale topic modelling is for corpora where structure has
   been *demonstrated* at more than one scale; structure is not yet established here
   at one. A second scale now would be a level that names a distinction the data
   cannot support -- a keyword cathedral by CLAUDE.md 0A.
4. **Two models share one graph and degrade it.** Node ids namespace per run
   (`sub-concept-topicfoundry-<run_id>-<topic_id>`), so two lineages write two
   disjoint topic clouds into `orion_substrate` with no edges between them:
   component count doubles and the 3 seeds become the only bridge. Every downstream
   consumer iterates the store generically -- decay, god-node ranking, and the
   `_RELATION_CLASSIFICATION_PAIR_CAP` LLM budget have no concept of "scale", so
   coarse and fine topics compete for the same slots. PR #1719/#1721 already
   established the precedent that two corpora in one graph warrants physically
   separate stores.

**When multi-scale is genuinely wanted**, it comes from one fit: HDBSCAN builds a
condensed cluster tree during training, and `cluster_selection_method` (`eom`
coarse / `leaf` fine) plus `cluster_selection_epsilon` yield coarse and fine cuts
from the same embedding pass. `topic_foundry_hdbscan_cluster_selection_method` is
already a setting. That is the follow-on, not a second corpus.

The legitimate part of the fused instinct -- that a question and its answer form one
unit of meaning -- is better served by conversation-level grouping via the existing
`segmentation_mode: time_gap` on top of split utterances. Same model, real
hierarchy, no second corpus. Deferred, not rejected.

### D2. Participation edges from provenance, not from NER.

Every segment already carries `provenance.row_ids` -> `chat_history_log.correlation_id`.
Topic -> segments -> rows -> speakers is a foreign-key join: no LLM, no enrichment
dependency, 100% recall, and it works on `stage=trained` runs.

### D3. Mention edges survive for Claude only.

Orion and Juniper get participation edges and their mention path is retired
outright -- not left as a fallback. CLAUDE.md 0A: "kill means kill, no fallback to
the thing being killed." Claude keeps mention edges because in this corpus Claude
genuinely is a mentioned subject rather than a speaker.

### D4. Model lineage: accept the orphaning, do not paper over it.

D1 changes the model spec, which changes `_topic_foundry_model_spec_fingerprint()`,
which mints a new model name and orphans all prior runs. There are already 6 models
across 10 runs from exactly this churn. Rejected: pinning the fingerprint to keep
continuity -- prior runs were trained on mis-segmented documents and are not a
baseline worth preserving. The old runs stay queryable in Postgres; nothing is
deleted. Accepted cost: the atlas starts from an empty topic set on first ingest
after this lands.

## Proposed changes

### Schema / bus / API

No new bus channels. No new substrate node kinds. Two additive changes:

1. `map_topic_foundry_run_to_substrate()` gains `segment_speaker_map`
   (`segment_id -> [speaker]`) and `speaker_concept_ids` (`speaker -> node_id`),
   parallel to the existing `landmark_concept_ids`. Emits `associated_with` edges
   from a topic concept to each golden seed that spoke in it, weighted by the count
   of contributing segments. `None` for both is a complete no-op -- every existing
   caller is unchanged.
2. `EnrichmentMeaning` pydantic model in `services/orion-topic-foundry/app/models.py`
   with `intent`, `outcome`, `questions`, `claims`, `next_steps`, `entities`.
   `_finalize_enrichment` coerces to it; a string `meaning` is preserved verbatim
   under a new `summary` field rather than discarded.

### Env/config

| key | change | service |
|---|---|---|
| `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_RUN_AT_STARTUP` | new, default `true` | orion-hub |
| `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_INTERVAL_SEC` | unchanged (86400) | orion-hub |

**Revised during build (branch 1).** This table originally proposed
`TOPIC_FOUNDRY_BLOCK_MODE` and `TOPIC_FOUNDRY_SPLIT_TEXT_COLUMNS` as
topic-foundry env keys. Dropped: a model's `windowing_spec` is *frozen into its
model row at creation* and the Hub sends that payload, so a topic-foundry-side
env key would be read only at model-creation time and then silently ignored --
an operator knob that looks live and is not. Windowing is a design decision, not
an operator knob; it lives in `_topic_foundry_windowing_spec()` in
`concept_atlas_routes.py` and is now folded into the model-name fingerprint, so
changing it always mints a new model instead of drifting. Zero new env keys for
branch 1, so no `.env_example` change and no sync needed.

Any `.env_example` touched by a later branch gets
`python scripts/sync_local_env_from_example.py` run in the same session
(CLAUDE.md 7).

### Files likely to touch

| path | why |
|---|---|
| `services/orion-topic-foundry/app/services/windowing.py` | split columns; `rows` default; drop fabricated role prefixes; fix or remove inert `_role_of` |
| `services/orion-topic-foundry/app/services/enrichment.py` | specify `meaning` shape in prompt; validate in `_finalize_enrichment` |
| `services/orion-topic-foundry/app/services/kg_edges.py` | stop swallowing `JSONDecodeError`; log and count |
| `services/orion-topic-foundry/app/models.py` | `EnrichmentMeaning` |
| `orion/substrate/adapters/topic_foundry.py` | speaker participation edges |
| `services/orion-hub/scripts/concept_atlas_routes.py` | build speaker map from provenance; pass to adapter |
| `services/orion-hub/scripts/topic_foundry_client.py` | fetch segment provenance |
| `services/orion-hub/scripts/main.py` | scheduler ticks at startup |
| `services/orion-hub/static/js/concept-atlas.js` | label gating on node count |
| `orion/substrate/*` (telemetry node producer) | stop emitting `"substrate:" + node_id` as a label |
| `topic_foundry_datasets` (data) | drop the dead `where_sql` |

## Non-goals

- Re-enabling `orion/spark/concept_induction` (the regex extractor). Stays off.
- Adding `journal_entries` or any corpus other than `chat_history_log`.
- A second topic model at a second granularity (see D1).
- Renaming either `concept_induction` module.
- Persisting `orion_substrate_aitown` (tracked separately; falkor-only by design today).
- Retuning `min_cluster_size` before measuring what ~508 documents actually cluster into.

## Acceptance checks

Runtime truth, not config truth (CLAUDE.md 0A). Each check names the live evidence
that closes it.

| # | check | evidence |
|---|---|---|
| A1 | a training run produces >= 2x the current document count | `topic_foundry_segments` count for the new run vs 62 |
| A2 | no document text contains a fabricated `User:`/`Assistant:` prefix | grep the run's `documents.jsonl` |
| A3 | `topic_foundry_edges` is non-empty after an enriched run | `SELECT count(*)` > 0 |
| A4 | `jsonb_typeof(meaning) = 'object'` for every newly enriched segment | SQL group-by |
| A5 | every non-outlier topic concept has >= 1 participation edge to a golden seed | Cypher on `orion_substrate` |
| A6 | the Orion and Juniper seed nodes each have degree > 0 from organic topics | Cypher; today both are 2.0, seed<->seed only |
| A7 | connected components drop below the current 12 | `/api/substrate/concepts/network` `component_count` |
| A8 | the scheduler logs an ingest tick within 60s of Hub start | `substrate_topic_foundry_scheduler_ingest_tick` in hub logs |
| A9 | no node in the network payload renders an empty label at default settings | payload + a UI check |
| A10 | no node label matches `^substrate:` | network payload scan |

A5/A6 are the ones that actually answer the original complaint. A3/A4 gate the
Claude mention path only.

## Metric quality gate (CLAUDE.md 0A)

The participation edge is a new signal entering a cognition-adjacent graph, so it
runs the gate before anything builds on it.

1. **Provenance.** `topic_foundry_segments.provenance.row_ids` (written by
   `windowing.py` `RowBlock.row_ids`), joined to `chat_history_log.correlation_id`
   (the dataset's declared `id_column`). Verified live: a real segment's provenance
   contains two row_ids resolving to two real rows.
2. **Independence.** Not independent of the mention edge for Claude -- both assert a
   topic/entity relation. It *is* independent of everything else in the graph:
   co-occurrence edges are text-derived, participation edges are metadata-derived.
   The Orion/Juniper mention edges are being retired precisely so the two are not
   both live for the same subject (D3).
3. **Theory anchor.** Not a proxy or a correlate: the row *is* the utterance, and
   the column *is* the speaker. This is a recorded fact, not a measurement.
4. **Live-data sanity.** Cannot be degenerate-at-zero the way an EWMA can, but it
   *can* be degenerate-at-saturation: if every topic links to both Orion and
   Juniper, the edge carries no discriminating information and the graph hairballs.
   This is the real risk and A7 is the check for it. If post-split every topic still
   links to both speakers uniformly, the edge is honest but uninformative, and the
   follow-on is weighting by segment share rather than deleting the edge.
5. **Existing mechanism.** Searched: `landmark_concept_ids` /
   `_GOLDEN_SUBJECT_ANCHOR_NODE_IDS` is the existing mechanism. This replaces its
   input rather than adding a parallel one.
6. **Reversibility.** Cheap. The edges are additive `associated_with` records with
   a distinct provenance `source_kind`; deleting them is one Cypher statement and
   the adapter params default to `None`.

## Live results (updated as branches land)

### Branch 1, verified 2026-08-28 on run `f9443362`

| check | result |
|---|---|
| A1 documents >= 2x | **394** vs 62 -- 6.4x |
| A2 no role labels in text | **0 / 394** snippets match `^(User|Assistant|juniper|orion):` |
| speakers recorded | **394 / 394** (220 orion, 174 juniper) |
| topics | **19** real clusters, up from 3 |

Contamination check (the review finding that the *true* speaker name in the
embedded text is as harmful as a fabricated one): 17 of 19 topics are
speaker-MIXED, 50-77% purity against ~56% chance for the 174/220 split. Only 2
are speaker-pure, both plausibly genuine Orion-only content. Had the label
still been in the text, purity would sit near 100% across the board.

### Branch 2, verified 2026-08-28 against the same run

36 participation edges: Orion -> 19 topics, Juniper -> 17.

**Metric quality gate item 4 (saturation) resolved: NOT saturated.** The
concern was that if every topic linked to both speakers uniformly, the edge
would be honest but carry no information. Measured shares span 0.00-1.00 and
track something real:

| topic | n | orion | juniper |
|---|---|---|---|
| General Greetings and Conversations | 30 | 0.23 | 0.77 |
| Athena's numeric signal | 14 | 1.00 | 0.00 |
| Conceptual Repair Pressure | 13 | 1.00 | 0.00 |
| Model Testing & Development | 13 | 0.31 | 0.69 |
| Code Review Process | 16 | 0.50 | 0.50 |

So the follow-on contemplated in the gate (weight by segment share rather than
drop the edge) is what shipped, and it was the right call.

### Branch 3, 2026-08-28

Scheduler now ticks ~30s after Hub start instead of sleeping a full 86400s
interval first (`SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_RUN_AT_STARTUP`, default
on). A1/A2/A5/A6 above were all verified by triggering runs BY HAND -- this is
what makes them recur.

Label fixes: the atlas declutter is gated on node count, not on the mere
existence of a god node (canonical seeds are god nodes unconditionally, so the
old check was always true and a 24-node graph hid 19 labels); substrate
prediction-error nodes get a real label (`node:substrate.harness_closure` ->
"Harness closure prediction error" rather than
`substrate:node:substrate.harness_closure`); and evidence nodes are named after
the concept they support instead of falling back to their raw node_id.

## New finding: the Hub's substrate store is a process-local cache

Found while verifying branch 1. `SUBSTRATE_SEMANTIC_STORE` is built once at
module import, and the served `/api/substrate/concepts/network` reflects only
writes made **in the Hub's own process**.

Evidence: an ingest run via `docker exec python3` wrote 19 concepts and 148
edges to FalkorDB (verified independently: `orion_substrate` went 18 -> 37
concept nodes), while the Hub endpoint kept serving the pre-ingest 24 nodes /
15 edges with `truncated: false, degraded: false`. Re-running the identical
ingest through the Hub's own HTTP route immediately returned 62 / 163.

Impact: any writer other than the Hub process -- an out-of-band script, a
second Hub replica, another service -- is invisible to the atlas until Hub
restarts, and the endpoint reports itself healthy while stale. The scheduler
runs in-process so the normal path is unaffected. Not fixed in this spec's
branches; needs its own patch (either a TTL/refresh on the read path, or an
explicit "this view is process-local" honesty field on the response).

## Missing questions

1. After the split, does every topic link to both Orion and Juniper (the
   saturation case in gate item 4)? Not answerable until a real run exists. If yes,
   weight by segment share -- do not drop the edge.
2. Is `min_cluster_size=8` right at ~508 documents? Deliberately unanswered until
   A1 lands; retuning before measuring would be guessing.
3. Does the 6000-char `max_chars` truncation bite differently on single utterances
   than on 2-exchange blobs? Orion's responses are long; a truncation that was
   harmless on a fused blob may now cut a whole utterance.

## Build order

Four branches off this spec, each independently verifiable.

1. **Windowing** -- split columns, `rows` default, drop fabricated prefixes, resolve
   `_role_of`. Gates A1, A2. Triggers the D4 model rename.
2. **Participation edges** -- provenance -> speaker map -> adapter -> store; retire
   the Orion/Juniper mention path. Gates A5, A6, A7.
3. **Freshness + atlas readability** -- scheduler startup tick; label gating;
   `substrate:` labels; evidence-node labels. Gates A8, A9, A10.
4. **Enrichment contract** -- `meaning` shape in prompt and validation; stop
   swallowing the decode error. Gates A3, A4. Last, because only the Claude mention
   path (8% of the corpus) depends on it.

## Recommended next patch

Branch 1. It is the only one that changes what the model sees, every other branch
is either presentation or a consumer of its output, and it forces the D4 model
rename that the rest should land on top of rather than under.

## Branch 4 result, 2026-08-29: the contract defect was bigger than this spec said

Branch 4 was scoped here as "last, because only the Claude mention path (8% of
the corpus) depends on it." That was wrong, and the reason is worth recording.

A live re-check of all ten acceptance checks before starting branch 4 found that
`GET /segments` was returning **HTTP 500** -- `pydantic_core.ValidationError: 2
validation errors for SegmentRecord ... dict_type` -- for every run containing a
segment whose `meaning`/`sentiment` were prose strings. Measured: 552 of 554
enriched rows, both columns.

That endpoint is not a Claude-only concern. It is the sole source of
`segment_speakers`, and `concept_atlas_routes` degrades a segments-fetch failure
to an empty map. So a live ingest run on 2026-08-29 returned:

```json
{"available": true, "run_id": "f9443362-...", "concepts_written": 19,
 "segments_fetched": 0, "segments_with_speakers": 0, "participation_edges": 0}
```

Branch 2's participation edges -- the mechanism that resolved the original
complaint -- were **dead on the live path**, with every test green, an
`available: true` response, and 19 concepts written. The 36 edges observed
during branch 2's own verification were real; they had simply stopped being
reproducible by the time the enrichment column filled up with prose. This is the
"no empty-shell cognition" failure mode described in CLAUDE.md 0A, arriving
through a schema-valid success response.

The `SegmentRecord` model declared `meaning: Optional[Dict[str, Any]]` all along.
Nothing enforced it: `_llm_prompt` named the six keys but never their shapes, so
the model answered with prose, and `_finalize_enrichment`'s `setdefault` is a
no-op for a key that is present-but-wrong-typed.

Three further consequences, all silent:

- `kg_edges._edges_from_segment` caught the resulting `JSONDecodeError` and
  substituted `{}`. That is the whole explanation for `topic_foundry_edges`
  holding 0 rows for all time.
- `repository.fetch_segments`'s `friction`/`valence` sort keys are
  `(sentiment->>'friction')::float`, which is NULL for a JSON string, so both
  sorts silently collapsed to a constant 0.
- Topic Studio and every other reader of those two columns saw a string where a
  dict was declared.

### What shipped

`app/services/enrichment_contract.py` -- the declared shape as constants, plus
`coerce_meaning`/`coerce_sentiment`. The prompt's shape block is **generated from
the same constants the coercion validates against**, so the instruction given to
the model and the shape the database accepts cannot drift apart. That is the
seam; the coercion is what makes the 552 already-written rows readable without
paying to re-enrich them.

Prose is preserved under `summary` with `unstructured: true` rather than
discarded or dressed up as structure. A non-numeric sentiment scalar is dropped
rather than defaulted to `0.0`, because a fabricated 0.0 valence is
indistinguishable from a measured neutral one.

Read side is enforced by `SegmentRecord` field validators rather than a
router-side fixup: `routers/segments.py` has two independent construction sites
and a miss in either is a 500 on the endpoint the participation edges depend on.

### Live results

| check | before | after |
|---|---|---|
| `GET /segments` | HTTP 500 | HTTP 200, 394/394 with speakers |
| ingest `segments_fetched` | 0 | 375 |
| ingest `segments_with_speakers` | 0 | 255 |
| ingest `participation_edges` | 0 | 34 |
| ingest `edges_written` | 19 | 170 |
| ingest `segment_topic_map_buckets` | 0 | 19 |
| **A3** `topic_foundry_edges` | **0 rows, all time** | **50 rows (14 `mentions`)** |
| **A4** newly enriched `jsonb_typeof` | 552 string / 2 object | **5/5 object, both columns** |

A live 5-segment re-enrichment through the fixed prompt produced populated
`entities` (`Athena`, `Circe`, `10G pipe`, `concept region`) and full numeric
`sentiment` on every row -- the entity path that had never once produced a row.

### Acceptance checks still open after branch 4

- **A7** (`component_count` below 12) is unreachable as written. Live decomposition
  is one 42-node main component (seeds + topics + evidence -- the win), one
  10-node orphan component from the stale 2026-08-19 run that nothing retires,
  and ten singleton substrate-telemetry concept nodes that connect to nothing by
  design. The check needs replacing with one that names those two groups, not
  satisfying.
- **A10** (no `^substrate:` label) fails on 3 nodes, but the branch-3 fix is
  working: 7 of the 10 telemetry nodes now carry real labels. The 3 that do not
  (`node:substrate.harness_closure`, `.transport`, `.chat`) are rows written to
  Falkor before that fix and never rewritten. This needs a backfill, not a code
  change. Separately, `node:substrate.transport` is `transport_prediction_error`,
  which CLAUDE.md 0A records as retired outright -- it is still holding a node.

### Follow-ups found while verifying, not fixed here

1. **The tick cannot land what it triggers.** `main.py` runs trigger -> enrich ->
   ingest back-to-back in one tick body (`:815`, `:845`, `:862`); training takes
   ~3 minutes and both endpoints are `BackgroundTasks`. At an 86400s interval the
   graph is always at least a day behind the newest run. Confirmed: the 2026-08-28
   startup tick queued run `68f0a3d1` (394 segments), which sat un-ingested until
   a manual call.
2. **Runs wedge with no watchdog.** `68f0a3d1` and `9160b074` sat in
   `running/enriching` for 20+ hours with nothing to time them out or retry.
3. **A failed training run is not retried.** Both 2026-08-29 02:34 runs failed on a
   transient `orion-athena-vector-host` DNS resolution error (the host was healthy
   again 20 minutes later). Nothing retried; at a 24h interval one DNS blip costs
   a day.
4. **`enriched_count` in the enrich response is stale.** `enrich_run_endpoint`
   reads `stats["segments_enriched"]` from the run row *before* queueing the
   background task, so the scheduler logs a number that predates the work it just
   asked for.
5. `topic-foundry` has no `evals/` directory. Coverage for this branch is gate
   tests plus the live verification above.
