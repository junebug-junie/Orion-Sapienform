## Summary
> Superseded 2026-10-14: the raw 0.65 floor and `run_reading_claim_objects_replay.py` are retired; see `2026-10-14-reading-link-targets-centered-pr.md` (centered floor, `run_reading_link_targets_replay.py`).

- When Orion finishes reading a page, Stage 2 may link the read's new concepts to concepts Orion already had. Until now it was only offered an existing concept when that concept's exact name appeared in the page. Atlas names are chat-topic phrases ("Health Crisis Burden") and earlier readings' concept names, so that almost never happened.
- Stage 2 now also offers the existing concepts closest in meaning to each new concept (embedding search over the whole concept atlas), up to the same 24-object cap, with exact-name hits always first.
- The rule that decides whether a link is actually made is unchanged: the cited quote must be verbatim in the retained text AND name the object. A claim to a meaning-only object is therefore recorded as `proposed` for review, not linked. (The single exception follows from the same rule: a concept ranked below the label path's 500-concept scan whose name IS in the text arrives via meaning search, and a quote naming it is linked exactly as a label hit would be. No such concept exists on the replayed reads.)
- The live replay shows what this buys and what it does not: 2 of 7 reads had something to link to before, 6 of 7 now; accepted links stay at 3 (proven unchanged by construction, see below).

## Outcome moved

Reads offered at least one existing concept (live replay, last 7 done reads with retained text): **2/7 -> 6/7**. The seventh has no stored concepts of its own (a 404 page), so nothing can be linked from it.

Accepted links possible under the quote rule: **3 -> 3**. Not a regression, a property: an accepted object's label must be inside a verbatim quote, so it is inside the retained text, so the exact-name path already offered it, unless it ranked below that path's 500-concept scan. On the replay, a full-atlas exact scan finds the same objects as the 500 scan on every read, so meaning search adds only `proposed` claims here.

## Current architecture

`orion/world_pulse_read/assertions.py` `object_candidates`: Recall's region-rank (top 500 concepts by salience) filtered to labels that occur in `what_i_learned` + retained text, cap 24. `plan_claims` accepts to `provisional` only when the quote (4+ words) is verbatim in a retained text, contains the object's stored label, and the predicate's domain rule holds; otherwise `proposed`. Hub's `AssertionProjector` writes links for accepted decisions.

Live 2026-10-13, all four reads that day logged `reading_claim_context ... objects=0 usable=False`. Replay shows why: three were 404 pages (215-char tool digest), one a 202-char digest of a JS-rendered page.

## Architecture touched

- `orion/world_pulse_read/concept_search.py` (new): index + query + re-gate.
- Stage 2 pipeline: a background index loop and a meaning pass after the label pass in `_build_claim_context`.
- Existing machinery reused, nothing new stood up: vector-host `/embedding` (bge-large-en-v1.5), Chroma, `orion/introspect/semantic_index.py` (`embed`, `nearest`, `index_docs`), bus channel `orion:vector:semantic:upsert` (orion-hub already a registered producer) -> orion-vector-writer.
- New Chroma collection `orion_atlas_concepts` (an index, never the record: every hit is re-read from the store and re-gated).

## Existing-mechanism check

- `ConceptNodeV1.embedding_ref` / `metadata["concept_embedding"]` (reconcile.py): only topic-foundry sets it in-flight, and it is never persisted to Falkor (0 of 2,823 concept/entity nodes carry it). Not usable for retrieval.
- Recall `concept_region` collector: label substring only, explicitly no embeddings. That is the path being widened.
- Chroma collections in use: dreams, chat, curiosity, reading results, Orion's Day, memory crystallizations; none holds atlas concepts. The introspect `semantic_index` plumbing is the house pattern for "embed once, upsert via vector-writer, re-gate every hit"; this reuses it verbatim (same as the Orion's Day collection, which also reuses another tool's Chroma/embedder URLs).

## Metric quality gate (cosine similarity deciding what Stage 2 sees)

1. **Provenance.** `semantic_index.embed` -> vector-host `POST /embedding` -> `BAAI/bge-large-en-v1.5`, 1024-d (verified live). Query similarity = Chroma distance converted by `semantic_index.similarity` (l2 on unit vectors: `1 - d/2`). Concept text = `concept_search.concept_doc`: label, or `label: definition`.
2. **Independence.** Not added to any model or aggregate. It only ranks objects that the label pass did not already offer (exact hits are excluded from the meaning pool), so it does not double-count the label signal.
3. **Theory anchor.** Dense retrieval: bge is contrastively trained so that cosine ranks texts by topical relatedness. That is the claim being made (related, not "states a relationship"); the relationship itself is still decided by the model plus the verbatim-quote rule.
4. **Live-data sanity.** 6,080 (subject, atlas concept) pairs from 6 reads, max over each read's subjects: median 0.531, p90 0.605, p95 0.630, p99 0.693, max 0.885. Not degenerate, and its rest point is ~0.53, not 0 (bge never rests near 0), so the floor is set against that rest point: 0.65 sits between p95 and p99. Effect per read: the two 404 reads get 2 and 3 offers (one is the sibling 404 concept at 0.885), the GPU read fills the cap (top 0.809), the NASA reads get 12 and 15 (0.65-0.69, mixed: "NASA Inspiration Tour" and "Freedom 250 F-5 flyover team" next to "Phase 1 proposal"). The 0.65-0.70 band is noisy; it is an offer list, and the quote rule gates links.
5. **Existing mechanism.** See above.
6. **Reversibility.** Cheap: a constant, a collection, one loop. Unset the two reading-search URLs (or `HUB_WORLD_PULSE_READ_ASSERTIONS_ENABLED=false`) and the path is gone; nothing is written to the graph or a schema.

## Replay (read-only, production Postgres + Falkor, vector-host; Chroma not touched)

`PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_claim_objects_replay.py`

| read | retained | subjects | by label (before) | by meaning (added) | acceptable before / after |
|---|---|---|---|---|---|
| 10-13 coreweave gpu-cpu-compute (404) | 215 | 0 | 0 | 0 | 0 / 0 |
| 10-13 coreweave architecture (404) | 215 | 1 | 0 | 2 (top 0.885 sibling 404 concept) | 0 / 0 |
| 10-13 coreweave infrastructure (404) | 215 | 1 | 0 | 3 | 0 / 0 |
| 10-13 coreweave gpu-compute | 202 | 2 | 0 | 24 (GPU/NVIDIA concepts, 0.735-0.809) | 0 / 0 |
| 10-11 nasa.gov/maxpower | 1,786 | 2 | 0 | 12 (0.650-0.691) | 0 / 0 |
| 10-11 NASA MetLife flyover | 2,863 | 4 | 1 | 15 (0.650-0.683) | 1 / 1 |
| 09-29 internal design doc | 5,225 | 6 | 2 | 22 (0.661-0.702, mostly chat-topic noise) | 2 / 2 |

"Acceptable" = offered objects whose stored label appears in the retained text, the necessary condition for any accepted link. It is an upper bound; it does not run the model.

## Recommendation on the acceptance rule

- **(a) strict, implemented.** Claims to meaning-only objects journal as `proposed`, no link. Replay: 0 additional accepted links (3 -> 3), by construction.
- **(b) label OR stored aliases / definition terms.** Measured live: 0 of 1,017 concept nodes carry aliases (only 7 of 1,806 entities do, and entities are not in the concept region), so the alias form accepts exactly what (a) accepts: 3 -> 3. A "definition terms" form has no principled shape: a definition is a sentence, not a name, so "names the object by a definition term" means word overlap, a hand-tuned list in disguise. Not recommended.
- **Recommended next step, not built:** keep (a), and let the `proposed` meaning claims this patch produces accumulate. Once Juniper has accepted/rejected a few dozen in review, that is a labelled set to calibrate a quote-to-object similarity rule (a score deciding acceptance needs its own metric gate on real reviewed claims, not borrowed constants).

## Latency on Stage 2's path

Measured live, read-only: one subject = one embed (77-107 ms) + one Chroma nearest(48) (12-79 ms, measured on `orion_reading_results`). Subjects are embedded concurrently (`asyncio.gather`), so a read costs roughly one round trip, ~100-190 ms, against a Stage 2 model turn of minutes. Logged per read as `related_ms=`. The index is built off the turn path: ~70 ms per concept, 200 per pass, about 2 minutes for the current 1,016 concepts, then only new or edited concepts.

## Files changed

- `orion/world_pulse_read/concept_search.py`: new. Index rows, index pass, per-read query, re-gate, merge under the cap.
- `orion/world_pulse_read/assertions.py`: `EndpointV1.via` ("label" | "meaning"); prompt lists meaning objects separately and says a claim to one is recorded for review; docstrings.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: `concept_search` parameter, index loop (start/stop), meaning pass in `_build_claim_context`, richer `reading_claim_context` log line.
- `services/orion-hub/scripts/main.py`: passes a `SearchConfig` built from the reading-search Chroma/embedder URLs.
- `services/orion-hub/tests/reading_assertion_fakes.py`: `FakeSemanticBackend` (vector-host + Chroma REST + vector-writer over `httpx.MockTransport`; the real `semantic_index` code runs against it).
- `services/orion-hub/tests/test_world_pulse_read_claim_related_objects.py`: new tests.
- `services/orion-hub/evals/test_reading_assertion_eval.py`: meaning batch (`meaning_offered`, `meaning_links`).
- `services/orion-hub/evals/run_reading_claim_objects_replay.py`: the live read-only replay above.
- `services/orion-hub/README.md`: docs.

## Schema / bus / API changes

- Added: Chroma collection `orion_atlas_concepts` (index only). `EndpointV1.via` (in-process dataclass, not a wire schema).
- Removed / Renamed: none.
- Behavior changed: Stage 2 prompt may list "objects related by meaning"; `reading_claim_context` log line gains `named= related= related_top= related_ms=`.
- Compatibility notes: bus channel and schema unchanged (`orion:vector:semantic:upsert`, `VectorUpsertV1`, orion-hub already a registered producer).

## Env/config changes

- Added / Removed / Renamed keys: none. Reuses `HUB_READING_SEARCH_CHROMA_URL`, `HUB_READING_SEARCH_EMBED_URL`, `HUB_READING_SEARCH_INDEX_INTERVAL_SEC` (both URLs already set live).
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest -q services/orion-hub/tests/test_reading_ingress.py services/orion-hub/tests/test_reading_turn_listener.py \
  services/orion-hub/tests/test_reading_postgres.py services/orion-hub/tests/test_world_pulse_read_*.py \
  services/orion-hub/tests/test_curiosity_introspect_listener.py orion/world_pulse_read/tests orion/introspect/tests \
  tests/test_world_pulse_read_*.py services/orion-hub/evals/test_reading_*_eval.py
-> 672 passed, 66 skipped (16 in the new test file)
Mutation check on concept_search.py (fence re-gate, own/offered exclusion, floor, index fence, cap): each mutation fails >= 1 test.
Static gates: check_settings_defaults orion-hub --example-drift OK; check_metric_lineage --gate PASS; --prompt-semantics PASS; check_definition_drift --gate PASS; check_scripts_dir_no_stdlib_shadow clean.
```

## Evals run

```text
pytest services/orion-hub/evals/test_reading_assertion_eval.py -s
  reading_assertion_eval cases=8 linked=2 unsupported_links=0 supported_recall=1.00 refused_writes=0
  reading_assertion_eval meaning_offered=1 meaning_links=0 proposed=2 reasons=quote_not_about_endpoints,quote_not_about_endpoints
run_reading_claim_objects_replay.py (live, read-only):
  pooled max-over-subjects similarity n=6080 p50=0.531 p90=0.605 p95=0.630 p99=0.693 max=0.885 floor=0.65
  totals reads=7 usable_before=2 usable_after=6 acceptable_before=3 acceptable_after=3
```

## Docker/build/smoke checks

```text
Not deployed from this branch (production deploys from the primary checkout on main).
Live path UNVERIFIED until Hub runs this code and the next feed batch (~12:00 UTC) produces reads:
expect `reading_concept_index indexed=200 pending=...` within a minute of start, then
`reading_claim_context ... related=N related_ms=...` on the next read.
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on d6d9aa6d8.

- Finding (SHOULD): docs said a meaning object can never be linked; false for labels under 3 characters (the quote check is a substring test, "ai" is in "said") and for concepts below the label path's rank-500 scan whose label is in the text.
  - Fix: meaning objects with labels under `MIN_LABEL_CHARS` are never offered; docs (module docstrings, README, this report) now state the rank-500 exception, which is correct behaviour under the unchanged rule.
  - Evidence: `test_a_label_too_short_to_name_anything_is_never_offered` (fails with the check removed), `test_a_low_ranked_concept_named_in_the_text_is_linked_by_the_same_quote_rule`.
- Finding (SHOULD): `related_objects` made synchronous store reads (Falkor round trips) on Hub's event loop.
  - Fix: subject-doc reads and re-gating run in `asyncio.to_thread`; only embed and Chroma calls stay on the loop.
  - Evidence: full reading suite green.
- Finding (SHOULD): index loop would re-embed the same batch every 5 s forever if vector-writer is down (upserts publish fine but never land).
  - Fix: hurry only while the backlog (indexed + pending) shrinks between passes; otherwise wait the full interval.
  - Evidence: `test_index_loop_backs_off_when_upserts_never_land`, `test_index_loop_hurries_only_while_the_backlog_drains` (fails with the old rule).
- Finding (SHOULD): test gaps (loop lifecycle, index-not-built path, `_record_claims` with a meaning id present and missing).
  - Fix: added `test_index_task_starts_only_with_search_and_assertions_and_stop_cancels_it`, `test_stage2_keeps_label_objects_before_the_index_exists` (fake Chroma now answers a never-written collection with Chroma 0.4.24's 500), `test_stage2_records_a_meaning_claim_as_proposed_and_refuses_one_no_longer_offered`.
- Finding (NIT): silent truncation past 5,000 concepts. Fix: `reading_concept_index_truncated` warning.
- Finding (NIT): docstrings said concept/entity; the region read is concept-only. Fix: reworded.
- Not fixed (NIT, follow-up): deleted/merged concepts stay in Chroma (re-gated away at query time, but they use nearest-neighbour slots; 48 per subject leaves headroom today); `related_ms` times embed + query + store reads together; the index pass reads full concept rows each pass.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: the 0.65-0.70 band is noisy (chat-topic labels like "orion / skye / message" on the internal-doc read). Mitigation: offers only; the quote rule gates links; the prompt says meaning objects are recorded for review.
- Severity: low. Concern: more `proposed` journal rows that nobody reviews. Mitigation: they carry no link and are the calibration set for any future relaxation.
- Severity: low. Concern: a Hub restart mid-turn rebuilds the context; a meaning-only object that no longer ranks is refused `unknown_object` (no proposal), where it would have been `proposed`. No link is lost either way.
- Severity: info. Concern: 3 of the 4 reads on 10-13 were 404 pages; the empty object lists that day were mostly correct. The upstream issue (world_pulse seeding dead CoreWeave URLs) is out of scope here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2627

🤖 Generated with [Claude Code](https://claude.com/claude-code)
