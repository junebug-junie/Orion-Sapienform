## Summary

Implements the remaining "Readability" items from `docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md` (the other two items in that section — `nodeDimensionsIncludeLabels`/`componentSpacing` collision handling — shipped separately in PR #1721).

- `concept_atlas_network()` computes connected-component ids via a plain union-find over the already-filtered node/edge lists, and reports `component_count` in the response envelope.
- `concept_atlas_network()` now surfaces `topic_id` (topic-foundry's HDBSCAN cluster assignment) — previously written into `ConceptNodeV1.metadata` by the ingest adapter but discarded before reaching the API response.
- `concept-atlas.js` colors non-god nodes by `topic_id` via a deterministic hash-to-hue function; god-node purple stays the priority signal.
- `concept-atlas.js` defaults node labels to god-nodes-only (dense graphs stacked every label illegibly); a "Show all labels" checkbox opts back into always-on labels.
- **Code review found and this branch fixed a real durability bug**: `topic_id` was never covered by the FalkorDB codec's metadata-to-Cypher-property translation, so on the real default backend it silently reverted to `None` within ~30 seconds of any write — see "Review findings fixed" below.

## Outcome moved

The Concept Atlas UI (`services/orion-hub`'s `/concept-atlas` tab) now gives real visual grouping for disconnected subgraphs and real cluster-based coloring, on the actual production backend (not just the in-memory test store) — the topic_id durability fix was necessary for the coloring feature to survive anything past the first render.

## Current architecture

`concept_atlas_network()` (`services/orion-hub/scripts/concept_atlas_routes.py`) serves up to 300 nodes / 600 edges from the shared substrate store to a Cytoscape.js-rendered graph (`concept-atlas.js`). Node dicts already carried `god_node: bool` (top-degree ranking, computed fresh per request); there was no grouping signal for disconnected components and no cluster/community signal at all, even though topic-foundry's HDBSCAN cluster id (`topic_id`) was already being written into each node's `metadata` dict by the ingest adapter (`orion/substrate/adapters/topic_foundry.py`) — just never read back out by this route.

## Architecture touched

- `services/orion-hub` (routes, static JS, template, tests) — the Concept Atlas feature itself.
- `orion/substrate` (`falkor_codec.py` + a new test file) — a genuine contract fix: `ConceptNodeV1.metadata["topic_id"]` now durably round-trips through FalkorDB, not just orion-hub's read path.

## Files changed

- `services/orion-hub/scripts/concept_atlas_routes.py`: new `_compute_connected_components()` helper (union-find); `component_id` + `topic_id` added to each node dict; `component_count` added to the response envelope (including the two `_unavailable(...)` degraded paths, for envelope-shape parity).
- `services/orion-hub/static/js/concept-atlas.js`: `topicColor()` hash-to-hue function; god-nodes-only default label visibility with a "Show all labels" checkbox toggle (falls back to showing all labels when the current subgraph has zero god nodes); `component_id`/`topic_id` passed through `graphToElements()` and shown in the node inspector; status line reports a component count.
- `services/orion-hub/templates/concept_atlas.html`: "Show all labels" checkbox; legend entries for cluster coloring; explanatory hint text.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: `_concept_node()` gained an optional `metadata` kwarg; two new tests (`test_network_connected_components_grouped_and_counted`, `test_network_passes_through_topic_foundry_cluster_id`); the empty-store degrade test now also asserts `component_count == 0`.
- `orion/substrate/falkor_codec.py`: new `TOPIC_FOUNDRY_METADATA_KEYS` allowlist, `_topic_foundry_properties_from_metadata()` (encode), `_topic_foundry_metadata_from_row()` (decode) — wired into `encode_node_properties()` and `decode_concept_node()`.
- `orion/substrate/tests/test_falkor_codec.py`: updated the pre-existing exact-dict-equality encode test for the new `topic_id: None` field.
- `orion/substrate/tests/test_falkor_codec_topic_id.py` (new): allowlist membership, encode/decode unit tests, and a full encode→decode round-trip test mirroring the existing `perception_staleness` regression-test precedent for this exact failure shape.

## Schema / bus / API changes

- Added: `concept_atlas_network()` response — `nodes[].component_id` (int), `nodes[].topic_id` (string or `null`), top-level `component_count` (int).
- Added: `ConceptNodeV1.metadata["topic_id"]` now durably persists through `FalkorSubstrateStore` (previously write-only to the in-process cache).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing fields; purely additive.
- Compatibility notes: additive-only API response fields; no consumer needs updating. The `orion/substrate/falkor_codec.py` change is a pure bugfix (a field that was supposed to persist now actually does) — no migration needed, no existing durable row loses data.

## Env/config changes

None.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests/test_concept_atlas_routes.py -q
  -> 19 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests -k "concept_atlas or topic_foundry" -q
  -> 83 passed, 1303 deselected

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest orion/substrate/tests -q
  -> 564 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-substrate-runtime/tests -q --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
  -> 267 passed, 16 failed
  Confirmed pre-existing and unrelated via git-stash comparison: the same 16
  test names fail identically with this branch's changes stashed out
  (baseline). test_grammar_consumer_integration.py itself fails to even
  collect on baseline too (ModuleNotFoundError: app.models -- an unrelated
  cross-service sys.path issue when run outside orion-substrate-runtime's
  own test invocation), so it's excluded from both runs for a fair
  comparison.

cd services/orion-hub && node --test static/js/*.test.js
  -> 39 passed, 22 skipped (pre-existing skips, unrelated to this branch)
```

## Evals run

No eval harness exists for `orion-hub`'s Concept Atlas feature or for `orion/substrate` beyond `tests/`. This is a pure interpretability/debug UI change plus a codec bugfix, not a cognition-loop change — judged not to need a new eval harness.

## Docker/build/smoke checks

No runtime/config/Docker-boot-path changes. Not run.

## Review findings fixed

Code review ran once (see `Skill("code-review")` on branch `feat/concept-atlas-readability`), found 4 findings:

- Finding (CONFIRMED, most severe): `topic_id` (and `run_id`/`doc_count`/`keywords`/`source`) were absent from `DYNAMICS_METADATA_KEYS`, the closed allowlist deciding which `ConceptNodeV1.metadata` keys survive a FalkorDB round-trip, so the new `topic_id` passthrough silently reverted to `null` on the live (`SUBSTRATE_STORE_BACKEND=falkor`, Hub's real default) backend within ~30 seconds of any write.
  - Fix: new, deliberately separate `TOPIC_FOUNDRY_METADATA_KEYS` allowlist + encode/decode pair in `falkor_codec.py`, not folded into `DYNAMICS_METADATA_KEYS` (different owner — topic-foundry ingest is the only writer, none of the concurrent-writer clobber risk `EXTERNALLY_OWNED_METADATA_KEYS` exists to guard against).
  - Evidence: `orion/substrate/tests/test_falkor_codec_topic_id.py` (new, 7 tests including a full encode→decode round-trip); `orion/substrate/tests -q` → 564 passed.
- Finding (CONFIRMED): the component-count status line had no defensive fallback for a missing `component_id`, unlike the adjacent `god_node_count` on the same line (guarded with `|| 0`) — a `Set` of all-`undefined` values collapses to size 1, so a stale cached bundle/backend response would confidently report "1 component(s)" instead of an honest unknown.
  - Fix: filter out `undefined`/`null` ids before building the `Set`.
  - Evidence: code inspection (no dedicated JS test — matches this file's existing convention of not unit-testing DOM-adjacent Cytoscape style/status logic; `edgeColor()`/`PREDICATE_COLORS` similarly have none).
- Finding (CONFIRMED): default label visibility was gated solely on the `god_node` flag, so any filtered/scoped subgraph with zero god nodes (e.g. every node degree-0 after an `anchor_scope` filter with no co-occurrence edges) showed zero labels by default — exactly the sparse/filtered view the readability effort was meant to help.
  - Fix: falls back to showing all labels when the currently-mounted subgraph has no god nodes at all.
  - Evidence: code inspection; `node --test static/js/*.test.js` still 39 passed (no test regressions).
- Finding (self-documented as an accepted approximation, not a bug — noted by the review itself): the displayed component count is computed after the client-side `promotion_state` filter removes nodes, but `component_id` was assigned server-side against the pre-filter graph, so the count can under-report real visual fragmentation if a filter happens to remove a cut vertex.
  - Not fixed: already had an explaining code comment before this review round; re-deriving connectivity client-side for an informational status line only was judged disproportionate. Documented here as an accepted, known limitation rather than silently left unexplained.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build
```

`orion-substrate-runtime` and any other service linking `orion/substrate/falkor_codec.py` should also be restarted so newly-ingested topic-foundry concept nodes get the fixed encode/decode path (existing durable rows written before this fix simply have no `topic_id` property yet — re-running the topic-foundry ingest for a given run re-populates it, no backfill migration needed).

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the component-count approximation after client-side `promotion_state` filtering (documented above) can under-report fragmentation in an edge case.
- Mitigation: informational status text only, not used for any decision logic; documented in-code and in this report.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/concept-atlas-readability
