# PR report: connect the golden seed concepts to the organic concept graph

Implements `docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md`.

## Summary

- Orion and Juniper's golden seed concepts were a fully isolated 3-node island in the Concept Atlas, disconnected from every organically-discovered topic-foundry concept — despite also ranking as top-degree "god nodes." Added a 4th golden seed, **Claude** (the Hub social room's live 3rd chat participant), and a real mechanism that connects all of them to the organic graph: when a topic-foundry mention exact-matches a golden seed's label, the mentioned entity now gets a second `associated_with` edge straight to that seed's real node_id.
- `SubstrateAnchorScopeV1` widened to add `"claude"` — the one real schema change in this patch.
- Fixed a pre-existing, related bug found during design: `concept_atlas_network()` (the graph API the Concept Atlas UI reads) was silently dropping every edge that touched a non-concept node (entity mentions), because `store.query_concept_region()` only ever returns concept-kind nodes and the route's own filter then required both edge endpoints to already be in that concept-only set. Added a bounded hydration pass that pulls in the off-slice node so the edge survives — this also resurrects the pipeline's pre-existing topic→entity mention edges, not just the new landmark edges.
- `orion/substrate/seed_concepts.yaml` gained a 4th entry (Claude), linked to both Orion and Juniper the same way the Orion-Juniper relationship node already is.

## Outcome moved

Before this patch, a topic-foundry run mentioning "Orion," "Juniper," or "Claude" produced an isolated entity node with no path back to the golden seed triad, and the network API silently dropped even that entity node's edges from the response. After this patch, that same mention produces a second edge straight to the real seed node, and the network API surfaces it — the seed nodes and the organically-discovered concept graph share a connected component instead of remaining permanently disconnected islands, live-verifiable via `GET /api/substrate/concepts/network` after a real ingestion run with a matching mention.

## Current architecture

Before this patch (all confirmed live/by-code in the design doc): 3 disconnected components in `orion_substrate` — golden seed triad (2 `associated_with` edges), 9 degree-0 `node:substrate.<domain>` telemetry singletons (untouched, out of scope), one organic topic-foundry cluster. `SubstrateIdentityResolver.canonical_node_key()` partitions identity by `node_kind`, so an `EntityNodeV1` mention can never identity-merge with a seed `ConceptNodeV1`, by design. `read_concept_region()` (the store-layer function every `query_concept_region()` caller ultimately hits) selects concept-kind nodes only.

## Architecture touched

- `orion/core/schemas/cognitive_substrate.py` — `SubstrateAnchorScopeV1` schema.
- `orion/substrate/seed_concepts.yaml` — golden seed fixture (data only).
- `orion/substrate/adapters/topic_foundry.py` — mention-edge → `EntityNodeV1` pipeline.
- `services/orion-hub/scripts/concept_atlas_routes.py` — Orion-graph ingestion route, network read route.

## Files changed

- `orion/core/schemas/cognitive_substrate.py`: widened `SubstrateAnchorScopeV1` to add `"claude"`.
- `orion/substrate/seed_concepts.yaml`: added the 4th golden seed entry for Claude, linked to Orion and Juniper.
- `orion/substrate/tests/test_seed_concepts.py`: updated the hardcoded "3 canonical concepts" expectation to 4; added assertions for Claude's label/scope/relationship edges.
- `orion/substrate/adapters/topic_foundry.py`: new optional `landmark_concept_ids` param on `map_topic_foundry_run_to_substrate()`/`_build()`; emits an extra `associated_with` edge from a matched mention's entity node to the landmark's real node_id (exact-normalized-label match only, deduped per entity per run — topic-independent).
- `services/orion-hub/scripts/concept_atlas_routes.py`:
  - `_VALID_ANCHOR_SCOPES` gains `"claude"`.
  - New `_NETWORK_HYDRATION_MAX_EXTRA_NODES` cap (100).
  - New `_landmark_concept_ids()` helper: reads `orion.substrate.seed.load_seed_concept_nodes()` (pure/read-only) into a `label.lower() -> node_id` map; degrades to `{}` on any fixture problem.
  - `_ingest_topic_foundry_run()` gains an optional `landmark_concept_ids` param, passed straight through to the adapter.
  - `concept_atlas_ingest_topic_foundry()` (Orion route/scheduler path) now calls `_landmark_concept_ids()` and passes it through — this covers both the manual route and the scheduler tick, since both call this same function.
  - `concept_atlas_ingest_topic_foundry_aitown()` deliberately unchanged — never passes `landmark_concept_ids` (no golden seed concepts exist in the AI Town store; non-goal, not an oversight).
  - `concept_atlas_network()` gains a bounded hydration pass: for each edge whose endpoint isn't already in the concept-only node set, look it up via `store.get_node_by_id()` and add it if it's non-concept — guarded so a concept node the `?scope=`/`?min_activation=` filters excluded on purpose is never silently readmitted.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: new hydration tests (entity node becomes visible + shares component_id with its concept neighbor, scope-filtered concept node is never readmitted, cap is enforced).
- `services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py`: new end-to-end tests (a matching mention produces a landmark edge to the real seed node_id, a non-matching mention doesn't, the AI Town route never wires landmarks, `_landmark_concept_ids()` reads the real 4-entry fixture).
- `tests/test_cognitive_substrate_topic_foundry_adapter.py`: new adapter-level unit tests (match, case-insensitive match, no match, `None` is a complete no-op, dedup across repeated mentions of the same entity).
- `docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md`: the design doc this implements (already committed to this branch during the design phase).

## Schema / bus / API changes

- Added: `SubstrateAnchorScopeV1` gains `"claude"` as a valid `anchor_scope` value.
- Added: `map_topic_foundry_run_to_substrate()` gains optional `landmark_concept_ids` param (default `None`, complete no-op).
- Added: `_ingest_topic_foundry_run()` gains optional `landmark_concept_ids` param (default `None`).
- Removed: none.
- Renamed: none.
- Behavior changed: `concept_atlas_network()` now returns non-concept nodes (most commonly `EntityNodeV1` mentions) that are reachable via an edge from an in-slice concept node, bounded by `_NETWORK_HYDRATION_MAX_EXTRA_NODES` (100). Previously such nodes and their edges were silently dropped. Hydrated nodes are exempt from the `?scope=`/`?min_activation=` filters (they were never eligible for them — the filters only ever operated on the concept-only node set fetched from the store).
- Compatibility notes: `SubstrateAnchorScopeV1` is a `Literal` type used only by `cognitive_substrate.py`'s own schemas (`BaseSubstrateNodeV1`, `SubstrateGraphRecordV1`) — confirmed by grep that several unrelated schemas (`orion/core/schemas/mentor.py`, `reasoning_policy.py`, `reasoning_summary.py`, `reasoning.py`, `orion/reasoning/lifecycle.py`) independently duplicate the same 5-value literal rather than importing this type alias. None of those consume `ConceptNodeV1`/`EntityNodeV1` and nothing in this patch constructs an instance of them with `anchor_scope="claude"`, so they're unaffected — flagging this as a known, pre-existing duplication debt, not touching it (out of this patch's scope; see design doc's non-goals discipline).

## Env/config changes

None. No `.env_example` changes, nothing to sync.

## Tests run

```text
cd <worktree> && .venv/bin/python -m pytest \
  orion/substrate/tests/test_seed_concepts.py \
  tests/test_cognitive_substrate_topic_foundry_adapter.py \
  services/orion-hub/tests/test_concept_atlas_routes.py \
  services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py -q
100 passed

cd <worktree> && .venv/bin/python -m pytest orion/substrate/tests/ -q
579 passed

cd <worktree> && .venv/bin/python -m pytest services/orion-hub/tests/test_topic_foundry_scheduler.py -q
36 passed
```

Note on test trustworthiness: an earlier run without an explicit `cd` into the worktree silently resolved `orion.*` imports against the main checkout instead of the worktree (Python inserts cwd at `sys.path[0]` for `-m` invocations; this sandbox's Bash cwd defaults to the main checkout between calls) — caught by intentionally observing a hardcoded "3 canonical concepts" assertion pass when it should have failed against the new 4-concept fixture, then confirming it correctly failed once import resolution was fixed. All numbers above are from runs with `cd <worktree> &&` in the same Bash call, verified against the worktree's actual code.

## Evals run

No dedicated eval harness exists for `orion/substrate` or `services/orion-hub`'s concept-atlas surface (same as prior PRs in this arc, e.g. PR #1760's report). Not adding one in this patch — out of proportion for a wiring change; the acceptance checks in the design doc (live network-API verification, entity/landmark edge correctness, cap enforcement) are covered by the unit/route tests above instead.

## Docker/build/smoke checks

Not applicable — no runtime config, port, dependency, or compose changes. Pure Python/schema/YAML change plus tests.

## Review findings fixed

Ran `/code-review medium` in a subagent against the full staged diff. Result: zero findings survived. The review confirmed: the hydration loop correctly excludes re-admitted scope-filtered concept nodes (guarded by `node_kind != "concept"`), the cap is enforced correctly, the AI Town caller deliberately never passes `landmark_concept_ids`, all live `SubstrateGraphStore` implementations (`InMemorySubstrateGraphStore`, `FalkorSubstrateStore`, `RoutedSubstrateGraphStore`) implement `get_node_by_id`, entity-node and landmark-edge dedup are correctly scoped, and the `SubstrateAnchorScopeV1`/`_VALID_ANCHOR_SCOPES` widening was applied everywhere it needed to be. No CLAUDE.md violations found — the new `"claude"` concept ships with a real producer (adapter + fixture), consumer (network route), and test in this same patch, clearing the no-keyword-cathedral gate.

## Restart required

```text
No restart required for this branch alone -- it's a pure code/schema/data change with no running-service dependency. Once merged and deployed, orion-hub picks it up on its normal restart/redeploy; no manual action needed beyond the normal deploy flow.
```

## Risks / concerns

- Severity: low
- Concern: `orion/core/schemas/mentor.py`, `reasoning_policy.py`, `reasoning_summary.py`, `reasoning.py`, and `orion/reasoning/lifecycle.py` independently duplicate the pre-widening 5-value anchor-scope literal instead of importing `SubstrateAnchorScopeV1`. They don't consume `ConceptNodeV1`/`EntityNodeV1` and are unaffected by this patch, but if a future patch ever wants a Claude-scoped mentor/reasoning artifact, those five spots would need the same widening by hand.
- Mitigation: documented here and in the design doc's schema-change note; not fixed in this patch since nothing in it touches those schemas (real scope creep to fix pre-existing duplication debt unrelated to the stated task).

## PR link

<paste-ready — `gh` not authenticated in this session; branch pushed, open a PR from `docs/concept-graph-landmark-connection-design` into `main` at:
https://github.com/junebug-junie/Orion-Sapienform/compare/main...docs/concept-graph-landmark-connection-design?expand=1>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
