## Summary

This is memory Stage 2 PR A: the shared "assertion core" that both memory referents (PR B) and #2497's reading pipeline need. Spec: `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md` (approved 2026-10-06).

- **Claims are first-class.** Orion's graph can now hold a claim that two things are related (an `assertion` node), with its own state (proposed / provisional / canonical / rejected / deprecated) and a revision counter.
- **A relationship is walkable only while its claim is accepted.** The shortcut edge drawn from a claim (`edge_role=semantic_projection`) is walked by the neighborhood read only while that claim is provisional or canonical, *at the same revision*. A rejected claim, or a half-finished projector run, can never pass a stale edge off as accepted. This holds in the in-memory backend, in real FalkorDB (Cypher join on `assertion_id`), and in the legacy SPARQL backend, which fails closed: it walks legacy edges only.
- **Every edge says what it is for** (`edge_role`). Old edges decode as `legacy_unreviewed` and behave exactly as before.
- **Memory nodes are fenced from silent merges.** Nodes from `memory.referents` are identified by id alone. Topic-foundry's same-named "circe" entity, or a concept with a close embedding, can never merge into them, in either direction.
- **New structure does not change cognition.** Dynamics, attention and the brain frame see exactly the graph they saw before (#2497 rule 8). A before/after fixture is identical, and a mutation twin shows the test can fail.
- **One append-only journal.** Postgres table `substrate_graph_journal` records proposal → decision → materialization. `AssertionProjector` applies decisions to Falkor, strictly in revision order, and records the real ids it got back. It fails closed: no placeholder nodes, and a persistent failure is logged once, not every tick.

## Outcome moved

Before: the substrate had no way to say "this relationship was claimed, by whom, on what evidence, and was it accepted", and every edge looked the same to every reader. After: the claim lifecycle exists end to end in code (journal → projector → graph → neighborhood gate). It is proven on real Postgres and real FalkorDB. It does not move a single number in dynamics or attention.

## Current architecture

- `SubstrateEdgeV1` had no acceptance state and no role (#2497 "Existing contracts").
- The codec persisted only concept/entity/evidence.
- `reconcile.py` merged entities by lowercased label and concepts by embedding cosine, for every producer.
- `dynamics.tick` and `substrate_pressure_signals` consumed every node and edge in the snapshot.
- #2497 proposed `ReadingGraph*V1` events, but shipped no code for them. This PR renames them to `SubstrateGraph*V1`, so nothing that shipped breaks.

## Architecture touched

- **Contracts:** `orion/core/schemas/cognitive_substrate.py` (assertion kind, two predicates, `edge_role`, assertion fields on edges, validators); new `orion/core/schemas/substrate_graph_journal.py`.
- **Store/codec:** `falkor_codec.py`, `falkor_store.py`; `graphdb_store.py` now writes an `orion:edgeRole` triple.
- **Identity:** `reconcile.py` (fence, projector-owned merge, role-aware edge identity).
- **Reads:** `neighborhood.py`, `neighborhood_backends.py`.
- **Cognition isolation:** new `eligibility.py`; `dynamics.py`, `attention_broadcast.py`, substrate-runtime `_brain_frame_tick`.
- **Journal + projector:** new `graph_journal.py`, `assertion_projector.py`, and a migration.
- **UI:** the Hub Concept Atlas labels an Assertion with its claim text.

## Files changed

- `orion/core/schemas/cognitive_substrate.py`: `AssertionNodeV1`; `assertion_subject`/`assertion_object` predicates; `SubstrateEdgeRoleV1`; `edge_role`/`assertion_id`/`assertion_revision` on edges, with validators.
- `orion/core/schemas/substrate_graph_journal.py` (new): `SubstrateGraph{Proposal,Decision,Materialization}V1`.
- `orion/core/schemas/__init__.py`, `orion/schemas/registry.py`: export and register `AssertionNodeV1` (`resolve("AssertionNodeV1")` verified).
- `orion/substrate/falkor_codec.py`, `falkor_store.py`: durable `assertion` kind; native edge-role and assertion properties; old rows decode as `legacy_unreviewed`.
- `orion/substrate/reconcile.py`: `IDENTITY_FENCED_PRODUCERS`, `is_identity_fenced`, fenced identity keys, embedding-candidate skip, projector-owned merge for fenced nodes and role-bearing edges, `projection|<assertion_id>` edge identity.
- `orion/substrate/eligibility.py` (new): `cognitive_view`, `is_cognitive_node`, `is_cognitive_edge`.
- `orion/substrate/dynamics.py`, `attention_broadcast.py`, `services/orion-substrate-runtime/app/worker.py`: consume the cognitive view.
- `orion/substrate/neighborhood.py`, `neighborhood_backends.py`, `graphdb_store.py`: `walkable_edge` gate (Python, Cypher, SPARQL).
- `orion/substrate/graph_journal.py` (new): asyncpg journal store with an expected-revision conflict.
- `orion/substrate/assertion_projector.py` (new): `AssertionProjector.run_once`.
- `services/orion-sql-db/manual_migration_substrate_graph_journal_v1{,_rollback}.sql` (new).
- `services/orion-hub/scripts/concept_atlas_routes.py`: Assertion label = `statement_text`.
- `orion/substrate/README.md` (new): Concepts table.
- Tests: `orion/substrate/tests/test_assertion_core.py`, `test_assertion_core_falkor.py`, `test_graph_journal_pg.py` (new); updated `test_falkor_codec.py`, `test_falkor_store.py`, hub `test_concept_atlas_routes.py`, substrate-runtime `test_brain_frame_worker.py`.
- `.github/workflows/substrate-neighborhood.yml`: Postgres service, new lanes, fail-on-skip.

## Concepts (each with a producer, a consumer and a test)

| Concept | Producer | Consumer | Test |
|---|---|---|---|
| `assertion` node kind | `AssertionProjector` | `walkable_edge` (Python + Cypher), `eligibility`, Hub Atlas label | `test_assertion_node_round_trips_through_the_codec`, Falkor walk tests, Atlas all-kinds test |
| `assertion_subject` / `assertion_object` | `AssertionProjector` | edge validator, `eligibility`, `walkable_edge` (never walks) | `test_edge_role_validators_refuse_unverifiable_shapes`, walk tests |
| `edge_role=legacy_unreviewed` | decode default + every existing writer | `walkable_edge`, `eligibility` | `test_edge_role_and_assertion_fields_round_trip_and_old_rows_decode_as_legacy` |
| `edge_role=semantic_projection` (+ `assertion_id`, `assertion_revision`) | `AssertionProjector` | `walkable_edge` / `_WALKABLE_EDGE_TAIL`, `canonical_edge_key` | 6-case walk test (memory), 4-case walk test (real Falkor, mutation-checked) |
| `edge_role=provenance` | memory referent projector (PR B) | `walkable_edge`, `eligibility`, `merge_edge` | `test_walkable_edge_refuses_provenance_and_missing_assertions`, `test_a_role_bearing_edge_takes_the_projector_s_validity_but_keeps_its_id` |
| `edge_role=assertion_structure` | `AssertionProjector` | `walkable_edge`, `eligibility` | walk tests |
| `IDENTITY_FENCED_PRODUCERS` | memory referent projector (PR B) | `canonical_node_key`, embedding scan, `merge_node`, `eligibility` | `test_a_fenced_referent_never_merges_with_a_same_label_topic_entity`, `test_embedding_identity_never_crosses_the_fence_in_either_direction` |
| `cognitive_view` | pure function | dynamics tick, attention signals, brain frame | `test_dynamics_is_identical_before_and_after_projecting_memory_structure` (+ mutation twin), `test_attention_never_offers_a_fenced_node_or_an_assertion`, `test_brain_frame_leaves_out_memory_referents_and_assertions` |
| `substrate_graph_journal` | `SubstrateGraphJournal.append` | `pending_decisions`, `proposal`, `latest_applied_revision`, `last_materialization` | `test_graph_journal_pg.py` (6 tests) |
| `SubstrateGraphProposalV1` | claim producers (PR B) | `AssertionProjector` | `test_accept_then_reject_projects_then_retracts_and_replays_idempotently` |
| `SubstrateGraphDecisionV1` | acceptance step (PR B's named policies) | `AssertionProjector` | `test_journal_append_is_idempotent_and_enforces_the_expected_revision`, `test_revisions_apply_in_order_even_when_the_later_one_is_read_first` |
| `SubstrateGraphMaterializationV1` | `AssertionProjector` | `pending_decisions`, `latest_applied_revision`, the projector's once-per-reason failure record | `test_a_missing_endpoint_fails_closed_once_and_mints_no_placeholder` |

**Cut, because nothing consumes them yet:**
- the `orion:substrate:graph:*` bus channels (the journal is Postgres-only until a subscriber exists);
- the `ontology_membership` edge role;
- the `visibility_scope` column;
- evidence `voice`/`channel` native properties. The first reader is recall (spec PR E/F); memory evidence voice stays in `episode_memory.voice` until then;
- `via`/`matched_text` on provenance edges (no reader before recall's `why`).

## Schema / bus / API changes

- **Added:**
  - `AssertionNodeV1` (registered);
  - predicates `assertion_subject`, `assertion_object`;
  - `SubstrateEdgeV1.edge_role` (default `legacy_unreviewed`), `assertion_id`, `assertion_revision`;
  - journal row contracts `SubstrateGraph{Proposal,Decision,Materialization}V1` (not bus payloads, so not registered);
  - Postgres table `substrate_graph_journal`.
- **Renamed (contract only):** #2497's proposed `ReadingGraph*V1` → `SubstrateGraph*V1` with `proposal_kind`. No code had shipped under the old names.
- **Behavior changed:**
  - neighborhood reads skip `provenance`/`assertion_structure` edges and unaccepted projections. This has no effect on today's data: every existing edge is legacy;
  - `reconcile` never merges `memory.referents` nodes;
  - dynamics, attention and the brain frame ignore assertions and fenced nodes (none exist yet).
- **Compatibility:**
  - `SubstrateEdgeV1`/node models are not on any bus channel (checked `orion/bus/channels.yaml`), so the extra-forbid field additions are in-process only;
  - Falkor hydration in OLD code fails closed on an unknown node kind or predicate. **So every substrate reader must run this PR's code before PR B writes the first Assertion or role edge** (see Restart).

### For the #2497 owner to review
1. The event rename and `proposal_kind` (the contract in `substrate_graph_journal.py`). Reading will add `proposal_kind` values with their consumers. The journal has `target_id` (the Assertion id) and typed relationship fields rather than a free payload.
2. The Cypher in `neighborhood_backends._WALKABLE_EDGE_TAIL`: an `OPTIONAL MATCH` on `e.assertion_id` after the focal filter, carried through `WITH`. Receipts are unchanged.
3. `reconcile.canonical_edge_key` now keys role-bearing edges by role, and projections as `projection|<assertion_id>` (your "projection identity is hash(assertion_id)").
4. The `AssertionProjector` revision semantics: `revision` = number of applied decisions; strict in-order apply; it records failure instead of writing placeholders.
5. The SPARQL backend walks legacy edges only (it has no Assertion nodes to verify against).

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` sync: not needed (no template changed).

## Tests run

```text
orion/substrate/tests/test_assertion_core.py                 19 passed
orion/substrate/tests/test_assertion_core_falkor.py           5 passed  (real FalkorDB 6.0.1, throwaway container)
orion/substrate/tests/test_graph_journal_pg.py                 6 passed  (real Postgres 16 + FalkorDB, throwaway containers)
orion/substrate/tests + orion/core (whole tree)            1095 passed, 3 failed
  -> the 3 are test_felt_state_self_definition_lane.py, failing identically on main @ 029322db2 (pre-existing)
substrate-neighborhood CI list (incl. test_falkor_direct.py, real Falkor)   all passed
services/orion-hub/tests (15 substrate/atlas/workbench files)   342 passed (after the Atlas fix)
services/orion-recall substrate tests 28 passed; substrate-runtime 29 passed (+1 new brain-frame test); cortex-exec 6 passed
Mutation checks (temporarily reverting each gate, then restoring):
  dynamics without cognitive_view            -> isolation test FAILS
  attention without is_cognitive_node        -> attention test FAILS
  Cypher gate accepting any assertion state  -> Falkor "rejected" walk case FAILS
Static gates (every python gate in orion-static-gates.yml, incl. check_definition_drift.py --gate): 22/22 PASS; git diff --check clean
```

## Evals run

```text
python -m orion.substrate.evals.run_neighborhood_eval        PASS
python -m orion.substrate.evals.run_complete_hydration_eval  PASS
python -m orion.substrate.evals.run_decay_since_last_eval    PASS
```

These are unchanged by this PR: the before/after isolation test is the eval-shaped check for #2497 acceptance 8. There is no live-data eval, because nothing writes Assertions until PR B.

## Docker/build/smoke checks

```text
No image rebuilt for this PR (no deploy requested). Real-store proof is the throwaway FalkorDB/Postgres
lanes above. Production Falkor/Postgres were not touched.
```

## Review findings fixed

Review of #2515 (2026-10-06). Prod check by the reviewer: 0 role edges and 0 assertions live today; the neighborhood join costs +5-10% on a 1,932-edge hub and uses the index.

- **Finding (SHOULD):** the walkability gate covered only the neighborhood read. Recall's concept region, graph compression, and the Hub typed-relation classifier would all have read assertion structure edges and rejected projections as ordinary relationships.
  - **Fix:**
    - in-memory region reads and `falkor_direct` apply the gate: a join-free role prefilter in Cypher, plus one batched assertion-state lookup after the cut. The cache does the same in the same order, so the #2505 equivalence holds;
    - the graph-compression federator applies the same gate;
    - the Hub classifier reads only the cognitive view.
  - **Evidence:**
    - `test_concept_region_drops_structure_and_unaccepted_projections_and_equals_cache` (real Falkor: direct == cache, only legacy + accepted);
    - `test_recall_region_reads_drop_structure_and_unaccepted_projections`;
    - `test_only_walkable_edges_reach_compression`;
    - `test_typed_relation_classifier_never_reads_projections_or_memory_nodes`;
    - the existing legacy equivalence lane (`test_full_slice_equals_hydrated_cache`, 5 cases) is unchanged and green.
  - **Found along the way:** a per-edge `OPTIONAL MATCH` in the concept-region cut timed out on the realistic fixture, hence the prefilter-plus-batch shape.
- **Finding (SHOULD):** old-vs-new safety depended on deploy order.
  - **Fix, part 1 (forward-tolerant readers):** unknown node kinds, predicates, endpoint kinds and roles are skipped and counted (and so are edges touching a skipped node), never fatal.
  - **Fix, part 2 (one mechanical gate):**
    - every reader advertises `assertion_core_v1` (`reader_capability.py`) from `bootstrap_substrate_reader`, which runs off the request path;
    - `AssertionProjector` requires a `readiness` check and writes nothing until every required reader has advertised;
    - the memory referent projector (#2520) uses the same gate.
  - **Evidence:** `test_a_shape_newer_than_this_reader_is_skipped_and_counted_not_fatal`, `test_reader_capability.py` (5), `test_nothing_is_written_until_every_reader_is_ready`.
  - **Regression caught by recall's own suite and fixed:** advertising first ran inside recall's no-network builder, costing 2 s per turn on a hung Falkor. It now rides recall's background index bootstrap.
- **Finding (SHOULD):** projector starvation.
  - **Fix:** the pending queue lists only the next revision of each target, drops terminal failures (stale revision, missing proposal, id held elsewhere), and orders never-attempted decisions first, then retries by oldest attempt.
  - **Evidence:** `test_101_stuck_decisions_cannot_starve_a_new_one` (fails under the old ordering, checked by mutation).
- **Finding (NIT):** the canonical-id mismatch was detected only after writing.
  - **Fix:** `_identity_taken` checks before writing. The after-write check stays as an error-level defense.
  - **Evidence:** `test_a_held_id_is_refused_before_anything_is_written`.
- **Finding (NIT):** append-only was a convention only.
  - **Fix:** a trigger refuses UPDATE and DELETE.
  - **Evidence:** `test_the_journal_is_append_only_and_ids_are_namespaced_by_kind`.
- **Finding (NIT):** failure ids collapsed A→B→A into two records, and event ids could collide across kinds.
  - **Fix:** a failed attempt is recorded each time the reason changes, with an attempt counter in its id; `event_id` is `<kind>:<id>`.
  - **Evidence:** `test_terminal_failures_leave_the_queue_and_a_b_a_is_three_records`, and the namespacing test.
- **Reported, not fixed (pre-existing):** `brain_frame_producer.py:319` edge samples are always empty.
- **Privacy (public repo):** real third-party names in this PR's tests and README were replaced with synthetic ones. They remain in commit `f44c3e948` (no force-push).
- **Also changed:** the Concept Atlas no longer reaches Assertion nodes, because region reads drop their structure edges. The `statement_text` label added earlier was dead code and was removed.

## Rollout (A + B together)

1. Apply the journal migration, then deploy this code to every substrate reader (substrate-runtime, hub, recall, cortex-exec incl. background, cortex-orch, spark-concept-induction, field-digester, world-pulse, meta-tags). Each one advertises `assertion_core_v1` at boot and skips unknown shapes from then on. Deploy order among them does not matter.
2. Deploy the writers (#2520). They check readiness before every pass and log `*_waiting reason=readers_not_ready missing=[...]` (also shown on consolidation `/health`) until step 1 is complete everywhere. Then they open by themselves.

## Restart required

Deploy order (nothing in this PR writes the new shapes, so it is safe to deploy alone):

1. Apply the migration (additive, safe any time before PR B):
   `psql "$DSN" -f services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql`
2. Rebuild every service that reads `orion_substrate`, BEFORE PR B's projector is enabled: orion-substrate-runtime, orion-hub, orion-recall, orion-cortex-exec (incl. background), orion-cortex-orch, orion-spark-concept-induction, orion-field-digester, orion-world-pulse, orion-meta-tags. Old code refuses to hydrate a graph that contains an Assertion node.
   `scripts/safe_docker_build.sh <service> up -d --build` (one service at a time)

## Risks / concerns

- Severity: high (deploy-order only)
  - Concern: an old reader hydrating a graph with an Assertion node fails its whole snapshot (fail-closed codec).
  - Mitigation: rollout step 2 before PR B; PR B's report repeats it.
- Severity: low
  - Concern: the SPARQL/GraphDB backend never walks projections.
  - Mitigation: it is not the live backend (`SUBSTRATE_STORE_BACKEND=falkor`); this fails closed, by design.
- Severity: low
  - Concern: the Falkor neighborhood adds one `OPTIONAL MATCH` per candidate edge.
  - Mitigation: it is indexed on `node_id` and runs after the focal filter; spec PR E measures latency.
- UNVERIFIED: the live runtime path. No Assertion exists in production until PR B runs; latency of the extra `OPTIONAL MATCH` on the live graph has not been measured.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2515

🤖 Generated with [Claude Code](https://claude.com/claude-code)
