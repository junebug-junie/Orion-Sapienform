# orion/substrate

Orion's one graph (FalkorDB graph `orion_substrate`): the things Orion knows about,
how they relate, and where each piece came from. Postgres holds text, evidence and
the append-only audit trail; Falkor holds what can be walked.

Design docs: `docs/plans/substrate/2026-10-06-reading-property-graph-design.md`,
`docs/plans/substrate/2026-10-06-neighborhood-read-contract.md`,
`docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`.

## Concepts

Every concept below has a producer, a consumer and a test. If you add one, add its
row here in the same PR; if a row loses its consumer, delete the concept.

| Concept | What it means in plain English | Producer | Consumer | Test |
|---|---|---|---|---|
| `assertion` node kind (`AssertionNodeV1`) | A claim that two things are related ("Quill and the spring retreat came up together"), with its own yes/no/maybe state and a revision counter. A relationship is only walkable while its claim is accepted. | `assertion_projector.AssertionProjector` | `neighborhood.walkable_edge` (and its Cypher twin in `neighborhood_backends.py`); `eligibility.is_cognitive_node` | `tests/test_assertion_core.py`, `tests/test_assertion_core_falkor.py`, `tests/test_graph_journal_pg.py` |
| `assertion_subject` / `assertion_object` edges | Which two things a claim is about. They describe the claim; they are never walked as a relationship. | `AssertionProjector` | edge validator in `SubstrateEdgeV1` (only from an Assertion); `eligibility` (never propagates) | `test_edge_role_validators_refuse_unverifiable_shapes`, `test_a_projection_walks_only_while...` |
| `edge_role` = `legacy_unreviewed` | Every edge that existed before roles: walked and propagated exactly as before. Old rows with no role decode as this. | the decode default (`falkor_codec.decode_edge`) and every pre-existing writer | `walkable_edge`, `eligibility.COGNITIVE_EDGE_ROLES` | `test_edge_role_and_assertion_fields_round_trip_and_old_rows_decode_as_legacy`, `test_cognitive_view_keeps_legacy_topology_exactly` |
| `edge_role` = `semantic_projection` (+ `assertion_id`, `assertion_revision`) | The walkable shortcut drawn from an accepted claim. Walked only while that claim is provisional/canonical *at the same revision*, so a half-finished projector run can never pass off a stale edge as accepted. | `AssertionProjector` | `walkable_edge` / Falkor `_WALKABLE_EDGE_TAIL`; recall's concept region (`store._read_by_node_predicate`, `falkor_direct` via `role_prefilter` + one batched `ASSERTION_STATE_CYPHER` lookup); graph-compression's Falkor federator; `reconcile.canonical_edge_key` (`projection|<assertion_id>`, never merged into a legacy edge) | parametrized walk tests in both lanes; `test_a_projection_never_merges_into_a_same_endpoint_legacy_edge` |
| `edge_role` = `provenance` | "This thing was seen in that piece of evidence." Never walked, never propagates. | `services/orion-memory-consolidation/app/referent_projector.py` | `walkable_edge` (refuses), `eligibility` (excludes), `reconcile.merge_edge` (projector's validity wins) | `test_walkable_edge_refuses_provenance_and_missing_assertions`, `test_a_role_bearing_edge_takes_the_projector_s_validity_but_keeps_its_id` |
| `edge_role` = `assertion_structure` | The claim's own wiring: claim → its two things, evidence → the claim it supports. Never walked. | `AssertionProjector` | `walkable_edge`, `eligibility` | as above |
| `IDENTITY_FENCED_PRODUCERS` (`memory.referents`) | Nodes from these producers are identified by their id alone. They are never merged with another node because the names match or the embeddings are close; only a recorded merge decision can join them. | `referent_projector.py` | `reconcile.canonical_node_key`, `_concept_embedding_match_key` candidate skip, `merge_node` (projector's lifecycle wins), `eligibility` | `test_a_fenced_referent_never_merges_with_a_same_label_topic_entity`, `test_embedding_identity_never_crosses_the_fence_in_either_direction` |
| Cognitive eligibility (`eligibility.cognitive_view`) | What takes part in Orion's dynamics, attention and brain frame: exactly the graph as it was before claims and memory nodes existed. New structure is visible to reads, not to cognition. | n/a (pure function) | `dynamics.SubstrateDynamicsEngine.tick`, `attention_broadcast.substrate_pressure_signals`, substrate-runtime `_brain_frame_tick` | `test_dynamics_is_identical_before_and_after_projecting_memory_structure` (+ its mutation twin), `test_attention_never_offers_a_fenced_node_or_an_assertion`, `services/orion-substrate-runtime/tests/test_brain_frame_worker.py::test_brain_frame_leaves_out_memory_referents_and_assertions` |
| `substrate_graph_journal` table | The append-only record of every claim, every decision about it, and what was actually written to the graph. The graph can be rebuilt from it. | `orion/memory/referents/store.py` (claims) and `AssertionProjector` via `graph_journal.SubstrateGraphJournal.append` | `SubstrateGraphJournal.pending_decisions` / `proposal` / `latest_applied_revision` / `last_materialization`, read by `AssertionProjector` | `tests/test_graph_journal_pg.py` |
| `SubstrateGraphProposalV1` (`proposal_kind=relationship_assertion`) | "I think X and Y are related, and here is the evidence." Nothing is walkable on a proposal alone. | `orion/memory/referents/cooccurrence.py` (`source_cooccurrence_v1`) | `AssertionProjector` (builds the claim node and edges from it) | `test_accept_then_reject_projects_then_retracts_and_replays_idempotently` |
| `SubstrateGraphDecisionV1` | Accept / reject / retract one claim, made against a known revision, naming the rule or reviewer that decided. Two decisions cannot claim the same next revision. | the acceptance step: `source_cooccurrence_v1` today, a reviewer later | `AssertionProjector` (applies state + revision, strictly in revision order) | `test_journal_append_is_idempotent_and_enforces_the_expected_revision`, `test_revisions_apply_in_order_even_when_the_later_one_is_read_first` |
| `SubstrateGraphMaterializationV1` | What the projector really wrote for a decision, with the ids the graph gave back, or why it refused. | `AssertionProjector` | `pending_decisions` (no applied row = still pending), `latest_applied_revision`, the projector's once-per-reason failure record | `test_a_missing_endpoint_fails_closed_once_and_mints_no_placeholder` |

Two store reads exist for single-writer projectors that should not hydrate the whole graph:
`FalkorSubstrateStore.prime_cache_for_producers` (load one producer's nodes/edges after a
restart) and `find_semantic_node_ids_by_label` (is a same-named node owned by someone else?).
Both are used by the referent projector and tested in
`services/orion-memory-consolidation/tests/test_referent_projector_pg.py`.

### Added after review (#2515, 2026-10-06)

| Concept | What it means in plain English | Producer | Consumer | Test |
|---|---|---|---|---|
| Reader capability `assertion_core_v1` (`reader_capability.py`, Redis key `orion:substrate:reader_capability:<reader>`) | "This process can read claims and skips shapes it does not know." Written once per reader, off the request path, by `falkor_store.bootstrap_substrate_reader` (store construction, `falkor_direct` builder, recall's background bootstrap). | every substrate reader built from this code | `reader_capability.readiness`, which `AssertionProjector` (and the memory referent projector) call before every write pass: nothing is written until every required reader (`SUBSTRATE_ASSERTION_REQUIRED_READERS`) has advertised | `tests/test_reader_capability.py`, `test_graph_journal_pg.py::test_nothing_is_written_until_every_reader_is_ready` |
| Forward tolerance (`falkor_codec.node_row_is_known` / `edge_row_is_known`) | A row of a kind, predicate, endpoint kind or edge role newer than this code is skipped and counted (`hydrate_skipped_unknown_nodes/edges`, log `falkor_substrate_hydrate_skipped_unknown_shapes`), never fatal. Edges touching a skipped node are skipped too. | hydration, `falkor_direct` | the reader keeps working on the shapes it knows | `test_complete_hydration.py::test_a_shape_newer_than_this_reader_is_skipped_and_counted_not_fatal`, `test_reader_capability.py::test_an_old_shape_reader_simulated_by_the_decoder_skips_new_rows` |
| Terminal failure reasons (`graph_journal.TERMINAL_FAILURE_REASONS`) | A decision that can never apply (stale revision, missing proposal, id held by another node) leaves the queue; transient ones are retried behind never-attempted decisions, so stuck rows cannot starve new ones. | `AssertionProjector._fail` | `SubstrateGraphJournal.pending_decisions` | `test_101_stuck_decisions_cannot_starve_a_new_one`, `test_terminal_failures_leave_the_queue_and_a_b_a_is_three_records` |
| Append-only trigger on `substrate_graph_journal` | UPDATE and DELETE are refused by the database itself. | the migration | Postgres | `test_the_journal_is_append_only_and_ids_are_namespaced_by_kind` |

Rollout (two phases, mechanical): (1) deploy this code to every substrate reader; each
advertises `assertion_core_v1` at boot and skips unknown shapes from then on. (2) Writers of
the new shapes open by themselves once every required reader has advertised; until then they
log `*_waiting reason=readers_not_ready missing=[...]` and write nothing. Rolling a reader back
to older code: delete its `orion:substrate:reader_capability:<reader>` key.
