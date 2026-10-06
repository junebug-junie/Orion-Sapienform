# orion/memory/referents

Memory Stage 2 (spec `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`,
approved 2026-10-06): the people, places, projects and ideas Orion's memories are about become
nodes in Orion's one graph, under Juniper's own names for them.

- **Resolution** (`resolve.py`, pure): which node a writer key like `project:hecate` means. A
  name only counts if Juniper said it. When a name could mean two things, Orion asks; nothing
  ever merges silently, and nothing is matched by vectors.
- **Store** (`store.py`): runs inside the episode persist (orion-durable-runs), in a savepoint,
  so a fault here never loses the memories. All writes are idempotent.
- **Co-occurrence** (`cooccurrence.py`): two things Juniper named in one sentence become a
  walkable "mentioned together" link.
- **Backfill** (`backfill.py`, CLI `scripts/backfill_referents_from_episodes.py`): recovers the
  names Stage 1 threw away from each saved distill run, then runs the same step.
- **Projection** to FalkorDB: `services/orion-memory-consolidation/app/referent_projector.py`.

## Concepts

| Concept | What it means in plain English | Producer | Consumer | Test |
|---|---|---|---|---|
| Referent node (a `key` row in `referent_alias`) | One thing Orion knows about. It exists because the writer named it with a key; its state (usable or still a question) is that row's state. | `store.persist_referents` (`resolve.resolve_referent`) | the next resolution (exact key); the projector builds the graph node from it | `tests/test_resolve.py`, `tests/test_referents_pg.py::test_persist_resolves_every_referent_and_keeps_juniper_s_names` |
| `episode_memory_referent.node_id` | Which node each memory is about. | `persist_referents` | projector memory pass (evidence node + provenance edges) | `test_persist_resolves_every_referent...` (no NULLs left) |
| Alias class `name` | A proper name ("Inspur NF5288M5"). Can resolve a later key to the same node. | `candidate_aliases` | resolution step 3; collision check | `test_a_grounded_name_resolves_a_later_key_to_the_same_node` |
| Alias class `descriptor` (relative name) | A name that starts with a word common in Juniper's own prompts ("my boss", "the Wade", "camera"). Decided by word frequency, not a word list. Usable at once, lapses 90 days after its last use, and never decides identity. | `aliases.alias_class` from first-word frequency in `chat_history_log` prompts | collision check (only while live); projector shows it only while live | `test_a_relative_name_resolves_now_and_lapses_90_days_after_last_use`, `test_a_collision_on_a_relative_name_becomes_a_question_never_a_merge` |
| `alias_grounding_v1` (kill switch `MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT`) | A name becomes usable only when it appears in Juniper's verified words. Anything else is kept but never used. | `resolve._admit_aliases` | resolution and co-occurrence only use live names; the projector shows only live names | `test_grounded_names_are_usable_and_ungrounded_ones_are_not`, `test_the_grounding_kill_switch_flips_grounded_names_to_proposed`, `test_the_kill_switches_flip_aliases_and_claims_to_proposed` |
| Kind refinement (`project` ↔ `service`) | The same slug filed under project and service is one thing, not two. | `resolve_referent` step 2 | resolution | `test_kind_refinement_files_the_same_thing_under_a_second_key` |
| Identity question (`memory_tension_shadow`, reason `referent_identity` / `alias_collision` / `label_collision`) | "Is X the same as Y?" or "Does 'my boss' mean Rachel or Dana?". People, places and events go to Juniper in conversation; anything else Orion can investigate itself. | resolution; projector (label collision with another producer's node) | the daily memory report, "Referents" section | `test_a_name_on_two_nodes_mints_a_proposed_node_and_asks`, `test_the_daily_report_shows_held_claims_names_by_rule_and_open_questions`, projector test |
| `source_cooccurrence_v1` (kill switch `MEMORY_COOCCURRENCE_AUTO_ACCEPT`) | Two things (not Juniper, not Orion) named in one of Juniper's verified sentences are linked as "mentioned together": accepted only if both are usable and the memory yields at most 6 such links. | `cooccurrence.cooccurrence_claims` (journal proposal + decision) | `AssertionProjector` → walkable `co_occurs_with`; the report counts the ones held back | `test_things_named_in_one_quote_become_accepted_co_occurrences`, `test_the_cooccurrence_kill_switch_leaves_proposals_only`, `test_more_than_six_claims_from_one_memory_are_all_held_for_review` |
| `MEMORY_REFERENTS_ENABLED` | Turns the whole referent step off; the memories are still written. | orion-durable-runs settings | `AdmissionRuntime._referent_policy` | `services/orion-durable-runs/tests/test_episode_distill_referent_policy.py` |
| Checkpoint alias recovery | Brings back the names Stage 1 dropped, from each saved distill answer. | `backfill.backfill_all` | `persist_referents` (same step as live) | `test_checkpoint_backfill_recovers_aliases_and_is_idempotent` (twice = identical) |
