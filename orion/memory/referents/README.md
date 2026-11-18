# orion/memory/referents

Memory Stage 2 (spec `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`,
approved 2026-10-06): the people, places, projects and ideas Orion's memories are about become
nodes in Orion's one graph, under Juniper's own names for them.

- **Resolution** (`resolve.py`, pure): which node a writer key like `project:hecate` means. A
  name only counts if Juniper said it. Only a name the distiller judged a *proper name* can tie
  a new key to an existing node; descriptions ("my boss") never can. Anything ambiguous becomes a
  question. Nothing merges silently, and nothing is matched by vectors or by word lists.
- **Store** (`store.py`): runs inside the episode persist (orion-durable-runs), in a savepoint,
  so a fault here never loses the memories. All writes are idempotent.
- **Co-occurrence** (`cooccurrence.py`): two things Juniper named in one sentence become a
  walkable "mentioned together" link.
- **Backfill** (`backfill.py`, CLI `scripts/backfill_referents_from_episodes.py`): recovers the
  names Stage 1 threw away from each saved distill run, then runs the same step. Dry run by
  default (computed, then rolled back), live progress log, per-episode errors counted.
- **Projection** to FalkorDB: `services/orion-memory-consolidation/app/referent_projector.py`.
  Rebuild only with `scripts/rebuild_referent_graph.py`.

## Concepts

| Concept | What it means in plain English | Producer | Consumer | Test |
|---|---|---|---|---|
| `alias_kind` (`proper_name` / `descriptor`) | The distiller's own judgment of each name, including the key's own name: a proper name names exactly one thing ("Hecate"); a descriptor can point at different things over time ("my boss"). No judgment means descriptor. | the distiller (prompt `memory_episode_distill.v4`), carried by `DistillReferentV1` and the validator | resolution (only proper names decide identity) and expiry (only descriptors lapse) | `tests/test_resolve.py::test_unjudged_names_are_descriptors`, the two review repros below |
| Referent node (a `key` row in `referent_alias`) | One thing Orion knows about; its state (usable, or still a question) is that row's state. | `store.persist_referents` | the next resolution (exact key); the projector builds the graph node from it | `test_referents_pg.py::test_persist_resolves_every_referent_and_keeps_juniper_s_names` |
| `episode_memory_referent.node_id` | Which node each memory is about. | `persist_referents` | projector memory pass | same (no NULLs left) |
| Identity rule | A key lands on an existing node only by its exact key, or by a grounded proper name matching exactly one node of the same kind, with the key's own proper name not saying otherwise. | `resolve_referent` | resolution | `test_the_same_descriptor_for_two_people_is_a_question_not_a_merge`, `test_a_misjudged_proper_name_still_cannot_override_the_key_s_own_name`, `test_a_descriptor_key_never_resolves_another_key`, `test_a_place_name_with_punctuation_never_splits_into_a_second_node` |
| Descriptor expiry (90 days after last use) | Relative names work at once (Juniper's decision) and lapse if unused; reuse extends them. | `_admit_aliases` | collision checks and the projector's displayed names only see live ones | `test_a_descriptor_lapses_90_days_after_last_use_and_is_refreshed_by_reuse` |
| `alias_grounding_v1` (kill switch `MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT`) | A name is usable only when it appears in Juniper's verified words. | `_admit_aliases` | resolution, co-occurrence, projector (live names only) | `test_the_grounding_kill_switch_flips_grounded_names_to_proposed`, `test_the_kill_switches_flip_aliases_and_claims_to_proposed` |
| Identity question (`memory_tension_shadow`, reason `referent_identity` / `alias_collision` / `label_collision`) | "Is X the same as Y?" or "Does 'my boss' mean Morgan or Taylor?". People, places and events go to Juniper in conversation; anything else Orion can investigate. | resolution; projector (a same-named node from another producer) | the daily memory report, "Referents" section | `test_a_proper_name_on_two_nodes_mints_a_proposed_node_and_asks`, `test_the_daily_report_shows_held_claims_names_by_rule_and_open_questions`, projector test |
| `source_cooccurrence_v1` (kill switch `MEMORY_COOCCURRENCE_AUTO_ACCEPT`) | Two things (never Juniper's or Orion's node) named in one of Juniper's verified sentences are linked as "mentioned together": accepted only if both are usable and the memory yields at most 6 such links. | `cooccurrence.cooccurrence_claims` | `AssertionProjector` → walkable `co_occurs_with`; the report counts the ones held back | `test_things_named_in_one_quote_become_accepted_co_occurrences`, `test_the_cooccurrence_kill_switch_leaves_proposals_only`, `test_more_than_six_claims_from_one_memory_are_all_held_for_review`, `test_juniper_and_orion_are_excluded_by_node_id_not_key` |
| `MEMORY_REFERENTS_ENABLED` | Turns the whole referent step off; the memories are still written. | orion-durable-runs settings | `AdmissionRuntime._referent_policy` | `services/orion-durable-runs/tests/test_episode_distill_referent_policy.py` |
| Checkpoint alias recovery | Brings back the names Stage 1 dropped, from each saved distill answer (old answers: every name a descriptor). | `backfill.backfill_all` | `persist_referents` | `test_checkpoint_backfill_recovers_aliases_and_is_idempotent`, `test_backfill_cli_dry_run_writes_nothing_logs_progress_and_survives_a_bad_episode` |
