## Summary

This is memory Stage 2 PR B, stacked on PR A (#2515). It makes the things Orion's memories are about (Hecate, Rachel, the Austin offsite) into nodes in Orion's one graph, under the names Juniper actually uses for them. Spec: `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md` (approved 2026-10-06).

- **Names stop being thrown away.** The validator now keeps the distiller's aliases. After each distill, every referent key is resolved to a graph node id, and Juniper's names for it are stored in Postgres (`referent_alias`).
- **A name only counts if Juniper said it** (`alias_grounding_v1`). A name found in one of her verified sentences is usable at once. Anything else (e.g. Orion's own "the new server") is kept but never used.
- **Relative names work as soon as she says them** (Juniper's decision). "my boss" and "my sister" are usable at once and lapse 90 days after their last use. A relative name never decides who someone is; it can only collide, which becomes a question. Relative names are recognised without a word list: by how common their first word is in Juniper's own prompts.
- **Nothing ever merges silently.** A name on two things, or our "circe" meeting topic-foundry's "circe", becomes an identity question (`memory_tension_shadow`). The new node waits as "proposed".
- **Things named together become walkable** (`source_cooccurrence_v1`). Two things in one verified sentence of Juniper's become a `co_occurs_with` link, through PR A's claim journal. The limits: never Juniper or Orion themselves, and at most 6 per memory.
- **The graph is built by a projector** in orion-memory-consolidation: referent nodes, one evidence node per memory, and "mentioned in" provenance edges with real start/end times. It is idempotent and can be rebuilt from Postgres. A checkpoint backfill recovers the names Stage 1 dropped.

## Outcome moved

Proven on a read-only copy of the live episode tables (31 memories, 4 distill runs), loaded into a throwaway Postgres and run through the real backfill:
- 15 referent nodes;
- 0 memory→referent links left unresolved;
- Hecate gets exactly its 4 grounded names ("hecate", "inspur nf5288m5", "agx-2 gpu", "8x smx2 gpus");
- 6 relative names ("my boss", "my sister", "a marriott", "camera", "ogden, utah", "austin team");
- 2 accepted "mentioned together" links (Rachel–Vincent, Vincent–Jackalope bar);
- a second run was byte-identical.

Recall does not use any of this yet (spec PR D–F). This PR only builds the graph it will walk.

## Current architecture

- Stage 1 stored `episode_memory_referent (memory_id, referent_key, role)`. `validate.py` dropped the writer's aliases.
- Nothing linked a key to a substrate node.
- `memory_tension_shadow` had a writer and no reader.
- PR A shipped the claim journal, `AssertionProjector`, the reconcile fence and the cognitive-isolation rule, with no live producer.

## Architecture touched

- **orion-durable-runs persist** (same transaction, in a savepoint) → `orion/memory/referents/store.py` writes:
  - `referent_alias`;
  - `episode_memory_referent.node_id`;
  - identity questions;
  - journal proposals/decisions.
- **orion-memory-consolidation**: new `referent_projector.py` loop (Postgres → Falkor `orion_substrate`, producer `memory.referents`), which then runs PR A's `AssertionProjector`.
- **orion/substrate/falkor_store.py**: two direct reads for single-writer projectors, so this one never hydrates the whole graph.
- **Daily memory report**: new "Referents" section. It is the reader of identity questions, of which rule admitted each name, and of claims held back.

## Files changed

- `orion/memory/referents/{aliases,resolve,cooccurrence,store,backfill}.py` (new) and `README.md` (Concepts table).
- `orion/memory/episode/validate.py`: `ValidatedMemory.referent_aliases`.
- `orion/memory/episode/store.py`: `referent_policy` kwarg; savepointed `_persist_referents`.
- `orion/memory/episode/report.py`: `render_referents`.
- `services/orion-durable-runs/app/{admission_runtime,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: three kill switches (all ON).
- `services/orion-memory-consolidation/app/{referent_projector,main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: projector loop, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH`, `MEMORY_REFERENT_PROJECTOR_{ENABLED,TICK_SEC}`.
- `orion/substrate/falkor_store.py`: `prime_cache_for_producers`, `find_semantic_node_ids_by_label`.
- `orion/substrate/README.md`, `orion/core/schemas/substrate_graph_journal.py`: producers named; docstring.
- `orion/schema_skew_discovery.py`: declares durable-runs as the writer of the journal schemas consolidation reads.
- `services/orion-sql-db/manual_migration_referent_alias_v1{,_rollback}.sql` (new).
- `scripts/backfill_referents_from_episodes.py` (new; dry run by default; the AGENTS.md §14 snapshot/progress/report protocol on `--apply`).
- Tests/evals: `orion/memory/referents/tests/{test_resolve,test_referents_pg}.py`, `services/orion-memory-consolidation/tests/test_referent_projector_pg.py`, `services/orion-memory-consolidation/evals/test_referent_graph_discipline_eval.py`, `services/orion-durable-runs/tests/test_episode_distill_referent_policy.py`.
- `.github/workflows/orion-memory-episode-tests.yml`: FalkorDB service, new lanes (still fails on any skip), redis 5.2.1.
- `services/orion-memory-consolidation/requirements.txt`: `redis==5.2.1` (the Falkor client on Python 3.12).

## Concepts (each with a producer, a consumer and a test)

| Concept | Producer | Consumer | Test |
|---|---|---|---|
| Referent node (`key` row in `referent_alias`) | `persist_referents` | next resolution; projector | `test_persist_resolves_every_referent_and_keeps_juniper_s_names` |
| `episode_memory_referent.node_id` | `persist_referents` | projector memory pass | same (0 NULLs) |
| alias class `name` | `candidate_aliases` | resolution step 3; collision check | `test_a_grounded_name_resolves_a_later_key_to_the_same_node` |
| alias class `descriptor` (relative name, 90 days) | first-word frequency in `chat_history_log` prompts | collision check while live; projector display while live | `test_a_relative_name_resolves_now_and_lapses_90_days_after_last_use`, `test_a_collision_on_a_relative_name_becomes_a_question_never_a_merge`, `test_a_thing_s_own_key_name_is_a_proper_name_however_often_juniper_says_it` |
| `alias_grounding_v1` + `MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT` | `_admit_aliases` | resolution, co-occurrence, projector (live names only) | `test_the_grounding_kill_switch_flips_grounded_names_to_proposed` (flips the outcome), `test_the_kill_switches_flip_aliases_and_claims_to_proposed` |
| kind refinement (project ↔ service) | `resolve_referent` | resolution | `test_kind_refinement_files_the_same_thing_under_a_second_key` |
| identity questions (`referent_identity` / `alias_collision` / `label_collision`) | resolution; projector | daily report "Referents" | `test_a_name_on_two_nodes_mints_a_proposed_node_and_asks`, `test_the_daily_report_shows_held_claims_names_by_rule_and_open_questions`, projector test |
| `source_cooccurrence_v1` + `MEMORY_COOCCURRENCE_AUTO_ACCEPT` | `cooccurrence_claims` | `AssertionProjector` → neighborhood; report counts held claims | `test_the_cooccurrence_kill_switch_leaves_proposals_only` (flips), `test_more_than_six_claims_from_one_memory_are_all_held_for_review` (flips at 7+), projector walk test |
| `MEMORY_REFERENTS_ENABLED` | settings | `AdmissionRuntime._referent_policy` | `test_referents_disabled_skips_the_step` |
| memory Evidence node + `observed_in` provenance edge | projector | `AssertionProjector` endpoint check; neighborhood/eligibility exclusion | projector test (6 provenance edges, closed on supersede) |
| `referent_projection` ledger | projector | projector | projector test (second tick writes nothing), discipline eval (rebuild) |
| label-collision demotion | projector | identity questions → report | projector test |
| checkpoint alias recovery | `backfill_all` | `persist_referents` | `test_checkpoint_backfill_recovers_aliases_and_is_idempotent` |

**Cut, because nothing reads them in this PR:**
- the `machine` kind. Nothing behaves differently for it yet, so Hecate stays `entity_type=project`, against the spec's PR B acceptance row;
- role normalization, and roles on the graph edge (roles stay in Postgres, which is unchanged);
- `via`/`matched_text` on provenance edges;
- evidence `voice`/`channel` (the memory's voice stays in `episode_memory.voice`);
- the `referent_identity` journal kind. Identity questions live in `memory_tension_shadow` until a Stage 3 decision applier exists;
- dormant graphify/git-log candidates (spec PR C).

## Schema / bus / API changes

- **Added:**
  - Postgres `referent_alias`, `referent_projection`, and the column `episode_memory_referent.node_id`;
  - `ValidatedMemory.referent_aliases` (an in-process dataclass);
  - `FalkorSubstrateStore.prime_cache_for_producers` / `find_semantic_node_ids_by_label`.
- **Bus:** none. Journal rows are Postgres-only (PR A).
- **Behavior changed:**
  - the persist now writes referent rows (savepointed);
  - consolidation writes `memory.referents` nodes and edges into Falkor `orion_substrate`;
  - the daily report gains a section.
- **Compatibility:** no extra-forbid bus model changed. `orion/schema_skew_discovery.py` now declares the journal's cross-service writer.

## Env/config changes

- **Added keys:**
  - orion-durable-runs: `MEMORY_REFERENTS_ENABLED=true`, `MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT=true`, `MEMORY_COOCCURRENCE_AUTO_ACCEPT=true`;
  - orion-memory-consolidation: `MEMORY_REFERENT_PROJECTOR_ENABLED=true`, `MEMORY_REFERENT_PROJECTOR_TICK_SEC=30`, `FALKORDB_URI=redis://orion-athena-falkordb:6379`, `FALKORDB_SUBSTRATE_GRAPH=orion_substrate`.
- `.env_example`, settings and compose updated for both services. All flags ship ON.
- Local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys orion-durable-runs orion-memory-consolidation`. All 7 keys were added to the primary checkout's `.env`; none were skipped. Three pre-existing diverged keys were left untouched.
- `check_service_env_compose_parity`:
  - durable-runs is OK;
  - consolidation shows the same 14 pre-existing missing keys as main, and all 4 new keys are exposed.

## Tests run

```text
orion/memory/referents/tests/test_resolve.py            20 passed (pure)
orion/memory/referents/tests/test_referents_pg.py         5 passed (throwaway Postgres 16 + FalkorDB 6.0.1)
services/orion-memory-consolidation/tests/test_referent_projector_pg.py   1 passed (real Falkor + Postgres)
memory-episode CI set (episode + referents + consolidation tests + evals)   494 passed, 0 skipped
durable-runs episode tests (CI set + new policy test)    19 passed; full durable-runs tests 273 passed / 71 env-skipped
orion/substrate/tests                                     1025 passed, 3 failed (pre-existing felt_state, same on main)
Static gates (every python gate in orion-static-gates.yml incl. check_definition_drift --gate,
  schema-skew discovery after declaring the journal writer): 22/22 PASS; git diff --check clean
Live-copy dry run (read-only copy of prod episode tables -> throwaway Postgres, real backfill):
  15 nodes, 0 unresolved, Hecate 4 grounded names, 6 descriptors, 2 accepted co-occurrences,
  second run identical. The copy was deleted afterwards.
```

## Evals run

```text
services/orion-memory-consolidation/evals/test_referent_graph_discipline_eval.py   1 passed
  spec eval 5: every walkable projection backed by an accepted assertion at its revision;
  no memory.referents node outside what the referent step minted; topic-foundry "circe" untouched;
  ledger + graph wiped -> rebuild from Postgres gives identical node and edge id sets.
```

Evals 1–4 and 6–9 need recall (spec PR F).

## Docker/build/smoke checks

```text
Not run: no deploy requested, and production Falkor/Postgres must not be written. The real-store
lanes above use throwaway FalkorDB/Postgres containers. Production Falkor was only READ
(GRAPH.RO_QUERY) to count label collisions.
```

## Review findings fixed

The orchestrator runs the review subagent. Found and fixed while building:
- **Finding:** a relative name ("my boss") resolved a *different* writer key (person:dana) onto Rachel. That is a silent merge.
  - **Fix:** descriptors never decide identity; they can only collide.
  - **Evidence:** `test_a_collision_on_a_relative_name_becomes_a_question_never_a_merge`.
- **Finding (live data):** "circe", "chicago" and "space" were classed as relative names because Juniper says them often.
  - **Fix:** a thing's own key name is always a proper name; the frequency rule applies only to extra aliases.
  - **Evidence:** `test_a_thing_s_own_key_name_is_a_proper_name_however_often_juniper_says_it`, and the live-copy rerun (descriptors went from 16 to 6, all genuinely relative or generic).

- **Finding (CI):** orion-memory-consolidation pinned `redis==5.0.7`. Its graph module imports `distutils`, which Python 3.12 (the consolidation image) removed, so the projector's Falkor client would have crashed in production.
  - **Fix:** the pin moves to `redis==5.2.1` (substrate-runtime's pin); the CI lane installs the same version.
  - **Evidence:** CI run 37440072022 failed on exactly this, and `pip install --dry-run -r services/orion-memory-consolidation/requirements.txt` resolves cleanly.

## Restart required

Deploy order (PR A #2515 first):

```bash
psql "$DSN" -f services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql   # PR A
psql "$DSN" -f services/orion-sql-db/manual_migration_referent_alias_v1.sql
# PR A step 2: rebuild EVERY substrate reader first (substrate-runtime, hub, recall, cortex-exec, cortex-orch,
# spark-concept-induction, field-digester, world-pulse, meta-tags) -- old code cannot hydrate an Assertion node.
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
python scripts/backfill_referents_from_episodes.py --dsn "$DSN"            # dry run, then add --apply
```

## Risks / concerns

- **Severity: high (product decision)**
  - **Concern:** the spec's rule "a slug equal to another producer's node label means an identity question, and our node waits as proposed" fires on 6 of today's 15 live names. A read-only Falkor check found hecate, circe, juniper, orion, ogden and space already present as topic-foundry or seed nodes. So Hecate and Juniper's own nodes would start as "proposed" (not walkable), and 6 questions would open.
  - **Mitigation:** implemented exactly as the approved spec says. The alternative, keep our node walkable and only ask, is a one-line change in `_demote_on_label_collision`. **Needs Juniper's call.**
- **Severity: medium**
  - **Concern:** a co-occurrence decided at persist time stays "accepted" even if an endpoint is later demoted by a label collision.
  - **Mitigation:** the neighborhood refuses proposed endpoints, so it is not walkable; the eval covers this.
- **Severity: medium**
  - **Concern:** a second memory naming the same pair adds a proposal but no new `supports` edge (only the first decided proposal's evidence is drawn).
  - **Follow-up:** add evidence on later proposals.
- **Severity: low**
  - **Concern:** relative-name detection depends on the prompt corpus. Borderline words such as "austin" (3 of 576 prompts, 0.52%) can flip as the corpus grows. A row's class is fixed at insert, so old rows do not change.
- **UNVERIFIED:**
  - the live persist path;
  - the projector against production Falkor;
  - Falkor write latency at production graph size;
  - the backfill `--apply` on production. Not run, by instruction.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2520

🤖 Generated with [Claude Code](https://claude.com/claude-code)
