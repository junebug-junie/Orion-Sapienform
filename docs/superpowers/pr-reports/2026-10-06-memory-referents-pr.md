## Summary

This is memory Stage 2 PR B, stacked on PR A (#2515). It makes the things Orion's memories are about into nodes in Orion's one graph, under the names Juniper actually uses for them. Spec: `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md` (approved 2026-10-06). Every name in this report and its tests is synthetic; live data is reported as counts only.

- **Names stop being thrown away, and the distiller says what kind each one is.** Prompt v4 asks for `alias_kind` on every name, including the key's own name:
  - `proper_name` is a name for exactly one thing, e.g. a machine's name or a town's name;
  - `descriptor` is a role or relation that can point at different things over time, e.g. "my boss".

  Code never classifies a name by its words. A name with no judgment (every answer from before v4) counts as a descriptor.
- **Only proper names can tie a key to an existing thing.** A key resolves onto an existing node only in two ways:
  - by its exact key;
  - by a grounded proper name that matches exactly one node of the same kind, provided the key's own proper name doesn't point elsewhere.

  Anything else that touches another node becomes a question. Descriptors never decide identity.
- **`alias_grounding_v1`**: a name is usable only if it appears in Juniper's verified words. Descriptors lapse 90 days after their last use, and reuse extends them.
- **`source_cooccurrence_v1`**: two things named in one of Juniper's verified sentences get a walkable "mentioned together" link. Juniper's and Orion's own nodes are excluded by node id, and a memory yields at most 6 such links.
- **A projector in orion-memory-consolidation** writes the nodes, one evidence node per memory, and "mentioned in" provenance edges.
  - It writes nothing until every substrate reader has advertised that it can read these shapes. This is the same gate PR A's `AssertionProjector` uses. `/health` shows which readers are missing.
  - A same-named node from another producer triggers a question; our node stays walkable.
  - The rebuild script replays accepted assertions too.
- **Checkpoint backfill** recovers dropped names under the §14 protocol: the dry run computes what would change and rolls it back, progress is logged live, and a failing episode is counted while the run continues.

## Outcome moved

Before this PR, the referent keys in Stage 1 memories led nowhere. After it, each key resolves to one graph node with Juniper's grounded names attached. Names only merge things on the distiller's explicit proper-name judgment, and every collision becomes a question.

Recall does not use any of this yet (spec PR D–F).

## Current architecture

- Stage 1 stored `episode_memory_referent (memory_id, referent_key, role)`.
- The validator dropped aliases.
- `memory_tension_shadow` had no reader.
- PR A provides the claim journal, `AssertionProjector`, the reconcile fence, cognitive isolation and the reader-capability gate.

## Architecture touched

- **Prompt and schema:** `memory_episode_distill.j2` is now v4 with `alias_kind`; `DistillReferentV1` and `DistillAliasV1` are lenient (`extra="ignore"`, bare strings coerced to descriptors).
- **orion-durable-runs persist,** inside a savepoint, writes:
  - `referent_alias`;
  - `episode_memory_referent.node_id`;
  - identity questions;
  - journal proposals and decisions.

  It no longer reads `chat_history_log` at all.
- **orion-memory-consolidation:** `referent_projector.py`, gated on reader readiness; `/health` gains `referent_projector`.
- **The daily memory report** gains a "Referents" section.
- **Scripts:** `scripts/backfill_referents_from_episodes.py` and `scripts/rebuild_referent_graph.py`.

## Files changed

- `orion/cognition/prompts/memory_episode_distill.j2`, `orion/schemas/memory_episode.py`: `alias_kind` and v4.
- `orion/memory/episode/validate.py`: keeps `(text, alias_kind)` per alias plus each key's own kind.
- `orion/memory/episode/store.py`: savepointed referent step.
- `orion/memory/episode/report.py`: Referents section.
- `orion/memory/referents/{aliases,resolve,cooccurrence,store,backfill}.py` and `README.md`.
- `services/orion-durable-runs/app/{admission_runtime,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-memory-consolidation/app/{referent_projector,main,settings}.py`, `.env_example`, `docker-compose.yml`, `requirements.txt` (redis 5.2.1), `README.md`.
- `orion/substrate/falkor_store.py`: `prime_cache_for_producers`, `find_semantic_node_ids_by_label`.
- `orion/schema_skew_discovery.py`: declares the journal's writer.
- `services/orion-sql-db/manual_migration_referent_alias_v1{,_rollback}.sql`.
- `scripts/backfill_referents_from_episodes.py`, `scripts/rebuild_referent_graph.py`.
- Tests/evals:
  - `orion/memory/referents/tests/{test_resolve,test_referents_pg}.py`;
  - `services/orion-memory-consolidation/tests/test_referent_projector_pg.py`;
  - `services/orion-memory-consolidation/evals/test_referent_graph_discipline_eval.py`;
  - `services/orion-durable-runs/tests/test_episode_distill_referent_policy.py`;
  - prompt-version pins in `test_distill_prompt.py` and `test_episode_distill_graph.py`.
- `.github/workflows/orion-memory-episode-tests.yml`: FalkorDB service and new lanes; fails on any skip.

## Concepts (each with a producer, a consumer and a test)

The full table, with plain-English meanings, is in `orion/memory/referents/README.md` and `services/orion-memory-consolidation/README.md`.

| Concept | Producer | Consumer | Test |
|---|---|---|---|
| `alias_kind` | distiller v4 → validator | identity rule; descriptor expiry | `test_unjudged_names_are_descriptors`, both review repros |
| identity rule | `resolve_referent` | resolution | `test_the_same_descriptor_for_two_people_is_a_question_not_a_merge`, `test_a_misjudged_proper_name_still_cannot_override_the_key_s_own_name`, `test_a_descriptor_key_never_resolves_another_key` |
| descriptor expiry | `_admit_aliases` | collisions and projector display (live only) | `test_a_descriptor_lapses_90_days_after_last_use_and_is_refreshed_by_reuse` |
| `alias_grounding_v1` + switch | `_admit_aliases` | resolution, co-occurrence, projector | `test_the_grounding_kill_switch_flips_grounded_names_to_proposed` (flips) |
| `source_cooccurrence_v1` + switch | `cooccurrence_claims` | `AssertionProjector`; report | `test_the_cooccurrence_kill_switch_leaves_proposals_only` (flips), 6-cap test, node-id self-exclusion test |
| identity questions | resolution; projector | daily report | report test, projector test |
| readiness gate | readers' advertisement (PR A) | `ReferentProjector`, `/health` | `test_projector_writes_nothing_until_every_reader_is_ready`, `test_health_reports_the_missing_readers` |
| `referent_projection` ledger + `rebuild()` | projector | projector | projector test; discipline eval (rebuild into an empty graph gives the same ids and adds no journal rows) |
| §14 backfill | `backfill_all` + CLI | `persist_referents` | `test_checkpoint_backfill_recovers_aliases_and_is_idempotent`, `test_backfill_cli_dry_run_writes_nothing_logs_progress_and_survives_a_bad_episode` |

**Removed in review:**
- the prompt-frequency classifier (`TokenFrequency`, `RARE_DF_FRAC`, the per-token `to_tsvector` scans in the persist transaction);
- kind refinement (project↔service). Cross-kind same names now become a question.

**Cut earlier, because nothing reads them yet:** `machine` kind, role normalization, provenance `via`/`matched_text`, evidence `voice`/`channel`, the `referent_identity` journal kind.

## Schema / bus / API changes

- **Added:**
  - Postgres `referent_alias`, `referent_projection`, and the column `episode_memory_referent.node_id`;
  - `DistillAliasV1`, plus `alias_kind` on `DistillReferentV1` (LLM-output models, `extra="ignore"`, old answers still parse);
  - prompt version `memory_episode_distill.v4`.
- **Bus:** none.
- **Behavior:**
  - the persist writes referent rows in a savepoint;
  - consolidation writes `memory.referents` shapes into Falkor, gated on reader readiness;
  - `/health` and the report show referent status.

## Env/config changes

- **orion-durable-runs:** `MEMORY_REFERENTS_ENABLED=true`, `MEMORY_ALIAS_GROUNDING_AUTO_ACCEPT=true`, `MEMORY_COOCCURRENCE_AUTO_ACCEPT=true`.
- **orion-memory-consolidation:** `MEMORY_REFERENT_PROJECTOR_ENABLED=true`, `MEMORY_REFERENT_PROJECTOR_TICK_SEC=30`, `FALKORDB_URI=redis://orion-athena-falkordb:6379`, `FALKORDB_SUBSTRATE_GRAPH=orion_substrate`, `SUBSTRATE_ASSERTION_REQUIRED_READERS=<the 9 reader services>`.
- All flags ship ON.
- **Local `.env`:** synced with `sync_local_env_from_example.py --all-keys`. Every key was added; none were skipped. The pre-existing diverged keys were left alone.
- **Env/compose parity:** durable-runs is OK. Consolidation shows the same 14 missing keys as main, and every new key is exposed.

## Tests run

```text
memory-episode CI set (episode + referents + consolidation tests + evals; real Postgres + FalkorDB)  498 passed, 0 skipped
  incl. orion/memory/referents/tests/test_resolve.py 20, test_referents_pg.py 6, projector 3, discipline eval 1
durable-runs tests                                       273 passed, 71 env-skipped (same as main)
orion/substrate/tests (after merging #2515's fixes)      1037 passed, 3 failed (pre-existing felt_state, also on main)
static gates (every python gate in orion-static-gates.yml, incl. check_definition_drift --gate)  all PASS; git diff --check clean
```

## Evals run

```text
services/orion-memory-consolidation/evals/test_referent_graph_discipline_eval.py   1 passed
  every walkable projection backed by an accepted assertion at its revision; no memory.referents
  node outside what the referent step minted; the other producer's same-named node untouched;
  ReferentProjector.rebuild() into an EMPTY graph -> identical node and edge ids, 0 new journal rows.
```

## Docker/build/smoke checks

```text
Not run: no deploy requested; production Falkor/Postgres not written. The backfill --apply was not
run anywhere near production. Real-store lanes use throwaway FalkorDB 6.0.1 / Postgres 16.
```

## Review findings fixed (#2520 review)

- **HIGH 1:** the word-frequency classification was inverted ("boss" became a permanent proper name, so two people with that name merged).
  - **Fix:** it is removed. The distiller emits `alias_kind`, defined in the prompt with examples (no word lists). Only a grounded proper name of the same kind can resolve a key; descriptors never do.
  - **Evidence:** `test_the_same_descriptor_for_two_people_is_a_question_not_a_merge`, plus `test_a_misjudged_proper_name_still_cannot_override_the_key_s_own_name` for the case where the distiller mislabels a name.
- **HIGH 2:** "a key's own name is always proper" let a descriptor key swallow a person.
  - **Fix:** the key's own name uses the distiller's `alias_kind` like any alias, and a descriptor-derived key never resolves another key.
  - **Evidence:** `test_a_descriptor_key_never_resolves_another_key`.
- **HIGH 3:** the projector flag was ON with no sequencing guard.
  - **Fix:** one mechanical gate shared with PR A. Readers advertise `assertion_core_v1` from their off-path bootstrap. The projector writes nothing until all of `SUBSTRATE_ASSERTION_REQUIRED_READERS` have advertised, and it logs `referent_projector_waiting reason=readers_not_ready missing=[...]`, which `/health` also shows.
  - **Evidence:** `test_projector_writes_nothing_until_every_reader_is_ready`, `test_health_reports_the_missing_readers`, and PR A's `test_reader_capability.py`.
- **HIGH 4 (privacy):** real names were in fixtures and docs.
  - **Fix:** fixtures, tests, docstrings, READMEs, this report and the PR body now use synthetic names, and live figures are counts only. The real names remain in already-pushed history (see "Pushed commits with real names").
- **MEDIUM 5:** a label collision demoted our node.
  - **Fix:** the node stays walkable and a question is asked; `admitted_by` is recorded on the question only. The question id is deterministic, so a rebuild re-asks nothing.
  - **Evidence:** projector test (Hecate→Circe is walkable).
- **MEDIUM 6:** normalization was inconsistent.
  - **Fix:** one rule for keys and aliases: all punctuation becomes a space.
  - **Evidence:** `test_one_normalization_for_keys_and_aliases`, and `test_a_place_name_with_punctuation_never_splits_into_a_second_node` (100 days later, the same node).
- **MEDIUM 7:** the backfill did not really follow §14.
  - **Fix:** live progress lines (percent, ETA, processed/total, rate, errors), per-episode savepoints with errors counted, a dry run that computes and rolls back, and `report.md` plus `before_after.csv` in both modes.
  - **Evidence:** `test_backfill_cli_dry_run_writes_nothing_logs_progress_and_survives_a_bad_episode`.
- **MEDIUM 8:** the rebuild was underspecified.
  - **Fix:** `ReferentProjector.rebuild()` and `scripts/rebuild_referent_graph.py`. Assertions are replayed from the journal's latest applied decisions, with no new journal rows.
  - **Evidence:** discipline eval.
- **LOW 9:** self-exclusion compared keys.
  - **Fix:** it compares node ids.
  - **Evidence:** `test_juniper_and_orion_are_excluded_by_node_id_not_key`.
- **LOW 10:** the per-token prompt scans were inside the persist transaction.
  - **Fix:** removed entirely; the persist no longer touches `chat_history_log`.
- **Earlier (CI):** the consolidation redis pin moves to 5.2.1, because redis 5.0.x imports `distutils`, which Python 3.12 removed.

## Pushed commits with real names

Not force-pushed. The history needs scrubbing separately:
- **PR B:** `2d39c3856`, `a9e749016`, `1e39bdbba`, and `cb79c338f` (the last one only because it left the old report file in place; fixed in the next commit).
- **PR A:** `f44c3e948`.
- **Spec branch (#2496):** the spec file itself names real people and places, including in my `4ae3279fb`, which only added the Decisions section.
- **Already on main before this work:** `services/orion-durable-runs/tests/test_episode_distill_graph.py` contains a real travel destination.

## Restart required

Rollout (A + B), mechanical rather than deploy-order dependent:

```bash
psql "$DSN" -f services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql
psql "$DSN" -f services/orion-sql-db/manual_migration_referent_alias_v1.sql
# phase 1: every substrate reader on #2515's code (order free); each advertises assertion_core_v1
scripts/safe_docker_build.sh <reader> up -d --build      # substrate-runtime hub recall cortex-exec cortex-orch spark-concept-induction field-digester world-pulse meta-tags
# phase 2: writers; they wait (logged, /health) until phase 1 is complete, then open on their own
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
python scripts/backfill_referents_from_episodes.py --dsn "$DSN"          # dry run (rolled back), then --apply
```

## Risks / concerns

- **Severity: medium**
  - **Concern:** old answers (prompt v3 and earlier, i.e. the backfill) carry no `alias_kind`, so every recovered name is a descriptor. They lapse after 90 days and never resolve a key; only exact keys link memories to nodes.
  - **Mitigation:** this is safe by design. New distills carry real judgments.
- **Severity: medium**
  - **Concern:** an alias's class is frozen at insert. A later proper/descriptor relabel by the distiller does not change it. A frozen descriptor still never decides identity.
- **Severity: medium**
  - **Concern:** the readiness gate is only as good as the required list. A substrate reader missing from `SUBSTRATE_ASSERTION_REQUIRED_READERS` is not waited for. A reader rolled back to old code keeps its key until someone deletes `orion:substrate:reader_capability:<name>`.
- **Severity: low**
  - **Concern:** a second memory naming the same pair adds a proposal but no new `supports` edge.
- **UNVERIFIED:**
  - the live persist path;
  - distiller v4 output quality on `alias_kind` (no live distill was run);
  - the projector against production Falkor and its write latency;
  - backfill `--apply` (not run, by instruction).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2520

🤖 Generated with [Claude Code](https://claude.com/claude-code)
