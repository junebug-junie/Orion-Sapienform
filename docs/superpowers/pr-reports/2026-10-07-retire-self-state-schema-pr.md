## Summary

- Deletes the last pieces of SelfStateV1 (Orion's old "how am I doing" self-report). Its producer was deleted on 2026-07-22; the schema module and every reader of its empty tables were still hanging around.
- Removes six "always empty" readers that ran every tick or every turn and never got data: the brain-frame Self-state region, the causal-geometry `self_state_predictions` source, the chat-stance self-state hazard, the Hub self-revision mutation lane, the `self_state` context-provenance entry, and a blank `self_state_id:` line in three probe prompts. In the probe prompts the blank line now shows the field tick id that actually arrives.
- Retires 5 env keys and their compose/settings wiring, and re-locks the metric definitions. The 5 removed metric nodes are the only change to the semantic layer.
- Does **not** drop any database table or column. A rollback-safe drop proposal is below.

## Outcome moved

Dead code paths that looked live are gone. Before this, a reader saw a "Self-state" tab in the Self brain EKG, a self-state hazard in chat stance, a self-revision mutation lane, and a `self_state_predictions` causal source. The four user-visible ones were wired up and always came back empty (six readers in total, per the table below). The substrate-runtime brain-frame tick also stops querying an empty table every 5 s.

## Current architecture

SelfStateV1's producer (`orion-self-state-runtime`), `orion/self_state/` and `config/self_state/` were deleted 2026-07-22 (bcc72f6a0 / PR #1266). Still present on main:

- `orion/schemas/self_state.py` (SelfStateV1, SelfStateDimensionV1, AttentionTargetSummaryV1). Only imported by `orion/schemas/registry.py`, `orion/inner_state_registry.py`, two dead scripts and a schema test. **None** of the modules in the original brief (collapse, consolidation motif/tensorize/windows, execution_dispatch, autonomy) import it. They only mention it in burn comments.
- `orion/schemas/identity_snapshot.py`, `orion/schemas/self_state_prediction.py`: not imported anywhere. The live `IdentitySnapshotV1` is a different class in `orion/core/schemas/drives.py`.
- Readers of empty tables: the substrate-runtime brain frame (`substrate_self_state`) and causal geometry (`self_state_predictions`).
- A dead belief projection in chat stance (it read `self:*` nodes from the deleted `self_state_ctx` adapter).
- A Hub self-revision stub that always returned `[]`, plus 3 env keys.

## Classification (every reference)

| Reference | Class | Evidence | Action |
|---|---|---|---|
| `orion/schemas/self_state.py` + registry entries | dead | importers listed above only | deleted |
| `orion/schemas/identity_snapshot.py`, `self_state_prediction.py` | dead | zero importers | deleted |
| `inner_state_registry` `self_state.v1` entry | dead | REHEARSAL, both ends gone, table 0 rows | removed (4 lineage URNs) |
| Brain-frame `self_state` region (Literal, producer, worker SQL, setting, env, lineage, Hub tab) | live-but-always-empty | `substrate_self_state` 0 rows; 0 of 4,743 frames in `substrate_brain_frame_log` carry it | removed (1 lineage URN) |
| Causal-geometry `self_state_predictions` source | live-but-always-empty | 0 rows; live snapshots note "returned 0 rows" every run | removed |
| `chat_stance._project_self_state_from_beliefs` + `SELF_STATE_STANCE_PRESSURE_THRESHOLD` | live-but-always-empty | producer adapter deleted 07-22; 0 of 53,885 `chat_stance_belief_log` rows (30 d) and 0 log hazard lines (14 d) | removed |
| `context_provenance` `self_state` entry | live-but-always-empty | 0 of 65 cortex-exec "Context Keys available" lines (7 d) carry the key | removed |
| Hub `_self_revision_signals_from_latest_self_state` + 3 `SUBSTRATE_AUTONOMY_SELF_REVISION_*` keys | live-but-always-empty | stub body is `return []` | removed |
| Probe prompts `self_state_id: {{ self_state_id }}` | live-but-always-empty | dispatch envelope sends `field_tick_id`, never `self_state_id` | now renders `field_tick_id` |
| `introspect_spark.j2` `spark_meta.self_state_v1` block | dead | nothing sets it | removed |
| `scripts/backfill_phi_corpus.py` | dead (already broken) | imports deleted spark-introspector file; source table empty | deleted |
| `scripts/analysis/measure_self_state_signal_quality.py` + test | dead | source table empty | deleted |
| `scripts/smoke_self_state_v1.sh` | dead | queries empty table | deleted; 3 other smokes now select `source_field_tick_id` |
| Proposal `target_kind="self_state"` (metacog/reverie/templates/proposal_frame Literal) | **live-and-meaningful** | 40,162 of 40,162 proposal frames in the last day contain a `self_state` candidate | kept |
| Field glossary `self_state_dimension` key | **live-and-meaningful** | documents `orion/field/pressure.py::CHANNEL_DIMENSION_MAP`, gated by `test_self_state_dimension_matches_channel_dimension_map_exactly`, rendered in the Hub glossary | kept (name is historical) |
| `CompositionStatus.COMPOSED = "composed_into_self_state"` | live label | string recorded in 4 lock notes | kept; docstring now says the name is historical |
| `InnerStateFeaturesV1.self_state_id`, `MoodArcCorpusRowV1.self_state_id` | historical-artifact readers | extra="forbid" corpus readers for old JSONL | kept |
| `hub_presence.py` | **live** | read by Hub substrate observability route | kept, docstring corrected |
| `scripts/analysis/measure_autonomy_gate.py` self-state section | live-but-always-empty | reads `substrate_self_state`; vendored into Hub image | kept, listed as follow-up (must go before the table drop) |
| `source_self_state_id` columns on 4 frame tables | always NULL | 0 non-null in samples of all 4 tables (rows retained since 2026-07-23); proposal frames 0 of 402,661 | kept, drop proposed below |
| Bus channels | none left | `orion:substrate:self_state` already removed 07-22 | no change |

## Consumer impact (before/after on real data)

- **Metric semantic layer** (`check_metric_lineage --json`, main vs branch): 698 to 693 URNs. Exactly 5 removed (`brain_region/.../self_state` and the 4 `inner_state/.../self_state.v1*` nodes), 0 added, **0 changed** on every other URN. Inner-state orphans went from 13 to 11; every other surface is unchanged.
- **Causal geometry** (`fetch_channels`, live DB, 6 h window): main had tables `{attention_salience_trace:177, orion_biometrics_summary:691, self_state_predictions:0, substrate_field_state:10241}` and 19 channels. The branch has the same three live tables and **19 channels** (the field_state count differs by 2 rows because the window slid between runs). The only output change is that the "self_state_predictions returned 0 rows" note disappears from snapshots.
- **Brain frame**: regions are identical. The self_state branch had already returned `[]` on every frame (0 of 4,743 logged frames).
- **Consolidation tensorize / motif / windows, proposal scoring, inner-state registry gate**: no code changes. Their SelfStateV1 iteration was already removed 07-22. The focused root suite passes and fails identically on main and branch (444 passed, same 4 pre-existing failures).
- **Probe prompts**: the provenance line used to render blank and now names the real field tick id. The LLM sees one more real id. No other prompt text changed.

## Files changed

- `orion/schemas/{self_state,identity_snapshot,self_state_prediction}.py`: deleted
- `orion/schemas/registry.py`, `orion/inner_state_registry.py`, `scripts/check_inner_state_registry.py`: entries removed
- `orion/schemas/brain_frame.py`, `services/orion-substrate-runtime/app/{brain_frame_producer,worker,settings}.py`, `.env_example`, `README.md`: self_state region retired
- `orion/metrics/lineage.py`, `config/metrics/metric_definitions.lock.json`: brain-region table and lock (re-locked; definition change)
- `orion/substrate/causal_geometry_engine.py`, `scripts/causal_geometry_report.py`: empty source removed
- `orion/schemas/context_provenance.py`: `self_state` entry removed
- `services/orion-cortex-exec/app/chat_stance.py`, `.env_example`: dead projection and key removed
- `services/orion-hub/scripts/api_routes.py`, `.env_example`, `docker-compose.yml`: self-revision lane removed
- `services/orion-hub/static/js/self-brain.js`, `scripts/self_brain_routes.py`: Self-state tab removed
- `orion/cognition/prompts/substrate_{inspect,observe,summarize}.j2`, `introspect_spark.j2`: dead variables
- `scripts/backfill_phi_corpus.py`, `scripts/analysis/measure_self_state_signal_quality.py` (+test), `scripts/smoke_self_state_v1.sh`: deleted
- `scripts/smoke_{proposal_frame,policy_decision_frame,consolidation}_v1.sh`: no longer read self-state table/column
- Tests updated or pinned: see "Tests run"
- Comment/doc pointers to deleted files: `mood_arc.py`, `hub_presence.py`, `sentience_striving_program/README.md`, analysis test docstrings, two backfill script comments, `run_scripts_platform_shadow_blast_radius_eval.py`

## Schema / bus / API changes

- Removed: `SelfStateV1`, `SelfStateDimensionV1`, `AttentionTargetSummaryV1`, the unused `orion.schemas.identity_snapshot.IdentitySnapshotV1`, `SelfStatePredictionV1`; `BrainRegionV1.dimension` value `"self_state"`; Hub mutation-scheduler summary keys `self_revision_enabled`/`self_revision_signals`; `context_provenance` key `self_state`.
- Behavior changed: probe prompts render `field_tick_id` where they used to render a blank `self_state_id`. Proven at the dispatch envelope (`orion/execution_dispatch/envelopes.py` sends `field_tick_id`) and in a unit render; a live rendered prompt showing it is UNVERIFIED (no deploy in this PR).
- Compatibility: narrowing the `BrainRegionV1.dimension` Literal is safe in either deploy order. No producer emits `self_state` (its table is empty) and no retained frame contains it. No bus channel changes. No Hub JS reads the removed summary keys.

## Env/config changes

- Removed keys: `SELF_STATE_STANCE_PRESSURE_THRESHOLD` (cortex-exec), `BRAIN_FRAME_SELF_STATE_CADENCE_SEC` (substrate-runtime), `SUBSTRATE_AUTONOMY_SELF_REVISION_ENABLED` / `_MIN_ERROR` / `_MAX_AGE_SEC` (hub, plus compose lines).
- `.env_example` updated: yes, 3 services.
- Local `.env` synced: ran `python scripts/sync_local_env_from_example.py orion-hub orion-cortex-exec orion-substrate-runtime` ("No changes needed"; the script only adds keys). Then removed the 5 retired keys by hand from the primary checkout's `.env` files. The values matched the old code defaults, so running main is unaffected.
- Skipped keys requiring operator action: none.

## Tests run

```text
check_metric_lineage.py --gate                PASS (693 URNs)
check_definition_drift.py --gate              PASS after --update (5 high: the 5 removals above, 0 other)
check_inner_state_registry.py                 OK (15 entries)
check_env_template_parity.py                  PASS
git diff --check                              clean
cortex-exec chat_stance/grounding/probe files all pass, except test_main_autonomy_graph_probe (5) and
  test_substrate_felt_state_reader (1), which fail identically on main
substrate-runtime tests+evals                 same 26 pre-existing DB-dependent failures/errors as main, nothing new
hub self_brain/observability/mutation/glossary 68 passed, 1 pre-existing failure (also on main)
node --test static/js/self-brain.test.js      7 pass
root focused suite (proposal/policy/feedback/consolidation/execution_dispatch/registry/...) 444 passed,
  same 4 pre-existing failures as main
tests/test_metric_lineage.py, orion/harness/tests, orion/schemas/tests: pass except
  test_context_provenance::test_static_ctx_assignments_covered (pre-existing on main: 7 unrelated new ctx keys)
```

## Evals run

```text
services/orion-substrate-runtime/evals/test_brain_frame_substance_eval.py   2 passed
```

## Docker/build/smoke checks

```text
scripts/smoke_{proposal_frame,policy_decision_frame,consolidation}_v1.sh against live DB: exit 0
No container build run (no deploy requested).
```

## Review findings fixed

Review ran as a subagent (`/code-review`, target `git diff origin/main...HEAD`). No blockers.

- Finding: Hub README still said the region-provenance route had 6 dimensions/entries.
  - Fix: changed to 5 / the other 4 / 5 entries.
  - Evidence: `services/orion-hub/README.md` region-provenance bullet.
- Finding: the claim that the probe prompt renders `field_tick_id` live was not runtime-proven.
  - Fix: marked UNVERIFIED above, and the envelope evidence is cited.
  - Evidence: "Schema / bus / API changes" section.
- Finding: dead test helper `_beliefs_with_strained_self_state`, and `import pytest` placed above the stdlib imports.
  - Fix: deleted the helper; moved the import.
  - Evidence: cortex-exec test files, all re-run green.
- Finding: the stale `inner_state_registry.md` note and the `lineage.py` "6 dimensions" docstring.
  - Fix: added a 2026-10-07 line to each.
- Finding: the proposal assert in the retired-lane test passes no matter what.
  - Fix: the docstring now says the hasattr and summary-key checks are the real guards.
- Finding: the reader counts in the Outcome section read as inconsistent.
  - Fix: reworded.

## Restart required

Rebuild and restart from the primary checkout on main after merge:

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build && scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## DB objects left for a follow-up (proposal, not applied)

Every object below is empty or all-NULL, with no foreign keys and no dependent views (checked live 2026-10-07):

- Tables: `substrate_self_state` (0 rows), `identity_snapshots` (0 rows), `self_state_predictions` (0 rows), plus their indexes.
- Columns: `source_self_state_id` on `substrate_proposal_frames`, `substrate_policy_decision_frames`, `substrate_execution_dispatch_frames`, `substrate_feedback_frames` (nullable, NULL on every sampled row).

Consumer-first order. Every reader has to be gone before anything is dropped:

1. Already done here: brain frame, causal geometry and the smoke scripts no longer read these.
2. Still reading, retire first: `scripts/analysis/measure_autonomy_gate.py` (self-state section, vendored in the Hub image). Also the proposal/policy/dispatch/feedback `*_store.py` legacy-row tests use `source_self_state_id` only inside JSON payload fixtures, not as the column.
3. Then a migration `manual_migration_retire_self_state_v1.sql`:

```sql
-- forward (each statement is independent; all objects are empty / all-NULL)
BEGIN;
DROP TABLE IF EXISTS substrate_self_state;
DROP TABLE IF EXISTS identity_snapshots;
DROP TABLE IF EXISTS self_state_predictions;
ALTER TABLE substrate_proposal_frames          DROP COLUMN IF EXISTS source_self_state_id;
ALTER TABLE substrate_policy_decision_frames   DROP COLUMN IF EXISTS source_self_state_id;
ALTER TABLE substrate_execution_dispatch_frames DROP COLUMN IF EXISTS source_self_state_id;
ALTER TABLE substrate_feedback_frames          DROP COLUMN IF EXISTS source_self_state_id;
COMMIT;

-- rollback: re-run the original create migrations (no data to restore):
--   services/orion-sql-db/manual_migration_self_state_v1.sql
--   services/orion-sql-db/manual_migration_identity_snapshot_v1.sql
--   services/orion-sql-db/manual_migration_self_state_prediction_v1.sql
-- and ALTER TABLE ... ADD COLUMN source_self_state_id text (nullable) on the 4 frame tables.
```

Precheck before running: a bounded `count(*) ... WHERE source_self_state_id IS NOT NULL` per frame table, and `count(*)` on the three tables. Abort if anything is non-zero. `DROP COLUMN` on the frame tables (2-7 GB each) only updates the catalog, but it takes an ACCESS EXCLUSIVE lock. Run it off-peak with `lock_timeout` set.

## Risks / concerns

- Severity: low. Concern: anyone with `SUBSTRATE_AUTONOMY_SELF_REVISION_ENABLED=true` in a host `.env` loses nothing, because the lane already produced nothing. Pinned by `test_scheduler_self_revision_lane_is_retired`.
- Severity: low. Concern: the names `CompositionStatus.COMPOSED` ("composed_into_self_state") and the glossary key `self_state_dimension` still carry the retired name. Renaming them is a definition change (lock notes, Hub glossary API), so it was deliberately left out.
- Severity: low. Concern: `orion/substrate/mutation_worker.py`'s generic `extra_signals` parameter now has no production caller. Kept because it is a generic, tested seam that does not depend on self-state.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2531

🤖 Generated with [Claude Code](https://claude.com/claude-code)
