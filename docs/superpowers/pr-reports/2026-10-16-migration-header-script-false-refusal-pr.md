## Summary

- The automatic migration step that runs after a pull refused a safe migration, because a script was mentioned in the file's comments. The step read that mention as "apply this file only through this script."
- Now a script counts as the apply route only in two cases. One is a new explicit marker line, `-- ORION-MIGRATION-APPLY-WITH: scripts/<name>.sh|.py`. The other is an apply verb directly before the path, such as "Run scripts/x.sh" or "apply it through scripts/x.py". A path that is only mentioned in a sentence does not count.
- The one migration that really must go through a script (`manual_migration_gpu_pool_stage5_drop_legacy_tables.sql`) now has the marker. Its old sentence is also still recognised.
- A bad marker value, or two markers that name different scripts, is reported as a marker error. It is not silently treated as "use plain psql."
- Added regression tests: the vision-split file is now applied, the real script migration is still refused, and a test runs over every migration file and pins which files name an apply script.

## Outcome moved

On 2026-10-16 03:16 UTC the post-pull auto-apply (PR #2703) refused `manual_migration_vision_split_v1.sql` with "its header says to apply it through scripts/undo_vision_split.py". Because of that, orion-sql-writer stayed blocked by the deploy gate until the file was applied by hand. That header names the script only as the undo tool. With this fix, that file is classified `apply`. The deploy gate's printed apply command for it is now the plain `docker exec ... psql` form.

Across all 145 migration files, 18 were read as naming an apply script. 17 of them were false positives:

| File | Script it only mentioned |
|---|---|
| manual_migration_attention_broadcast_log_v1.sql | scripts/analysis/measure_ast_hot_reducer.py (analysis tool) |
| manual_migration_causal_geometry_snapshots_v1.sql | services/orion-hub/scripts/api_routes.py (reader) |
| manual_migration_curiosity_spend_v1.sql | services/orion-hub/scripts/curiosity_offer_decisions.py (writer) |
| manual_migration_drive_audits_v2.sql | scripts/drive_history_reflection_synthesis.py (consumer) |
| manual_migration_endogenous_outreach_decisions_v1.sql | scripts/endogenous_outreach.py (writer) |
| manual_migration_execution_dispatch_frame_v1.sql | scripts/check_sql_migrations_applied.py (checker) |
| manual_migration_gateway_capacity_v1.sql | scripts/gpu_pool_stage5_snapshot_and_drop.sh (test fixture; superseded) |
| manual_migration_gpu2_elastic_v1.sql | scripts/gpu_pool_stage5_snapshot_and_drop.sh (test fixture; superseded) |
| manual_migration_orion_day_correction_v1.sql | services/orion-hub/scripts/orion_day_correction_listener.py (writer) |
| manual_migration_policy_decision_frame_v1.sql | scripts/check_sql_migrations_applied.py (checker) |
| manual_migration_proposal_frame_v1.sql | scripts/check_sql_migrations_applied.py (checker) |
| manual_migration_reverie_visual_chain.sql | services/orion-hub/scripts/chat_attachments.py (incident reference) |
| manual_migration_temporal_self_event_v1.sql | scripts/check_sql_migrations_applied.py (`Check:` line) |
| manual_migration_temporal_self_v1.sql | scripts/check_sql_migrations_applied.py (`Check:` line) |
| manual_migration_thought_decision_day_citations_v1.sql | services/orion-hub/scripts/chat_turn_trace_routes.py (reader) |
| manual_migration_vision_presence_transition_v1.sql | scripts/check_sql_migrations_applied.py (`Check:` line) |
| manual_migration_vision_split_v1.sql | scripts/undo_vision_split.py (undo tool) — **the live refusal** |

Kept: `manual_migration_gpu_pool_stage5_drop_legacy_tables.sql` -> `scripts/gpu_pool_stage5_snapshot_and_drop.sh`. No other file changed classification.

## Current architecture

`orion/sql_migration_drift.py` collected the comment lines from the first 60 lines of each migration. It took the first `scripts/...\.(sh|py)` match as `FileResult.header_script`. Two places use that value. `apply_command()` prints the script as the apply route in the deploy gate and the drift card. `scripts/apply_post_pull_migrations.py::classify` refuses to auto-apply any file that has one.

## Architecture touched

- `orion/sql_migration_drift.py`: new `header_apply_script(text) -> (script, marker_error)`, the new `ORION-MIGRATION-APPLY-WITH` marker, and a narrowed prose regex. The marker is documented in the module docstring with the other in-file markers.
- `scripts/apply_post_pull_migrations.py`: docstring only. The behaviour change comes through `FileResult.header_script`.
- No bus, schema, env, or Docker changes.

## Files changed

- `orion/sql_migration_drift.py`: the detection fix and the marker.
- `scripts/apply_post_pull_migrations.py`: the docstring describes the new rule.
- `services/orion-sql-db/manual_migration_gpu_pool_stage5_drop_legacy_tables.sql`: one comment line added (the marker). No SQL changed.
- `tests/test_sql_migration_drift_gate.py`: corpus test, phrasing table, marker tests, and the vision-split regression.
- `scripts/check_migration_required_by.py`: the CI check now fails on a bad or dangling APPLY-WITH marker.
- `tests/test_deploy_migration_gate.py`: tests for that CI check.
- `tests/test_post_pull_apply_migrations.py`: end-to-end classify on the real vision-split file (`apply`) and on the real script migration (refused with its script).

## Schema / bus / API changes

- Added: in-file marker `-- ORION-MIGRATION-APPLY-WITH: scripts/<name>.sh|.py`.
- Removed: none.
- Renamed: none.
- Behavior changed: a script mentioned in prose no longer marks a migration as "apply through a script."
- Compatibility notes: the old "Run scripts/x.sh" phrasing still works when "Run" starts a sentence. A broken marker now blocks the file through the existing marker-error path, the same way a broken SUPERSEDED-BY or ABSENT-OK marker does. In that case the printed apply command says to fix the marker.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_sql_migration_drift_gate.py tests/test_post_pull_apply_migrations.py \
       tests/test_deploy_migration_gate.py tests/test_post_pull_apply_migrations_postgres.py -q
256 passed
pytest services/orion-gpu-pool/tests/test_schema_drift_gate.py -q
4 passed
New tests against the pre-fix orion/sql_migration_drift.py: 19 failed (including the
vision-split end-to-end classify), so they catch the bug.
```

## Evals run

```text
No eval harness for the migration tooling. The corpus test over every real migration file
fills that role. It pins which files name an apply script.
```

## Docker/build/smoke checks

```text
None. This is host-side tooling with no container change. Live post-pull behaviour is
UNVERIFIED until the next pull that brings a migration.
```

## Review findings fixed

- Finding (should-fix): any plain "run" still counted as an apply verb. So "Undo: run scripts/undo.py", "To verify, run scripts/check…", and "Do not run scripts/x.sh" would bring back the same false refusal.
  - Fix: "run" counts only when it starts a sentence (the imperative "Run scripts/x.sh"). "apply … via/through/with/using scripts/x" counts anywhere. A "run" after a label colon ("PRODUCTION: run …") is no longer read as an apply instruction. No real file uses that phrasing. The one existing synthetic test that used it now uses the marker.
  - Evidence: the five phrasings from the review are `None` rows in `test_phrasings`. The corpus classification is unchanged (only the gpu-pool file is detected).
- Finding (should-fix): a typo in the APPLY-WITH marker became a marker error, but the alert still printed the raw `docker exec … psql` command. That is the bypass the marker exists to prevent.
  - Fix: the marker value is now lenient. Backticks, a leading `./`, and trailing punctuation are accepted. A broken marker sets `FileResult.apply_with_error`. When that field is set, `apply_command()` returns "none until the marker is fixed: …" and never a psql line. `scripts/check_migration_required_by.py` (the CI check) now fails on a bad marker or on a marker whose script does not exist.
  - Evidence: `test_broken_marker_never_prints_a_psql_apply_command`, `test_ci_check_flags_a_bad_or_dangling_apply_with_marker`, `test_ci_check_real_corpus_apply_with_markers_are_valid`.
- Finding (low): a marker below line 60 was ignored, so the file could be applied with plain psql.
  - Fix: the marker is now read from the whole file. Legacy prose is still read only from the header.
  - Evidence: `test_marker_counts_past_line_60`.
- Finding (low): an APPLY-WITH error overwrote an earlier SUPERSEDED-BY or ABSENT-OK marker error.
  - Fix: the error messages are now joined.
  - Evidence: `test_apply_with_error_keeps_an_earlier_marker_error`.
- Finding (low): wordings outside the prose rule are not recognised.
  - Fix: the module comment says so and points to the marker. The auto-apply allowlist still refuses anything that is not additive DDL.
- Reviewer confirmed with a read-only check of the live database: editing the already-applied gpu-pool file has no side effect. The file reads APPLIED (its tables are absent), so post-pull skips it. Replay order uses the date the file first landed, and this edit does not change that date.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: a future migration that says "apply via scripts/x" with a wording the regex does not cover would be auto-applied with plain psql.
- Mitigation: use the explicit `ORION-MIGRATION-APPLY-WITH` marker. The auto-apply still refuses any DROP, DELETE, or DO block on its own, and the gpu-pool file is refused for its DROPs regardless of this marker.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
