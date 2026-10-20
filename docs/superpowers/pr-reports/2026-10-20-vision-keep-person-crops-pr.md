## Summary

- Walkway person crops no longer get deleted after 7 days. Their rows and embeddings are the labelled examples the person-learning design (PR #2767, Phase 0) needs. Without this change, the examples Juniper labelled were being deleted every day.
- A new setting, `VISION_CROP_RETENTION_KEEP_LABELS=person` (orion-sql-writer), lists the crop labels the prune skips. It ships ON. `0` or empty brings back the old rule. Vehicles, bikes, strollers and dogs still get deleted after 7 days.
- Review fix: the split audit field `sightings_out_of_retention` now counts only old sightings whose crops are really gone. Before, it would have counted kept person sightings that the split actually used.
- Comments and docstrings that said every crop is deleted after 7 days now say person crops are kept.

## Outcome moved

The crop prune keeps person crops. Labelled people stop losing their examples. Live 2026-10-20: 6,206 person crops, the oldest from 2026-10-15. On the old rule they would start disappearing from 2026-10-22.

## Current architecture

`app/vision_individuals.py` `_prune` ran every individuals tick (60 s) and deleted up to 20,000 `vision_crop_observation` rows older than `VISION_CROP_RETENTION_DAYS` (7), whatever their label.

## Architecture touched

orion-sql-writer only (plus a comment in orion-hub). No schema change, no migration, no bus change.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: `IndividualsConfig.crop_retention_keep_labels`; the prune skips those labels.
- `services/orion-sql-writer/app/vision_individuals_loop.py`: parses the comma list (`0`/empty means none).
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `README.md`: the new key.
- `services/orion-sql-writer/app/vision_split.py`: `sightings_out_of_retention` counts only sightings with no surviving crop.
- `services/orion-sql-writer/tests/test_vision_confirm_asks.py`: tests that the prune keeps or deletes by label (real Postgres), and tests the env parsing.
- `services/orion-sql-writer/tests/test_vision_split.py`: regression test for the audit count.
- `services/orion-sql-writer/app/vision_motion_loop.py`, `evals/run_vision_body_match_replay_eval.py`, `scripts/backfill_story_evidence_links.py`, `services/orion-hub/scripts/motion_story_routes.py`: comments only.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the crop prune skips the labels in the keep list. `sightings_out_of_retention` is now an exact count.
- Compatibility notes: crops deleted before this deploy are gone. The change only protects crops that still exist.

## Env/config changes

- Added keys: `VISION_CROP_RETENTION_KEEP_LABELS=person` (orion-sql-writer)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: yes, added by hand to the primary checkout's `services/orion-sql-writer/.env`. The sync script runs against the worktree, and the worktree has no `.env`.
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-sql-writer/tests -k "vision or settings"   -> 287 passed, 76 skipped (no PG)
ORION_TEST_PG_URI=<throwaway pg16> pytest tests/test_vision_confirm_asks.py  -> 42 passed
ORION_TEST_PG_URI=<throwaway pg16> pytest tests/test_vision_split.py         -> passed (incl. new test)
Mutation: prune without the label filter -> keep test fails; old oor query -> split test fails.
Wider vision suite on PG: the same 10 failed + 5 errors in story-strip tests on main and on this branch
(those migrations were applied out of order in the ad-hoc database). Not caused by this patch.
python scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
None. This is a retention rule with no quality dimension. The Phase 1 eval (run_person_recognition_eval.py) is where the kept crops get measured.
```

## Docker/build/smoke checks

```text
Not built. No dependency or compose change.
Live read-only: EXPLAIN of the new prune query uses vision_crop_observation_stream_time_idx with a label filter.
Person crops/day (live): 4778 (10-15 burst), 1036, 69, 49, 259; ~5 KB/row -> about 1-2 GB/year.
```

## Review findings fixed

- Finding: `sightings_out_of_retention` assumed every crop older than 7 days was gone. It was wrong for person individuals.
  - Fix: count only sightings with no surviving crop in their time span.
  - Evidence: `test_out_of_retention_counts_only_sightings_whose_crops_are_gone` fails on the old query and passes on the new one.
- Finding: stale comments said every crop is deleted after 7 days.
  - Fix: updated in 5 files.
  - Evidence: diff.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d
```

Phase 0 acceptance after restart: `SELECT min(observed_at) FROM vision_crop_observation WHERE label='person'` stops moving forward once 7 days pass, and the vehicle minimum keeps moving forward.

## Risks / concerns

- Severity: low
  - Concern: the prune scans past every kept old person row on each 60 s tick (one heap fetch per row).
  - Mitigation: about tens of ms per tick at 100k rows (review estimate). If it grows, add a partial index on `observed_at` that excludes person rows.
- Severity: low
  - Concern: a split now loads every crop of a person individual (no 7-day cap). A frequent visitor could have about 10k crops (about 50 MB) after a year.
  - Mitigation: fine for months. Cap or page the split's member query if it shows up in timings.
- Severity: low
  - Concern: individuals unseen for 90 days are still deleted. Their kept crops then point to an individual that no longer exists.
  - Mitigation: no reader joins on that id today. Phase 1 albums reference crop_id, not individual_id.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
