## Summary

- The walkway matcher now decides "same person?" from the body in two lanes. A crop close to someone seen in the last 10 seconds (within 1.5 box heights) joins them at body likeness 0.88. Recognising someone after a gap needs 0.90 (was 0.80).
- The face lane, the same-frame rule, and the cannot-link rule from PR #2709 are unchanged.
- Kill switch: `VISION_INDIVIDUALS_CONTINUATION_ENABLED=false` plus `VISION_INDIVIDUALS_MATCH_THRESHOLD=0.80` gives the old rule back, byte for byte (golden test against main).
- New read-only replay eval drives the real `apply_batch` over the live crops. It reproduces the proposal exactly.
- New operator script `scripts/reanchor_vision_individuals.py` rebuilds each named individual's body average from that individual's own face crops. It is a dry run by default. `--apply` snapshots first and writes an audit row. `--undo` puts it back. It has not been run with `--apply`.

Design: `docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md` (PR #2713, approved by Juniper 2026-10-16).

## Outcome moved

Replay of the 4,988 live walkway person crops (read-only, pinned window up to 2026-10-16 03:11 UTC):

| Rule | Individuals | With >= 5 crops | Mixed (lower is better) | Passing split (lower is better) | Re-seen | Top sizes |
|---|---|---|---|---|---|---|
| Old: body 0.80 | 40 | 27 | 13.6% | 17.6% | 96% | 1166 / 758 / 738 |
| New: continuation 0.88 + re-id 0.90 | 493 | 88 | 6.3% | 15.6% | 55% | 1139 / 699 / 255 |

Two people seen together now end up in one individual half as often. A single passing is split slightly less often than before. The cost is about 500 small individuals a day instead of about 40 (Juniper accepted this).

Re-anchor dry run on live data (read-only). The last column counts how many of the last 24 h of person crops score >= 0.90 against the old body average and then the new one:

```text
individual label          copied  n old  n new stored  face     cos  last 24 h >= 0.90: old -> new (of total)
26b5a26f   Esme           F           2      -      0     0       -  623 -> - (of 4989)   skip: no face crop passes the rule
4354c877   Amanda Feld    T          84     45      0    45  0.9973  81 -> 79 (of 4989)
62cb5fcc   Maeve          F         756     24      0    24   0.918  690 -> 35 (of 4989)
9bea16d5   Amanda Feld    F          54     21      0    21  0.9806  60 -> 31 (of 4989)
cb65248a   Esme           F           1      3      1     2  0.9013  3 -> 12 (of 4989)
f5242d40   Teddy          F         296     20      0    20  0.9685  246 -> 58 (of 4989)
```

62cb5fcc stops pulling in the child in the chair: 690 crops pass against its old average, and 35 pass against the new one.

## Current architecture

`apply_batch` had one body rule. A crop without a confident face match joined the nearest same-kind body average at cosine >= 0.80, at any time and any place, or it opened a new individual. Named individuals built under that rule carry body averages mixed from several people.

## Architecture touched

- orion-sql-writer individuals loop (`apply_batch`, `_one_stream`, config wiring).
- One new audit table, used only by an operator script. No service reads it.
- No bus channel and no schema registry change.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: adds the continuation lane, `assigned_by_continuation` / `assigned_by_reid`, the `[VISION_MATCH]` per-batch log line, the fragmentation warning (> 1,500 new person individuals in 24 h), and the new `IndividualsConfig` fields.
- `services/orion-sql-writer/app/vision_individuals_loop.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the four new keys and the 0.90 default.
- `services/orion-sql-writer/evals/run_vision_body_match_replay_eval.py`: the replay eval.
- `services/orion-sql-writer/evals/run_vision_split_replay_eval.py`: pinned to the old rule, because it rebuilds individuals that the old rule clustered.
- `services/orion-sql-writer/tests/test_vision_body_match.py`, `tests/_body_match_fixture.py`: gate tests, the rollback golden test, and real-Postgres cycle tests.
- `services/orion-sql-writer/tests/test_vision_reanchor.py`: re-anchor tests on real Postgres.
- `services/orion-sql-writer/tests/test_vision_walkway_zones_thumbs_cursor.py`: the fake connection gains `scalar()` for the new count query.
- `scripts/reanchor_vision_individuals.py`: the re-anchor, the snapshot, and the undo.
- `services/orion-sql-db/manual_migration_vision_reanchor_v1.sql`: the `vision_individual_reanchor` audit table.

## Schema / bus / API changes

- Added: table `vision_individual_reanchor` (new table plus an index on it; passes `auto_apply_refusals`; `REQUIRED-BY: none`, because only the operator script uses it).
- Added: `apply_batch(continuation_threshold=None, continuation_gap_sec=10, continuation_max_disp=1.5)`. `None` means the lane is off, so existing callers keep their behaviour.
- Removed / renamed: none.
- Behaviour changed: the re-identification line moves 0.80 -> 0.90, and the continuation lane is on.
- Compatibility notes: only new crops are affected. Existing individuals keep their averages until the re-anchor runs.

## Env/config changes

- Added keys: `VISION_INDIVIDUALS_CONTINUATION_ENABLED=true`, `VISION_INDIVIDUALS_CONTINUATION_THRESHOLD=0.88`, `VISION_INDIVIDUALS_CONTINUATION_GAP_SEC=10` (capped at the merge gap), `VISION_INDIVIDUALS_CONTINUATION_MAX_DISP=1.5`.
- Changed default: `VISION_INDIVIDUALS_MATCH_THRESHOLD` 0.80 -> 0.90.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer`: yes. The 4 new keys were added.
- Skipped keys requiring operator action: `VISION_INDIVIDUALS_MATCH_THRESHOLD` was reported as diverged (local value 0.80). It was changed by hand to 0.90 in `/mnt/scripts/Orion-Sapienform/services/orion-sql-writer/.env`.

## Tests run

```text
pytest tests/test_vision_body_match.py tests/test_vision_reanchor.py   (with ORION_TEST_PG_URI)  -> all pass
pytest tests -k vision (orion-sql-writer, real Postgres)                -> all pass
pytest tests (orion-sql-writer, full)                                   -> 14 failed, the same 14 that fail on main (biometrics / chat_history / notify / gpu_pool / journal / subscription), none vision
check_env_key_single_source, check_migration_required_by --base origin/main, check_metric_lineage --gate,
check_definition_drift --gate, check_scripts_dir_no_stdlib_shadow, check_service_hostname_refs,
check_compose_no_relative_mounts, check_service_env_compose_parity orion-sql-writer -> PASS
```

## Evals run

```text
python evals/run_vision_body_match_replay_eval.py --live postgresql://...@localhost:55432/conjourney
crops=4988 faces=155 track_pairs=2538 tracklets=2450 different_people_pairs=1561
rule        inds   /day  >=5  mixed  split  re-seen  top
baseline      40   40.0   27  0.136  0.176     0.96  [1166, 758, 738]
deployed     493  493.0   88  0.063  0.156    0.548  [1139, 699, 255]
REPRODUCED
```

The pinned window holds every crop that landed by 2026-10-16 03:11 UTC. Crops are pruned after 7 days, so from about 2026-10-22 the eval exits 2 ("cannot replay"). After that, use `--since/--until` on a new window; that mode prints the table without asserting.

## Docker/build/smoke checks

```text
No image build: the change is Python only, with no new dependency. Live path UNVERIFIED until deploy.
```

## Review findings fixed

The code review ran in a subagent. It confirmed the kill switch against origin/main itself (seeds 1, 2, 3 and 11, all identical) and reproduced the eval.

- Finding (must): a re-anchor set `centroid_n` to the face-crop count. A later split of the same individual then subtracted the other person's crops a second time. It could also write `centroid_n=0`. Undoing an earlier split added those crops back with a very large weight.
  - Fix: `vision_split.reanchored_body()`. While a re-anchor is in force, split and undo-split use only the moved crops that belong to the re-anchored centroid: the run's `crop_ids`, or crops that landed after the run.
  - Evidence: `test_a_split_after_a_reanchor_subtracts_only_crops_in_the_reanchored_centroid`.
- Finding (must): undo restored every row in the snapshot. That included targets the run skipped, and runs that never committed. It also silently dropped crops the matcher had folded in since the run.
  - Fix: undo now works from the run's committed audit rows only. It refuses, and writes nothing, when a centroid changed since the run; `--force` overrides. The dry run only reads.
  - Evidence: the skipped-target assertion in `test_apply_snapshots_writes_audits_and_undo_restores_byte_for_byte`, plus `test_undo_refuses_when_crops_joined_since_unless_forced` and `test_undo_of_a_run_that_never_committed_refuses`.
- Finding (should): a late crop was compared with the person's last box, not with where they were at that moment.
  - Fix: `apply_batch(prior_observations=...)`. `_one_stream` loads stored crop membership around the batch, and the lane uses each individual's observation nearest in time to the crop. (A first attempt used "any observation within 10 s". The live replay showed it made things worse: mixed 6.6%, split 15.9%. It was replaced with nearest-in-time, which reproduces 6.3% / 15.6%.)
  - Evidence: `test_a_late_crop_is_compared_with_where_its_person_was_at_that_moment`.
- Finding (should): the new Postgres tests never ran in CI.
  - Fix: `.github/workflows/orion-sql-writer-tests.yml` now applies the reanchor migration and runs both new files in the must-not-skip Postgres step and in the unit step.
- Finding (should): stationary fixtures got the lower continuation line for passers-by.
  - Fix: stationary individuals are no longer continuation candidates. On the replay, the skeleton prop goes from 1144 to 1139 crops; nothing else moves.
  - Evidence: `test_a_stationary_fixture_does_not_get_the_continuation_line`.
- Finding (should): `nearest_among` rebuilt its id-to-row map on every crop.
  - Fix: the index keeps that map up to date in `add()`.
- Finding (should): mixed embedding sizes crashed the re-anchor.
  - Fix: face centroids are grouped by size, and crops whose body size differs from the target's centroid are skipped.
- Finding (nit): the gap cap lived only in the caller.
  - Fix: `apply_batch` clamps the gap to the merge gap itself.
  - Evidence: `test_the_continuation_gap_is_capped_at_the_merge_gap_inside_apply_batch`.
- Finding (nit): the fragmentation warning fired on every tick.
  - Fix: it now runs at most once an hour per stream.
- Finding (nit): the new settings had no bounds.
  - Fix: added `ge`/`le`/`gt` limits.
- Not changed (nit): when a cannot-link refuses a crop's face match, the continuation lane can still choose that same individual. This matches PR #2709 ("falls through to the body rule"). Not changed (nit): the eval's `deployed` row uses the shipped defaults (= `.env_example`), not the live env.

## Restart required

Deploy from the clean primary `main` checkout after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_reanchor_v1.sql && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Then check the live path: `docker logs orion-athena-sql-writer --since 5m 2>&1 | grep VISION_MATCH` should show non-zero `continuation=` and `reid=` while people pass.

Re-anchor (the coordinator runs it after the merge; it is a production write):

```bash
python3 scripts/reanchor_vision_individuals.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney          # dry run
python3 scripts/reanchor_vision_individuals.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney --apply  # write
```

Delete `/tmp/vision-reanchor/<ts>/` after 7 days (biometric-adjacent).

## Risks / concerns

- Severity: medium. Concern: about 500 small individuals a day means fewer people reach the ask gates (>= 10 sightings, >= 5 days), so Orion asks about fewer people. Mitigation: this is the safe direction (pieces can be joined later; a mixed individual cannot be separated). `[VISION_MATCH] ... fragmenting` warns above 1,500 a day.
- Severity: low. Concern: a re-anchor uses face crops of unknown membership (crops older than the split migration) by the "nearest face centroid" rule. Mitigation: a face-cosine guard of >= 0.60 to the target, a dry-run review, and `--undo`.
- Severity: low. Concern: `--undo --force` drops crops that joined after the re-anchor from the centroid. Mitigation: without `--force`, undo refuses and names those individuals.
- Severity: medium (needs Juniper). Concern: the "Esme" individual 26b5a26f has 2 crops and no face. Its body average still scores >= 0.90 against 623 of the last 24 h of person crops, so it is a magnet at the new line too. The re-anchor cannot fix it, because it has no face crops. Mitigation: none in this patch. Options: clear its label, or let it be found and split.
- Severity: low. Concern: about 12x more individuals load per tick (30 days of candidates). Mitigation: the index lookups are vectorised or O(1); watch tick time after deploy.

## PR link

(filled in below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
