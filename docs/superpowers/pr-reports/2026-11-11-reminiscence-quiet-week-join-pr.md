## Summary

- A week with no commit on main is no longer visited alone. The nightly reminiscence now joins a run of
  quiet weeks onto the next week that has a commit, and Orion visits that as one longer period.
- The read says so in code-written words: "covers 5 quiet weeks: no commit on main from 2025-07-29 until
  the week from 2025-09-02, so they are joined onto that week and read as one period."
- The dig (`dig reminisce`) and the overnight tick (orion-durable-runs) use one rule and one git count
  (`rm.git_commits_in`), so the brief and the dig always name the same period.
- Weeks 1-4 (already visited, three of them quiet) stay valid as written. Nothing in the ledger is rewritten.
- orion-durable-runs now mounts the dig clone `/mnt/orion-fcc/repo` read-only and its image installs `git`.

## Outcome moved

Orion spent three nights (weeks 2, 3, 4) writing "nothing happened, the tree is frozen". 14 of 72 weeks are
quiet. With one night a day, the backlog from tonight is about 58 nights instead of about 71, and every
night has real code change to read.

Real plan (read-only, real ledger `/mnt/orion-fcc/self/reminiscence.jsonl` + real clone `/mnt/orion-fcc/repo`):

```text
episode 2026-11-11 (tonight, 03:00 MST) -> 2025-07-22..2025-07-28  (week 5, 2 commits, not joined)
episode 2026-11-12                      -> 2025-07-29..2025-09-08  (weeks 6-11: 5 quiet weeks + week 11, 2 commits)
episode 2026-11-13                      -> 2025-09-09..2025-09-22  (week 12 quiet + week 13, 11 commits)
episode 2026-11-14                      -> 2025-09-23..2025-09-29  (week 14, 14 commits)
```

## Current architecture

`rm.pick_period(records, today)` returned the oldest 7-day week with no complete episode. The dig and
`app/reminiscence_ledger.py` both called it. Nothing looked at commits, so a quiet week was visited
like any other. `ReminiscenceRunBriefV1` required the period to be exactly 7 days.

## Architecture touched

- `orion/body_atlas/reminiscence.py`: `pick_period(records, today, commits_in)` joins quiet weeks;
  `complete_weeks` covers every week inside a complete period; new `first_open_week`, `period_weeks`,
  `quiet_weeks`, `git_commits_in`. Read model gains `quiet_weeks`.
- `orion/dig/__main__.py`: the read names the joined period and the quiet-weeks note; a git failure refuses
  the pick (`git UNREAD`) instead of guessing.
- `services/orion-durable-runs`: the ledger reader counts commits in the read-only clone; the tick records
  `history_unreadable` / `waiting_for_commit` and submits nothing in those cases; the hop prompt names the
  joined weeks.
- `orion/schemas/reminiscence_run.py`: brief accepts whole grid weeks.

## Files changed

- `orion/body_atlas/reminiscence.py`: the join rule and the shared git count.
- `orion/dig/__main__.py`: joined period in the read, write receipt and help line; fail-closed pick.
- `orion/schemas/reminiscence_run.py`: brief validator accepts whole grid weeks; `quiet_weeks` property.
- `services/orion-durable-runs/app/reminiscence_ledger.py`: commit count source, `quiet_wait_from`, `HistoryUnreadable`.
- `services/orion-durable-runs/app/reminiscence_poll.py`: new decisions `history_unreadable`, `waiting_for_commit`; `quiet_weeks` in the tick result.
- `services/orion-durable-runs/app/reminiscence_graph.py`: hop/cross-read prompt wording for joined periods.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`, `app/main.py`: `REMINISCENCE_REPO_DIR`, read-only mount, `git` in the image.
- `services/orion-durable-runs/README.md`, `docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md` (new section 1a): docs.
- `tests/test_organ_names_partition.py`: the ledger reader may return `date | None` (a week start date, no hop text).
- Tests: `orion/body_atlas/tests/test_reminiscence.py`, `orion/dig/tests/test_reminisce_cmd.py`, `services/orion-durable-runs/tests/test_reminiscence_poll.py`.
- Evals: `orion/body_atlas/evals/run_reminiscence_source_truth_eval.py`, `services/orion-durable-runs/evals/reminiscence_daily_replay.py`.

## Schema / bus / API changes

- Added: `rm.PeriodOverlap`; tick decision `period_refused`; `ReminiscenceReadV1.quiet_weeks` (default 0); `ReminiscenceRunBriefV1.quiet_weeks` (derived property, not a field); tick decisions `history_unreadable`, `waiting_for_commit`; `TickResult.quiet_weeks`.
- Removed: none.
- Renamed: none.
- Behavior changed: `ReminiscenceRunBriefV1` accepts a period of whole weeks on the 2025-06-24 grid (was exactly 7 days). A complete episode covers every week inside its period.
- Compatibility notes: every stored 7-day brief still validates. Old ledger lines (single weeks) are read unchanged. No bus channel or registry change.

## Env/config changes

- Added keys: `REMINISCENCE_REPO_DIR=/mnt/orion-fcc/repo` (orion-durable-runs).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, `services/orion-durable-runs/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`services/orion-durable-runs/.env` line `REMINISCENCE_REPO_DIR=/mnt/orion-fcc/repo`).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest -q orion/dig/tests orion/body_atlas/tests     -> 412 passed
cd services/orion-durable-runs && pytest tests -q                  -> 835 passed, 130 skipped
pytest tests/test_organ_names_partition.py                          -> 10 passed
every run step of .github/workflows/orion-static-gates.yml          -> pass
pytest services/orion-hub/tests/test_orion_abilities_catalog.py    -> 10 passed
```

## Evals run

```text
python orion/body_atlas/evals/run_reminiscence_source_truth_eval.py
  oracle: 16 reviewed cases, re-check, label truth, quiet-week periods -> PASS
  mutation 'a quiet week is visited alone': CAUGHT (periods: night 2 visited 2025-07-01..2025-07-07: a quiet week visited alone)
  mutation 'a quiet run joins past the first busy week': CAUGHT
  (5 earlier mutations: CAUGHT)
python -P services/orion-durable-runs/evals/reminiscence_daily_replay.py
  oracle: PASS, joined ['2025-07-01..2025-07-28', '2025-07-29..2025-09-08', '2025-09-09..2025-09-22']
  mutation 'quiet week visited alone': CAUGHT (8 quiet: ... ends on a week with no commit)
  mutation 'join runs past the first busy week': CAUGHT
  (3 earlier mutations: CAUGHT)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile -t <scratch tag> .     -> built (scratch image removed after)
docker run --rm -v /mnt/orion-fcc/repo:/mnt/orion-fcc/repo:ro -v /mnt/orion-fcc/self:/mnt/orion-fcc/self:ro <scratch> \
  python -c "ReminiscenceLedger(self, repo).week_for(2026-11-11)"
  uid 0 -> WeekPick(2025-07-22, 2025-07-28, quiet_weeks=0, weeks_complete=4, weeks_total=72)
  commits 2025-07-29..2025-09-08: 2; week 2025-07-29: 0
dig read for the night of 2026-11-12 on a copy of the real ledger (clock patched, body offline):
  "reminiscence 2026-11-12 (today): the weeks 2025-07-29..2025-09-08 (...; 5 of 72 weeks visited)"
  "covers 5 quiet weeks: ..."  "code_then: commit 741c2c678 (2025-09-08), 2 commits on main in that period"
```

Note: `scripts/safe_docker_build.sh orion-durable-runs build` from the worktree printed
"couldn't find env file .../.env" and exited 0 without building (a worktree has no root `.env`). The plain
`docker build` above is the evidence.

## Review findings fixed

A code review ran in a subagent against commit c0048b821. It found no blockers. It checked git as root on the
athena-owned clone mounted read-only (works with `safe.directory`), and the `--after` cut-off on all 72 weeks
of the live clone (0 mismatches).

- Finding: a joined period could overlap a week that is already visited (out-of-order or hand-edited ledger).
  - Fix: `pick_period` raises `PeriodOverlap`; the tick records `period_refused` and submits nothing; the dig refuses the pick.
  - Evidence: `test_a_period_never_overlaps_a_visited_week`, `test_a_period_that_would_overlap_a_visited_week_is_refused`.
- Finding: the write check compared only `period_start` with today's first hop, so a hop with the same start and another end was accepted.
  - Fix: compare start and end against the first hop by `written_at`; the dig's "fixed by your first hop" also uses the earliest hop now (same rule as the tick).
  - Evidence: `test_a_hop_with_the_same_start_but_another_end_is_refused`.
- Finding: `--before` is inclusive, so a commit at exactly 00:00:00 Denver counted in two weeks.
  - Fix: end one second before the next Denver day.
  - Evidence: midnight case in `test_git_commits_in_counts_first_parent_commits_in_denver_days`.
- Finding: the git test did not cover `origin/main` versus HEAD or first-parent with a merge.
  - Fix: `test_git_commits_in_reads_origin_main_first_parent_only` (side-branch and local-only commits are not counted).
  - Evidence: passes.
- Finding: an operator-set host path that does not exist becomes an empty root-owned dir.
  - Fix: noted in `.env_example` (the tick then reads `history_unreadable`).
  - Evidence: n/a (doc).
- Finding (not changed): `git_commits_in` falls back to HEAD when `origin/main` is missing (same as the old dig rule); the partition allowlist accepts any `date | None` return (a date carries no hop text). Recorded, left as is.
- Finding (deploy skew, rollback): see Risks.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run from the clean primary `main` checkout after merge (the post-pull rebuild does this). The new image
has `git` and the new read-only mount. The dig side is live for Orion's turns once the FCC sandbox sync
moves `/mnt/orion-fcc/repo` to the merged main.

## Risks / concerns

- Severity: low. Concern: if the dig clone is unreadable (sandbox reset mid-tick), the tick records
  `history_unreadable` and submits nothing that tick. Mitigation: the tick runs every 10 min across a 6 h window.
- Severity: low. Concern: `where_im_going` reads alike across nights because it reads the same
  `dig capacities gaps` list each night. Not changed here; it needs its own material (e.g. what changed in
  the gap list since the last reminiscence), which is a design choice, not a one-line fix.
- Severity: low. Concern: deploy skew. The new tick can name a joined period while the FCC sandbox
  (`/mnt/orion-fcc/repo`) still has the old dig, which would show one quiet week. The dig decides, so
  nothing breaks, but the prompt would not match the read. Mitigation: before the first joined night
  (episode 2026-11-12), check `git -C /mnt/orion-fcc/repo log -1 --oneline` shows the merge.
- Severity: low. Concern: rolling durable-runs back while a joined-period run is unfinished makes the old
  7-day validator reject its stored brief at the next step (the run fails; the dig re-picks next night).
- Severity: low. Concern: only quiet weeks open with no later busy week ended yet makes the tick wait
  (`waiting_for_commit`). That cannot happen until the backlog reaches the present.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
