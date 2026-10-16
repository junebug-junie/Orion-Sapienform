## Summary

- The walkway retina "not recreated after the PR #2728 pull" report was a rebuild still running, not a skipped service. The post-pull rebuild did list `orion-vision-retina`, did run the walkway profile pass, and did build a new walkway image (18:57:01 UTC).
- Docker Compose v5 creates the replacement container first, named `<old-id>_orion-vision-retina-walkway`, then waits for the old container to stop. The old container had a 30 s stop grace period. During that window the old container still showed `StartedAt 18:13:41`, and the not-yet-started replacement already carried the new env keys and the new state-dir bind. That matches every detail of the report.
- Nothing on disk recorded what the hook did. Docker keeps about one minute of events on athena, and the hook output goes only to the terminal that ran `git pull`. So a running recreate could not be told apart from a missed one, and a manual `--force-recreate` at 18:57:30 raced the hook's own recreate.
- Fix: `scripts/rebuild_services_from_git_diff.sh` now appends one START and one END line (with exit code and seconds) per compose step to `~/.orion/post_pull_rebuild.log` (`ORION_REBUILD_LOG` overrides). Best effort: a log write never fails or changes the rebuild.
- New invariant test: a service's own bind-mounted Python code must also be COPY'd into its image. The planner plans `build` (not `recreate`) for own-dir directory mounts, and that restarts the process only because the image changes. A mount-only Python app would keep running old code after every pull; the test now fails if one appears.

## Outcome moved

"Did the post-pull rebuild recreate X, fail, or is it still running?" is now answerable from a file, after the fact. Before: unanswerable once the terminal scrolled and about one minute passed.

## Current architecture

`git pull` on the primary checkout -> `post-merge` hook -> `rebuild_services_from_git_diff.sh` -> `rebuild_affected_services.py --plan` -> one `safe_docker_build.sh <svc> up -d --build [--force-recreate]` per service, plus one `--profile <p> up -d --build <name>` per running profile service (PR #2680). Output went only to stdout.

## Architecture touched

- `scripts/rebuild_services_from_git_diff.sh` only. No planner, schema, bus, or env template change.

## Files changed

- `scripts/rebuild_services_from_git_diff.sh`: run log (RUN / START / END rc / DONE lines), log path printed at the end.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: harness points the log at `tmp_path` (tests never touch `~/.orion`); the fake build can fail; four new tests (PR #2728 case logs both passes, failed step rc logged, dry-run writes no log, unwritable log never fails the rebuild).
- `tests/scripts/test_rebuild_affected_services.py`: PR #2728 path set plans `orion-vision-retina\tbuild` and the Dockerfile COPYs `app/`; invariant that own-dir Python mounts are COPY-baked (allowlist: `services/orion-analytics`, a dbt dir whose one `.py` is an operator helper).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the rebuild script writes `~/.orion/post_pull_rebuild.log`.
- Compatibility notes: none.

## Env/config changes

- Added keys: none in any `.env_example`. `ORION_REBUILD_LOG` is an optional shell override for the host script (tests use it), not a service env key.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/scripts/test_rebuild_services_from_git_diff.py -q   16 passed
pytest tests/scripts/test_rebuild_affected_services.py -q        69 passed
mutation: emptying the analytics allowlist makes the invariant test fail (offender services/orion-analytics)
bash -n scripts/rebuild_services_from_git_diff.sh: clean (shellcheck not installed on this host)
```

## Evals run

```text
None. Host deploy tooling has no eval harness; behavior is covered by the script tests above.
```

## Docker/build/smoke checks

```text
Read-only reproduction on the primary checkout:
  rebuild_affected_services.py --plan --base 5fade7cc4   -> orion-vision-retina	build
  rebuild_affected_services.py --running-profile-services orion-vision-retina -> walkway	orion-vision-retina-walkway
Agent board (deploy heartbeats from safe_docker_build.sh, no session id = git hook):
  18:56:53 up -d --build (orion-vision-retina)
  18:56:59 --profile walkway up -d --build orion-vision-retina-walkway
  18:57:30 --profile walkway up -d --force-recreate orion-vision-retina-walkway (manual, session 03df7418)
Images: walkway 9c4d93de3fc4 created 18:57:01; base 6f9c454ab5e6 created 18:56:55.
Compose v5 recreate order probe (throwaway project, SIGTERM-ignoring container, 6 s grace):
  t+1..t+5: "<id>_orion-recreate-probe Created" + old "Up"; inspect-by-name -> old StartedAt, PROBE_KEY=old
  t+6: replacement renamed and started, PROBE_KEY=new
Live log write not exercised on the host yet: UNVERIFIED until the next pull.
```

## Review findings fixed

- (filled in after review)

## Restart required

```text
No restart required. The next `git pull` on main picks up the script.
```

## Risks / concerns

- Severity: low
- Concern: the log grows without bound (about 2 lines per compose step per pull).
- Mitigation: a full-fleet pull is about 140 lines; years of pulls stay in the low MB. Rotate by hand if ever needed.

- Severity: low
- Concern: who started the walkway container at 18:13:41 was also the post-merge hook (pull 0e6efb908 at 18:03:26 rebuilt the whole fleet; walkway step heartbeat 18:13:38). That pull predates PR #2728, so that container could not have had the PR #2728 env keys; those were on the not-yet-started replacement container seen at about 18:57.
- Mitigation: none needed; recorded here for the record.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
