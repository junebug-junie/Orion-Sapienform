## Summary

- Post-pull rebuilds no longer skip orion-hub / orion-retro when a retro game is live.
- Removed the guard block from `scripts/rebuild_services_from_git_diff.sh`, its helper `scripts/check_retro_session_live.py`, its test, and CI references.
- Reverts the behavior added in the 2026-10-20 rebuild-guard PR. Juniper: "i dont care if games are running."

## Outcome moved

A pull that touches orion-hub rebuilds it immediately instead of reporting `deferred (retro game live)` and rebuilding 0 services.

## Env/config changes

- Removed: `ORION_REBUILD_IGNORE_LIVE_RETRO` (script-only override, never in any `.env_example`).

## Tests run

```text
pytest tests/test_post_pull_apply_migrations.py tests/scripts/test_rebuild_services_from_git_diff.py  -> 174 passed
bash -n scripts/rebuild_services_from_git_diff.sh -> ok
```

## Evals run

None; script-only removal, no eval harness applies.

## Restart required

No restart required.

## Risks / concerns

- Severity: low. Concern: a rebuild during a game ends that game. Mitigation: accepted by Juniper.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
