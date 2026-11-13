## Summary
- Orion's containers (orion-hub, orion-harness-governor) stop mounting the operator's `~/.config/gh` and `~/.ssh`. Orion acts on GitHub as `orion-sapienform` via `ORION_GITHUB_TOKEN` (`GH_TOKEN`, and `GITHUB_PAT` for the MCP server and sandbox pushes).
- Governor entrypoint no longer copies the operator's gh login into `GITHUB_PAT`; it strips any stale value first.
- `sandbox_sync` scrubs the old token from `remote.origin.url`/`pushurl` and from `url.*x-access-token*.insteadOf` keys, then rewrites SSH and HTTPS GitHub URLs to Orion's token.
- Image git identity email is now Orion's GitHub noreply address.
- `CODEOWNERS` gets a catch-all `* @junebug-junie` first line, so only the owner's approval counts on every path.
- Design: PR #3141. Env stub and never-sync: PR #3144 (first commit here is a cherry-pick of it).

## Outcome moved
Orion's PRs, comments and pushes were `junebug-junie` (an admin that `main` protection does not bind). After deploy they are `orion-sapienform` (Write role, bound by protection).

## Current architecture
Hub and governor bind-mounted the operator's gh login and SSH keys; the governor entrypoint copied the gh token into `~/.fcc/.env` as `GITHUB_PAT`; the sandbox clone's `origin` URL held the operator's token in plaintext.

## Architecture touched
`services/orion-hub`, `services/orion-harness-governor`, `orion/fcc/sandbox_sync.py`, `.github/CODEOWNERS`.

## Env/config changes
- Added keys: `ORION_GITHUB_TOKEN` (hub, governor `.env_example`, empty). Never-sync via #3144.
- Local `.env` for hub and governor updated by hand with the live token (never-sync key).
- Skipped keys requiring operator action: none.

## Tests run
```text
pytest tests/test_orion_own_github_identity.py orion/fcc/tests/test_sandbox_sync.py -> 29 passed
pytest tests/scripts/test_sync_local_env_from_example.py -> pass
check_env_template_parity.py -> PASS
```
Pre-existing failure, also on main, not touched here: `orion/fcc/tests/test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv`.

## Evals run
None; no eval harness covers GitHub identity. Live proof is the checks below.

## Docker/build/smoke checks
UNVERIFIED until deploy. After deploy: `docker exec orion-athena-hub gh api user --jq .login` and the same on the governor must print `orion-sapienform`; `git -C /mnt/orion-fcc/repo config --local --list` must contain no `gho_`/`ghp_`; Orion posts one comment on a throwaway PR and the author reads back `orion-sapienform`.

## Review findings fixed
- Finding: empty token would silently skip push auth. Fix: log is now a warning; repo is public so fetch still works anonymously.
- Finding: scrub missed `remote.origin.pushurl`. Fix: scrubbed, test added (`test_sync_with_no_token_still_scrubs_the_old_one`).
- Finding: entrypoint truncate-then-write race. Fix: write sibling file and `mv`.
- Finding: stale docstring in `claude_spawn.py`. Fix: updated.
- Finding: weak mount test. Fix: matches `.ssh` / `.config/gh` anywhere in the volume.

## Restart required
From the clean primary `main` checkout after merge:
```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns
- Medium: Orion loses everything the operator's token allowed beyond Write (intended).
- Medium: the operator's old token is still in the sandbox clone and `~/.fcc/.env` until the first sync/restart; rotate it afterwards (`gh auth logout && gh auth login`).
- Settings only the owner can change: enable "Dismiss stale approvals" and "Require approval of the most recent push" on `main`.
