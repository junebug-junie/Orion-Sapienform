# fix(fcc): stop denying `Bash(gh *)` — it closed Orion's only route to opening a PR

## Summary

- Orion (the headless FCC harness agent) could not create GitHub PRs. Root-caused to a single hardcoded argv string, confirmed live from the running `orion-athena-harness-governor` container's own rendered argv.
- `orion/fcc/claude_spawn.py`'s `mcp_disallowed_tool_patterns()` unconditionally appended `Bash(gh *)` to `--disallowedTools` whenever the github MCP server was present. A deny rule **beats** `--permission-mode bypassPermissions`, so `gh` was the one command Orion could not run while holding otherwise-unrestricted Bash.
- Simultaneously the github MCP server renders read-only (`GITHUB_READ_ONLY=1`), so it exposes no `create_pull_request` tool. **Both** PR-creation routes were closed at once — which is why the failure looked unfixable from inside a turn.
- The deny's stated justification ("gh not installed in headless Hub") is factually false today: `gh` 2.63.2 is bind-mounted into both spawning containers and authenticated as `junebug-junie` with `repo` scope.
- Deleted the deny outright (kill means kill) rather than narrowing it, added a regression gate that asserts on the whole emitted argv, and corrected the stale README claim.

## Outcome moved

Orion can run `gh pr create` (and the rest of the `gh` CLI) from a headless FCC turn. Previously: hard denial on every attempt, with no MCP fallback. This is the capability Juniper reported ~20 PRs of failed attempts against.

## Current architecture

`orion/fcc/claude_spawn.py` builds the argv for the `claude -p` subprocess that every FCC turn spawns. Two production callers run it as root in containers:

- `orion/harness/fcc_motor.py` (service `orion-harness-governor`)
- `services/orion-hub/scripts/fcc_claude_bridge.py` (service `orion-hub`)

Both get `--permission-mode bypassPermissions` (full Bash, no prompt) via `claude_permission_argv()`, and both got `--disallowedTools 'Bash(gh *)'` via `extend_mcp_argv()`.

## Architecture touched

Only the argv builder and its tests/docs. No bus channel, schema, or env key changed.

## Files changed

- `orion/fcc/claude_spawn.py`: deleted `mcp_disallowed_tool_patterns()` and the `--disallowedTools` emission; documented on `extend_mcp_argv()` why no deny list exists and what to verify before re-adding one.
- `orion/fcc/tests/test_claude_spawn.py`: dropped the test asserting the deny; added `test_extend_mcp_argv_never_denies_gh` regression gate; fixed two argv equality assertions.
- `orion/harness/tests/test_fcc_motor_mcp.py`: `--disallowedTools` no longer expected in the governor's argv.
- `services/orion-hub/tests/test_fcc_claude_bridge_mcp.py`: same, for the Hub bridge argv.
- `services/orion-hub/README.md`: corrected the documented argv contract.

## Schema / bus / API changes

- Added: none
- Removed: `mcp_disallowed_tool_patterns()` (public helper in `orion.fcc.claude_spawn`; no callers outside `extend_mcp_argv` and its own test)
- Renamed: none
- Behavior changed: the `claude -p` argv no longer carries `--disallowedTools`
- Compatibility notes: the FCC subprocess already ran under `bypassPermissions`, so this widens the tool surface by exactly one CLI (`gh`) and nothing else.

## Env/config changes

None. No `.env_example` touched, so no sync required.

Noted but deliberately **not** changed: `GITHUB_READ_ONLY` is absent from `~/.fcc/.env`, so `orion/fcc/mcp_config.py:156` defaults it to `1`. Setting it to `0` would be the other way to grant PR creation (via the MCP `create_pull_request` tool), and needs no rebuild — but it also unlocks `merge_pull_request` and the file-write tools in the `repos,pull_requests` toolsets. Left read-only on purpose: reads go through MCP, writes go through the authenticated `gh` CLI where they land in git history. Flag for Juniper, not a silent decision.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-fcc-gh-pr-unblock
PYTHONPATH=. pytest orion/fcc/tests/ orion/harness/tests/test_fcc_motor_mcp.py \
  services/orion-hub/tests/test_fcc_claude_bridge_mcp.py \
  services/orion-hub/tests/test_fcc_mcp_config.py -q \
  --ignore=orion/fcc/tests/test_context_mode_hooks_smoke.py
-> 91 passed in 4.43s

PYTHONPATH=. pytest orion/fcc/tests/test_context_mode_hooks_smoke.py -q
-> 14 passed in 0.06s
```

Regression gate verified failable (not tautological): temporarily re-introducing the deny inside `extend_mcp_argv` produced

```text
FAILED orion/fcc/tests/test_claude_spawn.py::test_extend_mcp_argv_never_denies_gh
FAILED orion/fcc/tests/test_claude_spawn.py::test_extend_mcp_argv_uses_per_server_patterns
FAILED orion/fcc/tests/test_claude_spawn.py::test_extend_mcp_argv_appends_extra_allowed_tools_after_server_patterns
FAILED orion/harness/tests/test_fcc_motor_mcp.py::test_run_fcc_turn_adds_mcp_config_when_enabled
4 failed, 37 passed
```

then reverted and re-run green.

Pre-existing, unrelated: collecting `orion/fcc/tests/` together with `services/orion-hub/tests/` errors with `ModuleNotFoundError: No module named 'scripts.context_mode_hooks_smoke'` (a pytest sys.path collision — orion-hub's test root shadows `scripts`). Reproduced on unmodified `main`, so not introduced here. Each path passes when collected separately.

## Evals run

No eval harness exists for `orion/fcc/`. The meaningful behavioral check here is the live runtime evidence below, not a scored eval.

## Docker/build/smoke checks

Live evidence from the running governor, **before** the fix — the actual argv the FCC subprocess receives, rendered through the real code path with the real `~/.fcc/.env`:

```text
claude -p --mcp-config /tmp/orion-fcc-mcp/argv-proof.json \
  --allowedTools mcp__github mcp__firecrawl mcp__gitnexus \
  --disallowedTools 'Bash(gh *)' \
  --permission-mode bypassPermissions
GITHUB_READ_ONLY: 1
GITHUB_TOOLSETS: repos,pull_requests
```

`gh` is present and authenticated in both containers that spawn `claude -p`:

```text
docker exec orion-athena-harness-governor gh --version   -> gh version 2.63.2
docker exec orion-athena-hub               gh --version   -> gh version 2.63.2
gh auth status -> Logged in to github.com account junebug-junie
                  Token scopes: 'gist', 'read:org', 'repo'
```

`gh` reaches the API from inside Orion's own workspace (read-only calls):

```text
docker exec orion-athena-harness-governor sh -lc 'cd /mnt/orion-fcc/repo && gh api user --jq .login'
-> junebug-junie
docker exec orion-athena-harness-governor sh -lc 'cd /mnt/orion-fcc/repo && gh pr list --limit 3'
-> 1619 / 1617 / 1212  (real open PRs)
```

No other layer denies `gh`: the subprocess loads `--setting-sources user,local`; `/root/.claude/settings.json` contains only hooks/plugins (no `permissions` block) and no `settings.local.json` exists in the workspace. `--disallowedTools` was the sole gate.

**UNVERIFIED at runtime.** `/app` is baked into the governor image (not a bind mount), so the deny is still live as of this writing:

```text
docker exec orion-athena-harness-governor grep -n disallowedTools /app/orion/fcc/claude_spawn.py
-> 51:        argv.append("--disallowedTools")
```

The patch is proven at the argv/unit level only. There is no end-to-end proof that `gh pr create` succeeds from a real FCC turn until the rebuild below runs and one real PR-creation turn is observed. Per the repo's runtime-truth rule this stays `UNVERIFIED`, not `DONE`.

## Review findings fixed

- Finding: `services/orion-hub/tests/test_fcc_claude_bridge_mcp.py:86` still asserted `Bash(gh *)` and would have failed CI — a third test module I missed on the first pass.
  - Fix: replaced with `assert "--disallowedTools" not in captured_argv` plus an explanatory comment.
  - Evidence: full affected-module run green afterwards (91 passed).

- Finding (review, SHOULD FIX): the restart block told the operator to run `scripts/safe_docker_build.sh` from `/mnt/scripts/Orion-Sapienform`. That wrapper refuses to run from the shared/primary checkout, so both rebuild commands would have aborted with `exit 1`.
  - Fix: rewrote "Restart required" to invoke the wrapper from a worktree, and explained why the primary-checkout `git pull` is still separately required (containers bind-mount that absolute path; `/app` comes from the build context).
  - Evidence: `scripts/safe_docker_build.sh:48-65` guard read directly.

- Finding (review, SHOULD FIX): the patch was labeled as if shipped, but the deny is still live in the baked image.
  - Fix: added an explicit `UNVERIFIED at runtime` block under Docker/build/smoke checks with the live `grep` output showing `--disallowedTools` still present in `/app`, and set the final status to `DONE_WITH_CONCERNS`.
  - Evidence: `docker exec orion-athena-harness-governor grep -n disallowedTools /app/orion/fcc/claude_spawn.py` → `51: argv.append("--disallowedTools")`.

- Finding (review, NIT): the regression gate's second assertion (`"gh" in a and a.startswith("Bash(")`) was a false-positive trap — a legitimate future allow-list entry like `Bash(gh pr create:*)` passed via `extra_allowed_tools` would have failed a correct change, since the scan cannot distinguish allow from deny in a flat argv.
  - Fix: gate the deny *flag* instead of scanning for the string `gh`, and check both `--disallowedTools` and `--disallowed-tools` spellings.
  - Evidence: `orion/fcc/tests/test_claude_spawn.py` re-run green; gate still fails against pre-fix code.

- Finding (review, NIT): `graphify-out/` still carries the deleted `mcp_disallowed_tool_patterns` symbol and its now-retired "gh not installed" docstring.
  - Fix: NOT APPLIED. Ran `scripts/safe_graphify_update.sh` and hit the known destructive-update failure mode again — the wrapper refused and auto-restored:
    ```text
    [safe-graphify-update] REFUSED: node count dropped from 28306 to 2475 (~91.26%, threshold 10%).
    ```
    Per CLAUDE.md ("If the wrapper refuses, do not just re-run it"), left the graph stale rather than forcing it. `git status --short graphify-out/` is clean, so nothing destructive was committed. The stale symbol is a search-hygiene wart only — no runtime effect.
  - Evidence: wrapper output above; `graphify-out/` unmodified in this branch.

- Reviewer coherence note (accepted, wording corrected): after this patch `GITHUB_READ_ONLY=1` no longer constrains *what* Orion can write to GitHub — only *which path* the writes take. `gh` with `repo` scope covers PR create/merge/close, branch push/delete, and arbitrary `gh api` writes. The read-only MCP setting should not be read as defense-in-depth going forward. Risks section reworded accordingly.

## Restart required

`/app` is baked into the image, so a plain restart is not enough — both services that spawn `claude -p` need an image rebuild.

`scripts/safe_docker_build.sh` **refuses to run from the shared/primary checkout** (`scripts/safe_docker_build.sh:48-65`), so the rebuild must be invoked from a worktree. The image's `/app` comes from the build context (the worktree), while the containers' read-only bind mounts point at the absolute path `/mnt/scripts/Orion-Sapienform` — so both need to carry the fix:

```bash
# 1. after PR #1621 merges, refresh the primary checkout (the containers bind-mount it)
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only

# 2. rebuild FROM A WORKTREE -- this one already contains the fix
cd /mnt/scripts/Orion-Sapienform-fcc-gh-pr-unblock
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Verify the deny is gone from the live image:

```bash
docker exec orion-athena-harness-governor \
  grep -c disallowedTools /app/orion/fcc/claude_spawn.py || echo "0 — deny gone"
```

## Risks / concerns

- Severity: low
  - Concern: Orion gains real GitHub write capability via `gh` (branch push, PR create, and with `repo` scope also merge/close). This is the requested capability, not an accident, and it is auditable in git history — but it is a genuine autonomy widening.
  - Mitigation: revert is a one-line restore of the deny. Note explicitly that `GITHUB_READ_ONLY=1` is **not** defense-in-depth here — after this patch it only decides which path writes take, not whether they are possible. `gh` with `repo` scope already covers PR create/merge/close, branch push/delete, and arbitrary `gh api` writes. The benefit of keeping the MCP read-only is traceability (writes land in git history), not containment.

- Severity: medium (pre-existing, out of scope for this patch)
  - Concern: the `origin` remote URL in `/mnt/orion-fcc/repo` embeds the GitHub token in plaintext, so it leaks into any `git remote -v`, clone error, or log line that echoes the URL. The same token is stored as `GITHUB_PAT` in `~/.fcc/.env`.
  - Mitigation: switch the remote to a bare `https://github.com/junebug-junie/Orion-Sapienform.git` and let the already-configured `gh` credential helper supply auth. Not changed here — it touches Orion's live workspace, which is Juniper's call.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1621

## Status

DONE_WITH_CONCERNS — see the UNVERIFIED block under Docker/build/smoke checks. The code fix and its gates are complete and green; the live path is not proven until both services are rebuilt and one real `gh pr create` turn is observed.
