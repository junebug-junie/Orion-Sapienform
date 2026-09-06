# Stop bind-mounting the host's `~/.claude.json` into Orion's containers

## Summary

- Orion's Claude sessions have opened with "configured MCP servers failed to connect: caveman (ENOENT)" since 2026-09-02. Orion's chat model misread that banner as a GitHub MCP outage and stopped using the GitHub tools it actually had.
- Root cause: `orion-harness-governor` and `orion-hub` bind-mounted the operator's `~/.claude.json` (a single file) into the container. That file holds the host-global MCP server list; `caveman` was installed on the host on 2026-09-01 and its binary does not exist in the container.
- Second defect from the same mount: Claude Code rewrites `~/.claude.json` by atomic replace, so the bind mount pins the inode from container start. The container's copy was dated 2026-09-03 while the host file was newer; host-side config edits were invisible to Orion until a restart.
- Removed the mount from both compose files. Live-verified inside the governor container that `claude -p` with a fresh `CLAUDE_CONFIG_DIR` and no host file answers normally with github, firecrawl and gitnexus all `connected`.
- Added a CI static gate (`scripts/check_compose_no_host_claude_json_mount.py`, wired into `orion-static-gates.yml`) that fails on the exact removed line. First draft was a pytest file under the governor's tests dir; the review pass noted no CI workflow runs that directory, so it became a `scripts/check_*.py` like the neighbouring relative-mount gate.
- Host-side, `caveman` was moved from global to project-local scope (`claude mcp remove -s user` + `add -s local`); not a repo change.

## Outcome moved

Orion's sessions no longer start with a false "MCP failed to connect" banner, and host-global MCP servers can no longer leak into Orion's runtime. Host config edits no longer silently diverge from what the container sees.

## Current architecture

Both containers mounted `${HOME}/.claude.json:/root/.claude.json:ro` (added 2026-07-05/06 with the agent-claude hardening). Governor's claude subprocess reads default config paths (`/root/.claude` volume + `/root/.claude.json`). Hub's subprocess already uses `HUB_AGENT_CLAUDE_CONFIG_DIR=~/.claude-fcc`, so its `.claude.json` lives there and the mount was already unused by the subprocess.

## Architecture touched

- `services/orion-harness-governor/docker-compose.yml`: volume removed, comment explains why.
- `services/orion-hub/docker-compose.yml`: volume removed, comment explains why.

## Files changed

- `services/orion-harness-governor/docker-compose.yml`: drop the mount; document the stale-inode + MCP-leak failure.
- `services/orion-hub/docker-compose.yml`: same.
- `scripts/check_compose_no_host_claude_json_mount.py`: static gate over every `services/*/docker-compose.yml`, with a `--self-test` that proves it bites on the removed line.
- `.github/workflows/orion-static-gates.yml`: runs the self-test and the gate next to the existing relative-mount gate.
- `docs/superpowers/pr-reports/2026-09-06-host-claude-json-mount-removal-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: Orion's claude subprocess no longer sees the host-global MCP server list or host-side trust/onboarding flags. Live test shows none are needed (`--permission-mode bypassPermissions` + `-p` skip both dialogs).
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: n/a
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest services/orion-harness-governor/tests -q
23 passed

.venv/bin/python -m pytest services/orion-hub/tests/test_service_logs.py \
  services/orion-hub/tests/test_fcc_claude_bridge_mcp.py \
  services/orion-hub/tests/test_fcc_mcp_config.py -q
15 passed

python3 scripts/check_compose_no_host_claude_json_mount.py --self-test  -> self-test PASS
python3 scripts/check_compose_no_host_claude_json_mount.py              -> PASS (89 compose files, 0 mounts)
Gate bites on origin/main's compose files:
  orion-harness-governor -> [(27, '${HOME}/.claude.json')]
  orion-hub              -> [(32, '${HOME}/.claude.json')]
python3 scripts/check_compose_no_relative_mounts.py                     -> PASS
python3 scripts/check_scripts_dir_no_stdlib_shadow.py                   -> clean
```

## Evals run

```text
No eval harness covers compose shape; the gate above is the deterministic check.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-harness-governor config   -> renders, 0 "claude.json" mounts
scripts/safe_docker_build.sh orion-hub config                -> renders, 0 "claude.json" mounts

Live proof (inside orion-athena-harness-governor, before the patch, simulating
the post-patch state with a fresh CLAUDE_CONFIG_DIR and no host file):
  claude mcp list                    -> "No MCP servers configured" (no caveman)
  claude -p "Reply with exactly: FRESH_CONFIG_OK" --mcp-config <rendered> ...
    init mcp_servers = github:connected, firecrawl:connected, gitnexus:connected
    result is_error=False text=FRESH_CONFIG_OK
```

## Review findings fixed

- Finding: the regression gate lived under `services/orion-harness-governor/tests/`, which no CI workflow runs, so it was a gate in name only.
  - Fix: rewritten as `scripts/check_compose_no_host_claude_json_mount.py` and added to `orion-static-gates.yml`.
  - Evidence: gate output above; bites on both origin/main compose files at the exact removed lines.
- Finding (second review pass, should): the gate's regex stopped at the `:` inside `${HOME:-/root}`, so a `${VAR:-default}` prefix slipped through while the comment claimed it was covered.
  - Fix: pattern now tolerates colons inside `${...}` and quotes; self-test covers `${HOME}`, `${HOME:-/root}` and a quoted absolute path.
  - Evidence: `--self-test` PASS with all three forms.
- Finding (second review pass, should): the glob skipped host-specific variants like `docker-compose.circe-qwen.yml`.
  - Fix: glob widened to `services/*/docker-compose*.yml` (89 files).
  - Evidence: gate output above.
- Second review pass: no must-fix findings; no code reads `.claude.json` directly; no README or non-historical doc still describes the mount; CI YAML parses and the step sits in the `static-gates` job.
- The first review subagent was cut off by a usage limit after raising the CI-coverage point.

## Restart required

The fix only takes effect when the containers are recreated with the new compose:

```bash
cd /mnt/scripts/Orion-Sapienform-claude-json-mount   # or main after merge
scripts/safe_docker_build.sh orion-harness-governor up -d
scripts/safe_docker_build.sh orion-hub up -d
# verify: no caveman line, only the context-mode plugin
docker exec -w /mnt/orion-fcc/repo orion-athena-harness-governor claude mcp list
```

## Risks / concerns

- Severity: low
- Concern: `/root/.claude.json` in the governor is now container-local and ephemeral (recreated on each container recreate). It only holds cosmetic first-run state; the durable plugin/session state stays in the `harness-claude-config` volume.
- Mitigation: live test above ran from an empty config dir and worked.

## PR link

(filled in after push)
