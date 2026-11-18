## Summary

- Orion's own turns can now hand a multi-step repo job to a sub-agent called `orion-repo-agent`. Before, that agent existed only for operator coding sessions and Orion's turns could not see it.
- The agent reaches Orion through the CLI's `--agents` flag on every turn that is not a reading-only turn. The repo's project settings (CLAUDE.md, hooks, settings.json) stay off, as before.
- Orion gets their own version of the agent (`orion/fcc/agents/orion-repo-agent.md`), not the operator's file. It carries the sandbox rules: make a branch in the sandbox clone, never push `main`, never deploy, no live database writes, no PR unless asked, never merge.
- The one-line take-on hint now offers `Explore` for read-only search and `orion-repo-agent` for repo work that needs Bash, tests, or edits on a branch.
- On/off switch `HARNESS_FCC_REPO_AGENT_ENABLED`, default on.

## Outcome moved

The Agent tool in an Orion turn lists 6 types instead of 5. Confirmed in the governor's own CLI: with the flag, the `system/init` line lists `claude, Explore, general-purpose, orion-repo-agent, Plan, statusline-setup`. Without it, the list is the same minus `orion-repo-agent`. A real take-on turn is UNVERIFIED until deploy (#3671).

## Current architecture

`build_claude_argv` (`orion/harness/fcc_motor.py`) spawns `claude --setting-sources user,local`. Commit 1b810d28e dropped the project scope on purpose: the repo's AGENTS.md is a coding contract that costs tokens against FCC's small local-model window, and the project hooks have nothing to protect behind the read-only repo mount. A side effect: `.claude/agents/` is project scope, so the repo agent was never loaded. The governor's user scope is the `harness-claude-config` volume. Nothing in the repo manages or tests that volume.

## Architecture touched

- Seam chosen: per-turn `--agents <json>`. Rejected options:
  - Adding `project` to the setting sources would bring back the CLAUDE.md, hooks, and settings that 1b810d28e removed.
  - An agent file in user scope would live in the operator-managed volume. That file is outside git and untested, and an image layer at `/root/.claude` would be hidden under the volume mount.
  - `--agents` is code in the repo. It is tested, ships with the image (`COPY orion ./orion`), and works the same for spawned turns and the warm pool.
- The warm-pool chat process uses the same `build_claude_argv`, so it gets the agent too. Reading-only turns do not get it.

## Files changed

- `orion/fcc/agents/orion-repo-agent.md`: new file, Orion's sandbox variant of the agent (`model: inherit`, no `Agent` tool, no MCP tools).
- `orion/fcc/claude_spawn.py`: `fcc_agents_json` / `fcc_agents_argv` and a strict frontmatter parser. If the agent file is broken or missing, the turn logs a warning and keeps only the built-in agent types. It never fails the turn.
- `orion/harness/fcc_motor.py`: passes `--agents` on non-reading turns.
- `orion/prime/intention.py`: `CARRY_SUBAGENT_TYPES` / `CARRY_SUBAGENT_HINT` now name `orion-repo-agent`.
- `services/orion-harness-governor/{.env_example,app/settings.py,docker-compose.yml,README.md}`: new switch key and a docs section.
- Tests: `orion/harness/tests/test_fcc_repo_agent_argv.py` (new), `orion/fcc/tests/test_claude_spawn.py`, `orion/prime/tests/test_prime_intention.py`.
- `orion/fcc/tests/test_claude_spawn.py`: also fixes a stale root-caller map that was failing on main (room-companion and two governor probes were missing from it).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the FCC `claude` argv gains `--agents`.
- Compatibility notes: none.

## Env/config changes

- Added keys: `HARNESS_FCC_REPO_AGENT_ENABLED=true` (orion-harness-governor).
- Removed/renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-harness-governor: +HARNESS_FCC_REPO_AGENT_ENABLED='true'`).
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/fcc/tests/test_claude_spawn.py orion/harness/tests/test_fcc_repo_agent_argv.py orion/prime/tests/test_prime_intention.py orion/harness/tests/test_fcc_motor_mcp.py orion/harness/tests/test_focus_anchor.py orion/harness/tests/test_session_resume.py orion/harness/tests/test_task_anchor.py orion/dig -> 489 passed
pytest services/orion-harness-governor/tests -> 1 failure, test_fcc_warm_pool_wiring::test_start_passes_settings_into_pool_config. It fails the same way on main: a host-only PermissionError on /home/athena/.fcc/.env.
scripts/check_env_template_parity.py -> PASS; scripts/check_integration_seam.py -> OK
```

## Evals run

```text
None. This change adds an argv flag and a prompt file, and the CLI probe below is its check. `python -P -m orion.dig take-ons` counts sub-agent calls by type, so it measures uptake after deploy (#3671).
```

## Docker/build/smoke checks

```text
Governor container (CLI 2.1.326), claude -p with a dead ANTHROPIC_BASE_URL, --setting-sources user,local:
  with --agents <this PR's JSON> -> init agents: ['claude','Explore','general-purpose','orion-repo-agent','Plan','statusline-setup']
  without                        -> init agents: ['claude','Explore','general-purpose','Plan','statusline-setup']
No image was built from the worktree, to avoid retagging the live image. The file ships through the existing `COPY orion ./orion` (no .dockerignore).
```

## Review findings fixed

- Finding: the sub-agent model was implicit, and some CLI versions default custom agents to `sonnet`, which FCC would route to a different lane.
  - Fix: `model: inherit` in the file. The parser passes it through.
  - Evidence: a test asserts `model == "inherit"`, and the live probe still lists the agent.
- Finding: a broken agent file would fail every non-reading turn.
  - Fix: `fcc_agents_argv` catches `OSError`/`ValueError`, logs `fcc_agents_unavailable`, and returns `[]`.
  - Evidence: `test_fcc_agents_argv_degrades_to_no_flag_when_the_file_is_missing`.
- Finding: the parser misread YAML block scalars and quoted values, and it rejected CRLF files.
  - Fix: it normalizes CRLF, strips matching quotes, and refuses block scalars.
  - Evidence: `test_fcc_agents_parser_refuses_block_scalars_and_strips_quotes`.
- Finding: the agent file said "gitignored files are lost". That was wrong: those files are not rescued and can be overwritten, but the sync does not delete them.
  - Fix: reworded.
- Finding: one test assertion was true by construction.
  - Fix: it now asserts that the operator file's "read AGENTS.md first" line is missing from the variant.
- Findings (nits): the built-in agent set in the hint test was hand-copied, the README check line was unverified, and nothing said the agent has no MCP tools.
  - Fix: added a snapshot comment to the test. The README check line now cites the live probe. The agent file and README now say it has no MCP tools.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Run from the clean primary `main` after merge. The post-pull rebuild does this on its own.

## Risks / concerns

- Severity: low.
  - Concern: the sub-agent has full Bash in a root container that has the docker socket mounted. Its rules are prompt rules, not mechanical gates, and Orion's parent turn already has the same access.
  - Mitigation: the variant forbids deploys, docker writes, DB writes, pushes to `main`, and merges. Branch protection still gates `main`.
- Severity: low.
  - Concern: about 100 tokens per turn for the agent's description in the Agent tool listing.
  - Mitigation: `HARNESS_FCC_REPO_AGENT_ENABLED=false`.
- Issues opened: #3671 (verify-live). Issues closed: none. #3650 is `Refs`, not `Fixes`, until a real take-on transcript proves it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3672
