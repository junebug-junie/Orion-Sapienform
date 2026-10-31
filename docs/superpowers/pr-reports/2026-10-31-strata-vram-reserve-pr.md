## Summary

- Both circe Strata lanes (chat on gpu0, agent on gpu1) now keep 1200 MiB of VRAM free instead of Strata setup's 700.
- The compose launch passes `--set "vram-reserve-mib=${STRATA_VRAM_RESERVE_MIB:-1200}"`; `strata launch` puts it over the run config's `--vram-reserve-mib` for that launch.
- New env keys `ATLAS_CHAT_STRATA_VRAM_RESERVE_MIB` and `ATLAS_AGENT_STRATA_VRAM_RESERVE_MIB` (default 1200, ON).
- Context is unchanged (262,144 chat, 131,072 agent). The reserve comes out of the expert cache.
- README section and a compose/env test.

## Outcome moved

Live 2026-10-31, both engines ran with 300-500 MiB free (nvidia-smi: gpu0 291 MiB, gpu1 299 MiB) and logged
`WARNING: only N MiB of VRAM free at the start of this request (below 512 MiB): ... the driver can page GPU memory to
system RAM, decode falls to a few tok/s` (4 times each since start; lowest 302 MiB). With 500 MiB more reserved, the
expected free VRAM is about 1000 MiB at load and about 800 MiB at the lowest seen so far, above the 512 MiB warning line.
UNVERIFIED until Juniper recreates the containers.

## Current architecture

`atlas-chat` / `atlas-agent` share the `x-strata-engine` compose anchor and run `strata launch --no-tui ... --save-preset`.
The engine reads the run config `/data/install/strata-iq3_s.json`, which setup wrote with `--vram-reserve-mib 700`
(and `--expert-cache auto`, `--kv-resident 32768`). Engine start log, both lanes:
`expert cache auto: 23.94 GiB free, 700 MiB reserved (+218 MiB for the draft head)` then
`494 MiB of VRAM free with everything loaded`.

## Evidence that this is the real flag (Strata 0.2.7-cuda12 image, read in the live container)

- Run config args on both lanes: `"--vram-reserve-mib", "700"`.
- `strata_cli/launch/params.py`: `"--vram-reserve-mib": "Graphics memory (in MiB) kept free for other programs."`
- The engine's own warning ends with `Add --vram-reserve-mib 768 to the config's args`.
- `strata launch --help`: `--set KEY=VALUE  an advanced engine setting ... e.g. prefill=65536`.
  `settings.split_setting` turns `vram-reserve-mib` into `--vram-reserve-mib`; `runner.derive_config` applies it with
  `tune_knobs.apply_to_argv`. Run in-container against the live chat run config: one `--vram-reserve-mib`, value `1200`,
  argument count unchanged (29 -> 29).

## Context impact

None. KV cells in VRAM are fixed by `--kv-resident 32768`; the rest streams from pinned RAM. `runner.desc_matches`
(the setup-rerun check) compares only `context`, `kv` and `images`, so a different reserve does not rerun setup.
The expert cache shrinks by ~500 MiB of ~23 GiB (about 2%, ~260 slots). `scripts/check_fcc_context_parity.py` passes.

## Architecture touched

`services/orion-llamacpp-host` compose (Strata engine anchor + two lanes), env template, README, tests. No bus, schema or API change.

## Files changed

- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: `--set vram-reserve-mib` in the launch; per-lane `STRATA_VRAM_RESERVE_MIB`.
- `services/orion-llamacpp-host/.env_example`: two new keys, 1200.
- `services/orion-llamacpp-host/tests/test_strata_apply_defaults.py`: both lanes pass a reserve >= 1200 that matches `.env_example`.
- `services/orion-llamacpp-host/README.md`: "VRAM reserve" section.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: Strata engines keep 1200 MiB VRAM free (was 700).
- Compatibility notes: the compose default `:-1200` covers a host `.env` without the keys.

## Env/config changes

- Added keys: `ATLAS_CHAT_STRATA_VRAM_RESERVE_MIB=1200`, `ATLAS_AGENT_STRATA_VRAM_RESERVE_MIB=1200`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: athena's `services/orion-llamacpp-host/.env` got both keys (the worktree has no `.env`, so the keys were added by hand, as the sync script would). circe's `.env` was not touched (read-only); the compose default applies there.
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-llamacpp-host && python -m pytest tests/test_strata_apply_defaults.py -q -p no:cacheprovider
7 passed
python3 scripts/check_fcc_context_parity.py   -> fcc-context-parity: ok
python3 scripts/check_env_template_parity.py  -> PASS (94 services)
docker compose --env-file .env_example -f docker-compose.atlas-workers.yml --profile agent-split config
  -> both lanes: STRATA_VRAM_RESERVE_MIB: "1200", --set "vram-reserve-mib=$${STRATA_VRAM_RESERVE_MIB:-1200}"
```

`tests/test_worker_announce.py` fails at collection on athena without `LLM_PROFILE_NAME` set (pre-existing, env-only); CI covers it.

## Evals run

```text
No eval harness for Strata engine config. The live check is the engine log line after recreate:
"strata serve: N MiB of VRAM free with everything loaded" should read ~1000, and no "WARNING: only N MiB" lines.
```

## Docker/build/smoke checks

```text
docker compose config (above). No build needed: the engine image is unchanged. Not deployed (Juniper deploys).
```

## Review findings fixed

Code review ran in a subagent (no blocking findings).

- Finding: an empty or missing env value in a future engine lane would become a bare `--vram-reserve-mib` flag.
  - Fix: shell default in the launch line, `$${STRATA_VRAM_RESERVE_MIB:-1200}`; the test now requires that on the launch's own line and fails if a third service merges the engine anchor without being added.
  - Evidence: `docker compose config` renders the default; 7 tests pass.
- Finding: README did not say the reserve now comes from the env, not the run config.
  - Fix: "VRAM reserve" section, including "editing the run config's value has no effect".
  - Evidence: `services/orion-llamacpp-host/README.md`.
- Finding: agent key had no comment; non-numeric values are not checked.
  - Fix: comment in `.env_example` ("Whole MiB only").
  - Evidence: `.env_example`.
- Finding (open): the free-VRAM gain is UNVERIFIED until deploy.
  - Follow-up: after recreate, check the engine log reads `1200 MiB reserved`, `N MiB of VRAM free with everything loaded` ~1000, no `WARNING: only` lines, and gpu0/gpu1 free stays above 512 MiB during a long-context request; note tok/s before and after.

## Restart required

From the clean primary `main` checkout on circe, after merge and pull (recreate, so the new env and command apply):

```bash
scripts/safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml --profile agent-split up -d atlas-chat atlas-agent
```

## Risks / concerns

- Severity: low
- Concern: ~2% smaller expert cache could lower the decode hit rate slightly.
- Mitigation: set the env keys lower (e.g. 900) and recreate; the engine's own suggestion was 768.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
