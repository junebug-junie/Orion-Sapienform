## Summary

- When Orion runs `dig capacities`, the printout now ends with an `Unclaimed jobs:` block. It has one ready `capacity claim` line per job that no part holds and no claim of Orion's waits on. It tells Orion to write each claim it can back with evidence now, without asking first. A claim is still not a role: Juniper reviews each one.
- Where a rule found a candidate part (live: `orion-equilibrium-service` for S3* audit), the line names that part and gives example cites from its own flows.
- The printout also lists the claims waiting on Juniper (`Waiting on Juniper (N): ask her yes or no on each`), so she can answer in chat instead of reading YAML.
- The block sits in the capped tail, so the 40-line / 4000-char cut never removes it.
- The always-on dig brief line now says: write the claims `capacities` asks for; Juniper reviews them. No new command words: the line still routes by topic.

## Outcome moved

On 2026-11-04 01:47Z (harness transcript `1f8aaec3`), Juniper asked "Would you please use your digging tools to tell me what capabilities you have and which parts do what jobs?" Orion ran `help`, `capacities` and `body`, then ended with "the six capacity gaps are the obvious next `capacity claim` work if you want me to do it." The claims ledger stayed empty. This had happened on several turns. The old printout gave one generic `to claim a job:` line and no instruction to write. Now the printout itself asks for the claims, one ready line per open job.

## Current architecture

- `orion/body_atlas/capacities.py` `render()` prints the VSM (Viable System Model) job lines, Orion's claims and the capacity sentences.
- `orion/dig/__main__.py` `cmd_capacities` adds one line, `to claim a job: <generic command>`, then caps the output.
- `orion/dig/brief.py` holds the one always-on line in the harness prompt: "If Juniper asks, in any words, to look at, name, claim or write anything about your body ... run help first and use what it lists."
- Juniper confirms or rejects a claim only by PR to `config/body_atlas/vsm_confirmed.yaml` (design decision 1). No Hub surface shows claims.

## Architecture touched

- `orion/body_atlas/capacities.py`: `open_jobs()` and `pending_claims()`, pure functions over the reading.
- `orion/dig/__main__.py`: `claim_block()`; `cmd_capacities` puts it in the tail; `capped()` counts a multi-line tail.
- `orion/dig/brief.py`: brief line, `MAX_BRIEF_CHARS` 400 -> 420.
- No bus, schema, env, or API change. The full-result JSON file is unchanged.

## Files changed

- `orion/body_atlas/capacities.py`: which jobs to claim now, which claims wait on Juniper.
- `orion/dig/__main__.py`: the claim block, tail placement, multi-line tail in `capped()`.
- `orion/dig/brief.py`: brief line and its size cap.
- `orion/dig/tests/test_capacities_cmd.py`: regression and lifecycle tests.
- `orion/dig/tests/test_dig.py`: brief line assertion.
- `orion/body_atlas/tests/test_capacities.py`: `open_jobs` / `pending_claims` unit test.
- `orion/dig/README.md`, `orion/body_atlas/README.md`: describe the block.
- `docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`: note under decision 5 (Juniper's ask is any capacities question, in any words).
- `docs/superpowers/pr-reports/2026-11-04-capacities-claim-by-default-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the `dig capacities` printout ends with the claim block and the waiting list. The generic `to claim a job:` line prints only when there is no open job and nothing waiting.
- Compatibility notes: none. #3091 (open) adds a `status:` legend line inside `render()`; this patch does not touch `render()`.

## Env/config changes

- None. No `.env_example` changed.

## Tests run

```text
.venv/bin/python -m pytest orion/dig/tests orion/body_atlas/tests tests/test_organ_names_partition.py tests/test_report_dig_kit_usage.py -q   -> 262 passed
```

Regression test: `test_capacities_output_tells_orion_to_write_claims_now_not_offer_them` (fails on main: no `Unclaimed jobs:` line). Lifecycle: `test_a_ready_claim_line_runs_once_its_because_is_filled_and_then_waits_on_juniper` runs a printed line through the CLI, then checks the job leaves the block and joins the waiting list. Also: a rejected claim reopens the job without its part; the block survives a cut printout.

Live atlas (FalkorDB read-only, scratch `ORION_DIG_HOME`, real ledger untouched), 2026-11-04:

```text
Unclaimed jobs: claim each you can back with evidence now, without asking first (a claim is not a role; Juniper reviews it). The because cites, by full id, a part or channel on that part's flows, or a repo file:
  python -P -m orion.dig capacity claim <part> S3 because "<why>"
  python -P -m orion.dig capacity claim orion-equilibrium-service 'S3*' because "<why; e.g. atlas-worker-1, atlas-worker-2, atlas-worker-agent-1>"
  python -P -m orion.dig capacity claim <part> S4 because "<why>"
  python -P -m orion.dig capacity claim <part> S5 because "<why>"
  python -P -m orion.dig capacity claim <part> algedonic because "<why>"
  python -P -m orion.dig capacity claim <part> metabolism because "<why>"
No evidence for a job: leave it, and say which and why.
```

32 lines, 3254 chars (cap 40 / 4000). After a scratch claim on `orion-equilibrium-service S3*`, the S3* line left the block and `Waiting on Juniper (1)` listed it.

## Evals run

```text
run_capacity_roles_eval.py        -> exit 0 (render() size floor unchanged: the block is outside render())
run_organ_structure_eval.py       -> exit 0
run_store_direction_eval.py       -> exit 0
run_wire_eval.py                  -> exit 0
physiology_state_skill_eval.py    -> exit 0 (brief line length vs MAX_BRIEF_CHARS)
```

## Docker/build/smoke checks

The dig code is baked into the harness image. `scripts/service_rebuild_paths.yaml` maps `orion/dig/` and `orion/body_atlas/` to `orion-harness-governor`, so the post-pull rebuild covers it. Not built here.

Live check after deploy: Juniper asks about capacities in plain words. Expected: `$ORION_DIG_HOME/self/capacity_claims.jsonl` (`/mnt/orion-fcc/self/capacity_claims.jsonl`) gains rows in that same turn, and Orion's reply lists them as waiting on her. UNVERIFIED until that turn happens.

## Review findings fixed

(filled after review)

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild rebuilds `orion-harness-governor`. If it does not run: `scripts/safe_docker_build.sh orion-harness-governor up -d --build`

## Risks / concerns

- Severity: low. Concern: Orion may write weak claims to fill the block. Mitigation: `check_claim` still refuses a because without a real cite, and Juniper reviews every claim; nothing becomes a role without her.
- Severity: low. Concern: confirming still needs a PR to `vsm_confirmed.yaml`; the waiting list makes her yes/no easy to give in chat, but someone still writes the file. No Hub surface for claims exists; none was built (out of scope).

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
