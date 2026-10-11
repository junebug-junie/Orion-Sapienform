## Summary

- New Orion tool `mcp__orion-introspect__record_correction`. After a conversation establishes that a claim in an Orion's Day letter was wrong, Orion records it. A correction holds:
  - the ref (e.g. `2026-10-09 ¶18`)
  - the claim's words
  - the correction
  - the evidence
  - the ids of the priors the claim rested on
- The Hub validates it and writes one `orion_day_correction` row:
  - the ref must resolve
  - every prior must exist, checked with a read-only `GRAPH.RO_QUERY`
  - the turn must be a chat turn

  It never writes the belief graph. Writing that graph is reserved to Orion's own curiosity runs, and the database enforces it, because `RO_QUERY` refuses writes.
- **Delivery.** Each correction is offered exactly once to the next investigate curiosity run, up to 5 per run, with the same take/release shape as carry-forward. The kickoff shows each named prior's current confidence and times_tested, and asks Orion to re-test it against the correction. A correction is evidence, not an override.
- **Proof.** At turn end the row is stamped with which named priors that run actually re-tested (`last_run_id` = that run, and times_tested went up) and which moved. A named prior the run never touched is in neither list.
- **The next letter.** Corrections are added to the next letter's material, the writer's digest (`[correction:<id>]`), and the email, under "Corrections from conversations".

## Outcome moved

Before, a falsehood found in conversation was lost, and the same wrong claim could be re-derived the following week. After, it becomes evidence that reaches Orion's beliefs through the only allowed door, and the row shows whether the belief was actually re-tested. A correction offered and never acted on stays visible as exactly that.

## Current architecture

- Patches 1 and 2 (#2606, #2613) gave letter parts, a claim check, and the `orion_day` reread tool.
- Carry-forward (`orion/orion_day/carry_forward.py`) is the only Orion's Day input to curiosity.
- Priors change only through curiosity-run writes (the PR #2199 boundary).

## Architecture touched

- **Contract:** `RecordCorrectionArguments`, `OrionDayCorrectionV1`, and `OrionDayMaterialV1.corrections` (default empty)
- **Channel:** `orion:introspect:record_correction:request`
- **New table:** `orion_day_correction`
- **Hub:** correction responder, curiosity take/release/stamp, kickoff section, email
- **Next letter:** gather and budget

## Files changed

- `services/orion-sql-db/manual_migration_orion_day_correction_v1.sql`: new table. Header `ORION-MIGRATION-REQUIRED-BY: orion-hub`.
- Contract and registration files:
  - `orion/schemas/introspect.py`
  - `orion/schemas/registry.py`
  - `orion/schema_skew_discovery.py`
  - `orion/introspect/transport.py`
  - `orion/bus/channels.yaml`
  - `config/metrics/metric_definitions.lock.json` (re-locked for the new channel)
- `orion/introspect/tools.py`, `orion/introspect/brief.py`: tool and exact-name guide line.
- `orion/orion_day/corrections.py` (new): take/release, `read_named_priors`, `correction_outcome`, stamping, kickoff section.
- `services/orion-hub/scripts/orion_day_correction_listener.py` (new), plus the `scripts/main.py` wiring.
- `services/orion-hub/scripts/curiosity_investigation.py`, `orion/curiosity/kickoff_prompt.py`: delivery and stamping.
- Next-letter path:
  - `orion/schemas/orion_day.py`
  - `orion/orion_day/gather.py`
  - `orion/orion_day/budget.py`
  - `orion/orion_day/letter_parts.py`
  - `services/orion-hub/scripts/orion_day_email.py` + template
  - `services/orion-hub/scripts/orion_day_letter.py`
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: flag, ON.
- READMEs (hub, governor) and spec status.
- Tests:
  - `orion/introspect/tests/test_record_correction_tool.py`
  - `orion/orion_day/tests/test_corrections.py`
  - `services/orion-hub/tests/test_orion_day_correction_listener.py`
  - shared fakes in `orion/orion_day/tests/fixtures.py`
- Eval: `services/orion-hub/evals/run_orion_day_correction_eval.py` (+ wrapper). Correction texts were added to the email eval's required texts.
- CI: `.github/workflows/orion-day-letter-hub-tests.yml`.

## Schema / bus / API changes

- Added:
  - `RecordCorrectionArguments`
  - `OrionDayCorrectionV1`
  - `OrionDayMaterialV1.corrections`
  - channel `orion:introspect:record_correction:request`
  - table `orion_day_correction`
- Removed: none
- Renamed: none
- Behavior changed: an investigate curiosity kickoff can include a "Corrections from conversations with Juniper" section.
- Compatibility notes: `OrionDayMaterialV1` forbids unknown fields. An empty corrections list is left out of the durable brief, so an older orion-durable-runs only fails on a day that actually has corrections. Deploy durable-runs before hub anyway.

## Env/config changes

- Added keys: `HUB_CURIOSITY_ORION_DAY_CORRECTIONS_ENABLED=true` (orion-hub)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-hub: +HUB_CURIOSITY_ORION_DAY_CORRECTIONS_ENABLED='true'`)
- skipped keys requiring operator action: none

## Tests run

```text
orion/orion_day + orion/introspect/tests + orion/curiosity + test_mcp_tool_names + test_schema_skew_discovery + test_harness_prefix:
  479 passed, 2 skipped
services/orion-hub: correction listener, orion_day listener, curiosity_investigation, orion_day_letter: 252 passed
orion/fcc/tests: 1 failed (test_no_undiscovered_root_callers_of_claude_permission_argv, pre-existing on main)
Static repo gates job, every step run locally: exit 0
  (definition drift, migration required-by, settings defaults, node tests, ...)
```

## Evals run

```text
run_orion_day_correction_eval.py  PASS. A planted correction on a fixture prior: that run's re-tested
  prior is stamped; an untouched named prior is absent; a prior moved by a different run is not credited.
  Mutation check: breaking the "stamped by this run" check fails it.
run_orion_day_reread_eval.py      PASS
run_orion_day_email_eval.py       PASS (now including the corrections section)
```

## Docker/build/smoke checks

```text
Not run. Acceptance checks 7-10 against the live DB, graph and bus are UNVERIFIED until the migration
is applied and the services are deployed.
```

## Review findings fixed

Code review in a subagent found 0 blockers and 7 should-fix. Fixed:

- Finding: a deploy-skew risk on the forbid-extra material model.
  - Fix: an empty corrections list is omitted from the brief.
  - Evidence: test
- Finding: a mid-turn cancel could re-offer a correction Orion had already seen.
  - Fix: only a cancel before the turn releases the claim.
  - Evidence: test
- Finding: "moved" and "re-tested" could contradict each other.
  - Fix: anything moved also counts as re-tested.
  - Evidence: test
- Finding: the tool and guide text promised a re-test would happen.
  - Fix: the text now says the correction is offered to a run.
- Finding: an outage reported "not recorded".
  - Fix: it now says "answer unknown".
- Finding: an impossible ref date was reported as an outage.
  - Fix: it is now a refusal.
- Finding: an oversized graph had no distinct message.
  - Fix: own message.

## Restart required

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_orion_day_correction_v1.sql
```

Then, from the main checkout after merge, in this order (durable-runs before hub):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-durable-runs orion-hub orion-harness-governor; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

## Risks / concerns

- Severity: medium
  - Concern: a run that takes corrections but whose durable registration is never confirmed leaves them offered and never stamped. Carry-forward has the same gap. The row shows it (`offered_at` set, `outcome_stamped_at` NULL), but nothing reaps it.
  - Mitigation: follow-up reaper checked against durable-runs.
- Severity: medium
  - Concern: live path UNVERIFIED (acceptance check 10).
  - Mitigation: after deploy, record one real correction in chat. Then confirm the named prior's `times_tested` rises, with `last_run_id` set to the offered run, and that the row is stamped.
- Severity: low
  - Concern: stamping repeats a full belief read at turn end.
  - Mitigation: bounded at 2,000 priors.
- Severity: low
  - Concern: a failed final turn is stamped "re-tested: none" without saying the turn failed.
  - Mitigation: none yet.
- Severity: low
  - Concern: prior ids containing spaces or quotes can't be named.
  - Mitigation: the tool description says so.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2618

🤖 Generated with [Claude Code](https://claude.com/claude-code)
