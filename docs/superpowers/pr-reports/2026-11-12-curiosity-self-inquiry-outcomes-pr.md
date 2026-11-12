## Summary

- Self-inquiry runs (Orion answering questions like "Am I alive?") now write a decision row and an outcome row, the same two rows an investigation writes.
- At dispatch the run writes a `curiosity_offer_decisions` row with `arm = 'self_inquiry'`, nothing offered, propensity 1.0 and no `arm_propensities`. Its turn is snapshotted at start and diffed at end, so `curiosity_run_outcomes` gets its row.
- The readers that model a choice among offered priors skip this arm: the value model's history, the personality choice lanes (Hub-side `choices.py` and durable-runs `curiosity_choice`) and the realized-nats replay. The three-arm readout never sees these rows (it reads only rows with `arm_propensities`).
- New flag `HUB_CURIOSITY_SELF_INQUIRY_OUTCOMES_ENABLED`, default on.
- Step 1 of 3 in the curiosity-findings-as-event-wants stack. Steps 2 and 3 are on hold: no local judge lane passed the agreement gate (see Risks).

## Outcome moved

- Before (live, 30 days to 2026-11-12): 0 of 163 finished self-inquiry runs had an outcome row (investigations: 161 of 164). In the spec's 3-week labelled set, self-inquiry held 97 of 100 philosophical findings. Orion's Day, Temporal Self arcs and reading credit could not see any belief move a self-inquiry run made.
- After: every self-inquiry turn writes one outcome row (realized nats, tested, formed, per-prior moves).

## Current architecture

- `_turn_result_for` measured a turn only when `source_tag == INVESTIGATION_TAG`. `load_turn_snapshot` found no decision row for a self-inquiry run, so `_spend_turn_ended` returned early.
- Spec: `docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md`, "The self-inquiry gap, and the fix".

## Architecture touched

- orion-hub (curiosity loop), orion-durable-runs (personality source query), shared `orion/personality/choices.py`, `orion/curiosity/value.py` (arm constant), one analysis script. No schema, bus or API change.

## Files changed

- `services/orion-hub/scripts/curiosity_investigation.py`: `_record_self_inquiry_decision` at both self-inquiry dispatch sites (`_self_inquire`, `_self_name`); `_measured_turn` measures self-inquiry turns; the correction and focus stamps stay investigation-only.
- `services/orion-hub/scripts/curiosity_offer_decisions.py`: `LOAD_HISTORY_SQL` skips outcomes whose decision row is `arm = 'self_inquiry'` (an outcome with no decision row is kept, as before).
- `orion/curiosity/value.py`: `ARM_SELF_INQUIRY`.
- `orion/personality/choices.py`: a self-inquiry decision row returns `self_inquiry_no_offer`, not a choice.
- `services/orion-durable-runs/app/personality_sources.py`: the `curiosity_choice` lane query filters `d.arm <> 'self_inquiry'`.
- `scripts/analysis/replay_curiosity_realized_nats.py`: the `--pg` read is investigations only.
- `scripts/analysis/measure_rest_drive_easing.py`: the curiosity consumer turns are investigation turns only.
- `.github/workflows/orion-hub-curiosity-spend-pg.yml` (new): runs the Hub Postgres test, must not skip.
- `services/orion-hub/app/settings.py`, `.env_example`, `scripts/main.py`, `README.md`: the flag.
- Tests: `services/orion-hub/tests/test_curiosity_spend_log.py`, `test_curiosity_self_inquiry.py`, `test_curiosity_self_naming.py`, `test_curiosity_self_inquiry_outcomes_pg.py` (new, real Postgres), `orion/personality/tests/test_choices.py`, `services/orion-durable-runs/tests/test_personality_loop.py`, `test_personality_postgres.py`, `scripts/analysis/tests/test_measure_rest_drive_easing.py`.

## Schema / bus / API changes

- Added: none (new value `self_inquiry` in the existing `curiosity_offer_decisions.arm` text column).
- Removed: none.
- Renamed: none.
- Behavior changed: self-inquiry runs write decision and outcome rows.
- Compatibility notes: readers that include these rows on purpose (spec reader table): Temporal Self (`orion/temporal_self/sources.py`, durable-runs `temporal_self_sources.py`), Orion's Day (`orion/orion_day/gather.py`), reading credit (`orion/world_pulse_read/findings.py`), the introspect listener and `temporal_fact.py`. Replica (`curiosity_offer_plans` join) and the three-arm readout (`arm_propensities IS NOT NULL`) cannot see them.

## Env/config changes

- Added keys: `HUB_CURIOSITY_SELF_INQUIRY_OUTCOMES_ENABLED=true` (orion-hub).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-hub/.env` has the key).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub: pytest tests -k "curiosity or self_inquiry or spend or naming" -> 663 passed, 6 skipped
services/orion-hub: ORION_TEST_PG_URI=<throwaway pg16> pytest tests/test_curiosity_self_inquiry_outcomes_pg.py -> 1 passed
orion/personality/tests + orion/curiosity/tests + scripts/analysis rest-drive and realized-nats tests -> 466 passed, 29 skipped
services/orion-durable-runs: pytest tests/test_personality_loop.py tests/test_personality_postgres.py (throwaway pg16) -> 39 passed
mutation checks: removing the dispatch call fails 2 tests; removing the durable-runs SQL filter fails
  test_full_loop_on_postgres (before the choices.py skip) and now the SQL assertion
```

## Evals run

```text
No new eval: this patch adds rows, it does not change a decision. The live check is the
acceptance query below after deploy.
```

## Docker/build/smoke checks

```text
Not run (no dependency, compose or boot change). Live path UNVERIFIED until deploy.
After deploy, the next self-inquiry run should show:
  SELECT d.run_id, d.arm, o.realized_nats, o.n_tested, o.n_formed
  FROM curiosity_offer_decisions d JOIN curiosity_run_outcomes o USING (run_id)
  WHERE d.arm = 'self_inquiry' ORDER BY d.decided_at DESC LIMIT 5;
and the Hub log line `curiosity_spend_outcome run=<id> ...` for that run.
```

## Review findings fixed

Code review ran in a subagent (read-only, on this diff). No blockers. Fixed:

- Finding: `scripts/analysis/measure_rest_drive_easing.py` counted self-inquiry turns as investigation turns (they now stamp `turn_started_at`).
  - Fix: its consumer query filters `arm <> 'self_inquiry'`.
  - Evidence: `scripts/analysis/tests/test_measure_rest_drive_easing.py` asserts the filter.
- Finding: the direct (non-cortex) self-inquiry turn wrote the decision row but was never measured.
  - Fix: that branch now snapshots and diffs like the investigation's direct branch.
  - Evidence: `test_a_self_inquiry_run_writes_one_self_inquiry_decision_row_and_is_measured` (runs with `kickoff_via_cortex=False`, asserts the snapshot write).
- Finding: no test covered the two dispatch sites.
  - Fix: tests drive `tick_self_inquiry` for a pool question and for a naming sitting.
  - Evidence: removing the `_record_self_inquiry_decision` call fails both tests (checked).
- Finding: the value-history exclusion was proven only in a skipped Postgres test.
  - Fix: new workflow `.github/workflows/orion-hub-curiosity-spend-pg.yml` runs it on a real Postgres and fails if it skips.
  - Evidence: run by hand on a throwaway postgres:16 (1 passed).
- Finding: the durable-runs Postgres test passed even without the SQL filter (choices.py also skips the arm).
  - Fix: `test_sql_is_keyset_capped_and_chat_is_naive_utc` asserts the SQL filter.
  - Evidence: `tests/test_personality_loop.py` passes; the assertion names the filter.
- Finding: `ARM_SELF_INQUIRY` was defined twice.
  - Fix: `choices.py` imports it from `orion/curiosity/value.py` (stdlib-only module).
- Finding: no test checked that the correction and focus stamps stay investigation-only.
  - Fix: the measured self-inquiry test asserts neither stamp runs.
- Finding: `value_arm_propensity=1.0` reads oddly for an arm with no draw.
  - Fix: comment at the write site (the spec's value; every weighting reader filters by arm or `arm_propensities`).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: a self-inquiry turn now takes two prior snapshots (up to 2,000 prior rows each) and one revision read inside its time budget, and each dispatch runs the snapshot prune. Mitigation: the same reads and prune investigations already pay; best-effort, never raise into the turn.
- Severity: info. Concern: steps 2 and 3 (finding register and curiosity event wants) are blocked. No local lane that is not a writer seat passed the judge agreement gate (kappa >= 0.8): metacog Qwen3-8B Q5 kappa 0.58, same with thinking 0.59, quick Qwen3-8B Q4 kappa 0.60. Mitigation: nothing ships a judge; the decision goes back to Juniper.

## PR link

PR_LINK_PLACEHOLDER
