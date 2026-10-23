# Strain arcs say which causes are present now

## Summary

- On 10-23 the cabinet cooled from about 33.5 °C to 28.4 °C, but Orion's chat cue still said "Strained since Sun 06:07 (cabinet too hot, GPU jobs backed up)". The arc label keeps every cause seen in the episode, and no row was written when a cause left.
- Fix:
  - The regulate step (orion-durable-runs) now also writes a strain row when the causes present differ from the last row's. This is limited to one row per 10 min, so a flapping cause can't flood the table.
  - The reducer keeps `causes_now` on the strain arc.
  - The stance cue now says what is true: "(now: GPU jobs backed up; earlier: cabinet too hot)", or "(no cause right now, settling; earlier: …)" while the strain only holds.
- Follow-up to #2817 (strain loop part 1).

## Outcome moved

Orion's day cue stops telling them a cleared cause (the cabinet) is still present.

## Current architecture

Strain rows were written on a level change, when a cause joined the episode, as a seed row, and as a 30-min keepalive. The label only grew. The cue rendered the label.

## Architecture touched

- **orion-durable-runs:** `_strain_causes_step`, new checkpoint key `strain_causes_now`, `CARRIED_KEYS`.
- **orion/temporal_self:** arcs, frame and stance cue.
- **Schema:** nullable `causes_now` on `TemporalSelfArcV1` and `ArcSummaryV1`.

## Files changed

- `services/orion-durable-runs/app/temporal_self_graph.py`: row on a cause change (`STRAIN_CAUSE_CHANGE_MIN_SEC` = 600). A pre-deploy checkpoint gets a row after 10 min, not 30.
- `services/orion-durable-runs/app/temporal_self_driver.py`: carry `strain_causes_now` over midnight.
- `orion/schemas/temporal_self.py`: `causes_now: list[str] | None`. `None` means not recorded, which is not the same as "none present".
- `orion/temporal_self/arcs.py`, `frame.py`: set the field from every strained row, copy it across midnight, and put it in the summary.
- `orion/temporal_self/stance_cue.py`: now/earlier wording. It keeps the plain label wording when nothing cleared, or when `causes_now` is None.
- Tests: `orion/temporal_self/tests/test_strain_arcs.py`, `services/orion-durable-runs/tests/test_temporal_self_regulate.py`.
- READMEs: durable-runs and temporal_self.

## Schema / bus / API changes

- **Added:** `causes_now` on `TemporalSelfArcV1` and `ArcSummaryV1` (optional, nullable).
- **Compatibility:** these models forbid unknown fields, and every stored frame now carries the key. **orion-thought and orion-hub must deploy before orion-durable-runs.** Otherwise the stance cue reads `invalid_frame` and goes dark until they are rebuilt. Rolling back durable-runs also needs a newer reader, because the stored state carries the field.
- No bus channel and no migration.

## Env/config changes

None.

## Tests run

```text
orion/temporal_self/tests                                   218 passed
services/orion-durable-runs/tests (disposable Postgres)     564 passed, 1 skipped (before review fixes)
services/orion-durable-runs/tests/test_temporal_self_regulate.py  35 passed, 3 skipped (after)
services/orion-thought stance cue + recent arcs             25 passed
services/orion-hub day + temporal-self routes               36 passed
Mutation checks: drop the "cause left" rule -> leave test fails; drop the post-deploy None rule -> deploy test fails
```

## Evals run

```text
python services/orion-durable-runs/evals/temporal_self_live_replay.py --step-sec 600   "passed": true
python orion/temporal_self/evals/run_arc_precision_eval.py                             exit 0
```

## Docker/build/smoke checks

```text
Gates: check_definition_drift --gate, check_inner_state_registry, check_metric_lineage --gate,
check_env_template_parity: all pass. git diff --check clean.
```

## Review findings fixed

- **Finding (MEDIUM):** deploying durable-runs before thought/hub darkens the stance cue.
  - **Fix:** the order is stated in the schema docstring and in the restart section below.
- **Finding (LOW):** after deploy, a cause that already left could wait 30 min for the keepalive.
  - **Fix:** a missing `strain_causes_now` counts as changed once 10 min have passed.
  - **Evidence:** `test_first_steps_after_deploy_record_causes_now_within_10_min` fails without it.
- **Finding (NIT):** a seed while the strain only holds renders "settling" with no label, and nothing tested it.
  - **Fix:** `test_seed_while_only_holding_renders_settling_with_no_label`.
- **Finding (LOW, by design):** "now" can be up to 10 min stale. Accepted, and stated in the README.

## Restart required

Consumer-first, from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** deploying in the wrong order darkens the cue.
  - **Mitigation:** the order above.
- **Not fixed here, on purpose:** Orion stays "strained" because their own background GPU backlog never drains below 2. That is routed to the body-economy design (#2855) through an agent-board finding, by Juniper's call on 10-23.
- **UNVERIFIED until deploy:** after durable-runs restarts, a strain row should land within 10 min, and the cue should read "Strained since Sun 06:07 (now: GPU jobs backed up; earlier: cabinet too hot)".

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
