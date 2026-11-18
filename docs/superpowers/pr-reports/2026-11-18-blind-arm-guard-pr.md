## Summary

Orion's read-only SQL can no longer read the stored arm of the two blinded experiments (curiosity three-arm test #3183, dream hypothesis dream/control). Fixes #3458.

- New hand-apply script `scripts/sql/2026-11-18_blind_experiment_arms_orion_readonly.sql`. It puts a view `orion_scope.temporal_self_event` in front of the table for the role `orion_readonly`. The view drops the `arm` key from `payload_json`, except on a `self_inquiry` curiosity run. Then it revokes the table grant. It is **not applied** by this PR.
- orion-durable-runs now stores each window's Temporal Self events in `event_id` order. Before, dream hypotheses were stored dream arm first, so row order told the arm.
- The curiosity readout prints the lift step once the test reads "Stopped".
- A real-Postgres test and a CI workflow check the guard, as `orion_readonly`.

## Outcome moved

Before: `orion_readonly` (the role behind `ORION_CURIOSITY_PG_DSN` in every turn and behind the dig kit's `ReadOnlyPsql`) could read `arm` on all 216 `curiosity_run` and 179 `dream_hypothesis` rows of `temporal_self_event`, and the same values through `pg_stats`. After apply: the role reads the same rows without the arm, and `public.temporal_self_event` is refused.

## Current architecture

- Orion's turns run as root in `orion-athena-harness-governor`, with raw `psql` and `ORION_CURIOSITY_PG_DSN` = `orion_readonly`. A code check in one runner cannot close a raw-psql path.
- `orion_readonly` has `search_path = orion_scope, public`. `orion_scope` already holds the AI Town filter views (`scope_orion_readonly_off_aitown.sql`).
- Live grants (checked 2026-11-18, read-only): `curiosity_offer_decisions`, `dream_hypothesis`, `dream_cycle` were **not** granted (the issue comment assumed they were). `temporal_self_event` was granted (11-16 roundup script). That was the only leak.
- Dreams' `BLIND_FORBIDDEN_COLUMNS` (`orion/dream/introspect_sql.py`) is a convention for the SQL that dream code writes. Only a test enforces it. It does not bind raw psql.

## Where the guard sits and why

At the database privilege layer, the same pattern as the AI Town scope. The view has the table's name and comes first on the role's search_path, so the dig kit's `FROM temporal_self_event` queries keep working unchanged. The REVOKE makes it a boundary. Every arm on every source kind is hidden by default. Only `self_inquiry` passes, because it is not an offer arm and the now already names it.

## Files changed

- `scripts/sql/2026-11-18_blind_experiment_arms_orion_readonly.sql`: the guard, with apply, verify, lift and undo lines.
- `scripts/sql/2026-11-16_grant_orion_readonly_organ_roundup.sql`: warning: re-running its temporal_self_event line reopens the arm.
- `services/orion-durable-runs/app/temporal_self_store.py`: `window_event_rows` stores rows in event_id order.
- `services/orion-durable-runs/tests/test_temporal_self_blind_row_order.py`: the order test.
- `scripts/analysis/curiosity_three_arm_readout.py` + its test: prints the lift step at the stop.
- `tests/test_blind_experiment_arms_sql.py`: real-Postgres test of the guard.
- `.github/workflows/orion-readonly-blind-guard.yml`: runs that test against a throwaway Postgres 15.

## Schema / bus / API changes

- Added: view `orion_scope.temporal_self_event` (after the manual apply).
- Removed: `orion_readonly` SELECT on `public.temporal_self_event` (after the manual apply).
- Behavior changed: Temporal Self event insert order within one window (no reader depends on it).
- Bus, schema registry, API: none.

## Env/config changes

- None. No `.env_example` changed.

## Tests run

```text
pytest tests/test_blind_experiment_arms_sql.py scripts/analysis/tests/test_curiosity_three_arm_readout.py orion/temporal_self/tests/test_now.py
  55 passed (guard test on a local initdb Postgres; also passed in a clean venv with only pytest/psycopg2/pydantic)
pytest services/orion-durable-runs/tests   1024 passed, 149 skipped
Mutation: the view without "- 'arm'" -> guard test fails; no sort in window_event_rows -> order test fails.
test_now.py::test_the_now_never_shows_a_blinded_experiment_arm passes.
```

## Evals run

No eval: the guard is a privilege boundary, and the real-Postgres test is its proof. After apply, run the verify lines in the SQL file (live check).

## Docker/build/smoke checks

None run. The durable-runs change is a sort before an existing insert.

## Review findings fixed

- Finding: dream arm leaked through stored row order (43 of 44 mixed cycles, aggregate check).
  - Fix: `window_event_rows` sorts by event_id (random hypothesis id); test added. Old rows keep their order until a table rewrite; documented, not done (locking production operation).
- Finding: the curiosity arm is a sha256 of the run id, so it can be re-derived.
  - Fix: not fixable by a view, and changing the draw changes a pre-registered randomization. Filed #3685 (needs-juniper); the guard file states it hides the stored arm only.
- Finding: test seeded hand-written payloads; a renamed arm key would pass.
  - Fix: seeds now come from the real producers `orion.temporal_self.sources.curiosity_run` / `dream_hypothesis`, and the test walks every JSON leaf for any arm name.
- Finding: `pg_stats` exposed payload values; not pinned.
  - Fix: test ANALYZEs, shows the stats are readable before, and asserts none after.
- Finding: CI path filter, ADD COLUMN regex, PG version pick, stale readout wording, lift-block column drift.
  - Fix: all done (paths `scripts/sql/**`, `orion/temporal_self/sources.py`; wider regex; prefers PG 15; "If ... still blind"; note in the LIFT section).

## Restart required

Apply the SQL by hand (orchestrator, from the primary checkout on main after merge):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-18_blind_experiment_arms_orion_readonly.sql
```

orion-durable-runs picks up the row-order change with the normal post-pull rebuild.

## Risks / concerns

- Severity: high. Concern: the arm is still knowable by re-derivation (#3685) and by superuser psql through the docker socket (#3680). Mitigation: both filed needs-juniper.
- Severity: low. Concern: old dream rows keep arm-ordered storage. Mitigation: documented; needs a table rewrite if Juniper wants it.
- Severity: low. Concern: re-running the 11-16 grant line reopens the leak. Mitigation: warning in that file; CI runs the guard test on any `scripts/sql/**` change (it does not detect a live re-run).
- Issues opened: #3680, #3685. Closed: #3458 (on merge).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3687
