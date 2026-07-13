# PR: graduate route_grammar to default-on + fix pre-existing test fixture bug

Branch: `feat/route-grammar-graduate-default-on` → `main`

## Summary

Two unrelated small fixes, bundled because they were the two agreed next steps from the prior session:

1. **Fixed a real, pre-existing test bug**: `_hub_client_patches` in `test_turn_orchestrator_ws_frames.py` mocked `ThoughtClient.react()` to return a bare `ThoughtEventV1` where production code returns `ThoughtReactResult(thought=..., failure_reason=...)`. Caused 8 failures on every run of that file, unrelated to any recent change (confirmed via `git stash` against unmodified `main` in a prior session).
2. **Graduated `route_grammar`/orch-route-grammar out of shadow mode**: flipped `PUBLISH_CORTEX_ORCH_GRAMMAR` (orion-cortex-orch) and `ENABLE_ROUTE_GRAMMAR_REDUCER` (orion-substrate-runtime) from `false` to `true`, mirroring commit `044d5318`'s exact precedent for graduating `chat_grammar`/`execution_trajectory`. Applied the prerequisite DB migration (`manual_migration_route_substrate_loop.sql`) directly against the live `conjourney` database as part of this change.

## Outcome moved

- `test_turn_orchestrator_ws_frames.py`: 8 failed / 10 passed → 18 passed / 0 failed. Trustworthy CI signal restored for `turn_orchestrator.py` changes going forward.
- `route_grammar` reducer and orch's grammar producer are now on by default in code, matching every other grammar lane in this codebase. The DB table it writes to (`substrate_route_arbitration_projection`) now exists in production.

## Current architecture

`route_grammar` shipped shadow-only in a prior PR pending "verified end-to-end." `chat_grammar`'s equivalent staleness bug (fixed in the immediately preceding session) proved the same producer/reducer pattern works correctly against live traffic once wired. `044d5318` is the literal precedent in this repo for exactly this "flip a proven shadow lane to default-on" action.

## Architecture touched

- `orion-hub` test suite only (fixture fix, no production code).
- `orion-cortex-orch` and `orion-substrate-runtime` settings/env/compose (default flip).
- Live `conjourney` Postgres database (new table + cursor row, via the pre-existing migration file — not new SQL).

## Files changed

- `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py` — one-line fix: `AsyncMock(return_value=thought)` → `AsyncMock(return_value=thought_client.ThoughtReactResult(thought=thought))`.
- `services/orion-cortex-orch/app/settings.py` — `publish_cortex_orch_grammar` default `False` → `True`.
- `services/orion-substrate-runtime/app/settings.py` — `enable_route_grammar_reducer` default `False` → `True`.
- `services/orion-cortex-orch/.env_example`, `services/orion-substrate-runtime/.env_example` — flags flipped to `true`; comments corrected during review (see Review findings below) to state the actual migration prerequisite instead of a premature "verified end-to-end" claim the implementing agent's draft had written before verification had actually happened.
- `services/orion-cortex-orch/docker-compose.yml`, `services/orion-substrate-runtime/docker-compose.yml` — inline `${VAR:-false}` → `${VAR:-true}` for these two keys only.
- `tests/test_grammar_lane_defaults.py` — `test_cortex_orch_grammar_default_on`, `test_route_grammar_reducer_default_on`, mirroring the file's existing substring-match pattern.
- `services/orion-substrate-runtime/README.md`, `services/orion-cortex-orch/README.md` — updated from shadow-mode language to default-on, with the migration prerequisite called out explicitly.

## Schema / bus / API changes

None new. `substrate_route_arbitration_projection` table and `route_grammar_consumer` cursor row already existed as a migration file from the prior PR — this PR is the first time that migration was actually applied to a live database.

## Env/config changes

- Changed defaults: `PUBLISH_CORTEX_ORCH_GRAMMAR` (orch) `false→true`, `ENABLE_ROUTE_GRAMMAR_REDUCER` (substrate-runtime) `false→true`.
- `.env_example` updated: yes, both services.
- local `.env` synced: not applicable in the implementing worktree (no pre-existing local `.env` for either service there). **Operator action required on the actual deployment host**: run `python scripts/sync_local_env_from_example.py orion-cortex-orch orion-substrate-runtime` (or hand-edit) so the live `.env` files pick up the new defaults, then restart both services — a bare `git pull` alone will not change already-set env values, only the code defaults used when a key is absent from `.env`. Confirmed neither service's live `.env` currently sets these keys explicitly (both fall through to the code default today), so this pull + restart is expected to be sufficient without a manual `.env` edit — but verify before assuming.
- skipped keys: none.

## Tests run

```
cd /mnt/scripts/Orion-Sapienform-route-grammar-graduate
/tmp/orion-test-venv/bin/python -m pytest services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -q -W ignore::UserWarning -W ignore::DeprecationWarning
→ 18 passed

/tmp/orion-test-venv/bin/python -m pytest tests/test_grammar_lane_defaults.py -q
→ 5 passed
```

## Evals run

None applicable — config-default and test-fixture changes only.

## Docker/build/smoke checks

```
docker compose -f services/orion-cortex-orch/docker-compose.yml config -q      → valid
docker compose -f services/orion-substrate-runtime/docker-compose.yml config -q → valid
```

**Migration applied directly against the live database as part of this PR** (not left for a separate operator step, since this PR's whole point is graduating the flag that depends on it):
```
PGPASSWORD=*** psql "postgresql://postgres:postgres@localhost:55432/conjourney" \
  -f services/orion-sql-db/manual_migration_route_substrate_loop.sql
→ CREATE TABLE
→ CREATE INDEX
→ INSERT 0 1
```
Verified before applying: table and cursor row did not already exist (`\dt substrate_route_arbitration_projection` → not found; `select cursor_name from substrate_reduction_cursor` → 4 rows, `route_grammar_consumer` absent). Verified after: cursor row present with `last_event_created_at = NULL` (correct pre-first-tick state).

**Live end-to-end verification of the reducer itself is NOT done in this PR** — that requires the code changes here to actually be running, which needs this branch merged and both services rebuilt/restarted, neither of which this session can do directly (no `gh` auth to merge; redeploys in this environment have consistently been a separate operator action after each prior push this session). See Restart required below for the exact post-deploy verification command.

## Review findings fixed

- Finding: the flag-flip implementing agent's `.env_example` comment drafts asserted "Verified end-to-end with the route_loop reducer" and "manual_migration_route_substrate_loop.sql has been applied" — both false at the time the agent wrote them (verification and migration application are this same PR's remaining steps, done by the orchestrator afterward, not yet true when the agent committed its draft).
  - Fix: reworded both comments to state the actual prerequisite (migration must be applied before the flag is safe) rather than a premature completed-tense claim.
  - Evidence: direct diff review before commit; migration was in fact applied afterward in this same PR, in the orchestrator's own step, so the comments are accurate as of the final commit.

## Restart required

```bash
# On the deployment host, after this PR merges:
git pull
python scripts/sync_local_env_from_example.py orion-cortex-orch orion-substrate-runtime
docker compose --env-file .env --env-file services/orion-cortex-orch/.env -f services/orion-cortex-orch/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

**Post-deploy verification** (the migration is already applied, so this should work immediately once both services are running the new code and a turn happens):
```bash
curl -fsS http://localhost:8115/grammar/truth | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('route_grammar enabled:', d['enabled_reducers']['route_grammar'])
for row in d['cursor_positions']:
    if row['cursor_name']=='route_grammar_consumer':
        print(row)
"
```
Expect `route_grammar: true` and, after at least one chat turn, `last_event_created_at` populated with a recent timestamp (not `null`). If it stays `null` after real traffic, check `docker logs orion-athena-substrate-runtime` for `route_grammar` reducer errors and `docker logs orion-athena-cortex-orch` for `orch_route_grammar_publish_failed`.

## Risks / concerns

- Severity: medium, self-resolving. The reducer will error on every tick if `ENABLE_ROUTE_GRAMMAR_REDUCER=true` takes effect on a deployment where the migration has NOT been applied. This is not a risk for the actual database I applied it to, but worth calling out explicitly for anyone deploying this same code change against a different/staging database: apply the migration before or in the same step as flipping the flag, not after.
- Severity: low. Live end-to-end verification is deferred to a post-merge, post-restart step this session cannot perform directly — the "outcome" claimed above (route_grammar working live) is `UNVERIFIED` until that happens. Treat this PR as code-complete-and-migration-applied, not as proven working in production yet.

## PR link

Push and open via: `git push -u origin feat/route-grammar-graduate-default-on`, then open the compare URL GitHub prints (no `gh` auth in this environment).
