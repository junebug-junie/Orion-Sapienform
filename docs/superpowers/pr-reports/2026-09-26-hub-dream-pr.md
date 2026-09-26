## Summary

- Add **Dream** to Hub's section launcher, with direct link `/#dream`.
- Show real sleep readiness, recent cycles, replay reasons, proposed links and waking offer state.
- Compare dream and random-control hypotheses using the existing scorecard.
- Explain which older memory paths participate, with links to Reverie and Curiosity.
- All operator routes are read-only; browsing never consumes an offer or starts sleep.

## Outcome moved

An operator can inspect a sleep from readiness through waking offers without
manual SQL or running the scorecard script. Failed reads are visibly unavailable,
not falsely empty. Enable flags and the minimum interval gate cannot be mistaken
for permission to sleep just because pressure is high.

## Current architecture

PR #2353 added `orion-dream`'s pressure/replay/recombination loop, three SQL tables,
a blind waking-offer seam in Hub curiosity, and a shared scoring function.
Hub already had a Reverie tab but no Dream surface. Hub uses FastAPI script
routers, `app/settings.py` (no root `settings.py`), `templates/index.html`, and
`static/js/app.js`; its entrypoint is `scripts/main.py`. Tests and evals both exist.

## Architecture touched

Hub only: GET routes proxy the dream HTTP contract, read its documented SQL
receipts, and query priors via `WorldviewReader`'s `GRAPH.RO_QUERY`. Existing
`POSTGRES_URI` and curiosity graph settings are reused. No bus subscriptions,
events, schema registry changes, sleep algorithms or waking prompts changed.

The surface displays existing observations; it introduces no metric into a
pipeline or cognition loop. Provenance: pressure is produced by
`services/orion-dream/app/cycle.py:read_pressure`; cycle receipts are written by
`app/cycle_store.py:persist_cycle`; offer state is written by
`orion/dream/hypotheses.py:take_hypotheses_for_offer`; scores use that module's
`score_hypotheses`. The UI's readiness boolean combines the existing enable,
pressure/idle and interval gates. It is not an independent signal or detector.

## Files changed

- `services/orion-hub/scripts/dream_routes.py`: four read-only APIs.
- `services/orion-hub/scripts/api_routes.py`: router registration.
- `services/orion-hub/static/js/dream-tab.js`: readiness, scoring, paging and details.
- `services/orion-hub/static/js/app.js`: Dream navigation/lifecycle.
- `services/orion-hub/templates/index.html`: operator panel and linked asset.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: dream URL contract.
- `services/orion-hub/tests/test_dream_routes.py`: gates, reads, errors, limits and rendering.
- `services/orion-hub/evals/dream_server.py`, `dream_browser.cjs`: isolated live smoke and fixture browser eval.
- `services/orion-hub/README.md`: API, scope, limitations and validation instructions.
- `.github/workflows/hub-dream.yml`: deterministic Dream gates.

## Schema / bus / API changes

- Added: GET `/api/dream/pressure`, `/cycles`, `/cycles/{cycle_id}`, `/scorecard`.
- Removed: none.
- Renamed: none.
- Behavior changed: human inspection only. Responses are `no-store`; requests stop on tab exit.
- Compatibility notes: requires existing dream v2 migration. Missing stores return 503.
  Cycle history is capped at 50/page with a timestamp-and-ID cursor; scorecard
  refuses more than 10,000 matching rows instead of silently truncating.

## Env/config changes

- Added keys: `HUB_DREAM_SERVICE_URL=http://127.0.0.1:8620` (Hub uses host networking).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes; settings and compose also updated.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  yes, then scoped `orion-hub --all-keys` to include the new URL outside the
  script's default prefixes. Confirmed value and gitignore; no secret committed.
- skipped keys requiring operator action: none for this patch. Protected
  `PUBLISH_CORTEX_EXEC_GRAMMAR` and host-specific bus settings were not changed.

## Tests run

```text
python -m pytest services/orion-hub/tests/test_dream_routes.py services/orion-hub/tests/test_reverie_routes.py tests/test_dream_hypotheses.py -q
40 passed (existing dependency deprecation warnings only).
Independent reviewer: 11 Dream route tests passed.
node --check services/orion-hub/static/js/dream-tab.js
node --check services/orion-hub/static/js/app.js
python scripts/check_env_template_parity.py orion-hub — PASS
python scripts/check_service_hostname_refs.py --service orion-hub — PASS
python scripts/check_async_routes_not_blocking.py — PASS
```

## Evals run

```text
python services/orion-hub/evals/dream_server.py --env-file <local Hub .env>
node services/orion-hub/evals/dream_browser.cjs http://127.0.0.1:18091
PASS: actual template/section launcher/deep link, pagination, details, refresh,
HTML escaping, hidden-tab silence, no duplicate listeners, empty/unavailable states,
and GET-only requests. Fixture evidence, distinct from the live smoke below.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build — PASS
Isolated loopback server, no Hub workers, live stores on 2026-09-26:
GET /api/dream/pressure — 200; enabled=true, ready=false, too_soon=true.
GET /api/dream/cycles — 200; one completed sleep.
GET /api/dream/cycles/<recorded ID> — 200; 12 replay items, 4 hypotheses.
GET /api/dream/scorecard — 200; dream offered=3, control offered=1,
adopted=0, tested=0; verdict too early (needs 20 / 5).
Production Hub deployment: UNVERIFIED; no container restart performed.
```

## Review findings fixed

- Finding: paging could retain old details when the next history page failed.
  - Fix: clear selection at page load and show explicit unavailable detail state on failure.
  - Evidence: browser paging/unavailable checks pass.
- Finding: zero hypotheses need not mean refusal or failure; there may be no pair.
  - Fix: render the escaped cycle note (including pair counts); use neutral empty wording.
  - Evidence: reviewed against cycle producer contract; browser escaping check passes.
- Independent requesting-code-review subagent found no critical or important issues.

## Restart required

After merging and refreshing this worktree to the merged revision, with the
root and Hub `.env` symlinks pointing to the operator's existing local files:

```bash
cd /mnt/scripts/Orion-Sapienform-hub-dream
scripts/safe_docker_build.sh orion-hub up -d --build
```

Merge first: Hub's template/static bind mounts point at the primary checkout,
which must also contain the merged patch before restart. The command is provided
for the operator; production restart has not been authorized or performed.

## Risks / concerns

- Severity: inherited limitation.
- Concern: the upstream pressure API treats individual source-read failures as
  empty sources; its successful response does not certify source health.
- Mitigation: documented explicitly; Hub does not add its own false zero fallback.
- Scorecard verdict is descriptive adoption comparison, not significance or proof
  of cognitive benefit; insufficient evidence and absent test outcomes stay explicit.
- Visual reverie is separate. Existing staged REM compaction is called only when
  its own flag is enabled. Active crystallizations contribute to replay; older
  narrative dream synthesis and memory synthesis/consolidation producers are not
  invoked by this loop. No canonical-memory or belief writes are added.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2363
