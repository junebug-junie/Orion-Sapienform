## Summary

- Remove Graph Workbench's separate HTTP Basic login.
- Make the launcher, Gephi assets, search, and read-only exports available to anyone who can already reach Hub.
- Retire the workbench username/password settings from every config surface and the ignored live Hub `.env`.
- Simplify the isolated browser eval and document Hub's trusted-network boundary explicitly.
- Keep the existing read-only queries, sensitivity filtering, export bounds, CSP, and no-store responses unchanged.

## Outcome moved

Opening **Graph Workbench ↗** no longer causes a second browser login prompt. The workbench now behaves like the rest of the single-user Hub while retaining its read-only data boundaries.

## Current architecture

Graph Workbench is a Hub router over local Gephi Lite assets and bounded worldview, substrate, and crystallization exports. The initial patch added a router-wide HTTP Basic dependency even though Juniper is the only Hub user and Hub is already restricted at the trusted network boundary.

## Architecture touched

Only `orion-hub`: router dependencies, settings/env/compose surfaces, tests, eval CLI, Docker commentary, and operator documentation. Storage readers, GEXF generation, source allowlisting, privacy filtering, CSP, and graph limits are unchanged.

## Files changed

- `services/orion-hub/scripts/graph_workbench_routes.py`: remove HTTP Basic imports, credential comparison, and router dependency; remove the now-unnecessary credentialed manifest rewrite.
- `services/orion-hub/app/settings.py`: remove workbench username/password settings.
- `services/orion-hub/.env_example`: remove retired workbench keys.
- `services/orion-hub/docker-compose.yml`: stop forwarding retired keys.
- `services/orion-hub/tests/test_graph_workbench.py`: prove all five workbench surfaces respond without a separate authentication challenge.
- `services/orion-hub/evals/graph_workbench_server.py`: remove credentials-file setup and fixture dependency.
- `services/orion-hub/evals/graph_workbench_browser.cjs`: remove browser authentication and make live double-fetch count assertions reflect independently changing snapshots.
- `services/orion-hub/README.md`: explain the inherited Hub network boundary and simplified eval commands.
- `services/orion-hub/Dockerfile`: correct the asset-serving boundary comment.
- This report: verification and rollout record.

## Schema / bus / API changes

- Added: none.
- Removed: HTTP Basic authentication requirement and `WWW-Authenticate` challenge from all Graph Workbench routes.
- Renamed: none.
- Behavior changed: anyone able to reach Hub can access `/graph-workbench`, `/gephi-lite/*`, and `/api/graph-workbench/*` without a second login.
- Compatibility notes: route URLs and response schemas are unchanged. No schema, migration, bus, producer, reducer, graph, or memory changes. Existing Authorization headers are harmlessly ignored.

## Env/config changes

- Added keys: none.
- Removed keys: `HUB_GRAPH_WORKBENCH_USERNAME`, `HUB_GRAPH_WORKBENCH_PASSWORD`.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: retired keys were removed directly from the ignored primary Hub `.env`; the sync helper only adds keys and cannot retire them.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests/test_graph_workbench.py -q
18 passed
node --check services/orion-hub/static/js/graph-workbench.js
PASS
node --check services/orion-hub/evals/graph_workbench_browser.cjs
PASS
python3 scripts/check_env_template_parity.py orion-hub
PASS; pre-merge checker warns about the two intentionally retired keys because it compares the primary checkout's still-current template.
python3 scripts/check_service_env_compose_parity.py orion-hub
N/A/pass: all template keys reach compose through env_file.
git diff --check
PASS
```

## Evals run

```text
python services/orion-hub/evals/graph_workbench_server.py \
  --env-file /mnt/scripts/Orion-Sapienform/services/orion-hub/.env \
  --assets /tmp/orion-gephi-smoke.rZypsc/gephi-lite --port 18090
PUPPETEER_MODULE=/mnt/scripts/Orion-Sapienform/services/orion-hub/node_modules/puppeteer \
node services/orion-hub/evals/graph_workbench_browser.cjs \
  http://127.0.0.1:18090 /tmp/orion-gephi-no-login.v5udAq
PASS
```

Anonymous-to-Hub browser flow loaded all three real sources with no browser errors: worldview 254 nodes / 41 edges, substrate 300 / 3,833, and crystallizations 300 / 268. Generic properties, node selection, and changed layout coordinates passed. The deterministic fixture retained two `a -> b` edges and one `b -> a` edge by ID. Private screenshots/report remain outside git. The isolated eval server was shut down.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build hub-app
PASS: image sha256:710f08de191bd668e78e699b99d894234481029730d59de01607ac89c10ec7b6
```

The running Hub was not changed during verification.

## Review findings fixed

- Finding: Dockerfile still described assets as protected by an operator login.
  - Fix: describe the actual Hub network boundary.
  - Evidence: focused search has no stale workbench credential/auth references outside intentional no-login tests/docs.
- Finding: the browser eval briefly captured progressive Gephi edge counts and assumed two independently fetched live snapshots were identical.
  - Fix: require a non-empty native import, then capture final native counts after interaction; launcher and loaded counts remain separately observable.
  - Evidence: final live browser eval captured full 3,833-edge substrate import and passed all interactions.
- Independent reviewer found no material issues and reran 18 passing tests.

## Restart required

After merge approval, rebuild and restart Hub from a clean deployment worktree:

```bash
scripts/safe_docker_build.sh orion-hub build hub-app
ORION_HOST_REPO_ROOT="$PWD" scripts/safe_docker_build.sh orion-hub up -d --no-deps hub-app
```

Then open **Graph Workbench ↗** and verify there is no credentials prompt.

## Risks / concerns

- Severity: privacy.
- Concern: Graph Workbench can display public/private canonical memories; anyone who can reach Hub can now access them.
- Mitigation: this is the explicit single-user behavior requested. Keep Hub on the trusted local/Tailscale network or protect the entire Hub with an authenticating HTTPS proxy. Intimate and unknown-sensitivity crystallizations remain excluded.
- Severity: rollback.
- Concern: there is no longer a workbench-specific password-disable switch.
- Mitigation: restrict Hub at the network boundary or revert this patch; storage is never mutated.

## PR link

Pending creation.
