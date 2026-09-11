## Summary

- Add a Gephi Lite workbench to Hub with the pinned official distribution served locally.
- Export bounded, read-only worldview, substrate, and canonical crystallization graphs, including memory lineage.
- Preserve arbitrary properties, directed relationships, and parallel edges without per-node UI adapters.
- Protect new routes with operator authentication and same-origin network restrictions.
- Add contract tests, isolated live-data browser evals, documentation, and focused CI.

## Outcome moved

Operators can open actual Orion graph slices in Gephi, inspect properties, select nodes, and change layouts. Real-browser verification imported 222 nodes / 40 edges from worldview, 3 / 2 from the default substrate neighborhood, and 300 / 269 from crystallizations (explicitly truncated). All three passed property inspection and layout-coordinate changes with no browser errors. A synthetic directed multigraph retained both parallel edges and the reverse edge.

Production deployment is **UNVERIFIED**: the running Hub has not been restarted. Verification used real backing stores through an isolated read-only server and the built image.

## Current architecture

Hub already serves FastAPI templates and custom graph atlases. Worldview/substrate reside in FalkorDB; canonical crystallizations and their provenance reside in PostgreSQL. Existing native analytics remain available. Crystallization projection references are written to canonical JSONB, not the separate empty projection-reference table.

## Architecture touched

Only the Hub service and its CI/documentation. A multi-stage Docker build copies Gephi assets into Hub; an authenticated router exposes the launcher, exports, search, and assets. No new service, port, cognition loop, metric, reducer, or write-back path.

## Files changed

- `services/orion-hub/scripts/graph_workbench.py`: bounded readers and generic GEXF serialization.
- `services/orion-hub/scripts/graph_workbench_routes.py`: authenticated read-only HTTP contract and private asset serving.
- `services/orion-hub/scripts/main.py`: router registration.
- `services/orion-hub/templates/graph_workbench.html`, `static/js/graph-workbench.js`, `static/css/graph-workbench.css`: source/search/neighborhood launcher.
- `services/orion-hub/templates/index.html`: workbench navigation link.
- `services/orion-hub/Dockerfile`, `Dockerfile.dockerignore`: pinned assets; exclude credentials and unnecessary build context.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: operator credentials contract.
- `services/orion-hub/tests/test_graph_workbench.py`: 19 deterministic contract checks.
- `services/orion-hub/evals/graph_workbench_server.py`, `graph_workbench_browser.cjs`: isolated live-source and directed-multigraph browser verification.
- `services/orion-hub/README.md`: operation, security boundaries, limitations, verification, rollout, rollback.
- `.github/workflows/hub-graph-workbench.yml`: path-scoped contract CI.
- This report: evidence and rollout instructions.

## Schema / bus / API changes

- Added: authenticated `GET /graph-workbench`, `/gephi-lite/{asset}`, `/api/graph-workbench/sources`, `/api/graph-workbench/search/{source}`, `/api/graph-workbench/export/{source}.gexf`.
- Removed: none.
- Renamed: none.
- Behavior changed: optional operator-only graph inspection. Source allowlist; depth 1–3; default 300 and maximum 1000 nodes; maximum 4000 edges; explicit truncation. Intimate and unknown-sensitivity crystals excluded from search and expansion.
- Compatibility notes: existing atlases/APIs unchanged. No storage migration, bus change, schema-registry change, or graph mutation. Falkor reads explicitly use `GRAPH.RO_QUERY`; PostgreSQL uses read-only transactions, repeatable-read exports, and bounded statement timeouts.

## Env/config changes

- Added keys: `HUB_GRAPH_WORKBENCH_USERNAME` (default `juniper`), `HUB_GRAPH_WORKBENCH_PASSWORD` (blank disables routes).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes; settings and compose match.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-hub --all-keys`: yes, primary checkout and worktree. Generated password stored only in ignored local configuration; no secret committed.
- Skipped keys requiring operator action: none for this patch. Existing unrelated local settings retained.

## Tests run

```text
python -m pytest services/orion-hub/tests/test_graph_workbench.py -q
  PASS: 19 tests, including auth, disabled mode, bounds, XML properties,
  direction/parallel edges, lineage, sensitivity filtering, read-only queries,
  sanitized errors, asset containment, private fonts and manifest auth.
node --check services/orion-hub/static/js/graph-workbench.js
node --check services/orion-hub/evals/graph_workbench_browser.cjs
  PASS
python scripts/check_env_template_parity.py orion-hub
  PASS
python scripts/check_service_env_compose_parity.py orion-hub
  N/A: service keys are supplied through env_file.
git diff --check
  PASS
python scripts/check_definition_drift.py --gate
  PASS: 648 definitions, no changes (using the repository virtualenv).
python scripts/check_metric_lineage.py --gate
  PASS: 648 URNs (using the repository virtualenv).
```

## Evals run

```text
PUPPETEER_MODULE=/mnt/scripts/Orion-Sapienform/services/orion-hub/node_modules/puppeteer \
node services/orion-hub/evals/graph_workbench_browser.cjs \
  http://127.0.0.1:18089 /tmp/orion-gephi-smoke.rZypsc/credentials.env \
  /tmp/orion-gephi-smoke.rZypsc
  PASS: all three live sources, actual imported counts, generic property columns,
  selected-node inspector, changed layout coordinates, no browser errors.
  PASS: native importer preserves two a->b edges and one b->a edge by ID.
```

Evidence is in `/tmp/orion-gephi-smoke.rZypsc/report.json` and screenshots alongside it. These contain private graph content and are deliberately not committed. The eval server binds localhost, skips Hub workers/database bootstrap, and uses read-only backing-store access. Default-slice browser coverage does not assert every upstream Gephi filter or metric.

Additional authenticated live HTTP smoke: search returned 20 choices per source; exporting the first search result at depth 2 / limit 30 succeeded for all three (worldview 1/0, substrate 1/0, crystallizations 10/10 nodes/edges). Isolated nodes are retained as legitimate neighborhoods.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build hub-app
  PASS after merging latest main: image sha256:588c7ff4385ff507da518c0aaa31d338abc8c92efe7e2150deb30db5b70711ab
Built-image read-only smoke against live FalkorDB
  PASS: 222 worldview nodes / 40 edges; Gephi index present; /app/.env absent.
```

Upstream distribution pinned to `ouestware/gephi-lite@sha256:cdbec4b88fbb7e261d45436e8eb0c63bc0384f1020c001fd290dc6ff1eb08d6f`. No live service restarted or production data written.

## Review findings fixed

- Finding: canvas presence and launcher counts did not prove native graph interaction.
  - Fix: assert imported counts, generic property columns, selected-node inspection, changed coordinates, and exact directed parallel-edge rows.
  - Evidence: final real-browser eval passed all sources plus the synthetic fixture.
- Finding: CI path filters omitted packaging and eval surfaces.
  - Fix: include Docker/config/evals/CSS/router/settings/navigation changes.
  - Evidence: workflow path review and final independent review.
- Independent review: requesting-code-review workflow run in a subagent; final review found no material regressions and reran all 19 tests successfully. Its remaining browser-fixture condition is now verified.

## Restart required

Requires Juniper's explicit production deployment approval. Run from the reviewed worktree (or a fresh deployment worktree after merge):

```bash
cd /mnt/scripts/Orion-Sapienform-gephi-lite-workbench
scripts/safe_docker_build.sh orion-hub build hub-app
ORION_HOST_REPO_ROOT="$PWD" scripts/safe_docker_build.sh orion-hub up -d --no-deps hub-app
```

Then open Hub's Graph Workbench link, authenticate as `juniper` using the password in the service `.env`, and smoke the three sources. Do not remove a worktree while deployment bind mounts reference it. To disable, clear the workbench password and restart Hub; graph data remains untouched.

## Risks / concerns

- Severity: operational. Production rollout remains pending approval; isolated browser and built-image checks are not a claim of deployed Hub behavior.
- Severity: privacy. Public/private memory is operator-visible; use HTTPS or trusted local/Tailscale transport. Downloads/workspaces are private artifacts. New endpoints are authenticated, no-store, and same-origin constrained; existing unrelated Hub endpoints are outside this patch.
- Severity: compatibility. Upstream requires `unsafe-eval`, confined to Gephi pages. External font imports are removed and the manifest opts into credentials. Repeat the browser eval before updating the pinned distribution.
- Severity: interpretation. Gephi analytics describe the bounded snapshot, not the complete live graph. The launcher preflights an export; Gephi fetches again, so a changing graph can produce different counts between reads. Projection references are provenance, not health assertions.
- No new metrics or cognition signals: the metric quality gate has no new candidate to assess. No Graphify artifacts modified or removed.

## PR link

[PR #2198](https://github.com/junebug-junie/Orion-Sapienform/pull/2198). Latest main merged without conflicts; live CI results are available in the PR's Checks tab. The final handoff reports the verified head's check status.
