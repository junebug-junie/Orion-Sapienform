## Summary

- Replace the external two-step Graph Workbench launcher with a lazy-loaded **Graphs** tab inside Hub.
- Immediately load a manageable 100-node Memory Crystallizations overview; no setup form, popup, or second open button.
- Add descriptive Graph and populated Focus dropdowns; typing filters named focus choices rather than requiring node IDs.
- Collapse hops, node limit, and lineage under **Advanced slice size** while preserving direct snapshot download.
- Remove the separate HTTP Basic login and retire its config keys; Hub's trusted network remains the access boundary.
- Preserve read-only stores, private-memory filtering, generic attributes, graph bounds, CSP, and native Gephi tools.

## Outcome moved

One click on **Graphs** now stays in Hub and displays an already-loaded Gephi graph. The normal flow is: open tab, look at the memory graph, optionally choose a different graph or a named Focus item. All technical slice settings are optional and hidden by default.

The final real-browser run confirmed that the page defaults to Memory Crystallizations, Advanced is closed, no second open control exists, Focus contains named choices, and Gephi runs inside the page. The prior browser-login challenge is gone.

## Current architecture

The initial integration put an external-link launcher in Hub. That launcher required graph selection, search, hops, node limit, and lineage configuration, followed by a second button that opened another tab. Gephi itself then opened outside Hub. A separate HTTP Basic dependency added another interruption despite Hub being single-user and network-restricted.

## Architecture touched

Only `orion-hub`. Hub's existing hash-tab controller now owns a lazy embedded workbench iframe. The workbench page owns a thin descriptive control bar and embeds same-origin Gephi. Backend search/export readers and the pinned Gephi distribution are unchanged except for removal of the Basic Auth dependency.

## Files changed

- `services/orion-hub/templates/index.html`: turn the external link into the `#graph-workbench` Hub tab and add its lazy iframe panel.
- `services/orion-hub/static/js/app.js`: route, show, style, and lazy-load the Graphs tab within Hub.
- `services/orion-hub/templates/graph_workbench.html`: replace the setup form with descriptive dropdowns, collapsed advanced settings, and embedded Gephi.
- `services/orion-hub/static/js/graph-workbench.js`: auto-load the default graph, populate/filter Focus choices, and reload in place on selection.
- `services/orion-hub/static/css/graph-workbench.css`: compact full-width embedded layout.
- `services/orion-hub/scripts/graph_workbench_routes.py`: remove HTTP Basic and its manifest workaround.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: retire workbench username/password config.
- `services/orion-hub/tests/test_graph_workbench.py`: assert no-login access, embedded defaults, no popup/open step, and Hub tab wiring.
- `services/orion-hub/evals/graph_workbench_server.py`, `graph_workbench_browser.cjs`: exercise the embedded UI without credentials and avoid false live-snapshot/progressive-count assumptions.
- `.github/workflows/hub-graph-workbench.yml`: include Hub tab controller syntax in focused CI.
- `services/orion-hub/README.md`: document the one-click flow, controls, access boundary, and eval.
- `services/orion-hub/Dockerfile`: accurately describe the Hub network boundary.

## Schema / bus / API changes

- Added: Hub hash route `#graph-workbench` and in-page Graphs panel.
- Removed: workbench-wide HTTP Basic requirement and `WWW-Authenticate` challenge.
- Renamed: Hub navigation label `Graph Workbench ↗` to `Graphs`.
- Behavior changed: `/graph-workbench` immediately loads an embedded 100-node crystallization graph; Graph/Focus changes reload that frame in place. Anyone who can reach Hub can access the workbench APIs and assets.
- Compatibility notes: HTTP API URLs and payloads remain compatible. Existing Authorization headers are ignored. No schema, migration, bus, metric, cognition, graph-write, or memory-write change.

## Env/config changes

- Added keys: none.
- Removed keys: `HUB_GRAPH_WORKBENCH_USERNAME`, `HUB_GRAPH_WORKBENCH_PASSWORD`.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: retired keys were removed directly from the ignored primary Hub `.env`; the helper only adds keys and cannot retire them.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests/test_graph_workbench.py -q
18 passed
node --check services/orion-hub/static/js/graph-workbench.js
PASS
node --check services/orion-hub/static/js/app.js
PASS
node --check services/orion-hub/evals/graph_workbench_browser.cjs
PASS
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_definition_drift.py --gate
PASS: no metric changes
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_metric_lineage.py --gate
PASS
/mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/check_async_routes_not_blocking.py
PASS
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
  http://127.0.0.1:18090 /tmp/orion-gephi-integrated.57rCkq
PASS
```

The final anonymous browser run imported worldview 100 nodes / 13 edges, substrate 100 / 1,123, and crystallizations 100 / 50, with zero browser errors. For every source it checked populated Focus choices, native property columns, node selection, and changed layout coordinates. A deterministic fixture retained two `a -> b` edges plus one `b -> a` edge by ID. Screenshot inspection confirmed the compact controls and embedded graph fit one viewport. Private evidence remains outside git; the eval server was shut down.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build hub-app
PASS: image sha256:11e0a8270c952b18680efa844cba99228a81461d5f974225e24d8e1701ebe01a
```

The running Hub remains unchanged and still serves the prior 401-gated external-link build until this PR is approved, merged, rebuilt, and restarted.

## Review findings fixed

- Finding: the original feature optimized backend safety but created a multi-page setup workflow with unexplained controls.
  - Fix: one lazy Hub tab, immediate memory default, descriptive/populated dropdowns, advanced controls collapsed, no popup.
  - Evidence: final browser eval and screenshots exercise the new primary flow.
- Finding: live graph counts can change or render progressively, making exact preflight-to-import comparisons flaky.
  - Fix: the workbench fetches one bounded snapshot, gives that exact blob to Gephi, and reports Ready only after the native import matches its counts.
  - Evidence: full 1,123-edge substrate slice captured after interaction.
- Finding: an iframe reload with the same graph counts could be incorrectly confirmed against the previous document.
  - Fix: bind each pending import to its exact blob navigation and stamp the newly loaded document with the request version before accepting counts.
  - Evidence: same-count Refresh regression requires a new blob URL and matching request/document confirmation versions.
- Finding: filtering Focus choices could reset the select to Overview while leaving the old focused graph displayed.
  - Fix: preserve the active Focus choice while filtering, including when it does not match the filter text.
  - Evidence: browser regression verifies both the selected Focus and displayed graph seed remain unchanged after a zero-result filter.
- Finding: the browser eval initially tested the direct workbench route rather than the real Hub tab.
  - Fix: render the actual Hub template/controller in the eval server and exercise deep linking, tab hide/restore, lazy loading, and popup count.
  - Evidence: the `hub-tab` eval result confirms all four behaviors.
- Finding: Dockerfile still described the retired operator login.
  - Fix: document the actual Hub network boundary.
  - Evidence: focused auth/config search.
- Final independent re-review: cleared with no remaining material findings.

## Restart required

After required PR approval and merge, build and restart Hub from a clean deployment worktree:

```bash
scripts/safe_docker_build.sh orion-hub build hub-app
ORION_HOST_REPO_ROOT="$PWD" scripts/safe_docker_build.sh orion-hub up -d --no-deps hub-app
```

Then click **Graphs** and verify Hub remains visible, Memory Crystallizations appears without configuration, and no login or popup occurs.

## Risks / concerns

- Severity: privacy.
- Concern: public/private crystallizations are visible to anyone who can reach Hub.
- Mitigation: explicit single-user behavior; keep Hub on trusted local/Tailscale networking or put the whole Hub behind an authenticating HTTPS proxy. Intimate and unknown-sensitivity records remain excluded.
- Severity: embedding.
- Concern: Gephi runs inside nested same-origin frames and can be visually dense on small screens.
- Mitigation: 100-node default, responsive full-width panel, direct `/graph-workbench` diagnostic route, and larger slices remain opt-in.
- Severity: interpretation.
- Concern: Gephi metrics describe the loaded slice, not the complete live graph.
- Mitigation: preserve explicit bounded exports and documentation; no analytics are written back.

## PR link

[PR #2202](https://github.com/junebug-junie/Orion-Sapienform/pull/2202)
