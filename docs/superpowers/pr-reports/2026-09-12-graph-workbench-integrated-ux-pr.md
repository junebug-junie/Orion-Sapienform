## Summary

- Replace the external two-step Graph Workbench launcher with a lazy-loaded **Graphs** tab inside Hub.
- Immediately load a manageable 100-node Memory Crystallizations overview; no setup form, popup, or second open button.
- Add descriptive Graph and populated Focus dropdowns; typing filters named choices instead of requiring node IDs.
- Collapse hops, node limit, and lineage under **Advanced slice size** while preserving direct snapshot download.
- Preserve read-only stores, private-memory filtering, generic attributes, graph bounds, CSP, and native Gephi tools.

## Outcome moved

One click on **Graphs** now stays in Hub and displays an already-loaded Gephi graph. The primary flow is: open the tab, inspect the memory graph, and optionally choose a different graph or named Focus item. Technical slice settings are optional and hidden by default.

## Current architecture

The original integration put an external-link launcher in Hub. It required graph selection, search, hops, node limit, and lineage configuration, followed by a second button that opened Gephi in another tab.

## Architecture touched

Only `orion-hub`. Hub's hash-tab controller now owns a lazy workbench iframe. The workbench page owns a thin descriptive control bar and embeds same-origin Gephi. Backend search/export readers and the pinned Gephi distribution are unchanged.

## Files changed

- `services/orion-hub/templates/index.html`: turn the external link into the `#graph-workbench` Hub tab and add its lazy iframe panel.
- `services/orion-hub/static/js/app.js`: route, show, style, and lazy-load the Graphs tab within Hub.
- `services/orion-hub/templates/graph_workbench.html`: replace the setup form with descriptive dropdowns, collapsed advanced settings, and embedded Gephi.
- `services/orion-hub/static/js/graph-workbench.js`: auto-load the default graph, populate/filter Focus choices, and reload in place on selection.
- `services/orion-hub/static/css/graph-workbench.css`: compact full-width embedded layout.
- `services/orion-hub/tests/test_graph_workbench.py`: assert embedded defaults, no popup/open step, and Hub tab wiring.
- `services/orion-hub/evals/graph_workbench_server.py`, `graph_workbench_browser.cjs`: exercise the actual Hub tab and embedded Gephi UI.
- `.github/workflows/hub-graph-workbench.yml`: include the Hub tab controller in focused CI.
- `services/orion-hub/README.md`: document the one-click flow, controls, and eval.

## Schema / bus / API changes

- Added: Hub hash route `#graph-workbench` and in-page Graphs panel.
- Renamed: Hub navigation label `Graph Workbench ↗` to `Graphs`.
- Behavior changed: `/graph-workbench` immediately loads an embedded 100-node crystallization graph; Graph/Focus changes reload it in place.
- Compatibility notes: HTTP API URLs and payloads remain compatible. No schema, migration, bus, metric, cognition, graph-write, or memory-write change.

## Env/config changes

None.

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
PASS: 652 definitions, no metric changes
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

The browser run exercised the real Hub deep link and tab controller without opening another page. It imported worldview 100 nodes / 13 edges, substrate 100 / 1,123, and crystallizations 100 / 50 with zero browser errors. Each source exposed native properties, node selection, and working layout changes. The eval also covers multiedge direction, Focus-filter preservation, and same-count iframe refreshes.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build hub-app
PASS: image sha256:11e0a8270c952b18680efa844cba99228a81461d5f974225e24d8e1701ebe01a
```

The running Hub was not changed during verification.

## Review findings fixed

- Finding: the primary workflow required two page transitions and unexplained technical inputs.
  - Fix: one lazy Hub tab, immediate memory default, descriptive/populated dropdowns, advanced controls collapsed, no popup.
  - Evidence: real Hub browser eval and screenshots exercise the new primary flow.
- Finding: filtering Focus choices could reset the select while leaving the old focused graph displayed.
  - Fix: preserve the active Focus choice while filtering, even when it does not match the filter text.
  - Evidence: zero-result browser regression preserves both the selection and displayed graph seed.
- Finding: an iframe reload with the same graph counts could be confirmed against the previous document.
  - Fix: bind each pending import to its exact blob navigation and stamp the newly loaded document with the request version before accepting counts.
  - Evidence: same-count Refresh regression requires a new blob URL and matching request/document versions.
- Finding: the initial eval bypassed the real Hub tab.
  - Fix: render the actual Hub template/controller and exercise deep linking, tab hide/restore, lazy loading, and popup count.
  - Evidence: `hub-tab` eval result confirms all four behaviors.
- Final independent re-review: cleared with no remaining material findings.

## Restart required

After required PR approval and merge, build and restart Hub from a clean deployment worktree:

```bash
scripts/safe_docker_build.sh orion-hub build hub-app
ORION_HOST_REPO_ROOT="$PWD" scripts/safe_docker_build.sh orion-hub up -d --no-deps hub-app
```

Then click **Graphs** and verify Hub remains visible, Memory Crystallizations appears without configuration, and no popup occurs.

## Risks / concerns

- Embedding: Gephi runs inside nested same-origin frames and can be visually dense on small screens. The 100-node default, responsive panel, and direct diagnostic route limit the impact.
- Interpretation: Gephi metrics describe the loaded slice, not the complete live graph. Limits remain explicit and no analytics are written back.

## PR link

[PR #2204](https://github.com/junebug-junie/Orion-Sapienform/pull/2204)
