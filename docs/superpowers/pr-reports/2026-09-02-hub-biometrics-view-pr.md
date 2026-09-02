# Hub Biometrics view: Cognitive EKG toggle + deep-inspection modal + Cabinet fold-in

## Addendum (same day, post-deploy feedback)

Juniper deployed and reported: "hard to read the cards, can't tell what's
good, bad, changing, important. Also, few trends??" Follow-up commit fixes:

- **Readability**: every tile now colors by value (good/warning/critical,
  reusing this repo's own emerald/amber/red dark-surface convention from
  `cabinet-sensors.js`'s `badge()`, not a new palette) and shows a trend
  arrow sourced from the already-computed EWMA `trend` field. Snapshot
  tiles sort worst-first. A color legend is on every panel.
- **More trends**: Trended history expanded from 4 hand-picked channels to
  all 14 the backend can chart; the GPU sparkline was reading a **5-sample**
  buffer (looked flat) -- bumped to 40.
- **Review findings from this follow-up, fixed**: (1) an unreachable node's
  status tile rendered gray/"neutral", identical to "no data yet" and the
  *only* visible tile once every value tile got filtered out -- now renders
  critical/red via a new `toneForNodeStatus()`. (2) expanding to 14 channels
  meant 14 concurrent, unpooled Postgres connections per modal open (real
  risk given this repo's connection-exhaustion incident history, PR #2010)
  -- consolidated into one `/history_multi` endpoint, one connection, one
  request from the client.
- 16 new/updated tests (77 total in the biometrics test set, all passing).
- Same open PR (#2027), pushed as a second commit -- not a new PR.

## Summary

- Cognitive EKG card (Hub landing tab) gets a toggle that swaps the `/spark/ui`
  Substrate Brain State iframe for a compact Athena + Circe biometrics preview,
  in the same card slot -- clicking the preview opens a near-fullscreen modal.
- Modal has 4 sub-tabs: Athena, Circe (current snapshot, 24h trended history,
  EWMA/induction detail), GPU (per-card index/name/routing-lane/util/mem/
  power/processes/realtime trend), and Cabinet (the old standalone Nano-sensor
  tab, relocated here instead of living on as its own top-level nav tab).
- New Hub read API (`/api/biometrics/preview/*`) reuses existing Postgres
  tables and the existing per-node `orion-biometrics` HTTP API rather than
  building new persistence.
- `orion-biometrics`' GPU collector now also captures per-GPU compute
  processes (paired `gpu_host_stats.sh` + `utils.py` change).
- GPU "lane" is a small hand-maintained config join (Orion's model-routing
  lane, not a PCIe/NVLink concept -- neither exists as a registry anywhere
  in this repo), not a scraper.

## Outcome moved

Biometrics was previously three disconnected surfaces in Hub: a 3-tile
Strain/Homeostasis/Stability composite widget, a standalone Cabinet tab for
physical enclosure sensors, and zero per-GPU-card visibility anywhere in the
UI (only fleet-normalized `gpu_util`/`gpu_mem` pressure channels). This patch
gives Athena and Circe host telemetry one coherent, clickable, inspectable
surface that shares screen real estate with the existing Brain State viz
instead of competing for a new tab slot.

## Current architecture

- Cognitive EKG card was a bare `/spark/ui` iframe, stacked above (not
  toggled with) a separate `#biometricsPanel` composite widget.
- Cabinet was a standalone top-level Hub tab (`#cabinetTabButton` /
  `#cabinet`), driven by `cabinet-sensors.js` (~1000 lines) reading
  `/api/cabinet/sensors/*` + `/api/cabinet/ambient/*`.
- GPU telemetry: `orion/sensors/gpu_host_stats.sh` (nvidia-smi CSV scrape) ->
  `orion-biometrics`' `collect_gpu_stats()` -> bus/Postgres, but no per-card
  breakdown or process list reached any Hub API or UI panel -- only the
  fleet composite (`gpu_util`, `gpu_mem` pressure channels).
- EWMA/trend already existed generically via `InductionTracker`
  (`orion/signals/normalization.py`) and
  `orion.substrate.metacog_trend_signals.latest_biometrics_induction_by_node`,
  but nothing in Hub exposed it per-node to the UI.
- An existing debug API (`substrate_biometrics_routes.py`) covers lineage/
  receipt-chain debugging, not a clean current-value/trend/EWMA UI shape --
  not reused directly, but its Postgres-access convention was consulted.

## Architecture touched

- `services/orion-hub` frontend (`templates/index.html`, `static/js/app.js`,
  new `static/js/biometrics-view.js`) and backend (new
  `scripts/biometrics_preview_routes.py`, `scripts/biometrics_node_client.py`,
  `app/settings.py`, `scripts/api_routes.py`).
- `services/orion-biometrics` (`app/utils.py`) and `orion/sensors/gpu_host_stats.sh`
  -- paired GPU-process-list extension, additive to the existing collector.

## Files changed

- `services/orion-hub/templates/index.html`: EKG card toggle + preview
  container; new Biometrics modal markup (4 sub-tabs); `#cabinetTabButton`
  nav link removed; `<section id="cabinet">` relocated bodily into the modal
  (id unchanged).
- `services/orion-hub/static/js/app.js`: removed all old cabinet-tab-switch
  wiring (consts, hash routing, click listener, `styleTabButton`); added
  generic `openBiometricsModal`/`closeBiometricsModal` mechanics (copies
  `openDebugPanelModal`'s pattern exactly) + scroll-lock/Escape wiring; hooks
  `OrionBiometricsView.deactivate()` into the hub-tab visibility toggle
  (review fix, see below).
- `services/orion-hub/static/js/biometrics-view.js` (new): card toggle +
  modal subview switching, `activate()/deactivate()/wireOnce()` lifecycle.
- `services/orion-hub/scripts/biometrics_preview_routes.py` (new):
  `/api/biometrics/preview/{snapshot,history,induction,gpu}`.
- `services/orion-hub/scripts/biometrics_node_client.py` (new): per-node
  `orion-biometrics` HTTP client (athena local, circe cross-host).
- `services/orion-hub/app/settings.py`: `CIRCE_BIOMETRICS_BASE_URL`,
  `BIOMETRICS_NODE_CLIENT_TIMEOUT_SEC`, `GPU_LANE_MAP_{ATHENA,CIRCE}_JSON`.
- `services/orion-hub/scripts/api_routes.py`: registers the new router.
- `services/orion-biometrics/app/utils.py` + `orion/sensors/gpu_host_stats.sh`:
  paired GPU compute-process capture, joined by `gpu_uuid`.
- Tests: `services/orion-hub/tests/{test_biometrics_preview_api,
  test_biometrics_node_client,test_biometrics_view_ui}.py` (new),
  `test_cabinet_sensors_panel.py` (updated for the relocation),
  `services/orion-biometrics/tests/test_gpu_collector.py` (new).

## Schema / bus / API changes

- Added: 4 new Hub REST endpoints under `/api/biometrics/preview/`. No new
  bus channels, no new schema registry entries -- this reads existing
  Postgres tables (`orion_biometrics_summary`, `orion_biometrics_induction`)
  and existing per-node `orion-biometrics` HTTP endpoints (`/snapshot`,
  `/raw/recent`).
- Removed: none.
- Renamed: none.
- Behavior changed: `orion-biometrics`' `collect_gpu_stats()` return shape
  gains a `processes` key per GPU row (additive, non-breaking).
- Compatibility notes: `atlas` is explicitly rejected (404) by every new
  endpoint, not silently dropped or forwarded as a doomed HTTP call.

## Env/config changes

- Added keys (`services/orion-hub/.env_example`):
  `CIRCE_BIOMETRICS_BASE_URL=http://100.112.254.99:8100`,
  `BIOMETRICS_NODE_CLIENT_TIMEOUT_SEC=5.0`, `GPU_LANE_MAP_ATHENA_JSON={}`,
  `GPU_LANE_MAP_CIRCE_JSON={}`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: **manually** -- `scripts/sync_local_env_from_example.py`
  reads the *primary checkout's* `.env_example`, so these worktree-added keys
  were invisible to it (known gap, see
  `feedback_env_sync_reads_example_from_primary_checkout` memory). Appended
  the same 4 keys directly to `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env`
  instead.
- Skipped keys requiring operator action: `GPU_LANE_MAP_ATHENA_JSON` /
  `GPU_LANE_MAP_CIRCE_JSON` ship as `{}` -- populating real GPU-index-to-lane
  labels is an operator edit against the live `CUDA_VISIBLE_DEVICES*`
  assignments across `orion-llamacpp-host`/`orion-vision-host`/
  `orion-diffusion-host`'s configs, deliberately not hand-guessed here.

## Tests run

```text
services/orion-hub:
  pytest tests/test_biometrics_preview_api.py tests/test_biometrics_node_client.py \
         tests/test_biometrics_view_ui.py tests/test_cabinet_sensors_panel.py -q
  -> 61 passed

  pytest tests/ -q
  -> 33 failed, 1959 passed, 5 skipped
     (all 33 failures confirmed identical on the unmodified primary checkout;
      none touch a file this patch changed except test_substrate_review_runtime_hub_debug.py,
      which fails identically -- 2 failed, 22 passed, 1 skipped -- on both trees)

services/orion-biometrics:
  pytest tests/test_gpu_collector.py -q
  -> 3 passed

  pytest tests/ -q --ignore=tests/test_power_intent_handler_wiring.py
  -> 12 failed, 132 passed
     (identical 12 failures, same names, on the unmodified primary checkout
      -- 129 passed there, the 3-test delta is exactly the new GPU-collector
      tests. test_power_intent_handler_wiring.py's collection error --
      ModuleNotFoundError: orion -- also reproduces unmodified on the primary
      checkout; pre-existing, unrelated to this change.)
```

## Evals run

No eval harness exists for `orion-hub` or `orion-biometrics` in this repo.
Not adding one for this patch -- it's a UI/read-API feature, not a
model-quality-bearing change; flagging the gap rather than skipping silently.

## Docker/build/smoke checks

Not run. This patch does not change container boot behavior, dependencies,
ports, health checks, or compose wiring -- only application code, a new
sibling telemetry file naming convention, and Hub settings fields with safe
defaults (`{}` / a real cross-host URL that degrades to a controlled
"unreachable" response, not a crash, per `biometrics_node_client.py`).
Manual UI exercise (toggle, modal open/close/Escape/backdrop, all 4
sub-tabs) was not performed in this session -- flagging as an open item
for Juniper before merge, see Risks below.

## Review findings fixed

- Finding: Switching away from the Hub landing tab never called
  `OrionBiometricsView.deactivate()`, so the EKG card's Biometrics-preview
  poll timer (once started via the toggle) kept firing
  `GET /api/biometrics/preview/snapshot` every 10s indefinitely, even with
  the card off-screen.
  - Fix: `app.js`'s `hubTabPanel.classList.toggle("hidden", !isHub)` now
    calls `window.OrionBiometricsView.deactivate()` when leaving the hub
    tab, matching every sibling lazy panel's `isX ? activate() : deactivate()`
    pair. `biometrics-view.js`'s `activate()` also now resumes the poll on
    return if the preview was left toggled on, so the fix doesn't strand the
    view silently reverted to "brain".
  - Evidence: `test_app_js_deactivates_biometrics_view_when_leaving_the_hub_tab`,
    `test_biometrics_view_js_resumes_preview_poll_on_reactivate`
    (`test_biometrics_view_ui.py`).
- Finding: `loadNodeDetail()` awaited its snapshot, 4-channel history, and
  induction fetches sequentially instead of concurrently, tripling worst-case
  load time for the Athena/Circe modal subviews on a slow/loaded node.
  - Fix: all three fetches now start before the first `await`.
  - Evidence: `test_biometrics_view_js_loads_node_detail_fetches_concurrently`
    (`test_biometrics_view_ui.py`).

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-biometrics/.env \
  -f services/orion-biometrics/docker-compose.yml \
  up -d --build
```

(orion-hub for the new routes/settings/frontend; orion-biometrics on both
athena and circe for the GPU process-list collector change.)

## Risks / concerns

- Severity: medium
  Concern: No live manual UI pass was done in this session (no running Hub
  instance exercised in-browser) -- the toggle/modal/sub-tab wiring is
  covered by static-content-contract tests (this repo's real UI-test
  convention) but not by an actual click-through.
  Mitigation: exercise the toggle, modal open/close/Escape/backdrop, and all
  4 sub-tabs against a live Hub instance before/after deploy; the restart
  commands above bring up both affected services.
- Severity: low
  Concern: `GPU_LANE_MAP_{ATHENA,CIRCE}_JSON` ship empty -- every GPU card
  will show "unassigned" until an operator populates the mapping by hand
  against the real `CUDA_VISIBLE_DEVICES*` assignments scattered across
  `orion-llamacpp-host`/`orion-vision-host`/`orion-diffusion-host`.
  Mitigation: deliberate -- ship honestly partial rather than guess; the doc
  comment above each key in `.env_example` says so.
- Severity: low
  Concern: GPU compute-process capture (`--query-compute-apps`) was not
  verified live against real athena/circe nvidia-smi output in this session
  -- `collect_gpu_stats()` degrades to `processes: []` on any failure there,
  so a driver-version quirk would show as an empty process list, not an
  error, until checked by hand.
  Mitigation: after deploy, curl `/api/biometrics/preview/gpu?node=circe`
  and confirm real process rows appear for at least one busy GPU.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2027
