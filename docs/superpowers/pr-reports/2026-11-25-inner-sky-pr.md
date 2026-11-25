## Summary

- The Hub's Cognitive EKG drawer is now the **Inner Sky**: a live 3D view where every light is one stored row.
- Each comet is one experience Orion had (`temporal_self_event`). Its words are the same `now_label` text Orion is shown.
- A gold arc means one part of Orion used another part's experience (`experience_use`).
- The outer stars are the services, and the blue pulses run along the cause-and-effect links between them (bus synaptic graph). Pulse speed is the real message rate, measured between two polls.
- New read-only routes: `GET /api/inner-sky/events` and `GET /api/inner-sky/state`. Full page at `/static/inner-sky.html`; the drawer uses `?embed=1`. The page has Live and Replay (last 12 h) modes.
- The old `/spark/ui` page is retired. It was a synthwave grid bent by three numbers. Its `/ws/tissue` socket and `/api/test-pulse` route go with it, plus the broadcast arguments of `feed_tissue()` that only fed the socket.

## Outcome moved

The old drawer squeezed Orion into three dials, then picked a wave height and a colour for them. Now each thing on screen is a thing that happened, and hovering it shows the real row. Things that are absent show as absent:

- Dead services are dark. vision-edge has been silent 20 days.
- Parts of Orion with no experience in the window are dim.
- When no attention arc is open, the page says "no open attention arc" and borrows no other arc.

It also shows a real gap plainly. In the last 12 hours, one part of Orion used another part's experience 5 times out of about 890 experiences.

## Current architecture

- The drawer was an iframe of `orion-vector-host`'s `/spark/ui` (`app/static/index.html` + `tissue_viz.js`).
- That page polled `/api/self-brain/frames/tail` and mapped honesty_metrics, coalition_stability and field_anomaly onto a synthwave grid's chaos, amplitude, speed and hue.
- The experience feed and the service graph had no visual surface.

## Architecture touched

- **orion-hub:**
  - New router `scripts/inner_sky_routes.py`, registered in `scripts/main.py`.
  - New page `static/inner-sky.html` and script `static/js/inner-sky.js` (three.js 0.160 via jsDelivr `+esm`).
  - New pure helpers `static/js/inner-sky-core.js`.
  - Drawer iframe and visible text changed in `templates/index.html`.
- **orion-vector-host:** `/ui`, `/ws/tissue`, `/api/test-pulse` and the static mount removed. `app/conn_manager.py` and `app/static/` deleted. `feed_tissue()` loses `doc_id`, `correlation_id` and `broadcast`. `GET /api/tissue/state` stays.
- **Reads, all read-only:** Postgres `temporal_self_event` and `experience_use`; FalkorDB `orion_bus_synapse`; durable-runs `GET /temporal-self/frame`.
- **Deliberately not read:** `GET /temporal-self/now`. That call stores an `experience_shown` row, so a debug page would register as Orion seeing something. A test pins this.

## Files changed

- `services/orion-hub/scripts/inner_sky_routes.py`: the two routes; `shape_event`, `shape_cite`, `read_body` and `focus_from_frame` are pure.
- `services/orion-hub/scripts/main.py`: registers the router.
- `services/orion-hub/static/inner-sky.html`, `static/js/inner-sky.js`: the page.
- `services/orion-hub/static/js/inner-sky-core.js`, `inner-sky-core.test.js`: sorted insert, count-delta rates, pulse rate, bins, liveness.
- `services/orion-hub/templates/index.html`: the drawer embeds Inner Sky; the visible "Cognitive EKG" text becomes "Inner Sky" (element ids unchanged).
- `services/orion-hub/static/js/biometrics-view.js`, `retro-tab.js`: stale `/spark/ui` comments updated.
- `services/orion-hub/tests/test_inner_sky_routes.py`, `test_inner_sky_page.py`: new tests.
- `services/orion-hub/evals/run_inner_sky_feed_eval.py`: a live empty-shell eval.
- `services/orion-hub/README.md`: new Inner Sky section.
- `services/orion-vector-host/app/main.py`, `app/tissue_feed.py`, `tests/*`, `README.md`: the retirement.
- `services/orion-substrate-runtime/README.md`: the stale `/spark/ui` sentence updated.

## Schema / bus / API changes

- **Added:** `GET /api/inner-sky/events?since=&cite_since=` and `GET /api/inner-sky/state` (Hub).
- **Removed:** vector-host `GET /ui` (served as `/spark/ui`), `WS /ws/tissue` and `POST /api/test-pulse`. Nothing outside vector-host used them.
- **Renamed:** none.
- **Behavior changed:** the drawer content.
- **Compatibility:** no bus channel or schema changes.

## Env/config changes

- **Added, removed or renamed keys:** none. The route uses existing settings: `POSTGRES_URI`, `FALKORDB_URI`, `FALKORDB_BUS_GRAPH`, `HUB_CURIOSITY_DURABLE_RUNS_URL`.
- **`.env_example` updated:** no.
- **Local `.env` sync:** not needed.

## Tests run

```text
pytest services/orion-hub/tests/test_inner_sky_routes.py test_inner_sky_page.py      16 passed
pytest (related Hub UI tests: biometrics view, ask routes, UI polish, layout pass)    166 passed, 1 skipped
pytest services/orion-hub/tests (full suite)   5273 passed, 217 skipped; 1 failed + 4 errors also fail on main (local env: fcc labels default, consult belief SQL DB)
node --test services/orion-hub/static/js/inner-sky-core.test.js                      6 passed
vector-host tests, inside the built image (torch needed)                             19 passed
```

## Evals run

```text
run_inner_sky_feed_eval.py logic: unit-tested (an empty shell FAILs, real rows PASS).
Live route smoke against production data (route functions in the built Hub image and on the host):
  events 887 (12 h), blank labels 0, unregistered kinds 0, bindings 5
  services 182, links live 24 h 246, day frame read OK (no open attention arc at check time)
  cursor overlap re-read: 2 events + 1 binding repeated (deduped by id), cursor held
Against the deployed Hub: UNVERIFIED until deploy. Run the eval after the rebuild.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-vector-host build   Built
scripts/safe_docker_build.sh orion-hub build           Built
In-image smoke of inner_sky_routes (host network, Hub .env): as above
Browser render: UNVERIFIED. The session's safety check blocked headless Chromium. The page has an on-screen
error box, so a failure shows its message instead of going blank.
```

## Review findings fixed

- **Finding:** the events cursor could skip rows for good. `ingested_at` is the transaction-start time, so an earlier transaction can commit after a later one was polled. Page boundaries can also cut through groups of equal timestamps.
  - **Fix:** both cursors re-read the last 60 s (`CURSOR_OVERLAP_SEC`). The page dedupes by id, and the cursors only move forward.
  - **Evidence:** `test_cursors_reread_an_overlap_...`, `test_cites_carry_a_stable_id_...`, plus the live re-read above.
- **Finding:** bindings were not deduped, and two polls could overlap.
  - **Fix:** each binding carries its `use_id` and the page dedupes on it. Each poll schedules the next one only after it finishes.
  - **Evidence:** code (`seenCites`, `loop()`), and the route test for `id`.
- **Finding:** counts drifted in Replay mode and never shrank in Live mode, and the page's arrays grew forever.
  - **Fix:** `noteEvent` counts only rows inside the window. Late rows are counted in both modes. Each minute, a prune drops rows that left the window and recounts.
  - **Evidence:** code (`recount`, `prune`).
- **Finding (low):** two live comets could share pool slots.
  - **Fix:** free lists for comet and arc slots.
- **Finding (low):** organs missing from the registry list got no node.
  - **Fix:** the ring is rebuilt when a new organ appears.
- **Nit:** hardcoded `?v=1` on the JS assets.
  - **Not changed:** `HubStaticFiles` sends `Cache-Control: no-cache`, so the browser revalidates every load.
- **Extra fix found during the live smoke:** attention fell back to whatever arc was open, which could show a curiosity arc as attention. Attention now reads only the attention arc.

## Restart required

```bash
# after merge, from the clean primary main checkout:
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-vector-host up -d --build
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** the page loads three.js from cdn.jsdelivr.net. With no internet the page shows its error box. The old page had the same dependency.
- **Severity: low.**
  - **Concern:** service pulse rates need two `/state` polls (20 s) before they show. Until then links draw dim and the tooltip says "measuring".
  - **Mitigation:** this is by design: no rate is guessed.
- **Severity: low.**
  - **Concern:** labels for `attention_loop_raised` and `visual_run` are only the kind's word (`now_label` has no template for them). The eval reports these and does not fail on them.
- **Issues opened:** #3891 (verify-live: drawer draws real rows after deploy), #3892 (follow-up: no `now_label` template for `attention_loop_raised` and `visual_run`). **Closed:** none.

## PR link

(filled in after push)
