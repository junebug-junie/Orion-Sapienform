## Summary

- **One light per real message.** A blue pulse is one real hop between services (`CAUSALLY_FOLLOWED_BY` count delta). A star flash plus a spark is one message that service published (`PUBLISHES` count delta). The page polls every 5 s instead of 20 s.
- **Drawer gains a wide state.** ▶ fills the row and hides chat until ◀. It is remembered under the existing key ('2'), and it exists only from md up.
- **The Widen link inside the sky works.** Before, it was dead: the overlay turns clicks off on itself. Now it asks the Hub to widen.
- **The Hub tells the sky its drawer state,** and the sky picks compact or full from that instead of guessing from its own width.
- **The center is labelled "now"** and has a hover explaining what it is.

## Outcome moved

Juniper: "it's quite slow... would think bus or cortex or chat steps would be pulsing." The hops were real but invisible. Measured live, the service graph sees about 3.5 hops/s and about 40 published messages/s in total. Spread over 246 links, that is about 0.01 hops/s per link, and the old `0.15*log2(1+rate)` formula drew almost nothing from it. Now each hop and each message is drawn. Per 5 s poll the drawing is capped at 40 hops per link and 30 messages per service. The rest are counted, not drawn.

## Current architecture

The PR #3893 Inner Sky. The drawer had two states, open and collapsed. The sky chose its compact view from `?embed=1`, and its in-frame "Open full sky" link was unclickable.

## Architecture touched

Hub static only:

- `static/js/inner-sky.js`, `inner-sky-core.js` (+ `countDeltas`, − `pulsesPerSecond`)
- `static/inner-sky.html`
- `static/js/hub-ekg-drawer.js` (three states + postMessage both ways, origin- and source-checked)
- `static/css/style.css`
- `templates/index.html` (▶ button; iframe without `?embed=1`)

No server, schema, bus or env changes.

## Files changed

- `services/orion-hub/static/js/inner-sky.js`: per-message pulses, flashes and sparks; 5 s state poll; "now" label and hover; camera re-placed only when the layout flips.
- `services/orion-hub/static/js/inner-sky-core.js`, `.test.js`: `countDeltas` (a first reading or a reset counter draws nothing).
- `services/orion-hub/static/inner-sky.html`: layout from the drawer state; working Widen link; legend.
- `services/orion-hub/static/js/hub-ekg-drawer.js`, `.test.js`: open, collapsed and wide; tells the sky its state; accepts widen only from its own frame.
- `services/orion-hub/static/css/style.css`: the wide rule, md-up only; ▶ hidden on phones.
- `services/orion-hub/templates/index.html`: the ▶ button.
- `services/orion-hub/tests/test_inner_sky_page.py`: wiring tests.
- `services/orion-hub/static/js/biometrics-view.js`: comment.
- `services/orion-hub/README.md`: Inner Sky table and layout text.

## Schema / bus / API changes

None. The page now calls `GET /api/inner-sky/state` every 5 s instead of every 20 s. That is 2 FalkorDB queries and 1 durable-runs `/temporal-self/frame` read per open tab per 5 s.

## Env/config changes

None.

## Tests run

```text
node --test services/orion-hub/static/js/*.test.js        456 pass, 0 fail
pytest test_inner_sky_page.py test_inner_sky_routes.py test_biometrics_view_ui.py test_hub_ui_layout_pass.py   81 passed
pytest (wider related Hub UI set, before the review fixes)   170 passed, 1 skipped
```

## Evals run

```text
Live measurement that motivated this: FalkorDB orion_bus_synapse totals sampled every 10 s for 60 s:
  CAUSALLY_FOLLOWED_BY sum +31..+46 per 10 s (~3.5 hops/s); PUBLISHES sum +339..+419 per 10 s (~38 msgs/s).
run_inner_sky_feed_eval.py is unchanged (server untouched).
```

## Docker/build/smoke checks

```text
Static-only change; no image rebuild needed beyond the normal post-pull Hub rebuild.
Browser render: UNVERIFIED here (headless Chromium blocked in this session); tracked by #3891.
```

## Review findings fixed

- **Finding:** the wide state hid chat on phones (the rule sat outside the md query) and was remembered there.
  - **Fix:** the rule moved inside `@media (min-width: 768px)`, and ▶ is hidden below 768 px.
  - **Evidence:** `test_drawer_has_a_widen_button_and_wide_hides_chat` checks that the rule is inside the md block.
- **Finding:** the sky guessed its layout from its own width, so the Widen link could be dead while the drawer was already wide, and a 2250 px monitor got the full layout in a half drawer.
  - **Fix:** the drawer posts its state to the sky on every change, on iframe load, and on request. The sky uses it, falling back to width only when opened on its own.
  - **Evidence:** `the drawer tells the sky its state...` test.
- **Finding:** every resize reset the camera, including each frame of the drawer's 260 ms transition.
  - **Fix:** the camera is re-placed only when the compact/full layout flips.
- **Finding:** other same-origin frames could send the widen message.
  - **Fix:** the drawer also checks `ev.source` against the sky's own frame.
  - **Evidence:** `the sky can ask to be widened, but only from its own frame...` test.
- **Finding:** a stale `?embed=1` comment.
  - **Fix:** updated.
- **Not changed:** a hidden tab's first poll after a long gap plays up to the cap in one 5 s window. The deltas overwrite rather than accumulate, so there is no runaway.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull rebuild does this on merge.

## Risks / concerns

- **Severity: low.**
  - **Concern:** the state endpoint runs 4× as often per open tab.
  - **Mitigation:** 2 small graph queries and one local HTTP read, about 0.2 s measured. Watch the Hub logs if many tabs are open.
- **Issues opened/closed:** none. #3891 (verify-live) also covers this.

## PR link

This PR.
