## Summary

- The composer's "Lend chat GPU" button (circe gpu0 only) is replaced by a **GPU leases** button that opens a small modal.
- The modal has one animated on/off switch per lendable card the pool reports: circe gpu0 (chat) and hecate-gpu0 (agent-deep) today. A new lendable card in `config/gpu_pool.yaml` appears with no Hub change.
- Each flip posts lend/unlend to the existing `POST /api/gpu-pool/control` and settles on the pool's reply. A refusal snaps back and shows the reason. A timeout re-reads the pool instead of guessing.
- The button badge shows how many cards are lent ("1 lent"), refreshed once a minute while the page is visible.
- A card whose server is down is marked, because lending it does nothing until it answers.

## Outcome moved

hecate's card could only be lent from a control buried at the bottom of the GPU pool tab (Juniper could not find it). It is now one click from the chat screen, next to the compute-lane picker, without taking up extra space.

## Current architecture

`app.js` rendered a gpu0-only toggle whose state came from the `chat-burst` route's `gate_open` in the 30 s `/api/llm-routes` poll. The general lend buttons existed only in `gpu_pool.js`'s Controls section.

## Architecture touched

Hub frontend only. No pool, schema, bus, or env change. It reuses `GET /api/gpu-pool/state` and `POST /api/gpu-pool/control` unchanged.

## Files changed

- `services/orion-hub/static/js/gpu_pool_leases.js`: new modal module (rows from pool state, switch, flip, focus trap, badge poll).
- `services/orion-hub/static/js/gpu_pool_leases.test.js`: node unit tests, plus a guard that the modal root has no display utility defeating `hidden`.
- `services/orion-hub/tests/test_gpu_leases_modal_browser_smoke.py`: Playwright smoke driving the real markup.
- `services/orion-hub/templates/index.html`: button + modal markup + script tag; old button removed.
- `services/orion-hub/static/css/style.css`: switch styles (slide + green track on `aria-checked`, reduced-motion respected).
- `services/orion-hub/static/js/app.js`: old gpu0 toggle code removed.
- `services/orion-hub/tests/test_llm_gateway_client_routes.py`: removed the source-level test of the deleted `chatBurstGateFromCatalog`. Its invariant ("unknown pool state keeps the last known value, never renders a guess") now lives in the module: a failed read keeps the badge and shows an error.
- `services/orion-hub/README.md`: section rewritten.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the composer lend control now covers every lendable card, not just gpu0.
- Compatibility notes: none

## Env/config changes

- None. `.env_example` is untouched, so no sync is needed. (`config/gpu_pool.yaml` was deliberately left alone: its pool `config_digest` hashes raw bytes, so even a comment edit would move the deployed digest.)

## Tests run

```text
node --test $(find services/orion-hub/static/js -name '*.test.js')   -> pass 254, fail 0
pytest services/orion-hub/tests/test_llm_gateway_client_routes.py services/orion-hub/tests/test_gpu_pool_routes.py -> 33 passed, 1 skipped
pytest services/orion-hub/tests/test_gpu_leases_modal_browser_smoke.py services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py -> 4 passed
  (run locally with Playwright pointed at the cached chromium-headless-shell 1243; the venv's Playwright wanted a newer build)
```

The browser smoke covers: open, the non-lendable card being hidden, the server-down mark, a click flip on and off (knob transform changes), the badge, a refused flip snapping back, Space on a focused switch keeping focus, Tab staying in the dialog, a 504 after the pool applied the flip showing the real state, and Escape to close.

## Evals run

```text
None. This is a UI control with no behavior to evaluate; the browser smoke is the behavioral check.
```

## Docker/build/smoke checks

```text
Not rebuilt. Static assets only; the live pool state was read from the running Hub (gpu0 + hecate-gpu0 lendable, both not lent) to confirm the data the modal renders.
```

## Review findings fixed

- Finding: a 504 that arrives after the pool applied the flip made the switch show the wrong state.
  - Fix: on error, re-read pool state and re-render; fall back to the old state only if that read fails or another flip is in flight.
  - Evidence: browser smoke "did apply" case.
- Finding: `disabled` on the focused switch dropped keyboard focus to `<body>`; no focus trap despite `aria-modal`.
  - Fix: `aria-disabled` + guard, Tab trap inside the dialog.
  - Evidence: browser smoke keyboard steps.
- Finding: "server down" ignored the pool's `static` status (healthy service roles).
  - Fix: uses the pool's own up set (`confirmed`, `static`).
  - Evidence: node test "a healthy service role (status static) counts as serving".
- Finding: a slow background poll could overwrite the badge after a fresh flip.
  - Fix: sequence counter; reads started before a flip, or landing during one, are dropped.
  - Evidence: code path; covered indirectly by the smoke's badge assertions.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the role line under each card lists every role covering it (gpu0 shows "chat, experiment").
- Mitigation: cosmetic; left as is.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
