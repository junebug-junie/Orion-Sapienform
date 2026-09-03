# Hub chat-column layout pass

## Summary

Seven operator-facing changes to the Hub chat column, all frontend-only.

- **Skill Runner and Container bring-up collapse into one bar.** They were two
  stacked full-width panels eating roughly 7rem of the chat column. They are now
  one "Operator tools" bar with a button that opens a modal holding both. Every
  element id `app.js` binds to is unchanged — only the chrome moved.
- **Every chat message gets an Expand button** that opens it full-screen, the
  same affordance the composer's message box already had. Copy in the modal
  yields the markdown source, not space-mangled `innerText`.
- **The Oríon's Voice card moved** out of the Cognitive EKG column into the chat
  column, sitting between the "Oríon + Juniper" rule and the transcript.
- **The voice visualizer is now actually constrained by its card.** It was sized
  from the padded container including the label row, so its drawing surface was
  taller than the space it had.
- **The Recall / No-write / TTS / Social room / Solo strip wraps instead of
  bleeding outside the card.**
- **An elapsed-turn timer** runs beside the status line while a turn is in
  flight and freezes on the final duration.
- **Dismiss-all** on Pending Attention and on Notifications, and a scroll window
  on Cognitive Loops matching the notification tray's.

## Outcome moved

Chat column height: two full-width operator panels became one bar, which is what
paid for the voice card moving in. Net column height is roughly unchanged while
the transcript keeps its space.

Two things that were visibly broken are fixed rather than restyled: the toggle
strip could not wrap and so drew outside its card at any normal window width,
and the voice canvas's backing store was taller than its box so the bars drew
past the bottom edge.

## Current architecture

The Hub UI is one large server-rendered template (`templates/index.html`, read
from disk per request by `render_hub_index_html`) plus one large IIFE
(`static/js/app.js`, ~14k lines) that binds behavior by `getElementById`. Small
pure view-model helpers are split into standalone modules
(`container-bringup-ui.js`, `cognitive-loop-card.js`) so their branching is unit
testable under `node --test` without a DOM harness.

Because behavior binds by id and not by position, moving markup is safe as long
as the ids survive — that is what made this pass a template reshuffle rather
than a JS rewrite.

## Architecture touched

Presentation only. No service, bus channel, schema, route, or env key changed.
One new pure module (`turn-timer.js`) follows the existing standalone-module
pattern.

## Files changed

- `services/orion-hub/templates/index.html`: voice card relocated; toggle strip
  split into two wrapping groups; the two operator panels replaced by one bar
  and re-parented into a new Operator tools modal; new per-message expand modal;
  turn-timer chip; two Dismiss-all buttons; scroll window on Cognitive Loops.
- `services/orion-hub/static/js/app.js`: open/close for the two new modals; the
  per-message Expand button in `appendMessage`; `setTurnInFlight()` plus timer
  lifecycle; `dismissAllPendingAttention()` / `dismissAllNotifications()`;
  canvas sizing fix and a `ResizeObserver`; Escape and backdrop wiring.
- `services/orion-hub/static/js/turn-timer.js`: new — pure elapsed formatting.
- `services/orion-hub/static/js/turn-timer.test.js`: new — 5 formatter cases.
- `services/orion-hub/tests/test_hub_ui_layout_pass.py`: new — 16 structural
  guards, one per claim above.
- `docs/operator_skill_prompt_catalogue.md`: one stale sentence — the bring-up
  panel is no longer "bottom of the operator controls".

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none. Dismiss-all on Pending Attention issues N of the
  existing `POST /api/attention/{id}/ack` calls rather than a new bulk endpoint,
  so the server sees no new shape.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable, no env key touched.
- local `.env` synced: not applicable, no env template changed.
- skipped keys requiring operator action: none.

## Tests run

```text
# Full Hub suite, this branch, in a worktree
pytest -c services/orion-hub/pytest.ini services/orion-hub/tests -q \
  --ignore=services/orion-hub/tests/e2e -p no:randomly
31 failed, 2046 passed, 3 skipped in 269.32s

# Same suite at the merge-base (b97a8531a) in a second worktree, same env state
31 failed, 2030 passed, 3 skipped in 298.45s

diff of the two FAILED sets: identical (0 lines)
```

The 31 failures are pre-existing and identical on both sides: they are Settings
validation errors from the gitignored `.env` being absent in a fresh worktree,
plus stale guards in `test_substrate_review_runtime_hub_debug.py` and
`test_substrate_effect_*.py` that also fail in the unmodified primary checkout.
The +16 passing is exactly the new `test_hub_ui_layout_pass.py`.

```text
# New structural guards
pytest -c services/orion-hub/pytest.ini \
  services/orion-hub/tests/test_hub_ui_layout_pass.py -q
16 passed

# Full JS suite (was 124/0 before this patch)
cd services/orion-hub && node --test static/js/
tests 129 | pass 107 | fail 0 | skipped 22
```

Structural analysis run alongside the suites, since a template reshuffle's real
risk is a dropped id rather than a failing assertion:

```text
ids removed by this patch:            none
getElementById lookups now broken:    NONE
duplicate ids introduced:             NONE  (66 pre-existing, unchanged)
div balance, chat card:               balanced (12667 bytes)
div balance, operator tools modal:    balanced (9045 bytes)
div balance, message expand modal:    balanced (1642 bytes)
div balance, mode/compute/toggles:    balanced (4765 bytes)
HTML nesting error signature:         identical to HEAD (3 pre-existing)
```

## Evals run

```text
None. services/orion-hub has no evals/ harness for presentation-layer changes,
and this patch adds no behavior an eval could score. Not claiming eval coverage.
```

## Docker/build/smoke checks

```text
None run, and deliberately so.

services/orion-hub/docker-compose.yml volume-mounts templates/ and static/ from
${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform} -- the PRIMARY checkout,
not this worktree. Bringing the Hub up from here would either serve main's UI
(proving nothing) or, if ORION_HOST_REPO_ROOT were pointed at this worktree,
pin a disposable worktree as production. Neither is worth doing for a
presentation change.

CI static gates, all 11 from .github/workflows/orion-static-gates.yml: PASS
  check_metric_lineage --gate, check_definition_drift --gate,
  check_inner_state_registry, check_scripts_dir_no_stdlib_shadow,
  check_service_hostname_refs, check_compose_no_relative_mounts,
  check_journal_dispatch_registry, check_daily_schedule_collisions,
  check_sentience_instruments --static-only, check_system_health_producers,
  check_control_surface_store_parity
```

## Review findings fixed

See "Review findings" section appended below.

## Restart required

```text
No restart and no rebuild.

templates/ and static/ are volume-mounted from the primary checkout, and
render_hub_index_html() re-reads index.html from disk per request. The
?v= cache-buster is derived from an rglob of static file mtimes
(_ui_asset_mtime_token), so the new turn-timer.js and the edited app.js bump it
on their own.

Once this merges, `git pull` in /mnt/scripts/Orion-Sapienform and reload the
browser.
```

## Risks / concerns

- Severity: low
  Concern: **The live path is UNVERIFIED.** Every claim here rests on the test
  suite, the id-resolution analysis, and div-balance checks — not on a browser
  actually rendering the new layout. That is a real gap, and it is structural:
  the Hub container mounts the primary checkout, so this UI cannot be loaded in
  a browser until it merges.
  Mitigation: the 16 structural guards plus the "no id removed, no lookup
  broken, no duplicate introduced" analysis cover the failure mode a template
  move actually has. Visual judgment — whether the voice card at `h-28` leaves
  the transcript enough room, whether the wrapped toggle strip reads well — is
  for Juniper on first load, and is a one-line class change either way.

- Severity: low
  Concern: `containerBringupStatus` now lives inside the modal, so a bring-up
  that is still running is invisible if the modal is closed.
  Mitigation: bring-up is operator-initiated and watched; mirroring status onto
  the collapsed bar would be speculative scope. Worth adding only if it actually
  bites.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2055
