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
31 failed, 2055 passed, 3 skipped in 196.87s   (after review fixes, post-rebase)

# Same suite at the merge-base (248e36470) in a second worktree, same env state
31 failed, 2030 passed, 3 skipped in 209.34s

diff of the two FAILED sets: identical (0 lines)
```

Re-baselined after rebasing onto `origin/main` at `248e36470` — a baseline
expires the moment the branch moves, so both numbers above come from runs at the
current merge-base, not the old one.

The 31 failures are pre-existing and identical on both sides: they are Settings
validation errors from the gitignored `.env` being absent in a fresh worktree,
plus stale guards in `test_substrate_review_runtime_hub_debug.py` and
`test_substrate_effect_*.py` that also fail in the unmodified primary checkout.
The +25 passing is the new `test_hub_ui_layout_pass.py` (24) and
`test_turn_timer_js.py` (1).

```text
# New structural guards
pytest -c services/orion-hub/pytest.ini \
  services/orion-hub/tests/test_hub_ui_layout_pass.py -q
24 passed

# Mutation harness over those guards -- 12 reversions attempted
CAUGHT  delete the whole Expand button block
CAUGHT  delete both Dismiss-all click handlers
CAUGHT  strip flex-wrap from the OUTER toggle row
CAUGHT  drop `hidden` from operatorToolsModalRoot
CAUGHT  revert canvas WIDTH to the container
CAUGHT  restore the start-timer early-return guard
CAUGHT  remove the paintTurnTimer module guard
CAUGHT  remove the dismiss-all in-flight flag
CAUGHT  remove stopTurnTimer's owner check
CAUGHT  remove the onclose freeze
CAUGHT  remove the empty-nodes guard in the expand modal
CAUGHT  remove max-h-56 from cognitiveLoopsList
SURVIVING MUTATIONS: NONE

# Full JS suite (was 124/0 before this patch)
cd services/orion-hub && node --test static/js/
tests 131 | pass 109 | fail 0 | skipped 22
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

Review ran in a subagent against `374dec625` and mutation-tested the suite. It
found one real bug, four timer-lifecycle holes, and — the important one — that
the tests were not load-bearing. Every finding below was reproduced by hand
before being fixed.

- Finding: `turn-timer.js` printed `60.0s`, the exact string its own test said
  could never appear. It branched on `secs < 60` but rendered `toFixed(1)`, so
  59.950s–59.999s passed the under-a-minute check and then rounded up. The
  fixture stopped at `59949` — one millisecond below the break — so it was
  written to the code rather than to the claim.
  - Fix: branch on the value actually displayed, not the raw one.
  - Evidence: `59950 -> "60.0s"` before, `-> "1m 00s"` after. The test now
    sweeps every millisecond in 59000–61000 and 119000–121000 asserting neither
    `60.0s` nor `1m 60s` can be produced.

- Finding: **four features could be deleted outright with all 16 tests green.**
  The suite asserted that functions were *defined* and that HTML *existed* — not
  that any button called anything. Deleting the Expand block, deleting both
  Dismiss-all click handlers, and stripping `flex-wrap` off the outer toggle row
  each left the suite passing.
  - Fix: rewrote all assertions to pin the call site inside a sliced function
    body. 24 tests now.
  - Evidence: 12 reversions attempted under a mutation harness, **all 12
    caught**, including the 4 that previously slipped through.

- Finding: the wrapping test anchored with `rindex("<div class=")`, which finds
  the *inner* toggle group, not the outer row that actually overflowed.
  - Fix: anchor on `border-t border-gray-700/80`, unique to the outer row.
  - Evidence: stripping `flex-wrap` from the outer row is now caught.

- Finding: a WebSocket reconnect stopped a clock it never started — landing
  mid-HTTP-turn, it froze that turn's timer partway and the `.finally` then
  no-opped.
  - Fix: each transport owns the clock it started and may only stop that one.
  - Evidence: `test_each_transport_owns_the_clock_it_started`; removing the
    owner check is caught.

- Finding: a WS turn outliving its socket kept counting through the entire
  outage, then repainted on reconnect — reporting a turn that died at 3s as
  `5m 12s`. A wrong number presented as a measurement.
  - Fix: `socket.onclose` freezes the chip (`repaint: false`) rather than
    letting reconnect repaint it.
  - Evidence: `test_socket_close_freezes_the_clock...`; removing it is caught.

- Finding: the `if (turnTimerHandle) return;` start guard let a turn that never
  reached idle hand the *next* turn its accumulated elapsed, permanently. Its
  stated justification — "a second lane joining the same turn" — describes a
  case that cannot occur, since the two lanes are the branches of one if/else.
  - Fix: every start restarts the clock, clearing any prior interval.
  - Evidence: `test_a_new_turn_cannot_inherit_the_previous_turn_start_time`.

- Finding: `paintTurnTimer` was the only unguarded optional-module call in
  app.js. It is reached from `updateStatusBasedOnState` inside
  `socket.onmessage`'s try/catch, so a missing `turn-timer.js` would have
  silently stopped `handleTtsFields` and error rendering for the rest of the
  frame — a cosmetic chip taking out voice output.
  - Fix: guarded like every other optional module in the file.
  - Evidence: `test_the_timer_chip_cannot_throw_out_of_the_websocket_frame_handler`.

- Finding: Dismiss-all re-enabled its own button on the first ack while the rest
  were still in flight, because each ack re-renders and the render re-derives
  the disabled state from the remaining list.
  - Fix: an in-flight flag that the re-render honours, cleared in `finally`.
  - Evidence: `test_attention_dismiss_all_stays_disabled_until_every_ack_settles`.

- Finding: Expand on a whitespace-only turn opened a full-screen modal over a
  detached, blank node. Such a turn is also a `workflowOnlyTurn`, so its body is
  never appended — an empty-shell UI state, which section 0A bans by name.
  - Fix: gate the button on trimmed text; the modal refuses to open on nothing.
    It now also clones the attachment strip, so it is the full view of the turn
    it claims to be.
  - Evidence: `test_expand_button_is_gated_on_trimmed_text`.

- Finding: the two new Escape branches were the only ones in a ~20-branch chain
  without a `return`, so one Escape closed two modals.
  - Fix: added `return`.

- Finding: dead code — `const visualizerContainer` lost its only use to the
  canvas fix, and `classList.remove('line-clamp-3')` targeted a class never
  applied to a message body.
  - Fix: both removed.

- Finding: `turn-timer.test.js` is run by no gate — not CI, not the Makefile.
  Eleven other `.test.js` files share this, so it is a pre-existing repo gap,
  but the commit advertised coverage nothing executed.
  - Fix: `test_turn_timer_js.py` runs it from the Python suite. It skips
    *loudly* when node is absent rather than passing, so a missing runtime can
    never read as a green formatter. Not attempting the repo-wide gap here.

- Finding (my own, found while fixing the above): the "modals start hidden"
  assertion passed with `hidden` stripped, because the roots carry
  `aria-hidden="true"` and a substring check matches it.
  - Fix: class assertions compare tokens, not substrings.
  - Evidence: that mutation is now caught.

Checked by review and clean: no element id dropped or duplicated, the toggle
strip's div nesting balanced in both revisions, `setTurnInFlight`'s stopButton
toggle changes no existing behavior at any of the four replaced sites, and the
ResizeObserver cannot feedback-loop.

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
  Concern: Dismiss-all on Notifications is client-side for any row without a
  `message_id`/`session_id` pair — it drops them from the local array, and the
  next `loadNotifications` replaces that array wholesale. This is identical to
  the existing per-item Dismiss, so it is not a regression, but it turns a
  one-row surprise into a fifty-row one.
  Mitigation: none applied; matching the existing per-item semantics was the
  right call for this pass. Worth revisiting as its own change if the rows
  coming back is actually annoying in practice.

- Severity: low
  Concern: `containerBringupStatus` now lives inside the modal, so a bring-up
  that is still running is invisible if the modal is closed.
  Mitigation: bring-up is operator-initiated and watched; mirroring status onto
  the collapsed bar would be speculative scope. Worth adding only if it actually
  bites.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2055
