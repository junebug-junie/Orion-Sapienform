# PR Report: Orion Hub Mode/Recall Profile cleanup + debug panel modal refactor

## Summary

- Killed 7 Hub chat "Mode" dropdown options (Auto, Grounded Small, Brain, Council, Agent Claude - Opus/Sonnet/Haiku); Orion remains the default, alongside Quick/Story/Agent.
- Wired Recall Profile to auto-follow Mode: Orion mode leaves `recall_profile` unset (backend default resolution); any other mode forces `assist.light.v1`.
- Killed "Auto" and "biographical.v1" from the Recall Profile dropdown.
- Refactored the Hub "Debug Panel" section: removed the outer accordion (rows always visible) and converted all 9 row headers (Memory, Agent Trace, Autonomy Runtime, Chat Stance, Substrate Review, Self Experiments, Autonomy Readiness, Recall Canary, Cognitive Review) from inline dropdown-expand to opening their modal directly. Added a real modal for Autonomy Readiness (previously had none). Resized all 10 debug-panel-adjacent modals to `75vw x 75vh` centered.
- Fixed two review findings: missing Escape-key handler for the new Autonomy Readiness modal, and a stray chat-feed "Open debug" button that still used the abandoned inline-expand pattern instead of opening the modal.

## Outcome moved

The Hub Mode/Recall Profile surface went from 12 dropdown options (several dead-end or redundant) down to a coherent 4+7 set with real, traceable behavior. Debug panel navigation changed from a two-level accordion (outer section + per-row inline expand, duplicated by a separate "Modal" button doing the same thing) to a single consistent interaction: click a row, get its content in a large modal.

## Current architecture

- `services/orion-hub/templates/index.html` renders the Hub chat UI; Mode/Compute/Recall Mode/Recall Profile selects live in a settings strip above the chat input. The "Debug Panel" section further down is a single `id="runtimeDebugPanel"` accordion nesting 9 sub-panel rows, each historically with its own inline-expand toggle *and* a separate "Modal" button doing largely the same thing.
- `services/orion-hub/static/js/app.js` owns `HUB_MODE_SPECS` (Mode -> verb/lane/fcc-model mapping), `applyHubModeSelection()` (Mode change handler), and one `toggle*Panel()` + `open*Modal()` pair per debug-panel row.
- `services/orion-hub/scripts/main.py` server-renders `{{HUB_AGENT_CLAUDE_MODE_OPTIONS}}` into the Mode select when `HUB_AGENT_CLAUDE_ENABLED` is set.
- Discovered mid-task (not previously documented in this form): "Orion" Mode and the `chat_general` cognition verb are two different pipelines — Orion routes through `orion.hub.turn_orchestrator.run_unified_turn()` (harness-governed unified turn), not the verb-dispatch path. Static-trace only (not live-verified): the harness's own recall-profile resolution (`services/orion-cortex-exec/app/recall_utils.py::resolve_runtime_default_profile`) falls through to a hardcoded `"agent"` `runtime_mode` default when nothing upstream sets `ctx["mode"]`, which resolves to `chat.general.v1`. Left as `UNVERIFIED` per house rules — did not pull a live trace to confirm before scoping this PR, since the user redirected mid-investigation to a different concrete ask (mode/recall-profile linkage + debug panel modals) rather than continuing that thread.

## Architecture touched

- `services/orion-hub/templates/index.html`: Mode select, Recall Profile select, entire Debug Panel section markup, 10 modal dialog shells (resized), new Autonomy Readiness modal shell.
- `services/orion-hub/static/js/app.js`: `HUB_MODE_SPECS`, `applyHubModeSelection`, new `setRecallProfileAutoState`, `normalizeRecallProfileDisplay` guard, 9 `toggle*Panel` functions, new `openAutonomyReadinessModal`/`closeAutonomyReadinessModal`/`ensureAutonomyReadinessModalRootOnBody`, `syncDebugModalScrollLock`, global Escape-key handler, one stray inline-expand button in the autonomy chat-feed summary card.
- `services/orion-hub/scripts/main.py`: removed dead `{{HUB_AGENT_CLAUDE_MODE_OPTIONS}}` template-fill block (placeholder no longer exists in the template).

## Files changed

- `services/orion-hub/templates/index.html`: Mode/Recall Profile option pruning, Debug Panel accordion removal, Autonomy Readiness modal added, 10 modal dialogs resized to 75vw/75vh.
- `services/orion-hub/static/js/app.js`: Mode spec pruning, Mode->Recall-Profile auto-linkage, 9 toggle->modal redirects, new Autonomy Readiness modal JS, Escape-key + stray-button fixes.
- `services/orion-hub/scripts/main.py`: removed dead agent-claude Mode-option template-fill code (backend `HUB_AGENT_CLAUDE_ENABLED` feature itself untouched, see Non-goals below).
- `services/orion-hub/tests/test_agent_trace_debug_panel.py`, `test_autonomy_runtime_ui_panel.py`, `test_chat_stance_debug_panel.py`, `test_memory_review_ui.py`: updated string-snapshot assertions to match the new dialog class/id/option set.
- `services/orion-hub/tests/test_llm_route_selector.py`: updated one stale assertion, added 3 new tests covering the surviving Mode option set, the pruned Recall Profile option set, and the new `setRecallProfileAutoState` wiring.

## Schema / bus / API changes

None. This is pure Hub frontend + one dead-code removal in the template renderer.

## Env/config changes

None.

## Non-goals (explicitly scoped out)

- Did **not** remove the deeper "Agent Claude" backend (`HUB_AGENT_CLAUDE_ENABLED` setting, `agent_claude_input.py`, `fcc_claude_bridge.py`, and their tests) — only its Mode-dropdown surface. That backend has real, separate infrastructure (a whole FCC Claude Bridge) that may be reachable outside the Hub Mode dropdown; killing it outright was a much larger, unrequested change.
- Did **not** sweep the now-permanently-hidden inline debug-panel body/caret DOM left behind by the toggle-to-modal conversion (8 of 9 rows). Several of these (confirmed: `autonomyDebugBody`) are still load-bearing — the Autonomy Runtime modal copies its content via `innerHTML` on open — so a blanket deletion would have broken a working modal. Flagged as a follow-up, not fixed here.
- Did not live-verify whether Orion mode's recall profile actually resolves to `chat.general.v1` at runtime (see Current architecture) — static trace only.

## Tests run

```
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests -q -p no:cacheprovider
# Branch:   31 failed, 1032 passed, 5 skipped
# Baseline (git stash to clean main, same command): 31 failed, 1028 passed, 5 skipped
# Diff: 0 branch-only failures (no regressions). All 31 failures pre-exist on main,
# unrelated to this change (DB/fixture-dependent tests, one unrelated JS-string
# assertion) -- confirmed via git-stash baseline comparison, not assumed.
```

Targeted re-run after review-finding fixes:
```
python -m pytest services/orion-hub/tests/test_autonomy_runtime_ui_panel.py \
  services/orion-hub/tests/test_llm_route_selector.py \
  services/orion-hub/tests/test_chat_stance_debug_panel.py \
  services/orion-hub/tests/test_agent_trace_debug_panel.py -q
# 71 passed, 3 failed (same 3 pre-existing context_exec_agent_bridge failures)
```

## Evals run

None — this service has no eval harness for Hub UI interaction; not something this narrow a UI cleanup would warrant standing one up for.

## Docker/build/smoke checks

Not run. This is a static template/JS change with no new dependencies, ports, env keys, or compose wiring; Docker rebuild is not required for these files to take effect (Hub serves `templates/`/`static/` directly). Manual browser verification of the modal interactions was not performed in this session — flagging per house rule rather than claiming UI verification that didn't happen.

## Review findings fixed

- Finding: New `autonomyReadinessModalRoot` modal had no Escape-key handler while every sibling debug-panel modal did.
  - Fix: Added an `Escape` branch calling `closeAutonomyReadinessModal()` alongside the other modal branches in the global keydown handler.
  - Evidence: `services/orion-hub/static/js/app.js`, keydown handler now has 18 modal branches (was 17), same pattern as the other 8 debug-panel-row modals.

- Finding: A chat-feed "Open debug" shortcut in the autonomy summary card still directly un-hid the old inline `autonomyDebugBody`/caret instead of opening the new modal — an inconsistent leftover from the toggle-to-modal refactor.
  - Fix: Replaced the 3-line manual un-hide with a single `openAutonomyDebugModal()` call, matching every other entry point into that content.
  - Evidence: `services/orion-hub/static/js/app.js` around the `createAutonomySummaryPanel`-style builder (~line 7814).

- Finding (not fixed, scoped out): 8 of 9 converted rows still carry permanently-hidden inline body/caret DOM that's dead from a user-interaction standpoint but, in at least one case, still load-bearing as a data source for its modal.
  - Disposition: left in place; documented as a follow-up requiring per-row verification before any deletion.

## Restart required

```text
No restart required.
```
Hub serves `templates/index.html` and `static/js/app.js` directly; changes take effect on next page load (may need a hard refresh / cache-bust if the browser cached the old `app.js`).

## Risks / concerns

- Severity: Low
  Concern: The new `w-[75vw] h-[75vh]` modal sizing has no min-width/min-height floor, so on very small viewports the dialogs shrink proportionally with no safety clamp.
  Mitigation: This was an explicit, deliberate user request ("cover 75% of the screen"); noted as a known trade-off rather than fixed, since a floor wasn't asked for and Hub is not currently used on small/mobile viewports as far as this session could determine.

- Severity: Low
  Concern: 8 of 9 converted debug-panel rows retain now-unreachable inline body/caret markup and the `update*Panel()` functions still write into it every time new data arrives, for no visible benefit.
  Mitigation: None applied this PR; flagged as a follow-up. A blanket sweep needs per-row verification since at least `autonomyDebugBody` is still read by its modal's open function.

- Severity: Info
  Concern: Whether Orion mode's recall profile genuinely resolves to `chat.general.v1` at runtime was traced statically only, not live-verified, mid-conversation before the user redirected to the concrete Mode/Recall-Profile/debug-panel asks actually implemented here.
  Mitigation: None needed for this PR's scope (Orion mode's *behavior* wasn't changed, only what recall_profile override the UI sends) — noted for whoever picks up that thread next.

## PR link

Not pushed / no PR opened yet — pending user confirmation before pushing to the remote per repo convention.
