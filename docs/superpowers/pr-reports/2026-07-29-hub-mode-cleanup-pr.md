# PR Report: Orion Hub Mode/Recall Profile cleanup + debug panel modal refactor

## Summary

- Killed 7 Hub chat "Mode" dropdown options (Auto, Grounded Small, Brain, Council, Agent Claude - Opus/Sonnet/Haiku); Orion remains the default, alongside Quick/Story/Agent.
- Wired Recall Profile to auto-follow Mode: Orion mode leaves `recall_profile` unset (backend default resolution); any other mode forces `assist.light.v1`.
- Killed "Auto" and "biographical.v1" from the Recall Profile dropdown.
- Restructured the Hub "Debug Panel" section into a single card with one "Modal" button. Clicking it opens one new 75vw x 75vh modal containing all 9 sub-panels (Memory, Agent Trace, Autonomy Runtime, Chat Stance, Substrate Review, Self Experiments, Autonomy Readiness, Recall Canary, Cognitive Review) exactly as they were — each still has its own inline expand/collapse toggle *and* its own separate per-item "Modal" button, unchanged. Added a real per-item modal for Autonomy Readiness (previously had none, only an unrelated "Policy Matrix" button). Resized all 10 debug-panel-adjacent modals (the new outer one plus the 9 existing/added per-item ones) to `75vw x 75vh` centered.
- Note: this PR went through one full revision of the Debug Panel approach mid-review. The first pass converted each of the 9 rows' own click target from inline-expand to opening its own modal directly and removed the outer accordion. That was **not** what was wanted — the actual ask was to keep every row's existing behavior completely unchanged and only wrap the *outer* section in one new modal. The first pass was reverted in place before this PR was finalized; what's described above and in the diff is the corrected version.
- Fixed two review findings from the (now-superseded) first pass: missing Escape-key handler, and a stray chat-feed "Open debug" button using an inconsistent pattern. Both fixes were carried forward/re-applied against the corrected structure (Escape-key branch for the new outer `debugPanelModalRoot`; the chat-feed button now opens the outer modal and expands the Autonomy Runtime row inline within it, matching the row's real behavior).

## Outcome moved

The Hub Mode/Recall Profile surface went from 12 dropdown options (several dead-end or redundant) down to a coherent 4+7 set with real, traceable behavior. The Debug Panel went from an always-expanded 9-row accordion sitting permanently in the page flow to a single collapsed card; all 9 rows' existing interactions are preserved verbatim, just relocated inside one on-demand modal.

## Current architecture

- `services/orion-hub/templates/index.html` renders the Hub chat UI; Mode/Compute/Recall Mode/Recall Profile selects live in a settings strip above the chat input. The "Debug Panel" section further down was a single `id="runtimeDebugPanel"` div always rendering all 9 sub-panel rows inline in the page, each with its own inline-expand toggle *and* a separate per-item "Modal" button.
- `services/orion-hub/static/js/app.js` owns `HUB_MODE_SPECS` (Mode -> verb/lane/fcc-model mapping), `applyHubModeSelection()` (Mode change handler), and one `toggle*Panel()` + `open*Modal()` pair per debug-panel row.
- `services/orion-hub/scripts/main.py` server-renders `{{HUB_AGENT_CLAUDE_MODE_OPTIONS}}` into the Mode select when `HUB_AGENT_CLAUDE_ENABLED` is set.
- Discovered mid-task (not previously documented in this form): "Orion" Mode and the `chat_general` cognition verb are two different pipelines — Orion routes through `orion.hub.turn_orchestrator.run_unified_turn()` (harness-governed unified turn), not the verb-dispatch path. Static-trace only (not live-verified): the harness's own recall-profile resolution (`services/orion-cortex-exec/app/recall_utils.py::resolve_runtime_default_profile`) falls through to a hardcoded `"agent"` `runtime_mode` default when nothing upstream sets `ctx["mode"]`, which resolves to `chat.general.v1`. Left as `UNVERIFIED` per house rules — did not pull a live trace to confirm before scoping this PR, since the user redirected mid-investigation to a different concrete ask (mode/recall-profile linkage + debug panel modal wrapping) rather than continuing that thread.

## Architecture touched

- `services/orion-hub/templates/index.html`: Mode select, Recall Profile select. Debug Panel: outer card simplified to a single "Modal" button; the entire original `runtimeDebugPanelBody` (all 9 rows, byte-for-byte their original markup) relocated as-is into a new `debugPanelModalRoot` modal shell. New per-item Autonomy Readiness modal added, mirroring the Autonomy Runtime modal's copy-on-open pattern. 10 modal dialogs (the new outer one, the new Autonomy Readiness one, and the 8 pre-existing per-item ones) resized to 75vw/75vh.
- `services/orion-hub/static/js/app.js`: `HUB_MODE_SPECS`, `applyHubModeSelection`, new `setRecallProfileAutoState`, `normalizeRecallProfileDisplay` guard. Debug panel: all 9 `toggle*Panel()` functions restored to their original inline expand/collapse behavior (unchanged from before this PR); new `openDebugPanelModal`/`closeDebugPanelModal`/`ensureDebugPanelModalRootOnBody` for the single outer modal; new `openAutonomyReadinessModal`/`closeAutonomyReadinessModal`/`ensureAutonomyReadinessModalRootOnBody` following the `autonomyDebugModal` copy-on-open pattern; `syncDebugModalScrollLock` and the global Escape-key handler both extended to cover the two new modal roots.
- `services/orion-hub/scripts/main.py`: removed dead `{{HUB_AGENT_CLAUDE_MODE_OPTIONS}}` template-fill block (placeholder no longer exists in the template).

## Files changed

- `services/orion-hub/templates/index.html`: Mode/Recall Profile option pruning; Debug Panel restructured into one card + one wrapping modal around the unchanged 9-row content; Autonomy Readiness per-item modal added; 10 modal dialogs resized to 75vw/75vh.
- `services/orion-hub/static/js/app.js`: Mode spec pruning; Mode->Recall-Profile auto-linkage; Debug Panel wrapped in one new modal with all 9 rows' original toggle behavior preserved; new Autonomy Readiness modal JS; Escape-key + scroll-lock coverage for both new modals.
- `services/orion-hub/scripts/main.py`: removed dead agent-claude Mode-option template-fill code (backend `HUB_AGENT_CLAUDE_ENABLED` feature itself untouched, see Non-goals below).
- `services/orion-hub/tests/test_llm_route_selector.py`: added tests covering the surviving Mode option set, the pruned Recall Profile option set, the `setRecallProfileAutoState` wiring, and the single-card/single-modal Debug Panel wrapping (row content lives inside the modal, main-page card has no inline row content, each row keeps its own toggle + per-item Modal button).

## Schema / bus / API changes

None. This is pure Hub frontend + one dead-code removal in the template renderer.

## Env/config changes

None.

## Non-goals (explicitly scoped out)

- Did **not** remove the deeper "Agent Claude" backend (`HUB_AGENT_CLAUDE_ENABLED` setting, `agent_claude_input.py`, `fcc_claude_bridge.py`, and their tests) — only its Mode-dropdown surface. That backend has real, separate infrastructure (a whole FCC Claude Bridge) that may be reachable outside the Hub Mode dropdown; killing it outright was a much larger, unrequested change.
- Did **not** change any of the 9 debug-panel rows' own interaction pattern (inline toggle + separate per-item Modal button, unchanged) — only wrapped the outer section in one new modal. An earlier draft of this PR did change per-row behavior; that draft was reverted before finalizing, see Summary.
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

An independent code-review subagent reviewed the first-pass version of this PR (per-row toggle-to-modal conversion). Two real findings from that pass were fixed at the time; both fixes were carried forward into the corrected structure below since they're still applicable:

- Finding: The new outer debug-panel-wrapping modal had no Escape-key handler while every other modal did.
  - Fix: Added an `Escape` branch calling `closeDebugPanelModal()` (and separately `closeAutonomyReadinessModal()` for the new per-item Autonomy Readiness modal) alongside the existing modal branches in the global keydown handler.
  - Evidence: `services/orion-hub/static/js/app.js` global keydown handler.

- Finding: A chat-feed "Open debug" shortcut in the autonomy summary card needs to reach into the now-modal-wrapped Autonomy Runtime row.
  - Fix: The button now calls `openDebugPanelModal()` first, then un-hides the Autonomy Runtime row's inline body within it — matching how a user would actually reach that row by hand (open Debug Panel modal, expand the row).
  - Evidence: `services/orion-hub/static/js/app.js`, the `debugButton` click handler in the autonomy chat-feed summary card builder (~line 7886).

- Finding (from the first pass, no longer applicable): 8 of 9 rows would have carried permanently-hidden, unreachable inline body/caret DOM under the per-row-modal design.
  - Disposition: moot — the corrected design keeps every row's original inline toggle fully live and reachable (nothing was made unreachable), since only the outer section was wrapped in a modal.

## Restart required

```text
No restart required.
```
Hub serves `templates/index.html` and `static/js/app.js` directly; changes take effect on next page load (may need a hard refresh / cache-bust if the browser cached the old `app.js`).

## Risks / concerns

- Severity: Low
  Concern: The new `w-[75vw] h-[75vh]` modal sizing has no min-width/min-height floor, so on very small viewports the dialogs shrink proportionally with no safety clamp.
  Mitigation: This was an explicit, deliberate user request ("cover 75% of the screen"); noted as a known trade-off rather than fixed, since a floor wasn't asked for and Hub is not currently used on small/mobile viewports as far as this session could determine.

- Severity: Info
  Concern: Whether Orion mode's recall profile genuinely resolves to `chat.general.v1` at runtime was traced statically only, not live-verified, mid-conversation before the user redirected to the concrete Mode/Recall-Profile/debug-panel asks actually implemented here.
  Mitigation: None needed for this PR's scope (Orion mode's *behavior* wasn't changed, only what recall_profile override the UI sends) — noted for whoever picks up that thread next.

## PR link

Not pushed / no PR opened yet — pending user confirmation before pushing to the remote per repo convention.
