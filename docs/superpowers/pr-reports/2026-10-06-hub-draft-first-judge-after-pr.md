## Summary

- Unified Hub chat now shows Orion's draft as soon as the reply writer finishes, while the finalize judge is still checking it (spec L8, approved 2026-10-06).
- When the judge rewrites the reply, the message is replaced where it sits and marked "revised: <reason>". When it doesn't, the draft is swapped for the same text with no mark.
- Sensitive turns are never shown early; they keep judge-before-display. "Sensitive" means the stance flagged a boundary, a trust rupture, high repair pressure, or a non-default repair overlay. The same `sensitive_turn_reason` helper now feeds the quick lane too.
- Only the final text is ever saved (chat history, memory, TTS, outreach). The draft is display-only, so a revision cannot create a duplicate turn.
- Hub flag `HUB_UNIFIED_DRAFT_FIRST_ENABLED`, shipped on. Set it to `false` to go back to judge-first for every turn.

## Outcome moved

Time until Juniper sees the first reply should drop by roughly the judge's time (spec estimate: about 11 s). This is UNVERIFIED live: nothing has been deployed.

Proof hooks, all keyed by correlation id:
- Hub logs:
  - `unified_turn_first_visible corr=… kind=draft_preview|final elapsed_ms=…`
  - `unified_turn_final_visible`
  - `unified_turn_revision corr=… reason=…`
- Governor logs:
  - `harness_draft_preview_published|held|revised`
- `harness_turn_trace.run_artifact` stores three fields: `draft_preview_text` (what was shown first), `final_text` (what it became), and `draft_preview_held_reason`.

## Current architecture

How a turn works before this patch:
1. The Hub's `execute_unified_turn` sends a bus request (RPC) to the harness governor.
2. The governor runs the motor (Claude Code), then `run_harness_finalize_chain`: substrate appraisal, the `harness_finalize_reflect` judge, and optional `orion_response_repair`.
3. The governor replies with `HarnessRunV1`.
4. The Hub sends a single `final` websocket frame.

While this happens, motor steps stream to the browser over `orion:harness:run:step`, through `HarnessStepRelay` and per-turn queues.

## Architecture touched

- **Contract:**
  - new event `HarnessRunDraftPreviewV1` on `orion:harness:run:draft_preview`, governor → hub;
  - additive `HarnessRunRequestV1.draft_preview`;
  - additive `HarnessRunV1.draft_preview_text` and `HarnessRunV1.draft_preview_held_reason`.
- **Governor:** publishes the draft after deterministic reading-receipt grounding (no LLM call) and before the judge.
- **Hub:**
  - the relay subscribes to both channels and puts the draft on the turn's queue;
  - `run_unified_turn` sends a `draft_preview` frame;
  - the final frame gets `replaces_draft`, `revised` and `revised_reason`.
- **Browser:** new `static/js/draft-revision.js` does the in-place swap. `appendMessage` now returns its node.

## Files changed

- `orion/schemas/harness_finalize.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: the contract.
- `orion/harness/finalize.py`: `sensitive_turn_reason` (extracted from the quick lane), `draft_preview_hold_reason`, `draft_preview_display_text`.
- `orion/harness/step_stream.py`: the publish helper.
- `services/orion-harness-governor/app/{bus_listener.py,settings.py}`, `.env_example`, `docker-compose.yml`, `README.md`: the governor side.
- `orion/hub/turn_orchestrator.py`, `services/orion-hub/scripts/{harness_step_relay.py,main.py}`, `app/settings.py`, `.env_example`, `README.md`: the Hub side.
- `services/orion-hub/static/js/{draft-revision.js,draft-revision.test.js,app.js}`, `templates/index.html`: the UI.
- Tests:
  - `services/orion-hub/tests/test_draft_first_display.py`
  - `services/orion-harness-governor/tests/test_draft_preview.py`
  - `orion/harness/tests/test_draft_preview_hold.py`
  - `tests/test_unified_turn_bus_catalog.py`
- `config/metrics/metric_definitions.lock.json`: re-locked because the new bus channel is a real definition change.

## Schema / bus / API changes

- **Added:**
  - channel `orion:harness:run:draft_preview` with kind `harness.run.draft_preview.v1`;
  - `HarnessRunDraftPreviewV1`;
  - request field `draft_preview`;
  - run fields `draft_preview_text` and `draft_preview_held_reason`;
  - final-frame keys `response_repair_ran`, `response_repair_reason`, `replaces_draft`, `revised`, `revised_reason`;
  - WS frame `{"type":"draft_preview","draft_text"}`.
- **Removed / renamed:** none.
- **Behavior changed:** interactive chat turns may show a draft first. Other callers are unchanged (curiosity, outreach, reading, collapse-mirror, HTTP), because `draft_preview` defaults to False.
- **Compatibility:**
  - All the models use pydantic's default of ignoring extra fields, so either service can be deployed first.
  - A browser with cached old JS ignores `draft_preview`. The text is sent under `draft_text`, not `llm_response`/`text`.
- **Metric gate:** this is a display event. It is not fed into any cognition loop.

## Env/config changes

- Added keys:
  - `CHANNEL_HARNESS_RUN_DRAFT_PREVIEW` (hub and governor);
  - `HUB_UNIFIED_DRAFT_FIRST_ENABLED=true` (hub).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes for the channel keys.
- Skipped keys: `HUB_UNIFIED_DRAFT_FIRST_ENABLED` falls outside SYNC_PREFIXES. It was added by hand to `services/orion-hub/.env` in the primary checkout.

## Tests run

```text
orion-harness-governor tests: 73 passed
hub: test_draft_first_display + relay + ws_frames + cockpit_hops + unified_turn_tts: 114 passed
orion/harness/tests + tests/test_unified_turn_bus_catalog.py: 432 passed
node --test draft-revision.test.js: 7 pass (fake DOM; jsdom is not installed on host)
check_bus_reply_channels, check_chat_route_poachers, check_async_routes_not_blocking, check_metric_lineage --gate, check_definition_drift --gate, check_inner_state_registry, check_env_template_parity: PASS
git diff --check: clean
Pre-existing failures, also on main: test_turn_orchestrator_utterance_origin::test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness, tests/test_channel_prefix_guardrail.py
```

## Evals run

```text
None. There is no eval harness for this behavior. The proof is the live revision rate: it should be about 19% of turns, the share where repair actually changed text. Run it after deploy from the Hub/governor log lines above.
```

## Docker/build/smoke checks

```text
Not run (no deploy requested). Live path UNVERIFIED.
```

## Review findings fixed

- **Finding:** when the substrate appraisal timed out (degraded path), the governor delivered the raw draft after the grounded draft had already been shown. The Hub then marked an unrevised turn as "revised".
  - **Fix:** the degraded final is now the text that was shown, whenever a preview was published.
  - **Evidence:** `test_degraded_path_delivers_the_grounded_text_that_was_shown` and `test_degraded_path_without_preview_keeps_the_raw_draft`.
- **Finding:** if the Hub raised an exception after the draft was shown, the bubble said "still being checked" forever.
  - **Fix:** the exception is held until the step drain has flushed, then a `turn_error` frame with `draft_shown=true` is sent before re-raising.
  - **Evidence:** `test_hub_exception_after_draft_still_settles_the_bubble`.
- **Finding (nit):** the governor recomputed the repair overlay instead of reusing it; its revision log compared text more strictly than the Hub; the "never persisted" wording was not literally true (bus-mirror copies the event).
  - **Fix:** the overlay is passed in, both checks compare stripped text, and the docstring and channels comment are reworded.
  - **Evidence:** the governor suite passes.

## Restart required

Deploy the governor first, then the Hub. Either order is safe, but this one avoids a window where the Hub asks for drafts that never arrive.
```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** a misaligned draft is visible for about 10–20 s before it is revised. Juniper approved this UX in principle; it has not been tried live.
  - **Mitigation:** sensitive turns stay judge-first, and `HUB_UNIFIED_DRAFT_FIRST_ENABLED=false` turns the feature off.
- **Severity: low.**
  - **Concern:** the Hub tests fake `execute_unified_turn`, so the real timing between the draft arriving and the governor's reply has no test.
  - **Mitigation:** a draft that arrives late is dropped rather than shown after the final, and the turn then looks exactly like today.
- **Severity: low.**
  - **Concern:** the "revised" marker is not saved to chat history, so it disappears on reload. `harness_turn_trace` keeps both texts.

## PR link

(filled after creation)
