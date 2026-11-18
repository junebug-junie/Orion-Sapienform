## Summary

- The current-turn signal probe (a small LLM read of "what did the user just say, and do they want a direct answer") now runs only on turns that carry a real human message.
- Orion's own turns (journal.compose, metacognition, render_scene, harness finalize, endogenous outreach, autonomous reading, curiosity investigation) skip it and never send the LLM request.
- The probe's route (`chat`), prompt, timeout and behavior on real chat turns are unchanged.
- Skipped turns get an explicit, distinguishable read state (`skipped: "not_human_turn"` plus the reason) instead of looking like a timeout or "found nothing"; the attention frame debug shows it as `turn_read_skipped`.
- No new env key, no schema/bus change.

## Outcome moved

Every Orion-initiated turn that built chat stance was sending an interactive-priority LLM request on route `chat`, the owner class of gpu0. Each one made the gpu-pool recall borrowed gpu0 holds from durable runs, which SIGKILLs harness turns and pushes some runs to the 12-takeback failure cap.

Live evidence (verified 2026-10-01, read-only):

- 2026-09-30: ~1,600 chat-class admits from `cortex-exec` in `gpu_pool_events`, 1,058 probe timeouts, 248 `recalled/owner_waiting`. Only 2 of 202 recall-triggering calls were real `chat_history_log` messages.
- `gpu_pool_events`, work_class `chat`, holder `cortex-exec`, admitted per day: 09-28 1,027 / 09-29 1,408 / 09-30 1,607.
- `recalled/owner_waiting` per day: 09-28 113 / 09-29 161 / 09-30 248.
- Last 6h of `cortex-exec-background` logs: 82 probe log lines against journal.compose (22), log_orion_metacognition (18), harness_finalize_reflect (5), render_scene (4), orion_response_repair (1) -- no human turns run on that container.

Expected after deploy: chat-class admits from `cortex-exec` and `recalled/owner_waiting` both drop sharply (roughly to real chat volume plus the reply calls themselves). UNVERIFIED until deployed.

Verify with:

```sql
-- chat-class admits from cortex-exec, per day
SELECT date_trunc('day', created_at)::date AS d, count(*)
FROM gpu_pool_events
WHERE work_class = 'chat' AND holder = 'cortex-exec' AND event = 'admitted'
  AND created_at > now() - interval '5 days'
GROUP BY 1 ORDER BY 1;

-- gpu0 borrow recalls because the owner was waiting, per day
SELECT date_trunc('day', created_at)::date AS d, count(*)
FROM gpu_pool_events
WHERE event = 'recalled' AND reason = 'owner_waiting'
  AND created_at > now() - interval '5 days'
GROUP BY 1 ORDER BY 1;
```

And in logs, autonomous turns should now show `current_turn_llm_signals_skipped ... reason=non_chat_verb:journal.compose` (INFO) instead of `current_turn_llm_signals_rpc_failed` warnings:

```bash
docker logs --since 1h orion-athena-cortex-exec-background 2>&1 | grep -c current_turn_llm_signals_skipped
```

## Current architecture

`chat_stance.py::build_chat_stance_inputs` runs for every `mode=brain` plan in cortex-exec (router.py prepares brain reply context for any brain verb), not just chat. Whenever `ORION_CURIOSITY_FRAME_ENABLED` is on it called `populate_current_turn_llm_signals(ctx)`, which sends the probe whenever `ctx["user_message"]` is non-empty -- and autonomous verbs carry their prompt there. The module docstring already said it was meant for "every real chat turn"; nothing enforced it.

## Architecture touched

Which signal decides "real human turn" (reused, no new flag):

1. `stance_inputs["utterance_origin"]`, set by the Hub's unified turn (`orion/hub/turn_orchestrator.py::execute_unified_turn`). Only the two human entry points pass `"juniper"` (websocket chat, `turn_orchestrator.py:1970`; HTTP `/api/chat`, `api_routes.py:3296`). Curiosity passes `"orion"`; endogenous outreach, autonomous reading and collapse-mirror replies pass nothing. It reaches cortex-exec inside the stance_react request context (`services/orion-thought/app/bus_listener.py::build_stance_react_context`).
2. For the legacy Hub -> cortex-gateway -> orch chat path (predates utterance_origin): the plan verb (`ctx["verb"]`, set by router.py) is one of the verbs cortex-orch already classifies as interactive chat (`chat_general` + `FAST_SINGLE_PASS_CHAT_VERBS`, mirroring `execution_lanes.py::resolve_execution_lane`), unless `policy_dispatch_only` is set -- the flag Orion's own dispatchers (orion-actions scheduler, cortex-orch workflows, durable runs, journaler, capability bridge) stamp.

Everything else is treated as Orion's own work.

## Files changed

- `services/orion-cortex-exec/app/current_turn_llm_signals.py`: `human_chat_turn_reason(ctx)` and `mark_current_turn_llm_skipped(ctx, reason)`.
- `services/orion-cortex-exec/app/chat_stance.py`: call site skips the probe on non-human turns and logs at INFO.
- `orion/substrate/attention_frame.py`: `debug.turn_read_skipped` so a skipped read is distinguishable from a failed one in the frame.
- `services/orion-cortex-exec/tests/test_current_turn_llm_signals.py`: gate truth table, skipped-state contract, policy still fails closed.
- `services/orion-cortex-exec/tests/test_attention_frame_integration.py`: existing tests now use a real human-turn ctx; new tests that a real chat turn calls the probe and that six autonomous ctx shapes never do.

## Schema / bus / API changes

- Added: optional `skipped` / `skip_reason` keys in the in-process `ctx["current_turn_llm_read"]` dict; `turn_read_skipped` in `AttentionFrameV1.debug` (free-form `dict[str, Any]`).
- Removed: none
- Renamed: none
- Behavior changed: probe not sent on non-human turns.
- Compatibility notes: `ok` stays `False` on skip, so `orion.substrate.attention.policy.direct_answer_cause` returns `"unavailable"` and fails closed exactly as it did on the dominant timeout outcome for these turns today.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-cortex-exec && PYTHONPATH=<worktree> pytest tests/test_current_turn_llm_signals.py tests/test_attention_frame_integration.py tests/test_attention_frame.py -q
-> 82 passed

Root/orion attention tests touching attention_frame/policy (per file): all pass except
tests/test_execution_dispatch_runtime_worker.py (39 failed) -- identical 39 failures on main, pre-existing.

Full cortex-exec suite cannot run as one process (pre-existing collection conflict:
"Verb already registered: legacy.plan"); ran with --continue-on-collection-errors on branch and main
and re-ran every branch-only failure in isolation: all pass except
tests/test_chat_stance_brief.py::test_build_chat_stance_inputs_falls_back_when_identity_missing,
which also fails in this worktree with the patch stashed (env-sensitive identity path) and passes
on the main checkout -- not caused by this patch.
```

## Evals run

```text
None run. services/orion-cortex-exec/evals/run_current_turn_signal_eval.py and
run_current_turn_disclosure_live_eval.py exercise the probe's prompt/parse on real chat text,
which this patch does not touch. The behavior change (gate) is covered by the deterministic tests above.
```

## Docker/build/smoke checks

```text
Not run: deploy/restart is out of scope for this task. Live after-deploy check is the SQL above.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. No blocking findings. Mutation check by the reviewer: forcing the gate to always-human makes 6 integration tests fail, so the tests catch the regression.

- Finding (should-fix): collapse-mirror replies are human-authored but were described in the docstring as "Orion's own work".
  - Fix: docstring corrected -- they are skipped on purpose (a form submission framed into a prompt, not a chat message); regression case added. Not tagging them `juniper`, because that also flips `record_user_turn` and the Mind origin note in the Hub -- a separate decision (see Risks).
  - Evidence: `test_autonomous_turn_never_calls_probe` collapse-mirror case passes.
- Finding (should-fix): legacy-lane human turns whose plan verb is rewritten away from a chat verb (Hub auto-route depth 1/2, single-verb override) now skip.
  - Fix: documented in the gate docstring and Risks. Low live impact: `HUB_AUTO_DEFAULT_ENABLED=false`, and Hub agent mode now goes through the unified turn.
  - Evidence: reviewer traced `decision_router.py:381-392`, `chat_request_builder.py:75-86`.
- Finding (nit): the harness finalize leg of a Juniper turn now always reads "unavailable".
  - Fix: documented; finalize prompt does not read the frame; frames carry `debug.turn_read_skipped` so they are not mistaken for probe failures. Regression case added.
- Finding (nit, latent): AI Town speech (`EMBODIMENT_SPEECH_UNIFIED_ENABLED`, false live) dispatches `chat_general` without `policy_dispatch_only` and would still fire the probe if turned on.
  - Fix: not changed here (adding `policy_dispatch_only` alters embodiment routing); listed as follow-up in Risks.
- Finding (nit): `human_chat_turn_reason` was computed outside the call site's `try`.
  - Fix: moved inside the `try`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: collapse-mirror replies (Juniper writes a collapse-mirror entry, Orion replies via the unified turn) carry no `utterance_origin`, so they now skip the probe. They are human-authored but not a chat message.
  - Mitigation: if wanted, pass `utterance_origin="juniper"` in `services/orion-hub/scripts/collapse_mirror_chat_reply.py` -- one-line follow-up, no cortex-exec change.
- Severity: low
  - Concern: a future human entry point that neither sets `utterance_origin="juniper"` nor uses a chat entry verb would silently lose the probe.
  - Mitigation: the skip is logged at INFO with the reason, and the frame debug shows `turn_read_skipped`.

- Severity: low
  - Concern: legacy-lane Hub turns that the router re-verbs (auto-route, single-verb override) skip the probe and fail closed.
  - Mitigation: low live use (`HUB_AUTO_DEFAULT_ENABLED=false`); a Hub-stamped human marker in `options` would cover them if needed.
- Severity: low (latent)
  - Concern: AI Town speech via `chat_general` would fire the probe if `EMBODIMENT_SPEECH_UNIFIED_ENABLED` is turned on.
  - Mitigation: follow-up -- mark that request as Orion-originated before enabling the flag.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2465

🤖 Generated with [Claude Code](https://claude.com/claude-code)
