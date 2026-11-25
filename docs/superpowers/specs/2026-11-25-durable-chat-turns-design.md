# Durable chat turns — design

Status: proposal (cognition-loop change; needs Juniper's go before Patch 1).
Patch 0 (rebuild drain) ships with this doc.

## Arsonist summary

A chat message dies when Hub or the governor restarts, and nothing tries it again.
Orion already runs consults, intention carries and session round-ups as durable runs:
they wait for the GPU, retry, survive restarts, and a killed attempt resumes its
`claude` session from the saved transcript. Chat is the one Orion turn that does none of
this. Moving chat onto the same graph is mostly wiring. The real work is in three places:
chat's writes must happen once per message under retries, the reply must reach a browser
that reconnected, and ordinary chat must not queue behind curiosity.

## Current architecture

- Browser → Hub websocket (`services/orion-hub/scripts/websocket_handler.py`) →
  `execute_unified_turn` (`orion/hub/turn_orchestrator.py:2297`): stance, recall, prompt,
  then `HarnessGovernorClient.run` over the bus → orion-harness-governor runs the FCC motor
  and the 5a/5b/5c finalize → Hub persists (`chat_history_log` upsert keyed by turn id,
  chat-turn envelope, promise ledger, mood, TTS) and sends the reply on the websocket.
- Every chat message starts a fresh `claude` session; only the step notes of the *previous
  finished* turn carry forward (`orion/harness/working_notes.py`).
- Durable Orion turns: Hub submits a `DurableRunRequestV1` straight to durable-runs over
  HTTP (`consult_loop.py:573`). durable-runs runs the one-step harness graph
  (`services/orion-durable-runs/app/harness_smoke_graph.py:build_single_step_harness_graph`),
  which asks Hub to run the turn on `orion:curiosity:turn:request` with the run's GPU hold.
  Hub runs it with `no_write`. A retry resumes the attempt's `claude` session
  (`orion/harness/session_resume.py`, `resume_key=<run_id>:<source_tag>`); a cut-short attempt's
  findings go into the retry prompt (`cut_short_text`).
- cortex-orch has a durable front door (`services/orion-cortex-orch/app/durable_runs.py`) that
  records the start and labels the run `durable:<workflow>`. Consults and round-ups skip it,
  so cortex does not see them.

## Missing questions

1. Latency budget: how much extra wait is acceptable before a chat turn starts (durable
   submit + GPU admission)? Proposed: chat uses `priority="interactive"` on the chat lane and
   must add < 2 s p50 to time-to-first-step. Measured in Patch 1 before cutover.
2. Agent-mode chat (Mode=Agent, Compute=Agent): same workflow, agent lane. Yes, by lane only.
3. What does the browser show during a retry? Proposed: the existing progress frame with
   "retry 2 of 3 (governor restarted)".

## Proposed schema / API changes

- `DurableWorkflowV1` += `"chat.orion_turn"` (additive, consumer-first: sql-writer, hub,
  actions, cortex-orch, then durable-runs; Hub submits last behind a flag).
- New `orion/schemas/chat_turn_run.py`: `ChatOrionTurnBriefV1` (extra=forbid, literal
  discriminator `chat: "chat.orion_turn"`): `chat_correlation_id` (the message's own id, stable
  across attempts), `session_id` (Juniper's Hub session), `message`, `mode`, `fcc_model_label`,
  `client_meta`, `attachments` refs, `timeout_sec`. Run id is deterministic:
  `chat-<chat_correlation_id>`.
- Submission goes through cortex-orch's durable front door, so cortex records it as
  `durable:chat.orion_turn`. Hub retries the submit if cortex-orch is restarting (the
  submit is idempotent: same run id, same bytes).
- `CuriosityTurnRequestV1` (the Hub turn RPC durable-runs already uses) gains
  `persist_as: Literal["chat"] | None`. When set, Hub runs the full chat turn (writes on, not
  `no_write`) and persists under `chat_correlation_id`, never the per-attempt id. Every write
  is keyed on that id: `chat_history_log` already upserts on it; the chat-turn envelope,
  promise-ledger line, mood, and TTS dispatch each get a "done for this id" check.
- Delivery: Hub writes the final reply to `chat_history_log` (already) and pushes it to any
  websocket on that session; on reconnect, `chat_history_rehydrate.py` shows a pending turn
  from the run's state (`GET /runs/{run_id}`) instead of nothing.
- Steps on retry: `resume_key=chat-<id>:chat`, so a killed attempt resumes its transcript
  (up to `DEFAULT_MAX_RESUMES`=3, refused at the context ceiling). `retry_cut_short=True`.

## Files likely to touch

- `orion/schemas/durable_run.py`, `orion/schemas/chat_turn_run.py` (new), `orion/schemas/registry.py`
- `services/orion-durable-runs/app/harness_smoke_graph.py`, `admission_runtime.py`, `run_acts.py`
- `services/orion-hub/scripts/websocket_handler.py`, `curiosity_investigation.py` (turn RPC
  handler), `chat_history_rehydrate.py`, `orion/hub/turn_orchestrator.py` (persist keyed on chat id)
- `services/orion-cortex-orch/app/durable_runs.py` (no change expected; verify ingress accepts it)
- `config/integration/organs.yaml` (chat is already the main organ; new workflow value → seam check)
- `services/orion-hub/.env_example`: `HUB_CHAT_DURABLE_ENABLED=true` (ship on, per standing rule)

## Non-goals

- Moving consults / round-ups to the cortex front door (separate ticket).
- Changing the finalize chain, stance, or recall.
- Exactly-once delivery to the browser: at-least-once with an idempotent row is the bar.

## Acceptance checks

- Gate tests: brief validation; deterministic run id; a retried attempt persists one
  `chat_history_log` row and one chat-turn envelope; `persist_as` absent keeps `no_write`.
- Live smoke 1: send a chat, `docker restart` the governor mid-turn → the reply arrives; the
  run's state rows show `retrying` then `completed`; the final frame shows `resumed=true`.
- Live smoke 2: same with Hub restarted → the reply shows after the page reconnects; one row.
- Live smoke 3: cortex-orch log shows `verb=durable:chat.orion_turn accepted` for the message.
- Latency: time-to-first-step p50 delta < 2 s over 20 turns vs. today.

## Recommended next patch

Patch 1: schema + durable-runs graph + Hub `persist_as` path behind
`HUB_CHAT_DURABLE_ENABLED`, with the idempotent-write tests. Patch 2: browser delivery on
reconnect and the retry progress frame, then cutover.

## Patch 0 (shipped with this doc)

`scripts/rebuild_services_from_git_diff.sh` builds orion-hub and orion-harness-governor
first, then waits (max `ORION_REBUILD_CHAT_DRAIN_WAIT_S`, default 600 s) until Hub's
`/api/runtime-activity` shows no `source == "chat"` turn running or queued, then
`up -d --no-build`. It never fails the run: a timeout or an unreachable Hub logs a
`CHAT_DRAIN` line and restarts anyway. Kill switch `ORION_REBUILD_CHAT_DRAIN=0`.
Gap it leaves: a turn that starts in the seconds between the check and the restart, and
Hub's pre-handoff phase (stance/recall, seconds) are not covered. Patch 1 covers both.
