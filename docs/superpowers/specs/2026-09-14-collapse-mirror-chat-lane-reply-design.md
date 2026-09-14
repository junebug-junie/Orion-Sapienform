# Collapse Mirror → live chat-lane reply

**Date:** 2026-09-14  
**Status:** approved design (implementation not started)  
**Owner seam:** Juniper Collapse Mirror reply delivery

## Arsonist summary

Today, when Juniper submits a Collapse Mirror, Orion answers on a background metacog LLM route and ships the text through Notify as `orion.chat.message` (toast/notification). That bypasses chat-lane stance, FCC harness/governor, and the Hub chat transcript.

Replace that path: Actions only signals Hub; Hub injects a **You** bubble (the mirror) into the **live** Hub chat session, runs `execute_unified_turn` on the **chat lane only**, and delivers Orion’s reply the same way endogenous outreach already does. Live on merge — no feature flag. Rollback = revert the PR.

## Problem

Juniper wants Orion’s reply to a causally dense (and any Juniper) Collapse Mirror to feel like a real conversation: visible in the chat window, present in chat history for later prompts, and produced by the full unified-turn stack (stance + harness/gov), not a short `metacog_background` consolation.

## Current architecture (as of design date)

1. Juniper mirror lands on collapse rails; `orion-actions` subscribes (default `orion:collapse:triage`).
2. Gate: `should_trigger` → `observer == juniper` (case-insensitive). Dedupe on `event_id`.
3. Actions builds `cortex.orch.request` for verb `actions.respond_to_juniper_collapse_mirror.v1`.
4. Verb (`services/orion-cortex-exec/app/verb_adapters.py`):
   - Recall via profile `collapse_mirror.v1` (no vector; SQL ~48h; small RDF/cards).
   - Reflective self-study consumer context.
   - One LLM call: route `metacog_background`, Atlas metacog profile, max_tokens 512, custom `[INTROSPECT]`/`[MESSAGE]` prompt.
   - Deliver via `NotifyClient` / `orion.chat.message`.
5. Hub UI treats that as a notification/toast, **not** an `orion_outreach` chat bubble.
6. Endogenous outreach (`services/orion-hub/scripts/endogenous_outreach.py`) is the existing unprompted chat seam: `execute_unified_turn` + `_deliver` (sockets + history). It already tracks live websocket `session_id` via `note_session` / `_active_session_id`.

### What Orion sees today on that reply

| Input | Present? |
|---|---|
| Mirror fields (trigger, summary, what changed, observer state, mantra, tags, …) | yes |
| Light `collapse_mirror.v1` recall | yes |
| Self-study reflective render | yes |
| Normal chat continuity / stance brief / FCC harness | no |

## Decisions (locked)

| Decision | Choice |
|---|---|
| Approach | Hub owns generation + delivery; Actions only signals |
| You bubble | Yes — mirror text is a user turn in the live session (feeds chat history / prompt continuity) |
| Generation lane | Force **chat lane** only (`fcc_model_label` unset). No agent/FCC-first attempt |
| Session | **Live Hub session only** (newest connected websocket with a `session_id`). No fixed `collapse_mirror` session fallback |
| Old metacog+Notify reply | Retire as the live Juniper reply path in the same patch |
| Context reuse | Mirror text = user message. Stance/recall come from unified turn. Do **not** re-run light `collapse_mirror.v1` + custom introspect prompt unless a later gap proves need |
| Feature flag | **None.** Ships live on merge. Rollback = revert |
| No live session | Fail closed: skip generation; audit `skipped:no_live_session`; optional quiet notify that a reply was held back. No You bubble |

## Proposed architecture / data flow

```text
Juniper submits Collapse Mirror
  → existing collapse intake / store / triage
  → orion-actions (juniper gate + dedupe)
  → NEW bus event: “reply to this mirror in chat”
  → Hub handler (beside endogenous outreach)
       → require live session_id from connected clients
       → persist + push You bubble (mirror text)
       → execute_unified_turn (chat lane, source=collapse_mirror_reply)
       → _deliver Orion reply (socket + history; outreach-style)
```

1. Collapse rails unchanged for storage/triage.
2. Actions keeps Juniper gate + dedupe; **stops** dispatching cortex-orch `actions.respond_to_juniper_collapse_mirror.v1` for this reply.
3. Actions publishes a thin Hub-bound envelope (new schema + channel + registry entry in the same changeset).
4. Hub consumes it, resolves live session only, injects You, runs chat-lane unified turn, delivers Orion reply.
5. Old verb is deleted or disabled in the same patch (no silent second brain). Prefer remove over leave-dead.

## Schema / API / bus changes (expected)

- **Added:** one thin envelope for Actions → Hub collapse-mirror chat reply request (fields at minimum: `event_id`, `correlation_id`, mirror text or full `CollapseMirrorEntryV2`, observer). Exact name/channel chosen at implementation; must register in `orion/bus/channels.yaml` + `orion/schemas/registry.py`.
- **Removed/retired:** live use of `actions.respond_to_juniper_collapse_mirror.v1` + Notify `orion.chat.message` as the primary Juniper collapse reply.
- **Behavior changed:** Juniper collapse replies appear in Hub chat transcript and chat history logs.
- **Compatibility:** Metacog/Orion-authored mirrors remain out of this path (`observer != juniper`). Journaling / other collapse consumers unchanged unless they incorrectly depended on the reply verb (verify in implementation).

## Files likely to touch

- `services/orion-actions/app/main.py`, `logic.py`, settings/README/tests
- `services/orion-hub/scripts/endogenous_outreach.py` (reuse delivery/session helpers) and/or a sibling Hub module for collapse-mirror reply
- Hub websocket/session registration path if needed for shared connection map
- `orion/bus/channels.yaml`, `orion/schemas/*`, `orion/schemas/registry.py`
- `services/orion-cortex-exec/app/verb_adapters.py` + verb yaml/tests for retiring the old reply verb
- Focused tests under `services/orion-actions/tests`, `services/orion-hub/tests`

## Failure modes

| Situation | Behavior |
|---|---|
| Observer ≠ juniper | Actions skips (unchanged) |
| Duplicate `event_id` | Actions dedupe skips (unchanged) |
| No connected Hub client with `session_id` | Hub skips; audit `skipped:no_live_session`; optional quiet notify; no You, no Orion text |
| Unified turn deferred / degraded / empty / timeout / error | No fabricated reply. Audit failure. You bubble may already be present — leave it. Optional system/notify note |
| Hub down at publish time | Bus redelivery + Hub idempotency on `event_id` so retries do not double You bubbles |
| Multiple Hub tabs / sessions | Newest connected session with a `session_id` (same rule as outreach). No multi-session fanout |

## Non-goals

- Agent-lane / FCC-first generation or fallback chain for this path
- Fixed `ACTIONS_SESSION_ID` / `collapse_mirror` session fallback
- Reintroducing `[INTROSPECT]`/`[MESSAGE]` metacog prompt for Juniper replies
- Keyword / phrase detectors on mirror content
- Feature / kill-switch env flags
- Changing Orion-authored (metacog) collapse mirror pipeline

## Acceptance checks

### Gate tests

- Actions: juniper mirror publishes the new Hub event; non-Juniper skipped; dedupe holds; **no** cortex-orch call to the old reply verb.
- Hub: mocked live session → You history write + chat-lane `execute_unified_turn` (no `fcc_model_label`) + outreach-style deliver.
- Hub: no live session → skip + audit; no generation; no You bubble.
- Hub: idempotent on `event_id` (second delivery does not double You).
- Schema / channel / registry parity for the new envelope.

### Manual smoke after deploy

1. Open Hub chat; submit a Juniper Collapse Mirror.
2. Chat shows **You** (mirror) then **Orion** in that live session.
3. Evidence: stance / harness-gov / chat lane — not `metacog_background` Notify toast as the reply brain.
4. Close Hub tabs; submit another mirror → skipped/audited; no fake reply.

## Recommended next patch

1. Contract: schema + channel + registry for Actions→Hub reply request.
2. Actions: swap dispatch from old verb to new publish; update audits/tests.
3. Hub: consumer + You inject + chat-lane unified turn + deliver; idempotency; tests.
4. Retire old verb path + fix any callers/docs.
5. Deploy Actions + Hub; run manual smoke above.

## Privacy / cognition note

This changes how Orion socially responds to Juniper’s private-ish mirror entries: they become first-class chat turns in the live session history. That is intentional. Do not widen audience beyond the live session’s normal chat persistence rules.
