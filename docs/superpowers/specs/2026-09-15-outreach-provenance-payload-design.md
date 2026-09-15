# Outreach provenance payload — design

Date: 2026-09-15  
Status: approved for planning (Juniper)  
Incident seed: `correlation_id=bfda9f83-385d-4d6d-9b32-c16b8eb8f52d` (unsolicited “crystallization gate / atmospheric content” outreach; follow-up confabulated collapse-mirror)

## Arsonist summary

Endogenous outreach already knows *why* it is speaking (full `build_outreach_prompt` + lane grounding). At deliver time that context is dropped: chat history only gets `client_meta.unsolicited=true`, and `endogenous_outreach_decisions` only stores lane booleans/counts — not the prompt. Orion cannot later answer “where did that come from?” honestly, and Hub has nothing to show under the bubble.

Ship one `outreach_provenance.v1` capsule: full generation prompt + lane summary, written to **both** chat `client_meta` and the decision row, pushed on the live WS payload, rendered as a collapsible “why I spoke” under outreach bubbles, and reinjected into the next unified-turn context when the previous assistant message was outreach.

## Current architecture

- **Producer:** `services/orion-hub/scripts/endogenous_outreach.py`
  - Builds `OutreachContext`, `build_outreach_prompt(ctx)`, generates via agent lane or chat fallback.
  - `grounding_summary(ctx)` → `endogenous_outreach_decisions.result_json.grounding` (booleans/counts only; prompt text deliberately omitted today).
  - `_deliver` → sockets (`kind=orion_outreach`) + `_publish_history` (`client_meta={"unsolicited": true}`) + notify.
- **Chat persistence:** `scripts/chat_history.py` `build_chat_history_envelope(..., client_meta=...)`.
- **Hub UI:** `services/orion-hub/static/js/app.js` handles `d.kind === 'orion_outreach'` via `appendMessage(..., { unsolicited: true })` — no provenance UI.
- **Follow-up turns:** `orion/hub/turn_orchestrator.py` unified turn does not read outreach `client_meta` into the prompt. Orion invents a frame when asked.

## Decisions (locked)

| Question | Choice |
|---|---|
| Who needs the pointer? | Both Orion and Juniper (UI) |
| How rich? | Full generation prompt snapshot |
| Privacy / retention hedge? | None — sole user; purposeful share |
| Where stored? | Both chat turn metadata and decision row |
| Hub presentation? | Collapsible “why I spoke”: `summary_line` collapsed, full `prompt_text` expanded |
| Approach | Capsule write + follow-up injection (not metadata-only, not ID-only lookup) |

## Proposed schema / API changes

### `outreach_provenance.v1` (JSON object)

```json
{
  "schema": "outreach_provenance.v1",
  "decision_id": "<uuid or decision row id>",
  "correlation_id": "<outreach corr>",
  "generated_at": "<ISO-8601>",
  "lanes": {
    "daydream": false,
    "daydream_age_sec": null,
    "curiosity_summaries": 2,
    "recent_turns": 3,
    "tension": false,
    "chat_presence": false,
    "embodied_presence": false,
    "priors_count": 3
  },
  "prompt_text": "<exact build_outreach_prompt output>",
  "summary_line": "<one short human line for collapsed UI>"
}
```

- `lanes` matches today’s `grounding_summary(ctx)` shape (extend if new lanes appear).
- `summary_line` is derived deterministically from lanes (e.g. “Open priors (3) + recent turns (3); no tension/daydream”) — not an LLM rewrite.
- `unsolicited: true` remains a sibling key on `client_meta`.

### Write surfaces

1. `chat_message` / chat history envelope: `client_meta.outreach_provenance = <capsule>`
2. `endogenous_outreach_decisions.result_json.provenance = <capsule>` (keep existing `grounding` for backward-compatible queries)
3. Live WS outreach payload: same capsule field so UI works before rehydrate

No new bus channel required for v1. No schema registry event unless we later publish provenance as a first-class bus artifact (non-goal).

## Data flow

```text
OutreachContext
  -> build_outreach_prompt(ctx)           # already exists
  -> build_outreach_provenance(...)       # NEW thin builder
  -> generate reply
  -> on send:
       decisions.result_json.provenance
       client_meta.outreach_provenance
       WS payload.outreach_provenance
  -> Hub: collapsible "why I spoke"
  -> later human turn:
       if latest assistant has outreach_provenance:
         inject into unified-turn / stance context
```

### Follow-up injection (consume path)

When starting a unified turn in the same session, if the most recent assistant history item has `client_meta.outreach_provenance`:

- Inject a fail-open context block that includes the stored `prompt_text` (and lanes/summary).
- Instruct Orion to answer “where did that come from?” from this block — do not invent collapse-mirror or other frames.
- Only the latest outreach in-session; missing/malformed provenance → no block.

Exact insertion point: prefer Hub `turn_orchestrator` / situation-or-history assembly before harness, so both stance and motor see it. Pin the choke point in the implementation plan after a one-file inspect of where recent assistant text is already folded in.

## Files likely to touch

- `services/orion-hub/scripts/endogenous_outreach.py` — build capsule; thread through `_deliver` / `_publish_history` / `_push_to_sockets` / decision write
- `services/orion-hub/scripts/chat_history.py` — only if envelope helpers need a typed pass-through (likely already fine)
- `orion/hub/turn_orchestrator.py` (and/or adjacent history assembly) — follow-up injection
- `services/orion-hub/static/js/app.js` — outreach bubble + rehydrate “why I spoke” UI
- `services/orion-hub/tests/test_endogenous_outreach.py` — deliver/meta/decision coverage
- New or extended test for follow-up injection
- Light UI/smoke or JS-adjacent assertion if the repo already has a pattern; otherwise a small DOM fixture test if one exists

## Hub UI

- Bubble body unchanged.
- Under outreach messages with provenance: collapsed control labeled **why I spoke**, showing `summary_line`.
- Expand: full `prompt_text` in a scrollable mono block.
- Messages without provenance (legacy): no control.
- Live WS and history rehydrate both populate the same control from the capsule.

## Non-goals

- Stopping outreach self-echo (re-chewing prior unsolicited lines). Separate follow-up.
- Changing fire gates (priors/tension/cooldown/caps).
- Collapse-mirror, curiosity, or other non-endogenous delivery paths.
- New bus channel / schema registry entry for v1.
- Spoken verbal “I am reaching out because…” forced into the bubble text (payload + UI + follow-up injection only).

## Acceptance checks

1. Forced or live endogenous outreach → Hub bubble shows collapsible “why I spoke” with the real generation prompt.
2. Postgres: both `chat_message` client_meta (or equivalent metadata path) and `endogenous_outreach_decisions.result_json` contain the same capsule for that `correlation_id`.
3. Ask “where did that come from?” on the next turn → Orion cites the stored prompt/lanes; does not invent collapse-mirror when provenance is present.
4. Legacy outreach rows without provenance still render; follow-up without provenance behaves as today (fail-open).
5. Collapse-mirror and solicited delivers do not get `outreach_provenance` or `unsolicited=true`.

## Risks / concerns

- **Prompt size in chat metadata:** full prompts can be large; acceptable per Juniper; watch WS payload size and JSONB row growth.
- **Rehydrate must preserve `client_meta`:** if rehydrate strips nested keys, UI and follow-up injection break after refresh — test that path.
- **Injection vs prompt budget:** full prior outreach prompt on every follow-up could be heavy; v1 injects only when the *immediately previous* assistant message was outreach (not every historical outreach).

## Recommended next patch

1. Worktree `docs/outreach-provenance-payload` → implementation plan via writing-plans.
2. Thin builder + deliver write path + tests first (observable in DB + WS).
3. Follow-up injection second (fixes the “huh?” confabulation).
4. Hub collapsible UI third (same capsule, no new backend).

## Related

- Incident: `bfda9f83-385d-4d6d-9b32-c16b8eb8f52d` (2026-09-15), session `orion_journal`, four unsolicited sends; first gate riff `a1847cef-…`.
- Prior art: `grounding_summary` / `docs/superpowers/pr-reports/2026-08-28-outreach-grounding-trace-pr.md` (lane booleans only).
- Tension outreach design: `docs/superpowers/specs/2026-08-16-tension-driven-outreach-design.md`.
