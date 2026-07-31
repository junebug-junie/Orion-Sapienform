# Recall: tag and correctly label ai-town-sourced chat memories

Status: implemented (2026-07-31, Juniper: "if those memories just come in raw
without any tag to differentiate them, orion will take it as our history...
i think we need to add the tag for the recall/memories that go into the chat
AND update the unified turn so orion will be on the lookout for those tags").

## Arsonist summary

The 2026-07-30 identity-bleed fix (PR #1526) added a prompt-level
`CURRENT CONTEXT` instruction telling the model to treat ai-town content in
`memory_digest` as a separate context. That's a mitigation on top of a
genuinely broken data pipeline, not a fix to the actual cause: traced to
`services/orion-recall/app/storage/sql_adapter.py`'s `fetch_sql_fragments`,
the SQL query that builds chat-history memory fragments never selects
`client_meta` at all, tags every single chat-history row with the identical,
non-differentiating `["dialogue"]`, and -- worse -- unconditionally renders
every row's text as `f"User: {prompt}\nOrion: {response}"`, regardless of
source. For an ai-town row, `prompt` is an NPC's line, not Juniper's --
labeling it "User:" is a factually false claim baked directly into what
reaches the model. Confirmed live: a hub reply about "tummy troubles" bled
into an ai-town-sourced row and vice versa -- this is bidirectional, not just
ai-town-into-hub.

`client_meta.external_room.platform`/`external_participant.participant_name`
(written by `services/orion-embodiment`'s conversation-memory feature, see
`docs/superpowers/specs/2026-07-30-aitown-social-memory-room-scope-design.md`
and neighboring specs) already carries exactly the distinguishing information
needed -- confirmed live, reliably populated on every recent aitown row.
Nothing currently reads it in the recall path.

## Current architecture

- `services/orion-recall/app/storage/sql_adapter.py:fetch_sql_fragments`:
  selects `trace_id, prompt, response, created_at` from `chat_history_log`
  only. Builds `Fragment(tags=["dialogue"], text=f"User: {prompt}\nOrion:
  {response}", meta={})` for every row.
- `services/orion-recall/app/service.py`: maps `Fragment.tags`/`.text`
  straight through to `MemoryItemV1.tags`/`.snippet` with no filtering or
  transformation -- `MemoryItemV1` already has a `tags: List[str]` field,
  unused for this purpose today.
- `services/orion-recall/app/render.py:render_items`: renders each item's
  `snippet` (already-clamped/truncated text) into the final `memory_digest`
  bullet list. Operates on whatever text it's given -- a tag/label baked
  into the snippet text survives this step automatically; nothing here
  needs to change.
- `orion/cognition/prompts/chat_quick.j2` / `chat_general.j2`: already have
  a `CURRENT CONTEXT` block (PR #1526) telling the model how to treat
  ai-town content *if it recognizes it as such* -- but nothing before now
  told the model definitively which lines those are.

## Missing questions

None outstanding -- root cause confirmed via live query, fix direction
confirmed with Juniper.

## Proposed schema / API changes

No new schema -- `MemoryItemV1.tags`/`Fragment.tags` already exist and are
already threaded through untouched. This is a values/content fix, not a
shape fix.

`fetch_sql_fragments`'s chat-history branch:
- SELECT gains `client_meta`.
- Parse `client_meta.external_room.platform` and
  `client_meta.external_participant.participant_name`.
- When platform == "aitown": `tags=["dialogue", "aitown"]`,
  `text=f"[ai-town] {participant_name or 'NPC'}: {prompt}\nOrion (in
  ai-town): {response}"`, `meta={"platform": "aitown", "participant_name":
  participant_name}`.
- Otherwise (hub/direct, or client_meta absent -- older rows predate this
  field): unchanged, `tags=["dialogue"]`,
  `text=f"User: {prompt}\nOrion: {response}"`, `meta={}`.

`orion/cognition/prompts/chat_quick.j2` / `chat_general.j2`: extend the
existing `CURRENT CONTEXT` block with a concrete instruction to look for the
literal `[ai-town]` marker in `memory_digest`/`message_history` lines and
treat those specifically per the existing "separate place" guidance --
replacing "guess whether this looks like ai-town" with "here is the exact
marker."

## Files likely to touch

- `services/orion-recall/app/storage/sql_adapter.py`: the actual fix.
- `services/orion-recall/tests/`: new tests covering both branches
  (aitown-tagged vs. untagged) of the chat-fragment query -- mocked cursor,
  no live DB needed, matching this service's existing test conventions.
- `orion/cognition/prompts/chat_quick.j2`, `chat_general.j2`: reference the
  concrete `[ai-town]` marker.
- `services/orion-cortex-exec/tests/test_chat_prompt_context_guardrails.py`:
  extend with a render test for the new marker-lookup instruction.

## Non-goals

- Not backfilling/relabeling existing already-recalled rows -- this only
  changes labeling for rows recalled *after* this ships; old rows already in
  `chat_history_log` get the fix applied retroactively for free the next
  time they're recalled (the fix is in the read path, not a data migration).
- Not touching `orion-recall`'s vector/RDF backends (`fetch_rdf_fragments`,
  any vector-store fragment path) -- scoped to the SQL chat-history path
  specifically, since that's the confirmed, demonstrated source of the bleed.
- Not adding a general-purpose "source platform" taxonomy for every possible
  future platform -- just the one real, confirmed distinction (aitown vs.
  not) that's actually driving errors today. More platforms can extend this
  pattern later if they ever need the same treatment.

## Acceptance checks

- A fresh chat-history memory recall for an ai-town-sourced row renders as
  `[ai-town] <ParticipantName>: ...` in `memory_digest`, not `User: ...`.
- A hub-sourced row's rendering is byte-identical to before this change.
- `MemoryItemV1.tags` contains `"aitown"` for ai-town-sourced items,
  inspectable independent of the text content.
- chat_quick.j2/chat_general.j2 explicitly reference the `[ai-town]` marker
  as the thing to look for.

## Recommended next patch

Implement as scoped above in a single PR on
`fix/recall-aitown-source-tagging`
(`/mnt/scripts/Orion-Sapienform-recall-aitown-source-tagging`).
