# AI Town conversation memory: verbatim recall + relationship synthesis

Status: implemented (2026-07-30, Juniper: "same as any other Orion-Juniper conversation" for
privacy handling, "both" for memory shape).

**Correction post-review (same day):** the first draft of this spec claimed
`orion:chat:social:turn` (`social.turn.v1`) itself lands in `chat_history_log`. That's wrong —
`orion-sql-writer` routes `social.turn.v1` to a *separate* table (`social_room_turns`, via
`SocialRoomTurnSQL`), which `orion-recall`'s verbatim-recall path deliberately excludes. Verbatim
recall actually requires the `orion:chat:history:turn` (`chat.history`) channel, which
`orion-sql-writer` genuinely routes to `ChatHistoryLogSQL`/`chat_history_log`. The implementation
below publishes to **both** channels per exchange (one correlation_id shared across both), which is
what actually delivers on "both" halves Juniper approved. Sections below are corrected to match.

## Arsonist summary

Orion's ai-town embodiment currently journals the *fact* that a conversation happened
(partner name + utterance count) but never the actual dialogue. The content is read
in-process to generate replies, then discarded. This patch wires two existing, already-live
pipelines to also carry ai-town conversation content, instead of building new infrastructure:

- **Verbatim recall**: reuse the existing `orion:chat:history:turn` (`chat.history`) pipeline that
  already feeds `chat_history_log` (via `orion-sql-writer`'s `ChatHistoryLogSQL` route) — the table
  `orion-recall` actually reads verbatim text from. `orion-embodiment` becomes a new producer on
  this one existing channel.
- **Relationship synthesis**: reuse the existing `orion:chat:social:turn` (`social.turn.v1`)
  pipeline, whose `social.turn.stored.v1` re-emission `orion-social-memory` already consumes to
  synthesize rolling per-participant relationship state. This is a *separate* table
  (`social_room_turns`) from `chat_history_log` — not a second verbatim copy, the synthesis half.
  `orion-embodiment` becomes a new producer here too.
- **Grounded journal**: `JournalTriggerV1.prompt_seed` already exists exactly for this — feed it
  the real transcript instead of leaving it unset, matching the precedent set by
  `build_discussion_window_journal_trigger` (`services/orion-actions/app/logic.py:111-123`), which
  already does the same thing for a different trigger source. `spawned_correlation_id` links the
  journal entry back to the shared correlation_id used on both bus publishes for the conversation's
  last exchange.

No new tables, no new top-level schemas. Two new producer registrations on two existing channels,
plus one field populated that a sibling producer already populates the same way.

## Current architecture

- `services/orion-embodiment/app/worker.py`'s `_perception_loop` (~line 621) drives, per tick:
  `_journal_from_perception` (conversation-completion journaling, gated on
  `EMBODIMENT_MEMORY_ENABLED`) and `_engage_conversation`/`_speak_once_safe`
  (~712-944, generates and injects Orion's replies via `messages:writeMessage`).
- `_journal_from_perception` (~398-445) builds a `JournalTriggerV1` with `trigger_kind="town_episode"`,
  `summary` = a templated one-liner from `orion/embodiment/salience.py`'s `_maybe_journal_episode`
  gate. `prompt_seed` is never set. This flows to `orion-actions`'s journaler
  (`orion/journaler/worker.py:364-397`, `build_compose_request`), which renders
  `orion/cognition/prompts/journal_compose_prompt.j2` — line 50 already interpolates
  `metadata.journal_trigger.prompt_seed` — into an LLM-drafted `JournalEntryWriteV1`, persisted by
  `orion-sql-writer` into `journal_entries` (no JSONB/raw-payload column — drafted prose only,
  by design; verbatim content lives elsewhere, see below).
- `orion/embodiment/perception.py` fetches ai-town's raw `human` field per player (present via
  `aitown_client.py:133-145`'s `list_players()` spread) but drops it while shaping `nearby_players`
  (~123-128) and `active_conversation.other` (~71-72) — so embodiment currently has no signal
  distinguishing an NPC partner from a human partner, even though ai-town already exposes it.
- `orion:chat:history:turn` (`chat.history`, `orion/bus/channels.yaml:976-982`) is currently
  produced only by `orion-hub` (`turn_orchestrator.py`'s `_publish_unified_turn_chat_history`, via
  `scripts/chat_history.py`). `orion-sql-writer` routes `chat.history` → `ChatHistoryLogSQL`
  (`app/settings.py:19`) → `chat_history_log`, the table `orion-recall` genuinely reads verbatim
  text from.
- `orion:chat:social:turn` (`social.turn.v1`, `orion/bus/channels.yaml:993-1000`) is currently
  produced only by `orion-hub` (`services/orion-hub/scripts/social_room.py:914-945`,
  `chat_history.py:585-595`), one event per single prompt/response exchange. `orion-sql-writer`
  routes `social.turn.v1` → `SocialRoomTurnSQL` (`app/settings.py:53`) → **`social_room_turns`**, a
  *separate* table from `chat_history_log` (confirmed: `orion-recall`'s verbatim path deliberately
  filters these rows out to avoid crowding chat.history recall). `orion-sql-writer` also re-emits
  `social.turn.stored.v1` on `orion:chat:social:stored`, which `orion-social-memory` consumes to
  synthesize rolling per-participant relationship state (`SocialParticipantContinuitySQL`, keyed on
  `platform:room_id:participant_id` — an `"aitown"` platform value cannot collide with any existing
  `"hub"`/`"callsyne"` relationship row). `client_meta.external_room`/`external_participant`
  (platform, room_id, participant_id, participant_kind) are free-form dict conventions, not
  schema-enforced — new values (`platform="aitown"`, `participant_kind="npc"`) need no schema
  change on the social-memory side.

**Privacy — explicitly not solved by anything today.** No `is_private`/`visibility` column exists
on `chat_history_log`; the nearest analog (`SocialRedactionScoreV1.recall_safe`/`redaction_level`,
computed in `orion-hub`'s `social_room.py:172-198`) is scored but **never consumed by any reader**
(confirmed by grep — zero downstream checks). So "same privacy treatment as any other
Orion-Juniper conversation" is, today, no enforced treatment at all. This patch does not invent
new protection that doesn't exist elsewhere — it tags `participant_kind="human"` consistently so a
human-participant town conversation is at least identifiable/filterable if privacy tooling is ever
added, matching (not exceeding) the current real state of the rest of the system.

## Missing questions

None outstanding — scope, privacy posture, and memory shape were confirmed with Juniper before
writing this spec. One open implementation judgment call, not blocking: whether to also wire
`list_agents()`/`world.agents[]` linkage for a stronger NPC-vs-human signal, or ship with the
already-available `human` field alone (presence/absence of `human` on a player already
distinguishes human-controlled from ai-town-agent-controlled players without needing the extra
agent-linkage call). Recommend shipping with `human` alone — it's already fetched, just currently
dropped, and is sufficient for the npc/human tag; add agent-linkage later only if a real need
for finer-grained agent identity surfaces.

## Proposed schema / API changes

- `orion/bus/channels.yaml`: add `orion-embodiment` to **both** `orion:chat:history:turn`'s and
  `orion:chat:social:turn`'s `producer_services` lists (each currently `["orion-hub"]` only). No
  new channels, no message-kind changes.
- `orion/schemas/embodiment.py` (`WorldPerceptionV1` and its nested player shapes): no schema change
  needed — `nearby_players`/`active_conversation` are already untyped `dict[str, Any]`, so
  `perception.py` can forward `is_human` as a new dict key without touching the schema.
- No changes to `SocialRoomTurnV1`, `ChatHistoryTurnV1`, `JournalTriggerV1`, `journal_entries`,
  `chat_history_log`, or `social_room_turns` schemas — all reused as-is with new field *values*,
  not new fields.

## Files likely to touch

- `orion/embodiment/perception.py` — forward `human`/`is_human` through nearby/active-conversation shaping
- `services/orion-embodiment/app/worker.py` — new method publishing both a `ChatHistoryTurnV1`
  (`orion:chat:history:turn`) and a `SocialRoomTurnV1` (`orion:chat:social:turn`) per Orion reply
  exchange, sharing one correlation_id (near `_inject_utterance`, ~905-944); extend
  `_maybe_journal_episode`/`_journal_from_perception` to populate `prompt_seed` with the
  accumulated transcript and `spawned_correlation_id` linking to that shared correlation_id
- `services/orion-embodiment/app/settings.py` — new feature flag
  `EMBODIMENT_CONVERSATION_MEMORY_ENABLED` gating the two new bus publishes, independent of
  `EMBODIMENT_MEMORY_ENABLED` (which still separately gates the journal-fact/prompt_seed half)
- `services/orion-embodiment/.env_example` **and** `docker-compose.yml`'s `environment:` list —
  both must carry the new key, not just `.env_example` (a compose-list miss was caught in review:
  the flag would otherwise never reach the container regardless of what's set on disk)
- `orion/bus/channels.yaml` — producer list update on both channels
- Tests: `services/orion-embodiment/tests/` — new coverage for both bus-turn event shapes,
  human/npc tagging, prompt_seed population, and the feature-flag gate
- `services/orion-embodiment/README.md` — document the new capability and its privacy caveat

## Non-goals

- Not building a new memory subsystem, table, or channel.
- Not touching ai-town's own NPC-native embeddings memory (`agent/memory.ts`) — separate system,
  out of scope, confirmed non-interoperating with Orion's embodiment.
- Not inventing new privacy enforcement (`recall_safe`/redaction consumption) — that's a
  system-wide gap bigger than this feature; flagging it, not fixing it here.
- Not wiring `list_agents()`/full agent-identity linkage unless the `human`-field-only signal
  proves insufficient.
- Not backfilling historical conversations — forward-looking only, from deploy time on.

## Acceptance checks

- A real town conversation between Orion and an NPC produces: (1) a `chat_history_log` row
  `orion-recall` can retrieve verbatim per exchange (via `orion:chat:history:turn`), (2) a
  `SocialParticipantContinuitySQL` row (or update) for that NPC under `platform="aitown"` (via
  `orion:chat:social:turn`), (3) a journal entry whose drafted body reflects real conversation
  content, not a template, with `spawned_correlation_id` cross-referencing the shared
  correlation_id used on both bus publishes for the conversation's last exchange.
- A conversation where a human player participates tags `participant_kind="human"` correctly on
  that participant's turns.
- Feature flag off restores exactly today's behavior (fact-only journaling, no social-turn publish).
- Existing `orion-hub`-sourced social-memory rows are unaffected (no key collision, confirmed via
  the `platform:room_id:participant_id` composite key).

## Recommended next patch

Implement as scoped above in a single PR (this spec + the code), on
`feat/aitown-conversation-memory` (worktree at
`/mnt/scripts/Orion-Sapienform-aitown-conversation-memory`).
