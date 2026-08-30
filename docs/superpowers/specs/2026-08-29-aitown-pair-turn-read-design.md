# AI Town: pair-turn read path (speaker-grounded)

Status: design — approved 2026-08-29. Juniper locked B: raw turns, town facts from **that speaker's other threads** only (not room omniscience).
Date: 2026-08-29

## Arsonist summary

NPC continue injects `/summary` topic bags (or nothing, after we drop the prefixes). The facts already sit in `social_room_turns`. Read those turns.

Keep the identity sheet. Stop calling the plan a quest. Do not add don’t-lists.

Town context is not “what happened in the room.” That is divination: Nico would recite Sofia↔Cam. Town context is **what this speaker was present for** in other pair threads.

## Current architecture

Write path (already shipped): Convex `POST /ingest-turn` → `social.turn.v1` → `social_room_turns`, keyed

```text
platform   = aitown
room_id    = aitown-town
thread_id  = "{slug_a}--{slug_b}"   # sorted, from orion/town_cast.py
```

Read path today: `GET /summary?participant_id={other}` on first continue, 400 characters of

- `recent_thread_summary` = `Recent room themes: {word-count topics}`
- `safe_continuity_summary` = `Recurring {kind} in {room}; recent shared topics include {topics}`
- `active_threads[].thread_summary` = `{name} · → {other} · {topics}`, 6-hour TTL, last 3 room threads

`extract_topics` is a stopworded word-count of prompt+response. After the Nico riddle chat that produced `secrets, elevator`.

This chat: `previousMessages` already on the LLM list.
Who they are: `compose_identity` essay + `plan` as `Your goals for the conversation`.
Same Postgres (`conjourney`) as sql-writer. No second store.

NPC ingest stores one row per NPC line: `prompt` = the other person's last line, `response` = this speaker's line, `external_participant` = the NPC. Human lines are not separate rows. Both sides of a pair must be reconstructed from those two fields.

## Locked decisions

| Decision | Choice |
|---|---|
| Approach | B — raw turns from `social_room_turns`, not `/summary` |
| Pair history | this `thread_id` only |
| Town facts | **speaker's other threads only** — both sides of those chats |
| Divination | forbidden. Sofia↔Cam never enters Nico's prompt |
| Identity | keep. Do not starve the sheet |
| `/summary` on NPC speech | remove |
| Embodiment `/summary` | unchanged |
| `LLM_MODEL` | `quick_background` |
| Fetch cadence | first continue / start only (`priorMessages.length <= 1`) |
| Fail-open | 3s abort, empty lists, speak as today |
| Keyword / don’t-lists | none |
| One stored row | two utterances when both `prompt` and `response` are non-empty (other, then speaker) |
| Caps | 8 pair utterances, 4 town utterances, oldest-first after sort |

## Read contract

```text
GET /town-continuity
  ?platform=aitown
  &room_id=aitown-town
  &thread_id=juniper-feld--nico-sable
  &speaker_id=nico-sable
```

`thread_id` is the pair talking **now**. `speaker_id` is the NPC about to speak (Convex `player.name` → town slug). Both required. 400 if either missing. `platform` and `room_id` also required (same as `/summary`).

### Pair turns

`client_meta.external_room.thread_id == thread_id`

Oldest-first, cap **8 utterances**. Both speakers. This is “related convos with this person.”

### Town turns (speaker-grounded)

Turns where **all** of:

1. same `platform` + `room_id`
2. `tags` contains `aitown`
3. `thread_id !=` the current pair
4. speaker is a participant on that thread: `thread_id` is `{a}--{b}` and `speaker_id` is `a` or `b`
5. `redaction.recall_safe` is not false

Oldest-first among the selected utterances, cap **4**.

Include both sides of those threads. Nico talking to Sofia, and Sofia answering Nico, are things Nico heard. Cam talking to Sofia is not.

Do not use room-wide “other threads.” Do not use `/summary` `active_threads`. Do not infer from topic overlap.

Slug match is exact on the two `--`-split parts. No substring `LIKE` on the raw string (avoids accidental collisions).

### Each utterance in the response

```text
{ speaker, other, text, thread_id, created_at }
```

`text` is one spoken line, clamp 160 characters.

From one stored row whose `external_participant` is the row speaker:

- `prompt` (if non-empty) → utterance by the **other** slug on that thread
- `response` (if non-empty) → utterance by the row speaker

Resolve slug → display name via the inverse of `orion.town_cast.TOWN_PARTICIPANT_SLUGS`. Unknown slug: use the slug itself. No `client_meta`, no journals, no mirrors, no topic-bag templates.

### Response schema

`TownContinuityReadV1`: `{ thread_id, speaker_id, pair_turns, town_turns }`
`TownContinuityTurnV1`: the row above.

Register both. Empty lists are valid.

## Convex inject (first continue / start)

Identity stays. Do not inject `/summary`. Do not label this as goals.

```text
Earlier with them:
Juniper: spill the tea
Nico: the pie sat out and the crumbs were sugar

From your other conversations:
Sofia: you still owe me for the trivia night
Nico: I'll settle it Friday
```

Copy only. No “you remember they like secrets.” Later continues keep this chat via `previousMessages`. Do not GET every line.

If both lists empty: omit both blocks.

`fetchTownContinuity(speakerName, otherName)` builds `thread_id` + `speaker_id` from `townThreadId` / `townSlug`. Fail-open on 4xx/5xx/timeout.

## Worked example

Nico speaking to Juniper (`speaker_id=nico-sable`, `thread_id=juniper-feld--nico-sable`).

| Source thread | In prompt? |
|---|---|
| `juniper-feld--nico-sable` | pair |
| `nico-sable--sofia-bell` | town (Nico was there) |
| `cam-lin--nico-sable` | town |
| `nico-sable--orion` | town (embodiment already ingests Orion↔Nico) |
| `cam-lin--sofia-bell` | **no** |
| `juniper-feld--sofia-bell` | **no** |

## Query

social-memory already has `DATABASE_URL` on `conjourney`. Read `social_room_turns` there. Do not add a Convex→sql-writer HTTP path. Do not import sql-writer models.

Filter in Python after a simple `SELECT` of candidate columns. Town volume is small; sequential scan is acceptable for v1. If the table is missing, return empty lists (fail-open). Optional same-patch index is **out of scope** for v1.

Do not invent if the JSON path is missing. Skip the row.

## Files likely to touch

- `orion/schemas/social_chat.py` + `orion/schemas/registry.py`
- `services/orion-social-memory/app/town_continuity.py` (new; pure filter)
- `services/orion-social-memory/app/main.py`, `service.py`, tests, README
- `services/orion-ai-town/patches/orion-town-continuity-ingest.patch`
- `services/orion-ai-town/tests/`, README

## Non-goals

- Synthesizer rewrite (topic bags may stay for Hub; they leave town speech)
- Second memory store
- Raising fast-worker ctx
- Wiping social-memory
- Banned-phrase lists / “don’t speak in riddles”
- Changing `LLM_MODEL`
- Starving `compose_identity`
- Room-omniscient town dump
- Private traces, journals, mirrors
- sql-writer index migration

## Acceptance checks

1. Two Juniper↔Nico chats, then a third: first continue contains a prior Juniper or Nico **line**, not `recent shared topics include`.
2. Fixture: Sofia↔Cam turns exist; Nico→Juniper continue must not contain those lines.
3. Fixture: Nico↔Sofia turns exist; Nico→Juniper continue may contain those lines under `From your other conversations`.
4. Empty table → empty lists → no extra prompt blocks.
5. Identity essay still present (unchanged `agentPrompts`).
6. Embodiment `GET /summary` unchanged.
7. Missing `thread_id` or `speaker_id` → 400 from the API; Convex fail-open (no inject).

## Recommended next patch

Schema + `GET /town-continuity` + Convex swap. One PR. Then talk to Nico.
