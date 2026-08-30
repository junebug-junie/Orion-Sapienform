# AI Town: cull four NPCs and give the remaining town CallsYne continuity

Status: design — approved direction from Juniper 2026-08-29
Date: 2026-08-29

Wipe is acceptable. Implementation does not start until Juniper reviews this file.

## Arsonist summary

Town dialogue is shitty for two independent reasons, both already visible in code:

1. **The cards never reach the model as people.** `compose_identity()` concatenates Orion-essays and signature lines and drops `role`, `daily_loop`, and `conversation_style`. Nico’s signature is literally about “better lighting.” A later prompt patch names `light, shadows, glow` as forbidden words. A small `quick_background` model recites the magnet.
2. **NPC speech has no access to the continuity store we already built.** `orion-social-memory` already treats `aitown-town` as one CallsYne room (live: 1235 turns, 4 speakers). Only Orion’s embodiment worker writes. `conversation.ts` never reads. NPC–NPC gossip dies in Convex. Conversations cannot grow.

The fix is not a second memory system. The town is one room. Pairs are sequential `N=2` threads inside that room — the thing social-memory was built for. Cull the four named characters, archive their cards, rewrite the remaining four as jobs, inject those jobs, and wire NPC lines onto the existing rail.

## Current architecture

### Cast

Source of truth: `services/orion-ai-town/cards/town_cards.yaml`.
Generator: `services/orion-ai-town/scripts/generate_descriptions.py`.
Live seed: `Descriptions` in `upstream/data/characters.ts` via `patches/orion-character.patch`.

Eight NPCs today: Mara Vale, Nico Sable, Dr. Elian Cross, Juno Park, Tessa Quinn, Vale Moreno, Sofia Bell, Cam Lin. Juniper is the human join. Orion joins externally.

`compose_identity()` currently emits: `public_description + deeper_bio + orion_dynamic + private_pressure + signature_line`.
It does **not** emit: `role`, `personality`, `daily_loop`, `conversation_style`, `memory_hooks`.
`plans:` are static Orion-centered goals (“test whether Orion has a self”).

### Two speech paths (do not conflate)

| Path | Code | Model | Memory today |
|---|---|---|---|
| NPC dialogue | `convex/agent/conversation.ts` → llm-gateway `quick_background` | circe-worker-fast-1 | AI Town embeddings + last conversation. No social-memory. |
| Orion speech | `orion-embodiment` → `chat_quick` | cortex quick lane | Writes `chat.history` + `social.turn.v1`. Reads `GET /summary`. |

NPC prompt patches already applied: short replies (`orion-town-chat-turns`), anti-repetition, concrete-grounding (the light-word list), proximity, cooldown.

### Social-memory (already the right primitive)

Keying, live and correct after 2026-07-30:

- Room: `platform=aitown`, `room_id=aitown-town` (stable venue, **not** `conversation_id`, **not** `world_id`).
- Peer: `aitown:aitown-town:<participant_id>`.
- Room row + participant rows + threads with `audience_scope` / `target_participant_id`.

This is the CallsYne N-person model. A pair talking is `N=2`. Four people pairing off over an afternoon is one room, sequential threads. Pairs do not need a different schema.

What is missing: NPC–NPC (and NPC–Juniper NPC-side) turns never publish. `participant_id` is Convex `p:12`, which dies on every wipe.

Embodiment publish contract to copy, not reinvent (`services/orion-embodiment/app/worker.py` `_publish_conversation_memory`):

```text
SocialRoomTurnV1
  source: <service>
  prompt: other person's line
  response: this speaker's line
  session_id: aitown:<conversation_id>          # verbatim grouping only
  tags: ["aitown"]
  client_meta.external_room: {platform: aitown, room_id: aitown-town}
  client_meta.external_participant: {id, name, kind}
```

Read-back already exists for Orion: `_fetch_participant_continuity` → `chat_quick.j2` `aitown_participant_continuity`. Fail-open. Do not rebuild.

## Locked decisions

| Decision | Choice |
|---|---|
| Dead | Juno Park, Tessa Quinn, Vale Moreno, Dr. Elian Cross |
| Live NPCs | Mara Vale, Nico Sable, Sofia Bell, Cam Lin |
| Dead cards | Archive, do not delete |
| World | Wipe + `init` + Orion re-bootstrap |
| Continuity store | Existing `orion-social-memory`, one room `aitown-town` |
| Pair scope | `thread_id` = sorted stable slugs (`cam-lin--sofia-bell`) |
| Parallel Convex dyad store | No |
| NPC speech HTTP-reading social-memory | Yes, **once per conversation start**, fail-open |
| Who publishes Orion↔anyone | Embodiment only (already does) |
| Who publishes NPC-generated lines when other is not Orion | New Convex → social-memory ingest |
| `participant_id` | Stable card slug (`sofia-bell`), not Convex `p:*` |

## Design

### 1. Cast cull + archive

Move the four YAML character blocks, their `sprites:` entries, and their `plans:` entries to:

```text
services/orion-ai-town/cards/archived/2026-08-29-retired-cast.yaml
```

`NPC_ORDER` becomes:

```python
NPC_ORDER = ["mara_vale", "nico_sable", "sofia_bell", "cam_lin"]
```

Keep Juniper and Orion in `town_cards.yaml`. Rewrite Orion’s `deeper_bio` / `town_presence` / `boundaries` so they do not name the dead four.

Generator, tests, README, and `patches/orion-character.patch` follow the new four. `init` seeds one agent per `Descriptions` entry (README: “seeds the 8 NPCs from Descriptions”). Four entries is the live cast. The unused sprite files in `data/characters.ts` may remain on disk; do not emit dummy `Descriptions` to keep eight slots.

### 2. Remaining cards become jobs

Rewrite Mara / Nico / Sofia / Cam so the prompt spine is work, objects, and today’s loop — not Orion-as-topic.

Required fields after rewrite, each card:

- `role` — one line job
- `public_description` — physical place + objects (diner, maps, flyers, devices)
- `daily_loop` — 3–5 concrete tasks
- `conversation_style` — how they talk
- `orion_dynamic` — **one sentence**, not an essay
- `signature_line` — not a metaphor the model will loop. **Nico’s lighting line is gone.**

`compose_identity(card)` must emit, in this order, collapsed to one paragraph:

1. `"{name} is the town's {role}."`
2. `conversation_style`
3. `"Today: {daily_loop[0]}."` (the live `plan` is this same item; see below)
4. `public_description` (keep; this is the object list)
5. One sentence `private_pressure` if present
6. One sentence `orion_dynamic`

Do **not** emit `deeper_bio` or `signature_line` into the NPC identity blob. Signatures stay on the card for humans and for Juniper/Orion blurbs. They have already proven they hijack `quick_background`.

`plans[cid]` is generated from `daily_loop[0]` in second person, e.g. Sofia: `"You are running the diner: coffee, pie, who sat where, who is lying."` Not `"You teach Orion to read soft evidence."`

Tests that must change:

- `test_compose_identity_is_rich_and_collapsed` — stop requiring 400+ chars and Mara’s signature. Require `role` + a `daily_loop` object and forbid the retired lighting / glow / “description of your logs” catchphrases in any live identity string.
- `test_render_descriptions_emits_eight_valid_sprites` — four, not eight.

### 3. Kill the light magnet

`patches/orion-concrete-grounding-prompt.patch`: delete the clause that names `light, shadows, glow, echoes, silence`.

Replace with a positive contract, no banned-word list:

```text
Answer as your job. Name a person, object, or task from your role or from what they just said. If you have nothing new to add, end the conversation.
```

Same file already has the anti-repetition line. Keep that. Do not add a new metaphor taxonomy.

### 4. Town continuity = existing social-memory

#### Keying

```text
platform     = aitown
room_id      = aitown-town
thread_id    = "{slug_a}--{slug_b}"   # sorted, e.g. cam-lin--sofia-bell
participant_id = card slug            # mara-vale | nico-sable | sofia-bell | cam-lin
                                      # juniper-feld for the human
participant_kind = npc | human
```

`thread_id` rides on `client_meta.external_room.thread_id`. `process_social_turn` already reads that field. Room stays one venue; threads distinguish pairs. That is CallsYne, not a new concept.

Stable slugs replace Convex `p:*` in **both** the new Convex ingest and embodiment’s `_publish_conversation_memory` / `_fetch_participant_continuity`. Map `player.name` → slug via a 6-row table (4 NPCs + Juniper + Orion). Unknown name: fail-open, skip publish/fetch rather than invent a key.

Wipe orphans the old `p:*` rows. Accepted. Do not migrate them.

#### Write path (NPC-generated lines, other is not Orion)

Convex already does HTTP (llm-gateway). Add one POST, not Redis from TypeScript.

New route on `orion-social-memory`:

```text
POST /ingest-turn
Authorization: Bearer $SOCIAL_MEMORY_INGEST_TOKEN
body: SocialRoomTurnV1
```

Handler: validate `SocialRoomTurnV1`, publish on `orion:chat:social:turn` / `social.turn.v1` via the existing `_publish` helper. sql-writer persists → `social.turn.stored.v1` → existing `process_social_turn`. Do **not** call `process_social_turn` directly (that skips `social_room_turns`).

`orion/bus/channels.yaml`: add `orion-social-memory` to `orion:chat:social:turn` `producer_services` (currently `orion-hub`, `orion-embodiment`).

When to POST, from `continueConversationMessage` / `leaveConversationMessage` after a successful NPC completion:

- Speaker is an NPC.
- Other player is **not** Orion (name match on `AITOWN_ORION_NAME` / “Orion”).
- Body: `prompt` = other’s last line, `response` = this NPC’s new line, `text` = `"{other}: {prompt}\n{speaker}: {response}"`, `session_id` = `aitown:{conversationId}`, `client_meta` as above with `thread_id` and speaker as `external_participant`.

Fail-open: ingest errors never block speech.

Do **not** double-publish Orion↔NPC. Embodiment keeps that dyad.

#### Read path (once per conversation start)

In `startConversationMessage` (and the first `continue` if start had no cache):

```text
GET /summary?platform=aitown&room_id=aitown-town&participant_id={other_slug}
```

Inject, budget-capped (~400 characters total), in this order:

1. `room.recent_thread_summary` or `room.current_thread_summary` if non-empty
2. other’s `safe_continuity_summary` if non-empty
3. if `thread_id` is present in room open threads, that thread’s one-liner

Cache the compact string on the in-memory conversation prompt context for the rest of that conversation. Do not GET on every line.

Fail-open: empty string, speak as today.

Orion’s existing GET stays. Add a gate test that `_fetch_participant_continuity` now queries `sofia-bell` (slug) not `p:12`. Live smoke after wipe: one Orion↔Sofia exchange, then `GET /summary` returns non-empty `safe_continuity_summary`.

### 5. Operator wipe (required for the cull to be real)

Descriptions changes do not evict live Convex players. After patches apply and Convex functions redeploy:

```bash
cd services/orion-ai-town/upstream
npx convex run testing:stop
npx convex run testing:wipeAllTables
npx convex run init
npx convex run testing:resume
# repo root
python services/orion-embodiment/scripts/bootstrap_orion_agent.py --write
```

Then confirm four NPC names on the map, none of the dead four, Orion present.

`SOCIAL_MEMORY_INGEST_TOKEN` + `AITOWN_SOCIAL_MEMORY_URL` go in social-memory `.env_example` / `.env` (sync script) and Convex env (document in README; Convex env is operator `npx convex env set`, not the Python sync script). `LLM_MODEL` stays `quick_background`.

## Proposed schema / API / env

- **Schema:** none. `SocialRoomTurnV1` and `client_meta.external_room.thread_id` already exist.
- **Bus:** `orion-social-memory` added as producer on `orion:chat:social:turn`.
- **HTTP:** `POST /ingest-turn` on `orion-social-memory`.
- **Env added:**
  - `SOCIAL_MEMORY_INGEST_TOKEN` — `services/orion-social-memory/.env_example` (empty placeholder) + local `.env` via `python scripts/sync_local_env_from_example.py`
  - Convex: `SOCIAL_MEMORY_URL`, `SOCIAL_MEMORY_INGEST_TOKEN`, `AITOWN_ORION_NAME=Orion` — operator-set, documented
- **Env meaning change:** embodiment `participant_id` for town NPCs becomes the card slug. Same flag, same room, new id values. Call it out in the embodiment README.

## Files likely to touch

- `services/orion-ai-town/cards/town_cards.yaml`
- `services/orion-ai-town/cards/archived/2026-08-29-retired-cast.yaml` (new)
- `services/orion-ai-town/scripts/generate_descriptions.py`
- `services/orion-ai-town/tests/test_generate_descriptions.py`
- `services/orion-ai-town/patches/orion-character.patch`
- `services/orion-ai-town/patches/orion-concrete-grounding-prompt.patch`
- `services/orion-ai-town/patches/orion-town-continuity-ingest.patch` (new; conversation.ts HTTP)
- `services/orion-ai-town/tests/test_concrete_grounding_prompt_patch.py`
- `services/orion-ai-town/tests/test_town_continuity_prompt_patch.py` (new)
- `services/orion-ai-town/README.md`
- `services/orion-social-memory/app/main.py`
- `services/orion-social-memory/app/service.py` (thin ingest publish)
- `services/orion-social-memory/app/settings.py`
- `services/orion-social-memory/.env_example`
- `services/orion-social-memory/tests/` (ingest auth + publish)
- `services/orion-social-memory/README.md`
- `orion/bus/channels.yaml`
- `services/orion-embodiment/app/worker.py` (name → slug)
- `services/orion-embodiment/tests/test_worker_conversation_memory.py`
- `services/orion-embodiment/README.md`

## Non-goals

- No new Convex memory tables, no dyad-fact store, no recall/crystallization for NPC speech.
- No `conversation.ts` GET on every line.
- No NPC–NPC turns through cortex / stance / Unified Turn.
- No migrating old `p:*` social-memory rows.
- No deleting archived card prose.
- No cooldown retune in this patch (optional follow-up once four NPCs are live).
- No Postgres-for-Convex migration (old 2026-07-29 finding; unrelated).
- No making social-memory synthesize on the ingest request (async via existing stored-event path only).

## Acceptance checks

1. `generate_descriptions.py` emits exactly four `Descriptions`. None of Juno / Tessa / Vale / Elian appear. Each identity string contains that card’s `role` and a concrete object from `public_description` or `daily_loop`.
2. Live identity strings do not contain “lighting”, “glow”, “shadows”, or “echoes” as prompt bait. The concrete-grounding patch no longer names those words.
3. After wipe+init: map roster is Mara, Nico, Sofia, Cam, Juniper, Orion only.
4. Sofia talks to Cam (neither is Orion). `social_room_turns` gains a row tagged `platform=aitown`, `room_id=aitown-town`, `thread_id=cam-lin--sofia-bell`, `participant_id=sofia-bell` (or Cam’s, for Cam’s line). `GET /summary?platform=aitown&room_id=aitown-town&participant_id=sofia-bell` is non-empty after `process_social_turn`.
5. Next Sofia–Cam conversation start prompt includes a fact from the previous exchange (room or peer summary), not only the last 4 in-chat lines.
6. Orion↔Sofia still publishes only from embodiment; no duplicate ingest from Convex. `_fetch_participant_continuity` uses `sofia-bell`.
7. Ingest without bearer token is 401. Ingest failure does not prevent the NPC line from appearing in town.
8. Env template parity passes. `python scripts/sync_local_env_from_example.py` run if `.env_example` changed. `PUBLISH_CORTEX_EXEC_GRAMMAR` skip reported if it fires.

## Risks / failure modes

- **Slug drift.** If a card `name` changes and the slug table does not, continuity silently forks. Keep slugs as an explicit map next to `NPC_ORDER`, not `name.lower().replace(" ", "-")` inferred at runtime from display names (`Dr. Elian Cross` already proved that is fragile — we are deleting him, do not reintroduce the pattern).
- **Orion name match.** If Orion’s town name is not exactly `Orion`, Convex will ingest Orion↔NPC and double-write. Pin `AITOWN_ORION_NAME` and test the skip.
- **`process_social_turn` LLM cost.** One pair, 8s message cooldown, one ingest per line — same order as a CallsYne chat. If the diner still floods the gateway, drop ingest to leave-conversation only. Do not add a new store.
- **Small model still collapses.** If cards + injection + continuity still yield light-talk, the next lever is the model route, not more prompt negatives.

## Recommended implementation order (one PR, two commits)

1. **Cards + prompts + cull.** Archive, rewrite four, generator, light-list, regenerate character patch, tests. Operator wipe can happen here so the map matches; continuity will be empty until commit 2.
2. **Ingest + start-of-conversation summary + slug keying.** social-memory POST, channel producer, conversation.ts patch, embodiment slug map, tests, env sync.

Do not ship commit 1 to a long-lived world without commit 2 if Juniper is standing in town waiting for memory — wipe once, after both are deployed.

## Restart required (when implemented)

```bash
# after commit 2
python scripts/sync_local_env_from_example.py
# recreate social-memory so ingest token + route load
# (print exact compose for Juniper; do not sudo)
# Convex: npx convex env set SOCIAL_MEMORY_URL / SOCIAL_MEMORY_INGEST_TOKEN / AITOWN_ORION_NAME
# Convex: npx convex dev --once
# then wipe+init+Orion bootstrap as above
# embodiment recreate so slug keying loads
```
