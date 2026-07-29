# AI Town: Orion conversational quality + latency — design

**Date:** 2026-07-29
**Status:** Proposal — no code changes yet
**Trigger:** Live incident response (AI Town broken/laggy, 2026-07-29) surfaced three follow-on threads once the immediate outage was fixed (PRs #1452, #1456, both merged).

---

## Arsonist summary

AI Town was down/laggy today for reasons unrelated to Orion's conversational quality (unbounded Convex revision-history bloat, a missing `~/.fcc/.env` bridge config, a too-short HTTP client timeout — all fixed live, PRs #1452/#1456). Once it was working again, the older complaint resurfaced: Orion's in-town dialogue is transactional and repetitive, not curious. Investigating "what can we lift from Unified Turn" turned up three separate, real findings:

1. A genuine bug in `orion-cortex-exec`'s identity-injection gate is silently starving Orion's AI Town speech (and any other non-brain-mode caller) of identity grounding that's already fully configured and ready to use. Small, isolated fix.
2. AI Town's backend mutation latency (15-25s, ~10-15x worse than upstream's own stated ~1.5s expectation) traces to SQLite single-writer serialization amplifying an already-contended write pattern. Postgres is a plausible, reversible lever — unverified, needs an isolated test before a live migration.
3. Orion has no persistent per-NPC conversational memory in AI Town. `orion-social-memory`'s read side is architecturally the right shape and cheap enough to use — but nothing writes AI Town turns into it today. Real feature build, not a lift.

None of these three collide with each other or with Orion Unified Turn (checked explicitly for #1 — see below).

---

## Current architecture

### AI Town's two LLM call paths (previously conflated, now confirmed distinct)

- **Orion's own embodied speech** (`services/orion-embodiment/app/worker.py` → `orion/embodiment/speech.py:build_speech_prompt()`) dispatches through `orion-cortex-exec`'s plan/verb system, verb `chat_quick`, `mode="quick"` (`EMBODIMENT_SPEECH_VERB=chat_quick`, `EMBODIMENT_SPEECH_LANE=quick` in `services/orion-embodiment/.env`). This is the one thing that goes through cortex.
- **The 8 NPCs' own dialogue** (a16z's own `upstream/convex/agent/conversation.ts`) calls `orion-llm-gateway` **directly** over plain OpenAI-compatible HTTP, deliberately bypassing cortex/mind entirely (wired by `services/orion-ai-town/scripts/wire_llm_gateway.sh`). Unaffected by anything in this doc.

### Finding 1 — identity injection never fires for AI Town's calls

`chat_quick.yaml` already declares `personality_file: orion/cognition/personality/orion_identity.yaml`, and `chat_quick.j2` already has template slots for `orion_identity_summary` / `juniper_relationship_summary` / `response_policy_summary` (which come from `orion/cognition/personality/identity_context.py:build_identity_context()`, a pure YAML-load + dict-transform already reused in 3 other places). The plumbing is real and complete.

The bug: `services/orion-cortex-exec/app/router.py:994` only calls the prep step that populates those variables (`prepare_chat_quick_reply_context` → `_inject_identity_context`, `executor.py:940-1009`) when `mode == "brain"`:

```python
if mode == "brain" and not _is_runtime_skill_verb(plan.verb_name):
    verb_lc = str(plan.verb_name or "").strip().lower()
    if verb_lc == "chat_quick":
        ...
        prepare_chat_quick_reply_context(ctx)
```

AI Town's embodiment worker builds its plan with `mode="quick"`, not `"brain"` — so for every one of Orion's town lines, this whole step is skipped and the template's identity variables render empty. `orion_identity.yaml` itself is well-written (explicit `banned_phrases` like `"It sounds like..."` / `"I'm here if you need anything"`, and the line *"on relational turns... situated curiosity is appropriate"*) — none of it currently reaches Orion's town voice.

**Checked for collision with Orion Unified Turn**: `stance_react.yaml` (a Unified Turn verb) also declares `personality_file`, and its prompt (`stance_react.j2:6-8`) consumes `orion_identity_summary` too. But `orion/thought/stance_react.py:100` defaults `llm_profile="brain"`, so stance_react is already dispatched brain-tier and already covered by the existing gate — unaffected either way. `harness_finalize_reflect.yaml` and `orion_voice_finalize.yaml` (the other two Unified Turn verbs) don't declare `personality_file` at all; `orion_voice_finalize`'s identity comes from a wholly separate mechanism (`grounding_capsule`, populated elsewhere in the finalize chain). **No shared surface, no regression risk.**

### Finding 2 — AI Town Convex backend latency

Already root-caused this session (see prior PR reports #1452/#1456 and this session's live investigation):

- `upstream/convex/aiTown/game.ts`'s step loop runs every 1000ms (`convex/constants.ts`) and does an **unconditional full `ctx.db.replace()`** of the singleton `worlds` doc and the singleton `engines` doc every single tick, regardless of whether anything changed (`game.ts:206-214,300`, `engine/abstractGame.ts:186`).
- Every input producer (player `moveTo`, 9 NPC agents' `agentSendMessage`, the step's own completion writes) reads/writes the same per-engine monotonic-counter index (`abstractGame.ts:140-152`) — constant OCC collisions with each other and with the once-a-second full-doc commit.
- Measured live: 318 OCC retry errors / 30 min, individual mutations taking 15-25s. Upstream's own `ARCHITECTURE.md:284-301` states ~1.5s expected at this step size — **10-15x worse than upstream's own disclosed ceiling**.
- `services/orion-ai-town/.env_example`'s `DATABASE_URL` is empty → running on embedded SQLite (single-writer). The `convex-local-backend` binary supports `-d postgres-v5`; Convex's own docs confirm switching requires a fresh empty Postgres DB + export/reimport (no in-place conversion) — the exact same shape as the already-shipped `scripts/compact_convex_data.sh`.
- An existing Postgres instance (`orion-athena-sql-db`, hosting `conjourney`) exists in the mesh, but reusing it would put AI Town's already-pathological write pattern on the same `max_connections=100` budget as the rest of Orion's cognition writes — an unsized risk. **Recommend a dedicated Postgres container**, not reuse.
- **Hypothesis is unverified.** Cheapest validation: an isolated `convex-local-backend` + throwaway Postgres, synthetic concurrent-mutation load matching the real contention pattern (9 agents + player + 1Hz step), before touching the live town.

### Finding 3 — no per-NPC conversational memory

`orion/embodiment/speech.py:_recent_lines()` only looks at the last 4 messages of the *current* conversation — nothing about who the NPC is, or what happened in prior conversations with them, even though `town_cards.yaml` describes each of the 8 NPCs as having an ongoing relational dynamic with Orion.

`services/orion-social-memory` is a real, running FastAPI service (port 8765) that already solves the general version of this problem for Juniper. Its read side is a good, cheap fit:

- `GET /summary?platform=&room_id=&participant_id=` (`app/main.py:77-85` → `app/service.py:966-1063`) does plain primary-key lookups (`sess.get()`) on already-materialized Postgres rows, keyed `f"{platform}:{room_id}:{participant_id}"`. No bus RPC, no LLM call in the read path — compatible with `chat_quick`'s latency budget.
- The schema (`SocialParticipantContinuityV1.safe_continuity_summary`, `.recent_shared_topics`, `.interaction_tone_summary`, `app/service.py:87-119`) is exactly "what do we remember about talking to NPC X."

**The gap**: nothing writes AI Town turns into it. Confirmed zero references to `social_memory`/`social_room` anywhere in `services/orion-embodiment/`. Data only exists in `orion-social-memory` after a turn flows through `orion:chat:social:stored` (`SocialRoomTurnStoredV1`, `orion/bus/channels.yaml`) and gets processed (`app/service.py:275-964`, `process_social_turn`). Using the read side means building that write side — a real, scoped feature, not a prompt tweak.

`chat_social_room.j2`/`.yaml` (Orion's peer-chat mode with Juniper specifically, 300s lane) is the wrong lane and wrong identity framing to lift wholesale for NPCs. The one reusable *idea* from it, not code: `build_social_context_window` (`app/synthesizer.py`) — a budgeted, freshness-banded, relevance-scored context-window selection pattern — worth knowing about if the write-side integration ends up needing its own context budgeting.

---

## Missing questions

1. **Finding 1**: are there other non-brain-mode callers of `chat_quick`/`chat_kids_story` today (the design doc's own mode table calls the quick lane "Fleet chores only — not Orion's voice")? Not exhaustively enumerated — worth a quick grep sweep before merging, to confirm the fix doesn't unexpectedly change behavior for some other fleet-chore caller. (Low risk either way: `_inject_identity_context` is idempotent and additive, never removes existing ctx keys.)
2. **Finding 2**: exact `POSTGRES_URL`/`DO_NOT_REQUIRE_SSL` env wiring for `services/orion-ai-town/docker-compose.yml` not yet drafted (currently only threads `DATABASE_URL` through, unused). Load-test methodology (how much synthetic concurrency, how long) not yet scoped.
3. **Finding 3**: exact `room_id`/`participant_id` keying scheme for AI Town NPCs not designed (candidate: `platform="ai_town"`, `room_id=<world_id>` or per-conversation, `participant_id=<npc player_id or stable name>`). Privacy/retention semantics of `orion-social-memory` were built around real people (Juniper, human contacts) — need to confirm fictional NPC personas don't need the same protections, or design an explicit, narrower policy for them. Whether Orion's *own* turns should also be summarized/tracked the same way, or just the NPC's side.
4. Sequencing/appetite: does Juniper want all three pursued, or just a subset, and in what order?

---

## Proposed schema / API changes

### Finding 1 (identity injection)

No new schema. Widen `services/orion-cortex-exec/app/router.py`'s gate so that a verb declaring `personality_file` gets `_inject_identity_context()` called regardless of `mode`, not only when `mode == "brain"`. Smallest form: an `elif` branch alongside the existing `mode == "brain"` block, or hoist the personality-file check above the mode check entirely.

### Finding 2 (Postgres)

No AI Town schema change — infra only. New `docker-compose.yml` service (dedicated Postgres container) + `POSTGRES_URL` wiring, following the same export → stop → reconfigure → restart → redeploy-functions → restore-env → reimport → heartbeat sequence `scripts/compact_convex_data.sh` already implements, retargeted at a fresh Postgres DB instead of a reset SQLite file.

### Finding 3 (social memory write-side)

New producer: AI Town's embodiment worker (or a small adapter alongside `orion/embodiment/aitown_client.py`) publishes `SocialRoomTurnStoredV1` on `orion:chat:social:stored` for each Orion↔NPC exchange it observes, keyed as above. New read call: `build_speech_prompt()` (or its caller in `worker.py`) makes one synchronous `GET /summary` call to `orion-social-memory` for the current NPC and folds `safe_continuity_summary`/`recent_shared_topics` into the prompt when non-empty. `orion/bus/channels.yaml` gains `orion-embodiment` as a new producer on `orion:chat:social:stored`.

---

## Files likely to touch

- **Finding 1**: `services/orion-cortex-exec/app/router.py`.
- **Finding 2**: `services/orion-ai-town/docker-compose.yml`, `.env_example`, `scripts/compact_convex_data.sh` (extended or a sibling script), a new isolated load-test harness (throwaway, not part of the live service).
- **Finding 3**: `orion/embodiment/speech.py`, `services/orion-embodiment/app/worker.py`, a new small `orion-social-memory` read client (or extend `aitown_client.py`), `orion/bus/channels.yaml`, `services/orion-embodiment/.env_example`.

---

## Non-goals

- Not migrating AI Town's actual game-engine architecture — accepting a16z's disclosed contention profile as a given; Postgres only mitigates the amplification on top of it.
- Not lifting Unified Turn's motor or its 5a/5b/5c multi-LLM finalize chain into AI Town — confirmed too heavy/slow for the town's latency budget; explicitly out of scope.
- Not building a generic multi-tenant social-memory platform abstraction beyond what AI Town's NPC keying needs.
- Not touching the NPCs' own dialogue path (still bypasses cortex, unchanged, out of scope).
- Not making the Postgres switch live before the isolated hypothesis test.

---

## Acceptance checks

1. **Finding 1**: live trace shows `identity_context_ready identity_kernel_source=configured_yaml` logged for an AI-Town-originated `chat_quick` call (traceable by correlation_id); a live AI Town conversation shows reduced verbatim repetition and at least one instance of situated curiosity (a follow-up question) in a companion-mode exchange.
2. **Finding 2**: isolated load test shows OCC retry rate materially below the measured 318/30min SQLite baseline before any live migration is attempted. Live migration (if pursued) preserves 100% of exported documents (same check already used for PR #1452's compaction: `npx convex import --replace-all` change-summary matches export counts).
3. **Finding 3**: after N turns with the same NPC across two separate conversations, `orion-social-memory`'s `/summary` for that `participant_id` returns non-empty `recent_shared_topics`/`safe_continuity_summary`, and a subsequent AI Town reply visibly references something from the earlier conversation rather than just the last 4 lines of the current one.

---

## Recommended next patch

Ship Finding 1 first — smallest, most isolated, fixes a real active bug, no dependency on the other two. Then run Finding 2's isolated load test (no live changes) to get a go/no-go decision on Postgres migration. Then scope Finding 3's write-side integration as its own PR once the other two are settled — it's a real feature build and deserves its own focused pass rather than being bundled in.
