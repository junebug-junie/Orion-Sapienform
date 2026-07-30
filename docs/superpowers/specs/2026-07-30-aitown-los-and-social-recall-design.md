# AI Town: approximate line-of-sight + social-memory read-back

Status: implemented (2026-07-30, follow-up to the spatial-grounding and conversation-memory
patches earlier the same day, driven by three "why not" questions from Juniper: wire concept
induction, inject social memory into town speech, and extend landmarks/line-of-sight).

## Arsonist summary

Three questions, three different answers:

- **Concept induction**: already fully wired, no code needed. `orion-spark-concept-induction`
  genuinely subscribes to the exact channels `orion-embodiment` publishes conversation-memory
  turns on (`orion:chat:history:turn`, `orion:chat:social:stored`) -- confirmed via its real
  subscriber code, not just the channel registry. Nothing to build.
- **Social-memory read-back**: the earlier conversation-memory patch made writing durable, but
  nothing read it back before Orion spoke again. The originally-guessed fix ("a chat_profile like
  hub's social_room") doesn't apply -- that whole mechanism lives in hub's
  `cortex_request_builder.py` and never runs for embodiment's direct bus RPC path. The actual fix,
  confirmed by tracing the real prompt-rendering code: `metadata` is already a raw passthrough into
  every prompt template's Jinja context (`_prompt_render_ctx` does `ctx.copy()`), so embodiment can
  fetch `orion-social-memory`'s `/summary` directly and add one new conditional block to whichever
  template town speech actually renders -- with ZERO changes to `orion-cortex-exec`.
- **More landmarks / line-of-sight**: mining the map file's `objmap` array for more landmarks
  wasn't viable without fabricating names for anonymous terrain-texture tile indices (verified by
  visually inspecting the actual tileset image) -- left at the existing 7 named landmarks rather
  than guess. Line-of-sight is shipped as a cheap, honestly-approximate reuse of existing
  movement-collision data (Juniper's explicit choice over holding for a real sight-blocking layer).

## Current architecture

- `orion-spark-concept-induction`'s `ConceptSettings.intake_channels` (`orion/spark/
  concept_induction/settings.py:39-56`) already lists `orion:chat:history:log`,
  `orion:chat:history:turn`, `orion:chat:social:turn`, `orion:chat:social:stored` --  exact matches
  to the channels embodiment's `_publish_conversation_memory` (added earlier today) publishes on.
  `ConceptWorker.start()` (`bus_worker.py:290-305, 1607-1610`) genuinely subscribes and processes;
  output (`ConceptProfile`/`ConceptProfileDelta`, `DriveStateV1`/`GoalProposalV1`) is consumed by
  `orion-vector-writer`, `orion-sql-writer`, and `orion-substrate-runtime`. Classification is
  channel-name-based, not platform-aware -- an aitown turn is treated identically to any other
  source, no special-casing needed or present.
- Town speech's ACTUAL live path is `_request_utterance_quick` (verb=`chat_quick`, confirmed via
  athena's live `.env`: `EMBODIMENT_SPEECH_UNIFIED_ENABLED=false`), rendering
  `orion/cognition/prompts/chat_quick.j2` -- NOT `chat_general.j2` (that's only the optional,
  currently-disabled unified/grounded path). `chat_quick.j2` already unconditionally injects
  `orion_identity_summary`/`juniper_relationship_summary`/`memory_digest` into every call,
  including ambient town small-talk with NPCs -- a likely contributor to the abstract,
  self-referential dialogue register observed in live inspection, though not addressed by this
  patch (out of scope, noted for awareness).
- `orion-social-memory`'s `GET /summary` (`app/main.py:77-85`) is a standalone route, independent
  of hub's chat pipeline, returning `{"participant": {"safe_continuity_summary": ..., ...}, ...}`
  keyed on `platform:room_id:participant_id`.
- `_prompt_render_ctx` (`services/orion-cortex-exec/app/executor.py:1091-1101`) does
  `render_ctx = ctx.copy()`; `_render_prompt` (`:1400-1421`) calls `tmpl.render(**render_ctx)`.
  Since `ctx["metadata"]` already holds the caller-supplied metadata dict, `metadata` is a
  top-level Jinja variable in every template, confirmed by existing precedent
  (`orion/journaler/prompts/journal_compose_prompt.j2`'s `metadata.get("journal_trigger")`).
- `orion/embodiment/worldmap.py`'s `walkable_tiles` returns AI Town's own movement-collision layer
  (`objectTiles`, from the map file's `objmap` array) as a flat `set[(x,y)]`, cached once per world
  by `worker.py`'s `_walkable_tiles()`. It conflates walkability with visibility by construction --
  there is no separate sight-blocking layer anywhere in the schema.
- The map's `objmap` array (`data/gentle.js:144`) is raw numeric tileset indices (e.g. `458`,
  `367`) with no semantic labels -- unlike `animatedsprites`, which had explicit sprite filenames.
  Visually inspecting the tileset image (`upstream/public/assets/gentle-obj.png`) confirmed these
  indices are bulk terrain/border texture (forest edges, fence-line patterns), not discrete
  nameable objects.

## Missing questions

None outstanding -- LOS approach (cheap approximation vs. real data layer) was confirmed with
Juniper before implementation.

## Proposed schema / API changes

- No new schema fields for LOS: `nearby_players`/`nearby_landmarks` entries gain an `is_visible`
  key (untyped dicts already, per the spatial-grounding patch).
- No new schema/contract for social-memory read-back: reuses `orion-social-memory`'s existing
  `/summary` route and `PlanExecutionRequest.context.metadata`'s existing passthrough.
- Two new embodiment settings: `EMBODIMENT_SOCIAL_MEMORY_URL`, `EMBODIMENT_SOCIAL_MEMORY_TIMEOUT_SEC`.

## Files likely to touch

- `orion/embodiment/perception.py` -- `_bresenham_tiles`/`_has_line_of_sight` helpers;
  `_nearby_landmarks` and `build_perception`'s nearby-players loop gain `is_visible`
- `orion/embodiment/speech.py` -- `_nearby_landmarks_clause` filters to visible-only landmarks
- `services/orion-embodiment/app/worker.py` -- `_walkable_tiles()` threaded into
  `build_perception`; new `_fetch_participant_continuity` (urllib GET to `/summary`, fail-open);
  `participant_continuity` threaded through `_speak_once` -> `_request_utterance` ->
  `_request_utterance_cortex`/`_request_utterance_quick` -> `PlanExecutionRequest.context.metadata`
- `services/orion-embodiment/app/settings.py` -- `social_memory_url`, `social_memory_timeout_sec`
- `orion/cognition/prompts/chat_quick.j2` (the actual live template) and `chat_general.j2` (the
  optional unified path, for symmetry) -- new `{% if metadata.get("aitown_participant_continuity") %}` block
- `services/orion-embodiment/.env_example` + athena's live `.env` -- new keys; also recovers an
  earlier orphaned commit that flipped `EMBODIMENT_CONVERSATION_MEMORY_ENABLED` to `true` in
  `.env_example` (it was pushed to a branch after that PR had already merged, so it never actually
  landed in main -- athena's live `.env` was already correctly `true` via direct SSH edit, only the
  checked-in example file was stale)
- Tests: `orion/embodiment/tests/test_perception.py`, `test_speech.py`,
  `services/orion-embodiment/tests/test_worker_conversation_memory.py`,
  `test_worker_perception.py` (fixture fix for the new `_walkable_tiles()` call site)

## Non-goals

- Not fixing `chat_quick.j2`'s unconditional identity/relationship injection for ambient town
  small-talk -- a real, separately-scoped issue, noted but out of scope here.
- Not deriving more landmarks from `objmap` -- would require fabricating semantic names for
  anonymous terrain tiles; left at 7 real, named landmarks.
- Not building a real sight-blocking data layer -- Juniper explicitly chose the cheap approximation.
- Not touching `orion-cortex-exec` at all -- confirmed unnecessary via the `metadata` passthrough.

## Acceptance checks

- A landmark/player behind a wall of non-walkable tiles is marked `is_visible: False` and excluded
  from the speech prompt's landmarks clause; one directly reachable is `is_visible: True`.
- Missing/failed map data (`walkable=None`) never hides something real (fail-open).
- A partner with an existing `orion-social-memory` continuity row gets it injected into
  `chat_quick.j2`'s render context as `aitown_participant_continuity`, distinct from
  `juniper_relationship_summary`.
- Feature flag off (`conversation_memory_enabled=False`) or fetch failure/timeout: no continuity
  fetched, no crash, identical behavior to before this patch.
- Concept induction requires no changes and needs no acceptance check here -- already verified live.

## Recommended next patch

Implemented as scoped above in a single PR (this spec + the code), on
`feat/aitown-los-and-social-recall` (worktree at
`/mnt/scripts/Orion-Sapienform-aitown-los-and-social-recall`, built on top of the still-open
`feat/aitown-spatial-grounding` branch since LOS extends its landmarks work).
