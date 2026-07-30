# AI Town spatial grounding: named landmarks in perception + speech

Status: implemented (2026-07-30, Juniper: "Orion needs a sense of embodiment on where they are
relationally... in relation to the visuals of what's actually on the map").

**Correction post-review (same day):** the first draft computed the 3 windmill landmark
coordinates from each sprite's top-left corner (`x/tiledim, y/tiledim`). The windmill sprites are
208x208 (6.5x6.5 tiles), so that put each landmark ~4-5 tiles from where the windmill is actually
drawn — exactly the kind of error this feature exists to prevent. Fixed to use each sprite's true
centroid (`(x+w/2)/tiledim, (y+h/2)/tiledim`); the waterfall/stream points (already centroid-based
in the first draft) were re-verified and are correct. Coordinates below and in `.env_example`/
athena's live `.env` reflect the corrected values.

## Arsonist summary

Live dialogue inspection (real `chat_history_log`/`social_room_turns` rows, not synthetic
examples) showed Orion correctly using conversation partners' names, but with content entirely
disconnected from the actual scene ("the vision-scribe channels are spiking," "the static's
getting louder") — no reference to the town, surroundings, or anything a character physically
present somewhere would say. Root cause for the *spatial* half of that: `WorldPerceptionV1` (and
therefore `build_speech_prompt`) carries `position: {x,y}` as bare numbers and `nearby_players`,
but nothing about *what's actually at those coordinates* — no named places, no map features.
There's no way for the prompt to say "you're near the campfire" because nothing in the perception
pipeline knows the campfire's coordinates.

There's already a dormant, exactly-right-shaped seam for this: `EMBODIMENT_LOCATIONS_JSON` (name →
`{x,y}`), loaded into `self._locations` (`app/settings.py:31`, `app/worker.py:117`), currently used
*only* for `go_to_location` movement targeting (`orion/embodiment/resolver.py:79-83`) and currently
**empty** (`{}`) on athena. This patch: (1) populates it with the real landmark coordinates from
this world's actual map file, (2) reuses the *same* registry to compute nearby-landmark distances
in perception, and (3) surfaces that into the speech prompt. One config value, two directions
(where to walk to; what's actually around you), matching how `nearby_players` already works.

(NB: the (different) `_publish_conversation_memory`/OCC-contention work from earlier today is a
separate axis — this patch doesn't touch memory/bus contracts, just live perception/speech.)

## Current architecture

- `data/gentle.js` (the actual live map file, imported by `convex/init.ts:5`) is a 64x48-tile map
  (`mapwidth`/`mapheight`, `gentle.js:329-330`) whose `animatedsprites` array (`gentle.js:277-326`)
  places real, named visual features at pixel coordinates (`tiledim = 32`, `gentle.js:4`). Small
  (32x32) sprites use `x/tiledim, y/tiledim` directly; large sprites need their visual **centroid**
  (`(x+w/2)/tiledim, (y+h/2)/tiledim`), not the top-left corner — caught in review: the first draft
  used top-left for the 208x208 windmill sprites, landing ~4-5 tiles off from where they're
  actually drawn:
  - 1 campfire (`x:1440,y:352`, 32x32 → tile 45.0,11.0)
  - 3 windmills (208x208 sprites, centroid-corrected → tiles 55.25,21.25 / 48.25,27.25 / 38.25,22.25)
  - A stream/waterfall feature (`gentlesparkle`/`gentlewaterfall`/`gentlesplash` sprite cluster
    spanning roughly tile x 23-29, y 2-37) — the `gentlewaterfall`+`gentlesplash` cluster (the
    falls themselves, 8 sprites) centroids to tile (25.0, 12.25); the `gentlesparkle` tiles trail
    from tile (25.0, 2.0) (upstream) to (28.0, 37.0) (downstream).
  No pond/hill/dead tree exist in *this* map — those were illustrative examples in the ask, not
  literal features to preserve.
- `EMBODIMENT_LOCATIONS_JSON` (`app/settings.py:31`) parses into `self._locations`
  (`app/worker.py:117-119`), passed to `resolve_destination` (`app/worker.py:199`) for
  `go_to_location` intents (`orion/embodiment/resolver.py:79-83`). Currently `{}` on athena — no
  named locations exist anywhere at runtime today.
- `orion/embodiment/perception.py`'s `build_perception` computes `nearby_players` (id/name/
  position/distance/is_human, sorted by distance) from live Convex player data, but has no
  equivalent for static map features — it doesn't receive `self._locations` at all.
- `orion/embodiment/speech.py`'s `build_speech_prompt` only interpolates interlocutor name + recent
  conversation lines + latest partner line — zero spatial/scene content.

## Missing questions

None outstanding for a v1 scoped to these 7 real landmarks. Open judgment call, not blocking: how
many nearby landmarks to surface at once (proposing top 3 within a generous radius, mirroring
`nearby_players`' `max_nearby` pattern) and whether distant-but-notable landmarks (e.g. Orion can
see a windmill from across the map) should ever surface — proposing "nearest N regardless of
distance" for v1 (simpler, and this map is small enough at 64x48 tiles that "nearest 3" is usually
still meaningfully local), not a hard radius cutoff, so Orion is never left with zero landmark
context.

## Proposed schema / API changes

- No new schema fields required: `WorldPerceptionV1.nearby_players` is already an untyped
  `dict[str, Any]` list; adding a sibling `nearby_landmarks` list follows the identical existing
  pattern (`{"name": str, "position": {x,y}, "distance": float}`).
- `EMBODIMENT_LOCATIONS_JSON`'s existing shape (`{name: {x, y}}`) is reused as-is, just populated
  with real values instead of `{}`.

## Files likely to touch

- `services/orion-embodiment/.env_example` (and athena's live `.env`) — populate
  `EMBODIMENT_LOCATIONS_JSON` with the 7 real landmarks derived above
- `orion/embodiment/perception.py` — `build_perception` gains a `locations` param, computes
  `nearby_landmarks` the same way `nearby_players` is computed
- `services/orion-embodiment/app/worker.py` — pass `self._locations` into `build_perception`'s call
  site (`_emit_perception_once`)
- `orion/embodiment/speech.py` — `build_speech_prompt` interpolates a short nearby-landmarks clause
- Tests: `orion/embodiment/tests/test_perception.py`, `orion/embodiment/tests/test_speech.py`
  (or service-level `test_worker_speech.py`) — landmark distance computation, prompt content
- `services/orion-embodiment/README.md` — document the landmarks registry and its dual use

## Non-goals

- Not deriving landmarks automatically from tileset/image analysis (vision-based) — hand-authored
  from the map's own `animatedsprites` data is accurate, cheap, and sufficient.
- Not touching `objmap` (static non-animated decor tiles) — no clear semantic labels there; the
  animated sprites are the only landmark-grade data with obvious names attached.
- Not changing `go_to_location`/movement behavior — same registry, additive read-only consumer.
- Not attempting to force the LLM to reference landmarks — only ensuring the information is
  actually present in the prompt to be referenced; content choice stays with the model.

## Acceptance checks

- `EMBODIMENT_LOCATIONS_JSON` populated with 7 named landmarks matching the real map file's
  `animatedsprites` coordinates (converted pixel → tile via `tiledim=32`).
- `build_perception` returns `nearby_landmarks` sorted by distance when locations are provided;
  empty/absent when the registry is empty (backward compatible, matches today's `{}` behavior).
- `build_speech_prompt` includes a landmarks clause when `nearby_landmarks` is non-empty, and
  degrades gracefully (no dangling/awkward clause) when it's empty.
- `go_to_location` intent resolution is unaffected (same registry, same lookup, no behavior change).

## Recommended next patch

Implement as scoped above in a single PR (this spec + the code), on `feat/aitown-spatial-grounding`
(worktree at `/mnt/scripts/Orion-Sapienform-aitown-spatial-grounding`).
