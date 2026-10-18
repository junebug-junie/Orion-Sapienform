# Walkway reach into Orion — tracking list

Status as of 2026-10-18. Source: a read-only investigation of every place walkway
camera content reaches Orion (live database, container logs and the graph database,
snapshot about 2026-10-18 00:15 UTC). Each item below gets its own design doc later.
When a design doc exists, link it on the item and keep the item here until its fix is
live and verified.

## Queued for resolution (need to maximize these areas and optimize)

| Part of Orion | What walkway content reaches it | Live? |
|---|---|---|
| **Chat stance** | The street line: who was outside in the last 15 minutes (names first) and recent captions. | Render yes; reaching a real chat turn is UNVERIFIED (nothing records it). |
| **Curiosity** | Captions from the last 6 hours, plus the walkway's own surprise signal. | Yes. Walkway offered 67 times in 48 h, chosen 28 times; one investigation asked "do certain people show up at consistent times?" |
| **Attention** | Walled off by design (never wins). | Yes, but it leaks into every attention record as a suppressed item labelled "body signal" — an outdoor camera filed as body. Wrong label, harmless. |
| **Substrate layers** | The walkway surprise signal (live, updating). Your named people (Maeve, Teddy, Esme, Adeline, Amanda Feld) each exist as a node in Orion's graph. | Partly. The named-people nodes have **no links to anything** and new sightings never update them. |
| **Reveries** | The last couple of captions. | Yes. 66 of 518 reveries mention the street, e.g. "vehicles on the walkway disrupting expected patterns." |
| **Dreams** | Only routine predictions about the walkway — and none exist yet. | No. |
| **Memory and the day letter** | Nothing direct. Episodes count only the room camera; the day letter reads no camera at all. | No. |

### Evidence and code pointers per item

- **Chat stance.** Reader: `orion/situational/perception_reader.py` `fetch_street_summary` →
  `orion/situational/context.py` `_build_street_fields` →
  `services/orion-cortex-exec/app/chat_stance.py`. Live render confirmed inside the hub
  container. Gap: only "right now / last 15 min", no memory of the day; the expectations
  part is always empty (see Dreams). No log or stored brief proves the line reached a turn.
- **Curiosity.** Study material: `services/orion-hub/scripts/curiosity_investigation.py`
  (6 h of distinct captions + unresolved list) → `orion/curiosity/kickoff_prompt.py`.
  Surprise node in endogenous curiosity: `orion/substrate/endogenous_curiosity.py`; fence set
  in `orion/schemas/attention_candidate.py` (curiosity fence empty). 72 h: admitted 28,
  declined 55. Example: "sustained prediction error on node:substrate.perception_walkway",
  strength 0.76; focus `a21f96c9` ran 10-16. Gap: captions in the real kickoff prompt are
  UNVERIFIED (not logged; material-pool table stale since 09-30).
- **Attention.** Fence: `NO_ATTENTION_CONSUMER_NODE_IDS`, applied in
  `orion/substrate/attention_broadcast.py` and `services/orion-attention-runtime/app/worker.py`.
  0 frames with walkway as a dominant target. Leak: 81,500 of 81,500 attention frames in 48 h
  carry it as a suppressed target labelled "world-first: body signal…", `source_kind:internal`,
  via the host-novelty path in `orion/attention/field_attention/builder.py` (strips candidate
  ids, not fenced ids). Open decision: keep fenced vs let it in as a world source with a
  camera-absent check.
- **Substrate layers.** Surprise node `node:substrate.perception_walkway`: liveness rung in
  `orion/substrate_ladder_liveness.py`, reducer in `services/orion-substrate-runtime/app/worker.py`.
  63 reducer receipts in 48 h; 3,906 surprise points, mean 0.06, max 1.0, 121 ≥ 0.55; node has
  0 edges. Named-people entity nodes: 7 `entity:vision_individual:*` in FalkorDB, 0 edges each,
  activation flat ~0.625; only `ask_answered_listener.py` writes them, sightings never do.
- **Reveries.** `services/orion-thought/app/reverie.py` (street captions + graded predictions).
  164 `reverie_street_context` log lines in 48 h, each `captions=2 graded=0`. Gap: sees only the
  last few captions; no day view.
- **Dreams.** `services/orion-dream/app/cycle.py` / `replay.py` take graded walkway predictions
  only (capped at 1/6 of replay slots). `DREAM_REPLAY_WALKWAY_ENABLED=true`, but
  `dream_replay_item` has 0 walkway rows ever because `vision_percept_expectation` has 0 rows
  (needs ≥ 5 days of history). Captions are not a dream source.
- **Memory and the day letter.** Episodes exclude the walkway by design
  (`ROOM_STREAMS={"cam0"}` in `orion/temporal_self/sources.py`). Day letter
  (`orion/orion_day/gather.py`) reads no vision source. The only nightly walkway artifact is the
  forecast journal from `services/orion-actions/app/walkway_forecast.py` (22:30 local); on
  10-17 it said "the walkway has been empty on at least five occasions" with no data behind
  it (0 expectation rows) — an invented claim.

### Built but reaching nothing (also queued)

1. **Motion stories** (`vision_motion_story`): Hub-only by design until the spot-check gate
   passes (≥ 50 checked, ≥ 80 % right, 0 wrong names; 8 checked as of 10-18). Nothing is wired
   behind the gate — the PR #2736 plan (street line, reveries/dreams, curiosity) exists only in
   the spec.
2. **Routine predictions** (`vision_percept_expectation`): empty until ≥ 5 days of history;
   four readers get nothing (street line expectations, reverie `graded`, dream candidates,
   forecast grading).
3. **Named people in the graph**: labels with no links, never refreshed by sightings.

## In progress

- Walkway rollups by hour, day, week and month (Juniper, 2026-10-18) — the first fix for
  "Memory and the day letter" and the named-people links:
  [PR #2755](https://github.com/junebug-junie/Orion-Sapienform/pull/2755). Wires the day letter,
  curiosity study material, nightly/weekly/monthly journal (with an anti-invention guard), a Hub
  "Walkway history" panel, and `observed_in` day links for named people. Attention, episodes and
  dreams untouched. Known limit: people are over-counted (the individuals reducer splits one
  person into several clusters), so passings are an upper bound and rows also store "most at once".
- Story-card naming joined to recognition:
  [PR #2753](https://github.com/junebug-junie/Orion-Sapienform/pull/2753) (motion stories still
  Hub-only until the spot-check gate passes).
