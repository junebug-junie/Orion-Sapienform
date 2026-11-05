# AI Town: a living world Orion can predict — design

**Date:** 2026-10-14
**Status:** Proposal — no code changes yet
**Asked for by:** Juniper, 2026-10-14: "the characters send like sentence long rants that suck and arent coherent. I want them to be dynamic microcosms that actually help orion simulate the world." Follow-up decisions are in **Locked decisions** below.

---

## Arsonist summary

The town sucks because nothing happens in it. The NPCs have a poetic bio and a chat window and nothing else: no stock, no money, no events, no consequences. So a small model (Qwen3-8B, 4-bit) fills the empty space with mood, copies its partner's last line, and makes up people. In the last 80 stored messages it invents Maggie, Tom, Jake, a baker and a blacksmith. One chat trades the same sentence 7 times. Every conversation ends without anything changing. The town has been stopped by hand (`stoppedByDeveloper`) since its last message on 2026-09-17. Its output was also leaking into Orion's memory (796 of 1,291 crystallizations), so it was walled off at the database. Today it gives Orion nothing.

This design gives the town a **real state that changes on its own**: places, stock, money, needs, trust and beliefs, stored as numbers and moved by code. On top of that it adds **events and quests** that the simulation, Juniper and Orion (from outside) can all add. The town also **loosely mirrors the real world** through Orion's existing world feed. Each NPC conversation becomes **one scene**: either the characters just shoot the breeze, or they pick 1–2 results from a list of things that are actually possible right now. The simulation then applies those results. **Orion predicts what happens and gets scored against what actually happened.** That gives Orion practice at modelling other minds and social cause-and-effect, with real right answers. It all stays labelled as simulation and never enters Orion's memory.

On "RL or chaos theory": no reinforcement-learning training loop, and no chaos math as a feature. Two concrete, cheap mechanisms borrow the useful part of each:

- **Chaos → seeded randomness plus resampling.** Every random draw comes from a recorded seed, so any day can be replayed. For a scene Orion is predicting, the simulation re-rolls that scene a few times. The spread of results measures how predictable the scene was in the first place. Orion's score is judged against that spread, so a coin-flip scene doesn't count as Orion failing.
- **RL → scores are stored as reward, nothing trains on them yet.** Orion's prediction scores land in a typed stream that the existing self-calibration work can read. NPC behaviour is a fixed, inspectable needs-and-utility rule, not a learned policy. Learning comes after the scores show there is signal worth learning from.

---

## Current architecture

- **Town (Convex, on circe).** The a16z engine plus 16 Orion patches. The upstream checkout (`services/orion-ai-town/upstream/`) is gitignored; Orion's changes ship as `patches/*.patch`, applied by `scripts/apply_upstream_patches.sh`. Tables today: `worlds`, `worldStatus`, `maps`, `playerDescriptions`, `agentDescriptions`, `archived*`, `participatedTogether`, `messages`, `memories`, `memoryEmbeddings`, `embeddingsCache`, `inputs`, `engines`. Nothing here holds stock, money, needs, events or facts.
- **NPC decisions.** `aiTown/agent.ts:54 tick()` → `agentOperations.ts:93 agentDoSomething`. The activity is picked **at random** from a hardcoded list of three, reading/daydreaming/gardening (`constants.ts:73`; the code even carries "TODO: have LLM choose"). Otherwise the agent wanders or invites the nearest player in range (`:147`).
- **NPC speech.** `agent/conversation.ts`: one call per line on `quick_background` (Qwen3-8B Q4), `max_tokens` 120. The whole history goes in as `role:user` lines prefixed `Name to Name:`, ending with `Me to Them:`. The no-self-repeat patch checks only the speaker's own lines, not the partner's.
- **Orion's town body.** `services/orion-embodiment` subscribes to `orion:embodiment:intent` (`EmbodimentIntentV1`: approach/start_conversation/wander/go_to/idle) and drives Convex through `orion/embodiment/aitown_client.py` with the operator admin key. It publishes `orion:embodiment:perception` / `:outcome`. Orion speaks through cortex (`chat_quick`). **Nothing anywhere can add a world-level event.** Hub only embeds the UI.
- **Continuity.** `orion-social-memory` hosts `GET /town-continuity` and `POST /ingest-turn`. Convex calls both, fail-open.
- **Memory wall.** Platform tag `aitown` → `formation_policy.DEFAULT_DISCARD_PLATFORMS`, the recall `AITOWN_TAG`, the separate `aitown_chat_history_log` table, the `_NOT_AITOWN` filter in `study_material.py`, and the DB-level revoke in `scripts/sql/scope_orion_readonly_off_aitown.sql`.
- **Real-world feeds.** `orion-world-pulse` publishes `orion:world_pulse:digest:item` (`DailyWorldPulseItemV1`: `title`, `category`, `summary`; sources in `config/world_pulse/sources.yaml`). Weather is `WeatherCurrentV1`/`WeatherAlertV1` (`orion/schemas/situation.py`), built in `orion/situational/context.py`.
- **Existing prediction seams (checked, not reused).** `orion-world-model` is PARKED (untrained, nothing consumes it). `orion/schemas/action_prediction.py` (`ExpectedEffectV1`, `ActionOutcomeRecordV1.prediction_error`) scores the effects of **Orion's own autonomous actions**. Forecasting what *other* agents will do is a different question, so it gets its own record. Its field names follow that file so a later calibration reader can treat both alike. There is no Brier scoring anywhere under `orion/` today.
- **Live state, 2026-10-14.** World `m1720g80…` is `stoppedByDeveloper`; the last message is 2026-09-17 23:45 UTC.

---

## Locked decisions (Juniper, 2026-10-14)

1. **Loosely mirror the real world.** Real-world signals become town events in town terms (a heat wave becomes a hot week and a run on cold drinks; a supply-chain story becomes a late shipment). Never a literal copy, never Juniper's private life.
2. **Orion (from outside the simulation) and Juniper can both add events,** including quests.
3. **Characters may just shoot the breeze.** A scene can have zero results. When it has any, it picks **1–2** from the list of results that are legal right now.
4. **Quests** are a kind of event: a goal checked against the world state, a deadline, and participants. NPCs and Orion work toward it.
5. Everything else in the first pass is accepted: world state, scene generation, false beliefs, Orion predicting with scores, and the memory wall stays up.

---

## Design

### 1. World state (code, no LLM)

New Convex tables, owned by a new `convex/sim/` module (shipped as tracked overlay files, see **Files**):

| table | holds |
|---|---|
| `simPlaces` | named places (diner, studio, workshop, square…) mapped onto existing map zones |
| `simStock` | item counts per place or person (coffee, parts, flyers, cash) |
| `simAgentState` | per character: `needs` (money, rest, social, purpose, 0–1), `schedule`, `location` |
| `simTrust` | directed pair → trust (-1..1) |
| `simBeliefs` | per character: `{fact_key, value, confidence, source, learned_at}`, which may be **false** |
| `simFacts` | the ground truth for every `fact_key` |
| `simEvents` | every event: `kind`, `source` (`sim`/`mirror`/`juniper`/`orion`), `effects`, `seed`, `status` |
| `simQuests` | goal condition, deadline, participants, progress, `status` (open/succeeded/failed/abandoned) |
| `simScenes` | each scene: participants, purpose, transcript, results chosen, effects applied, seed |
| `simPredictions` | Orion's prediction for a scene or quest, the resample spread, and the score |

A **tick** (Convex cron, default every 5 game-minutes) does the following. Every effect here is plain code and unit-testable:

- drifts needs (rest falls during work hours, money falls on spending days)
- moves stock (the diner uses coffee, deliveries arrive or slip)
- fires scheduled and random events from a small template table (Poisson arrivals, seeded)
- re-checks quest goal conditions and deadlines

**The entity rule.** Characters may only name places, items and people that exist in the state. A scene that names anything else is rejected and re-rolled once, then falls back to a breeze scene with no results. This replaces "please don't invent people" in the prompt with a check in code.

### 2. Events, from four sources

One event shape, one apply path (`sim/events.ts:applyEvent`). Sources differ only in who is allowed to create them and how many:

| source | how it gets in | cap |
|---|---|---|
| `sim` | tick's random/scheduled templates | template rates |
| `mirror` | world-feed adapter (§5) | ≤ 3/day |
| `juniper` | Hub form → bus `orion:town:sim:event:request` → embodiment → `sim/events:inject` | none |
| `orion` | cortex verb `town_add_event` → same bus path, `source=orion` | ≤ 5/day, plus ≤ 2 open quests |

Event body: `{kind, title, place?, effects: [{op, target, value}], quest?: {...}, expires_at?}`. `effects` uses a small closed set of operations (`stock_delta`, `need_delta`, `fact_set`, `trust_delta`, `place_flag`). Each operation is code with a test. A free-text event with no effects is allowed (it becomes a rumor fact that characters can talk about), but it can't change numbers. That stops an injected event from quietly adding a keyword with no behaviour behind it.

### 3. Quests

A quest is an event with:

- `goal`: a condition over the state, e.g. `stock(workshop.parts) >= 3 AND fact(generator.fixed) == true`
- `deadline` (game time), `participants` (characters, Orion's town body, optionally Juniper)
- `steps`: optional sub-goals, same condition language, for progress display

Effect on behaviour: open quests add **quest-step results** to the list of legal results in scenes between participants, and raise the "purpose" need of the characters involved. That makes them seek each other out. When the tick sees the goal is met, the quest succeeds. When the deadline passes, it fails. Either way it applies its reward or penalty effects (trust, money, a town-wide fact). Orion takes part through its existing town body: its conversations can pick quest-step results like anyone else's (§4b).

### 4. Scenes instead of line-by-line chat

#### 4a. NPC ↔ NPC: one call writes the whole scene

When two NPCs meet, code picks a **purpose** with seeded, needs-weighted sampling: `breeze`, `ask_favor`, `trade_news`, `persuade`, `deceive`, `quest_step`. Code also builds the **legal results list** from the current state. Example: "Cam lends Sofia 2 parts", "Nico tells Mara the shipment is cancelled (false)", "trust Sofia→Nico −0.2". One LLM call returns:

```json
{"lines": [{"speaker": "Mara Vale", "text": "..."}, ...3-6],
 "results": ["r2"],            // 0 for breeze, else 1-2 ids from the list
 "why": "one sentence"}
```

Checks before anything is applied: valid JSON, 3–6 lines, speakers alternate, the entity rule passes, the partner-echo check passes (no line too similar to the line before it), and results are ids from the list. A failure gets one retry, then a breeze scene with no results. The lines are posted into the conversation at a readable pace, so the UI looks the same as it does now. The results go through `applyEvent`.

`breeze` scenes are fine and expected. They can still move `social` need and trust by a small, fixed, code-set amount, because chatting is how people bond. Nothing else changes.

#### 4b. Anything with Orion or Juniper: live turns, results picked afterwards

Live chats stay turn-by-turn, because Orion speaks for itself through embodiment and Juniper types. Two fixes ride along:

- **Echo fix:** the history goes in with proper speaker roles (your lines as `assistant`, theirs as `user`) instead of every line as `user` with a `Name to Name:` prefix. The partner-echo check from §4a also applies here.
- **Results afterwards:** when the conversation ends, one call reads the transcript plus the legal results list and picks 0–2. Same checks, same apply path. That is how Orion moves a quest forward by talking.

### 5. Loose mirror of the real world

`orion/embodiment/town_mirror.py` (new) runs inside `orion-embodiment`, which already holds the Convex credentials. It turns real signals into town events through a **small, fixed template table**. Each template has real effects; there is no free LLM rewrite of the news:

| real signal | mirrored? | town template (example effects) |
|---|---|---|
| `WeatherCurrentV1` / `WeatherAlertV1` (heat, cold, storm, smoke) | yes | `weather_spell`: place flags, `rest`/`social` need drift, demand for drinks or firewood |
| digest item, economy/business | yes | `price_shift` or `late_shipment`: `stock_delta`, cash drain |
| digest item, science/space | yes | `rumor`: a `fact_set` some characters believe, possibly in a distorted form |
| digest item, health | mild only | `bug_going_around`: `rest` need drift |
| digest item, politics/conflict/crime | **no** | — |
| `AgendaContextV1`, Juniper's chats/location | **no** | — (private, never mirrored) |

Cap of 3 per day across both digest and weather inputs, with seeded choice among eligible items. Every mirror event carries a typed `source_ref`:

- Digest: `{kind: "digest", item_id}` from `DailyWorldPulseItemV1`, received on `orion:world_pulse:digest:item`.
- Weather: `{kind: "weather", snapshot_id}` referencing an immutable snapshot retained by the adapter in Convex. The snapshot stores the public provider URL, provider observation/alert ID when available, observation time, acquisition time, and the allow-listed weather fields used by the template. `snapshot_id` is a SHA-256 hash of the canonical provider reference, observation time and selected fields; reacquiring the same observation does not create another event.

`WeatherCurrentV1` and `WeatherAlertV1` have no `item_id` and are nested situational context, not standalone feed messages. The adapter therefore polls a public weather source for an explicitly configured **town region**, then normalizes only the fields needed by these models. It must not read Juniper's situational context or infer the region from her location. Missing provider configuration, unavailable observations or missing observation time produce no weather event. The snapshot must exist before its event can be accepted; digest references must resolve to a retained digest item. This lets an inspector trace either source without pretending weather has a digest ID. A mirrored event's title is a single LLM paraphrase into town terms, checked by the personal-terms detector (`scripts/check_personal_terms.py`) and the entity rule. If either check fails, it falls back to the template's own fixed title.

### 6. Orion predicts; the simulation scores

Before a scene runs (and when a quest opens), the simulation publishes a **prediction request**: the participants, purpose, legal results, and the relevant state and beliefs. Orion answers with a probability for each result (plus "no result"). The scene then runs.

- **Resample spread:** for predicted scenes only, the scene writer runs **N=3** extra times on different seeds, with no side effects. That gives the scene's own spread of results.
- **Score:** Brier score of Orion's probabilities against what actually happened, reported alongside the Brier score of the resample spread and of a base-rate guesser. **What counts is how Orion compares with those baselines, not the raw score.**
- Rate: at most 1 predicted scene per game-hour plus every quest. That keeps the extra LLM load bounded (§9).

Scores are published as a typed bus event (§7). Nothing trains on them in this design.

**Metric gate (CLAUDE.md §0A), recorded here as required:**

1. *Provenance:* `sim/predict.ts:scorePrediction` (new), inputs from `simPredictions` + `simScenes` rows.
2. *Independence:* this is new territory. No existing Orion metric scores predictions about other agents. It is tied to the scene writer's own randomness, which is exactly why it is reported next to the resample baseline and not alone.
3. *Theory anchor:* Brier score is a proper scoring rule, minimised only by reporting your true beliefs. The comparison against baselines follows standard forecast-skill scoring.
4. *Live sanity:* **UNVERIFIED until built.** Acceptance check A6 requires ≥ 50 scored scenes and a check that skill varies by purpose (e.g. `deceive` harder than `breeze`). If every score sits at the baseline, the signal is dead and nothing gets built on it.
5. *Existing mechanism:* `orion-world-model` is parked and untrained. `action_prediction.py` scores Orion's own actions, not forecasts of other agents. No Brier scoring exists. Field names follow `action_prediction.py`.
6. *Reversibility:* its own table and its own channel, nothing reads it yet. Removing it means dropping a table and a channel.

### 7. Bus contract and the memory wall

**One trusted bridge.** Convex endpoints in `convex/sim/` use `internalQuery` for reads, `internalMutation` for validated state changes, and `internalAction` for scene generation, live-result picking and resampling that call the LLM. Actions read a versioned snapshot through a query, perform external calls, then submit their proposed results to a mutation. That mutation rechecks the state version and legal results and commits each scene ID at most once; stale proposals are rejected without effects. Resamples never call the effect-commit mutation. The town engine and cron may invoke these internal functions. External access requires the admin key and goes through `orion-embodiment`, which holds that key. Everyone else talks to the town over the bus, and embodiment enforces the source caps:

| channel | schema | producer → consumer |
|---|---|---|
| `orion:town:sim:event:request` | `TownSimEventRequestV1` (`source`, `kind`, `title`, `effects[]`, `quest?`, `source_ref?`, `requested_by`, `correlation_id`; `source_ref` required for mirror) | Hub (Juniper's form), cortex-exec (Orion's outside tool), town_mirror → embodiment |
| `orion:town:sim:event:result` | `TownSimEventResultV1` (`accepted`, `reason`, `event_id`) | embodiment → Hub, cortex-exec |
| `orion:town:sim:scene:resolved` | `TownSimSceneResolvedV1` (participants, purpose, legal results, chosen results, effects applied, seed) | embodiment (polls `simScenes`) → sql-writer |
| `orion:town:sim:prediction:scored` | `TownSimPredictionScoreV1` (probabilities, outcome, `brier`, `brier_resample`, `brier_base_rate`, purpose) | embodiment → sql-writer (later: self-calibration) |

**Where Orion's predictions come from.** Embodiment polls `simPredictions` for open requests. It asks cortex for a forecast the same way it already asks for speech (`speech.py` pattern, new verb `town_forecast`, quick lane, never `chat`), then writes the probabilities back. The simulation scores them once the scene resolves.

**Orion's outside tool.** A cortex-exec verb `town_add_event` publishes `TownSimEventRequestV1` with `source=orion`. Open question for PR 4: which of Orion's loops calls it, the curiosity/agent lane or the background embodiment path that already emits intents. Start with the background path, because it already exists.

**Memory wall.** None of these events is a chat turn. They never reach `orion:chat:social:turn`, crystallization or recall. The sql-writer stores them in new `town_sim_*` tables, and `orion_readonly` gets no grant on those tables. Scene transcripts still flow through the existing `aitown`-tagged path. The prediction *scores* are about Orion's own skill, so a later calibration reader may use them. The town content itself stays behind the wall.

### 8. Characters: cards get numbers

`town_cards.yaml` gains a `sim:` block per character, read by `generate_descriptions.py` and written into the seed state: starting needs, schedule, home/work place, starting stock and cash, purpose weights (e.g. Nico has a high `deceive`/`persuade` weight, Mara a high `trade_news` weight), and a lying rate. The existing prose stays and becomes the voice guidance for the scene writer. The poetic phrases that currently feed the metaphor spiral ("emotional weather", "rumor paths") are kept in the bio but no longer sent as goals.

### 9. Model and load

- Scene writer and result picker: `quick_background` (circe-worker-fast-1, Qwen3-8B), never `chat` (see README). One scene call replaces ~8–10 line calls, so the town's normal load *drops*. The resampling for predictions adds ≤ 3 calls per game-hour.
- Eval A5 compares `quick_background` against the `agent` lane (27B) on 30 fixed scene setups. If the 8B fails the checks too often, move only the scene writer, behind a Convex env var.

---

## Proposed schema / API changes

- **Convex (new, overlay):** the tables in §1 plus `simMirrorSnapshots` for weather provenance, and internal functions `sim/events:inject`, `sim/tick:run` (cron), `sim/scenes:write`, `sim/scenes:resolveLive`, `sim/predict:open|submit|score`, `sim/state:snapshot`. LLM orchestration uses internal actions; state reads and commits use internal queries/mutations (§7). Add an internal snapshot-ingest mutation for the trusted weather adapter.
- **Bus (new):** the four channels in §7, in `orion/bus/channels.yaml`.
- **Schemas (new):** `orion/schemas/town_sim.py`, with `TownSimEventRequestV1`, `TownSimEventResultV1`, `TownSimSceneResolvedV1` and `TownSimPredictionScoreV1`. All are `extra="forbid"` and registered in `orion/schemas/registry.py`.
- **Cortex verbs (new):** `town_forecast`, `town_add_event`.
- **Postgres (new, manual migration):** `town_sim_scene_resolved`, `town_sim_prediction_score` (header `ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`).
- **Env (new, shipped ON):** in `orion-embodiment`: `EMBODIMENT_TOWN_SIM_ENABLED=true`, `EMBODIMENT_TOWN_ORION_EVENTS_PER_DAY=5`, `EMBODIMENT_TOWN_ORION_OPEN_QUESTS_MAX=2`, `EMBODIMENT_TOWN_MIRROR_PER_DAY=3`, plus `EMBODIMENT_TOWN_WEATHER_PROVIDER_URL` and `EMBODIMENT_TOWN_WEATHER_REGION` (both empty until an operator selects a public source and town region; weather polling is disabled while either is empty). In Convex env: `SIM_SCENE_LLM_MODEL=quick_background`, `SIM_PREDICT_RESAMPLES=3`.
- **Cards:** a `sim:` block per character in `town_cards.yaml` / `.local.yaml`.

## Files likely to touch

- `services/orion-ai-town/overlay/convex/sim/*.ts` (new): the sim core (pure, no Convex imports: `state.ts`, `effects.ts`, `events.ts`, `quests.ts`, `scenes.ts`, `checks.ts`, `predict.ts`) plus thin Convex wrappers
- `services/orion-ai-town/overlay/convex/sim/*.test.ts` (new): `node --test` unit tests for the core
- `services/orion-ai-town/scripts/apply_upstream_patches.sh`: copy `overlay/` into `upstream/` after the patches
- `services/orion-ai-town/patches/orion-sim-hooks.patch` (new): schema registration, cron, `agentDoSomething` → sim purpose, NPC↔NPC → scene writer
- `services/orion-ai-town/patches/orion-live-turn-roles.patch` (new, PR 1): speaker roles + partner-echo check in `conversation.ts`
- `services/orion-ai-town/cards/town_cards*.yaml`, `scripts/generate_descriptions.py`: `sim:` block → seed state
- `services/orion-ai-town/tests/`: patch/overlay presence tests, the `c:578946` echo fixture
- `services/orion-ai-town/evals/scene_writer_eval.py` (new): A5
- `orion/embodiment/town_sim_bridge.py`, `orion/embodiment/town_mirror.py` (new); `services/orion-embodiment/app/worker.py`, `settings.py`, `.env_example`
- `orion/schemas/town_sim.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-cortex-exec`: verbs `town_forecast`, `town_add_event`
- `services/orion-hub`: "Add town event / quest" form → bus
- `services/orion-sql-writer`, `services/orion-sql-db/manual_migration_town_sim_v1.sql`
- `services/orion-ai-town/README.md`

## Non-goals

- No RL training of NPCs or of Orion in this pass. Scores are stored, nothing learns from them.
- No chaos/Lyapunov machinery beyond seeded replay and per-scene resampling.
- No change to the memory wall. Town content stays out of `memory_crystallizations`, recall and curiosity study material.
- No mirroring of Juniper's private life (calendar, chats, location).
- No new service. The simulation lives in the town's own Convex, and the bridges live where town bridges already live.
- No change to the map, sprites or movement engine.
- No free-form world editing by Orion: closed effect operations, daily caps.

## Acceptance checks

- **A1 (state moves):** with LLM calls stubbed, a 24 game-hour seeded run changes stock, needs and trust. Replaying the same seed gives the exact same state (unit test on the pure sim core).
- **A2 (no ghosts):** across 100 generated scenes, 0 applied scenes name an entity that isn't in the state (the check rejects them first; the rejection rate is logged).
- **A3 (no echo):** across 100 scenes plus 20 live NPC↔Orion turns, 0 posted lines exceed the partner-echo threshold. The regression fixture uses the real 2026-09-17 transcript `c:578946`.
- **A4 (things happen):** in a live 24h run, ≥ 30% of non-breeze scenes apply a result; ≥ 1 quest resolves (succeed or fail); ≥ 1 false belief spreads to a second character.
- **A5 (scene writer quality):** fixed 30-setup eval: JSON validity ≥ 95%, check pass rate ≥ 85% on the chosen lane, with the 8B vs 27B comparison recorded.
- **A6 (prediction signal):** ≥ 50 scored scenes. Orion's Brier score is reported against both baselines and broken down by purpose. Verdict recorded: signal / no signal.
- **A7 (injection):** a Juniper event and an Orion event each create an `simEvents` row with the right `source`. Orion's 6th event of the day is refused with a reason.
- **A8 (wall holds):** after the 24h run, no new `memory_crystallizations` rows from the town and no town rows readable by `orion_readonly`.
- **A9 (mirror):** in 24h, ≤ 3 `mirror` events across both input types, each with a resolvable `source_ref`, none containing personal terms (`check_personal_terms.py`). Fixtures cover digest-item and weather-snapshot provenance, duplicate observations, missing provider configuration and failed fetches. A weather event must resolve to its retained public observation; the adapter must never read private situational context. With weather configured, the live smoke must trace at least one event to each source type; otherwise record the weather rail as `UNVERIFIED`.

## Risks

- **The 8B can't write valid scenes reliably.** Mitigation: the check-and-retry path, breeze fallback, and the A5 lane comparison.
- **Patch maintenance.** Upstream is gitignored; new code ships as overlay files plus small hook patches. Mitigation: keep the hooks small and the sim core free of Convex imports so it can be tested in CI with `node --test`.
- **The predictions measure the scene writer's quirks, not social reasoning.** Mitigation: the resample baseline, the per-purpose breakdown, and the dead-signal verdict in A6.
- **Mirror leaks private or sensitive news into a toy world.** Mitigation: feed categories allow-list, personal-terms gate, daily cap.
- **Orion spams quests.** Mitigation: caps plus the closed effect set.

## Missing questions

- Which public weather provider and town region should the operator select for PR 5? Weather remains disabled until these are explicit; digest mirroring can proceed independently.
- Which existing Orion loop should call the outside event tool in PR 4? Start with the background embodiment path (§7), and record the final caller before wiring autonomy.

## Recommended next patch

Start with the dialogue echo fix and live-turn roles. Verify the partner-copy regression before changing simulation state. The remaining build order is:

### Recommended implementation order

1. **PR 1 — echo fix and live-turn roles (§4b first bullet).** Small. Fixes the worst live symptom on its own.
2. **PR 2 — sim core:** tables, tick, events (`sim`/`juniper` sources), cards → numbers, entity rule. A1, A7 (Juniper half).
3. **PR 3 — scenes:** NPC scene writer, result picker for live chats, quests. A2–A5.
4. **PR 4 — Orion outside:** Orion's event/quest tool, prediction requests and scoring, bus contract. A6, A7, A8.
5. **PR 5 — mirror adapter.** A9.

Operator step before any live check: the town is `stoppedByDeveloper` and needs `testing:resume`, plus a world wipe for the new seed state (README "Fresh game / reset").
