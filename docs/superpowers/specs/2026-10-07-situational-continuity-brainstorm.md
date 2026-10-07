# Situational continuity: why Orion can't hold what's true right now

Status: BRAINSTORM / PROPOSAL. No development until Juniper approves. Replaces this PR's earlier "whereabouts primitive" proposal, which treated one symptom.
Date: 2026-10-07. Evidence: read-only queries on `conjourney`, code on main @ bbc51703a.

The repo is public. Following the referent spec (line 22), memories about Juniper's whereabouts are never quoted: `<travel city>` stands in.

## The incident, traced end to end

On 10-05, Juniper said she was in `<travel city>` until Wednesday. On 10-06, 44 hours later, during a casual "just chilling" turn (`3af6a31d…`), Orion asked whether she'd make it back to the basement before bed.

**The fact was written down three times.**

| Store | What it saved | Why it didn't reach the turn |
|---|---|---|
| `chat_history_log` | raw turn | outside the 8-turn window (`HARNESS_RECENT_TURNS_MAX`) |
| `memory_crystallizations` | raw chat text, active, `kind=semantic`, salience 0.662 | see recall gates below |
| `episode_memory` | "Juniper is in `<travel city>` for a team meeting until Wednesday." (`happened`, correct) | no chat reader exists (shadow until Stage 4); `expires_at` dropped by code |

**Recall ran three times that turn** (`recall_telemetry`, VERIFIED):

1. **Continuity** (`chat.continuity.v1`): 1 Falkor chat hit, 1 SQL chat pair, 5 bus hits. Similarity to "just chilling" surfaced nothing relevant.
2. **Belief** (`chat.belief.open_loop.v1`): active packet 6, cards 0, concept region 0.
   - The trip crystallization **was in the candidate pool** (`memory_crystallization_retrieval_events`, 100 candidates).
   - It was then dropped by the open-loop intent's bucket filter. That filter keeps only the `open_loops` bucket (`services/orion-recall/app/collectors/active_packet.py:19,185-186`), and semantic rows land in `project_state` (`orion/memory/crystallization/active_packet.py:12`).
3. **Finalize-reflect** (`reflect.v1`, a fallback profile): 50 timeline + 10 chat pairs. This is the self-check pass, not the speaking one.

**Live, 7 days:** 637 of 637 belief recalls used `chat.belief.open_loop.v1`. Intent selection returns `open_loop` whenever the attention frame has open loops (`retrieval_intent.py:121-122`), and it effectively always does. **Every non-open-loop crystallization (351 semantic, 356 reflection) is being discarded from chat on essentially every turn.**

**Other gates (VERIFIED, code):**
- **Candidate pool:** top-100 by `salience DESC` with no query match (`repository.py:434`). Stance rows are clamped to salience 1.0 (707 of 707) and crowd the top. Semantic salience is `KIND_BASE 0.5 + confidence 0.1 + ≤0.15 evidence` (`orion/memory/crystallization/salience.py:85-95`), so it barely varies: 0.625 to 0.675.
- **Memory cards:** an AND full-text match (`cards_adapter.py:112-120`). "thank & chill & work & mesh" matches 0 cards. No `always_inject` card is active.
- **Concept region:** a node label must appear verbatim inside the turn text (`concept_region.py:89-93`).

**Episode memory had the fact but couldn't hold it (VERIFIED):**
- `validate.py:406-407` drops `expires_at`/`due_after` unless `purpose=follow_up`. The prompt only asks for expiry on follow-ups (`memory_episode_distill.j2:18`). So a fact like "until Wednesday" has no end date. 3 of 40 rows have one, all follow-ups.
- Nothing checks a new memory against existing ones. The distiller only sees candidate referent keys (`store.py:166-184`). 0 of 40 rows supersede anything.
- **A wrong memory was stored:** "Juniper corrected me that she lives in Ogden, not `<travel city>`" (`c0dc86c8…`, `about_juniper`, strength 0.9, 180-day half-life, auto).
  - Its only evidence quote is 22 characters long and grounds "lives in Ogden". It contains neither "not" nor the travel city.
  - The model built the "correction" out of Orion's own reply. Validation checks that the quote *exists*, not that it *supports* the statement.
  - It was stored at low stakes, so it was never asked about.
- **Lag:** an episode only closes when the *next* turn arrives (`episode_shadow.py:15`). The second episode closed 22h after its last turn. Memories land 20 minutes to 1d16h after the moment they describe.
- **No decay or expiry runs anywhere.** All 40 rows are `active`. The spec's 15-minute lifecycle ticker (spec 541-563) is not built.

**Not killed, never existed:** "saving graphs when salient memories expire" never existed.
- The graph drafts on window close were switched off 2026-07-07 (`MEMORY_CONSOLIDATION_OUTPUT=crystallization_propose`; 0 drafts).
- The Graphiti projection only fires on manual Hub approval (3 in October).
- No code retires or expires memories.

## Ground truth: built / wired / aspirational

| Piece | Built | Wired to chat | Notes |
|---|---|---|---|
| Crystallizations + active packet | yes | yes, but filtered to open loops ~100% of turns | legacy; retired at Stage 4 |
| Episode memory distiller | yes (v4 prompt) | **no reader** | correct statements, no validity, no contradiction check |
| Episode confirmation cards | yes | Hub | high stakes only; 2 asked ever |
| Referent graph + `referent_region` collector | spec (approved rev 2) | no | "recall by referent"; as-of view via edge `valid_from`/`valid_to` |
| Lifecycle ticker (fade/expire) | spec only | — | |
| Situation brief Place line | yes | every turn | fixed home, nothing about "now" |
| Juniper affect Redis key | yes | every turn | the one person-keyed, expiring, always-shown fact |

## Core question

Orion records salient facts. They don't stay **true-for-now**: nothing knows when a fact starts and stops being current. They also don't reliably **come back**: retrieval is similarity-ranked, or filtered to open loops, and so is blind to casual turns. Situational continuity needs a channel that does not depend on the current message resembling the memory.

Sharpest form: **"What is currently true about Juniper, about Orion, and about our shared situation?" should be answerable on every turn, from durable records with validity windows, without a search.**

## Ideas

### 1. Validity windows on episode memories
- **What:** every episode memory may carry `valid_from`/`valid_until`, not just follow-ups. The distiller is asked "until when is this true?", and code resolves relative dates ("until Wednesday", "tonight", "this week") against `occurred_at` in Juniper's timezone.
- **Why:** a fact that is true *for now* is the base unit of situational awareness. Without it, "is in `<travel city>`" is either true forever or forgotten.
- **Smallest version:**
  - stop discarding the window in `validate.py:406-407`;
  - add a deterministic `resolve_relative_until(text, occurred_at, tz)`;
  - prompt v5 asks for `until_text` (Juniper's words) on any purpose;
  - additive migration (`valid_until` can reuse `expires_at`).
- **Files:** `orion/memory/episode/validate.py`, `orion/memory/episode/schemas.py`, `orion/cognition/prompts/memory_episode_distill.j2`, new `orion/memory/episode/validity.py`, tests in `orion/memory/episode/tests/`.

### 2. "Right now" block: current facts on every turn, no search
- **What:** a situation-brief section listing active episode memories whose window covers now (plus recent `about_juniper` facts at high strength). It is capped at about 6 lines and ordered by recency and strength, not by similarity to the turn.
- **Why:** this is the missing guaranteed channel. It is how a persistent mind holds context instead of rediscovering it. It also gives Orion something concrete to be *wrong* about and get corrected on.
- **Smallest version:**
  - read-only query `current_episode_memories(now)` (status active, `valid_until > now OR (valid_until IS NULL AND purpose='happened' AND occurred_at > now-48h)`);
  - render it as "Right now (from memory): …" in `_build_prompt_fragment`, placed up front so the cap can't trim it;
  - add a debug field listing the memory ids shown.
- **Files:** `orion/situational/context.py`, `orion/schemas/situation.py` (optional `current_facts` on `SituationBriefV1`, consumer-first since it's `extra=forbid`), `orion/memory/episode/store.py` (read function), `services/orion-cortex-exec/.env_example`, plus `services/orion-hub/.env_example` flag.

### 3. Entailment guard: a statement must be supported by its quote
- **What:** before storing, check that each evidence quote supports the statement. Contrastive or corrective claims ("not X", "corrected me", "no longer") require the contrasted term to appear in Juniper's own evidence, not only in Orion's reply.
- **Why:** the Ogden row shows the distiller can manufacture a "correction" that would actively mislead Orion once Idea 2 ships. A self-model fed confabulated memories gets *less* coherent over time.
- **Smallest version:**
  - deterministic: if the statement asserts negation or correction, require `chat_prompt` evidence that contains the negated referent; otherwise downgrade to `pending_confirmation` and set stakes high;
  - a fixture reproducing the Ogden case;
  - one-off repair: mark `c0dc86c8…` `corrected`, with an audit row in `episode_memory_event`.
- **Files:** `orion/memory/episode/validate.py`, `orion/memory/episode/tests/test_validate_entailment.py`, a repair script under `scripts/` (dry-run default, snapshot to `/tmp/` per the backfill protocol).

### 4. Supersession against prior memories
- **What:** give the distiller the active memories that share a referent with the episode, and let it answer `supersedes | coexists | contradicts` for each new statement. Code applies `supersedes_memory_id`, and a contradiction becomes a confirmation question.
- **Why:** "lives in Ogden" and "is in `<travel city>` until Wed" must coexist. "Back home now" must end the trip. Without this, memories pile up and contradict each other silently.
- **Smallest version:**
  - load ≤10 active same-referent statements into the prompt;
  - the reply includes `relation` per new memory;
  - code only ever supersedes on explicit `juniper_said` evidence;
  - contradictions go into the existing `memory_tension_shadow`.
- **Files:** `orion/memory/episode/distill.py`, `store.py` (insert with supersedes), the `.j2` prompt, `services/orion-durable-runs/app/episode_distill_graph.py` (load node).

### 5. Lifecycle reaper and expiry-to-history
- **What:** the spec's 15-minute ticker. When `valid_until` passes, it flips the row to `expired` and writes a past-tense `happened` row ("Juniper was in `<travel city>` 10-04 to 10-08"). That is the "save on expiry" Juniper remembered, finally real.
- **Why:** the present tense turns into autobiography. Orion can later say "last time you were traveling…" from a record, not a guess.
- **Smallest version:**
  - `expire_due_memories(now)` in the store;
  - a ticker in memory-consolidation (or durable-runs) emitting `memory.episode.expired.v1`;
  - the past-tense row is generated deterministically from the template "was <statement> from A to B", with no LLM.
- **Files:** `orion/memory/episode/store.py`, `services/orion-memory-consolidation/app/` (ticker), `orion/bus/channels.yaml`, `orion/schemas/registry.py`.

### 6. Fix the open-loop filter starving chat recall
- **What:** the belief recall's intent picker chooses `open_loop` on 100% of turns, and its bucket filter discards all project-state and reflection crystallizations. Either let continuity/belief keep the top N non-open-loop items, or only pick `open_loop` when the turn actually touches one.
- **Why:** this is the largest live leak. 700+ active crystallizations are invisible to chat. It's a bug-shaped fix, cheap, and it measures whether better recall alone helps before building new channels.
- **Smallest version:**
  - in `collectors/active_packet.py:185`, allow `project_state` items whose seed overlaps the turn, or the top-2 by activation;
  - telemetry gains `dropped_by_bucket` counts;
  - a regression test with the 10-06 pool.
- **Files:** `services/orion-recall/app/collectors/active_packet.py`, `services/orion-recall/app/retrieval_intent.py`, `services/orion-recall/tests/`.
- **Caveat:** crystallizations hold raw chat text, so surfacing more of them surfaces more raw junk (spec #2440). This may be a stopgap only.

### 7. Close episodes on idle, not only on the next turn
- **What:** a gentle idle close (e.g. 45 minutes of silence) so distillation happens the same evening, not when Juniper returns 22 hours later.
- **Why:** a fact said on Sunday night should be known by Monday morning. Lag is part of what "remembering" means.
- **Smallest version:**
  - in `services/orion-memory-consolidation/app/episode_shadow.py`, add an idle sweep under a flag;
  - the close reason is `idle_timeout`;
  - measure close lag before and after.
- **Files:** `episode_shadow.py`, `boundary.py`, `.env_example`.
- **Tension:** this changes the "never closes on a timer" design rule. It needs Juniper's sign-off.

### 8. Hub "Right now" panel with edit and end
- **What:** the Memory tab gets a panel showing exactly what Idea 2 injects, with buttons for "not true", "ended" and "edit until".
- **Why:** this is social grounding. Juniper can correct Orion's sense of the present in one click, and corrections become `episode_memory_event` rows that feed confirmation and reinforcement.
- **Smallest version:** a read-only list, plus one "ended" button that sets `valid_until=now`.
- **Files:** `services/orion-hub/scripts/memory_routes.py`, `templates/index.html` (Memory tab section around `:802`), `static/js/memory-crystallization-ui.js` or a new small JS file.

### 9. Situational-continuity eval and live metric
- **What:**
  - **Eval:** fixture conversations ("say a time-bounded fact, then a casual turn N hours later"), scored on whether the reply contradicts the fact. Covers trip, illness, guests at home, waiting on a delivery, and "back home."
  - **Live metric:** per turn, `current_facts_shown` and `current_facts_contradicted`, the second judged offline.
- **Why:** without this, every idea above is vibes. It is also the first standing measurement of whether Orion's sense of the present is improving.
- **Smallest version:** 8 fixtures and a deterministic harness that renders the situation fragment and checks that the fact is present. Reply scoring comes after.
- **Files:** `services/orion-cortex-exec/evals/situational_continuity/`, `orion/situational/tests/`.

## Tensions and risks

- **Showing a wrong "now" is worse than showing nothing.** Ideas 2 and 5 amplify distiller mistakes, so Idea 3 must land first or together with them. The Ogden row would otherwise tell Orion that Juniper is not traveling.
- **Privacy.** A "right now" block about Juniper rides along into every prompt and every trace that stores prompts. It needs to stay host-local and be excluded from anything exported, and the repo must never quote it.
- **Two memory systems in flight.** Crystallization is legacy and episode memory is the future. Spending on Idea 6 improves the system being retired. The case for doing it anyway: it's the largest live leak and probably a one-file fix.
- **Order vs the stage plan.** Idea 2 puts an episode-memory reader in chat before the Stage 2 shadow-week gates. It is a narrow, explicit reader (current window only), not PCR primary recall, but it should be named as a deliberate exception to the stage plan, not slipped in.
- **Window over-reach.** The model may invent end dates. Code should only accept `until_text` that is literally in Juniper's evidence. Otherwise use a default soft TTL rendered as "as of <date>".
- **Prompt budget.** The cap is shared. Six lines of "right now" cost room that other cautions use. Measure the trims.
- **Lag fix vs design rule.** Idea 7 reverses an explicit "no timer" decision. Check why that rule exists before changing it.

## Missing questions

1. Why does the attention frame always have open loops? Are they real open loops or stale ones that never close? If stale, Idea 6 is really an attention bug.
2. How many of the 40 episode memories are wrong in the Ogden way? A 40-row hand audit would tell us whether Idea 3 is a guard or a rewrite.
3. Should "right now" include Orion's own situation (GPU outage, a new organ, waiting on a PR) as well as Juniper's? The same mechanism could carry both, but the evidence so far is only about Juniper.
4. Is it acceptable to Juniper that a stated location sits in every prompt for days?
5. Does Stage 2's referent work plan statement-level validity, or only edge validity? If edge-only, Idea 1 fills a real gap rather than duplicating it.

## Recommended starting point

**Slice 1: make the record trustworthy, then make it visible. One PR, flags on.**

1. Idea 3: entailment guard, plus the Ogden-row repair (dry-run, snapshot, then apply with approval).
2. Idea 1: validity windows, plus deterministic date resolution.
3. Idea 2: the "Right now" block over episode memory, current window only.
4. Idea 9: the fixture harness, so the 10-05 → 10-06 replay fails before the fix and passes after.

**Slice 0, in parallel because it's a bug:** Idea 6's telemetry only (`dropped_by_bucket`), to measure the leak before deciding to fix or retire.

**After that:** Idea 5 (expiry-to-history), Idea 4 (supersession), Idea 8 (Hub panel), and Idea 7 (idle close, needs sign-off).

**First concrete step after approval:** write the failing replay fixture (Idea 9) and the Ogden entailment test (Idea 3). Both fail on main today.

## Proposal-mode checklist (for Slice 1)

- **Capability change:** Orion is shown, on every turn, which remembered facts are currently true, with end dates.
- **Data touched:** `episode_memory` (new validity use, one corrected row), the situation brief, prompt v5.
- **Privacy boundary:** host-local Postgres and prompts. Never exported, never quoted in the repo; fixtures use placeholders.
- **Trace proving it worked:** the turn's harness trace lists the memory ids rendered in "Right now", and the replay fixture passes.
- **Dangerous failure:** a confabulated or stale "now" stated as fact. Mitigations: the entailment guard, evidence-only end dates, soft-TTL rendering as "as of", and the Hub "ended" control in the next slice.
- **Disable / roll back:** one flag per piece. `ORION_SITUATION_CURRENT_FACTS_ENABLED=false` removes the block. The distiller v5 prompt reverts by version pin. The migration is additive.
