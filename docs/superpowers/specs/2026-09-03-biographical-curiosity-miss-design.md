# Biographical curiosity miss — celebrate without probing

**Date:** 2026-09-03
**Mode:** design / proposal (AGENTS.md §0A)
**Status:** Ready for implementation planning
**Incident turn:** `correlation_id=bfe0e731-f5fd-40bd-8bbc-ae4366c0f074` (2026-09-03, Hub chat)

**Problem:** Juniper shared a major life milestone (promotion to Team Lead while keeping AI/ML Architect / FDE work). Orion celebrated warmly and then explicitly refused to ask anything. The system graded that refusal as a perfect alignment. Curiosity machinery *noticed* the novelty and still produced no probe.

---

## Arsonist summary

Orion has two curiosity systems that do not cover this case.

1. **Relational curiosity** was built to stop solutioning on hard / presence-seeking turns. It celebrates and leaves space. It has no "good news that revises my model of you — ask one grounded thing" path.
2. **Worldview / investigation curiosity** was built so Orion can test claims about *its own* stores and substrate. Live priors are about crystallization gates and Atlas prediction-error territory. None are about Juniper's job, role, or life. Situation injection of those priors is color about Orion, not a reason to ask Juniper anything.

On this turn the attention frame *did* extract `Team Lead` as a novel `plan` loop (salience 0.562), then parked it at `watch` because `min_ask=0.65`. Stance invented `celebrate_promotion` / `leave_space_without_offer` and **dropped** `situated_curiosity`. Speech obeyed. Finalize praised the obedience.

The miss is not warmth. The miss is: **there is no primitive whose job is "someone I know just changed the shape of their life; get curious about the content of that change."**

---

## Current architecture

### What the incident turn actually did (live evidence)

Queried against conjourney Postgres + `orion_worldview` FalkorDB for `bfe0e731-f5fd-40bd-8bbc-ae4366c0f074`.

| Layer | What fired | Outcome |
|-------|------------|---------|
| Attention frame (`ORION_CURIOSITY_FRAME_ENABLED=true`) | Two `attention_salience_trace` rows: `Team Lead for 3-5 directs`, then `Team Lead`; `target_type=plan`; salience **0.562**; `why_it_matters` = novel / unresolved | Score below `min_ask` (0.65) → `watch`, not `ask` |
| Thought stance (`thought_decision.stance_harness_slice`) | `task_mode=playful_exchange`, `interaction_regime=relational`, `answer_strategy=grounded_witness_and_presence` | Priorities: `celebrate_promotion`, `validate_dual_role`, `stay_present`, `avoid_task_tracking`. **No `situated_curiosity`.** Closing move: `leave_space_without_offer` |
| Speech / FCC draft | Explicit "No advice asked for, no next steps needed" | Rhetorical "Architect *and* Team Lead?" is not a probe |
| Finalize | Alignment notes: "Draft matches imperative… Avoids task-tracking… Relational frame preserved" | Miss graded as success |
| Worldview priors | Live pool: crystallization-gate bias, intake routing, Atlas PE territory, substrate isolation | Zero Juniper-career priors |
| Crystallization | Same turn written as `proposed` stance | Not a prior revision |

### Primitives that exist (cross-reference)

| Primitive | Where | What it is for | Why it did not help here |
|-----------|-------|----------------|--------------------------|
| In-turn attention / curiosity frame | `ORION_CURIOSITY_FRAME_ENABLED`, `current_turn_llm_signals.py`, `attention_frame.py`, `select_actions` | Extract novel phrases; score open loops; optionally select `ask` | Found novelty; scored 0.562 &lt; 0.65 → `watch`. Even an `ask` would use plan template: "unresolved constraint around Team Lead?" |
| Stance `situated_curiosity` | `chat_stance_brief.j2`, `chat_general.j2`, `compile_speech_contract` | Ask 1–2 grounded questions on relational connection-seek turns | Synthesizer dropped the tag; inventing celebrate tags instead. Contract only injects "Ask one grounded question…" when the tag is present |
| Companion closing moves | `end_with_a_wondering`, `leave_space_without_offer`, `ground_observation`, `be_with_silence` | Positive close for relational turns (v2) | Chose leave-space (hard-news / no-probe posture) for good news |
| Worldview `:Prior` + situation injection | `orion/curiosity/`, `ORION_SITUATION_CURIOSITY_ENABLED` | Orion's claims about its world, shown as chat color | All priors are Orion-machinery; situation line is flavor, not a probe driver. Sentience README §15b still claims graph does not reach chat — **stale**; Hub situation path does |
| Curiosity investigation loop | Hub tick, 4h cooldown, daily cap | Unsolicited time to test Orion's own priors | Clock-triggered; cannot interrupt this turn; wrong object |
| Endogenous / frontier curiosity | `endogenous_curiosity.py`, frontier evaluator | Substrate PE / repair / open loops → concept_graph seeds | Explicitly not self/relationship zone |
| Chat prediction error | `chat_prediction_error()` | Diff turn pressure hints vs prior turn | Not "this fact revises my model of Juniper" |
| World-pulse curiosity | `orion-world-pulse` | External coverage gaps + bus-synaptic surprise | Not biographical |
| Agent-lane curiosity hint | `curiosity_hint.py` | Advisory line on agent REPL | Not Hub chat |
| Social-memory curiosity float | `social_memory` schema | Room-tone dial | Not a probe generator |
| Identity policy | `orion_identity.yaml` | Allows situated curiosity on relational turns when stance selects it | Stance did not select it |

### Structural choke points (name them)

1. **`select_actions` / `min_ask`** in `orion/substrate/attention/policy.py` — absolute cutoffs vs Borda scores that cluster near 0.5 (flagged 2026-07-31, still uncalibrated).
2. **`compile_speech_contract`** in `services/orion-cortex-exec/app/chat_stance.py` — ask instruction gated solely on `situated_curiosity` in priorities.
3. **Stance synthesizer vocabulary** in `chat_stance_brief.j2` / `stance_react.j2` — `connection_seek` vs instrumental only; no world-model-update / celebration+curiosity dimension. Sanctioned closing moves exist; nothing prefers `end_with_a_wondering` on good-news novelty.
4. **Stance brief attention rule** — if `selected_action` ≠ `ask`, prefer `curiosity:watch` and do not force a question. That vetoes relational curiosity even when the loop is a novel life fact.
5. **`question_for`** in `orion/substrate/attention/questions.py` — plan/activity/anomaly templates only; no role / people / dual-hat / "how does this land" shapes.
6. **Prior authorship** — only the investigation loop writes `:Prior` nodes, and only about Orion's own research objects. Chat cannot form a Juniper-prior when a durable biographical fact arrives.

### Related prior art (do not reinvent)

- `docs/superpowers/specs/2026-06-26-orion-relational-stance-design.md` — anti-solutioning / companion presence. This miss is the *complement*: anti-solutioning without curiosity.
- `docs/superpowers/specs/2026-06-30-orion-relational-stance-v2-design.md` — `companion_closing_move`, late speech contract. Closing move chosen wrong; contract incomplete.
- `docs/superpowers/specs/2026-08-26-orion-priors-and-worldview-design.md` — priors as falsifiable claims. Object of claims is Orion's world, not Juniper's life.
- Conversational anti-slop rule (`.cursor/rules/conversational-behavior-anti-slop.mdc`) — **no** keyword lists for life events; wire through stance → enforce → speech; test through `enforce_chat_stance_quality` / chat_general path.

---

## Missing questions

1. **Should Juniper-priors live in `orion_worldview` or a separate graph/label?** Same graph with `about=juniper` (or equivalent) keeps overlay mechanics; separate graph keeps autonomy boundary cleaner. Needs a call.
2. **Is celebration+curiosity a new `interaction_regime` value, a new response_priority tag, or both?** Smallest cut is a sanctioned priority tag + closing-move preference without a new regime enum.
3. **May chat write priors in-turn, or only propose them for the investigation loop?** Writing in-turn is the only way the *next* mention becomes a test rather than another first sighting. Privacy: Juniper-priors are about Juniper; they must stay inspectable and deletable.
4. **Does `min_ask` recalibration happen in the same patch as stance vocabulary, or first as a measured fix alone?** Live rows now exist (`scope=chat` in `attention_salience_trace`); recalibration has data.
5. **Hard news vs good news:** `leave_space_without_offer` must remain correct for grief / overload / explicit no-probe. How does stance distinguish without keyword cathedrals? Prefer semantic inference over whole-turn meaning (same as `connection_seek`), not phrase lists.

---

## Proposed schema / API changes

### Thin cut (recommended first patch)

No new bus channels. No new services.

1. **Stance vocabulary (prompt + contract, existing fields only)**
   - Teach stance that high-information biographical updates (role change, family, health, major project shift — inferred semantically, not keyword-matched) are a distinct posture from hard connection-seek and from instrumental work.
   - On that posture: keep celebration / presence; **require** `situated_curiosity` (or add one sanctioned tag, e.g. `biographical_curiosity`, if `situated_curiosity` must stay reserved for connection-seek); prefer `companion_closing_move=end_with_a_wondering`.
   - Hazards stay: `avoid_task_tracking`, `avoid_next_steps`, `avoid_transactional_closers`. Curiosity ≠ advice.

2. **`compile_speech_contract`**
   - When regime is relational **and** priorities include situated/biographical curiosity: keep "Ask one grounded question from this thread."
   - When closing move is `end_with_a_wondering`: ensure that instruction wins over "Leave space. Do not close with an offer."

3. **Attention frame policy (measured, not vibe)**
   - Recalibrate `min_ask` (and/or watch band) against live `attention_salience_trace` `scope=chat` distribution so novel life-role loops at ~0.56 can become `ask` when askability is high.
   - Soften the stance-brief rule: `selected_action=watch` must **not** veto situated curiosity when the open loop is novel and not `already_known`.

4. **Tests (required)**
   - Fixture: promotion-shaped user message + synthetic stance brief that *should* carry curiosity → assert `enforce_chat_stance_quality` / `compile_speech_contract` emit ask instruction and do not force leave-space alone.
   - Fixture: grief / explicit no-probe → leave-space still wins; no question pile-on.
   - Regression: connection-seek venting without milestone still gets hold_space without advice.

### Deeper cut (follow-on; proposal-mode cognition)

5. **Juniper-model priors** — claims about Juniper's world (career, role, people, projects) with confidence/status, formable from chat, testable later. Same prior shape as `orion_worldview`; different *about*. Privacy boundary explicit; inspectable; no keyword ontology of life events.

6. **In-turn prior formation** — chat may write or enqueue a prior when a durable biographical fact arrives, so the next mention is a test.

7. **Life-event question templates** in `question_for` — role / people / dual-responsibility / "how does this land," selected by `target_type` or stance tag, not by regex on "promotion."

8. **Live-signal trigger for investigation loop** (already named O2 gap) — orthogonal; does not fix this turn class.

---

## Files likely to touch

### Thin cut

- `orion/cognition/prompts/chat_stance_brief.j2`
- `orion/cognition/prompts/stance_react.j2`
- `orion/cognition/prompts/chat_general.j2` (only if speech contract alone is insufficient)
- `services/orion-cortex-exec/app/chat_stance.py` (`compile_speech_contract`, possibly relational upgrade path)
- `orion/substrate/attention/policy.py` (`min_ask` / select_actions — after live distribution check)
- `orion/cognition/prompts/chat_stance_brief.j2` attention-frame instructions (watch must not veto biographical curiosity)
- `services/orion-cortex-exec/tests/test_chat_relational_stance.py` (extend; do not invent keyword detectors)
- Optional: `orion/sentience_striving_program/README.md` §15b stale "does not reach chat" claim

### Deeper cut

- `orion/curiosity/worldview.py` / kickoff prompts (Juniper-about priors)
- `orion/situational/context.py` (what situation shows from Juniper-priors)
- `orion/substrate/attention/questions.py`
- Schema/docs for prior `about` / privacy

---

## Non-goals

- Keyword / phrase detectors for "promotion", "Team Lead", surgery, etc. (banned by relational-stance and anti-slop rules)
- Global temperature bump as a substitute
- Turning every celebration into an interrogation (cap: one grounded question)
- Replacing investigation-loop priors or endogenous curiosity
- Claiming sentience or felt pride; this is a stance/contract miss with inspectable traces
- Auto-outreach about Juniper's job from the curiosity loop

---

## Acceptance checks

Falsifiable, in order:

1. **Replay of this turn class.** Fixture with the promotion transcript (or equivalent synthetic brief) through `enforce_chat_stance_quality` + `compile_speech_contract` asserts: curiosity priority present; speech contract contains grounded-question instruction; closing move is not leave-space-alone.
2. **Hard-news control.** Fixture with explicit no-probe / overload still gets leave-space / no open-ended pile-on.
3. **Live smoke after deploy.** A similar biographical update produces `?` in the assistant reply *or* a recorded `selected_action.action_type=ask` with a non-constraint question — and turn-trace alignment notes do not praise "no questions" as success for that class.
4. **No keyword cathedral.** Diff adds no `if "promotion" in user_message` and no YAML list of life events.
5. **Attention calibration evidence.** Before changing `min_ask`, publish the live `scope=chat` salience distribution (or cite it in the PR) so the new cutoff is measured.
6. **(Deeper cut)** At least one Juniper-prior exists after a biographical turn and is readable on a later turn's situation/prior surface.

---

## Recommended next patch

**Thin cut only:** stance vocabulary + speech-contract wiring + tests that replay this miss; then a measured `min_ask` / watch-veto tweak from live chat salience rows.

Do **not** start Juniper-priors in the same PR. That is a cognition-loop / privacy proposal of its own and needs an explicit go.

---

## Evidence appendix (incident)

```text
correlation_id: bfe0e731-f5fd-40bd-8bbc-ae4366c0f074
user: promotion + Team Lead for 3-5 directs + keep Architect/FDE work
attention_salience_trace: Team Lead… salience=0.562 plan novel
stance: playful_exchange / relational / leave_space_without_offer
         priorities without situated_curiosity
finalize: praised no next-steps / no oversolve
worldview live priors: Orion machinery only
```

---

## Agent self-check (anti-slop)

- Does **not** propose keyword triggers for promotion/surgery/feelings.
- Names choke points: `select_actions`/`min_ask`, `compile_speech_contract`, stance vocabulary, attention watch-veto.
- Separates signals: attention already inferred novelty; stance/post-process destroyed curiosity.
- Requires structural tests through enforce/speech contract, not substring checks in `.j2` alone.
