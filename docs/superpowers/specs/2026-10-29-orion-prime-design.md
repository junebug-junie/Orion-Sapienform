# Orion Prime: an octopus mind on the substrate grammar

**Design specification · 2026-10-29, rev 3 on 2026-10-30 · Proposal mode, design only, no code**
**Split note:** Phase 1 of the original two-phase doc (development session orchestration) is now `2026-10-29-dev-session-orchestrator-design.md`. Neither doc depends on the other.
**Builds on:** the thinking architecture (`2026-10-14-orion-thinking-architecture-design.md`, PR #2669), the body economy (`2026-10-23-organism-body-economy-design.md`, PR #2855), and the act ledger (`orion/grammar/acts.py`, `orion/schemas/grammar.py`).

## Arsonist summary

Orion measures a lot and chooses almost nothing. The 09-20 sentience audit found "every ingredient exists as a measurement, none as a consequence." The thinking architecture found the same thing at the lattice: nothing reaches action selection, and nothing compares an intention with an outcome.

Prime is the part that chooses. It is a warm local model that keeps thinking between conversations. It holds unfinished concerns, makes forecasts, and proposes acts. It is shaped like an octopus: the central brain sets the overall tone, and each arm (organ) runs its own loop, can refuse, and can talk to other arms directly.

This revision makes four decisions:

1. **Prime is not a new layer.** It occupies the slow loop of the substrate lattice, which already exists. It reads attention (L5) and writes act candidates (L7). Those candidates go through policy (L8) and the allocator (L9), get graded by feedback (L10), and are fed back by consolidation (L11).
2. **Everything Prime and the organs do is written in the grammar.** Every signal and every call is an act with an `act_opened` and an `act_closed`, using the closed verb vocabulary that already exists.
3. **Chosen acts go through cortex. Autonomic acts do not.** This is the body economy's rule: you choose to look; you do not choose to sweat.
4. **Prime pays for its own thinking.** It bids in the body economy like every other want. When the cabinet is cool, thinking is cheap and Prime thinks often. When it is hot, Prime thinks less. Nothing vetoes it.

The first build has a live effector. It does not run in shadow. The "narrator" control (same thinking, no effect) runs on the allocator's existing holdback arm.

## 1. What Prime is for

A test of one hypothesis: a long-running generative process, coupled both ways to self-organizing organs, produces choices and continuity that isolated parts cannot. Examples are concerns that survive restarts, forecasts that get checked, and acts that change because of what the organs found.

This is a test of functional organization. It is not a claim about experience or personhood.

Three kinds of continuity, all three required:

1. **Process.** A small deterministic runtime is always alive. It keeps Prime's state, wakes Prime, checkpoints, and resumes. It does not depend on a GPU lease.
2. **Generative.** A warm local model is called again and again, carrying a curated working set forward.
3. **Causal.** What Prime thought earlier changes what Orion does later. Without this, the monologue is theater.

Prime's identity is not the model weights or a transcript. It is the durable history of Prime's concerns, forecasts, acts and their outcomes.

## 2. Where Prime sits

### 2.1 On the thinking-architecture rungs

| Rung (PR #2669) | State 2026-10-30 | What Prime does there |
|---|---|---|
| 1. Attention faces the world | open (#2528) | Reads the winner. Prime does not need this rung to start, but it is weaker without it. |
| 2. Doing on one track (act ledger) | **live** | Every Prime act and every hormone is an act row. |
| 3. Bind the winner, compete across doers | **live** (#2830, #2840) | Prime's act candidates enter the allocator. The holdback arm is the narrator control. |
| 4. Gain and one want | open | Prime's regulate signals are the gain dial (seam 3). Prime's hunger to think is a want. |
| 5. Felt state and situational goals | open | Prime reads felt state from the body economy (Organ 5). It cites `goal_ref` on its acts. |
| 6. Forecast and working memory | open | **Prime is the generator and holder here.** The design names "the stream, reverie, looking and painting" as generators that plug in at this rung. Prime is that slot. |
| 7. Consolidation and metacognition steer | open (mutation pipeline frozen) | Prime proposes `mutate` acts through the existing mutation review gate. |

So Prime is not a third attention system or a new scheduler. It fills rungs 4–7, which are open.

### 2.2 On the lattice

```text
 L1-2 grammar ─► L3 reducers ─► L4 field ─► L5 attention ──┐
                                                           ▼
                                         ┌──── PRIME (slow loop) ────┐
                    organs ◄── calls ──► │ working set · concerns    │
                    (arms)  ◄─ regulate─ │ forecasts · choice        │
                                         └─────────────┬─────────────┘
                                                       ▼ act candidates
                                L7 proposals ─► L8 policy ─► L9 allocator / cortex-orch bid
                                                       ▼
                                 cortex-exec or durable run ─► act_closed ─► L10 feedback
                                                                                 ▼
                                                    L11 consolidation ─► Prime's concerns
```

`make substrate-ladder-check` stays the liveness gate for this path.

## 3. The octopus: central brain, thinking arms

Semi-hierarchical. Each arm (organ) keeps its own loop and its own source truth. The central brain (Prime) has broad reach but no monopoly:

- **Arms act locally without asking.** Curiosity picks questions. Dream runs cycles. Vision perceives. None of that waits on Prime.
- **Arms talk to each other directly.** Curiosity can ask memory something without going through Prime.
- **Arms can refuse Prime.** A refused signal closes as a refused act, with a reason. That is data, not failure.
- **Prime sets the overall tone** with broadcast regulate signals and with what it bids for.
- **Prime reaches into one arm** only through that arm's published calls.

There are two ways to talk, which are described next: **hormones** (a few common signals) and **calls** (arm-specific requests).

## 4. Hormones: a few common signals, written as acts

### 4.1 No new verbs

The grammar already has a closed verb vocabulary, "the same at every level" (`ActVerb` in `orion/schemas/grammar.py`). Three verbs are reserved and have no producer yet: `regulate`, `maintain`, `wait`. Hormones are those verbs. The hormone name is the act's object. It is not a new verb.

| Hormone | Act | What it asks the receiving organ to do | Existing handler it maps onto |
|---|---|---|---|
| `quicken` / `slow` | `regulate`, direction `increase` / `decrease`, object = organ cadence | change how often it runs | per-organ cooldowns and caps (curiosity cooldown, outreach cadence); later, body-economy prices |
| `wake` | `regulate` increase, object = organ or host activity | come out of rest | `physiology.recommend_wake` (hecate); GPU pool service start |
| `sleep` | `regulate` decrease, plus `dream_cycle` verb | enter the rest/consolidation regime | dream cycle; rest gating |
| `hibernate` | `wait`, long horizon | stop ticking except for the heartbeat | GPU pool `pool-stopped` services |
| `metabolize` | `maintain`, object = store or backlog | spend effort on upkeep (compaction, distillation, consolidation) | memory distill, compaction, L11 consolidation |
| `mutate` | `regulate`, object = one parameter of the organ | propose a change to its own settings | substrate mutation pipeline (`orion/substrate/mutation_*`), under its existing review gate |

Not a hormone: **alarms.** The 34 °C reflex, Juniper's turn, quiet hours and caps stay in the spinal cord, as the body economy defines it. A hormone never loosens a spinal veto.

### 4.2 Receivers declare what they respond to

Each organ declares which hormones it handles. For organs that run as verbs, the declaration goes in a `receptors` list next to the existing `physiology` block in the verb YAML. For autonomic organs, it goes in one small manifest.

- A hormone sent to an organ that does not declare it is a **visible error**. It is never dropped silently.
- An organ that declares a receptor must have a handler and a test in the same patch.

### 4.3 The anti-cathedral gate is the act ledger

Each hormone opens an act with an expectation: this organ's measured cadence or activity moves in this direction within this horizon. The act closes with what actually happened.

- **This needs one new `ActSignal`**, called `target_moved` here as a proposed name. It resolves true if the receiving organ's own measured rate moved in the stated direction by the horizon. The receiving organ's existing rate counters are the provenance. The metric quality gate must be run on this signal before it is wired in.
- **A hormone whose acts never move anything gets retired.** The ledger shows that directly. It is not a judgment call.

## 5. Calls: arm-specific requests (the ADK-like layer)

Hormones are coarse. Calls are specific: "memory, have we tested this prior?" or "vision, what changed in the walkway zone?"

- **Each organ publishes its calls** the way an ADK agent publishes tools: name, typed input, typed output. Organs that already run as cortex verbs publish those verbs. For example, `skills.perception.look_at_camera.v1`, `ask_camera` and `skills.physiology.state.v1` already exist.
- **Any organ can call any published call.** Prime is one caller among several.
- **Every call is an act.** Verb `ask` or `recall` or `look`, with `parent_act_id` linking it to the act that caused it. This gives the cross-organ causal graph without a new store.
- **A caller's prose is never an instruction to the callee.** Authority comes from the runtime, not from the text.

Start with two or three calls that already exist as verbs. Add new ones only when a real caller needs them.

## 6. Which path each act takes

This is the body economy's execution seam, applied to Prime and the hormones.

| Who decided? | Path | Bid? | Grammar |
|---|---|---|---|
| **A mind chose it:** Prime, or an organ's model worker | cortex verb → cortex-orch (single bid site) → cortex-exec, or a durable run for heavy work | yes, at cortex-orch | `act_opened` / `act_closed` emitted by cortex-orch (`services/orion-cortex-orch/app/grammar_emit.py`), level `deliberate` or `faculty` |
| **A rule fired:** an organ's reflex to another organ's state | bus message between organs, no LLM, no cortex round-trip | no | `act_opened` / `act_closed` at level `organ` or `metabolic`, written in bulk or as rollups (thinking arch §3.2) |
| **Spinal cord:** heat reflex, Juniper's turn, caps | unchanged | never | unchanged |

So a hormone Prime chooses to send goes through cortex and costs a bid. The same hormone fired by curiosity's own rule ("backlog empty → slow") does not. The question is always whether a mind chose it.

Durable runs carry anything long or GPU-heavy: Prime's deep-thinking episodes, a model worker an organ recruits, or a dream synthesis. They use the existing retries, checkpoints and GPU-pool leases. Prime adds no queue and no scheduler.

## 7. Prime pays for its own thinking

Prime is a customer of the body economy. It is not exempt from it.

- **The heartbeat is autonomic and is not a verb.** The deterministic runtime keeps state, reads the field and checks "did anything change?" on CPU. It never waits on cortex.
- **Each generative cycle is a verb**, `prime.think` as a proposed name. Its `physiology` block covers the two kinds of thinking:
  - **Shallow thinking is Tier 1, with a reserved share.** That reserve is what makes Prime near-continuous. The reserved share follows the body economy's rule: p75 of its own use, capped at 50%.
  - **Deep thinking (a leased bigger model, tool use, wide context) is Tier 2.** It bids on the open market.
- **Hunger.** Prime's want to think is a `WantV1` whose intensity grows with open concerns and with unexplained surprise. It also grows while unfunded, as every want does.
- **Price sets frequency. Kev sets depth.** On a cool afternoon prices are low and Prime thinks every few tens of seconds. When the cabinet heats up, the price rises and Prime thinks less. No veto is involved. Kev decides whether a cycle stays shallow or escalates to a deep run.
- **Prime's acts bid too.** Prime's act candidates compete in the allocator against every other doer, with the same lane prices. Prime gets no back door.

### 7.1 Where Prime runs: circe chat lane 0 (Juniper, 10-30)

Prime's baseline thinking runs on circe's chat lane 0 by default. The model is already warm there, so Prime needs no new allocation and has no cold starts.

That lane also serves Juniper's turns, so the rules are:

- **Juniper's turn always goes first.** Interactive and urgent requests never bid and are never shed (spinal cord). Prime's cycles request the lane at a lower GPU-pool priority.
- **Prime yields mid-cycle.** If a Juniper turn arrives while Prime is generating, the Prime call is cancelled or deprioritized, and the cycle is recorded as `preempted`. It is never recorded as a thought.
- **The lane price carries the contention.** When chat is busy, the chat lane's price rises, and Prime's shallow cycles wait. Nothing is hard-coded.
- **Deep thinking may move elsewhere.** A Tier 2 deep episode can bid for another lane (for example the agent lane) through the GPU pool. Lane 0 is the default for baseline thinking, not a lock.
- **E9 is the main risk.** The first thing to measure is chat latency (time to first token and total) with Prime cycling on the same lane against Prime off.

**Result:** Prime's cadence is not hand-tuned. It emerges from heat, load and hunger, which is the point of the body economy.

## 8. What Prime does each cycle

1. **Attend.** Build a bounded working frame: the attention winner, felt state, open concerns, forecasts awaiting outcomes, Temporal Self arcs, curiosity questions, calls and challenges from organs. Mark each field's freshness. A missing field is shown as a gap, never as calm.
2. **Think.** The model produces competing readings and at most one or two proposals. A no-op is a valid output.
3. **Forecast.** Register any forecast **before** acting, as an act with an `expect`.
4. **Propose.** Send act candidates to L7 → allocator, hormones via cortex verbs, and calls to organs.
5. **Integrate.** Read closed acts: what was admitted, refused, or challenged, and what happened. Update concerns.

Concerns are a small, versioned working set (1–3 held items, as in thinking-arch rung 6). Each concern links to the acts and evidence behind it. A model swap or context compression must not create or drop a concern without an act that says so.

## 9. Experiments that would count

Kept from rev 2, now measured on the act ledger:

| Gate | Test | Evidence |
|---|---|---|
| E1 Persistence | run unprompted for days; restart the model | concerns survive restarts and later change acts; outages appear as gaps |
| E2 Origination | days with no user request | self-originated acts that trace to cross-organ evidence and close with a checkable outcome |
| E3 Lateral coordination | ambiguous cross-organ evidence | organ-to-organ calls change behavior; the effect disappears when those calls are ablated |
| E4 Both directions | — | acts where an organ refusal or challenge changed Prime's next proposal, **and** Prime hormones that closed `target_moved = true` |
| E5 Forecasts | forecasts registered before outcomes | calibration beats running-average surprise |
| E7 Equal budget | coupled vs Prime-only vs organs-only, same joules | coupling still wins |
| E8 Not performative | injected false narratives, gaps, loops | Prime does not invent certainty; loop and budget guards hold |
| E9 Juniper first | normal chat under Prime load | no regression in latency, quality, consent or repair |

**The narrator control is the holdback arm.** On held-back cycles Prime thinks the same way, but its proposals are logged and not submitted. If held-back and treated outcomes do not differ, Prime is narration, not a cause.

## 10. Rollout

| Step | What ships | Proof before the next step |
|---|---|---|
| **2A Prime live, one effector** | runtime on circe chat lane 0; `prime.think` verb with its `physiology` block; act candidates into the allocator; holdback as the narrator arm | Prime acts admitted and graded; held-back vs treated difference over ≥ 7 days; cadence follows price on cool vs warm days |
| **2B Hormones v1** | `regulate` to two organs (curiosity cadence, dream timing); receptor declarations; `target_moved` signal (after the metric gate) | regulate acts close with measured movement; unhandled-hormone count is 0 or visibly reported |
| **2C Calls between organs** | 2–3 existing verbs published as calls; organ refusals recorded as closed acts | E3 and E4 both directions observed |
| **2D Longitudinal** | add memory, body and relationship organs; run ablations | E1–E9 |

Feature flag per step, shipped ON per standing rule, with instant fallback to current behavior.

## 11. Proposal mode (§0A)

- **Capability change:** Orion gains a persistent chooser that proposes acts and regulates organs.
- **Data touched:** reads field, attention, Temporal Self, curiosity and the act ledger. Writes act rows, want and bid rows, and concern state.
- **Privacy boundary:** none between Juniper and Orion (standing decision). No credentials or terminal access are shared with the development orchestrator.
- **Trace that proves it worked:** `grammar_acts` rows with `actor = prime` that close with outcomes, and the holdback comparison.
- **Dangerous failure:** runaway rumination or a regulate loop (Prime slows curiosity, curiosity's hunger rises, Prime slows it again). Guards: concern TTL, refractory interval per receiver, bid budget, and loop detection on `parent_act_id` chains.
- **Rollback:** one flag per step. Organs ignore hormones when the flag is off. Prime's runtime can stop without affecting any organ.

## 12. Missing questions

1. ~~Which model and lane holds warm Prime?~~ **Decided (Juniper, 10-30): circe chat lane 0 by default.** See section 7.1.
2. Shallow thinking: Tier 1 with a reserved share (recommended), or Tier 2 open market?
3. Is `prime.think` one verb with a depth parameter, or two verbs (shallow and deep)?
4. Is `target_moved` one signal with an object, or one signal per receptor?
5. Which existing process owns the Prime runtime? Candidates are extending orion-mind or orion-thought, or a new small service. Inspect before choosing.

## 13. Non-goals

- No new attention ranking, scheduler, GPU allocator or job queue.
- No new verbs in `ActVerb`.
- No always-on large model per organ. Organs recruit model workers through durable runs when they need them.
- No loosening of spinal vetoes.
- No claim of sentience from this design.

## 14. Acceptance checks for the design

- Every new contract named here has a producer, a consumer and a test in the patch that adds it.
- `make substrate-ladder-check` stays green at every step.
- `check_body_economy_contract.py` and `check-verb-physiology` pass with `prime.think` registered.
- The held-back vs treated comparison is reported, whatever its result.

## Recommended next patch

Measurement only, on circe chat lane 0: run a synthetic Prime cycle (a short prompt and a short output) at several cadences. Record Juniper-turn latency (time to first token and total), lane occupancy and watts against a Prime-off baseline, on a cool day and on a warm day. The result sets the Tier 1 reserved share. Then build 2A.
