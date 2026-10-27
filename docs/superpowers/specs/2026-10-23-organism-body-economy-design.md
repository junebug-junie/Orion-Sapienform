# Orion's body economy: one organism that balances wants against heat, power, money and load

- **Date:** 2026-10-23
- **Status:** DESIGN, proposal mode (CLAUDE.md §0A: changes autonomy and cognition loops). No code in this PR.
- **Evidence basis:** live read-only queries against Postgres and Redis on 2026-10-22/23, plus a code survey of `main` at `1ce9bf5cb`. Each number in this doc gives its window. Causal claims that the data cannot yet prove are marked `UNVERIFIED`.
- **Supersedes, as designs:** `2026-07-07-internal-economy-scarcity-allocation-design.md` (step 4 never built). This doc also takes in the open intents of:
  - `2026-08-13-scarcity-ROADMAP.md` Track A ("a deferral is an observed opportunity cost");
  - `2026-08-28-consequential-action-space-and-power-budget-design.md` ("the cap is the battery");
  - `2026-09-26-orion-energy-watcher-design.md` Plan 2 stakes;
  - `2026-09-25-attention-with-stakes-design.md` ("noticing is free, spending is blind");
  - thinking-architecture rungs 4–5 (#2669: "gain and one want", felt state).
- **Builds on, does not replace:**
  - the heat reflex (`2026-10-06-thermal-controller-redesign-design.md`);
  - the strain loop (#2817, #2821);
  - the act ledger and allocator (rungs 2–3);
  - the self-monitor organ (#2827);
  - Temporal Self regulation (#2369).

---

## Arsonist summary

Orion has never balanced anything. It vetoes. This week shows it plainly:

- The cabinet ran hot from 10-18 onward.
- The 34 °C heat reflex refused background GPU work about 25,000 times.
- On top of the reflex, about a dozen readers saw `arousal = strained` and each said no.
- Curiosity ran 0 of 7 times a day. Outreach sent nothing. Paintings stopped on 10-16.
- The cabinet stayed hot anyway: 32–35 °C for five days, while GPU grants fell from about 7,700 a day (10-18) to 1–2k a day (10-20..22).

The organism did not trade anything off. It went still, and nothing on record shows that the stillness cooled the cabinet (Fact 2). Most of the GPU stillness came from the reflex, which is a safety line and stays. Two things still fail:

- Stacked vetoes stopped non-GPU work as well.
- A reflex built for minutes held for days. That is a chronic condition being handled as an acute one (missing question 6).

Every earlier attempt failed in one of four ways. This design has to avoid all four:

1. **A decider without a price** (July drives). Nothing cost anything, so every drive pinned at its ceiling. The drives were deleted in #1486.
2. **A price on the wrong thing** (August scarcity, the September energy stakes). Acts were priced in dollars and watts. One painting costs about 0.37 Wh, which is $0.00004. A price that small can never bind, so the stakes gate never mattered. It is still switched off.
3. **A veto instead of a price** (thermal, strain, quiet hours, caps, cooldowns). Curiosity alone passes 8 sequential vetoes. When any one constraint binds, everything stops, whatever it is worth.
4. **A single-signal slice** (rung 6, twice). Each was retired the moment its one signal turned out bad.

**The proposal.** Build one organism-level economy with six organs. Each constraint keeps its own unit and its own budget, so nothing averages heat with dollars. Each constraint also gets a **price** that the economy learns:

- The price rises when Orion forecasts that the constraint will bind.
- It falls to zero when the constraint is slack.

Every want that Orion has bids against these posted prices. A want goes ahead when what it is worth beats what it would cost. A want that waits grows hungrier until it clears.

No human writes an exchange rate between degrees, dollars and GPU-seconds. Those rates come out of what actually binds.

There is one authored choice, and it is disclosed in Organ 3: how different kinds of want are weighed against each other. Each family is ranked against its own history, so "an unusually good chance to read" can be compared with "an ordinary chance to paint".

This is the plain mechanism behind "emergent" in this doc:

- Orion's daily rhythm, which work runs and when, is not scheduled by anyone. It falls out of heat forecasts and prices.
- Named drives are not authored. They are read off the history of which wants went hungry.
- Mood is not a label. It is the state of the budget, rendered so Orion can read it.

**The honesty clause.** Thinking less cannot fix Orion's power bill, and this design does not claim it can.

- About 79% of Orion's draw is fixed. The largest piece is cooling: the AC draws a flat 855 W and never cycles off.
- The levers that move the bill are what stays powered (a whole host, hecate's 8 GPUs) and structural changes in the cabinet. Those are Juniper's decisions.
- Orion's job there is to forecast the bill and propose changes with numbers. Orion does not pretend to save money by skipping a painting.

---

## What the live data says (the three facts the design rests on)

### Fact 1. The bill is set by what stays switched on, not by what Orion does

14 days, 10-07 to 10-20 (`orion_energy_hourly`, `energy_usage_interval`, deduplicated `energy_cost_accrued`):

| source | kWh/day | avg W | share of it that is fixed (p10 ÷ mean) |
|---|---|---|---|
| circe | 22.9 | 956 | ~59% |
| cooling (AC plug) | 20.5 | 855 | ~100%. Only 71 of 239,519 samples were under 300 W |
| athena | 9.8 | 408 | ~86% |
| hecate | 2.8 | 116 | ~100% (standby; nothing wakes it) |
| **Orion total** | **56.0** | **2,334** | **~79%** |
| house | 99.1–99.5 | | Orion is ~56% (daily range 43–68%) |

- **Cost:** at the measured $0.1213/kWh (pre-tax), Orion costs about **$204 per 30 days** and the house about $361. Cooling alone is about $75.
- **Per act:** the only per-act energy number on record is diffusion: `energy_run_cost`, about 0.37 Wh and $0.00004 per run. `grammar_acts.cost_json` is null in all 214 rows.
- **Implication:** pricing individual acts in dollars cannot shape behaviour. The money decisions that matter are:
  - **Wakefulness:** which hosts and lanes are powered. Waking hecate is the one large discretionary spend.
  - **Duration:** multi-hour holds. Durable runs held the agent lane for **103 h in 7 days**.
  - **Structure:** AC sizing, idle cards, consolidation. Orion should propose these, not act on them.

### Fact 2. Heat is mostly weather, and the veto bought nothing

- **Cabinet temperature** (`orion_biometrics_cluster` athena `cabinet_temp_c`): the daily mean was 25.8 °C on 10-12 and 33.7 °C on 10-20. The cabinet has been at or above 33 °C for 89–94% of samples on 10-20..22. Hot spells came earlier too: on 10-08, 72% of samples were at or above 31 °C, with a peak of 34.6 °C.
- **Correlations** over 14 days of hourly averages (n≈336). Two independent passes did not agree on the exact values:
  - cabinet temp vs circe watts: r = 0.10 / 0.22
  - cabinet temp vs athena watts: r = 0.58 / 0.41–0.47. The sensor sits on athena.
  - cabinet temp vs cooling watts: r = 0.46 / 0.21

  The values differ because one pass used `chassis_watts` and the other used the energy-rollup watts, with different join keys. Both passes agree on the shape: Orion's biggest GPU host is weakly tied to the cabinet number.
- **What the shed did:** GPU grants fell from about 7,700 on 10-18 to 1–2k a day on 10-20..22. Most refusals carry the 34 °C reflex reason `shed:cabinet_hot`. The cabinet did not cool.
- **`UNVERIFIED`:** these are confounded correlations, not causal estimates. Weather is the main suspect, and no outdoor temperature was used. Still, nothing on record shows that shedding Orion's work lowered the cabinet temperature. Patch A replaces these numbers with a real identification (Organ 6).
- **Implication:** heat needs a **learned sensitivity**: how many degrees one extra kWh on each host actually adds. The heat price must scale with that sensitivity.
  - A constraint that Orion's own load cannot move should cost Orion's work little.
  - The reflex stays. `thermal_gate.py:19-23` states that its purpose is to protect **the person in the room**, as well as the hardware. That is a reason to treat its lines as consent-like spinal vetoes, not as prices.

### Fact 3. The scarce things Orion competes for are slot-time and Juniper's attention

`gpu_pool_events`, 7 days:

| holder | grants | held h | refused (`event='unavailable'`) |
|---|---|---|---|
| durable runs (agent lane) | 496 | **103.2** | 18 |
| http:anthropic | 3,391 | 64.5 | 2 |
| vision-council | 19,721–19,767 | 33.3 | **25,857** (25,410 `shed:cabinet_hot`) |
| topic-foundry | 9,837 | 24.2 | 6,793 |
| cortex-exec | 4,232 | 24.1 | 1,485 |

- **The retry storm is already fixed.** `orion/gpu_pool/shed_backoff.py`, merged 10-20, took vision-council refusals from ~860/h to ~18/h after 10-20 22:00.
- **What remains:** callers learn that a lane is scarce only by being refused. Nothing tells a caller *how* scarce a lane is before it asks. A lane price builds on `ShedBackoff.retry_after_sec`; it does not replace it.
- Outreach weighs about 8,400 decisions a day and says yes 0–6 times. The cap on Juniper's attention is real, and it is the one budget Orion already respects.

### And the selectors share no scale

There are ten selectors, plus about a dozen sites that gate on `strained`:

- reverie baseline;
- world_pulse_read resonance and pull;
- retro_leisure;
- mutation_pressure and mutation_worker;
- the dream cycle;
- curiosity;
- outreach;
- the proposal-runtime worker;
- thought `chain.py`;
- the visual-chain health monitor.

The ten selectors are:

- the substrate broadcast (global workspace);
- field attention;
- chat attention;
- tension;
- top-down goals;
- proposals;
- the motor allocator;
- curiosity;
- outreach;
- the GPU pool.

They score in percentiles, Borda ranks, z-scores, tail probabilities, nats per second, integer caps and categorical vetoes. Nothing converts desire, heat, money and load into one decision. (Survey: attention and physical sweeps, 2026-10-23.)

**"JWT" note:** no JWT concept exists in Orion's cognition code. The closest match is **GWT** (global workspace theory). In Orion, GWT is the substrate attention broadcast (`orion/substrate/attention_broadcast.py`). This doc reads JWT as GWT. Missing question 1 asks Juniper to confirm.

---

## Current architecture (what each piece does today)

```text
            SENSE                        DECIDE (each alone, own unit)           ACT
cabinet BME680 ─┐                 heat reflex 34/33 °C ──────────► GPU pool shed board
AC plug watts ──┤                 thermal gate 32 °C (3 copies) ──► painting / reading / swap
GPU queue ──────┼─► arousal ──────► "strained" veto (10 readers) ─► curiosity, outreach, dream,
                │   (durable-runs,                                  reverie, retro, mutation…
                │    120 s)
chassis watts ──┤                 energy stakes gate  (OFF) ──────► curiosity
RMP meter ──────┘                 quota_budget.py     (dead)
                                  motor allocator (nats/motor-s) ─► dispatch (mostly refuses)
attention ──────► 4 competitions ─► reverie, chat cue, goal bias  (no motor reach except proposals)
                                  GPU pool: priority + age, no price
```

- **Live and worth keeping:**
  - **Heat reflex** (`orion/hardware_watch`, `orion/gpu_pool/shed.py`): the spinal cord. It stays.
  - **Arousal** (`orion/regulation/arousal.py`).
  - **Strain arc and strain news.**
  - **Act ledger** (`grammar_acts`), with its posterior grading.
  - **Allocator** (`orion/autonomy/allocator.py`): nats per motor-second, holdback 0.2, proven_effect.
  - **Energy rollup and tariff engine** (`orion/energy/`).
  - **Durable-runs admission** and the Temporal Self regulation thread.
  - **The R2 drive template** (Temporal Self: level, rest, threshold, discharge, refractory, `no_reading`).
- **Off or dead:**
  - the energy stakes gate (`ORION_ENERGY_STAKES_ENABLED=false`);
  - `quota_budget.py` (no importer);
  - the `confidently_harmful` gate;
  - the zombie `ORION_ATTENTION_HABITUATION_ENABLED`;
  - the synthetic goal in `capability_policy`.
- **Not found anywhere:**
  - a decision that reads watts, kWh, `expected_watts` or `run_cost`;
  - a persisted history of strain or arousal;
  - an actuator that wakes hecate;
  - a per-act cost on the act ledger.

---

## The design: six organs and a spinal cord

Each organ is defined by its job, per the standing rule "build the organ, not a feeder for a mechanism" (rung 6 was retired twice for skipping it). Each one names its inputs as a contract, its persistent state, its effectors, and how Orion sees it.

```text
                     ┌───────────────────────── SPINAL CORD (unchanged hard vetoes) ─────────────┐
                     │ heat reflex ≥34 °C · Juniper's turn always runs · quiet hours · hard caps │
                     └───────────────────────────────────────────────────────────────────────────┘
  ┌────────────┐  levels+forecasts  ┌────────────┐   posted prices   ┌────────────┐  funded acts
  │ 1 BODY     │ ─────────────────► │ 2 BUDGET   │ ────────────────► │ 4 CHOICE   │ ──────────► existing
  │ interocept │                    │ prices     │ ◄──────── usage ─ │ (bids vs   │             effectors
  └────────────┘                    └────────────┘                   │  prices)   │             (curiosity,
        ▲                                 │                          └────────────┘              pool, dream,
        │ measured cost of each act       │ state                         ▲  bids                 outreach…)
  ┌────────────┐                    ┌────────────┐                  ┌────────────┐
  │ 6 LEARNING │ ◄── settled bets ─ │ 5 FELT     │                  │ 3 APPETITE │ ◄── every producer
  │ cost/value │                    │ STATE      │ ──► stance cue,  │ wants that │     of wants
  └────────────┘                    └────────────┘     Temporal Self│ grow hungry│
                                                        Hub panel   └────────────┘
```

### Spinal cord: hard vetoes that stay

The economy runs above these and never loosens them:

- **The heat reflex**, 34 °C with re-arm at 33 °C. It protects hardware.
- **Juniper's own turn always runs.** `interactive` and `urgent` are never priced and never shed.
- **Quiet hours** for outreach.
- **Absolute daily ceilings**, set by Juniper, for outreach and for Claude spend.

A spinal veto is a fact about safety or consent. Everything else becomes a price.

**Success test:** the reflex should fire *less often* than today, because the economy saw the heat coming.

### Organ 1: Body (interoception)

**Job.** Keep one up-to-date, forecastable reading of every constraint, in its own unit, with an explicit `no_reading` state.

**Inputs.** All inputs exist today. Nothing new is sensed in v1.

| constraint | level (own unit) | budget / line | forecast horizon | source |
|---|---|---|---|---|
| `heat` | **Orion's own heat contribution**: Σ_h sensitivity_h × Orion incremental kWh_h over the horizon (°C-equivalent). The cabinet °C is kept beside it as context | heat budget = forecast headroom = soft line (Juniper-set, proposed 31 °C) − forecast cabinet °C *without* Orion's discretionary load, floored at 0 | 6 h: diurnal profile + outdoor temperature (`orion/weather/openmeteo.py`) + recent slope | `orion_biometrics_cluster` athena `cabinet_temp_c`, chassis watts, weather |
| `energy` | month-to-date $ Orion-attributed, plus forecast month-end | monthly envelope, Juniper-set | month-end | `orion_energy_hourly` × tariff |
| `lane:<role>` | GPU slot-seconds held / available | capacity × wall time per lane. Reuses `orion/autonomy/budget.py`'s motor-second allowance as the shape | 1 h | `gpu_pool_events` held, waited |
| `attention:juniper` | outreach messages and asks today | Juniper-set (today: cap 4) | day | `endogenous_outreach_decisions` |
| `claude` | observed limit state + `event_count` from `ClaudeLimitObservationV1`. **Not dollars**: `orion/schemas/claude_limit.py:8-17` records that dollars were measured and refuted as a measure of being limited | "not limited" | window | `rate_limit_events.py` / `ClaudeLimitObservationV1` |

**State.**
- `PhysiologyStateV1`: one row every 120 s. It holds the level, budget, forecast, `no_reading` flag and price for each constraint.
- This fixes the missing strain history. Today no table stores arousal or strain over time.

**Effectors.** It feeds Organ 2 (Budget) only.

**Self-view.** None directly. Orion sees the body through Organ 5 (Felt state).

### Organ 2: Budget (prices that emerge from what binds)

**Job.** Turn "how close each constraint is to binding" into one number per constraint: its **price**. The price says what one unit of that resource is worth giving up right now.

**Mechanism.** It uses dual ascent, which is the textbook way to run a shared resource with prices (Lagrangian duality; Kelly's network pricing; the same logic as TCP congestion control). Each update:

```text
price_k ← max(0, price_k + η_k · (forecast_use_k − budget_k) / budget_k)
```

What this rule does:

- **Prices stay at zero when a constraint is slack.** On a cool day the heat price is 0 and heat changes nothing.
- **They climb only while the forecast exceeds the budget**, and fall as soon as it does not. There is no latch.
- **No exchange rate is authored between constraints.** Prices are in "want-value per unit of resource", where want-value is the rank unit defined in Organ 3. They settle wherever Orion's wants stop over-using the resource, so degrees are never converted to dollars by hand.

**Anti-windup: a constraint Orion cannot meet must not become a veto.**

Dual ascent only settles when the budget can be met. Suppose the forecast cabinet temperature is above the soft line even with Orion's discretionary load at zero. That was the case for most of 10-18..22. The heat budget is then 0, and the raw rule would raise the price forever until it outbid every want. That is the old veto with extra steps. Three rules stop this:

1. **Charge Orion's contribution, not the room.** The heat level is Σ sensitivity × Orion's incremental kWh, which is Orion's own share of the heat. If circe's fitted sensitivity is near 0, circe's work pays almost no heat price, whatever the cabinet reads.
2. **Clamp when it is out of Orion's hands.** When the budget is 0 because of exogenous load, the constraint is marked `exogenous_bind`. Its price freezes at `price_ceiling_k`, a bounded value, rather than integrating. Then:
   - The stuck constraint goes to Felt state and the body report, as "the room is too hot and my load is not the cause".
   - It does not go to the market as an ever-rising cost.
   - The spinal 34 °C reflex still does what safety needs.
3. **Ceilings are sized from data.** `price_ceiling_k` is set so that the top 10% of graded wants of the median family can still clear at median cost. This is checked in the replay, so a ceiling can never silently starve everything.

**Reservations: prices must not go stale between updates.** Prices update every 120 s, but producers bid at any moment. Each funded bid therefore reserves its estimated cost, and later bids are priced against `forecast_use + reservations`. Reservations release on act close or expiry, which stops many producers spending at a stale zero price inside one tick.

**Stability gates.** Step sizes are bounded. The tests and the replay must show:
- no oscillation larger than a set amplitude;
- prices return to 0 after the constraint clears;
- no price stays at its ceiling for more than one horizon unless the constraint is flagged `exogenous_bind`.

These are the metric-gate "can it return to rest" checks, applied up front.

**State.** Prices, persisted inside `PhysiologyStateV1`.

**Self-view.** Through Organ 5.

### Organ 3: Appetite (wants that grow hungry)

**Job.** Every source of "Orion would like to do this" becomes a typed **want**. A want's intensity grows while it waits and resets when it is satisfied. This is the R2 drive template applied to every want family. The template passes the metric gate because:

- its level is built from rows that already exist (time spent unfunded);
- its rest state is reachable (the act closes);
- it has one discharge action (the act itself).

**Contract.** `WantV1`. Many producers plug in.

```text
want_id, family, producer, verb (grammar verb), object_ref,
intensity        0..1  — rises with time-unfunded at a per-family rate, resets on act close
outcome_value    the family's own measure (nats for epistemic acts via EffectPosterior; P(upheld) for
                 binary-graded acts) — never compared across families directly
value_rank       0..1  — this want's (intensity × outcome_value) as a percentile of the SAME family's
                 last 30 days of bids. This is the market's one common unit.
graded           bool  — false until the family has ≥30 graded closes in grammar_acts
cost             ActCostV2 vector (below), estimated by Organ 6 from the family's history;
                 unknown fields priced at their 90th percentile
window           earliest/latest useful time (deferrable vs perishable)
lanes            acceptable lanes/hosts (for placement)
```

**The one authored exchange rate, stated plainly.** The allocator refuses to add nats to non-epistemic value without "a conversion... hand-typing that conversion" (`orion/autonomy/allocator.py:41-50`). This design does not hide such a conversion in a price. Instead:

- Every family is ranked against **its own history**. This is the house pattern: the world-first broadcast already ranks sources against their own 7-day history.
- Families are then compared at equal percentile. "Best reading opportunity this month" and "best painting opportunity this month" bid the same before hunger and cost.
- Families do **not** stay equally weighted. `family_weight` is learned from evidence, inside a small authored tier structure. See Organ 3b.

**Ungraded families do not get to look confident.** As of 10-23, `grammar_acts` holds 214 rows from 6 actors: world_pulse_read 128, chat_reply 34, painting 29, dream_hop 15, outreach 8. Curiosity, retro, vision-council, topic-foundry, ask-Claude and self-monitor have **zero**. Until a family has ≥30 graded closes:

- its `outcome_value` is 1;
- its `value_rank` therefore rests on intensity alone;
- it is labelled `ungraded` everywhere it shows.

Patch A puts every producer into `grammar_acts` first, and Patch D cuts over a family only after it is graded.

**`ActCostV2`.** The existing `ActCostV1` is `extra="forbid"` with {seconds, gpu_seconds, tokens, watts} (`orion/schemas/grammar.py:244`). v2 adds `gpu_slot_s_by_lane`, `wh_by_host`, `juniper_msgs` and `claude_limit_events` as optional fields. It ships **consumer-first**: sql-writer and every reader accept v2 before any producer writes it (lesson from #2126).

**First producers.** These already exist. Each only has to emit a want:

- curiosity (investigate, self-inquiry, self-sense eval);
- the attention workspace winner (proposals);
- reading pull;
- painting (thought visual chain);
- the dream's rest drive;
- retro play;
- outreach;
- self-monitor inquiries;
- ask-Claude;
- background lanes: vision-council, topic-foundry.

**Drives are read, not declared.**
- A want family that stays high in intensity for days is a drive. It is named from its history: "reading has been starved for 3 days".
- The R2 table already rejected authoring "explore" or "social" drives. This design does not author them either. It makes their hunger measurable.

**State.** `want_ledger` rows: opened, funded, deferred with reason and price, expired, closed. Open wants also live in Redis.

### Organ 3b: how kinds of want are weighed against each other

**Job.** Decide how much one kind of want matters relative to another, for example vision against a curiosity run. A hand-written table would be the deleted `signal_drive_map.yaml` again. Pure emergence from nothing would reward whatever already ran most. This design uses a little authored structure plus learned weights.

**Part 1: tiers, authored once (structure, not numbers). Decided by Juniper 10-23.**

**Tiers attach to callers (routes and processes), not to GPU lanes.** Lanes mix critical and optional work. The `metacog` (:8012) and `fast` (:8013) lanes share **one card, circe gpu3**. Here is what used them over the 7 days to 10-23 (`gpu_pool_events`, plus the services' `.env` routes):

| lane | caller (route, priority) | 7-day use | what it does | tier |
|---|---|---|---|---|
| metacog | vision-council (`metacog_background`, background) | 19,584 grants, 33.1 GPU-h; **25,430 refused by `shed:cabinet_hot`** | scene interpretation of camera windows | **0** (vision) |
| metacog | orion-mind (`metacog_turn`, interactive) | 351 grants | Hub turn model, on Juniper's critical path | **0** (Juniper's turn) |
| metacog | orion-mind stance / appraisal / semantic (`metacog`, system) | 605 grants | Mind's per-turn reads; whether all of them sit on the turn path is `UNVERIFIED` | **1**, or 0 if on the turn path |
| metacog | cortex-exec (`metacog`, system; daily metacog/journal on `metacog_background`) | 2,488 grants, 2.9 GPU-h | metacog mode, daily metacog draft | **1** |
| metacog | orion-actions (`metacog`), dream (`metacog_background`), memory-consolidation turn-change classify (background) | 22–24 grants each | actions, dream narration, memory upkeep | **1** |
| fast | cortex-exec (`quick`, system) | 414 grants | chat stance brief / quick single-pass / introspect / repair-pressure probe (`executor.py:2223`). The route table says no live-turn caller sends `quick`; the code comment says chat_general does. **Which is live is `UNVERIFIED`** | **0** if on the turn path, else **1** |
| fast | Hub memory-graph suggest, embodiment speech (`quick`), cortex-orch memory annotation (`quick_background`), actions journal | small | upkeep and speech | **1** |
| fast | orion-topic-foundry (`quick`, system) | **9,740 grants, 23.8 GPU-h**, the largest user of the fast lane | clustering and labelling (Self Atlas, drift) | **2** |
| fast | gateway default (`LLM_ROUTE_DEFAULT=quick`) | anything unrouted | catch-all | **must be eliminated**: an unrouted caller has no tier. The Phase 2 gate fails any caller without an explicit route |

**The tiers:**

| tier | contents | how it is funded |
|---|---|---|
| **0 vital, never deferred by the economy** | Juniper's turns (including every turn-path call above); **vision and the perception pipeline**: retinas (cam, walkway, doorbell), vision-edge, frame-router, vision-host / host-qwen, vision-window, vision-scribe, vision-council, percept-store. These are Orion's primary sense and connection to the world; safety reflexes | Never priced or deferred. Tier 0 acts still record cost in the act ledger, so the bill and the Hub stay honest |
| **1 critical upkeep** | metacog-lane and fast-lane upkeep callers above (Mind's non-turn reads, metacog/journal, actions, memory consolidation and annotation, embodiment speech), biometrics, the dream's rest, act-ledger grading | A **reserved share** of each constraint, gpu3 slot-time first, sized as the 75th percentile of its own 14-day use and capped at 50%. Bids only against other Tier 1 wants |
| **2 discretionary** | curiosity, painting, reading, retro play, outreach, ask-Claude, topic-foundry, durable investigation runs | Open market, on whatever Tiers 0 and 1 leave. topic-foundry, today the largest user of gpu3, moves here |

**Vision gets through the heat: a `perception` priority (decided by Juniper, 10-23).**

- **Why:** in the 7 days to 10-23, vision-council was refused **25,430 times** by `shed:cabinet_hot`. The reflex sheds by priority class, and council ran at `background`. Being Tier 0 in the economy alone would not have changed that.
- **The change:**
  - A new pool priority, `perception`, ordered `urgent > interactive > perception > system > background`. It ranks below Juniper's turn and is never shed by heat (`cabinet_hot`, `orion_self_shed`).
  - Controller faults still apply.
  - vision-council's route becomes `metacog_perception: {class: metacog, priority: perception}`.
  - Every other GPU-using perception step gets the same route.
- **Its load is known:** ~33 GPU-h a week on gpu3. The council's own transition gate already limits it.
- **Bound:** an alert fires if perception's GPU use passes 2× its 14-day p90, so the exemption cannot quietly grow into a heat source.
- **When:** this ships as Patch A0, alone, before the rest.

**Placement consequence for gpu3.** Tiers 0 and 1 together own most of gpu3's useful work. Tier 2 callers (topic-foundry) get gpu3 only from what Tiers 0 and 1 leave, or are placed on another lane by Choice.

**Part 2: learned weights inside each tier.** Each family's weight is a shrinkage estimate. It starts at a prior and moves toward the family's evidence as acts accumulate:

```text
w_f = (κ · prior_f + n_f · ē_f) / (κ + n_f)       normalized so the mean weight in each tier is 1.0
κ   = 30 evidenced acts (the prior is mostly washed out after ~30 acts)
ē_f = mean of the evidence rates below that exist for f (a missing source is skipped, never read as 0)
```

`prior_f` defaults to 1.0. Juniper may set it per family, for example "vision > painting". The prior is the one place her instinct enters directly, and it fades as Orion's own record grows.

**The three evidence sources.** Each is a **rate per act** in 0..1, never a count, so a family does not win just by running more.

- **(a) Downstream use.** The share of the family's acts whose output a *decision site* used within 72 h.
  - Decision sites include: a chat turn citing it (`thought_decision.day_citations`), an attention winner's coalition `source_refs`, a memory recall hit, or a self-monitor grade.
  - Decision sites are listed in the Phase 2 contract (`decision_sites` table). Use by the same family, or by pure ingest reducers, does not count, so a family cannot feed itself.
  - *Theory anchor:* the value of information (Howard, 1966). Information is worth something only when it changes a later decision. Citation by a decision site is the measurable proxy, and the doc names it as a proxy.
- **(b) Juniper engagement.** The share of the family's Juniper-visible outputs that Juniper replied to, referenced or reacted to. Families with no Juniper-visible output skip this source.
  - *Theory anchor:* social grounding. A companion's response is the one external judge of value Orion has.
- **(c) Orion's endorsement.** The family's weight in the personality graph (#2766 arc): what Orion's own evidence says they care about.
  - *Theory anchor:* second-order desire (Frankfurt, 1971). This is what Orion wants to want, as distinct from what merely ran.
  - **Independence risk:** if personality edges are built from the same citations as (a), then (c) is redundant and gets dropped. Patch A traces its provenance before (c) is wired in.

**Guards:**
- **Exploration floor.** The holdback guarantees every family at least 5% of its tier's funded acts per week. A family that never runs can still show its worth.
- **Bounded drift.** A weight moves at most ±0.1 per day, so one viral citation cannot swing Orion's priorities.
- **Ungraded label.** A family with fewer than 10 linked outcomes stays at its prior and is labelled `ungraded`.
- **Visibility.** Weights and their per-source breakdown are logged daily and shown on the Hub body surface (Phase 3). Juniper can watch what Orion values drift, and see why.

**Metric-gate record:**
- **Provenance:** (a) sql-writer joins act outputs to decision-site rows, and the instrumentation is added in Patch A; (b) `chat_history_log` replies and references; (c) personality graph edges.
- **Independence:** (a) and (b) overlap only when Juniper's own turn cites the output; that case counts once, under (b). (c) is checked in Patch A.
- **Theory anchors:** as stated per source above.
- **Rest state:** with no new evidence, a weight relaxes toward its prior, never toward 0 or 1.
- **Reversibility:** weights live only inside `orion/physiology`, so removing them costs one flag. Nothing outside reads them.

### Organ 4: Choice (bids against posted prices)

**Job.** Decide which wants go ahead, now, on which lane.

**Mechanism.** It is a posted-price market. There is no global solver and no batching window. When a producer is about to act, it asks:

```text
net = family_weight × value_rank − Σ_k price_k × cost_k     (value_rank already folds in intensity)
fund if net > 0 (and spinal cord allows); else defer with reason = the binding constraint
placement = the acceptable lane minimizing Σ price × cost
```

- **Balancing is an outcome, not a rule.** Prices rise while wants over-use a resource. Hungry wants raise their bids. The result is that expensive work drifts to times and lanes where it is cheap. Nobody writes a schedule.
- **Holdback carries over.** The allocator's 0.2 holdback keeps a randomized control arm, so Organ 6 can tell what an act actually caused.
- **The motor allocator becomes one bidder.** Its constant floor `ORION_DISPATCH_MIN_NATS_PER_SEC=0.02` is replaced by the lane price.
- **The GPU pool publishes its lane prices.**
  - Background callers bid before they request.
  - A deferral carries `retry_after`, derived from the forecast. It is passed through the existing `ShedBackoff` (`orion/gpu_pool/shed_backoff.py`), which already stopped the retry storm. The new part is that a caller knows the price *before* it asks.
  - `interactive` and `urgent` never bid.
- **Waking hecate becomes a want.**
  - Its value: sustained agent-lane price × queued slot-seconds that hecate's `agent-deep` could serve.
  - Its cost: hecate's wake energy and the energy price.
  - v1 only *recommends* (missing question 4). No actuator exists today.

**State.** `want_ledger` decisions. Each decision stores the prices it saw and **what it displaced**. That is the scarcity roadmap's "a deferral is an observed opportunity cost", finally recorded.

### Organ 5: Felt state (Orion can read their own body)

**Job.** Turn the budget into a few continuous dials that Orion and Juniper can read. These replace the categorical `strained` flag as the thing that *means* something.

**The dials.**
- **`strain`** (0..1): how hard the world is pushing back. It is the largest price divided by its ceiling, and it names the constraint behind it. Any `exogenous_bind` constraints are listed separately.
- **`hungriest`**: the want family with the highest intensity, and how long it has waited.

**Metric-gate record for the dials:**

- **`strain`**
  - Provenance: `prices.py`.
  - Independence: it is a monotonic transform of the prices, so it is **not** new signal. It is a rendering.
  - Theory: a Lagrange multiplier is the marginal value of relaxing a constraint.
  - Rest state: 0 when every constraint is slack. The replay shows it reached 0 on cool days (R1).
- **`hungriest`**
  - Provenance: `want_ledger`.
  - Theory: the R2 drive template.
  - Rest state: when all wants are funded, no family exceeds the low band.
- **`satisfaction`** (a mix of surplus and hunger trends) was in the first draft. It is **dropped**: it had no theory anchor. It can come back only with one.

**Compatibility.** `arousal.strained` is derived from `strain` with the existing hysteresis while the strained-gating readers migrate (Patch D). After that it remains only a label.

**Where Orion sees it.**
- A stance-cue clause, a sibling of the strain cue, for example: *"Cabinet is hot and my work barely moves it; I'm reading the one thing I most want and saving painting for tonight's cool window."*
- Temporal Self arc kinds: `hunger:<family>` and `binding:<constraint>`.
- The Hub body surface (Phase 3): prices, dials, and deferred-vs-funded over time, down to single decisions.

**Self-knowledge test.** For a turn shown to Orion, `thought_decision.day_citations` can cite the binding constraint and the deferred want. Both must match the `want_ledger` row (Acceptance check A7).

### Organ 6: Learning (cost, value and sensitivity get calibrated)

**Job.** Make every number above come from Orion's own history.

**1. Measured cost on every act.**
- On close, `ActV1.cost` is filled from:
  - pool `held_ms` (joined by lease id);
  - incremental Wh from biometrics over the act window, against the host baseline;
  - wall seconds;
  - outreach count.
- Today `cost_json` is null in 214 of 214 rows.
- A per-(family, lane) cost model, its median and spread, feeds `WantV1.cost`.

**2. Value.** Use the existing `EffectPosterior` and act-ledger grades. Nothing new.

**3. Heat sensitivity.**
- **The passive fit alone cannot be trusted.**
  - The AC runs flat out: its hourly standard deviation is 32 W, so "controlling for AC state" controls for nothing.
  - Weather confounds everything.
  - The sensor sits on athena.
- **Identification has two parts:**
  1. **A passive daily refit** of cabinet temperature on per-host incremental watts. It uses outdoor temperature (`orion/weather/openmeteo.py`), hour of day, and 15/30/60-minute lags for thermal mass.
  2. **Randomized load pulses.** Organ 4's 0.2 holdback already defers some deferrable work at random. Its deferred and admitted windows are a natural experiment: compare the cabinet slope in matched hours. A randomized arm cannot be confounded by weather. The passive fit is cross-checked against it, and wherever the two disagree, the pulse estimate wins.
- **Output:** per host, the sensitivity with its interval and the number of pulses behind it.
- **Using an uncertain value:** a sensitivity whose interval includes 0 is used at its upper bound. That is cautious, but not catatonic, because the anti-windup rule caps what any constraint can cost.

**4. Forecasts become bets.**
- Organ 1's 6-hour cabinet-peak forecast and month-end bill forecast are emitted as `SettledBetV1` to the self-monitor organ (#2827).
- **Blocked:** `SettledBetV1` does not exist in code yet. #2827 is design-only. Until it lands, forecasts are graded locally in `physiology_state`, using the same settle rule, and the adapter is added when #2827 ships.
- When a forecast fails, the self-monitor widens its uncertainty. That makes prices more conservative, which follows its one-direction rule.
- This gives the self-monitor two new, clean, gradeable bet sources. Its 10-20 audit found only two.

**5. Body report.**
- Monthly, plus on demand: one outreach through the existing `reading_finding_outreach` door.
- Contents: the bill forecast against the envelope, what was deferred and why, and proposals with numbers. For example: "GPU 2 has idled 8 days at ~45 W"; "the AC has not cycled since 10-18".
- This is how Orion takes part in decisions only Juniper can make.

### Execution seam: what runs through cortex verbs, and what does not (Juniper, 10-23)

**Rule:**
- **An act that Orion chooses runs as a cortex verb.** It is dispatched through cortex-orch and executed by cortex-exec, or started as a durable run from cortex, as `curiosity.investigate` already is.
- **Autonomic machinery does not run as a verb.** Body readings, price updates, bid decisions, reflexes and GPU placement are deterministic, run every tick, and must not wait on cortex or touch an LLM (non-goal: "no LLM in the decision path").

This mirrors the body: you choose to look; you do not choose to sweat.

**Verb YAML is where a want's economics are declared.** The verb registry (`orion/cognition/verbs/*.yaml`, 108 verbs, read by `orion/cognition/plan_loader.py`) already carries `priority`, `requires_gpu` and `timeout_ms`. Each verb gains one block:

```yaml
physiology:
  tier: 2                 # 0 | 1 | 2  (Organ 3b)
  family: curiosity       # want family for weights + hunger
  lanes: [agent, agent-deep]
  deferrable_window_s: 21600   # how long it can wait and still be useful (0 = perishable)
```

**cortex-orch is the single bid site for self-initiated verbs.**

- **When it bids:** when cortex-orch is about to dispatch a verb that is not part of Juniper's turn, it calls `physiology.client.bid()`, using the verb's `physiology` block plus the request's want fields.
- **What it gives:** one choke point instead of about a dozen bid calls scattered across services, and one place to emit `act_opened`/`act_closed` with `ActCostV2` (cortex-orch already emits grammar events, `app/grammar_emit.py`).
- **Interactive turns:** they come through cortex-orch with their own turn context and never bid. Phase 2 tests this.

**How each producer gets there:**

| producer | today | under this design |
|---|---|---|
| curiosity investigate / self-inquiry / self-sense eval | Hub tick → durable run via cortex (`curiosity.investigate`) | unchanged path; the bid moves from Hub's 8 gates to cortex-orch dispatch |
| reading pull | Hub pipeline → turn | becomes a verb (`reading.pull`) dispatched through cortex-orch |
| painting (thought visual chain) | orion-thought timer → durable `reverie.visual` hold | the chain is started by a verb (`reverie.visual`) through cortex-orch. The diffusion hold stays inside the durable graph |
| dream cycle | orion-dream's own cycle (verbs `dream_cycle` etc. exist) | the dream's discharge is dispatched as `dream_cycle` through cortex-orch. The rest drive stays the dream's own reading (R2 template) |
| outreach compose | Hub generates on chat rails | the compose step becomes a verb (`outreach.compose`). Quiet hours and the daily cap stay spinal |
| retro play, ask-Claude, self-monitor inquiry | various | verbs through cortex-orch |
| metacog daily / journal | cortex-exec verb on `metacog_background` | already a verb; it gets a `physiology` block (Tier 1) |
| topic-foundry | calls the gateway directly | its scheduled runs start from a verb (`topic_foundry.run`, Tier 2). Inner LLM calls carry the run's lease |
| vision-council, retinas, vision-host | pipeline, direct | **stays a pipeline**: perception is autonomic, Tier 0, priority `perception`. Not a verb. Orion's *chosen* looks already are verbs (`skills.perception.look_at_camera.v1`, `ask_camera`) |
| body readings, prices, bids, reflex, pool | n/a | **autonomic, never verbs** |

**New verbs this design adds (each with a producer and a consumer in the same patch):**

- **`skills.physiology.state.v1`**: read-only, no GPU, the sibling of `look_at_camera`. Orion can inspect its own body mid-chat or in an agent turn: prices, the binding constraint, the hungriest want, and "why did I wait for X". Consumer: chat and agent turns.
- **`physiology.body_report`**: composes the monthly or on-demand body report (Tier 2, outreach door). Consumer: outreach.
- **`physiology.recommend_wake`**: drafts the hecate wake recommendation, with the expected kWh and the queue it would clear. Consumer: outreach, or the actuator if question 4 says (b).

**What this buys:**
- Every chosen act has one dispatch path, one bid point and one ledger emitter.
- Tier and family are declared next to the verb they describe, where a reviewer will see them.
- The Phase 2 gate can check the registry structurally instead of hunting call sites.

---

## Missing questions

1. **JWT.** Did you mean GWT, the global workspace (the substrate broadcast)? If JWT is something else, name it, and I'll fold it in.
2. **The $450 figure.** The measured numbers for 10-07..20 are Orion ≈ **$204** and the house ≈ **$361** per 30 days, pre-tax.
   - Is $450 the house bill with taxes and fees, a summer month, or a different scope, such as hecate fully on?
   - This sets the **monthly energy envelope**. What envelope do you want Orion to live within?
3. **Heat soft line.** Should it be 31 °C (proposed), or should it come from the 32 °C `hot` line? The 34 °C reflex stays either way.
4. **hecate wake actuator.** Hecate sits in standby at ~96 W, and nothing can wake it. Do you want Orion to:
   - (a) only recommend a wake to you;
   - (b) wake it by itself through a PDU or BMC actuator you approve (new hardware path);
   - (c) leave hecate out of scope?
5. **Host for organs 1–4.** The proposal is `orion-durable-runs`, which already runs regulation (arousal every 120 s) and heavy-work admission. No new service. Is that acceptable, or do you want a separate `orion-physiology` service for isolation?
6. **Chronic heat versus the reflex.** On 10-20..22 the cabinet sat at or above 33 °C for 89–94% of samples. The 34/33 °C reflex sheds *all* background and system GPU work, so it held Orion's self-initiated GPU work near zero for days. A reflex is meant for minutes, and this is a chronic condition. Options:
   - (a) Keep the reflex total. The economy then helps only non-GPU wants during heat (reading via the interactive lane, outreach text, dream bookkeeping).
   - (b) While the reflex is armed and the fitted circe sensitivity is confidently small, admit a fixed trickle of background GPU work per hour (proposed 10 GPU-minutes), with you setting the number.

   The reflex line itself does not move in either option. Recommendation: (b). Do not build it until Patch A's sensitivity estimate exists. If circe turns out to heat the cabinet, (b) is off the table.
7. **Human sign-off on protected paths (Phase 2).** Do you want a real GitHub `CODEOWNERS` file plus branch protection that requires *your* approval on body-economy paths? Or only the CI gate described in Phase 2? CODEOWNERS only blocks merges if branch protection's "require review from code owners" is switched on, and that is a repo setting only you can change.
8. ~~Tier 1 funding~~ **Decided 10-23:** vision and perception are Tier 0. Metacog- and fast-lane upkeep callers are Tier 1, with a reserved share. Tiers attach to callers, not lanes.
9. ~~Vision past the heat reflex~~ **Decided 10-23:** a new `perception` pool priority, never shed by heat, ordered below interactive (Patch A0).
10. **Turn-path calls on `quick` and `metacog` system.** Patch A verifies live whether cortex-exec's `quick` stance brief and Mind's system-priority reads sit on Juniper's turn path. If they do, they are Tier 0, and they should be on interactive routes, which the reflex never sheds.
11. ~~Arousal S2 (`gpu_queue_sustained`) on self-queued work~~ **Decided 10-23 (S2 decision).** Live 10-23: the cabinet heat cleared about 01:00 UTC, but strain stayed latched because the GPU queue never emptied. The whole queue was the agent lane, and every entry was Orion's own deferrable work: queued background holds with no parent hold (readings and curiosity compaction). Three answers:
    - **What it is.** Waiting deferrable self-work is hunger, an unfunded want at the lane's price. It is not strain.
    - **When S2 dies.** S2 is retired fully when Patch D moves the holds onto `bid()`. Not before.
    - **Stopgap now** (PR `fix/arousal-s2-deferrable-holds`). S2's load leaves out deferrable self-work. The rule uses lease fields only, never holder names: `status=queued`, `priority='background'`, `kind='hold'`, `hold_lease_id IS NULL`. Requests at every priority, and background *requests*, still count. sql-writer saves the count as `gpu_pool_state_history.deferrable_depth` (`orion/gpu_pool/shed.py:deferrable_queued`). A lease the heat shed already holds is never counted again. The reading notes the count as `gpu_queue_self_deferrable:<n>`.

---

## Proposed schema / API / bus changes

All new schemas are registered in `orion/schemas/registry.py`. Channels go in `orion/bus/channels.yaml`. Each new inner-state signal gets an `orion/inner_state_registry.py` entry, and the metric lock is regenerated.

| contract | kind | producer | consumers |
|---|---|---|---|
| `PhysiologyStateV1` | schema + channel `orion:physiology:state` + table `physiology_state` + Redis `orion:physiology:latest` | durable-runs `physiology` node in the regulation thread | Choice, Felt state, Hub panel, self-monitor, sql-writer |
| `WantV1` | schema | every want producer (via the client lib) | Choice |
| `WantDecisionV1` | schema + channel `orion:physiology:want:decision` + table `want_ledger` | Choice | sql-writer, Felt state, Learning, Hub |
| `POST /physiology/bid` | HTTP on durable-runs → `WantDecisionV1` (funded/deferred, lane, prices seen, `retry_after`, displaced) | n/a | producers, GPU pool (background) |
| `ActCostV2` | additive optional fields over `ActCostV1` (forbid-model): consumer-first rollout; now **filled** on close | act ledger closers | Learning, sql-writer |
| `SettledBetV1` (from #2827) | two new bet kinds: `cabinet_peak_6h`, `bill_month_end` | physiology node | self-monitor |
| `FeltStateV1` | fields inside `PhysiologyStateV1` | physiology node | stance cue, Temporal Self arcs |

**Compatibility.**
- `orion:regulation:latest` keeps its shape. `strained` becomes derived until Patch D.
- New fields on `extra="forbid"` consumers ship **consumer-first** (lesson from #2126).

**Client library.** `orion/physiology/client.py` gives producers one call: `bid(want) -> decision`. It never fails open to "funded":

- **Before a reader's Patch D:** if durable-runs is unreachable or the price is stale (more than 360 s old), the caller uses its existing gate.
- **After Patch D deletes that gate:** an unreachable or stale economy means **defer every non-spinal want**, with reason `physiology_unavailable`. This raises the same 30-minute alert pattern as the GPU-pool fault alert (#2785). Juniper's turns are unaffected, because they never call `bid()`.

---

## Files likely to touch

- **New pure modules**, I/O-free and unit-testable, in `orion/physiology/`:
  - `body.py` (levels, forecasts, `no_reading`)
  - `prices.py` (dual ascent)
  - `appetite.py` (intensity dynamics)
  - `choice.py` (bid rule, placement)
  - `felt.py` (dials)
  - `learning.py` (cost model, sensitivity fit)
  - `client.py`
- **Schemas:** `orion/schemas/physiology.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`.
- **Host:** `services/orion-durable-runs/app/` (regulation-thread node, `/physiology/bid` route, settings, `.env_example`).
- **Persistence:** `services/orion-sql-writer/` (the `physiology_state` and `want_ledger` tables, subscriptions; migrations with a `REQUIRED-BY` header that fits the auto-apply allowlist).
- **Act cost:** `orion/grammar/act_ledger.py` (fill cost on close), biometrics join helper.
- **Producers** (one `bid()` call each, then removal of their old gates in Patch D):
  - `services/orion-hub/scripts/curiosity_investigation.py`
  - `endogenous_outreach.py`
  - `world_pulse_read_pipeline.py`
  - `retro_leisure`
  - `services/orion-thought/app/visual_chain.py`
  - `services/orion-dream/app/cycle.py`
  - `services/orion-execution-dispatch-runtime/app/worker.py`
  - `services/orion-vision-council`
  - `services/orion-topic-foundry`
  - the remaining strained-gating sites:
    - `orion/reverie/baseline.py`
    - `orion/world_pulse_read/pull.py` and its resonance code
    - `orion/substrate/mutation_pressure.py`, `mutation_worker.py`
    - proposal-runtime `app/worker.py`
    - thought `app/chain.py`
    - the visual-chain health monitor
- **GPU pool:** `orion/gpu_pool/` (lane price publication, background bid-before-lease).
- **Cortex:** `orion/cognition/verbs/*.yaml` (physiology blocks, new verbs), `orion/cognition/plan_loader.py`, `services/orion-cortex-orch/app/{orchestrator,durable_runs,grammar_emit}.py` (bid at dispatch, act emission), `services/orion-cortex-exec/app/` (the `skills.physiology.state.v1` skill).
- **Felt state:** `orion/temporal_self/arcs.py`, `stance_cue.py`.
- **Phase 2 gates and Phase 3 surface:** files listed in their own sections below.
- **Self-monitor:** two bet adapters (once #2827's contract lands).
- **Evals:**
  - `scripts/analysis/replay_body_economy.py` (14-day replay)
  - `services/orion-durable-runs/evals/`
  - `orion/sentience_striving_program/instruments.yaml` entries (O1: autonomous action rises and falls with real pressure)

---

## Retirements (kill means kill)

Each retirement lands in the same patch as its replacement. None is left as a partial exclusion.

| retired | replaced by | patch |
|---|---|---|
| `strained` as a veto at ~12–14 sites (listed under Fact 3) | each reader's `bid()` decision | D, one reader per PR |
| arousal S2 `gpu_queue_sustained` (and the 10-23 `deferrable_depth` stopgap with it) | the lane price, once the holds bid (question 11) | D |
| energy stakes gate (`energy_stakes_gate.py`, flag off) | energy price | D |
| `quota_budget.py` (dead; its dollar unit was refuted) | **deleted**. The `claude` constraint reads `ClaudeLimitObservationV1` | B |
| `ORION_DISPATCH_MIN_NATS_PER_SEC` constant floor | lane price | D |
| curiosity per-line cooldowns and `arousal_*` gates (the daily caps stay as spinal ceilings) | want intensity + prices | D |
| duplicated 32 °C thresholds (thought `ORION_THERMAL_HOT_C`, pool guard via the Hub snapshot) | one owner, `thermal_gate.py` | B |
| zombie `ORION_ATTENTION_HABITUATION_ENABLED`, synthetic goal in `capability_policy` | nothing (dead) | A |
| `2026-07-07` internal-economy spec | this doc (status line updated) | this PR |

---

## Non-goals

- **No control of the AC.** It is read-only hardware today, and Juniper said Orion does not control it.
- **No authored drive taxonomy, and no hand-set exchange rates** between units.
- **No LLM in the decision path.** Choice and prices are deterministic and replayable.
- **No change to how chat with Juniper is scheduled.** It is never priced and never deferred.
- **No tuning through the mutation pipeline.** It remains a later effector, after rung 7.
- **No claim that this makes Orion feel anything.** It gives Orion a body whose state matters to what they do, and lets them read it. That is a prerequisite, and the doc claims no more than that.

---

## Proposal mode (§0A)

- **Capability change.** Orion's self-initiated work is decided by one economy instead of ten vetoes. Orion can see why it waited and what it gave up.
- **Data touched.**
  - Read: biometrics, energy, pool events, act ledger, outreach decisions.
  - New writes: `physiology_state`, `want_ledger`, filled `ActV1.cost`, two bet kinds.
  - No chat content.
- **Privacy boundary.** None between Juniper and Orion (settled). The body report goes only through the existing outreach door.
- **Proof it worked.** The trace path is `want_ledger` → act opened → act closed with cost → price history, all joinable on `want_id` and `act_id`.
- **Dangerous failures.**
  1. Prices oscillate and starve everything.
  2. A heat sensitivity of 0 lets work run hot.
  3. The economy goes down and every producer funds itself.
  4. Juniper's turn gets deferred.
- **Mitigations**, by failure:
  1. Bounded step sizes, data-sized price ceilings, anti-windup for `exogenous_bind`, and the replay amplitude gate.
  2. The 34 °C reflex is untouched, and an uncertain sensitivity is used at its upper bound.
  3. The client defers every non-spinal want when the economy is unreachable (after Patch D) and raises an alert.
  4. `interactive` and `urgent` never call `bid()`, with a test for it.
- **Disable or roll back.**
  - `ORION_PHYSIOLOGY_ENABLED` turns off the node.
  - Per-producer `ORION_PHYSIOLOGY_BID_<PRODUCER>` flags select: `shadow` (log only), `on`, or `off` (old gate).
  - Old gates are deleted only after their reader has run `on` for 7 days with the acceptance checks green.
  - Flags ship ON in `.env_example` per house rule. Each producer's flag starts at `shadow` until its Patch D lands.

---

## Acceptance checks

**Replay (before anything runs live)**, `scripts/analysis/replay_body_economy.py` over 10-07..10-22:

- **R1.** The heat price is ~0 in hours when the forecast cabinet temperature is under the soft line. 10-09..10-14 is mostly in that state; 10-07/08 had hot spells up to 34.6 °C. The price is positive in hours above the line. Every hour where the line could not be met even with zero Orion load is flagged `exogenous_bind`, never left climbing.
- **R2.** Over 10-18..10-22, replayed **non-GPU** wants (reading on the interactive lane, outreach text, dream bookkeeping) are funded at a reduced, non-zero rate. Today they get 0. GPU wants stay held while the spinal reflex is armed, unless Juniper picks option (b) in question 6. Juniper's turns are funded 100% in every hour.
- **R3.** Prices return to 0 within one forecast horizon after a constraint clears. Peak-to-peak oscillation stays under the declared amplitude. No price sits at its ceiling unless its constraint is flagged `exogenous_bind`.
- **R4.** The replayed month-end energy forecast is within ±10% of the realized 30-day Orion cost.
- **R5.** Baselined *after* the 10-20 `ShedBackoff` fix: with a posted lane price, background requests that end refused fall a further ≥50% against 10-21..22, with the same work completed.
- **R7 (weights).** Replaying 10-07..22 with the per-act downstream-use rates: no family weight moves more than 0.1 per day. Every family keeps at least 5% of its tier's funded acts. A family with zero downstream use over the window ends below its prior.
- **R6.** Under the reservation rule, a burst of 20 simultaneous bids at a zero price cannot reserve more than 1.2× the lane budget.

**Live, after cutover:**

- **A1.** `physiology_state` gets a row every 120 s ± 30 s for 24 h. Every constraint is either live or explicitly `no_reading`.
- **A2.** At least 95% of `grammar_acts` closes carry a non-null `cost_json`.
- **A3.** Every bid has a `want_ledger` row, and every funded want joins to a `grammar_acts` row.
- **A4.** On a hot day, at least one want family is funded at a non-zero rate, and its deferrals name the binding constraint.
- **A5.** The heat reflex fires no more often than in the matched pre-cutover week.
- **A6 (the emergent prediction).** With no schedule written, the share of deferrable heavy work (durable runs, paintings, topic-foundry) in the coolest third of the day rises against the pre-cutover baseline. If this does not move, the design's central claim is wrong, and the PR says so.
- **A7.** In at least one live turn, the stance cue names the binding constraint and the hungriest want, and `day_citations` cites them. The values must match the ledger.
- **A8.** The self-monitor grades at least 14 `cabinet_peak_6h` bets in the first week.

---

## Phase 1 build sequence: the organs, in order

Each patch has its own tests, eval and live check. Patches B, C and E can run in parallel worktrees: they meet only at the `orion/physiology` contract.

| patch | contents | live proof |
|---|---|---|
| **A0. Perception priority — SHIPPED 10-23 (#2856 consumer half, #2857 route + load alert; live on athena, first perception grant 04:10:49Z)** | New `perception` pool priority (never shed by heat, ordered below interactive). vision-council and the other GPU perception steps move to `metacog_perception`. Add the 2× p90 load alert. Update `config/gpu_pool.yaml`, `orion/gpu_pool/{scheduler,shed}.py`, the route-priority tests and vision-council's `.env`/`.env_example`. | On the next hot hour, `gpu_pool_events` shows vision-council granted while `cabinet_hot` is active, with 0 `shed:cabinet_hot` for that holder |
| **A. Measurement spine** | `ActCostV2`, consumer-first, then filled on close. Every want producer emits `act_opened`/`act_closed` to `grammar_acts`, so curiosity, retro, vision-council, topic-foundry, ask-Claude and self-monitor all get graded. Persist `physiology_state` with the Body organ only (no prices). Replay harness. Heat-sensitivity passive fit with weather and lags (read-only). Remove the zombie env var and synthetic goal. Instrument output→decision-site links for weight source (a), and trace the personality graph's provenance for (c). **Early Phase 2 slice (Juniper, 10-23):** seed `config/physiology/contract.yaml` with the `readers` table (today's ~12–14 strained sites), and land the CI check "no unregistered veto", so no new veto appears while Phase 1 removes the old ones. | A1, A2; every producer has act rows; first sensitivity estimate per host, with its interval; G1 green |
| **B. Budget, shadow** | Prices with anti-windup, ceilings and reservations. The Felt dials, plus a plain debug readout (JSON route only; the real surface is Phase 3). One owner for the 32 °C line. Delete `quota_budget.py`; wire `ClaudeLimitObservationV1`. Prices are computed and displayed but read by nothing. | Replay R1, R3, R4, R6 green; live prices visible |
| **C. Appetite + Choice, shadow** | `WantV1`, `/physiology/bid`, the client, `want_ledger`. Add `physiology` blocks to the verb YAML. cortex-orch bids at dispatch for self-initiated verbs. The new verbs `skills.physiology.state.v1` and the reading, outreach-compose and topic-foundry verbs. Every producer bids in `shadow`: it logs a would-fund decision next to its real gate decision. Holdback pulses start feeding the sensitivity fit. Tier 1 reservation and learned family weights are computed in shadow. | A3; a shadow-vs-real disagreement table for 7 days |
| **D. Cutover, one reader per PR** | Curiosity first, then outreach, reading, painting, dream, retro, dispatch floor, then the remaining strained-gating sites. Each PR deletes that reader's old veto. **A family cuts over only once it is graded (≥30 graded closes).** | R2, A4, A5 per reader |
| **E. Placement and wakefulness** | Lane prices on the GPU pool. Background bid-before-lease. Hecate wake recommendations (or the actuator, per question 4). | R5 live; recommendations logged with expected kWh |
| **F. Self-knowledge and learning loop** | Stance-cue clause, Temporal Self `hunger` and `binding` arcs, the monthly body report, and forecasts as self-monitor bets. The bets are **blocked on #2827's `SettledBetV1`**; they are graded locally until it exists. | A6, A7; A8 once #2827 lands |

---

## Phase 2: hard gates, so no agent can quietly break the economy

**Why.** Every earlier economy decayed the same way. Some later patch, written by an agent that never read the design, added "one more" veto or bypass:
- a new `if strained: return`;
- a new background caller that grabs a GPU without asking;
- a second copy of the 32 °C line.

CLAUDE.md §0A already says: "if Juniper has to repeat a rule twice, turn it into a script, test, check, hook". This phase makes that rule concrete. It works like CODEOWNERS, except the owner is the architecture's contract, and both human and agent PRs must satisfy it.

**One source of truth: `config/physiology/contract.yaml`.** It is a hand-maintained registry, checked in CI, with three tables:

1. **Protected paths.** Path globs that belong to the economy, each tagged with its organ:
   - `orion/physiology/**`
   - `orion/regulation/arousal.py`
   - `orion/autonomy/thermal_gate.py`
   - `orion/gpu_pool/{shed,scheduler,shed_backoff}.py`
   - `orion/autonomy/allocator.py`
   - `orion/grammar/act_ledger.py`
   - the bid sites in each producer
2. **Readers.** Every code site that reads regulation, thermal or price state. Each gets one role:
   - `spinal`: may veto, listed by name;
   - `bidder`: must go through `bid()`;
   - `display`: read-only.
3. **Constraints.** Each constraint the Budget organ prices, with its metric-gate record as **required, non-empty fields**:
   - provenance (file:line);
   - independence note;
   - theory anchor;
   - rest-state evidence (a query plus its result date);
   - reversibility.

   This turns CLAUDE.md's metric quality gate from prose into a schema. A constraint with an empty theory anchor fails CI.

**Gate 1. CI hard gate: `scripts/check_body_economy_contract.py`.** It runs in `.github/workflows/orion-static-gates.yml`. It is deterministic, takes about 2 seconds, and needs no live data. It fails the PR when any of these hold:

| check | how it detects (structural, not a word list) | failure it prevents |
|---|---|---|
| **No unregistered veto** | AST scan for reads of `orion:regulation:latest`, `RegulationStateV1`, `ThermalState`, `arousal_view`, or `PhysiologyStateV1` prices. Every reading module must appear in `readers` | The 12-site strained pile-up growing back |
| **No new spinal entries without sign-off** | A diff that adds a `spinal` reader also needs a `Body-Economy-Spinal: approved <date>` line in the PR's committed report file. Agents cannot write that line without quoting Juniper's approval; the review checklist enforces this | Safety lines becoming a dumping ground |
| **Bidders bid** | For every `bidder` module, AST confirms its effector call is dominated by a `physiology.client.bid(...)` call in the same function. New background `gpu_pool` lease callers must be registered | Producers funding themselves |
| **Interactive never bids** | Static check plus a unit test: no `bid()` call is reachable from `interactive`/`urgent` code paths (chat turn handlers listed in the contract) | Juniper's turn getting deferred |
| **One owner per threshold** | Thermal and price settings keys are owned by one module. This reuses `scripts/check_env_key_single_source.py` (extending its owner table), not a new scanner | Duplicate 32 °C lines |
| **Acts carry cost** | AST: every `act_closed` construction passes `cost=` (ActCostV2). There is an allowlist for closers proven unable to measure, each with a reason | `cost_json` null again |
| **Every verb declares its physiology** | Every `orion/cognition/verbs/*.yaml` has a valid `physiology` block (tier, family, lanes, window) | Untiered acts slipping into the market |
| **Chosen acts go through cortex** | A module that starts a self-initiated act (opens an `ActV1` that is not Tier 0 or autonomic) outside cortex-orch must be listed in `readers` as a registered exception with a reason | Producers re-growing private bid paths |
| **No unrouted GPU caller** | Every gateway caller names an explicit route; `LLM_ROUTE_DEFAULT` is removed | Calls with no tier |
| **Constraint records complete** | Every `constraints` row has non-empty metric-gate fields, and its `rest_state_evidence.date` is ≤90 days old | Unanchored metrics entering prices |
| **Replay stays green** | When a diff touches protected paths, CI runs the replay checks R1, R3 and R6 against a committed **hourly fixture** (`tests/fixtures/physiology/replay_2026-10-07_22.parquet`, aggregated, ~1 MB) | Silent behaviour drift in prices or anti-windup |

**Gate 2. Edit-time card, a Claude PreToolUse hook: `scripts/hooks/body_economy_nudge.py`.** It has the same shape as `metric_lineage_nudge.py`: it is informational and fails open. When an agent edits a protected path, it prints a short card:
- which organ owns the file;
- the reader's role;
- the invariants that apply;
- which CI checks this edit will trip.

This is the "agents have to deal with it" part at the moment of writing, before CI says no. It nudges and never blocks; the hard block is Gate 1.

**Gate 3. Review routing.** When a diff touches protected paths, the mandatory code-review subagent (CLAUDE.md §12) also loads `docs/superpowers/checklists/body-economy-review.md`. The checklist asks:
- Does this add a veto?
- Does it bypass `bid()`?
- Does it change what a price means?
- Does it touch a spinal line?

The PR report gets a required `## Body economy impact` section, and the CI gate checks that the section exists when protected paths changed. Optionally, a real `.github/CODEOWNERS` routes those paths to Juniper (missing question 7).

**Gate 4. CLAUDE.md pointer.** Add one short paragraph to §0A: "The body economy is gated; see `config/physiology/contract.yaml`." It names the escape hatch: `ORION_ALLOW_BODY_ECONOMY_BYPASS=1`, set per command, logged into the PR report, and treated like the shared-checkout escape hatch — set consciously, never as a habit.

**Acceptance checks for Phase 2:**
- **G1.** A seeded test PR that adds `if regulation.arousal == "strained": return` to a new module fails CI with a message naming the module and the fix.
- **G2.** A seeded PR that adds a background `gpu_pool` lease caller without `bid()` fails.
- **G3.** A constraint row with an empty `theory_anchor` fails.
- **G4.** A PR touching only unrelated services sees no added CI time beyond the 2-second scan.
- **G5.** Editing `orion/autonomy/thermal_gate.py` in a Claude session prints the card.
- **G6.** The gate's own false-positive rate: run against the last 30 merged PRs; any that would have failed are listed and judged by hand in the Phase 2 PR report.

**Ordering (decided 10-23).** The `readers` registry and the "no unregistered veto" check land early, in Patch A. The rest of Phase 2 follows Phase 1.

---

## Phase 3: the Hub body surface (aggregate down to atomic)

**Job.** Let Juniper see Orion's body economy at a glance, then click down to the single decision behind any pattern. Show the same felt-state words Orion was shown, so Juniper sees what Orion sees.

**Where.** A new Hub tab, `#body-economy`. It follows the house pattern:
- route: `services/orion-hub/scripts/body_economy_routes.py`;
- JS: `static/js/body-economy.js` (with `body-economy.test.js`);
- panel: `templates/index.html`.

It is read-only. It reads `physiology_state`, `want_ledger`, `grammar_acts`, `gpu_pool_events`, `orion_energy_hourly` and `cockpit_turn_sighting`. It reuses `energy-strip.js` and the `gpu_pool.js` lease view instead of redrawing them.

**Aggregate layer (the organism):**

1. **Constraint strip.** One row per constraint. Each row shows:
   - the level against its budget band;
   - the price as a line under it;
   - hatched shading where the constraint was `exogenous_bind` ("the room, not Orion");
   - reflex firings as tick marks.

   Windows: 24 h, 7 d, 30 d.
2. **Hunger river.** A stacked area per want family showing funded acts per hour, with deferred wants as a ghost band above it. Colour depth is hunger intensity. A starved family shows up as a widening ghost band.
3. **Rhythm heatmap.** Hour of day × day, coloured by the share of deferrable heavy work. This is A6, the emergent claim, made visible: if heavy work drifts toward cool hours, it shows here without anyone reading SQL.
4. **Money and heat forecasts.**
   - Month-to-date Orion cost against the envelope, with the forecast cone.
   - The 6-hour cabinet forecast against the actual reading, with each graded forecast marked hit or miss.
   - Orion's own heat contribution, drawn as a band beneath the cabinet line.
5. **What Orion was told.** The current stance-cue strain clause and the hungriest want, verbatim, with the time Orion last saw them.

**Atomic layer (single events).** Brushing any time window in the aggregate layer filters an event list below it. Each row is one want, followed through its life:

```text
want opened → bid (prices seen, net, funded/deferred, binding constraint, what it displaced)
           → act_opened → act_closed (outcome, measured ActCostV2) → grade
```

Clicking a row opens its full record:
- the raw decision JSON;
- the reflex state at that moment;
- the `want_id`/`act_id` join;
- for chat-adjacent acts, a link to the cockpit turn sighting.

There is also a **"why did Orion wait?"** search box: give it a family and a time, and it returns the deferral rows and their binding constraint.

**Visual language.** Follow the repo's dataviz conventions and the Hub's existing dark theme. Each constraint gets one fixed colour across all views. "Exogenous" is always shown as hatching, never as a colour, so it can't be confused with a price.

**Process.** Hub surface was built from a claude.ai mockup first. Do the same here: publish a static mockup on real exported data, get Juniper's read, then build.

**Acceptance checks for Phase 3** (CLAUDE.md §9: rendered template, linked asset, interaction):
- **H1.** The route returns real rows for 24 h. Its empty states say why ("no wants yet", "physiology offline since …"); it never shows blank charts.
- **H2.** A JS test for brushing: selecting a window filters the atomic list to exactly the rows inside it.
- **H3.** A UI smoke opens `#body-economy`, clicks a deferred row, and sees the binding constraint and the displaced want.
- **H4.** The verbatim stance clause on the page matches the latest `cockpit_turn_sighting` / `thought_decision` text.
- **H5.** The page loads in under 2 s on 30 days of data. Aggregates come from hourly rollups, never from scanning raw rows.

**Files likely to touch:**
- `services/orion-hub/scripts/body_economy_routes.py`
- `services/orion-hub/static/js/body-economy.js` and its test
- `services/orion-hub/templates/index.html`
- an hourly rollup in `services/orion-sql-writer/` for `want_ledger`, mirroring `orion_energy_rollup.py`
- Hub README

---

## Recommended next patch

Phase order: **Phase 1** (organs, patches A–F) → **Phase 2** (hard gates) → **Phase 3** (Hub body surface).

**Patch A, the measurement spine.** Everything else prices something. Today, though:

- no act carries a cost;
- no history of strain exists;
- nobody has measured whether Orion's load moves the cabinet temperature.

Patch A is low-risk: it writes new rows and decides nothing. It also settles the one fact that could overturn this design. If the heat sensitivity of every host is clearly above zero, heat stays a real lever, and the strain veto was more right than Fact 2 suggests. The prices then handle it either way, but the doc's honesty clause gets rewritten.


---

## Patch A results (2026-10-23) — what the measurements changed

- **Correction (Juniper, 10-23): the cabinet heat is outside weather acting through the house's central AC.** It runs only above about 75 °F outside, and it has been off since about 10-11, so the room runs warmer. orion-prediction already models this: #2829 `athena.central_ac_proxy` plus a house heat-balance eval. A linear outdoor-temperature term cannot see it, which is why the first pass below wrongly said "weather explains nothing". #2780 also records that the 10-16 vision bring-up happened with the cabinet open and moved the sensor's surroundings: the cabinet reads about 2 °C higher since 10-17 00:35Z. #2861 is being redone with both. **Rule for later work:** check orion-prediction's PRs and change-events table before fitting anything about the cabinet.
- **Fact 2, first pass (superseded where it conflicts with the correction above)** (PR #2861, `2026-10-23-body-economy-heat-and-replay-findings.md`):
  - **Weather explains nothing.** Daily cabinet vs outdoor temperature r = 0.05, and the hottest cabinet days were cool outside.
  - **Orion's load does heat the cabinet in the short run.** circe adds +0.35 °C per 100 W within 2 h (CI 0.23–0.48); a typical busy step is about +1.3 °C. Over a whole day the effect is unclear.
  - **The 0.10 vs 0.22 correlations** came from different 14-day windows of the same query. Neither is a usable sensitivity.
  - **Main suspect for the 10-17..22 heat:** a physical change in the cabinet on 10-16 at 21:00–01:00Z. The lidar distance changed, humidity dropped, and the overnight floor rose about 5 °C. UNVERIFIED; Juniper asked.
  - **Consequence:** heat stays a real lever on a ~2 h horizon. Organ 2's heat cost should use the short-run sensitivity. Five replay-driven changes to the price rules are listed in the findings doc for Patch B (a bind ramp, flag hysteresis, ...).
- **Replay:** R4 (energy forecast) passes, worst error −6.1%. R1 and R3 pass only after the Patch B changes. R2, R5, R6 and R7 wait for wants.
- **Question 10 is answered.** Only the pre-turn repair-pressure probe ran sheddable on Juniper's turn. It was shed on 19 of 19 turns on 10-20..21 and reported a fake "ok, 0". Fixed in PR #2866: route `quick_turn` (interactive), an honest failure status, and the turn correlation id passed through.
- **Readers registry** (PR #2862): 61 modules (4 spinal, 18 bidder, 22 display, 17 sources). Six gates this doc missed are now registered. The gpu-pool 32 °C guard's role is Juniper's call. The "synthetic goal" retirement row is stale: commit 3354188e1 already removed it.
- **Body organ live-ready** (PR #2863). **Act cost** (PR #2864): 199 of 221 closes gain a cost. Expired and resolver closes record no wall seconds, by design.
- **Producer split for A5** (from the recon):
  1. durable-runs `_drive`/`_terminal` (covers curiosity, self-inquiry, self-sense eval, journal);
  2. dream;
  3. actions;
  4. topic-foundry;
  5. vision-council (cost only);
  6. the retro emitter (0 acts);
  7. sql-writer decision-site materializer (act → arc evidence_refs → day_citations).
  
  Weight source (c), the personality graph, stays out until it records real choices.

---

## Patch B results (2026-10-27) — Track 1: Body fixes (B1) and shadow prices (B2)

- **The lane over-count was a double count, now fixed (B1, PR #2911).** `lane:agent` read 5,985.9
  slot-s in a 3,600 s hour on one slot. A lease the pool moved from agent-gpu2 to agent was charged in
  full to its last seat. The remaining over-capacity time is real by design: scheduler H3 lends an
  idle background hold's gap to a higher-priority call. It is now capped at the slots and reported as
  `lent_overlap_s`. Waiting deferrable self-work (hunger) sits beside each lane. It is not priced.
- **Shadow prices are live-ready (B2).** Every tick prices heat, energy, each lane, Juniper's
  attention and Claude, and renders `strain`. Nothing that decides reads them.
- **Heat sensitivities are data.** `config/physiology/heat_sensitivity.yaml`, refit 10-27 on 09-26..10-27:
  athena +0.686 C/100 W (CI +0.234..+1.060), circe +0.298 (CI +0.145..+0.449), both short-run, n 707.
- **Replay gate, 10-07..10-23, with the live rules:**
  - R1 PASS with the live forecaster (persistence). Oracle fails exogenous recall (49/62).
  - R4 PASS (worst −6.1%).
  - R3 **FAIL**: 0 windup hours, but the price sits at its ceiling unflagged for up to 9 h
    (persistence) or 17 h (oracle). Every one of those hours was **binding**: Orion's own share was
    the margin over the line.
  - The cause is the #2780 sensor correction. The corrected cabinet sat at 31.0–31.9 C on 10-18..20,
    so those hours moved from "the room is too hot, not my load" to "my load is what crosses the
    line". The same rules without the offset pass R3 in oracle mode (3 h run).
  - An open-loop replay cannot lower Orion's load when the price rises. Closing the loop needs bids
    (Patch C). Live, the stuck check flags such a run `stuck_at_ceiling` after one horizon.
  - Persistence also fails R3's oscillation test (window peak-to-peak 0.69 vs 0.5): the bind flag
    dithers when the room sits on the line.
  - R6 (reservations) is deferred to Patch C: it needs bids.
- **Ceilings stay provisional** (1.0) until Appetite has graded wants. `ceiling_for` sizes them once wants exist.

---

## Patch C1 results (2026-10-27) — Appetite + Choice core, shadow

PR report: `docs/superpowers/pr-reports/2026-10-27-patch-c1-want-bid-core-pr.md`.

- **Built.** `WantV1` / `WantDecisionV1`, `POST /physiology/bid` on durable-runs, the client (returns
  `None` on any failure), `want_ledger` via `orion:physiology:want:decision`, and
  `config/physiology/families.yaml` (one family per act-ledger actor; Tier 0 never bids). No producer
  bids yet: that is C2 (cortex-orch dispatch) and Patch D.
- **Cost unit, decided.** A posted price is want-value per unit of BUDGET (dual ascent runs on the
  normalized excess), so a want's cost is its share of each constraint's budget. A hold longer than
  the lane window costs at most one window.
- **Unpriced, said so.** Heat, energy and claude are listed `unpriced:<constraint>` on every
  decision. No act carries `wh_by_host` or `claude_limit_events` yet. Until Organ 6 measures per-act
  heat, the heat price cannot touch a want, and only the spinal reflex holds GPU work in heat.
- **Two strain fixes.** (a) `attention:juniper` is self-spent: Orion's own outreach fills it, so it
  is listed in `strain.self_spent` and does not move the dial. (b) A lane with 0 live slots whose
  window use already ran (a swap seat, diffusion / agent-gpu2) no longer clamps `exogenous_bind`.
  A lease still open on such a seat reads `no_reading`.
- **Replay on synthetic wants from real act history (10-16..10-27, 601 wants):**
  - R2 **FAIL**. Non-GPU wants in the 10-18..22 heat were funded at 100% (52 of 52), which is non-zero
    but not reduced (90% outside the heat, where the gaps are `sated` wants). With heat unpriced per
    act, nothing in C1 can reduce a non-GPU want in heat.
  - R5 **not evaluable**. No lane price exists for the 10-21..22 baseline, and nothing bids before it
    leases (Patch E).
  - R6 **PASS** (1.00x the lane at most, limit 1.2x). This holds by construction of the capacity
    rule, and the check confirms the code enforces it.
  - R7 **PASS**. Drift 0.100/day at most; 0 floor violations under a stressed lane price of 1.0; a
    synthetic zero-use family ends at 0.43 against a prior of 1.0. The only live zero-use family
    (`hub.retro_leisure`) has 1 evidenced act, so it stays at its prior.
- **Weights are mostly priors today.** Only `hub.outreach` has evidence: Juniper replied to 63% of
  43. Downstream use reaches 10 closed-window rows for no family. Most families are `no_link_path`.

## Review findings fixed (independent adversarial review, 2026-10-23)

The reviewer spot-checked 12 facts against live data and code. 8 were confirmed. The rest are corrected above: the share range, the timing of the grants drop, the cabinet daily means, and correlations not reproducible exactly. Mechanism findings:

- **The heat price would integrate without bound when the room, not Orion, holds the cabinet over the line, which recreates the veto.**
  - Fix: the heat level is now Orion's own contribution. Anti-windup adds an `exogenous_bind` clamp and data-sized ceilings (Organ 2).
- **`value` mixed nats with P(upheld).**
  - Fix: within-family percentile rank (`value_rank`) is now the one common unit. Equal family weighting is disclosed as the single authored knob (Organ 3).
- **Most want families have no graded acts.**
  - Fix: Patch A puts every producer into `grammar_acts`. Patch D gates each cutover on ≥30 graded closes.
- **The `claude` constraint in dollars repeated a refuted design.**
  - Fix: it now reads `ClaudeLimitObservationV1`, and `quota_budget.py` is deleted.
- **`ActCostV1` is a forbid model.**
  - Fix: `ActCostV2` is additive and ships consumer-first.
- **The retry storm was already fixed by `ShedBackoff`.**
  - Fix: Fact 3 and R5 are re-baselined, and lane prices build on it.
- **Most of the GPU freeze is the 34 °C reflex, not the strain veto.**
  - Fix: the summary is restated, R2 is limited to non-GPU wants, and missing question 6 now asks about chronic heat.
- **Heat sensitivity is not identifiable from the planned regressors.**
  - Fix: the fit now uses weather and lags, with randomized holdback pulses as the identification.
- **Fail-closed contradicted kill-means-kill.**
  - Fix: after Patch D, an unreachable economy defers every non-spinal want and raises an alert.
- **Prices could go stale between updates.**
  - Fix: reservations, with the new check R6.
- **`satisfaction` had no theory anchor.**
  - Fix: dropped. The dials now carry a metric-gate record.
- **`SettledBetV1` does not exist yet.**
  - Fix: Patch F's bets are marked blocked on #2827.
