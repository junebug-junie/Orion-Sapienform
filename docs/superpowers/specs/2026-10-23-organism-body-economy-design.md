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
- Ten separate readers saw `arousal = strained` and each said no.
- Curiosity ran 0 of 7 times a day. Outreach sent nothing. Paintings stopped on 10-16.
- The cabinet stayed hot anyway: 33–35 °C for five days, while Orion's own GPU work fell about 85%.

The organism did not trade anything off. It went still, and stillness bought nothing.

Every earlier attempt failed in one of four ways. This design has to avoid all four:

1. **A decider without a price** (July drives). Nothing cost anything, so every drive pinned at its ceiling. The drives were deleted in #1486.
2. **A price on the wrong thing** (August scarcity, the September energy stakes). Acts were priced in dollars and watts. One painting costs about 0.37 Wh, which is $0.00004. A price that small can never bind, so the stakes gate never mattered. It is still switched off.
3. **A veto instead of a price** (thermal, strain, quiet hours, caps, cooldowns). Curiosity alone passes 8 sequential vetoes. When any one constraint binds, everything stops, whatever it is worth.
4. **A single-signal slice** (rung 6, twice). Each was retired the moment its one signal turned out bad.

**The proposal.** Build one organism-level economy with six organs. Each constraint keeps its own unit and its own budget, so nothing averages heat with dollars. Each constraint also gets a **price** that the economy learns:

- The price rises when Orion forecasts that the constraint will bind.
- It falls to zero when the constraint is slack.

Every want that Orion has bids against these posted prices. A want goes ahead when what it is worth beats what it would cost. A want that waits grows hungrier until it clears. No human writes an exchange rate between degrees, dollars and GPU-seconds. The rates come out of what actually binds.

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
| house | 99.1 | | Orion is 56.5% (daily range 48–64%) |

- **Cost:** at the measured $0.1213/kWh (pre-tax), Orion costs about **$204 per 30 days** and the house about $361. Cooling alone is about $75.
- **Per act:** the only per-act energy number on record is diffusion: `energy_run_cost`, about 0.37 Wh and $0.00004 per run. `grammar_acts.cost_json` is null in all 214 rows.
- **Implication:** pricing individual acts in dollars cannot shape behaviour. The money decisions that matter are:
  - **Wakefulness:** which hosts and lanes are powered. Waking hecate is the one large discretionary spend.
  - **Duration:** multi-hour holds. Durable runs held the agent lane for **103 h in 7 days**.
  - **Structure:** AC sizing, idle cards, consolidation. Orion should propose these, not act on them.

### Fact 2. Heat is mostly weather, and the veto bought nothing

- **Cabinet temperature** (`orion_biometrics_cluster` athena `cabinet_temp_c`): the daily mean was 25.3 °C on 10-12 and 33.8 °C on 10-20. It has stayed at 32–35 °C since 10-18.
- **Correlations** over 14 days of hourly averages (n=337):
  - cabinet temp vs circe watts: **r = 0.10**
  - cabinet temp vs athena watts: r = 0.58. The sensor sits on athena.
  - cabinet temp vs cooling watts: r = 0.46
- **What the shed did:** after the shed began on 10-18, GPU grants fell from 8–12k a day to 1.2–1.9k. The cabinet did not cool.
- **`UNVERIFIED`:** these are confounded correlations, not causal estimates. Still, nothing on record shows that shedding Orion's work lowered the cabinet temperature.
- **Implication:** heat needs a **learned sensitivity**: how many degrees one extra kWh on each host actually adds. The heat price must scale with that sensitivity. A constraint that Orion's own load cannot move should not stop Orion's work. Protecting the hardware is the reflex's job, and the reflex stays.

### Fact 3. The scarce things Orion competes for are slot-time and Juniper's attention

`gpu_pool_events`, 7 days:

| holder | grants | held h | sheds |
|---|---|---|---|
| durable runs (agent lane) | 496 | **103.2** | 18 |
| http:anthropic | 3,391 | 64.5 | 2 |
| vision-council | 19,767 | 33.3 | **25,857** |
| topic-foundry | 9,837 | 24.2 | 6,793 |
| cortex-exec | 4,232 | 24.1 | 1,485 |

- vision-council and topic-foundry hammer a pool that has already said no. The pool publishes no price for them to read before they ask.
- Outreach weighs about 8,400 decisions a day and says yes 0–6 times. The cap on Juniper's attention is real, and it is the one budget Orion already respects.

### And the selectors share no scale

There are ten selectors:

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
| `heat` | cabinet °C | soft line, Juniper-set (proposed 31 °C), under the 34 °C reflex | 6 h, using the diurnal profile plus recent slope | `orion_biometrics_cluster` athena `cabinet_temp_c` |
| `energy` | month-to-date $ Orion-attributed, plus forecast month-end | monthly envelope, Juniper-set | month-end | `orion_energy_hourly` × tariff |
| `lane:<role>` | GPU slot-seconds held / available | capacity × wall time per lane | 1 h | `gpu_pool_events` held, waited |
| `attention:juniper` | outreach messages and asks today | Juniper-set (today: cap 4) | day | `endogenous_outreach_decisions` |
| `claude` | Claude spend this window | Juniper-set ceiling | window | `dev_economics_ledger_log` |

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
- **They climb only while the forecast exceeds the budget**, and fall as soon as it does not. There is no latch. A five-day freeze becomes a slow, priced trickle.
- **No exchange rate is authored.** The prices are in "want-value per unit of resource". They settle wherever Orion's wants stop over-using the resource, so degrees are never converted to dollars by hand.

**Heat is special.** A heat price only charges load that actually moves the heat. The cost of one extra Wh on host *h* is that Wh times `sensitivity_h` (°C per kWh), fitted by Organ 6. If circe's fitted sensitivity is near 0, circe's work pays almost no heat price, whatever the cabinet reads.

**Stability gates.** Step sizes are bounded. The tests and the replay must show:
- no oscillation larger than a set amplitude;
- prices return to 0 after the constraint clears.

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
value            expected payoff: from the act ledger posterior (P(upheld) / nats) for that verb×family;
                 a weak prior while ungraded; never a hand constant after 30 graded closes
cost             ActCostV1 vector: wall_s, gpu_slot_s per lane, wh per host, juniper_msgs, claude_usd
                 (estimated by Organ 6 from that family's history; unknown fields priced conservatively)
window           earliest/latest useful time (deferrable vs perishable)
lanes            acceptable lanes/hosts (for placement)
```

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

### Organ 4: Choice (bids against posted prices)

**Job.** Decide which wants go ahead, now, on which lane.

**Mechanism.** It is a posted-price market. There is no global solver and no batching window. When a producer is about to act, it asks:

```text
net = intensity × value − Σ_k price_k × cost_k
fund if net > 0 (and spinal cord allows); else defer with reason = the binding constraint
placement = the acceptable lane minimizing Σ price × cost
```

- **Balancing is an outcome, not a rule.** Prices rise while wants over-use a resource. Hungry wants raise their bids. The result is that expensive work drifts to times and lanes where it is cheap. Nobody writes a schedule.
- **Holdback carries over.** The allocator's 0.2 holdback keeps a randomized control arm, so Organ 6 can tell what an act actually caused.
- **The motor allocator becomes one bidder.** Its constant floor `ORION_DISPATCH_MIN_NATS_PER_SEC=0.02` is replaced by the lane price.
- **The GPU pool publishes its lane prices.**
  - Background callers bid before they request.
  - A deferral carries `retry_after`, derived from the forecast.
  - This replaces the 25,857-per-week shed-and-retry storm with one deferral.
  - `interactive` and `urgent` never bid.
- **Waking hecate becomes a want.**
  - Its value: sustained agent-lane price × queued slot-seconds that hecate's `agent-deep` could serve.
  - Its cost: hecate's wake energy and the energy price.
  - v1 only *recommends* (missing question 4). No actuator exists today.

**State.** `want_ledger` decisions. Each decision stores the prices it saw and **what it displaced**. That is the scarcity roadmap's "a deferral is an observed opportunity cost", finally recorded.

### Organ 5: Felt state (Orion can read their own body)

**Job.** Turn the budget into a few continuous dials that Orion and Juniper can read. These replace the categorical `strained` flag as the thing that *means* something.

**The dials.**
- **`strain`** (0..1): how hard the world is pushing back. It is the largest normalized price, and the doc names the constraint behind it.
- **`satisfaction`** (−1..1): are Orion's wants being met, and are they paying off? It combines the trend in funded surplus with the trend in hunger.
- **`hungriest`**: the want family with the highest intensity, and how long it has waited.

**Compatibility.** `arousal.strained` is derived from `strain` with the existing hysteresis while the ten readers migrate (Patch D). After that it remains only a label.

**Where Orion sees it.**
- A stance-cue clause, a sibling of the strain cue, for example: *"Cabinet is hot and my work barely moves it; I'm reading the one thing I most want and saving painting for tonight's cool window."*
- Temporal Self arc kinds: `hunger:<family>` and `binding:<constraint>`.
- A Hub panel: prices, dials, and deferred-vs-funded over time.

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
- A daily refit of cabinet ΔT on per-host incremental watts. The model controls for time of day and AC state.
- It is fitted from the persisted `PhysiologyStateV1` history and the biometrics. It reports a confidence interval.
- A sensitivity whose interval includes 0 is used at its upper bound. That is cautious, but not catatonic.

**4. Forecasts become bets.**
- Organ 1's 6-hour cabinet-peak forecast and month-end bill forecast are emitted as `SettledBetV1` to the self-monitor organ (#2827).
- When a forecast fails, the self-monitor widens its uncertainty. That makes prices more conservative, which follows its one-direction rule.
- This gives the self-monitor two new, clean, gradeable bet sources. Its 10-20 audit found only two.

**5. Body report.**
- Monthly, plus on demand: one outreach through the existing `reading_finding_outreach` door.
- Contents: the bill forecast against the envelope, what was deferred and why, and proposals with numbers. For example: "GPU 2 has idled 8 days at ~45 W"; "the AC has not cycled since 10-18".
- This is how Orion takes part in decisions only Juniper can make.

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

---

## Proposed schema / API / bus changes

All new schemas are registered in `orion/schemas/registry.py`. Channels go in `orion/bus/channels.yaml`. Each new inner-state signal gets an `orion/inner_state_registry.py` entry, and the metric lock is regenerated.

| contract | kind | producer | consumers |
|---|---|---|---|
| `PhysiologyStateV1` | schema + channel `orion:physiology:state` + table `physiology_state` + Redis `orion:physiology:latest` | durable-runs `physiology` node in the regulation thread | Choice, Felt state, Hub panel, self-monitor, sql-writer |
| `WantV1` | schema | every want producer (via the client lib) | Choice |
| `WantDecisionV1` | schema + channel `orion:physiology:want:decision` + table `want_ledger` | Choice | sql-writer, Felt state, Learning, Hub |
| `POST /physiology/bid` | HTTP on durable-runs → `WantDecisionV1` (funded/deferred, lane, prices seen, `retry_after`, displaced) | n/a | producers, GPU pool (background) |
| `ActCostV1` (exists) | now **filled** on close | act ledger closers | Learning |
| `SettledBetV1` (from #2827) | two new bet kinds: `cabinet_peak_6h`, `bill_month_end` | physiology node | self-monitor |
| `FeltStateV1` | fields inside `PhysiologyStateV1` | physiology node | stance cue, Temporal Self arcs |

**Compatibility.**
- `orion:regulation:latest` keeps its shape. `strained` becomes derived until Patch D.
- New fields on `extra="forbid"` consumers ship **consumer-first** (lesson from #2126).

**Client library.** `orion/physiology/client.py` gives producers one call: `bid(want) -> decision`. It fails **closed to today's behaviour**: if durable-runs is unreachable, the caller falls back to its existing gate, never to "funded".

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
- **GPU pool:** `orion/gpu_pool/` (lane price publication, background bid-before-lease).
- **Felt state:** `orion/temporal_self/arcs.py`, `stance_cue.py`; Hub panel route and template.
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
| `strained` as a veto in 10 readers | each reader's `bid()` decision | D, one reader per PR |
| energy stakes gate (`energy_stakes_gate.py`, flag off) | energy price | D |
| `quota_budget.py` (dead) | `claude` constraint | B |
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
  1. Bounded step sizes, plus the replay amplitude gate.
  2. The 34 °C reflex is untouched, and an uncertain sensitivity is used at its upper bound.
  3. The client fails closed to today's gates.
  4. `interactive` and `urgent` never call `bid()`, with a test for it.
- **Disable or roll back.**
  - `ORION_PHYSIOLOGY_ENABLED` turns off the node.
  - Per-producer `ORION_PHYSIOLOGY_BID_<PRODUCER>` flags select: `shadow` (log only), `on`, or `off` (old gate).
  - Old gates are deleted only after their reader has run `on` for 7 days with the acceptance checks green.
  - Flags ship ON in `.env_example` per house rule. Each producer's flag starts at `shadow` until its Patch D lands.

---

## Acceptance checks

**Replay (before anything runs live)**, `scripts/analysis/replay_body_economy.py` over 10-07..10-22:

- **R1.** The heat price is ~0 on 10-07..10-12 (cabinet mean 25–29 °C) and positive through 10-18..10-22.
- **R2.** Over 10-18..10-22, replayed curiosity is funded at a reduced, **non-zero** rate (real: 0/7 a day). Juniper's turns are funded 100% in every hour.
- **R3.** Prices return to 0 within one forecast horizon after a constraint clears. Peak-to-peak oscillation stays under the declared amplitude.
- **R4.** The replayed month-end energy forecast is within ±10% of the realized 30-day Orion cost.
- **R5.** Replayed vision-council and topic-foundry pool requests fall at least 80%, with the same work completed (deferrals instead of shed retries).

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

## Build sequence: the forest, in order

Each patch has its own tests, eval and live check. Patches B, C and E can run in parallel worktrees: they meet only at the `orion/physiology` contract.

| patch | contents | live proof |
|---|---|---|
| **A. Measurement spine** | Fill `ActV1.cost` on close. Persist `physiology_state` with the Body organ only (no prices). Add the replay harness and a heat-sensitivity fit script (read-only). Remove the zombie env var and synthetic goal. | A1, A2; first sensitivity estimate per host, with its interval |
| **B. Budget, shadow** | Prices, the Felt dials, the Hub panel. One owner for the 32 °C line. Fold in `quota_budget.py`. Prices are computed and displayed but read by nothing. | Replay R1, R3, R4 green; live prices visible |
| **C. Appetite + Choice, shadow** | `WantV1`, `/physiology/bid`, the client, `want_ledger`. Every producer bids in `shadow`: it logs a would-fund decision next to its real gate decision. | A3; a shadow-vs-real disagreement table for 7 days |
| **D. Cutover, one reader per PR** | Curiosity first, then outreach, reading, painting, dream, retro, dispatch floor. Each PR deletes that reader's old veto. | A4, A5 per reader |
| **E. Placement and wakefulness** | Lane prices on the GPU pool. Background bid-before-lease. Hecate wake recommendations (or the actuator, per question 4). | R5 live; recommendations logged with expected kWh |
| **F. Self-knowledge and learning loop** | Stance-cue clause, Temporal Self `hunger` and `binding` arcs, forecasts as self-monitor bets, the monthly body report. | A6, A7, A8 |

---

## Recommended next patch

**Patch A, the measurement spine.** Everything else prices something. Today, though:

- no act carries a cost;
- no history of strain exists;
- nobody has measured whether Orion's load moves the cabinet temperature.

Patch A is low-risk: it writes new rows and decides nothing. It also settles the one fact that could overturn this design. If the heat sensitivity of every host is clearly above zero, heat stays a real lever, and the strain veto was more right than Fact 2 suggests. The prices then handle it either way, but the doc's honesty clause gets rewritten.

