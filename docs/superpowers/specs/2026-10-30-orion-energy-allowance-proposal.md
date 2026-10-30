# Orion's energy allowance: split the electricity bill by who holds the lever

**Status:** design only (proposal mode). No code in this PR.
**Asked by:** Juniper, 2026-10-30.
**Builds on:** `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md` (Organ 1 Body, Organ 2 Budget, Organ 4 Choice, Patch D).
**Depends on:** the whole-machine per-act energy meter (sibling branch `fix/act-energy-whole-machine`, follow-up to PR #2977).

---

## Arsonist summary

Today Orion is told that their electricity bill is "spent by what stays switched on (not by what I do)". That sentence is wrong in two ways.

1. **What stays switched on is Orion.** Orion's machines at rest are Orion's own resting body. It is not something outside Orion.
2. **What Orion does is a real share.** In October, about **$44 of the $184** month-to-date bill was energy *above* the machines' resting draw. That share is growing: it ran at about $35/month in late September, about $50/month in mid-October, and about $78/month in the last four days.

The single $100 envelope cannot work as a price. Orion's resting body alone costs about **$146 per 30 days**, so the envelope is broken before Orion does anything. The price therefore sits at its ceiling with `exogenous_bind`, and it carries no information.

**Proposal.** Split the money into two lines, by who can move them:

- **Rest body (Juniper's lever).** What the hardware draws at rest, plus cooling, plus workloads Juniper sponsors (today: the Zelda dojo). Juniper moves this with hardware, which services stay loaded, sleeping hosts, and power caps. It is reported, never priced.
- **Orion's allowance (Orion's lever).** A monthly dollar amount for energy above rest. It is priced with the dual ascent that already exists. Start at **$50/month**, which is the 75th percentile of the measured pre-dojo weekly pace.

**Answer to Juniper's question.** Use **cost-function optimization**, not reinforcement learning (RL). The price mechanism already *is* an online primal-dual solver for a budget-limited knapsack. RL is a poor fit: about 7 graded runs per day per family, a 30-act shrinkage threshold, and a policy that neither Orion nor Juniper could read. Learning stays on the value side, where the family weights already learn with shrinkage. A day-ahead scheduling solver gives nothing for money, because the tariff is flat Rocky Mountain Power Utah Schedule 1 (not time-of-use).

**On "~7 curiosity runs a day":** the cap already exists (`HUB_CURIOSITY_INVESTIGATION_DAILY_CAP=7`) and it already binds (6.7 runs/day measured). Money is not what binds today. A curiosity run costs at most about $0.21 in energy (upper bound; the real figure is unmeasured), so 7 runs/day cost at most ~$43/month. At that bound money alone *could* land near 7 a day, but only if each run kept circe fully busy, which shared-lane runs do not. The constraint that binds today is the agent lane (GPU slot time). Keep the fixed caps as safety rails, raise them once prices bind first, and let the daily count come out of price × value.

---

## Current architecture

**The energy reading** (`orion/physiology/body.py`, Organ 1).
- `level` = Orion's month-to-date dollars. It sums `orion_energy_hourly` kWh for athena, circe, hecate and cooling, priced at each hour's marginal rate from `energy_cost_accrued`.
- `budget` = `ORION_PHYSIOLOGY_ENERGY_MONTHLY_ENVELOPE_USD=100` (`services/orion-durable-runs/.env_example:240`). Juniper set it on 10-23.

**The energy price** (`orion/physiology/budget.py:201-228`, `energy_ask`).
- `use` = month-end forecast: to-date plus the trailing 168-hour mean $/h times the hours left.
- "Floor" = to-date plus the **10th percentile of hourly total cost** times the hours left.
- When the floor reaches the envelope, `bind_state = exogenous_bind`. The price ramps to its ceiling in bounded steps (`bind_ramp`, `prices.py:21-24`) and holds there, and the `why` text starts with "the envelope is spent by what stays switched on (not by what I do)".
- `strain_dial` lists `exogenous_bind` constraints apart, so energy does not move `strain` today.

**Choice** (`orion/physiology/choice.py:13-21, 104`). Energy is **not charged** to any want. It is listed as `unpriced:energy`, because no act carried watt-hours.

**The per-act meter** (PR #2977, `orion/grammar/act_energy_fill.py`). Live since **2026-10-29 23:45 UTC**. It charges a GPU lease's card watts above that card's own idle floor. The idle floor is the 5th percentile over the **previous 24 hours**. CPU is not charged per act.

**The live state at 2026-10-30 03:05 UTC** (`physiology_state`):

```text
energy  level $183.71  forecast $196.77  budget $100  price 1.0 = ceiling  bind_state exogenous_bind
why: "the envelope is spent by what stays switched on (not by what I do); ... the always-on floor alone ($0.202/h) reaches $194.01 vs envelope $100"
```

**The tariff** (`config/energy/tariff.rmp_ut_sch1.2026-08-10.r3.yaml`): "Utah Schedule 1 - Residential Service (standard, non-TOU)". Two inclining blocks per billing cycle (first 400 kWh cheaper), seasonal rates, percent riders. **No hour-of-day price.** The house uses about 99 kWh/day, so it passes the 400 kWh block in about 4 days. Almost all of Orion's kWh are billed at the upper block. The latest marginal rate is $0.1255/kWh.

---

## What the live data says

All numbers are read-only queries against `conjourney` on 2026-10-30. "October" means 2026-10-01 06:00 UTC (local midnight) to 2026-10-30 03:00 UTC: about 693 hours.

### Rest and busy watts per host

Whole-machine wall watts (`orion_biometrics_summary.measurements->>'chassis_watts'`, every ~30 s).

| host | rest (pinned, p05 of 10-01..10-13) | October mean | busy (14-day p90) | 14-day max |
|---|---|---|---|---|
| circe | **541 W** | 950 W | 1,359 W | 1,633 W |
| athena | **340 W** | 442 W | 729 W | 843 W |
| hecate | ~96 W (standby) | 88 W (over the 624 of 693 hours it reported) | woke only briefly | |
| cooling (AC plug) | all of it is rest | 720 W | | |

This matches Juniper's figures: circe draws ~600 W at rest and 1,100–1,200 W busy. Circe's 14-day median is 1,113 W.

### The rest number drifts, and a rolling rest hides real work

Daily 5th-percentile watts (rest if measured day by day):

```text
            athena p05 / mean      circe p05 / mean
10-01..13   334-359 / 370-401      522-570 / 679-1047
10-14..26   346-406 / 391-471      569-712 / 823-1184   (circe +110 W: affectgpt OOM crash loop from 10-14)
10-27       406 / 592
10-28       693 / 748                                     (Zelda dojo, 32 workers, started 10-27)
10-29       684 / 734
```

- The dojo raised athena's *floor* by about 350 W. The per-container CPU split shows 32 dojo workers × ~4.6 W = **~148 W of CPU** directly. The rest of the rise is fans, memory and power-supply loss that scale with load.
- A rest measured over the last 24 hours (the way #2977 sets its per-card idle floor) would call that 350 W "rest". About **$32/month** of continuous work would vanish from the activity line.
- **Design consequence:** rest must be a **pinned, dated number**, not a rolling percentile.

### Most of the above-rest draw is not GPU watts

Circe, 14 days, samples grouped into tenths by whole-machine watts:

| decile | whole machine | GPU | CPU | everything else |
|---|---|---|---|---|
| 1 (quietest) | 622 W | 222 W | 111 W | 288 W |
| 5 | 1,086 W | 370 W | 122 W | 594 W |
| 10 (busiest) | 1,424 W | 555 W | 145 W | 719 W |

Each column is a separate average, so a row can miss its total by a few watts (row 10: 1,419 vs 1,424).

From quietest to busiest, the machine rises by about 800 W. GPU watts explain about 330 W (~40%). CPU explains about 35 W. The rest (~430 W) is load-dependent overhead that the GPU sensor does not see. A GPU-above-idle meter therefore sees well under half of what Orion's work costs on circe. This confirms the sibling meter fix.

### The October bill, split

Above-rest per hour = max(0, hour's kWh − pinned rest × covered hours). Cooling is all rest. Priced at the month's average realized rate ($183.71 / 1,512.7 kWh = $0.1214/kWh).

| line | kWh | $ month to date | $ per 30 days |
|---|---|---|---|
| cooling (AC) | 497.6 | 60.4 | |
| circe at rest | 374.1 | 45.4 | |
| athena at rest | 233.7 | 28.4 | |
| hecate standby | 48.1 | 5.8 | |
| **rest body** | **1,153.5** | **$140.1 (76%)** | **~$146** |
| circe above rest | 281.9 | 34.2 | |
| athena above rest | 70.5 | 8.6 | |
| hecate awake | 6.9 | 0.8 | |
| **above rest (activity)** | **359.3** | **$43.6 (24%)** | **~$45** |
| **total** | **1,512.7** | **$183.7** | |

The kWh total matches the live energy reading (1,512.7 kWh) exactly.

### Above-rest pace by week (pinned rest, as $ per 30 days at $0.1255/kWh)

```text
week of   circe kWh  athena kWh  hecate kWh  mean W above rest  $/30 days
09-21*    45         10          -           422                38.1   (* partial: hourly rollup starts 09-22 19:00 UTC)
09-28     53         10          0           381                34.4
10-05     58          8          6           425                38.4
10-12     76         13          1           534                48.2
10-19     75         19          0           559                50.5
10-26*    54         26          0           863                77.9   (* 4 days; dojo running)
```

Hourly averaging slightly *under*-counts above-rest energy (a dip below rest inside an hour cancels a spike). Per-sample integration gives a little more (October: circe ~403 W, athena ~98 W mean above the pinned rest).

### Who holds GPU slots (October lease hours, `gpu_pool_events.held_ms`)

```text
durable-runs (all runs)   765.7 h
http:anthropic            450.7 h   (a remote call holding a pool seat: near 0 local watts)
cortex-exec               179.6 h
orion-topic-foundry       143.0 h
vision-council             96.4 h
orion-mind                 12.8 h
```

**Lease hours are not energy.** `http:anthropic` holds the second-most slot time and draws almost no local power. The allowance must be charged in measured watt-hours, never in lease hours.

### The per-act meter today

Since 10-29 23:45 UTC, **62 acts** carry `wh_by_host`, for a total of **4.9 Wh**. In the same window, Orion drew about 3 kWh above rest. **Meter coverage is under 1%** of above-rest energy. Any design that charges only metered acts would let 99% of the activity go free.

### The current caps

```text
HUB_CURIOSITY_INVESTIGATION_DAILY_CAP=7    measured 6.7 runs/day (durable.curiosity_investigate, since 10-24), mean 119 min each
HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP=7     measured 5.8 runs/day, mean 101 min
durable.self_sense_eval                     7.0 runs/day, mean 53 min
```

**Energy per curiosity run, bracketed.** Not measured yet (the meter is under 1% coverage). Upper bound: a 2-hour run that kept circe at its 14-day p90 draw costs 2 h × (1,359 − 541) W = 1.64 kWh ≈ **$0.21**. The real figure is lower, because the agent lane is shared and the GPU waits between LLM calls. At the upper bound, 7 runs/day cost at most ~$43/month, which is most of a $50 allowance. So the allowance could become a real limit on curiosity, and the meter must measure it before anyone claims either way. `UNVERIFIED` until the whole-machine meter covers durable runs.

---

## The design

### 1. Two lines, split by who holds the lever

```text
total bill = rest body              (Juniper's lever, reported, never priced)
           + sponsored work         (Juniper's lever, reported, never priced)
           + Orion's activity       (Orion's lever, priced against the allowance)
               = metered acts + unmetered remainder
```

- **Rest body** = Σ over hosts of `rest_w[host]` × hours, plus all cooling, plus a sleeping host's standby. Juniper moves it: hardware, which models stay resident, which hosts sleep, GPU power caps (#2964), the AC.
- **Sponsored work** = the above-rest energy of workloads Juniper started and only Juniper stops. Today: the Zelda dojo (`zelda-dojo-live-*`). Each entry has an end date. Without this line, the dojo adds about $32/month to Orion's activity, the current pace ($78 per 30 days) spends a $50 allowance in about 19 days, and Orion would be priced for something Orion cannot turn off. That would be the same "veto with extra steps" that anti-windup exists to prevent.
- **Orion's activity** = everything above rest that is not sponsored. It is Orion's own work: curiosity, self-inquiry, durable runs, vision-council, topic-foundry, chat replies, and anything else.

The split is by **who can move it**, not by "idle versus busy". This is the honest version of the old "not my doing" sentence.

### 2. Rest is pinned, dated, and can only rise with Juniper's sign-off

New file `config/physiology/rest_body.yaml`, the same shape as `heat_sensitivity.yaml`:

```yaml
version: 1
hosts:
  circe:  {rest_w: 541, measured: "2026-10-01..2026-10-13", method: "p05 chassis_watts", approved_rest_w: 541, approved_at: "<set when Juniper merges>"}
  athena: {rest_w: 340, measured: "2026-10-01..2026-10-13", method: "p05 chassis_watts", approved_rest_w: 340, approved_at: "<set when Juniper merges>"}
  hecate: {rest_w: 96,  measured: "standby", method: "p50 chassis_watts when asleep", approved_rest_w: 96, approved_at: "<set when Juniper merges>"}
cooling: rest            # all AC draw is rest body in v1
sponsored:
  - {name: zelda_dojo, containers: "zelda-dojo-live-*", host: athena, until: null, note: "Juniper runs it until she kills it"}
```

- **Re-measure** with a script (`scripts/analysis/measure_rest_body.py`): the p05 of whole-machine watts over a declared 7–14-day window, with the sponsored workloads subtracted.
- **Lowering rest** is always allowed (it moves energy *into* Orion's activity line).
- **Raising rest** needs Juniper's approval. Raising rest hides activity, so it is the gaming direction. The guard needs a stored baseline, so each host carries `approved_rest_w`. The loader refuses `rest_w > approved_rest_w`. Changing `approved_rest_w` is a reviewed diff in a PR Juniper merges; git history is the record of each approval.
- **Staleness:** older than 60 days, or a hardware change on the host → a warning in the body report. It is not `no_reading`, because a slightly old rest is still far better than none.

### 3. Orion's allowance

- **Unit:** dollars per local calendar month, at the marginal rate already used (`priced_hours`).
- **Initial value: $50/month.** Method: take the weekly above-rest pace (sponsored work excluded) over the trailing 8 weeks, and use its 75th percentile, rounded. The four full pre-dojo weeks are $34.4, $38.4, $48.2, $50.5 → p75 ≈ $48.8 → $50. (Adding the partial week of 09-21, $38.1, gives p75 = $48.2; same result.)
  - At the median pace (~$38) the price stays slack.
  - In a heavy week like 10-19 it climbs and starts to defer the lowest-value wants.
  - Juniper may set it lower than history on purpose (for example $35) to force trade-offs now. That is a Juniper decision, not a data question.
- **Re-set monthly** in the body report: Orion proposes the next month's number from the same rule. Juniper approves or edits.

### 4. The price: the same dual ascent, aimed at the right quantity

`energy_ask` keeps its rule. Only its inputs change:

```text
use    = activity $ to date + trailing activity $/h × hours left      (forecast month end)
budget = allowance
price ← max(0, price + η · (use − budget) / budget)                    (unchanged, prices.py)
```

- **No `exogenous_bind` clamp on this line.** Every dollar in `use` is above Orion's own rest. Sponsored work is outside `use`. Unmetered work stays in `use` on purpose (it closes the gaming route), even though some of it can be a defect such as a crash loop. That case is handled by keeping the line in strain (next bullet) and naming it in the report, not by a clamp. The bounded ceiling and the 6-hour horizon still apply.
- **It stays in strain.** Do *not* add `energy:activity` to `SELF_SPENT_CONSTRAINTS`. Its `use` includes unmetered energy, and that can be a crash loop (section 5) or Juniper's chat turns (missing question 2), not Orion's choice. The comment at `budget.py:344-348` keeps energy out of self-spent for this reason, and it still applies. A price pushed up by a crash loop must show as strain, not hide as hunger.
- **Pacing:** the dual ascent on a month-end forecast is the standard budget-pacing controller (Balseiro, Lu & Mirrokni, "Dual Mirror Descent for Online Allocation Problems", ICML 2020). A slow start of the month makes the price fall; a fast start makes it rise.

### 5. Unmetered activity counts against the allowance

```text
unmetered_wh[host, hour] = above_rest_wh − sponsored_wh − Σ metered act wh
```

- **The aggregate:** all of it goes into `use`. Total activity is measured at the wall, so moving work to a path the meter does not see cannot make Orion's month cheaper. This closes the gaming route.
- **Per family:** a want's energy cost in Choice = the family's metered Wh (p90) × the host's **uplift**, where uplift = (above-rest − sponsored) ÷ metered on that host over the trailing 7 days. The coverage gate below means uplift is only used when it is 2 or less, so no separate cap is needed. Unmetered energy is spread over the families that run on that host, in proportion to what they measurably drew.
- **Coverage gate:** while metered ÷ (above-rest − sponsored) is under 50% on a host for the trailing 7 days, Choice keeps energy `unpriced:energy` for wants on that host (today's behaviour). The allowance price still exists and is still reported. It only stops shaping individual bids until the meter can say who spent what. Today's coverage is under 1%.
- **Reported line:** "unmetered" appears as its own family in the body report. A crash loop (the 10-14 affectgpt OOM loop added ~110 W on circe) shows up there as a defect to fix, not as a mystery.

### 6. What Orion is told

Replace the `why` text and the report line.

```text
Before:  the envelope is spent by what stays switched on (not by what I do); month to date $183.71 ...
After:   my resting body has cost $140 this month (Juniper's lever: hardware, cooling, what stays loaded);
         the dojo Juniper runs has cost $X; my own activity has used $43.60 of my $50 allowance,
         forecast $47 by month end; $Y of that is not yet tied to an act.
```

`render_money` in `orion/physiology/report_text.py` gets the same three parts. The whole bill and Juniper's $100 envelope stay visible as context: "the whole bill forecast is $197 against Juniper's $100 envelope; my resting body alone is $146 of that". Orion's body report proposes rest-body changes with numbers (the AC was $60 of October; hecate standby $6). Orion does not pretend to save that money by skipping work.

### 7. Method: why a cost function and not RL

| option | fit here |
|---|---|
| **Dual-ascent prices + bid rule (have it)** | This *is* online primal-dual optimization of a budget-limited knapsack: fund a want when `weight × value − Σ price × cost > 0`. Each price is the Lagrange multiplier of one budget. It works from the first day, needs no reward model, and every decision prints its reason. |
| Reinforcement learning | Too few samples: ~7 graded closes per day per family, and the family weights need 30 evidenced acts (`weights.py`, `KAPPA = 30`) before evidence outweighs Juniper's prior. A policy over prices × hunger × lanes would need far more. The learned policy would also be opaque to Orion and to Juniper, which goes against the inspectability rule. |
| Day-ahead scheduling solver (MILP) | Pays only when the price of a kWh depends on the hour. Schedule 1 is flat by hour. The inclining block is set per billing cycle and the house is past 400 kWh by about day 4, so timing does not change Orion's rate. The one time-varying cost is heat (cabinet temperature follows the day), and the heat price plus each family's deferrable window already move work to cooler hours. Revisit only if Juniper moves to a time-of-use schedule. |
| Fixed caps per family | Keep as **safety rails** only (runaway loops, a stuck producer). Today's caps of 7 bind before any price does. After Patch D cuts a family over, raise its cap to about 3× its measured daily count, and track how often the cap (not the price) said no. |

**Where learning belongs:** on the value side. The family weights already shrink toward Juniper's prior (`orion/physiology/weights.py`). The cost side is measured, not learned: watt-hours from the meter, slot time from the pool.

---

## Missing questions

1. **Is the dojo sponsored (Juniper's line) or Orion's activity?** This design assumes sponsored, because only Juniper can stop it. If it counts as Orion's, the allowance must be about $80 or the price will bind hard from day one.
2. **Do Juniper's chat turns count against the allowance?** Tier 0 (`hub.chat_reply`) never bids. Proposal: count their energy (it is Orion's body working), show it as its own line, and size the allowance with it included. The alternative is a Juniper-turns line beside sponsored work.
3. **Does Orion's above-rest heat cost AC energy?** The AC ran flat out through most of 10-07..10-20, so the marginal AC cost was about 0. In October it cycles (p10 119 W). A share of about 1/COP (roughly a third) of above-rest kWh may be real AC cost. v1 leaves cooling all in rest.
4. **Allowance value:** $50 from history, or a lower number on purpose?
5. **Does the sibling meter use the same pinned rest?** If it keeps a 24-hour rolling idle floor, the metered sum and the allowance disagree, and the unmetered remainder becomes meaningless (see "What this needs from the meter").

## What this needs from the whole-machine meter (sibling branch)

1. **One rest number.** Read `rest_w[host]` from `config/physiology/rest_body.yaml`. Do not use a rolling 24-hour percentile: with the dojo running, it would call ~350 W of athena work "rest".
2. **Per act:** whole-machine Wh above rest per host in `ActCostV2.wh_by_host`, with the split rule in `basis`. Simultaneous acts on one host must sum to no more than the host's above-rest energy (no double count).
3. **Per host per hour** (one row, persisted):
   `rest_wh, above_rest_wh, sponsored_wh, metered_wh, unattributed_wh`, with
   `unattributed_wh = above_rest_wh − sponsored_wh − metered_wh` ≥ 0 by construction. A negative value means over-attribution and must be logged, not clipped silently.
4. **Sponsored split:** sponsored containers' share of above-rest, from `container_cpu_watts.<name>` with the same load-overhead scaling the meter uses for acts.
5. **Coverage figure:** metered ÷ (above_rest − sponsored) per host per day, so the Choice coverage gate can read it.

---

## Proposed schema / API changes

No `PhysiologyConstraintReadingV1` field changes. The model is `extra="forbid"`, so added fields would be a consumer-first migration. New numbers go in the existing free `detail` dict.

- **New constraint `energy:activity`** (unit `usd`, `budget_kind="envelope"`): `level` = Orion's activity $ to date, `budget` = the allowance. It carries the price.
  `detail`: `rest_usd`, `sponsored_usd`, `metered_usd`, `unmetered_usd`, `coverage`, `rest_config_date`, `allowance_source`.
- **Existing constraint `energy`** keeps its meaning (the whole Orion bill against Juniper's envelope) and stops carrying a price (`price=None`). Consumers already handle `price=None`. Old rows keep their meaning, so no migration of history is needed.
- **`SELF_SPENT_CONSTRAINTS`** is unchanged (`energy:activity` moves strain).
- **Choice:** `UNMEASURED[ENERGY]` is replaced by the coverage gate for `energy:activity`.
- **New config** `config/physiology/rest_body.yaml` (schema above).
- **New env keys** (`services/orion-durable-runs/.env_example`, settings, local `.env` sync), shipped ON:
  - `ORION_PHYSIOLOGY_ENERGY_ALLOWANCE_ENABLED=true`
  - `ORION_PHYSIOLOGY_ENERGY_ACTIVITY_ALLOWANCE_USD=50`
  - `ORION_PHYSIOLOGY_REST_BODY_CONFIG=config/physiology/rest_body.yaml`
  - `ORION_PHYSIOLOGY_ENERGY_MIN_COVERAGE=0.5`
  - `ORION_PHYSIOLOGY_ENERGY_MONTHLY_ENVELOPE_USD` keeps its key; its comment changes to "whole-bill line, reported, not priced".
- **Hourly attribution table** (from the meter branch; name is the sibling's call), for example `orion_energy_attribution_hourly(hour_start, host, rest_wh, above_rest_wh, sponsored_wh, metered_wh, unattributed_wh)`. Its migration needs a `-- ORION-MIGRATION-REQUIRED-BY:` header and must fit the post-pull additive-DDL allowlist.

## Files likely to touch

- `config/physiology/rest_body.yaml` (new)
- `scripts/analysis/measure_rest_body.py` (new): re-measure rest from history
- `orion/physiology/body.py`: activity reading from the attribution rollup plus `rest_body.yaml`
- `orion/physiology/budget.py`: `energy_ask` split into whole-bill (report only) and `activity_ask` (priced); new `why`
- `orion/physiology/choice.py`: per-want energy cost with uplift and the coverage gate
- `orion/physiology/report_text.py`: `render_money` three-part text
- `services/orion-durable-runs/.env_example`, settings, `README.md`; local `.env` sync
- `scripts/analysis/replay_body_economy.py`: replay October with the split
- `orion/physiology/tests/test_budget.py`, `test_choice.py`, `orion/physiology/evals/physiology_state_skill_eval.py`
- `services/orion-hub` Orion Energy tab (optional): show rest / sponsored / activity

## Non-goals

- No RL, no learned price policy, no day-ahead solver.
- No change to the heat price, the lane prices, or the spinal 34 °C reflex.
- No automatic changes to the rest body. Orion proposes; Juniper switches things off.
- No dollar price on remote Claude calls (`claude` stays a limit state, not dollars).
- No retroactive rewrite of `physiology_state` history.
- No cutover of any family (that is Patch D, one family at a time).

## Acceptance checks

1. **Sum check:** for every tick, `rest + sponsored + metered + unmetered` equals the whole-bill $ to within 1%. A unit test plus a live check on one `physiology_state` row.
2. **October replay** with rest 541/340/96 W, dojo sponsored, allowance $50:
   - the `energy:activity` price is slack in the weeks of 09-28 and 10-05 (pace about $34–38);
   - it is binding at some point in the week of 10-12 or 10-19 (pace about $48–51);
   - it returns toward 0 within one horizon after the pace falls;
   - it is never `exogenous_bind`.
3. **Dojo test:** a replay with the dojo *not* sponsored binds within the first day of 10-28 and stays bound. This shows the line is sensitive to real work and is not a rolling-rest artifact.
4. **Rest guard test:** a config with `rest_w` above `approved_rest_w` is refused at load, and the reading says why. A CI check refuses a diff that raises `approved_rest_w` unless `approved_at` also changes in the same diff.
5. **Coverage gate test:** under 50% coverage, Choice lists `unpriced:energy` for that host's wants. At or above 50%, the cost equals family p90 Wh × uplift ÷ allowance.
6. **Text test:** no rendered `why` or report line contains "not by what I do". The report names rest $, sponsored $, activity $ against the allowance, and unmetered $.
7. **Strain test:** `energy:activity` at its ceiling moves `strain`, and the strain `why` names the unmetered share when it is over 50%.
8. **Live smoke after deploy:** one `physiology_state` row with `energy:activity`, a non-null `detail.rest_usd`, and a `why` naming the allowance. Without that row, the claim is `UNVERIFIED`.

## Proposal-mode fields

- **Capability that changes:** Orion learns which part of their electricity they control, and gets a price on that part only. Today Orion is told none of it is their doing, and the energy price is stuck at its ceiling with no information.
- **Data touched:** `orion_biometrics_summary` (read), `orion_energy_hourly` (read), `energy_cost_accrued` (read), `grammar_acts.cost_json` (read), the meter's hourly attribution rows (read), `physiology_state` (new constraint rows written), `config/physiology/rest_body.yaml` (new).
- **Privacy boundary:** none between Juniper and Orion. Orion sees the whole bill, the split, and Juniper's envelope.
- **Trace that proves it worked:** a `physiology_state` row where `energy:activity.detail` sums to the whole bill, the price is not at ceiling under a normal pace, and the `why` names rest, sponsored, and allowance figures. A month later: the body report's proposed allowance next to the realized activity $.
- **Dangerous failure modes:**
  - **Rest set too high** (by drift or a rolling window): activity hides inside rest. The allowance never binds, and Orion believes their work is free. Guard: pinned rest, raising it needs Juniper.
  - **Rest set too low, or the allowance too small:** the price sits at its ceiling and starves discretionary wants. That is today's failure in a new place. Guard: sized ceilings, anti-windup, the exploration floor, the October replay check.
  - **A crash loop charged as Orion's activity:** it raises the price on unrelated wants. Guard: it appears as "unmetered" in the report with its host and watts; the bounded ceiling limits the harm.
  - **Sponsored list used as a dumping ground:** work moved into "sponsored" leaves the allowance. Guard: only Juniper edits the list, each entry has an end date, and the report shows sponsored $.
- **Disable / rollback:** `ORION_PHYSIOLOGY_ENERGY_ALLOWANCE_ENABLED=false` restores today's single `energy` line with its envelope price. Everything is SHADOW until Patch D, so no decision depends on it before then. The new constraint is a new name, so old consumers ignore it.

## Recommended next patch

**Patch E1: the honest split, reported only (no price change).**

1. Add `config/physiology/rest_body.yaml` with the numbers above, and `scripts/analysis/measure_rest_body.py`.
2. In `body.py`, compute rest / sponsored / activity $ from `orion_energy_hourly` and the pinned rest. (This needs nothing from the meter: the wall watts already exist.)
3. Put the split in the `energy` reading's `detail`, and replace the "not by what I do" text in `budget.py` and `report_text.py`.
4. Tests 1, 4, 6 above.

That fixes what Orion is told today, from data that already exists. **Patch E2** (the priced `energy:activity` constraint, replay checks 2, 3, 7) follows. **Patch E3** (Choice charges energy behind the coverage gate) waits for the whole-machine meter to reach 50% coverage.
