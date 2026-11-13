# Attention patch 3: the want market starts deciding

Date: 2026-11-07. Status: approved 2026-11-10 (all defaults; see "Decided" below). Docs only, no code. This changes what Orion chooses to spend
effort on, so proposal mode comes first (CLAUDE.md 0A). Juniper chose this patch on 2026-11-07.

Arc: `docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md` (PR #3089).
Patch 1: PR #3101. Patch 2: PR #3122. Ladder retirement: PR #3121's spec.

Every number below was read live on 2026-11-07 about 05:50Z. The window is 14 days
(2026-10-24 .. 11-07). The want ledger only starts on 10-27, so ledger numbers cover 11 days.
Read-only throughout: Postgres `conjourney` and CSV extracts in a scratch directory. The replay
method for each table is in "How the replays were run" at the end.

## In one paragraph

The market should choose between wants when a lane is contested. Today it does not, but the reason
is not the one patch 1 named. Patch 1 said "capacity is first come, first served in the reservation
book". The live data says the reservation book almost never decides anything: of 2,041 capacity
deferrals in 14 days, only 80 named any reservation that blocked them, and 73 of those 80 were the
same family blocking itself. The real contest happens one step later, in **the GPU pool's queue**,
which serves work by priority class and then by arrival time. The biggest user of that queue is
**reading**. Reading runs as durable runs that never bid. It holds 1,099 of the 1,442 background
agent-lane holds (214 of 301 hold-hours), and its own acts record no lane cost at all. So this patch
does three things, in this order:
1. it gives reading its real, already-measured cost;
2. it makes reading bid where curiosity already bids;
3. it lets the pool's background queue serve the higher-value want first.

A fourth piece, the consent gate, is a separate small contract whose first user is dream compaction.

## Decided questions (Juniper, 2026-11-04 and 11-05; not reopened here)

1. Binding order: Prime, curiosity, reading, camera asks, outreach, journals, painting/dreams, play.
2. A binding refusal means "not now". The want stays open.
3. Juniper's turns always fund. They never bid (Tier 0).
4. A family binds only after a 14-day replay shows it is not starved (5% floor), then a live week
   with an off switch.
5. Dream compaction apply stays off until the market's consent gate exists.
6. Gates that protect Juniper and the body stay where they are: the heat reflex, quiet hours and
   outreach caps.
7. Self-monitor evidence (source (d)) stays off until a world-graded success signal exists.

## Arsonist summary

- **"No value can win a contested lane" is true, but the book is the wrong place to fix it.**
  - 2,041 capacity deferrals in 14 days. 1,961 (96%) named no reservation: the lane was full of
    work that had already started.
  - Of the 80 that named a holder, 73 were blocked by their own family.
  - Replay A: letting a higher-net want displace an unopened reservation changes the winner
    **2 times in 2,041**. Both holders were durable runs that had already opened, so under the
    "unopened only" rule the real count is **0**.
- **The two big numbers from patch 1 were bursts, not steady contention.**
  - Prime's 910 deferrals: 593 on 10-31 alone, and 0 since 11-04. The mindloop refactor (11-04/05)
    now releases Prime's reservation after each thought. Before that, Prime's own finished thought
    kept blocking the next one: on 11-03 a reservation outlived its act by 35 minutes.
  - Outreach's 856 deferrals: 809 on 10-29 alone.
- **Where the contest really is.** The pool's background agent queue. Background holds wait
  (1,542 of 2,707 background agent grants waited more than a minute; p90 1.3 h). System-priority
  work does not wait (p90 1.4 s). Inside the background queue, the order is arrival time.
- **The free rider is reading, and it is not free.**
  - Reading's 807 acts in 14 days record `gpu_slot_s_by_lane = null`, and families.yaml says "lanes:
    [] ... non-GPU as far as the ledger can see".
  - The pool measured it: reading's turns held **131.6 slot-hours** of agent-lane time (median
    1,057 slot-seconds per read, p90 2,586). Its durable-run holds are 1,099 of the 1,442 background
    agent holds.
  - The ledger misses it because the act's `correlation_id` is the seed's `request_id`, while the
    leases carry the seed's `trace_id` and `stage2_trace_id`.
  - This is a join bug, not a missing measurement.
- **Charging reading at today's posted prices changes almost nothing.** The agent lane runs at 82%
  average use, but its price averages 0.045 and is above 0.05 only 8% of the time. The price only
  rises when use goes over the budget, and queued demand is deliberately "hunger, not priced".
  Replay C: reading's replayed charge has a median of 0.000. Selection will come from **queue order
  by value**, not from the price.
- **The 5% floor counts the wrong thing.** It counts funded *bids*. Retro leisure bids on a
  5-minute timer, so it is 73% of Tier 2 funded wants this week, from 5 sessions. Counted by acts
  that started, reading is 51% of Tier 2 effort, and retro is 0.6%. A floor over bids measures who
  bids most often, not who got effort.
- **Self-grades are 87% of all closes** (2,566 of 2,940 in 14 days; resolver 311, expired 63). Patch
  1's saturation finding still holds.

## Current architecture (live numbers)

### What binds today

| Family | Bid site | Rule on capacity deferral | 14-day decisions |
|---|---|---|---|
| `prime` | `orion/mindloop/runtime.py` tick | Anything but `funded` waits. (2026-11-13: now `queued_at_pool` like the others, `goes_ahead`.) | 331 funded, 910 capacity (0 since 11-04) |
| `durable.curiosity_investigate` | `services/orion-durable-runs/app/run_bids.py` `RunBidder.bind` | `capacity_reserved` with net > 0 is `queued_at_pool` (admitted) | 17 funded, 32 capacity (all admitted) |
| `dream.cycle` | `services/orion-dream/app/physiology_bind.py` | same rule as curiosity | 14 funded |
| `self_regulate.shed` | `services/orion-durable-runs/app/cooling_act.py` | only `funded` sheds | 0 wants in 14 days (no warming event qualified) |

Every other family bids in shadow. The shared rule is `orion/physiology/binding.py` `bind_verdict`.

### Where capacity is decided

1. **The market's book** (`orion/physiology/choice.py` `ReservationBook`, `decide`). A bid is
   deferred `capacity_reserved` when the lane's trailing-hour use, plus the open reservations, plus
   its own cost, exceed the lane's line. In 14 days:

   | Family | Capacity deferrals | Named a holder | Median net |
   |---|---|---|---|
   | prime | 910 | 34 (all Prime) | 0.59 |
   | hub.outreach | 856 | 29 (all outreach) | 0.34 |
   | thought.painting | 72 | 4 | 0.52 |
   | durable.self_sense_eval | 54 | 1 | 0.51-0.55 |
   | durable.curiosity_investigate | 50 | 3 | 0.65-0.68 |
   | durable.self_inquiry | 51 | 0 | 0.54-0.73 |
   | thought.dream_hop | 35 | 9 | 0.56 |
   | durable.pr_field_note | 14 | 0 | 0.58-0.63 |

   Since 11-04, the agent-lane families are deferred on most bids. Outreach was deferred on 18 of
   18 bids, self-inquiry on 18 of 21, self-sense on 17 of 22, and curiosity on 16 of 21. None of
   these deferrals named a holder.
2. **The GPU pool's queue** (`services/orion-gpu-pool/app/runtime.py` `_position`). Leases are
   served by `(priority rank, created_at)`. The priority order is urgent, interactive, perception,
   starved, system, background (`config/gpu_pool.yaml`). Inside one class, the first to arrive is
   served first. The pool knows nothing about value.

### Who uses the agent lanes (gpu_pool_leases, 14 days, slot-hours)

| Lane | Bidding durable runs | Reading (traced by seed) | Harness at system priority, not reading | cortex-exec, system |
|---|---|---|---|---|
| agent | 121.5 | 56.8 | 21.2 | 9.2 |
| agent-gpu2 | 154.6 | 74.7 | 86.7 | 14.3 |
| chat | 26.2 | 17.8 | 3.6 | 0.8 |

Background agent-class **holds** (the seat a durable run waits for):

| Holder | Holds | Hold-hours |
|---|---|---|
| reading runs (`durable-runs:reading-*`, never bid) | 1,099 | 213.6 |
| runs that bid (curiosity, self-inquiry, self-sense, PR field notes) | 301 | 87.3 |
| upkeep with no bid (compactors, curiosity compaction, orion-day) | 42 | 2.7 |

Note: the "harness, system, not reading" rows are turns on the Anthropic-compatible gateway route
that are not traced to a reading seed. This design does not trace them further.

### Reading today

- Hub's loop (`services/orion-hub/scripts/world_pulse_read_pipeline.py` `_tick_unlocked`) claims up to
  `HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT=3` seeds every 300 s. Each read is a durable run
  (`orion/world_pulse_read/durable.py`, run id `reading-<uuid5(seed:stage:attempt)>`): stage 1, then
  stage 2. There were 903 stage-1 and 818 stage-2 durable turns (attempts included) in 14 days.
- Reads per day: 15 to 130, with a median of about 50. There were 807 read acts in 14 days.
- Reading's runs pass through durable-runs admission, the same seam where curiosity binds. But
  `services/orion-durable-runs/app/run_acts.py` `run_act_family` returns None for the reading
  workflow, so reading never bids there.
- The only reading bids are patch 2's *after-the-read* event wants (`orion-hub.reading_events`):
  142 decisions, all funded.
- Hub reclaims a claim older than `HUB_WORLD_PULSE_READ_TIMEOUT_SEC=3500`.

### Charges today

| Constraint | Average price, 14 days | Price above 0.05 | Average use / budget |
|---|---|---|---|
| lane:agent (1 slot) | 0.045 | 8% of ticks | 0.82 |
| lane:agent-gpu2 (2 slots) | 0.003 | 0.9% | 0.60 |
| lane:chat (1 slot) | 0.023 | 4% | 0.24 |
| attention:juniper (messages) | 0.475 | 50% | 0.55 |

- **Reading** pays nothing because it declares no lanes and its acts carry no lane cost (the join bug
  above).
- **Camera asks** (`vision.ask`) pay nothing because they declare no lanes and no
  `juniper_msgs`. An ask is a card Juniper answers when she likes. There were 211 camera asks in 14
  days (8 to 45 a day; about 8 a day since 10-31, the sql-writer batch), and about 86% were answered.

### The exploration floor today

The floor is in `choice.py`: a Tier 2 family under 5% of Tier 2's funded wants over 7 days is
funded when only the price said defer. Live 7-day shares, three ways:

| Family | Share of funded bids | Share of funded family-hours | Share of acts started |
|---|---|---|---|
| hub.retro_leisure | 72.9% | 34.9% | 0.6% (5 acts) |
| hub.world_pulse_read | 9.4% | 11.8% | 50.9% (461 acts) |
| thought.painting | 4.6% | 18.2% | 8.7% |
| hub.outreach | 0.7% | 1.1% | 13.3% |
| durable.curiosity_investigate | 1.1% | 3.5% | 5.4% |
| durable.self_inquiry | 0.8% | 3.2% | 5.2% |
| durable.self_sense_eval | 0.9% | 3.8% | 5.4% |

### Risk, consent and reversibility checks today (inventory)

| Check | Where it lives | Protects | What it does |
|---|---|---|---|
| Heat reflex | `services/orion-gpu-pool/app/runtime.py` (thermal shed) | body | sheds background grants when the cabinet is hot |
| Cooling act eligibility | `orion/autonomy/self_shed.py`, `services/orion-durable-runs/app/cooling_act.py` | body | gates Orion's own shed request |
| Cabinet-hot refusals | `services/orion-hub/scripts/retro_leisure.py`, `services/orion-thought/app/chain.py`, `services/orion-topic-foundry/app/services/llm_client.py` | body | per-doer refusal |
| Arousal "strained" | `orion/regulation/arousal_view.py`, read by Hub outreach, retro and curiosity, `services/orion-dream/app/cycle.py`, the thought visual chain and `orion/reverie/baseline.py` | body | defers discretionary work |
| Juniper-first yield | `orion/mindloop/runtime.py` (`turns.active()`); retro `human_recent` | Juniper | Orion's own thinking waits for her turn |
| Quiet hours, daily cap, cooldown | `services/orion-hub/scripts/endogenous_outreach.py`; `services/orion-notify/app/policy.py` | Juniper | message timing and volume |
| Juniper attention price | `attention:juniper` in Choice | Juniper | prices outreach messages against her daily cap |
| Session caps | retro (daily cap, cooldown), AI Town controller (7,200 s/day, 1,800 s session) | body and Juniper | volume |
| Camera ask batch | sql-writer ask producer (about 8 a day) | Juniper | volume |
| Self-monitor act gate | `orion/self_monitor/gate.py` `act_gate`, used by mindloop, Hub retro, embodiment, actions | Orion | refuses retries of a failing faculty |
| Dream compaction hot gate | `services/orion-dream/app/compaction_applier.py` (flag, consent callable, injected store) | memory | apply is off, and consent always says no |

All of these except the last answer "is now a good time" or "is this too much". None of them asks
"is this act allowed to change something that cannot be put back, and who agreed?". That was the
retired policy rung's job. Today the only thing that needs it is dream compaction apply.

## Proposal

### 1. Capacity by value: the queue serves the higher-value want first

**What it does.** When two of Orion's discretionary wants wait for the same kind of GPU seat, the
pool serves the one the market valued higher, instead of the one that arrived first. Value only
reorders work **inside one priority class**. Juniper's turns (interactive), perception and system
work keep their places.

**Where.**
- The binding rule already sends a capacity deferral with positive net to the pool's queue
  (`queued_at_pool`). So the pool's queue is where the choice is made.
- A binding bid's lease request carries the decision's `want_id` and `net`.
- In `services/orion-gpu-pool/app/runtime.py`, the background queue sorts by
  `net + aging × hours waited`, with arrival time as the tie-break. `_position` and the grant pick use
  the same key.

**What "displace" means.**
- A displaced hold is not cancelled and not refused. It stays queued ("not now"), and it is served
  when it reaches the front.
- A hold that was already granted is never taken back. Started work is never preempted.
- The pool logs one `pool_queue_reordered` event with both lease ids, both nets and the margin, so
  every reorder can be inspected.

**Anti-thrash rules.**
- **Margin:** value reorders only when the gap is at least 0.10 net. A smaller gap keeps arrival
  order.
- **Aging:** a waiting hold gains 0.10 net per hour waited. A gap of 0.3 closes in 3 hours, so no
  hold waits forever behind a stream of slightly better wants.
- **Same family:** two holds of the same family keep arrival order. Their nets come from the same
  appetite, so reordering them is churn, not choice.
- **Continuations:** a stage-2 reading run (or any later step of one started job) carries the net of
  its job's first bid. It is not protected, and it is not re-bid.
- **Upkeep:** a hold with no bid (compactors, orion-day, curiosity compaction) keeps its arrival
  place ahead of every bid that came after it. Without this rule, upkeep would always lose to any
  positive net.

**Fairness with the floor.** The queue never refuses, so it cannot push a family's funded share
below the floor. It can only make a family wait longer. Two changes keep that honest:
- The floor counts **acts started** (the act ledger, 7 days) per Tier 2 family, not funded bids.
  Funded bids measure how often a family bids (retro: 73% of bids, 5 sessions).
- The live-week check watches per-family p90 queue wait. No family's p90 wait may rise by more than
  1 hour against its 14-day baseline.

**The reservation book.** Book displacement is answered by replay A, and it is **not built**. It
would change 0 to 2 of 2,041 decisions. The Prime self-block that inflated patch 1's count is
already fixed by the mindloop release.

#### Replay A: displacing unopened reservations in the book (14 days)

Rule replayed:
- the holder is another family;
- the holder's tier is not higher;
- the holder was not funded by the floor;
- each holder is displaced at most once;
- the challenger's net is at least the holder's net + margin;
- the freed amount covers the overshoot.

| Margin | Capacity deferrals | Named a holder | Same family | Margin not met | Winner changes |
|---|---|---|---|---|---|
| 0.10 | 2,041 | 80 | 73 | 4 | 2 (curiosity and self-sense, one each way) |
| 0.00 | 2,041 | 80 | 73 | 4 | 2 |

Both holders were durable runs that had already opened. Pending durable reservations never count as
holders. So under "unopened only", the count is 0.

#### Replay D: the background agent hold queue by value (14 days, 1,442 holds)

Reading's holds carry the net that replay C gives each read. Bidding runs carry their last recorded
net. Upkeep holds are protected.

| Aging | Grants with another hold waiting | Granted family changes | Share of all holds |
|---|---|---|---|
| 0.10 net/h | 655 | 213 | 14.8% |
| 0.00 | 655 | 220 | 15.3% |

At aging 0.10, these families would be served sooner on these counts: PR field notes 46, upkeep 54,
reading stage 2 35, self-inquiry 24, curiosity 20, reading stage 1 19, self-sense 15. These would
wait longer: reading stage 1 74, reading stage 2 49, self-sense 29, self-inquiry 27, PR field notes
22, curiosity 12. The top pairs were PR field note over reading (35) and self-inquiry over reading
(12+). Reading over self-sense (11) and reading over PR field notes (11) also happen.

This is a rank-order approximation. Service times are kept as they really were, so it counts
changes. It does not simulate the new waits. The winner really changes on about 1 in 7 background
holds, and both directions occur. That is choosing, not a fixed priority in disguise.

### 2. Charges that make selection real

**Smallest honest fix: a measured cost, not a new price.**

1. **Reading's cost join.** The act-cost fill (`orion/grammar/act_ledger.py` `fill_close_cost`,
   `orion/grammar/act_energy_fill.py`) charges a lease to the act whose `correlation_id` equals the
   lease's `turn_correlation_id`. Reading's act uses the seed's `request_id`, but its leases carry
   `trace_id` and `stage2_trace_id`. The fix: Hub's reading act close names its cost correlation ids
   (the two trace ids), and the fill sums leases over all of them. This is the same fill, with a
   list of ids instead of one. Backfill 30 days, so appetite's p90 lane cost exists on day one.
2. **Reading declares its lanes.** In `config/physiology/families.yaml`, `hub.world_pulse_read`
   changes from `lanes: []` to `lanes: [agent, agent-gpu2]`, `placement: one_of`. The comment that
   says "non-GPU" is replaced with the measured numbers.
3. **No price change.** Posted prices stay as they are. Replay C shows that, with the real cost,
   reading's charge is still about 0 (median 0.000, p90 0.0002), because the agent lane's price is
   slack 92% of the time. Selection comes from section 1's queue order. We do not invent a price
   that families would not yet pay. Pricing queued demand is a separate question for the budget
   organ, and it is not in this patch.
4. **Camera asks: no charge in patch 3.** Their real cost is Juniper's attention per card, and the
   sql-writer batch already caps it (about 8 a day). `attention:juniper` counts messages, not cards.
   Folding cards into it would change that line's meaning. Camera asks are 4th in the binding order,
   so their charge is decided with their binding, after a metric gate on "cards answered per day".

#### Metric quality gate: reading's lane cost (slot-seconds per read)

1. **Provenance.** `gpu_pool_leases` (granted_at .. release) and `gpu_pool_events.held_ms` on
   `released`, written by the GPU pool. Joined by `turn_correlation_id IN (world_pulse_read_seed.trace_id,
   stage2_trace_id)`. Cross-checked: the lease spans sum to 131.6 h, and `held_ms` over the same
   4,676 releases also sums to 131.6 h.
2. **Independence.**
   - Hunger is time since the last read closed. Weight is downstream use and Juniper engagement.
     Salience is links made. Cost is lane time: a different quantity.
   - Against salience: the correlation of per-read slot-seconds with accepted links is **r = -0.02**
     (362 reads). Reads with no link averaged 1,350 s, and reads with links averaged 1,184 s.
   - It is not a transform of any input reading already has.
3. **Theory.** Opportunity cost. A slot-second reading holds is a slot-second curiosity cannot use.
   That is Choice's own definition of a lane charge (`price × cost_frac`, from dual ascent on the
   lane's budget).
4. **Live data.**
   - 364 reads with lease time: median 1,057 s, p90 2,586 s, mean 1,301 s. Not flat, not
     saturated.
   - Rest point: a read that makes no model call holds 0. Every traced read made calls; none read 0.
   - 443 of the 807 read acts have no traced lease in this extract. Not investigated: failed fetches,
     retries and seeds outside the window are the candidates. Without a measured cost they fall back
     to appetite's p90 for the family.
5. **Existing mechanism.** `ActCostV2.gpu_slot_s_by_lane` and the act-cost fill. No new field and no
   new signal.
6. **Reversibility.** The fill change is additive (more leases joined). The families.yaml lanes
   line is one config revert. No schema change.

### 3. The consent gate

**What it does.** Before a binding fund of an act that changes something hard to put back, the
market asks one pure function: "is this act allowed, and who agreed?". The function answers
`allow`, `refuse` or `needs_consent`, with the rule it used. The answer is stored with the decision.

**What it unifies.** It takes over only the retired policy rung's job: the act's reversibility class
and its consent requirement. It does **not** absorb heat, arousal, caps, quiet hours, the
Juniper-first yield or the self-monitor act gate (decided item 6). Those stay where they are, because
they answer "is now a good time", and the market and the doers already ask that.

**Contract.**
- `config/physiology/consent.yaml`: one entry per (family, verb):
  - `reversibility`: `reversible` | `snapshot_restorable` | `irreversible`;
  - `consent`: `none` | `standing:<id>` | `refused`;
  - `requires`: a list of named preconditions the caller attests in the request.
- Standing consents are entries Juniper merges in a PR. The merge is her consent, and git history is
  the record.
- `orion/physiology/consent.py` `consent_verdict(want, request, rules, now) -> ConsentVerdictV1`.
  It is pure and deterministic. An unknown family/verb pair with `reversibility` other than
  `reversible` gets `refuse` (fail closed). A `reversible` act with no entry gets `allow`, so today's
  families are unchanged.
- `ConsentRequestV1`: `want_id`, `family`, `verb`, `object_ref`, `payload_digest`, `attests` (for
  example `snapshot_dir_writable`, `downscale_only`).
- `ConsentVerdictV1`: `verdict`, `rule_id`, `reason`, `consent_ref`.
- **Trace:** `WantDecisionV1` gains an optional `consent` field (the verdict). A `refuse` or
  `needs_consent` verdict makes the decision `deferred`, reason `consent`. `bind_verdict` treats
  reason `consent` as **never admit**, even at the window's end, unlike every other deferral. It
  logs `physiology_consent verdict=... rule=... want=...`.
- **v1 has no per-act ask.** `needs_consent` means "a standing consent is missing". Asking Juniper
  per act through `orion_ask` is a later patch, if one is ever needed.

**First consumer: dream compaction apply.**
- `services/orion-dream/app/compaction_applier.py` already takes an injected `consent` callable,
  which says no today.
- It becomes a call to the market (`POST /physiology/bid` with a `consent_request`), for family
  `dream.compaction`, verb `compact`, `reversibility: snapshot_restorable`.
- The rule `dream.compaction.downscale_only` requires the attests `snapshot_dir_writable` and
  `downscale_only`.
- Prune is `irreversible` and `refused`.
- `ORION_DREAM_COMPACTION_APPLY_ENABLED` stays false. This patch builds the gate. Turning apply on
  is Juniper's separate decision.

### 4. Binding the next family: reading

**What binding changes.**
- A stage-1 reading run bids at its first resource request in durable-runs admission
  (`RunBidder.bind`), exactly as curiosity does.
- If it is funded or queued at the pool, the run asks the pool as before.
- If it is deferred (`price`, `sated`, or `consent`), the run is held (`waiting_resource`,
  `run.economy_held` event) and bids again at `retry_at`. It never asks for a seat while held.
- Stage 2 never bids. It is the second half of a funded read.
- New producer: `orion-durable-runs.admission.reading.binding`, added to `BINDING_PRODUCERS`.
- The bid map is a separate `BIND_ONLY` entry for the reading workflow. `run_act_family` is not
  changed, because Hub already opens reading's act, and a second act would double-count.

**What changes in Hub's loop.** Nothing in its code path. A held run is an in-flight read, so it
counts toward `MAX_IN_FLIGHT=3` and Hub polls it. That is the back-pressure: while reads are held,
Hub starts no new ones. The one coupling: Hub reclaims a claim after 3,500 s. So reading's hold end
is `min(family window 10,800 s, 3,500 - 600) = 2,900 s` after the run's request. After that the run
goes anyway (`window_end`, held but never dropped). That is the same rule curiosity uses.

**Replay C** (real `choice.decide`, real `physiology_state`, real act history with reading's measured
cost patched in, 807 real read starts):

| Rule | Funded | Queued at pool | Held at all | Held > 1 h | Reading's share of Tier 2 funded wants (7 d) |
|---|---|---|---|---|---|
| shipped `bind_verdict` | 146 | 661 | 2 (0.2%) | 0 | 7.4% (floor 5%: passes) |
| hold capacity deferrals with net < 0.15 | 146 | 661 | 2 (0.2%) | 0 | 7.4% |

- Reading's replayed net: p10 0.51, median 0.74, p90 0.81.
- Weight: median 1.09. Rank: median 0.68.
- Reading is not starved, and binding does not slow it.
- The 82% capacity deferral rate (663 of 807 bids) is what section 1's queue order acts on.
- Binding reading before the queue change would only add bookkeeping. **So the queue change ships
  first.**

**Live week.**
- Flag `ORION_PHYSIOLOGY_BID_READING` in durable-runs: `on` (default since 2026-11-12; was `shadow` until the live week) or
  `on`. This follows `ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE`.
- Off switch: set it to `shadow`. That restores today's behaviour exactly.
- Daily checks (counts go in the PR report):
  - reads started per day within the 14-day range (15 to 130);
  - share held more than 1 h at or under 5%;
  - reading's share of Tier 2 acts started at or above 5%;
  - `physiology_bind run=reading-... obeyed=true` lines;
  - `want_ledger` rows with the binding producer;
  - read acts that close with a non-null lane cost.

### 5. Self-grade saturation: what would unblock source (d)

Source (d) stays off. It needs a **world-graded success** close: the act's output was used by
something other than its producer within the act's horizon, and that use is recorded as the grade.

The existing table already has the shape. `act_downstream_use` has 337 linked uses in 14 days, and
114 of them are `used`. Two concrete candidates:
1. A reading link (`accepted_provisional` receipt) that a later chat turn, curiosity run or Prime
   concern cites.
2. A curiosity finding that Prime or Juniper acts on.

**Unblock condition:** at least 3 Tier 2 families have at least 30 world-graded closes each in 14
days, and fewer than half of them read 0.98 or more. Then (d) runs as "world-graded only". This
design does not build the signal.

## Missing questions (each with a recommended default)

1. **Does the reservation-book displacement get built anyway, for completeness?**
   Default: **no**. Replay A says it changes 0 to 2 decisions in 14 days. Building it would be a rule
   with no measurable behaviour.
2. **What margin and aging rate does the queue use?**
   Default: margin **0.10 net**, aging **0.10 net per hour**. Replay D: the reorder count barely
   moves between aging 0 and 0.1 (220 vs 213). Aging is there for the tail, not the median.
3. **Does reading's pool priority stay at the current mix (background holds, system steps)?**
   Default: **yes**. Value reorders only inside the background hold queue, where reading and the
   bidding runs already meet. Moving reading's steps to background is a larger change to Hub's
   reading speed, and it is not needed for selection.
4. **What does the exploration floor count?**
   Default: **acts started per Tier 2 family over 7 days** (the act ledger), not funded bids. This
   changes Choice's `funded_share` source from `want_ledger` to `grammar_acts`. It is the one change
   to the floor in this patch. The floor still only applies when the price said defer.
5. **Does upkeep with no bid keep its queue place?**
   Default: **yes**, arrival order ahead of later bids. Tier 1 upkeep should bid in its own patch,
   rather than compete with net 0.
6. **What is reading's hold end?**
   Default: **2,900 s**, from Hub's 3,500 s reclaim minus a 600 s margin. No Hub change.
7. **Does `bind_verdict` keep admitting capacity deferrals to the pool?**
   Default: **yes** (`queued_at_pool`). With the queue ordered by value, the pool queue is the
   choice, and a second queue in front of it held 43 of 46 curiosity runs for hours (Patch D replay).
8. **Where does the consent gate run, and what does an unknown act get?**
   Default: in the market process (durable-runs), as a pure function. An unknown act that is
   `reversible` gets `allow`, and an unknown act of any other class gets `refuse`.
9. **Does the consent gate ever ask Juniper per act?**
   Default: **not in v1**. Standing consents in YAML, merged by her PR. Per-act asks wait until a
   consumer needs one.
10. **Camera asks' charge?**
    Default: **none in patch 3**. Decided when camera asks bind, after a metric gate on cards answered
    per day.
11. **Pricing queued demand on a lane?**
    Default: **not here**. It is the budget organ's question. Replay C shows price is not what
    selects today.

## Decided (Juniper, 2026-11-10): all defaults approved

Juniper approved every recommended default in "Missing questions". Patch 3a builds next.

1. Reservation-book displacement: **not built** (replay A changes 0 to 2 decisions in 14 days).
2. Queue margin and aging: margin **0.10 net**, aging **0.10 net per hour**.
3. Reading's pool priority: **unchanged** (background holds, system steps). Value reorders only
   inside the background hold queue.
4. Exploration floor: counts **acts started per Tier 2 family over 7 days** (`grammar_acts`), not
   funded bids.
5. Upkeep with no bid: **keeps its arrival place** ahead of later bids.
6. Reading's hold end: **2,900 s** (Hub's 3,500 s reclaim minus 600 s). No Hub change.
7. `bind_verdict` keeps admitting capacity deferrals to the pool: **yes** (`queued_at_pool`).
8. Consent gate: runs **in the market process** (durable-runs) as a pure function; an unknown
   `reversible` act gets `allow`, any other unknown act gets `refuse`.
9. Per-act consent asks: **not in v1**. Standing consents in YAML, merged by Juniper's PR.
10. Camera asks' charge: **none in patch 3**; decided when camera asks bind.
11. Pricing queued demand: **not in this patch**; it is the budget organ's question.

## Proposed schema / API changes

All additive. `WantDecisionV1` is `extra=forbid`, so for `consent` the consumers go first:
sql-writer, Hub, dream, actions, cortex-orch, mind and durable-runs are rebuilt before any decision
carries it. The new field is omitted when unset, as patch 1 did.

- `orion/schemas/physiology_want.py`
  - `WantDecisionV1.consent: Optional[ConsentVerdictV1]` (omitted when None).
  - `ConsentRequestV1`, `ConsentVerdictV1` (new).
  - `WantV1.consent_request: Optional[ConsentRequestV1]` (omitted when None).
- GPU pool lease request (`GpuLeaseRequestV1`, `orion/schemas/`):
  - optional `want_id: str`, `want_net: float`. The pool ignores them outside the background class.
  - The pool event `pool_queue_reordered` is registered in `orion/bus/channels.yaml` and
    `orion/schemas/registry.py` if it goes on the bus. Otherwise it is a `gpu_pool_events` row
    (preferred: no new channel).
- `orion/physiology/binding.py`: reason `consent` is never admitted; a new producer joins
  `BINDING_PRODUCERS`.
- `orion/physiology/choice.py`: the floor's `funded_share` reads acts started; `decide` consults
  `consent_verdict` for a want with a `consent_request`.
- Config:
  - `config/physiology/families.yaml`: reading's lanes; new family `dream.compaction` (Tier 2, no
    lanes, no receptors).
  - `config/physiology/consent.yaml` (new).
- Env: `ORION_PHYSIOLOGY_BID_READING` (durable-runs), and `GPU_POOL_VALUE_ORDER_ENABLED`,
  `GPU_POOL_VALUE_MARGIN`, `GPU_POOL_VALUE_AGING_PER_H` (gpu-pool). Per the standing rule, the value
  order ships **on**. The reading flag ships `shadow` until its live week starts, because decided
  item 4 requires a deliberate live week.

## Files likely to touch

- `orion/grammar/act_energy_fill.py`, `orion/grammar/act_ledger.py`: a cost join over a list of
  correlation ids.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py` (act close names the trace ids), plus a
  30-day backfill script under `scripts/` (backfill protocol, snapshot first).
- `config/physiology/families.yaml`, `config/physiology/consent.yaml`.
- `orion/physiology/{choice,binding}.py`, new `orion/physiology/consent.py`.
- `orion/schemas/physiology_want.py`, the GPU pool lease schema.
- `services/orion-gpu-pool/app/runtime.py` (`_position` and the grant pick), `config/gpu_pool.yaml`,
  and its settings and `.env_example`.
- `services/orion-durable-runs/app/run_bids.py` (the reading bind map, hold end),
  `app/physiology_market.py` (consent), settings and `.env_example`.
- `services/orion-dream/app/compaction_applier.py` (the consent callable calls the market).
- Tests next to each. Evals:
  - `services/orion-durable-runs/evals/reading_binding_replay.py` (replay C);
  - `services/orion-gpu-pool/evals/value_queue_replay.py` (replay D);
  - `services/orion-durable-runs/evals/consent_gate_fixture.py`.

## Non-goals

- No change to Juniper's turns, perception, or any interactive or system lease order.
- No preemption of started work.
- No new price, and no change to posted prices.
- No reservation-book displacement (replay A).
- No change to heat, arousal, quiet hours, caps or the self-monitor act gate.
- No per-act consent asks.
- No source (d), and no world-graded signal built here.
- No binding for camera asks, outreach, journals, painting/dreams or play.
- Dream compaction apply stays off.

## Acceptance checks

1. **Cost join:** after deploy and backfill, at least 90% of reading acts with a traced lease close
   with non-null `gpu_slot_s_by_lane`. Their 14-day sum matches the pool's reading slot-hours within
   2%.
2. **Queue order, unit:**
   - with value order off, the pick equals today's `(priority, created_at)` for every replayed grant;
   - a gap under the margin keeps arrival order;
   - the same family keeps arrival order;
   - upkeep keeps its place;
   - aging lets a 0.3-lower hold win after 3 h;
   - interactive and system order are untouched.
3. **Queue order, replay:** replay D re-run in CI's eval lane reports the change share. It must be
   above 0, which shows the queue really chooses.
4. **Consent gate:**
   - a fixture shows `irreversible` is refused, an unknown non-reversible act is refused, and a
     missing attest is refused;
   - `downscale_only` with both attests is allowed;
   - a `consent` deferral is never admitted at the window's end;
   - with no consent request, every decision is byte-identical to today's.
5. **Reading replay gate:** before the flag goes `on`, replay C on the latest 14 days shows reading's
   Tier 2 acts-started share at or above 5%, and the share held more than 1 h at or under 5%.
6. **Floor:** the floor's share for each family equals a hand-computed `grammar_acts` count on a
   fixture.
7. **The live path moved.** Over the live week (counts in the PR report):
   - `pool_queue_reordered` rows exist, with both directions of family pairs;
   - reading's runs show `physiology_bind ... obeyed=true` lines and binding-producer `want_ledger`
     rows;
   - reads per day stay within 15 to 130;
   - no family's p90 background wait rises by more than 1 h against baseline;
   - the dream applier logs `not_approved` with `rule=` on the consent verdict while apply stays off.
   A flag set to on is not evidence. If any count is missing, the status is `UNVERIFIED`.

## Recommended next patch

**Patch 3a: reading's real cost (no behaviour change).**
1. Fix the reading cost join (a list of correlation ids in the act-cost fill).
2. Back-fill 30 days under the backfill protocol.
3. Declare reading's lanes.
4. Commit replays C and D as evals.

This is the smallest slice. It changes no decision of any binding family (reading is still shadow).
It turns "reading is free" into a measured fact that every later step reads. Verify: a reading act
closed after the deploy carries `gpu_slot_s_by_lane`.

Then, in order:
- **3b:** the pool queue orders the background class by value, plus the floor counting acts.
- **3c:** reading binds (live week).
- **3d:** the consent gate, with dream compaction as its consumer.

3d does not depend on 3a to 3c. It can go in parallel.

## Proposal-mode checklist

- **Capability that changes:** when Orion's own discretionary jobs wait for the same GPU seat, the
  more valuable one goes first. Reading starts bidding for its effort. Acts that cannot be undone
  need a recorded consent.
- **Data touched:**
  - Reads: `gpu_pool_leases`, `gpu_pool_events`, `grammar_acts`, `want_ledger`,
    `world_pulse_read_seed`.
  - Writes: reading acts' `cost_json` (backfill, snapshot first), want decisions, pool queue events.
- **Privacy boundary:** none (Juniper's standing rule). Nothing new is shown to anyone.
- **Trace that proves it worked:**
  - `pool_queue_reordered` rows;
  - `physiology_bind run=reading-*` lines;
  - non-null reading lane cost;
  - consent verdicts on decisions;
  - the live-week counts in acceptance check 7.
- **Dangerous failure modes:**
  1. **Reading starves curiosity's seat**, or the reverse. Mitigation: the margin, aging, the p90
     wait check, and the floor on acts started.
  2. **Upkeep loses to bids.** Mitigation: upkeep keeps its arrival place (tested).
  3. **A held read is reclaimed by Hub mid-hold.** Mitigation: the hold end of 2,900 s is under Hub's
     reclaim.
  4. **The consent gate fails open on dream compaction.** Mitigation: an unknown non-reversible act
     is refused, a `consent` deferral is never admitted at the window's end, and the apply flag stays
     off.
  5. **The backfill writes wrong costs.** Mitigation: snapshot to `/tmp/reading-cost-backfill/`
     first; restore from it.
- **Disable or roll back:**
  - `GPU_POOL_VALUE_ORDER_ENABLED=false` restores arrival order.
  - `ORION_PHYSIOLOGY_BID_READING=shadow` restores today's reading.
  - Reverting the families.yaml lanes line removes reading's charge.
  - The consent gate is inert for any want without a consent request. The dream flag is off.

## How the replays were run

All are read-only, over CSV extracts of `want_ledger`, `gpu_pool_leases`, `grammar_acts`,
`act_downstream_use`, `physiology_state`, `world_pulse_read_seed` and `reading_durable_turn`
(14 days; history 46 days).

- **A:** every `capacity_reserved` row, with its `displaced_by` holders looked up as funded
  decisions in the ledger. The rules are as stated in section 1.
- **C:**
  - real `orion.physiology.choice.decide` and `build_context`, run from this worktree;
  - the `physiology_state` row in force at each read start;
  - reading's acts patched with their traced agent and agent-gpu2 slot-seconds;
  - families.yaml with reading's lanes set to agent and agent-gpu2, `one_of`;
  - the book seeded with real funded wants;
  - each real read start bids, and re-bids at `bind_verdict`'s `retry_at`.
  - Pessimistic: the lane levels include reading's own real use.
- **D:** at each real grant of a background agent hold, the holds waiting at that moment, picked by
  `net + aging × hours waited` instead of arrival. Service times are kept as real, so this counts
  changes and does not simulate new waits.

The scripts are not committed in this docs PR. Patch 3a commits C and D as evals.
