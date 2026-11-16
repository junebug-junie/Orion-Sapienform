# Family C first experiment, rewritten around the body economy and the tiers

Status: DESIGN (proposal mode). Nothing here is built or run. Replaces the proposal in
`docs/superpowers/pr-reports/2026-11-15-pe-family-c-slice1-pr.md` ("Proposal: the first new switchback
experiment"). Tracking issue: #3482 (needs Juniper).

## Arsonist summary

The old proposal wanted to cap circe GPU3 at 120 W with a new root helper. Three things were wrong
with it, and a live check found a fourth:

1. GPU3 is not a spare card. It is the `fast` lane (fast + metacog work, :8013) and it hosts Kev
   (System One appraisal). Seven tier-1 upkeep families run only there, and `actions.journal` uses
   it too (it also spans the agent lanes). A cap slows all of them, and
   the old proposal measured only heat, not that cost.
2. Juniper decided on 11-08 that Family C actuators are the attention economy (prices, bids) and run
   caps set by a runtime override with a TTL, not env and not hardware.
3. circe already has a root job that holds per-card caps (`orion-gpu-power-limits`). A new root helper
   would be a second one.
4. The card cannot go to 120 W. `nvidia-smi -i 3` on circe today: Tesla V100-SXM2-32GB, limit 300 W,
   default 300 W, **min 150 W**, max 300 W. The old ON arm would have been refused by the driver.

The new first experiment is a software switchback on one tier-2 family: **AI Town sessions**
(`aitown.session`). The experiment randomly holds a town session that was about to start, for 40
minutes, and measures what the session costs the body (GPU3 watts and heat) and what it costs the
lane (how much slower tier-1 work on `fast` gets, and Kev's latency). ON is what the town does today.
OFF only removes load. Tier-1 work is never held. The total play time per day does not change: the
2 h daily cap is still reached, only later.

The power-cap experiment moves to second place. It runs through the existing root job as a TTL
override, and it needs Juniper to reverse one earlier decision: the 10-29 power-cap design says gpu3
is never capped, because tier-0 perception (vision council) runs on it.

## Current architecture

- **Family C slice 1** (live, #3485): `services/orion-prediction/app/family_c.py`. The effect table
  (`GET /effects/table`) has one row per (actuator, Family A reading, window). `choose()` returns only
  `randomised` rows. The efference copy adds a row's effect to the newest Family A forecast and scores
  the real reading.
- **Experiment runner**: `services/orion-prediction/app/experiment_runner.py`. One actuator,
  `vision.focus:<stream>`. ON = a Redis key with a 90 s TTL that the runner refreshes every tick;
  the vision frame router reads it. Heat guard, a Redis lock, boot recovery (`recover()` cancels a
  half-done run). The only experiment ever run (10-29) moved nothing measurable, and half its OFF
  blocks were already in the triggered tier (#3484).
- **Body economy**: `orion/physiology/` (families, choice, prices, appetite, prime_charge) and the
  market `POST /physiology/bid` in `services/orion-durable-runs/app/physiology_market.py`. Decisions land
  in `want_ledger`. Every row carries `shadow=true` by schema (`orion/schemas/physiology_want.py`
  documents this); whether a decision was obeyed is told by the producer name (`*.binding`).
- **Tiers**: `config/physiology/families.yaml`. 0 never bids. 1 critical upkeep with a reserved lane
  share. 2 discretionary open market. Changing a tier needs Juniper.
- **Existing TTL override mechanisms** (existing-mechanism check):
  - Prime's regulate charge: Redis hash `orion:prime:regulate:<family>`, an expiring slow/quicken
    charge on a family's bids, with a built-in `control` (holdback) arm
    (`orion/physiology/prime_charge.py`). It acts only on families whose bids bind and that declare a
    receptor. Today: `dream.cycle` (slow) and `durable.curiosity_investigate` (slow). Neither is a
    tier-2 family on `fast`.
  - The vision-focus key above.
  - **No generic run-cap override exists.** Run caps are env values read at boot
    (`EMBODIMENT_TOWN_DAILY_CAP_SEC`, `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`, ...).
- **Power caps**: `scripts/ops/gpu_power_limits.py` (root, stdlib only), `config/gpu_power_limits.json`
  (gpu1 and gpu2 at 150 W; gpu3 not listed), `scripts/ops/orion-gpu-power-limits.timer` (every 5 min).
  The 10-29 design (`docs/superpowers/specs/2026-10-29-gpu-power-cap-design.md`, "Missing questions"
  1 and "Proposals, not writes") says gpu3 is never capped and the economy never writes the file.
- **Lane cost already recorded** (no new producer needed):
  - `gpu_pool_leases`: one row per lease, with `role`, `holder`, `work_class`, `priority`,
    `queued_since`, `granted_at`, `updated_at` (release time for `released` rows; 19,535 of 19,537 fast
    rows in 14 days are `released`), `want_family`. Grant wait = `granted_at - queued_since`. Hold =
    `updated_at - granted_at`.
  - `substrate_system_one_appraisal.frame_json->>'latency_ms'`: Kev's per-appraisal latency
    (6,899 rows in 14 days).
  - `grammar_acts.cost_json->'gpu_slot_s_by_lane'->'fast'`: GPU slot-seconds per act.
  - Temporal source `gpu_wait` (body organ) already turns long waits into experience.

## Choosing the family (live data, 2026-11-02 .. 2026-11-16)

Tier-2 families that may use `fast`: `topic_foundry.run`, `hub.outreach`, `hub.retro_coplay`,
`aitown.session`, `concept_induction`, `execution_dispatch.substrate`, `thought.reverie`.

Acts on the act ledger in 14 days (`grammar_acts`, by actor):

| Family | Acts | Per day | Shape |
|---|---|---|---|
| hub.outreach | 177 | 12.6 | messages Juniper; outcome lands in hours or days; also spans agent/chat |
| aitown.session | 58 | 4.1 | 30-min sessions on embodiment's own timer, ~95 min apart |
| topic_foundry.run | 39 | 2.8 | 2-3 runs in ONE batch per night (16 batches) |
| the other four | 0 | 0 | nothing to experiment on |

`hub.outreach` is ruled out: its effect is on Juniper, not on the body, and the 11-08 rule puts it
last (day-long blocks). `topic_foundry.run` has the biggest dose but one batch per day, so one block
per day. That leaves `aitown.session`.

What a town session does to the body and the lane. These are **not randomised** comparisons. They
justify the experiment; they are not its result.

- **The town is most of `fast`'s background traffic.** 3,975 `fast` leases in 14 days are held by
  `http:openai` at `background` priority (the town's NPC dialogue). 3,958 of them started inside a
  town session.
- **GPU3 watts.** Each session vs the 30 minutes before it: 126.7 W vs 113.6 W, +13.1 W (SD of the
  paired difference 15.9 W, n = 58). Same UTC hours (07-12), 30-min windows with a session vs
  without: 125.3 W (81 windows) vs 122.3 W (80), +3.0 W (SD 14.4). The true effect is somewhere
  between 3 and 13 W, and the two comparisons disagree because sessions sit at fixed night hours.
  That disagreement is the reason to randomise.
- **Slot use.** 10 to 1,288 GPU slot-seconds on `fast` per session (`cost_json`), rising over the
  last week.
- **Tier-1 lane cost.** Tier-1-class calls on `fast` (holder `cortex-exec`, class `metacog`, priority
  `system`: metacog.log, spark.introspect and friends) hold their slot 11.6 s on average inside a
  session vs 4.2 s outside (n 259 vs 1,631). Grant wait 0.53 s vs 0.29 s; p95 2.32 s vs 1.30 s.
  Paired, per session with such calls in both windows: +6.9 s (SD 2.4, n = 12).
- **Kev.** Appraisal latency p50 275 ms inside a session vs 224 ms in the same hours outside
  (n 555 vs 1,137). Paired +24 ms, but SD 114 ms: too noisy to be a primary outcome.

`topic_foundry.run`, for comparison: +18.3 W during a batch vs the 30 min before (SD 15.7, n = 16).
Bigger, but 1 block per day means ~6 weeks for the same power. It is the natural second family.

## Experiment 1 (recommended first): hold a town session, randomly

### What it does, in plain words

When a town session is about to be allowed to start, the runner takes the next arm from a
random list drawn in advance. ON: nothing happens, the town starts as it would today. OFF: the runner tells the town controller "no new
session for 40 minutes". When the 40 minutes end, the session is due again, and the runner takes
the next arm. Each decision is one block.

### Actuator: a run-cap TTL override (not a price multiplier)

A price or bid multiplier was the other option. It is wrong for this family today: `aitown.session`
bids are shadow (`embodiment` places the bid and never obeys it), so a price change on it moves
nothing real. Prime's existing regulate charge has the same limit (it needs a binding family with a
receptor). A run-cap override acts on the controller's own gate, which is real.

The override follows the 11-08 rules exactly:

- **The knob.** The town has no session-count cap to lower. `start_block_reason` has only the daily
  cap, the gap, the minimum session and the enabled switch (env `EMBODIMENT_TOWN_*`). So the knob is
  an honest boolean, `hold_starts`: env value `false` (today), override value `true`. It is the
  start gate only.
- **Runtime, not env.** Redis key `orion:physiology:run_cap:aitown.session`, value
  `RunCapOverrideV1` JSON `{family, knob: "hold_starts", value: true, experiment_id, block_index,
  set_at, expires_at}`, Redis `EX` TTL 90 s, refreshed by the runner every tick (30 s). The
  controller reads override-else-env. A dead runner falls back to env within 90 s.
- **`set_at` never moves.** It is the block's start. The runner never rewrites it on refresh. The
  controller obeys the key only while `now - set_at <= max_hold_s`. So a runner that is alive but
  stuck in its refresh loop still cannot hold the town longer than one block.
- **Bounds per knob, in config.** New `config/physiology/run_cap_overrides.yaml`:
  `aitown.session: {knob: hold_starts, env_value: false, allowed: [true], max_hold_s: 2700}`
  (40 min block + 5 min slack). A key with no TTL, a value not in `allowed`, or `now - set_at >
  max_hold_s` is ignored and counted (`run_cap_override_refused`), never obeyed.
- **The override can only hold.** It never loosens any of the town's env caps.
- **Never stops a running session.** It gates the start only (new refusal reason
  `run_cap_override` in `orion/embodiment/town_sessions.py` `start_block_reason`). Its
  `retry_after_sec` is the key's Redis TTL (90 s or less), not `expires_at`. The self-start loop
  holds a refusal in memory without re-checking (`town_self_start.py` step 3), so a longer retry
  would break the "town resumes within one tick" rollback. Stopping Orion mid-play is not part of
  this design.
- **Side effects the build must handle.** The self-start places its shadow bid before the gate, so
  each OFF block writes a few shadow `aitown.session` wants that end blocked by `run_cap_override`.
  The build tags them, so appetite readers do not read them as hunger. The build also confirms that
  this refusal does not count against the town in the self-monitor track record (step 4 gate). If
  it did, the town would be held as broken and ON blocks would be contaminated.

### The experiment is itself a tier-2 act

- New family `prediction.experiment` in `config/physiology/families.yaml`: tier 2 (**Juniper's
  sign-off**), `lanes: []` (the runner holds no GPU slot; an OFF block only removes load),
  `deferrable_window_s: 2400`, no receptors.
- Before each decision the runner bids (`orion.physiology.client.bid`, want id = the block id) and
  **obeys**: producer `orion-prediction.experiment.binding`, added to `BINDING_PRODUCERS` in
  `orion/physiology/binding.py`. Without that, the market treats the bid as shadow and Prime's charge
  never applies. Funded: take the next arm and run the block. Deferred: no block; the town starts
  normally and nothing is logged as an arm. **No answer: also no block.** This differs on purpose
  from the repo's binding convention (binding producers admit on no answer, `NO_ANSWER_REASONS` in
  `binding.py`): an experiment that cannot reach the market is the one act that should not run.
- Honest limit: a lane-less tier-2 bid is priced only by its own family weight and any Prime charge.
  The bid makes the experiment visible in the market, the body report and Prime's view, and lets
  Prime slow it later. It is not what keeps the body safe. The stop rules below do that.
- Tier-1's reserved lane share is never touched: the experiment holds only `aitown.session`
  (tier 2) and never sets anything on another family or on the pool.

### Blocks

- **Eligibility comes from the town, not from copied env.** Embodiment's `/health` gains
  `town_start_eligible_at`. It is the controller's own answer: the end of the gap, or the next local
  midnight after the daily cap, or `null` while a session is open, the town is disabled, or the
  self-monitor gate holds it. The runner never re-derives the town's caps from env.
- **No race at the decision point.** The town ticks every 300 s on its own clock and the runner
  polls every 30 s, so the runner could see "eligible" after the town already started. To close
  that gap, the runner sets a *pending* hold (the same key, with the coming block's index) from 60 s
  before `town_start_eligible_at`. At eligibility it takes the next arm. ON deletes the key, and the
  town starts on its next tick (within 5 min). OFF keeps the key as the block's hold (`set_at` = the
  block start). No eligible moment is missed, and no session can slip into an OFF block. A pending
  hold that does not become a block (a refused decision) is deleted at once and delays the town's
  start by at most 60 s.
- **Arms.** One experiment per night: a run is one night, not 16 nights. At the night's start the
  runner draws that night's seeded arm sequence. The sequence is balanced and is re-shuffled until it
  has no 3 OFF in a row (the play floor). The seed is logged. Each decision takes the next arm, which
  is logged before the key changes. A post-pull rebuild of orion-prediction then costs one night,
  not the whole run, because `recover()` cancels only that night. The 16 nights are analysed pooled,
  as 16 strata, with a permutation test under the same restriction.
- **ON block**: no key. The town's own timer starts the session.
- **OFF block**: the key is held for 40 min, then removed.
- **Outcome window**: block start + [2, 35) min. 35 min covers a 30-min session plus the timer's
  start lag. Blocks start on a 5-min grain boundary, like today's runner.
- **Carryover.** After an ON block the next decision point is at least ~65 min later (the town's
  own gap), so the session's GPU tail is gone. After an OFF block nothing ran, so there is no tail.
- **Window of the night**: decision points only between 00:00 and 08:00 America/Denver (the town's
  sessions start at local midnight today: 07:00, 08:35, 10:10, 11:45 UTC). Outside it the town runs
  normally and nothing is randomised. Expected time per decision: 0.5 x 95 + 0.5 x 40 = 67.5 min, so
  ~7 decisions per night, ~3.5 per arm. This is an upper bound with no slack. The daily cap allows
  at most 4 ON sessions (7200 / 1800 s), and 3.5 + 3.5 blocks fill ~472 of 480 minutes.
- **Time of night is a confounder.** OFF blocks push later decisions later, so the two arms do not
  land at the same hours. The analysis uses `analyze_switchback`'s existing time-of-day `profile`
  and stratifies by night. Patch 0 simulates the real decision process (cap, window, tick lag,
  refusal rate) and reports the expected n per arm and the hour balance before any live run.

### Outcomes

Family A (already registered in `config/prediction/family_a.v1.yaml`; the effect table reads these
today): `circe.gpu3_watts` (primary), `circe.gpu3_temp_c`, `circe.chassis_watts`, the cabinet reading.

Lane cost (new effect-table targets, computed from existing tables; metric gate below). The effect
table can hold them only after `services/orion-prediction/app/series.py` gains two source kinds,
`pool_leases` and `kev_appraisal`:

- `lane.fast.tier1_hold_s`: mean hold of `cortex-exec`/`metacog`/`system` leases on role `fast`
  started in the window (primary lane cost).
- `lane.fast.grant_wait_s`: mean grant wait of every non-town lease on `fast` in the window
  (holder != `http:openai`/background), and its p95.
- `lane.kev.latency_ms_p50`: median Kev appraisal latency in the window (secondary, noisy).

### Power, from the live variance above

Minimum detectable effect at 80 % power, two-sided 5 %: MDE = 2.8 x SD x sqrt(2 / n_per_arm).

- 16 nights x ~3.5 per arm ≈ **56 blocks per arm** (an upper bound; see the window note above).
- GPU3 watts, block SD ≈ 15 W: MDE ≈ 2.8 x 15 x 0.189 ≈ **7.9 W**. The +13 W reading is detectable.
  The +3 W reading is not: if that is the truth, the result is an honest null with an interval about
  ±8 W wide, which is still a usable effect-table row (it says "a town session costs GPU3 less than
  ~8 W").
- Tier-1 hold: only some blocks contain tier-1 calls (5 per session on average; 12 of 58 sessions
  had them in both paired windows). Assume ~40 % of blocks, ~22 per arm. With the paired SD 2.4 s,
  MDE ≈ 2.0 s against +6.9 s seen. The unpaired SD will be larger than the paired one; Patch 0
  (below) recomputes it before the run starts, and the run is extended (or the outcome dropped)
  if the replay says it is underpowered.
- Kev: SD 114 ms, MDE ≈ 60 ms against +24 ms seen. Reported, not claimed.

### Manipulation check (the 10-29 lesson, #3484)

Per block, from the act ledger and the pool, not from the runner's own state:

- town session-seconds in the window (`grammar_acts` actor `aitown.session`, overlap with the window)
  and `http:openai`/`background` lease count on `fast`.
- Pass: ON minus OFF session-seconds > 0 with a 95 % interval excluding 0, and **every** OFF block
  has 0 session-seconds started inside it. One OFF block with a session started inside it fails the
  check: the override did not hold, and the run is marked `contaminated`, not analysed as clean.
- Report intention-to-treat (all blocks) and the share of ON blocks where no session started (the
  self-monitor gate, `town_unreachable`, a manual session).
- **Manual sessions.** A session Juniper opens by hand is not gated by the override, and it loads
  `fast` too. Any block with a manual session in its window is flagged, excluded from the primary
  analysis, and reported separately.

### Stop rules

Experiment 1 cannot add load: ON is what the town does today. Its stop rules protect data quality,
Orion's play, and the rule that the experiment yields to tier-1 work.

- **Refuse a decision** (no arm taken, town starts normally, refusal logged with its reason) when:
  - any lease on role `fast` with priority `urgent`, `interactive`, `perception` or `system` has been
    queued more than 5 s (`gpu_pool_leases.status = 'queued'`), i.e. tier-0/1 work is waiting;
  - the cabinet reading is at or above `thermal_gate.hot`, or missing, or strain is latched (the
    heat reflex sheds `background`, which is the town's NPC priority, so the arm would be
    contaminated anyway);
  - the market defers the experiment's own bid.
- **Abort the night** (key deleted, abort row with arm NULL, status `cancelled`, the existing
  runner path) on: a failed guard read, a lost lock, or the first contaminated OFF block (the same
  line the manipulation check uses).
- **Play floor**: never more than 2 OFF blocks in a row. This is built into the arm sequence
  (above), so no block is ever overridden after the draw. A refused decision does not consume an
  arm.
- **Dead runner**: key TTL 90 s. `recover()` at boot cancels the half-done night. Today it deletes
  the key through `stream_of()`, which knows only `vision.focus` (the error is swallowed), so for
  this key only the TTL would clear it. The build makes the key name depend on the action.
- **Blind spots of the stop rules.** The pool does not schedule Kev, so the queued-lease rule cannot
  see a Kev backlog. Kev latency is only an outcome. The build confirms that orion-prediction can
  read `gpu_pool_leases`.

## Experiment 2 (later): GPU3 power cap through the existing root job

Only after Experiment 1 has a row, and only if Juniper reverses the 10-29 "gpu3 is never capped"
decision (question 3).

### Design

- **ON = 150 W** (the card's floor; also gpu1/gpu2's level). Not 120 W: the driver refuses it.
  Live 14 days of GPU3 samples (38,558, ~31 s apart): median 78 W, p75 115 W, p90 214 W, p95 239 W,
  p99 265 W, max 295 W, mean 104 W. 16.3 % of samples are above 150 W. The cap can remove at most
  the mean of max(w - 150, 0) = **10.8 W** on average (an upper bound; the work then takes longer).
- **Where the override lives**: the existing job, not a new helper.
  - `config/gpu_power_limits.json` gains `"overridable": {"3": {"floor_watts": 150, "max_ttl_s":
    180, "max_hold_s": 1500}}`. A card not listed there can never be overridden.
  - The root job reads `/run/orion-gpu-power-limits/override.json` (`{card, watts, experiment_id,
    set_at, expires_at}`) on every check. The runner refreshes `expires_at` every tick (TTL 180 s).
    `set_at` is the block start and never moves. The job applies the override only when all of these
    hold: the card is overridable; `floor_watts <= watts <= the card's base target`;
    `now < expires_at <= now + max_ttl_s`; `now - set_at <= max_hold_s` (20 min block + 5 min).
    Otherwise it applies the base config. If the runner dies or a clear request is lost, the cap
    lasts at most ~3 min past its last refresh, not an hour. (The lane controller handles one request
    at a time, so a clear can wait behind a launch plan.)
  - **Root reads a file the container can write.** Open it with `O_NOFOLLOW`, refuse anything over
    4 KB, check the owner uid and mode, and parse a strict schema. Any failure means "no override".
  - **State and errors.** The job records the override it applied in
    `/run/orion-gpu-power-limits/applied.json`. On any error, including a `ConfigError` (which today
    leaves a card untouched), a card with an applied override goes back to its default limit. An
    applied override never outlives an error.
  - **The directory** is created by a `tmpfiles.d` entry with a fixed owner and mode, before the lane
    controller starts. Not by `RuntimeDirectory=`: that is deleted when the service stops and would
    leave the container's bind mount on a dead inode. If the directory is missing, Docker creates it
    as root.
  - The timer goes from 5 min to 1 min (`OnUnitActiveSec=1min`). One `nvidia-smi` query per minute.
  - **Install step.** The job runs as a copy in `/usr/local/sbin`, and its units are copies in
    `/etc/systemd/system`. Neither the code nor the 1-min timer arrives by `git pull`. The operator
    re-runs `sudo .../gpu_power_limits.py install` once. An old installed script ignores the
    `overridable` block, which is safe: it never applies an override.
  - Crash safety: `/run` is tmpfs, so a reboot clears the file. Stopping the service still restores
    defaults (`ExecStop`).
  - **Writer**: `orion-gpu-lane-controller` (already on circe, already the pool's actuator). New
    `GpuActuateV1` action `power_cap_override` with a TTL; the controller writes the file atomically
    through a new rw bind mount of `/run/orion-gpu-power-limits`. The runner on athena never touches
    circe directly. This is a second writer to the cap, next to the PR; that exception is question 3.
- **Scheduled as a tier-2 bid** (`prediction.experiment`, same as Experiment 1). It is refused **at
  decision time** while any tier-0/1 lease is queued on `fast` (same rule). An ON block is not ended
  early when tier-1 work queues. That queueing is the cost being measured: cutting it off would push
  the estimate toward zero and make blocks differ in length by arm. The only early end is heat
  (cabinet at or above `thermal_gate.hot`). The analysis window stays fixed (intention-to-treat),
  and the early-end rate per arm is reported as its own outcome.
- **Lane latency is a measured cost**: the same lane-cost targets as Experiment 1, plus the server's
  own tokens per second on `fast` if the llama.cpp worker reports it (UNVERIFIED that it does).
- **Blocks**: 20 min, outcome window [3, 20) min. Effect of a cap is immediate on watts; heat lags.

### Exact file changes for Experiment 2

- `scripts/ops/gpu_power_limits.py`: `load_overridable()`, `read_override()` (O_NOFOLLOW, size,
  owner, schema), `effective_target()`, the `applied.json` state, restore on error. `apply()` uses
  `effective_target()`. `status` prints the override and its expiry.
- `scripts/ops/orion-gpu-power-limits.tmpfiles.conf` (new); `install()` copies it to
  `/etc/tmpfiles.d/`.
- `config/gpu_power_limits.json`: `overridable` block for card 3 (and add `"3": {"watts": "default"}`
  so the card is managed at all).
- `scripts/ops/orion-gpu-power-limits.timer`: `OnUnitActiveSec=1min`.
- `tests/scripts/test_gpu_power_limits.py`: expired override ignored, out-of-range ignored, card not
  overridable ignored, `expires_at` too far out ignored, `set_at` too old ignored, symlink and
  oversize files refused, restore on expiry, restore on `ConfigError` with an applied override.
- `services/orion-gpu-lane-controller/app/actuator_bus.py`: `power_cap_override` action, atomic
  file write; `docker-compose.yml`: rw bind mount `/run/orion-gpu-power-limits`.
- `orion/schemas/` (`GpuActuateV1` action literal) and `orion/schemas/registry.py` if the action
  changes the schema; `orion/bus/channels.yaml` only if a new reply channel is needed (none planned).
- `services/orion-prediction/app/experiment_runner.py`: actuator `gpu.power_cap:circe.gpu3`.
- `docs/runbooks/2026-10-29-circe-gpu-power-limits.md`: the override, how to clear it by hand
  (`rm /run/orion-gpu-power-limits/override.json`, then wait one minute).

## Proposal mode

- **Capability change.** Orion gains a second real actuator it can randomise: holding its own AI
  Town session for 40 minutes. Family C gets its first lane-cost rows, so Orion can learn what its
  play costs its own upkeep, not only its heat. Experiment 2 (later) adds a time-limited power cap
  on one card.
- **Data touched.** Reads: `orion_biometrics_summary` (circe GPU readings), `gpu_pool_leases`,
  `substrate_system_one_appraisal`, `grammar_acts`, the town's Redis session ledger, the cabinet
  reading. Writes: one Redis key per block, `prediction_experiment`/`prediction_intervention` rows
  (existing tables), `want_ledger` rows for the experiment's bids, effect-table rows. Experiment 2
  also writes one file on circe's tmpfs.
- **Privacy boundary.** No chat text, no Juniper messages, no camera data. The town's NPC dialogue
  is not read; only lease counts and timings are.
- **Trace that proves it worked.**
  - Per block: the `prediction_intervention` row (arm, experiment_id), the Redis key's set/clear log
    lines with the experiment id, the controller's `aitown_session_refused ... refusal=run_cap_override`
    log line in OFF blocks, and the `want_ledger` row for the experiment's bid.
  - Per run: the manipulation check passes, and `GET /effects/table` has rows with
    `actuator = run_cap:aitown.session` and `method = randomised`.
- **Dangerous failure mode.** Experiment 1: a stuck override silently stops Orion playing (a key
  with no TTL, or a controller that obeys a stale value). Guarded three ways: the controller ignores
  a key with no TTL or past `max_hold_s`; Redis expires it 90 s after the runner dies; the play floor
  caps consecutive holds. Experiment 2: a stuck cap slows every tier-1 job on `fast`. Guarded by the
  root job's own expiry check every minute, tmpfs, and `ExecStop`.
- **Disable / roll back.**
  - Stop a run: `POST /experiments/runner/abort` (existing) or delete the key
    (`redis-cli DEL orion:physiology:run_cap:aitown.session`); the town resumes within one 300 s tick.
  - Disable the reader: remove `aitown.session` from `config/physiology/run_cap_overrides.yaml`;
    the controller then ignores any key.
  - Experiment 2: `rm /run/orion-gpu-power-limits/override.json` on circe, or remove the
    `overridable` block by PR.

## Integration gate

The runner becomes a producer of Orion experience: it adds a family (`prediction.experiment`), and
a finished experiment is something Orion did and learned. It belongs to the existing `prediction`
organ in `config/integration/organs.yaml`.

- **Emit.** New `TemporalSourceKind` `experiment_result` (in `orion/schemas/temporal_self.py`), one
  item per analysed experiment (not per block): "I held my town sessions to see what they cost:
  GPU3 +x W (interval), tier-1 calls +y s slower". Reader `experiment_result` in
  `services/orion-durable-runs/app/temporal_self_sources.py` over `prediction_experiment` rows with
  status `analyzed` joined to their effect estimates. Salience magnitude: |estimate| / SE of the
  primary outcome (a new `MAGNITUDE_RULES` entry `effect_z`), because a clear effect is more
  notable than a null, and z does not drift with age.
- **Read the now: exemption.** The prediction organ already has `exempt_read`. Add to its reason:
  "the experiment runner must not read experience: a randomised arm that depends on Orion's state
  is no longer randomised; its stop rules read sensors and the pool only". A reviewer reads this.
- Registry changes: `prediction` organ gets `families: [prediction.experiment]` and
  `temporal_sources: [..., experiment_result]`. `scripts/check_integration_seam.py` must pass.

## Metric gate for the lane-cost outcomes

Run for `lane.fast.tier1_hold_s`, `lane.fast.grant_wait_s`, `lane.kev.latency_ms_p50`:

1. **Provenance.** `gpu_pool_leases` rows written by the GPU pool on grant and release
   (`granted_at`, `queued_since`, `updated_at` at release). Kev's `latency_ms` is set in
   `orion/substrate/system_one_appraisal.py` (~line 359) from the Kev response and stored in
   `substrate_system_one_appraisal.frame_json`.
2. **Independence.** Hold time on `fast` and GPU3 watts share one cause (the card's load), so
   tier1_hold_s is not independent of gpu3_watts as a *signal*. It is a different *outcome*: the
   cost to other work, not energy. Grant wait measures queueing, hold measures service time; they
   are separate stages of the same lease. Kev latency is a different process on the same card.
3. **Theory anchor.** Queueing: with a fixed number of server slots, added concurrent requests
   raise both wait (Little's law, W = L / lambda) and per-request service time on a shared GPU
   (batched decode slows each stream). The live numbers move in that direction.
4. **Live sanity.** Not degenerate: hold 4.2 s outside sessions, 11.6 s inside; grant wait mean
   0.29 s, p95 1.30 s, so a calm state exists and is near zero, not floored. Caveat: hold mixes
   prompt length with contention; the switchback randomises the town, not the prompt mix, so the
   estimate is the town's total effect on hold, which is the quantity wanted.
5. **Existing mechanism.** `gpu_wait` (body organ) already turns long waits into experience; these
   outcomes reuse the same table and add no producer.
6. **Reversibility.** They are effect-table targets computed at analysis time from existing
   tables. Removing one means removing its series spec and its rows. No schema field, no stored
   series. (That this is one line is UNVERIFIED until the series kinds exist.)

## Missing questions

None blocking the design. The four decisions that need Juniper are at the end.

## Proposed schema / API changes

- `RunCapOverrideV1` (new, `orion/schemas/physiology_run_cap.py`): `{family, knob, value,
  experiment_id, block_index, set_at, expires_at}`. Stored in Redis only; register in
  `orion/schemas/registry.py` (no bus channel).
- `TemporalSourceKind` gains `experiment_result`.
- Town controller refusal reason `run_cap_override` (new value in the refusal set; surfaced in
  embodiment `/health` refusal counts).
- Experiment runner: action `run_cap:<family>` accepted by `POST /experiments/run` next to
  `vision.focus:<stream>`.
- `config/physiology/families.yaml`: `prediction.experiment` (tier 2, lanes []).
- New `config/physiology/run_cap_overrides.yaml`.
- No new bus channel. No new env key for Experiment 1 (the override is the point: not env).

## Files likely to touch (Experiment 1)

- `services/orion-prediction/app/experiment_runner.py`: run-cap actuator, decision-point loop with
  the pending hold, one experiment per night, stop rules, restricted arm sequence, bid before each
  decision. Key name and `recover()` cleanup by action (today `stream_of()` knows only
  `vision.focus`).
- `services/orion-prediction/app/causal.py`: blocks that start at an event (today
  `design_switchback` makes fixed clock blocks), a restricted-permutation test (today
  `analyze_switchback` permutes freely), pooling across nights.
- `services/orion-prediction/app/effects.py`: several outcome series per experiment (today
  `analyze_experiment` matches fixed block times and one `series_id`).
- `services/orion-prediction/app/series.py`: source kinds `pool_leases` and `kev_appraisal`.
- `services/orion-prediction/app/family_c.py`: lane-cost targets, manipulation check from the act
  ledger.
- `orion/physiology/binding.py`: `BINDING_PRODUCERS` gains `orion-prediction.experiment.binding`.
- `services/orion-embodiment/app/main.py`: `/health` field `town_start_eligible_at`.
- `services/orion-prediction/tests/`, `services/orion-prediction/evals/`: replay eval (Patch 0).
- `orion/embodiment/town_sessions.py`, `services/orion-embodiment/app/town_sessions.py`: read the
  override, `run_cap_override` refusal.
- `services/orion-embodiment/tests/`.
- `orion/physiology/run_cap_override.py` (new, pure: parse, bounds, TTL rules; reusable for the
  next knob), `config/physiology/run_cap_overrides.yaml` (new).
- `config/physiology/families.yaml`, `config/integration/organs.yaml`.
- `orion/schemas/physiology_run_cap.py`, `orion/schemas/registry.py`,
  `orion/schemas/temporal_self.py`, `services/orion-durable-runs/app/temporal_self_sources.py`,
  `orion/temporal_self/now.py` (`effect_z` rule).

## Non-goals

- No hardware lever in Experiment 1. No root, no `nvidia-smi`.
- No change to any tier-1 family, its reserved share, or the GPU pool's priorities.
- No price or bid multiplier on `aitown.session` until its bids bind.
- Never stopping a running town session.
- No new root helper (Experiment 2 extends the existing job).
- No experiment on `hub.outreach` (its effect is on Juniper; day-long blocks; later if ever).
- No auto-start: a run starts only by `POST /experiments/run`, as today.

## Acceptance checks

Patch 0 (replay, before anything is built to run live):

- `services/orion-prediction/evals/aitown_switchback_replay.py` recomputes, from the last 14 days,
  the per-block SDs of every outcome and the MDE at 56 blocks per arm, and prints them. Go only if
  the GPU3-watt MDE is <= 10 W.

Patch 1 (build):

- Unit: the controller refuses with `run_cap_override` under a valid key, with `retry_after_sec` no
  more than the key's TTL. It ignores a key with no TTL, a value not in `allowed`, a `set_at` older
  than `max_hold_s`, and a family not in the YAML.
- Unit: the pending hold is set 60 s before `town_start_eligible_at`, and an ON arm deletes it.
- Unit: the runner refuses a decision with tier-0/1 queued > 5 s, when hot, when its bid is
  deferred; never produces 3 OFF in a row; logs the arm before setting the key.
- Unit: the manipulation check marks a run `contaminated` when an OFF block has a session start.
- `scripts/check_integration_seam.py`, `test_family_lanes.py`, schema registry checks pass.

Live (after Juniper's yes, VERIFIED only with these):

- One OFF block: the key exists with a TTL, the controller logs `refusal=run_cap_override`, no
  `aitown.session` act opens in the block, and a session opens within 5 min after the key goes.
- Kill the runner mid-OFF block: the town starts a session within 90 s + one 300 s tick.
- After 16 nights: manipulation check passes; `GET /effects/table` shows
  `run_cap:aitown.session` rows for `circe.gpu3_watts` and `lane.fast.tier1_hold_s` with
  `method = randomised`; one `experiment_result` item appears in `GET /temporal-self/now`.

## Recommended next patch

Patch 0: the replay eval above. It needs no approval beyond this design (read-only) and decides
whether 16 nights is enough.

## Questions for Juniper (yes/no)

1. **Experiment 1 on AI Town**: yes/no to randomly holding Orion's town sessions for 40 min (never
   more than 2 holds in a row, never stopping a running session, same 2 h a day) for ~16 nights?
2. **New family `prediction.experiment` at tier 2**, with bids that bind (the experiment yields when
   the market defers it)?
3. **Experiment 2 later**: yes/no to reversing the 10-29 rule "gpu3 is never capped" for
   time-limited 150 W blocks, written by the lane controller as a second writer next to the PR, and
   refused while tier-0/1 work waits on `fast`?
4. **Lane cost as an outcome**: yes/no to counting tier-1 slowdown on `fast` (hold and wait) as an
   effect-table target next to the Family A readings?
