# Retire the proposal, policy and dispatch rungs (L7–L9): replace first, then switch off

Date: 2026-11-04. Status: proposal (docs only). This is step 2 of the ladder demolition in
`2026-11-04-attention-as-want-competition-design.md` (PR #3089). Step 1, feedback (L10) and
consolidation (L11), shipped in PR #3115 and is live. Juniper (11-04): "dispatch next", and
**every act the rung carries gets its replacement wired in the same change that turns it off.**
Nothing is switched off without a working replacement.

Every number was read live on 2026-11-04 (14-day window unless stated).

## In one paragraph

Proposal, policy and dispatch run as one pipeline: about 40,000 frames a day per rung, 18 GB of
frames with no retention on two of the tables. Most of it is boilerplate:
- summarize and inspect calls that only the ladder reads;
- docker prunes that decline 93% of the time ("nothing to reclaim").

But dispatch is also the **only trigger** for five real acts:
- retro play;
- painting;
- AI Town sessions;
- docker image pruning;
- the "pause background GPU when the cabinet warms" world act.

It also triggers about 3% of reading pulls. So the plan has three steps:
1. **Move each real act** to a trigger of its own, through the body-economy want market, one act
   per PR. In the same PR, the act's dispatch template comes out of the proposal policy, so the
   act never has two triggers and never has none.
2. **Retire the three services together,** once nothing real runs through them.
3. **Drop their tables** 2 days later, after a backup.

## What dispatch carries today

| Act | Trigger today | Doer | Other way to start | Sends / real result (14 d) | Want-ledger bid |
|---|---|---|---|---|---|
| Retro play | offered on every warranted tick (`action_warrant >= 0.5`); allocator `play` | Hub `retro_leisure_listener` → `retro_leisure.should_start` | **none** ("There is no scheduler here") | 2,314 sends → **4 sessions** (refused: session active 1,640, arousal strained 420, human recent 82, browser 80, cabinet hot 72) | `hub.retro_leisure`, shadow, placed *after* dispatch arrives |
| Painting | field `resource_pressure` (allocator) + visual-baseline schedule (5,400 s) | cortex-exec `render_scene` → orion-thought visual chain | **none live**: `ORION_VISUAL_CHAIN_ENABLED=false` | 185 sends → 125 painting acts (101 allocator, 24 baseline) | `thought.painting`, shadow |
| AI Town | offered every warranted tick | orion-embodiment `town_sessions` | Juniper's manual Unfreeze only | 43 sends → 22 sessions | `aitown.session`, shadow, placed after dispatch |
| Reading pull | offered every warranted tick | Hub `world_pulse_read` pull | **yes**: Hub's own read loop does 693 of 713 reads (97%) | 20 pulls | none separate |
| Docker prune (images, build cache, stopped containers) | field `resource_pressure` + `deviation_pressure` | cortex-exec `skills.runtime.*` | none automatic (hand-run skill catalogue) | 156 → 145 declined "nothing to reclaim" | `execution_dispatch.maintain` declared, 0 rows |
| Cooling world act (`shed_background_gpu`) | attention workspace winner on the cabinet/biometrics node + `self_shed` eligibility; 50% held back as a control group | gpu-pool `orion:gpu_pool:shed:request` (dispatch is its **only** sender) | the pool's thermal reflex is separate and stays | 3 sheds; 20 episodes total, last 10-25; scored in sql-writer since #3115 | none |
| Self-study analysis | offered every warranted tick | cortex-exec `skills.self_study.analyze.v1` | **curiosity now owns self-study** (`a41aa36ea`, 10-30) | 326 runs → about 52 journal entries, read as reverie/visual context | none |
| Summarize current state, inspect capability/node | field pressure dimensions | cortex `substrate.summarize` / `substrate.inspect` | — | 228 summarize ("The target is in a stable state…"), 41 inspect; read **only** by the ladder's own posteriors | none |
| Reverie → action (`spontaneous_thought_to_candidate`) | `ORION_REVERIE_PROPOSE_ENABLED=true` | proposal arena | — | **0** reverie-originated dispatches in 14 days | none |

## Hidden couplings outside the ladder (each needs its fix in the retirement PR)

1. **The retention floor.** sql-writer's `_substrate_chain_floor` still probes `policy_pending`
   and `dispatch_pending`. Both are DEFAULT TRUE markers that only these rungs clear. If one rung
   stops before the other, the floor pins and proposal pruning stops silently. If a probed table
   is dropped, the probe raises and the floor refuses to prune forever. **Fix:** remove both probes
   and the proposal-frame retention job in the retirement PR.
2. **The act ledger imports `self_shed` lazily** (`orion/grammar/act_ledger.py:129,168`).
   `self_shed` imports the proposal schema's `AttentionWinnerRefV1`. Deleting the schema would
   break act grading at call time, not at boot. **Fix:** retire the act ledger's workspace-winner
   binding (rung-3 binding: 0 attention winners bound in 24 h; `event_ref` in the want market
   replaces it). Move `AttentionWinnerRefV1` only if another reader remains (hardware-watch reads
   `self_shed`).
3. **The field digester imports `orion.proposals.scoring`** (`digestion/precision.py:17`,
   `orion/field/action_warrant.py:161`). **Fix:** move those functions into `orion/field/` before
   the package is deleted.
4. **mood_arc enrichment reads `action_warrant` from proposal frames**
   (`orion/mood_arc/corpus_enrichment.py:95`). **Fix:** read it from the field, which is where
   `action_warrant` is computed. Otherwise it goes empty silently.
5. **Temporal Self reads ladder-kind outcome rows** (`temporal_self_sources.py:128`). The rows
   simply stop; act rows continue. **Fix:** filter the source to act rows, so it never reads a
   kind that cannot arrive.
6. **The dream compaction policy gate** reads `PolicyDecisionFrameV1`. It is idle live
   (`ORION_DREAM_COMPACTION_APPLY_ENABLED=false`). **Fix:** remove the gate, and before anyone
   turns apply on, give it the body-economy consent gate instead (open question 6).
7. **The shared tables stay:** `substrate_action_outcomes`, `substrate_action_effect_posterior`
   (act ledger + world scorer) and `substrate_signal_control_cells`. Only the ladder-kind
   posterior rows go stale.
8. **The rest, as in the feedback burn:** Hub routes and lattice page (show L7–L9 "retired"),
   ladder liveness and metrics liveness, channels/registry/inner-state registry, the physiology
   contract, CI (`attend-act-loop`, `thinking-rung3`, `visual-baseline`), the env sync script,
   smoke scripts, `Makefile:510`, analysis scripts, and the two dispatch-backed evals.

## The plan: one switch-over PR per real act, then one retirement PR

Each switch-over PR does three things together:
- (a) builds the act's own trigger;
- (b) removes its template from `config/proposals/proposal_policy.v1.yaml`;
- (c) shows live that the act still runs at about today's rate.

The want-ledger bid moves *before* the act, so the market sees the want and not the aftermath. Bids
stay shadow, so the act's real gates decide, as they do today. Binding comes family by family, in
attention patch 3, in the order Juniper chose.

### PR 1. Retro play: a Hub timer replaces a 5-minute poller
- **What changes:** the Hub retro listener gets its own timer at dispatch's effective cadence
  (300 s). Each tick calls the existing `should_start` (the doer's own gates: arousal, cabinet hot,
  Juniper recently active, browser open, cooldown, daily cap, session active) and bids before it
  starts.
- **Removed:** the `retro_leisure_session` template, and the RPC channel pair once nothing sends
  on it.
- **Live check:** sessions started per day are within the 14-day range, about 2/week. Refusal
  reasons are the same ones.

### PR 2. Painting: orion-thought paints on its own rhythm
- **What changes:** orion-thought's visual chain starts itself. It runs the existing baseline
  schedule (5,400 s, arousal deferral) plus a hunger bid on `thought.painting`, keeping the thermal
  gate, the diffusion lock and the GPU-pool lease.
- **Rate:** the target is today's rate, about 9 paintings a day (open question 2). We do not just
  flip `ORION_VISUAL_CHAIN_ENABLED` at its 600 s default, which could paint far more.
- **Removed:** the `render_scene` template.
- **Live check:** paintings per day are within ±30% of the 14-day mean, and the act ledger still
  closes them.

### PR 3. AI Town: embodiment starts its own sessions
- **What changes:** `town_sessions` gets a timer that bids, then starts under its existing
  controller caps (7,200 s/day, 3,600 s gap, 1,800 s session). Juniper's manual Unfreeze stays.
- **Removed:** the `aitown_session` template.
- **Live check:** sessions per day are within the 14-day range.

### PR 4. Docker pruning: watch disk, prune when it is needed
- **What changes:** hardware-watch gets a disk-usage rule. At or above the image-prune skill's own
  threshold (≥75% / ≥40 GB), it runs `skills.runtime.image_prune.v1`. Build cache and stopped
  containers stay on the hand-run catalogue (open question 4).
- **Removed:** the three `prune_*` templates.
- **Live check:** the rule fires only above the threshold; replay it on 14 days of disk history.

### PR 5. The cooling world act: the market decides, the pool acts
- **What changes:** the decision moves from "dispatch saw an attention winner" to a want: "the
  cabinet is warming" is an unusual body signal, so it posts an **event want**
  (`family: self_regulate.shed`, `event_ref` = the cabinet reading, `salience` from
  `cabinet_heat_pressure`), with the same `self_shed` eligibility rules. A funded want sends the
  same `orion:gpu_pool:shed:request`.
- **What carries over:** the treated/control episode ledger (`world_episodes`, holdback 0.5) moves
  with it unchanged, so the sql-writer scorer keeps working.
- **Where it runs:** this is the first family that binds by design, because the act exists to be
  decided by the market (open question 1).
- **Kept as it is:** the pool's thermal reflex stays separate.
- **Removed:** the `shed_background_gpu` template and its dispatch path.
- **Live check:** the replay over 14 days of cabinet history gives about the same eligible count.
  The next real shed produces a `[WORLD_SETTLEMENT] scored=` line.

### No replacement needed (removed in the retirement PR, with the reason stated)
- **Reading pull:** Hub's own loop already does 97% of reads. Its small treated/control experiment
  (20 treated, 6 control) ends. Reading findings become event wants in attention patch 2.
- **Self-study analysis:** curiosity owns self-study since 10-30. **Check before removal:**
  reverie/visual context still gets self-study text from curiosity's output. If it does not, PR 6
  repoints that context read first.
- **Summarize current state, inspect capability/node:** nothing outside the ladder reads them.
  They are boilerplate.
- **Reverie → action:** 0 dispatches in 14 days. Its successor is reverie posting event wants
  (attention patch 2+).
- **Policy's job** (risk, consent, reversibility): today nothing reaches policy except these
  templates. Its checks become the market's one consent gate, built with binding in attention
  patch 3.

### PR 6. Retire proposal, policy and dispatch
- Delete the three services and their packages, keeping the shared pieces:
  - the allocator's math, if a market caller needs it (today only dispatch calls it, so it goes);
  - `act_episodes`, if PR 5 keeps a reader;
  - `prediction`, `contrast` and `world_episodes`.
- Apply every coupling fix above.
- The Hub lattice page and the ladder check show L7–L9 as "retired". The sensing half (M1–M5)
  stays checked.
- Run the same no-orphans sweep as #3115: env keys, CI, contracts, docs.

### PR 7 (2 days after PR 6 deploys). Drop the tables, after a backup
The tables are `substrate_proposal_frames` (2.3 GB), `substrate_policy_decision_frames` (7.4 GB),
`substrate_execution_dispatch_frames` (11 GB), `substrate_execution_dispatch_inflight` and
`substrate_dispatch_results` (514 MB). This is a hand-applied DESTRUCTIVE migration with a
`pg_dump` first, merged in the same sitting, as with #3116.

## Missing questions (defaults in brackets)

1. **Cooling act: does it bind in the market from day one?** [Yes. It is a single, reversible,
   capped act; the 50% control group stays; the reflex stays separate.]
2. **Painting rate after the move?** [Today's rate, about 9 a day, with the baseline schedule plus
   a hunger bid. The market can raise or lower it later when painting binds.]
3. **Reading pull's treated/control experiment:** keep it somewhere, or end it? [End it. The read
   loop runs anyway, and patch 2 makes reading an event source.]
4. **Pruning:** hardware-watch rule for images only, or for all three prunes? [Images only. 145 of
   156 runs found nothing; the others stay hand-run.]
5. **Order of the switch-over PRs?** [Cooling act (5) first, because it is the most important act;
   then painting (2), retro (1), AI Town (3), pruning (4); then retire (6).]
6. **Dream compaction gate:** when apply is turned on one day, what approves a delta? [The
   market's consent gate from attention patch 3. Apply stays off until then.]

## Proposal-mode checklist

- **Capability that changes:** each of the five acts starts itself through the want market, not
  through the ladder. The ladder's boilerplate thinking stops.
- **Data touched:** new want-ledger rows (shadow, except the cooling act). The ladder's tables stop
  growing and are dropped after a backup.
- **Privacy boundary:** none.
- **Trace that proves it worked:** for each switch-over PR, the act's live rate per day matches its
  14-day range, with the act ledger closing the acts, and the retired template sends 0 times.
  After PR 6: the ladder check is green, with L7–L9 retired; proposal pruning and the act ledger
  are unchanged.
- **Dangerous failure modes:**
  1. An act loses its only trigger. Mitigation: the same PR removes the template and adds the
     trigger, with a live rate check.
  2. An act gets two triggers (double painting). Mitigation: the same PR does both.
  3. The retention floor pins. Mitigation: the probes come out in PR 6, before any table drop.
  4. The cooling act stops silently. Mitigation: a replay, the next real shed scored, and the
     scorer's hourly "alive" line.
- **Disable or roll back:** each switch-over PR reverts alone, which puts its template back.
  PR 6 reverts as one; the tables are kept until PR 7.

## Acceptance checks

1. After each switch-over PR, a live week shows the act's rate in its 14-day range, and the
   template's dispatch count is 0.
2. After PR 6: 0 frames written to the three tables; `make substrate-ladder-check` is GREEN with
   L7–L11 retired; the oldest proposal row stops changing (no retention job left); act-ledger
   outcomes per day are within ±25%.
3. The cooling act: a 14-day replay of its event wants, and one real shed scored after PR 5.
4. A repo-wide grep finds no reader of the three schemas, tables or channels outside history and
   migrations.

## Recommended next patch

PR 5, the cooling world act: it is the most important act, and dispatch is its only sender.
