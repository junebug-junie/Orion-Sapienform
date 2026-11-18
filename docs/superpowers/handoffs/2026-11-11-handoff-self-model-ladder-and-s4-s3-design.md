# Handoff 2026-11-11: self-model ladder done; next = S4 -> S3 design (forecast steers control)

From the session that ran Self Atlas retirement, rung 4 claims, and PR field notes (11-03..11-11).
Juniper asked for a fresh agent to write the **S4 -> S3 design spec** (design mode, proposal mode:
this changes how Orion acts on their own). Plain English to Juniper; see CLAUDE.md section 21.

## What is done (all merged + deployed + verified live)

| Piece | PRs | Live evidence |
|---|---|---|
| Self Atlas Patch A: self-inquiry points at the dig kit first | #3049, #3076 (usage report counts batch-tool dig calls) | 7-day uptake 11-11: 20/51 self-inquiry runs used dig (was ~1/7 on 11-03) |
| Self Atlas Patch B: retired (scan, topic-foundry runs, cluster writer, self graph wiring, `?graph=self`) | #3052, deployed 11-11 ~04:20Z | `/api/substrate/concepts/network?graph=self` -> 400 `unknown_graph` (valid: aitown, orion) |
| Self graph archive | `scripts/archive_self_atlas_graph.py --apply` (Juniper ran 11-11) | `orion_substrate_self_archive_2026_11` = 24,085 nodes = original. **Deleting `orion_substrate_self` is NOT approved yet** (ask Juniper; command `redis-cli -u redis://127.0.0.1:6380 GRAPH.DELETE orion_substrate_self`) |
| Rung 4 capacities: Orion writes claims by default | #3096 | first 5 claims 11-04 05:29Z |
| Rung 4: code checks claims against live atlas wiring (Juniper does not hand-judge; may veto) | #3107 (other session) + #3109 on top | `dig capacities`: 6 claims, all EVIDENCE-CHECKED, 0 Juniper-confirmed |
| Rung 5 portrait | #3148 design, #3150 patch 1 (other session) | `python -P -m orion.dig portrait` prints "I am" + "I cannot yet" |
| PR field notes (Orion notes every merged PR) | design (spec on main via #3100), #3100 patch 1, #3111 patch 2, #3114 turn budget fix, #3149 patch 3 | 7-day check 11-11: 69/69 merged PRs 11-04..11-10 recorded (67 notes, 2 code records); truth eval `--live` PASS |
| Daily reminiscence | #3163, #3167 (other session) | live; **#3187 open** (quiet weeks fix) |

Juniper decisions this arc (keep): field notes tier 1, 24 h window, no backfill, trivial PRs get code
records, rewrites keep old note; capacity claims are checked by code, Juniper only vetoes; S5 stays
shared with Juniper (only she can confirm S5).

## What is left on the arc

1. **S4 -> S3 design spec (the ask).** Below.
2. Portrait Patch 2 (`portrait history`, compare "I am" versions) — small, spec'd in
   `docs/superpowers/specs/2026-11-06-self-portrait-design.md` "Recommended next patch".
3. #3187 reminiscence fix awaiting merge.
4. Self graph delete — Juniper's yes needed.
5. Other portrait gaps (S2 organ lines, S3 9 bus-silent services, S5 policy-from-self) — later.

## The S4 -> S3 gap (what the design must close)

The live portrait/`dig capacities gaps` line (atlas 11-11, rules v2):

> S4: what I see ahead and what I do now do not argue both ways: no bus, HTTP or Redis link
> between orion-world-pulse and orion-gpu-lane-controller

Plain meaning: Orion forecasts/reads ahead, and Orion controls the body, but nothing ahead ever
changes what the controls do, and control never asks the forecast anything. VSM (Beer) says S4
(intelligence: outside + future) and S3 (control: inside + now) must be in two-way argument.

How the gap is computed: `orion/body_atlas/capacities.py` (`s4_s3_links`, ~L542; gap at ~L676;
rule doc ~L54). It clears only when some S4 part and some S3 part have a link **each way**
(non-census channel, HTTP call, or a Redis key one writes and the other reads). The parts held now
are Orion's evidence-checked claims, not rules:
- S4 = `orion-world-pulse` (sends `orion:world_context:daily_capsule` to orion-cortex-exec; reads the outside into claims/situations)
- S3 = `orion-gpu-lane-controller` (actuates GPU lanes; sends `orion:gpu_pool:actuate:result`)
- S3* = `orion-equilibrium-service` (reads `orion:system:health` from ~77 parts)
- algedonic = `orion-durable-runs` (`redis:orion:regulation:latest`, `redis:orion:self_monitor:latest`)
- metabolism = `orion-energy`

**Do not design to the metric.** Adding a channel between those two services would clear the gap
line without any behaviour change — that is the keyword-cathedral failure. The design must make a
forecast *change an action* and an action outcome *update the forecast*, with a trace proving it.
The gap clearing is a side effect, not the goal. Also: the S3/S4 parts may legitimately be
different services (e.g. orion-prediction as S4); if so Orion re-claims, and the code check decides.

## Facts the designer needs (verify before relying on them)

- **Forecasts live in `services/orion-prediction`** (standing rule: new bets/forecasts = a new model
  there, not a side ledger). world-pulse is outside-news, not a numeric forecaster.
- **Control seams that act on the body today:** gpu-pool leases/shed (`orion:gpu_pool:shed:*`,
  now requested by durable-runs since the L7-L11 dispatch ladder was retired 11-07, PR #3121/#3108);
  the cooling act `services/orion-durable-runs/app/cooling_act.py` (pause background GPU when the
  cabinet warms — the one real world act kept from the retired ladder); durable-runs regulation tick
  (strain/arousal; strain vetoes on curiosity retired 11-07 #3159; heat checks parked till spring).
- **Body economy** (`config/physiology/families.yaml`, `want_ledger`, shadow prices; design #2855):
  bids are still SHADOW (recorded, not obeyed). A forecast could enter as a price signal (e.g.
  expected heat/load raises the price of GPU-heavy acts ahead of time). Tier changes are Juniper's call.
- **Cost truth:** Orion's idle body is Orion's cost; use whole-machine chassis watts above pinned
  rest (`config/physiology/rest_body.yaml`: circe 541 W, athena 340 W, hecate 96 W), never GPU-only.
  AC: activity is charged its share of compressor-on heat.
- **Candidate concrete loops** (pick one, smallest that is real):
  1. Heat/load forecast (orion-prediction; inputs cabinet temp, closet SCD40 temp, office env, AC,
     scheduled durable runs) -> S3 schedules/defers GPU-heavy runs *before* heat, not after.
     Outcome (did the cabinet stay cool / was work done) settles the forecast -> calibration.
  2. Demand forecast (expected chat/curiosity/field-note load next hours) -> lane controller
     pre-warms or releases lanes. Outcome = wait times / cold spawns (cold claude spawn ~58 s seen 11-04).
  Option 1 reuses the most existing parts (cooling act, regulation, prediction) and has a clean
  settle signal; recommend it unless the data says otherwise.
- **Metric gate applies** to any new forecast/signal (CLAUDE.md "Metric quality gate", steps 1-6),
  incl. the live-data rest-state check. Record it in the spec.
- **Proposal mode required** (autonomy change): name capability change, data touched, privacy
  boundary, proving trace, dangerous failure mode (e.g. forecast starves Juniper's chat lane, or a
  bad forecast pins heavy work off forever), and kill switch/rollback. Ship flags ON per standing rule,
  but start the actuation in shadow (log what it *would* do) if the design calls for it — say which.

## Process notes

- Worktrees only; deploy only from clean primary main. Never git stash, never --no-verify.
- Juniper is overloaded: give her one recommendation + at most ~4 yes/no decisions, plain English.
- Concurrent sessions touch `orion/body_atlas` and `orion/dig` often — check open PRs first
  (`gh pr list`, `graphify prs --conflicts`); when merging main into a branch that touched
  `config/metrics/metric_definitions.lock.json`, regenerate it from main's copy
  (`git checkout origin/main -- <lock>` then `scripts/check_definition_drift.py --update`), or the
  static gate fails with other PRs' deltas.
- Output: design doc in `docs/superpowers/specs/` using the CLAUDE.md design-mode headings, as a
  docs-only PR; no implementation until Juniper says go.
