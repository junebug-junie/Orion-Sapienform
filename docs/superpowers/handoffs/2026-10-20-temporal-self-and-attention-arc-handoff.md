# Handoff: Temporal Self, attention and reverie arc (2026-10-20)

This file hands the arc to a new agent session. Read it from top to bottom before you do anything.

The arc started as the substrate lattice audit. Next it became Temporal Self rev 4 (PR #2369: sense, remember over time, regulate). Then it moved to "garbage in, garbage out": Orion's attention fed reverie a bare label, and reverie repeated it four times. Most of the work is live. This file lists what is live, what is verified, and what is still open.

## 1. Standing rules from Juniper (keep these)

- Juniper merges. The agent deploys and verifies ("you handle the deploys").
- Deploy only from the clean primary checkout on `main`. Use `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh <svc> up -d --build`.
- A `git pull` on primary main runs the post-pull hook. That hook rebuilds every service changed on main, so a pull is a deploy. Expect this.
- Ship flags ON (settings default and `.env_example`).
- Never use `git stash`. Never run `sudo`; print the command for Juniper instead. Give operator commands on one line.
- Changes to cognition, attention, memory or self-model start as a proposal unless Juniper asks to build.
- Enforce evidence with a mechanism. Do not list "hallucination" as a risk. Orion may talk about its own plumbing. Do not add word filters or banned-phrase lists.
- The unified turn (`stance_react`) is the only production chat path. Do not wire into `chat_general`. `daily_metacog_v1` stays off.
- Use plain English, in the spirit of ASD-STE100: say what a thing does first, and give identifiers after.
- Parked: rest-drive tiredness must not feed the field/lattice until Juniper says so.
- No keyword cathedrals. Run the metric quality gate (CLAUDE.md) for every new signal.

## 2. What is live and verified

| Area | PRs | Live evidence |
|---|---|---|
| Focus runs, replays, rest drive, arousal, chronology | #2555 #2576 #2584 #2594 #2597 #2605 | durable-runs `temporal_self` thread, `orion:regulation:latest` |
| Arousal readers, GPU fault vs load | #2624 #2628 #2635-2637 | readers log `arousal_view` |
| Stance cue, day citations, letter cites arcs | #2616 #2622 #2623 #2631 | `day_citations` rows labelled verified |
| proposal-runtime `requests` fix | #2639 | 0 tick failures |
| Immune sweep and rest ranges | #2643 | sweeps run while idle, 0 findings so far |
| Hub Temporal Self tab | #2657 | routes return 200. Browser render UNVERIFIED |
| Dream replays closed arcs | #2658 | sleep at 10-14 04:43 replayed 3 arcs |
| Curiosity kickoff "today so far" | #2659 | not yet seen in a live kickoff |
| Reverie "earlier today" block, daydreams grouped | #2660 #2673 | verified in container on the live frame |
| Camera company arcs (S1) | #2665 | rows land. No company arc seen yet |
| Reverie honest input + train of thought (other session) | #2707 | after deploy: 0 "The coalition" openers, `next_focus` filled, 0 "unattended for" |
| atlas retired as a host | #2714 | atlas gone from both projections, 0 atlas emissions |
| Sustained rule for continuous body signals | #2718 | last 72 h: bus_synaptic 1.0%, biometrics 0.6%, no winner 81.2% |

Last 72 h attention winners (substrate_attention_broadcast_log): none 81.2%, execution 4.0%, perception 3.7%, world:chat 3.4%, cabinet 2.6%, route 1.7%, chat 1.6%, bus_synaptic 1.0%, biometrics 0.6%, codebase 0.2%.

## 3. Check first (possible regressions)

0. **Arousal has been stuck on "strained" since 2026-10-18 12:07 UTC (found 10-20 22:00).** While it reads strained, the arousal readers block curiosity (`curiosity_investigation_blocked reason=arousal_strained`, `curiosity_self_inquiry_blocked`). The last curiosity run finished 10-18 05:10. This probably also explains part of the reverie volume drop (reverie chain is arousal reader 5). The reason is `gpu_queue_sustained`, from `gpu_pool_state_history` (circe). From 10-18 ~12:00 the agent lane holds 3-10 waiting requests in every hour (10 at 22:00 on 10-20), and `memory_distill` holds 2. At the same time `gpu_pool_events` shows a tight loop: holder `vision-council`, work_class `metacog`, about 900 per hour, admitted -> unavailable (`shed:cabinet_hot`) -> cancelled. Note that the arousal reading has `cabinet_temp_c` 32.26 and `cabinet_reflex` null, so the pool's "cabinet hot" shed and arousal's cabinet view disagree. Questions:
   - Is the agent-lane queue real load or jobs that can never be served?
   - Why does the pool shed for cabinet_hot at 32 C?
   - Why does vision-council retry with no backoff?
   - Should `shed:*` / `unavailable` count as a fault in `GPU_FAULT_EVENTS` (`orion/regulation/arousal.py:62`) so that a jammed queue is not read as strain? This is the same fault-vs-load seam as #2628.
1. **Reverie volume is falling.** Thoughts per day: 10-15 620, 10-16 314, 10-17 205, 10-18 118, 10-19 55, 10-20 24 (partial). Some drop is expected: calm ticks no longer start a chain (#2707), chains stop when settled, and the sustained rule (#2718) removed most body and bus winners. A steady fall to near zero is not explained by that. Find out whether reverie is starving or broken. Compare reverie skips (`no winner this tick`), chain terminal reasons in `substrate_reverie_chain`, and winners per day. Perception winners dropped from about 8% to 3.7%, so check perception too.
2. **Daydream steps still paraphrase.** After #2707 a step names `next_focus` ("investigate the specific nodes…"), but the next step gets no new information, so it restates. The next lever: give the step the real "what changed" detail for its winner (which pressure hint moved, which bus links went anomalous, the caption difference for perception). Measure with `services/orion-thought/evals/` (the #2707 continuity eval): first-to-last bigram overlap must fall below the same-subject cross-chain baseline.
3. **Cabinet wins 2.6%.** Cabinet is the room sensors, and the prompt labels it "body unusual for itself". See open decision 4.2.

## 4. Open decisions for Juniper

1. **World candidates for attention (PR #2722, docs, open).** Recommendation: add `world:curiosity`. A finished curiosity run competes for 300 s, but only when its belief change (`realized_nats` in `curiosity_run_outcomes`) is in the top 10% of its week. It carries its question, the priors it moved and a finding excerpt. It yields to `world:chat`. Replay: about 2 wins a day. Reading, paintings and activity rates failed the metric gate (reasons are in the doc). Second patch: attach Juniper's latest message as evidence on `world:chat` and remove the message count from its label. The count changes the loop id every message, which is a live bug. **Waiting for Juniper's go.**
2. **Room sensors inside biometrics.** Six `cabinet_*` keys from athena (proximity, vibration, audio…) sit in the machine-body mean. Taking them out changes what biometrics measures, so it needs Juniper's yes.
3. **Dream replay favours interoception arcs.** A body arc runs for hours and its "returns" count sensor flicker, so it wins every time. Proposed fix: at most one arc per kind per sleep. #2718 may already reduce this, because fewer body arcs open now. Re-check one dream before building.
4. **LangGraph checkpoint growth.** During the nightly `pg_dump`, a checkpoint read took more than 80 s, and the chronicle stepped every 6 min, not 2 min. `orion:regulation:latest` has a 360 s TTL, so readers could see "unknown". Proposed: prune old checkpoints and check the index. This is housekeeping only.
5. **Close PR #2705.** It is the design doc for #2718, which is now live. Merge it or close it.

## 5. Small open items

- **Vision-split migration header.** `services/orion-sql-db/manual_migration_vision_split_v1.sql` says to apply it with `scripts/undo_vision_split.py`, but that script undoes splits. The table exists live now, so someone applied it. Correct the header so the next reader is not misled.
- **FIXED in PR #2809 (verify after deploy):** stuck pressure and hecate standby. Check for `biometrics_silent_nodes excluded=hecate`, receipts with `definition_version: 2`, hecate `pressure_score` 0. Expect biometrics to win attention 1.8-3.8% (replay) until the 7-day history refills.
- **(was) Suppression clean-up loop** (found in #2714). `node_pressure_suppressed` maps to `strain` only, so `availability` on a stale expected-offline node never clears, and frozen hints re-add `strain`. atlas was the only live case. Fix it in `pressure_reducer.py` / `pressure_organ.py`.
- **hecate frozen keys.** While hecate is off (on-demand), its 9 frozen keys water down the biometrics mean. Should the mean skip stale nodes?
- **Old atlas names on circe.** `docker-compose.atlas-workers.yml`, `ATLAS_*` env keys, `orion-atlas-llamacpp-*` containers, `gpu_pool.yaml` / `llm_profiles.yaml` keys. Rename them in their own patch, with restarts on circe. `services/orion-bus/docker-compose.atlas-standby.yml` has no deployment target. Ask Juniper whether to delete it.
- **`REDIS_PROJECTION_EXCLUDE`.** Clear it now: dream, thought and proposal-runtime have all been rebuilt since #2643.
- **Curiosity runs over about 35 min.** They may miss arcs (`LATE_PROBE` 1800 s, no curiosity override). A subagent's read-only query on this was permission-denied, so Juniper must decide whether to run it.
- **`self_inquiry` runs.** 53 runs a week write no `curiosity_run_outcomes` row, so they cannot compete under #2722.
- **Optional:** remove `daily_metacog_v1` completely. Log when a turn skips chat history (`no_write`).

## 6. Verifications still owed (not yet observed live)

- The first real company arc and its stance-cue clause ("Someone has been here since…").
- The curiosity "TODAY SO FAR" line in a kickoff (`curiosity_today_fact_offered` in Hub logs).
- The first real `immune_finding` row.
- A live `arousal_strained` refusal by a reader.
- Hub Temporal Self tab in a browser.
- The R1a habituation decision: re-run `services/orion-attention-runtime/evals/replay_focus_runs.py` from 10-10 06:00. The session cron for this was session-only, so assume it did not run.
- Arousal threshold re-fit, due about 10-24.

## 7. Where things are

- Memory: `project_temporal_self_rev4_regulation_pr2369_2026-10-09.md`, `project_reverie_coalition_garbage_in_audit_2026-10-14.md` in `~/.claude/projects/-mnt-scripts-Orion-Sapienform/memory/`.
- Specs: `docs/superpowers/specs/2026-10-16-attention-continuous-signal-quota-design.md`, `…-attention-world-candidates-design.md` (PR #2722 branch), `2026-07-14-reverie-narration-continuity-design.md`.
- Replay scripts: `scripts/analysis/replay_attention_sustained_body.py`, `replay_attention_world_candidates.py` (PR #2722 branch).
- Key code: `orion/attention/world_first.py`, `orion/attention/sustained.py`, `orion/substrate/attention_broadcast.py`, `services/orion-thought/app/{reverie,chain,store}.py`, `orion/cognition/prompts/reverie_narrate.j2`, `services/orion-durable-runs/app/temporal_self_*.py`.
- Live DB, read-only: `docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "<SELECT>"`.
- Worktrees from this arc are merged and can be pruned: `make prune-merged-worktrees` (dry run first).

## 8. Suggested first move for the new session

1. Run check 3.0 (stuck strained arousal blocks curiosity), then 3.1 (reverie volume).
2. Ask Juniper for a go on #2722 (`world:curiosity`). If she says go, build it, then build the `world:chat` message-evidence patch.
3. Then do check 3.2 ("what changed" detail for reverie steps), as a proposal first, because it changes what reverie is given.
