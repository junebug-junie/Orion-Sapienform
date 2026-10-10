# Attention faces the world first; the body interrupts only when it is unusual for itself

## Summary

Until now Orion's attention always looked inward and always crowned a winner. The main contest rescaled five hardcoded prediction-error nodes so the top one read 1.0 on every frame, even when every one of them was calm. This patch makes three changes:

- **The world competes by default.** World sources are Juniper's chat activity and the camera's surprise.
- **The body interrupts only when unusual for itself.** A body signal competes only when its prediction error is high or unusual against its own last 7 days.
- **A calm tick can have no winner.**

What is in the patch:

- **New candidate shape.** `AttentionCandidateV1` is one shape for anything that can compete. It carries:
  - internal (body) or external (world);
  - how unusual the reading is against that source's own week (the existing `PredictionErrorMagnitudeV1`, reused as-is);
  - an `absent` flag, so silence is never read as calm.
- **The ranking rule** (`orion/attention/world_first.py`):
  - The world is eligible when it is busier than its own usual.
  - The body is eligible only at band high or unusual, in the bad direction. That direction comes from the glossary semantic layer, never invented here.
  - Eligible candidates rank by their own percentile.
  - An empty eligible set is an explicit no-winner frame.
- **Both background contests are wired.** The field contest (attention-runtime) and the substrate broadcast (substrate-runtime) run the rule. The field contest drops the hardcoded five-node list and the min-max rescale that guaranteed a winner.
- **Chat is the first world source.** It counts Juniper's turns in the last 15 minutes, against the same count over her last 7 days. Camera surprise (`node:substrate.perception`) moved from body to world.
- **Downstream fixes in the same patch:**
  - self-modification proposals and goals only ever bind to a body target;
  - an empty coalition no longer "activates";
  - reverie no longer narrates an empty coalition;
  - the self-model no longer says "'None' selected";
  - curiosity skips world-only loops.
- **Flag.** `ATTENTION_WORLD_FIRST_ENABLED` ships on. `false` reproduces the old ranking byte for byte, checked against golden output generated from origin/main.

## Outcome moved

Offline replay of the last 3 days (read-only, `services/orion-attention-runtime/evals/replay_world_first.py`, 1-minute grid, 4,321 ticks, window 2026-10-07 03:44 to 2026-10-10 03:44 UTC):

| | Old ranking (122,963 stored frames) | World-first (replay) |
|---|---|---|
| Frames whose winner is a calm body signal (raw error < 0.05) | **40.6%** | **0.28%** |
| Frames with no winner | 0% | **32.9%** |
| World winners (chat + camera) | 0% | 24.9% (camera 24.0%, chat 0.9%) |
| Body winners | 100% | 42.2% |

The spec quoted 49% calm winners. That figure came from an earlier 3-day window. This window measures 40.6% from the stored frames.

**Real body spikes still win:**

- **Biometrics surges ≥ 0.25 in the window:** 108 of 110 won the tick right after they landed (band "unusual", percentile ≥ 0.998).
- **Execution readings ≥ 0.9:** 51 of 86 won. The other 35 were eligible but outranked:
  - The execution signal saturates at 1.0 (3 sigma), and 4% of its week sits at exactly 1.0. So a maximal execution reading is only at about the 96th percentile ("high", not "unusual").
  - A camera blip or bus reading above the 96th percentile of its own week ranks above it. See Risks.
- **Synthetic 5-hour storm.** This is a stand-in for the 2026-09-28 timeout-storm class, which predates the stored PE history (that history starts 2026-10-03). The storm injects execution readings of 1.0 every 135 s, replacing the real readings in its window. STORM_RESULT

## Current architecture

Three contests, all inward-facing, all crowning a winner (spec, "The attention contests"):

- **Field attention** (`services/orion-attention-runtime`). The candidates were 5 hardcoded PE nodes. Scoring was precision-weighted error followed by **min-max across the five**, so the top one was always 1.0. Host and capability novelty targets sat alongside.
- **Substrate broadcast** (`orion/substrate/attention_broadcast.py`). Any graph node with `dynamic_pressure` at or above the floor competed, Borda-ranked. Magnitude was attached to loops but "never read by scoring".
- **Per-turn chat frame.** Three detectors with hardcoded constant saliences.

No contest had a world input. The camera was excluded from the field contest.

## Architecture touched

**Contracts:**
- `orion/schemas/attention_candidate.py`, new. Registered in both registries and verified with `resolve("AttentionCandidateV1")`.
- `scripts/check_inner_state_registry.py`: AttentionCandidateV1 is covered as a sub-object of the two frame entries.

**Shared logic:**
- `orion/attention/world_first.py`: the rule, the chat source, and polarity from the glossary.
- `orion/attention/pe_history_cache.py`: attention-runtime's 7-day window over `substrate_node_prediction_error_history`.

**Field contest:**
- `orion/attention/field_attention/builder.py`: `_build_world_first_frame`.
- `orion/attention/field_attention/selectors.py`: `world_first_targets`, `field_target_source_kind`.
- `services/orion-attention-runtime/app/worker.py`: candidates.
- `services/orion-attention-runtime/app/store.py`: two read-only queries.

**Broadcast:**
- `orion/substrate/attention_broadcast.py`: `world_first_signals`, `_rank_loops_world_first`, and the hysteresis fix.
- `services/orion-substrate-runtime/app/worker.py`: the chat candidate.
- `services/orion-substrate-runtime/app/store.py`: the chat query.

**Blast radius:**
- `orion/proposals/builder.py`
- `orion/attention/field_attention/goal_provenance.py`
- `orion/substrate/attention_self_model.py`
- `services/orion-thought/app/reverie.py`
- `orion/substrate/endogenous_curiosity.py`

**Per-turn detectors.** `orion/substrate/attention/detectors/*`: the constants are named, each signal is tagged with `source_kind`, and the "why they stay" note is in `base.py`.

## Files changed

- `orion/schemas/attention_candidate.py`: the candidate contract.
- `orion/schemas/registry.py`: registration in both maps.
- `orion/attention/world_first.py`: ranking rule, chat source, semantic-layer polarity, flag.
- `orion/attention/pe_history_cache.py`: incremental 7-day PE history window for attention-runtime.
- `orion/attention/field_attention/builder.py`, `selectors.py`: world-first field frame. The flag-off path is untouched.
- `orion/attention/field_attention/goal_provenance.py`: goals only name internal `node:substrate.*` targets.
- `orion/proposals/builder.py`: self-modification proposals skip external winners.
- `orion/substrate/attention_broadcast.py`:
  - world-first broadcast;
  - an empty coalition never activates;
  - updated docstring ("always one winner" and "magnitude never read" now describe flag-off only).
- `orion/substrate/attention_self_model.py`: a no-winner tick gets a no-winner narrative.
- `orion/substrate/endogenous_curiosity.py`: a world-only loop seeds no `concept_expand`.
- `orion/substrate/attention/detectors/{base,current_turn,concept_induction,situation}.py`: named rank priors, `source_kind` provenance, and the audit note.
- `services/orion-attention-runtime/{app/worker.py,app/store.py,app/settings.py,.env_example,docker-compose.yml,Dockerfile}`: wiring, the flag, and the glossary copied into the image.
- `services/orion-substrate-runtime/{app/worker.py,app/store.py,app/settings.py,.env_example,docker-compose.yml,Dockerfile}`: same.
- `services/orion-thought/app/reverie.py`: skip a no-winner tick.
- `scripts/check_inner_state_registry.py`: covered-name entry.
- `services/orion-attention-runtime/evals/replay_world_first.py` and `test_world_first_replay.py`: the replay eval.
- Tests (all new except where noted):
  - `tests/test_attention_world_first.py`
  - `tests/test_attention_world_first_field_frame.py`
  - `tests/test_attention_world_first_parity.py` with `tests/fixtures/world_first_parity/`
  - `tests/test_attention_pe_history_cache.py`
  - `orion/substrate/tests/test_attention_broadcast_world_first.py`
  - `services/orion-attention-runtime/tests/test_world_first_worker.py`
  - `services/orion-substrate-runtime/tests/test_worker_world_first_broadcast.py`
  - `services/orion-thought/tests/test_reverie_no_winner_skip.py`
  - additions to `orion/substrate/tests/test_attention_self_model.py` and `test_endogenous_curiosity.py`
  - three existing worker tests pinned to flag-off.

## Schema / bus / API changes

- **Added:** `AttentionCandidateV1`, registered and not on the bus.
- **Removed / Renamed:** none.
- **Behavior changed:**
  - With the flag on, both contests rank world-first and may produce no winner.
  - A field frame can carry a `world:chat` target with kind `channel`, and has `warnings=["world_first_no_winner"]` when empty.
  - A broadcast loop can carry `source_refs=["world:chat"]`.
- **Compatibility: no `extra="forbid"` model changed.** This is step 1 of the spec's rollout. `source_kind` rides in existing free-form fields:
  - `FieldAttentionTargetV1.evidence_refs` (`source_kind:internal|external`);
  - `AttentionSignalV1` / `OpenLoopV1.provenance["source_kind"]`;
  - `AttentionFrameV1.debug["world_first"]`, which holds every candidate's verdict, band, percentile and absent state.

  Every consumer validates the new frames unchanged, so there is no consumer-first ordering. Typed fields later on `FieldAttentionTargetV1` / `OpenLoopV1` / `AttentionFrameV1` would be consumer-first (deploy readers before producers). That is not done here.

## Env/config changes

- **Added keys:** `ATTENTION_WORLD_FIRST_ENABLED=true` in orion-attention-runtime and orion-substrate-runtime. Each is set in `.env_example`, `settings.py` and compose.
- **Removed / renamed keys:** none.
- **`.env_example` updated:** yes (both services).
- **Local `.env` synced:** yes, with `python scripts/sync_local_env_from_example.py --all-keys orion-attention-runtime orion-substrate-runtime`, run from the worktree. It wrote the primary checkout's `.env` for both services: `+ATTENTION_WORLD_FIRST_ENABLED='true'`.
- **Skipped keys requiring operator action:** none.
- **Dockerfiles** for both services now `COPY config/field`. The glossary is read for each node's value_kind and polarity. Verified inside both built images: `node_prediction_error_semantics('node:substrate.cabinet') == ('trigger', None)`.

## Metric quality gate: chat activity (new world source)

Gate run 2026-10-10 on live data.

**1. Provenance**
- The source is `chat_history_log` rows with `source LIKE 'hub%'` and a non-empty prompt. These are Juniper's Hub turns. Orion-initiated rows (outreach, metacog background) have no source and an empty prompt.
- The query is `orion/attention/world_first.py:CHAT_TURN_TIMES_SQL`, shared by both contests.
- The value is the number of turns in the trailing 15 minutes (`chat_rate_magnitude`).
- `created_at` is naive UTC (database timezone is `Etc/UTC`, checked). It is written after the reply, so a turn registers about one reply latency late.

**2. Independence from `node:substrate.chat` prediction error** (the chat load and repair-pressure movement z-score):
- Only 4.1% of chat-PE readings (566 in 7 days) fall within 15 minutes after a Juniper turn.
- 1,619 grid minutes had a nonzero chat PE with no Juniper turn nearby. Chat PE mostly fires on Orion-side turns.
- Pearson correlation between the 15-minute rate and the max chat PE in the same window (5-minute grid) is **0.13**.
- They are not the same signal. Chat PE stays internal.

**3. Theory anchor**
- The orienting response (Sokolov 1963): an organism orients to a stimulus in proportion to how much it departs from its habituated model of that channel.
- An event rate scored as a percentile against the channel's own week is a non-parametric form of "departure from habituated expectation".
- This anchors "busier than usual". It does not anchor "important".

**4. Live sanity, including rest**
- 28 Juniper turns in 7 days (47 in 14).
- On the minute grid the windowed count is 0 for 97.3% of minutes, 1 for 1.7%, 2 for 0.7% and 3 for 0.4%.
- **Rest is reachable and exact:** with nobody talking the value is 0, percentile 0.0, band quiet. Live right now it reads 0.0 / quiet.
- 1 message gives percentile 0.973 (high), 2 give 0.989 and 3 give 0.996 (unusual).
- With fewer than 5 turns in 7 days the band reads `insufficient_history` and the source cannot compete.

**5. Existing mechanism**
- There is no Juniper-turn-rate metric.
- `conversation_load` / `repair_pressure` measure Orion's load, not world arrival.
- The cockpit `cockpit_turn_sighting` table is turn-stage telemetry, not a rate.

**6. Reversibility**
- Nothing is persisted except traces inside frames.
- There is no schema field, no lock entry and no training default.
- Setting the flag to false removes it from both contests.

**Turn novelty (Juniper's second chat input): FAILS step 4. Not wired.**
- `spark_meta.novelty` is a TOPIC-shift classifier probability from memory-consolidation.
- 23 of 28 Juniper turns in 7 days read ≥ 0.99; 1 reads < 0.5.
- It is saturated and binary in practice. A percentile over values like 0.99999998 versus 0.9999999 is noise.
- Next step: a novelty measure with real spread, for example embedding distance from her recent turns. That is a separate gate.

## Metric quality gate: camera surprise (`node:substrate.perception`, moved body → world)

Semantic layer: `config/field/field_channel_glossary.v1.yaml` node-qualified entry, and `check_metric_lineage.py --metric node:substrate.perception.prediction_error`:
- value_kind `level`, polarity `higher_is_worse`, sparsity `designed_sparse`.
- rest: "0.0 = at or below this stream's usual surprise. Also 0.0 while warming and when the frame is stale."
- absent_means: "written as 0.0 every tick rather than dropped".

**1. Provenance:**
- `orion/substrate/prediction_error.py:1354` computes 1 − cos(embedding, the stream's EWMA), z-scores it against its own EWMA, clips negatives, and scales so 3 sigma = 1.0.
- `services/orion-substrate-runtime/app/worker.py:_perception_prediction_error_tick` writes it every tick and stores `embedding_staleness` on the node.

**2. Independence.** It is a separate sensor stream (cam0 embeddings). It shares no upstream computation with the body PE nodes. Its only coupling is to camera health, which is exactly what `absent` handles.

**3. Theory anchor.** Predictive-coding surprise: the scene's departure from its own running expectation.

**4. Live sanity:**
- 7-day history has 14,563 readings, 73.8% exact zeros, p90 0.149, p99 0.79.
- Rest is reachable: 0 means at or below usual.
- `absent` comes from staleness, never from the value:
  - broadcast: node `embedding_staleness ≥ 1.0`;
  - field: vision organ `vision_frame_staleness ≥ 1.0`, missing, or a reading older than 180 s.
- **Caveat found:** 2026-10-04 and 10-05 read exact 0.0 for two full days (2,344 and 2,333 readings), while `substrate_perception_embedding_baseline` shows embeddings arriving normally (about 16.7k per day).
  - The cause is UNVERIFIED: the container restarted today and the logs are gone.
  - A stuck-at-zero perception reads as "quiet", so it cannot *win* on a false value.
  - It does mean the world can look calm when it is not.
  - Those 2 days of zeros also lower the bar for any nonzero reading.

**5. Existing mechanism.** Endogenous curiosity already reads it. The field contest excluded it.

**6. Reversibility.** Flag off restores its old role exactly.

**Presence / home-camera Juniper sighting (PR #2558): FAILS step 4 (no history). Documented as the next source.**
- Nothing persists sightings: `vision_individual_sighting` has 0 rows.
- PR #2558's own report says the match "lived only in the camera's presence row… nothing persisted it". There is no week to be unusual against.
- Gap: persist sighting events, then score their change rate against their own normal.

**Cabinet mic loudness (PR #2569): FAILS step 2 (independence). Documented as the next source.**
- `cabinet_ambient_audio_activity` is a pressure hint that feeds `node:substrate.biometrics`' prediction error (`orion/substrate/cabinet_ambient_spike_consumer.py`, `orion/telemetry/ambient_audio.py`). As a world source it would double-count a body input.
- The mic also sits in the server cabinet, so fans and the room are mixed.
- History exists: 19,186 `orion_biometrics_summary` rows in 7 days, p50 5,935 and p90 8,441 RMS.
- Gap: split room sound out of the body node, then gate it.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -q -p no:cacheprovider -W ignore <paths>

tests/test_attention_world_first.py                         31 passed
tests/test_attention_world_first_field_frame.py             12 passed
tests/test_attention_world_first_parity.py                   2 passed  (golden from origin/main dc8eab06c)
tests/test_attention_pe_history_cache.py                     3 passed
orion/substrate/tests/test_attention_broadcast_world_first.py 9 passed
services/orion-attention-runtime/tests                      54 passed, 5 skipped
services/orion-substrate-runtime/tests/test_worker_world_first_broadcast.py 3 passed
services/orion-thought/tests                                518 passed, 1 failed (pre-existing on origin/main: test_settings_mind_enrichment)
tests/test_attention_*.py, tests/test_proposal_*.py         all passed (179)
orion/substrate/tests/test_attention_broadcast*.py, test_prediction_error_magnitude.py (unchanged, passing),
  test_attention_self_model.py, test_endogenous_curiosity.py, test_voluntary_attention_wiring.py  passed
services/orion-substrate-runtime/tests                      17 failed / 1 collection error -- identical set on origin/main (pre-existing; cursor/quarantine/reducer tests)
Static gates: check_metric_lineage --gate PASS, --prompt-semantics PASS, check_definition_drift --gate PASS (0 changed),
  check_inner_state_registry OK, check_env_template_parity PASS, check_service_env_compose_parity (attention-runtime OK;
  substrate-runtime same 17 pre-existing gaps as main), tests/test_agent_trace_schema_registry.py PASS
```

**Mutation check: 15 of 15 mutants killed.** Each was applied to the real file, the matching test run, and the file restored:

| # | Mutant | Result |
|---|---|---|
| 1 | Internal admits "usual" | killed |
| 2 | `absent` ignored | killed |
| 3 | No-winner forced to a winner | killed |
| 4 | External admits "quiet" | killed |
| 5 | Proposals bind external | killed |
| 6 | Goals accept external | killed |
| 7 | Field frame keeps min-max | killed |
| 8 | Broadcast ignores world-first | killed |
| 9 | Empty coalition activates | killed |
| 10 | Stale internal allowed | killed |
| 11 | Placeholder allowed | killed |
| 12 | Reverie narrates empty | killed |
| 13 | Flag-off field ranking perturbed | killed (parity golden) |
| 14 | Flag-off broadcast perturbed | killed (parity golden) |
| 15 | Flag default off | killed |

## Evals run

```text
services/orion-attention-runtime/evals (pytest): 3 passed (focus-run replay + world-first replay arithmetic)
Live read-only replay: python services/orion-attention-runtime/evals/replay_world_first.py --days 3 --step-sec 60
  -> numbers in "Outcome moved"; variant below
VARIANT_RESULT
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-attention-runtime build   -> Built
scripts/safe_docker_build.sh orion-substrate-runtime build   -> Built
scripts/safe_docker_build.sh orion-thought build             -> Built
(temporary .env symlinks to the primary checkout, removed afterwards; nothing deployed)

In-image import + glossary: attention-runtime and substrate-runtime resolve value_kind/polarity from /app/config/field.
Live read-only smoke (attention-runtime image on app-net, real Postgres, no writes):
  seed of 7-day PE history 0.96 s, incremental refresh + all candidates 0.09 s
  winner node:substrate.perception (external, percentile 0.891, band usual); every body node quiet/usual -> not eligible;
  harness_closure insufficient_history (placeholder); world:chat quiet (nobody talking).
```

Runtime proof of the deployed path is **UNVERIFIED** until deploy. To check it after deploy:
- `docker logs orion-athena-attention-runtime | grep "world_first=True"` shows `winner=none` on calm ticks.
- `SELECT frame_json->'warnings' FROM substrate_attention_frames ORDER BY generated_at DESC LIMIT 20` shows `world_first_no_winner` rows.
- `substrate_attention_broadcast_log` rows show `frame.debug.world_first`.

## Review findings fixed

REVIEW_FINDINGS

## Proposal-mode items (CLAUDE.md §0A)

Juniper approved implementation on 2026-10-07 ("hit that seam") and 2026-10-10 ("ok go").

- **Capability change.** Attention defaults to the world. The body interrupts only when unusual for itself. Calm ticks may have no winner. Goal provenance and self-modification proposals are restricted to body targets.
- **Data touched.** Read-only reads of:
  - `substrate_node_prediction_error_history`;
  - `chat_history_log` (timestamps only, no content);
  - `substrate_field_state`.

  Writes are unchanged: the same frame tables, with new trace fields inside the existing JSON.
- **Privacy boundary.** Unchanged. Only turn timestamps are counted; no message text is read.
- **Trace.** Every field frame target carries `source_kind:` plus band, percentile and reason. Every broadcast frame carries `debug.world_first` with each candidate's verdict, including absent and uncalibrated ones.
- **Dangerous failure modes:**
  - **A real body alarm is outranked.** Saturating signals (execution at 1.0) sit at about the 96th percentile and can lose to a camera blip; 35 of 86 execution spikes were outranked in the replay. They stay eligible, ranked second, but are not the broadcast winner.
  - **Over-reacting to world noise.** The camera is nonzero about 26–39% of the time, and the "usual" floor admits it whenever nonzero; it won 24% of replayed ticks.
  - **A world winner bound to self-modification.** Guarded by the proposals filter and the goal filter, both tested and mutation-checked.
  - **Silence read as calm.** Handled by `absent`. Not caught: perception stuck at exact 0 with fresh embeddings, as on 10-04/05.
- **Disable / roll back.** `ATTENTION_WORLD_FIRST_ENABLED=false` in either service's `.env`, then restart. This restores the old ranking exactly.

## Restart required

Deploy order: there are no forbid-model changes, so any order is safe. The recommended order is thought (reverie skip) → substrate-runtime → attention-runtime. Run from the PRIMARY checkout on main, after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-thought orion-substrate-runtime orion-attention-runtime; do ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh $s up -d --build || break; done
```

Other services that import the changed shared code pick it up on their next rebuild. Nothing breaks if they are not rebuilt:
- cortex-exec: detectors' provenance tag;
- hub: reads frames, unchanged shape.

## Risks / concerns

- **Severity: major. The body still interrupts on 42% of ticks.**
  - **What happens:** each body node is in its own top decile ("high") 10% of the time by definition. With about 9 nodes, the union is large.
  - **Why it still passes:** the spec's acceptance metric (calm winners under 5%) is met at 0.28%. Those interruptions are not calm in raw terms either.
  - **The knob:** the internal band. Variant: VARIANT_SHORT.
  - **Mitigation:** band cuts are knobs to grade on 48 h of live data. Not changed here.
- **Severity: major. Camera surprise wins 24% of ticks.**
  - **What happens:** the "usual" floor (at or above its own median) admits any nonzero reading.
  - **Mitigation:** a per-source external floor is a knob. The 48 h live check should grade it against whether those frames correspond to real scene changes.
- **Severity: medium. Saturating body signals cap near the 96th percentile.**
  - **What happens:** this happens whenever a signal's ceiling is common in its own history. 35 of 86 execution spikes ≥ 0.9 were outranked, though still eligible.
  - **Mitigation:** none in this patch. Report it. A tie-aware percentile, or "unusual" by value, is a follow-up decision.
- **Severity: medium. Perception sat at exact 0 for 48 h on 10-04/05 with embeddings arriving.**
  - **Status:** the cause is UNVERIFIED and `absent` cannot see it.
  - **Mitigation:** a follow-up issue on the perception scorer.
- **Severity: low.**
  - The `attention_saturated_execution` consolidation motif needs host targets among attended targets. It stays unreachable under world-first. It has already not fired since 2026-09-29 (1,228 fires before that), so this patch does not kill a live motif. Follow-up: retire or re-anchor it.
  - The Hub lattice "transport gate" and the self-brain spotlight render `world:chat` as if it were a graph node. Cosmetic.
- **Severity: low. 2 of 9 broadcast candidates are reads of other services' tables** (chat_history_log, field_state). If those reads fail, the source is absent, never calm, and never a fallback.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
