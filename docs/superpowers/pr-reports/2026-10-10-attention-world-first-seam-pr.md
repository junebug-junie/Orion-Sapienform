# Attention faces the world first; the body interrupts only when it is unusual for itself

## Summary

Until now Orion's attention looked only inward, and every tick crowned a winner. The main contest rescaled five hardcoded prediction-error nodes so the top one read 1.0 on every frame, even when all five were calm. This patch makes three changes:

- **The world competes by default.** Juniper's chat activity and the camera's surprise now compete.
- **The body interrupts only when it is unusual for itself.** A body signal competes only when it sits in the high or unusual band of its own last 7 days, in the bad direction.
- **A calm tick can have no winner.**

What is in the patch:

- **`AttentionCandidateV1`.** One shape for anything that can compete. It carries:
  - internal (body) or external (world);
  - how unusual the reading is against that source's own week (`PredictionErrorMagnitudeV1`, reused as-is);
  - an `absent` flag, so silence is never read as calm.
- **The ranking rule** (`orion/attention/world_first.py`):
  - The world is eligible when it is busy for itself, meaning its own top decile.
  - The body is eligible only at band high or unusual, in its bad direction. Polarity and value_kind come from the glossary semantic layer.
  - Eligible candidates rank by their own percentile. Ranking uses the mid-rank form, so a signal pinned at its ceiling still wins ties.
  - An empty eligible set produces an explicit no-winner frame.
- **Both background contests are wired.**
  - The field contest (attention-runtime) drops the hardcoded five-node list and the min-max rescale that guaranteed a winner.
  - The substrate broadcast (substrate-runtime) now uses magnitude as a gate rather than a description.
- **Chat is the first world source.** It counts Juniper's turns in the last 15 minutes against the same count over her last 7 days. Camera surprise (`node:substrate.perception`) moved from body to world.
- **Blast-radius fixes:**
  - self-modification proposals, goals and reverie proposals never bind to a world winner;
  - goals stay limited to the five native nodes;
  - an empty coalition no longer "activates" or reports dwell;
  - reverie skips no-winner ticks;
  - the self-model no longer narrates "'None' selected", and it names sources it could not read;
  - curiosity skips world-only loops.
- **Flag.** `ATTENTION_WORLD_FIRST_ENABLED` ships on. With `false`, both contests reproduce the old ranking byte for byte, checked against golden output from origin/main.

## Outcome moved

**Offline replay.** Read-only, `services/orion-attention-runtime/evals/replay_world_first.py`, 1-minute grid, 4,321 ticks, 2026-10-07 04:57 to 2026-10-10 04:57 UTC, final rules after review:

| | Old ranking (122,985 stored frames) | World-first (replay) |
|---|---|---|
| Frames whose winner is a calm body signal (raw error < 0.05) | **41.5%** | **0.6%** |
| Frames with no winner | 0% | **43.9%** |
| World winners | 0% | 13.8% (camera 13.1%, chat 0.7%) |
| Body winners | 100% | 42.3% |

The spec quoted 49% calm winners from an earlier window. This window measures 41.5% from the stored frames.

**Real body spikes still win:**

- **Biometrics surges ≥ 0.25:** 108 of 110 won the next tick, at band "unusual".
- **Execution readings ≥ 0.9:** 59 of 86 won. The other 27 were eligible and ranked, but lost near-ties (all about 0.96) to bus_synaptic, route, chat or camera readings at the same percentile.
- **Synthetic 5-hour storm.** This stands in for the 2026-09-28 timeout-storm class, which predates the stored PE history (that history starts 2026-10-03). It injects execution readings of 1.0 every 135 s, replacing the real readings in its window. The node stayed eligible (band "high") on **60 of 60** ticks, so its own baseline did not absorb the storm, and it won 73% of them.
- **Rejected variant:** a body band of "unusual" only. It cuts body winners to 6.4%, but every execution spike then loses (0 of 85), because saturated readings never reach "unusual". This is the real-alarm-suppressed failure, so the variant is not used.

## Current architecture

Three contests, all inward-facing, and all crowning a winner:

- **Field attention** (`services/orion-attention-runtime`). Five hardcoded PE nodes, scored by precision-weighted error, then min-max rescaled so the top one was always 1.0. Host and capability novelty targets sat alongside.
- **Substrate broadcast** (`orion/substrate/attention_broadcast.py`). Any graph node with `dynamic_pressure` at or above the floor, Borda-ranked. Magnitude was attached to loops but never read by scoring.
- **Per-turn chat frame.** Three detectors with constant saliences.

None of the three had a world input, and camera surprise was excluded from the field contest.

## Architecture touched / files changed

**Contract**
- `orion/schemas/attention_candidate.py` (new). It also holds `is_world_source_id` for thin consumers.
- `orion/schemas/registry.py`: registered in both maps, verified with `resolve()`.
- `scripts/check_inner_state_registry.py`: AttentionCandidateV1 is covered as a sub-object of the two frame entries.

**Rule**
- `orion/attention/world_first.py`: ranking, chat source, camera absence, glossary polarity.
- `orion/attention/pe_history_cache.py`: attention-runtime's memoized 7-day window over `substrate_node_prediction_error_history`.

**Field contest**
- `orion/attention/field_attention/{builder,selectors,goal_provenance}.py`
- `services/orion-attention-runtime/app/{worker,store,settings}.py`, `.env_example`, `docker-compose.yml`, `Dockerfile` (adds `COPY config/field`)

**Broadcast**
- `orion/substrate/attention_broadcast.py`
- `services/orion-substrate-runtime/app/{worker,store,settings}.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`

**Blast radius**
- `orion/proposals/builder.py`
- `orion/reverie/proposal.py`
- `services/orion-thought/app/reverie.py`
- `orion/substrate/attention_self_model.py`
- `orion/substrate/endogenous_curiosity.py`

**Per-turn detectors** (`orion/substrate/attention/detectors/*`)
- The constants are now named.
- Each signal carries `source_kind` provenance.
- `base.py` has the audit note on why they stay fixed rank priors: there is no per-detector history to calibrate against, the values only act as Borda rank order, and the frame exists only when a turn has arrived.

**Eval:** `services/orion-attention-runtime/evals/replay_world_first.py` and `test_world_first_replay.py`.

**Tests:** listed below. Three existing worker tests are pinned to flag-off.

## Schema / bus / API changes

- **Added:** `AttentionCandidateV1`, registered and not on the bus.
- **Removed / Renamed:** none.
- **Behavior changed, with the flag on:**
  - Both contests rank world-first and can produce no winner.
  - A field frame can carry `world:chat` (kind `channel`), and has `warnings=["world_first_no_winner"]` when empty.
  - A broadcast loop can carry `source_refs=["world:chat"]`.
  - With world-first, `loop.salience` is the rank percentile. The Borda value moves to `provenance["borda_salience"]`, which is what reverie's `derive_salience` reads, so reverie salience keeps its calibration.
- **Compatibility:**
  - No `extra="forbid"` model changed. This is step 1 of the spec's rollout.
  - `source_kind` rides in existing free-form fields: `FieldAttentionTargetV1.evidence_refs` (`source_kind:…`), `OpenLoopV1.provenance`, and `AttentionFrameV1.debug["world_first"]` (every candidate's verdict, band, percentile, rank score, plus `absent_sources`).
  - Deploy order is therefore free. Typed fields later would be consumer-first, and are not done here.

## Env/config changes

- **Added keys:** `ATTENTION_WORLD_FIRST_ENABLED=true` in orion-attention-runtime and orion-substrate-runtime, across `.env_example`, settings and compose.
- **Removed / renamed keys:** none.
- **`.env_example` updated:** yes, both services.
- **Local `.env` synced:** yes. `python scripts/sync_local_env_from_example.py --all-keys orion-attention-runtime orion-substrate-runtime` was run from the worktree and wrote both primary-checkout `.env` files (`+ATTENTION_WORLD_FIRST_ENABLED='true'`).
- **Skipped keys needing operator action:** none.
- **Not flag-gated (stated on purpose).** These bug fixes are correct with the flag off too:
  - an empty coalition never activates or reports dwell;
  - reverie skips a no-winner tick;
  - the self-model no-winner narrative.

  The flag restores **ranking** exactly. These three stay fixed.

## Semantic-layer trace (standing rule)

Checked with `config/field/field_channel_glossary.v1.yaml` and `scripts/check_metric_lineage.py --metric node:substrate.<x>.prediction_error` (PR #2579 fields):

| Node | value_kind | polarity | Notes |
|---|---|---|---|
| perception | level | higher_is_worse | rest 0.0 = at/below usual, **also 0 while stale/warming**, so absence comes from staleness, never the value |
| chat | level | higher_is_worse | designed_sparse: zeros are by design |
| biometrics | level | higher_is_worse | per_tick; rest about 0.02–0.03 |
| cabinet | trigger | none | "fired" is the event; treated as up |
| harness_closure | placeholder | — | never eligible: a fixed 0.65, not a measurement |
| route | count | — | designed_sparse |
| execution, codebase | level | — | designed_sparse |

- **Polarity** is `orion.metrics.semantics.derived_channel_polarity("prediction_error", value_kind)`, the same derivation PR #2579 uses. Nothing is invented here.
- **Containers.** Both images copy `config/field`. Verified in-image: cabinet → `('trigger', None)`, harness_closure → placeholder.
- **Glossary read failures** are not cached, and are logged.

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
- Rest is reachable: 0 means at or below usual. Nonzero 26% of the time and flat around the clock, so under the review fix it competes only in its own top decile (≥ 0.149), not whenever nonzero.
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
tests/test_attention_*.py + tests/test_voluntary_attention_wiring.py           266 passed
  (incl. new test_attention_world_first.py 40, _field_frame.py, _parity.py (golden from origin/main dc8eab06c), _pe_history_cache.py)
tests/test_proposal_*.py                                                         73 passed
orion/substrate/tests/test_attention_broadcast*.py (incl. new _world_first.py), test_prediction_error_magnitude.py (unchanged),
  test_attention_self_model.py, test_endogenous_curiosity.py, override/verdict/goal tests, orion/reverie/tests/test_proposal_world_winner.py   201 passed
services/orion-attention-runtime/tests + evals                                   57 passed, 5 skipped
services/orion-thought/tests                                                     519 passed, 1 failed (pre-existing on origin/main: test_settings_mind_enrichment)
services/orion-substrate-runtime/tests                                           17 failed + 1 collection error: identical set on origin/main (cursor/quarantine/reducer/self-model-tick env tests)
orion/reverie/tests/test_proposal.py                                             8 failed on origin/main too (stale self_state_id kwarg), untouched
Static gates: check_metric_lineage --gate PASS, --prompt-semantics PASS, check_definition_drift --gate PASS (0 changed, no re-lock needed),
  check_inner_state_registry OK, check_env_template_parity PASS, check_service_env_compose_parity (attention-runtime OK; substrate-runtime
  same 17 pre-existing gaps as main), tests/test_agent_trace_schema_registry.py PASS
```

**Mutation check: 23 of 23 mutants killed.** Each was applied to the real file, run against its test, then restored.

| Group | Mutants killed |
|---|---|
| Eligibility | internal admits "usual"; external admits "quiet"; external admits "usual"; stale internal allowed; placeholder allowed |
| Absent ≠ calm | `absent` ignored |
| No-winner | no-winner forced to a winner; field frame keeps min-max; broadcast ignores world-first |
| Ranking | strict-below-only ranking (ceiling ties) |
| Self-modification guards | proposals bind external; goals accept an external-marked native id; goals widened past the native five; reverie proposes on a world coalition |
| Hysteresis | empty coalition activates; empty tick reports dwell |
| Reverie | narrates an empty coalition |
| Glossary | failure is cached |
| Parity | flag-off field ranking perturbed; flag-off broadcast perturbed (both caught by the golden fixtures) |

## Evals run

```text
pytest services/orion-attention-runtime/evals -> 3 passed (focus-run replay + world-first replay arithmetic on synthetic data)
python services/orion-attention-runtime/evals/replay_world_first.py --days 3 --step-sec 60 (live, read-only) -> "Outcome moved"
Variant (internal band "unusual" only, 5-min grid): body winners 6.4%, no-winner 58.4%, execution spikes won 0/85 -> rejected
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-attention-runtime build -> Built
scripts/safe_docker_build.sh orion-substrate-runtime build -> Built
scripts/safe_docker_build.sh orion-thought build           -> Built
(temporary .env symlinks to the primary checkout, removed right after; nothing deployed)
Live read-only smoke (attention-runtime image on app-net, real Postgres, no writes):
  7-day PE history seed 0.86 s; repeat candidate build 0.004 s (memoized; was 0.09 s)
  winner node:substrate.execution (internal, percentile 0.922, band high); camera and chat quiet; harness_closure insufficient_history
```

The deployed path has not been checked yet (**UNVERIFIED**). After deploy:
- `docker logs orion-athena-attention-runtime | grep world_first=True` should show `winner=none` on calm ticks.
- `substrate_attention_frames.frame_json->'warnings'` should contain `world_first_no_winner`.
- `substrate_attention_broadcast_log` rows should carry `frame.debug.world_first`.

## Review findings fixed

The code review ran as a subagent against `origin/main...HEAD`. A separate subagent audited the downstream consumers. Findings and what was done about them:

- **Finding:** a "usual" external floor reduced to "value > 0". The camera won 24% of ticks on readings as small as 5e-05, with its nonzero rate flat around the clock.
  - **Fix:** the external floor is now band "high" (its own top decile).
  - **Evidence:** camera share fell from 24.0% to 13.1% in the replay. Test `test_tiny_nonzero_camera_reading_on_a_zero_heavy_week_is_not_eligible`, and the mutant "external admits usual" is killed.
- **Finding:** a ceiling-pinned body signal capped at about 0.965 and lost to rare-fire nodes.
  - **Fix:** a mid-rank `rank_percentile` on the candidate, used for ranking only. Eligibility stays strict-below, so all-zero still reads as rest.
  - **Evidence:** execution spike wins went from 51/86 to 59/86, and storm wins from 63% to 73%. Tests `test_ceiling_ties_rank_by_mid_rank_not_strict_below` and `test_mid_rank_breaks_a_ceiling_tie_in_the_broadcast`.
- **Finding:** world winners leaked into self_state proposals through reverie, and percentile salience would inflate reverie salience.
  - **Fix:** `spontaneous_thought_to_candidate` refuses a world-only coalition, and `derive_salience` reads the parked Borda value.
  - **Evidence:** `orion/reverie/tests/test_proposal_world_winner.py`, `test_derive_salience_keeps_borda_meaning_under_world_first`.
- **Finding:** the goal set widened beyond the five native nodes (cabinet and codebase).
  - **Fix:** a goal must now be internal **and** native.
  - **Evidence:** `test_goal_set_is_not_widened_beyond_the_native_five` and `test_goal_refuses_a_native_id_marked_external`.
- **Finding:** flag-off did not restore downstream behaviour.
  - **Fix:** this is now stated in the docstring and in this report. Those three changes are deliberate bug fixes and are not flag-gated.
  - **Evidence:** the parity golden covers ranking.
- **Finding:** the two contests used different definitions of camera absence.
  - **Fix:** one shared `perception_absent_reason`.
  - **Evidence:** `test_one_definition_of_camera_absence`.
- **Finding:** a glossary read failure was cached forever.
  - **Fix:** only successful reads are cached.
  - **Evidence:** `test_glossary_failure_is_not_cached`.
- **Finding:** late rows could be missed by the 10-minute overlap (live max lag 355 s).
  - **Fix:** the overlap is now 30 minutes. `recorded_at` has no index, and scanning it every 2 s is not worth it.
  - **Evidence:** `test_row_committed_late_within_the_overlap_is_still_picked_up`.
- **Finding:** about 120 ms of recompute ran on every 2 s tick.
  - **Fix:** magnitudes are memoized per minute and the chat grid per minute and turn set. Only the reading's age is refreshed.
  - **Evidence:** 0.004 s live repeat, and `test_magnitudes_are_memoized_within_a_minute_but_age_moves`.
- **Finding:** a tick where the sources failed looked calm.
  - **Fix:** the trace carries `absent_sources`, and the self-model narrative names them.
  - **Evidence:** `test_trace_lists_absent_sources_so_failure_is_not_calm`, `test_unreadable_sources_are_named_in_the_trace`, `test_no_winner_with_unreadable_sources_says_so`.
- **Finding:** an empty tick still reported the old coalition's dwell and stability.
  - **Fix:** an empty tick now reports dwell 0 and stability 0.3, without touching the hysteresis state.
  - **Evidence:** `test_no_winner_tick_reports_no_dwell_while_the_old_coalition_decays`.
- **Finding:** `world_first_enabled()` was dead code.
  - **Fix:** removed. Runtime settings own the flag.
- **Audit:**
  - reverie narrated empty coalitions (438 of 2,251 live reveries in 3 days);
  - the self-model said "'None' selected";
  - curiosity could seed `concept_expand` on `world:chat`.

  All three are fixed and tested. Cosmetic issues are left as risks below.

## Proposal-mode items (CLAUDE.md §0A)

Juniper approved this on 2026-10-07 ("hit that seam") and 2026-10-10 ("ok go").

- **Capability change:**
  - Attention defaults to the world. The body interrupts only when it is unusual for itself, and calm ticks may have no winner.
  - Goals, self-modification proposals and reverie proposals stay body-only.
- **Data touched:** read-only reads of `substrate_node_prediction_error_history`, `chat_history_log` (timestamps only), and `substrate_field_state`. Writes are unchanged: the same frame tables, with new trace data inside the existing JSON.
- **Privacy boundary:** unchanged. No message text is read.
- **Trace:**
  - Every field target carries a `source_kind:` marker plus band, percentile and reason.
  - Every broadcast frame carries `debug.world_first`, with each candidate's verdict and `absent_sources`.
- **Dangerous failure modes:**
  - **A real body alarm can be outranked.** Execution at 1.0 loses near-ties: 27 of 86 spikes in the replay. Those losses stay eligible and ranked.
  - **Over-reacting to world noise.** The camera wins 13% of ticks even with the top-decile floor.
  - **A world winner bound to self-modification.** Guarded at three points (proposals, goals, reverie proposals). All three are tested and mutation-checked.
  - **Silence read as calm.** `absent` covers it, with one gap: perception stuck at exact 0 while embeddings arrive (10-04/05) is not caught.
- **Disable / roll back:** set `ATTENTION_WORLD_FIRST_ENABLED=false` in either service's `.env` and restart.

## Restart required

No forbid-model changes are involved, so any deploy order is safe. Recommended order: thought, then substrate-runtime, then attention-runtime. Run from the PRIMARY checkout on main, after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-thought orion-substrate-runtime orion-attention-runtime; do ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh $s up -d --build || break; done
```

Other services import the shared code and pick it up on their next rebuild. Nothing breaks if they are not rebuilt: proposal-runtime (the reverie proposal refusal), cortex-exec (detector provenance) and hub.

## Risks / concerns

- **Severity: major. The body still wins 42% of ticks.**
  - **Why:** each of about 9 body nodes is in its own top decile 10% of the time, so their union is large.
  - **Spec check:** met. Calm winners are 0.6%, against a target of under 5%.
  - **Knob:** the body band. The tighter "unusual"-only option was rejected because it suppresses execution alarms.
  - **Next step:** grade the knob on 48 h of live data.
- **Severity: medium. The camera wins 13% of ticks.**
  - **Next step:** grade those frames against real scene change during the 48 h live check.
- **Severity: medium. Perception sat at exact 0 for 48 h on 10-04/05 while embeddings kept arriving.**
  - **Status:** cause UNVERIFIED. `absent` cannot see this case.
  - **Next step:** a follow-up on the perception scorer.
- **Severity: low. Turn novelty is not wired** (it is a saturated classifier). **Presence/sighting is not wired** (no history exists). **Mic loudness is not wired** (not independent of biometrics PE). See the gate records above.
- **Severity: low.**
  - The `attention_saturated_execution` motif stays unreachable, but it has already not fired since 2026-09-29.
  - The Hub transport gate and the self-brain spotlight draw `world:chat` as if it were a graph node. Cosmetic.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2583

🤖 Generated with [Claude Code](https://claude.com/claude-code)
