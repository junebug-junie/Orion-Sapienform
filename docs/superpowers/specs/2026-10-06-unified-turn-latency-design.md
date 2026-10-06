# Unified chat turn latency: corrected findings and design

Date: 2026-10-06. Status: **design. Items L1 and L2 are ready to implement. Items marked PROPOSAL need Juniper's yes/no first.**

Supersedes the recommendations in `docs/superpowers/design/2026-10-06-unified-turn-latency-handoff.md`. That doc's diagnosis of where time goes stands. Two of its four "do now" fixes would have backfired or done nothing (see "Corrections"), and it missed the larger levers.

Companion spec: `2026-10-06-thermal-controller-redesign-design.md` (branch `docs/thermal-controller-redesign`). Tonight's slowest turns were caused by a false cooling shed, not by anything in this doc.

## Arsonist summary

A unified Hub turn takes about 68 s, made of serial steps:

| Step | Time |
|---|---|
| appraisal | 1 s |
| orion-mind (3 calls) | 11 s |
| stance context build | 9 s |
| stance model call | 10 s |
| recall | 1 s |
| Claude Code cold start | 3–4 s |
| reply writer | 10–31 s |
| a second, unused context build | 5–12 s |
| polishing judge | 11 s |

Changes in this doc:
- **Ready now:** remove the unused second build, using the skip flag that already exists, and stop the build freezing cortex-exec.
- **Design:** run orion-mind and the context build at the same time (the build never uses mind's output); keep the Claude Code reply writer warm between turns.
- **Proposals:** fix the substrate decay loop before touching the concept re-save; stop the reply writer loading a shared, ungoverned auto-memory; run the polishing judge after the reply is shown.

Rough, untimed estimate if all land: 68 s → mid-30s. Each item carries its own before/after measurement.

## Corrections to the handoff (verified 2026-10-06, read-only)

1. **The concept re-save is load-bearing (handoff F2/Rec 1 was wrong).**
   - `SubstrateDynamicsEngine.tick()` runs every 30 s in orion-substrate-runtime.
   - Every tick it decays the *already-decayed* stored activation by the full time since `observed_at` (`orion/substrate/dynamics.py:118-125`, `orion/core/activation_decay.py:8-20`). The loss compounds, about 2% per 30 s.
   - With the default 30-day half-life (`cognitive_substrate.py:91`) and a seed about 23 h past `observed_at`, each 30 s tick multiplies by about 0.978. That matches the live reads. The per-tick loss grows as a node ages. `decay_floor` defaults to 0.0 (`cognitive_substrate.py:84`).
- The re-save writes stale stored values back (max-merge, `orion/substrate/reconcile.py:299-312`), which accidentally undoes the decay.
   - Live Falkor reads: `sub-concept-seed-juniper` activation went 1.0 → 0.978 → 0.956 → 0.934, then reset to 1.0 when a stance build ran (03:33:40 UTC). The relationship seed did the same.
   - Making `concept_induction` ephemeral, as the handoff proposed, would remove the only thing currently undoing that compounding: the seeds fall toward 0 (the default floor) and flip `dormant` within hours. After the decay fix they would fade on the true 30-day half-life instead, so the re-save can go only once decay is fixed.
2. **Skipping the build in `router.py` alone saves nothing (handoff Rec 3).**
   - The step-time check `_should_prepare_brain_reply_context` (`services/orion-cortex-exec/app/executor.py:4828`, called at `:3120`) does not skip these verbs, so the build would run there instead.
   - The existing `skip_brain_reply_context` flag is honored by both places (`router.py:1046`, `executor.py:4841-4844`).
3. **The whole build cannot move to a thread (handoff Rec 2).** `build_chat_stance_inputs` awaits bus publishes and the probe partway through (`chat_stance.py:2575,2654,2693,2700`). Only the synchronous parts can move.
   - Also, `run_in_executor` does not copy contextvars; nothing uses them today.
4. **Raising the 30 s snapshot ceiling (handoff Rec 7) does nothing for the second reload.** That reload is caused by this process's own write bumping its in-memory generation counter (`falkor_store.py:666,686`), not by the ceiling.
5. **The finalize step's reflection is not a no-op.** It triggers response repair on about 1 in 4 runs.

## Current architecture

```text
Hub turn_orchestrator
  → orion-thought run_stance_react (services/orion-thought/app/bus_listener.py:442-445)
      → orion-mind: 3 serial calls, metacog lane (Qwen3-8B, gpu3)        ~11 s
      → cortex-exec stance_react (chat lane, 35B on gpu0+gpu3)
           router.py:1034-1056 prepare_brain_reply_context
             → build_chat_stance_inputs: felt state + beliefs_for_stance
               (2 full Falkor snapshot() rehydrates, ~4.5 s each, ON the event loop)   ~9 s
           → stance LLM call                                                            ~10 s
  → harness-governor: fresh `claude -p` subprocess per turn (orion/harness/fcc_motor.py:1052-1060,1127)
      cwd /mnt/orion-fcc/repo → auto-memory /root/.claude/projects/-mnt-orion-fcc-repo/memory/
      → reply writer via gateway Anthropic passthrough                                  10–31 s
  → run_harness_finalize_chain (orion/harness/finalize.py:1471)
      → cortex-exec harness_finalize_reflect: SAME build again (unused)                 5–12 s
      → reflection LLM call (verdict, not text)                                         ~11 s
      → response repair if misaligned/uncertain/strain (rewrites text)
  → Hub receives final text
```

## Items

### L1. Skip the unused build before finalize and repair (ready)

- **Change:**
  - Extract a verb-name helper, `brain_reply_context_skipped(verb, ctx, options)`, from `_should_prepare_brain_reply_context` (`executor.py:4828`), which takes an `ExecutionStep` the router doesn't have.
  - Router (`router.py:1046-1048`) and step-time check (`executor.py:4839-4844`) both call the helper, so they cannot drift again.
  - Add `harness_finalize_reflect` and `orion_response_repair` to the helper's skip set.
  - Today the router checks the `skip_brain_reply_context` flag only in `ctx`. The executor checks `ctx` and `options`, and `options` is where `services/orion-hub/scripts/memory_graph_suggest.py:354` sets it. The helper reads both.
  - This is a cortex-exec-only deploy. The alternative, setting the flag in `finalize.py:426`, would need a harness-governor redeploy and keep two mechanisms alive.
  - Skipping also skips `_inject_identity_context`. The templates don't read identity keys; the PR must grep for any downstream reader of those keys on these verbs.
- **Why safe:**
  - Neither template nor YAML reads stance keys.
  - Both callers read text only (`finalize.py:454,921`).
  - `orion_response_repair` runs only after a stance (`finalize.py:1471`, only caller in production).
  - The PCR pre-recall gate (`router.py:1124-1129`) runs only for `chat_general`/`chat_quick`/`stance_react`, so these verbs never reach recall.
- **Downstream effects** (all duplicates today: 60/60 finalize correlation ids are also stance ids):
  - `chat_stance_belief_log` and `substrate_attention_schema` (`cortex_turn`) roughly halve.
  - **Recent-attention cue** (`orion/substrate/recent_attention_cue.py:18-24`, newest 3 rows, no dedup) frees one slot per turn for a real item.
  - **Hub surface panel** (`hub-surface.js:32`) shows half as many `cortex_turn` rows.
  - **Self-study** (`self_study.py:674`, reads a row count) and **self-inquiry SQL** (`orion/curiosity/self_inquiry.py:124-125,625-626`) see more distinct turns per window. Their contents change; note it in the PR.
  - **`substrate_tier_outcomes_events`** retention is per correlation id (`db.py:35-55`). After this change the stance event is the one kept, not the finalize event.
  - The `instruments.yaml` `count(*)` baseline moves; its `count(DISTINCT …)` does not.
  - The metric lock is definition-only and does not need re-locking.
- **Tests:**
  - (a) the router and the step-time check both skip these verbs, and the step-time path does **not** rebuild (assert `build_chat_stance_inputs` is not called);
  - (b) the stance leg still writes exactly one `cortex_turn` row per turn.
- **Expected:** about 9 s off the finalize leg. It is on the critical path.

### L2. Take the synchronous half of the build off the event loop (ready)

- **Change:** run `hydrate_felt_state_ctx` + `_unified_beliefs_for_stance` (`chat_stance.py:2518-2519`), `_project_recent_dispatch_actions` (`:2618`) and `build_attention_frame` on a **dedicated single-worker** `ThreadPoolExecutor` owned by the cortex-exec process. The async publishes stay on the loop.
- **Why single worker:** it preserves today's one-at-a-time ordering. It also covers the unlocked lazy globals:
  - `_UNIFICATION_LAYER` `chat_stance.py:209`;
  - `_READER` `felt_state_reader.py:335`;
  - the lived-answers cache `:223`;
  - `_last_materialized_at` `layer.py:110`.

  It also avoids the default pool, which about 38 `asyncio.to_thread` calls in cortex-exec share (`feedback_a_shared_default_executor_is_a_hidden_fifo_across_lanes`).
- **Thread safety checked:**
  - SQLAlchemy engine with a pool (`felt_state_reader.py:166`);
  - Falkor store `_snapshot_lock` (`falkor_store.py:315`);
  - `beliefs_for_stance` is documented thread-safe (`layer.py:92`);
  - no loop calls in the synchronous path.

  The same image runs as 4 containers; each gets its own executor.
- **A new race that the move creates.** Falkor store **writes** are not covered by `_snapshot_lock` (`falkor_store.py:404-414`), and the generation counter is a plain `+= 1` (`:666,686`). Today a write cannot overlap a snapshot, because the build blocks the whole loop. Once the build is off the loop, a loop task can write mid-snapshot. The fix ships with L2:
  - list every caller of the layer's store (`chat_stance.py:212` `_UNIFICATION_LAYER`'s store) from loop code;
  - either route those calls through the same single worker, or take `_snapshot_lock` around `upsert_node` and the counter bump.

  Test: a write issued during a snapshot never produces a cache missing that write.
- **Test:** a concurrent RPC is answered while a build is in progress (a fake slow build of 2 s, with an assert that a second handler completes in well under 2 s).
- **Expected:** no seconds saved on the turn itself. cortex-exec stops freezing reverie, metacog and journal for about 9 s per build (handoff F1 freeze evidence: reply published 18:11:00.482, received 18:11:08.073).

### L3. Small fixes bundled with L1/L2 (ready)

- **identity_yaml ephemeral** (`SNAPSHOT_EPHEMERAL`, `pull_on_cold=False`).
  - It removes the every-cold-turn `durable writes support concept, evidence, entity nodes only; got node_kind='state_snapshot'` failure (7 per 6 h live) and the `degraded` mark it puts on the orion anchor (`layer.py:224-225,270`).
  - Verified: the projected identity lines match today's fallback, same keys, same strip, cap 10 (`identity_yaml.py:66-71` vs `chat_stance.py:827-844,643`).
  - Same latent issue, same treatment: `self_definition` (state_snapshot) and `autonomy` (GoalNodeV1) are write-through producers with non-durable kinds.
- **Probe timeout:** pass `gateway_read_timeout_sec` from `current_turn_signal_probe_timeout_sec` in `current_turn_llm_signals.py:428-436`. Today the gateway assumes 700 s while the caller gives up at 3 s.
- **Reverie reader de-duplication:** `orion/situational/reverie_reader.py:89-109` returns the newest N rows with no dedup, so one repeated reverie fills every slot. That is why the "fixated on bus synaptic prediction error" line appeared twice in the reply writer's prompt. Dedupe on normalized text.

### L4. Run orion-mind and the stance context build at the same time (design)

- **Fact:** the build never reads orion-mind's output.
  - `mind_coloring` is used only by `stance_react.j2:37-54`.
  - On the Hub path, `mind_appraisal_text` is not passed (only `curiosity_investigation.py:3382` passes it).
  - The build's only per-turn input is the raw user message, which exists before orion-mind starts.
  - Mostly CPU, Postgres and Falkor, **except** the per-turn probe `populate_current_turn_llm_signals` (`chat_stance.py:2654`). On human turns it calls the `chat` route (`current_turn_llm_signals.py:38,428`), i.e. the 35B on gpu0+gpu3, while mind uses the 8B on gpu3. Overlapping them shares gpu3 compute. The PR measures mind's call time with and without the overlap.
- **Design:**
  - orion-thought fires two requests together: the orion-mind call, and a new cortex-exec RPC `stance_context_prepare` (correlation id + user message + the same ctx `stance_react` would send).
  - cortex-exec builds `chat_stance_inputs` and stores it in a small in-process TTL cache (120 s) keyed by correlation id.
  - When `stance_react` arrives, `prepare_brain_reply_context` finds it through the existing short-circuit (`executor.py:4813`) and skips the build.
- **Constraint:** both requests must land on the **same cortex-exec container** (cortex-exec vs cortex-exec-chat lane routing). Either:
  - (a) send `stance_context_prepare` on the same lane channel `stance_react` will use, decided once in orion-thought; or
  - (b) store the prepared inputs in Redis keyed by correlation id so any container can pick them up.

  Recommend (a). The inputs hold live objects and per-process caches; serializing them is a new contract for no gain.
- **Must set `ctx["verb"]="stance_react"`** in the prepare request. The probe and the attention frame read it (`current_turn_llm_signals.py:156`).
- **No double build on a miss.** `stance_react` must not start a second build while a prepare for the same correlation id is in flight on that container; it awaits the in-flight one. Otherwise:
  - the duplicate `chat_stance_belief_log`/`cortex_turn`/salience rows that L1 removes come back;
  - a second chat-lane probe call is made.

  Only an actual failure or a timeout (prepare absent at stance arrival + 2 s) falls back to building inline. The cache entry is marked consumed so it is used once.
- **Contract:** a new bus channel/kind in `orion/bus/channels.yaml` + `orion/schemas/registry.py` (CLAUDE.md §6), with producer and consumer tests.
- **Failure mode:** if the prepare call fails, or never arrives on this container, `stance_react` builds as it does today after a 2 s wait. The worst case is today's latency + 2 s.
- **Measure:** orion-thought logs the gap between mind done and stance start. The target is about 9 s → about 0–1 s on turns where mind ≥ build.
- **Expected:** up to about 9 s per turn (UNVERIFIED until timed).

### L5. Keep the reply writer warm between chat turns (design)

- **Fact:** every turn spawns a fresh `claude -p … --output-format stream-json` (`fcc_motor.py:1052-1060,1127`) and re-runs the container's SessionStart hook and context-mode plugin. That costs 2.4–4.4 s.
- **Pattern to reuse:** `services/orion-room-companion/app/claude_session.py:56-59` (`--session-id` first turn, `--resume` after).
- **Design question:** what does "warm" carry? `--resume` carries the full prior conversation into the next turn. That duplicates and fights the Hub's own turn context and recall, and it grows without bound.
- **Two options:**
  - **(a) Warm process, cold context:** a long-lived `claude` process per chat conversation, using the streaming-input mode (`--input-format stream-json`). Each turn is sent as a new user message, but the history is reset per turn. Needs a check that the CLI supports a per-turn reset without respawning. UNVERIFIED.
  - **(b) Cut the startup instead of the process:** for chat turns, skip the SessionStart hook and plugin load (`--setting-sources` scoped, or a chat-specific config dir without the plugin). Measure the remaining spawn time.
- **Recommend:** measure first. Time spawn → first stream event with and without the hook and plugin. If (b) recovers most of the 2.4–4.4 s, ship it and stop. A persistent process adds lifecycle, crash and concurrency handling. That is only worth it if (b) leaves more than about 1.5 s.
- This interacts with L7: a chat-specific config dir solves both.
- **Expected:** 1.5–4 s per turn.

### L6. PROPOSAL: fix the substrate decay loop, then stop the concept re-save

- **Capability change:** activation decays once per unit of real time instead of compounding. Seed concepts stop being held up by an accidental re-save.
- **Fix:**
  - In `SubstrateDynamicsEngine.tick()` (`dynamics.py:118-125`), decay by the time since the **last decay applied**: stamp `metadata["activation_decayed_at"]` on persist, and fall back to `observed_at` when absent. Today it decays by the full time since `observed_at` every tick.
  - Equivalent alternative: closed form from a stored `activation_at_observation`. The stamp is the smaller change.
  - `_compute_activations` (`dynamics.py:296`) takes `max(seed, stored)`. Keep it.
- **Then, in a separate step, the concept re-save:**
  - make `concept_induction` non-write-through;
  - dedupe beliefs by node id at `layer.py:274-276`, so the ephemeral copy and the durable node are not both shown;
  - fix or remove the adapter's own store, which is frozen since process boot (`concept_induction_ctx.py:84`, never calls `snapshot()`).
  - Expected about 4.5 s off each build that still runs.
- **Also missed by the handoff:** the relationship and juniper anchors read "cold" every turn because their freshness falls back to a seed's `observed_at` that is about 23 h old (`layer.py:165-170`). Fix that alongside, so anchors are judged by `_last_materialized_at`, not by the seed's age.
- **Data touched:** activation, recency and `dormant` on every substrate node. No private content.
- **Proof it worked:**
  - Falkor reads of the two seed concepts every 30 s for 30 min **with the re-save disabled** show a smooth half-life curve, not a 2%/tick cliff.
  - A count of nodes flipping `dormant` per day before and after.
  - The first wave of nodes crossing dormancy will be a real behavior change for curiosity and attention consumers (`graph_cognition/features.py:85`). Name them in the PR.
- **Dangerous failure:** a wrong stamp could freeze decay (nothing ever fades) or double it. Unit tests must cover a missing stamp, a clock going backwards, and a node written mid-tick.
- **Rollback:** env flag `SUBSTRATE_DYNAMICS_DECAY_MODE=since_last|legacy`, shipped `since_last`.
- **Order matters:** decay fix live and verified → then the concept producer goes ephemeral. Never the reverse.

### L7. PROPOSAL: turn off the reply writer's auto-memory for chat turns

- **What happens today:**
  - Claude Code keeps an auto-memory per working directory and loads its index into every session.
  - Chat replies, curiosity investigations, urgent investigations, self-inquiry and mutation runs all share `cwd=/mnt/orion-fcc/repo`, so they share one memory: 24 topic files, 71 KB, "prediction_error" ×19. 39 transcripts wrote to it, mostly investigation runs.
  - Every chat reply reads those notes with no gate, review or provenance.
  - This is one of two real sources of the "watching the prediction errors dance on the bus" talk. The other is the reverie line, L3.
- **Correction to the handoff:** it is not loaded via `HARNESS_FCC_SETTING_SOURCES`. It is keyed by working directory.
- **Change:** set `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` in `_build_subprocess_env` **only for chat-reply turns**. Investigations keep theirs.
  - Alternative that pairs with L5(b): a chat-specific `CLAUDE_CONFIG_DIR`.
- **Capability change:** chat replies lose a side channel to Orion's investigation notes. Anything that should carry into conversation must come through recall or crystallization, which are governed.
- **Data touched:** none deleted. The memory directory stays for investigation runs.
- **Proof:**
  - the reply writer's transcript for a chat turn shows no memory index load;
  - the count of "prediction error" or "bus synaptic" phrases in Hub replies over 7 days, before and after (`harness_turn_trace.run_artifact->>'final_text'`).
- **Dangerous failure:** a chat answer that relied on a recent investigation finding loses it. Mitigation: the investigation's findings already reach recall via its journal.
- **Rollback:** remove the env var.
- **Open for Juniper:** should investigation runs keep an ungoverned shared memory at all? That is a separate question and out of scope here.

### L8. PROPOSAL: show the draft first, judge after

- **Facts** (`harness_turn_trace`, last 7 days, 483 finalize runs):

  | Outcome | Runs |
  |---|---|
  | aligned, no change | 298 |
  | aligned, text changed by deterministic grounding/cleanup | 104 |
  | misaligned/uncertain → repair rewrote | about 70 |
  | judge LLM failed, fallback | 11 |
  | quick-lane skip fired | 1 |

  - Repair reasons: strain_unresolved 81, misaligned 36, uncertain 11.
  - Real catches: "You —" → "Juniper —"; an overclaim about running containers softened to what was actually checked.
  - The judge is worth keeping; loosening its gate is withdrawn.
- **Change:**
  - Deterministic grounding/cleanup still runs before display; it costs no LLM call.
  - The Hub shows the draft as soon as the reply writer finishes. The judge then runs. If it triggers repair, the Hub replaces the message in place and marks it "revised", showing the reason.
- **Capability change:** Juniper sometimes sees a reply, then a revision. Today she sees only the final.
- **Proof:**
  - time-to-first-visible-reply drops by about 11 s (Hub log);
  - revision rate matches the rate at which repair actually changed text (94 of 483 runs, about 19%).
- **Dangerous failure:** a misaligned draft is visible for about 10–20 s before revision. For identity- or boundary-sensitive turns (`trust rupture`, `boundary flag` inputs already computed in `finalize.py:275-352`), keep today's judge-before-display.
- **Rollback:** a Hub flag back to today's behavior.
- **Open for Juniper:** whether seeing revisions is acceptable UX.

## Missing questions

1. L7: auto-memory off for chat replies, yes or no?
2. L8: show the draft first and revise in place, yes or no? And should sensitive turns stay judge-first (recommended)?
3. L6: approve the decay fix as written (proposal mode: cognition-loop change)?

## Proposed schema / API changes

- **L4:** new bus RPC `stance_context_prepare` (request/reply) on cortex-exec, with a channel in `orion/bus/channels.yaml`, a schema in `orion/schemas/` and a registry entry.
- **L8:** a Hub-side message-revision event to the client (existing websocket), plus a `revised_reason` field. Contract to be named in the L8 implementation PR.
- **L6:** new node metadata key `activation_decayed_at` (additive, inside the existing metadata dict).
- **Env:**
  - `SUBSTRATE_DYNAMICS_DECAY_MODE` (L6);
  - possibly `HARNESS_FCC_CHAT_DISABLE_AUTO_MEMORY` (L7, shipped on);
  - all `.env_example` updates synced to local `.env` (CLAUDE.md §7).
- L1–L3: no schema, bus or env change.

## Files likely to touch

- **L1–L3:**
  - `services/orion-cortex-exec/app/{router.py,executor.py,chat_stance.py,current_turn_llm_signals.py,main.py}`
  - `orion/substrate/relational/registry.py`
  - `orion/situational/reverie_reader.py`
  - tests under `services/orion-cortex-exec/tests/`
- **L4:**
  - `services/orion-thought/app/bus_listener.py`
  - `services/orion-cortex-exec/app/{main.py,executor.py}`
  - `orion/bus/channels.yaml`, `orion/schemas/`
- **L5/L7:** `orion/harness/fcc_motor.py`, `orion/fcc/claude_spawn.py`, `services/orion-harness-governor/{.env_example,app/settings.py,docker-compose.yml}`
- **L6:**
  - `orion/substrate/dynamics.py`, `orion/core/activation_decay.py`
  - `orion/substrate/relational/{layer.py,registry.py,adapters/concept_induction_ctx.py}`
  - `services/orion-substrate-runtime`
- **L8:** `orion/harness/finalize.py`, `orion/hub/turn_orchestrator.py`, Hub templates/JS

## Non-goals

- No change to field-attention normalization (parked by Juniper 10-02), the attention broadcast, the stance prompt, or orion-mind's internals.
- No change to which other verbs run the build (journal, skills, self_study); that is a follow-up.
- No thinking budget for stance_react (Juniper said no). No change to the 8,000 max_tokens cap.
- The cooling shed is fixed in the thermal spec, not here.

## Acceptance checks

1. **L1:** `router_skip_prepare_brain_reply_context` logged for both verbs. Finalize-leg time from request to cortex-exec result drops by about 9 s, median over 20 turns before and after. One `cortex_turn` row per turn.
2. **L2:** no cross-request reply delay during a build. The gateway publish → cortex-exec receive gap stays under 100 ms for concurrent corrs.
3. **L3:**
   - zero `producer_materialize_failed identity_yaml` lines;
   - stance identity lines unchanged (11 Orion lines);
   - the probe sends `gateway_read_timeout_sec`;
   - the reverie block has no repeated line.
4. **L4:** mind-done → stance-start gap is under 1 s on turns where mind takes ≥ 9 s.
5. **L5:** spawn → first stream event, measured before and after.
6. **L6:** a smooth decay curve on the seed concepts with the re-save off; then the build drops from about 9 s to about 4.5 s.
7. **L7/L8:** per their own proof lines above.
8. **Whole turn:** median Hub turn end-to-end, measured over a week, from the existing hub timing logs. Measured on nights **without** a cooling shed, so the thermal fix doesn't contaminate the number.

## Recommended next patch

One PR: **L1 + L2 + L3** (cortex-exec plus two small shared-library edits). No proposal needed; no schema, bus or env changes. Then L5's measurement spike, then L4. L6, L7 and L8 wait for Juniper's answers above.
