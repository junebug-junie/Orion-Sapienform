# Handoff: why a unified chat turn takes about a minute, and what to fix

Date: 2026-10-06. Author: Claude (session with Juniper). Status: **diagnosis done, nothing changed yet.**
> **SUPERSEDED (2026-10-06):** recommendations corrected by `docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md`. Do not implement Rec 1 or Rec 3 as written here: Rec 1 removes the only thing currently undoing a compounding decay bug, and Rec 3 saves nothing because the build re-runs at step time.

Scope: only the production path, the unified Hub turn (Hub → orion-thought stance → harness/FCC reply writer → final polishing step). chat_general, chat_quick and chat_kids_story are debug-only and out of scope.

## The problem in plain words

1. **A turn is slow.** A typical unified turn takes about 68 seconds. About 13 of those seconds is pure overhead. Part of that overhead also freezes cortex-exec for everyone else while it runs.
2. **Orion keeps bringing up "prediction errors"** in casual check-ins ("watching the prediction errors dance on the bus").
3. **Already fixed (context only):** on 10-02 a turn failed outright ("Turn deferred: stance_react exec result missing thought payload"). The stance model looped while copying a 36-character turn ID until it hit its 8,000-token cap. PR #2478 fixed that and is deployed and verified live.

## Where the minute goes

Reference turn `d6479024-3bab-4bf5-a4d3-ffe7e5501b72`, 2026-10-02 18:10 UTC, 68s end to end. Shape confirmed on other turns (`3a23320f`, `8622fa5e`, `ebbc3329`, `1e362242`, `5482a5c2`).

| Step | What it is for | Time | Real or waste |
|---|---|---|---|
| Turn appraisal | classifies the turn | 0.9s | real |
| orion-mind, 3 serial small-model calls | Orion's attention coloring for the stance | 10.7s | real |
| **Stance context build: memory-graph reload, twice** | builds the stance's inputs | **~9s** (p50 8.7–9.2s over 60+ turns) | **mostly waste** |
| Stance model call | how to approach the turn | 10.4s | real |
| Memory recall (two lookups) | grounding | ~1s | real |
| Claude Code CLI startup in harness-governor | reply-writer boot | 2.4–4.4s | unclear, cause UNVERIFIED |
| Reply-writer model call (FCC motor via gateway Anthropic passthrough) | writes the draft | 10–31s | real |
| **Same context build again before polishing** | output never used | **4.5–12s** | **waste** |
| Final polishing model call (`harness_finalize_reflect`) | polishes the draft | ~11s | real |

## Findings, with evidence

### F1. The stance context build reloads Orion's whole memory graph twice and freezes cortex-exec

- **Path:** `services/orion-cortex-exec/app/router.py:1056` → `prepare_brain_reply_context` (`executor.py:4804`) → `build_chat_stance_inputs` (`chat_stance.py:2514`). `hydrate_felt_state_ctx` and `_unified_beliefs_for_stance` run **synchronously on the event loop**.
- **Two reloads:** `orion/substrate/relational/layer.py` `beliefs_for_stance` calls `self._store.snapshot()` before the producer fan-out (~:151) and after it (~:257). `FalkorSubstrateStore.snapshot()` (`orion/substrate/falkor_store.py:686`) fully rehydrates the graph (4,946 nodes / 37,305 edges). One read-only rehydrate measured in the container took **4.05s**.
  - **First reload:** the 30s refresh ceiling. `SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC=30.0` is live, and turns are minutes apart, so it always fires.
  - **Second reload:** the `concept_induction` producer (write-through tier) re-saves concepts through the store, and `upsert_node` bumps the write-generation counter unconditionally (`falkor_store.py:664/684`). The next `snapshot()` then rehydrates again. The `relationship` and `juniper` anchors are cold in 64/64 turns, so this happens every turn.
- **Freeze evidence:** during the reference turn, a concurrent reverie reply was published by the gateway at 18:11:00.482 but received by cortex-exec at 18:11:08.073, the instant the build finished. A `metacog_trend_cue_fetch_timeout` set to 0.8s fired after 8.4s on cortex-exec-background.
- **Timing split:** the two halves are about 4.3–4.75s and about 4.3–4.5s (e.g. corr `5a240233`). The p10 of about 4.5s matches turns where only one reload ran.

### F2. The concept "re-save" adds nothing

Juniper asked: "isn't the resave how concepts cement through repetition?" Checked: **no.**

- `orion/substrate/relational/adapters/concept_induction_ctx.py` reads up to 64 **existing** concept nodes (`query_concept_region(limit_nodes=64)`) and returns them unchanged, apart from a tier label (`tier_rank=3`) and a default `concept_type`.
- The durable materializer merges each one into itself through `orion/substrate/reconcile.py` `merge_node`. That function keeps the **max** of existing vs incoming for confidence, salience, activation, recency and observed_at. Both sides hold the same values, so nothing increments. The `materialization_lineage` it appends is not persisted to Falkor (it is not among the stored node keys).
- Real reinforcement lives elsewhere (`contributing_turn_ids_json`, `promotion_state` on the node) and is not touched by this path.
- **Confidence:** proven from code. A **live before/after check of a concept's stored values across one turn has NOT been done**. Do that first (see acceptance checks).

### F3. The context build before the polishing step is unused

- `router.py:1034-1056` runs `prepare_brain_reply_context` for every brain-mode verb not on a skip list, including `harness_finalize_reflect` and `orion_response_repair`.
- **Nothing reads it:**
  - **Templates:** `harness_finalize_reflect.j2` and `orion_response_repair.j2` use only `draft_text`, `grammar_receipts`, `substrate_appraisal`, `thought_event`, `tool_execution`, `user_message` and the two overlays. Neither verb YAML declares `personality_file`.
  - **Callers:** `orion/harness/finalize.py:454` (`extract_finalize_reflection_payload`) and `:921` read **text only**, so result metadata (autonomy slice, grounding capsule) is dropped.
  - **Probe:** `current_turn_llm_signals` is skipped on these verbs (`reason=non_chat_verb`).
  - **Memory writes:** producer write-through is a TTL refresh only, and the next stance step does it anyway.
  - **Memory block:** the "relevant memory" prompt block comes from recall, not this build.
  - **Hub panel:** the Hub's `extract_autonomy_payload` ran 0 times across 58 harness turns since the 10-05 restart.
- **Side effects that disappear, all duplicates:** over 72h, all 60 finalize correlation ids are also stance correlation ids.

  | Table | Rows (60 turns) | Rows lost by the skip | Note |
  |---|---|---|---|
  | `chat_stance_belief_log` | 131 | ~71 | duplicates; `self_study.py:674` reads it with no dedup, so they halve its window |
  | `substrate_attention_schema` `cortex_turn` | 131 | ~71 | `shift_kind` identical in 60/60; `attended_id` differs in 2/60 |
  | `attention_salience_trace` | 62 | ~2 | |

  Juniper's standing requirement of one `cortex_turn` row per real chat turn (`docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md` ~535-548) is still met by the stance leg.
- The polishing step's build also does the double reload, so F1 and F3 overlap (see Expected outcome).

### F4. The identity producer fails on every cold turn

- The `identity_yaml` producer (operator_static tier, write-through) fails with `durable writes support concept, evidence, entity nodes only; got node_kind='state_snapshot'` when the `orion` anchor is cold: 34/64 cortex-exec turns and 75 on background.
- It raises before any write, so **it does not cause the reloads.** The stance still gets identity lines from the ctx fallback (`orion_count=11`), only by accident.
- Accepting `state_snapshot` durably would make things slower: one more write and one more reload. It would also need new codec work: `DURABLE_NODE_KINDS` was narrowed deliberately in 8491943b6 / ac67621e9.

### F5. Where the "prediction error" talk actually comes from

The reply-writer's transcript for the 10-02 04:08 turn was recovered (`a1b478f2-9b36-43ab-af9f-1935cab1f123`, harness-governor container `/root/.claude/projects/-mnt-orion-fcc-repo/10c4fa26-….jsonl`).

- The phrase "watching the prediction errors dance on the bus" was already in the draft; the polishing step did not add it.
- The reply-writer's prompt contains "Biometrics" 0 times and "suppress" 0 times. The per-turn attention frame's concept-induction channel **did not reach the reply.**
- **Real sources:**
  1. **The situation brief's reverie line**, appearing twice: "The coalition is fixated on … bus synaptic prediction error" (`orion/situational/context.py:2302`). Upstream, field attention normalizes salience across prediction-error targets only, so a calm node always scores 1.0 (`orion/attention/field_attention/selectors.py`, `normalize_across_targets`). Over 6h of frames: bus_synaptic was dominant 9,530 times, biometrics 7,753, chat 349. That feeds reverie.
  2. **The reply-writer's own auto-memory file** `/root/.claude/projects/-mnt-orion-fcc-repo/memory/MEMORY.md`, loaded every turn via `HARNESS_FCC_SETTING_SOURCES=user,local`. It contains "Bus synaptic oscillation verified", `_write_prediction_error_node()` and similar.
- Separately, the per-turn attention frame selected `suppress` 4,744 times in 7 days (`substrate_attention_schema`, `process='cortex_turn'`), always on "Biometrics prediction error", via the concept-induction detector. `node:substrate.biometrics` is stored as `node_kind=concept` (852 concepts). This is noise, but it is **not** what makes Orion say it.

### F6. Smaller items

- **Abandoned probe holding the GPU (10-02 08:42).** The `current_turn_llm_signals` probe (`current_turn_llm_signals.py:425-455`) sends no `gateway_read_timeout_sec`, so the gateway assumes 700s (`llm_backend.py:156-167`).
  - The real waste is a probe that queues for a lease after cortex-exec already gave up at 3s, then generates for nobody. The stance call behind it waited 67s.
  - Since 10-01 the probe runs only on human chat turns: 4 runs and 0 failures since 10-05. **Rare.**
  - Whether cutting the socket stops llama.cpp generating is UNVERIFIED.
- **Reply-writer call shows `corr=None` in gateway logs.** This is only the log line. 3,603/3,657 `http:anthropic` leases carry `hold_lease_id`, which joins to `turn_correlation_id`. No new tagging is needed.
- **Why 154 tokens took 70s on 10-02.** UNVERIFIED. Lead: the 35B chat model is split across GPU0+GPU3, GPU3 is shared with the fast/metacog models (which topic-foundry was flooding at the time), and the pool counts chat as gpu0 only. Needs `nvidia-smi` on circe.
- **orion-mind "missing result".** Not a failure. `semantic_synthesis` uses the native-completion path (`llm_backend.py:789-920`), which does not log `provider_result`.

## Recommendations

### Do now, as one PR (no change to what Orion thinks)

1. **Stop the concept re-save.** Make the `concept_induction` producer non-write-through (`SNAPSHOT_EPHEMERAL` in `orion/substrate/relational/registry.py`). It copies substrate→substrate, so writing back serves nothing (F2). This removes the generation bump, so the second `snapshot()` hits cache: same data, about 4.5s saved per build.
2. **Run the build off the event loop, on a dedicated single-worker executor.** Not the shared default `asyncio.to_thread` pool. A single worker keeps today's one-at-a-time ordering, and avoids concurrent `beliefs_for_stance` races (unlocked `_last_materialized_at`, cache swap during upsert) and the hidden FIFO with other `to_thread` users. This stops the freeze. Thread safety was checked: the work is synchronous Postgres plus redis-py, and `to_thread`/executor copies contextvars.
3. **Skip the build for `harness_finalize_reflect` and `orion_response_repair`** in `router.py` (gate strictly by verb name). Add a test that the stance leg still writes exactly one `cortex_turn` row per turn.
4. **Make `identity_yaml` ephemeral** (`SNAPSHOT_EPHEMERAL`, `pull_on_cold=False`). This removes the every-turn error (F4). Before shipping, check that `_project_identity_from_beliefs` output matches today's ctx-fallback identity lines (line caps, self-definition stripping), since the stance would start reading the projected path.

### Optional, low priority

5. The probe passes `gateway_read_timeout_sec=current_turn_signal_probe_timeout_sec` (one line). Rare event. Do **not** add a gateway-wide default: the envelope `ttl_ms` is not set or read anywhere.
6. The gateway copies the hold's `turn_correlation_id` onto the child `http:anthropic` lease log line, purely for log readability.
7. Decide whether to raise the 30s snapshot refresh ceiling. That trades staleness from other processes' writes for about 4s more per turn. Needs Juniper's call.

### Separate small job: the prediction-error talk (F5)

- Look at the situation brief's reverie line (`orion/situational/context.py:2302`) and the reply-writer's auto-memory file. These are what actually put the words in Orion's mouth.
- The deeper cause, field-attention normalization always crowning a prediction-error node, was parked by Juniper on 10-02 ("not ready to touch the broader attention architecture").

### Rejected after adversarial review (do not redo)

- **Move the per-turn attention frame into the Hub / merge it with the attention broadcast.** The frame build is about 15ms, so the move saves nothing. Its placement is deliberate:
  - `docs/superpowers/specs/2026-07-11-autonomy-v2-closed-loop-wiring-design.md:29,74`;
  - `2026-09-28-stance-imperative-scope-design.md:70`;
  - `2026-09-04-attention-schema-surface-design.md:535-548` ("cortex must still be the kickoff").

  It is also the only source of a selected `ask` (the broadcast is `max_asks=0`, `attention_broadcast.py:11,223`). It picks the recall profile: 63/63 turns `open_loop`, via `pcr_chat_memory.py:274` → `retrieval_intent.py`. And merging it into the broadcast would loosen evidence validation (`orion/thought/coalition.py:17`).
- **Remove the concept-induction attention detector to stop prediction-error talk.** It does not reach the reply (F5). Might still be worth retiring as noise, but that is a cognition change and needs a proposal.
- **Lower the 8,000 max_tokens cap.** Since #2478, 31 stance runs all finished on their own (p50 291 tokens, max 6,008). The cap is a runaway brake, not the problem.
- **Thinking budget for stance_react.** Juniper said no.

## Expected outcome

- Item 1 alone: about 4.5s off each of the two builds (stance and polishing).
- Item 3 alone: about 9s off the polishing leg (its whole build).
- Items 1 + 3 together: about **13.5s per human turn** (they overlap; not additive to 18s). The polishing leg is on the critical path: the Hub gets the reply right after it returns.
- Item 2: no seconds saved for the turn itself, but cortex-exec stops freezing other work (reverie, metacog, journal) for about 9s per build.

## Acceptance checks

1. **Before any code:** read one concept node's `confidence, salience, activation, recency_score, observed_at` in `orion_substrate` (FalkorDB container `orion-athena-falkordb`), let one stance turn run, read again. Expect no change from the re-save. If values change, stop: F2 is wrong.
2. **Timing:** median of `plan_start` → `chat_stance_inputs_ready` on stance turns drops from about 9s to about 4.5s, and the polishing leg's build disappears (log line `router_skip_prepare_brain_reply_context` for those verbs). Measure the same log timestamps before and after.
3. **Freeze:** no cross-request reply delays during a stance build. Compare gateway publish vs cortex-exec receive timestamps for concurrent corrs.
4. **No loss:**
   - exactly one `cortex_turn` row per real chat turn;
   - concept beliefs still present in the stance's unified-beliefs lineage;
   - identity lines in the stance prompt unchanged (11 Orion lines).
5. **Error gone:** no `producer_materialize_failed identity_yaml` lines.

## Non-goals

- No change to field-attention normalization, the attention broadcast, orion-mind, or the stance prompt.
- No change to which verbs get the build beyond the two polishing verbs. Other background verbs (journal 191/24h, skills, substrate, self_study) also run it. That is a follow-up, untraced.

## Rollback

Each item is a small, independent revert:
- items 1 and 4: registry tier change;
- item 2: executor wrapper;
- item 3: verb skip set.

No schema, bus, or env changes are expected. If one turns out to be needed, the env-parity rules apply.

## Related

- PR #2478: stance turn-ID spiral fix (deployed 10-02, verified).
- Main commits since 10-02 that touch this area: probe human-turn-only gate (`fix/probe-real-chat-only`, merged 10-01), #2476 FCC prompt-cache fix.
- Proposal #2474 (open): reverie prediction-error magnitude.
