# DriveEngine + concept-induction — full trace and deactivation considerations

**Mode:** Design / decision brief. No code changed while producing this doc. Written to consolidate a full session of live tracing before Juniper decides whether to burn the whole `orion/spark/concept_induction/` package down.

**Status of claims below:** everything here was independently traced against running containers (`docker logs`/`docker exec`/live debug endpoints) on 2026-07-11, not inferred from code alone. Two earlier passes in this same investigation got specific consumer claims wrong (see §5) — corrected versions only are included below; the wrong intermediate claims are noted so they don't get re-asserted later.

---

## Arsonist summary

Three things bundled into one Redis-bus worker (`ConceptWorker`, `orion/spark/concept_induction/bus_worker.py`) that are not equally real:

1. **`DriveEngine`** — real, live, running on every accepted bus event, feeding real persisted goals. Not theater.
2. **Concept extraction/clustering** (`ConceptProfile`) — technically live but its output is dominated by literal pronouns ("i" at weight 1.0), diluting real content phrases with noise. Weak signal, not fabricated.
3. **Identity snapshotting** — a direct transform of DriveEngine's pressures; inseparable from #1 by construction.

The finding that changes the whole calculus: **neither #1 nor #2's output currently reaches anything Orion says or decides in chat.** Both of the two real integration points into chat cognition are switched off in the live config (`CONCEPT_PROFILE_REPOSITORY_BACKEND=local` not `graph`; `AUTONOMY_GRAPH_BACKEND=disabled`), confirmed via 6 hours of live `unified_beliefs_for_stance` lineage logs on the running cortex-exec-chat container — neither `"autonomy"` nor `"concept_induction"` appears as a contributing producer in that window. This is a config/wiring fact independent of whether the underlying math or taxonomy is good.

So the honest framing is not "should we kill something load-bearing" — it's "this stopped reaching Orion at some point (deliberately or by drift), and everything downstream of that point has kept running and writing state into a system nothing currently reads at inference time."

---

## Part A — What the numbers mean and where they come from (background)

### A.1 The six drives — no conceptual grounding exists

`DRIVE_KEYS = (coherence, continuity, capability, relational, predictive, autonomy)` (`orion/spark/concept_induction/drives.py:10`). No docstring, design note, or spec anywhere in the repo argues for this specific set over any other. First commit predates the 2026-05-22 `autonomy-drives-automation` PR; every later spec treats it as fixed ("Six drives stay" — hard constraint in `docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md`).

**Operational semantics** (derived from what actually fires each one — no docstring states an intended meaning):

| Drive | Actual triggers |
|---|---|
| `coherence` | drop in self-state/turn `coherence` score, `spark_signal.coherence` dip |
| `continuity` | novelty spikes, uncertainty deltas, biometric volatility, mesh-health drops |
| `capability` | energy/resource/execution pressure, biometric strain, failure severity |
| `relational` | valence drops, social-hazard signals (cooldown loops, self-message loops) |
| `predictive` | coherence deltas, uncertainty, novelty, world-coverage gaps |
| `autonomy` | novelty, uncertainty, low feedback scores |

`coherence`, `continuity`, and `predictive` draw from largely the same underlying tensions (self-state coherence/uncertainty deltas, novelty) with different weight vectors — three views on one signal, not obviously three distinct constructs. `autonomy` is named for a capacity for self-initiation, but its actual inputs are the same generic distress signals as three other drives; the one mechanism that would earn the name — `orion/autonomy/endogenous_origination.py` — is real, independent math (reads `SelfStateV1` directly, zero coupling to `DriveEngine`) but is `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` by default and its only integration point is `ConceptWorker` itself.

No self-preservation drive exists; the homeostatic-drives spec names this gap explicitly and declines to solve it, mapping somatic/biometric signals onto `capability`/`continuity` as an acknowledged workaround.

**Not concept-induction-derived**, despite the directory name: zero references to `ConceptProfile`/`ConceptProfileDelta` anywhere in `drives.py`/`tensions.py`/`goals.py`. Pure naming-proximity artifact — drives and concept extraction are directory-mates, not a real dependency.

### A.2 Two independent drive-pressure computations share the same six keys

- **`orion.spark.concept_induction.drives.DriveEngine`** — leaky integrator (rebuilt 2026-07-07 to kill a flat-0.731 fixed-point bug — the math is honest now). Durable, persisted. Feeds `GoalProposalEngine.propose()` unconditionally on every accepted bus event.
- **`orion.autonomy.reducer.reduce_autonomy_state`** (`AutonomyStateV2`) — a *pure function, no I/O*. Its `drive_pressures` live only in `ctx["chat_autonomy_state_v2"]` for one chat turn, feeding the chat-stance prompt inputs and a router export, then discarded. This is deliberate: `docs/autonomy_state_v2_reducer.md` states outright *"Not an input to phi features, `build_self_state`, or homeostatic `DriveEngine`"*, and names "Dual pipelines" as an accepted limitation, not a bug. `_DRIVE_KEYS` is independently hardcoded a second time in `orion/autonomy/reducer.py:33` (different tuple order than `drives.py`'s `DRIVE_KEYS`).

A parallel implementation effort (`docs/superpowers/plans/2026-07-11-autonomy-v2-closed-loop-wiring.md`, in progress as of this doc) is making `AutonomyStateV2` persistent and giving it two real consumers (the stance-decision prompt + FCC/Claude harness prefix) — but explicitly keeps it separate from `DriveEngine`/goal-pipeline unification (stated non-goal). After that ships, there will still be two parallel drive-pressure systems, not one — this doc doesn't resolve whether that's the right end state; it's a live open question for whoever owns that plan.

### A.2b Exact drive-pressure math (`DriveEngine.update`, `orion/spark/concept_induction/drives.py:44-89`)

Per drive `d`, on each tick at wall-clock time `now`, given `previous_ts`:

```
elapsed = (now - previous_ts).total_seconds()          # 0 if no previous tick
decay   = exp(-elapsed / τ)                             # τ = decay_tau_sec = 1800s (30 min)
base_d  = prev_pressure_d * decay                        # decays toward 0 with no input

impulse_d = clamp01( Σ_over_tensions  clamp01(tension.magnitude) * clamp01(tension.drive_impacts[d]) )

pressure_d = clamp01( base_d + impulse_d * (1 - base_d) )   # headroom-scaled accrual, bounded [0,1]

activation_d = pressure_d >= 0.62   (if was inactive)  — hysteresis
             = pressure_d >= 0.42   (if was active)
```

This is the leaky-integrator rebuild (2026-07-07) that replaced the old `soft_saturate(x) = 1 - exp(-1.8x)` formula, which had a stable non-zero fixed point at ≈0.731 that pinned every drive under frequent ticks. The legacy path is still present (`leaky_math_enabled=False` branch) for rollback but is not the configured default.

**What actually produces a `tension.magnitude` and `drive_impacts` weight vector** (`orion/spark/concept_induction/tensions.py`) — the five hardcoded tension types, each firing only on a directional delta (never on a value merely being present):

| Tension | Fires when | `magnitude` | `drive_impacts` (fixed weights, not learned) |
|---|---|---|---|
| `tension.contradiction.v1` | turn-effect `coherence` delta `< 0` | `\|coherence_delta\|` | `{coherence: 1.0, predictive: 0.65}` |
| `tension.distress.v1` | turn-effect `valence` delta `< 0` | `\|valence_delta\|` | `{relational: 0.95, continuity: 0.55}` |
| `tension.identity_drift.v1` | turn-effect `novelty` delta `> 0` | `novelty_delta` | `{continuity: 0.8, autonomy: 0.6, predictive: 0.4}` |
| `tension.cognitive_load.v1` | turn-effect `energy` delta `< 0` | `\|energy_delta\|` | `{capability: 0.9, coherence: 0.4}` |
| `tension.drive_competition.v1` | max−min pressure spread across drives `≥ 0.06`, no turn-effect needed | `spread` | `{top_drive: 0.9, runner_up_drive: 0.75}` |

The `drive_impacts` weights (0.4, 0.55, 0.6, 0.65, 0.75, 0.8, 0.9, 0.95, 1.0) are hand-picked constants in source, not fit or learned from data — this is the concrete mechanism behind the §A.1 observation that `coherence`/`continuity`/`predictive` overlap: `tension.contradiction.v1` alone feeds both `coherence` (1.0) and `predictive` (0.65) from the same single delta.

When no `turn_effect` field exists on the payload, `_extract_turn_effect` (`tensions.py:31-56`) falls back to computing the same four deltas from `phi_before`/`phi_post_after` snapshots — i.e., on the fallback path, drive tensions inherit whatever fragility exists in the phi corpus pipeline (a separate, previously-documented issue: 6/11 seed-v3 phi dims frozen). How often the fallback triggers vs. the primary `turn_effect` field being present was not measured in this pass — UNVERIFIED.

### A.3 Concept induction's actual output — real signal diluted by noise, not fabricated

Live state pulled directly from the running container (`orion-athena-spark-concept-induction`, `/data/concept-induction-state.json`) on 2026-07-11:

```
subject: orion, revision 18171 — 14 concepts / 10 clusters
  i (1.0), food > code (0.34), you (0.34), one (0.33), half (0.33), that (0.33),
  one way (0.33), my attention (0.33), half an allotment (0.33), a quick chat (0.33),
  this (0.33), the generous upgrade (0.33), what (0.33), your mind (0.33)

subject: relationship, revision 2773 — 20 concepts / 15 clusters
  (same set, plus:) lunch, fcc, ooops, your fcc-governor context window,
  some food lol food > code
```

This traces to a real recent conversation (splitting lunch/food, sizing an FCC-governor context window) — the extraction is reading real content, not emitting noise. But the top-weighted "concept" is always the literal pronoun `i` at weight 1.0, with function words (`you, it, that, this, what, one`) ranking at or above genuine content phrases.

**A.3b Exact extraction/weighting math — traced, and the missing filter is confirmed absent, not just suspected.**

*Candidate extraction* (`SpacyConceptExtractor.extract`, `extractor.py:58-88`): for each input text (sentence/turn), run spaCy (`en_core_web_sm`, falls back to a blank tokenizer-only pipeline if the model fails to load), take **NER entities first, then all noun chunks** not already captured as an entity (`_chunk_candidates`, `extractor.py:38-53`); if spaCy yields nothing (blank-model/parser-less case), fall back to a bare regex word-splitter (`_regex_candidates`). Each candidate is lowercased/whitespace-normalized and accumulates a running frequency:
```
weight_for_this_text = 1.0 + (text_index * 0.01)     # tiny recency bump, ~cosmetic
freq[candidate] += weight_for_this_text                # summed across all texts in the window
```
Top 50 by summed frequency survive (`top_k=50`).

**The root cause**: spaCy's `noun_chunks` treats pronouns as noun chunks. `"i"`, `"you"`, `"it"`, `"this"`, `"that"`, `"what"`, `"one"` are syntactically valid one-word noun chunks in nearly every sentence, so they accumulate the highest raw frequency by construction — a first-person conversational transcript will always have `"i"` in nearly every sentence.

*Salience/weight normalization* (`inducer.py:174-199`, this is the `0.33`/`1.0` figures actually stored):
```
max_freq  = max(all candidate raw counts, default 1.0)
salience  = raw_count / max_freq                                    # pure min-max normalize, 0..1
confidence = min(1.0, 0.5 + (0.15 if has_embedding else 0.0) + 0.35 * salience)
```
This is the entire weighting formula: **frequency, min-max normalized to whatever the single most frequent token was that window.** No TF-IDF, no inverse-document-frequency down-weighting of common tokens, no part-of-speech filter, no stopword list. `grep -n "stopword\|is_stop\|PRON"` across every file in `orion/spark/concept_induction/` returns **zero hits** — confirmed absent, not merely uncaught. Since `"i"` is nearly always the single most frequent token in a chat transcript, it becomes the `max_freq` denominator and therefore always scores `salience=1.0` by construction, and its `confidence` is also inflated by the same `0.35 * salience` term — the noise gets the highest confidence too, not just the highest weight.

*Clustering* (`ConceptClusterer.cluster`, `clusterer.py:33-86`): single-pass greedy, order-dependent, no global optimization. Each candidate is compared only against each existing cluster's **first member** (the "anchor") — first cluster whose anchor clears the threshold wins, otherwise start a new cluster:
```
if embeddings available:  join if cosine(candidate_vec, anchor_vec) >= 0.8
else:                      join if jaccard(candidate_words, anchor_words) >= 0.6    # word-overlap ratio
```
This explains the observed clusters like `half, half an allotment` and `you, your mind` (word-overlap) — and also explains why clustering quality is downstream-fragile to the extraction step above: garbage candidates (pronouns) cluster just as confidently as real ones, since nothing in the clustering step evaluates candidate quality, only pairwise similarity.

**Bottom line on the metrics themselves**: the drive-pressure math (§A.2b) is honest arithmetic over hand-picked-but-legible weights — you can read a `drive_impacts` table and know exactly what produced a given pressure value. The concept-extraction math (above) is honest arithmetic too, but over an *upstream candidate set with no relevance filter at all* — the formula is simple and correct, the input to it is not curated. A five-line stopword/pronoun filter in `extractor.py:_chunk_candidates` or `inducer.py`'s salience computation (excluding `spacy`'s built-in `token.is_stop`/`token.pos_ in {"PRON", "DET"}`) would likely fix this without touching the frequency/clustering math at all — worth knowing as a cheap alternative to full deletion of extraction, independent of what happens to `DriveEngine`.

### A.4 Live/dead status — corrected twice during this investigation, final state below

Two earlier claims made during this trace were wrong and are recorded here so they don't get re-asserted:

1. **Wrong claim**: "`AutonomyStateV2` is RDF-persisted and feeds `capability_policy` drive-origin gating." **Correct**: it's a pure function with no I/O at all; its output never leaves `ctx` for one chat turn.
2. **Wrong claim**: "The real `capability_policy` consumer is gated at `orion-social-room-bridge/app/service.py:801`." **Correct**: that line constructs an unrelated `PolicyContext` class (governs whether Orion speaks in a social room; no `.goal` field, no connection to drives at all). The real consumer is `CapabilityEvaluationContext.goal`, used by `orion/autonomy/policy_act.py` and `orion-world-pulse/curiosity.py`.

**Final, verified state**: `DriveEngine` is live — 1016+ real bus events processed in one hour on the running container, zero exceptions, `GoalProposalEngine.propose()` called unconditionally on every accepted event, real `GoalProposalV1` objects persisted to RDF with a computed `drive_origin`. **Not dead.** But:

- `orion-substrate-runtime`'s attention goal-context listener propagates the *real* computed `drive_origin` into `GoalContextStore` — wired, but nothing reads that store (`ORION_ATTENTION_TOPDOWN_ENABLED=false` live). Inert.
- `policy_act.py`/`curiosity.py`'s capability gates are live and flag-enabled, but both hardcode `drive_origin="predictive"` into a synthetic goal object — DriveEngine's actual per-tick computation never reaches this gate either.
- `orion-hub`/`orion-cortex-exec` promote/execute API can turn a proposed goal into `planned`/`executing`, but requires an operator token — human-gated, not autonomous.

**Lesson recorded for future tracing in this area**: this subsystem has enough near-identically-named classes (two unrelated `PolicyContext` classes, two `_DRIVE_KEYS` tuples, two unrelated `IdentitySnapshotV1` classes — see §5) that static grep-only tracing produces confident-sounding wrong answers. Verify with live runtime evidence (container debug endpoints, logs) before asserting a consumer chain.

---

## Part B — Deactivation considerations

### B.1 The one hard blocker

`orion/spark/concept_induction/graph_query.py` (`GraphQueryClient`/`GraphQueryConfig`/`GraphQueryError`) is a **generic SPARQL HTTP client with no concept/drive-specific logic**. `orion/autonomy/repository.py:14` imports it at module top level; `services/orion-cortex-exec/app/chat_stance.py:23` imports `orion.autonomy.repository` eagerly, not lazily. **Deleting the package wholesale breaks import of `chat_stance.py` — cortex-exec's chat-serving module won't boot.** This must be extracted (e.g. to `orion/autonomy/` or a new shared `orion/graph/` module) before any deletion, independent of how much else is removed.

### B.2 Full downstream consumer map

**Identity snapshotting** (`identity.py`) is a direct transform of `DriveEngine`'s pressures (`drive_pressures=drive_state.pressures`, `identity.py:160`) — not separable from DriveEngine by construction. Its output (`orion.core.schemas.drives.IdentitySnapshotV1`, channel `orion:memory:identity:snapshot`) is materialized to RDF by `orion-rdf-writer` and read by `orion/autonomy/repository.py` (sets `latest_identity_snapshot_id`) and `orion/reasoning/adapters/autonomy.py`. If it stops, `orion/autonomy/reducer.py`'s `"no_identity_snapshot"` unknown-flag becomes permanent — a visible, non-crashing degradation.

**Naming collision to not confuse during removal**: there is a second, unrelated `IdentitySnapshotV1` at `orion/schemas/identity_snapshot.py`, produced independently by `orion-self-state-runtime` with its own Postgres table, zero involvement from concept-induction. Untouched by this change either way.

`dossier.py` (`TurnDossierV1`) is a correlation-bookkeeping layer over artifacts produced entirely by DriveEngine/GoalProposalEngine/identity.py in the same worker tick — structurally dead without them, no independent value.

**Concept-profile RDF/vector output**: registered as a chat-context producer (tier `CONCEPT_INDUCED`) in both `chat_stance.py` and `orion/cognition/projection_builder.py` — genuinely wired to matter, *if* `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph` were ever the active backend. It is not, live (`=local`), and cortex-exec-chat has no shared store file with the worker container — so today this is a dead end at runtime by config, not by absence of wiring. No independent memory-recall/search system reads concept-profile RDF nodes.

**Drives/goals in chat context** (`orion/substrate/relational/adapters/autonomy_ctx.py`): same pattern — wired into the same two chat-facing registries, but `AUTONOMY_GRAPH_BACKEND=disabled` live means it's currently skipped (`autonomy_graph_backend_blocked`, `fallback=skip_adapter`, confirmed in live logs). Same dark-by-config status as concept profiles.

**Hub UI**: the Autonomy Debug Panel (`services/orion-hub/templates/index.html#autonomyDebugPanel`, rendering logic in `static/js/app.js`) is a real, visible, but debug-only UI element — shows dominant drive, top drives, goal presence, and an alignment note per chat turn. If DriveEngine stops, it doesn't break; it permanently shows its existing degraded-state copy ("unavailable — Orion drives skipped or timed out").

**`capability_policy.py`'s `required_drive_origins` gate**: continues to function unaffected by DriveEngine's removal either way, since both live callers already hardcode a synthetic `drive_origin="predictive"` rather than reading the real computed value.

**`endogenous_origination.py`**: independent math (reads `SelfStateV1` directly), but `ConceptWorker` is its only caller anywhere in the repo. Currently flag-disabled, so no live behavior change from removal — but the capability disappears for future re-enablement unless rehomed to a different worker first.

**Bus channel fan-in**: most of `ConceptWorker`'s ~19 subscribed channels have other real consumers and stay meaningful without it (`orion:chat:*`, `orion:collapse:sql-write`, `orion:metacognition:tick`, `orion:cognition:trace`, `orion:substrate:self_state`, `orion:signals:*`). One channel becomes a produce-into-the-void: `orion:feedback:frame` (`orion-feedback-runtime` is its sole producer, `ConceptWorker` its only documented consumer).

### B.3 Registry/contract cleanup inventory (list only, not fixed)

- **Bus channels** (`orion/bus/channels.yaml`), all sole-produced by `orion-spark-concept-induction`: `orion:memory:concepts:profile`, `orion:memory:concepts:delta`, `orion:memory:drives:state`, `orion:memory:drives:audit`, `orion:memory:identity:snapshot`, `orion:memory:goals:proposed`, plus dossier/graph-materialization channels.
- **Schema registrations** (`orion/schemas/registry.py`) that lose their only producer: `ConceptProfile`/`ConceptProfileDelta`, `DriveStateV1`/`DriveAuditV1`/`IdentitySnapshotV1` (concept_induction's), `GoalProposalV1`, `TurnDossierV1`. **Caution**: `TensionEventV1` must stay registered — it's also actively constructed by `orion/autonomy/signal_tension.py`, `substrate_metabolism.py`, and `endogenous_origination.py`, independent of concept-induction.
- **RDF predicates** in `services/orion-rdf-writer/app/autonomy.py` become write-once-then-permanently-stale: `ORION.anchorStrategy`, `ORION.snapshotSummary`, `ORION.drivePressure`, `ORION.driveDimension`, `ORION.hasDriveAssessment`, `ORION.goalStatement`, `ORION.driveOrigin`, `ORION.proposalSignature`, etc.

---

## Part C — Proposed unification direction (Juniper's lean, v1, still riffing — no code written)

Setting `DriveEngine`/concept-induction deletion aside. Comparing purely on mechanism (§ recent chat discussion, not reproduced in full here): `DriveEngine`'s wall-clock leaky-integrator decay is the theoretically correct shape for a "drive" (relaxes toward baseline absent renewed cause); `AutonomyStateV2`'s reducer has **no decay term at all** — pressure only ever accumulates (`_fold_tension_into_pressures`, `min(1.0, out[drive] + added)`), reset only on a cold start. Conversely, `AutonomyStateV2`'s typed evidence contract (`user_turn`/`infra_health`/`reasoning_quality`/`relational_signal`, each with an explicit emit-condition and an explicit "moves pressure?" column) is a more disciplined, better-designed input signal than `DriveEngine`'s own diffuse five-tension sourcing — and its candidate-impulse / inhibited-impulse / `unknowns` structure is a richer appraisal-and-inhibition model than `DriveEngine` has at all.

**The lean**: keep `AutonomyStateV2` computing exactly what it computes today (evidence, candidate impulses, inhibitions, unknowns — "preserve current function in the chat session"), but stop letting it fold its own no-decay pressure. Instead, route the `TensionEventV1` objects it already produces (`chat_evidence_to_tension()` in `signal_tension.py` already emits the same schema `DriveEngine.update()` consumes — this is a routing change, not a rewrite) into `DriveEngine`'s tension stream, so the leaky-integrator decay becomes the one source of pressure truth for both systems. `dominant_drive`/`active_drives` in `AutonomyStateV2` would then read from `DriveEngine`'s resulting pressure rather than being locally folded.

**Why the "hard isolation" rule exists, traced from Juniper directly (not previously documented in code)**: the constraint in `docs/autonomy_state_v2_reducer.md` — *"Not an input to phi features, `build_self_state`, or homeostatic `DriveEngine`"* — exists specifically to avoid **multicollinearity in the phi autoencoder's training corpus**. If chat-turn evidence moves homeostatic drive state, and that state (or something derived from it) correlates with features the phi encoder trains on, you get redundant/correlated training features. Quick sanity check done during this pass: `drive_pressures`/`AutonomyStateV2` do not currently appear anywhere in phi feature-building code (`orion/spark_encoder/`, `orion-spark-introspector`) — so the risk isn't "drive pressure is literally a phi input today." The more likely mechanism is indirect: both phi's seed features and `DriveEngine`'s tensions draw from overlapping raw material (turn-effect deltas, `SelfStateV1` dimensions) — routing V2 evidence into `DriveEngine` could make the same underlying turn-level signal reach the phi corpus through two correlated derivation paths instead of one. **Not traced yet**: exactly which `SelfStateV1` dimensions phi's seed features consume vs. which ones `DriveEngine`'s `extract_tensions_from_self_state` consumes — that overlap (or lack of it) is the concrete next step to actually resolve this, not just mitigate it.

**Agreed v1 approach**: async only — `AutonomyStateV2`'s routed tensions publish to the bus for `DriveEngine`/`ConceptWorker` to consume on its own cadence, not a synchronous same-turn call. This does **not** resolve the multicollinearity risk (going async changes *when* correlated signal arrives, not *whether* it's correlated) — it's a deliberately conservative v1 specifically so the isolation rule isn't silently violated before the actual multicollinearity question is answered. Must be labeled loudly as v1/experimental in code and docs so this distinction doesn't get lost. Synchronous same-turn coupling (V2's evidence visibly moving pressure within the same conversation) is a later, separate step that additionally requires moving `DriveEngine`'s pressure store off its current per-container local JSON file to something `cortex-exec` can also safely read/write (Postgres/Redis) — real infra work, not attempted in v1.

**Cooldown**: `GOAL_PROPOSAL_COOLDOWN_MINUTES=180` was tuned for `DriveEngine`'s original diffuse bus-event cadence; loosening it to admit V2's higher-frequency per-turn evidence is a low-risk parameter change, no objection raised.

**Explicitly out of scope for this direction, needs its own proposal-mode treatment**: removing the human-operator-token gate on goal promote/execute (`orion-hub`/`orion-cortex-exec`'s `X-Orion-Operator-Token`-gated endpoints, §B.2). This is a materially different decision — expanding autonomous execution scope, not routing internal metric computation — and per CLAUDE.md's "Proposal mode before invasive cognition changes" needs its own explicit brief naming what capability changes, what failure mode would be dangerous, and how to disable/roll back. It should not ride along inside a metrics-unification patch even though it came up in the same conversation.

## Missing questions — resolved 2026-07-11

### Q1 — RESOLVED: two different stories, not one

**`AUTONOMY_GRAPH_BACKEND=disabled` is a deliberate, documented rollback.** Commit `e9b233e9`, 2026-06-19, direct-to-main (no PR wrapper), message: *"Disable autonomy Fuseki reads and graph churn by default. Chat stance no longer depends on SPARQL autonomy graphs; archive and RDF materialization are off until re-enabled."* Coordinated multi-service change landed alongside two mesh-stability hotfixes the same session (`1790bd96` landing-pad `/ready` flapping, `8309c61e` mesh-health noise) — also flipped `SUBSTRATE_STORE_BACKEND` sparql→in_memory, disabled nightly Fuseki goal-archive on two services, and expanded `orion-rdf-writer`'s `RDF_SKIP_KINDS` to drop drive/goal/identity materialization, with inline comments explicitly framing this as load/stability-motivated, not a specific incident postmortem. `docs/superpowers/pr-reports/2026-07-08-endogenous-origination-leaky-refresh-pr.md`, written ~3 weeks later, independently notices the fallout without connecting it to this commit ("drive-audit persistence has been broken since ~June 19").

**`CONCEPT_PROFILE_REPOSITORY_BACKEND=local` is pure config drift — never anything else.** Introduced at `local` by commit `5d1e663e`, 2026-03-27 (*"Add concept induction backend override and fallback policy settings"*), explicitly commented as a Phase 3A **shadow rollout** default (`CONCEPT_PROFILE_PARITY_*` settings built for shadow-compare-then-promote). Checked every commit touching this key since: **never once changed to `graph`, ever** — not promoted, not rolled back, just never finished. No graduation criteria, parity results, or Phase-3A-completion decision found anywhere in `docs/superpowers/`.

**Consequence for scope**: "burn it down" and "revive it" are not symmetric anymore. `AUTONOMY_GRAPH_BACKEND` was turned off on purpose for a real reason (Fuseki load) that would need addressing before reviving it (e.g., cheaper query patterns, caching) — reviving is nontrivial. `CONCEPT_PROFILE_REPOSITORY_BACKEND` was simply never finished — flipping it to `graph` might be the *cheaper* option to actually evaluate concept induction's real value (with the pronoun-filter fix from §A.3b applied first) before deciding to delete it. Worth actually trying before deleting, specifically for concept-profile, not for the autonomy-graph side.

### Q2 — carries forward given Q1's answer

Given concept-profile's backend flag is unfinished drift rather than a deliberate call, the pronoun/stopword filter (§A.3b) plus flipping `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph` in a non-prod environment is now the cheapest way to actually evaluate whether concept extraction is worth keeping, before deciding to delete it. Still not started — flagged as the recommended next patch below.

### Q3 — RESOLVED: the enable-gate was already run, and it's NO-GO, but the result was never committed

`docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md` requires the measurement gate (`scripts/analysis/measure_autonomy_gate.py`, plan: `docs/superpowers/plans/2026-07-07-autonomy-origination-measurement-gate.md`) to pass before `ORION_ENDOGENOUS_ORIGINATION_ENABLED` can flip. **The gate was run on 2026-07-08** against 120 days of live Postgres self-state history (128,610 rows) and 444,943 live Fuseki `DriveAudit` records. Both required verdicts came back **NO-GO**: drift-during-exogenous-silence `mean_abs_trajectory=0.0026` (required ≥0.03 — self-state is essentially flat when nothing is happening), drive co-activation `coactivation_frac=0.0004` (required ≥0.10). Per the spec's own stated rule, this NO-GO means the origination signal is inert as designed and would need to source dynamics from unresolved-pressure persistence instead.

**This verdict exists only in `/tmp/autonomy-gate/report.md` — never transcribed into any committed doc.** `docs/superpowers/pr-reports/2026-07-08-endogenous-origination-leaky-refresh-pr.md` (same date) is framed as a handoff asking someone to *still run* the gate, suggesting whoever wrote it didn't know it had already run. `docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-audit-design.md:33` (this session, same day as this doc) still frames enablement as an open question — written without knowledge of this result. **Action needed independent of this doc's scope**: commit the gate verdict to the repo (closing that question definitively as NO-GO under current architecture) so it stops getting re-asked. Given the NO-GO, `endogenous_origination.py`'s only integration point can be deleted along with the rest of `ConceptWorker` with no near-term loss — nothing is waiting to use it.

### Q4 — RESOLVED: the routing change itself is safe on this axis; a different, pre-existing overlap is the real exposure

Traced precisely: `AutonomyStateV2`'s evidence contract (`compile_autonomy_evidence`, `evidence_compiler.py:47-153`) never reads `SelfStateV1.dimensions` or turn-effect deltas at all — only two of its four evidence kinds (`reasoning_quality`, `relational_signal`) even produce a tension, and both route through categorical labels (`fallback`, hazard strings) matched against a closed typed map, not continuous self-state scores. **Routing V2's tensions into `DriveEngine` does not add a new correlated path into the phi corpus** — the raw material is categorically different from what phi trains on.

But tracing `DriveEngine.extract_tensions_from_self_state` (reads `coherence, agency_readiness, social_pressure, uncertainty, resource_pressure, execution_pressure`) against phi's **live** seed-v4 trainable feature set (`agency_readiness, execution_pressure, reasoning_pressure` + `overall_intensity` + cognitive slots — confirmed via `INNER_FEATURES_VERSION=seed-v4` in the live `.env`) found a **genuine, pre-existing overlap that has nothing to do with this proposal**: `agency_readiness` and `execution_pressure` are simultaneously live `DriveEngine` tension inputs and live phi-encoder-trainable features, today, with zero `AutonomyStateV2` involvement. Under seed-v3 (the code-level default, not what's live) the overlap is wider — `coherence`, `social_pressure` also included, since seed-v3 lacks seed-v4's "theater" exclusions.

**Net**: the isolation rule as originally stated (protect against `AutonomyStateV2`/drive-pressure becoming a phi input) doesn't actually block this specific routing proposal — go ahead per §Part C's async-v1 plan without that specific fear. But there's a separate, real, already-live multicollinearity surface (`agency_readiness`/`execution_pressure` double-read) worth its own investigation regardless of what happens to concept-induction/DriveEngine — not caused by anything proposed here, but real today.

### Q5 — ANSWERED by Juniper directly

Not currently looked at — "it's garbage" (agreed, matches its own "unavailable"/degraded copy being the common case today). Decision: **keep the panel's code, don't delete it, repurpose it once real metrics exist** (post §Part C unification). Not urgent, no action needed now.

### Decision — concept induction: keep the code, turn it off (not delete)

Juniper's call: *"let's leave it but turn it off."* This is cleaner than it sounds — `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED` (`services/orion-spark-concept-induction/.env_example:39`, default `true`) already exists and gates exactly the concept-profile/clustering trigger (`bus_worker.py:1091`), checked **after** `DriveEngine.update()`, `GoalProposalEngine.propose()`, identity snapshotting, and dossier all already run earlier in the same event handler (`bus_worker.py:836-1074`). Setting it to `false`:
- Stops all concept extraction/clustering (no more pronoun-soup `ConceptProfile` generation) — zero code deleted, zero risk of the `graph_query.py` import-breakage blocker (§B.1) since nothing is removed.
- Has **zero effect on `DriveEngine`, `GoalProposalEngine`, identity snapshotting, or dossier** — confirmed they all run earlier in the handler, unconditional on this flag.
- Fully reversible — flip back to `true` any time, no data loss (concept-profile store file just stops being written to).

This resolves the concept-induction half of this doc's original scope with a one-key config change, no code change, no restart-required deletion. **Not yet applied live** — this doc records the decision; applying it means updating `services/orion-spark-concept-induction/.env_example` (default → `false`) + syncing local `.env` + restarting that container, which is an operational action outside "just design/riffing" scope and needs its own explicit go-ahead.

`endogenous_origination.py`'s integration point is unaffected by this flag either way (it hooks earlier in the handler, same as DriveEngine) — its NO-GO gate result (Q3) is the separate, real reason nothing is lost by it going along with whatever eventually happens to the rest of `ConceptWorker`.

### Q6 — RESOLVED: no new lane needed

Traced `bus_worker.py:836-935`: `DriveEngine.update()` runs **unconditionally on every accepted event, with no cooldown or subject-selection gate at all**. The `concept_induction_generation_skipped decision=skipped_due_to_cooldown` messages seen in live logs are a *different*, earlier gate — the concept-profile generation cycle's own 300s cooldown, which never touches `drive_engine.update()`. The only cooldown that exists near this path is `GOAL_PROPOSAL_COOLDOWN_MINUTES=180`, applied later inside `goal_engine.propose()` for goal creation specifically — already agreed to loosen. So: routing V2's tensions into `DriveEngine.update()` hits no gate at all; no new lane or threshold needed.

## Non-goals

- Not proposing a replacement drive taxonomy.
- Not deciding here whether to remove the whole package or just concept extraction — Q1's answer reframes this: concept-profile revival (flip `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph` + fix the stopword filter) is now a real third option worth trying before deletion, distinct from the autonomy-graph side which was deliberately disabled for a real load reason.
- §Part C is a direction, not a plan — no code, no schema change, no flag flip proposed yet. It does now intentionally touch the in-progress `AutonomyStateV2` closed-loop-wiring plan's stated non-goal ("no graph-drives unification") — that plan's owner needs to know this direction exists before finishing that work under the old assumption.
- §Part C explicitly excludes removing the human-approval gate on goal promotion/execution — that is a separate, higher-stakes autonomy decision requiring its own proposal-mode brief, not bundled here.
- Not investigating the `agency_readiness`/`execution_pressure` phi/DriveEngine overlap found in Q4 — real, but orthogonal to everything else in this doc; needs its own pass.

## Decisions made (2026-07-11) — this doc is now closed

- **Concept induction**: keep the code, turn it off via `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false`. Not deleted, not revived. `graph_query.py` extraction (§B.1) is no longer urgent — nothing is being deleted, so the import-breakage blocker doesn't apply. Revisit revival (stopword fix + `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph`) only if something later wants concept-profile output again.
- **`endogenous_origination.py`**: NO-GO gate result stands, goes dark along with the rest of `ConceptWorker`'s unused trigger, no loss.
- **Hub Autonomy Debug Panel**: keep the code, not used today, repurpose once §Part C produces real unified metrics.
- **`DriveEngine`**: stays live, unaffected by the concept-induction flag flip (confirmed it runs earlier in the same handler, unconditional on that flag). §Part C's V2→DriveEngine unification proceeds as its own separate track, cleared by Q4/Q6.

## Recommended next patch

Three independent, low-cost items, no ordering dependency:

1. **Apply the concept-induction decision**: set `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` in `services/orion-spark-concept-induction/.env_example` (new default) + sync local `.env` + restart that container. Operational action, needs its own explicit go-ahead — not yet applied.
2. **Commit the endogenous-origination gate verdict** (Q3) — copy `/tmp/autonomy-gate/report.md`'s NO-GO result into the repo (e.g. append to `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`). Purely a documentation patch, zero risk, stops the same open question from being re-asked.
3. **§Part C's V2→DriveEngine routing** — separate implementation track, async-v1, per the plan above. Needs its own scoping session when Juniper's ready to move from design to a plan doc.

## 2026-07-14 update — concept-profile graph deleted, revival option no longer available as described

The "revival" option this doc floats above (Q1/Q2, §Non-goals: flip `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph` + fix the stopword filter, evaluate before deleting) is no longer available in its original form: the underlying `spark/concept-profile` RDF graph was deleted from the live Fuseki store on 2026-07-14 (backed up first: 1.2GB, 5,611,466 triples). It had no live readers — the backend was always `local`, confirmed again during this pass — and separately, its `graph` read query pattern was measured live and timed out at 60 seconds even against a single-subject-filtered query, so it was never going to scale to a live per-turn read path even if the backend had been flipped on. See `docs/superpowers/specs/2026-07-14-autonomy-spark-graph-reenable-cost-trace.md` for the query-cost measurements. A replacement store is planned but not yet built; this doc's `graph`-backend revival option should be read as historical until that replacement exists.
