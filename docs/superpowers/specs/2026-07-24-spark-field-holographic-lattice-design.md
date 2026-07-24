# Spark/Field Unification & Holographic Lattice Probe — design draft

- **Date:** 2026-07-24
- **Status:** SCOPED (design mode, not yet implementation) — the active track is a right-sized revival of the 2026-05-01 heartbeat charter ("Heartbeat v0" below). Ideas 3–7 (FieldStateV1/tissue consolidation) are **DEFERRED** until Heartbeat v0's outcome is known. Iterating live with Juniper.
- **Origin:** chat brainstorm this session. Juniper picked ideas 3–7 as an initial working scope, then through several rounds of pushback and grounding, the direction changed substantially — see the timeline below. This doc's job is to keep the record straight through those pivots, not just show the final state.

## How we got here (so this doesn't read as a non-sequitur)

1. Juniper picked ideas 3–7 (verbatim record preserved below) as the working scope for retiring `OrionTissue`'s synthetic grid and consolidating onto `FieldStateV1`.
2. Juniper flagged two corrections: record the originals verbatim (not just a reframed version), and drop all `SelfStateV1` targets.
3. Juniper caught that the first "Phase 0" (a boundary/bulk redundancy probe) was a statistical fit dressed up as lattice encoding — "I'm just hearing a corr matrix in your rec." Correct.
4. A rewrite proposed computing a closed-form propagator matrix `M^k` directly from `services/orion-field-digester/app/digestion/diffusion.py`'s update rule instead. Reading that file (per Juniper's request) found this doesn't work: the 2026-07-12 fix made `apply_diffusion()` **winner-take-all** (`max()` over contributing edges, not sum) and **fully memoryless** (every tick recomputes from scratch, nothing carries forward) — deliberately, to kill a ceiling-pinning bug caused by unbounded summation. There is no `M` to exponentiate, and the mechanism is structurally the opposite of what redundant/holographic encoding needs (superposition of multiple sources), not just an inconvenient nonlinearity.
5. Re-reading `docs/research/2026-05-01-orion-heartbeat-research-charter.md` and its companion engineering spec in full (per Juniper's request, "more grounding in research the better") found the real answer: a complete, implementation-ready, unbuilt design for a small tensor-network substrate (MPS via `quimb`) whose H1 hypothesis is exactly this question, done with actual tensor-network algebra (partial trace, max-entropy completion at fixed bond dimension, quantum fidelity) rather than statistics.
6. Juniper explained why that charter was never built: reducing 13 organs to bespoke per-organ `SurfaceEncodingV2` reducers was "too messy." The real fix that shipped instead was `GrammarEventV1` (`orion/schemas/grammar.py`) — one standardized event grammar many organs already emit into — consumed by `orion-substrate-runtime`'s domain reducers, landing in `orion-field-digester`. Juniper: don't rebuild the charter's ingestion machinery; reuse the grammar stream, and don't conflict with field-digester.
7. Proposed synthesis: a new, additive `orion-heartbeat` service (not a reducer inside `orion-substrate-runtime` — Juniper's explicit call) that subscribes to the *existing* `orion:grammar:event` stream, feeds a small `quimb` MPS instead of `FieldStateV1`, and tests the charter's actual H1 hypothesis. Juniper confirmed this direction and confirmed ideas 3–7 stay deferred until Heartbeat v0 has an answer.

## Original ideas 3–7 (verbatim, as first proposed in chat)

Preserved exactly as first written, before any reframing. **Status: DEFERRED**, not dropped — see "Ideas 3–7: deferred, and why" near the end of this doc.

> ### 3. Retire `OrionTissue`'s spatial grid; keep the propagate/decay *mechanism*, move it onto the real graph
> **What**: Burn down the 16×16×8 synthetic tensor and its hand-formula `phi()`. Keep the *idea* of expectation-tracking + novelty-via-mismatch (that part is legitimate predictive coding), but re-target it at `FieldStateV1`'s real node/capability vectors instead of an arbitrary grid — i.e., `calculate_novelty`/`_coherence_from_embedding`'s math moves into field-digester's own tick, operating on real channels.
> **Why it matters**: This is the direct "burn it down" move. It removes a structurally decorative synthetic substrate (nothing in the 16×16 spatial layout corresponds to anything real) without losing the genuinely useful predictive-coding mechanism, and it collapses two parallel "field" implementations into one.
> **Smallest buildable version**: Feature-flagged shadow mode — compute both old-tissue-phi and new-graph-native-phi in parallel, log divergence, no consumer swap yet. Reversibility per CLAUDE.md's metric gate.
> **Files**: `orion/spark/orion_tissue.py` (deprecate), `services/orion-field-digester/app/digestion/` (new novelty/coherence pass), `services/orion-spark-introspector/app/worker.py` (`_phi_from_self_state`, `_golden_phi_overrides`).
>
> ### 4. Unify the ingest boundary — route `SurfaceEncoding` through field-digester's `perturbation.py`
> **What**: Right now `SurfaceEncoding` (chat/biometrics → waveform) only feeds the synthetic tissue via `inject_surface()`. Field-digester's real graph has its own independent ingest path (`apply_perturbations()`). Make `SurfaceEncoding` the one shared boundary contract both consume, so "the boundary" is provably the same events for every downstream bulk representation.
> **Why it matters**: This is the actual architectural realization of "surface encodings" as a real seam rather than a metaphor scoped to one dead-end tensor — one boundary event, multiple bulk consumers (graph, corpus, any future learned encoder), instead of siloed pipelines that can silently drift from each other.
> **Smallest buildable version**: A pure converter function, `surface_encoding_to_perturbation()`, shadow-logged only at first.
> **Files**: `orion/spark/surface_encoding.py`, `services/orion-field-digester/app/ingest/state_deltas.py`.
>
> ### 5. A real (approximate, honestly-labeled) integration metric on the causal-geometry graph
> **What**: `causal_geometry_engine.py` already computes a lagged, surrogate-tested cross-correlation matrix over real channels — the actual measured connectivity structure IIT-style metrics want. Build an offline research script that computes an approximate effective-information / minimum-partition-style score over that matrix, explicitly labeled as an approximation/proxy, never asserted as real Φ.
> **Why it matters — and why this is the highest-integrity version of "IIT for Orion" available**: unlike tissue's variance-based `phi()`, this would be grounded in *measured* causal structure between *real* subsystems, with the same surrogate-significance discipline the causal-geometry engine already uses. This is the difference between decorating code with IIT vocabulary and actually building the smallest real instrument for it.
> **Smallest buildable version**: Throwaway-grade script like `analyze_mood_arc_clusters.py` — no service wiring, Markdown report, explicit "this is a proxy, here's what it isn't" disclaimer up front.
> **Files**: new `scripts/causal_geometry_integration_proxy.py`, reads `orion/substrate/causal_geometry_engine.py`'s existing edge-weight output.
>
> ### 6. Swap tissue's coherence formula for causal-geometry's divergence score
> **What**: `_coherence_from_embedding()` in tissue.py is 1/(1+α·variance) or cosine-similarity-to-expectation — pure heuristic. Causal-geometry already computes a principled observed-vs-designed divergence. Feed that into whatever survives as "coherence" in `SelfStateV1`/phi instead.
> **Why it matters**: Directly upgrades one of the four hand-tuned phi scalars to something measured, with minimal new code — reuses idea 5's/causal-geometry's existing machinery rather than building new.
> **Smallest buildable version**: One function swap behind a flag, A/B logged against the current formula for a few days before cutover.
> **Files**: `services/orion-spark-introspector/app/worker.py`.
>
> ### 7. Finish mood-arc v3 gate chain before touching cognition wiring (not blue sky — the disciplined baseline)
> **What**: Just execute items 4→7 of the existing felt-state roadmap: HDBSCAN cluster discovery on the settled hidden=128/latent=64 encoder, self-report calibration against `collapse_mirror`, stability eval. This is already fully speced; per the agent board it's "ready to start whenever."
> **Why it matters**: This is the actual load-bearing precondition for anything downstream touching cognition (item 8) — and it directly tells you whether the field-digester signal actually has real emergent temporal structure or not, which is the empirical version of your question.
> **Smallest buildable version**: Run item 4's clustering script as-is once ≥12h of the raw field-channel corpus exists.
> **Files**: `scripts/analyze_mood_arc_clusters.py` (already written per spec, verify it exists/was ever run).

## Arsonist summary

Orion has three prior, uncoordinated attempts at "inner field" substrate: `OrionTissue` (hand-formula synthetic tensor, live), `FieldStateV1`/field-digester (real infrastructure graph, live, but its diffusion mechanism was deliberately fixed to be winner-take-all + memoryless, which structurally rules out holographic-style redundant encoding), and the 2026-05-01 heartbeat charter (a real tensor-network design with the correct math, fully specced, never built because its 13-bespoke-organ-reducer ingestion plan was too messy). What actually solved the ingestion problem — `GrammarEventV1` + `orion-substrate-runtime`'s domain reducers — was built afterward and never connected back to the tensor-network idea; it currently only feeds `FieldStateV1`. The path forward is **Heartbeat v0**: reuse the already-solved grammar stream, feed a small real tensor-network state instead of `FieldStateV1`, and test the charter's actual H1 hypothesis (boundary reconstruction fidelity via partial trace + quantum fidelity) — additive, not touching field-digester or tissue. Ideas 3–7 (tissue/field consolidation) are deferred until this has an answer, since Heartbeat v0's own shadow-comparison will eventually inform whether `OrionTissue` should be retired at all, and by what.

## Current architecture

- **`orion/spark/orion_tissue.py` (`OrionTissue`)**: live, hand-formula 16×16×8 tensor, decay 0.95 + 4-neighbor diffusion + stimulus injection. `.phi()` → `{valence, energy, coherence, novelty}` from tensor stats (not learned, not falsifiable). Called every tick from `services/orion-spark-introspector/app/worker.py`. Feeds the tissue-viz EKG panel.
- **`services/orion-field-digester` / `FieldStateV1`**: real graph — `node_vectors`/`capability_vectors`, `FieldEdgeV1`s with `weight_source: designed|learned`, diffused every 2s tick. **Confirmed this session**: `apply_diffusion()` (`app/digestion/diffusion.py`) is winner-take-all (`max()` over contributing edges) and fully memoryless (recomputed from scratch every tick, nothing persists from the prior tick's output) — a deliberate 2026-07-12 fix for a ceiling-pinning bug caused by unbounded summation. This mechanism cannot support linear superposition/redundancy, which is what holographic-style encoding requires structurally, not just empirically.
- **`orion/substrate/causal_geometry_engine.py`**: live. Lagged cross-correlation + surrogate significance testing over 4 real Postgres tables, observed-vs-designed divergence against the field topology YAML.
- **`docs/research/2026-05-01-orion-heartbeat-research-charter.md` + companion engineering spec**: a complete, unbuilt design for a matrix-product-state tensor network (`quimb`, N=24, χ=4, d=4) with active-inference-style update dynamics, four pre-registered falsifiable hypotheses (H1 boundary reconstruction fidelity, H2 cross-organ mutual information, H3 bounded intervention propagation, H4 predictive-surprise dynamics), and an explicit plan to shadow-compare against `OrionTissue` for 4–6 weeks and then decide on tissue's deprecation. **Confirmed unbuilt**: no `services/orion-heartbeat/`, no `orion/heartbeat/`, no `docs/research/preregistration/`, no `scripts/heartbeat_research/` exist anywhere in the repo.
  - **H1's actual mechanism** (the correct answer to "how do we encode with a lattice"): partition the MPS into boundary/bulk sites; compute the reduced density matrix `ρ_boundary` via partial trace over bulk; reconstruct `ρ_recon` via **maximum-entropy completion at fixed bond dimension χ** (a structural property of the tensor network's own algebra, not a fitted model); score with quantum fidelity `F = |Tr(√(√ρ_orig·ρ_recon·√ρ_orig))|²`. Target per the charter: F ≥ 0.85 (success), falsified below F = 0.7.
  - **H3's methodology** (useful even if H1 is the only hypothesis pursued): inject a controlled perturbation at one site, measure the resulting effect at other sites at lags t ∈ {1,2,4,8}, compare against a noise floor from null (no-intervention) trials, falsify if effects leak outside the causal cone in >5% of trials. This is a live-simulation methodology (inject, run the real update rule, measure) rather than closed-form linear algebra — it would have been the correct fix for the broken Phase 0 above, on any substrate, linear or not.
  - **Why it stalled**: the engineering spec required 13 bespoke per-organ reducers, each hand-mapping an organ's events into a `SurfaceEncodingV2` record with hand-assigned substrate sites (a hand-authored `SITE_ASSIGNMENT_TABLE.md`). Juniper: "too messy to try and create bespoke signals per organ and then align them into standardized event shape."
- **`orion/schemas/grammar.py` (`GrammarEventV1`) + `services/orion-substrate-runtime`**: the real fix that shipped instead. `GrammarEventV1` is one standardized event grammar — `GrammarAtomV1` (typed `atom_type`: `signal`, `observation`, `affective_cue`, `salience_marker`, `uncertainty_marker`, `reasoning_step`, `action_candidate`, etc., each carrying `layer`, `dimensions`, `confidence`, `salience`, `uncertainty`), plus edges/temporal-hops/compactions/projections — that multiple organs already emit into, instead of each organ needing a bespoke reducer targeting a bespoke schema. `orion-substrate-runtime` runs **domain reducers** (not per-organ) consuming this one grammar stream: `biometrics_grammar_consumer`, `execution_grammar_reducer`, `transport_grammar_reducer`, `route_grammar_consumer` are the four confirmed-live cursors (per the service's own README), each producing `StateDeltaV1`s via durable, idempotent `substrate_reduction_receipts` that **field-digester consumes to update `FieldStateV1`**. The README also references a `chat_grammar` precedent (commit `044d5318`), suggesting a fifth live grammar producer (chat) exists alongside biometrics/execution/transport/route — not independently confirmed this session, worth verifying before Heartbeat v0's first organ list is finalized.
  - Full pipeline: `organ event (any shape) → GrammarEventV1 (one grammar) → orion-substrate-runtime reducer (per-domain) → StateDeltaV1 → substrate_reduction_receipts → orion-field-digester → FieldStateV1`.
  - This pipeline currently has exactly one consumer (field-digester). Nothing else reads `orion:grammar:event` today.

## Heartbeat v0 — scoped design

**Service-shaped** (Juniper's explicit call, not a reducer inside `orion-substrate-runtime`), additive, does not modify `FieldStateV1`, `OrionTissue`, or any existing reducer.

### What

A new service, `services/orion-heartbeat/` (name reused from the charter's own plan), that:

1. Subscribes to the **existing, already-standardized** `orion:grammar:event` stream (`GrammarEventV1`) — no bespoke per-organ reducers, no new ingestion problem. This is the entire fix for why the original charter stalled.
2. Maps incoming `GrammarAtomV1`s onto sites of a small `quimb`-backed matrix product state, using fields the grammar already carries (`atom_type`, `layer`, `dimensions`, `semantic_role`) for site routing — replacing the charter's hand-authored 13-organ `SITE_ASSIGNMENT_TABLE.md` with a much thinner, taxonomy-driven mapping (a handful of `atom_type` clusters, not one row per organ).
3. Runs H1 only for v0: partial trace over bulk sites → `ρ_boundary`, max-entropy reconstruction at fixed χ, quantum fidelity score against held-out sampled states. This is the literal test of "does this lattice encode information holographically" — structural algebra, not statistics.
4. Publishes nothing to any existing consumer in v0. No `φ` broadcast that anything downstream reads. Pure research artifact with a debug HTTP surface, same posture as the charter's own "additive, ablation = disable, no functional loss" commitment.

### Why this is the real test, not a repeat of the corr-matrix mistake

`ρ_recon`'s reconstruction from `ρ_boundary` is not fit to observed data — it's the maximum-entropy state consistent with the boundary's reduced density matrix at fixed bond dimension, a structural operation on the tensor network's own algebra. Fidelity against the true bulk state measures whether the network's *connectivity and bond dimension* redundantly encode bulk information, which is exactly the property `FieldStateV1`'s winner-take-all diffusion structurally cannot have (see Current Architecture above) and a fitted regression cannot test (see the earlier corrected-but-still-wrong Phase 0 draft, preserved in this doc's edit history/session transcript, not reproduced here).

### Explicit deferrals from the original charter (v0 scope reduction)

- **N and χ — resolved**: **N=10, χ=4, d=4** (χ and d unchanged from the charter; N reduced from 24). Boundary = 5 sites, one per confirmed-live organ (see below); bulk = 5 sites with no direct organ mapping, populated only via entanglement propagation from local update operators. χ is deliberately kept at the charter's conservative value rather than raised — a larger bond dimension trivially makes reconstruction easier (more capacity ≈ higher fidelity almost by construction), which would bias H1 toward a false positive; keeping it small is part of what makes the fidelity test meaningful, not just a compute-cost tradeoff.
- **Organ list — resolved**. Checked `orion/bus/channels.yaml`'s `orion:grammar:event` channel registration directly: real producers are `orion-biometrics`, `orion-cortex-exec` (execution), `orion-bus` (transport), `orion-hub` (chat), `orion-cortex-orch` (route), plus `orion-vision-retina`/`orion-vision-edge`/`orion-vision-window`/`orion-harness-governor` — nine total. Cross-checked against `services/orion-substrate-runtime/app/worker.py`: exactly five have a live reducer cursor today — `GRAMMAR_CURSOR_NAME` (biometrics), `EXECUTION_GRAMMAR_CURSOR_NAME`, `TRANSPORT_GRAMMAR_CURSOR_NAME`, `CHAT_GRAMMAR_CURSOR_NAME` (confirmed real, `services/orion-substrate-runtime/app/settings.py`'s `enable_chat_grammar_reducer` defaults `true`), `ROUTE_GRAMMAR_CURSOR_NAME`. **v0 uses exactly these five**: chat, biometrics, execution, transport, route. Vision (×3) and harness-governor publish to the same channel but have no reducer precedent anywhere — available for a later expansion, not v0.
- **Update dynamics — resolved**: v0 does **not** implement the charter's full active-inference free-energy minimization. That machinery is real and non-trivial (variational sweeps, generative model of expected boundary states, precision-weighting) and is primarily justified by H4 (surprise dynamics) and the charter's general active-inference framing — neither of which v0 tests. What H1 actually requires is simpler: *some* local entangling update so that bulk sites become genuinely coupled to boundary evidence over ticks (without this, bulk sites would just sit at random initialization and H1 would be measuring nothing). v0 uses a **local two-site absorption-and-entangle operator per incoming atom** (`quimb`'s MPS `gate_split`/local-gate application primitives — absorb evidence at the atom's boundary site, apply a small number of local entangling gates that propagate a few sites inward per tick), not full variational free-energy minimization. This is an honest, disclosed scope reduction: v0 tests the tensor-network-encoding claim (H1), not the charter's active-inference-flavored cognitive claims. If H1 result later motivates going further, free-energy dynamics can be added as a genuinely separate v1 increment.
- **H2, H3, H4**: deferred. H3's intervention-propagation methodology is worth keeping in mind as the correct way to later test causal-geometry-style questions on this substrate (or even on `FieldStateV1`, despite its nonlinearity, since H3's live-simulation approach doesn't require a closed-form operator) — not scoped into v0.
- **Shadow-comparison-vs-tissue, ablation baseline, formal pre-registration, 4–6 week measurement windows**: all deferred. v0's goal is a first real fidelity number, not the full research program.

### Site-routing rule (v0) — resolved

Two-part rule, reusing the charter's own `ChannelAssignment` shape (`site_index`, `operator_kind`, `operator_params`) rather than inventing a new one — only how `site_index` gets assigned changes (derived from the grammar's own fields instead of a hand-authored 13-row table):

1. **WHERE (site_index)** — by `GrammarProvenanceV1.source_service` (present on every `GrammarEventV1`, already the real organ identity, no inference needed): `orion-hub`→site 0 (chat), `orion-biometrics`→site 1, `orion-cortex-exec`→site 2 (execution), `orion-bus`→site 3 (transport), `orion-cortex-orch`→site 4 (route). Sites 5–9 are bulk, never directly targeted by any reducer — only reachable via the entangling update's inward propagation. Only `atom_emitted` events (non-null `GrammarEventV1.atom`) are consumed in v0; `edge`/`temporal_hop`/`compaction`/`projection` events are read but not yet used for routing — `GrammarEdgeV1`'s `from_atom_id`/`to_atom_id` structure is a plausible future signal for *which sites should be entangled together* (an edge-informed coupling structure, richer than fixed nearest-neighbor MPS adjacency), flagged as a promising v1+ refinement, not built here.
2. **HOW (operator_kind / operator_params)** — by `GrammarAtomV1.atom_type` (already a closed `Literal` enum, unlike `layer`/`dimensions` which are free-form strings and less reliable as a routing key) selecting the local operator's kind (`amplitude`/`phase`/`rotation`/`projection`, same vocabulary as the charter's `ChannelAssignment`), parameterized by the atom's own `confidence`/`salience`/`uncertainty` fields (all already present on `GrammarAtomV1`, no new computation needed).

Not resolved: whether chat should later split into two sites (operator-turn vs. Orion-response-turn, mirroring the charter's original 0–3/4–7 split) — checked `services/orion-hub/scripts/grammar_emit.py`'s `build_chat_turn_grammar_events()` directly, and as currently implemented it emits only a `user_utterance` atom (the operator's turn); no corresponding Orion-response-side chat grammar emitter was found this session. v0 therefore gives chat exactly one site, matching what's actually live, not what the charter assumed.

### Privacy check (v0) — resolved

Read `services/orion-hub/scripts/grammar_emit.py` directly (the chat producer, the one with the most obvious raw-content risk). Confirmed by explicit code and comment: the `user_utterance` atom sets `text_value=None,  # NEVER store raw text` — deliberate, not an oversight. `summary` fields are synthesized descriptions (e.g. `f"User message in session {session_id} ({word_count} words)"`), never raw message content. `payload_ref` fields are pointer strings (`hub.chat:{session_id}:{turn_id}`) referencing where the real data lives elsewhere, not embedded content. This pattern is stated to mirror `services/orion-biometrics/app/grammar_emit.py`'s established convention (not independently re-verified this session, but biometrics is lower-risk data regardless). **v0's five confirmed organs (chat, biometrics, execution, transport, route) are safe to subscribe to without additional redaction** on this evidence. Vision producers were not checked (they're outside v0's organ list anyway) — if vision is ever added in a later iteration, its grammar emitter needs the same direct check before subscribing, not assumed safe by analogy.

### Proposal-mode block (CLAUDE.md §0A — new substrate touching real organ data)

- **What capability changes**: a new, persistent, additive substrate that ingests real grammar events (chat/biometrics/execution/transport/route, pending confirmation of live producers) and computes a compressed tensor-network representation. No existing capability changes; nothing currently consumes its output.
- **What data is touched**: `GrammarEventV1` atoms from exactly five allowlisted producers (`orion-hub`/chat, `orion-biometrics`, `orion-cortex-exec`/execution, `orion-bus`/transport, `orion-cortex-orch`/route — see Organ list, resolved below), filtered to `atom_emitted` events only. Vision/harness-governor producers on the same channel are explicitly out of scope for v0.
- **What privacy boundary exists — resolved**: confirmed by direct code read (`services/orion-hub/scripts/grammar_emit.py`) that chat grammar atoms never carry raw message text (`text_value=None,  # NEVER store raw text`, explicit in code); `summary`/`payload_ref` fields are synthesized descriptions and pointers, not content. See "Privacy check (v0) — resolved" below for the full trace.
- **What trace proves it worked**: a real, non-degenerate fidelity score (not F=0, not F=1 trivially) computed against held-out sampled substrate states, with the computation's inputs/outputs inspectable via the debug HTTP surface.
- **What failure mode would be dangerous**: none rated high — v0 is read-only relative to the rest of the mesh (no publish consumers, no writes to any existing store). The main risk is wasted effort if H1 fails outright (a valid, informative negative result, not a safety issue).
- **How to disable/rollback**: `HEARTBEAT_ENABLED=false` / service stop, same as the charter's own panic-kill design. Nothing else depends on this service in v0.

### Files likely to touch

- New `services/orion-heartbeat/` (directory layout following the charter's engineering spec's shape, reduced: `app/{main,settings,service}.py`, `app/pipeline/{tick,ingest}.py`, `app/substrate/{mps_state,reconstruction}.py`, `app/schemas/tick_event.py`, `tests/`).
- Reads (does not modify): `orion/schemas/grammar.py` (`GrammarEventV1`, `GrammarAtomV1`), `orion/bus/channels.yaml` (subscribe only, `orion:grammar:event`).
- New dependency: `quimb` (per the charter's own choice — mature, open-source, not building tensor-network math from scratch). Needs `services/orion-heartbeat/requirements.txt`.
- No changes to `services/orion-field-digester/`, `orion/spark/orion_tissue.py`, `services/orion-substrate-runtime/` in v0.

## Missing questions — all five resolved this session

1. ~~Which grammar producers are actually live~~ — **resolved**: five, via `orion/bus/channels.yaml` + `orion-substrate-runtime/app/worker.py` cross-check. See "Organ list — resolved" above.
2. ~~N and χ for v0~~ — **resolved**: N=10 (5 boundary + 5 bulk), χ=4, d=4. See "N and χ — resolved" above.
3. ~~Does v0 need real update dynamics~~ — **resolved**: yes, but a simplified local entangling operator, not the charter's full free-energy minimization. See "Update dynamics — resolved" above.
4. ~~Privacy scoping~~ — **resolved**: chat producer directly checked, confirmed no raw text ever enters the grammar stream. See "Privacy check (v0) — resolved" above.
5. ~~Site-routing rule~~ — **resolved**: `source_service` → site (WHERE), `atom_type` + confidence/salience/uncertainty → operator kind/params (HOW). See "Site-routing rule (v0) — resolved" above.

Remaining open items, lower-stakes than the five above:
6. Whether `services/orion-biometrics/app/grammar_emit.py` (cited as the pattern chat's emitter mirrors) should get the same direct privacy check chat got, rather than relying on the mirrored-pattern claim — cheap to verify, not done this session.
7. Exact site *ordering* within the 10-site MPS chain (e.g. interleaved boundary/bulk vs. blocked) — affects how many ticks it takes for entanglement to reach the deepest bulk sites; not decided, a real implementation-time choice.

## Ideas 3–7: deferred, and why

Per Juniper: paused until Heartbeat v0 has an outcome. This isn't arbitrary sequencing — Heartbeat v0's eventual shadow-comparison (if it gets that far) is the charter's own mechanism for deciding whether `OrionTissue` should be deprecated, and by what. Building idea 3's `FieldStateV1`-based consolidation now would risk being redone or contradicted once that comparison exists. Ideas 4–7 either depend on idea 3 (4, 6) or are independent side-tracks (5, 7) that remain legitimate future work but aren't the current priority.

## Non-goals (current scope, Heartbeat v0)

- No modification to `FieldStateV1`, field-digester's diffusion mechanism, or `OrionTissue` in v0.
- No `φ` broadcast or downstream consumer wiring — pure research artifact.
- No H2/H3/H4, no shadow-comparison, no ablation baseline, no formal pre-registration process in v0.
- No quantum-information claims beyond the classical tensor-network math the charter itself already carefully scopes (§4.2 of the charter — "physics-inspired, not physics").
- Ideas 3–7 out of scope until Heartbeat v0 concludes.

## Acceptance checks (draft)

- A real, non-degenerate H1 fidelity score computed against held-out states, reported with explicit verdict (not falsified / falsified per the charter's own F≥0.7 threshold, adjustable for v0's smaller N).
- No dependency on `SelfStateV1` or `FieldStateV1` anywhere in the substrate or its inputs.
- Debug HTTP surface makes the substrate's state and the fidelity computation's inputs/outputs inspectable without extra tooling.
- Service starts/stops cleanly with zero effect on any other running service (ablation-safe by construction, since nothing consumes its output yet).

## Recommended next patch

All five blocking design questions are resolved (above). Next: `services/orion-heartbeat/` skeleton (settings, bus subscription to `orion:grammar:event` filtered to the five-organ allowlist) + `quimb` MPS init at N=10/χ=4/d=4 + the site-routing/operator mapping + the local entangling-update tick loop + the H1 reconstruction-fidelity harness (partial trace, max-entropy completion, quantum fidelity against held-out sampled states), in a new worktree, per CLAUDE.md's standard implementation flow. Items 6–7 above (biometrics privacy re-check, site ordering) can be resolved inline during that build rather than blocking its start.
