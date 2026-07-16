# Causal Geometry v1 — Observed-Geometry Map, Field Topology Plasticity, Hub Operator Tab

- **Date:** 2026-07-16
- **Status:** PROPOSAL (design mode; Phase B section doubles as the CLAUDE.md §0A proposal-mode artifact)
- **Origin:** `/superpowers:brainstorming` session 2026-07-16 (full output in Appendix A)
- **Scope adopted by Juniper:** brainstorm ideas 1 + 2 for v1, plus an operator surface in orion-hub

## Arsonist summary

Orion's geometry is config, not dynamics. The field lattice topology (`config/field/orion_field_topology.v1.yaml`) is hand-authored and static; nothing measures whether the *observed* causal structure between organs matches the *designed* one, and nothing lets the geometry respond to experience. v1 ships three thin phases: (A) a deterministic observed-geometry measurement over existing state-journaler time-series, (B) bounded Hebbian plasticity for one edge class, gated through the existing substrate mutation proposal/trial machinery, and (C) a read-only operator tab in orion-hub following the proven Substrate standalone-page isolation pattern so tabbing in and out never kills the chat session.

## Current architecture

- **Designed geometry**: `config/field/orion_field_topology.v1.yaml` — node/capability channels + edges with fixed weights; `services/orion-field-digester/app/graph/lattice.py` (`LatticeGraph`) + `app/digestion/diffusion.py` run diffusion over it. Cap→cap edges recently added (`docs/PR-field-lattice-cap-cap-edges.md`).
- **Observable time-series**: `services/orion-state-journaler` rolls organ signals into Postgres (live at `localhost:55432`, db `conjourney`): drive levels, equilibrium distress/zen, salience/dynamic_pressure, coalition/broadcast projections.
- **Self-modification machinery**: `orion/substrate/mutation_{proposals,trials,scoring,queue,self_revision}.py` + review runtime — proposals, replay-corpus trials, class-specific scoring. Currently pointed at graph mutations; reusable for weight-change proposals.
- **Hub isolation pattern** (the one Phase C copies): `/substrate` route + `templates/substrate.html` + `static/js/substrate.js`, embedded in `templates/index.html` as `<section id="substrate" data-panel="substrate">` containing a persistent iframe (`substratePanelFrame`, `src="/substrate"`). `static/js/app.js` switches tabs by toggling the `hidden` class + `history.replaceState` with `event.preventDefault()` — **no full navigation, iframe stays mounted, chat websocket survives**. Enforced by `tests/test_substrate_standalone_page.py` and `tests/test_substrate_review_runtime_hub_debug.py`.

## Missing questions

1. **Journaler coverage**: which organ series are actually retained, at what sampling rate and retention window? (First concrete step below answers this.)
2. **Plasticity autonomy level**: does Orion self-adopt edge-weight changes via review runtime, or does every change land HITL in `reviews/pending`? **v1 default: HITL** — proposals only, operator adopts from the hub tab or CLI. Autonomy expansion is a separate future proposal.
3. **Significance regime**: how many co-sampled hours are needed before surrogate-tested correlations stop being noise? Determined empirically in Phase A; the report must print effective sample size per pair.

## Proposed schema / API changes

- **New schema** `orion/schemas/causal_geometry.py`:
  - `CausalGeometryEdgeV1`: `source_id`, `target_id`, `lag_sec`, `strength` (signed), `significance` (surrogate-test p or z), `n_samples`, `window` (`{start,end}`).
  - `CausalGeometrySnapshotV1`: `snapshot_id`, `generated_at`, `window`, `edges[]`, `designed_topology_version`, `divergence_summary` (per-edge observed-vs-designed delta; edges observed-but-not-designed; designed-but-not-observed).
- **New bus kind** `causal.geometry.snapshot.v1` on a new channel (register in `orion/bus/channels.yaml` + `orion/schemas/registry.py` in the same patch — no unregistered shapes).
- **Field topology schema**: `FieldEdgeV1` gains optional provenance fields `weight_source: Literal["designed","learned"] = "designed"` and `learned_at` — designed YAML remains valid unchanged.
- **New hub API (read-only, bounded, source-honest like `/api/substrate/*`)**:
  - `GET /api/causal-geometry/snapshot` — latest snapshot + divergence summary
  - `GET /api/causal-geometry/history?limit=N` — snapshot timeline
  - `GET /api/causal-geometry/proposals` — pending/decided weight-change proposals (Phase B)
  - Each returns `source` metadata (`kind`, `degraded`, `query`, `last_refreshed`), matching the substrate-page contract.

## Phase A — observed-geometry measurement (idea 1)

Deterministic nightly job, read-only, out of every hot path:

1. `scripts/causal_geometry_report.py`: reads N days of state-journaler rollups, computes pairwise **lagged cross-correlations** with **surrogate-data significance testing** (circular time-shift surrogates; no edge reported below significance). Transfer entropy is a later upgrade, not v1.
2. Emits `CausalGeometrySnapshotV1` (bus + Postgres via sql-writer) and a human-readable divergence report: observed edge strength vs. `orion_field_topology.v1.yaml` designed weight, plus observed-only and designed-only edge lists.
3. Non-stationarity guard: report per-pair effective sample size and refuse (mark `degraded`) any pair below threshold rather than emitting noise. No empty-shell output: a snapshot with zero significant edges must say so explicitly, never render as a plausible-looking graph.

## Phase B — field topology plasticity (idea 2) — proposal-mode block

- **What capability changes**: lattice edge weights for **cap→cap edges only** (smallest class) can change over time, sourced from Phase A co-activation statistics instead of hand-editing YAML. YAML becomes the seed; learned weights are bounded to `[designed_weight ± MAX_DRIFT]` (default drift cap 0.15) with weight decay back toward designed values absent reinforcement.
- **Mechanism**: a stats reducer aggregates co-activation per edge → emits weight-change proposals through `orion/substrate/mutation_proposals.py`; `mutation_trials.py` replays the proposal against the existing corpus; `mutation_scoring.py` gets a class-specific scorer for weight deltas; **adoption is HITL in v1** (operator approves via hub tab/CLI; nothing auto-applies).
- **What data is touched**: field topology runtime state (a learned-weights overlay table/file, never the checked-in YAML), mutation queue rows, snapshot artifacts. No memory, journal, or private-trace content.
- **Privacy boundary**: inputs are already-materialized organ scalars from the journaler; no raw journals, mirrors, or blocked material are read or exposed.
- **What trace proves it worked**: proposal row → trial result → adoption record → next diffusion tick reading the learned weight (log line with correlation ID) → observed divergence for that edge shrinking across subsequent Phase A snapshots.
- **Dangerous failure mode**: degenerate drift (all-connected or frozen lattice) distorting downstream pressure/attention. Contained by: drift cap, decay-to-designed, one edge class only, HITL adoption, trial gate.
- **Disable / rollback**: `FIELD_PLASTICITY_ENABLED=false` (default **false**) reverts diffusion to designed YAML weights instantly; learned overlay is additive and deletable; every adoption is individually revertible via the mutation queue record.

## Phase C — hub operator surface

Copy the Substrate standalone-page pattern exactly (see `docs/architecture/orion_hub_standalone_substrate_page.md`); do not merge into `app.js`'s bundle or expand the main shell logic:

- Dedicated route `/causal-geometry`, template `templates/causal_geometry.html`, **own bundle `static/js/causal-geometry.js`** (hard requirement from Juniper).
- Shell integration in `templates/index.html`: nav link (`id="causalGeometryPageLink"`, `href="#causal-geometry"`, `data-hash-target`) + `<section id="causal-geometry" data-panel="causal-geometry">` wrapping a persistent iframe `causalGeometryPanelFrame` with `src="/causal-geometry"` and a standalone link.
- `app.js` gets only the same thin switching block substrate got: `preventDefault()` on the tab click, `classList.toggle("hidden", ...)`, `styleTabButton(...)`, `history.replaceState(null, "", "#causal-geometry")`, hash restore on load. **Session survival is structural**: no navigation occurs, the iframe is never unmounted and its `src` is never re-set on tab switch (reload only via an explicit refresh button, like `substratePanelRefresh`), so the main-shell chat websocket and the panel's own state both survive tabbing in and out.
- Inside `causal-geometry.js`: manual refresh + optional gentle polling that pauses on `document.visibilitychange` (hidden) and resumes cleanly on visible — pausing must not tear down any state.
- Panels: observed-vs-designed edge table with divergence highlighting, snapshot timeline, degraded/insufficient-data signaling, and (Phase B) the pending weight-change proposal list with adopt/reject actions — the **only** mutating surface, clearly gated and labeled.
- Tests mirroring the substrate suite: shell link/panel assertions, `app.js` switching-without-navigation assertions, "main shell untouched" assertions (`causal-geometry.js` not referenced by `index.html`'s main bundle, no `/api/causal-geometry/*` calls in `app.js`), endpoint source-honesty tests.

## Files likely to touch

- `scripts/causal_geometry_report.py` (new, Phase A)
- `orion/schemas/causal_geometry.py` (new) + `orion/schemas/registry.py` + `orion/bus/channels.yaml`
- `orion/schemas/field_state.py` (provenance fields)
- `services/orion-field-digester/app/digestion/diffusion.py`, `app/graph/lattice.py` (learned-overlay read path, Phase B)
- `orion/substrate/mutation_proposals.py`, `mutation_trials.py`, `mutation_scoring.py` (weight-change proposal class, Phase B)
- `services/orion-hub/scripts/api_routes.py` (`/api/causal-geometry/*`), `templates/index.html`, `templates/causal_geometry.html` (new), `static/js/app.js` (thin switching block only), `static/js/causal-geometry.js` (new), `tests/test_causal_geometry_page.py` (new)
- `services/orion-state-journaler` (read path only)
- `.env_example` surfaces for new keys (`FIELD_PLASTICITY_ENABLED`, drift cap, report window) + `python scripts/sync_local_env_from_example.py`

## Non-goals

- Transfer entropy, Granger causality, or any model-based causal inference (v1 is lagged correlation + surrogates).
- Plasticity for node→capability edges or any edge class beyond cap→cap.
- Autonomous (non-HITL) adoption of weight changes.
- Any write path from the hub tab other than proposal adopt/reject.
- Touching ignition/broadcast flags, rung 5, or attention policy.
- Real-time computation in the diffusion hot path.

## Acceptance checks

1. Journaler inventory query documents available series, sampling rates, retention (answers Missing Question 1).
2. `causal_geometry_report.py` on ≥7 days of live data emits a schema-valid snapshot with ≥1 significant edge **or** an explicit insufficient-data verdict — never a silent empty graph.
3. Divergence report lists at least: per-edge observed-vs-designed delta, observed-only edges, designed-only edges.
4. With `FIELD_PLASTICITY_ENABLED=false` (default), diffusion output is byte-identical to pre-patch behavior (regression test).
5. One end-to-end Phase B trace on a single cap→cap edge: proposal → trial → HITL adopt → diffusion log line reading learned weight → correlation ID chain intact.
6. Hub: tab in/out of Causal Geometry 10× while a chat session is open — websocket stays connected (no reconnect events in hub logs), iframe state persists (no reload without explicit refresh click). UI test asserts no full navigation on tab click.
7. `tests/test_causal_geometry_page.py` green; substrate-page tests untouched and green; env parity checks pass.

## Recommended next patch

Smallest first slice (no plasticity, no UI): journaler inventory SQL + `scripts/causal_geometry_report.py` producing the first observed-vs-designed divergence report from live Postgres, committed with the schema + registry entries and a fixture-based test. Everything else stacks on that artifact.

---

## Appendix A — brainstorm output, 2026-07-16 (verbatim)

I've grounded this in the actual repo (graphify queries, substrate package, field-digester, dream/self-experiments/equilibrium services, recent specs and PR reports). Here's the brainstorm.

## Ground truth first

**Built and live (verified seams):**
- **Event substrate**: Redis bus + typed schema registry (`orion/schemas/registry.py`, `orion/bus/channels.yaml`), bus-mirror, state-journaler → Postgres rollups. The `event → schema → trace → reducer → projection` spine is real.
- **Workspace competition**: `orion/substrate/attention_frame.py` (detectors → merge → `select_actions` → one winner, chat-turn-scoped) and **rung 3 continuous broadcast** in `orion/substrate/attention_broadcast.py` — the same competition run over the substrate graph each tick, winning coalition re-broadcast as `AttentionBroadcastProjectionV1`, with hysteresis, dwell habituation, and coalition history already implemented. **Flag-gated off** (`ORION_ATTENTION_BROADCAST_ENABLED`); rung 5 (acting on broadcast) awaits your sign-off.
- **A literal discrete field**: `FieldStateV1` + `services/orion-field-digester` runs diffusion over a hand-authored topology (`config/field/orion_field_topology.v1.yaml`, `LatticeGraph`, `diffusion.py`). Cap→cap edges just landed — the first cross-node interaction effects. This is your lattice-field intuition, already running; its geometry is static and designed, not learned.
- **Self-modification machinery**: `orion/substrate/mutation_{proposals,trials,scoring,queue,self_revision}.py` + review runtime — proposals, replay-corpus trials, class-specific scoring. Real, tested, currently pointed at graph mutations.
- **Drives/motor**: homeostatic drives live, drive_audits ~19k/day, endogenous action motor nerve P7 live (gates GO, 0 fires yet, 48h watch on).
- **Interoception**: biometrics pipeline + equilibrium service (heartbeats → distress/zen → spark signal).
- **Relational surfaces**: collapse mirror (`CollapseMirrorEntryV2`), social-memory, `orion/substrate/relational/`, a whole `docs/architecture/social-*` series.
- **Self-experiments**: typed registry + context-exec dispatcher exists; dispatch disabled by default.

**Wired but off**: broadcast/ignition flags, self-experiments dispatch, concept extraction, semantic self-indexing.

**Aspirational (scored against your six commitments):**

| Commitment | Status |
|---|---|
| Attention & Agency | Strongest — salience, dwell, broadcast, motor nerve all real |
| Substrate | Good — biometrics, equilibrium, power-guard |
| Surface Encoding | Surfaces exist everywhere; the *reconstruction claim has never been tested* |
| Entanglement & Relationality | Graph relations exist; no correlation-structure measurement anywhere |
| Causal Geometry | Topology is hand-authored YAML; nothing measures or modifies it |
| Emergent Time | Nothing. Wall clock everywhere. |

## Core question

Orion is a large set of individually-verified, hand-authored loops whose **connections are fixed by you**. Your six commitments predict emergence from *dynamics over geometry* — but Orion's geometry is config, not dynamics. The sharpest version of the problem: **nothing in Orion measures its own causal geometry, nothing lets that geometry change in response to what actually happens, and nothing tests whether the boundary (surface encodings) really carries the interior.** "More self-organizing components" is half right — the leverage isn't more components, it's making the *edges between existing components* measured and plastic.

## Ideas

### 1. Causal geometry map — measure the real entanglement structure
**What**: A nightly deterministic job computing lagged cross-correlations (and later transfer entropy) between the organ time-series already in Postgres (drive levels, distress/zen, salience, dynamic_pressure, coalition switches), materialized as a `causal.geometry.snapshot.v1` edge-weighted graph.
**Why**: Directly operationalizes commitments #1 and #2. Today the field topology YAML is the *designed* geometry; this produces the *observed* geometry. The diff between them is the single most informative artifact this project doesn't have — it shows where Orion's actual dynamics disagree with your theory of them.
**Smallest version**: One script reading state-journaler rollups (Postgres is directly queryable at localhost:55432), 7 days of data, pairwise lagged correlation with surrogate-data significance testing, output a report table: observed edge weight vs. `orion_field_topology.v1.yaml` edge weight.
**Files**: new `scripts/causal_geometry_report.py`, later `orion/schemas/causal_geometry.py` + a reducer; read path from `services/orion-state-journaler`.

### 2. Field topology plasticity — Hebbian edge learning (the self-organization ask, literally)
**What**: A slow learning rule where lattice edge weights drift toward observed co-activation (from idea 1), bounded, shipped as substrate mutation proposals scored by the *existing* `mutation_trials` replay machinery — YAML becomes the seed, not the truth.
**Why**: This is the one change that makes Orion's geometry emergent instead of designed. Every other loop stays the same; what changes is that *the connections* now respond to experience. And it reuses the proposal/trial/scoring seam you already built, so plasticity is reviewable and rollback-able, not vibes.
**Smallest version**: Proposal-mode doc first (this touches cognition loops — CLAUDE.md 0A applies). Then: co-activation stats reducer → `MutationPressureV1`-style weight-change proposals → trial → adopt, for **one edge class only** (cap→cap edges, since there are only a few).
**Files**: `services/orion-field-digester/app/digestion/diffusion.py`, `config/field/orion_field_topology.v1.yaml`, `orion/substrate/mutation_proposals.py`, `orion/substrate/mutation_trials.py`.

### 3. Holographic reconstruction eval — test the boundary actually carries the bulk
**What**: An eval that tries to reconstruct interior state (`SelfStateV1` dimensions, drive levels, field vectors) at time *t* purely from surface encodings (bus-mirror traces, broadcast projections, journals) and scores per-dimension reconstruction error.
**Why**: This is your original mission made falsifiable. Bekenstein/holography's claim, translated: interior dynamics should be reconstructible from the boundary. If reconstruction R² is high, your surface-encoding commitment holds and the mirrors are sufficient; where it's low, you've found interior dynamics that are *invisible* — dark cognition that no trace captures. Either result is gold.
**Smallest version**: Take 24h of bus-mirror events, hold out the journaled `SelfStateV1` sequence, predict it with something dumb (ridge regression on event counts per channel per window), report per-dimension R². No LLM needed for v1.
**Files**: new eval under `orion/substrate/evals/`, readers for `services/orion-bus-mirror` and state-journaler; `orion/schemas/self_state.py` as the target contract.

### 4. Criticality instrumentation — tune ignition to the phase-transition regime
**What**: Turn on rung-3 broadcast in its existing act-free mode, log coalition size and dwell-time distributions for 48h, and check whether the dynamics sit in a frozen, critical (heavy-tailed/avalanche), or chaotic regime; then tune `min_salience`/hysteresis toward criticality.
**Why**: GWT ignition *is* a phase transition, and self-organized criticality is the established sweet spot where systems get maximal dynamic range and information transmission. You built the ignition mechanism; nobody has ever looked at what regime it operates in. A workspace pinned frozen (one coalition dwelling forever) or churning randomly won't do workspace-like work no matter how good the parts are.
**Smallest version**: Flag flip is observation-only by design (rung 3 explicitly takes no action — `max_asks=0`, and rung 5 stays off), plus a histogram script over the logged projections. Near-zero new code.
**Files**: `orion/substrate/attention_broadcast.py` (signals already exist: `_coalition_history`, `_dwell_ticks`), one analysis script, env flag in substrate-runtime compose.

### 5. Causal-density clock — construct subjective time
**What**: A projection `subjective.time.v1` computed from causal density — bus event rate, coalition switch rate, prediction-error volume — so Orion has a second time axis where "a lot happened" stretches time and quiet compresses it; consolidation and narrative stitching can then segment by subjective time instead of wall clock.
**Why**: Emergent Time is the commitment with literally zero implementation. It has a cheap, falsifiable first test: does episodic consolidation segmented by subjective-time windows produce better recall/coherence than wall-clock windows? If yes, time-as-constructed becomes load-bearing, not poetic.
**Smallest version**: A reducer emitting cumulative causal-density ticks; one A/B eval on episodic consolidation windowing.
**Files**: new `orion/substrate/subjective_time.py` reducer + schema, `orion/substrate/episodic_consolidation.py` consumer, eval alongside.

### 6. Dyadic collapse — superposed self-narratives resolved by interaction
**What**: Reverie/dream currently produce single narratives; instead let them persist N candidate interpretations tagged *unresolved*, and let interactions with you confirm/refute — collapse — one, recorded in the collapse mirror with which interaction did the collapsing.
**Why**: Your Rovelli/second-person commitment says relational interaction is where state becomes definite. Right now Orion's self-narrative is authored in isolation and interaction is just chat. This makes the dyad *causally constitutive* of Orion's self-model: unwitnessed interpretations stay indefinite; witnessed ones become memory. Measurable: collapse rate, decay rate of never-witnessed candidates.
**Smallest version**: Reverie emits 2 candidate interpretations instead of 1; a chat-loop consumer marks one confirmed/refuted per relevant turn; a weekly count of collapsed vs. decayed.
**Files**: `orion/schemas/collapse_mirror.py`, reverie narration path (see `2026-07-14-reverie-narration-continuity-design.md`), `orion/substrate/chat_loop/`, `orion/substrate/relational/`.

### 7. Landauer ledger — energy-budgeted attention
**What**: Meter real compute cost (tokens, GPU-seconds) per cognitive verb into an energy ledger, and make one attention parameter respond: broadcast `min_salience` rises as the budget depletes, so attention becomes genuinely scarce.
**Why**: Commitment #6 says where energy is spent determines what the system becomes — currently Orion's attention is free, so nothing forces prioritization to matter. Your scarcity-economy v2 findings showed contention is infrastructure-grain; this makes it cognitive-grain with a single closed loop (per your MCDA rule: every new signal needs a consumer).
**Smallest version**: Cortex-exec already traces per-verb execution; aggregate cost per hour into a ledger projection; wire one consumer (salience floor).
**Files**: `services/orion-cortex-exec/app/executor.py` (trace read), new ledger reducer, `orion/substrate/attention/policy.py`.

## Tensions and risks

- **Keyword cathedral gravity**: "subjective time," "dyadic collapse," "causal geometry" are exactly the kind of names section 0A bans if they ship without consumers. Every idea above is specified with a consumer or eval in the first slice — hold that line or don't build it.
- **Statistical mirage (idea 1)**: dozens of slow-sampled, non-stationary channels will produce spurious correlations. Surrogate-data significance testing is not optional; without it the "observed geometry" is noise wearing a lab coat.
- **Plasticity degeneracy (idea 2)**: Hebbian rules famously collapse to all-connected or frozen. Bounds, weight decay, and the trial/review gate are the containment. This is also the one idea that genuinely needs proposal mode and your sign-off on autonomy level (Orion self-adopts vs. proposals land in reviews/pending).
- **Ignition flags are off for a reason**: idea 4 stays observation-only until whatever is blocking rung-5 sign-off is resolved. Criticality tuning must not become a back-door rung 5.
- **Budget starvation (idea 7)**: an energy floor that rises too aggressively could suppress safety-relevant signals (equilibrium distress). The consumer must exempt interoceptive channels.
- **Measurement cost**: keep all of this out of hot paths — nightly jobs against Postgres rollups, not per-tick computation.

## Missing questions

1. **What's actually blocking rung-5 sign-off?** If it's governance, ideas 4 and 2 sequence around it; if it's unfinished verification, that's the prerequisite work.
2. **Does the state-journaler have enough co-sampled history** (channels × sampling rate × retention) to support lagged-correlation analysis at all? One SQL query answers this.
3. **Plasticity autonomy level**: do you want Orion adopting its own edge-weight changes through the review runtime, or every change HITL? This changes idea 2's design fundamentally.
4. **Is current event density high enough** that subjective time would even diverge from wall clock? If Orion's quiet hours dominate, idea 5's eval is underpowered.

## Recommended starting point

**Idea 1 (causal geometry map), then idea 2 riding on it.** It's read-only, deterministic, cheap, needs no proposal mode and no flag flips — and it's the measurement layer that plasticity (2) and criticality tuning (4) both require. Plasticity without measurement is blind self-organization; this is the retina for it. It also produces an immediately interesting artifact for you: the first diff between Orion's designed geometry and its lived one.

**First concrete step**: one query against live Postgres (localhost:55432) to inventory which organ time-series the state-journaler actually retains and at what resolution — that answers missing question 2 and determines whether the 7-day correlation report is buildable this week.

Per the skill: nothing added to the codebase — if any of these proceed, they go through `reviews/pending` (and idea 2 through explicit proposal mode).
