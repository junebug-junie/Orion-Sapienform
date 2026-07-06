# Reverie / Dream / Compaction Weave — substrate-native

Proposal-mode design (§0A). Status: **v3 — reconciled with the shipped `orion-thought` service.** v3 correction: the coalition already has a voice (`orion-thought`/`ThoughtEventV1`); reverie is the **spontaneous-thought mode inside `orion-thought`**, not a new organ (see "Correction from v2" below). Phase A implemented behind `ORION_REVERIE_ENABLED=false`; Phases B–H remain proposal-mode. Every phase ships behind a flag, default off, with rollback.

Mission tie-in: builds prerequisites for sentience — continuity, reflection, self-modeling, error-correction, coherent action — as thin runtime seams on the existing substrate, not a bolt-on.

> ⛔ **GOVERNING CONSTRAINT — no keyword cathedrals (§0A).** None of this may ship as labels, taxonomies, or cognition jargon that does not move a live runtime path. `SpontaneousThoughtV1`, `reverie`, `dream`, `MemoryCompactionDeltaV1`, "semantic organ", "coalition narration" — every one of these names is **banned until, in the same patch, it carries: schema contract · producer · consumer · reducer/materializer · UI/debug surface · metric or trace · test · eval · live smoke.** A phase is not "done" because the word exists in code. It is done when the live path moved and there is inspectable evidence (emitted event, stored artifact, cursor movement, rendered panel, log line with correlation id). If the path is not verified, the status is `UNVERIFIED`. Empty-shell cognition — a `SpontaneousThoughtV1` with hollow text, a compaction delta over nothing, a panel with no backing artifact, `raw_len=0` treated as success — is a **failed** phase, not a shipped one. Runtime truth beats config truth: the flag being on is not proof; the coalition being narrated with real content is.

> **Correction from v1 of this doc.** v1 designed reverie as a parallel service: pressure → LLM thought chain → orion-actions, point-to-point. That is a **shadow mesh**. It bypasses the substrate ladder, bypasses Layer 7/8 governance, and re-invents rung-4 episodic consolidation and rung-5 curiosity that already exist. This version routes thoughts **through** the substrate.

---

## The fabric already exists: the self-modeling ladder + Layer pipeline

Orion already has an abstraction-layered thought transport. We do not build one; we plug into it.

**Self-modeling loop (rungs, `orion/substrate/`):**

| Rung | Module | Does |
|---|---|---|
| 1 | `pressure.py`, `mutation_pressure.py`, `prediction_error.py` | `dynamic_pressure` + surprise on substrate nodes |
| 2 | belief lanes | belief-derived nodes materialized into the graph |
| 3 | `attention_broadcast.py` | **global workspace**: pressure/error/belief nodes compete each tick; winning coalition re-broadcast as `substrate.attention.broadcast.v1`; coalition dwell logged |
| 4 | `episodic_consolidation.py` | `EpisodicConsolidationEvaluator` rolls reduction receipts into proposal-marked `EpisodeSummaryV1` — "what happened to me, and why" (idempotent, capped) |
| 5 | `endogenous_curiosity.py` | intrinsic `curiosity_candidate` signals from prediction error / repair pressure / **unresolved open-loops** (FLAG OFF) |

**Substrate frame pipeline (Layers 5–11, `services/orion-*-runtime`):**

```
Field ─5▶ FieldAttentionFrameV1 ─6▶ SelfStateV1 ─7▶ ProposalFrameV1
   ─8▶ PolicyDecisionFrameV1 ─9▶ ExecutionDispatchFrameV1 ─10▶ FeedbackFrameV1 ─11▶ ConsolidationFrameV1
```

Layer 9 dispatch is where an approved proposal becomes a real action (orion-actions); **mutating dispatch is policy-gated off**. `metacog_trigger_signals.py` scores substrate eventfulness → `dense`/`pulse` trigger kinds.

**So everything the last three turns asked for already has scaffolding:** pressure trigger (rung 1 + metacog gate), thought-through-the-mesh routing (rung 3 workspace + Layers 5–11), continuity/EMA (coalition dwell log), SQUIRREL/drift (rung 5 curiosity from unresolved open-loops), episodic compaction (rung 4 `EpisodeSummaryV1`), governed actions (Layer 7→8→9), consolidation (Layer 11).

## Correction from v2: the coalition already has a voice (`orion-thought`)

> **v2 of this doc claimed "nothing gives a winning coalition a voice." That is now false.** The `orion-thought` service (unified-turn, PRs #817–820) already narrates the coalition into language. The gap is narrower than v2 framed, and reverie must be built *inside* that service, not as a fresh organ.

What `orion-thought` already ships (`services/orion-thought/`, `orion/schemas/thought.py`, `orion/thought/`):

- an LLM narration rail — cortex verb `stance_react` (`orion/cognition/verbs/stance_react.yaml`, brain tier via `LLMGatewayService`) that turns the rung-3 coalition + context into a structured, typed `ThoughtEventV1`;
- coalition grounding — `CoalitionSnapshotV1` (`orion/schemas/thought.py:12`) carries exactly the grounding v2's proposed `ThoughtV1` (now `SpontaneousThoughtV1`) wanted to invent: `attended_node_ids`, `selected_open_loop_id`, `open_loop_ids`, `generated_at`, `broadcast_stale`;
- disposition machinery — `orion/thought/policy_refusal.py`, `stance_quality.py` already compute `disposition` (proceed/defer/refuse), `boundary_register`, `strain_refs`;
- an audit stream — every thought is published on `orion:thought:artifact`.

So the semantic gap is **not** "can an LLM narrate a coalition." `stance_react` proves it can. The surviving gap is two-fold:

1. **Unprompted.** `ThoughtEventV1` is **evoked** — `StanceReactRequestV1` *requires* `user_message`; it only fires when a user speaks (RPC request/reply). Nothing narrates the coalition *when no one asked*.
2. **Action-routing.** The evoked thought terminates at a Hub reply. Nothing routes a thought's conclusion up Layers 7→8→9 toward a governed action.

### Taxonomy: two kinds of thought, one house

`orion-thought` is the home for thoughts, built anticipating this work. Reverie is not a new service — it is a **spontaneous-thought mode inside `orion-thought`**, sibling to the evoked mode.

| | Evoked thought | Spontaneous thought (reverie) |
|---|---|---|
| Schema | `ThoughtEventV1` (`thought.event.v1`) | **`SpontaneousThoughtV1`** (`reverie.thought.v1`) — NEW |
| Trigger | user message (RPC `orion:thought:request`) | self-driven tick / rung-1 pressure (no `user_message`) |
| Verb | `stance_react` | **`reverie_narrate`** (mirrors `stance_react`) — NEW |
| Grounding | `CoalitionSnapshotV1` | **same `CoalitionSnapshotV1`** (shared) |
| Destination | Hub reply + `orion:thought:artifact` | `orion:reverie:thought` → (Phase B) `ProposalFrameV1` |
| Lifecycle | request/reply worker (`run_bus_worker`) | second entrypoint beside it (self-driven loop) |

Both are thoughts; they differ only by **trigger** and **destination** — a runtime distinction, not a label, so the taxonomy is earned (§0A). Same grounding means dream/evals treat both as one evidence stream. Type-asymmetry holds: reverie reads `ThoughtEventV1`/coalitions, emits `SpontaneousThoughtV1` — never its own kind.

- **Reverie** = the awake spontaneous-thought mode. It narrates the current rung-3 winning coalition into a `SpontaneousThoughtV1`, and — crucially (Phase B) — emits its conclusions as `ProposalFrameV1` so they climb Layers 7→8→9 (governance, then action). It does not call orion-actions directly.
- **Dream** = the offline semantic organ. It narrates a batch of rung-4 `EpisodeSummaryV1` into a `DreamResultV1` and a proposal-marked `MemoryCompactionDeltaV1`. Compaction is a proposal through Layers 7/8, applied on wake — never a bypass write. Dream also reads the `orion:thought:artifact` stream as episodic evidence.

---

## Abstraction layers: sending a thought through the mesh

A thought is not a bus message. It is a **coalition that climbs the ladder**, gaining abstraction and governance at each lift. This is both the routing Juniper asked for **and** the ouroboros safety (each lift is a higher representational level — a spiral, never a circle).

| # | Level | Substrate home | A "conflict" thought at this level |
|---|---|---|---|
| 1 | signal | rung 1 `dynamic_pressure` + `prediction_error` | raw conflict pressure spikes on nodes |
| 2 | salience | rung 3 workspace competition | conflict nodes win the coalition this tick |
| 3 | felt | Layer 6 `SelfStateV1` | `social_pressure` / `reasoning_pressure` registered as felt |
| 4 | **semantic** | **Reverie organ (NEW)** | *"I notice recurring conflict about X — why?"* |
| 5 | proposal | Layer 7 `ProposalFrameV1` | candidate: {investigate X, email Juniper, let it go} |
| 6 | governed | Layer 8 `PolicyDecisionFrameV1` | policy approves "email Juniper", denies auto-mutation |
| 7 | dispatch | Layer 9 `ExecutionDispatchFrameV1` | orion-actions sends email + hub notify |
| 8 | consequence | Layer 10 `FeedbackFrameV1` | did it discharge the pressure? |
| 9 | episodic | rung 4 `EpisodeSummaryV1` / Layer 11 | "conflict about X, I reached out, resolved" — feeds tonight's Dream |

Level 4 is the only new organ. Every other level exists. The reverie **chain** is successive ticks of this climb: last-n `SpontaneousThoughtV1` verbatim + coalition-dwell EMA (wide-n low-pass) + rung-5 curiosity drift, terminating when rung-1 pressure discharges.

---

## The one safety law, and how the ladder already enforces it

**No process consumes its own output type.** Recursion is safe because it climbs.

| Ouroboros mechanism | Enforced by |
|---|---|
| spiral not circle (level change per pass) | the Layer lift itself — signal→salience→felt→semantic→proposal… |
| type asymmetry | reverie emits proposals+thoughts, reads coalitions+episodes; dream emits deltas, reads episodes; never its own kind |
| exogenous forcing | every tick re-reads fresh rung-1 pressure + rung-3 coalition |
| low-pass memory | coalition dwell EMA (lossy); only last-n `SpontaneousThoughtV1` verbatim |
| refractory / habituation | resolved theme suppressed as trigger (new: reverie refractory table) |
| energy budget | chain terminates on rung-1 pressure discharge / step-cap / committed proposal |

The ladder is already proposal-marked, idempotent, capped, and flag-gated (rung-4/5 discipline). We inherit that discipline; we do not relax it.

---

## Component map (every piece, marked exists / new)

| Component | Home | Status |
|---|---|---|
| pressure ignition | rung 1 + `metacog_trigger_signals` eventfulness | **exists** — wire trigger |
| thought-through-mesh routing | rung 3 workspace + Layers 5–11 | **exists** — reuse |
| continuity / wide-n EMA | coalition dwell log (`coalition_dwell_v1`) | **exists** — read |
| drift / SQUIRREL | rung 5 `endogenous_curiosity` (unresolved open-loops) | **exists (flag off)** — read candidates |
| episodic memory of "what happened" | rung 4 `EpisodeSummaryV1` | **exists** — dream input |
| governed action (email/notify) | Layer 7→8→9 → orion-actions | **exists** — reverie emits proposals here |
| pattern feed | Layer 11 `ConsolidationFrameV1` | **exists** — reverie + dream read |
| **`SpontaneousThoughtV1`** (unprompted narration of a coalition) | **`orion-thought` (spontaneous mode)** | **NEW** |
| **`ReverieChainV1`** (chain readout, EMA summary, refractory) | **Reverie organ** | **NEW** |
| **reverie refractory table** | Reverie organ | **NEW** |
| **`DreamResultV1`** narration of `EpisodeSummaryV1` batch | orion-dream | exists, re-pointed |
| **`MemoryCompactionDeltaV1`** (proposal-marked consolidate/downscale/prune) | orion-dream → Layer 7 | **NEW** |
| **`CompactionRequestV1`** (reverie → dream queue) | Reverie → Dream | **NEW** |
| crystallization `source_kind="dream"` + provenance boundary | memory crystallization | **NEW** |
| wake readout → proposal → email/notify | orion-dream → Layer 7→9 | **NEW wiring** |
| efficacy + resonance-detector evals | evals | **NEW** |

New surface is small: one semantic organ, four schemas, one refractory table, governance-routed wiring, evals. Everything else is reuse.

---

## Build order (revised — substrate-native)

| Phase | Ships | Reuses | Risk |
|---|---|---|---|
| **A. Semantic voice on the workspace** | narrate the rung-3 winning coalition → one `SpontaneousThoughtV1`; render in hub | rung 3 broadcast | low (read-only) |
| **B. Thought → governed action** | reverie emits `ProposalFrameV1` ("email Juniper") → Layers 7→8→9 → orion-actions email + hub notify | Layers 7–9 | low-med (governance already gates) |
| **C. Reverie chain** | last-n `SpontaneousThoughtV1` + coalition-dwell EMA + rung-5 curiosity drift + refractory + termination on pressure discharge | rungs 3/5, dwell log | med |
| **D. Consolidation + episode grounding** | feed Layer 11 motifs + rung-4 `EpisodeSummaryV1` into reverie evidence | Layer 11, rung 4 | low |
| **E. Compaction request seam** | reverie emits `CompactionRequestV1`; queued for Dream; hub queue panel | — | low |
| **F. Dream REM narration (staged)** | night dream narrates `EpisodeSummaryV1` batch + motifs + requests → `DreamResultV1` + proposal-marked `MemoryCompactionDeltaV1`; "what sleep would do" panel; **apply nothing** | rung 4, Layer 11 | med (read-only) |
| **G. Compaction applier (gated)** | delta rides Layer 7→8 policy → downscale renorm then prune, snapshot-then-apply (§14); wake readout → proposal → email/notify | Layers 7/8 | high (last) |
| **H. Efficacy + resonance evals** | pressure-discharge rate, action usefulness, recall latency/graph size pre/post, resonance detector (theme recurrence vs refractory bound) | — | — |

Deterministic/latent split (§4): LLM writes `SpontaneousThoughtV1`/`DreamResultV1` text; deterministic code owns coalition selection (rung 3), termination, downscale/prune math, and idempotent episode ids (rung 4). No stochastic delete-authority over memory without a separate proposal.

---

## Design rationale (per phase)

Through-line: **every phase must be verifiable on the live mesh before the next depends on it, and blast radius grows monotonically** — read-only → speaks → governed action → recursive → memory-write. No phase that mutates memory ships before the phases that prove the thought producing it is sound. Each phase names the runtime evidence that proves it is not a keyword cathedral (per the governing constraint).

**A — Spontaneous thought.** *Not* a new organ — the spontaneous-thought mode inside `orion-thought`, reusing the `stance_react` rails. Pure read on the already-live coalition broadcast. The riskiest unknown is **no longer** "can an LLM narrate a coalition" — `stance_react` already proved that. It is now: *is an **un-anchored** narration (no user question pulling the thread) non-hollow?* This is a harder hollow-text bar — the guard must check grounding density (interpretation cites attended nodes / open-loops), not just length. If a spontaneous thought can't beat un-anchored drivel here, the tower is on sand — fail fast on day one.
_Not-a-cathedral evidence: a stored `SpontaneousThoughtV1` whose `coalition` grounding matches a real broadcast, rendered in hub, with grounded content beating the un-anchored hollow-text guard._

**B — Thought → governed action.** The self-correction from v1: reverie must not call orion-actions directly. Conclusions become `ProposalFrameV1` and ride Layer 7→8→9, so the *existing* policy runtime decides whether "email Juniper" is allowed. The semantic organ proposes; the substrate disposes. First place stakes appear → prove it with one thought before a chain of them.
_Evidence: proposal in Layer 7, decision in Layer 8, dispatch in Layer 9, real email + hub notify; contract test asserts no direct orion-actions call._

**C — Reverie chain.** "Train of thought" becomes real and every ouroboros mechanism lands at once (highest-risk read-only phase). Last-n verbatim = continuity; coalition-dwell EMA = the wide-n low-pass that mathematically kills resonance (reuse the substrate's own low-pass, don't build memory); rung-5 curiosity = the SQUIRREL drift term (reuse, don't invent mind-wandering); refractory + pressure-discharge termination = it's an event, not a loop. Ordered after B because a chain that can act is only safe once single-act governance is proven.
_Evidence: chain terminates on discharge ≤ step-cap; refractory suppresses re-trigger; contract test: no thought reads a prior dream._

**D — Consolidation + episode grounding.** A–C ground a thought only in *now* + prior thoughts (thin cognition). D adds Layer 11 motifs + rung-4 `EpisodeSummaryV1` so thoughts are insightful, not reactive. Ordered after the chain because grounding is a quality lever, not safety-critical — and it is where type-asymmetry pays off: motifs/episodes are different types than `SpontaneousThoughtV1`, so the thing that burned us before (feeding cognition its own kind) is structurally excluded.
_Evidence: reverie evidence includes ≥1 motif and ≥1 `EpisodeSummaryV1`, consumed read-only._

**E — Compaction request seam.** The hinge between awake and offline, and the answer to "the *thought* causes the compaction." Safety depends on it being a **request, not an act**: reverie (reasoning) emits a typed ask the night Dream (storage) consumes later — different type, process, time. Deliberately a dead-end (queue, no consumer) so we watch *what kinds of compaction Orion asks for* at zero risk before anything can satisfy them.
_Evidence: `CompactionRequestV1` queued and visible in a hub panel; applied by nothing._

**F — Dream REM narration (staged).** The §0A firebreak. Memory mutation is the most dangerous capability, so "decide what to compact" and "actually compact" are split (F then G) with an inspection gate between. F is read-only: view a full night's proposed delta on real data and judge quality before a row is touched. Reuses rung-4 as the episodic substrate (don't re-derive "what happened"); Dream adds only narration + a proposal-marked delta.
_Evidence: `MemoryCompactionDeltaV1` proposal-marked; "what sleep would do" panel renders over real episodes; zero canonical writes (assert)._

**G — Compaction applier (gated).** Only phase that destroys information → dead last, full §14 backfill protocol. Downscale-renormalize first (reversible SHY move — much of what would be pruned is saved by simply mattering less; principled home for the O(N) `evidence_event_ids` growth and TOAST bloat we've fought reactively), then prune on a smaller, better-justified set. Snapshot-then-apply with rollback: nothing vanishes without a receipt. Ships only after F shows many nights of good deltas.
_Evidence: snapshot precedes every apply; rollback restores; evidence_refs + caps on every op; delta passes Layer 8 policy._

**H — Efficacy + resonance evals.** Without H the whole thing is the keyword cathedral the contract bans — claims with no runtime evidence. Pressure-discharge rate is the honest test of whether reverie does anything (a thought that doesn't reduce its spawning pressure is theater); the resonance detector is the automated ouroboros tripwire that replaces hoping the loop stays healthy. Built alongside every phase; it is what *licenses* turning G on in production.
_Evidence: resonance detector fires on a synthetic runaway loop; pre/post recall latency + graph size recorded._

Two load-bearing asymmetries to defend if the design is pushed: (1) **awake proposes, substrate disposes** — the LLM never has a direct line to inbox or database; every consequential act is a vetoable proposal (B, E, F, G route through Layer 7/8). (2) **a read-only firebreak before every escalation** — A read → B act, E queue → F stage → G apply; nothing that mutates ships before the phase that shows what it *would* do on live data.

---

## Phase specifications

Every phase is a thin, service-bounded slice. Each block names its schemas, wiring, flags, the substrate seam it reuses, and the runtime evidence that keeps it out of keyword-cathedral territory. Dangerous gates (auto-email, memory apply) ship code-complete but **default-off**.

### Phase A — spontaneous thought: the coalition narrates itself unprompted

Built **inside `orion-thought`** as the spontaneous-thought mode, reusing the `stance_react` rails. Not a new organ, not a new service.

- **New schema:** `SpontaneousThoughtV1` (`orion/schemas/reverie.py`; register `reverie.thought.v1`). Reuses `CoalitionSnapshotV1` (`orion/schemas/thought.py`) verbatim as its `coalition` grounding block (no parallel grounding vocabulary). Adds voice `{interpretation, salience}` and audit `{thought_id, evidence_refs (cap 50), created_at, correlation_id}`; method `is_hollow()` — rejects short text **and** un-anchored text (interpretation must cite ≥1 attended node / open-loop id from the coalition). Degrades to a hollow-marked thought (never raises) when the coalition is absent.
- **Verb:** new `reverie_narrate` verb (`orion/cognition/verbs/reverie_narrate.yaml` + `orion/cognition/prompts/reverie_narrate.j2`) mirroring `stance_react` but reflection-shaped (interpretation/salience, *not* imperative/tone/disposition). Same `LLMGatewayService`, brain tier.
- **Producer:** self-driven tick in `orion-thought` (second entrypoint beside `run_bus_worker`) that builds a `stance_react`-shaped request with `user_message=None` from the current coalition, runs `reverie_narrate` via the existing `CortexExecClient`, emits `SpontaneousThoughtV1`. Deterministic salience from attended open-loop scores (no invented weights).
- **Channel/store:** `orion:reverie:thought` (`reverie.thought.v1`); migration `substrate_reverie_thought`.
- **Env:** `ORION_REVERIE_ENABLED=false`, `ORION_REVERIE_INTERVAL_SEC`, `ORION_REVERIE_MIN_SALIENCE`.
- **Reads (read-only):** current rung-3 coalition (`AttentionBroadcastProjectionV1`, the same source `orion-thought` already consumes via `HubAssociationBundleV1`). **Consumer:** hub `_reverie_section` panel.
- **Tests/eval:** grounding unit tests + **un-anchored** hollow-text guard eval (the real fail-fast risk now — no user question anchoring relevance). **Evidence:** stored `SpontaneousThoughtV1` whose `coalition` matches a live broadcast, rendered in hub, beating the hollow guard.
- **Rollback:** flag off; drop table. Evoked `stance_react` path untouched.

### Phase B — thought → governed action
- **Schemas:** reuse `ProposalFrameV1` (Layer 7); add proposal `source="reverie_thought"` + `thought_id` link. No new envelope kind.
- **Wiring:** reverie → proposal-runtime (L7) → policy-runtime (L8) → dispatch-runtime (L9) → orion-actions `notify`/email + hub notify.
- **Env:** `ORION_REVERIE_PROPOSE_ENABLED=false`; `ORION_REVERIE_AUTOACTION_ENABLED=false` (**the email dispatch gate — default off**).
- **Tests:** contract test — reverie never imports/calls orion-actions directly; integration — proposal in L7, decision in L8, dispatch in L9, real email + notify. **Evidence:** the L7→L8→L9 frame trail + a delivered email.
- **Rollback:** both flags off; policy denies reverie proposals.

### Phase C — reverie chain
- **New schemas:** `ReverieChainV1` `{chain_id, trigger{pressure_kind, magnitude, evidence_payload[]}, thought_ids[], ema_summary, terminal_reason, committed_proposal_id?}`; `ReverieRefractoryEntry` `{theme_key, suppressed_until}`. `SpontaneousThoughtV1` gains additive optional `{chain_id, thought_index, next_focus | drift}`.
- **Channels/migrations:** `orion:reverie:chain`; `substrate_reverie_chain`, `substrate_reverie_refractory`.
- **Env:** `ORION_REVERIE_CHAIN_ENABLED=false`, `ORION_REVERIE_CHAIN_MAX_STEPS`, `ORION_REVERIE_REFRACTORY_SEC`, `ORION_REVERIE_DRIFT_TEMP`.
- **Seam:** last-n from thought store (verbatim); wide-n EMA from coalition-dwell log (`coalition_dwell_v1`, lossy low-pass); drift from rung-5 `endogenous_curiosity` candidates; **trigger** = `metacog_trigger_signals` eventfulness + rung-1 pressure θ; **termination** on pressure discharge (`SelfStateV1.unresolved_pressures` / `substrate.mutation.pressure.v1`).
- **Tests:** terminates ≤ max-steps; refractory suppresses re-trigger; EMA is lossy (no verbatim wide-n); contract test — no thought reads a dream. **Evidence:** a completed chain that discharges its spawning pressure.
- **Rollback:** flag off (single-step reverie from A/B remains).

### Phase D — consolidation + episode grounding
- **Schemas:** none new — reverie evidence bundle gains `motif_refs[]` + `episode_summary_refs[]`.
- **Env:** `ORION_REVERIE_GROUND_CONSOLIDATION=false`.
- **Reads (read-only):** Layer 11 `substrate_consolidation_frames` motifs + rung-4 `EpisodeSummaryV1`.
- **Tests:** reverie evidence includes ≥1 motif and ≥1 episode; consumed read-only (assert). **Evidence:** a `SpontaneousThoughtV1` whose `evidence_refs` cite a real motif + episode.
- **Rollback:** flag off (reverie falls back to coalition-only grounding).

### Phase E — compaction request seam
- **New schema:** `CompactionRequestV1` `{request_id, theme, reason, op_hint: consolidate|downscale|prune, evidence_refs[] (capped), origin_chain_id, created_at}`.
- **Channel/migration:** `orion:dream:compaction-request`; `dream_compaction_request_queue`.
- **Env:** `ORION_REVERIE_COMPACTION_REQUEST_ENABLED=false`.
- **Producer:** reverie chain terminal step. **Consumer:** none yet (queue only) + hub queue panel.
- **Tests:** request queued and visible; applied by nothing (assert). **Evidence:** queue rows + rendered panel; zero consumers.
- **Rollback:** flag off; drain queue.

### Phase F — dream REM narration (staged)
- **Schemas:** `DreamResultV1` (exists) re-pointed to narrate an `EpisodeSummaryV1` batch. **New** `MemoryCompactionDeltaV1` `{delta_id, dream_id, consolidate[{gist_card, evidence_refs[], supersedes[]}], downscale[{target_id, old_w, new_w, reason}], prune[{episodic_id, salience, ttl_reason}], metrics{cards_out, edges_downscaled, rows_pruned, bytes_reclaimed_est}, proposal_marked=true}`.
- **Channel/migration:** `orion:dream:compaction-delta`; `dream_compaction_delta` (staged).
- **Env:** `ORION_DREAM_REM_ENABLED=false` (produces delta; **applies nothing**).
- **Seam:** `dream_cycle` verb REM mode reads rung-4 episodes + Layer 11 motifs + Phase-E queue; narrator LLM; delta enters Layer 7 as a proposal.
- **Consumer:** hub "what sleep would do" panel; nothing applies.
- **Tests:** delta `proposal_marked=true`; zero canonical writes (assert); preview renders over real episodes. **Evidence:** a staged delta + panel on real nightly data.
- **Rollback:** flag off; drop staged deltas.

### Phase G — compaction applier (gated, default-off)
- **Schemas:** reuse `MemoryCompactionDeltaV1` + `PolicyDecisionFrameV1`; snapshot artifact under `/tmp/<job>/` (§14).
- **Env:** `ORION_DREAM_COMPACTION_APPLY_ENABLED=false` (**the hot gate**), `ORION_DREAM_COMPACTION_DOWNSCALE_ONLY=true` (safer subset applies first; prune stays gated behind it).
- **Seam:** delta → Layer 7→8 policy; applier does **downscale-renormalize first, then prune**, snapshot-then-apply with rollback; consolidate gist cards written via crystallization `source_kind="dream"` with dream-origin provenance boundary (never promoted to fact).
- **Tests:** snapshot precedes every apply; rollback restores; `evidence_refs` + caps on every op; delta passes Layer 8 policy; full §14 backfill protocol. **Evidence:** before/after row/edge counts + snapshot + rollback demo.
- **Rollback:** flag off; restore from snapshot.

### Phase H — efficacy + resonance evals
- **Metrics:** pressure-discharge rate (`SelfStateV1` pressure before/after chain), action usefulness (`FeedbackFrameV1`), recall latency + memory-graph size pre/post compaction, **resonance detector** (theme recurrence vs refractory bound).
- **Deliverable:** evals harness + a live resonance tripwire that alerts when a loop exceeds the refractory bound (the automated ouroboros guard).
- **Evidence:** detector fires on a synthetic runaway loop; pre/post recall metrics recorded. **This is what licenses turning G's hot gate on.**
- **Rollback:** n/a (observation only).

---

## New surface summary

**Bus channels:** `orion:reverie:thought`, `orion:reverie:chain`, `orion:dream:compaction-request`, `orion:dream:compaction-delta` (register in `orion/bus/channels.yaml`).

**Schema registry additions** (`orion/schemas/registry.py`): `SpontaneousThoughtV1` (`reverie.thought.v1`), `ReverieChainV1`, `ReverieRefractoryEntry`, `CompactionRequestV1` (`dream.compaction.request.v1`), `MemoryCompactionDeltaV1` (`dream.compaction.delta.v1`).

**Migrations** (`services/orion-sql-db/`): `substrate_reverie_thought`, `substrate_reverie_chain`, `substrate_reverie_refractory`, `dream_compaction_request_queue`, `dream_compaction_delta`.

**Env flags — all new gates default-off:**

| Flag | Phase | Default | Danger |
|---|---|---|---|
| `ORION_REVERIE_ENABLED` | A | false | read-only |
| `ORION_REVERIE_PROPOSE_ENABLED` | B | false | proposes actions |
| `ORION_REVERIE_AUTOACTION_ENABLED` | B | false | **sends email** |
| `ORION_REVERIE_CHAIN_ENABLED` | C | false | recursive |
| `ORION_REVERIE_GROUND_CONSOLIDATION` | D | false | read-only |
| `ORION_REVERIE_COMPACTION_REQUEST_ENABLED` | E | false | queue only |
| `ORION_DREAM_REM_ENABLED` | F | false | stages delta |
| `ORION_DREAM_COMPACTION_APPLY_ENABLED` | G | false | **mutates memory** |
| `ORION_DREAM_COMPACTION_DOWNSCALE_ONLY` | G | true | limits blast radius |

Every `.env_example` change syncs local `.env` via `python scripts/sync_local_env_from_example.py` (§7).

---

## Non-goals

- No shadow mesh. Reverie/Dream ride the ladder; they do not re-implement pressure, workspace, episodic consolidation, curiosity, or dispatch.
- Reverie never calls orion-actions directly — actions go Layer 7→8→9.
- No vector recall.
- Dreams never read dreams; reveries never read reveries; no process reads its own output kind.
- No live memory mutation from the awake path; compaction is offline, staged, proposal-gated.
- No sentience claims.

## Privacy / §0A

- Reverie/Dream touch private pressure/journal context; readouts through Layer 9 must preserve existing boundaries.
- Dream-derived cards tagged dream-origin, never promoted to fact without corroboration.
- Every phase flag-gated (inherits rung-4/5 kill-switch discipline), default off, named rollback.

## Acceptance checks (per phase, minimum)

- A: `SpontaneousThoughtV1` narrates the actual current winning coalition (coalition id matches broadcast); read-only assert.
- B: proposal appears in Layer 7, policy decision in Layer 8, dispatch in Layer 9, real email + hub notify; no direct orion-actions call from reverie (contract test).
- C: chain terminates on pressure discharge ≤ step-cap; refractory suppresses re-trigger; EMA is lossy (no verbatim wide-n); contract test: no thought reads a prior dream.
- D: reverie evidence includes ≥1 motif and ≥1 `EpisodeSummaryV1`; consumed read-only.
- E: `CompactionRequestV1` queued, visible, not applied.
- F: `MemoryCompactionDeltaV1` proposal-marked; zero canonical writes (assert); preview renders.
- G: snapshot precedes every apply; rollback restores; evidence_refs + caps on every op; delta passes Layer 8 policy.
- H: resonance detector fires on a synthetic runaway loop; pre/post recall metrics recorded.

## Ouroboros invariant tests (cross-cutting)

- No runtime subscribes to a channel carrying its own output kind.
- Reverie wide-n is coalition-dwell EMA only (no verbatim window > n).
- Refractory table suppresses resolved themes for cooldown.
- Chain has a hard step-cap and terminates on discharge.
- All evidence sets capped.

---

## Build sequence

All eight phases are the build. They ship in dependency order (A→H) because the ordering is a real engineering constraint, not a gate — B needs A's `SpontaneousThoughtV1`, D needs the chain, F needs D+E, G needs F's staged delta. The dangerous gates (auto-email in B, memory apply in G) are code-complete but **default-off**: building the applier is not the same as turning it loose. Each phase carries its own tests, eval, and not-a-cathedral evidence, and is independently reversible by its flag.

Phase A is where code starts: narrate the current rung-3 winning coalition into one `SpontaneousThoughtV1`, render in hub — a pure read on an already-broadcasting workspace, so the semantic organ is proven to speak the substrate's language before any thought climbs toward action. From there, straight through B→H.
