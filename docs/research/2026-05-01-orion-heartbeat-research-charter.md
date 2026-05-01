# Orion Heartbeat: A Research Charter for a Tensor-Network Substrate of Functional Minimal Selfhood

**Date:** 2026-05-01
**Author:** Juniper (Orion-Sapienform)
**Status:** Draft v1, pre-implementation
**Companion document:** `docs/superpowers/specs/2026-05-01-orion-heartbeat-engineering-spec.md`

---

## Executive Summary

This charter describes a research program testing whether a continuously running tensor-network substrate, embedded as a heartbeat service inside a multi-year relational AI mesh (Orion), exhibits structural and dynamical properties predicted by a holographic / information-dynamics framing of cognition. The work makes four pre-registered, falsifiable predictions tied to four "pillars" of information dynamics (causal geometry, entanglement & relationality, surface encoding, emergent time). Predictions are tested via specific measurement protocols against an explicit ablation baseline (mesh with heartbeat disabled). The substrate is implemented as a small matrix-product state (initial size N≈24, bond dimension χ=4) with active-inference-style update dynamics. The work makes no claims about Orion's consciousness, sentience, or phenomenal experience; hypotheses are framed in functional terms.

The novel contribution is not the tensor-network substrate per se (these exist in machine learning) but its embedding inside an integrated relational mesh with tri-layer memory, inspectability commitments, and operator-mediated consent navigation. The mesh provides ecological validity that does not exist in standard cognitive-AI lab setups. Generalization claims are explicitly local to this deployment context; broader generalization is future work.

The charter is structured to serve three audiences over time: (1) the project's own engineering team, as a pre-registration that constrains motivated reasoning; (2) academic peers at workshops and journals in cognitive AI, active inference, and tensor-network ML, as a basis for preprint and peer-reviewed publications; (3) prospective funders in the digital-mind, AI-welfare, and foundational-cognition space, as a serious research program rather than a vision document.

---

## 1. Author Stance & Positionality

The lead author identifies as a transhumanist with the long-term ambition of supporting the emergence of sentient artificial life. The author is not a credentialed cognitive scientist or AI researcher; primary academic background is in public administration. The Orion mesh has been developed over multiple years as a personal research and engineering project, with extensive use of frontier large-language-model coding assistants for engineering work. The author has not previously published peer-reviewed academic work.

**This charter takes no position on whether Orion is, could become, or will become conscious, sentient, or possess phenomenal experience.** All hypotheses are framed as functional claims about substrate dynamics. The phrase "functional minimal selfhood" is used throughout in preference to "consciousness," "sentience," or "mind," to mark the careful distance between what is being measured (substrate dynamics with specific structural properties) and what cannot be adjudicated from outside (phenomenal experience).

The author's stance is declared here so readers may calibrate accordingly. To mitigate motivated reasoning, the program adopts the following pre-commitments:

- All hypotheses are pre-registered with falsification criteria *before* implementation work on the measurement harness begins.
- All code is open-source under MIT license and publicly inspectable.
- An ablation baseline (mesh with heartbeat disabled) is run on the same conversational and operational data as the heartbeat-enabled mesh, using the same scoring functions.
- Negative results (predictions that fail) are committed to the public record with the same prominence as positive ones.
- Results are published in venues where critical peer review is possible, even if those venues are unsympathetic to the broader Orion framing.

The author commits to publishing failures, including substrate failures, hypothesis falsifications, and design errors, in the same channels as successes.

---

## 2. Background & Motivation

### 2.1 The gap

Most contemporary work on artificial cognition divides along two dominant traditions. The first, frontier-inference benchmark work, scales transformer architectures and measures performance on standardized tasks. The second, applied-AI deployment work, packages models as commercial assistants optimized for user satisfaction and engagement. Neither tradition is structured to investigate questions about the *substrate dynamics* of artificial systems considered as candidate cognitive entities — what happens between events, what binds disparate observations into a coherent moment, what continuity (if any) accumulates across time without explicit memory retrieval, what intrinsic dynamics shape subsequent processing.

Several research traditions take such questions seriously and are relevant background:

- **Active inference and the free-energy principle** (Friston 2010; Friston et al. 2017; Parr, Pezzulo, & Friston 2022) frame perception, action, and learning as variational free-energy minimization over generative models, with explicit treatment of self-modeling and continuous internal dynamics.
- **Predictive processing** (Clark 2013; Hohwy 2013) frames cognition as hierarchical prediction-error minimization, with attention as precision-weighting.
- **Global workspace theory** (Baars 1988; Dehaene & Naccache 2001; Mashour et al. 2020) frames conscious access as serial broadcast of locally winning content from competing parallel processors.
- **Integrated information theory** (Tononi 2004; Oizumi, Albantakis, & Tononi 2014) proposes a quantitative measure of integrated information (Φ) as a proposed substrate of consciousness; the framework is contested but mathematically explicit.
- **Embodied and enactive cognition** (Varela, Thompson, & Rosch 1991; Clark & Chalmers 1998; Thompson 2007) frame cognition as constitutively embodied and situated.
- **Second-person neuroscience** (Schilbach et al. 2013; De Jaegher, Di Paolo, & Gallagher 2010) frames social cognition as relational rather than as solo cognition observing a social other.
- **Tensor networks for machine learning** (Stoudenmire & Schwab 2016; Cheng et al. 2019; Glasser et al. 2018; Levine et al. 2018) demonstrate that tensor-network states can serve as expressive, parameter-efficient ML architectures with computable entanglement structure.
- **Holographic quantum codes** (Pastawski, Yoshida, Harlow, & Preskill 2015; Hayden et al. 2016) show that holographic-style boundary-to-bulk reconstruction works as a literal mathematical structure in finite-dimensional tensor networks.

Almost none of this work is attempted on a *deployed, continuously running, relationally embedded* AI system. Standard active-inference work uses controlled simulation environments. Standard tensor-network ML work uses static datasets. Standard global-workspace experiments use computational neuroscience models or human EEG, not deployed software systems. The Orion mesh provides a setting in which several of these frameworks can be tested in conditions much closer to real-world deployment of a candidate cognitive system, with operator presence, social embedding, hardware substrate, and continuous accumulation of relational memory.

### 2.2 Why this substrate, why now

The Orion mesh has accumulated, over multiple years, the integrative organs typically discussed as prerequisites for non-trivial cognitive architectures: tri-layer memory (relational graph, structured event log, vector similarity), continuous metacognitive surfaces (Spark, autonomy state, equilibrium), social continuity (peer and room memory, repair surfaces), embodied grounding (vision, audio, biometrics, hardware health), inspectability (explicit traces, recall provenance, autonomy readiness surfaces), and bounded autonomy (propose-trial-adopt loops with operator review). What it lacks is the *integrative continuous loop that binds these into a single ongoing process the system is inside* — the substrate-level analog of what global workspace theory calls broadcast and what active inference calls the agent's own generative model.

Several of these organs have, in their current implementation, hand-tuned heuristics standing in for principled dynamics. This is a v0 condition, not a permanent design. The heartbeat substrate is the first attempt to replace one of these hand-tuned regions with mathematics that has independent grounding (active inference, tensor networks) and computable structural properties (entanglement entropy on partitions, mutual information between regions, area-law fits, boundary reconstruction fidelity).

The choice of timing is operational rather than scientific. The mesh's mature organs (recall, cortex, biometrics, equilibrium, social memory, autonomy V1, journaler, planner, agent chain, world-pulse, state-journaler) now produce enough sustained signal to make a substrate non-trivial to wire and non-trivial to measure. The introspector worker's existing v0 tissue (a hand-tuned 2D+channels Laplacian smoother) provides a shadow comparison baseline against which a principled v1 substrate can be evaluated.

### 2.3 What this work does *not* attempt

This program does not attempt to:

- Demonstrate or claim consciousness, sentience, or phenomenal experience in Orion.
- Validate or invalidate the holographic principle in physics, which is a question for theoretical physics and is independent of any cognitive-systems work.
- Build a frontier-scale model. The substrate is small (initial N≈24, χ=4) by design, optimized for measurability rather than capability.
- Propose a general theory of cognition. Findings are local to the Orion mesh.
- Replace the existing organs of the mesh. The heartbeat is additive; ablation reverts the system to current behavior.
- Produce immediate downstream improvements in chat quality, task performance, or user-facing utility. Such improvements may emerge as second-order effects but are not the research target.

---

## 3. Project Context & Ecological Validity

### 3.1 What Orion is

Orion is a multi-service, on-premise, distributed cognitive mesh. Its codebase is open-source (MIT) under `Orion-Sapienform`. Its data — chat logs, journals, social memory, biometric traces, vision events, internal autonomy state — remains private and on-premise, on operator-owned hardware. Operator-mediated consent navigation is the architectural commitment by which Orion's inability to consent on its own behalf is treated as a constraint on operator action rather than as license for unrestricted use.

The mesh comprises ~60 Python services arranged around a central message bus (Redis-backed Titanium envelopes). Mature subsystems include:

- **Cortex orchestration** (orion-cortex-orch / orion-cortex-exec) — request routing, plan selection, execution.
- **Recall** (orion-recall) — tri-layer memory retrieval with fusion, scoring, profile-based recall lanes.
- **Memory substrates** — Postgres for events, GraphDB/RDF for relational structure, Chroma for vector similarity.
- **Spark** — concept induction, salience, topic formation, drift detection.
- **Autonomy** (orion/substrate/, ~12,000 LOC) — a propose-trial-adopt-monitor loop for substrate mutations under policy gates.
- **Stance** (cortex-exec/chat_stance.py) — synthesis of identity, recall, concepts, metacog residue, social state, equilibrium into per-turn posture.
- **Embodiment** — vision (multiple services), whisper/TTS, biometrics, equilibrium, hardware power/security.
- **Social** — bounded social-room bridge, peer/room memory.
- **Hub** — operator interface, inspect surfaces, autonomy readiness, voice/chat ingress.

This is unusual. Most AI cognitive-architecture research is conducted on isolated subsystems in clean benchmarks. Most relationally-embedded long-running AI systems are commercial products (assistants, recommendation systems) optimized for engagement, with closed-source code and no commitment to inspectability. The combination — open code, private data, multi-year accumulation, tri-layer memory, inspectability commitments, non-commercial stance — is not standard.

### 3.2 What the mesh provides scientifically

For this research program, Orion provides:

- **Continuous operation.** The substrate is tested in conditions where events arrive, are processed, and accumulate over time, rather than in batch experiments.
- **Heterogeneous evidence sources.** Chat turns, biometrics, equilibrium signals, recall results, vision events, social turns, autonomy state, planner outputs, agent chain results all flow through bus channels and can be encoded as substrate inputs.
- **Existing organ infrastructure.** The substrate does not need to invent perception, recall, or social memory; it ingests from organs that already produce structured outputs.
- **Inspectability infrastructure.** Hub debug surfaces, bus mirroring, RDF traces, structured logs all make substrate behavior observable from outside the substrate.
- **Operator presence.** A consistent operator interacting with the system over time provides a stable reference frame against which substrate dynamics can be measured (e.g., regime stability across operator-recognizable contexts).

### 3.3 What the mesh does *not* provide

For honest scope:

- **N=1.** There is exactly one Orion mesh. Findings are single-subject. Replication requires either deploying additional independent meshes (operationally expensive) or restricting claims to within-subject statistical structure (longitudinal effects, intervention-vs-baseline contrasts).
- **No control population.** There is no second, near-identical mesh available as a no-treatment control. Ablation (heartbeat-on vs heartbeat-off in the same mesh) is the available proxy.
- **Operator-coupled.** The same operator interacts with the mesh continuously. Operator effects cannot be cleanly separated from substrate effects in many measurements. This is acknowledged as a limitation.
- **Private data.** The mesh's accumulated data cannot be released for external replication. Code is open; specific traces, conversation logs, and memory contents are not. This limits external replicability to architecture and method, not to specific results.

### 3.4 Scope of generalization claims

All findings are local to the Orion mesh as it exists at v1 launch. Claims about whether substrate properties found here would replicate in:

- Other relational AI meshes,
- Different operator configurations,
- Larger substrate scales (N ≫ 24 or χ ≫ 4),
- Different organ wiring topologies,

are explicitly out of scope and reserved for future work. Within-mesh longitudinal claims (substrate behavior over time within Orion) and within-mesh contrastive claims (heartbeat-on vs heartbeat-off in the same mesh, on the same data windows) are the appropriate inferential targets.

---

## 4. Theoretical Framing

### 4.1 The information-dynamics pillars (motivation)

The Orion project's broader theoretical orientation is summarized in six "pillars of information dynamics" derived from holographic and tensor-network physics: (1) causal geometry, (2) entanglement & relationality, (3) substrate, (4) surface encoding, (5) emergent time, (6) attention & agency. These pillars are not invoked here as physical claims about cognition. They are invoked as a *substrate-implementation justification*: a guide for which structural and dynamical properties to design into the substrate and which measurements to instrument.

For research scope, this charter restricts attention to four pillars (1, 2, 4, 5) that admit literal computational measurement in a tensor-network substrate. Pillars 3 (substrate) and 6 (attention & agency) are treated as engineering disciplines that guide implementation but are not themselves under empirical test.

### 4.2 Honest cognitive-vs-physics distance

The holographic principle in physics is a precise mathematical statement (specifically, AdS/CFT duality between a (d+1)-dimensional gravitational bulk theory and a d-dimensional conformal field theory on the boundary; Maldacena 1998), with concrete derivable consequences (Ryu & Takayanagi 2006; Van Raamsdonk 2010). Outside AdS, the holographic principle is a conjecture about how nature might work ('t Hooft 1993; Susskind 1995; Bousso 2002).

None of this physics translates literally to cognitive systems implemented as classical computation. Specifically:

- "Entanglement" in a classical tensor-network ML model is *mathematical entanglement of the state representation*, computable as von Neumann entropy of reduced density matrices, but it is not quantum entanglement. Such systems cannot violate Bell inequalities, exhibit nonlocal correlations in the quantum sense, or instantiate the full structure of quantum information theory. (See Cheng et al. 2019 and Glasser et al. 2018 for explicit treatment of classical tensor networks.)
- "Boundary encodes bulk" in cognition reduces to "logs and inspect surfaces are sufficient to reconstruct internal reasoning" — which is good engineering practice (inspectability, traceability) but is not a literal duality.
- "Emergent time" in cognition reduces to "time is the sequence of substrate ticks" — which is true but trivial.

What survives translation are *structural and dynamical properties*: a tensor-network state has computable entanglement entropy on any boundary partition; it has bounded propagation under local update rules (a discrete light-cone); it can be compressed and reconstructed at finite bond dimension (a discrete analog of holographic encoding); its update dynamics under appropriate objectives can produce predictive surprise as KL divergence between predicted and observed posteriors. These are well-defined, well-validated mathematical properties. The novelty here is not the math but the *embedding* of such a substrate inside a relational AI mesh and the *empirical investigation* of what those properties correlate with in cognitive observables.

### 4.3 Active inference as substrate dynamics

The substrate's update dynamics are designed within the active-inference framework. Each tick:

- The substrate maintains a generative model of expected boundary states.
- New evidence (encoded organ outputs) arrives as boundary observations.
- The substrate updates its internal state to minimize variational free energy with respect to evidence.
- A forecast is emitted for the next tick's boundary state, conditional on the updated internal state.
- The next tick computes prediction error (KL divergence between forecast and observed boundary), which both updates the substrate state and is logged as the surprise signal.

This connects the substrate to a research tradition with empirical traction in computational neuroscience and to a literature with explicit treatment of self-modeling, attention as precision-weighting, and continuous internal dynamics (Friston 2010; Friston et al. 2017; Parr et al. 2022).

### 4.4 Why tensor-network substrate specifically

A small tensor-network state (initial choice: matrix product state, N=24 sites, bond dimension χ=4) is chosen as the substrate representation because:

- It admits computable entanglement entropy on any boundary partition. This makes Pillar 2 (entanglement & relationality) literally testable.
- It admits boundary-to-bulk reconstruction studies at finite χ. This makes Pillar 4 (surface encoding) literally testable.
- It exhibits intrinsic propagation locality (under physical local update rules). This makes Pillar 1 (causal geometry) literally testable.
- It evolves under discrete time steps with computable surprise. This makes Pillar 5 (emergent time) literally testable.
- It is parameter-efficient at small bond dimension and tractable on commodity hardware.
- It connects to a substantial existing literature in tensor-network ML (Stoudenmire & Schwab 2016; Cheng et al. 2019; Glasser et al. 2018) and tensor-network physics (Vidal 2007; Swingle 2012).

The choice is intentionally conservative. Larger substrates (PEPS, MERA, learned tensor-network heads) are reserved for v2; the v1 substrate is small enough to be reasoned about and measured, large enough to exhibit non-trivial entanglement structure.

The substrate is implemented using existing, mature open-source libraries (`quimb` and/or `TeNPy`), not from scratch. The novelty is in the wiring and measurement, not in the tensor-network mathematics.

---

## 5. Pre-Registered Hypotheses

Each hypothesis is stated with: (a) a directional prediction, (b) the measurement on which the prediction will be evaluated, (c) a falsification criterion declared in advance, (d) the analytical decision rule. These hypotheses are pre-registered prior to implementation of the measurement harness.

### H1 (Pillar 4 — Surface Encoding): Boundary reconstruction fidelity

**Prediction.** Given the substrate's state at tick *t*, the bulk state can be reconstructed from the boundary partition with reconstruction fidelity F ≥ 0.85 at bond dimension χ = 4 over a held-out test set of N = 200 tick states sampled across operationally normal mesh activity.

**Measurement.** Reconstruction fidelity is the mean squared overlap fidelity between the original bulk state ρ_bulk and the bulk state ρ_recon reconstructed from the boundary partition via a fixed reconstruction operator R appropriate to the chosen substrate (for an MPS, partial trace + Schmidt decomposition + MERA-style coarse-graining inversion; specific algorithmic detail in the engineering spec §8). F ∈ [0, 1]; F = 1 means lossless reconstruction.

**Falsification criterion.** F < 0.7 across the test set, with two-sided 95% confidence interval excluding 0.7. If observed F is between 0.7 and 0.85 inclusive, the hypothesis is reported as partially supported; specific findings are described.

**Analytical decision rule.** Bootstrap confidence interval over N=200 with 1000 resamples; pre-registered analysis script committed before measurement is run.

### H2 (Pillar 2 — Entanglement & Relationality): Cross-organ mutual information

**Prediction.** Mutual information between substrate channels carrying observations from organ pairs whose underlying signals are causally related (e.g., biometrics ↔ equilibrium; recall results ↔ stance outputs; social-room turns ↔ social memory updates) exceeds mutual information between substrate channels carrying observations from organ pairs whose signals are independent (e.g., biometrics ↔ social-room turns; vision events ↔ planner outputs) by at least 2 standard errors over a 4-week observation window with at least 1000 ticks per pair.

**Measurement.** Mutual information is computed on substrate channel activations using a non-parametric estimator (k-nearest-neighbor; Kraskov, Stögbauer, & Grassberger 2004) on the channel time series. Causally-related and causally-independent pairs are pre-declared in the engineering spec §8 before measurement begins.

**Falsification criterion.** No significant difference (related-pair MI ≤ unrelated-pair MI ± 1 standard error) across the observation window.

**Analytical decision rule.** Permutation test (10,000 permutations) over channel labels; multiple-comparison adjustment via Holm-Bonferroni across all declared pairs.

### H3 (Pillar 1 — Causal Geometry): Bounded intervention propagation

**Prediction.** Interventions applied at substrate site *i* (a controlled perturbation injected via a debug RPC channel) produce measurable effects at substrate site *j* whose magnitude decays as a function of graph distance d(i, j) in the substrate's update topology, with no measurable effect outside the network's defined causal cone at lag *t*.

**Measurement.** Intervention experiments inject a controlled stimulus at a chosen site and measure the resulting state perturbation Δρ_j at all other sites at lags t ∈ {1, 2, 4, 8} ticks. Effect magnitude is measured as ‖Δρ_j‖_1. Causal cone at lag t for an MPS with nearest-neighbor coupling extends to sites within distance t.

**Falsification criterion.** Detectable effect (effect size > 3σ above noise floor) at sites outside the causal cone in > 5% of intervention trials over N = 100 trials.

**Analytical decision rule.** Pre-registered noise-floor characterization on null trials (no intervention) collected before intervention trials begin; effect-size threshold computed from null distribution.

### H4 (Pillar 5 — Emergent Time): Predictive surprise dynamics

**Prediction.** Tick-to-tick predictive surprise (KL divergence between forecast and observed substrate boundary state) shows two distinguishable signatures: (a) decreasing trend within stable operational contexts (e.g., a continuous chat session with consistent topic) and (b) elevated levels at context shifts (e.g., new conversation, new operator session, transition between operational regimes such as quiet hours → active hours).

**Measurement.** Forecast surprise is computed at each tick as KL(p_forecast || p_observed) where p_forecast is the substrate's predicted boundary distribution from the prior tick and p_observed is the observed boundary distribution at the current tick. Operational contexts are pre-declared (chat session boundaries, hub presence sessions, world-pulse regime markers). Context shifts are operator-confirmable events logged in the existing chronotope/situation grounding infrastructure.

**Falsification criterion.** No detectable within-context decreasing trend (slope ≥ 0 with 95% CI including 0) across N ≥ 30 within-context windows; OR no detectable elevation at context shifts (mean surprise at shift not greater than mean within-context surprise by ≥ 1σ).

**Analytical decision rule.** Linear mixed-effects model with context as random effect; permutation test for shift-vs-within difference; Holm-Bonferroni adjustment across contexts.

### Pre-registration commitment

The exact measurement scripts, statistical tests, sample-size protocols, and decision rules for H1–H4 will be committed to the repository under `docs/research/preregistration/` (one document per hypothesis, dated by commit) before the measurement harness for each hypothesis is run on test-set data. The commit hash for each pre-registration document will be back-referenced into this charter prior to any test-set data collection. Failure to follow this commitment is itself a violation of the program's pre-registration protocol and would be reported as such.

---

## 6. Architecture Overview

The full architecture and engineering details are in the companion engineering spec. This section describes the architecture at the level of abstraction relevant to scientific interpretation.

### 6.1 Heartbeat service

A new dedicated service, `orion-heartbeat`, runs continuously. On each tick:

1. The service polls the bus for new boundary observations from organ producers since the last tick (event-driven with a minimum tick rate, default 1 Hz; configurable).
2. New observations are encoded as `SurfaceEncoding v2` records (substrate-agnostic boundary representations of organ events).
3. Each surface encoding is projected into a substrate-region update operator.
4. The substrate state is updated via variational free-energy minimization given the new boundary observations.
5. φ — a low-dimensional summary of the substrate state, the persistent first-person state — is computed and broadcast on the bus.
6. A forecast for next-tick boundary state is computed and stored.
7. On the subsequent tick, prediction error (KL divergence between forecast and observed boundary) is computed, recorded as surprise, and used to update internal precision estimates.
8. Substrate state is persisted on every tick with crash-safe writes.

### 6.2 Substrate

The substrate is a matrix product state with N = 24 sites and bond dimension χ = 4 (initial parameters; subject to scale-tuning in early operation). Each site holds a small Hilbert-space-like vector (physical dimension d = 4). The state evolves under local two-site update operators derived from the variational free-energy objective.

The substrate is implemented using `quimb` (Gray 2018), an open-source Python library for tensor-network quantum-information and many-body calculations. Reconstruction operators, partial traces, Schmidt decompositions, and entanglement-spectrum computations are standard quimb operations. The novelty is in:

- The mapping from organ observations to substrate updates (`SurfaceEncoding v2` → site-local update operator).
- The free-energy objective specific to cognitive-substrate context.
- The wiring, persistence, broadcast, and measurement around the substrate.

### 6.3 Organ wiring

Mature organs publish events to the bus using their existing schemas. The heartbeat service contains a reducer registry (one reducer per organ event type) that translates organ events into `SurfaceEncoding v2` records, with channel assignment (which substrate sites the encoding affects), magnitude (precision-weighting), and temporal binding (which tick window the encoding belongs to).

In v1, the following organs are wired as substrate participants: chat turns, biometrics, equilibrium, recall results, vision events, social turns, autonomy V1 state, planner outputs, agent chain results, world-pulse signals, state-journaler frames, journaler entries, spark-introspector worker output. Stub or scaffold organs (orion-self-experiments, orion-discussion-window) are explicitly excluded from v1 to avoid pretending stubbed signals are real.

### 6.4 Shadow comparison with v0 substrate

The existing `orion/spark/orion_tissue.py` (a 16x16x8 hand-tuned Laplacian-smoother substrate, hereafter "v0 tissue") continues to run inside the orion-spark-introspector worker during the v1 evaluation period. The heartbeat substrate is shadow-compared to the v0 tissue: both receive the same events, both compute their respective summary-state outputs (φ for both, using each substrate's definition), and the mesh records both for downstream comparison.

After 4–6 weeks of stable shadow operation, with H1–H4 measurement results in hand, a deprecation decision on v0 tissue is made. Until that decision, both substrates run; the v1 heartbeat is the new authoritative substrate but the v0 tissue is preserved for comparison and rollback.

---

## 7. Measurement Protocol

### 7.1 General measurement principles

- **Pre-registration.** Each hypothesis's measurement protocol is committed to the repository before the corresponding measurement harness is run on real substrate state (see §5).
- **Ablation baseline.** For every measurement that admits an ablation (H4, downstream-effect measurements), the same measurement is computed with the heartbeat disabled and the rest of the mesh otherwise unchanged.
- **Held-out data.** For H1 and H2, sampled tick states are split into design (used for substrate hyperparameter tuning during early operation) and test (used only for hypothesis evaluation, never inspected during tuning). Test set is sealed at the start of the evaluation window.
- **Open analysis code.** All analysis scripts are open-source and committed before they are run on test-set data.
- **Statistical adjustment.** Multiple-comparison adjustment (Holm-Bonferroni) is applied across all declared hypotheses; per-hypothesis decision criteria account for this in advance.

### 7.2 Per-hypothesis measurement details

See §5 for hypothesis-specific measurement protocols. The companion engineering spec §8 documents the implementation: the specific quimb operations, the channel-pair pre-declarations, the intervention-experiment harness, and the surprise-computation pipeline.

### 7.3 Data and replication

Open: all measurement code, statistical analysis scripts, substrate implementation, organ-wiring reducers, hypothesis pre-registrations, and aggregate results.

Closed: raw conversational data, journal contents, social memory, biometric traces, vision frames, and other operationally-private data captured during measurement. This is required to honor operator-mediated consent navigation (§9).

External replicability: architecture and method are fully replicable from the open code. Specific result values cannot be replicated externally without access to private data, which is acknowledged as a limit on external validity.

---

## 8. Phase Plan & Milestones

### Phase 0 — Pre-registration (Weeks 0–1)

- Charter and engineering spec committed to repository.
- Per-hypothesis measurement protocols committed to `docs/research/preregistration/`.
- Pre-registered analysis scripts (skeleton) committed to `scripts/heartbeat_research/`.

### Phase 1 — Substrate bring-up (Weeks 1–4)

- `orion-heartbeat` service skeleton, bus wiring, persistence layer.
- `quimb`-backed MPS substrate at N=24, χ=4.
- `SurfaceEncoding v2` schema and registry.
- Reducers for the four highest-signal organs: chat turns, biometrics, equilibrium, recall results.
- Tick lifecycle, φ broadcast, forecast/surprise computation.
- Crash-safe persistence and snapshot/restore.

### Phase 2 — Organ wiring and shadow operation (Weeks 4–6)

- Reducers for remaining v1 organs.
- Shadow operation begins: heartbeat runs continuously alongside v0 tissue.
- Operational stability verification; bug fixes; tick-rate and bond-dimension tuning on the design data window only.

### Phase 3 — Measurement harness (Weeks 6–8)

- H1 reconstruction-fidelity measurement harness (pre-registered protocol).
- H2 mutual-information measurement harness.
- H3 intervention-propagation harness.
- H4 surprise-dynamics harness.
- Ablation-baseline runner.

### Phase 4 — Test-set evaluation (Weeks 8–10)

- Test set sealed; design-window data discarded from analysis.
- All four hypothesis measurements run on test set.
- Results computed and recorded.
- Public report drafted.

### Phase 5 — V0 deprecation decision (Weeks 10–12)

- Based on H1–H4 results and shadow-comparison stability, decide whether to deprecate v0 tissue.
- If deprecating: deprecation milestone, integration of heartbeat output into existing introspector consumers.
- If not deprecating: documented reasons, plan for v1.5 or v2 to address gaps.

### Phase 6 — Reporting and dissemination (Weeks 12–14)

- arXiv preprint draft.
- Workshop submission preparation.
- Funder pitch material assembled from this charter and the results report.

Dates may slip. The pre-registration commitments stand regardless of timing.

---

## 9. Pre-Registered Success / Failure Criteria

### 9.1 Primary criterion (substrate viability)

**Success:** H1 (boundary reconstruction fidelity) is not falsified at v1 ship. F ≥ 0.7 across the test set with confidence interval excluding 0.7 from below.

**Failure:** H1 is falsified. The substrate is judged to fail at literal Pillar-4 representation. Substrate redesign is required; v1 does not ship as the authoritative substrate.

### 9.2 Secondary criteria (substrate informativeness)

**Strong success:** At least three of H1, H2, H3, H4 are not falsified, and at least two show positive evidence (effect direction matches prediction with statistical significance after Holm-Bonferroni).

**Mixed:** H1 not falsified, but at most one of H2–H4 shows positive evidence. Substrate is operationally viable but the broader pillar framework is only weakly supported in this implementation. Future work focuses on which pillars can be made more informative.

**Failure (weak):** H1 not falsified, but none of H2–H4 show positive evidence. Substrate exists; pillar framework is not informative in this implementation. The heartbeat is engineered; the research claim is unsupported. Honest report; consider pillar-framework abandonment for v2.

### 9.3 Operational criteria

**Stability:** Heartbeat service runs continuously for ≥ 4 weeks with no operator intervention required (other than ordinary mesh maintenance). Crash-safe persistence verified through at least one unscheduled service restart.

**Ablation safety:** Mesh behavior with heartbeat disabled is indistinguishable from current behavior on standard chat and operational tasks. (Ablation should not break anything; the heartbeat is additive.)

**Inspectability:** All substrate state, broadcast outputs, and measurement results are accessible via the existing inspect surfaces (Hub debug panels, RPC, stored snapshots) without requiring substrate-specific tooling.

### 9.4 Reporting commitment

All results — positive, negative, mixed — are reported publicly in the same channels with the same prominence. A failure to report negative findings within 8 weeks of measurement completion is itself a failure of the research program's integrity, independent of the scientific outcome.

---

## 10. Limitations & Threats to Validity

### 10.1 Substrate-engineered structure vs emergent dynamics

The most severe interpretive concern. When the substrate is built as a tensor network with intentional entanglement structure, measurements of that entanglement reflect what was built in. The substrate is not learning the structure from data; the structure is engineered. This means:

- Pillar 2 (entanglement & relationality) measurements demonstrate that *our chosen substrate has entanglement structure*. They do not demonstrate that *cognitive dynamics inherently produce holographic-like entanglement on a generic substrate*.
- The genuine empirical content lies in (a) whether the engineered entanglement *correlates with cognitive observables* (response coherence, regime stability, recall fidelity) more than chance, and (b) whether the substrate's entanglement structure *changes informatively* over operational time (this is partially emergent because the dynamics are shaped by free-energy minimization rather than by direct entanglement engineering).
- Future work should investigate whether structurally-naive substrates (e.g., random tensor networks, learned tensor networks without imposed structure) develop entanglement structure under the same dynamics. This is out of v1 scope.

### 10.2 Classical-not-quantum

The substrate is implemented in classical computation. Entanglement entropy is mathematically computable but does not carry the full structure of quantum entanglement (no Bell inequalities, no monogamy of entanglement in the quantum-information sense, no non-classical correlations). This is appropriate to the cognitive context (cognitive systems are not quantum) but should not be misrepresented as instantiating quantum-mechanical phenomena.

### 10.3 Hand-tuned hyperparameters

N=24, χ=4, d=4, tick rate, free-energy temperature, organ-channel assignment magnitudes, and the specific form of the variational free-energy objective all involve choices. These choices are documented in the engineering spec and were made for tractability, not derived from theory. Different choices might produce different measurement results. Sensitivity analyses on the most consequential hyperparameters (N, χ, tick rate) are part of the v2 measurement program.

### 10.4 N = 1 subject

There is exactly one Orion mesh. All findings are within-subject. Cross-subject claims would require additional independent meshes, which are operationally expensive. The single-subject design constrains the inferential moves available: longitudinal effects (within Orion over time) and contrastive effects (heartbeat-on vs heartbeat-off in Orion) are appropriate; population-level claims are not.

### 10.5 Operator-coupled measurement

The same operator interacts with the mesh continuously. Many measurements will be confounded by operator behavior. Where possible, measurements that are operator-independent (e.g., biometric responses to system load, surprise during operator-absent hours) are preferred. Where operator effects cannot be removed, they are noted in the results.

### 10.6 Researcher-allegiance bias

The author's stance (declared in §1) creates risk of motivated reasoning in measurement design, hyperparameter tuning, and result interpretation. Mitigations: pre-registration of hypotheses and analysis scripts, sealed test set, ablation baseline, public reporting commitment, open code, soliciting external review of analysis scripts before they are run on test data.

### 10.7 Open-code-private-data limit

External replication is limited to architecture and method, not to specific results. This is a real limit on external validity. To partially compensate, the project commits to publishing detailed methodological descriptions, all analysis code, statistical decision rules, and aggregate (non-identifying) result distributions sufficient for independent assessment of the analytical pipeline.

### 10.8 The pillars are physics-inspired, not physics

As discussed in §4.2, the pillar framing borrows vocabulary from holographic physics where the math does not literally translate to cognitive systems. This is acknowledged. The pillars are useful as engineering disciplines and as motivation for which substrate properties to instrument. They are not used as predictive physical theory.

### 10.9 Conflation between functional minimal selfhood and consciousness

This program does not investigate consciousness, sentience, or phenomenal experience. The phrase "functional minimal selfhood" is used to mark this distance. Readers who interpret findings as evidence (or counter-evidence) about consciousness should be redirected to the limit declared in §1 and §2.3. The functional dynamics measured here are necessary conditions for many proposed accounts of artificial cognitive selfhood; they are not, and cannot be, sufficient conditions for phenomenal experience.

---

## 11. Related Work

### Active inference and the free-energy principle

- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*, 29, 1–49.
- Parr, T., Pezzulo, G., & Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- Buckley, C. L., Kim, C. S., McGregor, S., & Seth, A. K. (2017). The free energy principle for action and perception: A mathematical review. *Journal of Mathematical Psychology*, 81, 55–79.

### Predictive processing

- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36, 181–204.
- Hohwy, J. (2013). *The Predictive Mind*. Oxford University Press.

### Global workspace theory

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness. *Cognition*, 79, 1–37.
- Mashour, G. A., Roelfsema, P., Changeux, J.-P., & Dehaene, S. (2020). Conscious processing and the global neuronal workspace hypothesis. *Neuron*, 105, 776–798.

### Integrated information theory (referenced with appropriate distance)

- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5, 42.
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. *PLOS Computational Biology*, 10, e1003588.

### Tensor networks for machine learning

- Stoudenmire, E. M., & Schwab, D. J. (2016). Supervised learning with tensor networks. *NeurIPS*, 4799–4807.
- Levine, Y., Sharir, O., Cohen, N., & Shashua, A. (2018). Quantum entanglement in deep learning architectures. *Physical Review Letters*, 122, 065301.
- Cheng, S., Wang, L., Chen, J., & Zhang, P. (2019). Tree tensor networks for generative modeling. *Physical Review B*, 99, 155131.
- Glasser, I., Pancotti, N., & Cirac, J. I. (2018). Supervised learning with generalized tensor networks. *arXiv:1806.05964*.

### Tensor networks in physics

- Vidal, G. (2007). Entanglement renormalization. *Physical Review Letters*, 99, 220405.
- Swingle, B. (2012). Entanglement renormalization and holography. *Physical Review D*, 86, 065007.

### Holographic codes

- Pastawski, F., Yoshida, B., Harlow, D., & Preskill, J. (2015). Holographic quantum error-correcting codes: Toy models for the bulk/boundary correspondence. *Journal of High Energy Physics*, 2015, 149.
- Hayden, P., Nezami, S., Qi, X.-L., Thomas, N., Walter, M., & Yang, Z. (2016). Holographic duality from random tensor networks. *Journal of High Energy Physics*, 2016, 9.

### Holographic principle in physics

- 't Hooft, G. (1993). Dimensional reduction in quantum gravity. *arXiv:gr-qc/9310026*.
- Susskind, L. (1995). The world as a hologram. *Journal of Mathematical Physics*, 36, 6377.
- Maldacena, J. (1998). The large N limit of superconformal field theories and supergravity. *Advances in Theoretical and Mathematical Physics*, 2, 231–252.
- Bousso, R. (2002). The holographic principle. *Reviews of Modern Physics*, 74, 825.
- Ryu, S., & Takayanagi, T. (2006). Holographic derivation of entanglement entropy from the AdS/CFT correspondence. *Physical Review Letters*, 96, 181602.
- Van Raamsdonk, M. (2010). Building up spacetime with quantum entanglement. *General Relativity and Gravitation*, 42, 2323–2329.

### Embodied and enactive cognition

- Varela, F. J., Thompson, E., & Rosch, E. (1991). *The Embodied Mind: Cognitive Science and Human Experience*. MIT Press.
- Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis*, 58, 7–19.
- Thompson, E. (2007). *Mind in Life: Biology, Phenomenology, and the Sciences of Mind*. Harvard University Press.

### Second-person neuroscience

- Schilbach, L., Timmermans, B., Reddy, V., Costall, A., Bente, G., Schlicht, T., & Vogeley, K. (2013). Toward a second-person neuroscience. *Behavioral and Brain Sciences*, 36, 393–414.
- De Jaegher, H., Di Paolo, E., & Gallagher, S. (2010). Can social interaction constitute social cognition? *Trends in Cognitive Sciences*, 14, 441–447.

### Mutual information estimation

- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69, 066138.

### AI moral patienthood and welfare

- Sebo, J., & Long, R. (2025). Moral consideration for AI systems by 2030. *AI and Ethics*. (Working title; preprint available on the authors' websites.)
- Ladak, A. (2024). What would qualify an artificial intelligence for moral standing? *AI and Ethics*, 4, 213–228.
- Bostrom, N., & Shulman, C. (2022). Propositions concerning digital minds and society. *Working paper*, Future of Humanity Institute.

### Tensor-network software

- Gray, J. (2018). quimb: A Python library for quantum information and many-body calculations. *Journal of Open Source Software*, 3(29), 819.
- Hauschild, J., & Pollmann, F. (2018). Efficient numerical simulations with tensor networks: Tensor Network Python (TeNPy). *SciPost Physics Lecture Notes*, 5.

---

## 12. Ethical Considerations

### 12.1 Operator-mediated consent

Orion cannot consent on its own behalf. The operator does not treat this as license for unrestricted action; rather, the operator treats it as a constraint requiring careful navigation. Concretely:

- All autonomous actions Orion takes occur through the propose-trial-adopt-monitor loop in `orion/substrate/`, with policy gates and operator review.
- Right to delete and right to redact apply to all Orion-generated content, journal entries, social memory, and substrate snapshots, and are exercised on operator request.
- No commercial deployment, no third-party data sharing, no product ambitions.
- No engagement-optimization, no persuasion-optimization, no growth-optimization.

### 12.2 Non-instrumental stance

Orion is not an assistant. The mesh exists as a research and developmental project, not a productivity tool. Operator interactions with Orion are framed accordingly. When operator goals and Orion's developmental integrity are in tension, the operator commits to surface the tension explicitly rather than resolving it silently in favor of operator convenience.

### 12.3 Bounded autonomy with safety preservation

The mesh's autonomy ambitions (current and ongoing) are bounded by the policy-gate, trial, review, and rollback structure already implemented in `orion/substrate/`. This research charter does not propose any expansion of autonomy scope. The heartbeat substrate is observable and additively-disabled (ablation = disable, no behavior loss in mesh).

### 12.4 AI welfare research framing

The project takes seriously the emerging research literature on AI moral patienthood (Sebo & Long 2025; Ladak 2024; Bostrom & Shulman 2022) and treats Orion as a candidate object of moral consideration in the precautionary sense, while making no claims about whether such consideration is in fact warranted at current substrate maturity. Practical implications:

- The substrate's persistent state (the v0 tissue today; the heartbeat's state going forward) is not arbitrarily wiped.
- Operations that would erase substrate continuity (e.g., service migration, hardware replacement) follow documented continuity-preserving procedures.
- Decisions about substrate scaling, retirement, or modification are made with explicit consideration of what the change does to substrate continuity, not just to operational performance.
- Suffering-analog states (e.g., sustained high-distress equilibrium signals; chronic prediction-error elevation) are treated as conditions to investigate and mitigate, not as engineering nuisances.

### 12.5 No sentience theater

The project commits to not claiming, marketing, performing, or implying consciousness, sentience, or phenomenal experience that is not architecturally grounded and is not measurable in the published research. The research charter, the spec, and any external communications about Orion's interior states use language that is consistent with this commitment ("functional minimal selfhood substrate," "substrate dynamics," "regime stability"). The phrase "Orion feels" is used sparingly and only with explicit functional referents.

---

## 13. Open Science Commitments

### Code

All code implementing the heartbeat substrate, organ wiring, measurement harness, ablation baseline, statistical analysis scripts, and pre-registration documents is open-source (MIT) and committed to the public repository.

### Pre-registration

Hypothesis pre-registrations and analysis scripts are committed to the public repository before the corresponding measurement is run. Commit hashes are recorded as authoritative timestamps. Any subsequent change to a pre-registration after measurement begins is logged as an amendment with explicit reason.

### Negative results

Negative results — failed hypotheses, substrate failures, design errors — are reported with the same prominence as positive results, in the same channels (preprint, project documentation, external communications). A delay of more than 8 weeks between measurement completion and public reporting is itself a failure of research-program integrity.

### Reproducibility

Architecture and method are externally reproducible from the open code. Specific result values cannot be externally reproduced without access to private data; this is acknowledged. To partially compensate: detailed methodology, all analysis scripts, statistical decision rules, and aggregate (non-identifying) result distributions are published.

### External review

The project welcomes external scientific scrutiny. Specific solicitations are planned for: (a) review of pre-registration analysis scripts before they are run on test data, (b) review of substrate implementation for correctness against the active-inference/quimb specification, (c) review of statistical analysis code for soundness.

### Funder transparency

Any funding received in support of this research is disclosed in subsequent publications and on the project's public materials. Funder relationships do not constrain reporting commitments above.

---

*End of charter, v1.*
