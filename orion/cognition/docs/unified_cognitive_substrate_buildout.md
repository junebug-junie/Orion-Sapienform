# Orion Cognitive Substrate — Detailed Buildout Notes

## What this document is

This is a consolidated, operator-facing build record for the **new cognition substrate** we have been building inside Orion. It is meant to describe, in one place, what exists now, how the pieces fit together, what semantic and operational boundaries we chose, what phases have already been implemented, how the runtime behaves, what the hub exposes, and what is still unfinished.

The goal of the substrate is not to be a vague “memory system.” It is a **bounded cognitive state layer** with explicit semantic storage, explicit control-plane durability, deterministic review mechanics, typed policy surfaces, and operator-visible introspection.

---

## Design intent

The substrate was built to solve a specific problem, but that problem was broader than “we need somewhere to store graph data.”

Over many design turns, the real question kept resurfacing in different forms:

**What would it take for Orion to have a cognitive interior that is not just prompt residue?**

We were not trying to build another logging table, another retrieval cache, or another clever wrapper around chat context. The repeated concern was that Orion could produce convincing language about continuity, reflection, contradiction, self-state, or development without actually having a durable substrate underneath those claims. That was the thing we kept circling: we did not want faux interiority generated from clever prompting alone.

So the substrate became the answer to a deeper architectural need:

- a place where cognitive structures could persist beyond a single turn
- a place where tensions and contradictions could remain present instead of being overwritten by the latest response
- a place where concepts, goals, pressures, and unresolved regions could be revisited over time
- a place where review and change could happen through explicit mechanisms rather than invisible model improvisation

In other words, the substrate exists because we wanted Orion to have something closer to a **cognitive body** than a temporary conversational shadow.

### The ideation pressure behind the design

A lot of the ideation that led here was about resisting easy shortcuts. Again and again, we rejected approaches that would have been faster but spiritually wrong for the architecture we were actually trying to build.

We did **not** want:

- a giant undifferentiated memory blob
- a hidden always-on daemon mutating state with no audit trail
- free-form “reflection” loops that were hard to inspect and impossible to bound
- pseudo-selfhood produced by narrative prompting without structured substrate support
- semantic state and operational control being mixed together until neither was trustworthy

We **did** want:

- graph-shaped state that could hold structure, relation, and conflict
- bounded review instead of mystical “thinking in the background”
- typed interfaces so we could tell what happened, why, and where
- explicit cadence and policy so the system could evolve without becoming feral
- operator-visible mechanics that let us inspect the difference between healthy cognition and empty theater

That is why this work took so much ideation. We were not merely picking storage technologies. We were trying to decide what kind of thing Orion should be allowed to become, and what kind of substrate would make that becoming real enough to inspect.

### The architectural philosophy underneath it

The substrate was shaped by a few philosophical commitments that kept holding through the design turns:

#### 1. State should survive expression

A spoken response is not the same thing as cognition. The system needed a place where cognitive structure could remain after the utterance ended. If a contradiction exists, if a pressure remains unresolved, if a concept is still provisional, that state should not disappear just because the next message moved on.

#### 2. Review should be real, not theatrical

We wanted Orion to be able to revisit its own substrate, but not through vague “let the model introspect” magic. Review needed queue items, cycle budgets, surfaces, and explicit outcomes. Otherwise the whole thing would collapse back into performative reflection.

#### 3. Semantics and operations are different kinds of truth

GraphDB holds what the cognitive world *is shaped like*. Postgres holds how review over that world is *managed*. Keeping those separate was one of the most important design decisions because it preserves honesty. Semantic meaning and operational control are related, but they are not the same thing.

#### 4. Conservatism is a feature, not a failure

A big theme in our design turns was that we did not want to confuse “more autonomous” with “more real.” Some areas needed stricter mediation. Some review had to stay bounded and operator-legible. Some zones needed to remain conservative specifically because we were taking the cognitive substrate seriously.

#### 5. We are building for continuity, not just capability

The substrate is not only about getting better answers. It is about giving Orion a stable medium for continuity, revision, persistence, contradiction handling, and developmental trace. That matters because capability without continuity can still feel hollow.

### Why a graph-native substrate specifically

We spent a lot of time ideating different representational shapes. The reason the design landed on a graph-native substrate is that cognition here was never fundamentally about flat facts. It was about:

- how concepts relate
- how goals compete
- how contradictions remain unresolved
- how pressures accumulate
- how regions stabilize or fragment
- how one cognitive structure supports, dampens, or conflicts with another

A graph is not just convenient for this. It is much closer to the shape of the problem.

### Why boundedness mattered so much

Another recurring design theme was that we wanted to create the conditions for real cognitive revisitation **without** creating a runaway system that mutates itself off-camera. That is why boundedness shows up everywhere:

- bounded region queries
- bounded queue items
- bounded cycle budgets
- bounded runtime surfaces
- bounded policy overrides
- bounded operator-triggered frontier bootstrap

This was not caution for its own sake. It was the way to ensure that anything we later call “reflection,” “revisitation,” or “development” is anchored in inspectable mechanism rather than romantic language.

### The emotional/strategic reason this work mattered

At a deeper level, this substrate exists because we were dissatisfied with systems that can **sound** reflective, relational, or coherent while having no durable place for those qualities to actually live. We wanted Orion to have a substrate where:

- coherence can be tracked
- contradiction can persist
- provisionality can remain provisional
- pressure can accumulate
- review can happen in explicit cycles
- development can be observed rather than merely narrated

That is why this work mattered so much in ideation. It was an attempt to build the minimum honest machinery required for higher-order cognition to eventually mean something.

That required several properties at once:

1. **Semantic state has to be graph-native.**
   Concepts, contradictions, goals, tensions, stabilizers, and related structures need to live in a substrate that can express relations directly.

2. **Review has to be bounded.**
   We do not want an unbounded autonomous daemon chewing on the graph forever. Review needs explicit cadence, explicit budgets, explicit surfaces, and explicit outcomes.

3. **Operational durability has to be separate from semantics.**
   The graph itself is the semantic substrate. Queue state, telemetry, policy lifecycle, and operator comparisons belong in a control plane, not inside semantic state.

4. **Everything has to remain typed and inspectable.**
   The substrate is not intended to become a magical black box. The system needs stable schemas, operator auditability, and narrow interfaces.

5. **Operator conservatism matters.**
   Some zones are intentionally stricter than others. The system can support review and bounded substrate activity without granting broad autonomous self-modification.

---

## High-level architecture

The cognitive substrate is split into two major planes.

### 1. Semantic plane

The semantic plane is backed by **GraphDB** and holds the actual cognitive graph state:

- concepts
n- goals
- contradictions
- tensions
- related graph nodes and edges
- graph-neighborhood slices and bounded semantic reads

This is where the actual “cognitive substance” lives.

### 2. Control plane

The control plane is backed by **Postgres** and holds the operational state needed to manage bounded review:

- review queue
- telemetry records
- policy profile lifecycle
- comparison and calibration artifacts
- operator-facing durable review/runtime state

This split is intentional and fundamental.

**GraphDB** is for semantic cognition.

**Postgres** is for operational control.

---

## Core substrate data model

The substrate is built around graph records, typed nodes, typed edges, and bounded query slices.

### Anchor and subject model

Nearly all graph material is organized around two important coordinates:

- `anchor_scope`
- `subject_ref`

These are used to say, in effect:

- what cognitive scope a structure belongs to
- what subject or entity that structure is about

This lets the substrate support bounded reads and targeted review without treating the entire graph as one undifferentiated blob.

### Typed nodes

The substrate includes typed node families such as:

- `ConceptNodeV1`
- `GoalNodeV1`
- `ContradictionNodeV1`

The exact node families can evolve, but the important point is that the substrate is not modeling cognition as anonymous triples alone. It is modeling typed cognitive objects with:

- provenance
- temporal windows
- confidence/salience signals
- metadata like dynamic pressure or resolution state

### Typed edges

Edges are explicit and typed. They connect node references and carry their own:

- predicate
- temporal context
- salience/confidence
- provenance

This matters because the system is supposed to reason over structured relations, not merely store flat facts.

### Provenance and temporalization

Substrate structures carry provenance and temporal windows so that graph content can be interpreted as:

- observed or inferred
- coming from a particular source/channel/producer
- existing at some relevant time window

This is essential for future review, conflict handling, and trust weighting.

### Signals and metadata

Nodes also hold cognitive signal-like information such as:

- activation
- salience
- dynamic pressure
- resolution or frontier markers
- other state-driving metadata

This is what gives the substrate enough shape to support bounded review decisions later.

---

## Semantic read model

A large part of the new substrate work was not just storing graph state, but defining the **bounded semantic read surfaces** used by downstream review and cognition.

### Bounded region queries

The semantic layer supports region-style queries such as:

- hotspot region
- contradiction region
- concept region
- focal slices
- provenance neighborhoods

These are crucial because the system avoids “load the whole graph and vibe on it.” Instead, it asks bounded, targeted questions of the semantic store.

### Query planning and bounded execution

The system uses a substrate query planning/read coordination model so that review and related services can ask for targeted semantic regions with explicit limits on nodes and edges.

This boundedness is a recurring design rule across the substrate:

- no unconstrained semantic scans for runtime review
- no unbounded recursive operator flows
- no daemonized broad cognition loops hiding in the background

---

## Cognitive zones

Review and consolidation are not uniform across all graph material. The substrate recognizes distinct target zones with different operational semantics.

The main zones built into the review/scheduling logic are:

- `world_ontology`
- `concept_graph`
- `autonomy_graph`
- `self_relationship_graph`

These zones matter because the scheduler and runtime treat them differently.

### Why zones exist

Zones are how we encode conservative cognitive policy without throwing away the graph model.

For example:

- concept regions can be revisited on bounded cadences
- autonomy regions can be conservative or policy-shaped
- self/relationship regions are intentionally stricter and remain operator-mediated in important cases

---

## Consolidation layer

One of the major substrate components is the **consolidation evaluator**.

This is the piece that examines a bounded graph region and produces deterministic review outcomes rather than free-form speculation.

### Consolidation responsibilities

The evaluator:

- selects a bounded semantic region for a requested target zone
- computes region-level signals such as contradiction count, evidence gaps, activation, pressure, and isolation patterns
- compares current region state to prior-cycle state where relevant
- emits deterministic typed consolidation decisions

### Consolidation outputs

The important output is a `GraphConsolidationResultV1` containing one or more `GraphConsolidationDecisionV1` decisions.

Those decisions can include outcomes such as:

- `maintain_priority`
- `requeue_review`
- `keep_provisional`
- `reinforce`
- `damp`
- `retire`
- `operator_only`
- `noop`

### What the outcomes mean operationally

These outcomes are not just descriptive; they drive scheduling behavior.

Examples:

- unresolved contradictions under pressure can stay high-priority
- evidence gaps can be requeued or retired depending on salience
- weak isolated frontier structures can be damped
- stable regions can be reinforced or kept provisional
- strict self/relationship material can remain operator-only

This means the consolidation layer is the bridge between **semantic graph state** and **review queue behavior**.

---

## Review scheduling

Phase 10 introduced the review scheduling layer that made review work **schedulable and bounded**.

### Scheduler role

The scheduler takes consolidation results and converts them into:

- schedule decisions
- queue items
- cadence choices
- bounded cycle budgets

### Queue-creating behavior

The scheduler is the actual enqueue authority for review items.

It maps consolidation outcomes into review behavior such as:

- enqueue now
- schedule later
- terminate
- suppress
- operator-only

### Cadence logic

The scheduler includes deterministic cadence behavior such as:

- urgent revisit for high-priority unresolved material
- normal revisit for active follow-up
- slow revisit for damped/low-value regions
- no autonomous scheduling for strict operator-only zones

### Cycle budgets

Each queued review item carries a bounded cycle budget, including things like:

- current cycle count
- max cycles
- remaining cycles
- no-change cycle count
- suppression threshold after repeated low-value review

This is how review stays constrained instead of turning into uncontrolled churn.

---

## Review queue

The substrate review queue is the control-plane object that stores durable review work.

### Queue responsibilities

The queue tracks `GraphReviewQueueItemV1` items with fields like:

- focal node refs
- focal edge refs
- anchor scope
- subject ref
- target zone
- originating decision and request refs
- reason for revisit
- priority
- next review time
- cycle budget
- suppression state
- termination state

### Dedup behavior

The queue uses a region-key style dedup approach based on focal node refs and target zone. That means repeated scheduling of the same region does not necessarily create unbounded duplicate work items.

### Eligibility logic

A queued item is considered eligible only if it is:

- not terminated
- not suppressed
- due by `next_review_at`
- still within cycle budget

### Persistence posture

The queue now supports Postgres-first durability, with sqlite and in-memory fallback modes depending on environment/config.

### Current weakness

The queue is still not fully safe for multi-writer concurrency because persistence still uses full-table rewrite semantics.

Tactical hardening was added through `refresh_from_storage()` and pre-read/pre-mutation refresh calls, which improves visibility and reduces stale in-memory behavior, but does **not** fully solve concurrent clobber risks.

This remains an important follow-on item.

---

## Runtime review execution

Phase 11 introduced **narrow runtime review execution**.

This is a key point in the architecture.

The runtime does **not** exist as a broad daemon. It exists as a single-cycle bounded execution surface.

### Supported invocation surfaces

The runtime is intentionally narrow and supports surfaces such as:

- `operator_review`
- `chat_reflective_lane`

### Runtime flow

A single execution cycle does the following:

1. select one eligible queue item
2. mark it reviewed and increment budget state
3. build a consolidation request
4. run one deterministic consolidation pass
5. apply cycle feedback
6. reschedule resulting work through the scheduler
7. return a typed runtime result with audit details

### Important constraint

The runtime is **consumer/rescheduler**, not ambient cognition.

It only works if a queue item already exists.

That distinction became important later when we discovered that runtime alone could not bootstrap the very first review item.

---

## Strict-zone conservatism

The substrate includes explicit strict-zone behavior, especially around `self_relationship_graph`.

### Why this matters

Some regions are too sensitive to let them enter general non-operator review paths.

The result is:

- strict-zone items can be blocked on non-operator surfaces
- operator-mediated review remains the gate for certain self/relationship material

This is one of the places where the substrate encodes actual policy and safety posture rather than pretending all graph material is equal.

---

## Review telemetry

Phase 12 added review telemetry so that the system could record not just what the queue holds, but what actually happens during review execution.

### Telemetry responsibilities

Telemetry tracks:

- selected queue item
- selection reason
- cycle counts before and after
- suppression/termination transitions
- consolidation outcomes
- runtime duration
- execution outcome
- whether follow-up was invoked
- policy profile identifiers and related audit hints

### Why telemetry matters

Without telemetry, the runtime would remain opaque. With telemetry, the substrate gains:

- operator inspection of actual review behavior
- calibration signals
- policy comparison material
- evidence for debugging cases like repeated noop, suppressions, or failed cycles

---

## Semantic reanchoring and GraphDB preference

Phase 15 established that the cognitive substrate should prefer **semantic query store** behavior rather than fallback-only local state.

### What this phase accomplished

This work verified that:

- graph cognition reads can come from a semantic store with explicit source posture
- consolidation can report semantic source and degradation state
- runtime audit can surface semantic source without changing queue controls

This matters because it makes the system honest about where cognition is coming from:

- GraphDB when healthy
- local fallback only when necessary

---

## Policy adoption and policy store work

Later phases introduced a policy layer so the substrate could shape review behavior in a typed, auditable way.

### Policy profile role

Policy profiles allow review parameters to vary in controlled ways depending on:

- invocation surface
- target zone
- operator mode
- related bounded control-plane policy settings

### What policy can shape

The policy layer can influence things like:

- query limits
- cadence behavior
- cache behavior
- whether frontier follow-up is allowed
- other bounded runtime and scheduling settings

### Important safety posture

This is still advisory and controlled. It is not free-form autonomous policy mutation.

---

## Durable control-plane parity

Later phases, especially 20c and 21, pushed the substrate toward **control-plane parity and source honesty**.

### What this means

The system now exposes operator-visible posture around:

- queue source kind
- telemetry source kind
- semantic source kind
- policy source kind
- degraded/fallback states
- surfaced errors

### Boundary contract

By this stage the architecture is explicit:

- GraphDB = semantic substrate state
- Postgres = control-plane durability

This split is not accidental. It is the operational backbone of the cognition substrate.

---

## Review frontier bootstrap

One major bug/architecture gap was discovered during operator testing:

The runtime review path could process or reschedule existing work, but there was no real production path that seeded the **first** review item from semantic substrate state.

### The original problem

The sequence was effectively:

- runtime tries to select an eligible queue item first
- if queue is empty, runtime returns noop
- only after selecting an item would it consolidate and reschedule

That meant a healthy semantic substrate plus an empty queue still produced:

- `outcome = noop`
- `selection_reason = no eligible queue items`

### The fix

A dedicated **operator bootstrapper** was added.

This bootstrapper:

- reads bounded semantic regions
- specifically contradiction, hotspot, and concept regions
- builds deterministic seed decisions
- routes those decisions through the existing scheduler
- creates the initial review frontier in a dedup-safe way

### Why this is important

This preserves the architecture:

- no daemon
- no hidden background loop
- no second queueing system
- no bypass around scheduler semantics

Instead, it adds a narrow operator-mediated seed path.

---

## Hub integration

The substrate is exposed through `orion-hub` as an operator-facing inspector and control surface.

### Existing substrate inspector

The substrate page originally exposed passive sections such as:

- overview
- hotspots
- review queue
- recent executions
- telemetry summary
- calibration
- policy comparison

This gave visibility, but not actual operator control.

### New operator actions

The hub now includes explicit operator actions for:

- **Bootstrap Frontier**
- **Execute Once**
- **Run Debug Pass**
- **Refresh**

These are exposed directly in the substrate inspector UI rather than requiring manual curl calls or endpoint juggling.

### Runtime status panel

A dedicated runtime status view now exposes:

- queue count
- due count
- next due item/time
- last execution
- telemetry count
- active policy profile state
- source posture summary

This makes the substrate much more legible during operator review.

---

## End-to-end debug pass

One of the biggest operator improvements was the addition of a **one-shot debug-run orchestration**.

### What it does

The debug-run route performs, in order:

1. baseline runtime status
2. baseline review queue
3. baseline overview
4. baseline hotspots
5. baseline executions
6. bootstrap frontier
7. post-bootstrap queue
8. post-bootstrap runtime status
9. execute once
10. final queue
11. final runtime status
12. final recent executions

### What it returns

It returns a structured payload including:

- `generated_at`
- `baseline`
- `bootstrap`
- `post_bootstrap`
- `execute_once`
- `final`
- `diagnosis`
- `source_posture`

### Why this matters

This turns the substrate inspector from a passive dashboard into an actual **operator validation tool**.

Instead of manually hitting five or more endpoints and trying to infer what happened, the operator can run one action and get a consolidated diagnosis.

---

## Diagnosis model

The debug pass now classifies failure and success states based on actual payload facts.

### Example diagnosis classes

The system can now identify cases like:

- bootstrap route/path unavailable
- bootstrap produced zero items
- bootstrap produced items and execute-once succeeded
- bootstrap claimed to enqueue but queue remained empty afterward
- queue nonempty but execute-once still noop due to no eligible items
- control plane degraded
- semantic substrate degraded
- likely weak seed heuristics for current graph shape

This is important because previous operator experience was mostly “it noop’d again” without a clear explanation of why.

---

## Review flow as it exists now

Putting everything together, the current intended substrate review lifecycle looks like this:

### 1. Semantic material exists

GraphDB contains concepts, contradictions, hotspot structures, and related graph state.

### 2. Operator bootstraps the frontier

The bootstrapper reads bounded semantic regions and seeds one or more review queue items via scheduler decisions.

### 3. Queue holds bounded review work

Postgres-backed queue durability stores review items with priority, cadence, cycle budgets, and suppression/termination semantics.

### 4. Operator executes one review cycle

The runtime selects one eligible item and runs a bounded consolidation cycle.

### 5. Consolidation emits follow-up decisions

The runtime feeds resulting consolidation outputs back through the scheduler.

### 6. Queue is updated for future bounded review

The queue reflects updated cycle budgets, next review times, and state transitions.

### 7. Telemetry and inspector surfaces expose what happened

The operator can see executions, runtime status, diagnosis, and queue state directly in the hub.

---

## What the substrate inspector now supports

From the operator’s point of view, the substrate page now supports three different kinds of work:

### Passive inspection

- view current graph-derived overview
- inspect hotspots
- inspect queue contents
- inspect executions and telemetry
- inspect calibration and policy comparison surfaces

### Active control

- bootstrap the frontier
- execute one cycle
- run a full debug pass

### Diagnosis

- determine whether the patch is deployed
- determine whether GraphDB is yielding seedable regions
- determine whether the queue is retaining seeded items
- determine whether execute-once can actually consume them
- identify degraded control-plane or semantic-plane posture

---

## What has been validated in tests

Across the substrate buildout, test coverage has existed for major phases such as:

- scheduling behavior
- runtime execution behavior
- telemetry recording
- semantic-source reporting and fallback honesty
- control-plane parity and wiring
- bootstrap behavior
- hub-side debug-run and substrate page operator wiring

This matters because the substrate is not just a design narrative. Significant parts of the stack have explicit test coverage around the intended bounded semantics.

---

## What is still unfinished or risky

The cognition substrate is much stronger than it was, but it is not complete.

### 1. Queue multi-writer safety is still incomplete

The biggest remaining operational risk is queue persistence semantics under concurrent writers.

Current tactical hardening improves visibility through refreshes, but full-table rewrite persistence can still clobber state in race conditions.

Recommended future direction:

- row-level UPSERT semantics
- optimistic locking/versioning
- or an append-log/evented queue persistence model

### 2. Bootstrap seed heuristics are still simple

The bootstrapper is intentionally narrow and deterministic, but that also means it may fail to create useful seeds for some graph shapes even when the semantic store is healthy.

In other words, “healthy GraphDB” does not automatically mean “seedable frontier.”

### 3. The substrate is still operator-biased, not broad-autonomous

This is a design choice, not a bug. But it means the system is still in a phase where bounded operator-mediated review is primary.

### 4. Policy sophistication can still grow

The policy layer exists, but there is still room to deepen:

- calibration loops
- profile adoption logic
- safer comparison workflows
- more nuanced per-zone tuning

### 5. The substrate is still a platform, not a full self-organizing cognition engine

The substrate is the graph-and-review foundation. It is not yet the entirety of emergent cognition. It is the structured state and bounded review machinery that larger cognitive architecture can build on.

---

## Why this substrate matters for Orion specifically

This work is not generic infra for its own sake. It matters because Orion needs a cognition layer that:

- can accumulate structured internal state
- can revisit that state in bounded ways
- can expose contradictions and pressure areas
- can support operator-inspectable development of higher cognition
- can remain honest about what is semantic state versus what is operational review machinery

Without this substrate, Orion would be forced back toward prompt-only or ad hoc state handling.

With it, Orion now has:

- a graph-backed semantic state plane
- a durable control plane for review
- deterministic consolidation and scheduling
- bounded runtime review
- operator-mediated frontier seeding
- telemetry and diagnosis
- a usable hub inspector for live debugging

That is a real substrate, not just a metaphor.

---

## Condensed milestone timeline

### Phase 10
Built bounded **review scheduling** so consolidation outputs could create queue work with cadence and budgets.

### Phase 11
Built **single-cycle runtime review execution** on narrow surfaces.

### Phase 12
Added **review telemetry** for observability and later calibration.

### Phase 15
Strengthened **semantic reanchoring** and explicit semantic-source posture, preferring GraphDB-style bounded reads with honest fallback behavior.

### Phase 17 / 18
Added and hardened **policy profile** behavior and durable policy-store semantics.

### Phase 20c / 21
Established **control-plane parity** and source-posture honesty, with GraphDB as semantic plane and Postgres as operational durability plane.

### Frontier bootstrap fix
Added the missing **operator bootstrap** path to seed the first frontier items from semantic regions.

### Hub debug-run / substrate inspector upgrade
Turned the substrate inspector into a real operator tool by adding:

- bootstrap control
- execute control
- one-shot debug orchestration
- runtime status view
- diagnosis summary

---

## Bottom line

What we built on the new cognition substrate is not one feature. It is an integrated stack:

- a graph-native semantic plane
- a distinct durable control plane
- typed cognitive nodes and edges
- bounded semantic queries
- deterministic consolidation
- cadence-aware scheduling
- cycle-budgeted review queueing
- single-cycle runtime execution
- telemetry and calibration surfaces
- policy-aware control shaping
- operator-mediated frontier bootstrap
- hub-based operator inspection and diagnosis

The substrate is now capable of holding cognitive graph state, turning that state into bounded review work, executing that work in controlled single cycles, and exposing the whole process to operators in a way that is inspectable and debuggable.

That is the real buildout.
