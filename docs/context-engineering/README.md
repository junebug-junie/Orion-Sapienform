# Context Engineering Pack: Substrate Trace Adoption

This folder is the portable context pack for agents working in Orion services.

Its purpose is to let an agent open a service folder, understand the service's substrate role, implement service-owned substrate trace emission, and reason about how that service's meaningful transitions move through Layers 1–11 of the Orion cognition substrate.

## Core doctrine

```text
Native payloads are local.
Substrate traces are shared.
Typed frames are compiled artifacts.
Every meaningful transition must be causally legible.
```

The current implementation schema for substrate trace events is `GrammarEventV1`. The design concept is **Substrate Trace**.

Use the phrase **substrate trace** in design docs. Use `GrammarEventV1` when referring to the concrete schema/code that currently implements it.

## Why this exists

Earlier design work already pointed here:

- The Organ Signal Gateway design defined common normalized signals, causal parents, dimensional representation, source event ids, and trace propagation.
- The meta-services architecture graph design defined evidence classes: Observed, Declared, Documented, and Inferred.
- The recent Layers 1–10 substrate work proved the pipeline: service traces → reducers → field → attention → self-state → proposal → policy → dispatch → feedback.

This pack turns that into an agent-readable harness.

## Read order for agents

1. `00_substrate_trace_doctrine.md`
2. `01_service_taxonomy.md`
3. `02_service_port_manifest_spec.md`
4. `03_substrate_trace_emitter_pattern.md`
5. `04_layer_1_to_11_pipeline.md`
6. `05_reducer_and_state_delta_pattern.md`
7. `06_frame_runtime_pattern.md`
8. `07_evidence_and_epistemic_classes.md`
9. `08_testing_and_live_proof.md`
10. `09_agent_handoff_template.md`

Then read the service-local files:

```text
services/<service>/AGENT_CONTEXT.md
services/<service>/SERVICE_PORTS.yaml
services/<service>/SUBSTRATE_TRACE_MAP.md
services/<service>/LAYER_PIPELINE_PLAN.md
```

If those files do not exist yet, use the templates under `docs/context-engineering/templates/` to create them.

## What this pack is not

This is not a central emitter service.

Each service owns its own substrate trace emitter because the service owns the semantics of its meaningful transitions.

Shared code may provide schemas, validators, publishing helpers, redaction helpers, and ID helpers. Shared code must not become a god mapper that decides what each service means.

## Layer map

```text
1. Organs / ingress
   local service transitions become substrate traces

2. Trace substrate
   GrammarEventV1 / substrate trace events are persisted

3. Reducers
   trace histories become StateDeltaV1 / ReductionReceiptV1

4. Field digestion
   state deltas perturb lattice/tensor field state

5. Attention
   field state selects what matters now

6. Self-state
   attention + field synthesize operating condition

7. Proposal
   operating condition becomes possible actions

8. Policy
   proposals are gated by risk/consent/scope

9. Dispatch
   approved decisions become dry-run/prepared/dispatch envelopes

10. Feedback
   consequences, failures, blocks, absences are captured

11. Consolidation
   repeated consequences become motifs, expectations, priors, sparse tensor slices, and schema candidates
```

## Service-level rule

Do not ask, "Is this whole service grammar-worthy?"

Ask:

```text
Which service ports produce substrate-relevant transitions?
```

For each port:

```text
What happened?
Where did it happen?
When did it happen?
What did it derive from?
What consequence did it produce?
What evidence supports it?
How should later layers consume it, if at all?
```

## Acceptance standard

A service is substrate-trace ready when:

- its service-local context files exist;
- each meaningful port is classified;
- trace emission is service-owned and fail-open;
- raw payloads, secrets, prompts, completions, and blobs are not mirrored into traces;
- all trace events validate against `GrammarEventV1`;
- source ids, evidence refs, and lineage are present;
- unit tests and live proof queries are documented;
- downstream reducer/field/frame/consolidation needs are explicit.