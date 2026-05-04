# Orion meta-services — architecture → property graph (GraphDB) — design

**Date:** 2026-05-04  
**Status:** Draft — operator review; **no implementation** until an implementation plan is approved after this spec.  
**Working name:** `services/orion-meta-services` (or equivalent).  
**Scope:** How **bounded repo/service context** is turned into **durable, honest claims** in the **existing GraphDB substrate path**, with **ephemeral markdown hydration** for humans and Orion prompts; explicit **limitations** and a **v2** path toward **observed interconnectivity** and **repo documentation discipline**.

---

## 1. Purpose

Orion needs a way to accumulate **structured architectural knowledge** (organs, themes, dynamics, constraints) that is **queryable alongside cognitive substrate** data, without pretending that stale READMEs or hand-wavy LLM prose are ground truth.

This spec defines:

1. A **consumer** pipeline: **`ExtractionScope` → grounded interpretation → validated projection → `SubstrateGraphMaterializer` → GraphDB** (same deployment patterns as today’s substrate store; see §5).
2. An **epistemic model** for every edge and node: **Observed**, **Declared**, **Documented**, **Inferred** (§3).
3. **v1** boundaries: what we will and will not claim (§4, §10).
4. **v2**: **telemetry-backed** coupling plus **engineering discipline** (repo `ARCH.md` / skills-style artifacts) to make the graph **more deterministic** (§8–§9).
5. **Determinism goals**: same inputs + same tool versions → **as close as practical** to the same graph delta (§7).

**Non-goals (v1):** PageIndex integration; treating chat memory as the sink for architecture prose; silent upgrade of inferred claims to observed truth; full OTel/bus-derived service graphs without a separate feeder design.

---

## 2. Problem statement

1. **Stale human docs:** READMEs and ad-hoc markdown drift; they must not be the only evidence for critical structure.
2. **Undefined “knobs”:** Configuration meaning is not uniformly machine-readable unless we adopt an explicit **contract surface** (e.g. `.env_example` as key schema, optional `ARCH.md` / manifest — see §8).
3. **Weak static interconnectivity:** Compose `depends_on`, shared networks, and code references are **priors**, not proof of runtime traffic.
4. **Thin “cognitive engine” hand-wave:** A frontier model alone does not “know” GraphDB. **Interpretation** must be paired with **ontology-locked projection** (`SubstrateGraphRecordV1` + materializer) and **provenance**.

---

## 3. Epistemic classes (normative)

Every materialized node or edge MUST carry enough metadata for consumers to answer: **how do we know this?**

| Class | Definition | Typical sources |
|--------|------------|-----------------|
| **Observed** | Measured from runtime or captured traffic (time-bounded) | OpenTelemetry service graph edges, aggregated bus publish/subscribe logs, health checks used only as *liveness*, not semantics |
| **Declared** | Machine-parseable, versioned artifacts | `docker-compose` links/`depends_on`/networks; bus **channel constants** and catalog; import/client references in code; `.env_example` **keys** (not secret values) |
| **Documented** | Human-authored text in repo | `README.md`, design docs — **optional** evidence, often stale |
| **Inferred** | Model-synthesized holistic claims | Ethos, “why this organ exists,” narrative coupling — **must** cite Declared/Observed/Documented evidence where asserting facts about behavior or coupling |

**Rules**

- **Inferred** MUST NOT be presented or queried as **Observed**.
- Assertions of **runtime interconnectivity** in v1 MUST have at least one **Declared** or **Observed** leg; otherwise they are **quarantined** (separate `anchor_scope`, `risk_tier`, or rejected — implementer chooses one policy and documents it).
- **Documented** claims SHOULD carry **source path + commit hash** from the extraction manifest when available.

---

## 4. v1 scope (what ships first)

### 4.1 Inputs

- **`ExtractionScope`**: produced **outside** this service by whoever has context (Cursor script, CI, operator, future orchestrator). Fields include at minimum `scope_path` (e.g. `services/orion-cats`), optional `hints.recent_paths`, optional `git_ref`.
- **Deterministic expansion** (policy-coded, not model-chosen): e.g. always include `README.md`, `docker-compose.yml`, `Dockerfile`, `.env_example`, primary `app/main.py` (or service entrypoint) **when present** under scope.

### 4.2 Processing

1. Build **`source_manifest`**: paths + content hashes (or blob ids) + `git_ref` when known.
2. **Gather** Declared facts (compose, env keys, optional static bus channel grep against allowlist).
3. **Optional:** attach **Documented** excerpts (bounded size).
4. **LLM (frontier or gateway):** emits **only** JSON (or equivalent) that decodes to an intermediate DTO; **no** direct SPARQL from the model.
5. **Validate** against schema; map through a **deterministic adapter** to **`SubstrateGraphRecordV1`** (same pattern as `map_concept_profile_to_substrate` — see `orion/substrate/adapters/concept_induction.py`).
6. **`SubstrateGraphMaterializer.apply_record`** → **`GraphDBSubstrateStore`** (or equivalent approved write path).

### 4.3 Outputs

- **Durable:** graph under a dedicated **named graph URI** or clearly namespaced IRIs (recommended: **separate named graph** from default substrate cognitive slice unless unified queries are explicitly desired — e.g. `http://conjourney.net/graph/orion/architecture` vs existing `SUBSTRATE_GRAPHDB_GRAPH_URI`).
- **Ephemeral:** **hydration** endpoint returns **markdown** synthesized from a **bounded subgraph** (template and/or secondary LLM pass **constrained** to not introduce new factual claims — policy TBD in implementation plan).

### 4.4 API sketch (illustrative)

- `POST /v1/architecture/snapshots` — body includes `ExtractionScope` + optional flags; response includes `snapshot_id`, manifest summary, materialization status, and **`hydrate_url`** (see prior brainstorm: client should not need SPARQL).
- `GET /v1/architecture/snapshots/{id}/hydrate.md` — returns markdown for injection or local save.

Sync vs async for long Frontier runs is left to the implementation plan; contract should support **202 + status** if needed.

---

## 5. Alignment with existing GraphDB / substrate

- **Write path:** Reuse **`SubstrateGraphRecordV1`** + **`SubstrateGraphMaterializer`** + **`GraphDBSubstrateStore`** (`orion/substrate/materializer.py`, `orion/substrate/graphdb_store.py`) unless a reviewed alternative (e.g. RDF bus to `orion-rdf-writer`) is chosen for a specific artifact type.
- **Ontology:** Either **reuse** existing `node_kind` values with **`subject_ref`** pointing at repo paths, or **introduce** narrow new kinds (e.g. `architecture_subsystem`, `architecture_claim`) — **decision in implementation plan** to avoid polluting cognitive `concept` queries. This spec requires **explicit** choice and Hub/query impact notes.
- **Env:** Same family as `SUBSTRATE_STORE_BACKEND`, `SUBSTRATE_GRAPHDB_ENDPOINT`, `GRAPHDB_URL` / `GRAPHDB_REPO` (`services/orion-hub/.env_example` documents these patterns).

---

## 6. Ingress and “who generates scope”

- **`ExtractionScope` is upstream.** The meta-service **consumes** it; it does not guess the business reason for the run.
- **Producers:** Cursor/local scripts, CI (changed paths), operators, future cortex verbs.
- **The “cognitive engine”** is **not** the HTTP layer: it is **grounded LLM + schema validation + substrate adapter + materializer** (§4.2).

---

## 7. Determinism (goals and limits)

**Goals**

- **Frozen extraction policy:** deterministic file expansion rules; capped bytes; stable ordering of inputs into the model context.
- **Record all versions in snapshot metadata:** `prompt_template_id`, `adapter_version`, `model_id`, tokenizer/API version if applicable.
- **Content-addressed manifest:** `source_manifest` hashes; reject or warn on dirty tree without explicit flag.
- **Pure adapter:** mapping from validated DTO → `SubstrateGraphRecordV1` is **code**, not model output.

**Limits**

- Frontier APIs may be **non-deterministic** across retries; treat LLM output as **stochastic**. Mitigations: **low temperature**, **structured decoding**, **optional** “second pass” consistency check, and **storing** raw model JSON in object storage or append-only log is **out of v1** unless added in plan.
- **Merge semantics:** `SubstrateGraphMaterializer` identity merge may make absolute determinism across **time** hard; snapshot boundaries and `graph_id` / provenance must document **which run** wrote what.

---

## 8. v2 — Observed interconnectivity

**Objective:** Add **Observed** edges for real coupling.

**Candidate feeders (non-exclusive)**

- **OpenTelemetry:** service graph, RPC edges, trace-linked spans (align with `docs/superpowers/specs/2026-05-03-hub-otel-traces-metrics-observability-design.md` and gateway instrumentation; join keys such as `otel_trace_id` must not be conflated with bus correlation ids).
- **Bus telemetry:** extend beyond **live UI** (`services/orion-bus-tap`) to **durable, aggregated** channel statistics (publisher identity, envelope kinds, rates) suitable for **time-windowed** graph edges.

**Requirement:** Observed edges MUST carry **time window**, **evidence refs** (trace id, rollup batch id), and **sampling caveats**.

---

## 9. v2 — Engineering discipline (repo artifacts)

**Objective:** Reduce reliance on stale READMEs by making **small, reviewable** architecture facts **part of normal commits**.

**Proposal (normative direction for v2; exact filenames are negotiable)**

- Per service (or per organ), maintain a **short, structured** file such as **`ARCH.md`** (and/or `docs/arch/<service>.md`) with **stable sections** that map cleanly to graph projections, for example:
  - **Role** (one paragraph, factual)
  - **Declared dependencies** (links to other services by name, not prose only)
  - **Bus channels** (publish/subscribe lists if applicable)
  - **Operational knobs** (reference `.env_example` keys with **one-line semantics** each)
  - **Non-goals / boundaries** (what this service must not do)

- Optionally align with **skills-style** conventions (machine-checkable headings or a tiny YAML front-matter block) so CI or a pre-commit hook can **validate presence** when certain paths change.

- **Commit policy:** changes under `services/<name>/` that affect external contracts SHOULD touch **`ARCH.md`** (or the chosen artifact) in the same change; bots/agents SHOULD read that file when generating **`ExtractionScope`** or when proposing graph deltas.

This does **not** replace telemetry; it **anchors Documented** class claims to something **reviewable in diff**.

---

## 10. Limitations (explicit callouts)

1. **v1 does not prove runtime coupling** beyond **Declared** priors (compose, code references, channel names in static form). Narrative interconnectivity without telemetry is **Inferred** or **Documented** only.
2. **README drift** remains a failure mode; v1 mitigates via epistemic labeling and v2 mitigates via **`ARCH.md` discipline** and telemetry.
3. **Config semantics** are incomplete without **per-key documentation** in `ARCH.md` or equivalent; `.env_example` alone lists keys, not meanings.
4. **LLM hallucination** is mitigated by schema validation, evidence refs, and quarantine rules — **not eliminated**.
5. **Hydrated markdown is ephemeral** and **must** carry snapshot id and generation timestamp when injected into Orion prompts; **do not** write it into **chat long-term memory** by default.

---

## 11. Security and operations

- Scope expansion must respect **path allowlists** (no arbitrary filesystem read outside repo root).
- Secrets must never enter prompts from `.env` — only **`.env_example` keys** and **non-secret** Declared facts.
- Rate limits and auth on `POST /v1/architecture/snapshots` are required for any network-exposed deployment (detail in implementation plan).

---

## 12. Open decisions (for implementation plan)

1. **New `node_kind` values vs reuse** of existing substrate kinds + `subject_ref`.
2. **Named graph URI** for architecture vs cognitive substrate default.
3. **Sync vs async** snapshot API for long Frontier runs.
4. **Quarantine mechanics** for under-evidenced inferred edges.
5. **Exact filename** and CI hook policy for **`ARCH.md`** (or chosen equivalent).

---

## 13. Related documents

- `docs/architecture/unified_cognitive_substrate_phase13_graphdb_persistence.md` — GraphDB ownership and substrate mapping.
- `docs/architecture/unified_cognitive_substrate_phase14_graphdb_reads.md` — bounded semantic reads.
- `docs/superpowers/specs/2026-05-03-hub-otel-traces-metrics-observability-design.md` — OTel / Hub direction (v2 feeder).
- `orion/spark/concept_induction/PHASE1_CONCEPT_PROFILE_GRAPH_MATERIALIZATION.md` — precedent for graph materialization patterns.
- `services/orion-bus-tap/README.md` — live bus observation (v2 feeder precursor).
