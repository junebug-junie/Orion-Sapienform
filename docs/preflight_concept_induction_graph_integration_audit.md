# Preflight Audit: Concept Induction Integration (Local Profile Path vs GraphDB/RDF)

Date: 2026-03-27 (UTC)
Scope: preflight-only architecture audit; no runtime behavior changes.

## 1) Executive summary

### Bottom line
- **`concept_induction_pass` currently reads only from `LocalProfileStore` JSON files via `CONCEPT_STORE_PATH`.** There is no GraphDB query in that workflow path. 
- **Spark Concept Induction currently writes profiles/deltas to local file + bus channels + vector write requests, but not to GraphDB through an implemented concept-profile RDF path.**
- **GraphDB concept read/write code does exist, but for a different concept domain (`self_study` induced/reflective concepts), not for `ConceptProfile` from Spark concept induction.**
- **There is no concept-profile graph read model and no RPC/service boundary for graph-backed retrieval of Spark concept profiles.**
- Therefore: this is **not** “80% wired with a missing final adapter”; it is a **representation and read-model gap** requiring a deliberate mapping layer and migration strategy.

### Highest-risk misconception corrected
“Graph schemas and GraphDB writes exist, so concept_induction_pass can be switched to graph reads” is false in current code. The workflow reads `LocalProfileStore` directly, and the RDF writer currently does not materialize `memory.concepts.profile.v1` / `ConceptProfile` into GraphDB.

---

## 2) What `concept_induction_pass` consumes today

### Alias resolution and workflow dispatch
- User prompt alias resolution is performed in Hub (`resolve_user_workflow_invocation`) and attached to `metadata.workflow_request`. 
- For matching prompts (including “Run through your concept induction graphs”), Hub sets `workflow_id=concept_induction_pass` and routes to workflow lane.
- Orch checks `has_explicit_workflow_request(req)` and routes to `execute_chat_workflow(...)`; workflow-specific branch dispatches to `_execute_concept_induction_pass(...)` when `workflow_id == "concept_induction_pass"`.

### Actual runtime data source
Inside `_execute_concept_induction_pass(...)`:
- Loads concept settings via `get_concept_settings()`.
- Calculates placeholder status with `settings.store_path == DEFAULT_CONCEPT_STORE_PATH`.
- Instantiates `LocalProfileStore(settings.store_path)`.
- Iterates configured `subjects` and calls `store.load(subject)`.
- Builds review rows from `ConceptProfile` fields (`profile_id`, `revision`, `concepts`, `clusters`, `state_estimate`, time window).
- Returns workflow metadata including:
  - `profile_store_path`
  - `profile_store_placeholder_path`
  - `profile_store_env_var = CONCEPT_STORE_PATH`
  - `profiles_reviewed`
- On empty results, returns explicit failure with `code=concept_profiles_unavailable` and an honesty message (including placeholder guidance when default `/tmp/...` path is in use).

### Result shape returned by workflow
- `CortexClientResult` with:
  - `ok`: true only if at least one profile reviewed
  - `status`: `success`/`fail`
  - `final_text`: workflow summary text
  - `error`: includes code `concept_profiles_unavailable` when no profiles
  - `metadata.workflow`: bounded structured workflow payload (no graph identifiers)

---

## 3) Local/light profile layer (LocalProfileStore)

### Data model + serialization
- `LocalProfileStore` is a JSON-file store with layout:
  - `profiles[subject] -> ConceptProfile JSON`
  - `_hashes[subject] -> profile hash`
  - additional local runtime state (`drive_states`, `goal_cooldowns`)
- `load(subject)` validates against canonical `ConceptProfile` schema.
- `save(subject, profile, profile_hash)` persists serialized `ConceptProfile` and hash.

### Writers of LocalProfileStore
- `ConceptInducer.run(...)` computes profile/delta and persists via injected `store_saver` (wired to `LocalProfileStore.save` in `ConceptWorker`).
- `ConceptWorker` also writes drive state and goal cooldown state into the same local JSON file via store helpers.

### Readers of LocalProfileStore
- `concept_induction_pass` workflow (`services/orion-cortex-orch`) reads profiles for summary review.
- `chat_stance` (`services/orion-cortex-exec/app/chat_stance.py`) reads concept profiles to construct stance buckets.
- `ConceptInducer` reads previous profile for revision increment + delta computation.
- Goal proposal / drive logic in concept worker reads/writes local cooldown/drive state.

### Cache vs canonical status (current reality)
- For Spark concept profiles in runtime workflow path, **LocalProfileStore is currently the only implemented read source**.
- It behaves as a **primary operational store** for this path, not a documented graph-derived cache.
- It also contains extra operational state (`drive_states`, `goal_cooldowns`) that is not represented as `ConceptProfile` graph entities.

---

## 4) GraphDB/RDF/schema layer relevant to concept induction

## 4.1 What exists

### RDF writer + GraphDB sink
- RDF writer consumes selected bus channels and writes N-Triples to GraphDB via `/repositories/{repo}/statements`.
- GraphDB write path is implemented and active for supported kinds.

### Graph-backed concept-like entities that exist (different domain)
- `self_study` in cortex-exec defines and persists:
  - authoritative self snapshot facts
  - induced self concepts (`orion:InducedSelfConcept`) in `orion:self:induced`
  - reflective findings in `orion:self:reflective`
- `self_study` also includes GraphDB SPARQL retrieval utilities (`_graphdb_*`) and typed retrieval models.

## 4.2 What does *not* exist for Spark `ConceptProfile`

### No RDF materialization for `memory.concepts.profile.v1` / `ConceptProfile`
- Spark concept worker publishes profile/delta events and vector writes.
- RDF writer `build_triples_from_envelope` has no branch for `memory.concepts.profile.v1` or `memory.concepts.delta.v1`.
- RDF writer subscribed channels do not include `orion:spark:concepts:profile` / `orion:spark:concepts:delta` by default.

### No graph-backed read model for Spark concept profiles
- No SPARQL/query utility found for `ConceptProfile` retrieval.
- No RPC/service contract found for “get concept profile(s) from graph”.
- No mapper from graph triples -> `ConceptProfile` used by workflow runtime.

### Config mismatch that can mislead
- Spark concept settings expose `BUS_RDF_OUT` (`forward_rdf_channel`) defaulting to `orion:rdf:write`, but concept worker does not publish RDF events on that channel in current implementation.
- RDF writer defaults to `CHANNEL_RDF_ENQUEUE=orion:rdf-collapse:enqueue` and known intake channels; not `orion:rdf:write`.

---

## 5) Overlap matrix

| Artifact | Local representation | Graph representation | Producer(s) | Consumer(s) | Current source of truth | Overlap type | Current read path | Current write path | Integration risk | Recommended future ownership |
|---|---|---|---|---|---|---|---|---|---|---|
| Spark concept profile | `ConceptProfile` in `LocalProfileStore` JSON (`profiles[subject]`) | **None implemented** for this artifact | `ConceptInducer` via `ConceptWorker` | `concept_induction_pass`, `chat_stance`, inducer delta logic | Local file | N/A (no graph twin) | Direct local file read | Local save + bus publish profile event | High if switching abruptly (no graph read model) | Graph-backed read model + compatibility local cache during migration |
| Spark concept profile delta | `ConceptProfileDelta` bus payload; optionally derivable from local prev profile | **None implemented** | `ConceptInducer` | downstream event consumers (if any) | Event stream (transient) / local derivation | N/A | No workflow read | Bus publish only | Medium (consumer ambiguity) | Keep event-based delta; derive from canonical profile read model |
| Spark drives/identity/goal artifacts | Local operational state (`drive_states`, `goal_cooldowns`) + typed events | RDF materialization exists for selected artifacts (`memory.identity.snapshot.v1`, `memory.drives.audit.v1`, `memory.goals.proposed.v1`) | `ConceptWorker` | RDF writer + SQL writer + others | Mixed (local runtime + event log) | Partial overlap | No workflow consumption in concept_induction_pass | Bus publish; RDF writer ingest supported kinds | Medium (state split) | Keep operational local state; publish canonical typed events |
| Self-study induced concepts | Not stored in LocalProfileStore | `orion:InducedSelfConcept` in `orion:self:induced` graph | `self_study` verbs | self-study retrieval path | Graph for persisted retrieval mode | Semantic-name overlap only (“concept”), different domain model | GraphDB SPARQL via self_study retrieval | RDF write requests via self_study | Medium misconception risk (false equivalence with Spark concept profiles) | Keep separate unless explicit unification project |
| Self-study reflective findings | Not stored in LocalProfileStore | reflective entities in `orion:self:reflective` graph | `self_study` verbs | self-study retrieval | Graph | Different artifact | GraphDB SPARQL via self_study retrieval | RDF write requests via self_study | Low for concept_pass, high for naming confusion | Keep separate from Spark concept profile lane |
| Workflow concept review output | `metadata.workflow.profiles_reviewed[]` summary synthesized from local profiles | None | Orch workflow runtime | Hub/UI/user | Ephemeral runtime result | Derived summary | In-memory runtime | N/A | Medium (can diverge from future graph source if mapping not explicit) | Keep as derived projection from canonical read model |

---

## 6) Read/write truth table (implemented reality)

| Path | Implemented today | Tested today | Used by workflow today | Used by another service today | Source-of-truth or derivative | Blocking gap for full integration |
|---|---|---|---|---|---|---|
| Workflow read from `LocalProfileStore` | Yes | Yes | Yes (`concept_induction_pass`) | Yes (`chat_stance`) | Primary operational source (today) | None for current local mode |
| Workflow read from GraphDB for Spark concept profiles | No | No | No | No | N/A | Missing read model + mapper + query + integration seam |
| Concept induction write to `LocalProfileStore` | Yes | Yes | Indirectly (workflow consumes output) | Yes (concept worker runtime) | Primary operational persistence for profiles | N/A |
| Concept induction write to GraphDB (Spark `ConceptProfile`) | No direct implemented materialization | No | No | No | N/A | Missing RDF materialization contract and writer support |
| Graph read-model/query surface for Spark concept profiles | No | No | No | No | N/A | Missing query model + typed adapters |
| RPC/service boundary for graph-backed concept profile retrieval | No | No | No | No | N/A | Missing service contract + ownership |
| Cache/materialization GraphDB -> local Spark profile | No | No | No | No | N/A | Missing backfill/sync/materializer |
| Self-study graph read/write for self concepts | Yes (separate domain) | Yes | Not by concept_induction_pass | Yes (`self_retrieve` etc.) | Graph-backed for self-study lane | Not directly reusable without domain mapping |

---

## 7) Config and environment audit

## 7.1 Local-only behavior requirements (current concept workflow)
- Required in practice for meaningful results:
  - `CONCEPT_STORE_PATH` must point to a real populated file.
- If unset, default `DEFAULT_CONCEPT_STORE_PATH=/tmp/concept-induction-state.json` is used and workflow explicitly reports unconfigured/missing profiles.

## 7.2 Graph-backed behavior requirements (if/where implemented today)
- GraphDB env variables exist and are used by RDF writer / self-study retrieval paths (e.g., `GRAPHDB_URL`, `GRAPHDB_REPO`, auth vars), but **not by `concept_induction_pass`**.

## 7.3 Hidden defaults and misleading placeholders
- `DEFAULT_CONCEPT_STORE_PATH` can make a deployment appear configured while actually empty/unseeded; workflow now surfaces this explicitly.
- Spark concept settings contain `BUS_RDF_OUT`, but current concept worker does not use it; this can imply a non-existent concept-profile RDF write path.
- Channel naming divergence exists (`orion:rdf:enqueue` in bus catalog vs RDF writer default `orion:rdf-collapse:enqueue`) and can obscure integration assumptions.

---

## 8) Tests and observability audit

### Covered today
- Workflow routing + `concept_induction_pass` behavior (success, missing profiles, placeholder honesty).
- Local profile storage and concept induction generation paths (including store round-trip, revisions/deltas, drive state persistence).
- Self-study GraphDB retrieval and persistence boundaries (separate concept domain).
- Workflow invocation resolution from Hub for concept_induction aliases.

### Missing for full Spark-profile graph integration
- No tests proving Spark `ConceptProfile` RDF materialization.
- No tests proving Spark concept profile SPARQL retrieval.
- No tests proving workflow parity between local and graph-backed profile reads.
- No contract/integration tests for RPC concept-profile retrieval service.

### Observability currently available
- Workflow logs include explicit source status (`configured_source_kind=local_profile_store`, placeholder flag, source path, availability).
- No analogous graph-source status for concept_induction_pass (because no graph read path exists).

---

## 9) Canonical-source recommendation

### Recommended long-term source of truth
**Graph-backed read model with an explicit materialized compatibility cache (temporary).**

Why:
1. Graph gives cross-service inspectability and consistent provenance surfaces.
2. Workflow/runtime should consume a stable typed read model, not raw triples.
3. Existing local consumers (`concept_induction_pass`, `chat_stance`) need low-risk continuity during migration.
4. Current local store also carries operational state; cache/materializer avoids forcing all operational state into graph prematurely.

### Not recommended
- Making LocalProfileStore permanent canonical source if cross-service graph introspection is a product requirement.
- Directly querying raw GraphDB triples from workflow runtime without typed mapping and compatibility contracts.

---

## 10) Narrow implementation-ready integration plan (no code in this pass)

## Phase 0 — contract + seam definition
1. Define a typed **ConceptProfileReadModelV1** (or reuse `ConceptProfile` if sufficient) as workflow-consumable abstraction.
2. Define `ConceptProfileRepository` interface used by workflow/chat stance:
   - `get_latest(subject)`
   - `list_latest(subjects)`
   - explicit freshness/provenance metadata.
3. Keep `concept_induction_pass` logic intact; swap only dependency seam (repository implementation).

## Phase 1 — graph materialization for Spark profiles
1. Add RDF writer support for `memory.concepts.profile.v1` (+ delta if needed).
2. Ensure RDF writer subscribes to concept profile channels (or route via explicit enqueue with stable kind contract).
3. Map `ConceptProfile` fields to stable graph predicates (including profile_id, subject, revision, concept items, clusters, window bounds, confidence/salience, evidence refs where available).

## Phase 2 — graph read model
1. Build query adapter (SPARQL + parser) that reconstructs typed profile read model.
2. Add deterministic mapping tests from sample triples -> model.
3. Add behavior parity tests comparing local JSON and graph read outputs for representative fixtures.

## Phase 3 — RPC/service boundary
1. Provide a bounded RPC surface (`concept.profile.get.v1` / `concept.profile.list.v1`) from a single owning service.
2. Orch workflow and chat stance use repository implementation backed by this RPC client.
3. Include timeout/error semantics with explicit “unavailable” vs “empty” distinction.

## Phase 4 — compatibility and rollout
1. Dual-read shadow mode (graph + local) with diff telemetry, no user-visible behavior change.
2. Alert on structural diffs (subject missing, revision mismatch, concept count drift).
3. Progressive rollout toggle (`CONCEPT_PROFILE_READ_BACKEND=local|graph|shadow`).
4. After parity confidence window, switch default to graph; keep local fallback behind explicit emergency flag.

## Phase 5 — deprecation
1. Deprecate direct workflow/local reads only after sustained parity and recovery playbook.
2. Retain local operational state for non-profile concerns (drive cooldowns) unless/until separately redesigned.

---

## 11) “Don’t do this” list

1. **Do not** read raw GraphDB entities directly in workflow runtime without a mapping layer.
2. **Do not** delete `LocalProfileStore` before proving graph/local equivalence in shadow mode.
3. **Do not** add fake/default profile fallback data to hide missing graph reads.
4. **Do not** create a second partially overlapping Spark concept profile format.
5. **Do not** conflate self-study induced concepts with Spark concept profiles—they are different domains and trust semantics.
6. **Do not** rely on schema/ontology presence as proof that runtime retrieval exists.

---

## 12) Full integration readiness assessment

Assessment: **larger integration, not a final adapter**.

Reasoning:
- Current workflow runtime is hard-wired to local JSON profile loading.
- Graph materialization for Spark `ConceptProfile` is absent in RDF writer path.
- Graph query/read-model/RPC surface for Spark concept profiles is absent.
- Existing graph concept retrieval code targets self-study concepts, not Spark concept profiles.

Therefore, integration requires: materialization + read-model + service boundary + migration/compatibility harness.

---

## 13) Exact files inspected

### Workflow/runtime
- `orion/cognition/workflows/registry.py`
- `services/orion-hub/scripts/cortex_request_builder.py`
- `services/orion-hub/tests/test_workflow_request_builder.py`
- `services/orion-cortex-orch/app/main.py`
- `services/orion-cortex-orch/app/workflow_runtime.py`
- `services/orion-cortex-orch/tests/test_workflow_lane.py`

### Local concept profile layer
- `orion/core/schemas/concept_induction.py`
- `orion/spark/concept_induction/settings.py`
- `orion/spark/concept_induction/store.py`
- `orion/spark/concept_induction/inducer.py`
- `orion/spark/concept_induction/bus_worker.py`
- `orion/spark/concept_induction/tests/test_concept_induction.py`
- `services/orion-cortex-exec/app/chat_stance.py`
- `services/orion-spark-concept-induction/README.md`
- `services/orion-spark-concept-induction/.env_example`
- `services/orion-spark-concept-induction/docker-compose.yml`

### GraphDB/RDF/schema + related concept domain
- `services/orion-rdf-writer/app/settings.py`
- `services/orion-rdf-writer/app/rdf_builder.py`
- `services/orion-rdf-writer/app/service.py`
- `services/orion-rdf-writer/README.md`
- `services/orion-rdf-writer/docker-compose.yml`
- `orion/bus/channels.yaml`
- `orion/schemas/registry.py`
- `orion/schemas/self_study.py`
- `services/orion-cortex-exec/app/self_study.py`
- `services/orion-cortex-exec/tests/test_self_study_graphdb.py`
- `services/orion-cortex-exec/tests/test_self_study_pass1.py`

### Supporting architecture docs
- `docs/architecture/chat_invoked_cognitive_workflows.md`
- `orion/spark/README.md`

---

## 14) Commands executed for this preflight

- `rg --files -g 'AGENTS.md'`
- `rg -n "concept_induction_pass|LocalProfileStore|CONCEPT_STORE_PATH|GraphDB|RDF|sparql|concept induction|concept_profile|profile store|profile_store|graph" /workspace/Orion-Sapienform`
- `rg -n "resolve_user_workflow_invocation|workflow_request|matched_alias|concept_induction_pass" ...`
- `rg -n "memory\.concepts\.profile\.v1|memory\.concepts\.delta\.v1|orion:spark:concepts:profile|ConceptProfile|ConceptProfileDelta" ...`
- `rg -n "forward_rdf_channel|BUS_RDF_OUT|_forward_rdf|rdf" orion/spark/concept_induction`
- and targeted `sed -n` reads across files listed above.

