# Codex Prompt — Architecture Review + Fixes + Validation (Anti-Drift)

You are working in the Orion monorepo.

## REQUIRED INPUTS (read first)

1. docs/platform_contract.md
2. docs/platform_codex_testing.md
3. orion/bus/channels.yaml

Treat these as binding requirements.

## HARD RULES

* Do NOT invent class/module names. Use only primitives that exist in this repo. If unsure, search the repo and use the actual names.
* Do NOT claim “no findings” without producing the required report artifacts proving the scan happened.
* Do NOT introduce new bus channels without adding them to orion/bus/channels.yaml.
* Do NOT publish/consume raw JSON over the bus. Use Titanium envelopes with schema_id and Pydantic-validated payloads.
* Do NOT bypass the execution spine:
  cortex-orch → orion:verb:request → cortex-exec(VerbRuntime) → orion:verb:result

---

## GOAL

Perform a full architecture review per the Platform Contract and Codex Testing Contract, implement fixes for all fatal/major drift, and produce runnable validation artifacts.

---

## A) ARCHITECTURE REVIEW (produce evidence + artifacts)

Create a new folder `reports/` (if missing) and generate ALL artifacts below.

### A1) Channel inventory + drift

1. Scan the repo for bus channel usage:

   * string literals containing `orion:` (and other known namespaces)
   * constants/config values that hold channel names
   * publish/subscribe calls (including pattern subscriptions)

2. Produce `reports/channel_inventory.json` with entries:

   * channel (string)
   * usage (publish | subscribe | pattern | unknown)
   * files (list of `path:line`)
   * service_guess (derived from file path)

3. Compare inventory against `orion/bus/channels.yaml` and produce `reports/channel_drift.json`:

   * unknown_channels (in code, not in catalog)
   * catalog_unused_channels (in catalog, not found in code)
   * pattern_subscriptions (PSUBSCRIBE / wildcards)
   * channels_missing_roles (catalog entry exists but missing producers/consumers)

4. Reconcile drift:

   * If a channel is legitimate: add/update its entry in channels.yaml.
   * If a channel is accidental drift: change code to use the canonical channel.

Success condition: unknown_channels is empty OR any remaining entries are explicitly marked legacy/test-only with a documented plan.

### A2) Schema inventory + drift

1. Scan all bus publish sites and identify:

   * how Titanium envelopes are built
   * schema_id values
   * payload models (if inferable)

2. Produce `reports/schema_inventory.json` with entries:

   * schema_id
   * files (list of `path:line`)
   * envelope_builder (name/path)
   * pydantic_model_resolves (true/false; verify the symbol exists)

3. Produce `reports/schema_drift.json` with:

   * missing_schema_id_sites
   * untyped_payload_sites
   * unresolved_schema_ids

Success condition: no bus publish site emits a message without Titanium + schema_id.

### A3) Spine audit (orch/exec/verbs)

Produce `reports/spine_audit.json` containing:

* all imports/usages of VerbRuntime (file:line)
* all publish sites to `orion:verb:request` (file:line)
* all subscribers/consumers of `orion:verb:request` (file:line)
* all publish sites to `orion:verb:result` (file:line)
* any bypass paths (verbs executed outside exec; direct exec calls; ad-hoc RPCs that violate contract)

Success conditions:

* VerbRuntime usage exists ONLY inside cortex-exec (test-only stubs allowed if isolated + documented).
* Planned verb invocation originates from cortex-orch (or clearly documented test harness only).

### A4) Config lineage audit

For each changed/new service, verify `.env_example → docker-compose.yml → app/settings.py`.

Produce `reports/config_lineage.json` containing per-service:

* env_example_exists (true/false)
* compose_wires_vars (true/false)
* settings_reads_vars (true/false)
* env_vars_added_or_changed (list)

Success condition: all services pass lineage.

### A5) Anti-pattern scan

Scan for platform drift anti-patterns and produce `reports/antipatterns.json`:

* raw Redis pub/sub usage outside OrionBus
* hardcoded channel strings not present in channels.yaml
* publishing/consuming non-Titanium messages
* schema_id missing
* VerbRuntime used outside exec

Success condition: no fatal/major anti-patterns remain in production code paths.

---

## B) IMPLEMENT FIXES (patch all fatal/major)

For every finding with severity fatal/major:

* Refactor to use OrionBus for bus I/O (no raw redis pubsub in production code paths).
* Normalize all bus messages to Titanium envelopes.
* Ensure schema_id resolves to an existing Pydantic model.
* Update channels.yaml as needed (and only as needed).
* Preserve the execution spine (orch orchestrates; exec executes; writers persist).

If a wildcard/pattern subscription is required (e.g., Bus Tap), implement one of:

* an allowlist filter derived from channels.yaml
* a restricted prefix pattern (domain-limited)
* explicit documentation + safe filtering rules

---

## C) VALIDATION HARNESS ARTIFACTS (Codex cannot run, but must create runnable scripts)

Create a minimal test harness consistent with docs/platform_codex_testing.md.

### C1) Compose

Create/update: `scripts/mth/docker-compose.mth.yml` containing at least:

* bus-core (Redis)
* cortex-orch (or minimal stub that can emit VerbRequestV1)
* cortex-exec (VerbRuntime)
* bus-tap (passive observer)

Optional: writers (sql/rdf/vector) if required by effects scenario.

### C2) Test verbs

Ensure test verbs exist and are registered:

* `orion.test.ping`
* `orion.test.effects`

### C3) Scripts

Create:

* `scripts/mth/run_mth.sh`
* `scripts/mth/run_scenarios.sh`

Scenarios:

* Ping end-to-end (orch → request → exec → result)
* Effects fanout (exec emits orion:effect:sql/rdf/vector)
* Catalog enforcement behavior (warn vs error mode)

Include expected observable outcomes in bus-tap output:

* correct channel names
* Titanium envelope shape
* schema_id present
* correlation_id propagated

---

## D) REVIEW REPORT (human-readable)

Create: `docs/architecture_reviews/<DATE>_review.md` (use <DATE> placeholder; do not hardcode timestamps).

Include:

1. Scope summary
2. What was scanned (patterns/tools)
3. Key findings (fatal/major/minor)
4. Fix summary (file list)
5. Remaining debt / follow-ups
6. Success criteria checklist + final verdict

### Required success criteria checklist

* ✅ Channels used are cataloged (channel_drift.json)
* ✅ Bus publishes are Titanium + schema_id (schema_drift.json)
* ✅ Spine enforced (spine_audit.json)
* ✅ No raw redis pub/sub outside OrionBus (antipatterns.json)
* ✅ Env lineage intact (config_lineage.json)
* ✅ MTH artifacts + scripts exist (scripts/mth/*)

Final verdict: PASS / PASS-WITH-DEBT / FAIL

---

## STOP CONDITIONS

* If any required artifact cannot be produced, explain exactly why and mark the review FAIL.
* Do not claim “review complete” without generating all artifacts listed
