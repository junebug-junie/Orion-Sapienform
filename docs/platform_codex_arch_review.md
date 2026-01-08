# **🧩 Comprehensive Architectural Review (Codex Section)**

During the Architectural Review phase, perform a **systematic, multi-dimensional analysis** of the repository aimed at detecting, explaining, and remediating deviations from the Orion Platform Contract.

This phase MUST be executed across the following dimensions:

---

## **1. Architecture → Bus Dimension**

**Verify that:**

* All inter-service communication routes through the `OrionBus` abstraction.
* No direct Redis clients, pub/sub from SDKs, or bespoke wrappers are used.
* All publish/subscribe operations use Titanium envelopes with declared `schema_id`.
* Request/Response verbs only travel via:
  `orion:verb:request` → cortex-exec → `orion:verb:result`

**Detect anti-patterns:**

* `redis = Redis(...)`
* `redis.publish(...)`
* `redis.subscribe(...)`
* ad-hoc JSON blobs without schema
* services coordinating via HTTP when the platform contract mandates bus

**Required remediation types:**

* Wrap raw bus calls in OrionBus
* Convert raw messages → Titanium envelopes
* Add missing schemas

**Artifacts:**

* Report list of all bus ops per service + compliance status

---

## **2. Architecture → Channel Dimension**

**Verify that:**

* Every channel used is declared in `orion/bus/channels.yaml`
* Producer + consumer roles match contract
* Stability metadata exists (e.g., `internal`, `stable`, `experimental`)
* No new channels were silently created

**Detect anti-patterns:**

* Hardcoded channel name strings
* Pattern/prefix subscriptions that bypass validation
* Channels that logically conflict with existing ones

**Remediation actions:**

* Add to catalog w/ schema + stability
* Update service to reference catalog enum or constant

---

## **3. Architecture → Schema Dimension**

**Verify that:**

* Every published message has a matching schema
* All Titanium envelopes include:

  * `schema_id`
  * `payload`
  * `timestamp`
  * `request_id` OR `correlation_id`
  * `producer`
  * optional `reply_to`

**Detect anti-patterns:**

* payload-only messages
* mismatched envelope casing
* schemas in code not matching schemas in catalog
* untyped effects

**Remediation:**

* Generate Pydantic schemas
* Normalize envelope generation
* Create schema mappings

---

## **4. Architecture → Verb/Orch Dimension**

**Verify that:**

* All verbs are defined via Verb SDK
* All verbs registered in VerbRegistry
* All invocations are orchestrated by cortex-orch
* No direct instantiation of verbs at runtime

**Detect anti-patterns:**

* calling a verb class directly
* embedding verb logic inside other services
* bypassing VerbRuntime

**Remediation:**

* Move logic into verbs
* Route through orch
* Patch planners

---

## **5. Architecture → Config Lineage Dimension**

**Verify `.env → docker-compose → settings.py` lineage**

* No config is hard-coded in code
* No config is only in python and missing compose
* No config is only in compose and missing env example
* No env var leaks credentials or secrets

**Remediation:**

* Add missing `.env_example`
* Correct compose → settings bindings
* Normalize naming

---

## **6. Architecture → Service Boundary Dimension**

**Check service responsibilities:**

* No leakage of cognition into writer services
* No business logic in bus services
* No introspection in storage
* No stateful reasoning in orch

We enforce “narrow brain / wide bus / explicit verbs”.

---

## **7. Architecture → Observability & Tap Dimension**

**Verify:**

* Bus tap conforms to platform contract
* Bus tap honors channel catalog
* Bus tap uses filtering/whitelisting switches
* Bus tap can be replayed for audits

Detect:

* taps with wildcard subscription (anti-pattern)
* taps with write-back side effects (major anti-pattern)

---

## **8. Cross-Cutting Invariants**

Check invariants across the whole mesh:

| Invariant                   | Must Hold |
| --------------------------- | --------- |
| Bus is source of truth      | yes       |
| Orch is only verb gateway   | yes       |
| Writers have no cognition   | yes       |
| Storage is non-interrupting | yes       |
| Taps are passive            | yes       |
| Effects are typed           | yes       |
| Schemas are canonical       | yes       |
| No global singletons        | yes       |
| No direct cycles            | yes       |

---

# **🧾 Review Outputs**

After running the review, Codex MUST produce:

### **A. Findings Report**

Structured as:

```
[Dimension]
  findings:
    - file:LINE
      issue:
      severity:
      suggested remediation:
```

Severity levels:

* `fatal` (violates platform contract)
* `major` (breaks invariants or cross boundary)
* `minor` (cosmetic / organizational)

---

### **B. Concrete Patches**

For all fatal & major findings

* Code modifications
* Schema additions
* Verb boundary fixes
* Channel catalog diffs

---

### **C. Updated Catalogs & Docs**

If needed:

* `channels.yaml` diff
* new/updated schema models
* updated `docs/platform_contract.md`
* updated `docs/services/*`

---

# **🎯 Success Criteria (Codex MUST validate against)**

The review is successful when ALL are true:

✔ All bus usages go through `OrionBus`
✔ All channels resolve in catalog & are typed
✔ All verbs invoked via orch pipeline
✔ No cross-service cognition leakage
✔ Configuration lineage is intact
✔ No raw Redis usage exists
✔ No untyped envelopes exist
✔ No undocumented channels remain
✔ Bus tap operates under contract
✔ Schemas + envelopes match reality
✔ Patches compile & test in minimal environment

---

# **🍒 Optional (but recommended)**

If gaps persist, Codex may propose:

* new verbs
* new pipeline packs
* new schemas
* new metrics
* new tap filters
* new enrichment or summarization primitives

