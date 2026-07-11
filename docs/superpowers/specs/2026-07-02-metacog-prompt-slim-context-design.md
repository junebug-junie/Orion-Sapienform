# Metacog prompt slim context — design

**Date:** 2026-07-02  
**Status:** Draft (awaiting review)  
**Scope:** `orion-cortex-exec` metacog publish lane (`log_orion_metacognition`), prompt templates, settings/env contract  
**Related:** PR #785 two-pass native logprob probe (merged); operator-owned metacog profile/model swap (out of scope for this change set)

---

## 1. Purpose

Metacog **draft** reliably fails JSON parse on live stack despite the two-pass routing fix. Root cause on trace `63f29e00`: **prompt bloat vs worker context window**, not native-completion detour.

Observed draft prompt: **11,420 chars** on **`llama3-8b-instruct-q4km-atlas-metacog`** with **`ctx_size: 4096`**. Gateway returned `finish_reason=length` at **104 completion tokens** — output budget exhausted mid-JSON.

The prompt ships the same substrate signals **multiple times** (prose + JSON), plus **~5.1k chars of template ritual** the model is told not to echo. Preflight passes because `CORTEX_METACOG_DRAFT_PROMPT_MAX_CHARS=16384` is unrelated to the worker's 4k token window.

**Goal:** Slim metacog context assembly so draft (and enrich) prompts fit the worker ctx budget while preserving mirror signal quality. Keep draft and enrich as **separate LLM calls** with distinct contracts.

**Success criteria (all required):**

| ID | Criterion |
|----|-----------|
| A | Draft prompt **≤ ~6,500 chars** on live baseline/manual trigger (down from ~11k) |
| B | `metacog_draft_mode=llm` with valid JSON on baseline or manual trigger (no `finish_reason=length` truncation at ~100 tokens) |
| C | Enrich still merges valid `MetacogEnrichScorePatchV1` (non-null `numeric_sisters` / `causal_density` when LLM succeeds) |
| D | Full biometrics remain in state/Postgres paths; prompts use **cluster-level cues only** (option C) |
| E | Draft ctx preflight tied to worker char budget (mirror enrich trim semantics) |

**Operator note:** Model/profile swap (e.g. 16k ctx Qwen3 or bumped Llama ctx) may proceed in parallel. This spec fixes structural bloat regardless of profile.

---

## 2. Problem anatomy (verified live)

Trace `63f29e00-0ed2-4a44-a703-f6161c2a7c30` — `metacog_publish_prompt_diagnostics`:

| Section | Chars | Notes |
|---------|------:|-------|
| Template shell (identity, ritual, example JSON) | ~5,150 | `log_orion_metacognition_draft.j2` |
| `biometrics_json` | 3,823 | Pretty-printed blob; not used by Python merge |
| `context_summary` | 1,183 | Prose digest; partially redundant |
| `spark_state_json` | 634 | Full snapshot JSON |
| `spark_phi_narrative` | ~550 | **Not logged** in `section_sizes` today |
| Turn-effect blobs | ~181 | Mostly empty |

**Critical finding:** `_apply_enrich_patch` and `MetacogEnrichScorePatchV1.model_validate` do **not** read `biometrics_json`. Enrich already sets `biometrics_json="{}"` on ctx overflow and continues — proving no Python dependency on the fat blob.

**Choke point:** `MetacogContextService` in `services/orion-cortex-exec/app/executor.py` builds `biometrics_json` via `json.dumps(biometrics_context, indent=2)` and both templates inject it verbatim.

---

## 3. Chosen approach

**Option C — phase-aware biometrics cues; shared slim context; template ritual preserved.**

| Approach | Verdict |
|----------|---------|
| A — cluster only everywhere | Rejected for enrich: loses per-node hot-spot nuance for risk/density scoring |
| B — cluster + one-liners on draft too | Rejected: draft needs tone cues; prose `context_summary` already covers this |
| **C — draft: cluster cue; enrich: cluster + compact node lines** | **Selected** |
| D — slash template ritual now | **Deferred** (operator values the ritual; revisit after injection fix) |

**Non-goals:**

- Merging draft and enrich into one LLM call
- Changing collapse-mirror Pydantic schemas or SQL writer contract
- Template ritual compression (this pass)
- Metacog profile/model selection (operator-owned)
- Qwen3 thinking-off verification (separate deploy gate; see §9)

---

## 4. Architecture

```text
MetacogContextService (executor.py)
  │
  ├─ state RPC → BiometricsContext (summary, induction, cluster)
  ├─ ctx["biometrics"]          ← internal dict (unchanged authority for Python)
  ├─ ctx["context_summary"]     ← prose digest (unchanged this pass)
  ├─ ctx["spark_phi_narrative"] ← keep for draft+enrich (phase 2 may dedupe vs spark JSON)
  │
  ├─ NEW: _metacog_biometrics_cue(ctx, phase="draft"|"enrich") → str
  │        draft:  cluster composites, constraint, freshness (~200–350 chars)
  │        enrich: draft cue + compact per-node lines (~400–600 chars max)
  │
  └─ ctx["metacog_biometrics_cue"]  ← template-facing (replaces fat biometrics_json in prompts)

MetacogDraftService / MetacogEnrichService
  ├─ render template with slim cue (not 3.8k biometrics_json)
  ├─ worker ctx preflight (draft: NEW; enrich: extend existing trim)
  └─ LLM → parse → Pydantic merge (unchanged)
```

**Data flow invariant:** `ctx["biometrics"]` retains full internal structure for `context_summary` biometrics line and telemetry. Only the **LLM-facing serialization** slimming changes.

---

## 5. Biometrics cue contract

### 5.1 Source of truth

Use existing aggregation paths — do **not** invent a new biometrics service:

| Signal | Source |
|--------|--------|
| Cluster composites (strain, homeostasis, stability) | `BiometricsContext.cluster` from state RPC (`biometrics.cluster.v1` ingest) |
| Cluster pressures / constraint | `BiometricsClusterV1` |
| Freshness | `BiometricsContext.status`, `age_ms` |
| Per-node one-liners (enrich only) | Derived from state store per-node summaries when available; cap at **4 nodes × ~80 chars** |

**Bugfix in scope:** `MetacogContextService` today does not copy `cluster` from state `BiometricsContext` into `biometrics_context` — it keeps `_default_biometrics_context()` zeros. Copy `cluster` from state reply so cues reflect live aggregation.

### 5.2 Draft cue shape (example)

Compact JSON, no `indent=2`, no nested induction trees:

```json
{"status":"fresh","constraint":"NONE","strain":0.42,"homeostasis":0.71,"stability":0.88,"freshness_s":12}
```

Target: **≤ 350 chars**.

### 5.3 Enrich cue shape (example)

Draft cue plus optional node lines when state has per-node summaries:

```json
{"cluster":{"strain":0.42,"constraint":"NONE"},"nodes":["atlas: gpu=0.82 strain=0.71","athena: ok"]}
```

Target: **≤ 600 chars**. If multi-node digest unavailable on global scope, enrich falls back to draft-level cluster cue only (still valid).

### 5.4 Template change

Replace in **both** `log_orion_metacognition_draft.j2` and `log_orion_metacognition_enrich.j2`:

```jinja
BIOMETRICS (JSON; may be missing with reason; do NOT paste into output):
{{ biometrics_json }}
```

with:

```jinja
BIOMETRICS CUE (compact; do NOT paste into output):
{{ metacog_biometrics_cue }}
```

Keep `context_summary` prose unchanged this pass (already includes biometrics one-liner).

---

## 6. Worker context preflight

### 6.1 Draft (new)

Mirror enrich semantics:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET` | `8000` | Trim/re-render before draft LLM when prompt exceeds budget |

Trim order when over budget:

1. Replace `metacog_biometrics_cue` with minimal `{"status":"trimmed"}`
2. Clear `spark_state_json` to `"{}"` (φ narrative + `context_summary` spark line remain)
3. If still over → `prompt_context_overflow` skip (same as enrich), telemetry records reason

Rationale: 8k chars ≈ ~2k tokens input on 4k ctx leaves ~2k tokens output; on 16k ctx profiles provides headroom without changing setting.

### 6.2 Enrich (extend)

Existing `CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET=12000` trim currently zeroes `biometrics_json`. Update to trim `metacog_biometrics_cue` first (same order as draft).

### 6.3 Section size diagnostics

Add to `_METACOG_DRAFT_CTX_LEN_KEYS` / enrich keys:

- `metacog_biometrics_cue`
- `spark_phi_narrative`

Deprecate logging `biometrics_json` size in diagnostics (keep internal `ctx["biometrics"]` out of prompt size accounting).

---

## 7. Draft / enrich separation (unchanged)

| Call | Contract | Extra inputs |
|------|----------|--------------|
| Draft | `MetacogDraftTextPatchV1` | Slim context bundle |
| Enrich | `MetacogEnrichScorePatchV1` | Slim context + `collapse_json` (draft output) |

Two-pass logprob probe (pass 1 chat JSON, pass 2 native probe) remains as merged in PR #785. This spec does not alter probe wiring.

---

## 8. Error handling & telemetry

| Condition | Behavior |
|-----------|----------|
| State RPC missing biometrics | Cue `{"status":"missing","reason":"..."}`; draft/enrich proceed |
| Prompt over worker ctx budget after trim | Skip LLM; fallback path; telemetry `metacog_*_fallback_reason=prompt_context_overflow` |
| Prompt over char limit (`CORTEX_METACOG_*_PROMPT_MAX_CHARS`) | Unchanged: skip before LLM |

New telemetry keys (optional, on existing publish diagnostics):

- `metacog_biometrics_cue_chars`
- `metacog_ctx_trim_applied` (list: `biometrics_cue`, `spark_state_json`)

---

## 9. Model / profile (operator-owned; acceptance notes)

This change set does **not** switch `atlas_metacog_profile_name`. Operator may swap models independently.

If metacog moves to **Qwen3** on llamacpp:

- Worker launch should use profile with `enable_thinking: false` + `reasoning_budget: 0` (see PR #528)
- **Recommended:** add per-request `chat_template_kwargs: {"enable_thinking": false}` on metacog draft/enrich options (pattern from `orion-memory-consolidation/app/classify.py`)
- Run `verify_atlas_quick_llamacpp_thinking_off.sh` against metacog worker URL before declaring Qwen3 metacog live

If staying on **Llama 8B**: bump profile `ctx_size` to 8192–16384 on V100 16GB — complementary to slim prompts, not a substitute.

---

## 10. Files likely touched

| File | Action |
|------|--------|
| `services/orion-cortex-exec/app/executor.py` | Cue builder, cluster copy fix, draft ctx preflight, trim order, diagnostics keys |
| `services/orion-cortex-exec/app/settings.py` | `cortex_metacog_draft_worker_ctx_char_budget` |
| `services/orion-cortex-exec/.env_example` | Document new key + cue semantics |
| `orion/cognition/prompts/log_orion_metacognition_draft.j2` | `metacog_biometrics_cue` slot |
| `orion/cognition/prompts/log_orion_metacognition_enrich.j2` | Same |
| `services/orion-cortex-exec/tests/test_metacog_publish_lane.py` | Cue size, draft trim, enrich trim update |
| `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py` | Fixture ctx uses cue key |

**Non-touch:** gateway, llm_profiles.yaml (unless operator changes separately), collapse mirror schemas.

---

## 11. Verification

### Unit (required before merge)

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py -q --tb=short
```

Assert:

- Draft render with fixture ctx: `metacog_biometrics_cue` ≤ 350 chars; total prompt ≤ 6500 chars
- Enrich cue ≤ 600 chars when node lines present
- Draft ctx trim fires before LLM when prompt exceeds draft worker budget
- Enrich trim targets cue (not legacy fat `biometrics_json`)

### Live (required for closure)

After deploy + operator model swap (if any):

1. Manual metacog trigger or wait for baseline
2. Cortex-exec log: `metacog_publish_prompt_diagnostics` shows `prompt_chars` ≤ 6500 for draft
3. Gateway: draft `finish_reason=stop` (not `length` at ~100 tokens)
4. Postgres `collapse_mirror`: `draft_mode=llm`, valid summary (not "Fallback mirror draft")
5. Enrich row has populated `numeric_sisters` / `causal_density` when enrich LLM succeeds

---

## 12. Phase 2 (explicitly deferred)

- Dedupe spark: drop `spark_state_json` from draft when `spark_phi_narrative` + `context_summary` suffice
- Template ritual compression (only if prompt still tight on 16k ctx)
- State service: expose compact multi-node digest on global scope for richer enrich cues without fat JSON

---

## 13. Spec self-review

| Check | Result |
|-------|--------|
| Placeholders / TBD | None |
| Internal consistency | Cue replaces prompt injection only; internal `ctx["biometrics"]` preserved |
| Scope | Single implementation plan target (cortex-exec + templates + tests) |
| Ambiguity | Enrich per-node lines best-effort from state; cluster-only fallback explicit |

---

**Next step after approval:** `writing-plans` → `docs/superpowers/plans/2026-07-02-metacog-prompt-slim-context.md`
