# Topic Foundry ↔ Orion Hub Topic Studio Contract Deep-Dive

## Scope and method
This report documents the **actual runtime contract** between:
- `orion-topic-foundry` (`/ready`, `/capabilities`), and
- Orion Hub Topic Studio (`/api/topic-foundry/*` proxy + Topic Studio JS parsing paths).

It is based on static code inspection of the current repository state.

---

## A) Foundry capability truth source

## A.1 Where `/capabilities` is built
`/capabilities` is constructed in `services/orion-topic-foundry/app/routers/capabilities.py` and returned as `CapabilitiesResponse`. The endpoint computes segmentation modes from `WindowingSpec.segmentation_mode`, hardcodes enricher modes, computes transport from bus settings, and includes defaults/introspection payloads.【F:services/orion-topic-foundry/app/routers/capabilities.py†L13-L42】

The response schema/type contract is defined in `CapabilitiesResponse`.【F:services/orion-topic-foundry/app/models.py†L364-L380】

## A.2 Field-by-field source, defaults, and gating

### Foundry Capabilities Contract

| JSON key | Type | Meaning | Missing/empty behavior | Env knobs affecting it |
|---|---|---|---|---|
| `service` | string | Service identity name. | Always present (required schema). | `SERVICE_NAME` via `settings.service_name` (default `orion-topic-foundry`).【F:services/orion-topic-foundry/app/settings.py†L12-L13】 |
| `version` | string | Service semantic/version string. | Always present. | `SERVICE_VERSION` (default `0.1.0`).【F:services/orion-topic-foundry/app/settings.py†L13-L13】 |
| `node` | string | Node hostname/name used by service. | Always present. | `NODE_NAME` or `HOSTNAME` (default `unknown`).【F:services/orion-topic-foundry/app/settings.py†L14-L14】 |
| `llm_enabled` | boolean | Whether Foundry allows LLM enrichment/LLM segmentation features. | Always present boolean; default false. | `TOPIC_FOUNDRY_LLM_ENABLE` (default `False`).【F:services/orion-topic-foundry/app/settings.py†L58-L61】【F:services/orion-topic-foundry/app/routers/capabilities.py†L29-L29】 |
| `llm_transport` | string (`bus`/`http`) | Declared transport mode for LLM client path. | Always present string. | Computed as `bus` only when `TOPIC_FOUNDRY_LLM_USE_BUS=true` **and** `ORION_BUS_ENABLED=true`, else `http`.【F:services/orion-topic-foundry/app/routers/capabilities.py†L17-L17】 |
| `llm_bus_route` | string/null | Optional LLM route passed through bus payload. | Can be `null`/empty; still present by schema. | `TOPIC_FOUNDRY_LLM_BUS_ROUTE` (default `None`).【F:services/orion-topic-foundry/app/settings.py†L54-L57】【F:services/orion-topic-foundry/app/routers/capabilities.py†L31-L31】 |
| `llm_intake_channel` | string/null | Bus intake channel for RPC request. | Explicitly `null` unless transport is bus. | `TOPIC_FOUNDRY_LLM_INTAKE_CHANNEL` / `CHANNEL_LLM_INTAKE` (default `orion:exec:request:LLMGatewayService`), but emitted only when transport=`bus`.【F:services/orion-topic-foundry/app/settings.py†L46-L49】【F:services/orion-topic-foundry/app/routers/capabilities.py†L32-L33】 |
| `llm_reply_prefix` | string/null | Prefix used to construct per-request reply channel. | Explicitly `null` unless transport is bus. | `TOPIC_FOUNDRY_LLM_REPLY_PREFIX` (default `orion:llm:reply`), emitted only when transport=`bus`.【F:services/orion-topic-foundry/app/settings.py†L50-L53】【F:services/orion-topic-foundry/app/routers/capabilities.py†L33-L33】 |
| `segmentation_modes_supported` | array<string> | UI-facing supported segmentation modes. | Always present; sourced from literal enum. | No env; sourced from `WindowingSpec.segmentation_mode` literals: `time_gap`, `semantic`, `hybrid`, `llm_judge`, `hybrid_llm`.【F:services/orion-topic-foundry/app/models.py†L39-L39】【F:services/orion-topic-foundry/app/routers/capabilities.py†L15-L15】 |
| `enricher_modes_supported` | array<string> | Enrichment modes advertised to clients. | Always present; currently hardcoded. | No env; hardcoded `['heuristic','llm']`.【F:services/orion-topic-foundry/app/routers/capabilities.py†L16-L16】 |
| `supported_metrics` | array<string> (optional in schema, always set in route) | Permitted clustering metrics. | Route always sets sorted list. | No env; from `SUPPORTED_METRICS = {euclidean, cosine, manhattan, l1, l2}`.【F:services/orion-topic-foundry/app/services/metrics.py†L6-L6】【F:services/orion-topic-foundry/app/routers/capabilities.py†L36-L36】 |
| `default_metric` | string | Default clustering metric. | Always set by route. | No env; from `ModelSpec.metric` default (`cosine`).【F:services/orion-topic-foundry/app/models.py†L67-L67】【F:services/orion-topic-foundry/app/routers/capabilities.py†L37-L37】 |
| `cosine_impl_default` | string | Cosine implementation mode used in training. | Always set by route. | `TOPIC_FOUNDRY_COSINE_IMPL` (default `normalize_euclidean`).【F:services/orion-topic-foundry/app/settings.py†L26-L29】【F:services/orion-topic-foundry/app/routers/capabilities.py†L38-L38】 |
| `defaults.embedding_source_url` | string | Suggested embedding URL for model spec. | Present inside `defaults`. | `TOPIC_FOUNDRY_EMBEDDING_URL` (default `http://orion-vector-host:8320/embedding`).【F:services/orion-topic-foundry/app/settings.py†L22-L25】【F:services/orion-topic-foundry/app/routers/capabilities.py†L20-L20】 |
| `defaults.metric` | string | Suggested metric default for model creation. | Present inside `defaults`. | No env; `ModelSpec.metric` default (`cosine`).【F:services/orion-topic-foundry/app/models.py†L67-L67】【F:services/orion-topic-foundry/app/routers/capabilities.py†L21-L21】 |
| `defaults.min_cluster_size` | integer | Suggested min cluster size for model creation. | Present inside `defaults`. | No env; `ModelSpec.min_cluster_size` default (`15`).【F:services/orion-topic-foundry/app/models.py†L66-L66】【F:services/orion-topic-foundry/app/routers/capabilities.py†L22-L22】 |
| `defaults.llm_bus_route` | string/null | Echoed route default to show clients current route value. | May be null. | `TOPIC_FOUNDRY_LLM_BUS_ROUTE`.【F:services/orion-topic-foundry/app/settings.py†L54-L57】【F:services/orion-topic-foundry/app/routers/capabilities.py†L23-L23】 |
| `introspection.ok` | boolean | Whether configured schemas list is non-empty. | Present in route payload object. | Derived from non-empty `TOPIC_FOUNDRY_INTROSPECT_SCHEMAS` list after split/trim. Defaults to `public` ⇒ usually true.【F:services/orion-topic-foundry/app/settings.py†L84-L87】【F:services/orion-topic-foundry/app/routers/capabilities.py†L18-L18】【F:services/orion-topic-foundry/app/routers/capabilities.py†L40-L40】 |
| `introspection.schemas` | array<string> | Schemas configured for introspection. | Empty array if env blank/whitespace. | `TOPIC_FOUNDRY_INTROSPECT_SCHEMAS`.【F:services/orion-topic-foundry/app/settings.py†L84-L87】【F:services/orion-topic-foundry/app/routers/capabilities.py†L18-L18】【F:services/orion-topic-foundry/app/routers/capabilities.py†L40-L40】 |
| `default_embedding_url` | string | Back-compat direct default embedding URL key. | Always set by route. | `TOPIC_FOUNDRY_EMBEDDING_URL`.【F:services/orion-topic-foundry/app/settings.py†L22-L25】【F:services/orion-topic-foundry/app/routers/capabilities.py†L41-L41】 |

### Runtime gating notes (LLM behavior)
- `llm_enabled` in capabilities is a **direct echo** of `TOPIC_FOUNDRY_LLM_ENABLE`; it does **not** require non-empty bus route/channel in this endpoint logic.【F:services/orion-topic-foundry/app/routers/capabilities.py†L29-L33】
- Runtime enrichment API hard-blocks explicit `enricher=llm` only when `TOPIC_FOUNDRY_LLM_ENABLE=false` (409).【F:services/orion-topic-foundry/app/routers/runs.py†L222-L226】
- LLM transport actual path in client uses bus only when `TOPIC_FOUNDRY_LLM_USE_BUS && ORION_BUS_ENABLED`; if false, requests are skipped and return `None`.【F:services/orion-topic-foundry/app/services/llm_client.py†L72-L73】【F:services/orion-topic-foundry/app/services/llm_client.py†L85-L88】
- Reply correlation is `reply_prefix + ':' + correlation_id`, with envelope `correlation_id` and `reply_to` set for RPC request/response binding.【F:services/orion-topic-foundry/app/services/llm_client.py†L134-L152】【F:services/orion-topic-foundry/app/services/llm_client.py†L154-L159】

---

## B) Hub: what Topic Studio expects

## B.1 Where Hub fetches and parses `/ready` + `/capabilities`

### Active split-pane Topic Studio path (current template)
- `initTopicStudioUI()` always calls `refreshTopicStudio()` for the Topic Studio tab; this is the active path for current UI mount flow.【F:services/orion-hub/static/js/app.js†L682-L696】
- `refreshTopicStudioCapabilities()` fetches `/capabilities`, parses capability keys, and updates segmentation/metric defaults + LLM UI state.【F:services/orion-hub/static/js/app.js†L5572-L5611】
- `refreshTopicStudioStatus()` fetches `/ready`, shows **Reachable/Unreachable** badge and check-level badges using `.checks.pg`, `.checks.embedding`, `.checks.model_dir`.【F:services/orion-hub/static/js/app.js†L5480-L5505】

### Proxy endpoint path
- All Topic Foundry requests are sent to `TOPIC_FOUNDRY_PROXY_BASE = apiUrl('/api/topic-foundry')` and fetched via `tfFetchJson(...)`.【F:services/orion-hub/static/js/app.js†L559-L559】【F:services/orion-hub/static/js/app.js†L1748-L1777】
- Hub backend forwards `/api/topic-foundry/{path}` to `TOPIC_FOUNDRY_BASE_URL` via generic proxy route.【F:services/orion-hub/scripts/api_routes.py†L199-L207】【F:services/orion-hub/app/settings.py†L50-L53】

## B.2 Parsed fields and fallback behavior

| Hub read location | Expected JSON path | Fallback if missing | Notes |
|---|---|---|---|
| `refreshTopicStudioCapabilities` | `segmentation_modes_supported` | `[]` via `?? []`; then fallback hardcoded set only on fetch failure. | Used to populate `#tsSegmentationMode`; LLM modes disabled when `llm_enabled=false`.【F:services/orion-hub/static/js/app.js†L5582-L5585】【F:services/orion-hub/static/js/app.js†L5508-L5533】 |
| `refreshTopicStudioCapabilities` | `supported_metrics` | `[]` when key missing; fetch failure fallback `['euclidean','cosine']`. | Used to populate metric select and default metric selection. 【F:services/orion-hub/static/js/app.js†L5583-L5586】【F:services/orion-hub/static/js/app.js†L5535-L5554】【F:services/orion-hub/static/js/app.js†L5598-L5601】 |
| `refreshTopicStudioCapabilities` | `default_metric` | none (empty string behavior in selector). | Only selected if present in metrics. 【F:services/orion-hub/static/js/app.js†L5535-L5553】【F:services/orion-hub/static/js/app.js†L5585-L5585】 |
| `refreshTopicStudioCapabilities` | `defaults.embedding_source_url` then `default_embedding_url` | empty string. | Used to prefill embedding URL. 【F:services/orion-hub/static/js/app.js†L5586-L5587】【F:services/orion-hub/static/js/app.js†L5556-L5563】 |
| `refreshTopicStudioCapabilities` | `defaults.metric`, `defaults.min_cluster_size` | no-op when absent. | Applies form defaults. 【F:services/orion-hub/static/js/app.js†L5564-L5569】 |
| `refreshTopicStudioCapabilities` | `llm_enabled` | `Boolean(undefined)` => false | Disables enrich button + tooltip "LLM disabled" and disables `llm_*` segmentation options. 【F:services/orion-hub/static/js/app.js†L5515-L5523】【F:services/orion-hub/static/js/app.js†L5584-L5592】 |
| `refreshTopicStudioStatus` | `ok`, `checks.pg.ok`, `checks.embedding.ok`, `checks.model_dir.ok` | null/`--` badges when missing; whole fetch failure => Unreachable | Reachable/Unreachable is about `/ready` fetch success, not capabilities parse success. 【F:services/orion-hub/static/js/app.js†L5480-L5503】 |

### Naming mismatch check
No proven naming mismatch for currently used keys:
- Foundry emits `segmentation_modes_supported`, `enricher_modes_supported`, `supported_metrics`, `default_metric`, `defaults`, `default_embedding_url`, `llm_enabled`.【F:services/orion-topic-foundry/app/routers/capabilities.py†L34-L41】
- Active Hub parser consumes `segmentation_modes_supported`, `supported_metrics`, `default_metric`, `defaults.embedding_source_url`, `default_embedding_url`, `llm_enabled`. It does **not** currently consume `enricher_modes_supported`.【F:services/orion-hub/static/js/app.js†L5582-L5592】

## B.3 Duplicate/legacy Topic Studio paths
- `refreshTopicStudioMvp()` exists and separately fetches `/ready` and `/capabilities`, parses introspection/default embedding, and writes to `ts-mvp-*` elements.【F:services/orion-hub/static/js/app.js†L2404-L2474】
- Current template does **not** include `ts-mvp-*` elements, so MVP path is effectively dormant in current UI render path.【F:services/orion-hub/templates/index.html†L512-L523】
- Current route/mount path calls `refreshTopicStudio()` (split-pane path) from `initTopicStudioUI()`; this is the active path in production template.【F:services/orion-hub/static/js/app.js†L682-L696】
- Recommendation: keep legacy path documented for now; removal is possible but exceeds strict doc-only goal unless explicitly requested.

---

## C) Hub ↔ Foundry mismatch matrix

| Capability key | Foundry provides? | Hub reads (active split-pane)? | Names match? | Type match? | Consequence if missing |
|---|---:|---:|---:|---:|---|
| `service` | yes | no | n/a | n/a | no UI impact. |
| `version` | yes | no | n/a | n/a | no UI impact. |
| `node` | yes | no | n/a | n/a | no UI impact. |
| `llm_enabled` | yes | yes | yes | yes | Hub disables LLM modes + enrich button, shows “LLM disabled”. |
| `llm_transport` | yes | no | n/a | n/a | no UI impact today; useful for ops visibility only. |
| `llm_bus_route` | yes | no | n/a | n/a | no UI impact today. |
| `llm_intake_channel` | yes (bus only else null) | no | n/a | n/a | no UI impact today. |
| `llm_reply_prefix` | yes (bus only else null) | no | n/a | n/a | no UI impact today. |
| `segmentation_modes_supported` | yes | yes | yes | yes | segmentation selector may become empty (or fallback only on fetch failure). |
| `enricher_modes_supported` | yes | no | n/a | n/a | no UI impact today (hardcoded enrich action behavior). |
| `supported_metrics` | yes | yes | yes | yes | metric selector may be empty; on fetch error fallback list shown. |
| `default_metric` | yes | yes | yes | yes | no default metric selected automatically. |
| `cosine_impl_default` | yes | no | n/a | n/a | no UI impact today. |
| `defaults.embedding_source_url` | yes | yes | yes | yes | embedding URL hint/default may not prefill. |
| `defaults.metric` | yes | yes | yes | yes | metric default prefill may not occur. |
| `defaults.min_cluster_size` | yes | yes | yes | yes | min-cluster prefill may not occur. |
| `defaults.llm_bus_route` | yes | no | n/a | n/a | no UI impact today. |
| `introspection.ok` | yes | no (active split-pane), yes (legacy MVP) | mixed by path | yes | no impact in active path; MVP toggles manual dataset mode if absent. |
| `introspection.schemas` | yes | no (active split-pane), yes (legacy MVP) | mixed by path | yes | no impact in active path; MVP schema select empty/manual mode. |
| `default_embedding_url` | yes | yes | yes | yes | embedding fallback key missing still okay if defaults key exists. |

### Proven mismatch findings
1. **Contract surface asymmetry (not a break):** Foundry exposes `enricher_modes_supported`, but active Hub does not consume it. This is not currently breaking but means Hub cannot adapt if Foundry ever removes/changes enrichers dynamically. 【F:services/orion-topic-foundry/app/routers/capabilities.py†L35-L35】【F:services/orion-hub/static/js/app.js†L5582-L5592】
2. **Legacy split in parsing behavior:** MVP path uses introspection fields from capabilities; active split-pane path independently fetches `/introspect/schemas`. Different expectations are present but current template uses split-pane path. 【F:services/orion-hub/static/js/app.js†L2404-L2466】【F:services/orion-hub/static/js/app.js†L5619-L5634】

---

## C.2 Actual LLM enablement logic + tailnet/proxy settings

### Effective logic
- **UI “LLM enabled”** in Topic Studio is based only on capabilities `llm_enabled` boolean. 【F:services/orion-hub/static/js/app.js†L5584-L5592】
- Foundry sets `llm_enabled` directly from `TOPIC_FOUNDRY_LLM_ENABLE`. 【F:services/orion-topic-foundry/app/routers/capabilities.py†L29-L29】
- Bus transport declaration (`llm_transport='bus'`) requires both `TOPIC_FOUNDRY_LLM_USE_BUS=true` and `ORION_BUS_ENABLED=true`. 【F:services/orion-topic-foundry/app/routers/capabilities.py†L17-L17】
- Bus request correlation uses `reply_prefix:correlation_id` and envelope correlation metadata. 【F:services/orion-topic-foundry/app/services/llm_client.py†L134-L152】

### Recommended env snippet (local + tailnet hub proxy)

```env
# Foundry LLM switch
TOPIC_FOUNDRY_LLM_ENABLE=true

# Force bus transport
TOPIC_FOUNDRY_LLM_USE_BUS=true
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://<tailnet-redis-host>:6379/0

# LLM gateway wiring
TOPIC_FOUNDRY_LLM_BUS_ROUTE=LLMGatewayService
TOPIC_FOUNDRY_LLM_INTAKE_CHANNEL=orion:exec:request:LLMGatewayService
TOPIC_FOUNDRY_LLM_REPLY_PREFIX=orion:llm:reply
TOPIC_FOUNDRY_LLM_TIMEOUT_SECS=60
TOPIC_FOUNDRY_LLM_MAX_CONCURRENCY=4

# Hub -> Foundry proxy target
TOPIC_FOUNDRY_BASE_URL=http://orion-topic-foundry:8615
```

Notes:
- `TOPIC_FOUNDRY_BASE_URL` is a Hub setting for proxy destination. 【F:services/orion-hub/app/settings.py†L50-L53】
- Hub frontend always calls `/api/topic-foundry/*` and does not call Foundry directly from browser. 【F:services/orion-hub/static/js/app.js†L559-L559】【F:services/orion-hub/static/js/app.js†L1748-L1751】

---

## D) Verification checklist (browser + curl)

### Browser checklist
1. Open Hub `#topic-studio` tab and verify status badge transitions to **Reachable** after `/ready` fetch. 【F:services/orion-hub/static/js/app.js†L5484-L5493】
2. Confirm segmentation dropdown options match capabilities `segmentation_modes_supported`. 【F:services/orion-hub/static/js/app.js†L5582-L5585】
3. If `llm_enabled=false`, verify `llm_*` segmentation options are disabled, enrich button disabled, and tooltip shows `LLM disabled`. 【F:services/orion-hub/static/js/app.js†L5515-L5523】【F:services/orion-hub/static/js/app.js†L5589-L5592】
4. Confirm copied URL buttons emit `/api/topic-foundry/ready` and `/api/topic-foundry/capabilities` links in same-origin context. 【F:services/orion-hub/static/js/app.js†L5980-L5985】

### Curl checklist
1. `GET /api/topic-foundry/ready` returns JSON with `ok` + `checks`. (Proxy route) 【F:services/orion-hub/scripts/api_routes.py†L199-L207】
2. `GET /api/topic-foundry/capabilities` returns `service`, arrays for `segmentation_modes_supported` and `supported_metrics`, and boolean `llm_enabled`. 【F:services/orion-topic-foundry/app/routers/capabilities.py†L25-L41】
3. If expecting bus LLM mode, verify `llm_enabled=true`, `llm_transport="bus"`, and non-empty `llm_bus_route`.

