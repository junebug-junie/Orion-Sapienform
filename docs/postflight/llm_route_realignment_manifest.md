# LLM Route Realignment Manifest (Merged Default + HELPER internal + QUICK user lane)

## Scope

This change set keeps logical route contracts intact (`chat`, `agent`, `metacog`, `helper`, `quick`) while changing default Atlas physical placement:

- `chat` → `atlas-worker-1` (`8011`)
- `agent` (logical lane preserved) → `atlas-worker-1` (`8011`) in default merged mode
- `metacog` → `atlas-worker-2` (`8012`) unchanged
- `helper` (internal bounded lane) → `atlas-worker-helper-1` (`8013`)
- `quick` (Hub-visible fast lane) → `atlas-worker-quick-1` (`8015`)

Optional split mode stays config-only:

- `agent` → `atlas-worker-agent-1` (`8014`) via `agent-split` compose profile + route table switch

## Route-table centered architecture

- Routing remains enforced by `LLM_GATEWAY_ROUTE_TABLE_JSON`.
- `served_by` remains observability metadata.
- No schema-level collapse of `chat`/`agent`.
- No Orch mode rewrite and no typed planner/agent RPC rewrites.

## Chat behavior guarantees

- `chat_general` remains two-pass:
  1) `synthesize_chat_stance_brief` on `route="helper"`
  2) `llm_chat_general` on `route="chat"`
- New `chat_quick` is a first-class single-pass verb:
  - one step
  - explicit `route="quick"`
  - no stance-brief dependency
  - still receives lightweight identity context from existing executor injection plumbing

## Non-goals preserved

- No new profile added to `config/llm_profiles.yaml` for QUICK.
- HELPER not exposed in Hub mode selector.
- Metacog routing behavior unchanged.
