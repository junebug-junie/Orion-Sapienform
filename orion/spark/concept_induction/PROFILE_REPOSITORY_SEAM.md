# Spark Concept Profile Repository Seam (Phase 0)

This seam decouples workflow/chat consumers from direct `LocalProfileStore` reads.

## Why it exists

`concept_induction_pass` and `chat_stance` now read concept profiles through a typed repository boundary so future backends (GraphDB read model or RPC retrieval) can be introduced behind one stable interface.

## Current backend

- Backend: `local`
- Implementation: `LocalConceptProfileRepository`
- Source of truth remains unchanged: local JSON profile state via `LocalProfileStore`

## Scope guardrails in this pass

- No GraphDB or RPC reads are implemented.
- No fake graph fallback is introduced.
- Operational local state (`drive_states`, `goal_cooldowns`) stays in `LocalProfileStore` and is intentionally outside this repository contract.
- Contract is limited to workflow-consumable concept profile retrieval (`get_latest`, `list_latest`) plus bounded source status metadata.

## Next-phase hook

Future graph/RPC adapters should implement the same repository interface and be selected in the provider/factory layer without changing workflow/chat consumer logic.
