# Source reference: substrate tier telemetry design

Imported from repo archive:
`docs/superpowers/specs/2026-05-14-substrate-tier-telemetry-persistence-design.md`

Key facts captured as claims:

- New service `orion-substrate-telemetry` subscribes to `orion:substrate:tier_outcomes`.
- Append-only Postgres persistence; orch optionally merges facet into MindRunRequest.
- Hub reads via HTTP proxy, not Redis bus subscription.
