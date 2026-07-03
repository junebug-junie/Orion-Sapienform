# Orion Self-Observability — Plan (v1 audit + v2 completion)

> **Provenance:** PR #783 (`feat/orion-self-observability-v1`) referenced this document but it was
> never committed. This file is the reconstructed plan, an audit of what v1 actually shipped,
> and the completion plan executed by `feat/orion-self-observability-v2`.

**Goal:** Make Orion visible to itself *and* to the operator. Close the loop:
observation → introspection → self-refinement. Every observable must be inspectable from the Hub.

## v1 audit (what PR #783 claimed vs. what shipped)

| Claim | Status in v1 | Gap |
|---|---|---|
| 1. Attention schema dimension in self-state | Schema fields only | Nothing populated `attention_schema_type` / `attention_dwell_ticks` / `attention_node_count`; tests exercised defaults only |
| 2–3. Curiosity signals as workspace beliefs | Adapter + registry wired | Felt-state lane reads `substrate_endogenous_curiosity_candidates`, a table no migration created and no code wrote — the belief node could never appear in production |
| 4. Curiosity → REPL focus hint ("partial") | Not shipped at all | No REPL/agent-lane file was touched in the PR |
| 5. Coalition dwell & hysteresis | Computed in `broadcast_projection_from_frame` | `coalition_history` always empty; `substrate_coalition_dwell_log` migration shipped but never written or read |
| 6. Hub presence observable | Schema field only | No producer anywhere; `hub_presence` was permanently `None` |
| 7. Registry wiring (15th producer) | Shipped and correct | — |
| Hub UI | — | Zero UI or API surface; operators had no way to see any of this |

## v2 completion scope

1. **Curiosity persistence** — migration for `substrate_endogenous_curiosity_candidates`
   (`generated_at`, `candidates_json`); substrate-runtime persists endogenous signals each
   curiosity tick (bounded, best-effort, prunes old rows). This makes the existing felt-state
   lane and `curiosity` producer live.
2. **Self-state population** — `build_self_state` accepts the latest attention broadcast
   projection and a hub-presence snapshot; derives `attention_schema_type`
   (`focused_single` = 1 node, `distributed` = 2–5, `open_loop` = unresolved/no stable
   coalition, `none` = idle), `attention_dwell_ticks`, `attention_node_count`, and
   `hub_presence`. Degrades to `None`/defaults on absent or stale inputs.
3. **Dwell log** — broadcast tick writes one bounded row per tick to
   `substrate_coalition_dwell_log`; `coalition_history` transitions populated (cap 10).
4. **Hub presence** — Hub records chat-turn timestamps in-process and upserts a single-row
   snapshot to `substrate_hub_presence` (`last_turn_age_sec`, `turns_per_minute`,
   `connection_health`); self-state runtime hydrates it.
5. **Hub observability surface** — `GET /api/substrate/observability/summary` aggregating
   self-state observability fields, attention broadcast dwell, curiosity candidates, and hub
   presence (each section independently null-degrading), plus a Self-Observability Hub panel.
6. **REPL curiosity hint (v1 Task 4)** — flag-gated (`HUB_AGENT_CURIOSITY_HINT_ENABLED`,
   default off): when the agent lane runs and fresh curiosity candidates exist, prepend one
   bounded focus-hint line to the agent request. Structural gate (lane + data freshness), no
   keyword classification, per anti-slop rules.

## Non-goals

- No new bus channels or message kinds (presence and candidates ride Postgres seams).
- No rung-6 governance changes; curiosity stays observational, hints stay advisory.
- No flag flips: `ORION_ENDOGENOUS_CURIOSITY_ENABLED` and broadcast flags keep their defaults.
- No auto-apply of anything; proposals remain proposals.

## Acceptance checks

- Curiosity tick persists candidates; felt-state reader hydrates them; adapter emits
  `curiosity:unresolved_gaps` from a real row round-trip.
- Self-state rows carry populated attention schema fields when a fresh broadcast exists, and
  `None`/defaults otherwise.
- Broadcast tick writes dwell rows; table stays bounded (prune > 24 h).
- Hub `/api/substrate/observability/summary` returns each section, null-degrading per source.
- Hub panel renders all four sections without errors when sources are absent.
- All collections capped; every new adapter/reader degrades to None; nothing raises on absent input.
