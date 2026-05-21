# PR: Runtime Trace → Signal Nexus (Milestone A + B)

**Branch:** `feat/runtime-trace-signal-nexus`  
**Worktree:** `.worktrees/feat-runtime-trace-signal-nexus`  
**Base:** `71232c75` (main)  
**Head:** `c09b5b16` (+ review fixes)  
**Design:** [2026-05-20-runtime-trace-signal-nexus-design.md](../specs/2026-05-20-runtime-trace-signal-nexus-design.md)  
**Plan:** [2026-05-20-runtime-trace-signal-nexus-implementation.md](../plans/2026-05-20-runtime-trace-signal-nexus-implementation.md)

## Summary

Joins existing cognition execution truth (`CognitionTracePayload`) into the signal nexus (`OrionSignalV1` + Hub) without duplicating trace truth.

**Milestone A** — turn drill-down:
- Multi-emission adapter contract + gateway processor flatten
- `CognitionTraceAdapter` (1 run + N step signals for `chat_general`)
- Hub `CognitionTraceCache` + correlation index APIs
- Execution Steps panel + Organ Signals correlation view with stub hiding
- §5.8 correlation ID propagation gate

**Milestone B** — live runtime mesh:
- Shared `stub_detection` + `ORGAN_LAYER` taxonomy
- Real adapters for equilibrium, recall, chat_stance, autonomy, spark_introspector, runtime services, and priority B6 organs
- Layer filter dropdown on Organ Signals live mesh

## Commits (25)

| Phase | Commit | Message |
|-------|--------|---------|
| A0 | `700b7e32` | test(gateway): document cognition trace channel preflight |
| A1 | `a8a0d361` | feat(signals): multi-emission AdapterResult contract |
| A1 | `1841b2ad` | feat(gateway): publish each signal from multi-emission adapters |
| A2 | `3ffab105` | feat(cortex-exec): enrich CognitionTracePayload metadata for signal nexus |
| A3 | `69d6130d` | feat(signals): add runtime organ registry entries |
| A3 | `92e1d953` | feat(signals): CognitionTraceAdapter multi-emission for chat_general |
| A4 | `95e14fb7` | feat(hub): CognitionTraceCache and cognition trace API |
| A5 | `fdd032af` | feat(hub): correlation index for signal chain API |
| A6 | `5ecd6d3f` | feat(hub): expose canonical correlation_id in chat turn metadata |
| A7 | `edbc312c` | feat(hub): Execution Steps panel in thought-process.js |
| A8 | `cf12e0cb` | feat(signals): stub detection helper for organ signals UI |
| A8 | `5a40674a` | feat(hub): organ signals correlation view with stub hiding |
| B0 | `8a46bb32` | feat(signals): organ layer taxonomy for mesh filters |
| B1 | `3e500772` | feat(signals): real equilibrium snapshot adapter |
| B2 | `fef9c0d4` | feat(signals): real recall exec result adapter |
| B3 | `b4e1a199` | docs(signals): chat_stance signal adapter bus contract |
| B3 | `e72f17ef` | feat(signals): real chat_stance adapter from ChatStanceBrief |
| B4 | `a8fbb4cd` | feat(signals): real autonomy adapter from exec summary/state |
| B4 | `4ff700b8` | feat(signals): real spark_introspector adapter for spark.signal.v1 |
| B5 | `c857e3a7` | feat(signals): runtime service adapters for gateway orch hub writers |
| B6 | `73b77911` | feat(signals): real collapse_mirror journaler social_memory world_pulse |
| B7 | `bd7d0b5a` | feat(hub): organ signals layer filter UI |
| — | `4fc42a65` | chore(hub): wire cognition trace cache env in docker-compose |
| — | `985afc9d` | fix(signals): align cognition.trace correlation propagation for §5.8 gate |
| — | `c09b5b16` | chore: restore unrelated files accidentally dropped from main |

## Key changes

| Area | Change |
|------|--------|
| `orion/signals/adapters/result.py` | `AdapterResult` + `normalize_adapter_result()` |
| `orion/signals/adapters/cognition_trace.py` | `CognitionTracePayload` → run + step signals |
| `services/orion-signal-gateway/app/processor.py` | Multi-emission publish loop; `_envelope_correlation_id` injection |
| `services/orion-cortex-exec/app/main.py` | `build_cognition_trace_metadata()` enrichment |
| `services/orion-hub/scripts/cognition_trace_cache.py` | Bus subscriber + redacted/debug API |
| `services/orion-hub/scripts/signals_inspect_cache.py` | `source_event_id` correlation index |
| `services/orion-hub/static/js/thought-process.js` | Execution Steps panel |
| `services/orion-hub/static/js/organ-signals-graph-ui.js` | Correlation mode + layer filter |
| `orion/signals/stub_detection.py` | Shared stub detection |
| `orion/signals/layers.py` | Organ layer taxonomy |
| Adapters B1–B6 | Real implementations replacing stubs for priority organs |

## Config / env

**Hub** (`services/orion-hub/.env_example` + `.env` + `docker-compose.yml`):

```env
COGNITION_TRACE_CACHE_ENABLED=true
COGNITION_TRACE_CACHE_MAX=200
COGNITION_TRACE_CACHE_TTL_SEC=300
COGNITION_TRACE_SUBSCRIBE_CHANNEL=orion:cognition:trace
COGNITION_TRACE_API_DEBUG=false
```

**Gateway** — `ORGAN_CHANNELS` extended for cortex/exec/chat/notify paths (settings defaults; no new env keys).

## New API routes

| Route | Purpose |
|-------|---------|
| `GET /api/cognition/trace/{correlation_id}` | Redacted cognition trace (debug when `COGNITION_TRACE_API_DEBUG=true`) |
| `GET /api/signals/correlation/{correlation_id}` | Signal chain for correlation drill-down |
| `GET /api/signals/layers` | Organ → layer map for mesh filters |

## §5.8 correlation gate

Chat responses include `trace_linkage` with canonical `correlation_id`. Gateway injects envelope correlation into cognition trace adapter via `_envelope_correlation_id`. Post-review fix (`985afc9d`) aligns `cognition.trace` bus kind (dot) with colon-pattern preflight.

**Staging checklist** (manual — not automated):

1. Hub chat response `correlation_id`
2. `CognitionTracePayload` on `orion:cognition:trace`
3. `orion:signals:cortex_exec` run signal `source_event_id`
4. `GET /api/cognition/trace/{id}`
5. `GET /api/signals/correlation/{id}`

## Tests

```bash
cd .worktrees/feat-runtime-trace-signal-nexus

# Gateway (31 passed)
./scripts/test_service.sh orion-signal-gateway services/orion-signal-gateway/app/tests/ -q

# Hub nexus (5 passed)
./scripts/test_service.sh orion-hub \
  services/orion-hub/tests/test_cognition_trace_api.py \
  services/orion-hub/tests/test_signals_inspect_api.py \
  services/orion-hub/tests/test_correlation_id_propagation.py \
  services/orion-hub/tests/test_organ_signals_correlation_mode.py -q

# Signals adapters + layers (45 passed)
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/signals/adapters/tests/ orion/signals/tests/ -q

# Hub UI contracts (24 passed — includes thought-process + organ signals tab)
./scripts/test_service.sh orion-hub \
  services/orion-hub/tests/test_chat_stance_debug_panel.py \
  services/orion-hub/tests/test_organ_signals_graph_hub_tab.py -q
```

| Command | Exit | Result |
|---------|------|--------|
| Gateway app tests | 0 | 31 passed |
| Hub nexus tests | 0 | 5 passed |
| Signals adapter tests | 0 | 45 passed |
| Hub UI contract tests | 0 | 24 passed |

Full `./scripts/test_service.sh orion-hub` suite: **6 pre-existing collection errors** in topic_foundry tests (unrelated to this PR).

## Milestone acceptance

### A (spec §6.1)

- [x] Multi-emission processor tests pass
- [x] Gateway preflight: cognition channel subscribed
- [x] Hub 3-step timeline API (Execution Steps panel)
- [x] Correlation graph uses `signal_id` nodes
- [x] Stub hiding + toggle
- [x] Correlation ID gate §5.8 (unit + propagation fix)
- [x] No PII in signals / default cognition API (tests)
- [ ] Staging smoke with live `chat_general` turn — **UNVERIFIED**

### B (spec §6.2)

- [x] Real adapters for equilibrium, recall, chat_stance, autonomy, spark, runtime services, B6 priority organs
- [x] Layer filters Runtime + Cognition in UI
- [ ] ≥8 non-stub nodes during normal chat — **UNVERIFIED** (staging)
- [ ] Missed parent rate <20% — **UNVERIFIED** (staging)

## Deferred / follow-up

- **B6 long tail** — stub adapters remain: `social_room_bridge`, `vision`, `agent_chain`, `planner`, `dream`, `state_journaler`, `topic_foundry`, `concept_induction`, `graph_cognition`, `power_guard`, `security_watcher`
- **`mind` adapter** — registry entry only; cognition trace steps may cover mind handoff
- **Hub `orion:hub:chat:turn`** — hub adapter uses `orion:chat:history:turn` from actual publish path
- **Integration test** — end-to-end chat → bus → gateway → correlation API not automated

## Test plan (operator)

- [ ] Rebuild `orion-hub`, `orion-signal-gateway`, `orion-cortex-exec`
- [ ] Copy cognition trace env keys from `.env_example` to `.env` if missing
- [ ] Send one `chat_general` turn; confirm Execution Steps shows 3 steps with latency
- [ ] Open Organ Signals `?correlation_id=<id>` — chain `graph_cognition → chat_stance → llm_gateway`
- [ ] Toggle stub hiding; confirm non-stub runtime/cognition nodes with layer filters
- [ ] Grep `orion:signals:*` bus payloads for PII (no `final_text` content)

## Remaining risks

- Live §5.8 and Milestone B staging acceptance require compose-up verification
- Correlation UI stub detection improved via `is_stub` on chain items (post-review)
- Some organs still emit stub signals until B6 long tail is completed
