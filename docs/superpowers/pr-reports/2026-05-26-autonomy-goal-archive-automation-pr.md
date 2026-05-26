# PR: Hub chat reliability, memory graph fixes, recall lane clamp, goal archive automation

**Branch:** `feat/autonomy-goal-archive-automation`  
**Base:** `main`

---

## Summary

This branch bundles operational fixes for day-to-day Hub chat and memory-graph workflows on Grounded Small / Quick lanes, plus autonomy goal-archive automation in `orion-actions` and spark.

| Area | What changed |
|------|----------------|
| **Hub chat UI** | Orion messages render in the conversation pane even when debug panels throw; WebSocket connects early, waits before HTTP fallback, no reconnect loop on fallback |
| **Recall (echo fix)** | Grounded Small + Quick auto-select `assist.light.v1`; exec clamps `hub_chat_lane` → echo-safe profile; Hub sends `recall_profile_explicit` |
| **Memory graph suggest** | Compact prompt, quick lane routing, `json_truncated` detection, higher token defaults |
| **Memory graph approve** | `utterance_text` backfill for SHACL; Postgres `jsonb` fix for `evidence` on card insert |
| **Deploy / env** | `PROJECT=orion-athena` in hub/cortex compose examples; `scripts/sync_local_env_from_example.py` |
| **Autonomy** | Goal archive drain on actions startup + nightly scheduler; spark post-publish trim |

---

## Hub chat

### Message display (`fix/hub-orion-message-display`, merged)

- Append Orion message body **before** autonomy/stance debug panel updates (failures in panels no longer block the reply bubble).
- Safer JSON stringify for chat-stance debug.

### WebSocket vs HTTP fallback

- **Root cause:** WS started only after slow `initSession` / library load; sends raced to HTTP. HTTP fallback called `setupWebSocket()` and closed a socket still in `CONNECTING`.
- **Fix:** Connect WS on page load; `waitForWebSocketOpen(5s)` before fallback; do not reconnect from fallback path.
- Resolve API/WS URLs from `__HUB_CFG__` overrides or app root (`/ws`, `/api`) — not inferred `/hub/ws` path prefix.
- Server: `/hub/ws` alias for path-prefixed proxies.
- HTTP fallback: one warning per session, `Processing (HTTP)...` status, clearer empty/error responses.

### Recall profile per lane

| Hub lane | Recall profile (auto) |
|----------|------------------------|
| Grounded Small | `assist.light.v1` (`render_transcript_user_only` — no pasted `OrionResponse` echo) |
| Quick | `assist.light.v1` |
| Brain | `recall.v1` |

- Hub dropdown syncs on mode change; `recall_profile_explicit: true` so Hub choice beats verb default `chat.general.v1`.
- Exec: `apply_hub_chat_lane_recall_clamp()` in router/supervisor/executor.

---

## Memory graph

- **Suggest:** Shorter `memory_graph_suggest_prompt.j2`; `default_mode=quick`, `skip_brain_reply_context`, route `llm_route=quick`; cortex skips brain reply context for suggest verb.
- **Approve / RDF:** `orion/memory_graph/utterance_text.py` — `ensure_draft_utterance_text` wired in `json_to_rdf`, `approve`, hub routes; JS coalesce merges bridge turn text.
- **Postgres:** `memory_cards.insert_card` — `json.dumps` for `evidence`, `time_horizon`, `subschema` jsonb columns.

---

## Autonomy goal archive (original branch scope)

- `orion-actions`: startup backlog drain + nightly 03:15 archive scheduler.
- `orion-spark-concept-induction`: post-publish goal trim when `AUTONOMY_GOAL_ARCHIVE_ENABLED=true`.
- `orion/autonomy/goal_archive.py`: `archive_subjects_drain()` for bootstrap rounds.

---

## Env / compose

| File | Notes |
|------|--------|
| `services/orion-hub/.env_example` | `PROJECT=orion-athena`, `HUB_API_BASE_OVERRIDE` / `HUB_WS_BASE_OVERRIDE` |
| `services/orion-cortex-exec/.env_example` | `PROJECT`, `LLM_CHAT_*`, memory graph token defaults |
| `services/orion-actions/.env_example` | goal archive scheduler flags |
| `AGENTS.md` | note to run `scripts/sync_local_env_from_example.py` after example edits |

```bash
python scripts/sync_local_env_from_example.py
```

---

## Deploy

```bash
cd services/orion-hub && docker compose up -d --build
cd services/orion-cortex-exec && docker compose up -d --build
cd services/orion-actions && docker compose up -d --build
cd services/orion-spark-concept-induction && docker compose up -d
```

Hard-refresh Hub after deploy (static JS cache).

---

## Test plan

- [ ] Hub: status shows **Connected.** before send; chat uses WS (no yellow HTTP warning)
- [ ] Hub: Orion reply visible in conversation pane (not only history log)
- [ ] Grounded Small: recall profile `assist.light.v1`; existential follow-up does not echo old `OrionResponse` from digest
- [ ] Memory graph: suggest returns JSON; approve passes SHACL; card insert succeeds
- [ ] Actions: `autonomy_goal_archive_drain` on startup; scheduler tick in logs

**Automated (run in CI / locally):**

```bash
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_hub_chat_lane_recall_clamp.py \
  services/orion-hub/tests/test_chat_stance_debug_panel.py \
  services/orion-hub/tests/test_memory_graph_suggest_escalation.py \
  tests/test_memory_graph_utterance_text.py -q
```
