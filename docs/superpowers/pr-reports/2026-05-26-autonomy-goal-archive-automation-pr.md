# PR: Autonomy goal archive automation (+ Hub message display fix)

**Branch:** `feat/autonomy-goal-archive-automation`  
**Includes:** Hub fix from `fix/hub-orion-message-display` (merged)

---

## Summary

| Service | Change |
|---------|--------|
| `orion-actions` | Startup backlog drain + nightly 03:15 goal archive (no host script) |
| `orion-spark-concept-induction` | Post-publish goal trim (`AUTONOMY_GOAL_ARCHIVE_ENABLED=true`) |
| `orion/autonomy/goal_archive.py` | `archive_subjects_drain()` for bootstrap rounds |
| `orion-hub` | Orion reply body renders even when autonomy/inspect panels throw |

---

## `.env_example` updates (committed)

| File | Key defaults |
|------|----------------|
| `services/orion-actions/.env_example` | `ACTIONS_DAILY_GOAL_ARCHIVE_RUN_ON_STARTUP=true`, `AUTONOMY_GOAL_ARCHIVE_BOOTSTRAP_MAX_ROUNDS=30`, Fuseki URLs + `RDF_STORE_USER`/`PASS` |
| `services/orion-spark-concept-induction/.env_example` | `AUTONOMY_GOAL_ARCHIVE_ENABLED=true`, graph URLs |
| `services/orion-actions/docker-compose.yml` | Same vars passed through |
| `services/orion-spark-concept-induction/docker-compose.yml` | `AUTONOMY_GOAL_ARCHIVE_ENABLED` default **true** |

Sync local `.env` from these after merge (not committed).

---

## Deploy

```bash
cd services/orion-actions && docker compose up -d --build actions
cd services/orion-spark-concept-induction && docker compose up -d
cd services/orion-hub && docker compose up -d --build
```

Logs: `autonomy_goal_archive_drain`, `autonomy_goal_archive_scheduler_result`, `autonomy_goal_archive_tick`.

---

## Test plan

- [ ] Hub: Orion text appears in main conversation pane after chat turn
- [ ] Actions: `autonomy_goal_archive_drain` on first tick after restart
- [ ] Spark: goals published → `autonomy_goal_archive` log lines (no host CLI)
