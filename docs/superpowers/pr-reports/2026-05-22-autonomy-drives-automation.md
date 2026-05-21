# PR: Autonomy drives timeout — archive automation + Phase 1b

**Branch:** `feat/autonomy-drives-automation`  
**Status:** Ready for review

---

## Summary

Implements the drives-timeout proposal (options A + B): automated goal-graph hygiene and faster chat_stance when Orion drives are slow or skipped.

| Area | Change |
|------|--------|
| `orion/autonomy/goal_archive.py` | Shared archive module (candidates, SPARQL apply, post-publish trim) |
| `scripts/autonomy/archive_stale_goal_proposals.py` | Thin CLI wrapper (`--apply`, `--all-subjects`) |
| `orion/spark/.../bus_worker.py` | Optional `maybe_archive_after_goal_publish` after goal publish |
| `orion/autonomy/repository.py` | Defer Orion drives for `chat_stance`; separate drives query timeout client |
| `services/orion-actions` | **Built-in scheduler** (45s poll loop, same as daily pulse) @ 03:15 local — **not** host `crontab`; needs `AUTONOMY_GRAPH_QUERY_URL` / `UPDATE_URL` on actions |
| Env | cortex-exec, spark-concept-induction, orion-actions `.env_example` + `.env` + docker-compose |

---

## Phase 1b (chat_stance)

- `AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=true` — Orion `drives` subquery skipped when consumer is `chat_stance`; diagnostic `status=deferred` so relationship fallback stays explicit.
- `AUTONOMY_DRIVES_SUBQUERY_TIMEOUT_SEC=12` — drives-only `GraphQueryClient` (default `min(12, AUTONOMY_GRAPH_TIMEOUT_SEC)`).

---

## Archive automation

| Flag | Default | Service |
|------|---------|---------|
| `AUTONOMY_GOAL_ARCHIVE_ENABLED` | false | spark (post-publish trim) |
| `AUTONOMY_GOAL_ARCHIVE_MAX_UPDATES_PER_TICK` | 25 | spark |
| `ACTIONS_DAILY_GOAL_ARCHIVE_ENABLED` | true | orion-actions (nightly) |
| `AUTONOMY_GOAL_ARCHIVE_SUBJECTS` | orion,relationship | actions + archive |

**One-time ops (still recommended after deploy):**

```bash
PYTHONPATH=. python scripts/autonomy/archive_stale_goal_proposals.py --apply --subject orion
PYTHONPATH=. python scripts/autonomy/archive_stale_goal_proposals.py --apply --all-subjects
```

---

## Tests

```bash
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest \
  orion/autonomy/tests/test_goal_archive.py \
  orion/autonomy/tests/test_repository_defer_orion_drives.py \
  scripts/autonomy/tests/test_archive_stale_goal_proposals.py -v
```

---

## Sign-off

- [x] Archive module + CLI refactor
- [x] Nightly actions scheduler
- [x] Phase 1b defer + drives timeout
- [ ] Phase 2 Fuseki index / latest-pointer (deferred)
