# PR Report: investigation_v2 evidence sweep (PR2)

**Branch:** `feature/investigation-v2-pr2`  
**Scope:** Context-exec evidence fanout — replaces PR1 skeleton with read-only per-source probes and `InvestigationReportV2`.

## Problem

PR1 plumbed neutral `investigation_v2` mode and profile-derived permissions, but returned only a skeleton artifact. A single organ failure (e.g. recall timeout) could still poison legacy investigation paths, and repo evidence depended on prompt keyword routing in old modes.

## Solution

1. **Schemas:** `SourceStatus`, `SourceResult`, `EvidenceBundle`, `InvestigationReportV2` in `orion/schemas/context_exec.py`; registered in `orion/schemas/registry.py`.
2. **Evidence sweep:** `run_investigation_v2` fans out repo, traces, recall, memory, runtime, and health probes via `safe_probe` (permission gate, timeout, exception isolation).
3. **Repo probe:** Reuses `_repo_search_terms` / grep primitives from alexzhang repo-impact path — no keyword routing.
4. **Result composition:** `answer_status` (`partial_grounding`, `answered_grounded`, `dependency_unavailable`, `no_reliable_evidence`, `blocked`) from per-source statuses; recall timeout → `recall.status=unavailable` without collapsing repo hits.
5. **Runner:** `_run_investigation_v2` replaces skeleton handler; artifact type `InvestigationReportV2`.
6. **Settings:** `CONTEXT_EXEC_INVESTIGATION_V2_PROBE_TIMEOUT_SEC` (default 15s).

## Per-source behavior

| Source | Permission gate | PR2 behavior |
|---|---|---|
| repo | `read_repo` | Grep using question-derived anchors |
| traces | `read_redis_traces` | `organ_runtime.traces_search` |
| recall | `read_recall` | Bus RPC via `recall_query`; timeout → unavailable |
| memory | `read_memory` | `namespace.memory.search_claims` |
| runtime | `read_runtime_logs` | `skipped` (not wired yet) |
| health | always | Shallow dependency snapshot |

## Tests added

- **A** Recall timeout isolation — repo still hits, `partial_grounding`
- **B** Repo runs without `"repo"` / `"impact"` prompt words
- **C** Per-source status preservation (hit / no_hit / unavailable)
- **D** `read_repo=False` blocks repo probe execution
- **E** Agent compat flag path unchanged
- **Regression** Trace hits through full runner build `findings_bundle`

## Files changed

- `orion/schemas/context_exec.py`
- `orion/schemas/registry.py`
- `orion/schemas/tests/test_context_exec_investigation_v2.py`
- `services/orion-context-exec/app/investigation_v2.py`
- `services/orion-context-exec/app/runner.py`
- `services/orion-context-exec/app/artifact_builder.py`
- `services/orion-context-exec/app/settings.py`
- `services/orion-context-exec/tests/test_investigation_v2.py`
- `services/orion-context-exec/.env_example`, `README.md`
- `services/orion-hub/README.md`

## Verification

```bash
cd .worktrees/feature/investigation-v2-pr2
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_investigation_v2.py \
  orion/schemas/tests/test_context_exec_investigation_v2.py \
  services/orion-hub/tests/test_investigation_v2_request.py -q
# exit 0 — 16 passed
```

Local `.env` synced (gitignored): `CONTEXT_EXEC_INVESTIGATION_V2_PROBE_TIMEOUT_SEC=15` added to `services/orion-context-exec/.env`.

## Non-goals (deferred to PR3+)

- Parallel probe fanout (`asyncio.gather`)
- Final reducers / Hub polished rendering
- Full runtime log / readiness probes
- Removing legacy modes or `_infer_context_exec_mode`

## Remaining risks

- Probes run sequentially; wall-clock latency is sum of probe times.
- Runtime probe intentionally `skipped` until PR4 wiring.
- Flag must stay consistent on Hub **and** context-exec for agent-compat paths.
