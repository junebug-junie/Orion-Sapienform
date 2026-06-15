# PR Report: investigation_v2 composite report (PR3)

**Branch:** `feature/investigation-v2-pr3`  
**Base:** `feature/investigation-v2-pr2`  
**Scope:** Reducers + deterministic composer + optional LLM synthesis + Hub rendering for `InvestigationReportV2`.

## Problem

PR2 returned a raw `EvidenceBundle` with per-source statuses but no human-useful composite report. Recall/LLM failures could dominate the headline even when repo evidence grounded an answer. Hub still rendered generic operator blobs.

## Solution

1. **Schema:** `InvestigationSectionV2` + expanded `InvestigationReportV2` (`sections`, `grounded_sources`, `unavailable_sources`, `limitations`, `next_actions`, `raw_evidence`). Registered in `orion/schemas/registry.py`.
2. **Reducers:** `investigation_v2_reducers.py` converts each `SourceResult` into a readable section. Repo reducer reuses `_agent_chain_risk` and repo-impact path/config/test anchor extraction.
3. **Deterministic composer:** `compose_investigation_report()` builds summary and conservative `answer_status` before any LLM pass (`answered_grounded`, `partial_grounding`, `dependency_unavailable`, `no_reliable_evidence`, `blocked`, `error`).
4. **Optional synthesis:** `investigation_v2` added to `SYNTHESIS_MODES`. Runner calls `run_agent_synthesis` after deterministic report; failure adds limitation `"LLM synthesis unavailable; deterministic evidence summary returned."` without erasing sections.
5. **Hub rendering:** `format_investigation_v2_report()` (Python + JS) shows answer status, summary, per-source sections, unavailable/failed/blocked lists, and limitations.

## Per-source reducer behavior

| Source | hit | no_hit | unavailable/error/blocked |
|---|---|---|---|
| repo | paths, risk, config anchors, tests, migration notes | searched, no anchors | preserved distinctly |
| traces | correlation handles + snippets | no matching traces | dependency/probe failure |
| recall | hit count + findings | searched, no hits | dependency failure (not headline collapse) |
| memory | claim hits | no namespace matches | blocked/unavailable preserved |
| runtime | shallow status | skipped (PR4 wiring) | blocked preserved |
| health | dependency snapshot | always shallow | n/a |

## Tests added (A–F)

- **A** Mixed evidence: repo hit + traces no_hit + recall unavailable + memory no_hit → `partial_grounding`/`answered_grounded`, recall in `unavailable_sources`
- **B** LLM synthesis failure → deterministic report + limitation preserved
- **C** Repo reducer includes affected paths and config anchors
- **D** All no_hit → `no_reliable_evidence` with searched sections
- **E** No hits + recall/traces unavailable → `dependency_unavailable`
- **F** Hub `format_investigation_v2_report` renders summary + sections

## Files changed

- `orion/schemas/context_exec.py`
- `orion/schemas/registry.py`
- `orion/schemas/tests/test_context_exec_investigation_v2.py`
- `services/orion-context-exec/app/investigation_v2_reducers.py` (NEW)
- `services/orion-context-exec/app/investigation_v2.py`
- `services/orion-context-exec/app/runner.py`
- `services/orion-context-exec/app/agent_synthesis.py`
- `services/orion-context-exec/app/artifact_builder.py`
- `services/orion-context-exec/tests/test_investigation_v2.py`
- `services/orion-context-exec/tests/test_investigation_v2_reducers.py` (NEW)
- `services/orion-context-exec/README.md`
- `services/orion-hub/scripts/context_exec_agent_bridge.py`
- `services/orion-hub/static/js/app.js`
- `services/orion-hub/tests/test_investigation_v2_request.py`
- `services/orion-hub/README.md`

## Verification

```bash
cd .worktrees/feature/investigation-v2-pr3
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_investigation_v2.py \
  services/orion-context-exec/tests/test_investigation_v2_reducers.py \
  orion/schemas/tests/test_context_exec_investigation_v2.py \
  services/orion-hub/tests/test_investigation_v2_request.py -q
# exit 0 — 22 passed

PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_agent_synthesis.py -q
# exit 0 — 16 passed
```

No new env keys in PR3 (`.env_example` unchanged).

## Non-goals (deferred to PR4)

- Remove `_infer_context_exec_mode` / keyword routers
- Parallel probe fanout
- Full runtime log / readiness probes
- Change default production behavior without v2 flag

## Remaining risks

- Probes still sequential (PR2 debt).
- Runtime probe intentionally `skipped` until PR4.
- v2 flag must stay consistent on Hub **and** context-exec.
