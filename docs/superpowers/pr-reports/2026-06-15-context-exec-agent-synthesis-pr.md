# PR: feat(context-exec): add route-bound Agent synthesis and live Hub probes

## Summary

Adds a **route-bound read-only synthesis pass** to context-exec Agent modes after deterministic organ collection, exposes `operator_summary` on runs for Hub, and renders a structured inline Agent response (mode, route, synthesis status, result, proposal, mutation none). Includes live probe script `scripts/context_exec_agent_route_probe.sh`.

Stacks on `feat/context-exec-llm-profile-bind` (includes Hub route selector + `llm_profile` binding).

## Files changed

| Area | Path |
|------|------|
| Schema | `orion/schemas/context_exec.py`, `orion/schemas/registry.py` |
| Synthesis | `services/orion-context-exec/app/agent_synthesis.py` (new) |
| Runner | `services/orion-context-exec/app/runner.py` |
| Settings / env / compose | `services/orion-context-exec/app/settings.py`, `.env_example`, `docker-compose.yml` |
| Docs | `services/orion-context-exec/README.md` |
| Hub bridge / UI | `services/orion-hub/scripts/context_exec_agent_bridge.py`, `static/js/app.js`, `README.md` |
| Probe | `scripts/context_exec_agent_route_probe.sh` (new) |
| Tests | `services/orion-context-exec/tests/test_agent_synthesis.py`, `services/orion-hub/tests/test_llm_route_selector.py` |

## Route-bound synthesis behavior

| Setting | Default | Behavior |
|---------|---------|----------|
| `CONTEXT_EXEC_AGENT_SYNTHESIS_ENABLED` | `true` | Run synthesis for supported modes after artifact validation |
| `CONTEXT_EXEC_AGENT_SYNTHESIS_REQUIRED` | `false` | When false, preserve deterministic artifact on LLM failure |
| `CONTEXT_EXEC_AGENT_SYNTHESIS_MAX_CHARS` | `4000` | Cap synthesis prompt size |
| `CONTEXT_EXEC_AGENT_SYNTHESIS_TIMEOUT_SEC` | `30` | Documented timeout (uses LLM RPC timeout today) |

**Supported modes:** `belief_provenance`, `trace_autopsy`, `repo_impact_analysis`, `patch_proposal`, `memory_correction_proposal`

**Runtime fields:** `model_synthesis_used`, `synthesis_fallback_used`, `synthesis_fallback_reason`, plus existing `llm_profile_*` / `route_used`. Route/profile `fallback_used` remains separate from synthesis fallback.

**Grounding:** synthesis output rejected if it introduces file paths or memory ids absent from artifact/user text (`synthesis rejected: ungrounded`).

**Safety:** read-only; no shell, writes, memory mutation, or proposal execution. `mutation_allowed=false` preserved.

## Hub Agent response behavior

Hub Agent mode inline text:

```text
Agent run complete
Mode: belief_provenance
Route: chat
Synthesis: used / skipped / fallback
Result: <summary>
Proposal: prop_xxx pending_review (if created)
Open Pending Decisions to review.
Mutation: none
```

Proposal runs surface Pending Decisions affordance text; no execution buttons added.

## Test results

```bash
PYTHONPATH=. orion_dev/bin/python -m pytest orion/schemas -q
# 20 passed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests -q
# 159 passed, 1 xfailed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_llm_route_selector.py -q
# 8 passed

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_proposal_review_hub.py -q
# 20 passed (run with route selector: 22 passed)

PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-llm-gateway/tests/test_route_catalog.py -q
# 2 passed

ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh
# BETA GATE PASS

ORION_PY=orion_dev/bin/python STORE=/tmp/orion-proposals.json ./scripts/repl/orion_fresh_main_smoke.sh
# PASS=20 FAIL=0
```

## Live probe results

Stack was up (`http://127.0.0.1:8080/health` OK) but running **pre-deploy** code without this branch. Probe failed route assertion (`route_used` missing from live response) — expected until context-exec/Hub containers are rebuilt with this branch.

After deploy, run matrix:

```bash
for r in chat quick agent metacog; do
  HUB_BASE_URL=http://127.0.0.1:8080 AGENT_ROUTE=$r \
    AGENT_TEXT="Where did the Denver belief come from?" \
    bash scripts/context_exec_agent_route_probe.sh || echo "ROUTE_DOWN or fail route=$r"
done
```

## Remaining risks

- Live operator experience still depends on LLM gateway/bus availability for synthesis; deterministic organ output remains when synthesis falls back.
- `CONTEXT_EXEC_AGENT_SYNTHESIS_TIMEOUT_SEC` is documented but synthesis reuses `CONTEXT_EXEC_LLM_TIMEOUT_SEC` for RPC today.
- PR stacks on unpublished `feat/context-exec-llm-profile-bind`; merge order should land profile binding before this branch.

## Local `.env` sync

New keys added to `services/orion-context-exec/.env_example` and synced to host `services/orion-context-exec/.env` (gitignored).
