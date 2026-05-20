# PR: Autonomy compact card degraded-state visibility

**Branch:** `feat/autonomy-compact-degraded-state`  
**Worktree:** `.worktrees/feat-autonomy-compact-degraded-state`  
**Commit:** `b1df7ae6`

## Summary

The Hub autonomy compact debug card was showing silent `--` for dominant drive, top drives, tensions, and expected posture when `selected_subject=orion` had a partial graph lookup: identity + goals succeeded but the drives SPARQL facet timed out. This PR adds structured degraded-state fields to the backend summary contract and renders explicit unavailable/degraded labels in the Hub UI instead of blank placeholders.

Proposal-only safety is preserved: relationship drives are never substituted for Orion drives; a context note explains when relationship context exists but was not used.

## Root cause

| Layer | Behavior |
|-------|----------|
| `GraphAutonomyRepository._query_subject` | Partial success → `availability=available`, `unavailable_reason=timeout`, state built without drive audit |
| `summarize_autonomy_state` | Treated missing drives like empty healthy state; `raw_state_present=true` |
| Hub `normalizeAutonomyModel` / `updateAutonomyDebugPanel` | Rendered empty strings as `--` with no facet health |

## Changes

### Backend contract (`orion/autonomy/`)

- **`models.py`** — `AutonomySummaryV1` gains `state_quality`, `stance_mode`, `degraded_reason`, `facet_health`, `context_note`, `selected_subject`
- **`summary.py`** — `summarize_autonomy_lookup()` derives degraded semantics from subquery diagnostics; explicit context note when relationship drives ok but Orion drives failed
- **`repository.py`**
  - Configurable `AUTONOMY_DRIVES_QUERY_LIMIT` (chat stance passes compact limit)
  - Subquery `elapsed_ms` measured inside worker (not queue wait)
  - Short-circuit only when preferred subject is fully healthy (not partial)

### Cortex exec wiring

- **`chat_stance.py`** — uses `summarize_autonomy_lookup`; exports degraded fields in summary + `autonomy_debug._runtime`

### Hub UI

- **`app.js`** — `formatAutonomyFieldLabel()` renders explicit unavailable reasons; overview shows `autonomy state`, facet health, context note, stance mode; inline card includes selected subject + degraded reason

### Config

- **`services/orion-cortex-exec/.env_example`** — `AUTONOMY_DRIVES_QUERY_LIMIT=80`
- **`docker-compose.yml`** — env passthrough
- Local **`services/orion-cortex-exec/.env`** — parity key added (gitignored)

## Acceptance scenario

Given:

- `selected_subject = orion`
- `orion.identity` ok · `orion.drives` timeout · `orion.goals` ok
- `relationship.drives` ok

Compact card now shows equivalent to:

```text
Autonomy state: degraded_drives_timeout
Selected subject: orion
Reason: Orion drives facet timed out

dominant drive: unavailable — Orion drives facet timed out
top drives: unavailable — drives facet failed
top tensions: unavailable — drive competition requires drives

Proposals:
- Clarify autonomy boundaries without executing any new action.

Stance:
proposal-only because selected subject drives were unavailable

Context note:
relationship drives are available, but were not substituted for Orion drives
```

## Tests

```bash
cd .worktrees/feat-autonomy-compact-degraded-state
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  tests/test_autonomy_summary_degraded.py \
  tests/test_autonomy_summary.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py \
  services/orion-cortex-exec/tests/test_autonomy_repository_list_latest_short_circuit.py \
  services/orion-cortex-exec/tests/test_autonomy_repository_graph_diagnostics.py \
  services/orion-hub/tests/test_autonomy_runtime_ui_panel.py \
  -q --tb=short
# 58 passed
```

New coverage highlights:

- `test_drives_timeout_partial_state_marks_degraded`
- `test_relationship_context_note_without_substitution`
- `test_chat_stance_partial_drives_timeout_exports_degraded_summary`
- `test_list_latest_does_not_short_circuit_on_partial_orion`
- Hub static contract for `formatAutonomyFieldLabel` + degraded labels

## Verification

| Command | Exit | Result |
|---------|------|--------|
| Targeted pytest suite (above) | 0 | 58 passed |

Live Hub stack verification not run in this session — deploy cortex-exec + hub and reproduce the original timeout scenario to confirm UI labels in browser.

## Remaining risks

1. **Drives query still heavy at default `AUTONOMY_DRIVES_QUERY_LIMIT=80`** — docker-compose passes 80; chat stance code defaults compact to 20 only when env unset. Lower env to 20 if timeout persists.
2. **Fully unavailable lookup** (`state_quality=unavailable`) still shows `--` for drive fields — follow-up if that path becomes common.
3. **Live timeout reproduction** — UNVERIFIED against running Fuseki stack in this session.

## Test plan

- [ ] Merge branch; rebuild `orion-cortex-exec` + `orion-hub`
- [ ] Trigger chat turn where `orion.drives` times out (or mock via test harness)
- [ ] Confirm compact card shows degraded labels, not `--`
- [ ] Confirm raw debug JSON includes `state_quality`, `facet_health`, `context_note`
- [ ] Confirm proposal-only headline still present; no relationship drive substitution in summary
