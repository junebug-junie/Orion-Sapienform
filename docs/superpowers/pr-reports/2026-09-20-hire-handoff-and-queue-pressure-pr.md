# Hire handoff + queue contention pressure — PR report

Branch: `feat/hire-handoff-impl`

Design parent: https://github.com/junebug-junie/Orion-Sapienform/pull/2259

## Summary

- Mind `deep` sittings now get a strong advisory `hire_cursor` nudge in role teach; kickoff Cypher example is neutral (`local_crawl|hire_cursor`), not crawl-first.
- ≥2 allow-listed access refusals in hop notes and Cursor `refused_budget` each produce progress lines (handoff now / resume hops — do not re-hire).
- Official digester EWMA `queue_contention_score` (0–10) + `queue_contention_driver` land on FieldState; Hub reads latest row for disclosure — no Hub Redis EWMA, no raw counts.
- Metric semantic layer + CI static gates registered; Hub splice accepts progress-only disclosure; FieldState gather runs via `asyncio.to_thread`.
- Docs cross-link curiosity + digester READMEs to the spec and score field names.

## Outcome moved

Orion’s hire teach can now surface deep-work handoff pressure, mid-run access refusal, budget-spent resume, and one official queue-weather score — instead of a wallpaper `deep` label, crawl-first examples, and no shared-queue signal. Python still does not auto-MERGE `:InvestigationRole` or `:HelpRequest`.

Live curiosity-turn splice against live FieldState remains **UNVERIFIED** until digester + hub restart and an operator inspects one Orion-origin motor prompt.

## Current architecture

Before this patch:

- Role teach listed Mind work-shape bullets without a strong deep→Cursor sentence; MERGE example favored `local_crawl`.
- No access-refusal or budget-spent progress lines in disclosure.
- No official queue-contention meter; no hire disclosure of shared seed/durable/gateway backlog weather.
- Hub splice required mind work-shape; no progress-only path.

## Architecture touched

- `orion/curiosity` — teach rewrite, disclosure formatters, refusal/budget/queue progress composition
- `orion/field` + `orion/schemas/field_state` — pure EWMA score math + additive FieldState scalars
- `orion-field-digester` — hot-tick producer after significance, before precision baseline
- Metric semantic layer — inner-state registry, definition lock, lineage consumer for Hub hire disclosure
- `orion-hub` — gather progress lines (hop notes, PeerBrief budget, FieldState read); splice; `asyncio.to_thread`
- Specs / plan / gate evidence under `docs/superpowers/`

## Files changed

- `orion/curiosity/kickoff_prompt.py`: neutral role choice example
- `orion/curiosity/role_teach_disclosure.py`: deep strong nudge; refusal/budget formatters
- `orion/curiosity/access_refusals.py`: allow-listed hop-note refusal counter
- `orion/curiosity/hire_progress.py`: compose progress lines fail-open per source
- `orion/curiosity/queue_contention_disclosure.py`: score+driver → one line
- `orion/field/queue_contention.py`: pure EWMA / max score
- `orion/schemas/field_state.py`: `queue_contention_*` fields
- `services/orion-field-digester/app/digestion/queue_contention.py`: digester producer
- `services/orion-field-digester/app/{tensor/update_rules,worker,store,settings}.py` + compose / `.env_example`: wire + config
- `orion/inner_state_registry.py` + `orion/metrics/lineage.py` + `config/metrics/metric_definitions.lock.json`: register consumer
- `orion/hub/queue_contention_field_read.py` + `orion/hub/turn_orchestrator.py`: read + splice + to_thread
- `services/orion-hub/scripts/curiosity_investigation.py`: attach hop/budget hints on payload
- Tests under curiosity, hub, digester, field, metric lineage
- `docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-pressure-design.md`
- `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md` (gate evidence)
- `docs/superpowers/plans/2026-09-20-hire-handoff-and-queue-pressure.md`
- `orion/curiosity/README.md` + `services/orion-field-digester/README.md`: cross-links

## Schema / bus / API changes

- Added: `FieldStateV1.queue_contention_score` (0–10), `queue_contention_driver`, `queue_contention_ewma`, `queue_contention_ewma_n`, `queue_contention_computed_at`
- Added: metric `metric://inner_state/orion-field-digester/field_state.v1#queue_contention_score` with Hub hire disclosure consumer
- Removed: none
- Renamed: none
- Behavior changed: Orion-origin unified turns may splice hire progress lines; digester writes score every tick
- Compatibility notes: additive FieldState fields default quiet (`score=0.0`); Hub fails open if Postgres/FieldState unavailable (omits queue line). No new bus channels.

## Env/config changes

- Added keys: `FIELD_QUEUE_CONTENTION_HALF_LIFE_SEC`, `FIELD_QUEUE_CONTENTION_FLOOR`, `FIELD_DIGESTER_LLM_GATEWAY_URL` (digester)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-field-digester/.env_example`)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (Task 6)
- skipped keys requiring operator action: none

No new Hub curiosity env keys. Existing `HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE` still gates soft disclosure.

## Tests run

```text
# Curiosity / Hub disclosure (Task 8 + follow-up)
PYTHONPATH=.:services/orion-hub pytest \
  orion/curiosity/tests/test_queue_contention_disclosure.py \
  orion/curiosity/tests/test_access_refusals.py \
  tests/test_role_teach_disclosure.py \
  services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py -q
→ 33 passed

PYTHONPATH=.:services/orion-hub pytest \
  services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py \
  orion/curiosity/tests/ \
  -q -k 'role_teach or hire_progress or queue_contention or access_refusal'
→ 20 passed, 33 deselected

# Digester / pure math (Task 6)
PYTHONPATH=.:services/orion-field-digester \
  pytest tests/test_queue_contention.py \
         services/orion-field-digester/tests/test_queue_contention_digestion.py \
         tests/test_field_state_schemas.py -q
→ 20 passed

pytest tests/test_metric_lineage.py::test_queue_contention_score_names_hire_disclosure_consumer_only -q
→ 1 passed (Task 7)
```

## Evals run

```text
No hire-handoff / queue-contention quality eval harness.
Acceptance (Orion writes hire_cursor more often under deep + elevated score)
remains UNVERIFIED until digester + hub restart and live motor_boot inspect.
```

## Docker/build/smoke checks

```text
Live digester write of queue_contention_* and live Hub splice: UNVERIFIED.
Unit/fake FieldState coverage only. Do not treat tests as runtime proof.
```

## Static gates (Tasks 7–8)

Gate evidence (metric quality §0A): `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md`

Commands and results (cited from `.superpowers/sdd/task-7-report.md` and `task-8-report.md`; Task 9 did not re-run — if bare python lacks pydantic in this environment, treat prior PASS as authoritative):

```text
python3 scripts/check_definition_drift.py --gate
→ definition drift gate: PASS (0 changed)

python3 scripts/check_inner_state_registry.py
→ inner_state_registry gate OK (16 entries checked)

python3 scripts/check_metric_lineage.py --gate
→ orphans: bus_channel=17, inner_state=13, organ_signal=160
→ metric lineage gate: PASS

python3 scripts/check_metric_lineage.py --metric queue_contention_score
→ BLAST RADIUS (discovered, non-test, high-confidence): 1
→ orion/hub/queue_contention_field_read.py:24  [attribute]

rg -n "orion:hire:queue_pressure|queue_pressure:ewma" services/orion-hub orion/hub \
  --glob '!**/tests/**' --glob '!**/node_modules/**'
→ OK (no matches)  # no Hub Redis EWMA
```

## Review findings fixed

- Finding: sync FieldState SQLAlchemy inside `execute_unified_turn` blocked Hub event loop
  - Fix: `await asyncio.to_thread(_gather_role_teach_progress_lines, payload)`
  - Evidence: `87fd59d13`
- Finding: (Tasks 1–7) no Critical/Important blockers left open in SDD reviews
  - Fix: n/a
  - Evidence: per-task `.superpowers/sdd/task-*-report.md`

## Restart required

```bash
# From this worktree (not the shared checkout). Print for Juniper — do not sudo.
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

After restart (still UNVERIFIED until run):

1. Confirm digester FieldState rows include `queue_contention_score` / `queue_contention_driver`.
2. One Orion-origin curiosity turn: inspect motor role-teach for deep nudge and (when applicable) refusal / budget / queue lines — never raw pending counts.
3. Confirm no Hub keys `orion:hire:queue_pressure` / `queue_pressure:ewma`.

## Risks / concerns

- Severity: should
- Concern: Live path UNVERIFIED (digester write + Hub splice against real FieldState).
- Mitigation: restart commands above; operator inspect one curiosity motor_boot / FieldState row.
- Severity: should
- Concern: FieldState Postgres read runs on every Orion-origin unified turn (not only curiosity); fails open when DB down.
- Mitigation: narrow gather later if noisy/slow; already fail-open.
- Severity: low
- Concern: Seed backlog is chronically elevated — score≈0 means “at backlog normal,” not empty queue (gate doc).
- Mitigation: documented in gate evidence; disclosure uses score+driver only.
- Severity: low
- Concern: Supervisor second consumer of the score deferred (registry note only).
- Mitigation: follow-up proposal; not in this PR.

## PR link

_(filled after `gh pr create`)_
