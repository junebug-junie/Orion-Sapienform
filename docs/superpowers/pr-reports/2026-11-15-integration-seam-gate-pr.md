# The integration gate: organ registry, CI check, live coverage report (integration patch 2)

Refs #3424 (epic #3434). **Merges after patch 1** (`feat/integration-experience-now`, Refs #3423): this
branch is stacked on it. Spec: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`
(PR #3416), Part A.6 and "Patch 2".

## Summary

- A new rule in `CLAUDE.md` ("Integration gate", the spec's text verbatim): any new producer of Orion
  experience must emit through the experience contract and read the now back, or say why not in the
  organ registry.
- `scripts/check_integration_seam.py` enforces it in CI (orion-static-gates). It fails when a family, a
  durable workflow, an intention adapter, a source kind, or a newly added bus channel belongs to no organ
  and has no exemption with a reason. It prints every new exemption for the reviewer.
- `config/integration/organs.yaml` now covers every producer the check can see: 16 organs, 42 families,
  17 workflows, 1 intention adapter, 28 source kinds; 3 families and 1 workflow exempt, each with a reason.
- `scripts/report_integration_coverage.py`: a read-only live report per organ (lived, now reads, cites
  made, cites received, closed) with the experience use rate. Its first run is the baseline below.
- The metric's SQL moved to `orion/integration/coverage.py`, so the durable-runs route and the report count
  the same way.

## Outcome moved

Before: nothing stopped a new organ from being added with a private pipe. After: CI refuses it until the
registry names the organ or an exemption says why. Proof it bites: after rebasing on main, the gate
failed on a family added there the same day (`orion.look_check`, #3347 part 2a) until it was registered
under the camera organ.

## Current architecture

Patch 1 added the registry with source kinds and salience magnitudes only, read by the now and the
chronicle. No check tied families, workflows, adapters or channels to an organ.

## Architecture touched

- `CLAUDE.md` (= `AGENTS.md`): new section under the hard mandates.
- `config/integration/organs.yaml`: families, workflows, intention_sources, reads_now / exempt_read,
  exempt_emit, exempt_families, exempt_workflows, exempt_channels.
- `scripts/check_integration_seam.py`, `tests/test_check_integration_seam.py`, `.github/workflows/orion-static-gates.yml`.
- `scripts/report_integration_coverage.py`, `orion/integration/coverage.py`,
  `services/orion-durable-runs/app/temporal_self_store.py` (imports the shared SQL).

## What the check detects (spec list, 1-6) and what it does not

1. families in `config/physiology/families.yaml`; 2. `DurableWorkflowV1` values; 3. modules in
`orion/prime/intention_sources/`; 4. `TemporalSourceKind` values (and an `exempt_emit` reason for organs
that emit nothing); 5. each organ's `reads_now` module exists and imports `orion.temporal_self.now` or names
the `/temporal-self/now` route (AST, so a comment does not count), else `exempt_read`; 6. channels added
to `orion/bus/channels.yaml` since the merge base with `origin/main`. Also: a registry entry naming a
producer that does not exist fails (no stale claims), a producer claimed by two organs fails, and an
exemption with an empty reason fails.

Not seen: a producer that writes a new table with no family, workflow, adapter or channel (the coverage
report shows such a kind as "unregistered" once it lands in `temporal_self_event`). "A new Prime concern
source" is in the CLAUDE.md text but not in the spec's detectable list, so it is not checked.

## Organ assignments worth a reviewer's eye

- `durable.self_inquiry` -> curiosity (the self-inquiry line runs as curiosity); `durable.pr_field_note`,
  `compactor.digest`, `topic_foundry.run`, `concept_induction` -> memory; `durable.consult_orion_turn`,
  `hub.outreach` -> chat; `actions.daily_pulse` -> reading; `situation.update` -> metacognition.
- Exempt: `execution_dispatch.{maintain,inspect,substrate}` (dispatch ladder retired 2026-11-07; no
  `grammar_acts` row for these actors ever, live 11-15) and `harness.echo_smoke` (a seam smoke).
- Every organ is `exempt_read` today: no organ reads the now until patch 3 (Prime, reveries, curiosity).

## Files changed

- `CLAUDE.md`: the section.
- `config/integration/organs.yaml`: the full registry.
- `scripts/check_integration_seam.py`: the gate. `tests/test_check_integration_seam.py`: its tests.
- `.github/workflows/orion-static-gates.yml`: runs both.
- `scripts/report_integration_coverage.py`: the live report. `orion/integration/coverage.py`: shared SQL.
- `services/orion-durable-runs/app/temporal_self_store.py`: imports the shared SQL (no behavior change).

## Schema / bus / API changes

- None. No schema, channel or route change.

## Env/config changes

- None. No env key.

## Tests run

```text
tests/test_check_integration_seam.py   14 passed (repo passes; planted: new family, new workflow, new
  intention adapter, new channel in the diff, new source kind, organ with no emit/read reason, fake
  reads_now, stale claim, double claim; exemption with reason passes, without fails)
the same under a venv with only the static-gates deps (pydantic, pydantic-settings, PyYAML, pytest): passed
python scripts/check_integration_seam.py: OK (16 organs; 42 families, 17 workflows, 1 intention adapters, 28 source kinds)
orion-static-gates.yml run step by step locally: 41 of 41 steps ok (integration gate included)
```

## Evals run

```text
python scripts/report_integration_coverage.py (live, read-only), 2026-11-15 05:51Z:
experience use rate: 0 / 2374 = 0.00%; closed organs: 0 of 16 (none)
lived per organ (24 h): metacognition 1357, body 367, camera 298, prediction 214, reverie 92, chat 14,
curiosity 14, painting 12, memory 4, dream 2; reading, play, prime, intention, day_letter, personality 0.
missing tables (migration not applied yet): experience_shown, experience_use.
```

This is the gate's baseline. Reading and Prime read 0 because their kinds start with patch 1's deploy;
patch 1's live read counted 2,980 experiences in 24 h once they are included.

## Docker/build/smoke checks

```text
No runtime change: no image, no restart.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```text
No restart required. Deploy patch 1 first (it carries the only runtime change in the stack).
```

## Risks / concerns

- Severity: low. Concern: the gate fails for any PR that adds a family or workflow without touching the
  registry, including agents' PRs already open. Mitigation: the error names the file and the fix; an
  exemption with a reason is one line.
- Severity: low. Concern: rule 6 needs a merge base; a shallow checkout skips it with a printed note.
  Mitigation: orion-static-gates already checks out with `fetch-depth: 0`.
- Issues: refs #3424. Opened: none. Closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
