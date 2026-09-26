## Summary

- Reconstruct contractor asks and motor actions from existing records without changing Orion's behavior.
- Show missing and unverifiable links instead of treating saved output as evidence of learning.
- Preserve the separate visual-outcome path; missing field scores on current images are intentional.
- Add offline replay, read-only live collection, regression tests and a focused CI workflow.

## Outcome moved

Patch 1 of the agency episode plan is complete: the question “which links can we actually join?” has a repeatable command and a captured replay. No learning or causal closure is claimed. The next patch can target durable pre-action expectations and later-decision receipts on the existing ask lane.

## Current architecture

Contractor HelpRequests and PeerBriefs live in the worldview graph; briefs also persist in SQL. Consumed markers record prompt offering, not a later decision. Dispatch frames, results and field outcomes already exist, and posteriors already feed dispatch/allocator consumers. Current image actions instead carry visual outcomes. The audit maps these existing seams; it does not replace them.

## Architecture touched

Local read-only analysis only: `orion/autonomy` reconstruction/reader and one command in `scripts/analysis`. No running service is changed. PostgreSQL is forced read-only with a statement timeout; graph access uses `GRAPH.RO_QUERY`. The event bus is not used.

## Files changed

- `orion/autonomy/agency_episode.py`: deterministic evidence projection, source references, temporal checks and explicit gaps.
- `orion/autonomy/agency_episode_reader.py`: bounded SQL/graph metadata collection with unavailable/truncated source reporting.
- `scripts/analysis/report_agency_episodes.py`: live audit and offline replay; optional new owner-only snapshot file.
- `orion/autonomy/tests/test_agency_episode*.py`: provenance, conflicts, replay, read-only enforcement, caps and privacy checks.
- `orion/autonomy/evals/run_agency_episode_eval.py` and fixture: end-to-end CLI replay of selected real metadata with evidence-removal/reordering scenarios.
- `.github/workflows/agency-episode-audit.yml`: focused tests and replay eval in CI.
- `docs/agency-episode-audit.md`: usage, current architecture, source-backed gaps and next patch.
- `docs/superpowers/specs/2026-09-26-agency-episode-plan.md`: previously prepared proposal included with the implementation branch.
- This report.

## Schema / bus / API changes

- Added: local `agency_episode_audit.v1` report format; not a bus event or persisted cognition schema.
- Removed: none.
- Renamed: none.
- Behavior changed: none in production.
- Compatibility notes: existing contracts and stores remain authoritative. Retention or unreadable sources cannot establish that an event never happened.

## Env/config changes

- Added keys: none. The standalone command reads the existing `ORION_PG_DSN` convention.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable; no template changes.
- skipped keys requiring operator action: none.

## Tests run

```text
python -m pytest orion/autonomy/tests/test_agency_episode.py orion/autonomy/tests/test_agency_episode_reader.py -q
28 passed
git diff --check
clean
```

## Evals run

```text
python orion/autonomy/evals/run_agency_episode_eval.py
10/10 evidence-honesty checks passed
```

This eval grades reconstruction, not Orion's learning quality. It cannot prove sentience, causal effects, or the absent experience-to-choice loop.

## Docker/build/smoke checks

No runtime service, boot configuration or dependency change; no Docker build/restart required. Live read-only CLI ran against host PostgreSQL and FalkorDB. Initial sample: 3 asks, 6 motor episodes combining recent results and older scores. Final replay fixture: 1 ask, 2 motor episodes, all selected sources readable.

Evidence:

- Ask `hr_6b3e0ad9fdc6_n6_menu_assembly` → `brief-27576141f7fd`: graph answer edge, SQL row and consumed mark exist; later choice remains UNVERIFIED.
- Recent `tick_3aeed7bbf7f9` image execution: persisted `visual_outcome=produced`, no field score by design. Independent artifact/perception trace remains UNVERIFIED.
- Historical `tick_d0db6ae013df` execution: result saved at `2026-09-08T08:31:33.913681Z`, prediction-bearing frame at `08:31:33.996928Z`; frame persistence followed execution. Old proposal frame not returned by exact-ID lookup.

Full trace metadata and source paths are documented in `docs/agency-episode-audit.md`. Local full sample: `/tmp/agency-episode-audit-20260926-v2.json`; final small sample: `/tmp/agency-episode-audit-20260926-v3.json`. Neither contains question/reply prose.

## Review findings fixed

- Finding: contradictory source records were excluded and then falsely reported as missing.
  - Fix: propagate conflict status into negative evidence claims.
  - Evidence: duplicate-brief regression requires `unverified`.
- Finding: conflicting dispatch candidates could still make their prediction appear missing.
  - Fix: report missing only when a candidate resolves uniquely and lacks a prediction.
  - Evidence: contradictory-prediction regression requires unverified selection and expectation.
- Audit correction: current visual actions deliberately bypass the field ledger.
  - Fix: retain visual outcomes and scope the score link to field evidence; do not report an outage from absence alone.
  - Evidence: live visual outcome replay and deferral test.

Review used the `review-agent` skill in a read-only subagent.

## Restart required

No restart required.

## Risks / concerns

- Severity: informational.
- Concern: bounded samples and non-atomic graph/SQL reads cannot prove global absence; older records may have expired. Existing source data lacks durable precommit and later-decision receipts.
- Mitigation: explicit scope, source status, referenced identities and UNVERIFIED verdicts; no signal feeds back into cognition. Future signals must pass the metric gate before wiring.

## PR link

Pending creation.
