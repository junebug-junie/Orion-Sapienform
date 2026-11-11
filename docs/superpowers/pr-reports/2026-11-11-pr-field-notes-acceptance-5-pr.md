## Summary

- Records the 7-day result of the PR field-notes coverage check (spec acceptance check 5). The result is a **PASS**.
- 69 of 69 owed PRs (merged 2026-11-04 .. 11-10, Denver days) have a record. 67 are notes by Orion (97%). 2 are code records by design. 0 PRs are missing.
- Truth eval PASS on all 75 notes. Merge to record latency: median 93 min, worst 11.0 h. Both are inside the 24 h bar.
- Docs only. No code changed, because nothing failed.

## Outcome moved

Acceptance check 5 is closed with evidence. The spec now states how stacked PRs count.

## Current architecture

orion-durable-runs polls GitHub every 15 min. For each merged PR it starts a `pr.field_note` run, or it writes a code record for tiny docs and dependency PRs. Orion writes the note into `/mnt/orion-fcc/self/pr_field_notes.jsonl` through `dig field-note`.

## Architecture touched

None. Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-04-pr-field-notes-design.md`: adds the "Acceptance check 5 result (2026-11-11)" section.
- `docs/superpowers/pr-reports/2026-11-11-pr-field-notes-acceptance-5-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
python scripts/check_personal_terms.py  -> No personal terms in tracked files
git diff --check                         -> clean
```

## Evals run

```text
docker exec orion-athena-harness-governor ... python -P -m orion.dig field-notes 7
  -> 2026-11-04 .. 2026-11-10: 67 field notes, 2 code records; coverage 67/69 (97%), 2 a code record, 0 neither
.venv/bin/python orion/body_atlas/evals/run_field_note_truth_eval.py --live
  -> 75 notes; pr_text refs 138/138, code_then 16/16, organs 75/75, read 75/75; numbers 186/188 (not gated); PASS
python3 services/orion-durable-runs/scripts/pr_field_note_cost.py
  -> 76 runs ended, 74 completed; median 331 s wall, 268 s GPU, 4.6 Wh, 162k prompt tokens
gh pr list --state merged --search "merged:>=2026-11-03"  -> same 69 PRs in the window, none missing from the ledger
docker logs --since 168h orion-athena-durable-runs | grep field.note -> 0 errors since the last restart (11 h)
```

## Docker/build/smoke checks

```text
None. Docs only.
```

## Review findings fixed

- No code review. The PR changes no code. The figures were re-derived from the live ledger, GitHub and the run table.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: `run_id` and `correlation_id` are still empty in every record. Mitigation: nothing reads them. The fix needs the harness step to pass the run id into Orion's turn. That is not a trivial change, so it is left open and stated in the spec.
- Severity: low. Concern: the container log covers only the last 11 h, because the container restarted. Mitigation: the failure count comes from the durable run table instead.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3181
