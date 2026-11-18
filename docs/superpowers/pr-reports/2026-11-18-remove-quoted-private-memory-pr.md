## Summary

- Removes quoted private memory text from 9 committed docs. Each removed span reads `[removed: private memory]`.
- Found with the stricter mesh-edge leak check (`orion/memory/mesh_edge.py`, verbatim + 8-word shingle) run against every tracked file, plus a hand sweep of the same files for passages the check missed.
- Sentences were reworded only where a doc stopped making sense without the removed text. A few place names became the doc's existing placeholders (`<trip city>`, `[trip city]`, `[spouse alias]`).
- Technical hits (host, hardware, repo and tool text) were left as they are.

## Outcome moved

The 11 flagged files go from 26 guard hits to 7. The 7 left are technical sentences that were judged not private.

## Current architecture

Design docs and PR reports under `docs/superpowers/` quoted chat text and memory rows directly as evidence. The repo is public.

## Architecture touched

Docs only. No code, config, schema, bus or env change.

## Files changed

Removed spans per file (markers added; rewording-only edits not counted):

- `docs/superpowers/pr-reports/2026-10-01-memory-intake-stop-junk-pr.md`: 1
- `docs/superpowers/pr-reports/2026-10-09-outreach-staleness-and-presence-pr.md`: 3
- `docs/superpowers/pr-reports/2026-11-13-graphiti-trial-report.md`: 44
- `docs/superpowers/pr-reports/2026-11-15-graphiti-trial2-report.md`: 44
- `docs/superpowers/specs/2026-08-14-crystallization-queue-auto-gate-analysis.md`: 3
- `docs/superpowers/specs/2026-09-29-recall-semantic-retrieval-pipeline-design.md`: 1
- `docs/superpowers/specs/2026-10-07-situation-graph-design.md`: 2
- `docs/superpowers/specs/2026-10-13-reading-resonance-design.md`: 1
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: 7
- `docs/superpowers/pr-reports/2026-11-18-remove-quoted-private-memory-pr.md`: this report.

Two flagged files had only technical hits and are unchanged: `2026-11-04-capacities-claim-by-default-pr.md` and `2026-11-08-prediction-error-suite-rebuild-design.md`.

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
mesh-edge guard (load_guard().find) over the 11 flagged files: 26 hits before, 7 after (all 7 technical)
git diff --check: clean
```

## Evals run

```text
None. Docs-only change.
```

## Docker/build/smoke checks

```text
None. Docs-only change.
```

## Review findings fixed

- Finding: the guard missed family, health and home details that were already partly placeholdered (for example place names, relatives' names and medical details in the Graphiti trial appendices).
  - Fix: hand sweep of all 11 files; those spans are removed too.
  - Evidence: keyword re-grep of the 11 files is clean for those details.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: high. Concern: git history still holds the old text. This PR only fixes the current tree.
  - Mitigation: none here. Rewriting history (and force-pushing) needs a separate decision by Juniper. This PR does not do it.
- Severity: low. Concern: the guard catches verbatim and near-verbatim text only; real rewording passes (#3699). The hand sweep covered the 11 flagged files only, not the rest of `docs/`.
  - Mitigation: none in this PR.
- Issues opened: none. Issues closed: none.

## PR link

See the PR this report ships in.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
