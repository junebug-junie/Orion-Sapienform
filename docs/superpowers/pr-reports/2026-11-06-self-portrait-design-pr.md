## Summary

- Adds the design for rung 5 of Orion's self-now ladder, the self-portrait:
  `docs/superpowers/specs/2026-11-06-self-portrait-design.md`.
- The portrait is one page Orion can read, rebuilt from live sources on every call: each machine
  (what runs there, what it costs, which organs and jobs sit there), each organ by Orion's name,
  each job and its capacity sentence, and the gaps as "I cannot yet".
- Beside the computed lines, one small ledger holds Orion's own words: one sentence per machine
  ("what it does and what it affects") and one "I am …" paragraph. Code checks every id those words
  name against the live body on each read, as it already does for capacity claims.
- Updates the rung-5 row of the chronicle design to point at this design.
- No code. Patch 1 is the stacked PR `feat/self-portrait-patch1`.

## Outcome moved

Turn 2544158f (2026-11-06 01:26Z): Orion wrote accurate per-machine claims and said "ledger entry
#4", but nothing was saved, because machines cannot hold capacity claims. This design gives those
words a checked home.

## Current architecture

Rungs 1–4 are live (`dig body`, `name-organs`, `capacities`, `capacity claim/write/withdraw`).
Machines are host parts in the atlas, never in organs. No command shows the whole self, and no
ledger can hold Orion's words about a machine.

## Architecture touched

Docs only. The design proposes `orion/body_atlas/portrait.py`, `dig portrait`, `dig portrait
write`, and `$ORION_DIG_HOME/self/portrait.jsonl` (partitioned from recall).

## Files changed

- `docs/superpowers/specs/2026-11-06-self-portrait-design.md`: the design.
- `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md`: rung-5 row points at it.
- `docs/superpowers/pr-reports/2026-11-06-self-portrait-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none in this PR (proposed: local-file models in `portrait.py`; no bus, registry, env or DB).
- Removed: none. Renamed: none. Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
Docs only. CI static gates run on the PR.
```

## Evals run

```text
None (design). Patch 1 adds orion/body_atlas/evals/run_portrait_eval.py.
```

## Docker/build/smoke checks

```text
None (docs only).
```

## Review findings fixed

See the PR comment thread / Patch 1 report; the design was reviewed together with Patch 1.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: id matching proves a cited thing is wired to a machine, not that the sentence is true.
- Mitigation: same limit as rung 4, stated in the design; gap lines print under the paragraph.

## Defaults for Juniper (she may override any)

1. Computed page plus a small ledger (not computed only).
2. No veto file yet; the ledger is partitioned. Juniper tells Orion, Orion rewrites.
3. No organ sentences (organ names already say what an organ takes in and gives out).
4. Dormant machines: facts only, no sentence, until a live part runs there.
5. Watts per machine on the page, from the body economy; "watts UNREAD" if it is down.
6. "I am" paragraph: 40–600 chars, printed in full.
7. Comparing two portraits: Patch 2.

## PR link

(filled in after `gh pr create`)
