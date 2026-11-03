# Design: Prime's concerns come from loops that close

## Summary

- A docs-only proposal: `docs/superpowers/specs/2026-11-04-prime-concern-sources-design.md`.
- Prime stops picking concerns from attention arcs. Those are mostly internal surprise gauges and
  body-economy binding lines. Instead it picks from **concern sources**: loops that already tie an
  event to an action to a checkable outcome. v1 sources: camera asks, reading findings, curiosity
  findings, missed visitor expectations, and failing skills.
- The self-monitor grades each source, and each source earns its share of Prime's attention from its
  own record.
- Two prerequisite fixes are named:
  - the street node pinned at 0.0000 in every frame;
  - an ungraded camera-ask loop.

## Outcome moved

No runtime change. It decides the shape before any code. Live evidence:
- 42 Prime concerns in 7 days, 0 from the world, Juniper, reading or curiosity; 40 dropped and 2
  produced something.
- The camera ask loop closed 495 times in 14 days and feeds nothing.

## Current architecture

See the spec. It covers the seven-month history of attention attempts and why they fizzled: they
re-scored the gauges and never added event sources with an action.

## Architecture touched

Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-04-prime-concern-sources-design.md`: the proposal.
- `docs/superpowers/pr-reports/2026-11-04-prime-concern-sources-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed: `ConcernCandidateV1`, a candidate set, a per-source act key)
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
None: docs only.
```

## Evals run

```text
Live read-only audit 2026-11-03/04, in the spec:
- grammar_acts prime predict: 42 bets, 2 produced
- substrate_attention_frames: 66.8% no winner; 57% of winners internal
- street node: 0.0000 in 122,342 frames
- orion_ask: 495 answered and applied, 33 dismissed in 14 days
- temporal_self_arc attention dwell: internal 56%, chat 28%, reading 16%
```

## Docker/build/smoke checks

```text
Not applicable.
```

## Review findings fixed

- Design review is Juniper's: see the spec's "Missing questions".

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: if camera asks already reach Juniper directly, a Prime concern on top may add nothing.
- Mitigation: per-source track records will show it, and such a source earns no share.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3089

🤖 Generated with [Claude Code](https://claude.com/claude-code)
