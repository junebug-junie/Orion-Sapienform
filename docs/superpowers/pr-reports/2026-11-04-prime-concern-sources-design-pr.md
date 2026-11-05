# Design: attention is the competition for Orion's effort (rev 2)

## Summary

- Rev 2 replaces rev 1, which built a second attention system beside the first (Juniper's
  challenge). The spec is now `docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md`.
- **The finding.** Attention feeds a substrate ladder that writes about 40k frames a day per rung and
  binds 0 winners to acts. Orion's real doing (about 90% of acts) runs through the body-economy want
  ledger and the act ledger, beside the ladder.
- **The proposal.** Attention becomes the want-ledger competition:
  - the menu is event wants, each keeping its event identity;
  - the bid is how unusual the event is × earned family weight − cost;
  - the self-monitor is one more evidence source for the weights;
  - binding moves one family at a time.
- **The ladder is retired in stages, not revived.** Proposal templates move to want producers.
  Policy's job becomes one gate. The allocator stays. Feedback and consolidation frames are retired.

## Outcome moved

No runtime change. Live evidence:
- 0 winners bound to an act in 24 h;
- 0 executions allowed out of about 121k "approved" decisions a day;
- 235 real acts a day through the producer path, all with outcomes;
- the want ledger is shadow everywhere except Prime.

## Current architecture

See the spec. It covers the seven-month history of attention attempts and why they fizzled: they
re-scored the gauges and never added event sources with an action.

## Architecture touched

Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md`: the proposal (rev 2; rev 1's file renamed and rewritten).
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
