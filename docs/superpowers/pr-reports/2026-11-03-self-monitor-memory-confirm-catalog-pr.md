# Self-monitor: memory-confirmation cards are not an attention-salience bug

## Summary

- The 10-20 audit said Orion's "attention asks" save their importance as 0, and called it a
  recording bug.
- A live re-check (11-03) shows otherwise. All 63 of those outcomes are memory-confirmation cards
  ("is this memory right?"), opened by `orion/memory/episode/confirmation.py`, and no importance or
  confidence number exists before the ask. So there is nothing to save, and no bug to fix.
- The self-monitor's catalog now says this honestly. The faculty is `memory.confirmation`, and it
  is ungradeable because Orion makes no bet before asking. The fix that would unlock it: a "chance
  Juniper confirms" bet, placed in orion-prediction when the card opens.

## Outcome moved

The organ no longer promises a fix that cannot exist. Its "why can't I grade this" line is now
true.

## Current architecture

`orion/self_monitor/catalog.py` listed `attention.ask` as `ungradeable:source_bug`.

## Architecture touched

`orion/self_monitor/catalog.py` only.

## Files changed

- `orion/self_monitor/catalog.py`: `attention.ask` (source_bug) becomes `memory.confirmation`
  (`no_stated_confidence`), with the evidence in a comment.
- This report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: catalog faculty `attention.ask` becomes `memory.confirmation` (report text only; nothing
  reads it yet)
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
pytest orion/self_monitor/tests -> 21 passed
```

## Evals run

```text
Live (11-03): 63/63 attention_loop_outcome rows with features_at_close.via='orion_is_asking' have
salience_at_close = 0; every loop_id starts with memory-confirm-. Juniper's panel verdicts: 0/57 at
zero. Implicit decay: 0/308 at zero.
```

## Docker/build/smoke checks

```text
Not applicable.
```

## Review findings fixed

- One-entry text change; the evidence comes from the source-fix agent's live re-check. No code
  review was needed.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: the act ledger also writes salience 0 when it is unknown (8 of 44 rows), so 0 means
  "unknown" there too.
- Mitigation: the organ does not read that field.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3051

🤖 Generated with [Claude Code](https://claude.com/claude-code)
