## Summary

- Retro's Zelda reader treated any nonzero RAM `$506` as "item held up, Link frozen". Now only 1..127 counts.
- A level-9 save (2fb75d76, 11-11) holds `$506 = 242` permanently while Link walks, so Retro never offered Orion a decision on it.
- Tests pin the 127/128 boundary and the live 242 value; README updated.

## Outcome moved

Orion's solo Zelda play (#3774). After #3799 marked the frozen 11-16 save, the next run (11-25 01:27Z) loaded 2fb75d76 and again got 0 offers in 60 s. With this fix that save reads controllable.

## Current architecture

`services/orion-retro/app/zelda/observe.py` decodes RAM; `controllable = mode in MAP_MODES and not frozen`. Navigator offers decisions only when controllable.

## Architecture touched

Retro Zelda reader only. No contract, bus, schema or env change.

## Files changed

- `services/orion-retro/app/zelda/observe.py`: lift freeze only for `$506` 1..127.
- `services/orion-retro/tests/test_zelda.py`: boundary + live-value asserts.
- `services/orion-retro/README.md`: controllable rule text.

## Schema / bus / API changes

- Behavior changed: `link.controllable` true when `$506 >= 128` in map modes.

## Env/config changes

None.

## Tests run

```text
services/orion-retro: pytest tests -q -> 190 passed
```

Live evidence (scratch CoreProbe in orion-athena-retro, save 2fb75d76): `$506` = 242 for 3,600 idle frames; 60 frames DOWN moved y 93->133, RIGHT moved x 48->88; old reader said controllable=False.

## Evals run

No eval harness change; reader is covered by deterministic tests.

## Docker/build/smoke checks

Not run; pure Python change picked up by the normal retro rebuild.

## Review findings fixed

- Finding: no boundary test at 127/128.
  - Fix: added asserts for 0x7F (frozen) and 0x80 (controllable).
  - Evidence: 190 passed.
- Finding: README still said any `$506` lift.
  - Fix: README states 1..127 and the live 242 case.
- Finding (not fixed, ticketed): triforce/heart-container lifts never probed; if they read >= 128, Link would read controllable mid-lift. #3887 (verify-live).

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
```

After deploy: remove 2fb75d76 from Redis `orion:retro:leisure:unplayable_saves` (it was marked by this bug).

## Risks / concerns

- Severity: medium (conditional). Concern: an unprobed lift kind above 127 would read controllable. Mitigation: #3887.
- Issues opened: #3887. Related: #3774 (stays open until a run makes moves).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3889

🤖 Generated with [Claude Code](https://claude.com/claude-code)
