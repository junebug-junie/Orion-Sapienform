## Summary

- Orion's cabinet sound line no longer shows dBFS numbers. It now says only whether the cabinet is louder than usual, quieter than usual, or about as loud as usual, compared with its own last 24 hours, and that the mic can't give decibels.
- The line is left out entirely when there is no 24-hour history, because a bare level means nothing.
- Fixes the restart command in the #2569 write-up. The old one is refused when run from the main checkout.

## Outcome moved

After #2569, Orion told Juniper "-16 dBFS is what I hear." dBFS means how close the sound is to this mic's maximum. The mic is uncalibrated, so that number isn't decibels. Putting it first in the line invited exactly that misreading. Real decibels will come from a calibrated sensor (PCB Artists I2C SPL module, planned).

## Files changed

- `orion/situational/context.py`: the render block now carries only the comparison. The `sound_*` dBFS fields stay on `CabinetContextV1` for debugging.
- `orion/situational/tests/test_situation_cabinet_sound.py`: asserts that no digits and no "dB" appear in the line, and that there is no line without history.
- `docs/superpowers/pr-reports/2026-10-09-cabinet-sound-situation-pr.md`: corrected restart command.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
orion/situational/tests  154 passed
```

## Evals run

None for the prompt line. After deploy, ask Orion how their cabinet sounds and check the reply.

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
