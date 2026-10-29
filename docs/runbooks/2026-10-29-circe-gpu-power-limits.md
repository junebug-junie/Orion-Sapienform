# circe GPU power limits (gpu2 at 150 W)

**[GO, Juniper, sudo]** Every command below that starts with `sudo`, or uses `ssh -t ... sudo`, is for
Juniper to run. Agents may not run sudo (CLAUDE.md section 8).

## What it does, plainly

gpu2 runs at 150 W instead of its default 250 W. That uses 20% less energy per word Orion writes
there, and writing is 14% slower (PR #2963). A small job on circe keeps the limit there:

- It sets the limit when circe boots.
- Every 5 minutes it checks again. If anything moved the limit (a driver reload, a GPU reset,
  someone typing `nvidia-smi -pl`), it puts it back and logs a `drift` line.
- It leaves a card alone while the power bench is running on it.
- It never touches a card that is not listed in `config/gpu_power_limits.json`. Today only gpu2 is
  listed.

Design: `docs/superpowers/specs/2026-10-29-gpu-power-cap-design.md`.

## Install (once, after the PR is merged and circe has pulled main)

```bash
ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/gpu_power_limits.py install
```

This copies the script to `/usr/local/sbin/orion-gpu-power-limits` and the three units to
`/etc/systemd/system/`. It enables the service and the timer, applies the limit, and prints the
status. Root runs the installed copy, so a later `git pull` changes the numbers it reads but never
the code it runs. Re-run `install` after a change to the script or the units.

## Check

These need no sudo:

```bash
ssh circe@circe nvidia-smi -i 2 --query-gpu=power.limit,power.draw --format=csv,noheader
ssh circe@circe /usr/local/sbin/orion-gpu-power-limits status
ssh circe@circe journalctl -u orion-gpu-power-limits -u orion-gpu-power-limits-check --since today --no-pager
ssh circe@circe systemctl list-timers orion-gpu-power-limits.timer --no-pager
```

Expected:
- `power.limit` reads `150.00 W`.
- `status` shows `"at_target": true`.
- The journal shows `gpu2: power limit 250 -> 150 W (apply)` once.

Biometrics samples on circe also carry `power_limit_watts` per card, in `/mnt/telemetry/gpu_stats/*.csv`
and in `gpus[]` of the biometrics sample.

## Change the number, or add a card

1. Edit `config/gpu_power_limits.json` in a PR. Keys are nvidia-smi indices. A target outside the
   card's min-max range is refused.
2. After the merge, pull on circe: `ssh circe@circe git -C /mnt/scripts/Orion-Sapienform pull --ff-only`.
3. The next 5-minute check applies it. To apply at once instead:
   `ssh -t circe@circe sudo systemctl restart orion-gpu-power-limits.service`

Removing a card from the file stops the job from managing that card, but it does not reset it. Run
`sudo nvidia-smi -i <N> -pl <default>` once, or stop the service before the edit (rollback below
resets every listed card).

## Rollback

```bash
ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/gpu_power_limits.py uninstall
```

Stopping the service runs `remove`, so every listed card goes back to its default limit (gpu2 to
250 W). Then the installed copies are deleted. To pause without uninstalling:

```bash
ssh -t circe@circe sudo systemctl disable --now orion-gpu-power-limits.timer orion-gpu-power-limits.service
```

## Running the power bench on a capped card

`scripts/gpu_power_bench.py` now treats the configured limit as the card's normal starting point,
and it restores that limit when it ends. While the bench runs it holds a per-card lock in `/run`,
and the 5-minute check skips that card. If a check is in the middle of setting the limit when the
bench starts, the bench refuses with "orion-gpu-power-limits mid-change"; run it again.
