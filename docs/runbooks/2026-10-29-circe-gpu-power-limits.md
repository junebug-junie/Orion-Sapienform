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

## Install (once, right after the PR is merged and circe has pulled main)

Install soon after the pull: from the moment circe has the config, the power bench expects gpu2 at
150 W and refuses to run while it is still at 250 W.

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

To un-cap a card, set its entry to `"watts": "default"`; the next check puts it back at its default
and keeps it there. Deleting the entry instead stops managing the card and leaves whatever limit it
had.

## Rollback

```bash
ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/gpu_power_limits.py uninstall
```

Uninstall turns the checks off, resets every listed card to its default (gpu2 to 250 W), and only
then deletes the installed copies. If a card cannot be reset (for example a power bench is still
running on it after 2 minutes), it says so and leaves everything installed; run it again later.

To pause without uninstalling (stopping the service resets the cards and also stops the timer):

```bash
ssh -t circe@circe sudo systemctl stop orion-gpu-power-limits.service
```

That lasts until the next boot or `systemctl start`. To keep it off across reboots, use `disable --now`
on both the service and the timer.

## Running the power bench on a capped card

`scripts/gpu_power_bench.py` now treats the configured limit as the card's normal starting point,
and it restores that limit when it ends. While the bench runs it holds a per-card lock in `/run`,
and the 5-minute check skips that card. If a check is in the middle of setting the limit when the
bench starts, the bench refuses with "orion-gpu-power-limits mid-change"; run it again.
