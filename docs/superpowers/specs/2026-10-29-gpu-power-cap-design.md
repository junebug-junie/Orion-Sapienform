# Permanent GPU power cap on circe (gpu2 at 150 W)

Status: design, 2026-10-29. Decision owner: Juniper ("I'll take 150w", 2026-10-29).
Evidence: PR #2945 (short bench), PR #2963 (long-context bench under pool holds).

## Arsonist summary

Juniper chose 150 W for gpu2. Today that is a hand-typed `nvidia-smi -pl 150`. It disappears on a
reboot or a driver reload, nothing notices when it goes, and nothing records that it was ever set.

This design makes the cap a checked-in setting. A small root job on circe applies it at boot,
checks it every 5 minutes, puts it back if it drifted, and logs every change. A column in the
GPU telemetry shows the limit next to the power draw, so anyone can see the cap working.

It is deliberately dumb: one number per card, no dynamic control. The body-economy agent's idea of
caps that move with demand (economy ceiling, holder hints) waits until a flat cap has been
measured live for a week. That is the agent's own order: "flat cap first; add the effort-guess
link only if data shows it beats a flat cap".

What the cap buys (long-context bench, both slots, 16k tokens in and 4k out per slot):

| | 250 W | 150 W | change |
|---|---|---|---|
| write speed one turn sees | 29.4 tok/s | 25.2 tok/s | 14% slower |
| card output, both slots | 45.0 tok/s | 38.1 tok/s | 15% slower |
| energy per written token, all-in | 4.89 J | 3.93 J | 20% less |
| card max temperature | 78 °C | 67 °C | 11 °C cooler |

The 250 W bench wrote at the same speed as real gpu2 work (30 tok/s over 72 h of server logs), so
these numbers describe real load.

What it does not buy: cabinet cooling. The economy agent measured circe's link to cabinet heat
as weak (r 0.10–0.22). This is an energy decision, not a heat fix.

## Current architecture

- **Nothing sets GPU power limits** except the bench, `scripts/gpu_power_bench.py:173`, which
  always restores the card's starting limit.
- **circe's lane controller** (`services/orion-gpu-lane-controller/`) cannot do it. It has no GPU
  device and no nvidia-smi. Its only route to the host is the Docker socket (`docker-compose.yml:39`),
  through an allowlist of `docker` subcommands (`app/launch_exec.py:70`). It has no periodic loop;
  it only answers pool actuation requests on the bus. The economy agent named it as the natural
  home, but giving it GPU access would widen the most powerful container on circe for one number.
- **orion-biometrics on circe** is privileged and runs nvidia-smi as root
  (`docker-compose.yml:17`, `orion/sensors/gpu_host_stats.sh:18`). It samples per-card
  `power.draw` about every 31 s but not `power.limit`. It is a sensor, and a sensor should not
  actuate.
- **Root host jobs on circe already exist as systemd units installed from the repo**:
  `orion-llm-port-gate.service`. It is a script copied to `/usr/local/sbin`, a oneshot unit, and a
  runbook (`scripts/ops/circe_llm_port_gate.sh`, `docs/runbooks/2026-10-01-circe-llm-port-firewall.md`).
  It is the only `orion*` unit running on circe today.
- **Card facts on circe** (nvidia-smi, 2026-10-29):

  | card | limit | default | min |
  |---|---|---|---|
  | gpu0, gpu1, gpu2 | 250 W | 250 W | 100 W |
  | gpu3 | 300 W | 300 W | 150 W |

## How the body economy reads this

The cap is not an economy act, by the economy's own rules. Its honesty clause
(`2026-10-23-organism-body-economy-design.md:60-65`) says structural power levers are Juniper's
decisions, and Orion's job is to forecast and propose with numbers. A fixed per-card power limit is
such a lever. The economy is also shadow-only today (Patches A, B, C1 and C3a record and price, and
control nothing), so a root setting on a card must not depend on it.

What the economy must do is read the cap as a body fact:

- **Capacity.** A capped lane writes fewer tokens per second (gpu2 at 150 W: 0.85x). Anything that
  prices or forecasts a lane's slot-seconds (`gpu_slot_s_by_lane` in the planned `ActCostV2`) must
  use the capped rate, not a 250 W rate.
- **Energy per act.** A capped lane spends less energy per token (gpu2 at 150 W: 0.80x). `wh_by_host`
  estimates must use the capped figure.
- **Where to read it.** The checked-in `config/gpu_power_limits.json` gives the intended limit. The
  biometrics `power_limit_watts` column gives the actual limit. When the two disagree, the card is
  in drift or a bench is running, and the economy should treat that lane's cost as unknown, not
  guess.
- **Proposals, not writes.** If Orion's data later says another number is better, Orion proposes a
  config change with the evidence (bench or live J/token). The economy never writes the file. There
  is one writer: a PR that Juniper merges.

## Missing questions

None blocking. Juniper picked the number and the card. Decided here by default, each easy to change:

1. **Only gpu2.** gpu0 is the chat card and is never capped. gpu3 is never capped either: it serves
   Tier 0 holders such as vision. gpu1 (the Q4 agent card) gets a cap only after gpu2's week of data.
   hecate is out of scope, because it is on-demand and on another host.
2. **Re-check every 5 minutes.** A driver reload, a GPU reset, or a person typing `-pl` can change
   the limit. Five minutes bounds how long a wrong limit can last, at the cost of one nvidia-smi
   call per check.
3. **Stopping the unit restores each listed card's default.** That makes rollback one command.

## Proposed schema / API changes

No bus, schema, or service API changes. Three small surfaces:

**1. `config/gpu_power_limits.json`** (new, checked in). JSON so the root job needs only the
Python standard library.

```json
{
  "comment": "Per-card power limits on circe. Applied by orion-gpu-power-limits (docs/superpowers/specs/2026-10-29-gpu-power-cap-design.md). A card not listed is never touched.",
  "host": "circe",
  "cards": {
    "2": {"watts": 150, "why": "Juniper 2026-10-29: 0.80x J/token for 0.86x write speed (PR #2963)"}
  }
}
```

Keys are nvidia-smi indices. They match `cards.<c>.index` in `config/gpu_pool.yaml`.

**2. `scripts/ops/gpu_power_limits.py`** (new, stdlib only). It is installed as
`/usr/local/sbin/orion-gpu-power-limits`, like the port gate. It has three verbs:

- **`apply`**: for each listed card, read (current, default, min, max). Then:
  - If the target is outside [min, max], or the host name is wrong: refuse that entry, log it, and
    leave the card alone.
  - If `/run/gpu-power-bench-<N>.lock` is held (a bench is running): skip the card and log it. The
    bench already takes this flock.
  - If the current limit differs from the target: set it and log
    `gpu2 power limit 250 -> 150 W (drift)`.
  - Exit 0 when every listed card ends at its target or was skipped for a bench. Exit 1 otherwise.
- **`remove`**: restore each listed card's default. Bench lock rules apply as above.
- **`status`**: print one JSON line per listed card, with target, current and default. It never writes.

The config is read from `/mnt/scripts/Orion-Sapienform/config/gpu_power_limits.json` on every run,
so changing a number is a PR plus `git pull` on circe. The code runs from `/usr/local/sbin`, so a
pull never changes what root executes; only a reinstall does. This is the port gate's trust
boundary, kept.

**3. systemd units** in `scripts/ops/`:

- `orion-gpu-power-limits.service`: oneshot. `ExecStart=... apply`, `ExecStop=... remove`,
  `RemainAfterExit=yes`, `After=nvidia-persistenced.service`.
- `orion-gpu-power-limits.timer`: `OnBootSec=1min`, `OnUnitActiveSec=5min`. It triggers a second
  service, `orion-gpu-power-limits-check.service` (`apply` only, no `ExecStop`), so a check never
  undoes the cap.

**4. Telemetry:** add `power.limit` to `orion/sensors/gpu_host_stats.sh` as a new **last** CSV column
`power_limit_watts`, following the convention at `:14-15`. orion-biometrics then carries
`gpus[].power_limit_watts` next to `power_draw_watts`. This is a readback of actuator state, not a
cognition signal, and no model or score consumes it. The metric gate's provenance is
nvidia-smi `power.limit`. It is not independent of the config, by design: it exists to check that
the actuator did its job.

**5. The bench learns about the cap.** `gpu_power_bench.py` refuses to start unless the card is at
its default limit (it treats anything else as "someone else may be capping it"). With a permanent
cap, it must instead accept the configured limit as the expected starting point and restore that.
It reads the same JSON. While the bench runs, the enforcer skips the card (point 2), so the two
never fight.

## Files likely to touch

- `config/gpu_power_limits.json`: new
- `scripts/ops/gpu_power_limits.py`: new
- `scripts/ops/orion-gpu-power-limits.service`, `-check.service`, `.timer`: new
- `docs/runbooks/2026-10-29-circe-gpu-power-limits.md`: new (install, check, rollback)
- `tests/scripts/test_gpu_power_limits.py`: new. Uses a fake nvidia-smi and covers: apply,
  drift, out-of-range refusal, wrong host, bench lock skip, remove, and status writing nothing.
- `orion/sensors/gpu_host_stats.sh`: one column
- `services/orion-biometrics/app/metrics.py`: parse the column (about line 469), plus a test
- `scripts/gpu_power_bench.py` and its test: configured limit as baseline

## Non-goals

- Caps that move with load, effort guesses, or economy prices. Not until a week of flat-cap data
  shows a gap that a dynamic cap would close.
- gpu0, gpu1, gpu3, or hecate.
- Any change to the lane controller, the GPU pool, or the bus.
- Heat control. The cap is chosen for energy per token.
- A UI panel. The journal, `status`, and the biometrics column are enough to start.

## Acceptance checks

1. After install: `nvidia-smi -i 2 --query-gpu=power.limit` reads 150 W, and `journalctl -u
   orion-gpu-power-limits` shows the 250 -> 150 line.
2. After a reboot of circe, gpu2 reads 150 W within 2 minutes of boot.
3. After `sudo nvidia-smi -i 2 -pl 250`, gpu2 is back at 150 W within 5 minutes, and the journal
   logs a drift line.
4. With `/run/gpu-power-bench-2.lock` held, a check skips gpu2 and says so.
5. A config entry of 90 W (below the card's minimum), or one under another host name, is refused
   and the card is untouched. Covered by a test, not run live.
6. `sudo systemctl disable --now orion-gpu-power-limits.service orion-gpu-power-limits.timer`
   puts gpu2 back at 250 W.
7. Biometrics samples on circe show `power_limit_watts=150` for gpu_index 2 and 250 for the others.
8. **One week later (outcome, not mechanism):**
   - From the server logs: gpu2 write speed is about 25 tok/s on long-context requests. The 72 h
     baseline at 250 W was a median of 30 tok/s.
   - From biometrics: gpu2 average draw while busy is at or under 150 W.
   - From pool leases: agent-lane stance p50, compared with the 2026-10-29 numbers (agent-gpu2
     147 s). If it got much worse than the 14% slower writing explains, queueing is amplifying the
     cap, and the number goes back to the table.

## Recommended next patch

One implementation PR with the config, the script, the units, the runbook, tests, and the
biometrics column. Then Juniper runs a one-line install on circe (copy the script to
`/usr/local/sbin`, copy the units, `systemctl enable --now` the service and the timer). Checks 1–7
run that day; check 8 runs on about 2026-11-05.

The bench change (proposed change 5) can ride in the same PR, or follow it. Until it lands, the
bench refuses to run on a capped card, which is safe.

For the body-economy agent: this is the "flat cap first" step from your answer. The config file
is the one place a card's limit is written. If a dynamic ceiling ever earns its keep, it should
change what writes that number, not add a second writer. Until then, see "How the body economy
reads this".

gpu1: it still serves the Q4 27B (one slot). Bonsai replaced the Q4 model only on gpu2 (#2477, around
2026-10-02). The gpu1 swap waits on the Bonsai replay quality check (#2554, `make
eval-bonsai-replay`), which has no recorded run. If gpu1 moves to Bonsai, gpu2's bench numbers
apply to it directly. Juniper's Hub Agent turns run there, so the 14% slower writing would be felt.
