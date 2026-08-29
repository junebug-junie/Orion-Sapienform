# Close the power-intent loop: four bugs, none of which had ever executed

## Summary

- `power_intent_settled` held **0 rows since the table was created**. Four separate defects sat between the producer and the table, and every one of them was in code that had merged but never once run.
- Recovers two dependency commits (`sentencepiece`, `protobuf`) pushed to `feat/diffusion-flux-schnell-fp16` **after PR #1926 merged** — main configured FLUX but could not load it.
- Fixes `bus.publish()` called with three positional arguments where the API takes two; the `TypeError` was swallowed by a broad `except` into a WARNING.
- Fixes `NameError: name 'bus' is not defined` in the settler, which discarded a fully-computed settlement inside a fire-and-forget task.
- Turns on `ORION_BUS_ENABLED` for orion-diffusion-host, where `publish()` early-returns silently when disabled, and adds a startup guard so that combination can never be quiet again.
- **Live-verified end to end**: first row ever written to `power_intent_settled`.

## Outcome moved

Orion can now declare an expected power draw before doing work, have the hardware measured against that declaration, and have the result persisted. Before this patch the loop was fully wired on paper, logged success on both sides, and delivered nothing.

First settlement, live on Circe 2026-08-29 21:13:07Z:

```
intent_id     7f621f19-6712-415c-b3e1-e3fe8a403f91
workload_kind reverie_diffusion    node circe    gpu_index 2
outcome       settled              sample_count  18
peak_watts    144.46               baseline      42.1
mean_watts    53.38                energy        1074.379 J
```

102 W above baseline, sampled at 0.894 Hz across the declared window. Measured, not estimated.

## Current architecture (before this patch)

```
orion-diffusion-host  --(orion:power:intent)-->  orion-biometrics  --(orion:power:intent:settled)-->  orion-sql-writer  -->  Postgres
```

Every hop existed in code. Channels were in `orion/bus/channels.yaml`, both schemas in `orion/schemas/registry.py`, the settler subscribed and healthy, the persister listening. Nothing had ever traversed it.

## Architecture touched

No contract changed. Four defects on the existing seam, plus one config flag and one guard.

### 1. FLUX could not load (dependency, never merged)

PR #1926 swapped sdxl-turbo → FLUX.1-schnell and merged. Two fixes found by a live GPU smoke on Circe were pushed to that branch **afterwards** and never came home:

```
f3fc9f66e  fix(diffusion): add missing sentencepiece dependency
438ef2bd4  fix(diffusion): add missing protobuf dependency
```

`settings.py` on main pointed at FLUX; `requirements.txt` on main lacked what FLUX needs. Circe had been running an image built from that branch's leftover worktree, so the box worked while main did not. Rebuilding from main reverted it:

```
Cannot instantiate this tokenizer from a slow version. If it's based on
sentencepiece, make sure you have sentencepiece installed.
... diffusion model load permanently failed after 3 attempts
```

The branch could not simply be deployed — it is 118 commits behind main and contains no power-intent producer. Cherry-picked instead.

**The tell:** main's own design doc already stated both dependencies were "found and fixed live." The prose describing the fix merged. The fix did not. Any review that reads documentation as evidence passes this.

### 2. Wrong publish arity (swallowed)

```python
await bus.publish("orion:power:intent", "power.intent.v1", intent)   # 3 args
async def publish(self, channel: str, msg: BaseEnvelope | Dict) -> None:   # takes 2
```

`TypeError` on every generation, caught by the function's own broad `except Exception` and logged as a WARNING. Generation kept working; the line read like an ordinary bus hiccup. The schema name belongs *inside* the envelope. Now uses `BaseEnvelope(kind=, source=, payload=)` — the same shape `orion-biometrics._publish()` already used on the settled channel.

### 3. Bus disabled (silent by design)

```python
async def publish(self, channel, msg):
    if not self.enabled:
        return          # no raise, no log
```

`ORION_BUS_ENABLED=false` for orion-diffusion-host. The producer logged `power_intent_declared` on every generation while nothing reached the wire. Confirmed by subscribing directly: 0 messages on the power channels, against a control of 239 `orion:system:health` messages in 25s on the same connection.

That service had never been on the bus at all — its `SystemHealthV1` heartbeat had never published either.

### 4. Settlement computed, then thrown away

```python
NameError: name 'bus' is not defined     # services/orion-biometrics/app/main.py:597
```

The publish referenced a bare `bus` that existed in **no enclosing scope**. `settle()` ran its full window against the real GPU, logged `power_intent_settled ... outcome=settled samples=18 peak=48.8 baseline=42.19`, and *then* raised — inside a task nobody awaited, so it surfaced only as `Task exception was never retrieved`.

Root cause is structural, not a typo: the handler was a closure inside `lifespan`, so nothing could import it and no test could reach it. `test_power_intent_settlement.py` has nine tests covering `settle()`/`summarize()` and stayed green through the entire outage — it tests the arithmetic, never the delivery.

Extracted to `make_power_intent_handler(get_bus)` at module scope. `get_bus` is a callable, not a bus, because the `Hunter` that owns the connection is constructed after the handler is built.

## Files changed

- `services/orion-diffusion-host/requirements.txt`: `sentencepiece==0.2.2`, `protobuf==7.36.0`
- `services/orion-diffusion-host/app/main.py`: envelope-wrap the intent; contradictory-config startup guard
- `services/orion-diffusion-host/.env_example`: `ORION_BUS_ENABLED=true` with the reason
- `services/orion-diffusion-host/tests/test_power_intent_publish.py`: **new**
- `services/orion-biometrics/app/main.py`: extract handler to module scope; loud publish failure
- `services/orion-biometrics/tests/test_power_intent_handler_wiring.py`: **new**

## Schema / bus / API changes

- Added / Removed / Renamed: none
- Behavior changed: `orion:power:intent` now actually carries traffic. Payload shape unchanged (`PowerIntentV1` inside `orion.envelope`), matching what the settler already validated.
- Compatibility: none required — nothing was ever published on this channel before.

## Env/config changes

- Added keys: none
- Changed default: `ORION_BUS_ENABLED` false → **true** for orion-diffusion-host
- `.env_example` updated: yes
- Local `.env` synced: **by hand, deliberately.** `scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary checkout*, not the worktree, so it could not see this change and silently reported nothing. Athena's `.env` and Circe's `.env` were both edited directly and verified still gitignored.
- Skipped keys requiring operator action: none

`settings.py` still defaults `ORION_BUS_ENABLED` to `False`, which is why the startup guard exists: a deploy that omits the key lands back in the silent state, and now says so at ERROR.

## Tests run

```text
services/orion-diffusion-host   28 passed
services/orion-biometrics      141 passed, 2 failed

The 2 failures (test_circe_expected_offline and
test_circe_node_availability_reflects_expected_offline) fail IDENTICALLY on
clean main -- baselined, not introduced here. They do reveal that the node
catalog still marks circe expected_offline while it runs live GPU workloads.
```

Every fix mutation-tested — reverted, confirmed exactly the intended test fails, restored:

```text
publish arity   reverted -> 1 failed (test_power_intent_is_published_as_an_envelope), 3 passed
settler NameError reverted -> 2 failed, 1 passed
config guard    removed  -> 1 failed (test_contradictory_config_is_loud), 4 passed
all restored -> 28 / 141 green
```

The arity test pins the fake bus against `inspect.signature(OrionBusAsync.publish)`. A permissive `*args` double would have passed against the broken code — that is the hole that let this ship.

The wiring test's `_drain()` **gathers** the detached task rather than sleeping, so an exception inside it fails the test instead of vanishing the same way it did in production.

## Evals run

```text
None. Neither service has an evals/ harness. Not created here -- these are
correctness fixes with deterministic pass/fail, which is what tests are for.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-biometrics    up -d --build   exit 0
scripts/safe_docker_build.sh orion-diffusion-host up -d --build  exit 0

GET /ready -> {"ready":true,"model_loaded":true,"load_error":null}
POST /generate -> HTTP 200, 41.2s, real PNG bytes

redis psubscribe orion:power:intent, orion:power:intent:settled
  -> 2 pmessages (intent AND settled)
  control: orion:system:health -> 239 messages / 25s

psql conjourney -> power_intent_settled: 0 rows before, 1 row after
```

Built from worktrees on both hosts. `safe_docker_build.sh` correctly refuses Circe's primary checkout; the escape hatch was **not** used.

## Risks / concerns

- **Severity: low.** `protobuf==7.36.0` is inherited from the original live-verified commit, not independently re-pinned. It loaded FLUX successfully here.
- **Severity: low.** Turning the bus on for diffusion-host also starts its heartbeat. Additive, and every comparable GPU-host service already does this.
- **Severity: medium, pre-existing, not addressed.** The repo has **no linter and no lint config**. Three of the four defects here (`NameError`, wrong arity) are exactly what `pyflakes` catches for free. That is the systemic fix; it is a repo-wide change and is deliberately not bundled into this patch.
- **Severity: low, pre-existing.** The node catalog marks `circe` `expected_offline` while it runs live workloads, which is what those 2 baselined failures assert.

## Restart required

Already applied on Circe. To reproduce elsewhere:

```bash
cd <worktree>
scripts/safe_docker_build.sh orion-diffusion-host up -d --build
scripts/safe_docker_build.sh orion-biometrics     up -d --build
```

## PR link

<filled in on open>
