# PR: three services have never published a heartbeat, and a gate so a fourth can't

## Summary

- Fixes `SystemHealthV1` construction in **three** services that omit its required
  `boot_id`/`last_seen_ts`: `orion-gpu-cluster-power`, `orion-bus-tap`, `orion-rag`.
- Adds `scripts/check_system_health_producers.py` -- an AST gate over every
  `SystemHealthV1(...)` call site in the repo -- plus `make check-system-health-producers`
  and a pytest wrapper.
- Found by deploying PR #1937 and checking the result instead of assuming it.

## Outcome moved

`orion-gpu-cluster-power` was failing its heartbeat every 30s tick, indefinitely, and
nothing downstream noticed. PR #1937 fixed the *first* error
(`'Settings' object has no attribute 'service_version'`) and uncovered a second underneath:

```
Heartbeat failed: 2 validation errors for SystemHealthV1
boot_id        Field required
last_seen_ts   Field required
```

So that service has **never** published a heartbeat. Auditing every producer found
`orion-bus-tap` and `orion-rag` with the identical defect (0 references to `boot_id`), and
`services/orion-whisper-tts/app/main.py` carrying a hand-written
`# FIX: Added boot_id and last_seen_ts to satisfy SystemHealthV1 schema` -- someone hit this
before and fixed exactly one service. That is a bug class, not three bugs.

## Current architecture

Every producer builds the payload inside a heartbeat loop shaped like:

```python
while True:
    try:
        payload = SystemHealthV1(...).model_dump(mode="json")
        await bus.publish("orion:system:health", ...)
    except Exception as e:
        logger.warning(f"Heartbeat failed: {e}")
    await asyncio.sleep(30)
```

A missing required field raises *inside* that `except`. The container stays `Up`, its
`/health` endpoint stays 200, `docker ps` looks perfect -- and it publishes nothing. This is
the same failure shape as the rest of this incident arc: **the system cannot tell "absent"
from "fine."**

## Files changed

- `services/orion-gpu-cluster-power/app/api.py`, `services/orion-bus-tap/app/main.py`,
  `services/orion-rag/app/main.py`: pass `boot_id` + `last_seen_ts`; add module-level
  `BOOT_ID = str(uuid.uuid4())` following whisper-tts's existing convention, so a consumer
  can distinguish a restart from continuous uptime
- `scripts/check_system_health_producers.py`: new gate
- `Makefile`: `check-system-health-producers` target
- `tests/test_system_health_producers.py`: 6 tests (new)

## Schema / bus / API changes

None. `SystemHealthV1` is unchanged; these producers now satisfy the contract it already had.
Payload gains `boot_id`/`last_seen_ts` **from services that were publishing nothing at all**,
so no consumer sees a changed message -- it sees a message for the first time.

## Env/config changes

None.

## Tests run

```text
python3 scripts/check_system_health_producers.py  -> OK (12 construction sites checked)
make check-system-health-producers                -> OK
pytest tests/test_system_health_producers.py -q   -> 6 passed
```

**Mutation-tested against the real files, not a synthetic fixture** (this repo has been
burned by a gate that passed a fixture while being inert on real code): removing `boot_id`
from each of the three services in turn makes the gate fail and name that exact file and
line. All three verified individually, then restored.

`orion-rag`, `orion-bus-tap` and `orion-gpu-cluster-power` have **no test directories at
all** -- which is why this survived. The gate is repo-wide precisely so it does not depend on
those services growing suites.

## Docker/build/smoke checks

```text
Not run -- no Dockerfile, compose, dependency or port changed.
Live evidence instead: docker logs orion-athena-gpu-cluster-power showed the
post-#1937 validation error once per 30s tick, which is what prompted this patch.
```

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-cluster-power up -d --build
# bus-tap and rag are not currently running; they pick the fix up on next start.
```

Verify with: `docker logs --since 5m orion-athena-gpu-cluster-power | grep -c "Heartbeat failed"`
-> expect **0**, and confirm a message actually lands on `orion:system:health`.

## Risks / concerns

- Severity: low. Purely additive to the payload; the three services were publishing nothing.
- Severity: low. `BOOT_ID` is generated at import time, so it changes on reload as well as
  restart. Same as whisper-tts's existing convention; consistency chosen over a private
  variant.
- Severity: low, NOT fixed. **13 Makefile targets use bare `python`, which does not exist on
  this host** -- verified: `make check-inner-state-registry` exits 127 without running. Those
  gates are inert here. My target uses `python3` so it actually runs; fixing the other 13 is
  out of scope but worth its own patch, since a gate that cannot run is the exact failure
  this PR is about.
- Severity: low, NOT fixed. `orion:system:health` has exactly **1** live subscriber. Nothing
  in this patch verifies anyone acts on a heartbeat's *absence* -- that is the same
  absence-blindness PR #1935 phase 2 addresses.
