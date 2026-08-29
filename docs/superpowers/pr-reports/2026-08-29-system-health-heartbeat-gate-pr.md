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
python3 scripts/check_system_health_producers.py  -> OK (11 AST-verified sites)
make check-system-health-producers                -> OK
pytest tests/test_system_health_producers.py -q   -> 7 passed
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

## Review findings fixed

8 findings, all fixed.

- **Finding 1 (HIGH) -- my fix would have made things worse.** All producers sleep 30s but
  passed no `heartbeat_interval_sec`, so the payload carried the schema default `10.0`.
  `orion-equilibrium-service` computes `grace = interval * EQUILIBRIUM_GRACE_MULTIPLIER (3.0)`
  = **30.0s** and marks a service `"down"` past it. A 30s period against a 30s grace is zero
  margin: any event-loop delay flips it to `down`, emits a spurious transition and pushes
  `distress_score`. The three services would have gone from silent to actively lying.
  - Verified the math myself at `service.py`'s status check before accepting.
  - Fix: widened well past the reported three. **Seven** producers had this
    (`gpu-cluster-power`, `bus-tap`, `rag`, `vision-edge`, `context-exec`, `whisper-tts`,
    `llamacpp-host`) -- including services publishing successfully today, surviving on
    latency luck. All now declare `30.0`. `vision-frame-router`, `vision-retina` and
    `graph-compression` now thread their real configured period instead of hardcoding.
  - The gate enforces it, so an eighth cannot regress.
- **Finding 2 (MEDIUM)** -- the gate exited 0 when it inspected nothing, and its count was a
  substring count that included `class SystemHealthV1(BaseModel):`.
  - Fix: count AST-verified calls (now correctly reports **11**, not 12) and fail below
    `MIN_EXPECTED_SITES`. `SKIP_PARTS` now matches paths relative to the repo root, so a
    checkout under a directory named `tests` no longer skips the entire repo.
  - Evidence: pointing `SEARCH_ROOTS` at a nonexistent dir now exits 1.
- **Finding 3 (MEDIUM)** -- aliased imports were invisible. Review's probe
  (`import SystemHealthV1 as SH`, then `SH(...)` missing both required fields) produced zero
  problems.
  - Fix: resolve aliases from the module's `Import`/`ImportFrom` nodes.
  - Evidence: re-running review's exact probe now reports both problems.
  - I did **not** flag `model_validate` as review suggested: it fully validates, and it is how
    the legitimate consumer reads heartbeats off the bus
    (`orion-equilibrium-service/app/service.py:1198`). Flagging it was a false positive in my
    first cut. Only `model_construct`, which genuinely skips validation, is flagged.
- **Finding 4 (MEDIUM)** -- the gate was not in `.github/workflows/orion-static-gates.yml`,
  the only workflow that runs gates, so nothing executed it unless a human typed the target.
  That is the exact "nudge you can skip" failure this patch is written against.
  - Fix: added as a step. It meets every criterion in that workflow's own header
    (stdlib-only, no live infra, green on main).
- **Finding 5** -- the anti-drift test grepped the gate's source text, which a docstring
  mention would satisfy.
  - Fix: imports the gate module and compares `set(REQUIRED_KWARGS)` to the model's required
    set **both directions**, so a stale entry is caught too.
- **Finding 6** -- the parametrization ran three byte-identical validations and the payload
  was hand-written, so it could not catch a producer drifting.
  - Fix: replaced with a test that reads the declared interval and the real `sleep()` out of
    each producer and asserts they agree. Mutation-verified: setting rag's interval back to
    `10.0` fails with `declares 10.0s but sleeps 30s`.
- **Finding 7** -- `pytest.raises(Exception)` around a call missing two fields.
  - Fix: `pytest.raises(ValidationError)`, omitting only `boot_id`.
- **Finding 8** -- target missing from `.PHONY`. Added.

## Risks / concerns

- Severity: low. Purely additive for the three that published nothing. For the four that
  already publish, the only change is a corrected `heartbeat_interval_sec`, which widens
  equilibrium's grace from 30s to 90s -- strictly fewer false `down` classifications.
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
