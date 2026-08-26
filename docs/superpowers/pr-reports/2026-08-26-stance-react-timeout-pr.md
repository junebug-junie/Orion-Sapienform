# Triple the stance_react timeout; leave the verb on the lane it was designed for

## Summary

- `STANCE_REACT_TIMEOUT_SEC` 120 → **360**, in all four places the value lives.
- `stance_react` **stays on the `chat` lane** (Qwen3.6-35B-A3B, circe-worker-1).
  That placement is deliberate and is not being second-guessed here: a stance is
  a judgement about whether to engage at all, and it is worth the good model.
- Measured cause, not inferred: a stance that completed **correctly** at ~122s
  was discarded at 120.006s.

## Outcome moved

An under-budgeted stance does not fail closed. It surfaces to Hub as
`turn_deferred` — which reads as *"Orion judged the moment wrong"* and is
indistinguishable from it at that layer. So the failure mode this removes is not
"a turn was slow", it is **a real infrastructure timeout wearing the costume of
a considered decision**.

## The evidence

First live curiosity run, 2026-08-26:

```text
06:33:17.823  thought -> cortex-exec   stance_react        (timeout_sec=120)
06:34:47.499  llm-gateway REPLIED to cortex-exec           (~90s, the model leg)
06:35:17.824  thought TIMED OUT                            elapsed_ms=120006.3
06:35:19.743  cortex-exec published the finished result    (~122s)  <-- 1.9s late
```

The answer was real and complete — `provider_finish_reason=stop`, 1,268 chars of
stance, 15,207 chars of reasoning. Nothing failed except the clock.

Why that call was slow, from the same logs:

| | completion tokens |
|---|---|
| stance completions, 24h median | **101** |
| stance completions, 24h min | 60 |
| this call | **4,708** |

The input was an ~11 KB self-addressed curiosity prompt — roughly 47× the median
stance input — and `chat` is a thinking model, so it reasoned in proportion.
120s left no headroom for that. The 90s model leg plus 32s of cortex-exec
post-processing is what crossed the line.

## Why 360, and why not more

Chosen against the chain rather than picked round:

```text
Hub -> thought                TIMEOUT_SEC=400     (services/orion-hub/.env_example)
thought -> cortex-exec        120 -> 360          (this patch)
cortex-exec -> llm-gateway    120                 (untouched -- see below)
```

360 must stay **under** Hub's own 400s outer wait, or Hub gives up first and the
extra budget is unreachable. 40s of margin is left for the reply hop.

**The next ceiling is named and deliberately not raised.** cortex-exec's own RPC
to `LLMGatewayService` is capped at 120s, and the model leg above measured 90s —
30s of headroom. If a stance prompt ever pushes the model itself past 120s, that
inner cap binds first and this key cannot help. That should be raised when it
actually fires, with its own measurement, not pre-emptively on a guess.

## Files changed

- `services/orion-thought/.env_example`: 360, with the incident recorded inline.
- `services/orion-thought/.env` (local, gitignored): synced by hand to 360.
- `services/orion-thought/docker-compose.yml`: fallback `:-120` → `:-360`, so an
  absent key does not silently reinstate the old value.
- `services/orion-thought/app/settings.py`: `Field` default 120.0 → 360.0, with
  the reason a *shorter* default is not the conservative choice here.

All four, because a value that lives in four places and is only changed in one
is how `ORION_ATTENTION_TOPDOWN_ENABLED` ended up absent from a running
container while every config surface said it was set.

## Blast radius

`stance_react_timeout_sec` has three consumers, all in `orion-thought`:
`bus_listener.py` (the stance path this incident is about) and `reverie.py` ×2.
Reverie's stance calls get the same 360s. Same verb, same lane, same reasoning —
stated rather than discovered later.

This raises a **ceiling**, not a duration: a stance that answers in 1.5s still
answers in 1.5s. Observed thought-RPC latencies over the period were
min 874ms / median 1,592ms / max 3,652ms. The only behaviour that changes is
what happens to calls that would previously have been thrown away.

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- Changed: `STANCE_REACT_TIMEOUT_SEC` 120 → 360.
- `.env_example` updated: yes. Local `.env` synced by hand and verified by
  key-set diff against `.env_example`.
- Pre-existing parity gap found and **not** fixed here (out of scope, flagged):
  `ORION_THOUGHT_MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC` exists in the live `.env`
  but not in `.env_example`.

## Tests run

```text
pytest services/orion-thought/tests -q
-> 3 failed, 234 passed

Same 3 failures on clean origin/main with the same env:
-> 3 failed, 234 passed
```

Pre-existing and unrelated — `test_settings_mind_enrichment`,
`test_settings_salience_flags`, `test_reverie_spontaneous_thought` assert
"defaults off" and see the operator's live `.env` through the test's env. Not
introduced by this patch; not fixed by it either.

No new test: this patch changes a single numeric default. A test asserting
`stance_react_timeout_sec == 360.0` would restate the constant rather than pin a
behaviour, which is the keyword-cathedral shape in test form.

## Evals run

```text
None. This is a timeout ceiling; there is no quality surface an eval measures.
```

## Docker/build/smoke checks

No image change (env + one default). Requires a restart of `orion-thought` to
take effect — see below.

## Review findings fixed

Self-review, one finding worth recording:

- **Finding:** the first pass changed only `.env_example` and the live `.env`,
  leaving `docker-compose.yml`'s `:-120` fallback and the pydantic `Field(120.0)`
  default in place. Both are silent reinstatements of the old value the moment
  the key goes missing — the exact shape of the `ORION_ATTENTION_TOPDOWN_ENABLED`
  incident, where every config surface said a flag was set and the running
  container did not have it at all.
  - **Fix:** all four surfaces carry 360.
  - **Evidence:** `grep -n STANCE_REACT_TIMEOUT_SEC` across all four.

## Restart required

```bash
cd <a worktree synced to main>
scripts/safe_docker_build.sh orion-thought up -d --force-recreate --no-build thought
```

`--force-recreate` specifically: a plain `up -d` does not re-read `env_file` on
an env-value-only change.

## Risks / concerns

- **Severity: low — a genuinely hung stance now blocks a turn for up to 6
  minutes instead of 2.** Acceptable for the unsolicited loops this mostly
  affects (nobody is waiting), and still bounded by Hub's own 400s. For an
  interactive chat turn a 6-minute stance would be a bad experience — but that
  case was already broken at 120s, it just failed faster and lied about why.
- **Severity: low — the inner 120s cortex-exec→gateway cap is untouched** and is
  now the binding constraint, at 30s above the measured model leg.

## PR link
