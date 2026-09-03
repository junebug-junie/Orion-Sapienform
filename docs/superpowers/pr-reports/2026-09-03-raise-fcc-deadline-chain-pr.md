# PR report: raise the FCC deadline chain for curiosity's slower lane

## Summary

- PR #2067 moved Orion's curiosity investigation loop from `harness`/`chat`
  (35B MoE, ~944 tok/s prompt processing) to `agent` (27B dense, ~491 tok/s) --
  measured ~1.9x slower. That PR's own risk section flagged the consequence:
  applied to the six most recent completed investigations, **3 of 6** would
  exceed the 1600s FCC ceiling, and the failure is silent (INFO log, no
  journal entry, daily slot still spent).
- This raises the whole nested deadline chain to absorb that slowdown:
  `HARNESS_FCC_TIMEOUT_SEC` 1600->2400, `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC`
  2160->2960, `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` 2700->3500. Derived
  from the chain, not picked: `2400 + finalize(485) = 2885`, 2960 leaves 75s
  margin (matching the old value's own 75s margin); `400(stance) + 2960 =
  3360`, 3500 leaves ~140s margin (matching the old value's own ~140s margin).
- `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC` (3600, hard ceiling) is
  **deliberately left unchanged**. Read `harness_governor_client.py` directly:
  the extension loop computes `remaining = max_wait_sec - elapsed` and clamps
  every wait to `min(poll_sec, remaining)`, so it cannot overshoot regardless
  of `poll_sec`'s size -- raising `RPC_TIMEOUT_SEC` to 2960 is safe against it
  by construction. The margin between the two shrank from 1440s to 640s; that
  is accepted, not overlooked -- see "Risks" for why.
- Every literal restatement of `HARNESS_FCC_TIMEOUT_SEC` updated in lockstep:
  the `.env_example` owner (gated by `check_env_key_single_source.py`), the
  governor's `settings.py` Field default, the compose shell default, and
  prose comments in `orion/harness/runner.py`. Same for both Hub `settings.py`
  Field defaults, one adjacent pre-existing stale comment (said `960s`,
  predates this patch, fixed opportunistically), and a stale doc table in
  `orion/curiosity/README.md` that had already drifted to `1500` before this
  patch touched it.
- **A real mistake made and reversed in this session, disclosed in full:**
  syncing these three values into the live `.env` with
  `sync_local_env_from_example.py --force` also force-flattened five
  unrelated diverged keys back to `.env_example` placeholders, including a
  live credential (`HUB_AITOWN_ADMIN_KEY`). Caught immediately from the
  sync tool's own output, all five restored via `Read`+`Edit` on the live
  files in the same turn, and verified live afterward -- see "Env/config
  changes" for the full list and final state.

## Outcome moved

Curiosity's move to the slower lane (PR #2067) no longer trades completion
rate for GPU isolation. Before this patch, roughly half of long investigations
on the new lane would have timed out silently. The new ceiling (2400s FCC)
covers all six of the most recently measured runs at the 1.9x slowdown with
margin (worst case 1176s * 1.9 = 2234s, well under 2400s).

## Current architecture (before this patch)

```
FCC motor (HARNESS_FCC_TIMEOUT_SEC=1600, governor process)
  -> finalize chain (substrate 5 + reflect 180 + voice 300 = 485s)
     -> Hub RPC wait (HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC=2160, soft;
        HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC=3600, hard, liveness-extendable)
        -> stance leg (TIMEOUT_SEC=400)
           -> curiosity's own outer wait (HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC=2700)

Sized for the 35B MoE `harness`/`chat` lane. PR #2067 moved curiosity's turns
to the ~1.9x slower `agent` lane without touching any of these four numbers.
```

## Architecture touched

Config and comments only. No code paths, schemas, or bus contracts changed.

- `services/orion-harness-governor/.env_example` -- the owner of
  `HARNESS_FCC_TIMEOUT_SEC` (single-source gated).
- `services/orion-harness-governor/app/settings.py`,
  `docker-compose.yml` -- the two code-level copies the gate requires to
  match the owner.
- `services/orion-hub/.env_example`, `app/settings.py` -- the two Hub-side
  keys in the same chain (not single-source gated, kept manually consistent).
- `orion/harness/runner.py`, `orion/schemas/harness_finalize.py`,
  `services/orion-hub/scripts/curiosity_investigation.py`,
  `orion/curiosity/README.md` -- prose that states these numbers as current
  fact, updated so nothing in the diff contradicts the new values.

## Files changed

- `services/orion-harness-governor/.env_example`: `HARNESS_FCC_TIMEOUT_SEC`
  1600->2400; comment rewritten with the new derivation and history.
- `services/orion-harness-governor/app/settings.py`: `fcc_timeout_sec` Field
  default 1600.0->2400.0; comment updated (was still citing 1600 verbatim,
  caught by review).
- `services/orion-harness-governor/docker-compose.yml`: shell default
  `${HARNESS_FCC_TIMEOUT_SEC:-1600}` -> `${HARNESS_FCC_TIMEOUT_SEC:-2400}`.
- `services/orion-hub/.env_example`: `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC`
  2160->2960, `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` 2700->3500.0; both
  comments rewritten with the new derivation.
- `services/orion-hub/app/settings.py`: both Field defaults raised to match;
  a stale, pre-existing `960s` comment beside
  `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` corrected to reference the field
  rather than restate a literal.
- `orion/harness/runner.py`: two prose comments citing `1600s` -> `2400s`.
- `orion/schemas/harness_finalize.py`: one prose comment citing `1600s` ->
  `2400s` (caught by review -- missed on the first pass despite an
  identically-worded twin comment in `runner.py` being fixed).
- `services/orion-hub/scripts/curiosity_investigation.py`: the "three nested
  deadlines" comment block and two adjacent literal citations updated.
- `orion/curiosity/README.md`: config table row for
  `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` corrected from a stale `1500`
  (real value was already `2700` before this patch) to the current `3500`.

## Schema / bus / API changes

None.

## Env/config changes

- Changed keys (live default, `.env_example`, and both code copies, all in
  lockstep): `HARNESS_FCC_TIMEOUT_SEC` 1600->2400,
  `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` 2160->2960,
  `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` 2700->3500.0.
- `.env_example` updated: yes, both files.
- **Local `.env` synced with `--force`, scoped to the two named services**
  (`sync_local_env_from_example.py orion-harness-governor orion-hub
  --force`) -- required because these are pre-existing keys whose *meaning*
  changed upstream, which the sync tool's normal non-destructive path
  correctly refuses to touch.
- **Incident, fully disclosed:** that `--force` call also flattened five
  unrelated diverged keys back to `.env_example` placeholders:
  `HARNESS_AITOWN_ENABLED` (true->false), `HARNESS_AITOWN_CONVEX_URL`
  (real host -> template host), `GRAPHITI_ADAPTER_URL` (real host ->
  container-name placeholder), `HUB_AITOWN_WORLD_ID` (real ID -> empty),
  `HUB_AITOWN_ADMIN_KEY` (real credential -> empty). Caught from the sync
  tool's own "Updated:" output before moving on to anything else. All five
  restored via `Read`+`Edit` directly on the live files (not scripted --
  each value hand-verified against what the tool had just printed), and the
  final live state was independently re-verified by the code-review
  subagent reading the files itself, not trusting this report:
  `HARNESS_AITOWN_ENABLED=false`,
  `HARNESS_AITOWN_CONVEX_URL=http://100.121.214.30:5173`,
  `GRAPHITI_ADAPTER_URL=http://127.0.0.1:8640`,
  `HUB_AITOWN_WORLD_ID=m1720g80td7...` (non-empty),
  `HUB_AITOWN_ADMIN_KEY=orion-aitown|011afe2d...` (non-empty). No data was
  lost past this session.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_curiosity_investigation.py -q
-> 117 passed

PYTHONPATH=. pytest orion/harness/tests/ orion/schemas/ -q
-> 364 passed, 4 failed

PYTHONPATH=.:services/orion-harness-governor pytest services/orion-harness-governor/tests/ -q
-> 20 passed

python scripts/check_env_key_single_source.py -> OK: 1 owned env key(s), no drifted copies.

docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml config -q -> exit 0
```

All 4 failures confirmed PRE-EXISTING by stashing this diff and re-running
the same tests against clean `main` -- identical failures, same names, same
line. Three (`test_grounding_capsule_consumers.py` x2,
`test_harness_runner.py::test_harness_runner_surfaces_fcc_error_code`) were
already documented pre-existing in PR #2062's own report. The fourth
(`test_context_provenance.py::test_static_ctx_assignments_covered`) is about
unclassified `orion-actions/executor.py` context keys, unrelated to anything
this patch touches; confirmed identical on clean `main` via `git stash`.

## Evals run

None applicable -- config values, not model or pipeline behavior.

## Docker/build/smoke checks

`docker compose config -q` validated against the live main-checkout `.env`
(read-only, no build/up run). No image content changed.

## Review findings fixed

- Finding: `services/orion-harness-governor/app/settings.py`'s comment
  directly above the changed `Field` default still said "1600, matching
  .env_example and the compose default" -- describing a state that no longer
  existed, in the exact file whose own point is "don't let this drift."
  - Fix: comment rewritten to state 2400 and reference the 2026-09-03 raise.
  - Evidence: `check_env_key_single_source.py` doesn't catch prose (only
    `KEY=`/`Field(N, alias=...)` literals), so this was a manual re-read, not
    gate-caught.
- Finding: `orion/schemas/harness_finalize.py:298` carries a near-identical
  comment to one fixed in `orion/harness/runner.py` in the same diff, and was
  missed on the first pass.
  - Fix: updated `1600s` -> `2400s`.
  - Evidence: re-grepped for the literal after the fix; zero remaining hits
    tied to this key outside deliberately-frozen PR-report/test-fixture text.
- Finding: `orion/curiosity/README.md`'s config table listed
  `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` as `1500` -- already stale before
  this patch (real value was `2700`), now doubly so.
  - Fix: corrected to `3500` with a note on the prior drift, since this patch
    was already touching this exact key's derivation three other places.
  - Evidence: table cell now matches `.env_example`.
- Finding (informational, no code change): `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC`
  is not curiosity-specific -- it's the default for every
  `HarnessGovernorClient.run()` call with no explicit override, including
  `execute_unified_turn()` on the ordinary chat path. Raising it means a
  genuinely stuck fast-lane chat turn now waits ~800s longer before Hub
  reports an error.
  - Response: accepted as correct scope, not a defect. Hub's own settings
    comment already documents that real turns have no hard Hub-side ceiling
    ("Juniper's UI just shows 'thinking' for as long as it takes"), so this
    changes how long a *failure* takes to surface, not user-facing latency on
    a working turn. Named here explicitly since the PR framing is
    curiosity-focused and this blast radius wasn't obvious from that framing
    alone.
- Finding (informational, no code change): the margin between
  `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` (now 2960) and the hard
  `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC` (3600, unchanged) shrank from 1440s
  to 640s -- the opposite direction of "this workload got 1.9x slower."
  - Response: deliberate, not an oversight. Confirmed by reading
    `harness_governor_client.py`: the extension loop clamps every wait to
    `min(poll_sec, remaining)`, so it never overshoots regardless of
    `poll_sec`'s size -- 2960 is safe against 3600 by construction, not by
    luck. `MAX_WAIT_SEC` is a shared ceiling across every RPC caller, not a
    per-lane value, and this patch's own arithmetic already covers the
    measured worst case (2234s) with the soft ceiling alone (2400s), well
    inside even the smaller 640s of remaining headroom. Raising it further
    was not required by any measured number and would have widened a
    system-wide ceiling to solve a lane-specific problem -- left unchanged on
    purpose.

## Restart required

```bash
sudo docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build
sudo docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: LOW.
  Concern: a stuck (not merely slow) fast-lane interactive chat turn now
  takes up to ~800s longer to surface as an error to Juniper, because
  `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` is shared rather than per-lane.
  Mitigation: none needed today -- Hub's own UI has no timeout-driven
  behavior at this ceiling (`settings.py`'s comment: "just shows 'thinking'
  for as long as it takes"), so this affects failure-surfacing latency, not
  the happy path. A future per-lane RPC timeout would remove this coupling
  entirely but is a larger change (the RPC call has no lane-aware parameter
  today) and is out of scope here.
- Severity: LOW.
  Concern: `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC` headroom shrank from 1440s
  to 640s. A future further slowdown (a third GPU lane change, a larger
  model) could re-open the gap this patch closes without anyone revisiting
  this specific margin.
  Mitigation: none required now -- see "Review findings fixed" above for why
  640s is still safe against every measured number. Worth re-deriving if
  `HARNESS_FCC_TIMEOUT_SEC` is raised again.

## PR link

<pending>
