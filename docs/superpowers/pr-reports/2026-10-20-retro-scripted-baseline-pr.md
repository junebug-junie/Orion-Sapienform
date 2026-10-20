## Summary

- Add an explicit 60-second browser handoff to `scripted-right-jump-v1`.
- Fix a CI-exposed media pipe deadlock after native-worker SIGKILL using a bounded shared slot.
- Keep native frame authority, neutral-frame human takeover, deadline limits and
  sponsor-disconnect cancellation; no automatic script restart.
- Attribute script actions in bounded local evidence and show active policy identity.
- Reconcile the stale handoff against merged Retro fixes and the original roadmap.
- Preserve the existing paused live game; verify new behavior in isolated containers.

## Outcome moved

Retro can now execute and identify a resident scripted controller on an explicitly
handed-off port. Human takeover returns actual applied input to the browser.
This is P3 infrastructure only: no perception, goal revision, learning, autonomy
admission or memory export is claimed.

## Current architecture

Browser → dedicated Hub Retro relay → private Unix socket → native worker.
Main already includes #2771, #2772, #2774, #2787 and #2793; the old 502 is no
longer reproducible. Public page and catalog loaded without a code. Deployed
control/session/client/relay source hashes matched main at audit time.

## Architecture touched

Retro service-local contracts, arbiter consumer and session projection; Hub
route allowlist, rendered page, linked client and native browser eval.
No cognition service or cross-service bus event is added.

## Files changed

- `services/orion-retro/app/scripted.py`: bounded fixed policy and lease lifecycle.
- `services/orion-retro/app/{contracts,main,worker,session}.py`: request validation,
  worker actuation, evidence attribution and status projection.
- `services/orion-retro/app/media.py`: nonblocking fixed-size shared packet slot.
- `services/orion-retro/tests/`: ownership, epochs, timing, cancellation, killed media writer and API checks.
- `services/orion-hub/scripts/retro_routes.py`: same-origin script route forwarding.
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js`: explicit
  handoff, policy label, takeover and human recovery.
- `services/orion-hub/tests/retro/test_boundary.py`, `evals/retro_browser.mjs`:
  boundary and real interaction coverage.
- Retro README, browser runbook and continuation-status spec: contracts,
  superseded work, original acceptance gaps, and next cognition experiment.

## Schema / bus / API changes

- Added: strict service-local `ScriptControl`; POST `sessions/{id}/scripted`.
- Removed: none.
- Renamed: none.
- Behavior changed: only a live owned human lease with the expected epoch can
  become a script lease. Human takeover revokes it permanently. Script commands
  expire after 100 ms without refresh; wall-clock budget is 5–1800 seconds.
- Compatibility notes: additive `scripted` status summaries and `policy_id` on
  script-applied evidence. Existing human input protocol is unchanged.
- No global schema registration or bus channel; no metric is wired to cognition.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  not applicable; no template changed. Private local build config copied from
  primary checkout; all `.env` files verified ignored.
- skipped keys requiring operator action: none.

## Tests run

```text
/tmp/orion-retro-service-venv/bin/python -m pytest tests/retro services/orion-retro/tests -q
85 passed
/tmp/orion-retro-service-venv/bin/python -m pytest services/orion-hub/tests/retro services/orion-hub/tests/test_hub_panel_reachability.py -q
8 passed
node --check services/orion-hub/static/js/retro-panel.js
node --check services/orion-hub/evals/retro_browser.mjs
git diff --check
PASS
```

The supplied local venv lacked Hub's already-pinned pydantic-settings; installed
2.5.2 there, then reran the Hub tests. No dependency file change was needed.

## Evals run

```text
retro_browser.py normal: PASS (/tmp/retro-scripted-slot-browser/browser.json)
retro_browser.py --ack-delay-ms 350: PASS (/tmp/retro-scripted-slot-lag/browser.json)
container_smoke.py: PASS (/tmp/retro-scripted-slot-container/result.json)
```

Browser checks include rendered template/linked asset, native frames and nonzero
PCM, keyboard/synthetic gamepad, script attribution, takeover, five-second
server budget, rejected-handoff recovery, pause/blur recovery, reconnect and tab-exit pause.
The first script eval's human-input assertion ran before asynchronous takeover
restored canvas focus. Native evidence showed the takeover; the eval now waits
for focus before pressing the key.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-retro build: PASS
container smoke: native input, pause/resume, lease expiry, worker kill,
storage quota and disabled network egress: PASS
public /retro browser: no code prompt, six games, existing paused game preserved
```

No new live deployment. Isolated eval containers were removed by their harness.
No ROM, credential, save, `.env`, or graph artifact is committed.

## Review findings fixed

- Finding: CI killed the emulator mid-media-packet; `Queue.get_nowait()` blocked
  the API waiting for the rest of a pipe packet. Local reproduction confirmed
  the reader still blocked after the writer died.
  - Fix: fixed-size shared slot with nonblocking locks; discard an abandoned
    slot with its session. No blocking thread fallback or timeout increase.
  - Evidence: spawned killed-writer regression, bounded-slot test and rebuilt
    native container fault smoke. Independent review found no material issue.

- Finding: delayed handoff could consume a newer human lease on the same connection.
  - Fix: strict `expected_epoch`, checked before transfer; stale stop also refuses
    to cancel a newer script grant.
  - Evidence: deterministic reordered-handoff regression and native takeover eval.
- Finding: script mode stranded recovery after timeout, pause/blur or failed start.
  - Fix: clear local mode on closed transport/confirmed paused revocation; explicit
    takeover resumes/reconnects human control. Failure names the recovery control.
  - Evidence: real-browser timeout/pause/blur recovery checks.

Independent re-review found no remaining material defects. The reviewer ran 55
focused tests; final normal and delayed-ack browser runs also passed.

## Restart required

After merge, from clean primary main synchronized with origin/main, and after
the current user game has ended:

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Retro restart loses the current paused game because save/load is unavailable.
The previously approved restart concerned an older session, so this patch
preserves the newly observed session.

## Risks / concerns

- Severity: medium. Concern: original P3 remains incomplete (open-loop script,
  no goal inbox/outcome reader/cognition loop, no independent 30-minute run).
  Mitigation: explicit UI label and
  [acceptance reconciliation](../specs/2026-10-20-retro-continuation-status.md).
- Severity: medium. Concern: P2 WebRTC/save-load/actual-client audio and performance
  gates remain open. Mitigation: preserve UNVERIFIED/deviation verdicts; do not
  equate server ack timing or automated PCM with end-user qualification.
- Severity: low. Concern: 100 ms script deadline can stop under host overload.
  Mitigation: stop safely, record reason, require a new explicit handoff.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2796
