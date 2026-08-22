# orion-juniper-affective-state

Thin CPU orchestrator in front of `orion-affectgpt-worker`. Two ways in:

- `POST /v1/juniper/affect/trigger` — given an already-written video+audio
  pair, does a real bus RPC round-trip to the worker.
- `POST /v1/juniper/affect/capture_and_assess` (2026-08-22) — the live
  path: bus RPC to `orion-vision-retina` (carbon) for a fresh clip, fetch
  both blobs from `orion-percept-store`, then the same worker round trip.
  This is what Hub's "Affect check" button calls.

Both wrap the result as `JuniperMultimodalAffectV1` and publish it to
`orion:affectgpt:assessment` — one event stream regardless of entry point.

**Deployed on circe, not athena.** `video_path`/`audio_path` fed to the
worker are resolved on the *worker's* filesystem, and circe/athena share no
filesystem (`/mnt/telemetry` is athena-local ext4, no NFS/exports;
`/mnt/scripts` is a separate per-host clone, not synced — see
`reference_circe_gpu_inventory_and_lane_map`). Co-locating sidesteps that
gap for the worker call; see `app/settings.py`'s `NODE_NAME` comment.

## Non-goal: still no ambient mode

Both entry points above are explicit, caller-triggered captures. There is
still no background/scheduled polling loop — a loop with nothing forcing a
capture to happen would be empty-shell cognition. Hub's button is a manual
turn-scoped trigger, not a toggle that starts continuous recording; add
ambient mode only once there's a real, named reason to poll on a schedule.

## The cross-host bridge (built 2026-08-22)

`capture_and_assess()` is the answer to the "Cross-host capture" gap this
README used to flag as future work: it bus-RPCs `orion-vision-retina`
(`orion:exec:request:RetinaClipCaptureService`, see that service's README)
for a live clip, fetches the resulting `video_sha256`/`audio_sha256` blobs
from percept-store with **hash verification on the fetched bytes**
(`_fetch_percept` — never trusts a reported hash without recomputing it),
and writes them to `AFFECTGPT_SCRATCH_DIR`
(`/mnt/scripts/orion-affectgpt-scratch` by default) — the SAME shared volume
`orion-affectgpt-worker` already mounts read-only at the identical
container path. That's the whole trick: a plain `tempfile.TemporaryDirectory()`
default would write somewhere private to *this* container and the worker
container could never see it. The temp dir (and its fetched bytes) is
removed once the worker call returns, success or failure.

A capture or fetch failure never reaches the worker at all — it's wrapped
straight into a failed `AffectGptAssessResultPayload`
(`error_code` in `{"capture_failed", "fetch_failed"}`) and published like
any other failed assessment.

## Bus contract

- Calls `orion:exec:request:AffectGptWorkerService` (RPC, via
  `OrionBusAsync.rpc_request`), reply on a per-request `orion:affectgpt:reply:<corr_id>`.
- Calls `orion:exec:request:RetinaClipCaptureService` (same RPC pattern),
  reply on `orion:retina:clip:reply:<corr_id>` — only from
  `capture_and_assess()`, not from `/trigger`.
- Publishes `orion:affectgpt:assessment` (`JuniperMultimodalAffectV1`) after
  every trigger, success or failure (the event's `ok`/`error` fields carry
  that — a failed assessment is still a real event, not a silent drop).

## Operator checklist

1. `GET /health`
2. `POST /v1/juniper/affect/trigger` — `{"video_path": "...", "audio_path": "...", "subtitle": "..."}` (paths must be readable inside the *worker's* container).
3. `POST /v1/juniper/affect/capture_and_assess` — optional `{"subtitle": "...", "user_message": "..."}`. Synchronous, 30-90s (real capture + real GPU inference) — use a generous client timeout, not a quick one.

## Tests

```bash
pytest services/orion-juniper-affective-state/tests -q
```

Bus-free by design (mocked `OrionBusAsync`) — no live worker or Redis
required.

## Evals

```bash
pytest services/orion-juniper-affective-state/evals -q
```

Requires a live bus + live `orion-affectgpt-worker`. Round-trips a real
trigger and checks the published `orion:affectgpt:assessment` event actually
lands.
