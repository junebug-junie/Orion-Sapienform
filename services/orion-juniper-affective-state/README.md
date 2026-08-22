# orion-juniper-affective-state

Thin CPU orchestrator in front of `orion-affectgpt-worker`. Given an
already-written video+audio pair, does a real bus RPC round-trip to the
worker, wraps the result as `JuniperMultimodalAffectV1`, and publishes it to
`orion:affectgpt:assessment`.

**Deployed on circe, not athena.** `video_path`/`audio_path` in a request are
resolved on the *worker's* filesystem, and circe/athena share no filesystem
(`/mnt/telemetry` is athena-local ext4, no NFS/exports; `/mnt/scripts` is a
separate per-host clone, not synced — see
`reference_circe_gpu_inventory_and_lane_map`). Co-locating sidesteps that
gap; see `app/settings.py`'s `NODE_NAME` comment.

## Non-goal: no ambient mode

There is currently no pipeline anywhere in this repo that captures Juniper's
webcam/mic. This service is a manual/turn-scoped trigger only —
`POST /v1/juniper/affect/trigger` with a video/audio path pair already on
disk. Building a background polling loop with nothing to poll would be
empty-shell cognition. Add ambient mode once a real capture source exists to
drive it; don't guess at its shape now.

## Cross-host capture (future, not built)

The moment a capture source lands on a host other than circe, this
service's request contract (bare filesystem paths) stops working — that's
the point where real upload/streaming needs designing, not before.

## Bus contract

- Calls `orion:exec:request:AffectGptWorkerService` (RPC, via
  `OrionBusAsync.rpc_request`), reply on a per-request `orion:affectgpt:reply:<corr_id>`.
- Publishes `orion:affectgpt:assessment` (`JuniperMultimodalAffectV1`) after
  every trigger, success or failure (the event's `ok`/`error` fields carry
  that — a failed assessment is still a real event, not a silent drop).

## Operator checklist

1. `GET /health`
2. `POST /v1/juniper/affect/trigger` — `{"video_path": "...", "audio_path": "...", "subtitle": "..."}` (paths must be readable inside the *worker's* container).

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
