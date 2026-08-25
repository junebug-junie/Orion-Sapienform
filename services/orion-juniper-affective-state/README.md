# orion-juniper-affective-state

Thin CPU orchestrator in front of `orion-affectgpt-worker`. Two ways in:

- `POST /v1/juniper/affect/trigger` — given an already-written video+audio
  pair, does a real bus RPC round-trip to the worker.
- `POST /v1/juniper/affect/capture_and_assess` (2026-08-22) — the live
  path: bus RPC to `orion-vision-retina` (carbon) for a fresh clip, fetch
  both blobs from `orion-percept-store`, then the same worker round trip.
  This is what Hub's "Check now" button AND its ambient toggle both call
  (`{"trigger": "manual"}` vs `{"trigger": "ambient"}` in the request body).

Both wrap the result as `JuniperMultimodalAffectV1` and publish it to
`orion:affectgpt:assessment` — one event stream regardless of entry point.

**Deployed on circe, not athena.** `video_path`/`audio_path` fed to the
worker are resolved on the *worker's* filesystem, and circe/athena share no
filesystem (`/mnt/telemetry` is athena-local ext4, no NFS/exports;
`/mnt/scripts` is a separate per-host clone, not synced — see
`reference_circe_gpu_inventory_and_lane_map`). Co-locating sidesteps that
gap for the worker call; see `app/settings.py`'s `NODE_NAME` comment.

## Ambient mode exists now, but it does not live here (2026-08-22)

Both entry points above are still single, explicit calls with no scheduling
logic of their own inside THIS service — every request here is still one
caller asking for one attempt. Recurring capture is real, though: Hub owns
that loop (`services/orion-hub/scripts/vision_affect_ambient.py`) and just
calls `/capture_and_assess` repeatedly with `trigger="ambient"` while its
toggle is on. This corrects an earlier version of this README, which
described Hub's button as "a manual turn-scoped trigger, not a toggle" --
that was true of the button that shipped first (2026-08-22, PR #1838), not
of the toggle that replaced it as the primary control the same day.

`trigger` (`"manual"` | `"ambient"`) and `correlation_id` on
`JuniperMultimodalAffectV1` (`orion/schemas/affectgpt.py`) exist so a
consumer can tell the two apart and, via `correlation_id`, join one
attempt's retina-RPC/worker-RPC/event legs together -- `capture_and_assess()`
generates ONE id per attempt and threads it through all three, rather than
each leg getting its own independently-generated one.

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

## Closing the loop into Orion's own chat turns (2026-08-25)

Before this date, `orion:affectgpt:assessment` had exactly one consumer:
`scripts/tap_assessments.py`, a manual debug CLI. Orion's own chat turns
never found out about a capture — only the Hub UI panel showed it to
Juniper. `_publish_event` now ALSO mirrors every successful capture (a
truncated excerpt of `raw_response`, capped at `_AFFECT_SUMMARY_MAX_CHARS`
= 300 chars — never the verbatim `transcript`) into a single Redis key
(`orion:juniper_affect:latest`, `orion/situational/juniper_affect_state.py`)
that `orion/situational/context.py` polls for every "orion" mode chat turn's
situation brief, gated on a configurable max-age (default 300s,
`ORION_SITUATION_AFFECT_MAX_AGE_SECONDS` in orion-hub/orion-cortex-exec).
Failed/empty captures are not mirrored — a failure should not overwrite a
real prior read, and the reader's own age gate ages that prior read out on
its own schedule. The mirror write is additive and fail-open: it runs after
the real `orion:affectgpt:assessment` publish already succeeded, and never
raises, so a Redis hiccup here cannot break the real event stream.

## Durable persistence (2026-08-25)

`orion:affectgpt:assessment` now has a second real consumer:
`orion-sql-writer` projects every event into `juniper_multimodal_affect_log`
(see that service's README). The Redis mirror above has a 1h TTL and is
the live-read path for chat turns; this table is the durable history for
any capture published while `orion-sql-writer` was actually connected to
the bus. Review finding, 2026-08-25: `OrionBusAsync.publish()` is plain
Redis pub/sub with no redelivery -- a capture published while
`orion-sql-writer` itself is disconnected (restart, DB pool exhaustion)
is dropped before it ever reaches this table too, same as any other
consumer on this bus. Once both the TTL key and that window have passed
with nothing durable written, the capture leaves no trace anywhere. Same
privacy boundary as the mirror: `transcript` is never persisted here
either (including on error -- see `JuniperMultimodalAffectSQL`'s
docstring for the fallback-path fix).

## Operator checklist

1. `GET /health`
2. `POST /v1/juniper/affect/trigger` — `{"video_path": "...", "audio_path": "...", "subtitle": "..."}` (paths must be readable inside the *worker's* container).
3. `POST /v1/juniper/affect/capture_and_assess` — optional `{"subtitle": "...", "user_message": "...", "trigger": "manual"|"ambient"}` (trigger defaults to "manual" if omitted). Synchronous, typically well under a minute but up to ~195s worst case (real capture + real GPU inference) — use a generous client timeout, not a quick one.

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
