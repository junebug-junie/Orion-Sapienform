# Curiosity compaction readout route correction

The deployed readout returned 404 at its documented `/dreams/curiosity-compactions`
path: `dream_api` repeated the `/dreams` prefix already supplied by `main.app`.
Remove the router-local prefix so the mounted public path matches the operator contract.
No memory, scheduling, schema, env, model prompt, or compaction behavior changes.

## Evidence and verification

- Read-only live deployment check: migration table exists, zero compaction rows,
  dream cycle enabled, `DREAM_CURIOSITY_COMPACTION_ENABLED=false`.
- Durable `/health` returned `ok=true` and `admission_enabled=true` after startup.
- Live OpenAPI exposed `/dreams/dreams/curiosity-compactions`; that path returned
  `{"items":[]}`. No compaction generation was claimed.
- Regression uses the actual `main.app` mount through TestClient without lifespan:
  failed with 404 before the fix; verifies 200, stored result and limit forwarding
  afterward, and 422 for an out-of-range limit.
- Focused compaction tests and offline chronology eval: 20 passed.
- Dream Docker build passed; network-isolated image TestClient smoke returned 200
  at the documented path with a mocked store (no production access).

## Review findings fixed

Independent code review approved the fix with no findings. Final CI is recorded in the PR. The patch is confined
to one route decorator and its mounted-app regression. No live services are restarted.

## Deployment and rollback

The route correction was merged to main and deployed. With explicit operator approval,
`DREAM_CURIOSITY_COMPACTION_ENABLED` is now `true` in the checked-in Dream environment
template and the ignored live Dream `.env`; Dream was recreated from main. Verify the
documented URL returns 200 and observe the next successful persisted sleep for a durable
admission. Its end-to-end generated summary remains UNVERIFIED until that occurs.
Roll back by setting the flag to `false` and recreating Dream; original journals and
compaction rows remain untouched.
