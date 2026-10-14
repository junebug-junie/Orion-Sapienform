## Summary
- Reverie (orion-thought) now gets today's last 3 closed Temporal Self arcs, most recent first, local time, each with `[arc:<id>]` (Temporal Self rev 4 later patch).
- New pure `orion/temporal_self/recent_arcs.py::project_recent_arcs` (same render gates as the stance cue: stale >900 s, other day, future, invalid → nothing). Cap 5 arcs / 400 chars.
- Reuses the stance cue's frame fetch (`fetch_frame_bounded`), run in the same `gather` as the other reverie reads; 1.0 s timeout; any failure → no block.
- Prompt: "EARLIER TODAY" background block; arc ids are not evidence. Stray arc ids in `evidence_refs` are stripped before the hollow guard (never counted as grounding).

## Outcome moved
Reverie had no day history; it can now connect a thought to what Orion was doing earlier today, with citations.

## Env/config changes
- Added `ORION_REVERIE_RECENT_ARCS_ENABLED=true`, `ORION_REVERIE_RECENT_ARCS_LIMIT=3`, `ORION_REVERIE_RECENT_ARCS_FETCH_TIMEOUT_SEC=1.0` (orion-thought). Local `.env` synced.

## Tests run
```text
orion/temporal_self/tests/test_recent_arcs.py 11 passed; services/orion-thought/tests/test_reverie_recent_arcs.py 13 passed
services/orion-thought/tests 606 passed, 28 skipped, 1 failed (test_mind_enrichment_defaults_off — also fails on main)
orion/schemas/tests/test_context_provenance.py::test_static_ctx_assignments_covered fails on main too (cortex-exec keys)
```
Live render from the real frame (day 2026-10-13):
```text
Recently closed today (most recent first):
- 20:37-20:38 (1 min) an image attempt [arc:d3c81aa8ef1585e4]
- 20:18-20:36 (18 min) attention on Codebase prediction error [arc:4670f089358ab896]
- 20:05-20:06 (1 min) attention on Bus synaptic prediction error [arc:2023af2918105f7f]
```

## Review findings fixed
- Stray arc citation would drop a grounded thought → stripped before the hollow guard (tests).
- Char cap could overrun on tiny caps → renders nothing (`cap_too_small`).

## Restart required
```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns
- Medium: watch `reverie_evidence_arc_ref_stripped` and hollow rate after deploy. Model use of the block is UNVERIFIED until a live tick.
- Low: interoception labels are raw node refs.
