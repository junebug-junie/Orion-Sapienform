## Summary
- Seam S1: the room camera's presence state machine (orion-vision-window) now appends one row per state change to new table `vision_presence_transition`, plus a same-state keepalive every 600 s while someone is present, and an `unknown` row on clean shutdown/dark camera (120 s with no frames).
- The Temporal Self reducer folds these into **company** arcs: open on `present`, `recent` is grace, close on `absent` (process_ended) or `unknown`/restart/lost close/no row for 1500 s (source_stale). The enrolled name is stored only on a `probable` face match (DB CHECK `label IS NULL OR identity_confirmed`).
- Stance cue gets one clause while a company arc is open: "Someone has been here since 21:02 [arc:id]" (name only when confirmed). A hard crash closes the arc after 1500 s, so the cue can't claim company indefinitely.

## Outcome moved
"Someone was with me from 9:02 to 11:15" is now replayable and citable; before, only the latest presence state existed (one overwritten row).

## Schema / bus / API changes
- Added table `vision_presence_transition` (`manual_migration_vision_presence_transition_v1.sql`, additive, header `ORION-MIGRATION-REQUIRED-BY: orion-vision-window, orion-durable-runs`).
- `orion/schemas/temporal_self.py`: source kind `presence_transition`, arc kind `company`, closed reason `source_stale`. These are forbid/Literal models: **old readers reject the new values** (old orion-thought skips the cue as `invalid_frame`; old hub day read degrades). Readers must deploy before durable-runs. Rolling durable-runs back after a company arc exists stalls the chronicle.
- Consumers checked: frame (company never displaces active/previous), day_block KIND_ORDER, citation_truth phrase "been here since", body NO_BODY_KINDS.

## Env/config changes
- Added `VISION_PRESENCE_TRANSITIONS_ENABLED=true`, `VISION_PRESENCE_KEEPALIVE_SEC=600` (orion-vision-window). Local `.env` synced with `--all-keys orion-vision-window`.

## Tests run
```text
vision-window + orion/temporal_self: 312 passed
services/orion-durable-runs full incl. Postgres (throwaway postgres:16): 492 passed, 1 skipped
orion/orion_day + thought stance/temporal tests: 174 + 66 passed
check_migration_required_by --base origin/main: OK; migration drift/schema skew 77 passed; deploy gate + vision contract 43 passed
```

## Evals run
```text
temporal_self_live_replay.py --step-sec 600: pass
run_arc_precision_eval.py: pass
run_day_citation_truth_eval.py: truth_rate 1.000
```

## Review findings fixed
- Lost closing row could merge two visits → reducer closes on from_state evidence; producer requeues failed writes (≤256).
- Weak (`possible`) face match counted as confirmed → name only on `probable`.
- Row after shutdown `unknown` → close() refuses later rows.
- Hard crash left company asserted indefinitely → keepalive + 1500 s reducer timeout evaluated at every clock advance (replay-identical).

## Restart required (order matters)
```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < /mnt/scripts/Orion-Sapienform/services/orion-sql-db/manual_migration_vision_presence_transition_v1.sql
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-thought up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-vision-window up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns
- Low: a stretch in its 2-min grace at midnight can leave a zero-length 00:00 arc.
- Low: "been here since" said while citing a conversation arc can be marked a citation mismatch.
- Live write path UNVERIFIED until deployed.
