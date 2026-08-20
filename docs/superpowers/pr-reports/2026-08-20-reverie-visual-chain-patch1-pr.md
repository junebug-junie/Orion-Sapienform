# Reverie Visual Chain — Patch 1 PR report

Date: 2026-08-20
Branch: `feat/reverie-visual-chain`
Design doc: `docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md`

## Summary

- New, parallel reverie chain alongside the live text chain (`substrate_reverie_chain`):
  generate an image about gathered context, re-observe it via captioning, feed the
  description into the next reverie — Patch 1 lays the schema/storage/service skeleton only.
- Two new Postgres tables (`reverie_visual_chain`, `reverie_visual_artifact`), deliberately
  not under the `substrate_` prefix (design doc §3/§5).
- Two new Pydantic schemas (`ReverieVisualChainV1`, `ReverieVisualArtifactV1`), registered in
  both `_REGISTRY` and `SCHEMA_REGISTRY`. `prior_description` is a real, enforced continuity
  field — deliberately not repeating the dead `SpontaneousThoughtV1.next_focus`/`drift`
  keyword-cathedral pattern found live in the text chain (design doc §2).
- New `orion-diffusion-host` service skeleton (health-check only — no `diffusers`/`torch`, no
  model loading, no bus wiring), modeled on `orion-vision-host`'s shape.
- New content-addressed image storage helper (`orion/reverie/visual_storage.py`), mirroring
  `services/orion-hub/scripts/chat_attachments.py`'s discipline (sha256 filename, magic-byte
  MIME sniffing, write-then-atomic-rename).

## Outcome moved

Nothing runtime-visible yet — this is schema/skeleton only, as scoped. The concrete outcome is
that Patch 2 (chain orchestration in `orion-thought`) and the eventual real diffusion/vision
wiring have a real table, a real schema contract, and a real storage helper to build against,
instead of starting from nothing.

## Current architecture

Before this patch: the text reverie chain (`orion-thought`, `substrate_reverie_chain` /
`substrate_reverie_thought`) was the only reverie mechanism — live, continuous,
narration-only. No image generation, no re-observation loop, no visual artifact storage
anywhere in the repo.

## Architecture touched

- `orion/schemas/` — two new schema models + registry entries.
- `orion/reverie/` — new package: content-addressed visual-artifact storage.
- `services/orion-diffusion-host/` — new service (skeleton only).
- `services/orion-sql-db/` — new manual migration.

No existing service's runtime behavior changed. `orion/bus/channels.yaml` untouched (no
producer/consumer exists yet — confirmed by diff).

## Files changed

- `docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md`: design doc, Patch 1 scope.
- `services/orion-sql-db/manual_migration_reverie_visual_chain.sql`: new — `reverie_visual_chain`
  / `reverie_visual_artifact` tables + indexes, per design doc §5.
- `orion/schemas/reverie_visual.py`: new — `ReverieVisualChainV1` / `ReverieVisualArtifactV1`.
- `orion/schemas/registry.py`: registers both new schemas.
- `orion/schemas/tests/test_reverie_visual_registry.py`: new — registration + round-trip tests,
  including a dedicated test pinning `prior_description` as a real field.
- `orion/reverie/visual_storage.py`: new — content-addressed image write/read, magic-byte MIME
  sniffing (PNG/JPEG/GIF/WebP), no image library dependency.
- `orion/reverie/tests/test_visual_storage.py`: new — sniffing, write, idempotency, rejection,
  round-trip tests.
- `services/orion-diffusion-host/{README.md,.env_example,docker-compose.yml,requirements.txt,
  Dockerfile,app/main.py,app/settings.py,tests/test_health.py}`: new service skeleton, health
  endpoint only.
- `services/orion-diffusion-host/tests/test_health.py`: fixed in a follow-up commit (see Review
  findings below) to import correctly from repo root.

## Schema / bus / API changes

- Added: `ReverieVisualChainV1` (`kind=reverie.visual.chain.v1`), `ReverieVisualArtifactV1`
  (`kind=reverie.visual.artifact.v1`) — registered in `orion/schemas/registry.py`.
- Removed: none.
- Renamed: none.
- Behavior changed: none (no consumer/producer wired yet).
- Compatibility notes: no bus channel added in this patch — `orion:exec:request:VisionHostService`
  will get a new `task_type` in a later patch when re-observation is actually wired
  (design doc §3), not this one.

## Env/config changes

- Added keys: `services/orion-diffusion-host/.env_example` (new file) — `LOG_LEVEL`, `NODE_NAME`,
  `HOST_PORT`, `ORION_BUS_ENABLED`, `ORION_BUS_ENFORCE_CATALOG`, `ORION_BUS_URL`,
  `HEARTBEAT_INTERVAL_SEC`, `MODEL_CACHE_DIR`, `HF_HOME`, `TRANSFORMERS_CACHE`,
  `CUDA_VISIBLE_DEVICES`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (new file).
- local `.env` synced: yes — `services/orion-diffusion-host/.env` created in the primary
  checkout (`/mnt/scripts/Orion-Sapienform`) as a direct copy of `.env_example` (no secrets in
  any key, so a direct copy is safe; `scripts/sync_local_env_from_example.py` doesn't
  auto-create `.env` for services outside its static `DEFAULT_SERVICES` list, so this was done
  by hand). Verified byte-identical by the review agent.
- skipped keys requiring operator action: none.

## Tests run

```text
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest \
    orion/reverie/tests/test_visual_storage.py \
    orion/schemas/tests/test_reverie_visual_registry.py \
    services/orion-diffusion-host/tests/test_health.py -q
15 passed, 16 warnings in 3.24s
```

(16 warnings are pre-existing `pydantic` `model_` protected-namespace warnings from unrelated
schemas already in the registry — not introduced by this patch.)

## Evals run

No eval harness exists for `orion-diffusion-host` (new service) or for the reverie-visual
storage/schema modules — expected, since Patch 1 is schema/skeleton only with no generative
behavior to evaluate yet. Flagging per CLAUDE.md §11: an eval harness for image-generation
quality/relevance belongs in the patch that actually wires the diffusion model, not this one.

## Docker/build/smoke checks

Not run. This patch adds no runtime behavior that changes container boot, ports, health checks,
or dependencies beyond the skeleton FastAPI app itself, and the target GPU (4th Circe V100
32GB) is not provisioned yet (design doc §3, explicitly decoupled from Patch 1). `docker-compose
config` validation deferred to the patch that actually brings this service up.

## Review findings fixed

- Finding: `services/orion-diffusion-host/tests/test_health.py` was missing the
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` line every sibling service's
  test file has (e.g. `orion-vision-host`). `from app.main import app, settings` only resolved
  when pytest was invoked with cwd = the service directory itself (`python -m pytest` adds cwd
  to `sys.path[0]`); it failed collection when run from the repo root, which is how CLAUDE.md
  §11 documents test invocation (`pytest services/<service_name>/tests -q`). Found by me before
  dispatching the review subagent.
  - Fix: added the missing `sys.path.insert` line, matching the sibling-service convention.
  - Evidence: re-ran the full 3-file suite from repo root after the fix — 15 passed (see Tests
    run above). Committed separately (`574ebe7`) with the reasoning in the commit message.
- A dedicated `orion-repo-agent` code-review pass (full diff, migration correctness, schema/
  registry consistency, skeleton-service honesty, storage-module edge cases, env parity, naming)
  found **no further material issues**. One informational note recorded, not fixed (not
  reachable in this patch): `visual_storage.py`'s temp filename is
  `.{sha256}.{pid}.part` — process-scoped, not thread-scoped. Nothing in Patch 1 calls this
  function from any concurrent request path (single-flight orchestration is Patch 2), so this
  isn't reachable today; flagged for whoever wires the first real caller in Patch 2.

## Restart required

```text
No restart required.
```

No service that is currently running was changed. `orion-diffusion-host` is a brand-new service
that has never been brought up.

## Risks / concerns

- Severity: low
- Concern: `orion-diffusion-host`'s eventual GPU target (a 4th physical V100 32GB on Circe,
  sharing with an as-yet-undefined "world model" component) is not provisioned and its sharing
  arrangement is unresolved — explicitly deferred per Juniper ("I don't recall which world
  model it is so get on with it"), not blocking Patch 1.
- Mitigation: none needed for this patch; flagged so it isn't forgotten before the patch that
  actually wires a real model.

- Severity: low
- Concern: no eval harness exists yet for image-generation quality/relevance once a real model
  is wired.
- Mitigation: build one in the patch that wires the diffusion model, per CLAUDE.md §11.

## PR link

<will be filled in after `gh pr create`>
