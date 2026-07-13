# PR report: surface drive_state.v1's already-computed provenance to substrate + chat stance

## Summary

- `DriveAuditV1` (`dominant_drive`, `tension_refs`, `tension_kinds`, a human-readable `summary`) is already built and already published on every drive-engine tick (`build_drive_audit()` in `orion/spark/concept_induction/audit.py`, called live twice in `bus_worker.py`). It was being silently dropped two hops downstream — the exact "real engineering thrown away one hop downstream" pattern this repo already found once with `field_attention_frame.v1`.
- Gap 1: `orion/substrate/adapters/autonomy.py`'s substrate projection accepted a `DriveAuditV1` param but never wrote its `dominant_drive`/`summary` into the `drive_state` `StateSnapshotNodeV1`'s metadata.
- Gap 2: `services/orion-cortex-exec/app/chat_stance.py::_project_autonomy_from_beliefs()` only collected substrate snapshots tagged `snapshot_source == "autonomy"`. Investigated whether that filter had ever matched anything real (it was suspected dead) — **it hadn't been dead**: `orion/substrate/relational/adapters/autonomy_ctx.py` is a real, still-wired legacy GraphDB/SPARQL path, gated off by default (`AUTONOMY_GRAPH_BACKEND=disabled`). Kept it, added `"drive_state"` alongside it rather than replacing it.
- Also found and fixed: `sl.drives` nodes (already read by this function) carry each drive's live pressure in `.signals.salience`, previously extracted only as a bare `drive_kind` label with the value discarded.
- New `inputs["drive_state"]` in the chat-turn prompt payload — a **sibling** of the existing `inputs["autonomy"]`, never merged, per `orion/self_state/inner_state_registry.py`'s explicit `DUPLICATE` note on `drive_state.v1` vs `autonomy_state_v2`. Ships behind `CHAT_STANCE_DRIVE_STATE_VISIBLE`, default off.

## Outcome moved

Real drive-audit data (which drive dominates, why, in plain language) that has existed and published on every tick now has a path to reach the LLM's own reasoning context — previously it was computed, published to the bus, written to RDF, and then invisible everywhere else. This is visibility only; no behavior changes live (flag off by default).

## Current architecture (before this patch)

`DriveAuditV1` had zero real cognition consumers per `orion/self_state/inner_state_registry.py`'s own accounting. The substrate ladder already carried `drive_state` pressures in `StateSnapshotNodeV1.dimensions`, but `chat_stance.py`'s belief-projection function filtered for a snapshot_source value (`"autonomy"`) that a *different*, legacy adapter produces — `drive_state`'s snapshots were present in the belief graph but never collected by this function.

## Architecture touched

`orion/substrate/adapters/autonomy.py` (substrate projection), `services/orion-cortex-exec/app/chat_stance.py` (belief projection + prompt input assembly), test files for both, `.env_example`/`README.md`.

## Files changed

- `orion/substrate/adapters/autonomy.py`: `map_autonomy_artifacts_to_substrate()` now stamps `metadata["dominant_drive"]`/`metadata["summary"]` onto the `drive_state` snapshot node when a `DriveAuditV1` is passed. Purely additive — `dimensions`, `snapshot_source`, and every other field untouched.
- `services/orion-cortex-exec/app/chat_stance.py`: `_project_autonomy_from_beliefs()` now collects `snapshot_source in ("autonomy", "drive_state")` (was `== "autonomy"` only); builds a `pressures: dict[str, float]` from `sl.drives[*].signals.salience` (previously discarded); returns a structurally separate `"drive_state"` key alongside its existing `summary`/`debug` (autonomy_state_v2-lineage) fields. At the `build_chat_stance_inputs` call site, behind `CHAT_STANCE_DRIVE_STATE_VISIBLE` (default off, documented in `.env_example`), sets `inputs["drive_state"]` as a sibling of `inputs["autonomy"]`.
- `services/orion-cortex-exec/tests/test_chat_stance_drive_state_projection.py` (new): snapshot-source collection, the "autonomy" filter still working alongside the new "drive_state" one, and — critically — explicit assertions that `inputs["autonomy"]` never gains `pressures`/`activations`/`drive_state` keys (the non-merge requirement, independently re-verified by the orchestrator).
- `tests/test_cognitive_substrate_phase2_domain_mappings.py`: two new tests for the substrate adapter metadata change (with and without a `DriveAuditV1` present).
- `.env_example` / `README.md`: new flag documented with the `DUPLICATE`-note citation.

## Schema / bus / API changes

- Added: `CHAT_STANCE_DRIVE_STATE_VISIBLE` env flag (default `false`/off).
- No schema changes — `StateSnapshotNodeV1.metadata` is already a free-form dict; no new Pydantic fields anywhere.
- Behavior changed: `_project_autonomy_from_beliefs()`'s return dict gains a new `"drive_state"` key (additive, existing keys unchanged).
- Compatibility notes: flag defaults off, so no live behavior change until explicitly flipped.

## Env/config changes

- Added keys: `CHAT_STANCE_DRIVE_STATE_VISIBLE` (services/orion-cortex-exec).
- `.env_example` updated: yes.
- Local `.env` synced: no local `.env` exists in this worktree (confirmed via `git check-ignore` — nothing to sync); run `python scripts/sync_local_env_from_example.py` on a host with a real `.env` before this flag is usable there.
- Skipped keys requiring operator action: none — this key is safe to sync normally (not a `NEVER_SYNC_KEYS` case), defaults to off.

## Tests run

```text
# Substrate adapter tests
$ python -m pytest tests/test_cognitive_substrate_phase2_domain_mappings.py -q
7 passed

# Full chat_stance suite
$ python -m pytest services/orion-cortex-exec/tests/test_chat_stance_drive_state_projection.py \
    services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py \
    services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py \
    services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py \
    services/orion-cortex-exec/tests/test_chat_stance_shared_spine.py -q
46 passed
```
Both runs independently re-executed by the orchestrator (not just the implementing agent), using `/tmp/orion-test-venv` for the cortex-exec suite.

## Evals run

Not applicable — pure plumbing/visibility patch, no behavior to eval yet (nothing downstream consumes `inputs["drive_state"]` in a prompt template in this patch; that's explicitly the next step, not this one).

## Docker/build/smoke checks

Not run — no runtime config, ports, or Docker-relevant surface touched. Pure Python + env flag.

## Review findings fixed

- Investigated and resolved an open question from the task brief: whether `snapshot_source == "autonomy"` had ever matched anything. Confirmed real (legacy GraphDB path, off by default) rather than dead — kept the filter rather than removing it, avoiding a regression the original task brief would have caused if blindly followed.
- Implementing agent ran its own code-review pass; one soft note (no downstream prompt-template consumer yet) — expected and correct for this patch's scope (surface, don't yet consume).
- Orchestrator independently re-read the full diff and re-ran all tests rather than trusting the agent's self-report — no discrepancies found between agent intent and agent output.

## Restart required

```text
No restart required for this patch alone (flag defaults off, no behavior change).
```
If/when `CHAT_STANCE_DRIVE_STATE_VISIBLE=true` is set later:
```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low — `inputs["drive_state"]` has no prompt-template consumer yet, so flipping the flag today would add data to the `inputs` payload with no visible effect until a follow-up patch actually renders it into the prompt. Intentional scoping (this patch is the visibility seam per the accepted spec, not the full feature) — flagging so it isn't mistaken for a bug if someone flips the flag and sees no change.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/drive-audit-substrate-chat-visibility?expand=1`
