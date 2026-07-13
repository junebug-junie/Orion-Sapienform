# PR report — single-consumer channel gate + dead equilibrium config removal

Date: 2026-07-13
Branch: `feat/single-consumer-channel-gate`

## Summary

- New deterministic live-bus gate: every channel marked `single_consumer: true` in `orion/bus/channels.yaml` must have exactly one live subscriber, or the gate fails. This turns the PR #994 class of bug (two containers both subscribed to `orion:verb:request`, every LLM call executed twice) into a failing check instead of a live incident.
- 31 request/dispatch channels annotated `single_consumer: true`, including one channel that was live on the bus but missing from the catalog entirely (`orion:exec:request:CollapseMirrorService`).
- A catalog test enforces the annotation going forward: any `kind: request` entry with one consumer and a concrete name must be annotated (empty exemption list), so new request channels cannot drift out of coverage.
- Code review found the first cut of the gate failed open (redis-cli error replies and parse drift degraded to benign 0-subscriber warnings); fixed fail-closed at the fetch seam.
- Removed the dead `channel_cortex_orch_request` / `CHANNEL_CORTEX_ORCH_REQUEST` config key from orion-equilibrium-service — defined since inception, never read by any code, and misleadingly implied equilibrium touches `orion:verb:request`.

## Outcome moved

The duplicate-LLM-execution failure mode is now regression-gated. Before: a second subscriber on any execute-once channel was invisible until someone traced duplicate `request_id`s in production (as happened in PR #994). After: `make check-single-consumer-channels` fails with the exact channel and count. First live run verified the whole mesh clean: 31 channels, 28 at exactly 1 subscriber, 3 warnings for idle services, 0 violations.

## Current architecture

The bus is Redis pub/sub — every subscriber executes every message; there are no competing-consumer queues. Execute-once semantics (RPC request channels served by `Rabbit`, dispatch channels consumed by a single `Hunter`) are purely conventional: nothing declared which channels carry those semantics and nothing checked subscriber counts. The catalog (`orion/bus/channels.yaml`) documented producers/consumers but had no single-consumer concept, and one live RPC channel was missing from it. `scripts/platform/audit_channels.py` exists but is static analysis only — it never touches the live bus.

## Architecture touched

- Contract: `orion/bus/channels.yaml` (new `single_consumer: true` field, 31 entries + 1 new entry)
- Gate: `scripts/check_single_consumer_channels.py` (new), `Makefile` target `check-single-consumer-channels`
- Tests: `tests/test_single_consumer_channels_gate.py` (new, 19 tests)
- Service config hygiene: `services/orion-equilibrium-service` (dead key removed)

## Files changed

- `orion/bus/channels.yaml`: 31 channels annotated `single_consumer: true`; added missing `orion:exec:request:CollapseMirrorService` entry (producer orion-cortex-exec, consumer orion-collapse-mirror, mirrors the MetaTagsService entry convention).
- `scripts/check_single_consumer_channels.py`: new gate. Reads the catalog (plain `yaml.safe_load`; missing PyYAML is a loud exit 2), queries live counts via `redis-cli PUBSUB NUMSUB` subprocess (no new Python dependency), fails on >1 subscriber, warns on 0 (`--strict-zero` promotes to failure). Fail-closed: Redis error replies (NOAUTH/ERR/LOADING/…) and any requested channel absent from the parsed reply are infra errors (exit 2) — never counted as 0. Exit codes: 0 clean, 1 violation, 2 infra.
- `tests/test_single_consumer_channels_gate.py`: pure decision core, NUMSUB parsing, fail-closed contract, main() exit paths, catalog wiring, and the annotation-coverage enforcement test. No live bus needed.
- `Makefile`: `check-single-consumer-channels` target beside `check-inner-state-registry`.
- `scripts/README.md`: gate documented (what it guards, how to run, expected output).
- `services/orion-equilibrium-service/app/settings.py`: removed dead `channel_cortex_orch_request` field.
- `services/orion-equilibrium-service/.env_example`: removed `CHANNEL_CORTEX_ORCH_REQUEST` line; comment trimmed accordingly.
- `docs/platform_routing_wiring_map.md`: bullet that flagged the key as drift-prone now records its removal.

## Schema / bus / API changes

- Added: `single_consumer: true` catalog field (documentation + gate input; no runtime payload change). New catalog entry `orion:exec:request:CollapseMirrorService` (channel already live; `GenericPayloadV1`, already registered in `orion/schemas/registry.py`).
- Removed: none at runtime.
- Renamed: none.
- Behavior changed: none at runtime — the gate is read-only against the bus (`PUBSUB NUMSUB` only).
- Compatibility notes: unknown catalog fields are ignored by existing consumers of channels.yaml (verified `scripts/platform/_common.load_channels_catalog` and audit scripts tolerate it; `tests/test_inner_state_registry_gate.py` still passes).

## Env/config changes

- Added keys: none.
- Removed keys: `CHANNEL_CORTEX_ORCH_REQUEST` (orion-equilibrium-service; dead, never read).
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-equilibrium-service/.env_example`).
- local `.env` synced: yes — targeted removal of the dead line from `services/orion-equilibrium-service/.env` (deliberately not via `sync_local_env_from_example.py`, which is additive-oriented and has previously reverted intentional live overrides; a one-key deletion is safer done surgically). Verified 0 matches remain.
- skipped keys requiring operator action: none.

## Tests run

```text
orion_dev/bin/python -m pytest tests/test_single_consumer_channels_gate.py \
  tests/test_inner_state_registry_gate.py \
  services/orion-equilibrium-service/tests -q
34 passed, 6 warnings in 5.68s
  (19 gate tests + 8 inner-state gate regression + 7 equilibrium)
```

## Evals run

```text
No eval harness applies: the gate is fully deterministic (catalog parse +
subscriber-count arithmetic); there is no model-quality surface to eval.
The live smoke below is the behavioral proof.
```

## Docker/build/smoke checks

```text
ORION_BUS_URL=redis://100.92.216.81:6379/0 python3 scripts/check_single_consumer_channels.py
→ 31 channels checked: 28 OK at exactly 1 subscriber; WARN 0 for
  orion:conversation:request, orion:exec:request:ContextExecService,
  orion:exec:request:CouncilService (services idle/down — liveness, not
  duplication); 0 violations; exit 0.

No Docker changes: no service runtime code touched (equilibrium change is a
dead-field deletion; settings use extra="ignore" so stale env values are
harmless).
```

## Review findings fixed

- Finding: gate failed open — redis-cli exits 0 on Redis error replies (NOAUTH/WRONGPASS/ERR/LOADING) with the error on stdout; the parse yielded `{}`, every channel was backfilled to 0, and the gate printed all-WARN and exited 0 while checking nothing (empirically confirmed by reviewer on redis-cli 7.0.15).
  - Fix: error-reply first lines raise; any requested channel absent from the parsed reply raises (NUMSUB always echoes every requested channel, so absence proves a non-NUMSUB reply). Both exit 2.
  - Evidence: `test_fetch_live_counts_error_reply_raises_not_zero_counts`, `test_fetch_live_counts_missing_channel_in_reply_raises`.
- Finding: one stray stdout line shifted the fixed-offset pair parsing, converting a real 2-subscriber violation into a benign 0-subscriber warning (confirmed by reviewer simulation).
  - Fix: same seam — the missing-channel check makes any parse drift an infra error; the `setdefault(ch, 0)` backfill was removed.
  - Evidence: the noisy-banner test above feeds exactly that scenario.
- Finding: 11 structurally identical request channels left unannotated, and nothing enforced the annotation for future channels.
  - Fix: annotated all 11 (verified each: `kind: request`, exactly one consumer, concrete name); added `test_every_request_kind_single_consumer_channel_is_annotated` with an empty exemption list.
  - Evidence: live run now covers 31 channels; enforcement test passes with zero exemptions.
- Finding (nit): `main()` re-derived FAIL/WARN/OK by hand, so printed status and exit code could drift.
  - Fix: `evaluate_counts` returns per-channel statuses; both derive from the one function.
- Finding (nit): shared catalog loader's regex fallback silently drops `single_consumer` when PyYAML is absent, producing a misleading "annotation missing?" diagnostic.
  - Fix: plain `yaml.safe_load` only; missing PyYAML is a clear exit-2 infra error.
- Finding (nit): TTY-only redis-cli output-format parsing branches were unreachable under `capture_output=True`.
  - Fix: removed, with their test.
- Finding (nit, not fixed here): the `scripts/platform/` stdlib-shadow `sys.path.pop` workaround is hand-copied across ~12 scripts, this one included. Repo-wide cleanup (rename `scripts/platform/` or a shared helper) is a separate chore — out of scope for this patch.

## Restart required

```text
No restart required. The gate is a repo script; no running service reads the
removed equilibrium key (never read by code) or the new catalog field. The
equilibrium container will simply boot without the dead field on its next
scheduled rebuild — no action needed now.
```

## Risks / concerns

- Severity: low
  - Concern: `single_consumer: true` semantics could drift if a channel legitimately gains a second, role-distinct consumer (fan-out by design).
  - Mitigation: that change must flip the annotation in the same patch — the gate failing is exactly the review conversation we want; the enforcement test's exemption list is the documented escape hatch.
- Severity: low
  - Concern: the 3 WARN-at-0 channels (conversation, context-exec, council) will keep warning while those services stay idle; warning fatigue could hide a real new WARN.
  - Mitigation: `--strict-zero` exists for a full-mesh-up check; warnings are listed per channel so new ones are visible in diffs of gate output.
- Severity: low
  - Concern: gate requires `redis-cli` on the runner.
  - Mitigation: missing binary is a loud exit-2 with an install hint; no silent pass.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/single-consumer-channel-gate
