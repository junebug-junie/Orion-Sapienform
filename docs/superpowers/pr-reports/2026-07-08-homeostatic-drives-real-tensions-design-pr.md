# PR: Homeostatic drives + real deviation-driven tensions (design)

**Status:** DESIGN — spec + implementation plan only. No runtime code changes. The substrate fix is not built by this PR; this PR captures the contract so the build lands as a follow-up with tests + live verify.

## Summary

- Adds the **homeostatic-drives design spec**: replaces the `soft_saturate` fixed-point pressure math (which pins all 6 drives to ~0.731 as a tick-cadence artifact) with a **cadence-invariant leaky integrator** that rests at zero.
- Adds a **deviation gate** (EWMA baseline → z-score → impulse) so tensions fire on *change*, not on the 55/s `scene_state` presence flood.
- Sources real tensions from already-built rails: **OrionSignalV1** bus (`orion:signals:*`), **failure events** (`system:error`, `exec_step_failed`, `rdf:error`, `vision:edge:error`), and **equilibrium health** — not synthetic ticks.
- Structural, non-lexical mapping via `config/autonomy/signal_drive_map.yaml` (typed `signal_kind → drive_impacts`). No keyword cathedral.
- Adds the **flags-on implementation plan** (10 tasks), reusing `is_stub_signal` and `compute_tick_attribution`.

## Outcome moved

Nothing at runtime yet (design PR). The outcome this *unblocks*: drive pressure becomes real, deviation-driven, and cadence-invariant — the precondition for the endogenous-origination / φ-reward / internal-economy / voluntary-attention arc, all of which currently assume a functioning substrate that does not exist on main.

## Current architecture (before)

- `orion/spark/concept_induction/drives.py:36` — `_soft_saturate` = `1 - exp(-gain·p)`; stable non-zero fixed point ~0.731 → all drives inflate identically under frequent ticks. Pressure is a cadence artifact, not cognition.
- Tensions fire ~0.064% of ticks; dominant drive was an alphabetical `max(sorted(...))` artifact on zero-vectors.
- Drive-audit persistence to Fuseki broke ~June 19 (loop still publishes on the bus; rdf-writer persistence is what stalled) — flagged separately, not fixed here.

## Architecture touched (by the plan, when built)

- `orion/spark/concept_induction/drives.py` — leaky integrator replacing `_soft_saturate`.
- `config/autonomy/signal_drive_map.yaml` — new structural signal→drive map.
- Deviation-gate module (EWMA/z-score) feeding `TensionEventV1`.
- Adapters over OrionSignalV1 + failure channels; reuse `orion/signals/stub_detection.py::is_stub_signal`.
- Flags: `ORION_HOMEOSTATIC_DRIVES_ENABLED`, `ORION_DRIVE_LEAKY_MATH_ENABLED` (latter is behavior-changing to `policy_act`).

## Files changed (this PR)

- `docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md`: the cohesive design spec.
- `docs/superpowers/plans/2026-07-07-homeostatic-drives-real-tensions.md`: flags-on implementation plan, 10 tasks, live runtime-truth acceptance gate.
- `docs/superpowers/pr-reports/2026-07-08-homeostatic-drives-real-tensions-design-pr.md`: this report.

## Schema / bus / API changes

- Added (planned, not in this PR): `config/autonomy/signal_drive_map.yaml`; two env flags.
- Reused: `OrionSignalV1`, `TensionEventV1`, `DriveStateV1`, `DriveAuditV1`, `compute_tick_attribution`, `is_stub_signal`.
- Compatibility: leaky math gated behind `ORION_DRIVE_LEAKY_MATH_ENABLED`; default path unchanged until Task 10 flips it live with verification.

## Env/config changes

- This PR: none (docs only).
- Planned by the plan: two flags added to the relevant `.env_example` + local `.env` sync via `python scripts/sync_local_env_from_example.py`.

## Tests run

```text
None — docs-only PR. The plan's Task N ships gate tests + eval + live smoke; those run in the implementation PR.
```

## Review findings fixed

```text
N/A for the design PR. Code review runs on the implementation PR per the contract.
```

## Restart required

```text
No restart required (docs only).
```

## Risks / concerns

- Severity: medium — `ORION_DRIVE_LEAKY_MATH_ENABLED` changes `policy_act` pressure reads. The plan gates it and makes Task 10 a live enable+verify, not a blind flip.
- The flat-0.731 bug remains on main until this plan is *built and merged*. This PR does not fix it; it authorizes the fix.

## PR link

<to be filled by `gh pr create` / GitHub UI>
