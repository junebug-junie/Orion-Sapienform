## Summary

Design only. Nothing is built or run.

- Rewrites the Family C first-experiment proposal (#3482) so it works through the body economy and the tiers, not around them.
- Experiment 1 (recommended first): randomly hold Orion's AI Town sessions (tier 2) for 40 min, using a run-cap override with a TTL. The outcomes are the GPU3 watts and heat, plus the slowdown that tier-1 work on the `fast` lane feels.
- Experiment 2 (later): a time-limited 150 W cap on GPU3. It runs as a TTL override inside the existing `orion-gpu-power-limits` root job. It does not add a new root helper.
- Live check found that the old 120 W ON arm was impossible. The V100-SXM2 minimum power limit is 150 W.
- 4 yes/no questions for Juniper.

## Outcome moved

The first real Family C experiment now has a safe actuator that it can randomise. It never adds load and never touches tier 1. It has a power estimate from live data and a manipulation check that would have caught the 10-29 contamination (#3484).

## Current architecture

The old proposal was a new root helper that capped circe GPU3 at 120 W. It measured heat only. GPU3 is the `fast` lane and also hosts Kev. Nine tier-1 families run only on it.

## Architecture touched

None. This is a doc. The design names the seams it would touch:
- the experiment runner in orion-prediction
- the town controller in orion-embodiment
- a new pure helper, `orion/physiology/run_cap_override.py`
- `families.yaml` and `organs.yaml`
- for Experiment 2: the power-limits root job and the lane controller

## Files changed

- `docs/superpowers/specs/2026-11-16-family-c-first-experiment-economy-native.md`: the design.
- `docs/superpowers/pr-reports/2026-11-16-family-c-first-experiment-economy-native-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed only: `RunCapOverrideV1`, `TemporalSourceKind` `experiment_result`, family `prediction.experiment`).
- Removed / Renamed / Behavior changed: none.

## Env/config changes

- None. `.env_example` not touched; no sync needed.

## Tests run

```text
.venv/bin/python -m pytest -q tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py
10 passed
git diff --cached --check: clean
```

## Evals run

```text
None (design doc). Patch 0 of the design is a read-only replay eval that recomputes the power numbers.
```

## Docker/build/smoke checks

```text
Not applicable (docs only).
Live read-only data pulled for the design (Postgres conjourney, 2026-11-02..16; circe nvidia-smi query).
```

## Review findings fixed

An independent review subagent read the doc against the code. Every must-fix finding is fixed in the doc.

- Finding: race between "eligible" and setting the OFF key (the town ticks every 300 s, the runner polls every 30 s); some decisions are missed; the runner copied the town's env caps.
  - Fix: a pending hold from 60 s before eligibility. Eligibility now comes from a new embodiment `/health` field `town_start_eligible_at`.
- Finding: a 16-night run dies on any rebuild (`recover()`). The key cleanup knows only `vision.focus`. The analysis code assumes fixed clock blocks, one series and a free permutation.
  - Fix: one experiment per night, pooled. Key cleanup is by action. The event-start blocks, the multi-series outcomes and the restricted permutation are now listed in the files to touch.
- Finding: the `max_open_sessions` knob does not exist.
  - Fix: an honest boolean knob, `hold_starts`.
- Finding: the `max_hold_s` guard does nothing if `set_at` is refreshed.
  - Fix: `set_at` is fixed at the block start, and the controller checks `now - set_at`. The same rule applies to Experiment 2.
- Finding: Experiment 2 ended an ON block early when tier-1 work queued, which hides the cost being measured.
  - Fix: refuse only at decision time. The analysis window is fixed (intention-to-treat). Heat is the only early end.
- Finding: in Experiment 2 a stuck cap could last up to 1 h.
  - Fix: a 180 s refreshed TTL plus a `set_at` limit of 25 min.
- Finding: Experiment 2's root-job mechanics: the install step, the directory lifetime, root reading a file the container writes, and an override left in place on `ConfigError`.
  - Fix: a re-install step, `tmpfiles.d`, `O_NOFOLLOW` with size, owner and schema checks, and an `applied.json` state file with restore on any error.
- Finding: the binding producer was not registered, and the doc's "no answer = no block" differs from the repo convention.
  - Fix: `BINDING_PRODUCERS` is added. The difference is now stated as deliberate.
- Smaller findings also fixed:
  - time-of-night confounder: Patch 0 simulates the decision process, and the analysis uses the time-of-day profile
  - abort on the first contaminated block
  - manual sessions are flagged
  - "nine" tier-1 families corrected to seven plus `actions.journal`
  - `retry_after_sec` is the Redis TTL
  - shadow wants and the self-monitor side effect are noted
  - Kev blind spot noted
  - the series source kinds are named
- Evidence: the review report is in the session. The diff is in the second doc commit on this branch.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
- Concern: the GPU3 effect of a town session reads +13 W one way and +3 W another (not randomised). At 56 blocks per arm, the smallest effect the experiment can detect is about 8 W, so the result may be an honest null.
- Mitigation: Patch 0's replay recomputes the numbers. It is a go only if the smallest detectable GPU3 effect is 10 W or less.
- Issues: #3482 updated with a comment. It keeps `needs-juniper`. No issues were opened or closed.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
