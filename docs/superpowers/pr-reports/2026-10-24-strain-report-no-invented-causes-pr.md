# Strain report: say how the GPU queue relates to the strain

## Summary

- On 10-23 at 22:47 UTC, Orion's first strain report told Juniper "the cabinet is hot at 33.6°C. This has also queued up nine GPU jobs." Only `cabinet_hot` was active, and nothing measured a link between the heat and the queue. The prompt listed the queue right after the cause and never said how the two relate, so the model filled the gap.
- The prompt now says it outright, with three cases:
  - The queue is one of the causes.
  - The queue is what keeps a holding strain from clearing (`strain_holding`: load still at or above the floor, not yet sustained).
  - The queue is NOT a cause right now (no queue cause, or `strain_clearing`).
- Orion's own deferrable work in the count is named as not counting as load.
- New rule: each cause is measured on its own, so never say one caused another.
- Follow-up to #2884 (strain loop part 3).

## Outcome moved

The strain report stops inviting a made-up causal link. This is UNVERIFIED on a live message, which needs the next long strain episode.

## Files changed

- `services/orion-hub/scripts/strain_report.py`:
  - `StrainReportFacts.queue_holding` (from `strain_holding`), plus `as_dict`.
  - The queue line in `build_strain_report_prompt`.
  - The no-causal-links instruction.
- `services/orion-hub/tests/test_strain_report_outreach.py`: 4 tests covering the cause, holding, clearing and deferrable cases.

## Schema / bus / API changes

None. `queue_holding` is added to the internal facts dataclass and its trace dict.

## Env/config changes

None.

## Tests run

```text
services/orion-hub/tests/test_strain_report_outreach.py      27 passed
strain + endogenous_outreach + decisions + provenance + arousal readers   364 passed
check_definition_drift --gate, check_body_economy_contract: pass; git diff --check clean
```

## Evals run

```text
None for model behaviour: whether the model obeys can only be judged on a live report.
```

## Review findings fixed

- **Finding (material):** in `strain_holding`, the new "NOT a cause" line would itself be false, because the queue is what holds the strain.
  - **Fix:** a separate branch says the queue keeps the strain from clearing.
  - **Evidence:** `test_prompt_says_the_queue_keeps_a_holding_strain_from_clearing`.
- **Finding (minor):** the raw queue count included deferrable self-work.
  - **Fix:** it is now named as not counting as load.
  - **Evidence:** `test_prompt_names_deferrable_self_work_in_the_count`.
- **Finding (minor):** "heat did not cause the jobs" can sometimes be true (heat sheds hold entries back). Kept as a "don't claim a link" rule, because the facts never carry the shed count.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** prompts are guidance, not a gate. The model may still invent links.
  - **Mitigation:** the fixed "Strain record" line under every message stays exact.
  - **Follow-up if it recurs:** a deterministic check on the composed text, not more prompt.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
