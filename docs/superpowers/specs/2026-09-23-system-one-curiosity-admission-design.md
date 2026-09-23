# System One curiosity admission — operational design

Date: 2026-09-23  
Status: implementation target for `feat/system-one-causal-influence`

## Arsonist summary

Promote **only** `curiosity_pull` from shadow observation into a **categorical
admission gate** over already-built endogenous curiosity candidates.

System One does not invent topics, choose investigation tasks, or replace
`FrontierCuriosityEvaluator`. It answers one cheaper question:

```text
Does the existing substrate evidence contain enough epistemic pull
to spend cognition on the full curiosity evaluation?
```

## Live evidence used to justify promotion

Window (not 24h — report actual):

- **n = 469** frames
- **first** `2026-09-23 17:36:26Z`
- **last** `2026-09-23 22:04:00Z`
- **span** ≈ 4.46 hours
- provider/model/question_set: `kev` / `kev-latest` / `orion.system_one.shadow.v1`
- malformed / missing answers: **0**

`curiosity_pull` argmax: **276 / 178 / 15** (levels 0 / 1 / 2).
Score mean 0.80, stdev 0.17, 252 distinct values — not flat/saturated.
Consecutive argmax persistence ≈ 68%.

`reverie_fit` argmax: **468 / 0 / 0** — **do not promote** (degenerate for current
evidence representation). Leave observational and recalibrate the question later.

`attention_interrupt` / `deliberation_need` remain observational (causal /
task-scope reasons; see original shadow spec promotion requirements).

## Operational contract for `curiosity_pull`

Use `argmax(probabilities)` over declared levels `"0"|"1"|"2"`.
Retain the full distribution in telemetry. Do **not** turn expected score into a
0..1 propensity. Do **not** aggregate across System One questions.

| Argmax | Meaning | Gate |
|--------|---------|------|
| 0 | Not worth additional cognition | `system_one_curiosity_noop` — preserve candidates + provenance; **skip** `FrontierCuriosityEvaluator` |
| 1 | Worth evaluating | `system_one_curiosity_admit` — run existing evaluator |
| 2 | Strongly worth evaluating | `system_one_curiosity_admit` — **same effect as 1**; no multiplier, priority bump, or bypass |

Levels 1 and 2 are identical for admission in this PR. Distinction is retained
for calibration.

Fallback when frame missing / expired / wrong question set / malformed /
incompatible: **`system_one_unavailable_fallback`** → legacy pre–System-One
behavior (call evaluator). Kev outage must not kill curiosity.

## Pipeline

```text
substrate evidence
→ endogenous_curiosity_candidates()
→ System One curiosity_pull admission (fresh frame)
→ noop | admit
→ FrontierCuriosityEvaluator (only if admitted or fallback)
→ existing governance / plans / outcomes
```

Hard authorities that System One never overrides:

- endogenous curiosity enable / kill switch
- candidate bounds / concept-zone restrictions
- budget / resource / capability policy
- evaluator invoke/defer/noop
- downstream governance

## Rollback

`SUBSTRATE_SYSTEM_ONE_CURIOSITY_GATE_KILL_SWITCH=true` restores legacy evaluator
behavior immediately. **Default is live (kill switch false).**  
Endogenous curiosity kill switch remains superior authority.

## Telemetry / lineage

Every seeded curiosity tick records (logs + optional `gate_json` on the
candidate-set row):

- candidate_set_id / evidence refs
- System One `frame_id`, question_set, model/provider, appraisal age
- full `curiosity_pull` probabilities + argmax level
- gate result: noop | admit | unavailable_fallback
- evaluator outcome when admitted
- decision_id / task type when present
- joinable later: investigation/run id and eventual outcome when those exist

No reward function. No cross-question soup.

## Non-goals

- No `reverie_fit` behavioral consumer (no honest discretionary seam; baseline
  floor must remain untouched)
- No behavioral use of `attention_interrupt` or `deliberation_need`
- No new state ontology / SelfState / propensity scalar
- No dependency of cognition on `GET /projections/system_one_appraisal`
- Grammar shadow projection remains; typed bus artifact is the operational payload

## Bus

Publish `SystemOneAppraisalFrameV1` on an explicit operational channel
(`orion:system_one:appraisal`) for future cross-service consumers. Same-service
curiosity gate reads the validated latest frame from the substrate store (or
in-process cache), not the debug HTTP surface.

## Acceptance checks

- Levels 0 → skip evaluator; 1 and 2 → admit evaluator (same)
- Missing/stale/malformed → fallback to legacy
- Endogenous kill switch still wins
- System One cannot mint candidates
- No accidental consumers for attention_interrupt / deliberation_need / reverie_fit
- Provenance in gate telemetry
- Eval script can report blocked vs admitted and conditional invoke rates
