# Phase 11: Offline Evaluation and Calibration for Endogenous Runtime

## Scope

Phase 11 is offline/read-oriented. It does **not** broaden live runtime surfaces or auto-apply threshold changes.

The same runtime surfaces remain:

- `chat_reflective_lane`
- `operator_review`

## Typed evaluation/calibration contracts

New typed contracts (`orion/core/schemas/endogenous_eval.py`) provide a canonical offline analysis shape:

- `EndogenousEvaluationRequestV1`
- `EndogenousMetricSummaryV1`
- `PromotionCalibrationSummaryV1`
- `ReasoningSummaryCalibrationSummaryV1`
- `EndogenousCalibrationRecommendationV1`
- `EndogenousCalibrationProfileV1`
- `EndogenousEvaluationResultV1`

These models capture evaluation windows/filters, metrics, recommendations, sample-size warnings, and advisory profile export.

## Offline evaluator

`orion/reasoning/evaluation.py` adds `EndogenousOfflineEvaluator` over durable runtime execution records.

Metrics include deterministic bounded rates/counts for:

- trigger/no-op/suppress/failure
- cooldown/coalesce/debounce
- mentor selected/invoked/disabled-suppressed
- workflow family rates (contradiction/concept/autonomy/reflective)
- materialization averages
- repeated subject density

Evaluator also computes bounded promotion and summary calibration summaries.

## Calibration engine

`orion/reasoning/calibration.py` adds `EndogenousCalibrationEngine`.

Recommendations are advisory-only and deterministic:

- hold on insufficient sample size
- stricter thresholds when over-triggering
- looser thresholds when under-triggering
- cooldown adjustments when cooldown hit rate is high
- mentor-gating recommendation when mentor-disabled suppressions are high
- summary/promotion guidance from summary/promotion calibration signals

No live threshold mutation occurs in this phase.

## Operator-facing outputs

`services/orion-cortex-exec/app/endogenous_runtime.py` adds:

- `run_endogenous_offline_evaluation(request)` returning:
  - typed evaluation result JSON
  - markdown operator report
  - advisory profile export payload

This gives operators inspectable calibration recommendations before any manual adoption.

## Safety posture

- Evaluation and calibration are read-oriented.
- No mutation of canonical reasoning state.
- Recommendations are explicit `advisory_only=True`.
- Runtime flow and live safety gates are unchanged.

## Manual adoption path

A later phase may manually apply generated profile overrides after review.

Phase 11 only produces reproducible analysis + recommendations, not runtime policy changes.
