# Unified Cognitive Substrate — Phase 12: Review Telemetry and Calibration

## Why Phase 12 exists

Phase 11 made bounded runtime review execution possible, but operators still needed structured answers about review quality, churn, and tuning posture. Phase 12 adds telemetry capture, bounded introspection, and advisory-only calibration guidance.

## Execution vs telemetry

- **Execution (Phase 11):** select one eligible queued item and run one bounded cycle.
- **Telemetry (Phase 12):** record what happened, summarize trends, and generate advisory recommendations.

Telemetry never mutates live review policy directly.

## Typed telemetry records

Each runtime attempt emits one `GraphReviewTelemetryRecordV1` entry including:

- invocation surface and queue item selection reason
- zone / subject / priority context
- cycle budget before/after
- suppression and termination transitions
- consolidation outcomes
- frontier follow-up usage
- final runtime outcome and duration

## Operator introspection

`GraphReviewTelemetryRecorder` supports deterministic bounded query filters by:

- surface
- zone
- subject
- outcome
- follow-up usage
- time window and limit

Summaries provide outcome/zone/surface/follow-up counts plus average cycle and runtime metrics.

## Advisory calibration

`GraphReviewCalibrationAnalyzer` converts summaries into advisory-only recommendations such as:

- increase cadence interval when requeue churn is high
- increase suppression threshold when suppression is too frequent
- decrease max cycles when failure ratios are high
- hold when insufficient data or balanced behavior

No automatic policy updates are performed.

## Safety posture

- runtime remains single-cycle bounded
- telemetry recorder failures are non-fatal
- strict-zone protections remain enforced by runtime execution rules
- calibration remains read-oriented and operator-facing only

## Forward path

Later phases can add durable storage backends and richer operator APIs while keeping calibration advisory and bounded.
