# Social-state inspection

## Purpose

Social-state inspection provides a compact, operator-facing snapshot of the social-room cognition state that shaped a turn.

It is designed to answer:

- what context was selected for the turn
- what was softened or excluded
- which claim, commitment, routing, repair, floor, calibration, freshness, resumptive, and epistemic signals mattered
- why those decisions mattered

## Inspection model

The inspection surface is built from three typed models:

- `SocialInspectionSnapshotV1` — top-level snapshot for a room/thread/participant
- `SocialInspectionSectionV1` — bounded section for a single cognition surface
- `SocialInspectionDecisionTraceV1` — compact “why this mattered” trace

Sections are intentionally summary-first:

- selected context
- important softened/excluded state
- short traces rather than raw dumps
- freshness / confidence hints where useful

## Access surfaces

Two existing operator/debug seams carry the inspection snapshot:

1. **Social-memory endpoint**: `GET /inspection`
   - returns the current bounded inspection snapshot for a room / participant
   - uses the same stored summary surfaces as `/summary`

2. **Hub routing debug payload**
   - `build_chat_request(...)` now adds `social_inspection` to the existing debug payload
   - this keeps the live turn path inspectable through the current Hub debug flow without introducing a new UI framework

## Safety boundary

- blocked / private / sealed material is omitted from inspection
- pending, deferred, or declined artifact dialogue is shown only as omitted/non-active state
- inspection does not enable tools or actions
- inspection stays within the same bounded social-room safety surface; it does not widen recall exposure

## Relationship to social-room cognition flow

Inspection sits on top of the current social-room stack:

- social-memory builds continuity, context windows, freshness, calibration, claims, deliberation, floor, and resumptive artifacts
- Hub adds live routing / repair / epistemic decisions
- inspection compacts those inputs into operator-readable sections and decision traces

This makes the turn path easier to audit without changing the reply logic itself.
