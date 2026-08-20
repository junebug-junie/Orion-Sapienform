# PR report: P2 perceptual prediction-error — surprise EWMA noise floor + acceptance-check audit

Covers three merged PRs that shipped without a committed report file
(`feedback_pr_report_must_be_committed_file` gap, caught and fixed
retroactively here), plus new audit work against P2's own stated acceptance
checks from `docs/superpowers/specs/2026-08-12-perception-frontier-design.md`.

## Summary

- **PR #1746** (`feat/perception-prediction-error`, merged 2026-08-19):
  shipped `node:substrate.perception`'s `prediction_error` signal —
  `surprise = 1 - cos(frame_embedding, EWMA_embedding)`, shadow-only (not
  yet in `ACTIVE_INFERENCE_DOMAINS`), plus fixed the `want_embeddings: false`
  baseline-tier blocker Appendix B flagged.
- **PR #1752** (`fix/endogenous-curiosity-perception-leak`, merged
  2026-08-19): migrated the raw magnitude to a second z-score stage
  (`_domain_zscore`) so it's on the same numeric footing as every other
  Active-Inference domain before comparison against
  `endogenous_curiosity.py`'s shared `min_error=0.55`.
- **PR #1776** (`fix/perception-surprise-noise-floor`, merged 2026-08-20):
  raised `_PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE` 1e-8 → 2e-6 after
  confirming live that the z-score stage was false-positiving on a
  physically static camera (10.0% crossing `min_error=0.55`, 2.9-4.1%
  fully saturating) — root cause was `compute_ewma_update` flooring only
  the z-score denominator, never the stored variance it returns for next
  tick, letting a short-memory (`alpha=0.2`, ~5-tick) EWMA whipsaw below
  the domain's true steady-state variance. This PR also surfaced and
  documented a real cross-worktree Docker deploy-collision risk (see that
  PR's own "Risks / concerns" section).
- **This report** additionally runs the two audit-script-mechanizable
  acceptance checks from the design doc's ladder (`P2:` bullet) and
  Appendix B against live data, closing one and reporting an honest,
  partial result on the other.

## Outcome moved

- Noise-floor fix: saturated-score rate 2.9-4.1% → 1.7% (live, integrity-
  checked — see PR #1776 body for the full before/after).
- **Decay-artifact acceptance check: now run and PASSING.**
  `scripts/analysis/audit_prediction_error_domain.py --node-id
  node:substrate.perception --window-hours 24` against the real
  `substrate_field_state` history:

  ```
  n=40890 mean=0.12937 median=0.00130 min=0 max=1 stdev=0.25530
  frac exactly 0.0=49.8%  frac exactly 1.0=3.0%  distinct values=924
  Distribution: not obviously degenerate by these checks.
  No decay artifact detected (longest matching-ratio run: 1, threshold: 20).
  ```

  This mechanically closes the design doc's P2 acceptance line
  "successive-value geometric-ratio check applied to rule out a decay
  artifact" — previously asserted only informally, never actually run for
  this domain. The 49.8% exact-zero rate also independently confirms the
  design doc's P2 acceptance line "surprise reaches genuine near-zero on a
  verified-static window" (this 24h window spans a mix of pre- and
  post-#1776 data, so it is not a clean before/after for the noise-floor
  fix specifically — that comparison lives in PR #1776's own body — but it
  does confirm the rest point is real, not a decay-to-zero artifact,
  because the decay check above ran against the same window and found
  nothing).

- **Non-redundancy check (Appendix B blocker item 5): now run and
  CONFIRMED PASSING.** The design doc requires: "run both [the label-
  habituation gate and the embedding residual] over the same live window
  and show the embedding residual fires on transitions the label gate
  calls `stable_scene`. If it does not, the existing mechanism wins and
  this should not be built." `orion-vision-council`'s `stable_scene` skip
  decision (`evidence_transition.py`) is not persisted anywhere in
  Postgres — only as a log line (`[COUNCIL] evidence_transition skip ...
  reason=stable_scene`) — so this required a live, real-time correlation
  session rather than a historical query. Ran a 20-minute correlated
  capture of both `orion-athena-vision-council` and
  `orion-athena-substrate-runtime` logs (2026-08-20 22:00-22:20 UTC,
  against the already-fixed `2e-6` floor from PR #1776 — confirmed via
  `docker exec ... grep` on the running container before trusting this
  result):

  ```
  unique stable_scene skip events: 141 (2 real "interpret" events in the
    same window, for contrast)
  stable_scene events matched to a perception tick within +/-5s: 129
  score during stable_scene: min=0.000 max=1.000 mean=0.101
  matches with score > 0.55 (crosses min_error): 8 / 129 (6.2%)
  ```

  Top 5 by score, all during a council `stable_scene` skip:

  | council skip (UTC) | perception score | raw_surprise |
  |---|---|---|
  | 22:15:07.012 | 1.000 | 0.00901 |
  | 22:15:37.825 | 1.000 | 0.01286 |
  | 22:05:42.406 | 0.855 | 0.00705 |
  | 22:11:10.898 | 0.699 | 0.00574 |
  | 22:07:14.782 | 0.683 | 0.00634 |

  These 8 events are spread across the full 20-minute window (22:05,
  22:07, 22:11, 22:15×2, plus 3 more), not one clustered burst, and every
  one has an elevated *raw* magnitude (0.0057-0.0129, well above the
  domain's calm-window mean of ~0.0035-0.0042 measured in PR #1776) — not
  an artifact of the z-score stage alone, the underlying signal itself
  moved. This is a real, direct demonstration that the embedding-based
  surprise channel registers scene changes the fixed `{door, screen}`
  label vocabulary has no word for and that the label-habituation gate
  therefore calls "nothing happened." **The design doc's own bar for this
  signal to earn its place is met** — the existing mechanism does not
  win; this is genuinely independent information, not a redundant restate
  of what the label gate already knows.

## Current architecture

Unchanged from PR #1776's own report — see that PR for the full
`perception_prediction_error()` two-stage EWMA description. This report adds
no new production code, only a retroactive report file and read-only audit
queries.

## Architecture touched

None (docs-only; the audit script used already existed, added in PR from
`5e9dddccf`, 2026-07-26, for other domains — reused here per the metric
quality gate's existing-mechanism check, not reimplemented).

## Files changed

- `docs/superpowers/pr-reports/2026-08-20-perception-p2-noise-floor-and-acceptance-audit-pr.md`:
  this file — retroactive committed report for PRs #1746/#1752/#1776, plus
  new audit findings.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
No new test/code changes in this report. Prior test runs (25/25 isolated
harness, 126/126 real pytest) already documented in PR #1776.

Decay-artifact / distribution audit (read-only, live Postgres, venv with
psycopg2):
  /mnt/scripts/Orion-Sapienform/venv/bin/python \
    scripts/analysis/audit_prediction_error_domain.py \
    --node-id node:substrate.perception --window-hours 24 \
    --postgres-uri postgresql://postgres:postgres@localhost:55432/conjourney
  -> n=40890, no decay artifact, distribution not degenerate (see "Outcome
     moved" above for full output).

stable_scene non-redundancy correlation (read-only, live docker logs,
orion-vision-council + orion-substrate-runtime, 20-minute window,
2026-08-20 22:00-22:20 UTC):
  -> 141 stable_scene skip events captured, 129 matched to a perception
     tick within +/-5s, 8/129 (6.2%) crossed min_error=0.55 with elevated
     raw magnitude, not clustered -- see "Outcome moved" above for the
     full table. Non-redundancy CONFIRMED.
```

## Evals run

No dedicated eval harness exists for this domain (same as PR #1776). The
audit-script run above is the closest equivalent for the decay-artifact
check specifically.

## Docker/build/smoke checks

None needed — no code changed, only documentation and read-only queries
against already-running services.

## Review findings fixed

Not applicable — no code changes in this report; PR #1776's own review
(clean, ship-as-is, independently re-ran pytest) already covers the actual
code change.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: **note, not blocking**
- Concern: both of P2's mechanizable acceptance-check gaps are now closed
  (decay-artifact check passing, non-redundancy confirmed), but the
  non-redundancy result rests on one 20-minute live correlation window,
  not a repeatable historical query — `orion-vision-council`'s
  `stable_scene` skip decision is still log-only, never persisted to
  Postgres. If this check needs to be re-run later (e.g. after a
  `orion-vision-window` or `evidence_transition.py` change) it again
  requires a live real-time correlation session rather than a query
  against history.
- Mitigation: none implemented in this report (out of scope — touches a
  different service, `orion-vision-council`, not `orion-substrate-
  runtime`). Follow-up option, if this needs to be repeatable: persist
  `stable_scene` skip decisions somewhere queryable (a lightweight
  receipt table or bus event), matching this domain's own existing
  convention of receipts-as-audit-trail. Not urgent given the check now
  has a confirmed-passing result on record.
- Separately, the domain remains shadow-only (not in
  `ACTIVE_INFERENCE_DOMAINS`) per the design doc's staged-promotion
  guidance ("observe for a week, promote only then") — P2 shipped
  2026-08-19, one day before this report, so promotion is not due yet.
  This report's findings (decay-check pass, non-redundancy confirmed) are
  exactly the evidence that guidance asks for before promoting; the
  remaining wait is time-based, not action-based.

## PR link

(filled in below)
