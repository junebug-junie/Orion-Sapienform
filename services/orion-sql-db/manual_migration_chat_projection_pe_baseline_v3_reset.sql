-- One-shot deploy step for chat prediction-error definition v3 (2026-09-29).
--
-- chat_prediction_error now averages over only the turns a batch touched
-- (orion/substrate/prediction_error.py). Its running EWMA lives on the chat
-- projection row and still holds v2-scale numbers (raw deltas ~300x smaller).
-- Scored against that leftover, the first v3 turns read as extreme surprise:
-- the replay showed 3 false 1.0 readings and 11 turns before it settles, and
-- those receipts would seed Candidate A's freshly reset chat baseline.
--
-- Zeroing the three fields gives v3 a normal cold start (first turn reads 0.0
-- and seeds the average). Turns and everything else on the row are untouched.
--
-- Run ONCE, while orion-substrate-runtime is stopped, right before starting the
-- v3 build (see docs/superpowers/pr-reports/2026-09-29-attention-input-honesty-pr.md).
-- Re-running later only costs a cold start; it destroys no turn data.

BEGIN;

-- Show what is being replaced (keep this output with the deploy notes).
SELECT projection_id,
       projection_json -> 'prediction_error_baseline_ewma'     AS ewma,
       projection_json -> 'prediction_error_baseline_ewma_var' AS ewma_var,
       projection_json -> 'prediction_error_baseline_ewma_n'   AS ewma_n
FROM substrate_chat_session_projection
WHERE projection_id = 'active_chat_session';

UPDATE substrate_chat_session_projection
SET projection_json = projection_json
    || '{"prediction_error_baseline_ewma": 0.0,
         "prediction_error_baseline_ewma_var": 0.0,
         "prediction_error_baseline_ewma_n": 0}'::jsonb
WHERE projection_id = 'active_chat_session';

COMMIT;
