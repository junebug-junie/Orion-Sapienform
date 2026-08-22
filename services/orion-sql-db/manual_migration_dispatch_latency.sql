-- Per-action cost (2026-08-21). The denominator a decision budget divides by.
--
-- Until this, there was NO per-action cost anywhere in Orion. `latency_ms`
-- existed as a field on ActionOutcomeRecordV1, a column on
-- substrate_action_outcomes, and a `_latencies()` reader in the feedback
-- runtime -- and was populated on 0 of 5,739 rows over 6 hours, because it was
-- dropped twice in series: the dispatch worker never wrote it, and
-- load_cortex_result_evidence() built its evidence dict from four hardcoded
-- keys that excluded it. A schema field, a column and a reader, none of which
-- could ever carry a value.
--
-- That is what stopped the decision budget being buildable: value with no
-- denominator is a ranking, not a budget.
--
-- Measured as wall-clock around the cortex send, not read off the verb's own
-- report -- it is the time the action actually occupied the motor path
-- (queueing and transport included, which is real spend), and it does not
-- depend on a verb choosing to report anything. skills.runtime.* verbs report
-- nothing.

ALTER TABLE substrate_dispatch_results
    ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION;

-- A budget aggregates cost per action per window on every allocation. Index
-- from day one rather than after it becomes the next 49.8%-of-buffer-traffic
-- query.
CREATE INDEX IF NOT EXISTS substrate_dispatch_results_latency_idx
    ON substrate_dispatch_results (created_at DESC)
    WHERE latency_ms IS NOT NULL;
