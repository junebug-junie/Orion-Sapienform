-- Read-only evaluation. Every *_delta column must be zero.
begin transaction read only;

with source_counts as (
    select
        (select count(*) from analytics_source.curiosity_run_transitions)::bigint
            as transition_rows,
        (
            select count(*)
            from (
                select run_id from analytics_source.curiosity_run_transitions
                union
                select run_id from analytics_source.curiosity_run_journals
            ) observed
        )::bigint as observed_runs,
        (select count(*) from analytics_source.curiosity_run_graph_writes)::bigint
            as graph_rows,
        (select count(*) from analytics_source.curiosity_run_material_pool)::bigint
            as material_rows
),
fact_counts as (
    select
        (select count(*) from analytics.fct_curiosity_run_transitions)::bigint
            as transition_rows,
        (select count(distinct transition_id) from analytics.fct_curiosity_run_transitions)::bigint
            as distinct_transition_ids,
        (select count(*) from analytics.fct_curiosity_runs)::bigint as observed_runs,
        (select count(distinct run_id) from analytics.fct_curiosity_runs)::bigint
            as distinct_run_ids,
        (select count(*) from analytics.fct_curiosity_graph_writes)::bigint as graph_rows,
        (select count(distinct graph_write_id) from analytics.fct_curiosity_graph_writes)::bigint
            as distinct_graph_ids,
        (select count(*) from analytics.fct_curiosity_material_pool)::bigint as material_rows,
        (select count(distinct material_pool_id) from analytics.fct_curiosity_material_pool)::bigint
            as distinct_material_ids
),
run_baseline as (
    select
        count(*) filter (where lifecycle_recorded_flag = 1)::bigint as lifecycle_runs,
        count(*) filter (where journaled_flag = 1)::bigint as journaled_runs,
        count(*) filter (where completed_flag = 1)::bigint as completed_runs,
        count(*) filter (where latest_nonterminal_flag = 1)::bigint as latest_nonterminal_runs,
        sum(failed_transition_count)::bigint as failed_transition_events,
        sum(resumed_transition_count)::bigint as resumed_transition_events,
        count(*) filter (where graph_written_flag = 1)::bigint as runs_writing_graph,
        count(*) filter (where graph_written_flag = 0)::bigint as runs_explicitly_writing_nothing,
        count(*) filter (where graph_written_flag is null)::bigint as runs_without_graph_evidence,
        round(avg(harness_step_count)::numeric, 2) as average_harness_steps,
        round(avg(whole_turn_seconds)::numeric, 2) as average_whole_turn_seconds,
        round(avg(harness_seconds)::numeric, 2) as average_harness_seconds,
        max(event_at) as newest_observed_run_at
    from analytics.fct_curiosity_runs
)
select
    source_counts.transition_rows as source_transition_rows,
    fact_counts.transition_rows as fact_transition_rows,
    source_counts.transition_rows - fact_counts.transition_rows as transition_source_fact_delta,
    fact_counts.transition_rows - fact_counts.distinct_transition_ids as transition_duplicate_delta,
    source_counts.observed_runs as source_observed_runs,
    fact_counts.observed_runs as fact_observed_runs,
    source_counts.observed_runs - fact_counts.observed_runs as run_source_fact_delta,
    fact_counts.observed_runs - fact_counts.distinct_run_ids as run_duplicate_delta,
    source_counts.graph_rows - fact_counts.graph_rows as graph_source_fact_delta,
    fact_counts.graph_rows - fact_counts.distinct_graph_ids as graph_duplicate_delta,
    source_counts.material_rows - fact_counts.material_rows as material_source_fact_delta,
    fact_counts.material_rows - fact_counts.distinct_material_ids as material_duplicate_delta,
    run_baseline.*
from source_counts, fact_counts, run_baseline;

select element_type, sum(element_count)::bigint as element_count
from analytics.fct_curiosity_graph_writes
group by element_type
order by element_count desc, element_type;

rollback;
