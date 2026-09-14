with transition_rollup as (
    select
        run_id,
        min(workflow) as workflow,
        min(event_at) as lifecycle_started_at,
        max(event_at) as latest_transition_at,
        max(event_at) filter (where status = 'completed') as completed_at,
        count(*)::bigint as transition_count,
        count(distinct node)::bigint as distinct_nodes_reached,
        count(*) filter (where status = 'failed')::bigint as failed_transition_count,
        count(*) filter (where status = 'resumed')::bigint as resumed_transition_count,
        max(attempt_count) as attempt_count,
        max(run_type) filter (where run_type is not null) as run_type
    from {{ ref('stg_curiosity_run_transitions') }}
    group by run_id
),

latest_transition as (
    select *
    from (
        select
            run_id,
            node as latest_node,
            next_node as latest_next_node,
            status as latest_status,
            resumed_from_node as latest_resumed_from_node,
            row_number() over (
                partition by run_id
                order by event_at desc, transition_id desc
            ) as row_number
        from {{ ref('stg_curiosity_run_transitions') }}
    ) ranked
    where row_number = 1
),

journal_rollup as (
    select
        run_id,
        count(*)::bigint as journal_record_count,
        min(journaled_at) as first_journaled_at,
        max(journaled_at) as latest_journaled_at
    from {{ ref('stg_curiosity_run_journals') }}
    group by run_id
),

latest_journal as (
    select *
    from (
        select
            run_id,
            run_type as journal_run_type,
            offered_concept_count,
            available_concept_count,
            offered_relation_count,
            available_relation_count,
            harness_step_count,
            grounding_status,
            whole_turn_seconds,
            harness_seconds,
            graph_write_status,
            row_number() over (
                partition by run_id
                order by journaled_at desc, journal_id desc
            ) as row_number
        from {{ ref('stg_curiosity_run_journals') }}
    ) ranked
    where row_number = 1
),

graph_rollup as (
    select
        run_id,
        sum(element_count)::bigint as graph_element_count
    from {{ ref('stg_curiosity_run_graph_writes') }}
    group by run_id
),

observed_run_ids as (
    select run_id from transition_rollup
    union
    select run_id from journal_rollup
)

select
    observed_run_ids.run_id,
    coalesce(transition_rollup.run_type, latest_journal.journal_run_type, 'unknown') as run_type,
    transition_rollup.workflow,
    case
        when transition_rollup.run_id is not null and journal_rollup.run_id is not null
            then 'lifecycle_and_journal'
        when transition_rollup.run_id is not null then 'lifecycle_only'
        else 'journal_only'
    end as observation_source,
    coalesce(transition_rollup.lifecycle_started_at, journal_rollup.first_journaled_at) as event_at,
    (
        coalesce(transition_rollup.lifecycle_started_at, journal_rollup.first_journaled_at)
        at time zone 'UTC'
    )::date as event_date,
    transition_rollup.lifecycle_started_at,
    transition_rollup.latest_transition_at,
    case
        when transition_rollup.latest_transition_at is not null
            then extract(epoch from (current_timestamp - transition_rollup.latest_transition_at)) / 60.0
        else null
    end as latest_state_age_minutes,
    transition_rollup.completed_at,
    journal_rollup.latest_journaled_at,
    latest_transition.latest_status,
    latest_transition.latest_node,
    latest_transition.latest_next_node,
    latest_transition.latest_resumed_from_node,
    coalesce(transition_rollup.transition_count, 0)::bigint as transition_count,
    coalesce(transition_rollup.distinct_nodes_reached, 0)::bigint as distinct_nodes_reached,
    coalesce(transition_rollup.failed_transition_count, 0)::bigint as failed_transition_count,
    coalesce(transition_rollup.resumed_transition_count, 0)::bigint as resumed_transition_count,
    transition_rollup.attempt_count,
    case when transition_rollup.run_id is not null then 1 else 0 end::smallint
        as lifecycle_recorded_flag,
    case when transition_rollup.completed_at is not null then 1 else 0 end::smallint
        as completed_flag,
    case
        when transition_rollup.run_id is not null
         and latest_transition.latest_status not in ('completed', 'cancelled', 'abandoned')
            then 1
        else 0
    end::smallint as latest_nonterminal_flag,
    case when journal_rollup.run_id is not null then 1 else 0 end::smallint as journaled_flag,
    coalesce(journal_rollup.journal_record_count, 0)::bigint as journal_record_count,
    case
        when transition_rollup.completed_at is not null
            then extract(
                epoch from (
                    transition_rollup.completed_at - transition_rollup.lifecycle_started_at
                )
            ) / 60.0
        else null
    end as lifecycle_minutes,
    latest_journal.offered_concept_count,
    latest_journal.available_concept_count,
    latest_journal.offered_relation_count,
    latest_journal.available_relation_count,
    latest_journal.harness_step_count,
    latest_journal.grounding_status,
    latest_journal.whole_turn_seconds,
    latest_journal.harness_seconds,
    latest_journal.graph_write_status,
    case
        when latest_journal.graph_write_status = 'wrote_elements'
            then coalesce(graph_rollup.graph_element_count, 0)
        when latest_journal.graph_write_status = 'wrote_nothing' then 0
        else null
    end::bigint as graph_element_count,
    case
        when latest_journal.graph_write_status = 'wrote_elements' then 1
        when latest_journal.graph_write_status = 'wrote_nothing' then 0
        else null
    end::smallint as graph_written_flag
from observed_run_ids
left join transition_rollup using (run_id)
left join latest_transition using (run_id)
left join journal_rollup using (run_id)
left join latest_journal using (run_id)
left join graph_rollup using (run_id)
