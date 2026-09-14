with example as (
    select E'The model says: Offered 999 of 999 approved concepts [private prose 999] and 999 of 999 relation judgements. Wrote to its own graph: Secret narrative 999.\n\n(Offered 12 of 40 approved concepts [open_loop 4, semantic 36] and 6 of 20 relation judgements, all sampled at random. Investigated over 104 harness steps, grounding: secret diagnostic text, whole turn 3499s (stance + harness + finalize), of which harness 2400s Wrote to its own graph: Hop 2, Prior 1.)'::text as body
),
footer as (
    select reverse(split_part(reverse(rtrim(body, E' \t\r\n')), E'\n', 1)) as value
    from example
),
parsed as (
    select
        (regexp_match(value, '^\(Offered ([0-9]+) of ([0-9]+) approved concepts'))[1]::integer
            as offered_concepts,
        (regexp_match(value, '^\(Offered ([0-9]+) of ([0-9]+) approved concepts'))[2]::integer
            as available_concepts,
        (regexp_match(value, '\. Investigated over ([0-9]+) harness steps'))[1]::integer
            as harness_steps,
        case
            when (regexp_match(value, ', grounding: ([^,.)]+)'))[1] in (
                'grounded', 'partial', 'failed', 'empty_draft', 'invalid_request',
                'fcc_bad_model_label', 'fcc_lane_context_too_small',
                'fcc_spawn_failed', 'fcc_stream_stalled', 'fcc_timeout',
                'fcc_stream_line_limit', 'fcc_draft_length_ceiling_exceeded',
                'fcc_nonzero_exit', 'fcc_context_overflow', 'fcc_mcp_github_missing'
            ) then (regexp_match(value, ', grounding: ([^,.)]+)'))[1]
            when (regexp_match(value, ', grounding: ([^,.)]+)'))[1] is not null then 'other'
            else null
        end as grounding_status,
        (regexp_match(value, ', whole turn ([0-9]+)s \(stance \+ harness \+ finalize\)'))[1]::integer
            as whole_turn_seconds,
        (regexp_match(value, ', of which harness ([0-9]+)s'))[1]::integer
            as harness_seconds,
        (regexp_match(value, ' Wrote to its own graph: ([^.]+)\.\)$'))[1]
            as graph_footprint,
        (
            regexp_match(
                value,
                '^\(Offered [0-9]+ of [0-9]+ approved concepts \[([^]]*)\] and [0-9]+ of [0-9]+ relation judgements, all sampled at random\.'
            )
        )[1] as material_pool
    from footer
)

select *
from parsed
where offered_concepts <> 12
   or available_concepts <> 40
   or harness_steps <> 104
   or grounding_status <> 'other'
   or whole_turn_seconds <> 3499
   or harness_seconds <> 2400
   or graph_footprint <> 'Hop 2, Prior 1'
   or material_pool <> 'open_loop 4, semantic 36'
