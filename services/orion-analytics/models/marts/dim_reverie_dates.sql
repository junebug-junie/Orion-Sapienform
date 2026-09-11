with bounds as (
    select min(event_date) as first_event_date
    from (
        select event_date from {{ ref('stg_reverie_chains') }}
        union all
        select event_date from {{ ref('stg_visual_reverie_chains') }}
    ) all_reverie_dates
)

select
    day_utc::date as date_key,
    extract(isodow from day_utc)::smallint as iso_day_of_week,
    extract(week from day_utc)::smallint as iso_week_of_year,
    date_trunc('month', day_utc)::date as month_start_date,
    date_trunc('quarter', day_utc)::date as quarter_start_date,
    extract(year from day_utc)::smallint as year_number
from bounds
cross join lateral generate_series(
    bounds.first_event_date::timestamp,
    (current_timestamp at time zone 'UTC')::date::timestamp,
    interval '1 day'
) as day_utc
where bounds.first_event_date is not null
