with bounds as (
    select min(date_key) as first_date, max(date_key) as last_date, count(*) as actual_dates
    from {{ ref('dim_reverie_dates') }}
)

select *
from bounds
where first_date is not null
  and actual_dates <> (last_date - first_date + 1)
