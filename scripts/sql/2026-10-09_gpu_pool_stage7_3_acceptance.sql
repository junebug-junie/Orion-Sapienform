-- GPU pool stage 7.3 live acceptance (docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md,
-- checks 4, 5, 6). Read-only. Run on athena:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v since="'<deploy UTC>'" \
--     < scripts/sql/2026-10-09_gpu_pool_stage7_3_acceptance.sql
-- For the "before" numbers use since = deploy - 7 days and compare windows of the same length.
-- gpu_pool_leases keeps ended leases for GPU_POOL_LEASE_RETENTION_HOURS (168 h): run within 7 days.

\echo '== check 4: two NON-URGENT durable-run holds granted on agent-gpu2 at once (urgent holds already stack, U3:'
\echo '   two urgent holds overlapped there on 10-06 04:24 before 7.3 -- excluded so this counts only what 7.3 allows)'
with g as (
  select lease_id, generated_at as start_at from gpu_pool_events
  where event = 'granted' and role = 'agent-gpu2' and holder like 'durable-runs:%'
    and priority <> 'urgent' and generated_at > :since::timestamptz),
iv as (
  select g.lease_id, g.start_at,
         coalesce((select min(e.generated_at) from gpu_pool_events e
                   where e.lease_id = g.lease_id and e.generated_at >= g.start_at
                     and e.event in ('released', 'aborted', 'expired', 'cancelled', 'unavailable')),
                  now()) as end_at
  from g)
select count(*) as overlapping_pairs,
       round(coalesce(sum(extract(epoch from least(a.end_at, b.end_at) - greatest(a.start_at, b.start_at))), 0)) as overlap_sec,
       max(greatest(a.start_at, b.start_at)) as latest_overlap_from
from iv a join iv b on a.lease_id < b.lease_id and a.start_at < b.end_at and b.start_at < a.end_at;

\echo '== check 5a: queued-hold wait (agent class), per role -- baseline 3 days to 09-30: p90 8,724 s on agent'
select role, count(*) as grants,
       round((percentile_cont(0.5) within group (order by waited_ms) / 1000)::numeric) as p50_s,
       round((percentile_cont(0.9) within group (order by waited_ms) / 1000)::numeric) as p90_s
from gpu_pool_events
where event = 'granted' and work_class = 'agent' and holder like 'durable-runs:%'
  and generated_at > :since::timestamptz
group by role order by role;

\echo '== check 5b: mean agent holds waiting, sampled every 10 min -- baseline 2.6 (max 13)'
with ts as (select generate_series(:since::timestamptz, now(), interval '10 min') as t),
ev as (select lease_id, event, generated_at from gpu_pool_events
       where work_class = 'agent' and holder like 'durable-runs:%'
         and generated_at > :since::timestamptz - interval '2 days')
select round(avg(n), 2) as mean_waiting, max(n) as max_waiting, count(*) as samples from (
  select ts.t, count(*) filter (where s.last_ev in ('admitted', 'queued', 'retried', 'backlogged')) as n
  from ts cross join lateral (
    select distinct on (lease_id) lease_id, event as last_ev from ev
    where ev.generated_at <= ts.t order by lease_id, generated_at desc) s
  group by ts.t) x;

\echo '== check 6a: one-off agent calls (no hold), pooled and per role -- baseline 7 d to 10-09: p90 49.5 s pooled, 0.19 s agent-gpu2'
select coalesce(e.role, 'ALL') as role, count(*) as grants,
       round((percentile_cont(0.5) within group (order by e.waited_ms) / 1000)::numeric, 2) as p50_s,
       round((percentile_cont(0.9) within group (order by e.waited_ms) / 1000)::numeric, 2) as p90_s
from gpu_pool_events e join gpu_pool_leases l using (lease_id)
where e.event = 'granted' and e.work_class = 'agent' and l.kind = 'request' and l.hold_lease_id is null
  and e.generated_at > :since::timestamptz
group by rollup(e.role) order by 1;

\echo '== check 6b: every system agent request (run calls + one-offs) -- baseline 7 d: p90 0.29 s agent, 0.25 s agent-gpu2'
select e.role, count(*) as grants,
       round((percentile_cont(0.9) within group (order by e.waited_ms) / 1000)::numeric, 2) as p90_s
from gpu_pool_events e join gpu_pool_leases l using (lease_id)
where e.event = 'granted' and e.work_class = 'agent' and l.kind = 'request' and e.priority = 'system'
  and e.generated_at > :since::timestamptz
group by e.role order by e.role;
