\set ON_ERROR_STOP on

-- Required psql variables. Values are supplied at execution time and are not
-- stored in this repository.
\if :{?analytics_transformer_password}
\else
  \echo 'missing -v analytics_transformer_password=...'
  \quit 2
\endif
\if :{?analytics_reader_password}
\else
  \echo 'missing -v analytics_reader_password=...'
  \quit 2
\endif

begin;

select format(
  'create role orion_analytics_transformer login password %L',
  :'analytics_transformer_password'
)
where not exists (select 1 from pg_roles where rolname = 'orion_analytics_transformer')
\gexec

select format(
  'alter role orion_analytics_transformer login password %L nosuperuser nocreatedb nocreaterole noinherit noreplication nobypassrls',
  :'analytics_transformer_password'
)
\gexec

select format(
  'create role orion_analytics_reader login password %L',
  :'analytics_reader_password'
)
where not exists (select 1 from pg_roles where rolname = 'orion_analytics_reader')
\gexec

select format(
  'alter role orion_analytics_reader login password %L nosuperuser nocreatedb nocreaterole noinherit noreplication nobypassrls',
  :'analytics_reader_password'
)
\gexec

-- A reused role name must not retain inherited privileges from an earlier use.
select format('revoke %I from %I', parent.rolname, member.rolname)
from pg_auth_members membership
join pg_roles parent on parent.oid = membership.roleid
join pg_roles member on member.oid = membership.member
where member.rolname in ('orion_analytics_transformer', 'orion_analytics_reader')
\gexec

-- Remove direct relation grants from every non-system schema before granting
-- the intentionally narrow source/mart access below. Grants held by PUBLIC
-- cannot be denied per-role, so the verification block also audits effective
-- privileges and fails if the database exposes anything broader.
select format(
  'revoke all privileges on all tables in schema %I from orion_analytics_transformer, orion_analytics_reader',
  nspname
)
from pg_namespace
where nspname <> 'information_schema'
  and nspname !~ '^pg_'
\gexec

select format(
  'revoke all privileges on all sequences in schema %I from orion_analytics_transformer, orion_analytics_reader',
  nspname
)
from pg_namespace
where nspname <> 'information_schema'
  and nspname !~ '^pg_'
\gexec

select format(
  'revoke all privileges on schema %I from orion_analytics_transformer, orion_analytics_reader',
  nspname
)
from pg_namespace
where nspname <> 'information_schema'
  and nspname !~ '^pg_'
\gexec

-- Privacy boundary for sources that contain narrative-bearing JSON or text.
-- These security-barrier views are owned by the operator running this script;
-- the transformer can read only their bounded structural projections and
-- never receives SELECT on the underlying lifecycle/journal tables.
create schema if not exists analytics_source;
alter schema analytics_source owner to current_user;
revoke all on schema analytics_source from public, orion_analytics_transformer, orion_analytics_reader;

create or replace view analytics_source.curiosity_run_transitions
with (security_barrier = true) as
select
  entry_id as transition_id,
  run_id,
  workflow,
  node,
  next_node,
  status,
  resumed_from_node,
  generated_at as event_at,
  created_at as stored_at,
  case
    when detail::jsonb ->> 'line' in ('investigate', 'self_inquiry')
      then detail::jsonb ->> 'line'
    else null
  end as run_type,
  case
    when jsonb_typeof(detail::jsonb -> 'attempts') = 'number'
      then (detail::jsonb ->> 'attempts')::integer
    else null
  end as attempt_count,
  case when detail::jsonb ? 'error' then 1 else 0 end::smallint as error_present_flag
from public.substrate_durable_run_state;

alter view analytics_source.curiosity_run_transitions owner to current_user;

create or replace view analytics_source.curiosity_run_journals
with (security_barrier = true) as
with footer_candidates as (
  select
    entry_id as journal_id,
    substring(source_ref from length('curiosity:') + 1) as run_id,
    created_at as journaled_at,
    case when title = 'Self-inquiry' then 'self_inquiry' else 'investigate' end as run_type,
    reverse(split_part(reverse(rtrim(body, E' \t\r\n')), E'\n', 1)) as footer
  from public.journal_entries
  where source_ref like 'curiosity:%'
),
footers as (
  select
    journal_id,
    run_id,
    journaled_at,
    run_type,
    case
      when footer ~ '^\(Offered [0-9]+ of [0-9]+ approved concepts \[[a-z0-9_, ]*\] and [0-9]+ of [0-9]+ relation judgements, all sampled at random\..*\.\)$'
        then footer
      else null
    end as footer
  from footer_candidates
),
parsed as (
  select
    footers.*,
    regexp_match(
      footer,
      '^\(Offered ([0-9]+) of ([0-9]+) approved concepts'
    ) as concept_counts,
    regexp_match(
      footer,
      'approved concepts \[[^]]*\] and ([0-9]+) of ([0-9]+) relation judgements, all sampled at random\.'
    ) as relation_counts,
    regexp_match(footer, '\. Investigated over ([0-9]+) harness steps') as step_count,
    regexp_match(footer, ', grounding: ([^,.)]+)') as grounding,
    regexp_match(
      footer,
      ', whole turn ([0-9]+)s \(stance \+ harness \+ finalize\)'
    ) as whole_turn,
    regexp_match(footer, ', of which harness ([0-9]+)s') as harness_turn
  from footers
)
select
  journal_id,
  run_id,
  journaled_at,
  run_type,
  concept_counts[1]::integer as offered_concept_count,
  concept_counts[2]::integer as available_concept_count,
  relation_counts[1]::integer as offered_relation_count,
  relation_counts[2]::integer as available_relation_count,
  step_count[1]::integer as harness_step_count,
  case
    when grounding[1] in (
      'grounded',
      'partial',
      'failed',
      'empty_draft',
      'invalid_request',
      'fcc_bad_model_label',
      'fcc_lane_context_too_small',
      'fcc_spawn_failed',
      'fcc_stream_stalled',
      'fcc_timeout',
      'fcc_stream_line_limit',
      'fcc_draft_length_ceiling_exceeded',
      'fcc_nonzero_exit',
      'fcc_context_overflow',
      'fcc_mcp_github_missing'
    ) then grounding[1]
    when grounding[1] is not null then 'other'
    else null
  end as grounding_status,
  whole_turn[1]::integer as whole_turn_seconds,
  harness_turn[1]::integer as harness_seconds,
  case
    when footer ~ ' Wrote to its own graph: [^.]+\.\)$' then 'wrote_elements'
    when footer ~ ' Wrote nothing to its own graph this run\.\)$' then 'wrote_nothing'
    else 'unavailable_or_unrecorded'
  end as graph_write_status
from parsed;

alter view analytics_source.curiosity_run_journals owner to current_user;

create or replace view analytics_source.curiosity_run_graph_writes
with (security_barrier = true) as
with footer_candidates as (
  select
    entry_id as journal_id,
    substring(source_ref from length('curiosity:') + 1) as run_id,
    created_at as journaled_at,
    case when title = 'Self-inquiry' then 'self_inquiry' else 'investigate' end as run_type,
    reverse(split_part(reverse(rtrim(body, E' \t\r\n')), E'\n', 1)) as footer
  from public.journal_entries
  where source_ref like 'curiosity:%'
),
footers as (
  select
    footer_candidates.*,
    (regexp_match(footer, ' Wrote to its own graph: ([^.]+)\.\)$'))[1] as footprint
  from footer_candidates
  where footer ~ '^\(Offered [0-9]+ of [0-9]+ approved concepts \[[a-z0-9_, ]*\] and [0-9]+ of [0-9]+ relation judgements, all sampled at random\..*\.\)$'
),
fragments as (
  select
    footers.*,
    trim(fragment) as fragment
  from footers
  cross join lateral regexp_split_to_table(footprint, ',') as fragment
  where footprint is not null
),
parsed as (
  select
    fragments.*,
    regexp_match(
      fragment,
      '^((-> )?[A-Za-z][A-Za-z0-9_]{0,62}) ([0-9]+)$'
    ) as fields
  from fragments
)
select
  md5(journal_id || ':' || fields[1]) as graph_write_id,
  journal_id,
  run_id,
  journaled_at,
  run_type,
  fields[1] as element_type,
  case when fields[1] like '-> %' then 'relationship' else 'node' end as element_family,
  fields[3]::integer as element_count
from parsed
where fields is not null;

alter view analytics_source.curiosity_run_graph_writes owner to current_user;

create or replace view analytics_source.curiosity_run_material_pool
with (security_barrier = true) as
with footer_candidates as (
  select
    entry_id as journal_id,
    substring(source_ref from length('curiosity:') + 1) as run_id,
    created_at as journaled_at,
    case when title = 'Self-inquiry' then 'self_inquiry' else 'investigate' end as run_type,
    reverse(split_part(reverse(rtrim(body, E' \t\r\n')), E'\n', 1)) as footer
  from public.journal_entries
  where source_ref like 'curiosity:%'
),
footers as (
  select
    footer_candidates.*,
    (
      regexp_match(
        footer,
        '^\(Offered [0-9]+ of [0-9]+ approved concepts \[([^]]*)\] and [0-9]+ of [0-9]+ relation judgements, all sampled at random\.'
      )
    )[1] as material_counts
  from footer_candidates
  where footer ~ '^\(Offered [0-9]+ of [0-9]+ approved concepts \[[a-z0-9_, ]*\] and [0-9]+ of [0-9]+ relation judgements, all sampled at random\..*\.\)$'
),
fragments as (
  select
    footers.*,
    trim(fragment) as fragment
  from footers
  cross join lateral regexp_split_to_table(material_counts, ',') as fragment
  where coalesce(material_counts, '') <> ''
),
parsed as (
  select
    fragments.*,
    regexp_match(fragment, '^([a-z][a-z0-9_]{0,62}) ([0-9]+)$') as fields
  from fragments
)
select
  md5(journal_id || ':' || fields[1]) as material_pool_id,
  journal_id,
  run_id,
  journaled_at,
  run_type,
  fields[1] as material_kind,
  fields[2]::integer as available_count
from parsed
where fields is not null;

alter view analytics_source.curiosity_run_material_pool owner to current_user;

revoke all on all tables in schema analytics_source from public, orion_analytics_transformer, orion_analytics_reader;
grant usage on schema analytics_source to orion_analytics_transformer;
grant select on all tables in schema analytics_source to orion_analytics_transformer;

create schema if not exists analytics authorization orion_analytics_transformer;
alter schema analytics owner to orion_analytics_transformer;
grant usage, create on schema analytics to orion_analytics_transformer;

select format(
  'grant connect on database %I to orion_analytics_transformer, orion_analytics_reader',
  current_database()
)
\gexec
grant usage on schema public to orion_analytics_transformer;
grant select on
  public.substrate_reverie_chain,
  public.reverie_visual_chain,
  public.reverie_visual_artifact
to orion_analytics_transformer;

grant usage on schema analytics to orion_analytics_reader;
grant select on all tables in schema analytics to orion_analytics_reader;
alter default privileges for role orion_analytics_transformer in schema analytics
  grant select on tables to orion_analytics_reader;

alter role orion_analytics_transformer set search_path = analytics, public;
alter role orion_analytics_transformer set statement_timeout = '60s';
alter role orion_analytics_reader set search_path = analytics;
alter role orion_analytics_reader set default_transaction_read_only = on;
alter role orion_analytics_reader set statement_timeout = '30s';

-- Fail closed if effective privileges do not match the intended boundary.
do $$
begin
  if not has_table_privilege(
    'orion_analytics_transformer',
    'public.substrate_reverie_chain',
    'select'
  ) or not has_table_privilege(
    'orion_analytics_transformer',
    'public.reverie_visual_chain',
    'select'
  ) or not has_table_privilege(
    'orion_analytics_transformer',
    'public.reverie_visual_artifact',
    'select'
  ) or not has_table_privilege(
    'orion_analytics_transformer',
    'analytics_source.curiosity_run_transitions',
    'select'
  ) or not has_table_privilege(
    'orion_analytics_transformer',
    'analytics_source.curiosity_run_journals',
    'select'
  ) or not has_table_privilege(
    'orion_analytics_transformer',
    'analytics_source.curiosity_run_graph_writes',
    'select'
  ) or not has_table_privilege(
    'orion_analytics_transformer',
    'analytics_source.curiosity_run_material_pool',
    'select'
  ) then
    raise exception 'analytics transformer cannot read every declared source';
  end if;

  if not has_schema_privilege(
    'orion_analytics_transformer',
    'analytics',
    'create'
  ) then
    raise exception 'analytics transformer cannot build analytics views';
  end if;

  if exists (
    select 1
    from pg_class relation
    join pg_namespace namespace on namespace.oid = relation.relnamespace
    where namespace.nspname <> 'information_schema'
      and namespace.nspname <> 'analytics'
      and namespace.nspname !~ '^pg_'
      and relation.relkind in ('r', 'p', 'v', 'm', 'f')
      and not (
        (
          namespace.nspname = 'public'
          and relation.relname in (
            'substrate_reverie_chain',
            'reverie_visual_chain',
            'reverie_visual_artifact'
          )
        )
        or (
          namespace.nspname = 'analytics_source'
          and relation.relname in (
            'curiosity_run_transitions',
            'curiosity_run_journals',
            'curiosity_run_graph_writes',
            'curiosity_run_material_pool'
          )
        )
      )
      and (
        relation.relowner = (
          select oid from pg_roles where rolname = 'orion_analytics_transformer'
        )
        or has_table_privilege(
          'orion_analytics_transformer', relation.oid, 'select'
        )
      )
  ) then
    raise exception 'analytics transformer can select an undeclared source relation';
  end if;

  if exists (
    select 1
    from pg_class relation
    join pg_namespace namespace on namespace.oid = relation.relnamespace
    where namespace.nspname <> 'information_schema'
      and namespace.nspname <> 'analytics'
      and namespace.nspname !~ '^pg_'
      and relation.relkind in ('r', 'p', 'v', 'm', 'f', 'S')
      and (
        relation.relowner = (
          select oid from pg_roles where rolname = 'orion_analytics_reader'
        )
        or (
          relation.relkind <> 'S'
          and has_table_privilege(
            'orion_analytics_reader', relation.oid, 'select'
          )
        )
        or (
          relation.relkind = 'S'
          and has_sequence_privilege(
            'orion_analytics_reader', relation.oid, 'usage,select,update'
          )
        )
      )
  ) then
    raise exception 'analytics reader can select a non-analytics relation';
  end if;

  if exists (
    select 1
    from pg_class relation
    join pg_namespace namespace on namespace.oid = relation.relnamespace
    where namespace.nspname <> 'information_schema'
      and namespace.nspname !~ '^pg_'
      and relation.relkind in ('r', 'p', 'v', 'm', 'f', 'S')
      and (
        relation.relowner = (
          select oid from pg_roles where rolname = 'orion_analytics_reader'
        )
        or (
          relation.relkind <> 'S'
          and has_table_privilege(
            'orion_analytics_reader',
            relation.oid,
            'insert,update,delete,truncate,references,trigger'
          )
        )
        or (
          relation.relkind = 'S'
          and has_sequence_privilege(
            'orion_analytics_reader', relation.oid, 'usage,update'
          )
        )
      )
  ) then
    raise exception 'analytics reader can modify or owns a user relation';
  end if;

  if exists (
    select 1
    from pg_namespace namespace
    where namespace.nspname <> 'information_schema'
      and namespace.nspname !~ '^pg_'
      and (
        namespace.nspowner = (
          select oid from pg_roles where rolname = 'orion_analytics_reader'
        )
        or has_schema_privilege(
          'orion_analytics_reader', namespace.oid, 'create'
        )
      )
  ) then
    raise exception 'analytics reader can create in or owns a user schema';
  end if;

  if exists (
    select 1
    from pg_namespace namespace
    where namespace.nspname <> 'information_schema'
      and namespace.nspname <> 'analytics'
      and namespace.nspname !~ '^pg_'
      and (
        namespace.nspowner = (
          select oid from pg_roles where rolname = 'orion_analytics_transformer'
        )
        or has_schema_privilege(
          'orion_analytics_transformer', namespace.oid, 'create'
        )
      )
  ) then
    raise exception 'analytics transformer can create in or owns an undeclared schema';
  end if;

  if exists (
    select 1
    from pg_roles
    where rolname in ('orion_analytics_transformer', 'orion_analytics_reader')
      and (rolsuper or rolcreatedb or rolcreaterole or rolinherit or rolreplication or rolbypassrls)
  ) then
    raise exception 'analytics role retains elevated role attributes';
  end if;

  if exists (
    select 1
    from pg_auth_members membership
    join pg_roles member on member.oid = membership.member
    where member.rolname in ('orion_analytics_transformer', 'orion_analytics_reader')
  ) then
    raise exception 'analytics role retains role memberships';
  end if;
end
$$;

commit;

select
  has_table_privilege('orion_analytics_transformer', 'public.substrate_reverie_chain', 'select')
    and has_table_privilege('orion_analytics_transformer', 'public.reverie_visual_chain', 'select')
    and has_table_privilege('orion_analytics_transformer', 'public.reverie_visual_artifact', 'select')
      as transformer_can_read_declared_sources,
  has_schema_privilege('orion_analytics_transformer', 'analytics', 'create') as transformer_can_build_views,
  not has_table_privilege('orion_analytics_reader', 'public.substrate_reverie_chain', 'select')
    and not has_table_privilege('orion_analytics_reader', 'public.reverie_visual_chain', 'select')
    and not has_table_privilege('orion_analytics_reader', 'public.reverie_visual_artifact', 'select')
      as reader_sources_are_denied,
  has_table_privilege('orion_analytics_transformer', 'analytics_source.curiosity_run_transitions', 'select')
    and has_table_privilege('orion_analytics_transformer', 'analytics_source.curiosity_run_journals', 'select')
    and has_table_privilege('orion_analytics_transformer', 'analytics_source.curiosity_run_graph_writes', 'select')
    and has_table_privilege('orion_analytics_transformer', 'analytics_source.curiosity_run_material_pool', 'select')
      as transformer_can_read_curiosity_safe_sources,
  not has_table_privilege('orion_analytics_transformer', 'public.substrate_durable_run_state', 'select')
    and not has_table_privilege('orion_analytics_transformer', 'public.journal_entries', 'select')
      as transformer_raw_curiosity_sources_are_denied,
  not has_schema_privilege('orion_analytics_reader', 'analytics_source', 'usage')
      as reader_curiosity_source_schema_is_denied,
  not has_schema_privilege('orion_analytics_reader', 'analytics', 'create') as reader_schema_write_is_denied;
