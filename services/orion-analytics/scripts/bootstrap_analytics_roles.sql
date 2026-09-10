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
        namespace.nspname = 'public'
        and relation.relname in (
          'substrate_reverie_chain',
          'reverie_visual_chain',
          'reverie_visual_artifact'
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
  not has_schema_privilege('orion_analytics_reader', 'analytics', 'create') as reader_schema_write_is_denied;
