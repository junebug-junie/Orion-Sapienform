create table if not exists grammar_traces (
    trace_id text primary key
  , trace_type text not null
  , session_id text
  , turn_id text
  , root_event_id text
  , started_at timestamptz not null
  , ended_at timestamptz
  , status text not null default 'open'
  , summary text
  , created_at timestamptz not null default now()
);

create table if not exists grammar_events (
    event_id text primary key
  , trace_id text not null references grammar_traces(trace_id)
  , parent_event_id text
  , root_event_id text
  , event_kind text not null
  , session_id text
  , turn_id text
  , correlation_id text
  , layer text
  , dimensions text[] not null default '{}'
  , emitted_at timestamptz not null
  , observed_at timestamptz
  , source_service text not null
  , source_component text
  , source_event_id text
  , payload_ref text
  , event_json jsonb not null
  , created_at timestamptz not null default now()
);

create table if not exists grammar_atoms (
    atom_id text primary key
  , trace_id text not null references grammar_traces(trace_id)
  , atom_type text not null
  , semantic_role text not null
  , layer text not null
  , dimensions text[] not null default '{}'
  , summary text not null
  , text_value text
  , confidence double precision
  , salience double precision
  , uncertainty double precision
  , time_start timestamptz
  , time_end timestamptz
  , source_event_id text
  , payload_ref text
  , renderer_hint text
  , atom_json jsonb not null
  , created_at timestamptz not null default now()
);

create table if not exists grammar_edges (
    edge_id text primary key
  , trace_id text not null references grammar_traces(trace_id)
  , from_atom_id text not null
  , to_atom_id text not null
  , relation_type text not null
  , confidence double precision
  , salience double precision
  , layer_from text
  , layer_to text
  , temporal_relation text
  , evidence_event_ids text[] not null default '{}'
  , edge_json jsonb not null
  , created_at timestamptz not null default now()
);

create table if not exists grammar_temporal_hops (
    hop_id text primary key
  , trace_id text not null references grammar_traces(trace_id)
  , from_atom_id text not null
  , to_atom_id text
  , hop_type text not null
  , direction text not null
  , reason text not null
  , confidence double precision
  , turn_distance integer
  , session_distance integer
  , target_time_start timestamptz
  , target_time_end timestamptz
  , hop_json jsonb not null
  , created_at timestamptz not null default now()
);

create table if not exists grammar_compactions (
    compaction_id text primary key
  , trace_id text not null references grammar_traces(trace_id)
  , source_atom_ids text[] not null
  , output_atom_id text not null
  , compaction_type text not null
  , method text not null
  , summary text not null
  , preserves text[] not null default '{}'
  , drops text[] not null default '{}'
  , confidence double precision
  , compaction_json jsonb not null
  , created_at timestamptz not null default now()
);

create table if not exists grammar_projections (
    projection_id text primary key
  , trace_id text not null references grammar_traces(trace_id)
  , source_atom_ids text[] not null
  , projection_type text not null
  , summary text not null
  , confidence double precision
  , expires_at timestamptz
  , projected_atom_id text
  , projection_json jsonb not null
  , created_at timestamptz not null default now()
);

create index if not exists idx_grammar_events_trace_id on grammar_events(trace_id);
create index if not exists idx_grammar_events_session_turn on grammar_events(session_id, turn_id);
create index if not exists idx_grammar_atoms_trace_id on grammar_atoms(trace_id);
create index if not exists idx_grammar_atoms_layer on grammar_atoms(layer);
create index if not exists idx_grammar_atoms_dimensions on grammar_atoms using gin(dimensions);
create index if not exists idx_grammar_edges_trace_id on grammar_edges(trace_id);
create index if not exists idx_grammar_edges_from on grammar_edges(from_atom_id);
create index if not exists idx_grammar_edges_to on grammar_edges(to_atom_id);
create index if not exists idx_grammar_temporal_hops_trace_id on grammar_temporal_hops(trace_id);
