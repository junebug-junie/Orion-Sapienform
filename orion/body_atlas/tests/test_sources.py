"""Whole-body source parsers and the gather step, on fixtures only (no DB, no docker, no network)."""
from __future__ import annotations

import textwrap

import yaml

from orion.body_atlas import sources as S

CATALOG_YAML = textwrap.dedent("""
    defaults: {expected_online: true, role: unknown}
    nodes:
      athena:
        aliases: [athena, athena.ts.net]
        role: orchestration
        hardware: HP DL380 G10
      hecate:
        role: gpu_server
        expected_online: false
    unmonitored_nodes:
      carbon-x1:
        role: juniper_laptop
        expected_online: false
        cameras: [carbon]
      pan-1:
        runs_services: [orion-office-env]
    retired_nodes:
      atlas: {retired_at: "2026-08-21"}
""")


def test_node_catalog_reads_both_sections_and_skips_retired():
    hosts, placed = S.parse_node_catalog(yaml.safe_load(CATALOG_YAML))
    by = {h.name: h for h in hosts}
    assert sorted(by) == ["athena", "carbon-x1", "hecate", "pan-1"]
    assert by["athena"].hardware == "HP DL380 G10" and by["athena"].monitored
    assert "athena.ts.net" in by["athena"].aliases
    assert by["hecate"].hardware is None and by["hecate"].expected_online is False
    assert not by["carbon-x1"].monitored and by["carbon-x1"].cameras == ["carbon"]
    assert by["pan-1"].role == "unknown"
    assert [(p.service, p.host, p.evidence) for p in placed] == [("orion-office-env", "pan-1", "declared_placement")]


def test_include_file_ignores_comments_and_blank_lines():
    text = "# header\n\norion-ai-town\n# orion-kev moved\norion-kev  # trailing\n"
    assert [p.service for p in S.parse_include_services("circe", text)] == ["orion-ai-town", "orion-kev"]


def test_frame_router_uses_streams_not_the_stale_cameras_section():
    data = {"streams": {"cam0": {"enabled": True}, "walkway": {}, "old": {"enabled": False}},
            "cameras": {"mock-cam-01": {}, "porch_eye": {}}}
    assert S.parse_frame_router_streams(data) == ["cam0", "walkway"]


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def test_stream_readers_by_rule(tmp_path):
    _write(tmp_path, "services/orion-vision-frame-router/docker-compose.yml", """
        services:
          orion-vision-frame-router:
            volumes: ["/repo/config/vision_frame_router.yaml:/app/config/vision_frame_router.yaml:ro"]
    """)
    _write(tmp_path, "services/orion-vision-retina/docker-compose.yml", """
        services:
          orion-vision-retina-walkway:
            environment:
              - RETINA_STREAM_ID=walkway
          orion-vision-retina:
            environment: {RETINA_STREAM_ID: "${RETINA_STREAM_ID:-x}"}
    """)
    _write(tmp_path, "services/orion-juniper-affective-state/.env_example", "AFFECT_TARGET_STREAM_ID=carbon\n")
    _write(tmp_path, "services/orion-other/.env_example", "STREAM_ID=not-a-stream\n")
    got = S.stream_readers(tmp_path, ["cam0", "carbon", "walkway"])
    names = {s: [svc for svc, _ in v] for s, v in got.items()}
    assert names["cam0"] == ["orion-vision-frame-router"]
    assert names["walkway"] == ["orion-vision-frame-router", "orion-vision-retina-walkway"]
    assert names["carbon"] == ["orion-juniper-affective-state", "orion-vision-frame-router"]


SETTINGS = """
DEFAULT_ROUTE_MAP: dict[str, str] = {
    "chat.history": "ChatHistoryLogSQL",
    "world.pulse.v1": "WorldPulseRun",
    "unmapped.kind": "NoSuchModel",
}
"""
MODELS = """
class ChatHistoryLogSQL(Base):
    __tablename__ = "chat_history_log"

class WorldPulseRun(Base):
    __tablename__ = "world_pulse_run"
"""


def test_route_map_joins_kind_to_table_to_producers():
    route_map = S.parse_route_map(SETTINGS)
    tables = S.parse_model_tables([MODELS])
    catalog = [
        {"name": "orion:chat:history", "message_kind": "chat.history",
         "producer_services": ["orion-hub", "*"], "consumer_services": ["orion-sql-writer"]},
        {"name": "orion:world:pulse", "message_kind": "world.pulse.v1",
         "producer_services": ["orion-world-pulse"], "consumer_services": ["orion-hub"]},  # not persisted
    ]
    writers, persisted = S.table_writers(route_map, tables, catalog, writer="orion-sql-writer")
    assert writers == {"chat_history_log": [("orion-hub", "orion:chat:history")]}
    assert persisted == {"orion:chat:history": "orion-sql-writer"}


def test_durable_workflows_and_their_runner_come_from_the_schema_and_catalog():
    text = textwrap.dedent("""
        DURABLE_RUN_STATE_KIND = "durable.run.state.v1"
        DurableWorkflowV1 = Literal["curiosity.investigate", "reading.turn"]
    """)
    assert S.parse_durable_workflows(text) == ["curiosity.investigate", "reading.turn"]
    catalog = [{"name": "orion:durable:run:state", "message_kind": "durable.run.state.v1",
                "producer_services": ["orion-durable-runs"], "consumer_services": []}]
    assert S.durable_runner(text, catalog) == "orion-durable-runs"
    assert S.durable_runner(text, []) is None


def test_scratch_graphs_are_dropped_by_rule():
    for g in ("t_assertion_7f51293ab9", "orion_bus_synapse_livetest2", "orion_bus_synapse_collapse_livetest"):
        assert S.is_scratch_graph(g), g
    for g in ("orion_substrate", "orion_bus_synapse", "graphiti_temporal", "orion_turns"):
        assert not S.is_scratch_graph(g), g


def test_postgres_text_timestamps_parse():
    assert S.parse_ts("2026-10-29 20:51:12.870855+00") == S.parse_ts("2026-10-29T20:51:12.870855+00:00")
    assert S.parse_ts("garbage") is None and S.parse_ts(None) is None


# --- gather with fake live readers --------------------------------------------------------

class FakeSql:
    def __init__(self, deny=("substrate_reduction_cursor",)):
        self.deny = deny

    def run(self, sql, flags):
        assert flags == ["--csv", "-q"]
        for table in self.deny:
            if table in sql:
                return f"psql:<stdin>:2: ERROR:  permission denied for table {table}\n", 3
        if "pg_stat_user_tables" in sql:
            return "relname,n_live_tup\nchat_history_log,813\nunrelated_table,5\n", 0
        if "orion_biometrics_summary" in sql:
            return "node,last_seen\nathena,2026-10-29 20:00:00+00\nhecate,2026-10-15 06:31:03+00\n", 0
        if "substrate_durable_run_state" in sql:
            return "workflow,runs,last_run\nreading.turn,4,2026-10-29 19:00:00+00\nbrand.new,1,2026-10-01 00:00:00+00\n", 0
        if "vision_events" in sql:
            return "stream_id,events,last_seen\nwalkway,900,2026-10-29 20:30:00+00\n", 0
        raise AssertionError(sql)


def _fake_graph(cmd, *args):
    if cmd == "GRAPH.LIST":
        return ["orion_substrate", "t_assertion_abc", "orion_empty"]
    graph = args[0]
    return [["count(n)"], [[0 if graph == "orion_empty" else 42]]]


def _fake_http(url):
    if url.endswith("/collections"):
        return [{"name": "orion_chat", "id": "c1"}]
    return 7


def _root(tmp_path):
    _write(tmp_path, "config/biometrics/node_catalog.yaml", CATALOG_YAML)
    _write(tmp_path, "config/vision_frame_router.yaml", "streams: {walkway: {}, carbon: {}}\n")
    _write(tmp_path, "services/orion-sql-writer/app/settings.py", SETTINGS)
    _write(tmp_path, "services/orion-sql-writer/app/models/chat.py", MODELS)
    _write(tmp_path, "orion/schemas/durable_run.py", 'DurableWorkflowV1 = Literal["reading.turn"]\n')
    _write(tmp_path, "mesh-utilities/common/include_services_circe.txt", "orion-kev\n")
    _write(tmp_path, "services/orion-chat-memory/.env_example",
           "CHAT_MEMORY_COLLECTION=orion_chat\nVECTOR_HOST_GPT_TURN_COLLECTION=orion_chat_gpt_turns\n")
    _write(tmp_path, "services/orion-recall/.env_example", "RECALL_GRAPH=orion_substrate\n")
    return tmp_path


CHANNELS = [{"name": "orion:chat:history", "message_kind": "chat.history",
             "producer_services": ["orion-hub"], "consumer_services": ["orion-sql-writer"]}]


def test_gather_reads_everything_and_reports_what_failed(tmp_path):
    live = S.LiveReaders(sql=FakeSql(), graph=_fake_graph, http_json=_fake_http,
                         docker=lambda argv: "athena\n" if argv[0] == "info" else "orion-athena-hub\nzelda-dojo-0\n")
    body = S.gather_body(_root(tmp_path), CHANNELS, live=live)
    reads = {r.name: r for r in body.reads}
    bad = reads["live:postgres substrate_reduction_cursor"]
    assert bad.status == "unread" and "permission denied" in bad.detail
    assert all(r.status == "read" for n, r in reads.items() if n != bad.name)
    hosts = {h.name: h for h in body.hosts}
    assert hosts["athena"].last_seen_epoch == S.parse_ts("2026-10-29 20:00:00+00")
    assert hosts["carbon-x1"].last_seen_epoch is None
    assert {(p.service, p.host, p.evidence) for p in body.placements} >= {
        ("orion-athena-hub", "athena", "observed_placement"), ("orion-kev", "circe", "declared_placement")}
    stores = {(s.store, s.name): s for s in body.stores}
    assert ("falkordb", "t_assertion_abc") not in stores
    assert stores[("falkordb", "orion_empty")].rows == 0
    assert stores[("falkordb", "orion_substrate")].links == [("orion-recall", "uses", "declared_env", "")]
    assert stores[("chroma", "orion_chat")].rows == 7
    assert stores[("postgres", "chat_history_log")].rows == 813
    assert ("postgres", "unrelated_table") not in stores  # no writer, no link: not a part
    assert body.missing_declared_stores == ["chroma:orion_chat_gpt_turns"]
    procs = {p.name: p for p in body.processes}
    assert procs["reading.turn"].declared and procs["reading.turn"].runs == 4
    assert not procs["brand.new"].declared  # seen running, not in the Literal: still a part
    walkway = next(s for s in body.sensors if s.name == "walkway")
    assert walkway.events == 900 and body.persisted == {"orion:chat:history": "orion-sql-writer"}


def test_live_readers_missing_are_unread_never_silent(tmp_path):
    body = S.gather_body(_root(tmp_path), CHANNELS, live=S.LiveReaders())
    unread = sorted(r.name for r in body.reads if r.status == "unread")
    assert "live:docker ps (this host)" in unread and "live:falkordb GRAPH.LIST" in unread
    assert any("pg_stat_user_tables" in n for n in unread)
    # declared tables stay, with rows unknown
    assert [s.rows for s in body.stores if s.name == "chat_history_log"] == [None]


def test_past_checkout_without_the_files_is_absent_not_unread(tmp_path):
    body = S.gather_body(tmp_path, [], live=None)
    assert body.hosts == [] and body.stores == [] and body.processes == []
    assert {r.status for r in body.reads} == {"absent"}


# --- review fixes (2026-10-29) -------------------------------------------------------------

def test_a_persisted_channel_whose_table_is_not_in_the_db_keeps_its_bus_flow(tmp_path):
    class NoTable(FakeSql):
        def run(self, sql, flags):
            if "pg_stat_user_tables" in sql:
                return "relname,n_live_tup\nunrelated_table,5\n", 0
            return super().run(sql, flags)

    body = S.gather_body(_root(tmp_path), CHANNELS, live=S.LiveReaders(sql=NoTable()))
    assert not [s for s in body.stores if s.name == "chat_history_log"]
    assert body.persisted == {}  # nothing replaces the flow, so nothing drops it


def test_read_with_no_rows_is_zero_not_unknown(tmp_path):
    class Empty(FakeSql):
        def run(self, sql, flags):
            if "substrate_durable_run_state" in sql:
                return "workflow,runs,last_run\n", 0
            if "vision_events" in sql:
                return "stream_id,events,last_seen\n", 0
            return super().run(sql, flags)

    body = S.gather_body(_root(tmp_path), CHANNELS, live=S.LiveReaders(sql=Empty()))
    assert [p.runs for p in body.processes if p.name == "reading.turn"] == [0]
    assert {s.events for s in body.sensors} == {0}


def test_a_compose_file_that_does_not_parse_makes_the_read_partial(tmp_path):
    root = _root(tmp_path)
    _write(root, "services/orion-broken/docker-compose.yml", "services: [unclosed\n")
    _write(root, "services/orion-vision-edge/.env_example", "STREAM_ID=walkway\n")
    body = S.gather_body(root, CHANNELS, live=None)
    read = next(r for r in body.reads if "STREAM_ID" in r.name)
    assert read.status == "partial" and "orion-broken" in read.detail
    walkway = next(s for s in body.sensors if s.name == "walkway")
    assert [svc for svc, _ in walkway.readers] == ["orion-vision-edge"]  # the rest still read


def test_compose_defaults_name_a_stream():
    assert S._stream_setting("RETINA_STREAM_ID=${RETINA_STREAM_ID:-walkway}") == ("RETINA_STREAM_ID", "walkway")
    assert S._stream_setting("STREAM_ID=${STREAM_ID}") is None


def test_psql_notices_are_not_rows():
    class Noisy:
        def run(self, sql, flags):
            return "node,last_seen\nathena,2026-10-29 20:00:00+00\nWARNING:  something\npsql:<stdin>:3: NOTICE: x\n", 0

    assert S.psql_rows(Noisy(), "x") == [{"node": "athena", "last_seen": "2026-10-29 20:00:00+00"}]


def test_scratch_rule_keeps_real_words():
    for g in ("orion_testimony", "orion_smoker", "foo_t_bar", "orion_tested_ideas"):
        assert not S.is_scratch_graph(g), g
    assert S.is_scratch_graph("orion_tests") and S.is_scratch_graph("orion_smoke2")


def test_dropped_scratch_graphs_are_named_in_the_read(tmp_path):
    body = S.gather_body(_root(tmp_path), CHANNELS, live=S.LiveReaders(graph=_fake_graph))
    read = next(r for r in body.reads if r.name == "live:falkordb GRAPH.LIST")
    assert "t_assertion_abc" in read.detail


def test_compose_only_tags_parse(tmp_path):
    # live 2026-10-29: orion-cortex-exec's compose uses `ports: !override []`, which safe_load refuses
    _write(tmp_path, "services/orion-vision-edge/docker-compose.yml", """
        services:
          orion-vision-edge:
            ports: !override []
            environment:
              - STREAM_ID=walkway
    """)
    assert S.stream_readers(tmp_path, ["walkway"])["walkway"][0][0] == "orion-vision-edge"
