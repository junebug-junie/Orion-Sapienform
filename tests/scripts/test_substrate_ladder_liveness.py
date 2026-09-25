"""Gate tests for the substrate ladder liveness check (freshness + schema skew).

The replay test uses timestamps recovered from live Postgres for the
2026-09-20 FieldStateV1 extra_forbidden incident, not a synthetic shape.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion import substrate_ladder_liveness as ll  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "substrate_ladder_2026-09-20_incident.json"
FIELD_STATE = ll.STRICT_SCHEMAS[0]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "check_substrate_ladder_liveness", REPO / "scripts" / "check_substrate_ladder_liveness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ts(s):
    return datetime.fromisoformat(s) if s else None


def _replay(fixture: dict) -> ll.LadderReport:
    now = _ts(fixture["now"])
    report = ll.LadderReport()
    report.rungs = ll.evaluate_ladder(
        {k: _ts(v) for k, v in fixture["newest"].items()},
        now,
        consolidation_motif_counts=fixture["consolidation_motif_counts"],
    )
    containers = [
        ll.RunningContainer(c["name"], c["service_dir"], _ts(c["image_created"]), schema_sha256=c.get("schema_sha256"))
        for c in fixture["containers"]
    ]
    report.skew = ll.evaluate_skew(
        FIELD_STATE,
        schema_commit_time=_ts(fixture["schema_commit_time"]),
        schema_sha256_on_ref=fixture.get("schema_sha256_on_ref"),
        consumer_services=[c["service_dir"] for c in fixture["containers"]],
        containers=containers,
    )
    return report


# ------------------------------------------------------------------ incident


def test_replay_of_2026_09_20_incident_is_red():
    report = _replay(json.loads(FIXTURE.read_text()))
    assert report.red
    red = {r.rung for r in report.red_rungs}
    # The dead rungs, and only those: field_state and everything under it kept writing.
    assert red == {"attention", "proposal", "policy", "dispatch", "feedback", "consolidation:motifs"}
    status = {r.rung: r.status for r in report.rungs}
    assert status["field_state"] == "fresh"
    assert status["consolidation"] == "fresh"  # it kept writing -- empty
    attention = next(r for r in report.rungs if r.rung == "attention")
    assert attention.age_sec > 47 * 3600
    skewed = {s.container for s in report.red_skew}
    assert skewed == {
        "orion-athena-attention-runtime",
        "orion-athena-proposal-runtime",
        "orion-athena-feedback-runtime",
    }
    msg = report.alert_message()
    assert "attention" in msg and "orion-athena-attention-runtime" in msg
    assert report.severity() == "critical"


def test_replay_is_red_on_the_timestamp_fallback_too():
    """Same incident when no container's schema bytes can be read."""
    fx = json.loads(FIXTURE.read_text())
    for c in fx["containers"]:
        c["schema_sha256"] = None
    report = _replay(fx)
    assert {s.container for s in report.red_skew} == {
        "orion-athena-attention-runtime",
        "orion-athena-proposal-runtime",
        "orion-athena-feedback-runtime",
    }


def test_feedback_runtime_shape_is_caught_by_skew_while_its_rung_is_fresh():
    """Live on 2026-09-25: feedback-runtime logged ~96k extra_forbidden/2h yet
    kept writing frames (field_before=None). Freshness is green; skew must not be."""
    fx = json.loads(FIXTURE.read_text())
    fx["newest"] = {k: fx["now"] for k in fx["newest"]}
    fx["consolidation_motif_counts"] = [3, 2, 3]
    fx["containers"] = [c for c in fx["containers"] if c["service_dir"] in ("orion-field-digester", "orion-feedback-runtime")]
    report = _replay(fx)
    assert not report.red_rungs
    assert report.red_keys() == ["skew:FieldStateV1:orion-athena-feedback-runtime"]


def test_incident_is_caught_by_freshness_within_the_limit_not_after_48h():
    """At 22:20Z on 09-20 (22 minutes in) attention is already past its 15m limit."""
    fx = json.loads(FIXTURE.read_text())
    fx["now"] = "2026-09-20T22:20:00+00:00"
    fx["newest"] = {k: ("2026-09-20T22:19:50+00:00" if k not in ("attention", "proposal", "policy", "dispatch", "feedback") else v)
                    for k, v in fx["newest"].items()}
    fx["consolidation_motif_counts"] = [2, 3, 2]
    report = _replay(fx)
    assert "rung:attention" in report.red_keys()


def test_healthy_ladder_is_green():
    now = datetime(2026, 9, 25, 0, 16, tzinfo=timezone.utc)
    newest = {r.name: now - timedelta(seconds=30) for r in ll.RUNGS}
    newest["consolidation"] = now - timedelta(minutes=16)
    rungs = ll.evaluate_ladder(newest, now, consolidation_motif_counts=[3, 2, 3])
    assert all(not r.red for r in rungs)
    assert len(rungs) == len(ll.RUNGS) + 1


# ----------------------------------------------------------------- freshness


def test_missing_row_in_window_is_stale():
    rung = ll.RUNGS[0]
    r = ll.evaluate_rung(rung, None, datetime.now(timezone.utc))
    assert r.status == "stale" and r.red and r.newest is None
    assert "no row" in r.summary()


def test_boundary_is_inclusive_and_naive_timestamps_are_utc():
    rung = ll.Rung("x", "t", "generated_at", timedelta(minutes=10))
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert ll.evaluate_rung(rung, datetime(2026, 1, 1, 11, 50), now).status == "fresh"
    assert ll.evaluate_rung(rung, datetime(2026, 1, 1, 11, 49, 59), now).status == "stale"


def test_freshness_sql_is_always_time_bounded():
    for rung in ll.RUNGS:
        sql = ll.freshness_sql(rung)
        assert f"WHERE {rung.ts_column} > now() - make_interval(secs => %s)" in sql
        assert sql.count("%s") == (2 if rung.lane_column else 1)


def test_rung_names_are_unique():
    names = [r.name for r in ll.RUNGS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "counts,red",
    [
        ([0, 0, 0], True),
        ([0, 0, 0, 4], True),
        ([0, 0, 2], False),  # longest normal empty run in 60 days of live data was 1
        ([0, 0], False),  # not enough frames to say
        ([], False),
        ([3, 0, 0], False),
    ],
)
def test_consolidation_empty_run(counts, red):
    assert ll.evaluate_consolidation_empty(counts).red is red


# ---------------------------------------------------------------------- skew

COMMIT = datetime(2026, 9, 20, 21, 56, 46, tzinfo=timezone.utc)
NEW, OLD = "n" * 64, "o" * 64


def _producer(sha=NEW, built=COMMIT + timedelta(minutes=30)):
    return ll.RunningContainer("digester", "orion-field-digester", built, schema_sha256=sha)


def _skew(consumer: ll.RunningContainer, producer=None, main_sha=NEW, consumers=None):
    containers = [consumer] + ([producer] if producer is not None else [])
    results = ll.evaluate_skew(
        FIELD_STATE,
        schema_commit_time=COMMIT,
        schema_sha256_on_ref=main_sha,
        consumer_services=consumers or [consumer.service_dir],
        containers=containers,
    )
    return {r.container or r.service_dir: r for r in results}


def test_matching_producer_bytes_is_ok_even_with_an_old_image():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT - timedelta(days=12), schema_sha256=NEW)
    assert _skew(c, _producer())["c"].status == "ok"


def test_older_consumer_with_different_bytes_is_red():
    c = ll.RunningContainer("c", "orion-feedback-runtime", COMMIT - timedelta(days=6), schema_sha256=OLD)
    r = _skew(c, _producer())["c"]
    assert r.status == "skew" and r.red


def test_producer_deployed_ahead_of_main_still_catches_old_consumers():
    """Review finding: comparing against main missed this. Producer runs a
    worktree build with new fields main does not have yet; the consumer matches
    main, so it looked fine -- but it cannot read what the producer writes."""
    producer = _producer(sha="w" * 64)
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT - timedelta(days=1), schema_sha256=NEW)
    got = _skew(c, producer, main_sha=NEW)
    assert got["c"].red
    assert got["digester"].status == "producer_differs_from_main" and not got["digester"].red


def test_merged_but_undeployed_schema_change_is_not_red():
    """Review finding: comparing against main raised a false alarm here. The
    change is on main but the producer still writes the old shape, so the old
    consumers are fine."""
    producer = _producer(sha=OLD, built=COMMIT - timedelta(days=3))
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT - timedelta(days=12), schema_sha256=OLD)
    got = _skew(c, producer, main_sha=NEW)
    assert got["c"].status == "ok"
    assert got["digester"].status == "producer_differs_from_main"
    assert not any(r.red for r in got.values())


def test_newer_consumer_with_different_bytes_is_reported_not_red():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT + timedelta(days=1), schema_sha256=OLD)
    r = _skew(c, _producer())["c"]
    assert r.status == "content_differs" and not r.red


def test_unreadable_bytes_fall_back_to_time():
    old = ll.RunningContainer("c", "orion-attention-runtime", COMMIT - timedelta(days=1))
    assert _skew(old, _producer(sha=None))["c"].red
    # Producer itself older than the change on main -> nothing new is being written.
    assert not _skew(old, _producer(sha=None, built=COMMIT - timedelta(days=2)))["c"].red
    new = ll.RunningContainer("c", "orion-attention-runtime", COMMIT + timedelta(days=1))
    assert not _skew(new, _producer(sha=None))["c"].red


def test_consumer_without_running_container_is_reported_not_red():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT + timedelta(days=1), schema_sha256=NEW)
    got = _skew(c, _producer(), consumers=["orion-attention-runtime", "orion-policy-runtime"])
    assert got["orion-policy-runtime"].status == "not_running" and not got["orion-policy-runtime"].red


def test_non_consumer_containers_are_ignored():
    c = ll.RunningContainer("c", "orion-bus", COMMIT - timedelta(days=30))
    results = ll.evaluate_skew(
        FIELD_STATE, schema_commit_time=COMMIT, schema_sha256_on_ref=None, consumer_services=[], containers=[c]
    )
    assert [r.service_dir for r in results] == ["orion-field-digester"]  # producer row only
    assert results[0].status == "not_running"


@pytest.mark.parametrize(
    "label,expected",
    [
        ("/mnt/scripts/Orion-Sapienform/services/orion-attention-runtime/docker-compose.yml", "orion-attention-runtime"),
        (
            "/mnt/scripts/Orion-Sapienform/.claude/worktrees/fix+x/services/orion-proposal-runtime/docker-compose.yml",
            "orion-proposal-runtime",
        ),
        ("/x/docker-compose.yml", None),
        (None, None),
    ],
)
def test_service_dir_from_compose_label(label, expected):
    assert ll.service_dir_from_compose_files(label) == expected


# ------------------------------------------------------ consumer derivation


def test_real_repo_consumers_include_the_incident_services():
    consumers = ll.schema_consumer_services(REPO, FIELD_STATE)
    # The two services that broke on 09-20, the one that is still broken
    # (feedback), the producer, and the transitive consumer (policy, via
    # orion.policy.builder).
    assert FIELD_STATE.producer_service in consumers  # the one declared fact must stay true
    for svc in (
        "orion-attention-runtime",
        "orion-proposal-runtime",
        "orion-feedback-runtime",
        "orion-field-digester",
        "orion-policy-runtime",
    ):
        assert svc in consumers, svc
    # The registry import path is excluded while FieldStateV1 is not on the bus,
    # otherwise ~60 services would be flagged.
    assert len(consumers) < 15


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_consumer_derivation_is_transitive_and_skips_tests(tmp_path):
    _write(tmp_path, "orion/schemas/thing.py", "class ThingV1: ...\n")
    _write(tmp_path, "orion/schemas/registry.py", "from orion.schemas.thing import ThingV1\n")
    _write(tmp_path, "orion/lib/a.py", "from orion.schemas.thing import ThingV1\n")
    _write(tmp_path, "orion/lib/b.py", "import orion.lib.a\n")
    _write(tmp_path, "orion/bus/channels.yaml", "channels: {}\n")
    _write(tmp_path, "services/svc-direct/app/x.py", "from orion.schemas.thing import ThingV1\n")
    _write(tmp_path, "services/svc-transitive/app/x.py", "from orion.lib import b\nimport orion.lib.b\n")
    _write(tmp_path, "services/svc-registry/app/x.py", "from orion.schemas.registry import resolve\n")
    _write(tmp_path, "services/svc-testonly/tests/test_x.py", "from orion.schemas.thing import ThingV1\n")
    _write(tmp_path, "services/svc-none/app/x.py", "import os\n")
    schema = ll.StrictSchema("orion/schemas/thing.py", "ThingV1", "svc-direct")
    got = set(ll.schema_consumer_services(tmp_path, schema))
    assert got == {"svc-direct", "svc-transitive"}

    # A longer name that merely contains the symbol is not the schema on the bus.
    _write(tmp_path, "orion/bus/channels.yaml", "channels:\n  x:\n    schema_id: ThingV1Delta\n")
    assert "svc-registry" not in ll.schema_consumer_services(tmp_path, schema)
    # Once the schema travels on the bus, registry resolution is a real path.
    _write(tmp_path, "orion/bus/channels.yaml", "channels:\n  x:\n    schema_id: ThingV1\n")
    assert "svc-registry" in ll.schema_consumer_services(tmp_path, schema)


# ------------------------------------------------------------ CLI / notify


class _FakeClient:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def attention_request(self, **kw):
        self.calls.append(kw)
        return type("A", (), {"ok": self.ok})()


def _red_report(keys=("attention",)):
    now = datetime.now(timezone.utc)
    rep = ll.LadderReport()
    rep.rungs = [ll.RungResult(k, "stale", None, None, 900.0) for k in keys]
    return rep


def test_notify_debounces_retries_and_rearms(tmp_path):
    cli = _load_cli()
    state = str(tmp_path / "s" / "state.json")

    failing = _FakeClient(ok=False)
    assert cli.notify(_red_report(), state_file=state, base_url="x", token=None, client=failing) is False
    ok = _FakeClient()
    # Undelivered -> retried next tick.
    assert cli.notify(_red_report(), state_file=state, base_url="x", token=None, client=ok) is True
    # Delivered and still red -> silent.
    assert cli.notify(_red_report(), state_file=state, base_url="x", token=None, client=ok) is None
    # A new rung joins -> one more card.
    assert cli.notify(_red_report(("attention", "proposal")), state_file=state, base_url="x", token=None, client=ok) is True
    assert ok.calls[-1]["context"]["new_keys"] == ["rung:proposal"]
    # Verified recovery (the rungs were checked and are fresh), then recurrence -> alerts again.
    green = ll.LadderReport(rungs=[ll.RungResult(k, "fresh", None, 1.0, 900.0) for k in ("attention", "proposal")])
    assert cli.notify(green, state_file=state, base_url="x", token=None, client=ok) is None
    assert cli.notify(_red_report(), state_file=state, base_url="x", token=None, client=ok) is True
    assert len(ok.calls) == 3


def test_parse_docker_nanosecond_timestamps():
    cli = _load_cli()
    assert cli._parse_ts("2026-09-23T21:27:59.408813151Z") == datetime(2026, 9, 23, 21, 27, 59, 408813, tzinfo=timezone.utc)
    assert cli._parse_ts("0001-01-01T00:00:00Z") is None
    assert cli._parse_ts(None) is None


def test_cli_exit_codes_without_db_or_docker():
    cli = _load_cli()
    assert cli.main(["--skip-db", "--skip-docker"]) == cli.EXIT_OK


def test_schema_commit_hash_matches_file_on_head():
    cli = _load_cli()
    try:
        when, short, sha = cli.schema_commit(str(REPO), "HEAD", FIELD_STATE.path)
    except Exception as exc:  # shallow CI clone without the history
        pytest.skip(f"git history unavailable: {exc}")
    assert when.tzinfo is not None and short
    assert sha == hashlib.sha256((REPO / FIELD_STATE.path).read_bytes()).hexdigest()


def test_flaky_read_does_not_rearm_a_delivered_card(tmp_path):
    """Review finding: a rung that vanished because its query errored counted as
    recovered, so the next good tick re-sent the card."""
    cli = _load_cli()
    state = str(tmp_path / "state.json")
    ok = _FakeClient()
    assert cli.notify(_red_report(), state_file=state, base_url="x", token=None, client=ok) is True
    errored = ll.LadderReport(cannot_check=["attention: QueryCanceled"])  # rung absent, not green
    assert cli.notify(errored, state_file=state, base_url="x", token=None, client=ok) is None
    assert cli.notify(_red_report(), state_file=state, base_url="x", token=None, client=ok) is None
    assert len(ok.calls) == 1


def test_severity_is_warning_only_for_empty_consolidation_alone():
    assert _red_report(("consolidation:motifs",)).severity() == "warning"
    assert _red_report(("consolidation:motifs", "attention")).severity() == "critical"


@pytest.mark.parametrize(
    "report,code",
    [
        (ll.LadderReport(), 0),
        (_red_report(), 1),
        (ll.LadderReport(cannot_check=["postgres: down"]), 2),
        (ll.LadderReport(rungs=_red_report().rungs, cannot_check=["docker: down"]), 1),  # red wins
    ],
)
def test_cli_exit_codes(monkeypatch, report, code, capsys):
    cli = _load_cli()
    monkeypatch.setattr(cli, "build_report", lambda args: report)
    assert cli.main([]) == code


class _Cursor:
    def __init__(self, fail_on: str):
        self.fail_on = fail_on
        self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if params and self.fail_on in params:
            raise RuntimeError("canceling statement due to statement timeout")
        self._last = sql

    def fetchone(self):
        return (datetime.now(timezone.utc),)

    def fetchall(self):
        return [(2,), (3,), (2,)]


class _Conn:
    def __init__(self, fail_on):
        self.fail_on = fail_on

    def cursor(self):
        return _Cursor(self.fail_on)


def test_a_failed_rung_query_is_cannot_check_not_fresh():
    cli = _load_cli()
    newest, motifs, errors = cli.read_freshness(_Conn("orion-cortex-exec"), 3600)
    assert "grammar:orion-cortex-exec" not in newest
    assert len(errors) == 1 and errors[0].startswith("grammar:orion-cortex-exec")
    rungs = ll.evaluate_ladder(newest, datetime.now(timezone.utc), consolidation_motif_counts=motifs)
    assert "grammar:orion-cortex-exec" not in {r.rung for r in rungs}
    assert len(rungs) == len(ll.RUNGS)  # every other rung + motifs
