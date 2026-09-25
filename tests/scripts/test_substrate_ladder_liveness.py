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
FIELD_STATE = ll.StrictSchema("orion/schemas/field_state.py", "FieldStateV1")


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
    assert skewed == {"orion-athena-attention-runtime", "orion-athena-proposal-runtime"}
    msg = report.alert_message()
    assert "attention" in msg and "orion-athena-attention-runtime" in msg


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

COMMIT = datetime(2026, 9, 20, 21, 15, 13, tzinfo=timezone.utc)


def _skew(container: ll.RunningContainer, ref_sha="a" * 64, consumers=None):
    return ll.evaluate_skew(
        FIELD_STATE,
        schema_commit_time=COMMIT,
        schema_sha256_on_ref=ref_sha,
        consumer_services=consumers or [container.service_dir],
        containers=[container],
    )


def test_matching_content_is_ok_even_with_an_old_image():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT - timedelta(days=12), schema_sha256="a" * 64)
    assert _skew(c)[0].status == "ok"


def test_old_image_with_different_content_is_red():
    c = ll.RunningContainer("c", "orion-feedback-runtime", COMMIT - timedelta(days=6), schema_sha256="b" * 64)
    r = _skew(c)[0]
    assert r.status == "skew" and r.red


def test_old_image_with_unreadable_content_falls_back_to_time():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT - timedelta(days=1), schema_sha256=None)
    assert _skew(c)[0].red


def test_newer_image_with_different_content_is_reported_not_red():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT + timedelta(days=1), schema_sha256="b" * 64)
    r = _skew(c)[0]
    assert r.status == "content_differs" and not r.red


def test_consumer_without_running_container_is_reported_not_red():
    c = ll.RunningContainer("c", "orion-attention-runtime", COMMIT + timedelta(days=1))
    results = _skew(c, consumers=["orion-attention-runtime", "orion-policy-runtime"])
    policy = next(r for r in results if r.service_dir == "orion-policy-runtime")
    assert policy.status == "not_running" and not policy.red


def test_non_consumer_containers_are_ignored():
    c = ll.RunningContainer("c", "orion-bus", COMMIT - timedelta(days=30))
    assert ll.evaluate_skew(
        FIELD_STATE, schema_commit_time=COMMIT, schema_sha256_on_ref=None, consumer_services=[], containers=[c]
    ) == []


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
    schema = ll.StrictSchema("orion/schemas/thing.py", "ThingV1")
    got = set(ll.schema_consumer_services(tmp_path, schema))
    assert got == {"svc-direct", "svc-transitive"}

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
    # Recovery, then recurrence -> alerts again.
    assert cli.notify(ll.LadderReport(), state_file=state, base_url="x", token=None, client=ok) is None
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
