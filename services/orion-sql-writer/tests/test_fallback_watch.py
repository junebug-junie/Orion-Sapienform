"""Tests for the bus_fallback_log backlog watcher.

The watcher decides whether Juniper's phone buzzes. Two failure modes matter
about equally and pull in opposite directions:

- staying silent while unrouted events pile up (the thing it exists to prevent --
  it happened twice on 2026-08-13/14, for hours each time), and
- re-sending an alert that was already sent, which is how a monitor gets
  filtered into a folder nobody reads and becomes the first failure again.

The escalation rule is therefore kept in two pure functions with no clock, no
database and no network, so both properties are asserted directly.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (SERVICE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.fallback_watch import (  # noqa: E402
    STATE_ROW_ID,
    format_alert,
    next_alert_threshold,
    reached_threshold,
)

STEP = 5


# --------------------------------------------------------------------------
# The escalation rule
# --------------------------------------------------------------------------


def test_quiet_below_the_first_threshold() -> None:
    assert next_alert_threshold(4, STEP, 0) == (None, 0)


def test_first_crossing_alerts_at_the_step() -> None:
    """`count >= threshold`, not `count > threshold`.

    A literal reading of "alert if it goes above 5" would wait for 6. Erring on
    the noisy side by one row is the right direction for a monitor whose whole
    history is failing silently, and the alert names the level it reached.
    """
    assert next_alert_threshold(5, STEP, 0) == (5, 5)


def test_no_second_alert_while_it_sits_between_levels() -> None:
    """The anti-spam property. At a 300s poll, a backlog parked at 7 for a week
    is one email rather than 2,016 of them."""
    assert next_alert_threshold(7, STEP, 5) == (None, 5)


def test_repeated_evaluation_at_an_unchanged_count_is_silent() -> None:
    mark = 0
    threshold, mark = next_alert_threshold(12, STEP, mark)
    assert threshold == 10
    for _ in range(50):
        threshold, mark = next_alert_threshold(12, STEP, mark)
        assert threshold is None
    assert mark == 10


def test_each_new_level_alerts_once() -> None:
    mark = 0
    fired = []
    for count in (3, 5, 6, 9, 10, 11, 14, 15):
        threshold, mark = next_alert_threshold(count, STEP, mark)
        if threshold is not None:
            fired.append(threshold)
    assert fired == [5, 10, 15]


def test_a_jump_past_several_levels_sends_one_alert_naming_the_level_reached() -> None:
    """A burst that goes 0 -> 17 between two polls must not produce three
    emails. One alert, naming 15 -- the level actually reached."""
    assert next_alert_threshold(17, STEP, 0) == (15, 15)


def test_draining_re_arms_the_lower_levels() -> None:
    """Recovery is what re-arms, and it must be silent while it happens."""
    threshold, mark = next_alert_threshold(2, STEP, 15)
    assert threshold is None
    assert mark == 0


def test_a_new_incident_after_recovery_alerts_again() -> None:
    """The point of re-arming: a backlog that drained and climbed again is a new
    incident, not a repeat of the old one, and must be reported as such."""
    _, mark = next_alert_threshold(12, STEP, 0)
    _, mark = next_alert_threshold(0, STEP, mark)
    assert mark == 0
    assert next_alert_threshold(5, STEP, mark) == (5, 5)


def test_a_partial_drain_does_not_re_arm_anything() -> None:
    """15 -> 11 holds the mark at 15. It does not ratchet down to 10.

    Ratcheting down level-by-level is the intuitive implementation and it is
    wrong, because the count is taken over a TRAILING window and drifts in both
    directions all day as rows age out. See the next test.
    """
    threshold, mark = next_alert_threshold(11, STEP, 15)
    assert threshold is None
    assert mark == 15


def test_oscillating_across_a_boundary_does_not_re_alert() -> None:
    """The bug this file shipped in its first version.

    Review replayed the escalation functions over the real 87 created_at_ts
    values in the live bus_fallback_log at the shipped 300s/86400s/step-5
    settings: 12 alerts across 20 days, SIX of them "crossed 5". A count
    alternating 14/15 between polls alerted every other poll -- about 144 emails
    a day. The window is trailing, so this oscillation is the normal state of
    the metric, not an edge case.
    """
    mark = 0
    fired = []
    for count in [15, 14, 15, 13, 15, 14, 15, 11, 15, 14, 15]:
        threshold, mark = next_alert_threshold(count, STEP, mark)
        if threshold is not None:
            fired.append(threshold)
    assert fired == [15], f"oscillation across the 15 boundary re-alerted: {fired}"


def test_only_a_full_drain_below_the_step_re_arms() -> None:
    _, mark = next_alert_threshold(22, STEP, 0)
    assert mark == 20
    for count in (19, 12, 7, 5):
        threshold, mark = next_alert_threshold(count, STEP, mark)
        assert threshold is None, f"count={count} re-alerted"
        assert mark == 20, f"count={count} ratcheted the mark down to {mark}"
    threshold, mark = next_alert_threshold(4, STEP, mark)
    assert (threshold, mark) == (None, 0)


@pytest.mark.parametrize("step", [0, -1])
def test_a_nonpositive_step_never_alerts_instead_of_dividing_by_zero(step: int) -> None:
    """Misconfiguration must degrade to silence, not to a crash inside the
    background loop."""
    assert reached_threshold(1000, step) == 0
    assert next_alert_threshold(1000, step, 0) == (None, 0)


def test_reached_threshold_floors_to_the_multiple() -> None:
    assert [reached_threshold(n, STEP) for n in (0, 4, 5, 9, 10, 24, 25)] == [0, 0, 5, 5, 10, 20, 25]


def test_a_custom_step_is_honoured() -> None:
    assert next_alert_threshold(20, 20, 0) == (20, 20)
    assert next_alert_threshold(19, 20, 0) == (None, 0)


# --------------------------------------------------------------------------
# The alert body
# --------------------------------------------------------------------------


def test_alert_body_names_kinds_and_counts() -> None:
    title, body = format_alert(13, 10, 86400, [("legacy.message", 8), ("juniper.affective_state.v1", 5)])
    assert "10" in title and "13" in title
    assert "legacy.message: 8" in body
    assert "juniper.affective_state.v1: 5" in body


def test_alert_body_caps_the_kind_list() -> None:
    """A pathological spread of kinds must not produce a thousand-line email."""
    by_kind = [(f"kind.{i}", 1) for i in range(40)]
    _, body = format_alert(40, 40, 86400, by_kind)
    assert "... and 30 more kind(s)" in body
    assert "kind.39" not in body


def test_window_label_is_readable_for_whole_hours() -> None:
    title, _ = format_alert(5, 5, 86400, [("x", 5)])
    assert "24h" in title


# --------------------------------------------------------------------------
# Lifecycle against a real database
# --------------------------------------------------------------------------


class _Settings:
    """Only the fields the watcher reads."""

    sql_writer_fallback_watch_window_sec = 86400
    sql_writer_fallback_watch_threshold_step = STEP
    sql_writer_fallback_watch_interval_sec = 300
    notify_service_url = "http://notify:7140"
    notify_api_token = ""


@pytest.fixture()
def db(monkeypatch: pytest.MonkeyPatch):
    """In-memory SQLite holding just the two tables the watcher touches.

    A real session, real SQL, real commits -- the arithmetic above is already
    covered, and what is left to get wrong is the part the pure functions cannot
    see: that the high-water mark actually round-trips through the database. An
    in-process variable would pass every test above and still re-send every
    alert on every restart.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.db as app_db
    from app.models.fallback_alert_state import BusFallbackAlertState
    from app.models.fallback_log import BusFallbackLog

    engine = create_engine("sqlite://")
    for model in (BusFallbackLog, BusFallbackAlertState):
        model.__table__.create(bind=engine)

    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(app_db, "remove_session", lambda: None)
    return engine, factory


def _add_fallback_rows(factory, count: int, kind: str = "legacy.message", *, age_seconds: int = 60) -> None:
    from app.models.fallback_log import BusFallbackLog

    session = factory()
    # SQLite's DateTime column is naive; use naive UTC so the comparison in
    # count_backlog is apples-to-apples here. Production is timestamptz and the
    # same expression is a plain range scan there.
    stamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=age_seconds)
    for _ in range(count):
        session.add(BusFallbackLog(kind=kind, correlation_id="c", payload={}, created_at_ts=stamp))
    session.commit()
    session.close()


def _evaluate(factory, sent: list):
    import app.fallback_watch as fw

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return fw.evaluate_once(_Settings(), factory, now=now)


@pytest.fixture()
def sent(monkeypatch: pytest.MonkeyPatch) -> list:
    import app.fallback_watch as fw

    calls: list = []

    def _fake_send(settings, title, body, threshold, count, window_seconds):
        calls.append({"threshold": threshold, "count": count, "title": title, "body": body})
        return "sent"

    monkeypatch.setattr(fw, "_send_alert", _fake_send)
    return calls


def test_backlog_below_threshold_sends_nothing(db, sent) -> None:
    _, factory = db
    _add_fallback_rows(factory, 4)
    result = _evaluate(factory, sent)
    assert result["count"] == 4
    assert sent == []


def test_first_crossing_sends_once_and_persists_the_mark(db, sent) -> None:
    from app.models.fallback_alert_state import BusFallbackAlertState

    _, factory = db
    _add_fallback_rows(factory, 6)
    result = _evaluate(factory, sent)

    assert result["threshold"] == 5
    assert [c["threshold"] for c in sent] == [5]

    session = factory()
    state = session.get(BusFallbackAlertState, STATE_ROW_ID)
    assert state.last_alerted_threshold == 5
    assert state.last_count == 6
    assert state.last_alert_status == "sent"
    session.close()


def test_the_mark_survives_a_restart_and_does_not_resend(db, sent) -> None:
    """The regression this whole table exists for.

    `evaluate_once` re-reads state from the database on every call, so a fresh
    session -- which is what a restarted container gets -- must find the mark
    already at 5 and stay quiet. An in-process high-water mark passes every pure
    test above and fails here, re-alerting on each of this service's many
    redeploys.
    """
    _, factory = db
    _add_fallback_rows(factory, 6)
    _evaluate(factory, sent)
    assert len(sent) == 1

    for _ in range(5):
        _evaluate(factory, sent)
    assert len(sent) == 1, "re-evaluating an unchanged backlog re-sent the alert"


def test_climbing_to_the_next_level_sends_again(db, sent) -> None:
    _, factory = db
    _add_fallback_rows(factory, 6)
    _evaluate(factory, sent)
    _add_fallback_rows(factory, 5)
    result = _evaluate(factory, sent)

    assert result["count"] == 11
    assert [c["threshold"] for c in sent] == [5, 10]


def test_events_outside_the_window_are_not_counted(db, sent) -> None:
    """Otherwise the count is cumulative-forever and every threshold is crossed
    exactly once in the table's lifetime, which is not a monitor."""
    _, factory = db
    _add_fallback_rows(factory, 20, age_seconds=90000)  # > 24h old
    _add_fallback_rows(factory, 2)
    result = _evaluate(factory, sent)

    assert result["count"] == 2
    assert sent == []


def test_rows_with_no_timestamp_are_excluded_rather_than_treated_as_now(db, sent) -> None:
    from app.models.fallback_log import BusFallbackLog

    _, factory = db
    session = factory()
    for _ in range(9):
        session.add(BusFallbackLog(kind="legacy.message", payload={}, created_at_ts=None))
    session.commit()
    session.close()

    result = _evaluate(factory, sent)
    assert result["count"] == 0
    assert sent == []


def test_no_payload_content_reaches_the_alert(db, sent) -> None:
    """The privacy boundary, tested where payloads are actually in reach.

    Fallback rows keep whatever the undelivered event held, and for the largest
    current contributor (`legacy.message`, 80 of the 87 rows present when this
    was written) that is Orion's own prompts, responses and reasoning traces.
    The alert goes to email, which leaves the box. Rows here carry a sentinel
    payload and the assertion is on the fully-rendered title and body, so the
    natural "make this alert more useful" edit -- pasting in an example row --
    fails loudly instead of mailing out cognition.
    """
    from app.models.fallback_log import BusFallbackLog

    _, factory = db
    session = factory()
    stamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=60)
    for _ in range(6):
        session.add(
            BusFallbackLog(
                kind="legacy.message",
                correlation_id="SENTINEL_CORRELATION",
                payload={"prompt": "SENTINEL_PROMPT", "response": "SENTINEL_RESPONSE"},
                created_at_ts=stamp,
            )
        )
    session.commit()
    session.close()

    _evaluate(factory, sent)
    assert len(sent) == 1

    rendered = sent[0]["title"] + "\n" + sent[0]["body"]
    assert "legacy.message: 6" in rendered, "the alert should still be useful"
    for sentinel in ("SENTINEL_PROMPT", "SENTINEL_RESPONSE", "SENTINEL_CORRELATION"):
        assert sentinel not in rendered, sentinel


def test_an_unconfigured_notify_url_does_not_consume_the_crossing(db, sent) -> None:
    """The one failure that must NOT advance the mark.

    A failed HTTP send does advance it, deliberately -- otherwise one unreachable
    notify service becomes an alert attempt every 300s. But an empty
    NOTIFY_SERVICE_URL costs no network to retry, and burning the only alert for
    a level because the key happened to be blank would lose it permanently.
    """
    import app.fallback_watch as fw
    from app.models.fallback_alert_state import BusFallbackAlertState

    _, factory = db

    class _Unconfigured(_Settings):
        notify_service_url = ""

    _add_fallback_rows(factory, 6)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    result = fw.evaluate_once(_Unconfigured(), factory, now=now)

    assert result["threshold"] is None
    assert sent == []
    session = factory()
    state = session.get(BusFallbackAlertState, STATE_ROW_ID)
    assert state.last_alerted_threshold == 0, "the crossing was consumed with nowhere to send it"
    session.close()

    # ...and once configured, the same backlog still alerts.
    result = fw.evaluate_once(_Settings(), factory, now=now)
    assert [c["threshold"] for c in sent] == [5]


def test_production_defaults_now_to_wall_clock(db, sent) -> None:
    """Every other DB test passes an explicit `now`, so the `now or
    datetime.now(timezone.utc)` default -- the only branch production takes --
    would otherwise never be exercised."""
    import app.fallback_watch as fw

    _, factory = db
    _add_fallback_rows(factory, 6, age_seconds=60)
    result = fw.evaluate_once(_Settings(), factory)

    assert result["count"] == 6
    assert [c["threshold"] for c in sent] == [5]


def test_by_kind_breakdown_is_busiest_first(db, sent) -> None:
    _, factory = db
    _add_fallback_rows(factory, 2, kind="thought.event.v1")
    _add_fallback_rows(factory, 6, kind="legacy.message")
    result = _evaluate(factory, sent)

    assert result["by_kind"] == [("legacy.message", 6), ("thought.event.v1", 2)]


def test_a_failed_send_does_not_retry_forever(db, monkeypatch: pytest.MonkeyPatch) -> None:
    """If notify is down, the mark still advances.

    Retrying on the next poll would turn one unreachable notify service into an
    alert attempt every 300s for as long as it stays down, and the moment it came
    back it would deliver the backlog of them. The failure is recorded on the
    state row and logged at ERROR instead.
    """
    import app.fallback_watch as fw
    from app.models.fallback_alert_state import BusFallbackAlertState

    _, factory = db
    attempts: list = []
    monkeypatch.setattr(
        fw, "_send_alert", lambda *a, **k: (attempts.append(1), "failed")[1]
    )

    _add_fallback_rows(factory, 6)
    _evaluate(factory, [])
    _evaluate(factory, [])

    assert len(attempts) == 1
    session = factory()
    state = session.get(BusFallbackAlertState, STATE_ROW_ID)
    assert state.last_alerted_threshold == 5
    assert state.last_alert_status == "failed"
    # A column named `last_alert_sent_at` must not be stamped by an alert that
    # was never sent -- otherwise the state row tells an operator the opposite
    # of what happened.
    assert state.last_alert_sent_at is None
    session.close()


def test_drain_then_climb_alerts_again_through_the_database(db, sent) -> None:
    from app.models.fallback_log import BusFallbackLog

    _, factory = db
    _add_fallback_rows(factory, 6)
    _evaluate(factory, sent)
    assert len(sent) == 1

    session = factory()
    session.query(BusFallbackLog).delete()
    session.commit()
    session.close()

    _evaluate(factory, sent)  # drained -> re-arm, silent
    assert len(sent) == 1

    _add_fallback_rows(factory, 5)
    _evaluate(factory, sent)
    assert [c["threshold"] for c in sent] == [5, 5]


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_requests_is_a_declared_dependency() -> None:
    """No unit test can catch this one -- the dev venv has `requests`.

    orion/notify/client.py imports it at module level, and this service's
    requirements.txt did not list it. Verified against the live image before this
    was added: `from orion.notify.client import NotifyClient` raised
    ModuleNotFoundError inside orion-athena-sql-writer. The watcher would have
    logged fallback_watch_tick_failed every 300s and never sent anything --
    the monitor built to end silent failures, failing silently.
    """
    text = (SERVICE_ROOT / "requirements.txt").read_text()
    declared = {
        line.split("==")[0].split(">=")[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "requests" in declared


def test_notify_client_import_chain_is_satisfiable() -> None:
    """The actual import the alert path performs, exercised eagerly rather than
    only at the moment the first real alert fires."""
    from orion.notify.client import NotifyClient

    assert NotifyClient is not None


def test_watcher_is_on_by_default() -> None:
    """A monitor that ships disabled is a monitor that is off in production, and
    the failure mode here is silence being mistaken for health."""
    from app.settings import Settings

    assert Settings.model_fields["sql_writer_fallback_watch_enabled"].default is True


def test_default_step_is_five() -> None:
    from app.settings import Settings

    assert Settings.model_fields["sql_writer_fallback_watch_threshold_step"].default == 5


def test_env_example_carries_every_watcher_key() -> None:
    """`.env_example` is the operator contract; a key that lives only in code
    silently takes its default on a machine whose `.env` predates it."""
    text = (SERVICE_ROOT / ".env_example").read_text()
    for key in (
        "SQL_WRITER_FALLBACK_WATCH_ENABLED",
        "SQL_WRITER_FALLBACK_WATCH_INTERVAL_SEC",
        "SQL_WRITER_FALLBACK_WATCH_WINDOW_SEC",
        "SQL_WRITER_FALLBACK_WATCH_THRESHOLD_STEP",
        "NOTIFY_SERVICE_URL",
    ):
        assert f"{key}=" in text, key


def test_notify_host_is_the_compose_service_name() -> None:
    """`orion-notify` (directory) and `orion-athena-notify` (container_name) have
    each already broken this exact URL in another service; only the compose
    service name `notify` resolves on the shared network. Verified live
    2026-08-14 from inside orion-athena-sql-writer."""
    text = (SERVICE_ROOT / ".env_example").read_text()
    assert "NOTIFY_SERVICE_URL=http://notify:7140" in text


_NOTIFY_PKG = "_orion_notify_under_test"


def _load_notify_module(modname: str):
    """Import a module from services/orion-notify/app under a private package name.

    Two constraints force this shape rather than a plain import:

    - orion-notify's package is ALSO called `app`. Importing it normally
      replaces this service's `app` in sys.modules and breaks every other test
      in the directory depending on collection order -- cross-test contamination
      is already a live problem here.
    - `email_delivery.py` does `from .policy import Policy`, so a bare
      spec_from_file_location load raises "attempted relative import with no
      known parent package". A synthetic parent package with __path__ set gives
      the relative import something to resolve against.

    Modules are registered in sys.modules before exec because @dataclass
    resolves string annotations through sys.modules[cls.__module__].
    """
    import importlib.util
    import types

    app_dir = REPO_ROOT / "services/orion-notify/app"
    if _NOTIFY_PKG not in sys.modules:
        pkg = types.ModuleType(_NOTIFY_PKG)
        pkg.__path__ = [str(app_dir)]
        sys.modules[_NOTIFY_PKG] = pkg

    full = f"{_NOTIFY_PKG}.{modname}"
    spec = importlib.util.spec_from_file_location(full, app_dir / f"{modname}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def _unload_notify_modules() -> None:
    for name in [n for n in sys.modules if n == _NOTIFY_PKG or n.startswith(_NOTIFY_PKG + ".")]:
        sys.modules.pop(name, None)


def test_error_severity_is_what_actually_triggers_the_email() -> None:
    """Asserted against the real gate on the real path.

    `POST /notify` -- the handler NotifyClient.send() calls -- never invokes
    Policy.evaluate. Email is decided solely by should_send_email() in
    app/email_delivery.py. An earlier version of this test asserted on
    rules.yaml, which is a real file that this path does not read: it would have
    kept passing if someone changed should_send_email to exclude `error`.
    """
    try:
        module = _load_notify_module("email_delivery")
        from orion.schemas.notify import NotificationRequest

        def _request(**kw):
            return NotificationRequest(
                source_service="orion-sql-writer",
                event_kind="sqlwriter.fallback_backlog",
                title="t",
                **kw,
            )

        sends, reason = module.should_send_email(_request(severity="error"))
        assert sends, reason
        # And the belt-and-braces channels_requested keeps it sending even if the
        # severity is later softened.
        sends, _ = module.should_send_email(
            _request(severity="warning", channels_requested=["email", "in_app"])
        )
        assert sends
        # Sanity: the gate is real, not always-true.
        assert not module.should_send_email(_request(severity="warning"))[0]
    finally:
        _unload_notify_modules()


def test_the_real_outbound_request_passes_the_real_email_gate() -> None:
    """End to end without a network call: build the actual request this watcher
    sends, then feed it to orion-notify's actual email gate."""
    from app.fallback_watch import build_alert_request

    try:
        module = _load_notify_module("email_delivery")
        request = build_alert_request("title", "body", 15, 17, 86400)
        sends, reason = module.should_send_email(request)
        assert sends, f"the alert would not be emailed: {reason}"
        assert "in_app" in (request.channels_requested or [])
    finally:
        _unload_notify_modules()


def test_the_outbound_request_context_carries_no_row_content() -> None:
    """`context` is echoed into the Hub card and the durable notify record."""
    from app.fallback_watch import build_alert_request

    request = build_alert_request("title", "body", 15, 17, 86400)
    assert set(request.context) == {"threshold", "count", "window_seconds"}
    assert all(isinstance(v, int) for v in request.context.values())


def test_severity_survives_quiet_hours() -> None:
    """`Policy.evaluate` drops channels to [] during quiet hours for anything
    that is not critical or error. A backlog at 3am is still a backlog, so this
    pins that `error` is on the exempt side of that check.

    Loaded by file path under a private module name rather than by import:
    orion-notify's package is also called `app`, and importing it normally would
    replace this service's `app` in ``sys.modules`` and break every other test
    in the suite depending on collection order. That cross-test contamination is
    a live problem in this directory already.
    """
    import importlib.util
    from datetime import datetime as dt

    policy_path = REPO_ROOT / "services/orion-notify/app/policy.py"
    name = "_orion_notify_policy_under_test"
    spec = importlib.util.spec_from_file_location(name, policy_path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: policy.py uses @dataclass, and dataclasses resolves
    # string annotations via sys.modules[cls.__module__], which is None for a
    # module loaded by path alone.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)

        policy = module.Policy.load(str(REPO_ROOT / "services/orion-notify/app/policy/rules.yaml"))
        at_2am_denver = dt(2026, 8, 14, 8, 0, tzinfo=timezone.utc)
        assert policy.is_quiet_hours(at_2am_denver)

        from orion.schemas.notify import NotificationRequest

        def _channels(severity: str) -> list[str]:
            request = NotificationRequest(
                source_service="orion-sql-writer",
                event_kind="sqlwriter.fallback_backlog",
                severity=severity,
                title="t",
            )
            return policy.evaluate(request, at_2am_denver).channels

        assert set(_channels("error")) >= {"email", "in_app"}
        # ...and the tempting "less alarmist" alternative is silently muted at 2am.
        assert _channels("warning") == []
    finally:
        sys.modules.pop(name, None)
