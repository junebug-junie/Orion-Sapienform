"""Self-signal -> outcome rollup (equilibrium adapter, spec 2026-10-07 section E): due hours,
read failures skip (never write calm), restart does not recompute persisted hours, the episode
open row carries the association only when one renders, settings/flag wiring."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orion.attention.self_signal_outcomes as sso  # noqa: E402
from app.pe_history_reader import NodePeHistoryReadError  # noqa: E402
from app.self_signal_episode_gate import (  # noqa: E402
    GateConfig,
    SelfSignalEpisodeGate,
    observe_signals,
)
from app.self_signal_outcome_rollup import (  # noqa: E402
    MAX_HOURS_PER_CYCLE,
    SelfSignalOutcomeRollup,
    association_payload,
    body_signal_ids,
)
from orion.metacog.evidence_map import map_trigger  # noqa: E402
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1  # noqa: E402
from orion.schemas.self_signal_outcome import SelfSignalOutcomeAssociationV1  # noqa: E402

NOW = datetime(2026, 10, 14, 5, 20, tzinfo=timezone.utc)
BIO = "node:substrate.biometrics"


class FakeReader:
    def __init__(self, persisted=(), fail_on: str = "", associations=()):
        self.persisted = list(persisted)
        self.associations = list(associations)
        self.fail_on = fail_on
        self.calls: list[str] = []

    def _maybe_fail(self, name):
        self.calls.append(name)
        if self.fail_on == name:
            raise NodePeHistoryReadError("boom")

    def fetch_hour_rows(self, since):
        self._maybe_fail("hours")
        return list(self.persisted)

    def fetch_association_rows(self, since, holds):
        self._maybe_fail("associations")
        return [a for a in self.associations if a.computed_at >= since]

    def fetch_turn_rows(self, start, end):
        self._maybe_fail("turns")
        return [(start + timedelta(minutes=5), "failed", True)]

    def fetch_rpc_rows(self, start, end):
        self._maybe_fail("rpc")
        return [(start, "orion:exec:request:LLMGatewayService", 10, 1)]


def _history(start: datetime, end: datetime):
    rows, t, k = [], start, 0
    while t < end:
        rows.append((t, 0.01 + (k * 7919 % 1000) / 10000.0))
        t += timedelta(seconds=35)
        k += 1
    return rows


HIST = {BIO: _history(NOW - timedelta(days=7), NOW), "node:substrate.harness_closure": [(NOW, 0.65)],
        "node:substrate.perception": [(NOW, 0.1)]}


def _cycle(rollup, reader, now=NOW, history_start=None):
    return rollup.run_cycle(
        now=now, reader=reader, history=lambda n: HIST.get(n, []), node_ids=list(HIST),
        history_start=history_start or (NOW - timedelta(days=7)),
    )


def test_body_signals_exclude_world_and_placeholders() -> None:
    assert body_signal_ids(HIST) == [BIO]


def test_due_hours_wait_for_grace_and_stay_inside_the_window() -> None:
    r = SelfSignalOutcomeRollup(grace_sec=600)
    due = r.due_hours(NOW, NOW - timedelta(days=30))
    # 05:20 - 10 min grace -> hour 04:00 is the last complete one.
    assert due[-1] == datetime(2026, 10, 14, 4, tzinfo=timezone.utc)
    assert due[0] >= sso.hour_floor(NOW - timedelta(days=7))
    assert len(due) <= MAX_HOURS_PER_CYCLE
    early = datetime(2026, 10, 14, 5, 5, tzinfo=timezone.utc)
    assert r.due_hours(early, NOW - timedelta(days=30))[-1] == datetime(2026, 10, 14, 3, tzinfo=timezone.utc)
    assert r.due_hours(NOW, None) == []


def test_a_cycle_writes_hour_rows_and_every_association() -> None:
    r = SelfSignalOutcomeRollup()
    res = _cycle(r, FakeReader())
    assert not res.skipped
    assert len(res.hour_rows) == len(r.known_hours) > 100
    row = res.hour_rows[0]  # the fake reader's turn and RPC rows sit in the first due hour
    assert [s.signal_id for s in row.signals] == [BIO]
    assert {o.outcome for o in row.outcomes} == {"turn_failed", "response_repair", "rpc_timeout"}
    assert len(res.associations) == 3  # one signal x three outcomes
    again = _cycle(r, FakeReader())
    assert again.skipped == "nothing_due"
    assert not r.needs_cycle(NOW, NOW - timedelta(days=7))


def test_a_read_failure_skips_the_cycle_and_writes_nothing() -> None:
    for where in ("hours", "associations", "turns", "rpc"):
        r = SelfSignalOutcomeRollup()
        res = _cycle(r, FakeReader(fail_on=where))
        assert res.skipped.startswith("read_failed"), where
        assert res.hour_rows == [] and r.latest == []
        assert r.known_hours == {} and r.history == []


def test_restart_does_not_recompute_persisted_hours() -> None:
    first = SelfSignalOutcomeRollup()
    persisted = _cycle(first, FakeReader()).hour_rows
    second = SelfSignalOutcomeRollup()
    res = _cycle(second, FakeReader(persisted=persisted))
    assert res.hour_rows == []  # every due hour is already in self_signal_hour
    assert len(res.associations) == 3  # but associations are recomputed for the episode gate


def _assoc(clause: str, verdict: str = "tends_more") -> SelfSignalOutcomeAssociationV1:
    return SelfSignalOutcomeAssociationV1(
        association_id="ssa-0123456789", computed_at=NOW, signal_id="node:substrate.bus_synaptic",
        outcome="turn_failed", n_high=30, events_high=18, n_normal=60, events_normal=6,
        rate_high=0.6, rate_normal=0.1, ratio=6.0, ci_low=2.7, ci_high=13.5, min_hours=24,
        verdict=verdict, clause=clause,  # type: ignore[arg-type]
    )


CLAUSE = ("when this is high, my next hour had a failed turn 6.0x as often at equal activity (60% of 30 high hours vs "
          "10% of 60 normal) [assoc:ssa-0123456789]")


def test_renderable_only_holds_clauses() -> None:
    r = SelfSignalOutcomeRollup(consecutive_holds=1)
    quiet = _assoc("", verdict="below_min_n").model_copy(update={"outcome": "rpc_timeout",
                                                                 "association_id": "ssa-9999999999"})
    r.latest = [_assoc(CLAUSE), quiet]
    assert dict(r.renderable(NOW)) == {}  # nothing published yet: an open row must not cite it
    r.published_ids = {"ssa-0123456789", "ssa-9999999999"}
    assert list(r.renderable(NOW)) == ["node:substrate.bus_synaptic"]


def _hourly(n: int, *, verdict: str = "tends_more", end: datetime = NOW) -> list[SelfSignalOutcomeAssociationV1]:
    """n hourly computations of the bus_synaptic x turn_failed pair, oldest first."""
    out = []
    for i in range(n):
        at = end - timedelta(hours=n - 1 - i)
        tends = verdict.startswith("tends")
        out.append(_assoc(CLAUSE if tends else "", verdict=verdict).model_copy(
            update={"association_id": f"ssa-{i:010d}", "computed_at": at}))
    return out


def _published(rows):
    r = SelfSignalOutcomeRollup()
    r.history, r.latest = list(rows[:-1]), [rows[-1]]
    r.published_ids = {rows[-1].association_id}
    return r


def test_open_row_clause_needs_six_consecutive_hourly_holds() -> None:
    assert SelfSignalOutcomeRollup().consecutive_holds == 6
    assert list(_published(_hourly(6)).renderable(NOW)) == [BUS]
    assert dict(_published(_hourly(5)).renderable(NOW)) == {}
    gap = _hourly(7)
    del gap[2]
    assert dict(_published(gap).renderable(NOW)) == {}
    flip = _hourly(6)
    flip[3] = flip[3].model_copy(update={"verdict": "no_clear_difference", "clause": ""})
    assert dict(_published(flip).renderable(NOW)) == {}


def test_a_stale_computation_stops_the_open_row_clause_like_thought() -> None:
    r = _published(_hourly(6))
    assert list(r.renderable(NOW + timedelta(hours=2, minutes=59))) == [BUS]
    assert dict(r.renderable(NOW + timedelta(hours=3, minutes=1))) == {}


def test_an_unpublished_current_row_never_falls_back_to_last_hour() -> None:
    rows = _hourly(7)
    r = _published(rows)
    r.published_ids = set()
    assert dict(r.renderable(NOW)) == {}


def test_equilibrium_and_the_shared_rule_agree() -> None:
    """orion-thought's reader is sso.index_by_signal over persisted rows (its tests pin that);
    equilibrium's open rows must say exactly the same thing over the same rows."""
    cases = [_hourly(6), _hourly(5), _hourly(9), _hourly(6, verdict="no_clear_difference")]
    flip = _hourly(8)
    flip[4] = flip[4].model_copy(update={"verdict": "tends_less"})
    cases.append(flip)
    for rows in cases:
        want = {k: v.association_id for k, v in sso.index_by_signal(rows).items()}
        got = {k: v.association_id for k, v in _published(rows).renderable(NOW).items()}
        assert got == want


def test_a_restart_seeds_the_run_from_persisted_rows_and_cycles_extend_it() -> None:
    """A run already going is not reset by a restart, and each published cycle joins the run."""
    first = SelfSignalOutcomeRollup()
    persisted = _cycle(first, FakeReader()).hour_rows
    seeded = _hourly(5, end=NOW - timedelta(hours=1))
    r = SelfSignalOutcomeRollup()
    reader = FakeReader(persisted=persisted, associations=seeded)
    _cycle(r, reader)
    assert [a.association_id for a in r.history] == [a.association_id for a in seeded]
    # The fake world has no clause of its own; pretend this hour's computation repeated the pair.
    r.latest = [_assoc(CLAUSE).model_copy(update={"association_id": "ssa-now", "computed_at": NOW})]
    r.published_ids = {"ssa-now"}
    assert r.renderable(NOW)[BUS].association_id == "ssa-now"
    # Next hour: the published row moves into history, old hours beyond the window are trimmed.
    nxt = NOW + timedelta(hours=1)
    r.computed = False
    r.run_cycle(now=nxt, reader=reader, history=lambda n: HIST.get(n, []), node_ids=list(HIST),
                history_start=NOW - timedelta(days=7))
    ids = [a.association_id for a in r.history]
    assert "ssa-now" in ids and all(a.computed_at >= nxt - timedelta(hours=8) for a in r.history)


# --- the episode open row ------------------------------------------------------------------------

T0 = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)
BUS = "node:substrate.bus_synaptic"
HOT = PredictionErrorMagnitudeV1(value=0.12, age_sec=5.0, percentile_now=0.97, n_readings_7d=5000,
                                 band="high", p50_7d=0.03, p90_7d=0.06)


def _open_row(tends_to):
    gate = SelfSignalEpisodeGate(GateConfig())
    gate.cold_start = False
    outs = []
    for k in range(12):
        now = T0 + timedelta(seconds=30 * k)
        obs = observe_signals({BUS: (HOT, now - timedelta(seconds=5), None)}, now=now)
        for out in gate.step(obs, zen_state="zen", pressure=0.1, recall_enabled=False, tends_to=tends_to):
            gate.mark_published(out, now=now.timestamp())
            outs.append(out)
    opens = [o for o in outs if o.phase == "open"]
    assert len(opens) == 1
    return opens[0].trigger


def test_open_row_carries_the_association_and_its_evidence_cites_it() -> None:
    trig = _open_row({BUS: association_payload(_assoc(CLAUSE))})
    assert trig.upstream["tends_to"]["association_id"] == "ssa-0123456789"
    assert trig.upstream["tends_to"]["source_table"] == "self_signal_outcome_association"
    mapped = map_trigger(trig.trigger_kind, trig.reason, trig.upstream)
    assert CLAUSE in mapped.evidence


def test_no_association_open_row_is_exactly_as_before() -> None:
    plain = _open_row(None)
    empty = _open_row({})
    assert "tends_to" not in plain.upstream
    assert plain.upstream == empty.upstream
    assert all("when this is high" not in e for e in map_trigger(plain.trigger_kind, plain.reason, plain.upstream).evidence)


# --- settings / flag -----------------------------------------------------------------------------


def test_flag_ships_on_and_is_in_every_config_surface() -> None:
    from app.settings import Settings

    s = Settings()
    assert s.self_signal_outcome_association_enabled is True
    assert s.self_signal_outcome_min_hours == 24
    assert s.self_signal_outcome_consecutive_holds == 6
    env = (ROOT / ".env_example").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    for key in ("EQUILIBRIUM_SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED=true",
                "EQUILIBRIUM_SELF_SIGNAL_OUTCOME_MIN_HOURS=24",
                "EQUILIBRIUM_SELF_SIGNAL_OUTCOME_GRACE_SEC=600",
                "EQUILIBRIUM_SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS=6",
                "CHANNEL_EQUILIBRIUM_SELF_SIGNAL_HOUR=orion:equilibrium:self_signal:hour",
                "CHANNEL_EQUILIBRIUM_SELF_SIGNAL_OUTCOME_ASSOCIATION=orion:equilibrium:self_signal:outcome_association"):
        assert key in env, key
        assert key.split("=")[0] in compose, key


def test_flag_off_builds_no_rollup(monkeypatch) -> None:
    from app import service as svc_mod

    monkeypatch.setattr(svc_mod.settings, "self_signal_outcome_association_enabled", False)
    assert svc_mod.EquilibriumService()._self_signal_rollup is None
    monkeypatch.setattr(svc_mod.settings, "self_signal_outcome_association_enabled", True)
    assert svc_mod.EquilibriumService()._self_signal_rollup is not None


def test_service_poll_publishes_hour_rows_then_associations(monkeypatch) -> None:
    """Rollup alone (episode gate off): one poll publishes the due hour rows on the hour channel
    and every association on the association channel; a failed hour publish is retried."""
    import asyncio

    from app import service as svc_mod

    monkeypatch.setattr(svc_mod.settings, "metacog_self_signal_episodes_enabled", False)
    monkeypatch.setattr(svc_mod.settings, "self_signal_outcome_association_enabled", True)
    svc = svc_mod.EquilibriumService()
    assert svc._self_signal_gate is None and svc._self_signal_rollup is not None
    monkeypatch.setattr(svc_mod, "_utcnow", lambda: NOW)

    class _Cache:
        def refresh(self, fetch, *, now):
            return 0

        def earliest(self):
            return min(r[0][0] for r in HIST.values())

        def current(self):
            return {n: (rows[-1][1], rows[-1][0]) for n, rows in HIST.items()}

        def history(self, node_id):
            return list(HIST.get(node_id, []))

    class _Reader(FakeReader):
        def fetch_since(self, since):
            return []

    svc._pe_history_cache = _Cache()
    svc._pe_history_reader = _Reader()
    sent: list[tuple[str, str]] = []
    fail_first = {"armed": True}

    async def _publish(channel, env):
        if fail_first["armed"] and channel == svc_mod.settings.channel_self_signal_hour:
            fail_first["armed"] = False
            raise ConnectionError("bus down")
        sent.append((channel, env.kind))

    svc.bus = type("B", (), {"publish": staticmethod(_publish)})()
    asyncio.run(svc._self_signal_episode_poll_once())
    kinds = [k for _, k in sent]
    assert kinds.count("self_signal.outcome_association.v1") == 3
    n_hours = kinds.count("self_signal.hour.v1")
    assert n_hours > 100
    # The one hour whose publish failed was forgotten, so the next poll recomputes and sends it.
    sent.clear()
    asyncio.run(svc._self_signal_episode_poll_once())
    assert [k for _, k in sent].count("self_signal.hour.v1") == 1
