"""Patch 3: HTTP calls and shared Redis keys in the atlas (build, sources) and the v2 gap rules."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.body_atlas import capacities as K
from orion.body_atlas import sources as S
from orion.body_atlas.build import build_atlas
from orion.body_atlas.model import FlowV1
from orion.body_atlas.wire_code import WireRef, WireRefs

NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)
HEALTH = "orion:system:health"


def _observed(names: list[str], extra: list[tuple[str, str]] = ()):
    from orion.body_atlas.build import ObservedBus

    t = NOW.timestamp() - 60
    return ObservedBus(publishes=[(n, HEALTH, 5000, t) for n in names] + [(n, ch, 5000, t) for n, ch in extra])


def _catalog(names: list[str]) -> list[dict]:
    return [{"name": HEALTH, "producer_services": names, "consumer_services": ["orion-eq"]},
            {"name": "orion:x:req", "producer_services": ["orion-a"], "consumer_services": ["orion-b"]}]


def _wire() -> WireRefs:
    return WireRefs(
        http=[WireRef("orion-a", "http://orion-athena-notify:7140", "calls", "services/orion-a/app/main.py"),
              WireRef("orion-b", "http://orion-notify:7140", "calls", "services/orion-b/app/main.py"),
              WireRef("orion-a", "http://orion-athena-b:1/x", "calls", "services/orion-a/app/main.py"),
              WireRef("orion-a", "http://127.0.0.1:8124", "calls", "services/orion-a/app/main.py"),
              WireRef("orion-notify", "http://orion-athena-notify:7140", "calls", "services/orion-notify/app/x.py")],
        redis={"orion:regulation:latest": [
                   WireRef("orion-w", "orion:regulation:latest", "writes", "services/orion-w/app/main.py"),
                   *(WireRef(r, "orion:regulation:latest", "reads", f"services/{r}/app/m.py")
                     for r in ("orion-r1", "orion-r2", "orion-r3"))],
               "orion:private:k": [WireRef("orion-a", "orion:private:k", "writes", "services/orion-a/app/main.py")]},
        dynamic={"orion-a": ["services/orion-a/app/main.py"]})


NAMES = ["orion-a", "orion-b", "orion-notify", "orion-w", "orion-r1", "orion-r2", "orion-r3", "orion-eq", "orion-quiet"]


def _built(wire: WireRefs | None = None):
    body = S.BodyInputs(hosts=[S.HostIn(name="athena")])  # orion-athena-notify is notify on host athena
    if wire is not None:
        body.stores.extend(S.apply_wire(body, wire))
    return build_atlas(_catalog(NAMES), _observed(NAMES, [("orion-a", "orion:x:req")]), now=NOW, body=body)


def test_apply_wire_keeps_only_shared_keys_as_parts():
    body = S.BodyInputs()
    keys = S.apply_wire(body, _wire())
    assert [k.name for k in keys] == ["orion:regulation:latest"]
    assert body.redis_private == {"orion-a": 1} and body.redis_dynamic == {"orion-a": 1}
    assert {c.host for c in body.calls} == {"orion-athena-notify", "orion-notify", "orion-athena-b", "127.0.0.1"}


def test_calls_flows_one_unit_per_callee_and_unresolved_hosts_are_listed():
    atlas = _built(_wire())
    calls = {(f.source, f.target): f for f in atlas.flows if f.relation == "calls"}
    to_notify = [f for (s, t), f in calls.items() if t == "part:orion-notify"]
    assert {f.source for f in to_notify} == {"part:orion-a", "part:orion-b"}  # the self-call is dropped
    assert sum(f.weight for f in to_notify) == pytest.approx(1.0, abs=1e-3)
    assert any("127.0.0.1 (named by orion-a)" in u for u in atlas.unmatched["declared_url"])


def test_a_pair_that_already_signals_keeps_its_relation_and_gains_the_call_evidence():
    atlas = _built(_wire())
    ab = next(f for f in atlas.flows if f.source == "part:orion-a" and f.target == "part:orion-b")
    assert ab.relation == "signals" and "declared_url" in ab.evidence and "orion:x:req" in ab.channels


def test_shared_key_is_a_store_part_with_writer_and_readers():
    atlas = _built(_wire())
    key = next(p for p in atlas.parts if p.part_id == "part:redis:orion:regulation:latest")
    assert key.kind == "store" and key.status == "declared_only" and key.facts["value"] == "not read"
    rel = {f.source: f.relation for f in atlas.flows if f.target == key.part_id}
    assert rel == {"part:orion-w": "writes", "part:orion-r1": "reads", "part:orion-r2": "reads", "part:orion-r3": "reads"}
    assert not any(p.name == "redis:orion:private:k" for p in atlas.parts)
    assert atlas.redis_private == {"orion-a": 1}


def test_v2_gaps_read_the_wires():
    plain = K.read_capacities(_built())
    assert plain.wires is None  # no wire source: the v1 texts stand
    assert any(g.gap_id == "gap:algedonic-unseen" for g in plain.gaps)
    assert "does not read HTTP" in next(g.text for g in plain.gaps if g.gap_id == "gap:s3-unseen")

    r = K.read_capacities(_built(_wire()))
    assert r.wires is not None and r.wires.keys == 1 and r.wires.private_keys == 1
    # orion-notify only heartbeats on the bus but is called over HTTP: seen. orion-quiet is not.
    assert "part:orion-notify" in r.wires.seen_by_wire
    assert "part:orion-quiet" in r.silent_services and "part:orion-notify" not in r.silent_services
    gaps = {g.gap_id: g.text for g in r.gaps}
    assert "no bus, HTTP or shared-Redis traffic" in gaps["gap:s3-unseen"]
    assert "gap:algedonic-unseen" not in gaps and "gap:algedonic-none" in gaps
    assert "no part holds S4 and S3" in gaps["gap:s4-to-s3"]


def test_broadcast_key_rule_needs_readers_in_two_organs():
    atlas = _built(_wire())
    idx = K._Index(atlas)
    # put the three readers in one organ, then in two
    one = {"part:orion-r1": "organ:x", "part:orion-r2": "organ:x", "part:orion-r3": "organ:x"}
    idx.organ.update(one)
    assert K.rule_broadcast_keys(atlas, idx) == []
    assert [b.part_id for b in K.rule_broadcast_keys(atlas, idx, organs_clause=False)] == \
        ["part:redis:orion:regulation:latest"]
    idx.organ["part:orion-r3"] = "organ:y"
    b = K.rule_broadcast_keys(atlas, idx)
    assert [x.part_id for x in b] == ["part:redis:orion:regulation:latest"] and b[0].reader_organs == 2


def test_s4_s3_gap_clears_only_with_confirmed_parts_linked_both_ways():
    atlas = _built(_wire())
    # live key parts (claimable needs live)
    atlas.parts = [p.model_copy(update={"status": "live"}) if p.name.startswith("redis:") else p for p in atlas.parts]
    yes = K.VsmConfirmedV1(confirmed=[K.ConfirmationV1(part="orion-w", role="S4", date="2026-10-31"),
                                      K.ConfirmationV1(part="orion-r1", role="S3", date="2026-10-31")])
    one_way = K.read_capacities(atlas, vsm=yes)
    gap = next(g for g in one_way.gaps if g.gap_id == "gap:s4-to-s3")
    assert "orion-w -> orion-r1: redis:orion:regulation:latest" in gap.text
    # an HTTP call back from S3 to S4 closes the loop
    atlas.flows.append(FlowV1(source="part:orion-r1", target="part:orion-w", weight=0.1, evidence=["declared_url"],
                              relation="calls"))
    both = K.read_capacities(atlas, vsm=yes)
    assert not any(g.gap_id == "gap:s4-to-s3" for g in both.gaps)
    assert both.wires.s4_s3 == ["orion-r1 -> orion-w: http", "orion-w -> orion-r1: redis:orion:regulation:latest"]


def test_a_confirmed_algedonic_part_covers_the_algedonic_gap_and_a_claim_does_not():
    atlas = _built(_wire())
    atlas.parts = [p.model_copy(update={"status": "live"}) if p.name.startswith("redis:") else p for p in atlas.parts]
    yes = K.VsmConfirmedV1(confirmed=[K.ConfirmationV1(part="redis:orion:regulation:latest", role="algedonic",
                                                       date="2026-10-31")])
    assert not any(g.gap_id == "gap:algedonic-none" for g in K.read_capacities(atlas, vsm=yes).gaps)
    claim = K.CapacityClaimV1(part_id="part:redis:orion:regulation:latest", role="algedonic", kind="claim",
                              because="b", cites=[], at="t", dig="d")
    r = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [claim], [])
    assert any(g.gap_id == "gap:algedonic-none" for g in r.gaps)


def test_printout_with_wires_still_fits():
    text = "\n".join(K.render(K.read_capacities(_built(_wire()))))
    assert "past the bus: " in text and "rules v2" in text
    assert len(text.splitlines()) <= 39 and len(text) <= 3900


# --- live Redis reads ---------------------------------------------------------------------------

def test_redis_reader_refuses_anything_but_its_read_commands():
    sent = []
    cmd = S._read_only_redis(lambda *a: sent.append(a) or "string")
    assert cmd("TYPE", "k") == "string"
    for bad in ("SET", "DEL", "EXPIRE", "FLUSHDB", "GET"):
        with pytest.raises(RuntimeError):
            cmd(bad, "k")
    assert sent == [("TYPE", "k")]


def test_read_redis_keys_present_absent_and_family():
    replies = {("TYPE", "a"): "string", ("OBJECT", "IDLETIME", "a"): 8, ("TTL", "a"): 301, ("TYPE", "b"): "none"}
    scans = iter([("5", ["p:1", "p:2"]), ("0", ["p:3"])])

    def cmd(*args):
        if args[0] == "SCAN":
            return next(scans)
        return replies[args]

    got = S.read_redis_keys(cmd, ["a", "b", "p:*"])
    assert got["a"] == (1, {"value": "present", "type": "string", "idle_s": "8", "ttl_s": "301"})
    assert got["b"] == (0, {"value": "absent"})
    assert got["p:*"] == (3, {"keys_found": "3"})


def test_gather_applies_redis_facts_all_or_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "code_wire_refs", lambda root: _wire())
    (tmp_path / "orion").mkdir()

    def broken(*args):
        if args[0] == "TYPE":
            raise ConnectionError("bus down")
        return 0

    body = S.gather_body(tmp_path, [], live=S.LiveReaders(redis=broken))
    key = next(s for s in body.stores if s.store == "redis")
    assert key.rows is None and key.facts == {}
    assert any(r.name.startswith("live:redis") and r.status == "unread" for r in body.reads)

    ok = S.gather_body(tmp_path, [], live=S.LiveReaders(redis=lambda *a: {"TYPE": "string", "OBJECT": 3, "TTL": -1}[a[0]]))
    key = next(s for s in ok.stores if s.store == "redis")
    assert key.rows == 1 and key.facts["type"] == "string"
