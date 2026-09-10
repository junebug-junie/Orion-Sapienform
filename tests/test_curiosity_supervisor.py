"""Patch 1 of docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md.

Read-only, on history: no live FalkorDB, no live cortex call. Fakes match the
shape `tests/test_curiosity_worldview.py` already established for
`WorldviewReader` (match on query substring, return canned rows) plus a small
fake bus/codec for `generate_readings_for_run`.
"""

from __future__ import annotations

import pytest

from orion.curiosity.supervisor import (
    build_reading_options,
    build_reading_prompt,
    build_run_order,
    generate_all_readings,
    generate_readings_for_run,
    group_hops_by_run,
    group_readings_by_prior,
    is_circling,
    parse_reading_batch,
)
from orion.curiosity.worldview import (
    ALL_HOPS_CYPHER,
    ALL_PRIORS_CYPHER,
    RECENT_RUNS_LIMIT,
    HopRecord,
    Prior,
    WorldviewReader,
    WorldviewUnavailable,
    read_all_hops,
    read_all_priors,
)
from orion.schemas.curiosity_supervisor import HopReadingBatchV1, HopReadingV1


class _FakeReader(WorldviewReader):
    """Answers by matching on the query text -- same technique
    test_curiosity_worldview.py uses for WorldviewReader."""

    def __init__(self, *, answers=None, raises=False) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises

    def query(self, cypher: str):
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        for needle, rows in self.answers.items():
            if needle in cypher:
                return rows
        return []


class _FakeDecoded:
    def __init__(self, *, ok=True, payload=None, error=None):
        self.ok = ok
        self.error = error
        self.envelope = type("Envelope", (), {"payload": payload})()


class _FakeCodec:
    def __init__(self, decoded):
        self._decoded = decoded

    def decode(self, data):
        return self._decoded


class _FakeBus:
    """Records the request it was given and returns a canned decode."""

    def __init__(self, *, decoded):
        self.codec = _FakeCodec(decoded)
        self.requests: list[dict] = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.requests.append(
            {"channel": channel, "envelope": envelope, "reply_channel": reply_channel}
        )
        return {"data": "irrelevant -- codec is faked"}


def _prior(pid="p1", claim="a claim", confidence=0.55, status="open", tested=1):
    return Prior(
        prior_id=pid, claim=claim, confidence=confidence, status=status, times_tested=tested
    )


def _cortex_result(readings: list[dict]) -> dict:
    """A CortexClientResult-shaped payload the way cortex-orch really returns
    one for a brain-mode structured-output call -- confirmed live during
    Patch 1's own verification: the model's JSON lands in `final_text` as a
    STRING, not as the top-level dict itself."""
    import json as _json

    return {"ok": True, "status": "success", "final_text": _json.dumps({"readings": readings})}


def _reading(*, run_id="r1", n=1, prior_id="p1", moved=None, conf=0.7):
    return HopReadingV1(
        hop_run_id=run_id,
        hop_n=n,
        about_prior_id=prior_id,
        kind="test",
        moved_the_claim=moved,
        reading_confidence=conf,
        reasoning="because",
    )


# --- HopReadingV1 -------------------------------------------------------


def test_hop_reading_v1_rejects_unknown_field():
    with pytest.raises(Exception):
        HopReadingV1(
            hop_run_id="r1", hop_n=1, kind="test", reading_confidence=0.5,
            reasoning="x", unexpected="nope",
        )


def test_hop_reading_v1_allows_null_about_prior_and_moved():
    r = HopReadingV1(
        hop_run_id="r1", hop_n=1, kind="bookkeeping",
        moved_the_claim=None, reading_confidence=0.2, reasoning="unrelated",
    )
    assert r.about_prior_id is None
    assert r.moved_the_claim is None


def test_hop_reading_batch_v1_json_schema_has_readings_array():
    schema = HopReadingBatchV1.model_json_schema()
    assert "readings" in schema.get("properties", {})


# --- worldview reads (all-hops / all-priors) -----------------------------


def test_read_all_hops_drops_rows_missing_run_id_or_note():
    reader = _FakeReader(answers={
        ALL_HOPS_CYPHER.split(" RETURN")[0]: [
            {"run_id": "abc123", "n": "1", "note": "found something"},
            {"run_id": "", "n": "2", "note": "orphan, no run_id"},
            {"run_id": "abc123", "n": "3", "note": ""},
        ]
    })
    hops = read_all_hops(reader)
    assert len(hops) == 1
    assert hops[0] == HopRecord(run_id="abc123", n=1, note="found something")


def test_read_all_hops_unavailable_returns_empty():
    assert read_all_hops(_FakeReader(raises=True)) == []


def test_read_all_priors_includes_closed_and_collapses_duplicates():
    reader = _FakeReader(answers={
        ALL_PRIORS_CYPHER.split(" RETURN")[0]: [
            {"prior_id": "p1", "claim": "c1", "status": "open", "times_tested": "1"},
            {"prior_id": "p2", "claim": "c2", "status": "refuted", "times_tested": "5"},
            # a fork: same id, lower times_tested -- collapse keeps the higher
            {"prior_id": "p1", "claim": "c1", "status": "open", "times_tested": "0"},
        ]
    })
    priors = read_all_priors(reader)
    by_id = {p.prior_id: p for p in priors}
    assert set(by_id) == {"p1", "p2"}
    assert by_id["p1"].times_tested == 1
    assert by_id["p2"].status == "refuted"


# --- prompt building -------------------------------------------------------


def test_build_reading_prompt_includes_prior_and_hop_content():
    priors = [_prior(pid="pXYZ", claim="the served model is anchorable")]
    hops = [HopRecord(run_id="r1", n=1, note="re-confirmed 338/411 named turns")]
    prompt = build_reading_prompt(priors, hops)
    assert "pXYZ" in prompt
    assert "the served model is anchorable" in prompt
    assert "re-confirmed 338/411 named turns" in prompt


def test_build_reading_options_carries_the_schema():
    options = build_reading_options()
    assert options["structured_output_schema_name"] == "HopReadingBatchV1"
    assert "readings" in options["structured_output_schema"]["properties"]


# --- response parsing --------------------------------------------------


def test_parse_reading_batch_happy_path():
    payload = {
        "readings": [
            {
                "hop_n": 1, "about_prior_id": "p1", "kind": "test",
                "moved_the_claim": False, "reading_confidence": 0.8,
                "reasoning": "tested, no movement",
            },
        ]
    }
    out = parse_reading_batch(payload, run_id="r1", hop_ns=[1])
    assert len(out) == 1
    assert out[0].hop_run_id == "r1"  # stamped, not present in the raw row
    assert out[0].hop_n == 1


def test_parse_reading_batch_overrides_model_hallucinated_hop_run_id():
    """Observed live: nothing in the prompt asks the model for hop_run_id
    (the caller already knows it), but the JSON schema requires the field,
    and the model filled it with a hallucinated placeholder ("n=1") instead
    of leaving it out. The caller's run_id must always win."""
    payload = {"readings": [
        {"hop_run_id": "n=1", "hop_n": 1, "kind": "test",
         "reading_confidence": 0.5, "reasoning": "x"},
    ]}
    out = parse_reading_batch(payload, run_id="the_real_run_id", hop_ns=[1])
    assert len(out) == 1
    assert out[0].hop_run_id == "the_real_run_id"


def test_parse_reading_batch_drops_hop_n_not_asked_about():
    payload = {"readings": [
        {"hop_n": 99, "kind": "test", "reading_confidence": 0.5, "reasoning": "x"}
    ]}
    assert parse_reading_batch(payload, run_id="r1", hop_ns=[1, 2]) == []


def test_parse_reading_batch_drops_invalid_rows_keeps_valid_ones():
    payload = {"readings": [
        {"hop_n": 1, "kind": "test", "reading_confidence": "not a float", "reasoning": "x"},
        {"hop_n": 2, "kind": "test", "reading_confidence": 0.5, "reasoning": "y"},
    ]}
    out = parse_reading_batch(payload, run_id="r1", hop_ns=[1, 2])
    assert len(out) == 1
    assert out[0].hop_n == 2


def test_parse_reading_batch_sorts_by_hop_n_regardless_of_model_order():
    """Found in review: nothing guarantees an LLM echoes a multi-hop batch
    back in the order it read them, and is_circling reads the LAST entries
    of a prior's reading list as "most recent" -- a reordered response would
    silently flip that verdict. Fed out of order on purpose."""
    payload = {"readings": [
        {"hop_n": 3, "kind": "test", "reading_confidence": 0.5, "reasoning": "c"},
        {"hop_n": 1, "kind": "test", "reading_confidence": 0.5, "reasoning": "a"},
        {"hop_n": 2, "kind": "test", "reading_confidence": 0.5, "reasoning": "b"},
    ]}
    out = parse_reading_batch(payload, run_id="r1", hop_ns=[1, 2, 3])
    assert [r.hop_n for r in out] == [1, 2, 3]


def test_parse_reading_batch_wrong_shape_returns_empty():
    assert parse_reading_batch("not a dict", run_id="r1", hop_ns=[1]) == []
    assert parse_reading_batch({"no_readings_key": []}, run_id="r1", hop_ns=[1]) == []


# --- generate_readings_for_run (fake bus) -------------------------------


@pytest.mark.asyncio
async def test_generate_readings_for_run_uses_fake_bus_and_parses_result():
    decoded = _FakeDecoded(ok=True, payload=_cortex_result([
        {"hop_n": 1, "about_prior_id": "p1", "kind": "test",
         "moved_the_claim": True, "reading_confidence": 0.9, "reasoning": "settled it"},
    ]))
    bus = _FakeBus(decoded=decoded)
    from orion.core.bus.bus_schemas import ServiceRef

    hops = [HopRecord(run_id="r1", n=1, note="tested the claim, confirmed")]
    priors = [_prior(pid="p1")]
    readings = await generate_readings_for_run(
        bus, run_id="r1", hops=hops, priors=priors,
        cortex_request_channel="orion:cortex:request",
        cortex_result_prefix="orion:cortex:result",
        source=ServiceRef(name="test", node="local", version="0.0.1"),
    )
    assert len(readings) == 1
    assert readings[0].moved_the_claim is True
    assert len(bus.requests) == 1
    assert bus.requests[0]["channel"] == "orion:cortex:request"


@pytest.mark.asyncio
async def test_generate_readings_for_run_raises_on_decode_failure():
    decoded = _FakeDecoded(ok=False, error="boom")
    bus = _FakeBus(decoded=decoded)
    from orion.core.bus.bus_schemas import ServiceRef

    with pytest.raises(RuntimeError):
        await generate_readings_for_run(
            bus, run_id="r1", hops=[HopRecord(run_id="r1", n=1, note="x")],
            priors=[], cortex_request_channel="c", cortex_result_prefix="p",
            source=ServiceRef(name="test", node="local", version="0.0.1"),
            max_attempts=1,  # skip the retry/backoff loop -- one shot is enough here
        )


@pytest.mark.asyncio
async def test_generate_readings_for_run_retries_a_not_ok_result():
    """Observed live during Patch 1's own verification: the exact same
    request can fail cortex-orch's verb-activation gate and succeed on a
    bare retry seconds later. This is the shape of that failure, faked."""
    from orion.core.bus.bus_schemas import ServiceRef

    calls = {"n": 0}
    not_ok = _FakeDecoded(ok=True, payload={
        "ok": False, "error": {"message": "inactive_verb:curiosity_hop_reading"},
    })
    good = _FakeDecoded(ok=True, payload=_cortex_result([
        {"hop_n": 1, "kind": "test", "reading_confidence": 0.5, "reasoning": "x"},
    ]))

    class _FlakyBus(_FakeBus):
        async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
            calls["n"] += 1
            self.codec = _FakeCodec(not_ok if calls["n"] == 1 else good)
            return await super().rpc_request(
                channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec
            )

    bus = _FlakyBus(decoded=not_ok)
    readings = await generate_readings_for_run(
        bus, run_id="r1", hops=[HopRecord(run_id="r1", n=1, note="x")],
        priors=[], cortex_request_channel="c", cortex_result_prefix="p",
        source=ServiceRef(name="test", node="local", version="0.0.1"),
        max_attempts=3, retry_delay_sec=0.0,
    )
    assert calls["n"] == 2
    assert len(readings) == 1


@pytest.mark.asyncio
async def test_generate_readings_for_run_empty_hops_short_circuits():
    bus = _FakeBus(decoded=_FakeDecoded(ok=True, payload={"readings": []}))
    from orion.core.bus.bus_schemas import ServiceRef

    out = await generate_readings_for_run(
        bus, run_id="r1", hops=[], priors=[],
        cortex_request_channel="c", cortex_result_prefix="p",
        source=ServiceRef(name="test", node="local", version="0.0.1"),
    )
    assert out == []
    assert bus.requests == []  # no RPC call made for a run with no hops


@pytest.mark.asyncio
async def test_generate_readings_for_run_rejects_max_attempts_below_one():
    from orion.core.bus.bus_schemas import ServiceRef

    bus = _FakeBus(decoded=_FakeDecoded(ok=True, payload={"readings": []}))
    with pytest.raises(ValueError):
        await generate_readings_for_run(
            bus, run_id="r1", hops=[HopRecord(run_id="r1", n=1, note="x")],
            priors=[], cortex_request_channel="c", cortex_result_prefix="p",
            source=ServiceRef(name="test", node="local", version="0.0.1"),
            max_attempts=0,
        )
    assert bus.requests == []  # rejected before any RPC call


# --- generate_all_readings: one bad run must not blank the sweep --------


@pytest.mark.asyncio
async def test_generate_all_readings_survives_one_failing_run(monkeypatch):
    from orion.core.bus.bus_schemas import ServiceRef

    reader = _FakeReader(answers={
        ALL_HOPS_CYPHER.split(" RETURN")[0]: [
            {"run_id": "run_ok", "n": "1", "note": "note A"},
            {"run_id": "run_bad", "n": "1", "note": "note B"},
        ],
        ALL_PRIORS_CYPHER.split(" RETURN")[0]: [
            {"prior_id": "p1", "claim": "c1", "status": "open", "times_tested": "1"},
        ],
    })

    calls = []

    async def _fake_generate(bus, *, run_id, hops, priors, **kwargs):
        calls.append(run_id)
        if run_id == "run_bad":
            raise RuntimeError("simulated RPC timeout")
        return [
            HopReadingV1(
                hop_run_id=run_id, hop_n=hops[0].n, about_prior_id="p1",
                kind="test", moved_the_claim=True, reading_confidence=0.9,
                reasoning="ok",
            )
        ]

    monkeypatch.setattr(
        "orion.curiosity.supervisor.generate_readings_for_run", _fake_generate
    )

    out = await generate_all_readings(
        object(), reader,
        cortex_request_channel="c", cortex_result_prefix="p",
        source=ServiceRef(name="test", node="local", version="0.0.1"),
    )
    assert sorted(calls) == ["run_bad", "run_ok"]
    assert len(out) == 1
    assert out[0].hop_run_id == "run_ok"


# --- run ordering ---------------------------------------------------------


def test_group_hops_by_run_orders_unknown_written_at_last():
    hops = [
        HopRecord(run_id="newer", n=1, note="a"),
        HopRecord(run_id="older", n=1, note="b"),
        HopRecord(run_id="unknown_time", n=1, note="c"),
    ]
    run_order = {"older": 100, "newer": 200}
    grouped = group_hops_by_run(hops, run_order=run_order)
    assert [run_id for run_id, _ in grouped] == ["older", "newer", "unknown_time"]


def test_build_run_order_unavailable_returns_empty():
    assert build_run_order(_FakeReader(raises=True)) == {}


def test_build_run_order_warns_when_the_recent_runs_bound_is_hit(caplog):
    """Found in review: `RECENT_RUNS_CYPHER` caps at RECENT_RUNS_LIMIT rows,
    sized for 'show Orion its recent thread' -- repurposed here for 'order
    all of history', with no warning if that bound is ever actually hit.
    A run past it doesn't error, it silently sorts as unknown-last in
    group_hops_by_run; this is what makes that visible."""
    rows = [
        {"run_id": f"r{i}", "written_at": str(i), "continue_note": "", "claim": None}
        for i in range(RECENT_RUNS_LIMIT)
    ]
    reader = _FakeReader(answers={"MATCH (t:TurnOutcome)": rows})
    with caplog.at_level("WARNING"):
        order = build_run_order(reader)
    assert len(order) == RECENT_RUNS_LIMIT
    assert any("curiosity_supervisor_run_order_truncated" in m for m in caplog.messages)


# --- group_readings_by_prior / is_circling --------------------------------


def test_group_readings_by_prior_drops_unattributed():
    readings = [
        _reading(prior_id="p1", n=1),
        _reading(prior_id=None, n=2),
        _reading(prior_id="p2", n=3),
    ]
    grouped = group_readings_by_prior(readings)
    assert set(grouped) == {"p1", "p2"}


def test_is_circling_none_below_min_hops():
    assert is_circling([_reading(moved=False)]) is None
    assert is_circling([_reading(moved=False), _reading(moved=False)]) is None


def test_is_circling_true_when_last_three_never_moved():
    readings = [
        _reading(n=1, moved=True),   # earlier movement doesn't save it
        _reading(n=2, moved=False),
        _reading(n=3, moved=False),
        _reading(n=4, moved=False),
    ]
    assert is_circling(readings) is True


def test_is_circling_false_when_recent_hop_moved_the_claim():
    readings = [
        _reading(n=1, moved=False),
        _reading(n=2, moved=False),
        _reading(n=3, moved=True),
    ]
    assert is_circling(readings) is False


def test_is_circling_false_when_recent_hop_is_ambiguous():
    """A None (could not tell) in the recent window breaks the circling
    claim -- ambiguity is not evidence of being stuck."""
    readings = [
        _reading(n=1, moved=False),
        _reading(n=2, moved=False),
        _reading(n=3, moved=None),
    ]
    assert is_circling(readings) is False
