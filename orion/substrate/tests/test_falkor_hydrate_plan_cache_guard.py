"""#3455: FalkorDB's cached plan for the exact native-node scan text returned
each row's node_id in the decay_floor column, so every reload failed. The
store re-reads a fully-undecodable page once with a fixed plan-cache-bypass
variant of the query text, and reports the in-memory vs durable node gap."""
from __future__ import annotations

import logging

from orion.substrate.evals.complete_hydration_fixture import CappedClient, build
from orion.substrate.falkor_store import FalkorSubstrateStore

SUFFIX = FalkorSubstrateStore._PLAN_CACHE_BYPASS_SUFFIX


def _is_node_scan(cypher: str) -> bool:
    return "RETURN n.node_id AS node_id" in cypher


class StalePlanClient(CappedClient):
    """The exact node-scan text returns node_id in decay_floor (the live #3455
    shape); the bypass variant returns the stored rows. ``variant_bad`` makes
    the variant shifted too."""

    def __init__(self, *, variant_bad: bool = False, **kw):
        super().__init__(**kw)
        self.variant_bad = variant_bad

    def graph_query(self, cypher, params=None):
        rows = super().graph_query(cypher, params)
        if not _is_node_scan(cypher):
            return rows
        if cypher.endswith(SUFFIX) and not self.variant_bad:
            return rows
        return [dict(row, decay_floor=row["node_id"]) for row in rows]


def _node_scan_texts(client):
    return [q for q, _ in client.calls if _is_node_scan(q)]


def test_shifted_cached_plan_is_retried_with_variant_and_reload_succeeds(caplog):
    client = StalePlanClient()
    with caplog.at_level(logging.WARNING, logger="orion.substrate.falkor_store"):
        store = build(client)
    assert store.last_hydrate_ok is True
    assert store.last_scan_receipt.complete and store.last_hydrate_node_count == 7
    assert store.hydrate_plan_cache_retries_total == 1
    assert store.hydrate_plan_cache_recovered_total == 1
    assert store.last_scan_receipt.plan_cache_retries == 1
    texts = _node_scan_texts(client)
    # One bad exact read, then the variant for the rest of the scan.
    assert not texts[0].endswith(SUFFIX) and all(t.endswith(SUFFIX) for t in texts[1:])
    # Fixed variant set: exactly two distinct texts ever sent for this scan.
    assert len(set(texts)) == 2
    suspect = [r.getMessage() for r in caplog.records if "falkor_hydrate_plan_cache_suspect" in r.getMessage()]
    assert len(suspect) == 1 and "field=decay_floor" in suspect[0] and "node0" in suspect[0]


def test_variant_also_bad_fails_as_before_and_keeps_old_cache():
    client = StalePlanClient()
    store = build(client)
    good_cache = store._cache
    client.variant_bad = True
    store._hydrate_from_durable()
    assert store.last_hydrate_ok is False
    assert store._cache is good_cache
    assert not store.last_scan_receipt.complete
    assert "could not convert" in store.last_scan_receipt.reason
    assert store.hydrate_plan_cache_retries_total == 2
    assert store.hydrate_plan_cache_recovered_total == 1


def test_partly_bad_page_is_not_retried():
    client = CappedClient(cap=1000)
    client._hydrate_node_rows[3]["decay_floor"] = "node3"
    store = build(client)
    assert store.last_hydrate_ok is False
    assert store.hydrate_plan_cache_retries_total == 0
    assert all(not q.endswith(SUFFIX) for q, _ in client.calls)


def test_clean_scan_never_sends_variant():
    client = CappedClient()
    store = build(client)
    assert store.last_hydrate_ok is True and store.hydrate_plan_cache_retries_total == 0
    assert all(not q.endswith(SUFFIX) for q, _ in client.calls)


def test_node_gap_reads_zero_on_full_reload():
    store = build(CappedClient())
    assert store.last_durable_node_count == 7
    assert store.last_node_count_gap == 0
    receipt = store.last_scan_receipt
    assert (receipt.durable_node_count, receipt.node_count_gap) == (7, 0)


def test_node_gap_alarms_when_failed_reload_leaves_partial_cache(caplog):
    client = CappedClient(nodes=0, edges=0)
    store = build(client)
    assert store.last_node_count_gap == 0
    # The durable graph grows to 200 nodes but every reload now fails, so the
    # in-memory graph stays empty: the #3455 starved-reader state.
    client._hydrate_node_rows = [dict(r) for r in CappedClient(nodes=200, edges=0)._hydrate_node_rows]
    for row in client._hydrate_node_rows:
        row["decay_floor"] = row["node_id"]
    with caplog.at_level(logging.WARNING, logger="orion.substrate.falkor_store"):
        store._hydrate_from_durable()
    assert store.last_hydrate_ok is False
    assert store.last_node_count_gap == 200
    assert store.last_scan_receipt.node_count_gap == 200
    assert any("falkor_substrate_hydrate_node_gap" in r.getMessage() and "gap=200" in r.getMessage()
               for r in caplog.records)


def test_node_count_error_leaves_gap_unknown_and_reload_ok():
    class NoCount(CappedClient):
        def graph_query(self, cypher, params=None):
            if "AS node_count" in cypher:
                raise ConnectionError("count down")
            return super().graph_query(cypher, params)

    store = build(NoCount())
    assert store.last_hydrate_ok is True
    assert store.last_node_count_gap is None and store.last_scan_receipt.node_count_gap is None
