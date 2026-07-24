"""Regression test: orion.graph package must be importable (including
falkor_client, the Falkor-only consumers' real entry point) even when
`requests` isn't installed. orion.graph.sparql_client is Fuseki/SPARQL-
specific and needs `requests`; orion/graph/__init__.py previously imported
it unconditionally, so ANY consumer of orion.graph -- including Falkor-only
ones with no SPARQL involvement at all (orion-bus-mirror, orion-graph-
compression, orion-meta-tags, orion-recall) -- crashed if `requests` wasn't
in their requirements.txt. orion-bus-mirror hit this live, 2026-07-24
(ModuleNotFoundError: No module named 'requests').
"""

from __future__ import annotations

import builtins
import sys


def test_falkor_client_importable_without_requests() -> None:
    real_import = builtins.__import__
    graph_related = [m for m in sys.modules if m == "requests" or m.startswith("orion.graph")]
    saved = {m: sys.modules[m] for m in graph_related}

    def fake_import(name, *args, **kwargs):
        if name == "requests":
            raise ModuleNotFoundError("No module named 'requests'")
        return real_import(name, *args, **kwargs)

    try:
        for m in graph_related:
            del sys.modules[m]
        builtins.__import__ = fake_import

        from orion.graph.falkor_client import RedisGraphQueryClient  # noqa: F401

        import orion.graph as g

        assert not hasattr(g, "GraphStoreClient")
        assert not hasattr(g, "SparqlHttpClient")
        # Non-SPARQL exports must still be present.
        assert hasattr(g, "GraphPersistenceRouter")
        assert hasattr(g, "resolve_graph_backend")
    finally:
        builtins.__import__ = real_import
        # Drop whatever got cached mid-test (the degraded orion.graph included)
        # and restore exactly what was present before this test ran, so later
        # tests in this session never see the degraded module.
        for m in list(sys.modules):
            if m == "requests" or m.startswith("orion.graph"):
                del sys.modules[m]
        sys.modules.update(saved)
        # `import orion.graph as g` resolves via attribute access on the parent
        # package object (`orion`), NOT a fresh sys.modules lookup -- restoring
        # the sys.modules entry above isn't enough by itself, since the parent
        # package's own `.graph` attribute still points at the degraded
        # submodule object created during this test. Confirmed empirically:
        # without this line, a later `import orion.graph as g` anywhere else in
        # the same process returns the stale degraded object even though
        # sys.modules['orion.graph'] itself is already correct.
        if "orion.graph" in saved and "orion" in sys.modules:
            sys.modules["orion"].graph = saved["orion.graph"]


def test_sparql_symbols_still_exported_when_requests_present() -> None:
    """Backward-compat: real SPARQL consumers (requests genuinely installed)
    see identical behavior to before this fix."""
    import orion.graph as g

    assert hasattr(g, "GraphStoreClient")
    assert hasattr(g, "SparqlHttpClient")
    assert hasattr(g, "SparqlQueryClient")
    assert hasattr(g, "SparqlUpdateClient")
    assert hasattr(g, "redact_http_url_for_log")
    assert hasattr(g, "resolve_substrate_sparql_http_basic_auth")
