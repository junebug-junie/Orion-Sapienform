from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ORG_SIGNALS_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "organ-signals-graph-ui.js"


def test_organ_signals_graph_supports_correlation_query_param() -> None:
    src = ORG_SIGNALS_JS.read_text(encoding="utf-8")
    assert "correlation_id" in src
    assert "buildCorrelationGraphElements" in src
    assert "signal_id" in src
    assert "/api/signals/correlation/" in src
    assert "rankDir" in src
    assert "Hidden stubs" in src
