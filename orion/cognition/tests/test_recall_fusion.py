from pathlib import Path
import sys
import types
import logging

ROOT = Path(__file__).resolve().parents[3]
APP_PATH = ROOT / "services" / "orion-recall"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(APP_PATH) not in sys.path:
    sys.path.append(str(APP_PATH))
if str(APP_PATH / "app") not in sys.path:
    sys.path.append(str(APP_PATH / "app"))

sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logging.getLogger("test")))

from fusion import fuse_candidates  # type: ignore  # noqa: E402


def test_fuse_dedupe_and_limit():
    candidates = [
        {"id": "1", "source": "vector", "text": "alpha", "score": 0.9},
        {"id": "1", "source": "vector", "text": "alpha duplicate", "score": 0.5},
        {"id": "2", "source": "vector", "text": "beta", "score": 0.8},
        {"id": "3", "source": "rdf", "text": "gamma", "score": 0.7},
        {"id": "4", "source": "rdf", "text": "delta", "score": 0.6},
    ]

    profile = {
        "profile": "test",
        "max_per_source": 1,
        "max_total_items": 2,
        "render_budget_tokens": 50,
    }

    bundle, _ranking = fuse_candidates(
        candidates=candidates,
        profile=profile,
        latency_ms=5,
        query_text="alpha",
        session_id=None,
    )

    assert len(bundle.items) == 2  # dedup + per-source cap
    assert all(item.id in {"1", "3"} for item in bundle.items)
    assert "- [vector]" in bundle.rendered


def test_self_factual_filters_exclude_induced_and_reflective_candidates():
    candidates = [
        {"id": "auth-rdf", "source": "rdf", "text": "authoritative self fact", "score": 0.8, "tags": ["trust:authoritative"]},
        {"id": "reflective-rdf", "source": "rdf", "text": "reflective self story", "score": 0.99, "tags": ["trust:reflective"]},
        {"id": "induced-sql", "source": "sql_timeline", "text": "induced self pattern", "score": 0.95, "tags": ["trust:induced"]},
        {"id": "vector-auth", "source": "vector", "text": "vector fact", "score": 0.99, "tags": ["trust:authoritative"]},
    ]

    profile = {
        "profile": "self.factual.v1",
        "max_per_source": 4,
        "max_total_items": 4,
        "render_budget_tokens": 50,
        "filters": {
            "allowed_sources": ["rdf", "sql_timeline"],
            "exclude_tags_prefixes": ["trust:induced", "trust:reflective"],
        },
    }

    bundle, _ranking = fuse_candidates(
        candidates=candidates,
        profile=profile,
        latency_ms=5,
        query_text="self fact",
        session_id=None,
    )

    assert [item.id for item in bundle.items] == ["auth-rdf"]
    assert "reflective self story" not in bundle.rendered
