from __future__ import annotations


def resolve_graphiti_adapter_url(settings) -> str:
    return (
        getattr(settings, "GRAPHITI_ADAPTER_URL", "") or
        getattr(settings, "GRAPHITI_URL", "") or
        ""
    ).strip()
