"""Harness usage brief for orion-introspect, appended only when the server is attached."""
from __future__ import annotations

from orion.schemas.introspect import IntrospectToolBindingV1


def introspect_brief_lines(binding: IntrospectToolBindingV1) -> list[str]:
    return [
        (
            "Introspect MCP (orion-introspect) is available: reading_results searches what you "
            "actually learned from sources read through your reading pipeline. Use query=<topic "
            "in plain words> to recall readings by meaning (items carry similarity; weak matches "
            "are dropped), url or request_id for one known reading, or nothing for your most "
            "recent finished reads. Call it before describing what you learned from reading "
            "instead of reconstructing it; reading_status says where a request is in the queue, "
            "reading_results says what came out of it. Results are source-attributed candidates, "
            "not settled beliefs. items=[] means nothing matched; a tool error means the answer "
            "is unknown -- say so, never report an error as nothing having happened. "
            "learned=false means no output exists yet: report its reading_status."
        ),
    ]


def append_introspect_harness_brief(
    parts: list[str], *, binding: IntrospectToolBindingV1 | None,
) -> None:
    if binding is None:
        return
    parts.extend(introspect_brief_lines(binding))
