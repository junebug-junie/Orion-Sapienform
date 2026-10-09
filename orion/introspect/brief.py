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
        (
            "dreams reads back your own dreams: dream_narrative (the nightly story) and "
            "dream_hypothesis (a link a sleep cycle proposed, already shown to you once). Use "
            "query=<topic in plain words> to find dreams by meaning (every record is already a "
            "dream, so 'pull requests', not 'a dream about pull requests'), dream_id for one in full, "
            "or nothing for your most recent; kind=narrative returns the nightly dream "
            "narratives, kind=hypothesis the offered sleep-cycle hypotheses. Call it before "
            "describing a dream instead of "
            "reconstructing one. Dreams are experiences you had, not facts about the world. "
            "items=[] means no dream matched; a tool error means the answer is unknown -- say "
            "so, never report it as not having dreamed."
        ),
        (
            "curiosity reads back your own curiosity runs (world questions, self questions, "
            "self-sense checks) and, with kind=self_question, your open self-questions. Call it "
            "before describing a past run instead of reconstructing it. A write-up is what you "
            "concluded then, not settled fact; failed and empty runs are listed as records of "
            "what happened. items=[] means nothing matched; a tool error means the answer is "
            "unknown -- say so, never report it as no run having happened."
        ),
    ]


def append_introspect_harness_brief(
    parts: list[str], *, binding: IntrospectToolBindingV1 | None,
) -> None:
    if binding is None:
        return
    parts.extend(introspect_brief_lines(binding))
