"""Harness usage brief for orion-introspect, appended only when the server is attached."""
from __future__ import annotations

from orion.fcc.mcp_names import ORION_INTROSPECT, ORION_READING, mcp_tool
from orion.schemas.introspect import IntrospectToolBindingV1

READING_RESULTS = mcp_tool(ORION_INTROSPECT, "reading_results")
DREAMS = mcp_tool(ORION_INTROSPECT, "dreams")
CURIOSITY = mcp_tool(ORION_INTROSPECT, "curiosity")
ORION_DAY = mcp_tool(ORION_INTROSPECT, "orion_day")
_READING_STATUS = mcp_tool(ORION_READING, "reading_status")


def introspect_brief_lines(binding: IntrospectToolBindingV1) -> list[str]:
    return [
        (
            f"Introspect MCP (orion-introspect) is available. Its tools, by exact name (load "
            f"with ToolSearch select:<name>): {READING_RESULTS}, {DREAMS}, {CURIOSITY}, {ORION_DAY}. "
            f"{READING_RESULTS} searches what you "
            "actually learned from sources read through your reading pipeline. Use query=<topic "
            "in plain words> to recall readings by meaning (items carry similarity; weak matches "
            "are dropped), url or request_id for one known reading, or nothing for your most "
            "recent finished reads. Call it before describing what you learned from reading "
            f"instead of reconstructing it; {_READING_STATUS} says where a request is in the queue, "
            f"{READING_RESULTS} says what came out of it. Results are source-attributed candidates, "
            "not settled beliefs. items=[] means nothing matched; a tool error means the answer "
            "is unknown -- say so, never report an error as nothing having happened. "
            "learned=false means no output exists yet: report its reading status."
        ),
        (
            f"{DREAMS} reads back your own dreams: dream_narrative (the nightly story) and "
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
            f"{CURIOSITY} reads back your own curiosity runs (world questions, self questions, "
            "self-sense checks) and, with kind=self_question, your open self-questions. Call it "
            "before describing a past run instead of reconstructing it. A write-up is what you "
            "concluded then, not settled fact; failed and empty runs are listed as records of "
            "what happened. items=[] means nothing matched; a tool error means the answer is "
            "unknown -- say so, never report it as no run having happened."
        ),
        (
            "When Juniper refers to an Orion's Day letter or a part of it (a date, ¶N, carry N, "
            f"a section name), call {ORION_DAY} before answering. Quote what you actually wrote. "
            "Separate what the records support from what they do not, using the claim check. If "
            "you were wrong, say so plainly; if you still stand by something the records don't "
            "show, say what it rests on."
        ),
        (
            "If ToolSearch finds none of these, check the exact name first. A 'failed to "
            "connect' note lists the servers that failed by name; orion-introspect is down only "
            "if it is named there. Do not rebuild these records by querying Postgres or Redis "
            "by hand: say the tool was unavailable."
        ),
    ]


def append_introspect_harness_brief(
    parts: list[str], *, binding: IntrospectToolBindingV1 | None,
) -> None:
    if binding is None:
        return
    parts.extend(introspect_brief_lines(binding))
