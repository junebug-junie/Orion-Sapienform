"""The prompt for an urgent curiosity run: an assignment, not an invitation.

`kickoff_prompt.build_kickoff_prompt` opens Orion's own time with material and
no subject. An urgent run is the opposite: Juniper (or a hardware rule) asked a
question about the machines right now, and the turn should answer it. So none
of the self-directed sections are here -- no priors, crystallizations, peer
notes, dreams, continuation, or hops -- only the question, the evidence Hub
collected, what to find out, how to look, where to write the verdict, and the
clock. The access guide and the clock are reused from `kickoff_prompt`, so the
credentials and deadline wording cannot drift between the two prompts.

The question is inserted verbatim. Nothing here branches on its words.

The verdict goes into one `:IncidentReport` node whose property names are read
back by `orion/curiosity/incident_report.py` -- change them in both places.
"""

from __future__ import annotations

import json

from orion.curiosity.incident_report import IS_REAL_VALUES, LABEL_INCIDENT_REPORT, SEVERITY_VALUES
from orion.curiosity.kickoff_prompt import GRAPH_URI, _access_section, _budget_section
from orion.curiosity.worldview import _RUN_ID_RE
from orion.schemas.curiosity_urgent import CuriosityUrgentSeedV1

EVIDENCE_CHAR_CAP = 24_000
TRUNCATION_MARKER = "[EVIDENCE TRUNCATED"

_HARDWARE_TABLES = (
    ("orion_biometrics_summary", "per-host temps, fans, power over time (node, timestamp)"),
    ("home_cooling_sample", "cabinet AC plug readings: watts, switch_on, stale (node, ts)"),
)


def _assignment_section(seed: CuriosityUrgentSeedV1) -> list[str]:
    opener = (
        "URGENT. Juniper asked you to investigate this now."
        if seed.trigger == "manual"
        else f"URGENT. A hardware rule fired ({seed.trigger}) and started this investigation."
    )
    lines = [
        opener,
        "This is an assignment, not your own time. Stay on it until you can answer.",
        "",
        "THE QUESTION:",
        "",
        seed.question,
        "",
    ]
    if seed.subject:
        lines += [f"Subject: {seed.subject}", ""]
    lines += [f"Requested at {seed.requested_at.isoformat()} by {seed.requested_by}.", ""]
    return lines


def _evidence_section(seed: CuriosityUrgentSeedV1) -> list[str]:
    if not seed.evidence:
        return [
            "EVIDENCE. No evidence was collected with this request. Go and read "
            "the machines yourself with the tools below.",
            "",
        ]
    text = json.dumps(seed.evidence, indent=2, ensure_ascii=False, default=str)
    if len(text) > EVIDENCE_CHAR_CAP:
        shown = text[:EVIDENCE_CHAR_CAP]
        text = (
            f"{shown}\n{TRUNCATION_MARKER}: {len(shown)} of {len(text)} characters "
            "shown. Query the rest yourself if you need it.]"
        )
    return [
        "EVIDENCE Hub collected when this was requested (a snapshot; it may "
        "already be out of date, and a section with an \"error\" key could not "
        "be read):",
        "",
        text,
        "",
    ]


def _checklist_section() -> list[str]:
    return [
        "WHAT TO FIND OUT, in this order:",
        "",
        "  1. Is it real, or a sensor fault? Check the reading against a second "
        "source before you trust it.",
        "  2. The likely cause.",
        "  3. How bad it is: low, high, or critical.",
        "  4. One concrete thing Juniper should do now.",
        "",
        "Cite only readings you actually saw -- in the evidence above or from a "
        "query you ran -- with the values they returned. If you cannot tell, the "
        "answer is unclear; that is a real answer.",
        "",
        "Look, do not touch. Do not restart services, change settings, or switch "
        "anything on or off. Juniper acts on what you recommend.",
        "",
    ]


def _hardware_access_lines(pool_url: str) -> list[str]:
    return [
        "  For this incident, the same psql also reads (if Juniper has applied "
        "the read-only grant; \"permission denied\" means not yet -- say so and "
        "use the evidence above):",
        *[f"      {name.ljust(36)} {what}" for name, what in _HARDWARE_TABLES],
        "  And the GPU pool, who holds which GPU right now:",
        f"    curl -s {pool_url}/v1/pool",
        "",
    ]


def _report_section(*, seed: CuriosityUrgentSeedV1, own_graph: str, run_id: str) -> list[str]:
    return [
        f"WRITE YOUR VERDICT AS ONE :{LABEL_INCIDENT_REPORT} in your own graph, "
        "as soon as you have one -- before you polish the prose. Hub reads it back "
        "and sends it to Juniper; if the clock runs out first, she gets no verdict.",
        "",
        f'    redis-cli -u "{GRAPH_URI}" \\',
        f"      GRAPH.QUERY {own_graph} '",
        f"      CREATE (:{LABEL_INCIDENT_REPORT} {{",
        f'        run_id: "{run_id}",',
        f'        incident_id: "{seed.incident_id}",',
        f'        is_real: "{"|".join(IS_REAL_VALUES)}",',
        '        likely_cause: "one sentence",',
        '        evidence: ["a reading you saw and its value", "a query you ran and what it returned"],',
        f'        severity: "{"|".join(SEVERITY_VALUES)}",',
        '        operator_action: "one concrete thing Juniper should do now",',
        "        confidence: 0.0,",
        "        written_at: timestamp()",
        "      })'",
        "",
        "Pick ONE value for is_real and for severity. confidence is your own "
        "belief, 0.0 to 1.0. evidence must be a list with at least one entry; a "
        "report without evidence is read as no verdict.",
        "",
        "QUOTING: the Cypher sits inside single quotes for the shell, and every "
        "value inside it uses double quotes, as above. Do not put an apostrophe "
        "or a double quote inside any value -- write \"does not\", not "
        "\"doesn't\". A stray quote breaks the command and nothing is written.",
        "",
        "Write exactly one. If you must correct it, write another; the newest is "
        "the one read. Check it landed:",
        "",
        f'    redis-cli -u "{GRAPH_URI}" \\',
        f"      GRAPH.QUERY {own_graph} 'MATCH (r:{LABEL_INCIDENT_REPORT} "
        f'{{run_id: "{run_id}"}}) RETURN r.is_real, r.severity, size(r.evidence)\'',
        "",
    ]


def _prose_is_report_section() -> list[str]:
    return [
        "THERE IS NO GRAPH TO WRITE TO THIS RUN, so your prose answer is the "
        "report. Make it complete on its own.",
        "",
    ]


_CLOSING = """\
ANSWER IN PLAIN PROSE. Lead with the verdict (real, sensor fault, or unclear),
then the likely cause, then the one action. Then the evidence you cited. Short
beats thorough."""


def build_urgent_prompt(
    seed: CuriosityUrgentSeedV1,
    *,
    run_id: str,
    own_graph: str = "orion_worldview",
    hub_url: str = "http://127.0.0.1:8080",
    pool_url: str = "http://orion-athena-gpu-pool:8127",
    graph_enabled: bool = True,
    atlas_graph: str = "orion_substrate",
) -> str:
    """Assemble the urgent investigation prompt.

    `pool_url` defaults to the pool's `app-net` name: verified 2026-09-28
    reachable from the harness-governor sandbox, where `127.0.0.1:8127` is not.
    The report template is only offered when a graph is configured and the run
    id is one the reader will accept; otherwise the prose is the report.
    """
    writable = graph_enabled and bool(_RUN_ID_RE.match(run_id or ""))
    lines = _assignment_section(seed)
    lines += _evidence_section(seed)
    lines += _checklist_section()
    lines += _access_section(
        own_graph=own_graph,
        atlas_graph=atlas_graph,
        hub_url=hub_url,
        graph_enabled=graph_enabled,
        writable=writable,
    )
    lines += _hardware_access_lines(pool_url)
    if writable:
        lines += _report_section(seed=seed, own_graph=own_graph, run_id=run_id)
    else:
        lines += _prose_is_report_section()
    # writable=False on purpose: that variant does not promise a continuation
    # note, and an urgent run has no :TurnOutcome to carry one.
    lines += _budget_section(writable=False)
    lines.append(_CLOSING)
    return "\n".join(lines)
