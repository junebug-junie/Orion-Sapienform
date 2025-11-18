# orion-cognition/planner/rdf_sync.py

"""
RDF Sync Utilities (YAML Verbs -> RDF Turtle)
---------------------------------------------

This module takes the in-repo cognitive configs (YAML verbs) and
turns them into RDF triples consistent with orion_cognition_ontology.ttl.

It does NOT depend on GraphDB directly; instead it can output a Turtle
string and (optionally) push it to a SPARQL endpoint (e.g., GraphDB)
via HTTP.

Key entrypoints:

    - generate_turtle_for_all(base_dir: Path) -> str
    - push_turtle_to_graphdb(ttl: str, endpoint_url: str, graph_uri: Optional[str])

This is still firmly in the cognition layer. Execution Cortex remains untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

try:
    import requests  # optional, only needed if you call push_turtle_to_graphdb
except ImportError:
    requests = None  # type: ignore

from .loader import VerbRegistry
from .models import VerbConfig, PlanStepConfig, SafetyRuleConfig


# -----------------------------
# IRI / name helpers
# -----------------------------


def _pascal_case(name: str) -> str:
    """
    Convert snake_case or kebab-case or spaced text to PascalCase.
    Example:
        'introspection_prompt' -> 'IntrospectionPrompt'
    """
    # replace non-alnum with space, then title-case and join
    cleaned_chars: List[str] = []
    for ch in name:
        if ch.isalnum():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = "".join(cleaned_chars)
    parts = [p for p in cleaned.split(" ") if p]
    return "".join(p.capitalize() for p in parts)


def _sanitize_local_name(name: str) -> str:
    """
    Turn arbitrary text into a safe local name for an IRI fragment.
    """
    out: List[str] = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    # avoid leading digits
    if out and out[0].isdigit():
        out.insert(0, "_")
    return "".join(out)


def verb_iri(verb: VerbConfig) -> str:
    """Return the orion: IRI for a verb."""
    return f"orion:{_pascal_case(verb.name)}"


def plan_iri(verb: VerbConfig) -> str:
    """Return the orion: IRI for the verb's Plan."""
    return f"{verb_iri(verb)}Plan"


def step_iri(verb: VerbConfig, step: PlanStepConfig) -> str:
    """Return the orion: IRI for a PlanStep."""
    v_name = _pascal_case(verb.name)
    s_name = _pascal_case(step.name)
    return f"orion:{v_name}_Step_{s_name}"


def category_iri(category: str) -> str:
    """Map category string (SelfModification, etc.) to an orion: VerbCategory IRI."""
    return f"orion:{_sanitize_local_name(category)}"


def service_iri(service_name: str) -> str:
    """Map service string to orion:Service IRI."""
    return f"orion:{_sanitize_local_name(service_name)}"


def system_state_iri(state_name: str) -> str:
    """Map system state string (HighLoad, Idle) to orion:SystemState IRI."""
    return f"orion:{_sanitize_local_name(state_name)}"


def safety_rule_iri(rule: SafetyRuleConfig, verb: VerbConfig) -> str:
    """Map safety rule to an orion:SafetyRule IRI."""
    base = f"SafetyRule_{rule.rule_name}"
    return f"orion:{_sanitize_local_name(base)}"


def prompt_template_iri(template_name: str) -> str:
    """
    Map a Jinja template filename to a PromptTemplate IRI.

    Example:
        'introspection_prompt.j2' -> 'orion:IntrospectionPromptTemplate'
    """
    # strip extension
    if "." in template_name:
        base = template_name.rsplit(".", 1)[0]
    else:
        base = template_name
    return f"orion:{_pascal_case(base)}PromptTemplate"


# -----------------------------
# Turtle generation
# -----------------------------


def _ttl_literal(value) -> str:
    """
    Render a Python value as a Turtle literal.
    Note: very minimal; enough for our use (strings, bools, ints).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # assume string
    # escape quotes
    s = str(value).replace('"', '\\"')
    return f"\"{s}\""


def _verb_turtle(verb: VerbConfig) -> str:
    """
    Build Turtle triples for a single Verb, its Plan, Steps, and SafetyRules.
    """
    lines: List[str] = []

    v_iri = verb_iri(verb)
    p_iri = plan_iri(verb)
    cat_iri = category_iri(verb.category)
    pt_iri = prompt_template_iri(verb.prompt_template)

    # Verb header
    lines.append(f"{v_iri}")
    lines.append(f"    rdf:type        orion:Verb ;")
    lines.append(f"    rdfs:label      {_ttl_literal(verb.display_label())} ;")
    lines.append(f"    orion:name      {_ttl_literal(verb.name)} ;")
    lines.append(f"    orion:description {_ttl_literal(verb.description)} ;")
    lines.append(f"    orion:hasCategory {cat_iri} ;")
    lines.append(f"    orion:priority  {_ttl_literal(verb.priority)} ;")
    lines.append(f"    orion:isInterruptible {_ttl_literal(verb.interruptible)}^^xsd:boolean ;")
    lines.append(f"    orion:canInterruptOthers {_ttl_literal(verb.can_interrupt_others)}^^xsd:boolean ;")
    lines.append(f"    orion:requiresGPU {_ttl_literal(verb.requires_gpu)}^^xsd:boolean ;")
    lines.append(f"    orion:requiresMemoryAccess {_ttl_literal(verb.requires_memory)}^^xsd:boolean ;")
    lines.append(f"    orion:timeoutMs  {_ttl_literal(verb.timeout_ms)} ;")
    lines.append(f"    orion:maxRecursionDepth {_ttl_literal(verb.max_recursion_depth)} ;")
    lines.append(f"    orion:usesPromptTemplate {pt_iri} ;")
    lines.append(f"    orion:hasPlan   {p_iri} .")
    lines.append("")  # blank line

    # Plan
    lines.append(f"{p_iri}")
    lines.append(f"    rdf:type        orion:Plan ;")
    lines.append(f"    rdfs:label      {_ttl_literal(verb.display_label() + ' Plan')} ;")
    lines.append(f"    orion:description {_ttl_literal('Plan for verb ' + verb.name)} ;")

    # Plan-level services
    for svc in verb.services:
        lines.append(f"    orion:requiresService {service_iri(svc)} ;")

    # We'll link steps with hasPlanStep below; terminate later
    # For now, no trailing '.' yet; we may append in a second block.

    # Steps
    step_lines: List[str] = []
    for step_cfg in sorted(verb.plan, key=lambda s: s.order):
        s_iri = step_iri(verb, step_cfg)

        # Step resource
        step_lines.append(f"{s_iri}")
        step_lines.append(f"    rdf:type        orion:PlanStep ;")
        step_lines.append(f"    rdfs:label      {_ttl_literal(step_cfg.name)} ;")
        step_lines.append(f"    orion:description {_ttl_literal(step_cfg.description)} ;")
        step_lines.append(f"    orion:stepOrder {_ttl_literal(step_cfg.order)} ;")
        step_lines.append(f"    orion:requiresGPU {_ttl_literal(step_cfg.requires_gpu)}^^xsd:boolean ;")
        step_lines.append(f"    orion:requiresMemoryAccess {_ttl_literal(step_cfg.requires_memory)}^^xsd:boolean ;")

        # Step-level services
        for svc in step_cfg.services:
            step_lines.append(f"    orion:requiresService {service_iri(svc)} ;")

        # Step-specific prompt template (optional)
        if step_cfg.prompt_template:
            s_pt_iri = prompt_template_iri(step_cfg.prompt_template)
            step_lines.append(f"    orion:stepPromptTemplate {s_pt_iri} ;")

        # Terminate resource
        # replace last ';' with '.'
        if step_lines[-1].strip().endswith(";"):
            last = step_lines[-1]
            step_lines[-1] = last.rstrip(" ;") + " ."
        else:
            step_lines.append("    .")
        step_lines.append("")  # blank

    # Now link plan to steps
    for step_cfg in sorted(verb.plan, key=lambda s: s.order):
        s_iri = step_iri(verb, step_cfg)
        lines.append(f"    orion:hasPlanStep {s_iri} ;")

    # terminate plan resource: replace last ';' with '.'
    if lines[-1].strip().endswith(";"):
        last = lines[-1]
        lines[-1] = last.rstrip(" ;") + " ."
    else:
        lines.append("    .")
    lines.append("")

    # Safety rules
    safety_lines: List[str] = []
    for rule in verb.safety_rules:
        r_iri = safety_rule_iri(rule, verb)
        safety_lines.append(f"{r_iri}")
        safety_lines.append(f"    rdf:type        orion:SafetyRule ;")
        safety_lines.append(f"    rdfs:label      {_ttl_literal(rule.rule_name)} ;")
        safety_lines.append(f"    orion:description {_ttl_literal('Safety rule for verb ' + verb.name)} ;")
        safety_lines.append(f"    orion:governs   {v_iri} ;")
        if rule.applies_when_state:
            safety_lines.append(
                f"    orion:appliesWhenState {system_state_iri(rule.applies_when_state)} ;"
            )
        # terminate
        if safety_lines[-1].strip().endswith(";"):
            last = safety_lines[-1]
            safety_lines[-1] = last.rstrip(" ;") + " ."
        else:
            safety_lines.append("    .")
        safety_lines.append("")

    all_lines = lines + step_lines + safety_lines
    return "\n".join(all_lines)


def generate_turtle_for_all(base_dir: Path) -> str:
    """
    Load all verbs from base_dir/verbs and generate a Turtle string
    describing them (Verbs, Plans, Steps, SafetyRules).

    This Turtle assumes the following prefixes are defined:

        @prefix orion: <http://orion.ai/ontology#> .
        @prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .

    You can prepend those yourself or include them here.
    """
    verbs_dir = base_dir / "verbs"
    registry = VerbRegistry(verbs_dir=verbs_dir)
    registry.load(reload=True)
    verbs: Dict[str, VerbConfig] = registry.all_verbs()

    header = [
        "@prefix orion: <http://orion.ai/ontology#> .",
        "@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "# Auto-generated Verb/Plan/Step/SafetyRule definitions",
        "",
    ]

    body_lines: List[str] = []
    for verb in verbs.values():
        body_lines.append(_verb_turtle(verb))
        body_lines.append("")

    return "\n".join(header + body_lines)

def push_turtle_to_graphdb(
    ttl: str,
    endpoint_url: str,
    graph_uri: Optional[str] = None,
) -> None:
    """
    Push Turtle to a SPARQL endpoint (e.g., GraphDB) using SPARQL UPDATE.

    Args:
        ttl: Turtle text to insert.
        endpoint_url: SPARQL update endpoint URL.
                      e.g. http://localhost:7200/repositories/orion/statements
        graph_uri: Named graph URI, or None for default graph.

    NOTE: Requires 'requests' to be installed.
    """
    if requests is None:
        raise RuntimeError(
            "The 'requests' library is not available. Install it to use push_turtle_to_graphdb."
        )

    # Basic SPARQL INSERT DATA wrapper
    if graph_uri:
        sparql = f"""
        INSERT DATA {{
            GRAPH <{graph_uri}> {{
                {ttl}
            }}
        }}
        """
    else:
        sparql = f"""
        INSERT DATA {{
            {ttl}
        }}
        """

    resp = requests.post(
        endpoint_url,
        data=sparql.encode("utf-8"),
        headers={"Content-Type": "application/sparql-update"},
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"SPARQL update failed (status {resp.status_code}): {resp.text}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate RDF Turtle from Orion cognition YAML verbs."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Path to orion-cognition root (containing verbs/).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="If provided, write Turtle to this file instead of stdout.",
    )
    parser.add_argument(
        "--push-endpoint",
        type=str,
        help="If provided, push Turtle to this SPARQL update endpoint.",
    )
    parser.add_argument(
        "--graph-uri",
        type=str,
        help="Named graph URI for SPARQL INSERT (optional).",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()

    ttl = generate_turtle_for_all(base_dir)

    if args.outfile:
        out_path = Path(args.outfile)
        out_path.write_text(ttl, encoding="utf-8")
        print(f"Wrote Turtle to {out_path}")
    else:
        # print to stdout
        print(ttl)

    if args.push_endpoint:
        push_turtle_to_graphdb(ttl, endpoint_url=args.push_endpoint, graph_uri=args.graph_uri)
        print(f"Pushed Turtle to {args.push_endpoint}")
