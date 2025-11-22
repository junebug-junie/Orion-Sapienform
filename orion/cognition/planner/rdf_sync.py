from __future__ import annotations

"""
RDF Sync Utilities (YAML Verbs -> RDF Turtle)

This reads verbs/*.yaml and generates a Turtle document that can be
imported into GraphDB. It does NOT depend on VerbRegistry or models,
only on the raw YAML files.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

PREFIXES = """@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix orion: <http://conjourney.net/orion#> .

# Core classes
orion:Verb a rdfs:Class .
orion:Step a rdfs:Class .

# Core properties
orion:category a rdf:Property .
orion:priority a rdf:Property .
orion:interruptible a rdf:Property .
orion:canInterruptOthers a rdf:Property .
orion:timeoutMs a rdf:Property .
orion:maxRecursionDepth a rdf:Property .

orion:ofVerb a rdf:Property .
orion:order a rdf:Property .
orion:service a rdf:Property .
orion:requiresGpu a rdf:Property .
orion:requiresMemory a rdf:Property .
orion:promptTemplate a rdf:Property .

# ===============================
# Cognition Step Execution schema
# ===============================

orion:CognitiveStepExecution a rdfs:Class ;
  rdfs:label "Cognitive Step Execution" ;
  rdfs:comment "A single execution of a semantic verb/step in Orion's cognition cortex." .

orion:verb a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range rdfs:Resource .

orion:step a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range rdfs:Resource .

orion:verbName a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:stepName a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:serviceName a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:service a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range rdfs:Resource .

orion:originNode a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range rdfs:Resource .

orion:originNodeName a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:status a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:latencyMs a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:int .

orion:timestampEpoch a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:double .

orion:argsJson a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:contextJson a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .

orion:resultPreviewJson a rdf:Property ;
  rdfs:domain orion:CognitiveStepExecution ;
  rdfs:range xsd:string .
"""


def _ttl_literal(value: Any) -> str:
    """
    Render a Python value as a Turtle-safe literal.

    We only use this for *strings* in this module. It:
    - escapes backslashes
    - escapes double quotes
    - converts newlines and carriage returns to '\n' / '\r'
    so we never emit raw newlines inside a quoted literal.
    """
    # Don't use this for typed numeric/boolean literals
    s = str(value)
    s = (
        s.replace("\\", "\\\\")   # backslash
         .replace('"', '\\"')     # double quote
         .replace("\r", "\\r")    # carriage return
         .replace("\n", "\\n")    # newline
    )
    return f"\"{s}\""


def _load_verbs(verbs_dir: Path) -> Iterable[Dict[str, Any]]:
    """Yield all verb definitions as dicts from verbs/*.yaml."""
    for path in sorted(verbs_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "name" not in data:
            data["name"] = path.stem
        data["__file__"] = str(path)
        yield data


def _verb_triples(verb: Dict[str, Any]) -> str:
    """Return Turtle triples for a single verb and its steps."""
    name = verb.get("name")
    if not name:
        return ""

    verb_id = f"orion:verb_{name}"

    label = verb.get("label") or name
    desc = verb.get("description") or ""
    category = verb.get("category")
    priority = verb.get("priority")
    interruptible = bool(verb.get("interruptible", True))
    can_interrupt = bool(verb.get("can_interrupt_others", False))
    timeout_ms = int(verb.get("timeout_ms", 0) or 0)
    max_depth = int(verb.get("max_recursion_depth", 0) or 0)

    lines: List[str] = []

    # Verb header
    po: List[tuple[str, str]] = [
        ("a", "orion:Verb"),
        ("rdfs:label", _ttl_literal(label)),
    ]
    if desc:
        po.append(("rdfs:comment", _ttl_literal(desc)))
    if category:
        po.append(("orion:category", _ttl_literal(category)))
    if priority:
        po.append(("orion:priority", _ttl_literal(priority)))
    po.append(
        ("orion:interruptible", f"\"{str(interruptible).lower()}\"^^xsd:boolean")
    )
    po.append(
        ("orion:canInterruptOthers", f"\"{str(can_interrupt).lower()}\"^^xsd:boolean")
    )
    if timeout_ms:
        po.append(("orion:timeoutMs", f"\"{timeout_ms}\"^^xsd:int"))
    if max_depth:
        po.append(("orion:maxRecursionDepth", f"\"{max_depth}\"^^xsd:int"))

    lines.append(
        f"{verb_id} "
        + " ;\n  ".join(f"{p} {o}" for p, o in po)
        + " ."
    )
    lines.append("")

    # Steps
    plan_defs = verb.get("plan", []) or []
    sorted_defs = sorted(plan_defs, key=lambda s: s.get("order", 0))

    for step_def in sorted_defs:
        step_name = step_def.get("name", "")
        order = int(step_def.get("order", 0))
        step_id = f"orion:step_{name}_{order}"

        step_desc = step_def.get("description", "")
        services = step_def.get("services", []) or []
        requires_gpu = bool(step_def.get("requires_gpu", False))
        requires_mem = bool(step_def.get("requires_memory", False))
        prompt_template = (
            step_def.get("prompt_template") or verb.get("prompt_template")
        )

        spo: List[tuple[str, str]] = [
            ("a", "orion:Step"),
            ("orion:ofVerb", verb_id),
            ("rdfs:label", _ttl_literal(step_name)),
            ("orion:order", f"\"{order}\"^^xsd:int"),
        ]
        if step_desc:
            spo.append(("rdfs:comment", _ttl_literal(step_desc)))
        for svc in services:
            spo.append(("orion:service", _ttl_literal(svc)))
        if requires_gpu:
            spo.append(("orion:requiresGpu", "\"true\"^^xsd:boolean"))
        if requires_mem:
            spo.append(("orion:requiresMemory", "\"true\"^^xsd:boolean"))
        if prompt_template:
            spo.append(("orion:promptTemplate", _ttl_literal(prompt_template)))

        # Terminate step: last ';' -> '.'
        step_lines: List[str] = []
        step_lines.append(
            f"{step_id} "
            + " ;\n  ".join(f"{p} {o}" for p, o in spo)
            + " ."
        )
        step_lines.append("")
        lines.extend(step_lines)

    return "\n".join(lines)


def generate_turtle_for_all(base_dir: Path) -> str:
    """
    Generate Turtle for all verbs in base_dir/verbs.

    base_dir is typically the cognition root:
      /mnt/scripts/Orion-Sapienform/orion/cognition
    """
    verbs_dir = base_dir / "verbs"
    if not verbs_dir.exists():
        raise FileNotFoundError(f"verbs directory not found: {verbs_dir}")

    chunks: List[str] = [PREFIXES.strip(), ""]
    for verb_def in _load_verbs(verbs_dir):
        t = _verb_triples(verb_def)
        if t.strip():
            chunks.append(t)

    return "\n\n".join(ch for ch in chunks if ch.strip())


def write_turtle_file(base_dir: Path, out_path: Path | None = None) -> Path:
    """
    Generate Turtle for all verbs and write it to ontology/orion_cognition_generated.ttl
    (or to out_path if provided). Returns the path written.
    """
    ttl = generate_turtle_for_all(base_dir)
    if out_path is None:
        out_path = base_dir / "ontology" / "orion_cognition_generated.ttl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ttl, encoding="utf-8")
    return out_path


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

    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    ttl = generate_turtle_for_all(base_dir)

    if args.outfile:
        out_path = Path(args.outfile)
        out_path.write_text(ttl, encoding="utf-8")
        print(f"Wrote Turtle to {out_path}")
    else:
        print(ttl)
