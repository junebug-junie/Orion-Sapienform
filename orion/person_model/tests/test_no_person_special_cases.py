"""Modularity gate (spec section 0): no code in orion/person_model/ names a person.

People are data (config/person_model/people.yaml). If a name from that file shows
up in module code outside tests/ and evals/, a special case has crept in.
"""

import re
from pathlib import Path

from orion.person_model.person import load_people

MODULE = Path(__file__).resolve().parents[1]
EXEMPT = {"tests", "evals"}


def _module_files():
    for p in sorted(MODULE.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".j2", ".json", ".md", ".txt"}:
            if EXEMPT & set(p.relative_to(MODULE).parts):
                continue
            yield p


def test_module_has_files():
    assert any(p.name == "linkage.py" for p in _module_files())


def test_no_known_person_name_in_module_code():
    names = {"juniper"} | {p.display_name.lower() for p in load_people() if p.role != "self"}
    hits = []
    for p in _module_files():
        text = p.read_text(encoding="utf-8").lower()
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", text):
                hits.append(f"{p.relative_to(MODULE)}: {n}")
    assert not hits, f"person-specific text in orion/person_model: {hits}"


def test_no_person_node_id_in_module_code():
    ids = {p.node_id for p in load_people()}
    for p in _module_files():
        text = p.read_text(encoding="utf-8")
        assert not any(i in text for i in ids), p
