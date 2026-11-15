"""Verb prompts print dicts and lists as JSON, never Python repr (executor._json_finalize).

stance_react printed association / repair_bundle / coalition_projection raw, so the model saw
``{'key': 'value'}`` and answered in kind: 11 stance_react Strata 502s on 2026-11-14, all
"Expecting property name enclosed in double quotes" (issue #3289).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]
STANCE = ROOT / "orion" / "cognition" / "prompts" / "stance_react.j2"


def _executor():
    app_dir = EXEC_ROOT / "app"
    pkg, app_pkg = "orion_cortex_exec_dict_json", "orion_cortex_exec_dict_json.app"
    for name, path in ((pkg, app_dir.parent), (app_pkg, app_dir)):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(path)]
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(f"{app_pkg}.executor", app_dir / "executor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dicts_and_lists_print_as_json_and_strings_unchanged():
    ex = _executor()
    out = ex._render_prompt(
        "a={{ a }} b={{ b }} c={{ c }} d={{ d | tojson }} n={{ n }}",
        {"a": {"mood": "Juniper's relieved", "n": 1}, "b": ["el Coco", "stitches"], "c": "plain 'text'",
         "d": {"x": 1}, "n": None},
    )
    assert "a={\"mood\": \"Juniper's relieved\", \"n\": 1}" in out
    assert 'b=["el Coco", "stitches"]' in out
    assert "c=plain 'text'" in out and 'd={"x": 1}' in out and "n=None" in out


def test_stance_prompt_shows_no_python_literal_dicts():
    ex = _executor()
    ctx = {
        "association": {"topics": ["el Coco", "stitches"], "juniper_mood": "relieved"},
        "repair_bundle": {"repair_pressure_level": 0.2, "notes": ["none"]},
        "coalition_projection": {"attended_node_ids": ["node:substrate.execution"], "dwell_ticks": 0},
        "mind_coloring": {"attention_frontier": ["walkway"], "reflective_themes": ["care"],
                          "curiosity_threads": ["vehicle records"], "self_relevance": "low",
                          "juniper_relevance": "high", "identity_salience": "low"},
        "stance_inputs": {"recent_conversation": ["Juniper: el Coco had his stitches out"]},
        "user_message": "el Coco just had his butt stitches removed, yay!",
    }
    out = ex._render_prompt(STANCE.read_text(encoding="utf-8"), ctx)
    assert "{'" not in out and "['" not in out, [l for l in out.splitlines() if "{'" in l or "['" in l][:3]
    line = next(l for l in out.splitlines() if l.startswith("- association:"))
    assert json.loads(line.split(":", 1)[1]) == ctx["association"]


def test_evals_and_replays_render_like_production():
    """The shared env the stance evals and model replay use prints the same JSON as cortex-exec."""
    from orion.cognition.prompt_render import prompt_environment

    ex = _executor()
    template, ctx = "x={{ x }}", {"x": {"a": None, "b": [True, "it's"]}}
    assert prompt_environment().from_string(template).render(**ctx) == ex._render_prompt(template, ctx)
    assert ex._render_prompt(template, ctx) == 'x={"a": null, "b": [true, "it\'s"]}'
