"""The ablation harness (integration patch 3): ablation removes exactly the cited line, refuses a cite that was not
shown, counts a stub that ignores the cite as hollow and one that uses it as changed, and reports shares right."""

from __future__ import annotations

import pytest

from orion.integration.evals import now_ablation_fixtures as fx
from orion.integration.evals import run_now_ablation_eval as ev

PROMPT = "HEAD\n- [experience:a:1] camera: a truck (5 min ago)\n- [experience:b:2] reading: a paper (9 min ago)\nTAIL"


def test_ablate_removes_exactly_the_cited_line():
    assert ev.ablate(PROMPT, "a:1") == "HEAD\n- [experience:b:2] reading: a paper (9 min ago)\nTAIL"


def test_ablate_refuses_a_cite_that_is_not_in_the_prompt():
    with pytest.raises(ev.AblationError):
        ev.ablate(PROMPT, "never:9")


def test_a_cite_the_output_uses_is_changed_and_one_it_ignores_is_hollow():
    cases = [{"case_id": "used", "organ": "prime", "prompt": PROMPT, "cited_event_ids": ["a:1"], "stub_uses": ["a:1"]},
             {"case_id": "ignored", "organ": "prime", "prompt": PROMPT, "cited_event_ids": ["b:2"],
              "stub_uses": ["a:1"]},
             {"case_id": "absent", "organ": "reverie", "prompt": PROMPT, "cited_event_ids": ["zz:0"]}]
    rep = ev.run(cases, ev.stub_model, mode="t")
    s = rep.summary()
    assert s["cites"] == 2 and s["changed_share"] == 0.5 and s["hollow_share"] == 0.5
    assert s["hollow"][0].startswith("ignored:b:2") and len(s["refused"]) == 1
    assert s["per_organ"]["prime"] == {"n": 2, "changed": 1, "changed_share": 0.5, "hollow_share": 0.5}


def test_an_original_output_is_used_instead_of_replaying_the_full_prompt():
    calls = []

    def model(case, system, prompt):
        calls.append(prompt)
        return "same words"

    ev.run([{"case_id": "c", "prompt": PROMPT, "cited_event_ids": ["a:1"], "original_output": "same words"}],
           model, mode="t")
    assert len(calls) == 1 and "[experience:a:1]" not in calls[0]


def test_the_offline_lane_passes_on_the_checked_in_fixtures():
    assert ev.main(["--offline"]) == 0


def test_a_prime_style_cite_in_refs_counts_as_naming_the_item():
    """Prime cites as {"kind": "experience", "id": "<id>"}: the bare id is what is matched, not "experience:<id>"."""
    def model(case, system, prompt):
        return '{"reading": "r", "refs": [{"kind": "experience", "id": "a:1"}]}' if "[experience:a:1]" in prompt \
            else '{"reading": "r", "refs": []}'

    rep = ev.run([{"case_id": "p", "organ": "prime", "prompt": PROMPT, "cited_event_ids": ["a:1"]}], model,
                 mode="t", threshold=0.99)
    [r] = rep.results
    assert r.names_ref_full and not r.names_ref_ablated and r.changed


def test_the_fixtures_are_what_the_merged_renderers_build_today():
    """A renderer changed and the fixtures were not rebuilt: run
    python -m orion.integration.evals.now_ablation_fixtures --write"""
    pytest.importorskip("jinja2")
    pytest.importorskip("httpx")  # the reverie renderer's module imports it
    assert fx.FIXTURES.read_text(encoding="utf-8") == fx.to_jsonl(fx.build_cases())


def test_every_fixture_shows_its_cited_item_on_exactly_one_line():
    for case in ev.load_cases(fx.FIXTURES):
        for eid in case["cited_event_ids"]:
            assert case["prompt"].count(f"[experience:{eid}]") == 1, (case["case_id"], eid)
