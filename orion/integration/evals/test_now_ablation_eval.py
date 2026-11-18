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


def test_the_offline_lane_passes_on_the_checked_in_fixtures_and_its_verdict_comes_from_distance():
    assert ev.main(["--offline"]) == 0
    assert ev.main(["--offline", "--threshold", "2.0"]) == 1  # no content change can reach 2.0: the check must fail


def _one(model, **kw):
    rep = ev.run([{"case_id": "c", "organ": "prime", "prompt": PROMPT, "cited_event_ids": ["a:1"]}], model,
                 mode="t", **kw)
    return rep


def test_only_the_cite_token_differing_is_hollow_in_every_shape_an_organ_cites_in():
    """The ablated prompt cannot show the id, so dropping only the cite is no evidence the item shaped anything."""
    for with_cite in ('{"reading": "the cabinet is warm", "refs": [{"kind": "experience", "id": "a:1"}]}',
                      '{"interpretation": "the cabinet is warm", "cites": ["experience:a:1"]}',
                      "The cabinet is warm [experience:a:1]."):
        def model(case, system, prompt, w=with_cite):
            if "[experience:a:1]" in prompt:
                return w
            return w.replace('{"kind": "experience", "id": "a:1"}', "").replace('"experience:a:1"', "").replace(
                " [experience:a:1]", "")

        rep = _one(model)
        [r] = rep.results
        assert r.names_ref_full and not r.names_ref_ablated and not r.changed, with_cite


def test_a_content_change_without_naming_the_id_is_changed():
    def model(case, system, prompt):
        return "a mail truck idled on the walkway" if "[experience:a:1]" in prompt else "nothing stood out today"

    [r] = _one(model).results
    assert r.changed and not r.names_ref_full


def test_an_empty_output_is_refused_never_scored_hollow():
    rep = _one(lambda case, system, prompt: "   ")
    assert rep.results == [] and "empty output" in rep.refused[0]


def test_live_noise_raises_the_bar():
    calls = {"n": 0}

    def drifting(case, system, prompt):
        calls["n"] += 1
        if "[experience:a:1]" not in prompt:
            return "alpha beta gamma delta"
        # two replays of the same full prompt differ a lot: the noise floor
        return "alpha beta gamma delta" if calls["n"] % 2 else "epsilon zeta eta theta alpha"

    [r] = _one(drifting, measure_noise=True).results
    assert r.noise is not None and r.threshold >= r.noise + ev.NOISE_MARGIN - 1e-9
    assert r.changed is (r.distance >= r.threshold)
    assert _one(drifting, measure_noise=True).summary()["noise"]


def test_an_id_is_matched_as_a_whole_token():
    assert ev.names("see a:10 and b", "a:10") and not ev.names("see a:10", "a:1")
    assert ev.names('"cites": ["experience:a:1"]', "a:1") and not ev.names("experience:a:10", "a:1")
    assert not ev.names("xa:1", "a:1")
    assert ev.strip_cite("x [experience:a:10] y", "a:1") == "x [experience:a:10] y"


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
