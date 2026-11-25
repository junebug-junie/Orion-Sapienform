"""The integration gate (scripts/check_integration_seam.py): the repo passes; each planted producer fails and
names its fix; an exemption with a reason passes; one without a reason fails."""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_integration_seam", REPO / "scripts/check_integration_seam.py")
seam = importlib.util.module_from_spec(_spec)
sys.modules["check_integration_seam"] = seam
_spec.loader.exec_module(seam)


@pytest.fixture(scope="module")
def real():
    return seam.gather(REPO)


def planted(real, **changes):
    inp = replace(real, registry=copy.deepcopy(real.registry), notes=[])
    for k, v in changes.items():
        setattr(inp, k, v)
    if "repo_root" in changes:
        # Organs that already read the now keep their real reader under the planted root, so a test
        # that plants one organ's reader sees only that organ's verdict.
        for spec in (real.registry.get("organs") or {}).values():
            rel = (spec or {}).get("reads_now")
            if rel and (real.repo_root / rel).is_file() and not (inp.repo_root / rel).exists():
                (inp.repo_root / rel).parent.mkdir(parents=True, exist_ok=True)
                (inp.repo_root / rel).write_text((real.repo_root / rel).read_text(encoding="utf-8"), encoding="utf-8")
    return inp


def test_the_repo_passes(real):
    assert seam.check(real) == []
    assert real.channels_base is not None, "no merge base with origin/main: rule 6 did not run (fetch origin main)"
    assert len(real.families) >= 41 and "intention.carry" in real.workflows and "turn_promise" in real.intention_sources


def test_a_new_family_with_no_organ_fails_and_names_the_fix(real):
    errs = seam.check(planted(real, families=real.families + ["thought.new_organ"]))
    assert len(errs) == 1 and "family 'thought.new_organ'" in errs[0]
    assert "config/physiology/families.yaml" in errs[0] and "config/integration/organs.yaml" in errs[0]


def test_a_new_workflow_value_fails(real):
    errs = seam.check(planted(real, workflows=real.workflows + ["project.step"]))
    assert len(errs) == 1 and "workflow 'project.step'" in errs[0] and "DurableWorkflowV1" in errs[0]


def test_a_new_intention_adapter_fails(real):
    errs = seam.check(planted(real, intention_sources=real.intention_sources + ["project_step"]))
    assert len(errs) == 1 and "intention source adapter 'project_step'" in errs[0]


def test_a_new_channel_in_the_diff_fails_and_an_old_one_is_not_relitigated(real):
    base = sorted(set(real.channels_now) - {"orion:some:old"})
    errs = seam.check(planted(real, channels_now=real.channels_now + ["orion:new:organ_output"], channels_base=base))
    assert len(errs) == 1 and "channel 'orion:new:organ_output'" in errs[0] and "merge base" in errs[0]
    # Every existing channel is in the base: none of the ~365 is checked.
    assert seam.check(planted(real, channels_base=list(real.channels_now))) == []


def test_a_new_temporal_source_kind_with_no_organ_fails(real):
    errs = seam.check(planted(real, temporal_kinds=real.temporal_kinds + ["project_step"]))
    assert len(errs) == 1 and "TemporalSourceKind 'project_step'" in errs[0]


def test_an_exemption_with_a_reason_passes_and_one_without_fails(real):
    inp = planted(real, families=real.families + ["thought.new_organ"],
                  channels_now=real.channels_now + ["orion:new:organ_output"], channels_base=list(real.channels_now))
    inp.registry["exempt_families"]["thought.new_organ"] = "a probe, not an organ's output; removed in two weeks"
    inp.registry["exempt_channels"] = {"orion:new:organ_output": "carries a health ping, no experience"}
    assert seam.check(inp) == []
    inp.registry["exempt_families"]["thought.new_organ"] = "  "
    errs = seam.check(inp)
    assert any("has no reason" in e for e in errs) and any("thought.new_organ" in e for e in errs)


def test_a_registered_channel_passes(real):
    inp = planted(real, channels_now=real.channels_now + ["orion:new:organ_output"], channels_base=list(real.channels_now))
    inp.registry["organs"]["reverie"]["channels"] = ["orion:new:organ_output"]
    assert seam.check(inp) == []


def test_an_organ_must_emit_or_say_why_and_read_or_say_why(real):
    inp = planted(real)
    # Since integration patch 7a every organ emits: plant an organ that emits nothing and gives no reason.
    inp.registry["organs"]["personality"]["temporal_sources"] = []
    inp.registry["organs"]["personality"].pop("exempt_emit", None)
    del inp.registry["organs"]["camera"]["exempt_read"]
    errs = seam.check(inp)
    assert any("organ 'personality' emits no TemporalSourceKind" in e for e in errs)
    assert any("organ 'camera' has no reads_now module and no exempt_read" in e for e in errs)


def test_reads_now_must_really_read_the_now_and_record_cites(real, tmp_path):
    files = {
        "reader.py": ("from orion.temporal_self.now import build_now, record_use\n", True),
        "route_reader.py": ("NOW = 'http://durable-runs:8121/temporal-self/now?organ=prime'\n"
                            "USE = 'http://durable-runs:8121/temporal-self/now/use'\n", True),
        # reads but records nothing
        "reads_only.py": ("NOW = 'http://durable-runs:8121/temporal-self/now?organ=prime'\n", False),
        # liars: a comment, a module docstring, a function docstring (none is code)
        "comment_liar.py": ("# reads /temporal-self/now and posts /temporal-self/now/use\nimport orion.temporal_self\n", False),
        "docstring_liar.py": ('"""TODO: read GET /temporal-self/now and POST /temporal-self/now/use here."""\n', False),
        "func_docstring_liar.py": ('def f():\n    """Unlike /temporal-self/now and /temporal-self/now/use ..."""\n', False),
    }
    for name, (text, _) in files.items():
        (tmp_path / name).write_text(text)
    for name, ok in [*((n, ok) for n, (_, ok) in files.items()), ("missing.py", False)]:
        inp = planted(real, repo_root=tmp_path)
        # Organs that already read the now in the real repo point at files that do not exist under
        # tmp_path: point them at the valid planted reader so only the organ under test is judged.
        for organ, spec in inp.registry["organs"].items():
            if organ != "prime" and (spec or {}).get("reads_now"):
                spec["reads_now"] = "reader.py"
        inp.registry["organs"]["prime"]["reads_now"] = name
        inp.registry["organs"]["prime"].pop("exempt_read", None)  # prime reads the now since patch 3
        errs = seam.check(inp)
        assert (errs == []) is ok, (name, errs)


def test_an_organ_may_not_hold_a_claim_and_its_exemption(real, tmp_path):
    (tmp_path / "reader.py").write_text("from orion.temporal_self.now import build_now, record_use\n")
    inp = planted(real, repo_root=tmp_path)
    inp.registry["organs"]["prime"]["reads_now"] = "reader.py"
    inp.registry["organs"]["prime"]["exempt_read"] = "planted"  # and an exemption beside it
    inp.registry["organs"]["reverie"]["exempt_emit"] = "why"  # it emits
    errs = seam.check(inp)
    assert any("'prime' both names reads_now and is exempt" in e for e in errs)
    assert any("'reverie' both emits and is exempt" in e for e in errs)


def test_a_stale_channel_entry_fails(real):
    inp = planted(real)
    # Keep the real exemptions: replacing them would unregister a channel added on this branch.
    inp.registry["exempt_channels"] = {**(inp.registry.get("exempt_channels") or {}), "orion:no:such:channel": "why"}
    assert seam.check(inp) == ["channel 'orion:no:such:channel' is named in config/integration/organs.yaml "
                               "but does not exist; remove it."]


def test_a_claim_on_a_producer_that_does_not_exist_fails(real):
    inp = planted(real)
    inp.registry["organs"]["memory"]["workflows"].append("no.such.workflow")
    errs = seam.check(inp)
    assert errs == ["workflow 'no.such.workflow' is named in config/integration/organs.yaml but does not exist; remove it."]


def test_a_producer_claimed_twice_fails(real):
    inp = planted(real)
    inp.registry["organs"]["dream"]["families"].append("thought.reverie")
    assert any("claimed by 2 organs" in e for e in seam.check(inp))


def test_new_exemptions_are_listed_for_the_reviewer(real):
    inp = planted(real, base_registry=copy.deepcopy(real.registry))
    assert seam.new_exemptions(inp) == []
    inp.registry["exempt_workflows"]["project.step"] = "why"
    assert seam.new_exemptions(inp) == ["exempt_workflows.project.step: why"]


def test_the_registry_parses_for_the_now(real):
    from orion.integration.organs import load_organs

    reg = load_organs()
    assert set(reg.organs) == set(real.registry["organs"])
