"""The durable-runs persist hands the referent step its kill switches (memory Stage 2)."""

from __future__ import annotations

from types import SimpleNamespace

from app.admission_runtime import AdmissionRuntime


def _policy(**settings):
    return AdmissionRuntime._referent_policy(SimpleNamespace(settings=SimpleNamespace(**settings)))


def test_referents_disabled_skips_the_step():
    assert _policy(memory_referents_enabled=False) is None


def test_each_acceptance_rule_has_its_own_switch():
    p = _policy(memory_referents_enabled=True, memory_alias_grounding_auto_accept=False,
                memory_cooccurrence_auto_accept=True)
    assert (p.grounding_auto_accept, p.cooccurrence_auto_accept) == (False, True)
    p = _policy(memory_referents_enabled=True, memory_alias_grounding_auto_accept=True,
                memory_cooccurrence_auto_accept=False)
    assert (p.grounding_auto_accept, p.cooccurrence_auto_accept) == (True, False)


def test_the_flags_ship_on():
    from app.settings import Settings

    fields = Settings.model_fields
    assert all(fields[name].default is True for name in (
        "memory_referents_enabled", "memory_alias_grounding_auto_accept", "memory_cooccurrence_auto_accept"))
