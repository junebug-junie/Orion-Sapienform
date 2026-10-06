"""Activation decay has one owner: orion-substrate-runtime's dynamics tick.

The Hub ran a second decay writer (decay_concept_activations, every 120 s)
against the same Falkor nodes until 2026-10-06. The two processes decayed from
their own caches and their writes landed out of order: live reads of
sub-concept-seed-juniper showed activation_decayed_at stepping back
(05:28:22 -> 05:28:18) and activation ticking up. Kill means kill: this guard
fails if any Hub module starts decaying activation or writing the decay stamp
again.
"""
from __future__ import annotations

import re
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"\bdecay_activation\b|\bACTIVATION_DECAYED_AT_KEY\b|\bactivation_decay_anchor\b|SUBSTRATE_DECAY_SCHEDULER"
)


def test_no_hub_module_decays_substrate_activation() -> None:
    offenders = []
    for path in sorted((HUB_ROOT / "scripts").rglob("*.py")) + sorted((HUB_ROOT / "app").rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path.relative_to(HUB_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == [], "Hub must not decay substrate activation:\n" + "\n".join(offenders)


def test_hub_env_template_has_no_decay_scheduler_keys() -> None:
    text = (HUB_ROOT / ".env_example").read_text(encoding="utf-8")
    assert not re.search(r"^\s*SUBSTRATE_DECAY_SCHEDULER_\w+=", text, re.M)
    assert not re.search(r"^\s*SUBSTRATE_DYNAMICS_DECAY_MODE=", text, re.M)
