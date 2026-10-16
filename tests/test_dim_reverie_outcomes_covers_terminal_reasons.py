"""Every visual chain terminal reason the schema can persist has a row in the analytics outcome
dimension. `resource_deferred` was missing (GPU pool parent spec, downstream readers item 8), so
deferred chains fell out of the dim join (assert_visual_reverie_joins_do_not_fan_out would fail on
the first one). Fixed in GPU pool stage 5.4; this keeps the next new reason from repeating it."""
from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from orion.schemas.reverie_visual import VisualTerminalReason

MART = Path(__file__).resolve().parents[1] / "services/orion-analytics/models/marts/dim_reverie_outcomes.sql"


def test_every_visual_terminal_reason_has_an_outcome_row():
    keys = set(re.findall(r"^\s*\('([a-z_]+)',", MART.read_text(), flags=re.M))
    missing = sorted(set(get_args(VisualTerminalReason)) - keys)
    assert not missing, f"dim_reverie_outcomes.sql lacks rows for {missing}"
    orders = [int(n) for n in re.findall(r",\s*(\d+)\)\s*,?\s*$", MART.read_text(), flags=re.M)]
    assert len(orders) == len(set(orders)), "sort_order must stay unique"


def test_every_text_terminal_reason_has_an_outcome_row():
    """Same guarantee for the text chain (thought_complete added 2026-10-16)."""
    from orion.schemas.reverie import TerminalReason

    keys = set(re.findall(r"^\s*\('([a-z_]+)',", MART.read_text(), flags=re.M))
    missing = sorted(set(get_args(TerminalReason)) - keys)
    assert not missing, f"dim_reverie_outcomes.sql lacks rows for {missing}"


def _accepted_values(node, out):
    if isinstance(node, dict):
        av = node.get("accepted_values")
        if isinstance(av, dict) and isinstance(av.get("values"), list):
            out.append(av["values"])
        for v in node.values():
            _accepted_values(v, out)
    elif isinstance(node, list):
        for v in node:
            _accepted_values(v, out)
    return out


def test_every_text_chain_accepted_values_list_covers_terminal_reason():
    """dbt accepted_values on the text-chain source, staging and fact columns must
    list every TerminalReason, or the first new reason fails `dbt test`."""
    import yaml
    from orion.schemas.reverie import TerminalReason

    models = MART.parents[1]
    targets = {
        models / "sources.yml": ("substrate_reverie_chain", "terminal_reason"),
        models / "staging" / "staging.yml": ("stg_reverie_chains", "outcome_key"),
        models / "marts" / "marts.yml": ("fct_reverie_chains", "outcome_key"),
    }
    want = set(get_args(TerminalReason))
    for path, (owner, column) in targets.items():
        doc = yaml.safe_load(path.read_text())
        owners = []
        for src in doc.get("sources", []) or []:
            owners += src.get("tables", []) or []
        owners += doc.get("models", []) or []
        node = next(o for o in owners if o.get("name") == owner)
        col = next(c for c in node["columns"] if c["name"] == column)
        lists = _accepted_values(col, [])
        assert lists, f"{path.name}:{owner}.{column} has no accepted_values"
        for values in lists:
            missing = sorted(want - set(values))
            assert not missing, f"{path.name}:{owner}.{column} lacks {missing}"
