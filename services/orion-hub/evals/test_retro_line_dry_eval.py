"""The line turn's dry eval (retro_line_dry_eval.py) passes: game block, near-copy grouping, CHECKS parsing."""
import importlib.util
from pathlib import Path


def test_retro_line_dry_eval_passes():
    path = Path(__file__).with_name("retro_line_dry_eval.py")
    spec = importlib.util.spec_from_file_location("retro_line_dry_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
