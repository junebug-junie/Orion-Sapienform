import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_settings(rel: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cortex_exec_grammar_default_on() -> None:
    src = (REPO / "services/orion-cortex-exec/app/settings.py").read_text()
    assert 'publish_cortex_exec_grammar: bool = Field(True' in src


def test_substrate_reducers_default_on() -> None:
    import re
    src = (REPO / "services/orion-substrate-runtime/app/settings.py").read_text()
    # enable_chat_grammar_reducer is single-line; execution_trajectory is multi-line
    # (Field(\n    False, ...)) so match across whitespace/newlines with a regex.
    assert 'enable_chat_grammar_reducer: bool = Field(True' in src
    assert re.search(
        r"enable_execution_trajectory_reducer[^=]*=\s*Field\(\s*True", src
    ), "ENABLE_EXECUTION_TRAJECTORY_REDUCER default must be flipped to True"


def test_hub_chat_grammar_default_on() -> None:
    src = (REPO / "services/orion-hub/app/settings.py").read_text()
    assert 'PUBLISH_HUB_CHAT_GRAMMAR: bool = Field(default=True' in src


def test_cortex_orch_grammar_default_on() -> None:
    src = (REPO / "services/orion-cortex-orch/app/settings.py").read_text()
    assert 'publish_cortex_orch_grammar: bool = Field(True' in src


def test_route_grammar_reducer_default_on() -> None:
    src = (REPO / "services/orion-substrate-runtime/app/settings.py").read_text()
    assert 'enable_route_grammar_reducer: bool = Field(True' in src
