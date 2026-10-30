"""Merge Orion's chat defaults into a Strata run config (stdlib only; runs on the Strata image's Python).

Strata's setup rewrites only its own keys (setup.py SETUP_KEYS) and keeps the rest, so a key merged
here survives restarts and re-setup. The defaults match the llama.cpp chat lane they replace:
thinking off unless a request asks, and its sampling.

usage: strata_apply_defaults.py <strata-<model>.json> <defaults.json>
"""
import json
import sys
from pathlib import Path


def merged(config: dict, defaults: dict) -> dict:
    return {**config, **defaults}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    path, defaults = Path(argv[1]), json.loads(Path(argv[2]).read_text())
    if not path.is_file():
        print(f"{path} does not exist: start Strata once so its setup writes it", file=sys.stderr)
        return 1
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    changed = sorted(k for k in defaults if config.get(k) != defaults[k])
    if changed:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(merged(config, defaults), indent=1), encoding="utf-8")
        tmp.replace(path)
    print(f"{path}: {'set ' + ', '.join(changed) if changed else 'already current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
