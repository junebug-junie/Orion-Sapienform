from __future__ import annotations

from importlib.util import find_spec
from typing import Iterable


REAL_RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "bitsandbytes",
    "peft",
    "trl",
)


def missing_runtime_packages(packages: Iterable[str] = REAL_RUNTIME_PACKAGES) -> list[str]:
    missing: list[str] = []
    for pkg in packages:
        if find_spec(pkg) is None:
            missing.append(pkg)
    return missing


def ensure_runtime_packages(packages: Iterable[str] = REAL_RUNTIME_PACKAGES) -> None:
    missing = missing_runtime_packages(packages)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing runtime packages for real QLoRA run: {joined}")
