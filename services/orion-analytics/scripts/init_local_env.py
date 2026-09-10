#!/usr/bin/env python3
"""Create the ignored analytics env and initialize required local secrets."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
REQUIRED_SECRET_KEYS = (
    "ORION_ANALYTICS_DBT_PASSWORD",
    "ORION_ANALYTICS_READER_PASSWORD",
    "LIGHTDASH_SECRET",
    "LIGHTDASH_METADATA_PASSWORD",
    "LIGHTDASH_S3_SECRET_KEY",
)


def _required_positions(lines: list[str], *, source: Path) -> dict[str, int]:
    positions: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, line in enumerate(lines):
        match = KEY_LINE.match(line)
        if not match or match.group(1) not in REQUIRED_SECRET_KEYS:
            continue
        key = match.group(1)
        if key in positions:
            duplicates.add(key)
        positions[key] = index
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate required key(s) in {source}: {names}")
    return positions


def initialize_env(
    example_file: Path,
    env_file: Path,
    *,
    token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(30),
) -> tuple[str, ...]:
    """Fill required blank values without changing any existing nonblank value."""
    if not example_file.is_file():
        raise FileNotFoundError(f"analytics env example not found: {example_file}")

    example_lines = example_file.read_text().splitlines()
    example_positions = _required_positions(example_lines, source=example_file)
    missing_from_example = set(REQUIRED_SECRET_KEYS) - set(example_positions)
    if missing_from_example:
        names = ", ".join(sorted(missing_from_example))
        raise ValueError(f"required key(s) missing from {example_file}: {names}")

    env_exists = env_file.exists()
    lines = env_file.read_text().splitlines() if env_exists else example_lines.copy()
    positions = _required_positions(lines, source=env_file)
    generated: list[str] = []

    for key in REQUIRED_SECRET_KEYS:
        if key not in positions:
            lines.append(f"{key}={token_factory()}")
            positions[key] = len(lines) - 1
            generated.append(key)
            continue

        index = positions[key]
        match = KEY_LINE.match(lines[index])
        assert match is not None
        if not match.group(2).strip():
            lines[index] = f"{key}={token_factory()}"
            generated.append(key)

    env_file.parent.mkdir(parents=True, exist_ok=True)
    if generated or not env_exists:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{env_file.name}.", dir=env_file.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write("\n".join(lines) + "\n")
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, env_file)
        finally:
            temporary_path.unlink(missing_ok=True)
    os.chmod(env_file, 0o600)
    return tuple(generated)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example-file",
        type=Path,
        default=SERVICE_ROOT / ".env_example",
        help="template to read (default: service .env_example)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=SERVICE_ROOT / ".env",
        help="ignored env file to initialize (default: service .env)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generated = initialize_env(args.example_file, args.env_file)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"analytics env initialization failed: {exc}")
        return 1

    if generated:
        print("initialized required analytics values: " + ", ".join(generated))
    else:
        print("required analytics values are already set")
    print(f"analytics env ready: {args.env_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
