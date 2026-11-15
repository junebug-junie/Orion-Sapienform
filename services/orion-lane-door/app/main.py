"""``python -m app.main``: run one lane door (settings from the environment)."""
from __future__ import annotations

import asyncio

from .door import serve
from .settings import load


def main() -> None:
    asyncio.run(serve(load()))


if __name__ == "__main__":
    main()
