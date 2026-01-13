from __future__ import annotations

import asyncio
import logging

from .service import EquilibriumService


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[equilibrium] %(levelname)s - %(message)s")
    svc = EquilibriumService()
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
