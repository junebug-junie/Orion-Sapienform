"""One-time Ring login for RETINA_SOURCE_TYPE=ring.

    docker exec -it orion-vision-retina-doorbell python -m app.ring_login

Prompts for the Ring account email, password and 2FA code, then writes the
refresh token (owner-only) under RETINA_RING_STATE_DIR. The password is never
stored. Restart the container afterwards so the source picks the token up.
"""

from __future__ import annotations

import asyncio
import getpass
import json
from pathlib import Path

from .ring_source import TOKEN_FILE, USER_AGENT, hardware_id, token_updater
from .settings import get_settings


async def _login(state_dir: str) -> None:
    from ring_doorbell import Auth, Requires2FAError

    auth = Auth(USER_AGENT, None, token_updater(state_dir), hardware_id=hardware_id(state_dir))
    email = input("Ring account email: ").strip()
    password = getpass.getpass("Ring password: ")
    try:
        try:
            token = await auth.async_fetch_token(email, password)
        except Requires2FAError:
            token = await auth.async_fetch_token(email, password, input("Ring 2FA code: ").strip())
    finally:
        await auth.async_close()
    token_updater(state_dir)(token)
    print(f"Saved Ring token to {Path(state_dir) / TOKEN_FILE}. Restart the container.")


def main() -> None:
    state_dir = get_settings().RETINA_RING_STATE_DIR
    asyncio.run(_login(state_dir))


if __name__ == "__main__":
    main()
